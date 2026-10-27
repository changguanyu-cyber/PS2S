from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from torch import Tensor
from ps2s.data.molgraph import MolGraph
from ps2s.featurizers.base import Featurizer, GraphFeaturizer
from ps2s.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from ps2s.utils.utils import is_cuikmolmaker_available
from ps2s.utils.utils import dct_matrix as DCT
from ps2s.featurizers.molgraph.utils import get_modified_distance_matrix
if is_cuikmolmaker_available():
    import cuik_molmaker
import random, math

def sample_virtual_edges(candidate_pairs, sigma=0.7, keep_minimum=1):
    sampled = []
    for a, b in candidate_pairs:
        x = random.gauss(0, sigma)
        p = math.exp(-x ** 2 / (2 * sigma * sigma))
        if random.random() < p:
            sampled.append((a, b))
    if len(sampled) < keep_minimum and len(candidate_pairs) > 0:
        sampled.append(random.choice(candidate_pairs))
    return sampled

@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(self, mol: Chem.Mol, atom_features_extra: np.ndarray | None=None, bond_features_extra: np.ndarray | None=None) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        from rdkit import Chem
        smiles = Chem.MolToSmiles(mol)
        distance_matrix = []
        distance_matrix = get_modified_distance_matrix(mol)
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))
        E = []
        edge_index = [[], []]
        bond_set = set()
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]))
            u, v = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            E.append(x_e)
            edge_index[0].append(u)
            edge_index[1].append(v)
            E.append(x_e)
            edge_index[0].append(v)
            edge_index[1].append(u)
            bond_set.add((u, v))
            bond_set.add((v, u))
        zero_bond_feat = np.zeros(self.bond_fdim, dtype=np.single)
        vin_candidates = []
        for frag in Chem.GetMolFrags(mol):
            frag = list(frag)
            for i in range(len(frag)):
                for j in range(i + 1, len(frag)):
                    a, b = (frag[i], frag[j])
                    if (a, b) not in bond_set:
                        vin_candidates.append((a, b))
        sampled_vin = sample_virtual_edges(vin_candidates, sigma=0.7)
        for a, b in sampled_vin:
            E.append(zero_bond_feat)
            edge_index[0].append(a)
            edge_index[1].append(b)
            E.append(zero_bond_feat)
            edge_index[0].append(b)
            edge_index[1].append(a)
        conn_atoms = list({nbr.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0 for nbr in atom.GetNeighbors()})
        for i in range(len(conn_atoms)):
            for j in range(i + 1, len(conn_atoms)):
                a, b = (conn_atoms[i], conn_atoms[j])
                E.append(zero_bond_feat)
                edge_index[0].append(a)
                edge_index[1].append(b)
                E.append(zero_bond_feat)
                edge_index[0].append(b)
                edge_index[1].append(a)
        E = np.array(E, dtype=np.single)
        edge_index = np.array(edge_index, dtype=int)
        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        V_mean = V.mean(axis=0)
        D = DCT(V.shape[1])
        c_DCT = D @ V_mean
        del D
        c_DCT_expand = np.tile(c_DCT, (V.shape[0], 1))
        V = V + c_DCT_expand
        return MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index, distance_matrix=distance_matrix, smiles=smiles)

@dataclass(repr=False, eq=False, slots=True)
class BatchCuikMolGraph:
    V: Tensor
    E: Tensor
    edge_index: Tensor
    rev_edge_index: Tensor
    batch: Tensor
    __size: int = field(init=False)

    def __post_init__(self):
        self.__size = self.batch[-1].item() + 1

    def __len__(self) -> int:
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)

@dataclass
class CuikmolmakerMolGraphFeaturizer(Featurizer[list[str], BatchCuikMolGraph]):
    atom_featurizer_mode: Literal['V1', 'V2', 'ORGANIC', 'RIGR'] = 'V2'
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0
    add_h: bool = False
    atom_fdim: int = field(init=False)
    bond_fdim: int = field(init=False)

    def __post_init__(self):
        if not is_cuikmolmaker_available():
            raise ImportError(f"CuikmolmakerMolGraphFeaturizer requires cuik-molmaker package to be installed. Please install it using `python {Path(__file__).parents[2] / Path('scripts/check_and_install_cuik_molmaker.py')}`.")
        atom_props_float = ['aromatic', 'mass']
        bond_props = ['is-null', 'bond-type-onehot', 'conjugated', 'in-ring', 'stereo']
        self.bond_fdim = 14
        self.atom_featurizer_mode = self.atom_featurizer_mode.upper()
        if self.atom_featurizer_mode == 'V1':
            atom_props_onehot = ['atomic-number', 'total-degree', 'formal-charge', 'chirality', 'num-hydrogens', 'hybridization']
            self.atom_fdim = 133
        elif self.atom_featurizer_mode == 'V2':
            atom_props_onehot = ['atomic-number-common', 'total-degree', 'formal-charge', 'chirality', 'num-hydrogens', 'hybridization-expanded']
            self.atom_fdim = 72
        elif self.atom_featurizer_mode == 'ORGANIC':
            atom_props_onehot = ['atomic-number-organic', 'total-degree', 'formal-charge', 'chirality', 'num-hydrogens', 'hybridization-organic']
            self.atom_fdim = 44
        elif self.atom_featurizer_mode == 'RIGR':
            atom_props_onehot = ['atomic-number-common', 'total-degree', 'num-hydrogens']
            atom_props_float = ['mass']
            bond_props = ['is-null', 'in-ring']
            self.atom_fdim = 52
            self.bond_fdim = 2
        else:
            raise ValueError(f'Invalid atom featurizer mode: {self.atom_featurizer_mode}')
        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_props_onehot)
        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_tensor(atom_props_float)
        self.bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(bond_props)
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(self, smiles_list: list[str], atom_features_extra: np.ndarray | None=None, bond_features_extra: np.ndarray | None=None) -> BatchCuikMolGraph:
        offset_carbon, duplicate_edges, add_self_loop = (False, True, False)
        atom_feats, bond_feats, edge_index, rev_edge_index, batch = cuik_molmaker.batch_mol_featurizer(smiles_list, self.atom_property_list_onehot, self.atom_property_list_float, self.bond_property_list, self.add_h, offset_carbon, duplicate_edges, add_self_loop)
        if atom_features_extra is not None:
            atom_features_extra = torch.tensor(atom_features_extra, dtype=torch.float32)
            atom_feats = torch.cat((atom_feats, atom_features_extra), dim=1)
        if bond_features_extra is not None:
            bond_features_extra = np.repeat(bond_features_extra, repeats=2, axis=0)
            bond_features_extra = torch.tensor(bond_features_extra, dtype=torch.float32)
            bond_feats = torch.cat((bond_feats, bond_features_extra), dim=1)
        return BatchCuikMolGraph(V=atom_feats, E=bond_feats, edge_index=edge_index, rev_edge_index=rev_edge_index, batch=batch)
