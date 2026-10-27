from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from rdkit.Chem import AllChem as Chem
from ps2s.featurizers import Featurizer
from ps2s.utils import make_mol
MoleculeFeaturizer = Featurizer[Chem.Mol, np.ndarray]

@dataclass(slots=True)
class _DatapointMixin:
    y: np.ndarray | None = None
    weight: float = 1.0
    gt_mask: np.ndarray | None = None
    lt_mask: np.ndarray | None = None
    x_d: np.ndarray | None = None
    x_phase: list[float] = None
    name: str | None = None

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.x_d is not None:
            self.x_d[np.isnan(self.x_d)] = NAN_TOKEN

    @property
    def t(self) -> int | None:
        return len(self.y) if self.y is not None else None

@dataclass
class _MoleculeDatapointMixin:
    mol: Chem.Mol

    @classmethod
    def from_smi(cls, smi: str, *args, keep_h: bool=False, add_h: bool=False, ignore_stereo: bool=False, reorder_atoms: bool=False, **kwargs) -> _MoleculeDatapointMixin:
        mol = make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms)
        kwargs['name'] = smi if 'name' not in kwargs else kwargs['name']
        return cls(mol, *args, **kwargs)

@dataclass
class _LazyMoleculeDatapointMixin:
    smiles: str
    _keep_h: bool = False
    _add_h: bool = False
    _ignore_stereo: bool = False
    _reorder_atoms: bool = False
    _mol_cache: Chem.Mol = field(default=None, repr=False, compare=False)

    @property
    def mol(self) -> Chem.Mol:
        if self._mol_cache is None:
            self._mol_cache = make_mol(self.smiles, self._keep_h, self._add_h, self._ignore_stereo, self._reorder_atoms)
        return self._mol_cache

@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    V_f: np.ndarray | None = None
    E_f: np.ndarray | None = None
    V_d: np.ndarray | None = None

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN
        super().__post_init__()

    def __len__(self) -> int:
        return 1

@dataclass
class LazyMoleculeDatapoint(_DatapointMixin, _LazyMoleculeDatapointMixin):
    V_f: np.ndarray | None = None
    E_f: np.ndarray | None = None
    V_d: np.ndarray | None = None

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN
        super().__post_init__()

    def __len__(self) -> int:
        return 1

@dataclass
class MolAtomBondDatapoint(MoleculeDatapoint):
    E_d: np.ndarray | None = None
    atom_y: np.ndarray | None = None
    atom_gt_mask: np.ndarray | None = None
    atom_lt_mask: np.ndarray | None = None
    bond_y: np.ndarray | None = None
    bond_gt_mask: np.ndarray | None = None
    bond_lt_mask: np.ndarray | None = None
    atom_constraint: np.ndarray | None = None
    bond_constraint: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        NAN_TOKEN = 0
        if self.E_d is not None:
            self.E_d[np.isnan(self.E_d)] = NAN_TOKEN

    @classmethod
    def from_smi(cls, smi: str, *args, keep_h: bool=False, add_h: bool=False, ignore_stereo: bool=False, reorder_atoms: bool=True, **kwargs) -> MolAtomBondDatapoint:
        mol = make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms=reorder_atoms)
        kwargs['name'] = smi if 'name' not in kwargs else kwargs['name']
        return cls(mol, *args, **kwargs)

@dataclass
class _ReactionDatapointMixin:
    rct: Chem.Mol
    pdt: Chem.Mol

    @classmethod
    def from_smi(cls, rxn_or_smis: str | tuple[str, str], *args, keep_h: bool=False, add_h: bool=False, ignore_stereo: bool=False, **kwargs) -> _ReactionDatapointMixin:
        match rxn_or_smis:
            case str():
                rct_smi, agt_smi, pdt_smi = rxn_or_smis.split('>')
                rct_smi = f'{rct_smi}.{agt_smi}' if agt_smi else rct_smi
                name = rxn_or_smis
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
                name = '>>'.join(rxn_or_smis)
            case _:
                raise TypeError('Must provide either a reaction SMARTS string or a tuple of reactant and a product SMILES strings!')
        rct = make_mol(rct_smi, keep_h, add_h, ignore_stereo)
        pdt = make_mol(pdt_smi, keep_h, add_h, ignore_stereo)
        kwargs['name'] = name if 'name' not in kwargs else kwargs['name']
        return cls(rct, pdt, *args, **kwargs)

@dataclass
class ReactionDatapoint(_DatapointMixin, _ReactionDatapointMixin):

    def __post_init__(self):
        if self.rct is None:
            raise ValueError('Reactant cannot be `None`!')
        if self.pdt is None:
            raise ValueError('Product cannot be `None`!')
        return super().__post_init__()

    def __len__(self) -> int:
        return 2
