import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath, FragmentOnBonds
from ps2sMob.ps2s import data, models, nn
from lightning import pytorch as pl
import torch

def trimerize_canonical_index(monomer_smiles: str) -> str:

    def connect(left_mol, right_mol):
        stars_l = [a for a in left_mol.GetAtoms() if a.GetSymbol() == '*']
        stars_r = [a for a in right_mol.GetAtoms() if a.GetSymbol() == '*']
        if len(stars_l) != 2 or len(stars_r) != 2:
            raise ValueError('Each molecule must contain exactly two [*] atoms.')
        star_l = max(stars_l, key=lambda a: a.GetIdx())
        star_r = min(stars_r, key=lambda a: a.GetIdx())
        nbr_l = star_l.GetNeighbors()[0]
        nbr_r = star_r.GetNeighbors()[0]
        combo = Chem.CombineMols(left_mol, right_mol)
        emol = Chem.EditableMol(combo)
        offset = left_mol.GetNumAtoms()
        emol.AddBond(nbr_l.GetIdx(), nbr_r.GetIdx() + offset, Chem.BondType.SINGLE)
        for idx in sorted([star_l.GetIdx(), star_r.GetIdx() + offset], reverse=True):
            emol.RemoveAtom(idx)
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    mol1 = Chem.MolFromSmiles(monomer_smiles)
    mol2 = Chem.MolFromSmiles(monomer_smiles)
    mol3 = Chem.MolFromSmiles(monomer_smiles)
    if mol1 is None or mol2 is None or mol3 is None:
        raise ValueError('Failed to parse monomer SMILES.')
    dimer = connect(mol1, mol2)
    trimer = connect(dimer, mol3)
    return Chem.MolToSmiles(trimer, canonical=True)

def find_star_atoms(mol):
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(stars) != 2:
        raise ValueError('Exactly two * atoms are required.')
    return stars

def get_backbone_path(mol, s1, s2):
    path = GetShortestPath(mol, s1, s2)
    return [i for i in path if mol.GetAtomWithIdx(i).GetSymbol() != '*']

def monomer_backbone_length(smiles):
    mol = Chem.MolFromSmiles(smiles)
    s1, s2 = find_star_atoms(mol)
    return len(get_backbone_path(mol, s1, s2))

def fragment_backbone_length(frag):
    stars = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == '*']
    if len(stars) != 2:
        return None
    path = GetShortestPath(frag, stars[0], stars[1])
    return sum((1 for i in path if frag.GetAtomWithIdx(i).GetSymbol() != '*'))

def force_build_period(trimer, backbone, start, L):
    if start <= 0 or start + L >= len(backbone):
        return None
    bond_l = trimer.GetBondBetweenAtoms(backbone[start - 1], backbone[start])
    bond_r = trimer.GetBondBetweenAtoms(backbone[start + L - 1], backbone[start + L])
    frag_mol = FragmentOnBonds(trimer, [bond_l.GetIdx(), bond_r.GetIdx()], addDummies=True, dummyLabels=[(0, 0), (0, 0)])
    for f in Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False):
        if sum((a.GetSymbol() == '*' for a in f.GetAtoms())) != 2:
            continue
        if fragment_backbone_length(f) == L:
            Chem.SanitizeMol(f)
            return Chem.MolToSmiles(f, canonical=True)
    return None

def sliding_periods_from_monomer(monomer_smiles: str):
    trimer_smiles = trimerize_canonical_index(monomer_smiles)
    trimer = Chem.MolFromSmiles(trimer_smiles)
    s1, s2 = find_star_atoms(trimer)
    backbone = get_backbone_path(trimer, s1, s2)
    L = monomer_backbone_length(monomer_smiles)
    results = []
    unique_periods = set()
    for start in range(1, len(backbone) - L):
        p = force_build_period(trimer, backbone, start, L)
        results.append((start, p))
        if p is not None:
            unique_periods.add(p)
    return (trimer_smiles, results, unique_periods)

def safe_datapoint_from_smi(smi, y):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return data.MoleculeDatapoint.from_smi(smi, y)
    except Exception:
        return None
csv_path = './data/Tm.csv'
df = pd.read_csv(csv_path)
smiles_all = df['Smiles'].tolist()
ys_dummy = np.zeros((len(smiles_all), 1))
datapoints = []
valid_smiles = []
for smi, y in zip(smiles_all, ys_dummy):
    dp = safe_datapoint_from_smi(smi, y)
    if dp is not None:
        datapoints.append(dp)
        valid_smiles.append(smi)
    else:
        print(f'[SKIP invalid SMILES] {smi}')
print(f'Valid SMILES: {len(valid_smiles)} / {len(smiles_all)}')
dset = data.MoleculeDataset(datapoints)
dataloader = data.build_dataloader(dset, shuffle=False)
loaded_model = Path('<PROJECT_ROOT>/ps2s-anonymous/ckpt/finetune_depth12_Tm.pt')
model = models.MPNN.load_from_file(loaded_model, strict=False)
model.isdecoder = False
trainer = pl.Trainer(devices=[0], logger=False, enable_checkpointing=False)
_ = trainer.predict(model, dataloader)
feature_npy_path = '<PROJECT_ROOT>/ps2s-anonymous/data/predict_features.npy'
atom_feature_npy_path = '<PROJECT_ROOT>/ps2s-anonymous/data/predict_atom_features.npy'
atom_batch = '<PROJECT_ROOT>/ps2s-anonymous/data/predict_batch.npy'
model.save_features(feature_npy_path)
model.save_atom_features(atom_feature_npy_path)
model.save_batch(atom_batch)
features = np.load(feature_npy_path)
print('features shape:', features.shape)
atom_features = np.load(atom_feature_npy_path)
print('atom_features shape:', atom_features.shape)
atom_batch = np.load(atom_batch)
print('atom_features shape:', atom_batch.shape)
print(len(valid_smiles))
molecule_atom_features = []
for i in range(len(valid_smiles)):
    idx = np.where(atom_batch == i)[0]
    molecule_atom_features.append(atom_features[idx])
print(len(molecule_atom_features))
np.savez('<PROJECT_ROOT>/ps2s-anonymous/data/smiles_atom_features_Tm.npz', smiles=np.array(valid_smiles), atom_features=np.array(molecule_atom_features, dtype=object))
print('Saved smiles_atom_features.npz.')
print(valid_smiles[0])
print(f'Number of molecules: {len(valid_smiles)}, example molecule atom count: {molecule_atom_features[0].shape[0]}')
