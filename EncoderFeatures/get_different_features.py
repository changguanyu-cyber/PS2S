import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath, FragmentOnBonds
from ps2s import data, models, nn
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
csv_path = '<PROJECT_ROOT>/ps2s/data/PSMILES_Tg_GREA.csv'
df = pd.read_csv(csv_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
smiles_list = df['PSMILES'].tolist()
all_period_smiles = []
for smi in smiles_list:
    _, _, periods = sliding_periods_from_monomer(smi)
    periods = list(periods)
    while len(periods) < 5:
        periods.append(periods[-1] if periods else smi)
    all_period_smiles.append(periods[:5])
flat_period_smiles = [s for p in all_period_smiles for s in p]
ys_dummy = np.zeros((len(flat_period_smiles), 1))
dset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(flat_period_smiles, ys_dummy)])
dataloader = data.build_dataloader(dset, shuffle=False)
loaded_model = Path('<PROJECT_ROOT>/ps2s-anonymous/ckpt/finetune_depth12_Tm.pt')
model = models.MPNN.load_from_file(loaded_model, strict=False)
model.isdecoder = False
trainer = pl.Trainer(devices=[0], logger=False, enable_checkpointing=False, max_epochs=1)
feature_npy_path = '<PROJECT_ROOT>/ps2s-anonymous/monomer_period_features/predict_features.npy'
predictions = trainer.predict(model, dataloader)
model.save_features(feature_npy_path)
features = np.load(feature_npy_path)
feature_dim = features.shape[1]
n_smiles = len(smiles_list)
features_reshaped = features.reshape(n_smiles, 5, feature_dim)
np.save('<PROJECT_ROOT>/ps2s-anonymous/ckpt/predict_features_reshaped.npy', features_reshaped)
print('Feature generation completed, shape:', features_reshaped.shape)
