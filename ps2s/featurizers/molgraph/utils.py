import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import math

def get_modified_distance_matrix(mol):
    if mol is None:
        raise ValueError('Invalid SMILES')
    dist_matrix = rdmolops.GetDistanceMatrix(mol)
    star_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(star_atoms) == 0:
        return dist_matrix
    connected_atoms = []
    for star in star_atoms:
        neighbors = [nbr.GetIdx() for nbr in star.GetNeighbors()]
        connected_atoms.extend(neighbors)
    connected_atoms = list(set(connected_atoms))
    if len(connected_atoms) >= 2:
        i, j = connected_atoms[:2]
        dist_matrix[i, j] = 1 + 2 * math.pi
        dist_matrix[j, i] = 1 + 2 * math.pi
    return dist_matrix
