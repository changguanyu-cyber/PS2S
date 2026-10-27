from __future__ import annotations
from enum import StrEnum
import os
from typing import Any, Callable, Iterable, Iterator, Type
import multiprocess
import numpy as np
import psutil
from rdkit import Chem
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

class EnumMapping(StrEnum):

    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name
        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {', '.join(cls.keys())}")

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())
import torch
import math

def dct_matrix(d: int, device=None, dtype=torch.float32):
    device = device or 'cpu'
    n = torch.arange(d, device=device, dtype=dtype)
    k = torch.arange(d, device=device, dtype=dtype).unsqueeze(1)
    D = torch.cos(math.pi / d * (n + 0.5) * k)
    D[0, :] = 1.0 / math.sqrt(d)
    D[1:, :] *= math.sqrt(2.0 / d)
    return D

def make_mol(smi: str, keep_h: bool=False, add_h: bool=False, ignore_stereo: bool=False, reorder_atoms: bool=False) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(smi, params)
    if mol is None:
        raise RuntimeError(f'SMILES {smi} is invalid! (RDKit returned None)')
    if add_h:
        mol = Chem.AddHs(mol)
    if ignore_stereo:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    if reorder_atoms:
        atom_map_numbers = tuple((atom.GetAtomMapNum() for atom in mol.GetAtoms()))
        new_order = np.argsort(atom_map_numbers).tolist()
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    return mol

def create_and_call_object(cls: Type, call_args: tuple=(), call_kwargs: dict=None, init_args: tuple=(), init_kwargs: dict=None) -> Any:
    if call_kwargs is None:
        call_kwargs = {}
    if init_kwargs is None:
        init_kwargs = {}
    return cls(*init_args, **init_kwargs)(*call_args, **call_kwargs)

def parallel_execute(exe_func: Callable, func_args: Iterable[tuple]=(), func_kwargs: Iterable[dict]=(), n_workers: int=0) -> list:
    func_args = list(func_args)
    func_kwargs = list(func_kwargs)
    if not func_kwargs:
        func_kwargs = [{}] * len(func_args)
    if not func_args:
        func_args = [()] * len(func_kwargs)
    combined = list(zip(func_args, func_kwargs))
    if n_workers >= 2:

        def wrapped_call(args, kwargs):
            return exe_func(*args, **kwargs)
        with multiprocess.Pool(n_workers) as p:
            results = p.starmap(wrapped_call, combined)
    else:
        results = [exe_func(*func_arg, **func_kwargs) for func_arg, func_kwargs in combined]
    return results

def pretty_shape(shape: Iterable[int]) -> str:
    return ' x '.join(map(str, shape))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return f'Memory usage: {memory_mb:.2f} MB'

def is_cuikmolmaker_available():
    try:
        import cuik_molmaker
        return True
    except ImportError:
        return False
