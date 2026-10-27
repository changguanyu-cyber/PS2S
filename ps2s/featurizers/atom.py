from enum import auto
from typing import Sequence
import numpy as np
from rdkit.Chem.rdchem import Atom, HybridizationType
from ps2s.featurizers.base import VectorFeaturizer
from ps2s.utils.utils import EnumMapping

class MultiHotAtomFeaturizer(VectorFeaturizer[Atom]):

    def __init__(self, atomic_nums: Sequence[int], degrees: Sequence[int], formal_charges: Sequence[int], chiral_tags: Sequence[int], num_Hs: Sequence[int], hybridizations: Sequence[int]):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}
        self._subfeats: list[dict] = [self.atomic_nums, self.degrees, self.formal_charges, self.chiral_tags, self.num_Hs, self.hybridizations]
        subfeat_sizes = [1 + len(self.atomic_nums), 1 + len(self.degrees), 1 + len(self.formal_charges), 1 + len(self.chiral_tags), 1 + len(self.num_Hs), 1 + len(self.hybridizations), 1, 1]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)
        if a is None:
            return x
        feats = [a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(), int(a.GetChiralTag()), int(a.GetTotalNumHs()), a.GetHybridization()]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        return x

    def num_only(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))
        if a is None:
            return x
        i = self.atomic_nums.get(a.GetAtomicNum(), len(self.atomic_nums))
        x[i] = 1
        return x

    @classmethod
    def v1(cls, max_atomic_num: int=100):
        return cls(atomic_nums=list(range(1, max_atomic_num + 1)), degrees=list(range(6)), formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)), hybridizations=[HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2])

    @classmethod
    def v2(cls):
        return cls(atomic_nums=list(range(1, 37)) + [53], degrees=list(range(6)), formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)), hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP2D, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2])

    @classmethod
    def organic(cls):
        return cls(atomic_nums=[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53], degrees=list(range(6)), formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)), hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3])

class RIGRAtomFeaturizer(VectorFeaturizer[Atom]):

    def __init__(self, atomic_nums: Sequence[int] | None=None, degrees: Sequence[int] | None=None, num_Hs: Sequence[int] | None=None):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums or list(range(1, 37)) + [53])}
        self.degrees = {i: i for i in degrees or list(range(6))}
        self.num_Hs = {i: i for i in num_Hs or list(range(5))}
        self._subfeats: list[dict] = [self.atomic_nums, self.degrees, self.num_Hs]
        subfeat_sizes = [1 + len(self.atomic_nums), 1 + len(self.degrees), 1 + len(self.num_Hs), 1]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)
        if a is None:
            return x
        feats = [a.GetAtomicNum(), a.GetTotalDegree(), int(a.GetTotalNumHs())]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = 0.01 * a.GetMass()
        return x

    def num_only(self, a: Atom) -> np.ndarray:
        x = np.zeros(len(self))
        if a is None:
            return x
        i = self.atomic_nums.get(a.GetAtomicNum(), len(self.atomic_nums))
        x[i] = 1
        return x

class AtomFeatureMode(EnumMapping):
    V1 = auto()
    V2 = auto()
    ORGANIC = auto()
    RIGR = auto()

def get_multi_hot_atom_featurizer(mode: str | AtomFeatureMode) -> MultiHotAtomFeaturizer:
    match AtomFeatureMode.get(mode):
        case AtomFeatureMode.V1:
            return MultiHotAtomFeaturizer.v1()
        case AtomFeatureMode.V2:
            return MultiHotAtomFeaturizer.v2()
        case AtomFeatureMode.ORGANIC:
            return MultiHotAtomFeaturizer.organic()
        case AtomFeatureMode.RIGR:
            return RIGRAtomFeaturizer()
        case _:
            raise RuntimeError('unreachable code reached!')
