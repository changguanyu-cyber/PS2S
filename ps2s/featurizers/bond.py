from typing import Sequence
import numpy as np
from rdkit.Chem.rdchem import Bond, BondType
from ps2s.featurizers.base import VectorFeaturizer

class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):

    def __init__(self, bond_types: Sequence[BondType] | None=None, stereos: Sequence[int] | None=None):
        self.bond_types = bond_types or [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
        self.stereo = stereos or range(6)

    def __len__(self):
        return 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            return x
        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1
        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2
        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1
        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        n = len(xs)
        return (xs.index(x) if x in xs else n, n + 1)

class RIGRBondFeaturizer(VectorFeaturizer[Bond]):

    def __len__(self):
        return 2

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            return x
        x[1] = int(b.IsInRing())
        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        n = len(xs)
        return (xs.index(x) if x in xs else n, n + 1)
