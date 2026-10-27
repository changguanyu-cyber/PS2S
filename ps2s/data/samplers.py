from itertools import chain
from typing import Iterator, Optional
import numpy as np
from torch.utils.data import Sampler

class SeededSampler(Sampler):

    def __init__(self, N: int, seed: int):
        if seed is None:
            raise ValueError("arg 'seed' was `None`! A SeededSampler must be seeded!")
        self.idxs = np.arange(N)
        self.rg = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        self.rg.shuffle(self.idxs)
        return iter(self.idxs)

    def __len__(self) -> int:
        return len(self.idxs)

class ClassBalanceSampler(Sampler):

    def __init__(self, Y: np.ndarray, seed: Optional[int]=None, shuffle: bool=False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)
        idxs = np.arange(len(Y))
        actives = Y.any(1)
        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]
        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)
        return chain(*zip(self.pos_idxs, self.neg_idxs))

    def __len__(self) -> int:
        return self.length
