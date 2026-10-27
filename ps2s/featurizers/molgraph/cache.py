from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, Iterable
import numpy as np
from ps2s.data.molgraph import MolGraph
from ps2s.featurizers.base import Featurizer, S
from ps2s.utils import parallel_execute

class MolGraphCacheFacade(Sequence[MolGraph], Generic[S]):

    @abstractmethod
    def __init__(self, inputs: Iterable[S], V_fs: Iterable[np.ndarray], E_fs: Iterable[np.ndarray], featurizer: Featurizer[S, MolGraph]):
        pass

class MolGraphCache(MolGraphCacheFacade):

    def __init__(self, inputs: Iterable[S], V_fs: Iterable[np.ndarray | None], E_fs: Iterable[np.ndarray | None], featurizer: Featurizer[S, MolGraph], n_workers: int=0):
        self._mgs = parallel_execute(featurizer, zip(inputs, V_fs, E_fs), n_workers=n_workers)

    def __len__(self) -> int:
        return len(self._mgs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._mgs[index]

class MolGraphCacheOnTheFly(MolGraphCacheFacade):

    def __init__(self, inputs: Iterable[S], V_fs: Iterable[np.ndarray | None], E_fs: Iterable[np.ndarray | None], featurizer: Featurizer[S, MolGraph]):
        self._inputs = list(inputs)
        self._V_fs = list(V_fs)
        self._E_fs = list(E_fs)
        self._featurizer = featurizer

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._featurizer(self._inputs[index], self._V_fs[index], self._E_fs[index])
