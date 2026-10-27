from abc import abstractmethod
from collections.abc import Sized
from typing import Generic, TypeVar
import numpy as np
from ps2s.data.molgraph import MolGraph
S = TypeVar('S')
T = TypeVar('T')

class Featurizer(Generic[S, T]):

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        pass

class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...

class GraphFeaturizer(Featurizer[S, MolGraph]):

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...
