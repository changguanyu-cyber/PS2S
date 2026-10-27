from abc import abstractmethod
from torch import Tensor, nn
from ps2s.data import BatchMolGraph
from ps2s.nn.hparams import HasHParams

class MessagePassing(nn.Module, HasHParams):
    output_dim: int

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None) -> Tensor:
        pass

class MABMessagePassing(nn.Module, HasHParams):
    output_dims: tuple[int | None, int | None]

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None, E_d: Tensor | None=None) -> tuple[Tensor | None, Tensor | None]:
        pass
