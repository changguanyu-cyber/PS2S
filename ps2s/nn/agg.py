from abc import abstractmethod
import torch
from torch import Tensor, nn
from ps2s.nn.hparams import HasHParams
from ps2s.utils import ClassRegistry
__all__ = ['Aggregation', 'AggregationRegistry', 'MeanAggregation', 'SumAggregation', 'NormAggregation', 'AttentiveAggregation']

class Aggregation(nn.Module, HasHParams):

    def __init__(self, dim: int=0, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.hparams = {'dim': dim, 'cls': self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        pass
AggregationRegistry = ClassRegistry[Aggregation]()

@AggregationRegistry.register('mean')
class MeanAggregation(Aggregation):

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(self.dim, index_torch, H, reduce='mean', include_self=False)

@AggregationRegistry.register('sum')
class SumAggregation(Aggregation):

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(self.dim, index_torch, H, reduce='sum', include_self=False)

@AggregationRegistry.register('norm')
class NormAggregation(SumAggregation):

    def __init__(self, dim: int=0, *args, norm: float=100.0, **kwargs):
        super().__init__(dim, **kwargs)
        self.norm = norm
        self.hparams['norm'] = norm

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return super().forward(H, batch) / self.norm

class AttentiveAggregation(Aggregation):

    def __init__(self, dim: int=0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.hparams['output_size'] = output_size
        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        dim_size = batch.max().int() + 1
        attention_logits = self.W(H).exp()
        Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(self.dim, batch.unsqueeze(1), attention_logits, reduce='sum', include_self=False)
        alphas = attention_logits / Z[batch]
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(self.dim, index_torch, alphas * H, reduce='sum', include_self=False)
