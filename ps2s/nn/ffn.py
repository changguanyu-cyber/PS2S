from abc import abstractmethod
from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
from ps2s.conf import DEFAULT_HIDDEN_DIM
from ps2s.nn.hparams import HasHParams
from ps2s.nn.utils import get_activation_function

class FFN(nn.Module):
    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass

class MLP(nn.Sequential, FFN):

    @classmethod
    def build(cls, input_dim: int, output_dim: int, hidden_dim: int=300, n_layers: int=1, dropout: float=0.0, activation: str | nn.Module='relu'):
        dropout = nn.Dropout(dropout)
        act = get_activation_function(activation)
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        blocks = [nn.Sequential(nn.Linear(dims[0], dims[1]))]
        if len(dims) > 2:
            blocks.extend([nn.Sequential(act, dropout, nn.Linear(d1, d2)) for d1, d2 in zip(dims[1:-1], dims[2:])])
        return cls(*blocks)

    @property
    def input_dim(self) -> int:
        return self[0][-1].in_features

    @property
    def output_dim(self) -> int:
        return self[-1][-1].out_features

class ConstrainerFFN(nn.Module, HasHParams, HyperparametersMixin):

    def __init__(self, n_constraints: int=1, fp_dim: int=DEFAULT_HIDDEN_DIM, hidden_dim: int=300, n_layers: int=1, dropout: float=0.0, activation: str='relu'):
        super().__init__()
        self.save_hyperparameters()
        self.hparams['cls'] = self.__class__
        self.ffn = MLP.build(fp_dim, n_constraints, hidden_dim, n_layers, dropout, activation)

    def forward(self, fp: Tensor, preds: Tensor, batch: Tensor, constraints: Tensor) -> Tensor:
        k = self.ffn(fp)
        expk = k.exp()
        n_mols = constraints.shape[0]
        index_torch = batch.unsqueeze(1).repeat(1, k.shape[1])
        per_mol_sum_expk = torch.zeros(n_mols, expk.shape[1], dtype=expk.dtype, device=expk.device).scatter_reduce_(0, index_torch, expk, reduce='sum', include_self=False)
        by_atom_or_bond_sum_expk = per_mol_sum_expk[batch]
        w = expk / by_atom_or_bond_sum_expk
        index_torch = batch.unsqueeze(1).repeat(1, preds.shape[1])
        per_mol_preds = torch.zeros(n_mols, preds.shape[1], dtype=preds.dtype, device=preds.device).scatter_reduce_(0, index_torch, preds, reduce='sum', include_self=False)
        pred_has_constraint = ~torch.isnan(constraints)[0]
        deviation = constraints[:, pred_has_constraint] - per_mol_preds[:, pred_has_constraint]
        corrections = w * deviation[batch]
        cor_shape_preds = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        cor_shape_preds[:, pred_has_constraint] = corrections
        return preds + cor_shape_preds
