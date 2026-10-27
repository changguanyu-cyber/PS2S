from abc import abstractmethod
from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
from ps2s.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from ps2s.data import BatchMolGraph
from ps2s.exceptions import InvalidShapeError
from ps2s.nn.message_passing.mixins import _AtomMessagePassingMixin, _BondMessagePassingMixin
from ps2s.nn.message_passing.proto import MessagePassing
from ps2s.nn.transforms import GraphTransform, ScaleTransform
from ps2s.nn.utils import Activation, get_activation_function

class _MessagePassingBase(MessagePassing, HyperparametersMixin):

    def __init__(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, bias: bool=False, depth: int=3, dropout: float=0.0, activation: str | nn.Module | Activation=Activation.TANH, undirected: bool=False, d_vd: int | None=None, V_d_transform: ScaleTransform | None=None, graph_transform: GraphTransform | None=None, gamma_s: float=0.45, gamma_b: float=0.45, alpha_init: float=0.5, convergence_tol: float=1e-06):
        super().__init__()
        ignore_list = ['V_d_transform', 'graph_transform']
        if isinstance(activation, nn.Module):
            ignore_list.append('activation')
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams['V_d_transform'] = V_d_transform
        self.hparams['graph_transform'] = graph_transform
        if isinstance(activation, nn.Module):
            self.hparams['activation'] = activation
        self.hparams['cls'] = self.__class__
        self.W_i, self.W_h, self.W_o, self.W_d = self.setup(d_v, d_e, d_h, d_vd, bias)
        self.W_s = nn.utils.parametrizations.spectral_norm(nn.Linear(d_h, d_h, bias=bias))
        self.W_b = nn.utils.parametrizations.spectral_norm(nn.Linear(d_h, d_h, bias=bias))
        self.W_x = nn.Linear(d_v, d_h, bias=bias)
        alpha_init = min(max(alpha_init, 1e-06), 1 - 1e-06)
        self.alpha_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(alpha_init)).item()))
        self.gamma_s = float(gamma_s)
        self.gamma_b = float(gamma_b)
        if self.gamma_s + self.gamma_b >= 1:
            raise ValueError('gamma_s + gamma_b must be smaller than 1.')
        self.convergence_tol = float(convergence_tol)
        self.depth = depth
        self.undirected = undirected
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.fixed_point_activation = torch.tanh
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_o.out_features

    @abstractmethod
    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, bias: bool=False) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]:
        pass

    @abstractmethod
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        pass

    @abstractmethod
    def message(self, H_t: Tensor, bmg: BatchMolGraph):
        pass

    def update(self, M_t, H_0):
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)
        return H_t

    def finalize(self, M: Tensor, V: Tensor, V_d: Tensor | None) -> Tensor:
        H = self.W_o(torch.cat((V, M), dim=1))
        H = self.tau(H)
        H = self.dropout(H)
        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H = self.W_d(torch.cat((H, V_d), dim=1))
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError('V_d', V_d.shape, [len(H), self.W_d.in_features - self.W_o.out_features])
        return H

    def normalized_adjacency_message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        source = bmg.edge_index[0]
        target = bmg.edge_index[1]
        ones = torch.ones(source.shape[0], dtype=H.dtype, device=H.device)
        degree = torch.zeros(len(bmg.V), dtype=H.dtype, device=H.device)
        degree.scatter_add_(0, target, ones)
        degree = degree.clamp_min(1.0)
        norm = degree[source].rsqrt() * degree[target].rsqrt()
        M = torch.zeros_like(H)
        M.index_add_(0, target, H[source] * norm.unsqueeze(1))
        return M

    def cdmp_update(self, H: Tensor, X: Tensor, bmg: BatchMolGraph) -> Tensor:
        alpha = torch.sigmoid(self.alpha_logit)
        A_H = self.normalized_adjacency_message(H, bmg)
        Z = self.gamma_s * self.W_s(H) + self.gamma_b * self.W_b(A_H) + self.W_x(X)
        H_next = (1 - alpha) * H + alpha * self.fixed_point_activation(Z)
        return H_next

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None) -> Tensor:
        bmg = self.graph_transform(bmg)
        X = bmg.V
        H = self.fixed_point_activation(self.W_x(X))
        steps = max(int(self.depth), 1)
        for _ in range(steps):
            H_next = self.cdmp_update(H, X, bmg)
            if self.convergence_tol > 0 and torch.mean(torch.abs(H_next - H)).detach().item() < self.convergence_tol:
                H = H_next
                break
            H = H_next
        return self.finalize(H, X, V_d)

class BondMessagePassing(_BondMessagePassingMixin, _MessagePassingBase):

    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, bias: bool=False):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h + d_e, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        return (W_i, W_h, W_o, W_d)

class AtomMessagePassing(_AtomMessagePassingMixin, _MessagePassingBase):

    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, bias: bool=False):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_h + d_e, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        return (W_i, W_h, W_o, W_d)
