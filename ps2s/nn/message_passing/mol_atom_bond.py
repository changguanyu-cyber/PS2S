from abc import abstractmethod
from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
from ps2s.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from ps2s.data import BatchMolGraph
from ps2s.exceptions import InvalidShapeError
from ps2s.nn.message_passing.mixins import _AtomMessagePassingMixin, _BondMessagePassingMixin
from ps2s.nn.message_passing.proto import MABMessagePassing
from ps2s.nn.transforms import GraphTransform, ScaleTransform
from ps2s.nn.utils import Activation, get_activation_function

class _MABMessagePassingBase(MABMessagePassing, HyperparametersMixin):

    def __init__(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, bias: bool=False, depth: int=3, dropout: float=0.0, activation: str | Activation=Activation.RELU, undirected: bool=False, d_vd: int | None=None, d_ed: int | None=None, V_d_transform: ScaleTransform | None=None, E_d_transform: ScaleTransform | None=None, graph_transform: GraphTransform | None=None, return_vertex_embeddings: bool=True, return_edge_embeddings: bool=True):
        super().__init__()
        self.save_hyperparameters(ignore=['V_d_transform', 'E_d_transform', 'graph_transform'])
        self.hparams['V_d_transform'] = V_d_transform
        self.hparams['E_d_transform'] = E_d_transform
        self.hparams['graph_transform'] = graph_transform
        self.hparams['cls'] = self.__class__
        self.return_vertex_embeddings = return_vertex_embeddings
        self.return_edge_embeddings = return_edge_embeddings
        self.W_i, self.W_h, self.W_vo, self.W_vd, self.W_eo, self.W_ed = self.setup(d_v, d_e, d_h, d_vd, d_ed, bias)
        self.depth = depth
        self.undirected = undirected
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.E_d_transform = E_d_transform if E_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dims(self) -> tuple[int | None, int | None]:
        return (None if not self.return_vertex_embeddings else self.W_vd.out_features if self.W_vd is not None else self.W_vo.out_features, None if not self.return_edge_embeddings else self.W_ed.out_features if self.W_ed is not None else self.W_eo.out_features)

    @abstractmethod
    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, d_ed: int | None=None, bias: bool=False) -> tuple[nn.Module, nn.Module, nn.Module | None, nn.Module | None, nn.Module | None, nn.Module | None]:
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

    def vertex_finalize(self, M: Tensor, V: Tensor, V_d: Tensor | None) -> Tensor:
        H = self.W_vo(torch.cat((V, M), dim=1))
        H = self.tau(H)
        H = self.dropout(H)
        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H = self.W_vd(torch.cat((H, V_d), dim=1))
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError('V_d', V_d.shape, [len(H), self.W_vd.in_features - self.W_vo.out_features])
        return H

    def edge_finalize(self, H: Tensor, E: Tensor, E_d: Tensor | None) -> Tensor:
        H = self.W_eo(torch.cat((E, H), dim=1))
        H = self.tau(H)
        H = self.dropout(H)
        if E_d is not None:
            E_d = self.E_d_transform(E_d)
            try:
                H = self.W_ed(torch.cat((H, E_d), dim=1))
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError('E_d', E_d.shape, [len(H), self.W_ed.in_features - self.W_eo.out_features])
        return H

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None, E_d: Tensor | None=None) -> tuple[Tensor | None, Tensor | None]:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)
        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2
            M = self.message(H, bmg)
            H = self.update(M, H_0)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce='sum', include_self=False)
        H_v = self.vertex_finalize(M, bmg.V, V_d) if self.return_vertex_embeddings else None
        H_e = self.edge_finalize(H, bmg.E, E_d) if self.return_edge_embeddings else None
        return (H_v, H_e)

class MABBondMessagePassing(_BondMessagePassingMixin, _MABMessagePassingBase):

    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, d_ed: int | None=None, bias: bool=False):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_vo = nn.Linear(d_v + d_h, d_h) if self.return_vertex_embeddings else None
        W_eo = nn.Linear(d_e + d_h, d_h) if self.return_edge_embeddings else None
        W_vd = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None
        return (W_i, W_h, W_vo, W_vd, W_eo, W_ed)

class MABAtomMessagePassing(_AtomMessagePassingMixin, _MABMessagePassingBase):

    def setup(self, d_v: int=DEFAULT_ATOM_FDIM, d_e: int=DEFAULT_BOND_FDIM, d_h: int=DEFAULT_HIDDEN_DIM, d_vd: int | None=None, d_ed: int | None=None, bias: bool=False):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_e + d_h, d_h, bias)
        W_vo = nn.Linear(d_v + d_h, d_h) if self.return_vertex_embeddings else None
        W_eo = nn.Linear(d_e + d_h, d_h) if self.return_edge_embeddings else None
        W_vd = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None
        return (W_i, W_h, W_vo, W_vd, W_eo, W_ed)
