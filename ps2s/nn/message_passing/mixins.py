import torch
from torch import Tensor
import torch.nn as nn
from ps2s.data import BatchMolGraph
import math

class _BondMessagePassingMixin:

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1))

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce='sum', include_self=False)[bmg.edge_index[0]]
        M_rev = H[bmg.rev_edge_index]
        return M_all - M_rev

class _AtomMessagePassingMixin:

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(bmg.V[bmg.edge_index[0]])

    def message(self, H: Tensor, bmg: BatchMolGraph, attention, omega, sinbias, W_q, W_k, W_v):
        H_e = torch.cat((H, bmg.E), dim=1)
        Q = self.W_q(H_e)
        K = self.W_k(H_e)
        V = self.W_v(H_e)
        attn_logits = attention(H_e).squeeze(-1)
        source = bmg.edge_index[0]
        target = bmg.edge_index[1]
        batch = bmg.batch
        attn = torch.zeros_like(attn_logits)
        attn = torch.scatter_reduce(attn, 0, target, torch.exp(attn_logits), reduce='sum', include_self=False)
        attn = torch.exp(attn_logits) / (attn[target] + 1e-12)
        H_weighted = H_e * attn[:, None]
        node_msg = torch.zeros(len(bmg.V), H_weighted.shape[1], device=H.device, dtype=H.dtype)
        node_msg.index_add_(0, target, H_weighted)
        return node_msg[bmg.edge_index[0]]
