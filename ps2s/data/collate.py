from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence
import numpy as np
import torch
from torch import Tensor
from ps2s.data.datasets import CuikBatchedDatum, Datum, MolAtomBondDatum
from ps2s.data.molgraph import MolGraph
from ps2s.featurizers.molgraph.molecule import BatchCuikMolGraph

@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    mgs: InitVar[Sequence[MolGraph]]
    V: Tensor = field(init=False)
    E: Tensor = field(init=False)
    edge_index: Tensor = field(init=False)
    rev_edge_index: Tensor = field(init=False)
    batch: Tensor = field(init=False)
    dist_matrix: Tensor = field(init=False)
    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        dist_matrixs = []
        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            dist_matrixs.append(mg.distance_matrix)
            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]
        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        self.dist_matrix = [torch.tensor(m).long() for m in dist_matrixs]

    def __len__(self) -> int:
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)

class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph | BatchCuikMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
    smiles: str | None

def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks, smiles = zip(*batch)
    return TrainingBatch(BatchMolGraph(mgs), None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(), None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(), None if ys[0] is None else torch.from_numpy(np.array(ys)).float(), torch.tensor(weights, dtype=torch.float).unsqueeze(1), None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)), None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)), smiles)

def collate_cuik_batch(batch: CuikBatchedDatum) -> TrainingBatch:
    bmg, V_d, X_d, Y, weights, lt_mask, gt_mask = batch
    return TrainingBatch(bmg, None if V_d is None else torch.from_numpy(V_d).float(), None if X_d is None else torch.from_numpy(X_d).float(), None if Y is None else torch.from_numpy(Y).float(), torch.tensor(weights, dtype=torch.float).unsqueeze(1), None if lt_mask is None else torch.from_numpy(lt_mask), None if gt_mask is None else torch.from_numpy(gt_mask))

@dataclass(repr=False, eq=False, slots=True)
class BatchMolAtomBondGraph(BatchMolGraph):
    bond_batch: Tensor = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        super(BatchMolAtomBondGraph, self).__post_init__(mgs)
        bond_batch_indexes = []
        for i, mg in enumerate(mgs):
            bond_batch_indexes.append([i] * len(mg.E))
        self.bond_batch = torch.tensor(np.concatenate(bond_batch_indexes)).long()

    def to(self, device):
        super(BatchMolAtomBondGraph, self).to(device)
        self.bond_batch = self.bond_batch.to(device)

class MolAtomBondTrainingBatch(NamedTuple):
    bmg: BatchMolAtomBondGraph
    V_d: Tensor | None
    E_d: Tensor | None
    X_d: Tensor | None
    Ys: tuple[Tensor | None, Tensor | None, Tensor | None]
    w: tuple[Tensor | None, Tensor | None, Tensor | None]
    lt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    gt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    constraints: tuple[Tensor | None, Tensor | None]

def collate_mol_atom_bond_batch(batch: Iterable[MolAtomBondDatum]) -> MolAtomBondTrainingBatch:
    mgs, V_ds, E_ds, x_ds, yss, weights, lt_maskss, gt_maskss, constraintss = zip(*batch)
    weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    weights_tensors = []
    for ys in zip(*yss):
        if ys[0] is None:
            weights_tensors.append(None)
        elif ys[0].ndim == 1:
            weights_tensors.append(weights)
        else:
            repeats = torch.tensor([y.shape[0] for y in ys])
            weights_tensors.append(torch.repeat_interleave(weights, repeats, dim=0))
    if constraintss[0][0] is None and constraintss[0][1] is None:
        constraintss = None
    else:
        constraintss = [None if constraints[0] is None else torch.from_numpy(np.array(constraints)).float() for constraints in zip(*constraintss)]
    return MolAtomBondTrainingBatch(BatchMolAtomBondGraph(mgs), None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(), None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(), None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(), [None if ys[0] is None else torch.from_numpy(np.vstack(ys)).float() for ys in zip(*yss)], weights_tensors, [None if lt_masks[0] is None else torch.from_numpy(np.vstack(lt_masks)) for lt_masks in zip(*lt_maskss)], [None if gt_masks[0] is None else torch.from_numpy(np.vstack(gt_masks)) for gt_masks in zip(*gt_maskss)], constraintss)

class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None

def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]
    return MulticomponentTrainingBatch([tb.bmg for tb in tbs], [tb.V_d for tb in tbs], tbs[0].X_d, tbs[0].Y, tbs[0].w, tbs[0].lt_mask, tbs[0].gt_mask)
