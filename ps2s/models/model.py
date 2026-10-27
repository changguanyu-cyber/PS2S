from __future__ import annotations
import torch
from pathlib import Path
import io
import logging
import traceback
from typing import Iterable, TypeAlias
import torch.nn.functional as F
import random
import os
from ps2s.conf import MAX_ATOM_NUM
from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim
import numpy as np
from ps2s.conf import LIGHTNING_26_COMPAT_ARGS
from ps2s.data import BatchMolGraph, MulticomponentTrainingBatch, TrainingBatch
from ps2s.nn import Aggregation, PS2SMetric, MessagePassing, Predictor, Decoder
from ps2s.nn.transforms import ScaleTransform
from ps2s.schedulers import build_NoamLike_LRSched
from ps2s.utils.registry import Factory
from ps2s.nn.decoder import Decoder
logger = logging.getLogger(__name__)
BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath, FragmentOnBonds
from rdkit import Chem
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def clean_smiles(s):
    if s is None:
        return None
    s = s.replace('<pad>', '').replace('<bos>', '').replace('<eos>', '').strip()
    if s == '':
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def pad_to_same_length(a, b):
    max_len = max(len(a), len(b))
    a = a + [0] * (max_len - len(a))
    b = b + [0] * (max_len - len(b))
    return (a, b)

def atomic_num_sorted_degrees(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_info = [(atom.GetAtomicNum(), atom.GetDegree()) for atom in mol.GetAtoms()]
    atom_info_sorted = sorted(atom_info, key=lambda x: x[0])
    degrees = [deg for _, deg in atom_info_sorted]
    return degrees

def smiles_to_sorted_atomic_numbers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atomic_nums.sort()
    return atomic_nums

def trimerize_canonical_index(monomer_smiles: str) -> str:

    def connect(left_mol, right_mol):
        stars_l = [a for a in left_mol.GetAtoms() if a.GetSymbol() == '*']
        stars_r = [a for a in right_mol.GetAtoms() if a.GetSymbol() == '*']
        if len(stars_l) != 2 or len(stars_r) != 2:
            raise ValueError('Each molecule must contain exactly two [*] atoms.')
        star_l = max(stars_l, key=lambda a: a.GetIdx())
        star_r = min(stars_r, key=lambda a: a.GetIdx())
        nbr_l = star_l.GetNeighbors()[0]
        nbr_r = star_r.GetNeighbors()[0]
        combo = Chem.CombineMols(left_mol, right_mol)
        emol = Chem.EditableMol(combo)
        offset = left_mol.GetNumAtoms()
        emol.AddBond(nbr_l.GetIdx(), nbr_r.GetIdx() + offset, Chem.BondType.SINGLE)
        for idx in sorted([star_l.GetIdx(), star_r.GetIdx() + offset], reverse=True):
            emol.RemoveAtom(idx)
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    mol1 = Chem.MolFromSmiles(monomer_smiles)
    mol2 = Chem.MolFromSmiles(monomer_smiles)
    mol3 = Chem.MolFromSmiles(monomer_smiles)
    if mol1 is None or mol2 is None or mol3 is None:
        raise ValueError('Failed to parse monomer SMILES.')
    dimer = connect(mol1, mol2)
    trimer = connect(dimer, mol3)
    return Chem.MolToSmiles(trimer, canonical=True)

def find_star_atoms(mol):
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(stars) != 2:
        raise ValueError('Exactly two * atoms are required.')
    return stars

def get_backbone_path(mol, s1, s2):
    path = GetShortestPath(mol, s1, s2)
    return [i for i in path if mol.GetAtomWithIdx(i).GetSymbol() != '*']

def monomer_backbone_length(smiles):
    mol = Chem.MolFromSmiles(smiles)
    s1, s2 = find_star_atoms(mol)
    return len(get_backbone_path(mol, s1, s2))

def fragment_backbone_length(frag):
    stars = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == '*']
    if len(stars) != 2:
        return None
    path = GetShortestPath(frag, stars[0], stars[1])
    return sum((1 for i in path if frag.GetAtomWithIdx(i).GetSymbol() != '*'))

def force_build_period(trimer, backbone, start, L):
    if start <= 0 or start + L >= len(backbone):
        return None
    bond_l = trimer.GetBondBetweenAtoms(backbone[start - 1], backbone[start])
    bond_r = trimer.GetBondBetweenAtoms(backbone[start + L - 1], backbone[start + L])
    frag_mol = FragmentOnBonds(trimer, [bond_l.GetIdx(), bond_r.GetIdx()], addDummies=True, dummyLabels=[(0, 0), (0, 0)])
    for f in Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False):
        if sum((a.GetSymbol() == '*' for a in f.GetAtoms())) != 2:
            continue
        if fragment_backbone_length(f) == L:
            Chem.SanitizeMol(f)
            return Chem.MolToSmiles(f, canonical=True)
    return None

def sliding_periods_from_monomer(monomer_smiles: str):
    trimer_smiles = trimerize_canonical_index(monomer_smiles)
    trimer = Chem.MolFromSmiles(trimer_smiles)
    s1, s2 = find_star_atoms(trimer)
    backbone = get_backbone_path(trimer, s1, s2)
    L = monomer_backbone_length(monomer_smiles)
    results = []
    unique_periods = set()
    for start in range(1, len(backbone) - L):
        p = force_build_period(trimer, backbone, start, L)
        results.append((start, p))
        if p is not None:
            unique_periods.add(p)
    return (trimer_smiles, results, unique_periods)

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 15)

    def forward(self, x):
        return self.fc(x)
from rdkit import Chem

def remove_star_and_connect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RWMol(mol)
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    if len(stars) != 2:
        raise ValueError('need exactly two *')
    neigh1 = mol.GetAtomWithIdx(stars[0]).GetNeighbors()[0].GetIdx()
    neigh2 = mol.GetAtomWithIdx(stars[1]).GetNeighbors()[0].GetIdx()
    if neigh1 != neigh2:
        if mol.GetBondBetweenAtoms(neigh1, neigh2) is None:
            mol.AddBond(neigh1, neigh2, Chem.rdchem.BondType.SINGLE)
    for idx in sorted(stars, reverse=True):
        mol.RemoveAtom(idx)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
smiles = '*C(=O)N(CCCCCCCC)c1ccc(*)cc1'
print(remove_star_and_connect(smiles))

class MPNN(pl.LightningModule):

    def __init__(self, message_passing: MessagePassing, agg: Aggregation, predictor: Predictor, decoder: Decoder, isencoder: bool=True, isdecoder: bool=False, batch_norm: bool=False, metrics: Iterable[PS2SMetric] | None=None, warmup_epochs: int=2, init_lr: float=0.0001, max_lr: float=0.001, final_lr: float=0.0001, X_d_transform: ScaleTransform | None=None):
        super().__init__()
        self._cached_features: list[Tensor] = []
        self._cached_atom_features: list[Tensor] = []
        self._cached_batch: list[Tensor] = []
        self.save_hyperparameters(ignore=['X_d_transform', 'message_passing', 'agg', 'predictor'])
        self.hparams['X_d_transform'] = X_d_transform
        self.hparams.update({'message_passing': message_passing.hparams, 'agg': agg.hparams, 'predictor': predictor.hparams})
        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.decoder = decoder
        self.isdecoder = isdecoder
        self.isencoder = isencoder
        self.passlayer = nn.Linear(1, 15)
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()
        self.metrics = nn.ModuleList([*metrics, self.criterion.clone()]) if metrics else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> PS2SMetric:
        return self.predictor.criterion

    def fingerprint(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        if self.trainer is not None and self.trainer.state.stage.name == 'PREDICTING':
            self._cached_atom_features.append(H_v.detach().cpu())
            self._cached_batch.append(bmg.batch.detach().cpu())
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def encoding(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None, i: int=-1) -> Tensor:
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None) -> Tensor:
        if self.trainer is not None and self.trainer.state.stage.name == 'PREDICTING' and (self.isdecoder == False) and (self.isencoder == True):
            Z = self.fingerprint(bmg, V_d, X_d)
            self._cached_features.append(Z.detach().cpu())
            property = self.predictor(Z)
            return property
        elif self.trainer is not None and self.trainer.state.stage.name == 'PREDICTING' and (self.isdecoder == True) and (self.isencoder == False):
            input_ids = bmg['input_ids']
            attention_mask = bmg['input_ids']
            labels = bmg['input_ids']
            return self.decoder(input_ids, attention_mask, labels)
        else:
            Z = self.fingerprint(bmg, V_d, X_d)
            return self.predictor(Z)

    def training_step(self, batch: BatchType, batch_idx):
        if self.isencoder == True:
            batch_size = self.get_batch_size(batch)
            if self.isdecoder == False:
                bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                preds = self.predictor.train_step(Z)
                l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
                self.log('train_loss', self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
                return l
            else:
                bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, smiles = batch
                period_smiles = []
                connected_smiles = []
                for s in smiles:
                    try:
                        new_s = remove_star_and_connect(s)
                        if new_s is None:
                            new_s = s
                    except Exception as e:
                        print(f'[Warning] connect failed on {s}: {e}')
                        new_s = s
                    connected_smiles.append(new_s)
                Z = self.fingerprint(bmg, V_d, X_d)
                property = self.predictor.train_step(Z)
                token_input = self.passlayer(property)
                out, gened_smiles = self.decoder(token_input, smiles, return_smiles=True)
                clean_gened_smiles = []
                for s in gened_smiles:
                    try:
                        clean_s = clean_smiles(s)
                        if clean_s is None:
                            clean_s = ''
                    except Exception:
                        clean_s = ''
                    clean_gened_smiles.append(clean_s)
                gened_smiles = clean_gened_smiles
                structure_losses = []
                for s1, s2 in zip(gened_smiles, connected_smiles):
                    try:
                        atomnum1 = smiles_to_sorted_atomic_numbers(s1)
                        atomnum2 = smiles_to_sorted_atomic_numbers(s2)
                        deg1 = atomic_num_sorted_degrees(s1)
                        deg2 = atomic_num_sorted_degrees(s2)
                        if deg1 is None or deg2 is None:
                            continue
                        if atomnum1 is None or atomnum2 is None:
                            continue
                        atomnum1, atomnum2 = pad_to_same_length(atomnum1, atomnum2)
                        deg1, deg2 = pad_to_same_length(deg1, deg2)
                        a1 = torch.tensor(atomnum1, dtype=torch.float32, device=Z.device)
                        a2 = torch.tensor(atomnum2, dtype=torch.float32, device=Z.device)
                        t1 = torch.tensor(deg1, dtype=torch.float32, device=Z.device)
                        t2 = torch.tensor(deg2, dtype=torch.float32, device=Z.device)
                        l1 = F.mse_loss(a1, a2)
                        l2 = F.mse_loss(t1, t2)
                        l = l1 + l2
                        structure_losses.append(l)
                    except Exception as e:
                        print(f'[Warning] structure loss failed: {e}')
                        continue
                if len(structure_losses) > 0:
                    structure_loss = torch.stack(structure_losses).sum()
                else:
                    structure_loss = torch.tensor(0.0, device=Z.device)
                loss = out.loss + 0.0005 * structure_loss
                self.log('train_loss', loss, batch_size=batch_size, prog_bar=True, on_epoch=True, sync_dist=True)
                return loss
        else:
            input_ids = batch['input_ids']
            attention_mask = batch['input_ids']
            labels = batch['input_ids']
            outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log('train_loss', loss, on_epoch=True, sync_dist=True)
            return loss

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def validation_step(self, batch: BatchType, batch_idx: int=0):
        self._evaluate_batch(batch, 'val')
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, smiles = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log('val_loss', self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int=0):
        self._evaluate_batch(batch, 'test')

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, smiles = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)
        if self.predictor.n_targets > 1:
            preds = preds[..., 0]
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f'{label}/{m.alias}', m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int=0) -> Tensor:
        if self.isencoder == False:
            V_d = []
            X_d = []
            return self(batch, V_d, X_d)
        else:
            bmg, V_d, X_d, *_ = batch
            return self(bmg, V_d, X_d)

    def save_features(self, save_path=None):
        if not self._cached_features:
            print('[WARN] No features cached!')
            return
        features = torch.cat(self._cached_features, dim=0).cpu().numpy()
        if save_path is None:
            save_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt/predict_features.npy'
        np.save(save_path, features)
        print(f'[INFO] Saved features to {save_path}')
        self._cached_features.clear()

    def save_atom_features(self, save_path=None):
        if not self._cached_atom_features:
            print('[WARN] No features cached!')
            return
        atom_features = torch.cat(self._cached_atom_features, dim=0).cpu().numpy()
        if save_path is None:
            save_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt/predict_atom_features.npy'
        np.save(save_path, atom_features)
        print(f'[INFO] Saved features to {save_path}')
        self._cached_atom_features.clear()

    def save_batch(self, save_path=None):
        if not self._cached_batch:
            print('[WARN] No features cached!')
            return
        atom_batch = torch.cat(self._cached_batch, dim=0).cpu().numpy()
        if save_path is None:
            save_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt/predict_batch.npy'
        np.save(save_path, atom_batch)
        print(f'[INFO] Saved features to {save_path}')
        self._cached_atom_features.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning('For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs.')
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch
        lr_sched = build_NoamLike_LRSched(opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr)
        lr_sched_config = {'scheduler': lr_sched, 'interval': 'step'}
        return {'optimizer': opt, 'lr_scheduler': lr_sched_config}

    def get_batch_size(self, batch):
        if isinstance(batch, dict):
            return len(batch['input_smiles'])
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(f"{traceback.format_exc()}\nModel loading failed (full stacktrace above)! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'ps2s convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint.")
        try:
            hparams = d['hyper_parameters']
            state_dict = d['state_dict']
        except KeyError:
            raise KeyError(f'Could not find hyper parameters and/or state dict in {path}.')
        if hparams['metrics'] is not None:
            hparams['metrics'] = [cls._rebuild_metric(metric) if not hasattr(metric, '_defaults') or (not torch.cuda.is_available() and metric.device.type != 'cpu') else metric for metric in hparams['metrics']]
        if hparams['predictor']['criterion'] is not None:
            metric = hparams['predictor']['criterion']
            if not hasattr(metric, '_defaults') or (not torch.cuda.is_available() and metric.device.type != 'cpu'):
                hparams['predictor']['criterion'] = cls._rebuild_metric(metric)
        submodules |= {key: hparams[key].pop('cls')(**hparams[key]) for key in ('message_passing', 'agg', 'predictor') if key not in submodules}
        return (submodules, state_dict, hparams)

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if 'metrics.0.task_weights' not in state_dict:
            metrics = hparams['metrics']
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f'metrics.{i_metric}.task_weights'] = torch.tensor([[1.0]])
            state_dict[f'metrics.{i_metric + 1}.task_weights'] = state_dict['predictor.criterion.task_weights']
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)
        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)
        return model

class MPNN_pretrain(pl.LightningModule):

    def __init__(self, message_passing: MessagePassing, agg: Aggregation, predictor: Predictor, batch_norm: bool=False, metrics: Iterable[PS2SMetric] | None=None, warmup_epochs: int=2, init_lr: float=0.0001, max_lr: float=0.001, final_lr: float=0.0001, X_d_transform: ScaleTransform | None=None):
        super().__init__()
        self._cached_features: list[Tensor] = []
        self.save_hyperparameters(ignore=['X_d_transform', 'message_passing', 'agg', 'predictor'])
        self.hparams['X_d_transform'] = X_d_transform
        self.hparams.update({'message_passing': message_passing.hparams, 'agg': agg.hparams, 'predictor': predictor.hparams})
        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()
        self.metrics = nn.ModuleList([*metrics, self.criterion.clone()]) if metrics else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> PS2SMetric:
        return self.predictor.criterion

    def fingerprint(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def encoding(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None, i: int=-1) -> Tensor:
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None=None, X_d: Tensor | None=None) -> Tensor:
        Z = self.fingerprint(bmg, V_d, X_d)
        if self.trainer is not None and self.trainer.state.stage.name == 'PREDICTING':
            self._cached_features.append(Z.detach().cpu())
        return self.predictor(Z)

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        self.log('train_loss', self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def validation_step(self, batch: BatchType, batch_idx: int=0):
        self._evaluate_batch(batch, 'val')
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log('val_loss', self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int=0):
        self._evaluate_batch(batch, 'test')

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)
        if self.predictor.n_targets > 1:
            preds = preds[..., 0]
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f'{label}/{m.alias}', m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int=0) -> Tensor:
        bmg, V_d, X_d, *_ = batch
        return self(bmg, V_d, X_d)

    def save_features(self, save_path=None):
        if not self._cached_features:
            print('[WARN] No features cached!')
            return
        features = torch.cat(self._cached_features, dim=0).cpu().numpy()
        if save_path is None:
            save_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt/predict_features.npy'
        np.save(save_path, features)
        print(f'[INFO] Saved features to {save_path}')
        self._cached_features.clear()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning('For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs.')
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch
        lr_sched = build_NoamLike_LRSched(opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr)
        lr_sched_config = {'scheduler': lr_sched, 'interval': 'step'}
        return {'optimizer': opt, 'lr_scheduler': lr_sched_config}

    def get_batch_size(self, batch: TrainingBatch) -> int:
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(f"{traceback.format_exc()}\nModel loading failed (full stacktrace above)! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'ps2s convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint.")
        try:
            hparams = d['hyper_parameters']
            state_dict = d['state_dict']
        except KeyError:
            raise KeyError(f'Could not find hyper parameters and/or state dict in {path}.')
        if hparams['metrics'] is not None:
            hparams['metrics'] = [cls._rebuild_metric(metric) if not hasattr(metric, '_defaults') or (not torch.cuda.is_available() and metric.device.type != 'cpu') else metric for metric in hparams['metrics']]
        if hparams['predictor']['criterion'] is not None:
            metric = hparams['predictor']['criterion']
            if not hasattr(metric, '_defaults') or (not torch.cuda.is_available() and metric.device.type != 'cpu'):
                hparams['predictor']['criterion'] = cls._rebuild_metric(metric)
        submodules |= {key: hparams[key].pop('cls')(**hparams[key]) for key in ('message_passing', 'agg', 'predictor') if key not in submodules}
        return (submodules, state_dict, hparams)

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if 'metrics.0.task_weights' not in state_dict:
            metrics = hparams['metrics']
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f'metrics.{i_metric}.task_weights'] = torch.tensor([[1.0]])
            state_dict[f'metrics.{i_metric + 1}.task_weights'] = state_dict['predictor.criterion.task_weights']
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs) -> MPNN:
        submodules = {k: v for k, v in kwargs.items() if k in ['message_passing', 'agg', 'predictor']}
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)
        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d['state_dict'] = state_dict
        d['hyper_parameters'] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)
        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **LIGHTNING_26_COMPAT_ARGS, **kwargs)
    from pathlib import Path
    import torch

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=False, **submodules):
        from pathlib import Path
        import torch
        model_path = Path(model_path)
        checkpoint = torch.load(model_path / 'mpnn_backbone.pt', map_location=map_location)
        state_dict = checkpoint['state_dict']
        hparams = checkpoint['hyper_parameters']
        decoder_path = model_path / 'decoder'
        submodules['decoder'] = nn.Decoder(decoder_path)
        hparams.update(submodules)
        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)
        return model
