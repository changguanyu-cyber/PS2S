from abc import abstractmethod
from numpy.typing import ArrayLike
import torch
from torch import Tensor
from torch.nn import functional as F
import torchmetrics
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat
from ps2s.utils.registry import ClassRegistry
__all__ = ['PS2SMetric', 'LossFunctionRegistry', 'MetricRegistry', 'MSE', 'MAE', 'RMSE', 'BoundedMixin', 'BoundedMSE', 'BoundedMAE', 'BoundedRMSE', 'BinaryAccuracy', 'BinaryAUPRC', 'BinaryAUROC', 'BinaryF1Score', 'BinaryMCCMetric', 'BoundedMAE', 'BoundedMSE', 'BoundedRMSE', 'MetricRegistry', 'MulticlassMCCMetric', 'R2Score', 'MVELoss', 'EvidentialLoss', 'BCELoss', 'CrossEntropyLoss', 'BinaryMCCLoss', 'BinaryMCCMetric', 'MulticlassMCCLoss', 'MulticlassMCCMetric', 'ClassificationMixin', 'BinaryAUROC', 'BinaryAUPRC', 'BinaryAccuracy', 'BinaryF1Score', 'DirichletLoss', 'SID', 'Wasserstein', 'QuantileLoss']

class PS2SMetric(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, task_weights: ArrayLike=1.0):
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer('task_weights', task_weights)
        self.add_state('total_loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor | None=None, weights: Tensor | None=None, lt_mask: Tensor | None=None, gt_mask: Tensor | None=None) -> None:
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = torch.ones(targets.shape[0], dtype=torch.float, device=targets.device) if weights is None else weights
        lt_mask = torch.zeros_like(targets, dtype=torch.bool) if lt_mask is None else lt_mask
        gt_mask = torch.zeros_like(targets, dtype=torch.bool) if gt_mask is None else gt_mask
        L = self._calc_unreduced_loss(preds, targets, mask, weights, lt_mask, gt_mask)
        L = L * weights.view(-1, 1) * self.task_weights * mask
        self.total_loss += L.sum()
        self.num_samples += mask.sum()

    def compute(self):
        return self.total_loss / self.num_samples

    @abstractmethod
    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        pass

    def extra_repr(self) -> str:
        return f'task_weights={self.task_weights.tolist()}'
LossFunctionRegistry = ClassRegistry[PS2SMetric]()
MetricRegistry = ClassRegistry[PS2SMetric]()

@LossFunctionRegistry.register('mse')
@MetricRegistry.register('mse')
class MSE(PS2SMetric):

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.mse_loss(preds, targets, reduction='none')

@MetricRegistry.register('mae')
@LossFunctionRegistry.register('mae')
class MAE(PS2SMetric):

    def _calc_unreduced_loss(self, preds, targets, *args) -> Tensor:
        return (preds - targets).abs()

@LossFunctionRegistry.register('rmse')
@MetricRegistry.register('rmse')
class RMSE(MSE):

    def compute(self):
        return (self.total_loss / self.num_samples).sqrt()

class BoundedMixin:

    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)
        return super()._calc_unreduced_loss(preds, targets, mask, weights)

@LossFunctionRegistry.register('bounded-mse')
@MetricRegistry.register('bounded-mse')
class BoundedMSE(BoundedMixin, MSE):
    pass

@LossFunctionRegistry.register('bounded-mae')
@MetricRegistry.register('bounded-mae')
class BoundedMAE(BoundedMixin, MAE):
    pass

@LossFunctionRegistry.register('bounded-rmse')
@MetricRegistry.register('bounded-rmse')
class BoundedRMSE(BoundedMixin, RMSE):
    pass

@MetricRegistry.register('r2')
class R2Score(torchmetrics.R2Score):

    def __init__(self, task_weights: ArrayLike=1.0, **kwargs):
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer('task_weights', task_weights)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask])

@LossFunctionRegistry.register('mve')
class MVELoss(PS2SMetric):

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, var = torch.unbind(preds, dim=-1)
        L_sos = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2
        return L_sos + L_kl

@LossFunctionRegistry.register('evidential')
class EvidentialLoss(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0, v_kl: float=0.2, eps: float=1e-08):
        super().__init__(task_weights)
        self.v_kl = v_kl
        self.eps = eps

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, v, alpha, beta = torch.unbind(preds, dim=-1)
        residuals = targets - mean
        twoBlambda = 2 * beta * (1 + v)
        L_nll = 0.5 * (torch.pi / v).log() - alpha * twoBlambda.log() + (alpha + 0.5) * torch.log(v * residuals ** 2 + twoBlambda) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        L_reg = (2 * v + alpha) * residuals.abs()
        return L_nll + self.v_kl * (L_reg - self.eps)

    def extra_repr(self) -> str:
        parent_repr = super().extra_repr()
        return parent_repr + f', v_kl={self.v_kl}, eps={self.eps}'

@LossFunctionRegistry.register('bce')
class BCELoss(PS2SMetric):

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='none')

@LossFunctionRegistry.register('ce')
class CrossEntropyLoss(PS2SMetric):

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()
        return F.cross_entropy(preds, targets, reduction='none')

@LossFunctionRegistry.register('binary-mcc')
class BinaryMCCLoss(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0):
        super().__init__(task_weights)
        self.add_state('TP', default=[], dist_reduce_fx='cat')
        self.add_state('FP', default=[], dist_reduce_fx='cat')
        self.add_state('TN', default=[], dist_reduce_fx='cat')
        self.add_state('FN', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor | None=None, weights: Tensor | None=None, *args):
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = torch.ones(targets.shape[0], dtype=torch.float, device=targets.device) if weights is None else weights
        if not (0 <= preds.min() and preds.max() <= 1):
            preds = preds.sigmoid()
        TP, FP, TN, FN = self._calc_unreduced_loss(preds, targets.long(), mask, weights, *args)
        self.TP += [TP]
        self.FP += [FP]
        self.TN += [TN]
        self.FN += [FN]

    def _calc_unreduced_loss(self, preds, targets, mask, weights, *args) -> Tensor:
        TP = (targets * preds * weights * mask).sum(0, keepdim=True)
        FP = ((1 - targets) * preds * weights * mask).sum(0, keepdim=True)
        TN = ((1 - targets) * (1 - preds) * weights * mask).sum(0, keepdim=True)
        FN = (targets * (1 - preds) * weights * mask).sum(0, keepdim=True)
        return (TP, FP, TN, FN)

    def compute(self):
        TP = dim_zero_cat(self.TP).sum(0)
        FP = dim_zero_cat(self.FP).sum(0)
        TN = dim_zero_cat(self.TN).sum(0)
        FN = dim_zero_cat(self.FN).sum(0)
        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-08).sqrt()
        MCC = MCC * self.task_weights
        return 1 - MCC.mean()

@MetricRegistry.register('binary-mcc')
class BinaryMCCMetric(BinaryMCCLoss):
    higher_is_better = True

    def compute(self):
        return 1 - super().compute()

@LossFunctionRegistry.register('multiclass-mcc')
class MulticlassMCCLoss(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0):
        super().__init__(task_weights)
        self.add_state('p', default=[], dist_reduce_fx='cat')
        self.add_state('t', default=[], dist_reduce_fx='cat')
        self.add_state('c', default=[], dist_reduce_fx='cat')
        self.add_state('s', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor | None=None, weights: Tensor | None=None, *args):
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = torch.ones((targets.shape[0], 1), dtype=torch.float, device=targets.device) if weights is None else weights.view(-1, 1)
        if not (0 <= preds.min() and preds.max() <= 1):
            preds = preds.softmax(2)
        p, t, c, s = self._calc_unreduced_loss(preds, targets.long(), mask, weights, *args)
        self.p += [p]
        self.t += [t]
        self.c += [c]
        self.s += [s]

    def _calc_unreduced_loss(self, preds, targets, mask, weights, *args) -> Tensor:
        device = preds.device
        C = preds.shape[2]
        bin_targets = torch.eye(C, device=device)[targets]
        bin_preds = torch.eye(C, device=device)[preds.argmax(-1)]
        masked_data_weights = weights.unsqueeze(2) * mask.unsqueeze(2)
        p = (bin_preds * masked_data_weights).sum(0, keepdims=True)
        t = (bin_targets * masked_data_weights).sum(0, keepdims=True)
        c = (bin_preds * bin_targets * masked_data_weights).sum(2).sum(0, keepdims=True)
        s = (preds * masked_data_weights).sum(2).sum(0, keepdims=True)
        return (p, t, c, s)

    def compute(self):
        p = dim_zero_cat(self.p).sum(0)
        t = dim_zero_cat(self.t).sum(0)
        c = dim_zero_cat(self.c).sum(0)
        s = dim_zero_cat(self.s).sum(0)
        s2 = s.square()
        cov_ytyp = c * s - torch.einsum('ij,ij->i', p, t)
        cov_ypyp = s2 - torch.einsum('ij,ij->i', p, p)
        cov_ytyt = s2 - torch.einsum('ij,ij->i', t, t)
        x = cov_ypyp * cov_ytyt
        MCC = torch.where(x == 0, torch.tensor(0.0), cov_ytyp / x.sqrt())
        MCC = MCC * self.task_weights
        return 1 - MCC.mean()

@MetricRegistry.register('multiclass-mcc')
class MulticlassMCCMetric(MulticlassMCCLoss):
    higher_is_better = True

    def compute(self):
        return 1 - super().compute()

class ClassificationMixin:

    def __init__(self, task_weights: ArrayLike=1.0, **kwargs):
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer('task_weights', task_weights)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask].long())

@MetricRegistry.register('roc')
class BinaryAUROC(ClassificationMixin, torchmetrics.classification.BinaryAUROC):
    pass

@MetricRegistry.register('prc')
class BinaryAUPRC(ClassificationMixin, torchmetrics.classification.BinaryPrecisionRecallCurve):

    def compute(self) -> Tensor:
        p, r, _ = super().compute()
        return auc(r, p)

@MetricRegistry.register('accuracy')
class BinaryAccuracy(ClassificationMixin, torchmetrics.classification.BinaryAccuracy):
    pass

@MetricRegistry.register('f1')
class BinaryF1Score(ClassificationMixin, torchmetrics.classification.BinaryF1Score):
    pass

@LossFunctionRegistry.register('dirichlet')
class DirichletLoss(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0, v_kl: float=0.2):
        super().__init__(task_weights)
        self.v_kl = v_kl

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        targets = torch.eye(preds.shape[2], device=preds.device)[targets.long()]
        S = preds.sum(-1, keepdim=True)
        p = preds / S
        A = (targets - p).square().sum(-1, keepdim=True)
        B = (p * (1 - p) / (S + 1)).sum(-1, keepdim=True)
        L_mse = A + B
        alpha = targets + (1 - targets) * preds
        beta = torch.ones_like(alpha)
        S_alpha = alpha.sum(-1, keepdim=True)
        S_beta = beta.sum(-1, keepdim=True)
        ln_alpha = S_alpha.lgamma() - alpha.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()
        dg0 = torch.digamma(alpha)
        dg1 = torch.digamma(S_alpha)
        L_kl = ln_alpha + ln_beta + torch.sum((alpha - beta) * (dg0 - dg1), -1, keepdim=True)
        return (L_mse + self.v_kl * L_kl).mean(-1)

    def extra_repr(self) -> str:
        return f'v_kl={self.v_kl}'

@LossFunctionRegistry.register('sid')
class SID(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0, threshold: float | None=None, **kwargs):
        super().__init__(task_weights, **kwargs)
        self.threshold = threshold

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)
        preds_norm = preds / (preds * mask).sum(1, keepdim=True)
        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)
        return (preds_norm / targets).log() * preds_norm + (targets / preds_norm).log() * targets

    def extra_repr(self) -> str:
        return f'threshold={self.threshold}'

@LossFunctionRegistry.register(['earthmovers', 'wasserstein'])
class Wasserstein(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0, threshold: float | None=None):
        super().__init__(task_weights)
        self.threshold = threshold

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)
        preds_norm = preds / (preds * mask).sum(1, keepdim=True)
        return (targets.cumsum(1) - preds_norm.cumsum(1)).abs()

    def extra_repr(self) -> str:
        return f'threshold={self.threshold}'

@LossFunctionRegistry.register(['quantile', 'pinball'])
class QuantileLoss(PS2SMetric):

    def __init__(self, task_weights: ArrayLike=1.0, alpha: float=0.1):
        super().__init__(task_weights)
        self.alpha = alpha
        bounds = torch.tensor([-1 / 2, 1 / 2]).view(-1, 1, 1)
        tau = torch.tensor([[alpha / 2, 1 - alpha / 2], [alpha / 2 - 1, -alpha / 2]]).view(2, 2, 1, 1)
        self.register_buffer('bounds', bounds)
        self.register_buffer('tau', tau)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        mean, interval = torch.unbind(preds, dim=-1)
        interval_bounds = self.bounds * interval
        pred_bounds = mean + interval_bounds
        error_bounds = targets - pred_bounds
        loss_bounds = (self.tau * error_bounds).amax(0)
        return loss_bounds.sum(0)

    def extra_repr(self) -> str:
        return f'alpha={self.alpha}'
