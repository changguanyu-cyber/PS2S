from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import Tensor
from torchmetrics.regression import SpearmanCorrCoef
from ps2s.utils.registry import ClassRegistry
UncertaintyEvaluatorRegistry = ClassRegistry()

class RegressionEvaluator(ABC):

    @abstractmethod
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        pass

@UncertaintyEvaluatorRegistry.register('nll-regression')
class NLLRegressionEvaluator(RegressionEvaluator):

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            errors = preds_j - targets_j
            nll = (2 * torch.pi * uncs_j).log() / 2 + errors ** 2 / (2 * uncs_j)
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)

@UncertaintyEvaluatorRegistry.register('miscalibration_area')
class CalibrationAreaEvaluator(RegressionEvaluator):

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor, num_bins: int=100) -> Tensor:
        bins = torch.arange(1, num_bins)
        bin_scaling = torch.special.erfinv(bins / num_bins).view(-1, 1, 1) * np.sqrt(2)
        errors = torch.abs(preds - targets)
        uncs = torch.sqrt(uncs).unsqueeze(0)
        bin_unc = uncs * bin_scaling
        bin_count = bin_unc >= errors.unsqueeze(0)
        mask = mask.unsqueeze(0)
        observed_auc = (bin_count & mask).sum(1) / mask.sum(1)
        num_tasks = uncs.shape[-1]
        observed_auc = torch.cat([torch.zeros(1, num_tasks), observed_auc, torch.ones(1, num_tasks)]).T
        ideal_auc = torch.arange(num_bins + 1) / num_bins
        miscal_area = 1 / num_bins * (observed_auc - ideal_auc).abs().sum(dim=1)
        return miscal_area

@UncertaintyEvaluatorRegistry.register('ence')
class ExpectedNormalizedErrorEvaluator(RegressionEvaluator):

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor, num_bins: int=100) -> Tensor:
        masked_preds = preds * mask
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        errors = torch.abs(masked_preds - masked_targets)
        sort_idx = torch.argsort(masked_uncs, dim=0)
        sorted_uncs = torch.gather(masked_uncs, 0, sort_idx)
        sorted_errors = torch.gather(errors, 0, sort_idx)
        split_unc = torch.chunk(sorted_uncs, num_bins, dim=0)
        split_error = torch.chunk(sorted_errors, num_bins, dim=0)
        root_mean_vars = torch.sqrt(torch.stack([chunk.mean(0) for chunk in split_unc]))
        rmses = torch.sqrt(torch.stack([chunk.pow(2).mean(0) for chunk in split_error]))
        ence = torch.mean(torch.abs(root_mean_vars - rmses) / root_mean_vars, dim=0)
        return ence

@UncertaintyEvaluatorRegistry.register('spearman')
class SpearmanEvaluator(RegressionEvaluator):

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        spearman_coeffs = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            errs_j = (preds_j - targets_j).abs()
            spearman = SpearmanCorrCoef()
            spearman_coeff = spearman(uncs_j, errs_j)
            spearman_coeffs.append(spearman_coeff)
        return torch.stack(spearman_coeffs)

@UncertaintyEvaluatorRegistry.register('conformal-coverage-regression')
class RegressionConformalEvaluator(RegressionEvaluator):

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        bounds = torch.tensor([-1, 1], device=mask.device)
        half_interval = uncs.unsqueeze(0) * bounds.view([-1] + [1] * preds.ndim)
        lower, upper = preds.unsqueeze(0) + half_interval
        covered_mask = torch.logical_and(lower <= targets, targets <= upper)
        return (covered_mask & mask).sum(0) / mask.sum(0)

class BinaryClassificationEvaluator(ABC):

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        pass

@UncertaintyEvaluatorRegistry.register('nll-classification')
class NLLClassEvaluator(BinaryClassificationEvaluator):

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            likelihood = uncs_j * targets_j + (1 - uncs_j) * (1 - targets_j)
            nll = -1 * likelihood.log()
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)

@UncertaintyEvaluatorRegistry.register('conformal-coverage-classification')
class MultilabelConformalEvaluator(BinaryClassificationEvaluator):

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        in_set, out_set = torch.chunk(uncs, 2, 1)
        covered_mask = torch.logical_and(in_set <= targets, targets <= out_set)
        return (covered_mask & mask).sum(0) / mask.sum(0)

class MulticlassClassificationEvaluator(ABC):

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        pass

@UncertaintyEvaluatorRegistry.register('nll-multiclass')
class NLLMulticlassEvaluator(MulticlassClassificationEvaluator):

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            targets_one_hot = torch.eye(uncs_j.shape[-1])[targets_j.long()]
            likelihood = (targets_one_hot * uncs_j).sum(dim=-1)
            nll = -1 * likelihood.log()
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)

@UncertaintyEvaluatorRegistry.register('conformal-coverage-multiclass')
class MulticlassConformalEvaluator(MulticlassClassificationEvaluator):

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=uncs.shape[2])
        covered_mask = torch.max(uncs * targets_one_hot, dim=-1)[0] > 0
        return (covered_mask & mask).sum(0) / mask.sum(0)
