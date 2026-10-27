from abc import ABC, abstractmethod
import logging
import math
from typing import Self
import numpy as np
from scipy.optimize import fmin
from scipy.special import expit, logit, softmax
from sklearn.isotonic import IsotonicRegression
import torch
from torch import Tensor
from ps2s.utils.registry import ClassRegistry
logger = logging.getLogger(__name__)

class CalibratorBase(ABC):

    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        pass

    @abstractmethod
    def apply(self, uncs: Tensor) -> Tensor:
        pass
UncertaintyCalibratorRegistry = ClassRegistry[CalibratorBase]()

class RegressionCalibrator(CalibratorBase):

    @abstractmethod
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        pass

@UncertaintyCalibratorRegistry.register('zscaling')
class ZScalingCalibrator(RegressionCalibrator):

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        scalings = np.zeros(uncs.shape[1])
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j].numpy()
            uncs_j = uncs[:, j][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()
            errors = preds_j - targets_j

            def objective(scaler_value: float):
                scaled_vars = uncs_j * scaler_value ** 2
                nll = np.log(2 * np.pi * scaled_vars) / 2 + errors ** 2 / (2 * scaled_vars)
                return nll.sum()
            zscore = errors / np.sqrt(uncs_j)
            initial_guess = np.std(zscore)
            scalings[j] = fmin(objective, x0=initial_guess, disp=False).item()
        self.scalings = torch.tensor(scalings)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return uncs * self.scalings ** 2

@UncertaintyCalibratorRegistry.register('zelikman-interval')
class ZelikmanCalibrator(RegressionCalibrator):

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        if not 0 <= self.p <= 1:
            raise ValueError(f'arg `p` must be between 0 and 1. got: {p}.')

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        scalings = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            z = (preds_j - targets_j).abs() / uncs_j.sqrt()
            scaling = torch.quantile(z, self.p, interpolation='lower')
            scalings.append(scaling)
        self.scalings = torch.tensor(scalings)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return uncs * self.scalings ** 2

@UncertaintyCalibratorRegistry.register('mve-weighting')
class MVEWeightingCalibrator(RegressionCalibrator):

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        scalings = []
        for j in range(uncs.shape[2]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j].numpy()
            uncs_j = uncs[:, mask_j, j].numpy()
            targets_j = targets[:, j][mask_j].numpy()
            errors = preds_j - targets_j

            def objective(scaler_values: np.ndarray):
                scaler_values = np.reshape(softmax(scaler_values), [-1, 1])
                scaled_vars = np.sum(uncs_j * scaler_values, axis=0, keepdims=False)
                nll = np.log(2 * np.pi * scaled_vars) / 2 + errors ** 2 / (2 * scaled_vars)
                return np.sum(nll)
            initial_guess = np.ones(uncs_j.shape[0])
            sol = fmin(objective, x0=initial_guess, disp=False)
            scalings.append(torch.tensor(softmax(sol)))
        self.scalings = torch.stack(scalings).t().unsqueeze(1)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return (uncs * self.scalings).sum(0)

@UncertaintyCalibratorRegistry.register('conformal-regression')
class RegressionConformalCalibrator(RegressionCalibrator):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.bounds = torch.tensor([-1, 1]).view(-1, 1)
        if not 0 <= self.alpha <= 1:
            raise ValueError(f'arg `alpha` must be between 0 and 1. got: {alpha}.')

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        self.qhats = []
        for j in range(preds.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            preds_j = preds[:, j][mask_j]
            half_interval_j = uncs[:, j][mask_j]
            interval_bounds = self.bounds * half_interval_j.unsqueeze(0)
            pred_bounds = preds_j.unsqueeze(0) + interval_bounds
            calibration_scores = torch.max(pred_bounds[0] - targets_j, targets_j - pred_bounds[1])
            num_data = targets_j.shape[0]
            if self.alpha >= 1 / (num_data + 1):
                q_level = math.ceil((num_data + 1) * (1 - self.alpha)) / num_data
            else:
                q_level = 1
                logger.warning('The error rate (i.e., `alpha`) is smaller than `1 / (number of data + 1)`, so the `1 - alpha` quantile is set to 1, but this only ensures that the coverage is trivially satisfied.')
            qhat = torch.quantile(calibration_scores, q_level, interpolation='higher')
            self.qhats.append(qhat)
        self.qhats = torch.tensor(self.qhats)
        return self

    def apply(self, uncs: Tensor) -> tuple[Tensor, Tensor]:
        return uncs + self.qhats

class BinaryClassificationCalibrator(CalibratorBase):

    @abstractmethod
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        pass

@UncertaintyCalibratorRegistry.register('platt')
class PlattCalibrator(BinaryClassificationCalibrator):

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor, training_targets: Tensor | None=None) -> Self:
        if torch.any((targets[mask] != 0) & (targets[mask] != 1)):
            raise ValueError("Platt scaling is only implemented for binary classification tasks! Input tensor must contain only 0's and 1's.")
        if training_targets is not None:
            logger.info('Training targets were provided. Platt scaling for calibration uses a Bayesian correction to avoid training set overfitting. Now replacing calibration targets [0, 1] with adjusted values.')
            n_negative_examples = (training_targets == 0).sum(dim=0)
            n_positive_examples = (training_targets == 1).sum(dim=0)
            negative_target_bayes_MAP = (1 / (n_negative_examples + 2)).expand_as(targets)
            positive_target_bayes_MAP = ((n_positive_examples + 1) / (n_positive_examples + 2)).expand_as(targets)
            targets = targets.float()
            targets[targets == 0] = negative_target_bayes_MAP[targets == 0]
            targets[targets == 1] = positive_target_bayes_MAP[targets == 1]
        else:
            logger.info('No training targets were provided. No Bayesian correction is applied.')
        xs = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            uncs_j = uncs[:, j][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()

            def objective(parameters):
                a, b = parameters
                scaled_uncs = expit(a * logit(uncs_j) + b)
                nll = -1 * np.sum(targets_j * np.log(scaled_uncs) + (1 - targets_j) * np.log(1 - scaled_uncs))
                return nll
            xs.append(fmin(objective, x0=[1, 0], disp=False))
        xs = np.vstack(xs)
        self.a, self.b = torch.tensor(xs).T.unbind(dim=0)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return torch.sigmoid(self.a * torch.logit(uncs) + self.b)

@UncertaintyCalibratorRegistry.register('isotonic')
class IsotonicCalibrator(BinaryClassificationCalibrator):

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        if torch.any((targets[mask] != 0) & (targets[mask] != 1)):
            raise ValueError("Isotonic calibration is only implemented for binary classification tasks! Input tensor must contain only 0's and 1's.")
        isotonic_models = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            uncs_j = uncs[:, j][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()
            isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            isotonic_model.fit(uncs_j, targets_j)
            isotonic_models.append(isotonic_model)
        self.isotonic_models = isotonic_models
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        cal_uncs = []
        for j, isotonic_model in enumerate(self.isotonic_models):
            cal_uncs.append(isotonic_model.predict(uncs[:, j].numpy()))
        return torch.tensor(np.array(cal_uncs)).t()

@UncertaintyCalibratorRegistry.register('conformal-multilabel')
class MultilabelConformalCalibrator(BinaryClassificationCalibrator):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        if not 0 <= self.alpha <= 1:
            raise ValueError(f'arg `alpha` must be between 0 and 1. got: {alpha}.')

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        return -preds

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        if targets.shape[1] < 2:
            raise ValueError(f'the number of tasks should be larger than 1! got: {targets.shape[1]}.')
        has_zeros = torch.any(targets == 0, dim=1)
        index_zeros = targets[has_zeros] == 0
        scores_in = self.nonconformity_scores(uncs[has_zeros])
        masked_scores_in = scores_in * index_zeros.float() + torch.where(index_zeros, torch.zeros_like(scores_in), torch.tensor(float('inf')))
        calibration_scores_in = torch.min(masked_scores_in.masked_fill(~mask, float('inf')), dim=1).values
        has_ones = torch.any(targets == 1, dim=1)
        index_ones = targets[has_ones] == 1
        scores_out = self.nonconformity_scores(uncs[has_ones])
        masked_scores_out = scores_out * index_ones.float() + torch.where(index_ones, torch.zeros_like(scores_out), torch.tensor(float('-inf')))
        calibration_scores_out = torch.max(masked_scores_out.masked_fill(~mask, float('-inf')), dim=1).values
        self.tout = torch.quantile(calibration_scores_out, 1 - self.alpha / 2, interpolation='higher')
        self.tin = torch.quantile(calibration_scores_in, self.alpha / 2, interpolation='higher')
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        scores = self.nonconformity_scores(uncs)
        cal_preds_in = (scores <= self.tin).int()
        cal_preds_out = (scores <= self.tout).int()
        cal_preds_in_out = torch.stack((cal_preds_in, cal_preds_out), dim=2)
        return cal_preds_in_out

class MulticlassClassificationCalibrator(CalibratorBase):

    @abstractmethod
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        pass

@UncertaintyCalibratorRegistry.register('conformal-multiclass')
class MulticlassConformalCalibrator(MulticlassClassificationCalibrator):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        if not 0 <= self.alpha <= 1:
            raise ValueError(f'arg `alpha` must be between 0 and 1. got: {alpha}.')

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        return -preds

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        self.qhats = []
        scores = self.nonconformity_scores(uncs)
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            scores_j = scores[:, j][mask_j]
            scores_j = torch.gather(scores_j, 1, targets_j.unsqueeze(1)).squeeze(1)
            num_data = targets_j.shape[0]
            if self.alpha >= 1 / (num_data + 1):
                q_level = math.ceil((num_data + 1) * (1 - self.alpha)) / num_data
            else:
                q_level = 1
                logger.warning('`alpha` is smaller than `1 / (number of data + 1)`, so the `1 - alpha` quantile is set to 1, but this only ensures that the coverage is trivially satisfied.')
            qhat = torch.quantile(scores_j, q_level, interpolation='higher')
            self.qhats.append(qhat)
        self.qhats = torch.tensor(self.qhats)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        calibrated_preds = torch.zeros_like(uncs, dtype=torch.int)
        scores = self.nonconformity_scores(uncs)
        for j, qhat in enumerate(self.qhats):
            calibrated_preds[:, j] = (scores[:, j] <= qhat).int()
        return calibrated_preds

@UncertaintyCalibratorRegistry.register('conformal-adaptive')
class AdaptiveMulticlassConformalCalibrator(MulticlassConformalCalibrator):

    @staticmethod
    def nonconformity_scores(preds):
        sort_index = torch.argsort(-preds, dim=2)
        sorted_preds = torch.gather(preds, 2, sort_index)
        sorted_scores = sorted_preds.cumsum(dim=2)
        unsorted_scores = torch.zeros_like(sorted_scores).scatter_(2, sort_index, sorted_scores)
        return unsorted_scores

@UncertaintyCalibratorRegistry.register('isotonic-multiclass')
class IsotonicMulticlassCalibrator(MulticlassClassificationCalibrator):

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        isotonic_models = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            uncs_j = uncs[:, j, :][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()
            class_isotonic_models = []
            for k in range(uncs.shape[2]):
                class_uncs_j = uncs_j[..., k]
                positive_class_targets = targets_j == k
                class_targets = np.ones_like(class_uncs_j)
                class_targets[positive_class_targets] = 1
                class_targets[~positive_class_targets] = 0
                isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                isotonic_model.fit(class_uncs_j, class_targets)
                class_isotonic_models.append(isotonic_model)
            isotonic_models.append(class_isotonic_models)
        self.isotonic_models = isotonic_models
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        cal_uncs = torch.zeros_like(uncs)
        for j, class_isotonic_models in enumerate(self.isotonic_models):
            for k, isotonic_model in enumerate(class_isotonic_models):
                class_uncs_j = uncs[:, j, k].numpy()
                class_cal_uncs = isotonic_model.predict(class_uncs_j)
                cal_uncs[:, j, k] = torch.tensor(class_cal_uncs)
        return cal_uncs / cal_uncs.sum(dim=-1, keepdim=True)
