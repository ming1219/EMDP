# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class GHM_Loss(nn.Module):
    """A :class:`GHM_Loss` class."""

    def __init__(self, bins=10, alpha=0.5):
        """
        Initializes the GHM_Loss module with the specified number of bins and alpha value.

        :param bins: (int) The number of bins to divide the gradient. Defaults to 10.
        :param alpha: (float) The smoothing parameter for updating the last bin count. Defaults to 0.5.
        """
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        """
        Maps gradient values to corresponding bin indices.

        :param g: (torch.Tensor) Gradient tensor.
        :return: (torch.Tensor) Bin indices for each gradient value.
        """
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        """
        Custom loss function to be implemented in subclasses.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth labels.
        :param weight: (torch.Tensor) Weights for the loss.
        :raise NotImplementedError: Indicates that the method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        """
        Custom gradient computation function to be implemented in subclasses.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth labels.
        :raise NotImplementedError: Indicates that the method should be implemented in subclasses.
        """
        raise NotImplementedError

    def forward(self, x, target):
        """
        Forward pass for computing the GHM loss.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth labels.
        :return: (torch.Tensor) Computed GHM loss.
        """
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = x.size(0) * x.size(1)

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = (
                self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            )
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        beta = beta.type_as(x)

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
    Inherits from GHM_Loss. GHM_Loss for classification.
    '''

    def __init__(self, bins, alpha):
        """
        Initializes the GHMC_Loss with specified number of bins and alpha value.

        :param bins: (int) Number of bins for gradient division.
        :param alpha: (float) Smoothing parameter for bin count updating.
        """
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        """
        Custom loss function for GHM classification loss.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth labels.
        :param weight: (torch.Tensor) Weights for the loss.

        :return: Binary cross-entropy loss with logits.
        """
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        """
        Custom gradient function for GHM classification loss.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth labels.

        :return: Gradient of the loss.
        """
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    '''
    Inherits from GHM_Loss. GHM_Loss for regression
    '''

    def __init__(self, bins, alpha, mu):
        """
        Initializes the GHMR_Loss with specified number of bins, alpha value, and mu parameter.

        :param bins: (int) Number of bins for gradient division.
        :param alpha: (float) Smoothing parameter for bin count updating.
        :param mu: (float) Parameter used in the GHMR loss formula.
        """
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        """
        Custom loss function for GHM regression loss.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth values.
        :param weight: (torch.Tensor) Weights for the loss.

        :return: GHMR loss.
        """
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        """
        Custom gradient function for GHM regression loss.

        :param x: (torch.Tensor) Predicted values.
        :param target: (torch.Tensor) Ground truth values.

        :return: Gradient of the loss.
        """
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


def MAEwithNan(y_pred, y_true):
    """
    Calculates the Mean Absolute Error (MAE) loss, ignoring NaN values in the target.

    :param y_pred: (torch.Tensor) Predicted values.
    :param y_true: (torch.Tensor) Ground truth values, may contain NaNs.

    :return: (torch.Tensor) MAE loss computed only on non-NaN elements.
    """
    mask = ~torch.isnan(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    mae_loss = nn.L1Loss()
    loss = mae_loss(y_pred, y_true)
    return loss


def FocalLoss(y_pred, y_true, alpha=0.25, gamma=2):
    """
    Calculates the Focal Loss, used to address class imbalance by focusing on hard examples.

    :param y_pred: (torch.Tensor) Predicted probabilities.
    :param y_true: (torch.Tensor) Ground truth labels.
    :param alpha: (float) Weighting factor for balancing positive and negative examples. Defaults to 0.25.
    :param gamma: (float) Focusing parameter to scale the loss. Defaults to 2.

    :return: (torch.Tensor) Computed focal loss.
    """
    if y_pred.shape != y_true.shape:
        y_true = y_true.flatten()
    y_true = y_true.long()
    y_pred = y_pred.float()
    y_true = y_true.float()
    y_true = y_true.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)
    y_true = torch.cat((1 - y_true, y_true), dim=1)
    y_pred = torch.cat((1 - y_pred, y_pred), dim=1)
    y_pred = y_pred.clamp(1e-5, 1.0)
    loss = -alpha * y_true * torch.pow((1 - y_pred), gamma) * torch.log(y_pred)
    return torch.mean(torch.sum(loss, dim=1))


def FocalLossWithLogits(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Calculates the Focal Loss using predicted logits (raw scores), automatically applying the sigmoid function.

    :param y_pred: (torch.Tensor) Predicted logits.
    :param y_true: (torch.Tensor) Ground truth labels, may contain NaNs.
    :param alpha: (float) Weighting factor for balancing positive and negative examples. Defaults to 0.25.
    :param gamma: (float) Focusing parameter to scale the loss. Defaults to 2.0.

    :return: (torch.Tensor) Computed focal loss.
    """
    y_pred = torch.sigmoid(y_pred)
    mask = ~torch.isnan(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    loss = FocalLoss(y_pred, y_true)
    return loss


def myCrossEntropyLoss(y_pred, y_true):
    """
    Calculates the cross-entropy loss between predictions and targets.

    :param y_pred: (torch.Tensor) Predicted logits or probabilities.
    :param y_true: (torch.Tensor) Ground truth labels.

    :return: (torch.Tensor) Computed cross-entropy loss.
    """
    if y_pred.shape != y_true.shape:
        y_true = y_true.flatten()
    return nn.CrossEntropyLoss()(y_pred, y_true)


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty Weighted Loss for multi-task learning.
    
    Based on the paper: "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics" by Kendall et al. (2018)
    
    For regression tasks with Gaussian likelihood:
    L = Σ (1/(2σ_i²)) * MSE_i + log(σ_i)
    
    Let s_i = log(σ_i²), then:
    L = Σ 0.5 * exp(-s_i) * MSE_i + 0.5 * s_i
    """

    def __init__(self, num_tasks=8):
        """
        Initializes the UncertaintyWeightedLoss module.

        :param num_tasks: (int) The number of tasks/targets. Defaults to 8.
        """
        super(UncertaintyWeightedLoss, self).__init__()
        # log_vars represents s = log(σ²), initialized to 0 (σ = 1, equal weights)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.num_tasks = num_tasks
        self._debug_first_call = True
        print(f"[DEBUG] UncertaintyWeightedLoss initialized with num_tasks={num_tasks}")

    def forward(self, y_pred, y_true):
        """
        Forward pass - computes uncertainty weighted loss for each task independently.
        
        Based on Kendall et al. (2018) for regression:
        L = Σ (1/(2σ_i²)) * MSE_i + log(σ_i)
        Where s_i = log(σ_i²), so: L = Σ 0.5 * exp(-s_i) * MSE_i + 0.5 * s_i
        """
        # Debug info on first call
        if self._debug_first_call:
            print(f"[DEBUG] UncertaintyWeightedLoss.forward called")
            print(f"[DEBUG] y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}, device: {y_pred.device}")
            print(f"[DEBUG] y_true shape: {y_true.shape}, dtype: {y_true.dtype}, device: {y_true.device}")
            print(f"[DEBUG] log_vars device: {self.log_vars.device}, requires_grad: {self.log_vars.requires_grad}")
            self._debug_first_call = False
        
        # Handle 1D case (single task)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        batch_size, num_tasks = y_pred.shape
        
        # Validate num_tasks matches initialized num_tasks
        if num_tasks != self.num_tasks:
            raise ValueError(
                f"Number of tasks in input ({num_tasks}) does not match "
                f"initialized num_tasks ({self.num_tasks}). "
                f"Please reinitialize UncertaintyWeightedLoss with correct num_tasks."
            )
        
        # Compute loss for each task independently
        total_loss = 0.0
        valid_tasks = 0
        
        for i in range(num_tasks):
            # Get mask for valid (non-NaN) values for this task
            mask_i = ~torch.isnan(y_true[:, i])
            
            if mask_i.sum() == 0:
                continue  # Skip tasks with no valid values
            
            # Get predictions and targets for this task
            pred_i = y_pred[:, i][mask_i]
            true_i = y_true[:, i][mask_i]
            
            # Compute MSE for this task (following Kendall et al. 2018 for regression)
            mse_i = torch.pow(pred_i - true_i, 2).mean()
            
            # Get the log_var for this specific task (each task has its own parameter)
            log_var_i = self.log_vars[i]
            
            # Compute precision (inverse variance) for this task
            precision_i = torch.exp(-log_var_i)
            
            # Uncertainty weighted loss for this task:
            # L_i = 0.5 * precision * MSE + 0.5 * log_var (regularization)
            loss_i = 0.5 * precision_i * mse_i + 0.5 * log_var_i
            
            total_loss = total_loss + loss_i
            valid_tasks += 1
        
        # Return average loss across valid tasks (or zero if no valid tasks)
        if valid_tasks == 0:
            return y_pred.sum() * 0.0  # Return tensor connected to computation graph
        
        return total_loss / valid_tasks

    def get_task_weights(self):
        """
        Returns the learned task weights (precision = exp(-log_var)).
        """
        with torch.no_grad():
            return torch.exp(-self.log_vars)

    def get_task_uncertainties(self):
        """
        Returns the learned task uncertainties (σ = exp(log_var/2)).
        """
        with torch.no_grad():
            return torch.exp(self.log_vars / 2)
