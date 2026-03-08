"""
Custom loss functions for XDenseQNet training.

Contains:
    - FocalLoss           -- standard focal loss for class imbalance
    - LabelSmoothingFocalLoss -- focal loss with label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Args:
        alpha:     Per-class weight tensor (optional).
        gamma:     Focusing parameter (default 2.0).
        reduction: ``'mean'`` or ``'none'``.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss


class LabelSmoothingFocalLoss(nn.Module):
    """Focal Loss combined with label smoothing.

    Args:
        num_classes: Number of target classes.
        alpha:       Per-class weight tensor (optional).
        gamma:       Focusing parameter (default 2.0).
        smoothing:   Label smoothing factor (default 0.1).
        reduction:   ``'mean'`` or ``'none'``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha=None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_labels = torch.zeros_like(inputs)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.log_softmax(inputs, dim=1)
        loss = (-smooth_labels * log_probs).sum(dim=1)

        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * loss

        if self.alpha is not None:
            class_weights = self.alpha[targets]
            focal_loss = focal_loss * class_weights

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss
