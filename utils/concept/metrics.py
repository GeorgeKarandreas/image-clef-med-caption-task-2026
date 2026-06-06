"""Calculation functions for Loss and Validation metrics"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class AsymmetricLoss(nn.Module):
    """
    ASL for multi-label classification.
    Expects:
        logits:  [B, C]
        targets: [B, C] with values 0/1 (float)
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction='{reduction}'. Must be one of "
                f"{{'none', 'mean', 'sum'}}."
            )

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Keep the loss computation in fp32 to avoid AMP/fp16 log underflow.
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs

        # asymmetric clipping for negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # basic CE terms
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
            loss = loss * focal_weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

        return loss


class CombinedMultiLabelLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        pos_weight: torch.Tensor | None = None,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = self.bce(logits, targets)
        asl = self.asl(logits, targets)

        return self.alpha * asl + (1.0 - self.alpha) * bce


def compute_primary_f1_score(y_true, y_pred):
    """
    Compute the primary ImageCLEF concept-detection score.

    Args:
        y_true (numpy.ndarray): Ground-truth binary label matrix.
        y_pred (numpy.ndarray): Predicted binary label matrix.

    Returns:
        float: Average per-image F1 score.
    """
    total_f1 = 0.0
    scored_images = 0

    for target_row, prediction_row in zip(y_true, y_pred):
        if target_row.sum() == 0:
            continue

        total_f1 += f1_score(target_row, prediction_row, average="binary")
        scored_images += 1

    if scored_images == 0:
        return 0.0

    return total_f1 / scored_images


ALLOWED_CUIS = {
    "C0002978",
    "C0040405",
    "C0024485",
    "C0032743",
    "C0041618",
    "C1306645",
    "C1140618",
    "C0037949",
    "C0030797",
    "C0023216",
    "C0037303",
    "C0817096",
    "C0006141",
    "C0000726",
    "C0920367",
}

def compute_secondary_f1_score(y_true, y_pred, cui_to_index):
    """
    Compute the secondary ImageCLEF concept-detection score.

    Args:
        y_true (numpy.ndarray): Ground-truth binary label matrix.
        y_pred (numpy.ndarray): Predicted binary label matrix.
        cui_to_index (dict): Mapping from CUI string to label index.

    Returns:
        float: Average per-image F1 score on the official secondary concept set.
    """
    cui_indices = [cui_to_index[cui] for cui in ALLOWED_CUIS if cui in cui_to_index]

    total_f1 = 0.0
    scored_images = 0

    for target_row, prediction_row in zip(
        y_true[:, cui_indices],
        y_pred[:, cui_indices],
    ):
        if target_row.sum() == 0:
            continue

        total_f1 += f1_score(target_row, prediction_row, average="binary")
        scored_images += 1

    if scored_images == 0:
        return 0.0

    return total_f1 / scored_images
