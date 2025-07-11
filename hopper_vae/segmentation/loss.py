"""
Loss functions for segmentation tasks.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "soft_dice_loss",
    "soft_dice_loss_with_gating",
    "focal_loss_with_logits",
    "soft_skel",
    "cldice_loss",
]


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    num_classes: int = 1,
) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.
    Args:
        logits: Predicted logits (N, C, H, W)
        target: Ground truth masks (N, C, H, W)
        eps: Small value to avoid division by zero
    Returns:
        Dice loss
    """
    if num_classes > 1:
        one_hot_target = (
            F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )
        probs = F.softmax(logits, dim=1)
        intersection = (probs * one_hot_target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot_target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_loss = (1.0 - dice).mean()
    else:
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_loss = (1.0 - dice).mean()

    return dice_loss


def soft_dice_loss_with_gating(
    logits: torch.Tensor,
    target: torch.Tensor,
    clipping_mask: torch.Tensor,
    eps: float = 1e-6,
    enable_gating=False,
) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.
    Args:
        logits: Predicted logits (N, C, H, W)
        target: Ground truth masks (N, C, H, W)
        clipping_mask: Clipping mask to constrain loss
        calc to specific region (N, C, H, W)
        eps: Small value to avoid division by zero
        enable_gating: Whether to apply the clipping mask
    Returns:
        Dice loss
    """
    probs = torch.sigmoid(logits)
    if enable_gating:
        probs = probs * clipping_mask
    TP = (probs * target).sum(dim=(2, 3))
    P = probs.sum(dim=(2, 3))
    G = target.sum(dim=(2, 3))
    dice = (2.0 * TP + eps) / (P + G + eps)
    # return 1.0 - dice
    return (1.0 - dice).mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
):
    """Binary focal loss on logits (Sigmoid + BCE)."""
    # BCE with logits, but keep per‑pixel loss
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    # p_t = sigmoid(logit) for positive class, 1‑p for negative
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)

    # focal modulation
    focal_factor = (1 - p_t).clamp(min=eps) ** gamma

    # alpha balancing
    if alpha is not None:
        alpha_factor = alpha * target + (1 - alpha) * (1 - target)
        focal_factor = focal_factor * alpha_factor

    loss = focal_factor * bce
    return loss.mean() if reduction == "mean" else loss


def soft_skel(x, iters: int = 5):
    """Differentiable morphological skeletonisation."""
    for _ in range(iters):
        eroded = -F.max_pool2d(-x, 3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        x = torch.clamp(x - (opened - eroded), min=0.0, max=1.0)
    return x


def cldice_loss(
    logits: torch.Tensor, target: torch.Tensor, iters: int = 5, eps: float = 1e-6
):
    """Soft clDice loss for binary segmentation."""
    probs = torch.sigmoid(logits)
    skel_pred = soft_skel(probs, iters)
    skel_gt = soft_skel(target.float(), iters)

    tp = (skel_pred * target).sum(dim=(2, 3))
    ts = (skel_gt * probs).sum(dim=(2, 3))

    numerator = 2 * tp * ts
    denominator = tp + ts

    # cldice = (numerator + eps) / (denominator + eps)

    # guard against all zeros yielding a perfect cldice
    cldice_per_item = torch.ones_like(numerator, dtype=torch.float32)
    _valid_mask = denominator > 0.0
    cldice_per_item[_valid_mask] = (numerator[_valid_mask] + eps) / (
        denominator[_valid_mask] + eps
    )

    # CLIP THE CLDICE VALUE TO PREVENT IT FROM EXCEEDING 1.0
    cldice = torch.clamp(cldice_per_item, min=0.0, max=1.0)

    # return 1 - cldice
    return (1.0 - cldice).mean()


dice_funcs = {
    "soft_dice": soft_dice_loss,
    "cldice": cldice_loss,
    "focal": focal_loss_with_logits,
}


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Args:
        eps: Small value to avoid division by zero
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(logits, target, self.eps)


class SoftDiceLossWithGating(nn.Module):
    """
    Soft Dice loss for binary segmentation with gating.
    Args:
        eps: Small value to avoid division by zero
        enable_gating: Whether to apply the clipping mask
    """

    def __init__(self, eps: float = 1e-6, enable_gating=False):
        super().__init__()
        self.eps = eps
        self.enable_gating = enable_gating

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, clipping_mask: torch.Tensor
    ) -> torch.Tensor:
        return soft_dice_loss_with_gating(
            logits, target, clipping_mask, self.eps, self.enable_gating
        )


class FocalLossWithLogits(nn.Module):
    """
    Binary focal loss on logits (Sigmoid + BCE).
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small value to avoid division by zero
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_logits(
            logits, target, self.alpha, self.gamma, self.reduction, self.eps
        )


class clDiceLoss(nn.Module):
    """
    Soft clDice loss for binary segmentation.
    Args:
        iters: Number of iterations for soft skeletonization
        eps: Small value to avoid division by zero
    """

    def __init__(self, iters: int = 5, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cldice_loss(logits, target, self.iters, self.eps)


class HopperNetCompositeLoss(nn.Module):
    """
    Composite loss function for multi-head semantic segmentation.
    """

    # registry
    loss_function_registry = {
        "ce": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "soft_dice": SoftDiceLoss,
        "soft_dice_with_gating": SoftDiceLossWithGating,
        "focal": FocalLossWithLogits,
        "cldice": clDiceLoss,
    }

    def __init__(self, loss_configs: Dict[str, Dict], device: str = "cpu"):
        #
        for head_name, config in loss_configs.items():
            for sub_loss_name, loss_params in config.items():
                if sub_loss_name not in self.loss_function_registry:
                    raise ValueError(
                        f"Loss function '{sub_loss_name}' is not registered."
                    )

        super().__init__()
        self.loss_configs = loss_configs
        self.loss_funcs = {}
        for head_name, loss_config in loss_configs.items():
            self.loss_funcs[head_name] = {}
            for sub_loss_name, sub_loss_specs in loss_config.items():
                # Initialize the loss function with parameters
                if sub_loss_name == "bce":
                    sub_loss_specs["params"]["pos_weight"] = torch.tensor(
                        [sub_loss_specs["params"]["pos_weight"]], device=device
                    )
                self.loss_funcs[head_name][sub_loss_name] = self.loss_function_registry[
                    sub_loss_name
                ](**sub_loss_specs["params"]).to(device)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        clipping_masks: Dict[str, torch.Tensor] = None,
        weights: Dict[str, float] = None,
    ):
        # ensure the keys match
        if inputs.keys() != targets.keys():
            raise ValueError("Predictions and target must have the same keys.")

        total_loss = 0.0
        head_losses = {}

        for head_name, logits in inputs.items():
            if head_name not in self.loss_funcs:
                raise ValueError(
                    f"Loss function for head '{head_name}' is not defined."
                )

            if weights[head_name] <= 0.0:
                continue

            target = targets[head_name]
            clipping_mask = clipping_masks.get(head_name) if clipping_masks else None

            # per batch head loss
            head_loss = 0.0

            for sub_loss_name, sub_loss_func in self.loss_funcs[head_name].items():
                # Apply the loss function
                _weight = self.loss_configs[head_name][sub_loss_name]["weight"]
                if (
                    hasattr(sub_loss_func, "enable_gating")
                    and sub_loss_func.enable_gating
                ):
                    head_loss += _weight * sub_loss_func(logits, target, clipping_mask)
                else:
                    head_loss += _weight * sub_loss_func(logits, target)

            head_losses[head_name] = head_loss
            total_loss += head_loss * weights[head_name]

        return total_loss, head_losses
