import pytest
import torch
from hopper_vae.segmentation.loss import (
    soft_dice_loss,
    soft_dice_loss_with_gating,
    focal_loss_with_logits,
    cldice_loss,
)

@pytest.mark.parametrize("num_classes", [1, 3])
def test_soft_dice_loss(num_classes):
    N, H, W = 2, 64, 64
    logits = torch.rand(N, num_classes, H, W)  # Shape (N, C, H, W)
    
    # Generate target with valid indices for multi-class segmentation
    if num_classes > 1:
        target = torch.randint(0, num_classes, (N, H, W))  # Shape (N, H, W)
    else:
        target = torch.randint(0, 2, (N, 1, H, W)).float()  # Binary mask (N, C, H, W)

    loss = soft_dice_loss(logits, target, num_classes=num_classes)
    assert loss >= 0, "Dice loss should be non-negative."
    assert loss <= 1, "Dice loss should not exceed 1."

@pytest.mark.parametrize("enable_gating", [True, False])
def test_soft_dice_loss_with_gating(enable_gating):
    logits = torch.rand(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    clipping_mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = soft_dice_loss_with_gating(logits, target, clipping_mask, enable_gating=enable_gating)
    assert loss >= 0, "Gated Dice loss should be non-negative."

@pytest.mark.parametrize("alpha, gamma", [(0.25, 2.0), (0.75, 1.0)])
def test_focal_loss_with_logits(alpha, gamma):
    logits = torch.rand(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = focal_loss_with_logits(logits, target, alpha=alpha, gamma=gamma)
    assert loss >= 0, "Focal loss should be non-negative."

@pytest.mark.parametrize("iters", [3, 5])
def test_cldice_loss(iters):
    logits = torch.rand(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = cldice_loss(logits, target, iters=iters)
    assert loss >= 0, "clDice loss should be non-negative."
    assert loss <= 1, "clDice loss should not exceed 1."