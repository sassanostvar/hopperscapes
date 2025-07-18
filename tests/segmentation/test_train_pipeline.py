import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

# Train
TRAIN_IMAGES_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "images"
)
TRAIN_MASKS_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "masks"
)

# Valid
VALID_IMAGES_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "valid" / "images"
)
VALID_MASKS_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "valid" / "masks"
)

SAVEDIR = (
    Path(__file__).parent.parent.parent
    / "outputs"
    / "test_outputs"
    / "test_model_training"
)


@pytest.mark.integration
def test_train_pipeline(debug=False):
    """
    Test the train loader.
    """
    from hopperscapes.segmentation.dataset import WingPatternDataset, hopper_collate_fn
    from hopperscapes.configs import SegmentationModelConfigs

    # Create a dataset instance
    train_dataset = WingPatternDataset(
        image_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        configs=SegmentationModelConfigs(),
    )

    valid_dataset = WingPatternDataset(
        image_dir=VALID_IMAGES_DIR,
        masks_dir=VALID_MASKS_DIR,
        configs=SegmentationModelConfigs(),
    )

    # Check the length of the dataset
    assert len(train_dataset) > 0, "Dataset is empty"
    assert len(valid_dataset) > 0, "Dataset is empty"


    # Check the first sample
    sample = train_dataset[0]
    assert "image" in sample, "Image not found in sample"
    assert "masks" in sample, "Masks not found in sample"
    sample = valid_dataset[0]
    assert "image" in sample, "Image not found in sample"
    assert "masks" in sample, "Masks not found in sample"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=hopper_collate_fn,
        shuffle=False,
        drop_last=False,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        collate_fn=hopper_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation import loss, models, train

    c = SegmentationModelConfigs()

    model = models.HopperNetLite(
        num_groups=c.num_groups,  # for GroupNorm
        out_channels=c.out_channels,
    )

    c.model_name = "train_test"
    c.savedir = SAVEDIR

    _savedir = os.path.join(
        c.savedir,
        c.model_name,
    )

    loss_criteria = loss.HopperNetCompositeLoss(
        loss_configs=c.loss_function_configs,
        device=c.device,
    )

    trainer = train.HopperNetTrainer(
        model=model,
        freeze_heads=c.freeze_heads,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criteria=loss_criteria,
        total_loss_weights=c.total_loss_weights,
        lr=c.learning_rate,
        weight_decay=c.weight_decay,
        start_epoch=0,
        num_epochs=1,
        checkpoint_every=c.checkpoint_every,
        log_every=c.log_every,
        clip_gradients=c.clip_gradients,
        max_norm=c.max_grad_norm,
        dice_scores_to_track=c.dice_scores_to_track,
        threshold_dice_scores=c.dice_thresholds_to_freeze_heads,
        device=c.device,
        savedir=_savedir,
    )
    trainer.train()
