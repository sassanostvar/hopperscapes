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

# Checkpoint
CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent / "checkpoints" / "HopperNetLite_demo.pth"
)

# CONFIGS
CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs" / "unified_lite.yaml"

SAVEDIR = (
    Path(__file__).parent.parent.parent
    / "outputs"
    / "test_outputs"
    / "test_model_training"
)


@pytest.mark.integration
def test_train_pipeline_fresh_start_one_epoch(debug=False):
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

    model = models.ModularHopperNet(
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

    c.epochs = 1
    trainer = train.HopperNetTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criteria=loss_criteria,
        start_epoch=0,
        start_iter=1,
        savedir=_savedir,
        configs=c,
    )
    trainer.train()


@pytest.mark.integration
def test_train_pipeline_resume_checkpoint():
    from hopperscapes.segmentation.dataset import WingPatternDataset, hopper_collate_fn
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation import loss, train, infer
    import torch

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

    c = SegmentationModelConfigs().from_yaml(CONFIGS_PATH)
    model = infer.load_model(checkpoint_path=CHECKPOINT_PATH, configs=c, device="cpu")

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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=c.device)

    c.epochs = 1
    trainer = train.HopperNetTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criteria=loss_criteria,
        start_epoch=0,
        start_iter=1,
        savedir=_savedir,
        configs=c,
    )

    trainer.load_control_flow_states_from_checkpoint(checkpoint)

    trainer.train()
