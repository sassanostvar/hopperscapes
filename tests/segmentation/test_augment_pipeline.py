from pathlib import Path

import pytest

IMAGES_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "images"
)
MASKS_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "masks"
)


@pytest.mark.integration
def test_default_augment_pipeline():
    """
    Create augmented views for test data and write to disk.
    """
    from hopper_vae.segmentation.dataset import WingPatternDataset, hopper_collate_fn

    # Create a dataset instance
    dataset = WingPatternDataset(
        image_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )

    from hopper_vae.segmentation.augment import augment

    configs = augment.AugmentConfigs()
