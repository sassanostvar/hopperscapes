from pathlib import Path

import pytest

IMAGES_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "images"
)
MASKS_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "masks"
)


@pytest.mark.integration
def test_composite_loss_func_minimal():
    from hopper_vae.segmentation.dataset import WingPatternDataset, hopper_collate_fn

    # Create a dataset instance
    dataset = WingPatternDataset(
        image_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )

    from hopper_vae.configs import SegmentationModelConfigs
    from hopper_vae.segmentation.loss import HopperNetCompositeLoss

    configs = SegmentationModelConfigs()
    composite_loss = HopperNetCompositeLoss(
        loss_configs=configs.loss_function_configs, device="cpu"
    )
