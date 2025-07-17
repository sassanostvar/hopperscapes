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
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.loss import HopperNetCompositeLoss

    # init class with default configs
    configs = SegmentationModelConfigs()
    _ = HopperNetCompositeLoss(
        loss_configs=configs.loss_function_configs, device="cpu"
    )
