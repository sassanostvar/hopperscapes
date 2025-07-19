from pathlib import Path

import pytest

ZARR_PATH = Path(__file__).parent.parent / "test_data" / "zarr_demo.zarr"


@pytest.mark.unit
def test_hopper_zarr_dataset_init():
    from torch.utils.data import Dataset

    from hopperscapes.segmentation.dataset import HopperZarrDataset
    from hopperscapes.configs import SegmentationModelConfigs

    ZarrDataset = HopperZarrDataset(ZARR_PATH, configs=SegmentationModelConfigs())
    assert isinstance(ZarrDataset, Dataset)


@pytest.mark.unit
def test_hopper_zarr_dataset_fetch_image():
    import torch
    from torch.utils.data import Dataset

    from hopperscapes.segmentation.dataset import HopperZarrDataset
    from hopperscapes.configs import SegmentationModelConfigs

    ZarrDataset = HopperZarrDataset(ZARR_PATH, configs=SegmentationModelConfigs())
    assert isinstance(ZarrDataset, Dataset)

    idx = 0
    sample = ZarrDataset[idx]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "meta" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["meta"], dict)