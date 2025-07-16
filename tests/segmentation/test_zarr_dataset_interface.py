from pathlib import Path

import pytest

ZARR_PATH = Path(__file__).parent.parent / "test_data" / "zarr_demo.zarr"


@pytest.mark.unit
def test_hopper_zarr_dataset_init():
    from torch.utils.data import Dataset

    from hopperscapes.segmentation.dataset import HopperZarrDataset

    ZarrDataset = HopperZarrDataset(ZARR_PATH)
    assert isinstance(ZarrDataset, Dataset)


@pytest.mark.unit
def test_hopper_zarr_dataset_fetch_image():
    import torch
    from torch.utils.data import Dataset

    from hopperscapes.segmentation.dataset import HopperZarrDataset

    ZarrDataset = HopperZarrDataset(ZARR_PATH)
    assert isinstance(ZarrDataset, Dataset)

    idx = 0
    sample = ZarrDataset[idx]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "meta" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["meta"], dict)


@pytest.mark.unit
def test_hopper_zarr_dataset_fetch_image_test_transform(debug=False):
    import torch
    from torch.utils.data import Dataset

    from hopperscapes.segmentation.dataset import HopperZarrDataset

    # with transform
    ZarrDataset = HopperZarrDataset(ZARR_PATH)
    assert isinstance(ZarrDataset, Dataset)

    idx = 0
    transformed_sample = ZarrDataset[idx]
    assert isinstance(transformed_sample, dict)
    assert "image" in transformed_sample
    assert "meta" in transformed_sample
    assert isinstance(transformed_sample["image"], torch.Tensor)
    assert isinstance(transformed_sample["meta"], dict)

    # without transform
    ZarrDataset = HopperZarrDataset(ZARR_PATH, transform=None)
    assert isinstance(ZarrDataset, Dataset)

    idx = 0
    raw_sample = ZarrDataset[idx]
    assert isinstance(raw_sample, dict)
    assert "image" in raw_sample
    assert "meta" in raw_sample
    assert isinstance(raw_sample["image"], torch.Tensor)
    assert isinstance(raw_sample["meta"], dict)

    if debug:
        import matplotlib.pyplot as plt

        print(transformed_sample["image"].shape)
        print(raw_sample["image"].shape)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(transformed_sample["image"].permute(1, 2, 0).detach().numpy())
        ax[1].imshow(raw_sample["image"].permute(1, 2, 0).detach().numpy())
        fig.tight_layout()
        plt.show()
