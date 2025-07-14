from pathlib import Path

import pytest
from torch.utils.data import DataLoader

IMAGES_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "images"
)
MASKS_DIR = (
    Path(__file__).parent.parent / "test_data" / "dataset" / "raw" / "train" / "masks"
)


@pytest.mark.unit
def test_train_loader(debug=False):
    """
    Test the train loader.
    """
    from hopper_vae.segmentation.dataset import WingPatternDataset, hopper_collate_fn

    # Create a dataset instance
    dataset = WingPatternDataset(
        image_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )

    # Check the length of the dataset
    assert len(dataset) > 0, "Dataset is empty"

    # Check the first sample
    sample = dataset[0]
    assert "image" in sample, "Image not found in sample"
    assert "masks" in sample, "Masks not found in sample"

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=hopper_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    if debug:
        # Inspect shapes in the training loop:
        for batch_idx, sample in enumerate(train_loader):
            images = sample["image"]
            masks = sample["masks"]
            print(f"\nBatch {batch_idx}:")
            print("images shape:", images.shape)
            print("masks keys:", masks.keys())
            print("wing_masks shape:", masks["wing"].shape)
