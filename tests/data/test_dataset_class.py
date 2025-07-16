from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from hopperscapes.segmentation.dataset import WingPatternDataset

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "train"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

SAVEDIR = Path(__file__).parent.parent.parent / "outputs" / "test_outputs"


@pytest.mark.unit
def test_custom_dataset_class(debug=False):
    loader = WingPatternDataset(
        image_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )
    print(f"Number of images: {len(loader)}")
    print(f"Number of masks: {len(loader.mask_ids)}")
    print(f"Valid: {loader.valid}")

    for i, image_id in enumerate(loader.image_ids):
        print(f"Image ID: {image_id}")

    # get a sample
    sample = loader[0]
    print(f'image shape: {sample["image"].shape}, image dtype: {sample["image"].dtype}')
    for mask_id, mask in sample["masks"].items():
        print(
            f"Mask ID: {mask_id}, Mask shape: {mask.shape}, "
            f"Mask dtype: {mask.dtype}, Min: {mask.min()}, Max: {mask.max()}"
        )

    # Visualize the first image and its masks
    if debug:
        fig, ax = plt.subplots(figsize=(10, 5), ncols=5)
        ax[0].imshow(sample["image"].permute(1, 2, 0).numpy())
        ax[0].set_title("Input Image")
        ax[0].axis("off")
        for i, (mask_id, mask) in enumerate(sample["masks"].items()):
            ax[i + 1].imshow(mask.squeeze().numpy(), cmap="viridis")
            ax[i + 1].set_title(f"{mask_id}")
            ax[i + 1].axis("off")
        fig.tight_layout()
        plt.show()

        fig.savefig(
            f"{SAVEDIR}/sample_record.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
