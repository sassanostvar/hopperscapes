from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from skimage.io import imread

from hopperscapes.imageproc import preprocess
from hopperscapes.segmentation.models import HopperNetLite

TEST_DATA_PATH = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)


@pytest.mark.unit
def test_network_forward_pass(debug=False):
    """
    Minimal test for the HopperNetLite model forward pass on sample image.
    """

    # Load a sample image
    sample_img_arr = imread(TEST_DATA_PATH)
    print(f"Sample image shape: {sample_img_arr.shape}")

    # resize and pad to square
    sample_img_arr = preprocess.resize_image(sample_img_arr, target_side_length=512)
    sample_img_arr = preprocess.make_square(sample_img_arr)

    print(f"Padded sample image shape: {sample_img_arr.shape}")
    print(f"Sample image dtype: {sample_img_arr.dtype}")

    # move the channel dimension to the front
    sample_image = np.moveaxis(sample_img_arr, -1, 0)
    print(f"Sample image shape after moveaxis: {sample_image.shape}")

    # Convert to tensor
    sample_image = torch.from_numpy(sample_image).float()

    from hopperscapes.configs import SegmentationModelConfigs

    heads = SegmentationModelConfigs().out_channels

    model = HopperNetLite(
        num_groups=1,  # for GroupNorm
        out_channels=heads,  # use the heads from the config
    )

    # Check if the model can process the sample image
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension
    print(f"Sample image shape after unsqueeze: {sample_image.shape}")
    output = model(sample_image)
    print(f"Output shape: {output['wing'].shape}")
    print(f"Output dtype: {output['wing'].dtype}")
    assert output is not None
    assert isinstance(output, dict)

    if debug:
        # Visualize the output
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 6, 1)
        plt.imshow(sample_img_arr)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 6, 2)
        plt.imshow(output["wing"][0].squeeze().detach().numpy(), cmap="gray")
        plt.title("Output wing")
        plt.axis("off")

        # veins
        plt.subplot(1, 6, 3)
        plt.imshow(output["veins"][0].squeeze().detach().numpy(), cmap="gray")
        plt.title("Output Veins")
        plt.axis("off")

        # spots
        plt.subplot(1, 6, 4)
        plt.imshow(output["spots"][0].squeeze().detach().numpy(), cmap="gray")
        plt.title("Output Spots")
        plt.axis("off")

        # domains
        plt.subplot(1, 6, 5)
        plt.imshow(output["domains"][0].squeeze().detach().numpy(), cmap="gray")
        plt.title("Output Domains")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
