import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage.io import imread

from hopperscapes.segmentation.dataset import preprocess

from pathlib import Path

TEST_DATA_PATH = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)
TEST_SEG_PATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "LD_F_TC_02024_0024_left_forewing_seg_veins.tif"
)


@pytest.mark.unit
def test_preprocess(debug=False):
    """
    Test the preprocess function.
    """

    # Load a sample image
    sample_img_arr = imread(TEST_DATA_PATH)
    print(f"Sample image shape: {sample_img_arr.shape}")
    print(f"sample img arr dtype: {sample_img_arr.dtype}")
    # load a sample segmentation
    sample_seg_arr = imread(TEST_SEG_PATH)
    print(f"Sample segmentation shape: {sample_seg_arr.shape}")
    print(f"sample seg arr dtype: {sample_seg_arr.dtype}")

    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(sample_img_arr)
        ax[0].set_title("Sample Image")

        ax[1].imshow(sample_img_arr)
        ax[1].imshow(sample_seg_arr, alpha=0.5)
        ax[1].set_title("Sample Segmentation")

        plt.show()

    # Preprocess the image
    resized_img = preprocess.resize_image(sample_img_arr, target_side_length=512)
    resized_img = preprocess.make_square(resized_img)
    print(f"Resized image shape: {resized_img.shape}")
    resized_seg = preprocess.resize_image(sample_seg_arr, target_side_length=512)
    resized_seg = preprocess.make_square(resized_seg)
    print(f"Resized segmentation shape: {resized_seg.shape}")

    # print the types and bounds of the images
    print(f"resized_img dtype: {resized_img.dtype}")
    print(f"resized_img min: {resized_img.min()}")
    print(f"resized_img max: {resized_img.max()}")
    print(f"resized_seg dtype: {resized_seg.dtype}")
    print(f"resized_seg min: {resized_seg.min()}")
    print(f"resized_seg max: {resized_seg.max()}")

    # plot after resizing
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(resized_img)
        ax[0].set_title("Resized Image")

        ax[1].imshow(resized_img)
        ax[1].imshow(resized_seg, alpha=0.5)
        ax[1].set_title("Resized Segmentation")

        plt.show()
