import numpy as np
import pytest
from skimage.io import imread
from skimage.draw import disk

from hopper_vae.imageproc.preprocess import (
    convert_to_hsv,
    resize_image,
    make_square,
    isolate_wing_mask,
    align_wing_with_yaxis,
    center_wing,
)


@pytest.fixture
def sample_rgb_image():
    return np.random.random((100, 100, 3))


@pytest.fixture
def sample_binary_mask():
    mask = np.zeros((100, 100), dtype=bool)
    rr, cc = disk((50, 50), 30)
    mask[rr, cc] = True
    return mask


def test_convert_to_hsv(sample_rgb_image):
    hsv_image = convert_to_hsv(sample_rgb_image)
    assert hsv_image.shape == sample_rgb_image.shape


def test_resize_image(sample_binary_mask):
    resized_mask = resize_image(
        sample_binary_mask, target_side_length=50, anti_aliasing=False
    )
    assert resized_mask.shape == (50, 50)
    assert resized_mask.dtype == sample_binary_mask.dtype


def test_make_square(sample_binary_mask):
    padded_image = make_square(sample_binary_mask)
    assert padded_image.shape[0] == padded_image.shape[1]


def test_isolate_wing_mask(sample_binary_mask):
    isolated_mask = isolate_wing_mask(sample_binary_mask)
    assert isolated_mask.shape == sample_binary_mask.shape


def test_align_wing_with_yaxis(sample_binary_mask):
    aligned_image, aligned_mask, angle = align_wing_with_yaxis(
        sample_binary_mask, sample_binary_mask
    )
    assert aligned_image.shape == sample_binary_mask.shape
    assert aligned_mask.shape == sample_binary_mask.shape
    assert isinstance(angle, float)


def test_center_wing(sample_binary_mask):
    centered_image, centered_mask, shift = center_wing(
        sample_binary_mask, sample_binary_mask
    )
    assert centered_image.shape == sample_binary_mask.shape
    assert centered_mask.shape == sample_binary_mask.shape
    assert isinstance(shift, tuple)
