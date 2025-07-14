import numpy as np
import pytest
from skimage import img_as_float32

from hopper_vae.segmentation.augment._augment_funcs import (
    random_blur_whole_image,
    adjust_brightness,
    adjust_color,
    shuffle_channels,
    random_aug,
)

@pytest.fixture
def sample_image():
    """Provide a sample RGB image."""
    return img_as_float32(np.random.random((100, 100, 3)))

@pytest.mark.unit
def test_random_blur_whole_image(sample_image):
    blurred_image = random_blur_whole_image(sample_image)
    assert blurred_image.shape == sample_image.shape

@pytest.mark.unit
def test_adjust_brightness(sample_image):
    brightened_image = adjust_brightness(sample_image, factor=1.2)
    assert brightened_image.dtype == sample_image.dtype

@pytest.mark.unit
def test_adjust_color(sample_image):
    adjusted_color = adjust_color(sample_image, factor=0.8)
    assert adjusted_color.shape == sample_image.shape

@pytest.mark.unit
def test_shuffle_channels(sample_image):
    shuffled_image = shuffle_channels(sample_image)
    assert shuffled_image.shape == sample_image.shape

@pytest.mark.unit
def test_random_aug(sample_image):
    augmented_image = random_aug(sample_image)
    assert augmented_image.shape == sample_image.shape