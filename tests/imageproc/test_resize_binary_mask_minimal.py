import numpy as np
import pytest


@pytest.mark.unit
def test_reisze_synthetic_binary_mask():
    """
    Test resize image on synthetic data and verify mask area fraction before and after.
    """
    from hopper_vae.imageproc.preprocess import resize_image

    test_image = np.zeros((1200, 1200), dtype=bool)
    test_image[300:900, 300:900] = True

    frac_area_before = np.sum(test_image) / (test_image.shape[0] * test_image.shape[1])

    resized_image = resize_image(
        test_image, target_side_length=512, anti_aliasing=False
    )

    frac_area_after = np.sum(resized_image) / (
        resized_image.shape[0] * resized_image.shape[1]
    )

    print(f"frac area before: {frac_area_before}, after: {frac_area_after}")

    assert test_image.dtype == resized_image.dtype
    assert np.isclose(frac_area_before, frac_area_after)
