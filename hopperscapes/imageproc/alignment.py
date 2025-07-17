"""
Methods to use segmentation masks for image alignment.
"""

from typing import Tuple

from numpy.typing import NDArray


def align_wing_with_yaxis(
    image: NDArray, mask: NDArray
) -> Tuple[NDArray, NDArray, float]:
    """
    Align the wing with the y-axis by rotating the image and mask.

    Args:
        image (NDArray): The input image.
        mask (NDArray): The binary mask of the region.
    Returns:
        Tuple[NDArray, NDArray, float]: The aligned image, the aligned mask,
        and the angle of rotation in degrees.
    """
    import numpy as np
    from scipy.ndimage import rotate
    from skimage.exposure import rescale_intensity
    from skimage.measure import label, regionprops

    _regions = regionprops(label(mask))

    assert len(_regions) == 1, "Expected exactly one region in the mask."

    # calculate the angle of the major axis
    angle = np.rad2deg(_regions[0].orientation)

    # rotate the image and mask
    aligned_image = rotate(image, -angle, reshape=False)
    aligned_mask = (
        rescale_intensity(
            rotate(
                rescale_intensity(mask.astype(np.uint8), out_range=(0, 1)),
                -angle,
                reshape=False,
            ),
            out_range=(0, 1),
        )
        > 0.5
    )

    return aligned_image, aligned_mask, angle


def shift_arr_to_center(arr: NDArray, shift: Tuple[int, int]) -> NDArray:
    """
    Shift the array to center it based on the provided shift values.

    Args:
        arr (NDArray): The input array.
        shift (Tuple[int, int]): The shift values for rows and columns.
    Returns:
        NDArray: The shifted array.
    """
    import numpy as np

    return np.roll(arr, shift, axis=(0, 1))


def center_region(
    image: NDArray, mask: NDArray
) -> Tuple[NDArray, NDArray, Tuple[int, int]]:
    """
    Move the region's centroid to the image centroid.
    Args:
        image (NDArray): The input image.
        mask (NDArray): The binary mask of the region.
    Returns:
        Tuple[NDArray, NDArray, Tuple[int, int]]: The centered image,
        the centered mask, and the shift applied.
    """
    from skimage.measure import label, regionprops

    mask_centroid = regionprops(label(mask))[0].centroid
    mask_centroid_int = [int(mask_centroid[0]), int(mask_centroid[1])]
    shift = (
        int(image.shape[0] / 2) - mask_centroid_int[0],
        int(image.shape[1] / 2) - mask_centroid_int[1],
    )
    _shifted_image = shift_arr_to_center(image, shift)
    _shifted_mask = shift_arr_to_center(mask, shift)
    return _shifted_image, _shifted_mask, shift
