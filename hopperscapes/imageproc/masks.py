"""
Methods for postprocessing binary masks.
"""

from numpy.typing import NDArray


def pick_largest_region(binary_mask: NDArray) -> NDArray:
    """
    Isolate the largest connected component in the binary mask.
    """
    from skimage.measure import label, regionprops

    if binary_mask.ndim != 2:
        raise ValueError("Input binary mask must be a 2D array.")

    if not binary_mask.dtype == bool:
        raise ValueError("Input binary mask must be of type: boolean.")

    labeled = label(binary_mask)
    regions = regionprops(labeled)

    if not regions:
        raise ValueError("No regions found in the mask.")

    largest_region = max(regions, key=lambda r: r.area)
    return labeled == largest_region.label
