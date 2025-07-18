"""
Methods for postprocessing binary masks.
"""

from numpy.typing import NDArray


def pick_largest_region(binary_mask: NDArray) -> NDArray:
    """
    Isolate the largest connected component in the binary mask.

    Args:
        binary_mask (NDArray): A 2D binary mask where non-zero pixels
                                indicate the regions of interest.

    Returns:
        NDArray: A binary mask with only the largest connected component
                  retained.
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


def denoise_mask(
    binary_mask: NDArray, min_speck_area: int, max_hole_area: int
) -> NDArray:
    """
    Remove noise from isolated wing mask.
    Args:
        binary_mask (NDArray): A 2D binary mask where non-zero pixels
                                indicate the regions of interest.
        min_speck_area (int): Minimum area of specks to be removed.
        max_hole_area (int): Maximum area of holes to be removed.
    Returns:
        NDArray: A denoised binary mask with small specks and holes removed.
    """
    from skimage.morphology import remove_small_holes, remove_small_objects

    denoised_mask = remove_small_objects(binary_mask, min_size=min_speck_area)
    denoised_mask = remove_small_holes(denoised_mask, area_threshold=max_hole_area)
    return denoised_mask


def seeded_watershed(image: NDArray, seed_mask: NDArray) -> NDArray:
    """
    Performs seeded watershed segmentation on a grayscale image given
    an initial segmentation.

    Args:
        image (NDArray): The input microscopy image (grayscale).
        seed_mask (NDArray): A binary mask where non-zero pixels
                             indicate initial seed locations for the spots.

    Returns:
        NDArray: A labeled image where each segmented spot is assigned a
                 unique integer label. The background is typically labeled 0.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label
    from skimage.segmentation import watershed
    from skimage.color import rgb2gray

    if image.ndim == 3 and image.shape[2] == 3:
        image_grayscale = rgb2gray(image)
    elif image.ndim == 2:
        image_grayscale = image
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")

    inverted_seed_mask = (seed_mask == 0).astype(np.uint8) * 255
    distance = distance_transform_edt(inverted_seed_mask)
    markers = label(seed_mask)[0]

    # Assuming any non-zero pixel in the grayscale image is
    # part of the region of interest.
    watershed_mask = image_grayscale > 0

    labels = watershed(-distance, markers, mask=watershed_mask)

    return labels


def seeded_watershed_local_maxima(
    objects_mask: NDArray,
    min_distance: int = 10,
    exclude_border: bool = False,
) -> NDArray:
    """
    Performs a seeded watershed segmentation on an initial binary mask
    that identifies the general regions of blob-like features. Computes
    a distance transform on the initial mask, finds distinct local maxima
    within the distance transform, and uses these as markers for the watershed
    algorithm. This approach is robust for separating touching or overlapping
    circular/blob-like objects.

    Args:
        objects_mask (NDArray): A binary mask where non-zero pixels
                                        indicate the general foreground regions
                                        containing the spots. This mask is used
                                          to compute the distance transform.

        min_distance (int): Minimum distance between local maxima in pixels.
                                Default is 10.

        exclude_border (bool): If True, local maxima on the border of the mask
                               will be excluded. Default is False.

    Returns:
        NDArray: A labeled image where each segmented spot is assigned a
                 unique integer label. The background is typically labeled 0.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    distance = distance_transform_edt(objects_mask)

    local_max_coords = peak_local_max(
        distance,
        min_distance=min_distance,
        exclude_border=exclude_border,
        labels=objects_mask,
    )

    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(local_max_coords.T)] = True
    labeled_markers, num_features = label(markers)

    labels = watershed(-distance, labeled_markers, mask=objects_mask)

    return labels
