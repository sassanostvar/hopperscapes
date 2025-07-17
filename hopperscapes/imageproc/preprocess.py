"""
Basic image prepprocessing: padding, resizing, etc. to prepare 
images for segmentation and analysis.
"""

from numpy.typing import NDArray


def resize_image(
    image: NDArray,
    target_side_length: int,
    order: int = 0,
    preserve_range: bool = True,
    anti_aliasing: bool = True,
    rebinarize: bool = True,
):
    """
    Resize the image to the specified height and width.
    """
    from skimage.transform import resize

    h, w = image.shape[0], image.shape[1]
    if h > w:
        new_h = target_side_length
        new_w = int(w * (target_side_length / h))
    else:
        new_w = target_side_length
        new_h = int(h * (target_side_length / w))
    resized_image = resize(
        image,
        (new_h, new_w),
        order=order,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
    )

    # re-binarize
    is_binary = isinstance(image.dtype, bool)
    if is_binary and rebinarize:
        resized_image = resized_image > 0

    return resized_image


def make_square(image: NDArray) -> NDArray:
    """
    Pad the image to make it square.
    """
    import numpy as np

    h, w = image.shape[0], image.shape[1]
    if h > w:
        pad = (0, h - w)
    else:
        pad = (w - h, 0)

    # figure out if image is RGB or grayscale
    if len(image.shape) == 3:
        # RGB image
        padded_image = np.pad(
            image,
            ((0, pad[0]), (0, pad[1]), (0, 0)),
            mode="edge",
            # constant_values=image.mean(axis=0),
        )
    else:
        # Grayscale image
        padded_image = np.pad(
            image,
            ((0, pad[0]), (0, pad[1])),
            mode="edge",
            # constant_values=image.mean(),
        )
    return padded_image
