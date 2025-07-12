import numpy as np
from scipy.ndimage import rotate
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb


def convert_to_hsv(image):
    """
    Convert an RGB image to HSV color space.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is RGB
        hsv_image = rgb2hsv(image)
        return hsv_image
    else:
        raise ValueError("Input image must be an RGB image with 3 channels.")



def resize_image(image, target_side_length=512, anti_aliasing=True):
    """
    Resize the image to the specified height and width.
    """
    h, w = image.shape[0], image.shape[1]
    if h > w:
        new_h = target_side_length
        new_w = int(w * (target_side_length / h))
    else:
        new_w = target_side_length
        new_h = int(h * (target_side_length / w))
    resized_image = resize(
        image, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=anti_aliasing
    )
    return resized_image


def make_square(image):
    """
    Make the image square by padding it with zeros.
    """
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


def isolate_wing_mask(prediction):
    """
    Isolate the largest connected component in the binary mask.
    """
    labeled = label(prediction > 0)
    regions = regionprops(labeled)

    if not regions:
        raise ValueError("No regions found in the mask.")

    largest_region = max(regions, key=lambda r: r.area)
    return regions == largest_region.label


def align_wing_with_yaxis(masked_gs_img, _mask):
    """
    Align the wing with the y-axis by rotating the image and mask.
    """
    angle = np.rad2deg(regionprops(label(_mask))[0].orientation)
    aligned_gs_img = rotate(masked_gs_img, -angle, reshape=False)
    aligned_mask = rotate(_mask, -angle, reshape=False)
    return aligned_gs_img, aligned_mask, angle


def center_wing(masked_gs_img, _mask):
    """
    Center the wing in the image.
    """
    mask_centroid = regionprops(label(_mask))[0].centroid
    mask_centroid_int = [int(mask_centroid[0]), int(mask_centroid[1])]
    shift = (
        int(masked_gs_img.shape[0] / 2) - mask_centroid_int[0],
        int(masked_gs_img.shape[1] / 2) - mask_centroid_int[1],
    )
    _shifted_image = np.roll(masked_gs_img, shift, axis=(0, 1))
    _shifted_mask = np.roll(_mask, shift, axis=(0, 1))
    return _shifted_image, _shifted_mask, shift