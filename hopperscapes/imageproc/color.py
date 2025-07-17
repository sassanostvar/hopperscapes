"""
Methods for color space conversions.
"""


def convert_to_hsv(image):
    """
    Convert an RGB image to HSV color space.
    """
    from skimage.color import rgb2hsv

    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is RGB
        hsv_image = rgb2hsv(image)
        return hsv_image
    else:
        raise ValueError("Input image must be an RGB image with 3 channels.")
