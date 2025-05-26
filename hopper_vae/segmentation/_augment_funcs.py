"""
    Methods for augmenting transmission light microscopy images 
    of insec wings.
"""

import random
from typing import Any, Dict, Tuple

from skimage import img_as_float32, img_as_ubyte, color
import matplotlib.pyplot as plt
import random

from hopper_vae.segmentation.data_io import WingPatternDataset

import numpy as np
from skimage.filters import gaussian


def random_blur_whole_image(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (0.5, 2.0),
    channel_axis: int = -1,
) -> np.ndarray:
    """
    Apply random Gaussian blur to the image.

    Args:
        image (np.ndarray): Input image.
        sigma_range (Tuple[float, float]): Range of sigma values for Gaussian blur.

    Returns:
        np.ndarray: Blurred image.
    """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    return gaussian(image, sigma=sigma, channel_axis=channel_axis)


def random_blur_tile(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (0.5, 2.0),
    channel_axis: int = -1,
) -> Dict[str, Any]:
    """
    Apply random Gaussian blur to a tile of the image.

    Args:
        image (np.ndarray): Input image.
        sigma_range (Tuple[float, float]): Range of sigma values for Gaussian blur.

    Returns:
        np.ndarray: Blurred tile of the image.
    """
    h, w = image.shape[:2]
    x1 = random.randint(0, w - 1)
    y1 = random.randint(0, h - 1)
    x2 = random.randint(x1 + 1, w)
    y2 = random.randint(y1 + 1, h)

    tile_bounds = (x1, y1, x2, y2)

    tile = image[y1:y2, x1:x2]
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    blurred_tile = gaussian(tile, sigma=sigma, channlel_axis=channel_axis)
    image[y1:y2, x1:x2] = blurred_tile

    # return sigma, tile_bounds, image
    return {
        "sigma": sigma,
        "tile_bounds": tile_bounds,
        "image": image,
    }


# ---------- core helpers ----------
def _ensure_float(img):
    """Return float32 image in [0,1]."""
    return img_as_float32(img)


# --- replace inside hopper_aug_sk.py ---
def _restore_dtype(img, orig_dtype):
    """Cast back to original dtype (uint8 -> uint8, float -> float32)."""
    if orig_dtype == np.uint8:
        return img_as_ubyte(img)  # keeps [0-255] scale
    return img.astype(orig_dtype, copy=False)


def adjust_brightness(img, factor):
    f = _ensure_float(img) * factor
    return _restore_dtype(np.clip(f, 0, 1), img.dtype)


def adjust_color(img, factor):
    hsv = color.rgb2hsv(_ensure_float(img))
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)
    out = color.hsv2rgb(hsv)
    return _restore_dtype(out, img.dtype)


def shuffle_channels(img):
    if img.ndim != 3 or img.shape[2] < 3:
        return img  # grayscale or single-channel
    perm = np.random.permutation(img.shape[2])
    return img[..., perm]


# ---------- one-shot random recipe ----------
def random_aug(img, p_bright=0.5, p_color=0.5, p_shuffle=0.3):
    if random.random() < p_bright:
        img = adjust_brightness(img, random.uniform(0.6, 1.4))
    if random.random() < p_color:
        img = adjust_color(img, random.uniform(0.6, 1.4))
    if random.random() < p_shuffle:
        img = shuffle_channels(img)
    return img


# ---------- quick grid viewer ----------
def show_grid(imgs, ncols=6, figsize=6):
    nrows = int(np.ceil(len(imgs) / ncols))
    plt.figure(figsize=(figsize, figsize * nrows / ncols))
    for i, im in enumerate(imgs, 1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(im, cmap=None if im.ndim == 3 else "gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    dataset = WingPatternDataset(image_dir="data/raw/train", masks_dir="data/raw/train")
    print(f"dataset length: {len(dataset)}")

    show_grid(
        [
            random_aug(dataset[i]["image"].permute(1, 2, 0).cpu().numpy())
            for i in range(12)
        ],
        ncols=3,
        figsize=6,
    )


if __name__ == "__main__":
    main()
