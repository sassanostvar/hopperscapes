"""
Prepare augmented datasets and write to disk.
(See _augment_funcs for details of transforms.)
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from hopper_vae.segmentation.augment._augment_funcs import (
    random_aug,
    random_blur_whole_image,
)
from hopper_vae.segmentation.dataset import WingPatternDataset

"""
Checklist:
- [ ] implement tile-wise blur
- [ ] implement focal blur
- [ ] implement tile-wise color inversion
"""


@dataclass
class AugmentConfigs:
    random_blur_prob: float = 0.4
    random_blur_sigma_min: float = 1.0
    random_blur_sigma_max: float = 3.0
    num_augment_per_image: int = 10
    p_brightness: float = 0.5
    brightness_range: Tuple[float] = (0.6, 1.4)
    p_color_saturation: float = 0.5
    color_saturation_range: Tuple[float] = (0.6, 1.4)
    p_channel_shuffle: float = 0.3


def export_augmented_dataset(
    dataset,
    out_root: Union[str, Path],
    n_aug_per_img: int = 2,
    aug_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: Optional[int] = 42,
):
    """
    Save augmented copies of images **and every mask in sample['masks']**.

    - Images:  <out_root>/images/{idx:05d}_aug{k}.png
    - Masks:   <out_root>/masks/<mask_key>/{idx:05d}_aug{k}.tif
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    out_root = Path(out_root)
    img_dir = out_root / "images"
    base_mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)

    if aug_fn is None:
        aug_fn = lambda x: x

    for idx in tqdm(range(len(dataset)), desc="Exporting"):
        sample = dataset[idx]
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()  # HWC

        # ---- images ----
        for k in range(n_aug_per_img):
            img_aug = aug_fn(img_np.copy())
            imageio.imwrite(
                img_dir / f"{idx:05d}_aug{k}.png",
                (img_aug * 255).astype(np.uint8)
                if img_aug.dtype != np.uint8
                else img_aug,
            )

        # ---- masks (dict) ----
        mask_dict: Mapping[str, torch.Tensor | np.ndarray] = sample.get("masks", {})
        for key, mask in mask_dict.items():
            key_dir = base_mask_dir / key
            key_dir.mkdir(parents=True, exist_ok=True)

            # → NumPy, channel-last, uint8/uint16
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()  # tensor → numpy
            if mask.ndim == 3 and mask.shape[0] == 1:  # squeeze C=1
                mask = mask[0]
            mask = mask.astype(np.uint8, copy=False)  # or uint16 if >255 labels

            for k in range(n_aug_per_img):
                imageio.imwrite(key_dir / f"{idx:05d}_aug{k}.tif", mask)

    print(f"✓ {len(dataset)*n_aug_per_img} augmented images exported to {img_dir}")


# ---------------- quick CLI / script usage ----------------
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Create an augmented dataset.")

    arg_parser.add_argument("--images_dir", default=None, help="path to raw images.")
    arg_parser.add_argument(
        "--masks_dir", default=None, help="path to segmentation masks."
    )
    arg_parser.add_argument(
        "--savedir", default=None, help="Path to save the augmented dataset."
    )

    args = arg_parser.parse_args()

    path_to_images = args.images_dir
    path_to_masks = args.masks_dir
    savedir = args.savedir

    # configs
    aug_configs = AugmentConfigs()

    # TODO: consider moving the inline function out of main
    def composite_aug(
        img: np.ndarray, _configs: AugmentConfigs = aug_configs
    ) -> np.ndarray:
        # apply base transforms
        img = random_aug(
            img,
            p_bright=_configs.p_brightness,
            brightness_range=_configs.brightness_range,
            p_color=_configs.p_color_saturation,
            color_saturation_range=_configs.color_saturation_range,
            p_shuffle=_configs.p_channel_shuffle,
        )
        # apply blur
        if np.random.rand() < _configs.random_blur_prob:
            img = random_blur_whole_image(
                img,
                sigma_range=(
                    _configs.random_blur_sigma_min,
                    _configs.random_blur_sigma_max,
                ),
            )
        return img

    ds = WingPatternDataset(path_to_images, path_to_masks)

    export_augmented_dataset(
        ds,
        out_root=savedir,
        n_aug_per_img=aug_configs.num_augment_per_image,
        aug_fn=composite_aug,
    )
