import glob
import os
from typing import Dict, Optional, Tuple

import torch
from numpy.typing import ArrayLike
from skimage.io import imread
from torch.utils.data import Dataset

from hopper_vae import configs
from hopper_vae.imageproc import preprocess

_GLOBAL_CONFIGS = configs.SegmentationModelConfigs()

SQUARE_IMAGE_DIMS = _GLOBAL_CONFIGS.square_image_size


def hopper_collate_fn(batch):
    """
    Custom collate function to work with the nested masks dictionary
    """
    images = []
    masks_dicts = []
    ids = []

    for sample in batch:
        images.append(sample["image"])
        masks_dicts.append(sample["masks"])
        ids.append(sample["id"])

    images = torch.stack(images, dim=0)  # [N, C, H, W]

    mask_keys = masks_dicts[0].keys()

    collated_masks = {}
    for mask_key in mask_keys:
        collated_masks[mask_key] = torch.stack(
            [md[mask_key] for md in masks_dicts], dim=0
        )

    return {"image": images, "masks": collated_masks, "ids": ids}


class ResizeToLongestSide:
    """
    Resize image to given dimensions and pad to make square.
    """

    def __init__(self, target_image_dims: Tuple = SQUARE_IMAGE_DIMS):
        self.img_side_length = target_image_dims

    def __call__(self, tensor: torch.Tensor, anti_aliasing=True) -> torch.Tensor:
        # no need to transform images with the right dimensions
        if tensor[-2:].shape == (self.img_side_length, self.img_side_length):
            return tensor

        # reorder the channels from (C, H, W) to (H, W, C)
        img_arr = tensor.permute(1, 2, 0).cpu().numpy()

        # TODO: do we need to change the interpolation order
        # for images vs. masks? also, we should consider re-binarizing
        # the masks after resizing
        img_arr = preprocess.resize_image(
            img_arr,
            target_side_length=self.img_side_length,
            order=0,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )
        img_arr = preprocess.make_square(img_arr)

        # reorder the channels from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        return img_tensor


class SharedTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = sample["image"]
        masks = sample["masks"]

        if self.base_transform:
            img = self.base_transform(img)
            for mask_id in masks.keys():
                masks[mask_id] = self.base_transform(masks[mask_id])

        sample["image"] = img
        sample["masks"] = masks
        return sample


class WingPatternDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        masks_dir: str,
        # metadata_dict: Dict[str, Any],
        transform: Optional[callable] = ResizeToLongestSide(),
        config=_GLOBAL_CONFIGS,
    ):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        # self.metadata = metadata_dict
        self.transform = SharedTransform(transform) if transform else None
        self.config = config
        self.valid = []
        self.image_ids = [
            f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ]

        self.expected_heads = set(self.config.out_channels.keys())
        self.mask_ids = []
        for img_id in self.image_ids:
            # TODO: re-running glob for each image seems inefficient; we could
            # scan the directory once and try to match the filenames instead.
            is_valid, masks = self.find_matching_masks(img_id, self.masks_dir)
            self.valid.append(is_valid)
            self.mask_ids.append(masks)

        # discard invalid records
        self.image_ids = [
            img_id for img_id, valid in zip(self.image_ids, self.valid) if valid
        ]
        self.mask_ids = [
            masks for masks, valid in zip(self.mask_ids, self.valid) if valid
        ]

    def find_matching_masks(self, img_id: str, mask_dir) -> Tuple[bool, Dict[str, str]]:
        """
        Find matching masks for a given image ID in the specified mask directories.
        """
        _record_id = img_id.split(".")[0]

        masks_glob_str = os.path.join(mask_dir, "**", f"{_record_id}*")
        masks_glob = glob.glob(masks_glob_str, recursive=True)  # enable recursion

        # no files found
        if len(masks_glob) == 0:
            return False, {}

        # incomplete or mismatched files found
        if len(masks_glob) != len(self.expected_heads):
            return False, {}

        masks = {}
        for mask_path in masks_glob:
            mask_id = os.path.basename(os.path.dirname(mask_path)).lower()
            if mask_id in self.expected_heads:
                masks[mask_id] = os.path.relpath(
                    mask_path, mask_dir
                )  # keep relative sub-dir

        # check for typos, e.g. "vein" instead of "veins", etc.
        if len(masks) != len(self.expected_heads):
            return False, {}

        return True, masks

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        TODO: consider moving some of these outside to reduce overhead on __getitem__
        """
        img_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, img_id)

        mask_ids_dict = self.mask_ids[idx]
        masks_paths = {
            mask_id: os.path.join(self.masks_dir, mask_path)
            for mask_id, mask_path in mask_ids_dict.items()
        }

        image = imread(image_path)
        image = image.astype(float) / 255.0

        # convert to HSV
        if self.config.convert_to_hsv:
            image = preprocess.convert_to_hsv(image)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        masks = {}
        for mask_id, mask_path in masks_paths.items():
            mask = imread(mask_path)
            masks[mask_id] = torch.from_numpy(mask).unsqueeze(0).float()

        sample = {
            "image": image_tensor,
            "masks": masks,
            # "meta": meta,
            "id": img_id,
        }

        if self.transform:
            sample = self.transform(sample)

        final_masks = {}
        for mask_id, mask_tensor in sample["masks"].items():
            if self.config.out_channels[mask_id] > 1:
                final_masks[mask_id] = mask_tensor.squeeze(
                    0
                ).long()  # TODO: is this correct?
            else:
                final_masks[mask_id] = mask_tensor

        sample["masks"] = final_masks

        return sample


if __name__ == "__main__":
    pass
