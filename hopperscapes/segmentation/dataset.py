import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np

import torch
import zarr
from skimage.io import imread
from torch.utils.data import Dataset
import torchvision.transforms as T

from hopperscapes.configs import SegmentationModelConfigs
from hopperscapes.imageproc import preprocess
from hopperscapes.imageproc.color import convert_to_hsv


def hopper_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to work with the nested masks dictionary.
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


class ConvertToHSV:
    """
    Convert RGB image to HSV.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return convert_to_hsv(image)


class ResizeToLongestSide:
    """
    Resize and pad the image into a square of specified side length.
    """

    def __init__(self, image_side_length: int):
        self.img_side_length = image_side_length

    def __call__(self, image: np.ndarray, anti_aliasing=True) -> np.ndarray:
        # # no need to reshape images with the right dimensions
        if image.shape[:2] == (self.img_side_length, self.img_side_length):
            return image

        # use interpolation order 0 for masks and 1 for images
        is_mask = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
        interpolation_order = 0 if is_mask else 1

        input_arr = preprocess.resize_image(
            image,
            target_side_length=self.img_side_length,
            order=interpolation_order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )
        return preprocess.make_square(input_arr)


class PrepareImageTensor:
    """
    Convert numpy array to scaled tensor with permuted channels.
    """

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply type conversion to float32, normalization, and channel permutation.
        Args:
            image (np.ndarray): Input image array. Assumed to be in (H, W, C) format.
        Returns:
            torch.Tensor: Processed image tensor, in the (C, H, W) format.
        """
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return image_tensor


_TRANSFORMS_REGISTRY = {
    "ConvertToHSV": ConvertToHSV,
    "ResizeToLongestSide": ResizeToLongestSide,
    "PrepareImageTensor": PrepareImageTensor,
}


class SampleTransformer:
    """
    Sample transformer to apply image and mask transformations.

    Args:
        configs (SegmentationModelConfigs): Configurations, including
        "data_transforms" and "mask_transforms".

    """

    def __init__(self, configs: SegmentationModelConfigs):
        self.configs = configs

        self.image_transforms = self._build_pipeline(configs.image_transforms)
        self.mask_transforms = self._build_pipeline(configs.mask_transforms)

    def _build_pipeline(self, transform_configs: Dict) -> Optional[Callable]:
        """
        Build a transformation pipeline based on the provided configurations.

        Args:
            transform_configs (Dict): Dictionary mapping transform names to their parameters.
        Returns:
            Optional[Callable]: A callable transformation pipeline or None if no transforms are specified.
        """
        transform_list = []
        for transform_name, params in transform_configs.items():
            if transform_name in _TRANSFORMS_REGISTRY:
                transform_list.append(_TRANSFORMS_REGISTRY[transform_name](**params))
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

        return T.Compose(transform_list) if transform_list else None

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the transformations to the sample.

        Args:
            sample (Dict[str, torch.Tensor]): Sample containing "image" and "masks".
        Returns:
            Dict[str, torch.Tensor]: Transformed sample with "image" and "masks".
        """
        if self.image_transforms:
            sample["image"] = self.image_transforms(sample["image"])

        # Expecting a tensor of shape (H, W, C) for the image
        assert isinstance(
            sample["image"], torch.Tensor
        ), f"Expected image to be a torch.Tensor, got {type(sample['image'])}"

        processed_masks = {}
        for mask_id, mask_data in sample["masks"].items():
            if self.mask_transforms:
                mask_data = self.mask_transforms(mask_data)

            num_classes = self.configs.out_channels[mask_id]

            if num_classes > 1:
                processed_masks[mask_id] = torch.from_numpy(
                    mask_data
                ).long()  # multi-class
            else:
                mask_tensor = torch.from_numpy(mask_data)
                processed_masks[mask_id] = (
                    (mask_tensor > 0).unsqueeze(0).float()
                )  # binary mask

        sample["masks"] = processed_masks
        return sample


# Zarr-facing Dataset
class HopperZarrDataset(Dataset):
    """
    Pytorch Dataset to interface with an OME-Zarr store prepared according to the
    specification in "data.zarr_store.py".

    Args:
        zarr_path (str): Path to zarr store.
        configs ("SegmentationModelConfigs"): Segmentation model configurations, including
                                                those for the transforms.
        pyramid_level (int): OME-Zarr pyramid level to access.
        include_metadata (bool): Whether to retrieve and include the image metadata
                                from the Zarr store (default is True).
    """

    def __init__(
        self,
        zarr_path: str,
        configs: "SegmentationModelConfigs",
        pyramid_level: int = 0,
        include_metadata: bool = True,
    ):
        super().__init__()

        self.zarr_root = zarr.open(zarr_path, mode="r")
        self.configs = configs
        self.include_metadata = include_metadata

        self.transform = SampleTransformer(configs=self.configs)

        self.image_paths = []

        # Helper method to walk through the Zarr store and find
        # the images. The search string is "/rgb/{pyramid_level}".
        def find_rgb_images(path, obj):
            if isinstance(obj, zarr.core.Array) and path.endswith(
                f"/rgb/{pyramid_level}"
            ):
                self.image_paths.append(path)

        self.zarr_root.visititems(find_rgb_images)

        # Guard against empty dataset
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in the Zarr store at {zarr_path} for pyramid level {pyramid_level}."
            )

        # Query to find masks if they exist:
        # ...

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Fetch the image at index "idx".
        """

        image_path = self.image_paths[idx]
        image_arr = self.zarr_root[image_path][:]

        # Ensure the channel axis is last for the transforms
        if image_arr.ndim == 3 and image_arr.shape[0] in [
            1,
            3,
            4,
        ]:  # channel axis is first
            image_arr = np.transpose(image_arr, (1, 2, 0))  # convert to (H, W, C)

        # load masks from zarr store if they exist
        # ...

        # metadata
        if self.include_metadata:
            try:
                metadata_path = Path(image_path).parent.parent
                metadata_group = self.zarr_root[str(metadata_path)]
                metadata = dict(metadata_group.attrs)
            except Exception as e:
                raise ValueError(
                    f"failed to retrieve metadata from {self.zarr_root} for index {idx}"
                ) from e
        else:
            metadata = {}

        sample = {"image": image_arr, "masks": {}, "id": idx, "meta": metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Directory-tree-facing Dataset
class WingPatternDataset(Dataset):
    """
    Pytorch Dataset to interface with a directory structure containing images and masks.

    Args:
        image_dir (str): Directory containing the images.
        masks_dir (str): Directory containing the masks.
        configs ("SegmentationModelConfigs"): Segmentation model configurations, including
                those for the transforms.
        metadata (Dict[str, Any]): (Optional) Metadata dictionary for the dataset.
    """

    def __init__(
        self,
        image_dir: str,
        masks_dir: str,
        configs: "SegmentationModelConfigs",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.metadata = metadata

        if configs is None:
            self.configs = SegmentationModelConfigs()
        elif isinstance(configs, SegmentationModelConfigs):
            self.configs = configs
        else:
            raise ValueError("configs must be an instance of SegmentationModelConfigs.")

        self.transform = SampleTransformer(configs=self.configs)

        _file_extensions = tuple(self.configs.image_file_exts)
        self.image_ids = [
            f for f in os.listdir(image_dir) if f.endswith(_file_extensions)
        ]

        self.expected_heads = set(self.configs.out_channels.keys())

        self.mask_ids = []
        self.valid = []
        for img_id in self.image_ids:
            # FIXME: re-running glob for each image seems inefficient; we could
            # scan the directory once and try to match the filenames instead.
            is_valid, masks = self._find_matching_masks(img_id, self.masks_dir)
            self.valid.append(is_valid)
            self.mask_ids.append(masks)

        # Discard invalid records
        self.image_ids = [
            img_id for img_id, valid in zip(self.image_ids, self.valid) if valid
        ]
        self.mask_ids = [
            masks for masks, valid in zip(self.mask_ids, self.valid) if valid
        ]

        # Guard against empty dataset
        if len(self.image_ids) == 0:
            raise ValueError("No valid images found and dataset is empty.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.image_ids[idx])

        sample = {
            "image": imread(image_path),
            "masks": {
                mask_id: imread(os.path.join(self.masks_dir, mask_path))
                for mask_id, mask_path in self.mask_ids[idx].items()
            },
            "id": self.image_ids[idx],
            "meta": {},
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _find_matching_masks(
        self, img_id: str, mask_dir
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Find matching masks for a given image ID in the specified mask directories.

        Args:
            img_id (str): Image ID to match masks against.
            mask_dir (str): Directory containing the masks.
        Returns:
            Tuple[bool, Dict[str, str]]: A tuple containing a boolean indicating if the
                                         masks are valid and a dictionary of mask IDs to paths.
        """
        _record_id = img_id.split(".")[0]

        masks_glob_str = os.path.join(mask_dir, "**", f"{_record_id}*")
        masks_glob = glob.glob(masks_glob_str, recursive=True)  # enable recursion

        # No files found
        if len(masks_glob) == 0:
            return False, {}

        # Incomplete or mismatched files found
        if len(masks_glob) != len(self.expected_heads):
            return False, {}

        masks = {}
        for mask_path in masks_glob:
            mask_id = os.path.basename(os.path.dirname(mask_path)).lower()
            if mask_id in self.expected_heads:
                masks[mask_id] = os.path.relpath(
                    mask_path, mask_dir
                )  # keep relative sub-dir

        # Check for typos, e.g. "vein" instead of "veins", etc.
        if len(masks) != len(self.expected_heads):
            return False, {}

        return True, masks
