"""
Methods to quantify morphological features of wing outlines/areas.
"""


from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_holes, remove_small_objects

import logging

logger = logging.getLogger("WingMorphometryLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()  # stdout
handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

_WING_PROPERTIES = [
    "label",
    "area",
    "area_bbox",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
    "centroid",
    "eccentricity",
    "moments_central",
    "perimeter",
    "perimeter_crofton",
    "orientation",
    "solidity",
]


__all__ = [
    'WingMorphometerConfigs',
    'WingMorphometer'
]

@dataclass
class WingMorphometerConfigs:
    """
    Configurations for wing mask post-processing and denoising.
    """

    max_hole_area: int = 100
    min_speck_area: int = 100
    properties: Tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.properties = _WING_PROPERTIES


class WingMorphometer:
    """
    Pipeline for post-processing and morphometry of wing masks.

    Args:
        configs ("WingMorphometerConfigs"): Configurations for wing mask post-processing and denoising.
    """

    def __init__(self, configs: Optional["WingMorphometerConfigs"] = None):
        if configs is None:
            self.configs = WingMorphometerConfigs()
        else:
            assert isinstance(configs, WingMorphometerConfigs), "Unrecognized input for WingMorphometer configs."
            self.configs = configs

    def _isolate_wing_region(self, wing_mask: np.ndarray):
        """
        Guard against segmentation errors leading to multiple connected components in the wing mask.
        """
        _labeled_mask = label(wing_mask)
        _regions = regionprops(_labeled_mask)

        logger.info("Found %d regions in wing mask", len(_regions))

        if len(_regions) > 1:
            _areas = [r.area for r in _regions]
            _labels = [r.label for r in _regions]
            _max_area = max(_areas)
            largest_region_label = _labels[_areas.index(_max_area)]
            return _labeled_mask == largest_region_label
        else:
            return wing_mask

    def _postprocess_isolated_wing_mask(self, wing_mask: np.ndarray):
        """
        Remove noise from isolated wing mask.
        """
        _specks_removed = remove_small_objects(
            wing_mask, min_size=self.configs.min_speck_area
        )
        _holes_removed = remove_small_holes(
            _specks_removed, area_threshold=self.configs.max_hole_area
        )
        return _holes_removed

    def run(self, wing_mask: np.ndarray, return_mask: bool = False) -> Dict:
        """
        Compute morphological features of the post-processed and denoised wing mask.

        Args:
            wing_mask (np.ndarray): Post-processed and denoised wing mask.

        Outputs:
            (pd.DataFrame): Pandas dataframe of morphological features output by
            skimage.morphology.regionprops_table.
        """
        if wing_mask.ndim != 2:
            raise ValueError(f"Expect 2D wing mask but got {wing_mask.ndim}")

        if wing_mask.dtype != bool:
            logger.warning('The mask is not a binary -- will enforce binary...')
            wing_mask = wing_mask > 0

        processed_mask = self._isolate_wing_region(wing_mask)

        assert (
            np.sum(processed_mask) > 0
        ), "Could not isolate the wing mask; yieled an empty mask."

        logger.info(
            "Found wing connected component with area: %d", np.sum(processed_mask)
        )

        processed_mask = self._postprocess_isolated_wing_mask(processed_mask)

        assert np.sum(processed_mask) > 0, "Denoising yieled an empty mask."

        _labeled = label(processed_mask)

        if return_mask:
            return (
                regionprops_table(_labeled, properties=self.configs.properties),
                _labeled,
            )
        else:
            return regionprops_table(_labeled, properties=self.configs.properties)
