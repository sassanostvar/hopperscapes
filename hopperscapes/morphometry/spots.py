"""
Methods to quantify morphological features of spot patterns.
"""


from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops_table

import logging

logger = logging.getLogger("SpotsMorphometryLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()  # stdout
handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

_SPOTS_PROPERTIES = [
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
    "euler_number",
]

__all__ = ["SpotsMorphometerConfigs", "SpotsMorphometer"]


@dataclass
class SpotsMorphometerConfigs:
    """
    Configurations for spots mask post-processing and denoising.
    """

    properties: Tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.properties = _SPOTS_PROPERTIES


class SpotsMorphometer:
    """
    Pipeline for post-processing and morphometry of spots masks.

    Args:
        configs ("SpotsMorphometerConfigs"): Configurations for spots mask post-processing and denoising.
    """

    def __init__(self, configs: Optional["SpotsMorphometerConfigs"] = None):
        if configs is None:
            self.configs = SpotsMorphometerConfigs()
        else:
            assert isinstance(
                configs, SpotsMorphometerConfigs
            ), "Unrecognized input for SpotsMorphometerConfigs configs."
            self.configs = configs

    def run(self, binary_spots_mask: np.ndarray) -> Dict:
        """
        Compute morphological features of the post-processed and denoised spots mask.

        Args:
            binary_spots_mask (np.ndarray): Post-processed and denoised binary spots mask.

        Outputs:
            (pd.DataFrame): Pandas dataframe of morphological features output by
            skimage.morphology.regionprops_table.
        """
        if binary_spots_mask.ndim != 2:
            raise ValueError(f"Expect 2D spots mask but got {binary_spots_mask.ndim}")

        if binary_spots_mask.dtype != bool:
            logger.warning("The mask is not a binary -- will enforce binary...")
            binary_spots_mask = binary_spots_mask > 0
        else:
            binary_spots_mask = binary_spots_mask

        _labeled = label(binary_spots_mask)

        return regionprops_table(_labeled, properties=self.configs.properties)
