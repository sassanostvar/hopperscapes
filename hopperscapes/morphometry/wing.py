"""
Wrappers and methods to quantify morphological features of wing outlines/areas.
"""


from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops_table

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


__all__ = ["WingMorphometerConfigs", "WingMorphometer"]


@dataclass
class WingMorphometerConfigs:
    """
    Configurations for wing mask post-processing and denoising.
    """

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
            assert isinstance(
                configs, WingMorphometerConfigs
            ), "Unrecognized input for WingMorphometer configs."
            self.configs = configs

    def run(self, wing_mask: np.ndarray) -> Dict:
        """
        Computes morphological features of binary wing mask (features
        defined in `self.configs.properties`).

        Args:
            wing_mask (np.ndarray): Post-processed and denoised binary wing mask.

        Outputs:
            (pd.DataFrame): Pandas dataframe of morphological features output by
            skimage.morphology.regionprops_table.

        """
        if wing_mask.ndim != 2:
            raise ValueError(f"Expect 2D wing mask but got {wing_mask.ndim}")

        if wing_mask.dtype != bool:
            logger.warning("The mask is not a binary -- will enforce binary...")
            wing_mask = wing_mask > 0

        logger.info("Found wing connected component with area: %d", np.sum(wing_mask))

        _labeled = label(wing_mask)

        return regionprops_table(_labeled, properties=self.configs.properties)
