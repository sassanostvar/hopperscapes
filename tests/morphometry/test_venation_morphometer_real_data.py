import pytest
from pathlib import Path

VENATION_NETWORK_MASK_PATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "LD_F_TC_02024_0024_left_forewing_seg_veins.tif"
)


def test_venation_network_morphometer():
    from hopperscapes.morphometry.venation import VenationMorphometer
    from skimage.io import imread

    mask = imread(VENATION_NETWORK_MASK_PATH) > 0
    morphometer = VenationMorphometer()
    graph = morphometer.run(mask, prune=True)
