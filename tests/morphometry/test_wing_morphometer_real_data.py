from pathlib import Path

import pytest

WING_MASK_FILEPATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "LD_F_TC_02024_0024_left_forewing_seg_wing.tif"
)


@pytest.mark.unit
def test_wing_morphometer_on_real_data():
    import numpy as np
    from skimage.io import imread

    from hopper_vae.morphometry.wing import WingMorphometer, WingMorphometerConfigs

    wing_mask = imread(WING_MASK_FILEPATH)

    wing_area = np.sum(wing_mask)

    configs = WingMorphometerConfigs()
    configs.max_hole_area = int(wing_area / 100)
    configs.min_speck_area = int(wing_area / 100)

    morphometer = WingMorphometer(configs)
    table = morphometer.run(wing_mask, return_mask=False)

    assert np.isclose(table["area"], wing_area)
