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

    from hopperscapes.morphometry.wing import WingMorphometer
    from hopperscapes.imageproc.masks import pick_largest_region, denoise_mask

    wing_mask = imread(WING_MASK_FILEPATH)

    wing_area = np.sum(wing_mask)

    # denoise
    wing_mask = wing_mask > 0
    wing_mask = pick_largest_region(wing_mask)
    wing_mask = denoise_mask(
        wing_mask,
        min_speck_area=int(wing_area / 100),
        max_hole_area=int(wing_area / 100),
    )

    morphometer = WingMorphometer()
    table = morphometer.run(wing_mask)

    assert np.isclose(table["area"], wing_area)
