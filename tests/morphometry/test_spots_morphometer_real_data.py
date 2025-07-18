from pathlib import Path

import pytest

SPOTS_MASK_FILEPATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "LD_F_TC_02024_0024_left_forewing_seg_spots.tif"
)


@pytest.mark.unit
def test_spots_morphometer_on_spots_segmentations(debug=False):
    from skimage.io import imread

    from hopperscapes.morphometry.spots import SpotsMorphometer

    spots_mask = imread(SPOTS_MASK_FILEPATH)
    
    morphometer = SpotsMorphometer()

    table = morphometer.run(spots_mask)

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=3)
        ax[0].hist(table["area"])
        ax[0].set_title("area")
        ax[1].hist(table["perimeter_crofton"])
        ax[1].set_title("perimeter_crofton")
        ax[2].hist(table["eccentricity"])
        ax[2].set_title("eccentricity")
        fig.tight_layout()
        plt.show()

    assert len(table["label"]) > 10
