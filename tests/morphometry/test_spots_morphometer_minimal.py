import pytest


@pytest.mark.unit
def test_spots_morphometer_io():
    import numpy as np

    from hopperscapes.morphometry.spots import SpotsMorphometer

    wrong_configs = {"key": 10}
    with pytest.raises(AssertionError):
        morphometer = SpotsMorphometer(wrong_configs)

    wrong_input = np.zeros((3, 10, 10))
    with pytest.raises(ValueError):
        morphometer = SpotsMorphometer()
        morphometer.run(wrong_input)


@pytest.mark.unit
def test_spots_morphometer_on_synthetic_data():
    import numpy as np

    from hopperscapes.morphometry.spots import SpotsMorphometer

    spots_mask = np.zeros((512, 512), dtype=bool)

    # add spots
    spots_mask[100:150, 100:150] = True
    spots_mask[200:250, 200:250] = True
    spots_mask[300:350, 300:350] = True

    morphometer = SpotsMorphometer()
    table = morphometer.run(spots_mask)

    assert isinstance(table, dict)
    assert len(table["label"]) == 3


@pytest.mark.unit
def test_spots_morphometer_on_synthetic_data():
    import numpy as np

    from hopperscapes.morphometry.spots import SpotsMorphometer, SpotsMorphometerConfigs

    spots_mask = np.zeros((512, 512), dtype=bool)

    # add spots
    spots_mask[100:150, 100:150] = True
    spots_mask[200:250, 200:250] = True
    spots_mask[300:350, 300:350] = True
    expected_area = 50 * 50

    # add specks
    spots_mask[0:10, 0:10] = True
    spots_mask[400:410, 400:410] = True
    speck_area = 11 * 11

    # add holes
    spots_mask[110:115, 110:115] = False
    spots_mask[210:215, 210:215] = False
    spots_mask[310:315, 310:315] = False
    hole_area = 6 * 6

    configs = SpotsMorphometerConfigs()
    configs.max_hole_area = hole_area + 1
    configs.min_speck_area = speck_area + 1

    morphometer = SpotsMorphometer(configs)
    table = morphometer.run(spots_mask)

    assert isinstance(table, dict)
    assert len(table["label"]) == 3
    for area in table["area"]:
        assert area == expected_area
