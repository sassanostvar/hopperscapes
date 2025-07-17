import pytest


@pytest.mark.unit
def test_wing_morphometer_io():
    import numpy as np

    from hopperscapes.morphometry.wing import WingMorphometer

    wrong_configs = {"key": 10}
    with pytest.raises(AssertionError):
        morphometer = WingMorphometer(wrong_configs)

    wrong_input = np.zeros((3, 10, 10))
    with pytest.raises(ValueError):
        morphometer = WingMorphometer()
        morphometer.run(wrong_input)


@pytest.mark.unit
def test_wing_morphometer_on_synthetic_data():
    import numpy as np

    from hopperscapes.morphometry.wing import WingMorphometer

    wing_mask = np.zeros((512, 512), dtype=bool)

    # add wing mask
    wing_mask[200:400, 200:400] = True

    morphometer = WingMorphometer()
    table = morphometer.run(wing_mask)
    assert isinstance(table, dict)


@pytest.mark.unit
def test_wing_morphometer_on_noisy_synthetic_data():
    import numpy as np

    from hopperscapes.morphometry.wing import WingMorphometer, WingMorphometerConfigs

    wing_mask = np.zeros((512, 512), dtype=bool)

    # add wing mask
    wing_mask[200:400, 200:400] = True
    expected_area = np.sum(wing_mask)

    # add noise
    wing_mask[2:10, 2:10] = True
    wing_mask[210:220, 210:220] = False

    configs = WingMorphometerConfigs()
    configs.max_hole_area = 101
    configs.min_speck_area = 101
    morphometer = WingMorphometer(configs)

    assert morphometer.configs.max_hole_area == 101, "Failed to customize configs."
    assert morphometer.configs.min_speck_area == 101, "Failed to customize configs."

    table, _final_mask = morphometer.run(wing_mask, return_mask=True)
    assert isinstance(
        table, dict
    ), f"morphometer output should be of type dict; got {type(table)}"

    # check area
    final_area = table["area"]
    assert np.isclose(
        expected_area, final_area
    ), f"final area {final_area} does not matched expected area: {expected_area}; denoising likely failed"
