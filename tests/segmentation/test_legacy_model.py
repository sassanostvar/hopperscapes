import pytest 

@pytest.mark.unit
def test_legacy_model():
    from hopperscapes.segmentation.legacy_models import HopperNetLite
    model = HopperNetLite(
        in_channels=3,
        out_channels={'wing': 1, 'spots': 1},
    )