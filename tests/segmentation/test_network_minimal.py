import pytest
from torchinfo import summary

from hopper_vae.configs import SegmentationModelConfigs
from hopper_vae.segmentation.models import HopperNetLite


@pytest.mark.unit
def test_network_minimal():
    """
    Minimal test for the HopperNetLite model.
    """
    c = SegmentationModelConfigs()
    model = HopperNetLite(out_channels=c.out_channels, num_groups=c.num_groups)
    summary(model, input_size=(1, 3, 512, 512))
    assert model is not None
    assert isinstance(model, HopperNetLite)
