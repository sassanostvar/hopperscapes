import torch
import pytest
from pathlib import Path
import yaml
from hopperscapes.segmentation.models import (
    ModularHopperNet,
    Encoder,
    Decoder,
    DEFAULT_ENCODER_CONFIGS,
    DEFAULT_DECODER_CONFIGS,
)

DEFAULT_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs" / "models" / "unet.yaml"
LEGACY_CONFIGS_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "models" / "unet_legacy.yaml"
)


@pytest.fixture
def sample_config():
    return {"num_groups": 1, "out_channels": {"wing": 1, "veins": 1}}


@pytest.mark.unit
def test_model_instantiation(sample_config):
    model = ModularHopperNet(**sample_config)
    assert isinstance(model, ModularHopperNet)


@pytest.mark.unit
def test_forward_pass_and_output_shapes(sample_config):
    """
    Tests if the model can perform a forward pass and if all output
    heads have the correct shape.
    """
    model = ModularHopperNet(**sample_config)
    dummy_input = torch.randn(4, 3, 256, 256)  # Use a batch size > 1
    output = model(dummy_input)

    assert isinstance(output, dict)
    # Check that all expected heads are present
    assert set(output.keys()) == set(sample_config["out_channels"].keys())

    # Check the shape of each output head
    for head_name, head_output in output.items():
        expected_channels = sample_config["out_channels"][head_name]
        assert head_output.shape == (4, expected_channels, 256, 256)


@pytest.mark.unit
def test_encoder_skip_connection_count():
    encoder = Encoder(DEFAULT_ENCODER_CONFIGS, num_groups=1)
    dummy_input = torch.randn(1, 3, 256, 256)
    _, skips = encoder(dummy_input)
    assert len(skips) == 2


@pytest.mark.unit
def test_encoder_modularity_with_deeper_config():
    deeper_encoder_configs = {
        "stem": {
            "out_channels": 16,
            "stride": 1,
            "concat": True,
        },
        "encoder0_ds": {
            "out_channels": 32,
            "stride": 2,
            "concat": False,
        },
        "encoder1_conv": {
            "out_channels": 32,
            "stride": 1,
            "concat": True,
        },
        "encoder1_ds": {
            "out_channels": 64,
            "stride": 2,
            "concat": False,
        },
        "encoder2_conv": {
            "out_channels": 64,
            "stride": 1,
            "concat": True,
        },
        "encoder2_ds": {
            "out_channels": 128,
            "stride": 2,
            "concat": False,
        },
    }
    encoder = Encoder(deeper_encoder_configs, num_groups=1)
    dummy_input = torch.randn(1, 3, 256, 256)
    _, skips = encoder(dummy_input)
    assert len(skips) == 3


@pytest.mark.unit
def test_model_with_multiclass_head():
    multiclass_config = {"num_groups": 1, "out_channels": {"wing": 1, "domains": 4}}
    model = ModularHopperNet(**multiclass_config)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output["domains"].shape[1] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.unit
def test_model_on_cuda(sample_config):
    """Tests if the model runs on a CUDA device."""
    device = torch.device("cuda")
    model = ModularHopperNet(**sample_config).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)
    assert output["wing"].device.type == "cuda"



@pytest.mark.parametrize(
    "config_path",
    [
        DEFAULT_CONFIGS_PATH,
        LEGACY_CONFIGS_PATH,
    ],
)
@pytest.mark.unit
def test_architectural_configs_from_yaml(config_path):
    """
    Performs a smoke test to ensure the model can be built and run
    with architectures defined in the YAML configuration files.
    """
    # 1. Load the configuration from the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_params = config["params"]

    # 2. Instantiate the model by unpacking the params dict
    model = ModularHopperNet(**model_params)

    # 3. The test is simply that this forward pass does not crash
    dummy_input = torch.randn(1, 3, 256, 256)
    try:
        output = model(dummy_input)
        assert isinstance(output, dict)
        # Check that the number of output heads is correct
        assert len(output) == len(model_params["out_channels"])
    except Exception as e:
        pytest.fail(
            f"Model forward pass failed for config '{config_path}' with error: {e}"
        )
