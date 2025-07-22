import pytest
from pathlib import Path

CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent / "checkpoints" / "HopperNetLite_demo.pth"
)
CONFIGS_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "unified_lite.yaml"
)


@pytest.mark.unit
def test_infer_load_model():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import load_model

    configs = SegmentationModelConfigs.from_yaml(CONFIGS_PATH)
    model = load_model(
        checkpoint_path=CHECKPOINT_PATH, configs=configs, device="cpu"
    )
    assert model is not None


def test_infer_load_model_invalid_checkpoint():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import load_model

    configs = SegmentationModelConfigs.from_yaml(CONFIGS_PATH)
    with pytest.raises(TypeError):
        load_model(checkpoint_path=12345, configs=configs, device="cpu")


    with pytest.raises(ValueError):
        load_model(
            checkpoint_path="non_existent_checkpoint.pth",
            configs=configs,
            device="cpu",
        )

    with pytest.raises(ValueError):
        load_model(
            checkpoint_path=CHECKPOINT_PATH,
            configs=configs,
            device="invalid_device",
        )  # Assuming the device is not valid

    with pytest.raises(ValueError):
        invalid_configs = SegmentationModelConfigs()
        load_model(
            checkpoint_path=CHECKPOINT_PATH,
            configs=invalid_configs,
            device="cpu"
        )  # Assuming the device is not valid


def test_infer_postprocess():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import (
        load_model,
        preprocess_image,
        post_process_predictions,
    )
    import numpy as np
    import torch

    image_arr = np.random.rand(512, 512, 3)
    image_tensor = preprocess_image(
        image_arr, device="cpu"
    )  # converts to tensor and adds batch dimension

    configs = SegmentationModelConfigs.from_yaml(CONFIGS_PATH)
    model = load_model(
        checkpoint_path=CHECKPOINT_PATH, configs=configs, device="cpu"
    )
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    processed_outputs = post_process_predictions(outputs, binary_threshold=0.5)
    assert isinstance(processed_outputs, dict)
