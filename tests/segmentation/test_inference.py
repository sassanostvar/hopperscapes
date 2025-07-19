import pytest
from pathlib import Path

CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent / "checkpoints" / "HopperNetLite_demo.pth"
)


@pytest.mark.unit
def test_infer_load_model():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import load_model

    model = load_model(CHECKPOINT_PATH, SegmentationModelConfigs(), device="cpu")
    assert model is not None


def test_infer_load_model_invalid_checkpoint():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import load_model

    with pytest.raises(TypeError):
        load_model(12345, SegmentationModelConfigs(), device="cpu")

    with pytest.raises(ValueError):
        load_model("non_existent_checkpoint.pth", SegmentationModelConfigs(), device="cpu")

    with pytest.raises(ValueError):
        load_model(
            CHECKPOINT_PATH, SegmentationModelConfigs(), device="invalid_device"
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

    model = load_model(
        CHECKPOINT_PATH, SegmentationModelConfigs(), device="cpu"
    )
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    processed_outputs = post_process_predictions(outputs, binary_threshold=0.5)
    assert isinstance(processed_outputs, dict)
