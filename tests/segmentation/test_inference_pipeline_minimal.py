from pathlib import Path

import pytest


SAMPLE_IMAGE_PATH = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)

@pytest.mark.unit
def test_inference_pipeline_checkpoint_io():
    from hopperscapes.configs import SegmentationModelConfigs
    from hopperscapes.segmentation.infer import load_model

    # no valid checkpoint to load
    with pytest.raises(ValueError):
        _ = load_model("./checkpoint.pth", SegmentationModelConfigs(), device="cpu")


@pytest.mark.unit
def test_inference_pipeline_postprocess():
    import torch

    from hopperscapes.segmentation.infer import preprocess_image, load_image

    image_arr = load_image(SAMPLE_IMAGE_PATH)
    image_tensor = preprocess_image(image_arr, device="cpu")

    assert isinstance(image_tensor, torch.Tensor)


@pytest.mark.unit
def test_inference_pipeline_postprocess_single_class():
    import numpy as np
    import torch

    from hopperscapes.segmentation.infer import post_process_predictions

    logits = torch.tensor([[[[4.0, -4.0], [0.0, 2.0]]]])
    res = post_process_predictions({"mask": logits})
    exp = np.array([[True, False], [False, True]])
    assert np.array_equal(res["mask"], exp)


@pytest.mark.unit
def test_inference_pipeline_postprocess_multi_class():
    import numpy as np
    import torch

    from hopperscapes.segmentation.infer import post_process_predictions

    logits = torch.tensor([[[[2.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [2.0, 2.0]]]])
    res = post_process_predictions({"seg": logits})
    exp = np.array([[0, 0], [1, 1]], dtype=np.int64)
    assert np.array_equal(res["seg"], exp)
