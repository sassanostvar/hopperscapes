"""
Use a pre-trained model in inference on a given image.
"""

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from hopper_vae.segmentation import models


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model from checkpoint.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    except Exception as e:
        raise TypeError(f"Failed to load model at {checkpoint_path} to {device}") from e

    try:
        model_configs = checkpoint["model_configs"]
    except Exception as e:
        raise ValueError(
            f"Could not recover `model_configs' from checkpoint {checkpoint_path}."
        ) from e

    assert hasattr(
        model_configs, "num_groups"
    ), "model_configs has no attribue `num_groups'"
    assert hasattr(
        model_configs, "in_channels"
    ), "model_configs has no attribue `num_groups'"
    assert hasattr(
        model_configs, "out_channels"
    ), "model_configs has no attribue `num_groups'"
    assert hasattr(
        model_configs, "upsample_mode"
    ), "model_configs has no attribue `num_groups'"

    model = models.HopperNetLite(**model_configs)

    try:
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    except Exception as e:
        raise ValueError(
            f"Could not load model_state_dict from {checkpoint_path}"
        ) from e

    model.eval()
    model.to(device)
    return model


def preprocess_image(
    image_path: str,
    device: str = "cpu",
    transform=T.ToTensor(),
) -> torch.Tensor:
    """
    Prepare image for inference.
    """

    from skimage.io import imread

    try:
        image = imread(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image at {image_path}") from e

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def run_inference(
    model: nn.Module, image_tensor: torch.Tensor
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs


def post_process_predictions(
    outputs: Dict[str, torch.Tensor], binary_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Convert logits to binary masks. Use sigmoid for single-class outputs and softmax for multi-class
    outputs.

    Args:
        outputs (Dict[str, torch.Tensor]): Dictionary mapping head_name -> head_logits.
        binary_threshold (float): Threshold value to use to binarize probabilities (default is 0.5).

    Outputs:
        Dict[str, np.ndarray]: Dictionary mapping head_name -> numpy array of head mask(s).
    """
    processed_outputs = {}
    for head_name, logits in outputs.items():
        # multi-class output
        if logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)
            H, W = probs.shape[2], probs.shape[3]
            labels_mask = np.zeros((H, W), dtype=np.int64)
            for class_id in range(probs.shape[1]):
                class_binary_mask = (
                    probs[0, class_id, :, :].detach().cpu().numpy() > binary_threshold
                )
                labels_mask[class_binary_mask] = class_id
            processed_outputs[head_name] = labels_mask

        # single-class output
        elif logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(0))
            binary_mask = probs.squeeze().detach().numpy() > binary_threshold
            processed_outputs[head_name] = binary_mask

    return processed_outputs


def infer(
    image_path: str, checkpoint_path: str, device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Inference pipeline.
    """

    model = load_model(checkpoint_path, device)

    image_tensor = preprocess_image(image_path, device)

    outputs = run_inference(model, image_tensor)

    predictions = post_process_predictions(outputs, binary_threshold=0.5)

    return predictions


def main(args):
    """
    CLI for inference.
    """
    import os
    from skimage.io import imsave

    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    device = args.device
    savepath = args.savepath
    overwrite = args.rewrite
    extension = args.file_extension

    os.makedirs(savepath, exist_ok=overwrite)

    record_id = image_path.split("/")[-1].split(".")[0]

    predictions = infer(image_path, checkpoint_path, device)

    for head_name, prediction in predictions.items():
        head_filepath = os.path.join(
            savepath, "{record_id}_seg_{head_name}.{extension}"
        )
        imsave(head_filepath, prediction)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Apply model to given image and save outputs to disk."
    )

    arg_parser.add_argument("--image-path", required=True, help="Path to image file.")
    arg_parser.add_argument(
        "--checkpoint-path", required=True, help="Path to model checkpoint."
    )
    arg_parser.add_argument(
        "--device", required=True, help="Device to use for inference."
    )
    arg_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite any existing outputs."
    )
    arg_parser.add_argument(
        "--file-extension",
        default="png",
        help="File extension for the outputs (default is PNG).",
    )

    args = arg_parser.parse_args()
    main(args)
