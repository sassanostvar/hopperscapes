"""
Use a pre-trained model in inference on a given image.
"""

import argparse
from typing import Dict, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from pathlib import Path

from hopperscapes.segmentation import models


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model from checkpoint.
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model on (default is "cpu").
    Returns:
        nn.Module: Loaded model.
    """
    if not isinstance(checkpoint_path, (str, Path)):
        raise TypeError(
            f"checkpoint_path should be a string or Path, got {type(checkpoint_path)}"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    except Exception as e:
        raise ValueError(
            f"Failed to load model at {checkpoint_path} to {device}"
        ) from e

    try:
        model_configs = checkpoint["model_configs"]
    except Exception as e:
        raise ValueError(
            f"Could not recover `model_configs' from checkpoint {checkpoint_path}."
        )

    if not isinstance(model_configs, dict):
        raise TypeError(
            f"model_configs should be a dictionary, got {type(model_configs)}"
        )

    if "num_groups" not in model_configs:
        raise ValueError(
            f"model_configs should contain `num_groups' key, got {model_configs.keys()}"
        )
    if "in_channels" not in model_configs:
        raise ValueError(
            f"model_configs should contain `in_channels' key, got {model_configs.keys()}"
        )
    if "out_channels" not in model_configs:
        raise ValueError(
            f"model_configs should contain `out_channels' key, got {model_configs.keys()}"
        )
    if "upsample_mode" not in model_configs:
        raise ValueError(
            f"model_configs should contain `upsample_mode' key, got {model_configs.keys()}"
        )

    model = models.HopperNetLite(**model_configs)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        raise ValueError(
            f"Could not load model_state_dict from {checkpoint_path}"
        ) from e

    model.eval()
    model.to(device)
    return model


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from a file path.

    Args:
        image_path (Union[str, Path]): Path to the image file.
    Returns:
        np.ndarray: Loaded image as a NumPy array.
    """
    from skimage.io import imread

    try:
        image = imread(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image at {image_path}") from e

    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError(f"Image at {image_path} must be 2D or 3D (grayscale or RGB).")

    return image


def preprocess_image(
    image: np.ndarray,
    device: str = "cpu",
    transform: Callable[[torch.Tensor], torch.Tensor] = T.ToTensor(),
) -> torch.Tensor:
    """
    Prepare image for inference.
    Adds batch dimension and applies transformations.

    Args:
        image (np.ndarray): Input image.
        device (str): Device to use for inference (default is "cpu").
        transform (Callable): Transforms to apply to the image (default is ToTensor).
    Returns:
        torch.Tensor: Preprocessed image tensor ready for inference.
    """

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def run_inference(
    model: nn.Module, image_tensor: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Run inference on the input image tensor using the provided model.

    Args:
        model (nn.Module): The pre-trained model for inference.
        image_tensor (torch.Tensor): Preprocessed image tensor with batch dimension.
    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping head_name to head_logits.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs


def post_process_predictions(
    outputs: Dict[str, torch.Tensor], binary_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Convert logits to binary masks. Use sigmoid for single-class
    outputs and softmax for multi-class outputs.

    Args:
        outputs (Dict[str, torch.Tensor]): Dictionary mapping head_name -> head_logits.
        binary_threshold (float): Threshold value to use to binarize probabilities (default is 0.5).

    Returns:
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

    image_arr = load_image(image_path)

    image_tensor = preprocess_image(image_arr, device)

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
    output_dir = args.output_dir
    overwrite = args.overwrite
    extension = args.file_extension

    os.makedirs(output_dir, exist_ok=overwrite)

    record_id = os.path.splitext(os.path.basename(image_path))[0]

    predictions = infer(image_path, checkpoint_path, device)

    for head_name, prediction in predictions.items():
        head_filepath = os.path.join(
            output_dir, f"{record_id}_seg_{head_name}.{extension}"
        )
        imsave(head_filepath, prediction)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Apply model to given image and save outputs to disk."
    )

    arg_parser.add_argument("--image_path", required=True, help="Path to image file.")
    arg_parser.add_argument(
        "--checkpoint_path", required=True, help="Path to model checkpoint."
    )
    arg_parser.add_argument(
        "--output_dir", required=True, help="Path to output directory."
    )
    arg_parser.add_argument(
        "--device", default="cpu", help="Device to use for inference."
    )
    arg_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite any existing outputs."
    )
    arg_parser.add_argument(
        "--file_extension",
        default="png",
        help="File extension for the outputs (default is PNG).",
    )

    args = arg_parser.parse_args()
    main(args)
