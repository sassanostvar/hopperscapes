"""
Use a pre-trained model in inference on a given image.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from hopperscapes.segmentation import models
from hopperscapes.configs import SegmentationModelConfigs
from hopperscapes.segmentation.dataset import SampleTransformer


def load_model(
        checkpoint_path: str, 
        configs: SegmentationModelConfigs,
        device: str = "cpu") -> nn.Module:
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

    if not isinstance(configs, SegmentationModelConfigs):
        raise TypeError(
            f"configs should be an instance of SegmentationModelConfigs, got {type(configs)}"
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

    model_configs = {
        "num_groups": configs.num_groups,
        "in_channels": configs.in_channels,
        "out_channels": configs.out_channels,
        "upsample_mode": configs.upsample_mode,
    }

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
    configs: Optional[SegmentationModelConfigs] = None,
) -> torch.Tensor:
    """
    Prepare image for inference.
    Adds batch dimension and applies transformations.

    Args:
        image (np.ndarray): Input image.
        device (str): Device to use for inference (default is "cpu").
        configs (Optional[SegmentationModelConfigs]): Model configurations for preprocessing.
    Returns:
        torch.Tensor: Preprocessed image tensor ready for inference.
    """

    if configs is None:
        configs = SegmentationModelConfigs()
    elif isinstance(configs, SegmentationModelConfigs):
        pass
    else:
        raise ValueError("configs should be an instance of SegmentationModelConfigs.")

    # verify channels

    _in_channels = configs.in_channels

    if _in_channels > 1 and len(image.shape) == 2:
        raise ValueError(
            f"Image at {image.shape} must be 3D (grayscale or RGB) for {_in_channels} channels."
        )

    if _in_channels > 1 and (image.shape[2] != _in_channels):
        raise ValueError(
            f"Image at {image.shape} must have {_in_channels} channels, got {image.shape[2]}."
        )
    
    sample_transformer = SampleTransformer(configs)
    sample = {
        "image": image,
        "masks": {},  # No mask for inference
        'id': 0  # Dummy ID for inference
    }

    return sample_transformer(sample)["image"].unsqueeze(0).to(device)


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
            labels_mask = torch.argmax(logits, dim=1)
            processed_outputs[head_name] = labels_mask.squeeze(0).detach().cpu().numpy()

        # single-class output
        elif logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(0))
            binary_mask = probs.squeeze().detach().numpy() > binary_threshold
            processed_outputs[head_name] = binary_mask

    return processed_outputs


def infer(
    image_arr: np.ndarray,
    configs: Optional["SegmentationModelConfigs"] = None,
    model: nn.Module = None,
    device: str = "cpu",
    binary_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Inference pipeline.
    """

    image_tensor = preprocess_image(image_arr, device, configs)

    outputs = run_inference(model, image_tensor)

    predictions = post_process_predictions(outputs, binary_threshold=binary_threshold)

    return predictions


def main(args):
    """
    Use CL args to run inference on a given image.
    Saves the outputs to the specified directory.
    """
    import os

    from skimage.io import imsave

    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    if args.configs_path:
        configs_path = args.configs_path
    else:
        configs_path = None
    device = args.device
    output_dir = args.output_dir
    overwrite = args.overwrite
    extension = args.file_extension

    # verify configs
    if configs_path is not None:
        from hopperscapes.configs import SegmentationModelConfigs

        configs = SegmentationModelConfigs.from_yaml(configs_path)
    else:
        configs = None

    os.makedirs(output_dir, exist_ok=overwrite)

    record_id = os.path.splitext(os.path.basename(image_path))[0]

    model = load_model(checkpoint_path, configs, device)

    image_arr = load_image(image_path)

    predictions = infer(image_arr, configs, model, device)

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
        "--configs_path",
        default=None,
        help="Path to model configs file. If not provided, defaults to SegmentationModelConfigs.",
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
        default="tiff",
        help="File extension for the outputs (default is TIFF).",
    )

    args = arg_parser.parse_args()
    main(args)
