"""
Apply a pretrained checkpoint to sample data and save the output visualizations,
comparing ground truth and predicted masks.
"""

import argparse
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

from hopperscapes.configs import SegmentationModelConfigs
from hopperscapes.segmentation.dataset import WingPatternDataset
from hopperscapes.segmentation.infer import infer, load_model


def create_mask_overlay(
    mask: np.ndarray,
    color: Union[List[float], Dict[int, List[float]]],
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Converts a binary mask into a colored RGBA overlay.
    Args:
        mask (np.ndarray): Binary mask array.
        color (Union[List[float], Dict[int, List[float]]]): RGB color to apply to the mask.
            If mask is binary, color should be a list of 3 RGB values.
            If mask is multi-class, color should be a dictionary mapping class IDs to RGB lists.
        alpha (float): Transparency level for the overlay.
    Returns:
        np.ndarray: RGBA image where the mask is colored.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got shape {mask.shape}.")

    # Create an empty RGBA image
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    if mask.dtype == bool:
        assert (
            isinstance(color, list) and len(color) == 3
        ), "Color must be a list of 3 RGB values for binary masks."
        # Apply the color and alpha to the mask only
        overlay[mask] = [*color, alpha]

    elif np.issubdtype(mask.dtype, np.integer):
        assert isinstance(
            color, dict
        ), "Color must be a dictionary for multi-class masks."
        for class_id, class_color in color.items():
            if class_id == 0:
                continue
            assert (
                isinstance(class_color, list) and len(class_color) == 3
            ), f"Color for class {class_id} must be a list of 3 RGB values."
            overlay[mask == class_id] = [*class_color, alpha]

    else:
        raise ValueError(
            "Mask must either be a binary or multi-class mask with integer values."
        )

    return overlay


def plot_inference_results(
    original_image: np.ndarray,
    ground_truth_masks: Dict[str, np.ndarray],
    predicted_masks: Dict[str, np.ndarray],
    config: Dict,
):
    """
    Plots a grid of ground truth vs. predicted masks.
    """

    rc_params = {
        "font.size": 5,
        "axes.titlesize": 5,
        "axes.labelsize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    }

    with plt.rc_context(rc_params):
        heads = list(predicted_masks.keys())
        fig, axes = plt.subplots(nrows=2, ncols=len(heads), figsize=(12, 6))

        for col_idx, head_name in enumerate(heads):
            # Row 0: Ground truths
            ax_gt = axes[0, col_idx]
            ax_gt.imshow(original_image)
            if head_name in ground_truth_masks:
                gt_mask = ground_truth_masks[head_name]
                gt_color = config.heads[head_name]["color"]
                overlay = create_mask_overlay(gt_mask, gt_color)
                ax_gt.imshow(overlay)
            ax_gt.axis("off")

            #  Row 1: Predictions
            ax_pred = axes[1, col_idx]
            ax_pred.imshow(original_image)
            pred_mask = predicted_masks[head_name]
            pred_color = config.heads[head_name]["color"]
            overlay = create_mask_overlay(pred_mask, pred_color)
            ax_pred.imshow(overlay)
            ax_pred.axis("off")

        fig.tight_layout()
        return fig


def main(args):
    """
    Apply pre-trained checkpoint to sample data and
    visualize the predictions and ground truths.
    """
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    checkpoint_path = args.checkpoint
    savedir = args.savedir
    device = args.device
    record_idx = args.record_index

    configs_path = args.configs_path
    if configs_path is None:
        configs = SegmentationModelConfigs()
    else:
        configs = SegmentationModelConfigs.from_yaml(configs_path)

    # Load the dataset
    dataset = WingPatternDataset(
        image_dir=images_dir, masks_dir=masks_dir, configs=configs
    )

    model = load_model(checkpoint_path=checkpoint_path, configs=configs, device=device)

    try:
        sample = dataset[record_idx]
    except IndexError:
        raise ValueError(f"Record index {record_idx} is out of bounds for the dataset.")

    image_arr = sample["image"].permute(1, 2, 0).numpy()
    gt_masks = {key: mask.squeeze().numpy() for key, mask in sample["masks"].items()}

    # making sure the binary masks are boolean
    for key in gt_masks:
        if configs.heads[key]["type"] == "binary":
            gt_masks[key] = gt_masks[key] > 0

    predictions = infer(
        image_arr=image_arr,
        configs=configs,
        model=model,
        device=device,
    )

    fig = plot_inference_results(
        original_image=image_arr,
        ground_truth_masks=gt_masks,
        predicted_masks=predictions,
        config=configs,
    )

    # save output to file
    plots_savedir = os.path.join(savedir, "plots")
    os.makedirs(plots_savedir, exist_ok=True)
    checkpoint_name = os.path.basename(checkpoint_path).replace(".pth", ".png")
    fig.savefig(os.path.join(plots_savedir, checkpoint_name), dpi=300)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Apply semantic segmentation model to sample data"
    )
    arg_parser.add_argument("--images_dir", default=None, help="Path to images.")
    arg_parser.add_argument(
        "--masks_dir",
        default=None,
        help="Path to segmentation masks.",
    )
    arg_parser.add_argument(
        "--checkpoint", default=None, help="Path to model checkpoint."
    )
    arg_parser.add_argument(
        "--configs_path",
        default=None,
        help="Path to model configs file. If not provided, defaults to SegmentationModelConfigs.",
    )
    arg_parser.add_argument(
        "--savedir",
        default=None,
        help="Path to save the output images: <savedir>/plots",
    )
    arg_parser.add_argument(
        "--device", default="cpu", help="Device to use during inference."
    )
    arg_parser.add_argument(
        "--record_index",
        default=0,
        type=int,
        help="Index pointing to the record to use as sample data.",
    )

    args = arg_parser.parse_args()

    main(args)
