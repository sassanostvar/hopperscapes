import sys

sys.path.append("../hopper-vae")


import os
import torch

from hopper_vae.segmentation.data_io import WingPatternDataset
from hopper_vae.segmentation.models import HopperNet

import torchinfo

import matplotlib.pyplot as plt

# change font sizes for all text, axis labels, etc.
plt.rcParams.update({"font.size": 5})
plt.rcParams.update({"axes.titlesize": 5})
plt.rcParams.update({"axes.labelsize": 5})
plt.rcParams.update({"xtick.labelsize": 5})
plt.rcParams.update({"ytick.labelsize": 5})


def main(checkpoint_path: str):
    # Load the dataset
    dataset = WingPatternDataset(
        image_dir="data/aug/train/images", masks_dir="data/aug/train/masks"
    )
    print(f"dataset length: {len(dataset)}")

    # load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    heads = checkpoint["heads"]
    out_channels = {head: 1 for head in heads}
    out_channels["domains"] = 3  # shouldn't hand-code here..

    # Load the model
    model = HopperNet(
        num_groups=1,  # for GroupNorm
        out_channels=out_channels,  # use the heads from the checkpoint
    )
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    # evaluate the model on the dataset
    model.eval()
    with torch.no_grad():
        sample = dataset[13]
        image = sample["image"].unsqueeze(0)
        masks = sample["masks"]
        output = model(image)

    fig, ax = plt.subplots(nrows=2, ncols=len(masks), figsize=(4, 2))

    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    from skimage.measure import label, regionprops, find_contours

    # show image
    for axi in ax.flatten():
        axi.imshow(sample["image"].permute(1, 2, 0))
        axi.axis("off")

    keys = ["wing", "domains", "spots", "veins"]
    rgb_colors = {
        "wing": [0.8, 0.8, 0.8],
        "domains": [0.5, 0.5, 1.0],
        "spots": [1.0, 0.5, 0.5],
        "veins": [0.5, 1.0, 0.5],
    }

    # show ground truth masks
    # for idx, key in enumerate(sample["masks"]):
    for idx, key in enumerate(keys):
        _mask = sample["masks"][key].squeeze().numpy()
        color = np.concatenate([rgb_colors[key], np.array([0.6])], axis=0)
        mask_image = np.expand_dims(_mask, axis=-1) * color
        ax[0, idx].imshow(mask_image)
        for label in np.unique(_mask):
            contours = find_contours(_mask == label, 0.5)
            for contour in contours:
                ax[0, idx].plot(
                    contour[:, 1], contour[:, 0], linewidth=0.5, color="red"
                )

    domain_colors = {
        0: np.array([0.0, 0.0, 0.0]),  # Background (black, or transparent)
        1: np.array([1.0, 133.0 / 256.0, 89.0 / 256.0]),  # Domain Class 1 (Red)
        2: np.array([0.0, 0.5, 1.0]),  # Domain Class 2 (Blue)
    }
    alpha = 0.6  # Transparency for overlay

    # show predicted masks
    for idx, key in enumerate(keys):
        if key == "domains":
            pred_logit = output[key]
            pred_prob = torch.softmax(pred_logit, dim=1)

            H, W = pred_prob.shape[2], pred_prob.shape[3]
            composite_mask_image = np.zeros((H, W, 4))

            # Iterate through each class (0, 1, 2) to build the composite image
            for class_id in range(pred_prob.shape[1]):  # Iterate 0, 1, 2
                class_mask_prob = pred_prob[0, class_id, :, :].detach().cpu().numpy()
                class_mask_binary = class_mask_prob > 0.5  # Threshold to binary mask

                # Get color for current class
                current_color = domain_colors[class_id]
                color_with_alpha = np.concatenate(
                    [current_color, np.array([alpha])], axis=0
                )

                # Apply color where mask is true
                composite_mask_image[class_mask_binary] = color_with_alpha

                # Plot contours for non-background classes
                if class_id > 0 and np.any(
                    class_mask_binary
                ):  # Don't contour background
                    contours = find_contours(class_mask_binary)
                    for contour in contours:
                        ax[1, idx].plot(
                            contour[:, 1], contour[:, 0], linewidth=0.5, color="blue"
                        )

            ax[1, idx].imshow(
                composite_mask_image
            )  # Display the final composite RGBA image

        else:
            pred_logit = output[key].squeeze()
            pred_prob = torch.sigmoid(pred_logit)
            pred_prob = pred_prob.squeeze().numpy()
            mask = pred_prob > 0.5

            color = np.concatenate([rgb_colors[key], np.array([0.6])], axis=0)
            mask_image = np.expand_dims(mask, axis=-1) * color
            ax[1, idx].imshow(mask_image)
            contours = find_contours(mask)
            for contour in contours:
                ax[1, idx].plot(
                    contour[:, 1], contour[:, 0], linewidth=0.5, color="blue"
                )

    fig.tight_layout(pad=0.0)

    from skimage.measure import label, regionprops

    print(
        f'number of spots: {len(regionprops(label(sample["masks"]["spots"].squeeze().numpy())))}'
    )

    return fig, ax


if __name__ == "__main__":
    import glob
    import os
    from tqdm import tqdm

    model_dir = "./outputs/models/hopper_net_aug2_test4"
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    all_checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pth")))

    plots_savedir = os.path.join(model_dir, "plots")
    os.makedirs(plots_savedir, exist_ok=True)

    for checkpoint_path in tqdm(all_checkpoints):
        print(f"Evaluating {checkpoint_path}")
        fig, ax = main(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path).replace(".pth", ".png")
        fig.savefig(os.path.join(plots_savedir, checkpoint_name), dpi=300)
