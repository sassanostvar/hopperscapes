import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def visualize_filters_concise(model, layer_name=None, num_filters=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and (layer_name is None or layer_name in name):
            weights = module.weight.data.cpu().numpy()
            display_filters = []
            for i in range(min(num_filters, weights.shape[0])):
                # For multiple input channels, taking the first one for visualization
                # Or use np.mean(weights[i], axis=0) to average across input channels
                filt = (
                    weights[i, 0, :, :] if weights.shape[1] > 0 else weights[i, 0, :, :]
                )
                display_filters.append(filt)

            if display_filters:
                fig, axes = plt.subplots(
                    1, len(display_filters), figsize=(1.5 * len(display_filters), 2)
                )
                if len(display_filters) == 1:
                    axes = [axes]  # Handle single filter case
                for i, filt in enumerate(display_filters):
                    filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                    axes[i].imshow(filt_norm, cmap="gray")
                    axes[i].set_title(f"F{i+1}")
                    axes[i].axis("off")
                fig.suptitle(f"Layer: {name}", y=1.05)
                plt.tight_layout()
                plt.show()


# Example Usage (replace with your model loading and layer names)
if __name__ == "__main__":
    import os
    import glob
    from hopperscapes.segmentation.models import HopperNetLite
    from hopperscapes.segmentation.data_io import WingPatternDataset

    model_dir = "./outputs/models/hopper_net_aug2_test4"
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    all_checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pth")))

    last_checkpoint = all_checkpoints[-1] if all_checkpoints else None

    # load the checkpoint
    checkpoint = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    heads = checkpoint["heads"]
    out_channels = {head: 1 for head in heads}
    out_channels["domains"] = 3  # shouldn't hand-code here..

    # Load the model
    model = HopperNetLite(
        num_groups=1,  # for GroupNorm
        out_channels=out_channels,  # use the heads from the checkpoint
    )
    model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])

    # model = SimpleUNet()
    # model.load_state_dict(torch.load('your_model.pth')) # Load your actual model weights
    # model.to('cuda' if torch.cuda.is_available() else 'cpu') # Move to device

    dataset = WingPatternDataset(
        image_dir="data/aug/train/images", masks_dir="data/aug/train/masks"
    )
    image = dataset[3]["image"].unsqueeze(0)

    # --- Call the visualization function ---
    print("Visualizing Encoder Filters:")
    visualize_filters_concise(model, layer_name="encoder1_downsample", num_filters=8)

    print("\nVisualizing Decoder Filters:")
    visualize_filters_concise(model, layer_name="decoder0_mixer", num_filters=8)
