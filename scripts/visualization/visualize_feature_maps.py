import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def visualize_feature_maps_concise(model, image_tensor, layer_name=None, num_maps=8):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU)) and (
            layer_name is None or layer_name in name
        ):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        model(image_tensor.to(next(model.parameters()).device))

    for name, activation_tensor in activations.items():
        if layer_name is None or layer_name in name:
            maps = activation_tensor[0].cpu().numpy()  # First item in batch

            fig, axes = plt.subplots(
                1,
                min(num_maps, maps.shape[0]),
                figsize=(1.8 * min(num_maps, maps.shape[0]), 2.5),
            )
            if min(num_maps, maps.shape[0]) == 1:
                axes = [axes]
            for i in range(min(num_maps, maps.shape[0])):
                map_norm = (maps[i] - maps[i].min()) / (
                    maps[i].max() - maps[i].min() + 1e-8
                )
                axes[i].imshow(map_norm, cmap="magma")
                axes[i].set_title(f"Map {i+1}")
                axes[i].axis("off")
            fig.suptitle(f"Activations: {name}", y=1.05)
            plt.tight_layout()
            plt.show()

    for hook in hooks:
        hook.remove()
    model.train()  # Set back to train mode


if __name__ == "__main__":
    import os
    import glob
    from hopper_vae.segmentation.models import HopperNetLite
    from hopper_vae.segmentation.data_io import WingPatternDataset

    # model_dir = "./outputs/models/hopper_net_aug2_test4"
    # checkpoints_dir = os.path.join(model_dir, "checkpoints")
    # all_checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pth")))
    # last_checkpoint = all_checkpoints[-1] if all_checkpoints else None

    last_checkpoint = "./outputs/models/hopper_net_aug2_test4/checkpoints/checkpoint_epoch_200.pth"

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
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = WingPatternDataset(
        image_dir="data/aug/train/images", masks_dir="data/aug/train/masks"
    )
    image = dataset[-1]["image"].unsqueeze(0)
    print(
        f"image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
    )

    # --- Call the visualization function ---
    print("Visualizing Encoder Activations:")
    visualize_feature_maps_concise(model, image, layer_name="stem", num_maps=8)

    print("\nVisualizing Decoder Activations:")
    visualize_feature_maps_concise(
        model, image, layer_name="encoder0_downsample", num_maps=8
    )
