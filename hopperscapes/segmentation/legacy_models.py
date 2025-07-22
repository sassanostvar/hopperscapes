"""
A compact U-Net-like model for multi-task semantic segmentation.
"""

from typing import Dict

import torch
import torch.nn as nn


class HopperNetLite(nn.Module):
    """
    A multi-head, U-Net-like architecture for multi-task semantic segmentation of
    wing structure and pigmentation patterns.

    Args:
        num_groups (int): Number of groups for GroupNorm layers. Default is 1.
        in_channels (int): Number of input channels (e.g. 3 for RGB images). Default is 3.
        out_channels (Dict[str, int]): Dictionary mapping each head's name to the
        number of output channels.
        upsample_mode (str): The algorithm used for upsampling in the decoder (
            "bilinear" or "nearest"). Default is 'bilinear'.

    Raises:
        ValueError: If `out_channels` is not a dictionary.
    """

    def __init__(
        self,
        num_groups: int = 8,
        in_channels: int = 3,
        out_channels: Dict[str, int] = None,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()

        if out_channels is None or not isinstance(out_channels, dict):
            raise ValueError(
                "Invalid `out_channels' paramter. Please provide a dict \
                    mapping head names to output channel counts."
            )

        self.configs = {
            "num_groups": num_groups,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "upsample_mode": upsample_mode,
        }

        heads = out_channels.keys()

        # --- Encoder ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 8, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder0_downsample = nn.Sequential(
            nn.Conv2d(8, 8, 3, 2, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 8, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder1_mix = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 16, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder1_downsample = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 32, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 64, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 64, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        self.decoder1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.decoder1_reduce = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, 1, bias=False),
            nn.GroupNorm(num_groups, 32, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.decoder1_mixer = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 32, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 16, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.decoder0_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.decoder0_mixer = nn.Sequential(
            nn.Conv2d(16 + 8, 16, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 16, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 16, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )

        # --- Task-specific Heads ---
        self.heads = nn.ModuleDict()
        for head in heads:
            self.heads[head] = nn.Sequential(
                nn.Conv2d(32, out_channels[head], 1, 1, 0, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing skip connections.

        Inputs:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W)
        Outputs:
            Dict[str, torch.Tensor]: Dict of output tensors for each head, of shapes (N, C, H, W)

        """
        # stem:
        # (N, 3, H, W) -> (N, 8, H, W)
        x = self.stem(x)
        skip_full_res = x

        # encoder 0:
        # (N, 8, H, W) -> (N, 8, H/2, W/2)
        x = self.encoder0_downsample(x)

        # encoder 1:
        # (N, 8, H/2, W/2) -> (N, 16, H/4, W/4)
        x = self.encoder1_mix(x)
        skip_half_res = x
        x = self.encoder1_downsample(x)

        # --- Bottleneck ---
        # (N, 32, H/4, W/4) -> (N, 64, H/4, W/4)
        x = self.bottleneck(x)

        # decoder 1:
        # (N, 64, H/4, W/4) -> (N, 16, H/2, W/2)
        x = self.decoder1_upsample(x)
        x = self.decoder1_reduce(x)  # 64 -> 32 channel reduction to save on compute
        x = self.decoder1_mixer(
            torch.cat([skip_half_res, x], 1)  # <-- first skip connection
        )

        # decoder 0:
        # (N, 16, H/2, W/2) -> (N, 32, H, W)
        x = self.decoder0_upsample(x)
        x = self.decoder0_mixer(
            torch.cat([skip_full_res, x], 1)  # <-- second skip connection
        )

        output = dict.fromkeys(self.heads.keys(), None)
        for head_key, head_net in self.heads.items():
            output[head_key] = head_net(x)

        return output


MODEL_REGISTRY = {"HopperNetLite": HopperNetLite}


def get_model(name: str, configs: Dict):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model name {name} not found in registry.")
    return MODEL_REGISTRY[name](**configs)


def main():
    """
    Show model summary for a given input shape.
    """
    from hopperscapes.configs import SegmentationModelConfigs
    import torchinfo

    configs = SegmentationModelConfigs()
    model = HopperNetLite(
        num_groups=configs.num_groups,
        out_channels=configs.out_channels,
    )
    torchinfo.summary(model, input_size=(1, 3, 512, 512))


if __name__ == "__main__":
    main()
