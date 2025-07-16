"""
Repository of semantic segmentation models for various wing morphology targets. 
"""

from typing import Dict

import torch
import torch.nn as nn
import torchinfo


class HopperNetLite(nn.Module):
    """
    Multi-head multi-class semantic segmentation of
    wing structure and pigmentation patterns.

    Args:
        num_groups (int): Numer of groups for GroupNorm (default is 1).
        in_channels (int): Numer of input channels (default is 3 for RBG images).
        out_channels (Dict[str, int]): Dict specifying the number of output channels for each head.
        upsample_mode (str): Upsample mode used in the decoder pass (default is 'bilinear').

    """

    def __init__(
        self,
        num_groups: int = 1,
        in_channels: int = 3,
        out_channels: Dict[str, int] = None,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()

        if out_channels is None or not isinstance(out_channels, dict):
            raise ValueError(
                "Invalid `out_channels' paramter. Please provide a dict mapping head names to output channel counts."
            )

        self.configs = {
            'num_groups': num_groups,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'upsample_mode': upsample_mode
        }

        heads = out_channels.keys()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=8,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.encoder0_downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=8,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.encoder1_mix = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=16,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.encoder1_downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=64,
                eps=1e-5,
                affine=True,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=64,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.decoder1_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder1_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=32 + 16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=16,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder0_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.decoder0_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=16 + 8,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.ReLU(inplace=True),
        )

        #
        # task-specific heads:
        # ---------------------------
        self.heads = nn.ModuleDict()
        for head in heads:
            self.heads[head] = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=out_channels[head],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing skip connections.

        Inputs:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W)
        Outputs:
            Dict[str, torch.Tensor]: Dict of output tensors for each head, of shapes (N, C, H, W)

        """
        # stem
        x = self.stem(x)
        stem = x

        # encoder 0
        x = self.encoder0_downsample(x)

        # encoder 1
        x = self.encoder1_mix(x)
        down16 = x
        x = self.encoder1_downsample(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder 1
        x = self.decoder1_upsample(x)
        x = self.decoder1_reduce(x)
        # first skip
        x = self.decoder1_mixer(torch.cat([down16, x], 1))

        # decoder 0
        x = self.decoder0_upsample(x)
        # second skip
        x = self.decoder0_mixer(torch.cat([stem, x], 1))

        output = dict.fromkeys(self.heads.keys(), None)
        for head_key, head_net in self.heads.items():
            output[head_key] = head_net(x)

        return output


def main():
    """
    Show model summary for a given input shape.
    """
    from hopperscapes.configs import SegmentationModelConfigs

    configs = SegmentationModelConfigs()
    model = HopperNetLite(
        num_groups=configs.num_groups,
        out_channels=configs.out_channels,
    )
    torchinfo.summary(model, input_size=(1, 3, 512, 512))


if __name__ == "__main__":
    main()
