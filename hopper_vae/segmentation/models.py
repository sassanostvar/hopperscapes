from typing import Dict

import torch
import torch.nn as nn
import torchinfo


class HopperNet(nn.Module):
    """
    Multi-head multi-class semantic segmentation of
    wing structure and patterns.
    """

    def __init__(
        self,
        num_groups: int = 1,
        out_channels: Dict[str, int] = None,
    ):
        super().__init__()

        heads = out_channels.keys() if out_channels is not None else None
        if heads is None:
            raise ValueError("Please provide a set of heads for the HopperNet model.")

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
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

        # self.decoder1_upsample = nn.ConvTranspose2d(
        #     in_channels=64,
        #     out_channels=32,
        #     kernel_size=2,
        #     stride=2,
        # )
        self.decoder1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
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

        # self.decoder0_upsample = nn.ConvTranspose2d(
        #     in_channels=16,
        #     out_channels=16,
        #     kernel_size=2,
        #     stride=2,
        #     # padding=1,
        #     # dilation=1,
        # )

        self.decoder1_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=32 + 16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                # dilation=1,
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
                # dilation=1,
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
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.decoder0_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=16 + 8,
                # out_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                # dilation=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=32,
                eps=1e-5,
                affine=True,
            ),
            nn.Conv2d(
                # in_channels=16,
                # out_channels=16,
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                # dilation=1,
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
        # self.heads = dict.fromkeys(heads, None)
        self.heads = nn.ModuleDict()
        for head in heads:
            self.heads[head] = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=out_channels[head],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    # dilation=1,
                ),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        forward pass implementing the skip connections
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

        # encoder 2
        # TODO: add more blocks

        # bottleneck
        x = self.bottleneck(x)

        # decoder 2
        # TODO: add more blocks

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
    model = HopperNet()
    torchinfo.summary(model, input_size=(1, 3, 512, 512))


if __name__ == "__main__":
    main()
