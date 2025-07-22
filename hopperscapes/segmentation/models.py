"""
Configurable U-Net-like model for multi-task semantic segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["Encoder", "Decoder", "ModularHopperNet"]

DEFAULT_ENCODER_CONFIGS = {
    "stem": {
        "out_channels": 16,
        "stride": 1,
        "concat": True,
    },
    "encoder0_ds": {
        "out_channels": 32,
        "stride": 2,
        "concat": False,
    },
    "encoder1_conv": {
        "out_channels": 32,
        "stride": 1,
        "concat": True,
    },
    "encoder1_ds": {
        "out_channels": 64,
        "stride": 2,
        "concat": False,
    },
}


DEFAULT_BOTTLENECK_CONFIGS = {
    "b_neck1": {"out_channels": 128},
    "b_neck2": {"out_channels": 64},
}

# Upsample -> Reduce (optional) -> Mix
DEFAULT_DECODER_CONFIGS = {
    "decoder1": {"reduce_channels": 32, "mixer_out_channels": [32, 16]},
    "decoder0": {"reduce_channels": None, "mixer_out_channels": [16, 16]},
}


class Conv2dBlock(nn.Module):
    """
    Basic convolutional block with Conv2d -> GroupNorm -> ReLU.

    Args:
        in_channels     (int): input channels.
        out_channels    (int): output channels.
        kernel_size     (int): Conv2d kernel size. Default is 3.
        stride          (int): Conv2d stride. Default is 1.
        padding         (int): Conv2d padding. Default is 1.
        num_groups      (int): number of groups for GroupNorm layers. Default is 8.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        optimal_num_groups = min(num_groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.GroupNorm(optimal_num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder assembled using Conv2dBlock units.

    Args:
        configs (Dict[str, ...]): configuration for encoder stages, including
            in_channels, out_channels, kernel size, stride, padding, num groups,
            and whether to use skip connections.
        num_groups (int): number of groups for GroupNorm layers.
    """

    def __init__(
        self, configs: Dict[str, Dict[str, Any]], num_groups: int, in_channels: int = 3
    ):
        super().__init__()
        self.configs = configs
        self.encoder = self.assemble(self.configs, num_groups, in_channels)

    def assemble(
        self, configs: Dict[str, Dict[str, Any]], num_groups: int, in_channels: int
    ) -> nn.ModuleDict:
        """
        Create a sequence of Conv2dBlock units (Conv2d -> GroupNorm -> ReLU).

        Args:
            configs (Dict[str, ...]): dictionary mapping stage names to
                Conv2dBlock configs and their skip connection status.
            num_groups (int): number of groups for GroupNorm layers.
            in_channels (int): input channels for the first stage.

        Returns:
            nn.ModuleDict: List of Conv2dBlock units.
        """
        encoder = nn.ModuleDict()
        ch_in = in_channels
        for stage_name, params in configs.items():
            encoder[stage_name] = Conv2dBlock(
                ch_in,
                params["out_channels"],
                stride=params["stride"],
                num_groups=num_groups,
            )
            ch_in = params["out_channels"]

        self.out_channels = params["out_channels"]
        return encoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): input tensor of shape (N, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: the final output tensor and
                a list of intermediate outputs for skip connections.
        """
        skip_connections = []
        for stage_name, stage in self.encoder.items():
            x = stage(x)
            if self.configs[stage_name]["concat"]:
                skip_connections.append(x)

        return x, skip_connections[::-1]  # Reverse the list for decoder


class DecoderBlock(nn.Module):
    """
    Decoder block with Upsample -> Reduce (optional) -> Mix.

    Args:
        stage_configs (Dict[str, ...]): configuration of the decoder stage, including
            in_channels, reduction channels (optional), and mixer channels.
        num_groups (int): number of groups for GroupNorm layers.
        upsample_mode (str): the algorithm used for upsampling ("bilinear" or "nearest").

    Returns:
        nn.Module: A sequential block of upsampling, optional reduction, and mixing layers.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        stage_configs: Dict,
        num_groups: int,
        upsample_mode: str,
    ):
        super().__init__()

        reduce_channels = stage_configs.get("reduce_channels")
        mixer_out_channels = stage_configs["mixer_out_channels"]

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
            nn.ReLU(inplace=True),
        )

        # Optional reduction layer:
        self.reduce = nn.Identity()
        post_reduce_channels = in_channels
        if reduce_channels:
            self.reduce = Conv2dBlock(
                in_channels=in_channels,
                out_channels=reduce_channels,
                kernel_size=1,
                padding=0,
                num_groups=num_groups,
            )
            post_reduce_channels = reduce_channels

        # Mixer
        # (assemble dynamically based on skip and post-reduction channels)
        mixer_in_channels = skip_channels + post_reduce_channels
        self.mix = nn.Sequential()

        channels_in = mixer_in_channels
        for i, channels_out in enumerate(mixer_out_channels):
            self.mix.add_module(
                f"mix_conv_{i}",
                Conv2dBlock(channels_in, channels_out, num_groups=num_groups),
            )
            channels_in = channels_out

    def forward(self, x, skip):
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor):       input tensor.
            skip (torch.Tensor):    skip connection tensor from the encoder.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.upsample(x)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.mix(x)


class Decoder(nn.Module):
    """
    Decoder assembled using DecoderBlock units.

    Args:
        configs (Dict[str, ...]):   configurations for the decoder stages.
        num_groups (int):           number of groups for GroupNorm layers.
        upsample_mode (str):        the algorithm used for upsampling ("bilinear" or "nearest").

    Returns:
        nn.ModuleDict: List of "DecoderBlock"s for the decoder.
    """

    def __init__(
        self,
        decoder_configs: Dict[str, Dict[str, Tuple[int]]],
        encoder_configs: Dict,
        bottleneck_channels_out: int,
        num_groups: int,
        upsample_mode: str,
    ):
        super().__init__()
        self.blocks = self.assemble(
            decoder_configs,
            encoder_configs,
            bottleneck_channels_out,
            num_groups,
            upsample_mode,
        )

    def assemble(
        self,
        decoder_configs: Dict[str, Dict[str, Tuple[int]]],
        encoder_configs: Dict,
        bottleneck_channels_out: int,
        num_groups: int,
        upsample_mode: str,
    ) -> nn.ModuleDict:
        """
        Create a sequence of DecoderBlock units.

        Args:
            configs (Dict[str, ...]):   dict mapping stage names to DecoderBlock configs.
            num_groups (int):           number of groups for GroupNorm layers.
            upsample_mode (str):        the algorithm used for upsampling ("bilinear" or "nearest").

        Returns:
            nn.ModuleDict: List of DecoderBlock units for the decoder.
        """
        blocks = nn.ModuleDict()

        # Read the channel counts from the encoder's skip connections
        skip_channels = [
            p["out_channels"]
            for p in encoder_configs.values()
            if p.get("concat", False)
        ][
            ::-1
        ]  # reverse order of skip connections

        # Figure out the input channels for each decoder block
        channels_in = bottleneck_channels_out

        for i, (stage_name, stage_configs) in enumerate(decoder_configs.items()):
            blocks[stage_name] = DecoderBlock(
                in_channels=channels_in,
                skip_channels=skip_channels[i],
                stage_configs=stage_configs,
                num_groups=num_groups,
                upsample_mode=upsample_mode,
            )
            channels_in = stage_configs["mixer_out_channels"][-1]

        # Record the final out channels for downstream units
        self.out_channels = channels_in
        return blocks

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder, applying each block sequentially with skip connections.

        Args:
            x (torch.Tensor):           input tensor from the bottleneck.
            skips (List[torch.Tensor]): list of skip connection tensors from the encoder.

        Returns:
            torch.Tensor: final output tensor.
        """
        for i, (name, block) in enumerate(self.blocks.items()):
            x = block(x, skips[i])
        return x


class ModularHopperNet(nn.Module):
    """
    A modular U-Net-like architecture for multi-task semantic segmentation.

    Args:
        num_groups (int):               number of groups for GroupNorm layers. Default is 1.
        in_channels (int):              in channels (e.g. 3 for RGB images). Default is 3.
        out_channels (Dict[str, int]):  dict mapping each head's name to the
                                        number of output channels.
        upsample_mode (str):            the algorithm used for upsampling in the decoder (
                                        "bilinear" or "nearest"). Default is 'bilinear'.

    Raises:
        ValueError: if `out_channels` is not a dictionary.
    """

    def __init__(
        self,
        num_groups: int = 1,
        in_channels: int = 3,
        out_channels: Dict[str, int] = None,
        upsample_mode: str = "bilinear",
        encoder_configs: Optional[Dict[str, Any]] = DEFAULT_ENCODER_CONFIGS,
        bottleneck_configs: Optional[Dict[str, Any]] = DEFAULT_BOTTLENECK_CONFIGS,
        decoder_configs: Optional[Dict[str, Any]] = DEFAULT_DECODER_CONFIGS,
    ):
        super().__init__()

        if out_channels is None or not isinstance(out_channels, dict):
            raise ValueError(
                "Invalid `out_channels' parameter. \
                    Please provide a dict mapping head names to output channel counts."
            )

        self.configs = {
            "num_groups": num_groups,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "upsample_mode": upsample_mode,
        }

        heads = out_channels.keys()

        # --- Encoder ---
        self.encoder = Encoder(encoder_configs, num_groups)
        bneck_in = self.encoder.out_channels

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential()
        for name, params in bottleneck_configs.items():
            self.bottleneck.add_module(
                name, Conv2dBlock(bneck_in, params["out_channels"])
            )
            bneck_in = params["out_channels"]

        decoder_in = bneck_in

        # --- Decoder ---
        self.decoder = Decoder(
            decoder_configs,
            encoder_configs,
            decoder_in,
            num_groups,
            upsample_mode,
        )

        # --- Task-specific Heads ---
        self.heads = nn.ModuleDict()
        head_in_channels = self.decoder.out_channels
        for head in heads:
            self.heads[head] = nn.Conv2d(
                head_in_channels, out_channels[head], 1, 1, 0, 1
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with skip connections.

        Args:
            x (torch.Tensor): input image tensor of shape (N, 3, H, W)

        Returns:
            Dict[str, torch.Tensor]: dict mapping head names to their output tensors.

        """
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)

        output = dict.fromkeys(self.heads.keys(), None)
        for head_key, head_net in self.heads.items():
            output[head_key] = head_net(x)

        return output


MODEL_REGISTRY = {"ModularHopperNet": ModularHopperNet}


def get_model(name: str, configs: Dict):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model name {name} not found in registry.")
    return MODEL_REGISTRY[name](**configs)


def main(args):
    """
    Show model summary for a given input shape.
    """
    import torchinfo
    import yaml

    configs_filepath = args.model_configs
    if configs_filepath:
        with open(configs_filepath, "r") as f:
            configs = yaml.safe_load(f)

        model_configs = configs["params"]
        model = get_model(name=configs["name"], configs=model_configs)
        torchinfo.summary(model, input_size=(1, 3, 512, 512))
    else:
        model = ModularHopperNet(in_channels=3, out_channels={"head1": 1, "head2": 2})
        torchinfo.summary(model, input_size=(1, 3, 512, 512))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modular HopperNet Model Summary")
    parser.add_argument(
        "--model_configs",
        type=str,
        required=False,
        help="Path to model configuration file (YAML).",
    )
    args = parser.parse_args()

    main(args)
