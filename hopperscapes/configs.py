"""
Configurations for segmentation model architecture, inputs, loss functions, and training.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SegmentationModelConfigs:
    """
    Configuration for data prep and model training
    """

    experiment_id: str = "hopperscapes_demo"
    savedir: str = "./outputs/demo/models"

    # -------------------------------
    # ------- DATASET CONFIGS -------
    # -------------------------------

    image_file_exts: Tuple[str] = field(default_factory=lambda: (".png", ".jpg"))

    image_transforms: Dict = field(
        default_factory=lambda: {
            "ResizeToLongestSide": {"image_side_length": 512},
            # "ConvertToHSV": {},
            "PrepareImageTensor": {},
        }
    )

    mask_transforms: Dict = field(
        default_factory=lambda: {
            "ResizeToLongestSide": {"image_side_length": 512},
        }
    )

    # -------------------------------
    # -------- MODEL CONFIGS --------
    # -------------------------------

    in_channels: int = 3  # RGB, HSV, ...
    # model heads and channel counts
    out_channels: Dict[str, int] = field(
        default_factory=lambda: {
            "wing": 1,
            "veins": 1,
            "spots": 1,
            "domains": 3,  # 2 + background
        }
    )

    # to replace "out_channels" ultimately:
    heads: Dict[str, Dict] = field(
        default_factory=lambda: {
            "wing": {
                "channels": 1,
                "type": "binary",
                "color": [0.8, 0.0, 0.1],
            },
            "veins": {
                "channels": 1,
                "type": "binary",
                "color": [0.5, 1.0, 0.5],
            },
            "spots": {
                "channels": 1,
                "type": "binary",
                "color": [1.0, 0.5, 0.5],
            },
            "domains": {
                "channels": 3,
                "type": "multiclass",
                "color": {1: [1.0, 0.52, 0.35], 2: [0, 0.5, 1.0]},
            },
        }
    )

    num_groups: int = 8  # for GroupNorm
    upsample_mode: str = "bilinear"  # "bilinear" or "nearest"

    encoder_configs: Dict = field(
        default_factory=lambda: {
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
    )

    bottleneck_configs: Dict = field(
        default_factory=lambda: {
            "b_neck1": {"out_channels": 128},
            "b_neck2": {"out_channels": 64},
        }
    )

    # Upsample -> Reduce (optional) -> Mix
    decoder_configs: Dict = field(
        default_factory=lambda: {
            "decoder1": {"reduce_channels": 32, "mixer_out_channels": [32, 16]},
            "decoder0": {"reduce_channels": None, "mixer_out_channels": [16, 16]},
        }
    )

    # -------------------------------
    # ------ TRAINING CONFIGS -------
    # -------------------------------
    device: str = "cpu"
    batch_size: int = 4
    valid_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 4
    epochs: int = 200
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    lr_scheduler_params: dict = field(default_factory=dict)
    warmup_epochs: int = 10
    warmup_lr: float = 1e-6
    checkpoint_every: int = 10
    log_every: int = 1
    save_best: bool = True
    clip_gradients: bool = True
    max_grad_norm: float = 1.0

    # composite loss weights
    total_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "wing": 1.0,
            "veins": 1.0,
            "spots": 1.0,
            "domains": 1.0,
        }
    )

    # Dice score thresholds to freeze heads
    dice_thresholds_to_freeze_heads: Dict[str, float] = field(
        default_factory=lambda: {
            "wing": 1.0,
            "veins": 0.95,
            "spots": 0.95,
            "domains": 0.95,
        }
    )

    # per-head loss function configs
    loss_function_configs: Dict[str, Dict] = field(
        default_factory=lambda: {
            "wing": {
                "bce": {"weight": 1.0, "params": {"pos_weight": 5.0}},
                "soft_dice": {"weight": 1.0, "params": {}},
            },
            "veins": {
                "bce": {"weight": 0.5, "params": {"pos_weight": 100.0}},
                "soft_dice": {"weight": 2.0, "params": {}},
                "cldice": {"weight": 2.0, "params": {}},
            },
            "spots": {
                "focal": {"weight": 1.0, "params": {"alpha": 0.85, "gamma": 2.0}},
                "soft_dice": {"weight": 1.0, "params": {}},
            },
            "domains": {
                "ce": {"weight": 1.0, "params": {}},
            },
        }
    )

    # training dynamics
    freeze_heads: Dict[str, bool] = field(
        default_factory=lambda: {
            "wing": False,
            "veins": False,
            "spots": False,
            "domains": False,
        }
    )

    # record-keeping
    dice_scores_to_track: Dict[str, str] = field(
        default_factory=lambda: {
            "wing": "soft_dice",
            "veins": "cldice",
            "spots": "soft_dice",
            "domains": "soft_dice",
        }
    )

    def to_yaml(self, yaml_path: str) -> None:
        """
        Serialize current configs and save to YAML.

        Args:
            yaml_path (str): Savepath for YAML file.
        """
        import yaml
        from dataclasses import asdict

        with open(yaml_path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SegmentationModelConfigs":
        """
        Load configs from YAML file.

        Args:
            yaml_path (str): Path to YAML configuration file.

        Returns:
            SegmentationModelConfigs: Corresponding configs dataclass object.
        """
        import yaml

        with open(yaml_path, "r") as file:
            yaml_configs = yaml.safe_load(file)

        valid_keys = SegmentationModelConfigs.__dataclass_fields__.keys()
        filtered_configs = {k: v for k, v in yaml_configs.items() if k in valid_keys}

        return cls(**filtered_configs)

    @classmethod
    def from_yaml_files(
        cls,
        model_configs_path: str,
        dataset_configs_path: str,
        training_configs_path: str,
    ) -> "SegmentationModelConfigs":
        import yaml

        with open(model_configs_path, "r") as f:
            model_configs = yaml.safe_load(f).get("params", {})

        with open(dataset_configs_path, "r") as f:
            dataset_configs = yaml.safe_load(f)

        with open(training_configs_path, "r") as f:
            training_configs = yaml.safe_load(f)

        # merge
        merged_configs = {**model_configs, **dataset_configs, **training_configs}

        # validate
        valid_keys = cls.__dataclass_fields__.keys()
        final_configs = {k: v for k, v in merged_configs.items() if k in valid_keys}

        return cls(**final_configs)


def main(args):
    savepath = args.savepath
    configs = SegmentationModelConfigs()
    configs.to_yaml(savepath)
    print(f"Configs saved to {savepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Save segmentation model configs to YAML."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="segmentation_model_configs.yaml",
        help="Path to save the YAML configuration file.",
    )

    args = parser.parse_args()
    main(args)
    print("Segmentation model configs saved successfully.")
