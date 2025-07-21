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

    model_name: str = "hopperscapes_demo"
    savedir: str = "./outputs/demo/models"

    in_channels: int = 3  # RGB, HSV, ...

    image_file_exts: Tuple[str] = field(
        default_factory=lambda: (".png", ".jpg")
    )

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

    # model heads and channel counts
    out_channels: Dict[str, int] = field(
        default_factory=lambda: {
            "wing": 1,
            "veins": 1,
            "spots": 1,
            "domains": 3,  # 2 + background
        }
    )
    num_groups: int = 8  # for GroupNorm
    upsample_mode: str = "bilinear"  # "bilinear" or "nearest"

    # training configs
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
    log_every: int = 10
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

    @staticmethod
    def from_yaml(yaml_path: str) -> "SegmentationModelConfigs":
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

        return SegmentationModelConfigs(**filtered_configs)


def main(args):
    savepath = args.savepath
    configs = SegmentationModelConfigs()
    configs.to_yaml(savepath)
    print(f"Configs saved to {savepath}")    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save segmentation model configs to YAML.")
    parser.add_argument("--savepath", type=str, default="segmentation_model_configs.yaml",
                        help="Path to save the YAML configuration file.")
    
    args = parser.parse_args()
    main(args)
    print("Segmentation model configs saved successfully.")