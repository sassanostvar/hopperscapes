from dataclasses import dataclass, field



@dataclass
class SegmentationConfigs:
    """
    Configuration for the multi-head segmentation model.
    """

    square_image_size = 1024

    convert_to_hsv = True

    out_channels = {
        "wing": 1,
        "veins": 1,
        "spots": 1,
        "domains": 3,  # 2 + background
        # "bricks": 1,
    }

    num_groups = 1 # for GroupNorm


@dataclass
class TrainingConfigs:
    """
    Configuration for model training
    """

    model_name: str = "test_model"
    savedir: str = "./outputs/models"
    device: str = "cpu"
    seg_configs: SegmentationConfigs = SegmentationConfigs()
    batch_size: int = 4
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

    # # global weights for the composite loss function
    total_loss_weights = {
        "wing": 1.0,
        "veins": 1.0,
        "spots": 1.0,
        "domains": 1.0,
        # "bricks": 1.0,
    }

    dice_thresholds_to_freeze_heads = {
        "wing": 1.0,
        "veins": 0.95,
        "spots": 0.95,
        "domains": 0.95,
    }

    loss_function_configs = {
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
            # "bce": {"weight": 1.0, "params": {"pos_weight": 1.0}},
            # "soft_dice": {"weight": 1.0, "params": {}},
        },
    }

    freeze_heads = {
        "wing": False,
        "veins": False,
        "spots": False,
        "domains": False,
        # "bricks": False,
    }

    dice_scores_to_track = {
        "wing": "soft_dice",
        "veins": "cldice",
        "spots": "soft_dice",
        "domains": "soft_dice",
    }
