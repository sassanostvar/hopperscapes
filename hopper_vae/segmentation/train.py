"""
    Train the multi-task semantic segmentation model.
"""

import argparse
import logging
import os
from typing import Dict

import random
import numpy as np

import torch
import torch.optim as optim
import torchinfo
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hopper_vae.configs import SegmentationModelConfigs
from hopper_vae.segmentation import loss
from hopper_vae.segmentation.dataset import WingPatternDataset, hopper_collate_fn
from hopper_vae.segmentation.models import HopperNetLite

"""
Checklist: 
- [ ] add validation
- [x] add CLI
- [x] set up configs yaml
- [x] test on GPUs
- [x] add logging
- [ ] log preprocessing
- [ ] checkpoint logs
- [ ] dynamic loss weight adjustments
- [ ] add support for distributed training
- [ ] add wandb integration
"""

logger = logging.getLogger("HopperNetTrainingLog")


def init_seeds(seed: int = 42):
    """
    Set seeds for reproducibility.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_datasets(
    images_dir: str,
    masks_dir: str,
    batch_size: int,
    valid_split: float = 0.2,
    seed: int = 42,
):
    """
    Create a random train/valid split and generate the DataLoaders.

    Args:
        images_dir (str): Path to image direcotry.
        masks_dir (str): Path to the masks root directory.
        batch_size (int): Batch size for DataLoaders.
        valid_split (float): Fraction of data used for validation.
        seed (int): Random seed (for reproducibility).

    Returns:
        train_loader (DataLoader): Train DataLoader.
        valid_loader (DataLoader): Valid DataLoader.
    """

    init_seeds(seed)

    dataset = WingPatternDataset(image_dir=images_dir, masks_dir=masks_dir)

    # split
    dataset_size = len(dataset)
    valid_size = int(dataset_size * valid_split)
    train_size = dataset_size - valid_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=hopper_collate_fn,
        shuffle=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=hopper_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader


class HopperNetTrainer:
    def __init__(
        self,
        model: nn.Module,
        freeze_heads: bool,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criteria: nn.Module,
        total_loss_weights: Dict[str, float],
        lr: float,
        weight_decay: float,
        start_epoch: int,
        num_epochs: int,
        checkpoint_every: int,
        log_every: int,
        clip_gradients: bool,
        max_norm: float,
        dice_scores_to_track: dict,
        threshold_dice_scores: dict,
        device: str,
        savedir: str,
    ):
        if savedir is None:
            raise ValueError("Please provide a valid directory to save the model.")

        self.model = model
        self.freeze_heads = freeze_heads
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criteria = criteria
        self.global_loss_weights = total_loss_weights or None
        self.loss_weights = (
            self.global_loss_weights.copy() if self.global_loss_weights else {}
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.clip_gradients = clip_gradients
        self.max_norm = max_norm
        self.dice_scores_to_track = dice_scores_to_track or {}
        self.threshold_dice_scores = threshold_dice_scores or {}
        self.device = device
        self.savedir = savedir

        self.checkpoints_dir = os.path.join(savedir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.logs_dir = os.path.join(savedir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.writer = SummaryWriter(self.logs_dir)

        self.model.to(device)

        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # freeze heads if specified
        if self.freeze_heads is not None:
            for head_name, freeze in self.freeze_heads.items():
                if freeze:
                    self.loss_weights[head_name] = 0.0
                    logger.info("freezing head: %s", head_name)
                    for p in self.model.heads[head_name].parameters():
                        p.requires_grad = False

        self.num_iters = 1
        self.epoch = start_epoch
        self.total_loss = None
        self.head_losses = None

        self.dice_scores = dict.fromkeys(self.model.heads.keys(), 0.0)

    def train(self) -> None:
        """
        Runs model training. If specified, dynamically freezes/unfreezes
        heads based on given threshold Dice scores.
        """
        for i in range(self.num_epochs):
            #
            self.epoch += 1
            logger.info("----- Starting epoch %d/%d -----", self.epoch, self.num_epochs)

            self.model.train()

            # dynamically freeze and unfreeze
            # model heads based on dice scores
            if self.num_epochs > 1:
                for head_name, threshold in self.threshold_dice_scores.items():
                    if not head_name in self.dice_scores:
                        continue

                    if self.freeze_heads[head_name] is True:
                        continue

                    if self.dice_scores[head_name] > threshold:
                        logger.info("freezing head: %s", head_name)
                        self.loss_weights[head_name] = 0.0
                        for p in self.model.heads[head_name].parameters():
                            p.requires_grad = False
                    else:
                        logger.info("unfreezing head: %s", head_name)
                        self.loss_weights[head_name] = self.global_loss_weights[
                            head_name
                        ]
                        for p in self.model.heads[head_name].parameters():
                            p.requires_grad = True

            self.train_epoch()

            # checkpoint
            if self.epoch % self.checkpoint_every == 0:
                self.save_checkpoint()

            # log training performance
            if self.num_iters % self.log_every == 0:
                self.log_training_performance()

            # validate epoch
            self.valid_epoch()

    def train_epoch(self) -> None:
        """
        Run one training epoch.
        """
        n_batches = 0
        for sample in tqdm(
            self.train_loader, ascii=True, desc="running training epoch..."
        ):
            images = sample["image"]
            masks = sample["masks"]

            # send to device
            images = images.to(self.device)
            for key in masks:
                masks[key] = masks[key].to(self.device)

            # zero grad
            self.optimizer.zero_grad()

            # forward pass
            logits = self.model(images)

            # compute losses
            self.total_loss, self.head_losses = self.criteria(
                inputs=logits,
                targets=masks,
                weights=self.loss_weights,
            )

            # back-prop
            self.total_loss.backward()

            # clip gradients
            if self.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            # update
            self.optimizer.step()

            # update dice scores
            self.update_dice_scores(logits=logits, targets=masks, clipping_masks=None)

            # log training performance
            self.log_training_performance()

            # incremet
            self.num_iters += 1
            n_batches += 1

    # ------------- VALID LOOP START -------------------------
    def valid_epoch(self) -> None:
        """
        Run one validation epoch.
        """
        self.model.eval()

        total_valid_loss = 0.0
        for sample in tqdm(
            self.valid_loader, ascii=True, desc="running validation epoch..."
        ):
            images = sample["image"].to(self.device)
            masks = {k: v.to(self.device) for k, v in sample["masks"].items()}

            with torch.no_grad():
                logits = self.model(images)

                # compute losses
                valid_loss, _ = self.criteria(
                    inputs=logits,
                    targets=masks,
                    weights=self.loss_weights,
                )
            total_valid_loss += valid_loss.item()

            self.writer.add_scalar(
                "loss/valid_loss",
                total_valid_loss,
                self.num_iters,
            )

    # ------------- VALID LOOP END -------------------------

    def update_dice_scores(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        clipping_masks: torch.Tensor = None,
    ) -> None:
        """
        Compute the Dice scores for each head.
        """
        with torch.no_grad():
            for head_name, head_logits in logits.items():
                if head_name not in self.dice_scores:
                    continue

                num_channels = head_logits.shape[1]
                dice_flavor_to_call_for_head = self.dice_scores_to_track[head_name]
                if dice_flavor_to_call_for_head == "soft_dice":
                    self.dice_scores[head_name] = (
                        1.0
                        - loss.dice_funcs[self.dice_scores_to_track[head_name]](
                            logits=head_logits,
                            target=targets[head_name],
                            num_classes=num_channels,
                        ).item()
                    )
                else:
                    self.dice_scores[head_name] = (
                        1.0
                        - loss.dice_funcs[self.dice_scores_to_track[head_name]](
                            logits=head_logits,
                            target=targets[head_name],
                        ).item()
                    )

    def save_checkpoint(self) -> None:
        """
        **NOTE**: the inference script is going to require 'model_configs' to instantiate
        the model for inference.
        """
        checkoint = {
            "epoch": self.epoch,
            "model_configs": self.model.configs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_loss": self.total_loss,
            "head_losses": self.head_losses,
            "dice_scores": self.dice_scores,
            "num_iters": self.num_iters,
            "freeze_heads": self.freeze_heads,
            "threshold_dice_scores": self.threshold_dice_scores,
            "device": self.device,
            "savedir": self.savedir,
            "training_parameters": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "clip_gradients": self.clip_gradients,
                "max_norm": self.max_norm,
            },
        }
        torch.save(
            checkoint,
            os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{self.epoch}.pth"),
        )
        logger.info(
            "Checkpoint saved at epoch %d to %s", self.epoch, self.checkpoints_dir
        )

    def log_training_performance(self) -> None:
        """
        Log training data for dashboard.
        """
        with torch.no_grad():
            for head_name, head_loss in self.head_losses.items():
                self.writer.add_scalar(
                    f"loss/{head_name}",
                    head_loss.item(),
                    self.num_iters,
                )

            self.writer.add_scalar(
                "loss/total_loss",
                self.total_loss.item(),
                self.num_iters,
            )

            for head_name, dice_score in self.dice_scores.items():
                self.writer.add_scalar(
                    f"dice_score/{head_name}",
                    dice_score,
                    self.num_iters,
                )


def main():
    """
    Run model training
    """

    args_parser = argparse.ArgumentParser(
        description="Train semantic segmentation model."
    )
    args_parser.add_argument(
        "--configs_path", default=None, help="Path to config YAML."
    )
    args_parser.add_argument("--images_dir", default=None, help="Path to images")
    args_parser.add_argument("--masks_dir", default=None, help="Path to masks")
    args_parser.add_argument(
        "--checkpoint_path", default=None, help="Path to checkpoint to resume."
    )

    args = args_parser.parse_args()

    configs_path = args.configs_path
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    checkpoint_path = args.checkpoint_path

    # ---------------------------
    # ----- Load configs --------
    # ---------------------------
    if configs_path is None:
        c = SegmentationModelConfigs() # load the default if no configs is provided
    else:
        c = SegmentationModelConfigs.from_yaml(configs_path)

    # ---------------------------
    # ----- Set up model --------
    # ---------------------------
    model = HopperNetLite(
        num_groups=c.num_groups,  # for GroupNorm
        out_channels=c.out_channels,
    )

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=c.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    torchinfo.summary(
        model,
    )
    logger.info("heads: %s", model.heads)

    # ---------------------------
    # ----- Load dataset --------
    # ---------------------------
    train_loader, valid_loader = init_datasets(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=c.batch_size,
        valid_split=c.valid_split,
        seed=c.random_seed,
    )

    savedir = os.path.join(
        c.savedir,
        c.model_name,
    )

    loss_criteria = loss.HopperNetCompositeLoss(
        loss_configs=c.loss_function_configs,
        device=c.device,
    )

    start_epoch = 0 if checkpoint is None else checkpoint["epoch"] + 1

    trainer = HopperNetTrainer(
        model=model,
        freeze_heads=c.freeze_heads,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criteria=loss_criteria,
        total_loss_weights=c.total_loss_weights,
        lr=c.learning_rate,
        weight_decay=c.weight_decay,
        start_epoch=start_epoch,
        num_epochs=c.epochs,
        checkpoint_every=c.checkpoint_every,
        log_every=c.log_every,
        clip_gradients=c.clip_gradients,
        max_norm=c.max_grad_norm,
        dice_scores_to_track=c.dice_scores_to_track,
        threshold_dice_scores=c.dice_thresholds_to_freeze_heads,
        device=c.device,
        savedir=savedir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
