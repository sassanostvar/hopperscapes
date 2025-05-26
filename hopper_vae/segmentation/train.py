import os

from typing import Dict

import pandas as pd
import torch
from torch import nn

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hopper_vae.configs import TrainingConfigs
from hopper_vae.segmentation.data_io import WingPatternDataset, hopper_collate_fn
from hopper_vae.segmentation.models import HopperNet
from hopper_vae.segmentation import loss

import torchinfo

from tqdm import tqdm


class HopperNetTrainer:
    def __init__(
        self,
        model: nn.Module,
        freeze_heads: bool = None,
        train_loader: DataLoader = None,
        # valid_loader: DataLoader = None,
        criteria: nn.Module = None,
        total_loss_weights: Dict[str, float] = None,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        start_epoch: int = 0,
        num_epochs: int = 100,
        checkpoint_every: int = 10,
        log_every: int = 10,
        clip_gradients: bool = True,
        max_norm: float = None,
        dice_scores_to_track: dict = None,
        threshold_dice_scores: dict = None,
        device: str = "cpu",
        savedir: str = None,
    ):
        if savedir is None:
            raise ValueError("Please provide a valid directory to save the model.")

        self.model = model
        self.freeze_heads = freeze_heads
        self.train_loader = train_loader
        # self.valid_loader = valid_loader
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
                    print(f"Freezing head: {head_name}")
                    print(f"loss weights: {self.loss_weights}")
                    for p in self.model.heads[head_name].parameters():
                        p.requires_grad = False

        self.num_iters = 1
        self.epoch = start_epoch
        self.total_loss = None
        self.head_losses = None

        self.dice_scores = dict.fromkeys(self.model.heads.keys(), 0.0)

    def train(self):
        for i in range(self.num_epochs):
            self.epoch += 1
            print(f"----- Starting epoch {self.epoch}/{self.num_epochs} -----")

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
                        self.loss_weights[head_name] = 0.0
                        for p in self.model.heads[head_name].parameters():
                            p.requires_grad = False
                    else:
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
            # self.validate_epoch()

    def train_epoch(self):
        n_batches = 0
        for sample in tqdm(self.train_loader):
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
                # clipping_masks={
                #     "veins": masks["wing"],
                #     # "domains": masks["wing"],
                # },
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

    def update_dice_scores(self, logits=None, targets=None, clipping_masks=None):
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

    def validate_epoch(self):
        pass

    def save_checkpoint(self):
        checkoint = {
            "epoch": self.epoch,
            "heads": self.model.heads,
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
        }
        torch.save(
            checkoint,
            os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{self.epoch}.pth"),
        )
        print(f"Checkpoint saved at epoch {self.epoch} to {self.checkpoints_dir}")

    def log_training_performance(self):
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
    apply training loop:
    """

    # dataset = WingPatternDataset(image_dir="data/raw/train", masks_dir="data/raw/train")
    dataset = WingPatternDataset(
        image_dir="data/aug2/train/images", masks_dir="data/aug2/train/masks"
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=hopper_collate_fn,
        shuffle=True,
        drop_last=False,
    )

    print(f"data loader length: {len(train_loader)}")
    print(f"data loader batch size: {train_loader.batch_size}")

    c = TrainingConfigs()
    c.model_name = "hopper_net_aug2_test4"

    SAVEDIR = os.path.join(
        c.savedir,
        c.model_name,
    )

    CRITERIA = loss.HopperNetCompositeLoss(
        loss_configs=c.loss_function_configs,
        device=c.device,
    )

    model = HopperNet(
        num_groups=1,  # for GroupNorm
        # heads=c.seg_configs.heads,
        out_channels=c.seg_configs.out_channels,
    )

    checkpoint = None
    # checkpoint_path = './outputs/models/hopper_net_aug2_test2/checkpoints/checkpoint_epoch_70.pth'
    # checkpoint = torch.load(checkpoint_path, map_location=c.device)
    # model.load_state_dict(
    #     checkpoiint["model_state_dict"]
    # )

    torchinfo.summary(
        model,
    )
    print(f"heads: {model.heads}")

    start_epoch = 0 if checkpoint is None else checkpoint["epoch"] + 1

    trainer = HopperNetTrainer(
        model=model,
        freeze_heads=c.freeze_heads,
        train_loader=train_loader,
        criteria=CRITERIA,
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
        savedir=SAVEDIR,
    )
    trainer.train()


if __name__ == "__main__":
    main()
