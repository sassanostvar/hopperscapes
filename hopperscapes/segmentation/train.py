"""
Train the multi-task semantic segmentation model.

Current features include:
    - learning rate warmup and annealing
    - optional dynamic head freeze/unfreeze based on
        validation Dice scores and pre-set thresholds
    - checkpointing
    - TensorBoard logging
    - best model selection based on validation loss
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import torchinfo
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from hopperscapes.configs import SegmentationModelConfigs
from hopperscapes.segmentation import loss
from hopperscapes.segmentation.dataset import WingPatternDataset, hopper_collate_fn
from hopperscapes.segmentation.models import ModularHopperNet

logger = logging.getLogger("HopperNetTrainingLog")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


_SCHEDULER_FACTORY = {
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "linear": optim.lr_scheduler.LinearLR,
}


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
    configs: SegmentationModelConfigs,
    batch_size: int,
    valid_split: float,
):
    """
    Create a random train/valid split and generate the DataLoaders.

    Args:
        images_dir (str): Path to image directory.
        masks_dir (str): Path to the masks root directory.
        configs (SegmentationModelConfigs): Model/dataset configurations.
        batch_size (int): Batch size for DataLoaders.
        valid_split (float): Fraction of data used for validation.

    Returns:
        train_loader (DataLoader): Train DataLoader.
        valid_loader (DataLoader): Valid DataLoader.
    """

    dataset = WingPatternDataset(
        image_dir=images_dir, masks_dir=masks_dir, configs=configs
    )

    # split
    dataset_size = len(dataset)
    valid_size = int(dataset_size * valid_split)
    train_size = dataset_size - valid_size

    # if configs.device == "cpu":
    #     num_workers = configs.num_workers
    # else:
    #     num_workers = 1

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=hopper_collate_fn,
        shuffle=True,
        drop_last=False,
        # num_workers=num_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=hopper_collate_fn,
        shuffle=False,
        drop_last=False,
        # num_workers=num_workers,
    )

    return train_loader, valid_loader


def _verify_configs(c: SegmentationModelConfigs):
    """
    Sanity checks on configs.
    """
    assert c.total_loss_weights, "Total loss weights must be defined in configs."
    if c.lr_scheduler is not None:
        assert c.lr_scheduler in _SCHEDULER_FACTORY, "Invalid learning rate scheduler."

    assert (
        set(c.dice_scores_to_track)
        == set(c.freeze_heads)
        == set(c.dice_thresholds_to_freeze_heads)
    ), "Dice scores to track, freeze heads, and thresholds must have the same keys."


class HopperNetTrainer:
    """
    Trainer for the ModularHopperNet model.

    Args:
        model (nn.Module): The ModularHopperNet instance.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        criteria (nn.Module): Loss function. Currently, loss.HopperNetCompositeLoss.
        savedir (str): Directory to save checkpoints and logs.
        configs (SegmentationModelConfigs): Model and training configurations.
        start_iter (int): Starting iteration number.
        start_epoch (int): Starting epoch number.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criteria: nn.Module,
        savedir: str,
        configs: SegmentationModelConfigs,
        start_iter: int,
        start_epoch: int,
    ):
        if savedir is None:
            raise ValueError("Please provide a valid directory to save the model.")

        if not isinstance(model, ModularHopperNet):
            raise ValueError("Model must be an instance of ModularHopperNet.")

        _verify_configs(configs)

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criteria = criteria
        self.savedir = savedir
        #
        self.device = configs.device

        # model heads can be frozen/unfrozen dynamically if specified;
        # these are the various trackers and switches that are used
        # to implement a feedback control system with patience/delay.
        # a next iteration of the code will likely implement a dedicated
        # "FreezeController" class to isolate the complexity.
        self.enable_dynamic_freeze = configs.enable_dynamic_freeze
        self.freeze_heads = configs.freeze_heads
        self.freeze_patience = configs.freeze_patience
        self.active_heads = dict.fromkeys(self.freeze_heads.keys(), None)
        self.epochs_above_threshold = dict.fromkeys(self.freeze_heads.keys(), 0)
        self.epochs_below_threshold = dict.fromkeys(self.freeze_heads.keys(), 0)
        self.dice_scores_to_track = configs.dice_scores_to_track or {}
        self.threshold_dice_scores = configs.dice_thresholds_to_freeze_heads or {}

        # "loss_weights" is a live copy of "total_loss_weights"
        # that is updated during training to reflect the current
        # state of the model heads.
        self.global_loss_weights = configs.total_loss_weights
        self.loss_weights = self.global_loss_weights.copy()
        #
        self.lr = configs.learning_rate
        self.weight_decay = configs.weight_decay
        self.lr_scheduler_kind = configs.lr_scheduler
        self.lr_scheduler_params = configs.lr_scheduler_params or {}
        self.warmup_epochs = configs.warmup_epochs
        self.warmup_lr = configs.warmup_lr
        #
        self.num_epochs = configs.epochs
        self.checkpoint_every = configs.checkpoint_every
        self.log_every = configs.log_every
        self.clip_gradients = configs.clip_gradients
        self.max_norm = configs.max_grad_norm

        self.checkpoints_dir = os.path.join(savedir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.logs_dir = os.path.join(savedir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.writer = SummaryWriter(self.logs_dir)

        self.model.to(self.device)

        # heads that are frozen at the start will remain frozen,
        # even if the dynamic freeze is enabled.
        if self.freeze_heads:
            for head_name, freeze in self.freeze_heads.items():
                if freeze:
                    self.loss_weights[head_name] = 0.0
                    logger.info("freezing head: %s", head_name)
                    for p in self.model.heads[head_name].parameters():
                        p.requires_grad = False
                    self.active_heads[head_name] = False
                else:
                    self.active_heads[head_name] = True

        # Start the optimizer with the model parameters, excluding frozen heads.
        # Any subsequent change in the model heads will trigger a reinitialization
        # of the optimizer in the `update_heads_configuration` method.
        self.optimizer = optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Chain a warmup scheduler and/or a main lr scheduler,
        # based on the configs.
        self._setup_lr_schedulers()

        self.num_iters = 1 if start_iter is None else start_iter
        self.start_epoch = start_epoch

        self.total_loss = None
        self.head_losses = None
        self.dice_scores = dict.fromkeys(self.model.heads.keys(), 0.0)
        self.best_avg_valid_loss = 1.0e5
        self.is_best = False

    def train(self) -> None:
        """
        Run model training with optional dynamic freezing/unfreezing of model
        heads based on pre-set threshold Dice scores.
        """
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.epoch = epoch
            _current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "----- Starting train epoch %d with lr %f -----",
                self.epoch,
                _current_lr,
            )

            # 1. train for an epoch
            self.model.train()
            self.train_epoch()

            # 2. log training performance
            if self.num_iters % self.log_every == 0:
                self.log_training_performance()

            # 3. run validation epoch
            self.valid_epoch()

            # 4. evaluate the heads configuration
            # and make adjustments if necessary
            if self.enable_dynamic_freeze and (
                self.epoch > (self.warmup_epochs + self.start_epoch)
            ):
                self.update_heads_configuration()

            # 5. checkpoint
            if self.epoch % self.checkpoint_every == 0:
                self.save_checkpoint()

        self.writer.close()

    def update_heads_configuration(self) -> None:
        """
        Accounting and readjustment of model heads based on the
        eval Dice scores.
        """
        # 1. update the Dice scores
        self._update_dice_score_crossover_counters()

        # 2. update the freeze status of heads
        state_changed = self._update_heads_freeze_status()

        # 3. sync model and optimizer if there was a change
        # (initializes a new optimizer with updated parameters)
        if state_changed:
            self._sync_model_and_optimizer()

    def _update_dice_score_crossover_counters(self) -> None:
        """
        Analyze the Dice scores for each head to determine if any heads
        should be frozen or unfrozen based on the configured thresholds.
        """

        for head_name, threshold in self.threshold_dice_scores.items():
            if head_name not in self.dice_scores:
                continue

            # if a head is frozen at the start,
            # it will remain frozen
            if self.freeze_heads[head_name] is True:
                continue

            if self.dice_scores[head_name] > threshold:
                self.epochs_below_threshold[head_name] = 0
                self.epochs_above_threshold[head_name] += 1
            else:
                self.epochs_above_threshold[head_name] = 0
                self.epochs_below_threshold[head_name] += 1

    def _update_heads_freeze_status(self) -> bool:
        """
        Update the freeze status of model heads based on the number of epochs
        spent above or below the threshold Dice score.
        """
        state_changed = False
        # for heads that have been above the threshold for long enough,
        # freeze them
        for head_name, epochs_above in self.epochs_above_threshold.items():
            if epochs_above >= self.freeze_patience:
                if self.active_heads[head_name] is True:
                    self._freeze_head(head_name)
                    self.active_heads[head_name] = False
                    self.epochs_above_threshold[head_name] = 0
                    state_changed = True

        # and for heads that have been frozen for too long, unfreeze them
        for head_name, epochs_below in self.epochs_below_threshold.items():
            if epochs_below >= self.freeze_patience:
                if self.active_heads[head_name] is False:
                    self._unfreeze_head(head_name)
                    self.active_heads[head_name] = True
                    self.epochs_below_threshold[head_name] = 0
                    state_changed = True

        return state_changed

    def _freeze_head(self, head_name: str) -> None:
        """
        Freeze a specific model head.
        """
        if head_name not in self.model.heads:
            raise ValueError(f"Head '{head_name}' does not exist in the model.")

        logger.info("freezing head: %s", head_name)
        self.loss_weights[head_name] = 0.0
        for p in self.model.heads[head_name].parameters():
            p.requires_grad = False

    def _unfreeze_head(self, head_name: str) -> None:
        """
        Unfreeze a specific model head.
        """
        if head_name not in self.model.heads:
            raise ValueError(f"Head '{head_name}' does not exist in the model.")

        logger.info("unfreezing head: %s", head_name)
        weight = self.global_loss_weights.get(head_name, None)
        if weight is None:
            raise ValueError(
                f"Head '{head_name}' has a freeze threshold but no global loss weight."
            )
        self.loss_weights[head_name] = weight

        for p in self.model.heads[head_name].parameters():
            p.requires_grad = True

    def _sync_model_and_optimizer(self) -> None:
        """
        Update the optimizer after model heads have been frozen/unfrozen.

        ** NOTE ** this assumes uniform hyperparameters across param groups.
        ** NOTE ** this is a hard freeze and unfreeze, meaning that
        a fresh optimizer is created with the updated model parameters.
        """

        # 1. re-initialize optimizer with updated parameters
        current_lr = self.optimizer.param_groups[0]["lr"]
        logger.info("reinitializing optimizer with current lr: %f", current_lr)

        old_optimizer_state = self.optimizer.state_dict()
        logger.info("reinitializing optimizer with updated model parameters.")
        self.optimizer = optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=current_lr,
            weight_decay=self.weight_decay,
        )

        # map id(param) -> param for all param groups in
        # the newly initialized optimizer
        current_id2param = {
            id(p): p for g in self.optimizer.param_groups for p in g["params"]
        }

        # only restore the params that are still trainable
        for param_id, state_data in old_optimizer_state["state"].items():
            # param = current_id2param[param_id]
            param = current_id2param.get(param_id, None)
            if param is None:
                continue

            self.optimizer.state[param] = {
                k: (v.to(param.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in state_data.items()
            }

        logger.info(
            "optimizer reinitialized with learning rate: %f",
            self.optimizer.param_groups[0]["lr"],
        )

        # 2. re-initialize the learning rate scheduler if it exists
        if self.lr_scheduler is None:
            return

        logger.info("reinitializing learning rate scheduler with old state.")
        old_lr_scheduler_state = (
            self.lr_scheduler.state_dict() if self.lr_scheduler else None
        )
        self._setup_lr_schedulers()
        self.lr_scheduler.load_state_dict(old_lr_scheduler_state)

        # IMPORTANT: sync the learning rate in the optimizer,
        # otherwise it will get reset to the initial learning rate.
        try:
            last_lr = self.lr_scheduler.get_last_lr()
        except Exception:
            last_lr = [g["lr"] for g in self.optimizer.param_groups]

        for g, lr in zip(self.optimizer.param_groups, last_lr):
            g["lr"] = lr

        logger.info("last lr from scheduler: %s", last_lr)
        logger.info("current lr in optimizer %f", self.optimizer.param_groups[0]["lr"])

    def _setup_lr_schedulers(self):
        """
        Pipeline a warmup scheduler and/or a main lr scheduler.
        """
        warmup_scheduler = None
        main_lr_scheduler = None
        self.lr_scheduler = None

        # record-keeping for the checkpoint
        self.lr_scheduler_configs = {"warmup": {}, "main": {}}

        # lr warmup scheduler
        if self.warmup_epochs > 0:
            _args = {
                "start_factor": self.warmup_lr / self.lr,
                "total_iters": self.warmup_epochs,
            }
            warmup_scheduler = _SCHEDULER_FACTORY["linear"](self.optimizer, **_args)
            self.lr_scheduler_configs["warmup"]["kind"] = "linear"
            self.lr_scheduler_configs["warmup"]["args"] = _args
        else:
            warmup_scheduler = None

        # main lr scheduler
        if self.lr_scheduler_kind is None:
            main_lr_scheduler = None
        else:
            main_lr_scheduler = _SCHEDULER_FACTORY[self.lr_scheduler_kind](
                self.optimizer,
                **self.lr_scheduler_params,
            )
            self.lr_scheduler_configs["main"]["kind"] = self.lr_scheduler_kind
            self.lr_scheduler_configs["main"]["args"] = self.lr_scheduler_params

        if warmup_scheduler and main_lr_scheduler:
            self.lr_scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_lr_scheduler],
                milestones=[self.warmup_epochs],
            )
        elif warmup_scheduler:
            self.lr_scheduler = warmup_scheduler
        elif main_lr_scheduler:
            self.lr_scheduler = main_lr_scheduler

    def setup_on_resume(self):
        """
        Make sure the heads configuration is restored
        correctly when resuming from a checkpoint.
        """
        logger.info("Setting up model and optimizer on resume...")
        for head_name, active in self.active_heads.items():
            if active is False:
                self._freeze_head(head_name)
            else:
                self._unfreeze_head(head_name)

        self._sync_model_and_optimizer()

    def train_epoch(self) -> None:
        """
        Run one training epoch.
        """

        for sample in self.train_loader:
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

            # increment
            self.num_iters += 1

        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def valid_epoch(self) -> None:
        """
        Run one validation epoch.
        """
        self.is_best = False
        self.model.eval()
        logger.info("----- Starting valid epoch %d-----", self.epoch)
        total_valid_loss = 0.0
        total_valid_dice = {k: 0.0 for k in self.dice_scores_to_track}
        n_valid_batches = len(self.valid_loader)
        for sample in self.valid_loader:
            images = sample["image"].to(self.device)
            masks = {k: v.to(self.device) for k, v in sample["masks"].items()}

            with torch.no_grad():
                logits = self.model(images)

                # compute losses and accumulate
                valid_loss, _ = self.criteria(
                    inputs=logits,
                    targets=masks,
                    weights=self.loss_weights,
                )
                total_valid_loss += valid_loss.item()

                # compute dice scores for batch and accumulate
                batch_dice_scores = self.compute_dice_scores(logits, masks)
                for head_name, score in batch_dice_scores.items():
                    total_valid_dice[head_name] += score

        # record the epoch average dice scores
        for head_name in total_valid_dice:
            if n_valid_batches > 0:
                self.dice_scores[head_name] = (
                    total_valid_dice[head_name] / n_valid_batches
                )

        if n_valid_batches > 0:
            _avg_valid_loss = total_valid_loss / n_valid_batches
        else:
            _avg_valid_loss = 0.0

        self.writer.add_scalar(
            "loss/valid_loss",
            _avg_valid_loss,
            self.epoch,
        )

        if _avg_valid_loss < self.best_avg_valid_loss:
            self.is_best = True
            self.best_avg_valid_loss = _avg_valid_loss

        for head_name, score in self.dice_scores.items():
            self.writer.add_scalar(
                f"dice_score/valid_{head_name}",
                score,
                self.epoch,
            )

    def compute_dice_scores(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        clipping_masks: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute the Dice scores for each head.

        Args:
            logits (Dict[str, torch.Tensor]): Model outputs.
            targets (Dict[str, torch.Tensor]): Ground truth masks.
            clipping_masks (Dict[str, torch.Tensor], optional): Masks to clip the Dice scores.

        Returns:
            Dict[str, float]: Dictionary mapping head_name to Dice score.
        """
        with torch.no_grad():
            batch_scores = {}
            for head_name, head_logits in logits.items():
                if head_name not in self.dice_scores_to_track:
                    continue

                num_channels = head_logits.shape[1]
                dice_flavor_to_call_for_head = self.dice_scores_to_track[head_name]
                if dice_flavor_to_call_for_head == "soft_dice":
                    score = (
                        1.0
                        - loss.dice_funcs[self.dice_scores_to_track[head_name]](
                            logits=head_logits,
                            target=targets[head_name],
                            num_classes=num_channels,
                        ).item()
                    )
                else:
                    score = (
                        1.0
                        - loss.dice_funcs[self.dice_scores_to_track[head_name]](
                            logits=head_logits,
                            target=targets[head_name],
                        ).item()
                    )
                batch_scores[head_name] = score
        return batch_scores

    def save_checkpoint(self) -> None:
        """
        Save the model checkpoint.
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_id": self.model.__class__.__name__,
            "model_configs": self.model.configs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_configs": self.lr_scheduler_configs,
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "total_loss": self.total_loss.item()
            if self.total_loss is not None
            else None,
            "head_losses": {k: v.item() for k, v in self.head_losses.items()}
            if self.head_losses is not None
            else None,
            "best_avg_valid_loss": self.best_avg_valid_loss,
            "is_best": self.is_best,
            "num_iters": self.num_iters,
            "device": self.device,
            "savedir": self.savedir,
            "training_parameters": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "clip_gradients": self.clip_gradients,
                "max_norm": self.max_norm,
                "global_loss_weights": self.global_loss_weights,
            },
            "control_state": {
                "enable_dynamic_freeze": self.enable_dynamic_freeze,
                "freeze_heads": self.freeze_heads,
                "freeze_patience": self.freeze_patience,
                "active_heads": self.active_heads,
                "epochs_above_threshold": self.epochs_above_threshold,
                "epochs_below_threshold": self.epochs_below_threshold,
                "loss_weights": self.loss_weights,
                "dice_scores": self.dice_scores,
                "threshold_dice_scores": self.threshold_dice_scores,
            },
        }
        torch.save(
            checkpoint,
            os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{self.epoch}.pth"),
        )
        logger.info(
            "Checkpoint saved at epoch %d to %s", self.epoch, self.checkpoints_dir
        )

        if self.is_best:
            best_checkpoint_path = os.path.join(
                self.checkpoints_dir, "best_checkpoint.pth"
            )
            torch.save(checkpoint, best_checkpoint_path)
            logger.info("Best checkpoint saved to %s", best_checkpoint_path)

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

            self.writer.add_scalar(
                "hyperparameters/lr",
                self.optimizer.param_groups[0]["lr"],
                self.num_iters,
            )

    def load_control_flow_states_from_checkpoint(self, checkpoint: Dict) -> None:
        """
        Load the optimizer, lr scheduler, and best model checkpointing states and
        controls from a checkpoint.
        """
        # load optimizer and lr scheduler states
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            "lr_scheduler_state_dict" in checkpoint
            and checkpoint["lr_scheduler_state_dict"] is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            logger.info(
                "Resume LRs: %s", [g["lr"] for g in self.optimizer.param_groups]
            )

        # restore best checkpoint data and control state
        self.best_avg_valid_loss = checkpoint.get("best_avg_valid_loss", 1.0e5)
        self.is_best = checkpoint.get("is_best", False)
        for attr in checkpoint["control_state"]:
            setattr(self, attr, checkpoint["control_state"][attr])

        # restore heads configuration and update
        # the optimizer and lr scheduler accordingly.
        self.setup_on_resume()


def main(args):
    """
    Run model training
    """

    configs_path = args.configs_path
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    checkpoint_path = args.checkpoint_path

    # ---------------------------
    # ----- Load configs --------
    # ---------------------------
    if configs_path is None:
        c = SegmentationModelConfigs()  # load the default if no configs is provided
    else:
        c = SegmentationModelConfigs.from_yaml(configs_path)

    # ---------------------------
    # ----- Set up model --------
    # ---------------------------
    model = ModularHopperNet(
        num_groups=c.num_groups,
        out_channels=c.out_channels,
        encoder_configs=c.encoder_configs,
        bottleneck_configs=c.bottleneck_configs,
        decoder_configs=c.decoder_configs,
    )

    # loading the model from a checkpoint if provided;
    # requires additional steps to load the optimizer
    # and lr scheduler states after the Trainer is initialized.
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=c.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    torchinfo.summary(
        model,
    )
    logger.info("heads: %s", model.heads)

    # init random seeds
    init_seeds(c.random_seed)

    # ---------------------------
    # ----- Load dataset --------
    # ---------------------------
    train_loader, valid_loader = init_datasets(
        images_dir=images_dir,
        masks_dir=masks_dir,
        configs=c,
        batch_size=c.batch_size,
        valid_split=c.valid_split,
    )

    savedir = os.path.join(
        c.savedir,
        c.experiment_id,
    )

    loss_criteria = loss.HopperNetCompositeLoss(
        loss_configs=c.loss_function_configs,
        device=c.device,
    )

    start_epoch = 0 if checkpoint is None else checkpoint["epoch"] + 1
    start_iter = 1 if checkpoint is None else checkpoint["num_iters"] + 1

    trainer = HopperNetTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criteria=loss_criteria,
        savedir=savedir,
        configs=c,
        start_epoch=start_epoch,
        start_iter=start_iter,
    )

    if checkpoint is not None:
        trainer.load_control_flow_states_from_checkpoint(checkpoint)

    trainer.train()


if __name__ == "__main__":
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

    main(args)
