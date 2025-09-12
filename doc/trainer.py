from __future__ import annotations

import time
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import track

from ..src.cvt.config import TrainingConfig
from .dataset import create_dataloaders
from .dbnet import DBNet
from .dbnet_loss import create_dbnet_loss
from .metrics import AverageMeter, MetricsTracker, compute_text_detection_metrics, format_metrics_string

logger = getLogger(__name__)


class DBNetTrainer:
    """DBNet++ trainer with automatic device selection and mixed precision"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        # Mixed precision setup
        if config.mixed_precision and self.device.type in ["cuda"]:
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False

        # Initialize components
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.early_stopping_counter = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")

    def _setup_device(self) -> torch.device:
        """Setup device automatically or from config"""
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        return device

    def _setup_model(self) -> nn.Module:
        """Initialize DBNet++ model"""
        model = DBNet(backbone_type=self.config.backbone)
        model = model.to(self.device)

        # Load checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        return model

    def _setup_criterion(self):
        """Initialize loss function"""
        return create_dbnet_loss(**self.config.loss_weights)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer"""
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _setup_scheduler(self):
        """Initialize learning rate scheduler"""
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train_one_epoch(self, train_loader, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()

        # Average meters for tracking
        total_loss_meter = AverageMeter("TotalLoss")
        prob_loss_meter = AverageMeter("ProbLoss")
        binary_loss_meter = AverageMeter("BinaryLoss")
        thresh_loss_meter = AverageMeter("ThreshLoss")

        for batch_idx, batch in enumerate(track(train_loader, description=f"Epoch {epoch}")):
            # Move data to device
            images = batch["image"].to(self.device)
            gt_text = batch["gt_text"].to(self.device)
            training_mask = batch["training_mask"].to(self.device)

            targets = {"gt_text": gt_text, "training_mask": training_mask}

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                from torch.cuda.amp import autocast

                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets)

                # Backward pass
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)

                # Backward pass
                losses["total_loss"].backward()
                self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            total_loss_meter.update(losses["total_loss"].item(), batch_size)
            prob_loss_meter.update(losses["prob_loss"].item(), batch_size)
            binary_loss_meter.update(losses["binary_loss"].item(), batch_size)
            thresh_loss_meter.update(losses["thresh_loss"].item(), batch_size)

            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss_meter.avg:.4f} | LR: {lr:.6f}"
                )

        # Return epoch summary
        return {
            "total_loss": total_loss_meter.avg,
            "prob_loss": prob_loss_meter.avg,
            "binary_loss": binary_loss_meter.avg,
            "thresh_loss": thresh_loss_meter.avg,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, val_loader, epoch: int) -> dict:
        """Validate model"""
        self.model.eval()

        total_loss_meter = AverageMeter("ValTotalLoss")
        prob_loss_meter = AverageMeter("ValProbLoss")
        binary_loss_meter = AverageMeter("ValBinaryLoss")
        thresh_loss_meter = AverageMeter("ValThreshLoss")

        all_metrics = []

        with torch.no_grad():
            for batch in track(val_loader, description=f"Validation {epoch}"):
                # Move data to device
                images = batch["image"].to(self.device)
                gt_text = batch["gt_text"].to(self.device)
                training_mask = batch["training_mask"].to(self.device)

                targets = {"gt_text": gt_text, "training_mask": training_mask}

                # Forward pass
                if self.use_amp:
                    from torch.cuda.amp import autocast

                    with autocast():
                        outputs = self.model(images)
                        losses = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets)

                # Update loss meters
                batch_size = images.size(0)
                total_loss_meter.update(losses["total_loss"].item(), batch_size)
                prob_loss_meter.update(losses["prob_loss"].item(), batch_size)
                binary_loss_meter.update(losses["binary_loss"].item(), batch_size)
                thresh_loss_meter.update(losses["thresh_loss"].item(), batch_size)

                # Compute detection metrics
                detection_metrics = compute_text_detection_metrics(outputs["binary_map"], gt_text)
                all_metrics.append(detection_metrics)

        # Average detection metrics
        avg_detection_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_detection_metrics[f"val_{key}"] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return {
            "val_total_loss": total_loss_meter.avg,
            "val_prob_loss": prob_loss_meter.avg,
            "val_binary_loss": binary_loss_meter.avg,
            "val_thresh_loss": thresh_loss_meter.avg,
            **avg_detection_metrics,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
        }

        # Save regular checkpoint
        if epoch % self.config.save_interval == 0:
            checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")

    def train(self):
        """Main training loop"""
        logger.info("Starting DBNet++ training...")
        logger.info(f"Config: {self.config}")

        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            batch_size=self.config.batch_size, num_workers=self.config.num_workers, img_size=self.config.img_size
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # Validation phase
            val_metrics = self.validate(val_loader, epoch)

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Check for best model
            current_val_loss = val_metrics["val_total_loss"]
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, all_metrics, is_best)

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch}/{self.config.epochs} Summary:")
            logger.info(f"  Time: {epoch_time:.2f}s")
            logger.info(f"  Train: {format_metrics_string(train_metrics, 'train')}")
            logger.info(f"  Val: {format_metrics_string(val_metrics, 'val')}")
            logger.info(f"  Best Epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})")

            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        logger.info(f"Best model: Epoch {self.best_epoch} with validation loss {self.best_val_loss:.4f}")

        return self.best_epoch, self.best_val_loss


if __name__ == "__main__":
    from .logger import setup_logger

    setup_logger()

    # Create config
    config = TrainingConfig(
        epochs=5,  # Short test run
        batch_size=4,
        learning_rate=1e-3,
        save_interval=2,
    )

    # Create trainer
    trainer = DBNetTrainer(config)

    # Start training
    best_epoch, best_loss = trainer.train()

    logger.info(f"Training finished! Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")
