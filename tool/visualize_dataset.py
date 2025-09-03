from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch

from cvt.dataset import create_dataloaders
from cvt.logger import setup_logger

logger = getLogger(__name__)


def visualize_synthetic_data(num_samples: int = 4):
    """Visualize synthetic text dataset samples"""
    logger.info("Visualizing synthetic dataset samples...")

    # Create dataloader
    train_loader, _ = create_dataloaders(batch_size=num_samples, num_workers=0)

    # Get one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    gt_masks = batch["gt_text"]

    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        # Denormalize image for display
        img = images[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # Convert to numpy and transpose for matplotlib
        img_np = img_denorm.permute(1, 2, 0).numpy()
        mask_np = gt_masks[i, 0].numpy()

        # Display image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Sample {i + 1}: Original Image")
        axes[0, i].axis("off")

        # Display mask
        axes[1, i].imshow(mask_np, cmap="gray")
        axes[1, i].set_title(f"Sample {i + 1}: Text Mask")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("./synthetic_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    logger.info("Saved visualization to ./synthetic_dataset_samples.png")
    logger.info(f"Image tensor shape: {images.shape}")
    logger.info(f"Mask tensor shape: {gt_masks.shape}")
    logger.info(f"Image value range: [{img_denorm.min():.3f}, {img_denorm.max():.3f}]")
    logger.info(f"Mask value range: [{gt_masks.min():.3f}, {gt_masks.max():.3f}]")


def analyze_dataset_statistics(num_batches: int = 10):
    """Analyze dataset statistics"""
    logger.info("Analyzing dataset statistics...")

    train_loader, val_loader = create_dataloaders(batch_size=8, num_workers=0)

    # Collect statistics
    text_ratios = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        gt_masks = batch["gt_text"]

        # Calculate text ratio for each sample
        for i in range(gt_masks.size(0)):
            mask = gt_masks[i, 0]
            text_pixels = (mask > 0.5).sum().item()
            total_pixels = mask.numel()
            text_ratio = text_pixels / total_pixels
            text_ratios.append(text_ratio)

    text_ratios = np.array(text_ratios)

    logger.info(f"Text ratio statistics (over {len(text_ratios)} samples):")
    logger.info(f"  Mean: {text_ratios.mean():.4f}")
    logger.info(f"  Std: {text_ratios.std():.4f}")
    logger.info(f"  Min: {text_ratios.min():.4f}")
    logger.info(f"  Max: {text_ratios.max():.4f}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(text_ratios, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Text Ratio (text pixels / total pixels)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Text Ratios in Synthetic Dataset")
    plt.grid(True, alpha=0.3)
    plt.savefig("./dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.show()

    logger.info("Saved statistics plot to ./dataset_statistics.png")


if __name__ == "__main__":
    setup_logger()  # 一度だけ呼ぶ

    # Visualize samples
    visualize_synthetic_data(num_samples=4)

    # Analyze statistics
    analyze_dataset_statistics(num_batches=20)
