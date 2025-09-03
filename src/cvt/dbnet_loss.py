import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # Dice coefficient
        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        # Dice loss (1 - dice coefficient)
        return 1.0 - dice


class MaskL1Loss(nn.Module):
    """L1 Loss with mask (for threshold supervision)"""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply mask and compute L1 loss
        loss = torch.abs(pred - target) * mask

        # Average over masked regions
        if mask.sum() > 0:
            return loss.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=pred.device)


class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross Entropy Loss for handling class imbalance"""

    def __init__(self, alpha: float = 5.0, beta: float = 10.0, ohem_ratio: float = 3.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive samples
        self.beta = beta  # Weight for negative samples
        self.ohem_ratio = ohem_ratio  # Online Hard Example Mining ratio

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Apply mask
        bce = bce * mask

        # Separate positive and negative samples
        positive_mask = (target > 0.5).float() * mask
        negative_mask = (target <= 0.5).float() * mask

        positive_count = positive_mask.sum()
        negative_count = negative_mask.sum()

        if positive_count > 0 and negative_count > 0:
            # Online Hard Example Mining for negative samples
            negative_loss = bce * negative_mask
            if negative_count > positive_count * self.ohem_ratio:
                # Select hardest negative samples
                hard_negative_count = int(positive_count * self.ohem_ratio)
                negative_loss_flat = negative_loss.view(-1)
                hard_negative_loss, _ = torch.topk(negative_loss_flat, hard_negative_count)
                negative_loss = hard_negative_loss.mean()
            else:
                negative_loss = negative_loss.sum() / negative_count

            # Positive loss
            positive_loss = (bce * positive_mask).sum() / positive_count

            # Balanced loss
            total_loss = self.alpha * positive_loss + self.beta * negative_loss
        elif positive_count > 0:
            total_loss = (bce * positive_mask).sum() / positive_count
        else:
            total_loss = torch.tensor(0.0, device=pred.device)

        return total_loss


class DBNetLoss(nn.Module):
    """Combined loss function for DBNet++"""

    def __init__(
        self,
        alpha: float = 1.0,  # Weight for probability map loss
        beta: float = 10.0,  # Weight for binary map loss
        gamma: float = 1.0,
    ):  # Weight for threshold map loss
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = BalancedBCELoss()
        self.l1_loss = MaskL1Loss()

    def _create_threshold_target_and_mask(
        self, gt_text: torch.Tensor, shrink_ratio: float = 0.4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create threshold supervision target and mask

        Args:
            gt_text: Ground truth text mask (B, 1, H, W)
            shrink_ratio: Ratio for shrinking text regions

        Returns:
            threshold_target: Target for threshold map
            threshold_mask: Mask indicating where to apply threshold supervision
        """
        batch_size, _, height, width = gt_text.shape
        threshold_target = torch.zeros_like(gt_text)
        threshold_mask = torch.zeros_like(gt_text)

        for i in range(batch_size):
            # Convert to numpy for processing
            text_mask = gt_text[i, 0].cpu().numpy()

            # Find text regions
            text_regions = (text_mask > 0.5).astype(np.uint8)

            if text_regions.sum() > 0:
                # Create distance transform from text boundaries
                from scipy.ndimage import binary_erosion, distance_transform_edt

                # Shrink text regions
                kernel_size = int(np.sqrt(text_regions.sum()) * shrink_ratio)
                if kernel_size > 1:
                    shrunk = binary_erosion(text_regions, iterations=kernel_size)
                else:
                    shrunk = text_regions

                # Distance from boundary
                distance_inside = distance_transform_edt(text_regions)
                distance_outside = distance_transform_edt(1 - text_regions)

                # Create threshold target (higher values inside text)
                threshold_map = np.zeros_like(text_mask)
                threshold_map[text_regions > 0] = distance_inside[text_regions > 0] / (
                    distance_inside[text_regions > 0].max() + 1e-6
                )

                # Create mask for supervision (focus on boundary regions)
                boundary_distance = np.minimum(distance_inside, distance_outside)
                boundary_mask = (boundary_distance <= 3).astype(np.float32)  # Focus on 3-pixel boundary

                threshold_target[i, 0] = torch.from_numpy(threshold_map).to(gt_text.device)
                threshold_mask[i, 0] = torch.from_numpy(boundary_mask).to(gt_text.device)

        return threshold_target, threshold_mask

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dictionary containing 'prob_map', 'binary_map', 'thresh_map'
            targets: Dictionary containing 'gt_text' (ground truth text mask)

        Returns:
            Dictionary of losses
        """
        pred_prob = predictions["prob_map"]  # (B, 1, H, W)
        pred_binary = predictions["binary_map"]  # (B, 1, H, W)
        pred_thresh = predictions["thresh_map"]  # (B, 1, H, W)

        gt_text = targets["gt_text"]  # (B, 1, H, W)

        # Create training mask (exclude ignore regions if any)
        if "training_mask" in targets:
            training_mask = targets["training_mask"]
        else:
            training_mask = torch.ones_like(gt_text)

        # 1. Probability map loss (segmentation loss)
        prob_dice_loss = self.dice_loss(pred_prob, gt_text)
        prob_bce_loss = self.bce_loss(pred_prob, gt_text, training_mask)
        prob_loss = prob_dice_loss + prob_bce_loss

        # 2. Binary map loss (final output loss)
        binary_dice_loss = self.dice_loss(pred_binary, gt_text)
        binary_bce_loss = self.bce_loss(pred_binary, gt_text, training_mask)
        binary_loss = binary_dice_loss + binary_bce_loss

        # 3. Threshold map loss (boundary supervision)
        thresh_target, thresh_mask = self._create_threshold_target_and_mask(gt_text)
        thresh_loss = self.l1_loss(pred_thresh, thresh_target, thresh_mask)

        # Combined loss
        total_loss = self.alpha * prob_loss + self.beta * binary_loss + self.gamma * thresh_loss

        return {
            "total_loss": total_loss,
            "prob_loss": prob_loss,
            "binary_loss": binary_loss,
            "thresh_loss": thresh_loss,
            "prob_dice": prob_dice_loss,
            "prob_bce": prob_bce_loss,
            "binary_dice": binary_dice_loss,
            "binary_bce": binary_bce_loss,
        }


def create_dbnet_loss(alpha: float = 1.0, beta: float = 10.0, gamma: float = 1.0) -> DBNetLoss:
    """Factory function to create DBNet loss"""
    return DBNetLoss(alpha=alpha, beta=beta, gamma=gamma)
