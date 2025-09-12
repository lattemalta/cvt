from logging import getLogger
from typing import Any

import lightning as L
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models.detection as detection_models
from torch.optim import Optimizer

logger = getLogger(__name__)


class PretrainedObjectDetector(L.LightningModule):
    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        num_classes: int = 80,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.lr = lr

        # Load pretrained model
        self.model = self._load_pretrained_model()

        # Set model to evaluation mode for inference
        self.model.eval()

        # For COCO evaluation
        self.val_predictions = []
        self.val_targets = []

    def _load_pretrained_model(self) -> nn.Module:
        """Load pretrained object detection model from torchvision."""
        if self.model_name == "fasterrcnn_resnet50_fpn":
            model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
        elif self.model_name == "retinanet_resnet50_fpn":
            model = detection_models.retinanet_resnet50_fpn(pretrained=True)
        elif self.model_name == "ssd300_vgg16":
            model = detection_models.ssd300_vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Forward pass for inference."""
        return self.model(images)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> dict[str, Any]:
        """Prediction step for inference."""
        images, targets = batch

        # Run inference
        predictions = self.model(images)

        # Post-process predictions
        processed_predictions = []
        for pred in predictions:
            # Filter by confidence threshold
            keep_idx = pred["scores"] > self.confidence_threshold

            processed_pred = {
                "boxes": pred["boxes"][keep_idx],
                "scores": pred["scores"][keep_idx],
                "labels": pred["labels"][keep_idx],
            }
            processed_predictions.append(processed_pred)

        return {"predictions": processed_predictions, "targets": targets, "images": images}

    def validation_step(self, batch: tuple[torch.Tensor, list[dict[str, Any]]], _batch_idx: int) -> dict[str, Any]:
        """Validation step for metric computation."""
        images, targets = batch

        # Set model to eval mode
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(images)

        # Store predictions and targets for COCO evaluation
        for pred, target in zip(predictions, targets, strict=True):
            # Convert predictions to COCO format
            image_id = target["image_id"].item()

            # Filter by confidence threshold
            keep_idx = pred["scores"] > self.confidence_threshold
            boxes = pred["boxes"][keep_idx]
            scores = pred["scores"][keep_idx]
            labels = pred["labels"][keep_idx]

            # Convert boxes from (x_min, y_min, x_max, y_max) to COCO format (x, y, w, h)
            coco_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box.tolist()
                coco_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

            # Store predictions
            for box, score, label in zip(coco_boxes, scores, labels, strict=True):
                self.val_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": label.item() + 1,  # Convert back to 1-based COCO IDs
                        "bbox": box,
                        "score": score.item(),
                    }
                )

        return {"loss": torch.tensor(0.0)}  # Dummy loss for compatibility

    def on_validation_epoch_end(self) -> None:
        """Compute and log COCO metrics at the end of validation epoch."""
        if len(self.val_predictions) == 0:
            return

        # This would require the COCO ground truth annotations for proper evaluation
        # For now, we'll just log the number of predictions
        self.log("val_num_predictions", len(self.val_predictions))

        # Clear predictions for next epoch
        self.val_predictions = []
        self.val_targets = []

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer (not used for inference-only)."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def visualize_predictions(
        self,
        image: torch.Tensor,
        predictions: dict[str, torch.Tensor],
        category_names: list[str],
        save_path: str | None = None,
        max_detections: int = 10,
    ) -> None:
        """Visualize object detection predictions on an image."""
        # Convert tensor to PIL Image
        if image.dim() == 4:
            image = image[0]  # Remove batch dimension

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)

        # Convert to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_np)

        # Draw predictions
        boxes = predictions["boxes"][:max_detections]
        scores = predictions["scores"][:max_detections]
        labels = predictions["labels"][:max_detections]

        for box, score, label in zip(boxes, scores, labels, strict=True):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            width = x_max - x_min
            height = y_max - y_min

            # Draw bounding box
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

            # Add label
            label_idx = label.item()
            category_name = category_names[label_idx] if label_idx < len(category_names) else f"Class {label_idx}"

            ax.text(
                x_min,
                y_min - 5,
                f"{category_name}: {score:.2f}",
                color="red",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            )

        ax.axis("off")
        ax.set_title(f"Object Detection Results ({len(boxes)} detections)")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def create_object_detector(
    model_name: str = "fasterrcnn_resnet50_fpn", confidence_threshold: float = 0.5
) -> PretrainedObjectDetector:
    """Factory function to create object detection model."""
    return PretrainedObjectDetector(model_name=model_name, confidence_threshold=confidence_threshold)
