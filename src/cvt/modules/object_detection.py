import lightning as L
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np


class LitObjectDetector(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        return {"val_loss": torch.tensor(0.0)}

    def visualize_image(self, image_id, coco, root_dir, confidence_threshold=0.5):
        """Visualize RT-DETR predictions on original image."""
        self.model.eval()

        # Get original image path
        img_info = coco.loadImgs(image_id)[0]
        img_path = root_dir / "val2017" / img_info["file_name"]

        # Load original image
        from PIL import Image

        original_img = Image.open(img_path).convert("RGB")

        # Run RT-DETR inference
        inputs = self.processor(images=original_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([original_img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        # Visualize predictions
        plt.figure(figsize=(12, 8))
        plt.imshow(original_img)

        # Draw prediction bounding boxes
        import matplotlib.patches as patches

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1

            # Draw rectangle
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none")
            plt.gca().add_patch(rect)

            # Add label
            label_name = self.model.config.id2label[label.item()]
            plt.text(
                x1,
                y1 - 10,
                f"{label_name}: {score:.2f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        plt.title(f"RT-DETR Predictions (Image ID: {image_id})")
        plt.axis("off")
        plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
