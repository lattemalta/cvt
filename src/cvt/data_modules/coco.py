from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset


class COCODataset(Dataset):
    def __init__(
        self,
        root_dir: Path | str,
        annotation_file: Path | str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = [cat["id"] for cat in self.categories]
        self.cat_names = [cat["name"] for cat in self.categories]

        # Create mapping from category ID to index (0-based)
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.image_ids[idx]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.root_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Parse annotations
        boxes = []
        labels = []
        areas = []
        iscrowds = []

        for ann in anns:
            # Get bounding box in COCO format (x, y, width, height)
            x, y, w, h = ann["bbox"]
            # Convert to (x_min, y_min, x_max, y_max) format
            boxes.append([x, y, x + w, y + h])

            # Map category ID to 0-based index
            labels.append(self.cat_id_to_idx[ann["category_id"]])
            areas.append(ann["area"])
            iscrowds.append(ann["iscrowd"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowds,
        }

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return cast(torch.Tensor, image), target

    def get_category_names(self) -> list[str]:
        """Get list of category names."""
        return self.cat_names


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Custom collate function for object detection."""
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return torch.stack(images), targets


class LitCOCO2017(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Path | str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: int = 640,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Download data if needed (already downloaded in our case)."""
        pass

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = COCODataset(
                root_dir=self.root_dir / "train2017",
                annotation_file=self.root_dir / "annotations" / "instances_train2017.json",
                transform=self.transform,
            )

        if stage == "fit" or stage == "validate" or stage is None:
            self.val_dataset = COCODataset(
                root_dir=self.root_dir / "val2017",
                annotation_file=self.root_dir / "annotations" / "instances_val2017.json",
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )
