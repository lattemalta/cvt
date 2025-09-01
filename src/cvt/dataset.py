from logging import getLogger

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from .logger import setup_logger

logger = getLogger(__name__)


class SyntheticTextDataset(data.Dataset):
    """Simple synthetic text dataset for testing DBNet++"""

    def __init__(
        self, size: int = 100, img_size: tuple[int, int] = (640, 640), transform: transforms.Compose | None = None
    ):
        self.size = size
        self.img_size = img_size
        self.transform = transform

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )

    def __len__(self) -> int:
        return self.size

    def _generate_text_image(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic text image and mask"""
        # Create blank image
        img = np.ones((*self.img_size, 3), dtype=np.uint8) * 255
        mask = np.zeros(self.img_size, dtype=np.uint8)

        # No seed - completely random each time

        # Generate 1-3 text regions
        num_texts = np.random.randint(1, 4)

        for _ in range(num_texts):
            # Random text properties
            text = f"TEXT{np.random.randint(0, 999)}"
            font_scale = np.random.uniform(0.8, 2.0)
            thickness = np.random.randint(2, 4)

            # Random position
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = np.random.randint(0, max(1, self.img_size[1] - text_size[0]))
            y = np.random.randint(text_size[1], self.img_size[0])

            # Random color
            color = tuple(np.random.randint(0, 256, 3).tolist())

            # Draw text on image and mask
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(mask, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness)

        return img, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate synthetic image and mask
        img, mask = self._generate_text_image(idx)

        # Convert to PIL for transforms
        img_pil = Image.fromarray(img)

        # Apply transforms to image
        img_tensor = self.transform(img_pil)

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)

        # Create training mask (all pixels are valid for synthetic data)
        training_mask = torch.ones_like(mask_tensor)

        return {
            "image": img_tensor,
            "gt_text": mask_tensor,
            "training_mask": training_mask,
            "image_path": f"synthetic_{idx}.jpg",
        }


def create_dataloaders(
    batch_size: int = 8, num_workers: int = 4, img_size: tuple[int, int] = (640, 640)
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders for synthetic data"""

    # Common transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Create synthetic datasets
    train_dataset = SyntheticTextDataset(size=800, img_size=img_size, transform=transform)
    val_dataset = SyntheticTextDataset(size=200, img_size=img_size, transform=transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    setup_logger()

    # Test synthetic dataset
    logger.info("Testing synthetic dataset...")
    train_loader, val_loader = create_dataloaders(batch_size=2)

    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Batch {batch_idx}:")
        logger.info(f"  Image shape: {batch['image'].shape}")
        logger.info(f"  GT text shape: {batch['gt_text'].shape}")
        logger.info(f"  Training mask shape: {batch['training_mask'].shape}")

        if batch_idx >= 2:  # Only show first few batches
            break

    logger.info(f"Total train batches: {len(train_loader)}")
    logger.info(f"Total val batches: {len(val_loader)}")
