from pathlib import Path

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class LitFashionMNIST(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        datasets.FashionMNIST(self.root_dir, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.FashionMNIST(
                root=self.root_dir,
                train=True,
                download=True,
                transform=self.train_transform,
            )
        if stage in ("validate", "fit") or stage is None:
            self.val_dataset = datasets.FashionMNIST(
                root=self.root_dir,
                train=False,
                transform=self.val_transform,
                download=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
