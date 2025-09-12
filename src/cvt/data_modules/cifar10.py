from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path


class LitCIFAR10(L.LightningDataModule):
    def __init__(self, root_dir: str | Path = "data", batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2335, 0.2616))]
        )
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(self.root_dir, download=True)

    def setup(self, _stage: str) -> None:
        self.train_dataset = torchvision.datasets.CIFAR10(self.root_dir, train=True, transform=self.transform)
        self.valid_dataset = torchvision.datasets.CIFAR10(self.root_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
