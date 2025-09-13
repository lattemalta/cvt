import logging
from logging import getLogger
from pathlib import Path

import lightning as L
from rich.logging import RichHandler

from cvt.data_modules.fashion_mnist import FashionMNISTDataModule
from cvt.modules.fashion_mnist import FashionMNISTModule

logging.basicConfig(level=logging.INFO, format="", handlers=[RichHandler()])
logger = getLogger(__name__)


def main() -> None:
    root_dir = Path(__file__).parents[1].resolve()

    fashion_mnist_data_module = FashionMNISTDataModule(
        root_dir=root_dir / "data",
        batch_size=32,
        num_workers=4,
    )
    fashion_mnist_module = FashionMNISTModule()

    trainer = L.Trainer(
        max_epochs=2,
    )

    trainer.fit(
        model=fashion_mnist_module,
        datamodule=fashion_mnist_data_module,
    )


if __name__ == "__main__":
    main()
