import logging
from logging import getLogger
from pathlib import Path
import matplotlib.pyplot as plt
from rich.logging import RichHandler

from cvt.data_modules.fashion_mnist import LitFashionMNIST

logging.basicConfig(level=logging.INFO, format="", handlers=[RichHandler()])
logger = getLogger(__name__)


def main() -> None:
    root_dir = Path(__file__).parents[1]
    dm = LitFashionMNIST(
        root_dir=root_dir / "data",
        batch_size=32,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup("fit")

    train_dataset = dm.train_dataset
    class_names = train_dataset.classes
    logger.info(f"Class names: {class_names}")

    logger.info(f"Train dataset targets: {train_dataset.targets.shape}")

    class_counts = dict.fromkeys(class_names, 0)
    for label in train_dataset.targets:
        class_name = class_names[label]
        class_counts[class_name] += 1

    logger.info(f"Class counts: {class_counts}")

    plt.figure(figsize=(12, 20), dpi=100)
    for i in range(32):
        image, label = train_dataset[i]
        plt.subplot(8, 4, i + 1)
        plt.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis("off")
    plt.tight_layout()

    output_dir = root_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "fashion_mnist_samples.png")


if __name__ == "__main__":
    main()
