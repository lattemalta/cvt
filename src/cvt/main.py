from logging import getLogger
from pathlib import Path

from typer import Typer

from cvt.data_modules.coco import LitCOCO2017
from cvt.modules.object_detection import LitObjectDetector

from .logger import setup_logger

logger = getLogger(__name__)


ROOT = Path(__file__).parents[2].resolve()


app = Typer()


@app.command(no_args_is_help=True)
def main(*, verbose: bool = False) -> None:
    setup_logger(verbose)

    coco_dm = LitCOCO2017(root_dir=ROOT / "data" / "coco2017", batch_size=4)
    coco_dm.setup("validate")

    model = LitObjectDetector()

    # Get first batch and visualize
    val_dataloader = coco_dm.val_dataloader()
    batch = next(iter(val_dataloader))
    _, targets = batch

    # Visualize RT-DETR predictions
    image_id = targets[2]["image_id"].item()
    model.visualize_image(image_id, coco_dm.val_dataset.coco, coco_dm.root_dir)
