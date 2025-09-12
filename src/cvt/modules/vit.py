import lightning as L
import torch
import torch.nn.functional as F

from cvt.models.vit import VisionTransformer


class LitVisionTransformer(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        patch_size: int = 16,
        num_inputlayer_units: int = 512,
        num_heads: int = 4,
        num_mlp_units: int = 512,
        num_layers: int = 6,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = VisionTransformer(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            num_inputlayer_units=num_inputlayer_units,
            num_heads=num_heads,
            num_mlp_units=num_mlp_units,
            num_layers=num_layers,
        )
        self.lr = lr

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self.model(x)
        loss = F.cross_entropy(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", (preds.argmax(dim=-1) == y).float().mean())
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> None:
        x, y = batch
        preds = self.model(x)
        loss = F.cross_entropy(preds, y)

        self.log("val_loss", loss)
        self.log("val_acc", (preds.argmax(dim=-1) == y).float().mean())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)
