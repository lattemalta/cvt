import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from cvt.models.resnet import Resnet18


class LitFashionMNIST(L.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = Resnet18(ch_in=1, num_classes=10)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.accuracy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> None:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.accuracy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
