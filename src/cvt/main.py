from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from rich import print
from rich.traceback import install
from torch.utils.data import DataLoader

from .config import ModelConfig
from .model import VisionTransformer

install(show_locals=True)
ROOT = Path(__file__).parents[2].resolve()


def evaluate(
    data_loader: DataLoader,
    model: nn.Module,
    loss_func: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    losses = []
    correct_preds = 0
    total_samples = 0

    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(device=device)
            y = y.to(device=device)

            preds = model(x)
            loss = loss_func(preds, y)
            losses.append(loss.item())

            _, predicted = torch.max(preds, 1)
            correct_preds += (predicted == y).sum().item()
            total_samples += y.size(0)

    average_loss = sum(losses) / len(losses)
    accuracy = correct_preds / total_samples

    return average_loss, accuracy


def main() -> None:
    config = ModelConfig()

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2335, 0.2616))

    train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10(
        root=ROOT / "data",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=ROOT / "data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = VisionTransformer(
        num_classes=len(train_dataset.classes),
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_inputlayer_units=config.num_inputlayer_units,
        num_heads=config.num_heads,
        num_mlp_units=config.num_mlp_units,
        num_layers=config.num_layers,
    )
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(config.num_epochs):
        model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            preds = model(x)
            loss = loss_func(preds, y)
            accuracy = (preds.argmax(dim=-1) == y).float().mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        val_loss, val_accuracy = evaluate(test_loader, model, loss_func, device)

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"  Training: loss = {avg_train_loss:.3f}, accuracy = {avg_train_accuracy:.3f}")
        print(f"  Validation: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config.num_epochs + 1), val_losses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, config.num_epochs + 1), val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
