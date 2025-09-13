import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int) -> None:
        super().__init__()

        self.cn1 = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=3,
            stride=stride,
        )
        self.cn2 = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            kernel_size=3,
            stride=1,
        )
        self.short = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=1,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cn1(x)
        out = self.cn2(out)
        out = out + self.short(x)
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self, ch_in: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvNormLayer(ch_in=ch_in, ch_out=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = self._make_layer(64, 64, 2, stride=1)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.layer5 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, ch_in: int, ch_out: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(BasicBlock(ch_in=ch_in, ch_out=ch_out, stride=stride))
        layers.extend(BasicBlock(ch_in=ch_out, ch_out=ch_out, stride=1) for _ in range(1, n_blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
