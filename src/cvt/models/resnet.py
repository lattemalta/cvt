import torch
import torch.nn as nn


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
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
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
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cn1(x)
        out = self.cn2(out)
        out += self.short(x)
        out = self.act(out)
        return out
