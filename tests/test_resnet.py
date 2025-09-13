import torch

from cvt.models.resnet import BasicBlock, ConvNormLayer, Resnet18


def test_conv_norm_layer() -> None:
    layer = ConvNormLayer(ch_in=3, ch_out=16, kernel_size=3, stride=1)
    x = torch.randn(1, 3, 32, 32)
    y = layer(x)
    assert y.shape == (1, 16, 32, 32)


def test_basic_block() -> None:
    block = BasicBlock(ch_in=16, ch_out=32, stride=2)
    x = torch.randn(1, 16, 32, 32)
    y = block(x)
    assert y.shape == (1, 32, 16, 16)


def test_resnet18() -> None:
    model = Resnet18(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 10)
