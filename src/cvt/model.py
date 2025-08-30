import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_inputlayer_units: int, num_heads: int, dropout: float = 0.3) -> None:
        super().__init__()

        if num_inputlayer_units % num_heads != 0:
            raise ValueError(
                f"num_inputlayer_units must be divisible by num_heads. "
                f"num_inputlayer_units: {num_inputlayer_units}, num_heads: {num_heads}"
            )

        self.num_heads = num_heads
        dim_head = num_inputlayer_units // num_heads

        self.expansion_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units * 3, bias=False)
        self.scale = 1 / (dim_head**0.5)

        self.headjoin_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, ns = x.shape[:2]

        qkv = self.expansion_layer(x)
        qkv = qkv.view(bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attn = q.matmul(k.transpose(-2, -1))

        attn = (attn * self.scale).softmax(dim=-1)

        attn = self.dropout(attn)

        x = attn.matmul(v)

        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.headjoin_layer(x)

        return x


class MLP(nn.Module):
    def __init__(self, num_inputlayer_units: int, num_mlp_units: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.linear1 = nn.Linear(num_inputlayer_units, num_mlp_units)
        self.linear2 = nn.Linear(num_mlp_units, num_inputlayer_units)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_inputlayer_units: int, num_heads: int, num_mlp_units: int) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(num_inputlayer_units, num_heads)
        self.mlp = MLP(num_inputlayer_units, num_mlp_units)

        self.norm1 = nn.LayerNorm(num_inputlayer_units)
        self.norm2 = nn.LayerNorm(num_inputlayer_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int,
        patch_size: int,
        num_inputlayer_units: int,
        num_heads: int,
        num_mlp_units: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        input_dim = 3 * patch_size**2

        self.input_layer = nn.Linear(input_dim, num_inputlayer_units)

        self.class_token = nn.Parameter(torch.zeros(1, 1, num_inputlayer_units))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_inputlayer_units))

        self.encoder_layer = nn.ModuleList(
            [EncoderBlock(num_inputlayer_units, num_heads, num_mlp_units) for _ in range(num_layers)]
        )

        self.normalize = nn.LayerNorm(num_inputlayer_units)
        self.output_layer = nn.Linear(num_inputlayer_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = x.shape
        x = x.view(
            bs,
            c,
            h // self.patch_size,
            self.patch_size,
            w // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(bs, (h // self.patch_size) * (w // self.patch_size), -1)

        x = self.input_layer(x)
        class_token = self.class_token.expand(bs, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x += self.pos_embed

        for layer in self.encoder_layer:
            x = layer(x)

        x = x[:, 0]
        x = self.normalize(x)
        x = self.output_layer(x)

        return x
