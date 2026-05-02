"""Reusable PyTorch layers and building blocks for the VAE."""

from __future__ import annotations

from typing import Optional, Tuple

from .runtime import F, nn, torch


def get_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


def get_norm_layer(
    num_channels: int,
    norm_type: str = "group",
    num_groups: int = 32,
) -> nn.Module:
    if norm_type == "group":
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6)
    if norm_type == "layer":
        return nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-6)
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        norm_num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = get_norm_layer(in_channels, norm_type, norm_num_groups)
        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = get_norm_layer(out_channels, norm_type, norm_num_groups)
        self.act2 = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        norm_type: str = "group",
        norm_num_groups: int = 32,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = get_norm_layer(channels, norm_type, norm_num_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, height, width)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_num_heads: int,
        norm_type: str,
        norm_num_groups: int,
        activation: str,
        has_downsample: bool,
    ):
        super().__init__()
        blocks = []
        for index in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attn = SelfAttention2d(
            channels=out_channels,
            num_heads=min(attention_num_heads, out_channels // 8),
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
        )
        self.downsample = Downsample(out_channels) if has_downsample else None

    def forward(
        self,
        x: torch.Tensor,
        current_res: int,
        use_attention_at: set[int],
    ) -> Tuple[torch.Tensor, int]:
        h = x
        for block in self.blocks:
            h = block(h)

        if current_res in use_attention_at:
            h = self.attn(h)

        if self.downsample is not None:
            h = self.downsample(h)
            current_res //= 2

        return h, current_res


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        next_channels: Optional[int],
        num_res_blocks: int,
        attention_num_heads: int,
        norm_type: str,
        norm_num_groups: int,
        activation: str,
    ):
        super().__init__()
        blocks = []
        for index in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attn = SelfAttention2d(
            channels=out_channels,
            num_heads=min(attention_num_heads, out_channels // 8),
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
        )
        if next_channels is not None:
            self.upsample = nn.Sequential(
                Upsample(out_channels),
                nn.Conv2d(out_channels, next_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = None

    def forward(
        self,
        x: torch.Tensor,
        current_res: int,
        use_attention_at: set[int],
    ) -> Tuple[torch.Tensor, int]:
        h = x
        for block in self.blocks:
            h = block(h)

        if current_res in use_attention_at:
            h = self.attn(h)

        if self.upsample is not None:
            h = self.upsample(h)
            current_res *= 2

        return h, current_res