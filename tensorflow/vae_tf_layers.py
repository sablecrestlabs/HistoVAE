"""Reusable Keras layers and building blocks for the TensorFlow VAE."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vae_tf_runtime import tf


class TorchConvKernelInitializer(tf.keras.initializers.Initializer):
    """Match PyTorch Conv2d default kaiming_uniform_(a=sqrt(5))."""

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
        dtype = dtype or tf.float32
        fan_in = np.prod(shape[:-1])
        bound = 1.0 / math.sqrt(float(fan_in))
        return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        return {}


class TorchConvBiasInitializer(tf.keras.initializers.Initializer):
    """Match PyTorch Conv2d default bias init derived from fan_in."""

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
        dtype = dtype or tf.float32
        fan_in = self.kernel_size * self.kernel_size * int(shape[0])
        bound = 1.0 / math.sqrt(float(fan_in))
        return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        return {"kernel_size": self.kernel_size}


def make_conv2d(
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
) -> tf.keras.layers.Conv2D:
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_initializer=TorchConvKernelInitializer(),
        bias_initializer=TorchConvBiasInitializer(kernel_size),
    )


class GroupNorm(tf.keras.layers.Layer):
    """Simple GroupNorm implementation for NHWC tensors."""

    def __init__(self, groups: int = 32, eps: float = 1e-6, **kwargs: Any):
        super().__init__(**kwargs)
        self.groups = groups
        self.eps = eps

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = int(input_shape[-1])
        groups = min(self.groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.groups = max(groups, 1)
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        grouped = tf.reshape(
            inputs,
            [batch_size, height, width, self.groups, channels // self.groups],
        )
        mean, variance = tf.nn.moments(grouped, axes=[1, 2, 4], keepdims=True)
        normalized = (grouped - mean) / tf.sqrt(variance + self.eps)
        normalized = tf.reshape(normalized, input_shape)
        return normalized * self.gamma + self.beta


def get_activation(name: str) -> tf.keras.layers.Layer:
    if name == "silu":
        return tf.keras.layers.Activation("swish")
    if name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.2)
    if name == "relu":
        return tf.keras.layers.ReLU()
    raise ValueError(f"Unknown activation: {name}")


def get_norm_layer(
    num_channels: int,
    norm_type: str = "group",
    num_groups: int = 32,
) -> tf.keras.layers.Layer:
    if norm_type == "group":
        return GroupNorm(groups=num_groups)
    if norm_type == "layer":
        return GroupNorm(groups=1)
    if norm_type == "batch":
        return tf.keras.layers.BatchNormalization(epsilon=1e-6)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class ResidualBlock(tf.keras.layers.Layer):
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
        self.out_channels = out_channels
        self.norm1 = get_norm_layer(in_channels, norm_type, norm_num_groups)
        self.act1 = get_activation(activation)
        self.conv1 = make_conv2d(out_channels, 3, padding="same")

        self.norm2 = get_norm_layer(out_channels, norm_type, norm_num_groups)
        self.act2 = get_activation(activation)
        self.dropout = (
            tf.keras.layers.Dropout(dropout)
            if dropout > 0
            else tf.keras.layers.Identity()
        )
        self.conv2 = make_conv2d(out_channels, 3, padding="same")

        if in_channels != out_channels:
            self.skip = make_conv2d(out_channels, 1, padding="same")
        else:
            self.skip = tf.keras.layers.Identity()

    def build(self, input_shape: tf.TensorShape) -> None:
        current_shape = tf.TensorShape(input_shape)
        self.norm1.build(current_shape)
        self.conv1.build(current_shape)

        current_shape = current_shape[:-1].concatenate(self.out_channels)
        self.norm2.build(current_shape)
        self.dropout.build(current_shape)
        self.conv2.build(current_shape)
        self.skip.build(input_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.norm1(x, training=training)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h, training=training)
        h = self.act2(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(tf.keras.layers.Layer):
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
        self.scale = self.head_dim**-0.5
        self.norm = get_norm_layer(channels, norm_type, norm_num_groups)
        self.qkv = make_conv2d(channels * 3, 1, padding="same")
        self.proj = make_conv2d(channels, 1, padding="same")

    def build(self, input_shape: tf.TensorShape) -> None:
        current_shape = tf.TensorShape(input_shape)
        self.norm.build(current_shape)
        self.qkv.build(current_shape)
        self.proj.build(current_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        h = self.norm(x, training=training)
        qkv = self.qkv(h)
        qkv = tf.reshape(
            qkv,
            [batch_size, height * width, 3, self.num_heads, self.head_dim],
        )
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 4])
        out = tf.reshape(out, [batch_size, height, width, self.channels])
        out = self.proj(out)
        return x + out


class Downsample(tf.keras.layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = make_conv2d(channels, 3, strides=2, padding="same")

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.conv(x)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = make_conv2d(channels, 3, padding="same")

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = tf.TensorShape(input_shape)
        upsampled_shape = input_shape[0:1].concatenate(
            [
                None if input_shape[1] is None else input_shape[1] * 2,
                None if input_shape[2] is None else input_shape[2] * 2,
                input_shape[3],
            ]
        )
        self.conv.build(upsampled_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        spatial_size = tf.shape(x)[1:3] * 2
        x = tf.image.resize(x, spatial_size, method="nearest")
        return self.conv(x)


class EncoderStage(tf.keras.layers.Layer):
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
        self.out_channels = out_channels
        self.blocks = []
        for index in range(num_res_blocks):
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                )
            )
        num_heads = max(1, min(attention_num_heads, out_channels // 8))
        self.attn = SelfAttention2d(
            channels=out_channels,
            num_heads=num_heads,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
        )
        self.downsample = Downsample(out_channels) if has_downsample else None

    def build(self, input_shape: tf.TensorShape) -> None:
        current_shape = tf.TensorShape(input_shape)
        for block in self.blocks:
            block.build(current_shape)
            current_shape = current_shape[:-1].concatenate(self.out_channels)

        self.attn.build(current_shape)

        if self.downsample is not None:
            self.downsample.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        current_res: int,
        use_attention_at: set,
        training: bool = False,
    ) -> Tuple[tf.Tensor, int]:
        h = x
        for block in self.blocks:
            h = block(h, training=training)

        if current_res in use_attention_at:
            h = self.attn(h, training=training)

        if self.downsample is not None:
            h = self.downsample(h)
            current_res //= 2

        return h, current_res


class DecoderStage(tf.keras.layers.Layer):
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
        self.out_channels = out_channels
        self.next_channels = next_channels
        self.blocks = []
        for index in range(num_res_blocks):
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                )
            )
        num_heads = max(1, min(attention_num_heads, out_channels // 8))
        self.attn = SelfAttention2d(
            channels=out_channels,
            num_heads=num_heads,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
        )
        if next_channels is not None:
            self.upsample = tf.keras.Sequential(
                [
                    Upsample(out_channels),
                    make_conv2d(next_channels, 3, padding="same"),
                ]
            )
        else:
            self.upsample = None

    def build(self, input_shape: tf.TensorShape) -> None:
        current_shape = tf.TensorShape(input_shape)
        for block in self.blocks:
            block.build(current_shape)
            current_shape = current_shape[:-1].concatenate(self.out_channels)

        self.attn.build(current_shape)

        if self.upsample is not None:
            self.upsample.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        current_res: int,
        use_attention_at: set,
        training: bool = False,
    ) -> Tuple[tf.Tensor, int]:
        h = x
        for block in self.blocks:
            h = block(h, training=training)

        if current_res in use_attention_at:
            h = self.attn(h, training=training)

        if self.upsample is not None:
            h = self.upsample(h, training=training)
            current_res *= 2

        return h, current_res