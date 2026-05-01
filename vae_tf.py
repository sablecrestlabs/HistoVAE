#!/usr/bin/env python3
"""
TensorFlow Variational Autoencoder (VAE) port of vae.py.

Mirrors the PyTorch implementation's architecture, random tile dataset,
regularization, KL scheduling, and training loop as closely as practical in
TensorFlow/Keras while remaining self-contained in a separate file.
"""

import argparse
import glob
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    TF_AVAILABLE = True
except ImportError as exc:
    TF_AVAILABLE = False
    tf = None
    mixed_precision = None
    TF_IMPORT_ERROR = exc

try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    openslide = None

try:
    from PIL import Image, ImageEnhance

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageEnhance = None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VAEConfig:
    """Configuration dataclass for the TensorFlow VAE model and training."""

    img_channels: int = 3
    img_size: int = 256

    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    latent_channels: int = 16
    downsample_factor: int = 16
    num_res_blocks_per_stage: int = 1
    use_attention_at: Tuple[int, ...] = ()
    attention_num_heads: int = 4

    norm_type: str = "group"
    norm_num_groups: int = 8
    activation: str = "silu"

    beta: float = 1.0
    kl_warmup_steps: int = 2000
    recon_loss_type: str = "l1"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    max_grad_norm: Optional[float] = 1.0
    use_amp: bool = True

    @property
    def latent_size(self) -> int:
        return self.img_size // self.downsample_factor

    def validate(self) -> None:
        expected_downsample = 2 ** len(self.channel_multipliers)
        if self.downsample_factor != expected_downsample:
            warnings.warn(
                f"downsample_factor ({self.downsample_factor}) should typically equal "
                f"2^len(channel_multipliers) = {expected_downsample}"
            )

        if self.img_size % self.downsample_factor != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"downsample_factor ({self.downsample_factor})"
            )


# =============================================================================
# Building Blocks
# =============================================================================


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
            tf.keras.layers.Dropout(
                dropout) if dropout > 0 else tf.keras.layers.Identity()
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


class Encoder(tf.keras.Model):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        use_attention_at: Tuple[int, ...] = (64,),
        attention_num_heads: int = 4,
        norm_type: str = "group",
        norm_num_groups: int = 32,
        activation: str = "silu",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.use_attention_at = set(use_attention_at)
        self.conv_in = make_conv2d(base_channels, 3, padding="same")

        self.stages = []
        in_ch = base_channels
        for index, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.stages.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    attention_num_heads=attention_num_heads,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                    has_downsample=index < len(channel_multipliers) - 1,
                )
            )
            in_ch = out_ch

        final_ch = base_channels * channel_multipliers[-1]
        self.norm_out = get_norm_layer(final_ch, norm_type, norm_num_groups)
        self.act_out = get_activation(activation)
        self.conv_out = make_conv2d(2 * latent_channels, 3, padding="same")

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        h = self.conv_in(x)
        current_res = int(x.shape[1])

        for stage in self.stages:
            h, current_res = stage(
                h,
                current_res=current_res,
                use_attention_at=self.use_attention_at,
                training=training,
            )

        h = self.norm_out(h, training=training)
        h = self.act_out(h)
        h = self.conv_out(h)
        mu, logvar = tf.split(h, num_or_size_splits=2, axis=-1)
        return mu, logvar


class Decoder(tf.keras.Model):
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        use_attention_at: Tuple[int, ...] = (64,),
        attention_num_heads: int = 4,
        norm_type: str = "group",
        norm_num_groups: int = 32,
        activation: str = "silu",
    ):
        super().__init__()
        self.use_attention_at = set(use_attention_at)
        channel_multipliers_rev = tuple(reversed(channel_multipliers))
        first_ch = base_channels * channel_multipliers_rev[0]
        self.conv_in = make_conv2d(first_ch, 3, padding="same")

        self.stages = []
        in_ch = first_ch
        for index, mult in enumerate(channel_multipliers_rev):
            out_ch = base_channels * mult
            next_channels = None
            if index < len(channel_multipliers_rev) - 1:
                next_channels = base_channels * \
                    channel_multipliers_rev[index + 1]
            self.stages.append(
                DecoderStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    next_channels=next_channels,
                    num_res_blocks=num_res_blocks,
                    attention_num_heads=attention_num_heads,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                    activation=activation,
                )
            )
            in_ch = next_channels if next_channels is not None else out_ch

        final_ch = base_channels * channel_multipliers_rev[-1]
        self.norm_out = get_norm_layer(final_ch, norm_type, norm_num_groups)
        self.act_out = get_activation(activation)
        self.conv_out = make_conv2d(out_channels, 3, padding="same")

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.conv_in(z)
        current_res = int(z.shape[1])

        for stage in self.stages:
            h, current_res = stage(
                h,
                current_res=current_res,
                use_attention_at=self.use_attention_at,
                training=training,
            )

        h = self.norm_out(h, training=training)
        h = self.act_out(h)
        x_recon = self.conv_out(h)
        return tf.tanh(x_recon)


class VAE(tf.keras.Model):
    def __init__(
        self,
        config: Optional[VAEConfig] = None,
        in_channels: int = 3,
        img_size: int = 512,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        use_attention_at: Tuple[int, ...] = (64,),
        attention_num_heads: int = 4,
        norm_type: str = "group",
        norm_num_groups: int = 32,
        activation: str = "silu",
        recon_loss_type: str = "l1",
    ):
        super().__init__()

        if config is not None:
            in_channels = config.img_channels
            img_size = config.img_size
            base_channels = config.base_channels
            channel_multipliers = config.channel_multipliers
            latent_channels = config.latent_channels
            num_res_blocks = config.num_res_blocks_per_stage
            use_attention_at = config.use_attention_at
            attention_num_heads = config.attention_num_heads
            norm_type = config.norm_type
            norm_num_groups = config.norm_num_groups
            activation = config.activation
            recon_loss_type = config.recon_loss_type

        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_channels = latent_channels
        self.recon_loss_type = recon_loss_type

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            use_attention_at=use_attention_at,
            attention_num_heads=attention_num_heads,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
            activation=activation,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            use_attention_at=use_attention_at,
            attention_num_heads=attention_num_heads,
            norm_type=norm_type,
            norm_num_groups=norm_num_groups,
            activation=activation,
        )

    def encode(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.encoder(x, training=training)

    def decode(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.decoder(z, training=training)

    def reparameterize(
        self,
        mu: tf.Tensor,
        logvar: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        logvar = tf.clip_by_value(logvar, -10.0, 5.0)
        mu = tf.clip_by_value(mu, -10.0, 10.0)
        if training:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std), dtype=std.dtype)
            return mu + eps * std
        return mu

    def call(
        self,
        x: tf.Tensor,
        kl_weight: float = 1.0,
        return_latent: bool = False,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        x = tf.clip_by_value(x, -1.0, 1.0)
        mu, logvar = self.encode(x, training=training)
        z = self.reparameterize(mu, logvar, training=training)
        z = tf.clip_by_value(z, -10.0, 10.0)
        x_recon = self.decode(z, training=training)
        x_recon = tf.clip_by_value(x_recon, -1.0, 1.0)

        recon_loss = reconstruction_loss(
            x, x_recon, loss_type=self.recon_loss_type)
        kl_loss = kl_divergence(mu, logvar)
        recon_loss = tf.clip_by_value(recon_loss, 0.0, 100.0)
        kl_loss = tf.clip_by_value(kl_loss, 0.0, 1000.0)
        loss = recon_loss + tf.cast(kl_weight, recon_loss.dtype) * kl_loss

        output = {
            "x_recon": x_recon,
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
        if return_latent:
            output["mu"] = mu
            output["logvar"] = logvar
            output["z"] = z
        return output

    def sample(self, num_samples: int, latent_size: Optional[int] = None) -> tf.Tensor:
        if latent_size is None:
            latent_size = self.img_size // (2 ** len(self.encoder.stages))

        z = tf.random.normal(
            [num_samples, latent_size, latent_size, self.latent_channels]
        )
        return self.decode(z, training=False)

    def reconstruct(self, x: tf.Tensor, deterministic: bool = False) -> tf.Tensor:
        mu, logvar = self.encode(x, training=False)
        z = self.reparameterize(mu, logvar, training=not deterministic)
        return self.decode(z, training=False)


# =============================================================================
# Losses and Scheduling
# =============================================================================


def kl_divergence(mu: tf.Tensor, logvar: tf.Tensor, free_nats: float = 0.5) -> tf.Tensor:
    mu = tf.cast(mu, tf.float32)
    logvar = tf.cast(logvar, tf.float32)
    logvar = tf.clip_by_value(logvar, -10.0, 5.0)
    mu = tf.clip_by_value(mu, -10.0, 10.0)
    kl = -0.5 * (1.0 + logvar - tf.square(mu) - tf.exp(logvar))
    kl = tf.reduce_sum(kl, axis=[1, 2, 3])
    free_nats_tensor = tf.cast(free_nats, kl.dtype)
    kl = tf.maximum(kl, free_nats_tensor)
    kl = tf.where(
        tf.math.is_finite(kl),
        kl,
        tf.fill(tf.shape(kl), free_nats_tensor),
    )
    return tf.reduce_mean(kl)


def reconstruction_loss(
    x: tf.Tensor,
    x_recon: tf.Tensor,
    loss_type: str = "l1",
) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)
    if loss_type == "l1":
        return tf.reduce_mean(tf.abs(x_recon - x))
    if loss_type == "l2":
        return tf.reduce_mean(tf.square(x_recon - x))
    raise ValueError(f"Unknown loss_type: {loss_type}")


class LinearKLScheduler:
    def __init__(self, beta: float, warmup_steps: int):
        self.beta = beta
        self.warmup_steps = warmup_steps

    def __call__(self, global_step: int) -> float:
        if self.warmup_steps <= 0:
            return self.beta
        progress = min(1.0, global_step / self.warmup_steps)
        return self.beta * progress

    def state_dict(self) -> Dict[str, Any]:
        return {"beta": self.beta, "warmup_steps": self.warmup_steps}


class CyclicKLScheduler:
    def __init__(self, beta: float, cycle_steps: int, ratio: float = 0.5):
        self.beta = beta
        self.cycle_steps = cycle_steps
        self.ratio = ratio
        self.warmup_steps = int(cycle_steps * ratio)

    def __call__(self, global_step: int) -> float:
        step_in_cycle = global_step % self.cycle_steps
        if step_in_cycle < self.warmup_steps:
            return self.beta * step_in_cycle / self.warmup_steps
        return self.beta

    def state_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "cycle_steps": self.cycle_steps,
            "ratio": self.ratio,
        }


# =============================================================================
# Training Utilities
# =============================================================================


def check_for_nan(loss: tf.Tensor, name: str = "loss") -> bool:
    if bool(tf.reduce_any(~tf.math.is_finite(loss)).numpy()):
        print(f"WARNING: {name} contains NaN or Inf!")
        return True
    return False


def has_nonfinite_gradients(gradients: Any, variables: Any) -> bool:
    for grad, variable in zip(gradients, variables):
        if grad is None:
            continue
        if bool(tf.reduce_any(~tf.math.is_finite(grad)).numpy()):
            grad_min = float(tf.reduce_min(grad).numpy())
            grad_max = float(tf.reduce_max(grad).numpy())
            grad_mean = float(tf.reduce_mean(grad).numpy())
            print(
                f"Non-finite gradients detected in '{variable.name}': "
                f"min={grad_min:.3e}, max={grad_max:.3e}, mean={grad_mean:.3e}"
            )
            return True
    return False


def log_images(
    writer: tf.summary.SummaryWriter,
    x: tf.Tensor,
    x_recon: tf.Tensor,
    step: int,
    prefix: str = "train",
    num_images: int = 4,
) -> None:
    x = tf.cast(tf.clip_by_value(
        (x[:num_images] + 1.0) / 2.0, 0.0, 1.0), tf.float32)
    x_recon = tf.cast(
        tf.clip_by_value((x_recon[:num_images] + 1.0) / 2.0, 0.0, 1.0),
        tf.float32,
    )
    combined = tf.concat([x, x_recon], axis=2)
    with writer.as_default():
        tf.summary.image(f"{prefix}/orig_vs_recon", combined,
                         step=step, max_outputs=num_images)


@tf.function(reduce_retracing=True)
def run_train_step(
    model: VAE,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Any]:
    with tf.GradientTape() as tape:
        outputs = model(x, kl_weight=kl_weight,
                        return_latent=True, training=True)
        loss = tf.cast(outputs["loss"], tf.float32)
        gradient_loss = optimizer.scale_loss(loss) if hasattr(optimizer, "scale_loss") else loss

    gradients = tape.gradient(gradient_loss, model.trainable_variables)
    return outputs, loss, gradients


@tf.function(reduce_retracing=True)
def run_eval_step(
    model: VAE,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    return model(x, kl_weight=kl_weight, return_latent=False, training=False)


def train_epoch(
    model: VAE,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    epoch: int,
    kl_scheduler: Any,
    global_step: int,
    max_grad_norm: Optional[float] = None,
    writer: Optional[tf.summary.SummaryWriter] = None,
    log_interval: int = 100,
    image_log_interval: int = 1000,
) -> Tuple[Dict[str, float], int]:
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, x in enumerate(dataset):
        kl_weight = kl_scheduler(global_step)

        if bool(tf.reduce_any(~tf.math.is_finite(x)).numpy()):
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input data")
            continue

        outputs, loss, gradients = run_train_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
            optimizer,
        )

        if check_for_nan(loss, "loss"):
            print(f"Skipping batch {batch_idx} due to NaN loss")
            continue

        grad_var_pairs = [
            (grad, variable)
            for grad, variable in zip(gradients, model.trainable_variables)
            if grad is not None
        ]
        if not grad_var_pairs:
            print(f"Skipping batch {batch_idx} due to missing gradients")
            continue

        gradients, variables = zip(*grad_var_pairs)
        gradients = list(gradients)
        variables = list(variables)

        if has_nonfinite_gradients(gradients, variables):
            print(f"Skipping batch {batch_idx} due to non-finite gradients")
            continue

        optimizer.apply_gradients(zip(gradients, variables))

        loss_value = float(loss.numpy())
        recon_value = float(tf.cast(outputs["recon_loss"], tf.float32).numpy())
        kl_value = float(tf.cast(outputs["kl_loss"], tf.float32).numpy())

        total_loss += loss_value
        total_recon_loss += recon_value
        total_kl_loss += kl_value
        num_batches += 1

        if writer is not None and global_step % log_interval == 0:
            with writer.as_default():
                tf.summary.scalar("train/loss", loss_value, step=global_step)
                tf.summary.scalar("train/recon_loss",
                                  recon_value, step=global_step)
                tf.summary.scalar("train/kl_loss", kl_value, step=global_step)
                tf.summary.scalar("train/kl_weight",
                                  kl_weight, step=global_step)
                tf.summary.histogram(
                    "train/mu", outputs["mu"], step=global_step)
                tf.summary.histogram(
                    "train/logvar", outputs["logvar"], step=global_step)

        if writer is not None and global_step % image_log_interval == 0:
            log_images(writer, x, outputs["x_recon"],
                       global_step, prefix="train")

        global_step += 1

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }
    return metrics, global_step


def evaluate(
    model: VAE,
    dataset: tf.data.Dataset,
    kl_weight: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, x in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break

        outputs = run_eval_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
        )
        total_loss += float(tf.cast(outputs["loss"], tf.float32).numpy())
        total_recon_loss += float(
            tf.cast(outputs["recon_loss"], tf.float32).numpy())
        total_kl_loss += float(tf.cast(outputs["kl_loss"], tf.float32).numpy())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }


# =============================================================================
# OpenSlide Tile Dataset
# =============================================================================


def has_content(
    img: Image.Image,
    min_std: float = 5.0,
    min_mean: float = 10.0,
    max_mean: float = 245.0,
) -> bool:
    arr = np.array(img, dtype=np.float32)
    if arr.std() < min_std:
        return False
    if arr.mean() < min_mean:
        return False
    if arr.mean() > max_mean:
        return False
    return True


class OpenSlideTileDataset:
    """Random tile dataset backed by OpenSlide for tf.data generators."""

    def __init__(
        self,
        data_root: str,
        tile_size: int = 256,
        tiles_per_epoch: int = 10000,
        level: int = 0,
        color_jitter: bool = False,
        color_jitter_strength: float = 0.05,
        seed: Optional[int] = None,
    ):
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python is required. Install with: pip install openslide-python"
            )
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required. Install with: pip install Pillow")

        self.data_root = data_root
        self.tile_size = tile_size
        self.tiles_per_epoch = tiles_per_epoch
        self.level = level
        self.color_jitter = color_jitter
        self.color_jitter_strength = color_jitter_strength
        self.seed = seed
        self._rng = random.Random(seed)

        self.tif_files = glob.glob(os.path.join(data_root, "*.tif"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.TIF"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.svs"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.SVS"))
        self.tif_files += glob.glob(os.path.join(data_root,
                                    "**", "*.tif"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root,
                                    "**", "*.TIF"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root,
                                    "**", "*.svs"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root,
                                    "**", "*.SVS"), recursive=True)
        self.tif_files = list(set(self.tif_files))

        if not self.tif_files:
            raise ValueError(f"No .tif/.svs files found in {data_root}")

        print(f"Found {len(self.tif_files)} TIF/SVS files in {data_root}")
        self._slide_cache: Dict[str, Any] = {}
        self._slide_dimensions: Dict[str, Tuple[int, int]] = {}
        self._invalid_slides = set()

    def clone(self, seed_offset: int = 0) -> "OpenSlideTileDataset":
        clone_seed = None if self.seed is None else self.seed + seed_offset + 1
        return OpenSlideTileDataset(
            data_root=self.data_root,
            tile_size=self.tile_size,
            tiles_per_epoch=self.tiles_per_epoch,
            level=self.level,
            color_jitter=self.color_jitter,
            color_jitter_strength=self.color_jitter_strength,
            seed=clone_seed,
        )

    def _get_slide_with_dims(
        self, tif_path: str
    ) -> Optional[Tuple[Any, Tuple[int, int]]]:
        if tif_path in self._invalid_slides:
            return None

        if tif_path not in self._slide_cache:
            try:
                slide = openslide.OpenSlide(tif_path)
                level = min(self.level, slide.level_count - 1)
                dims = slide.level_dimensions[level]

                if dims[0] < self.tile_size or dims[1] < self.tile_size:
                    slide.close()
                    self._invalid_slides.add(tif_path)
                    return None

                self._slide_cache[tif_path] = slide
                self._slide_dimensions[tif_path] = dims
            except Exception as exc:
                print(f"Warning: Could not open {tif_path}: {exc}")
                self._invalid_slides.add(tif_path)
                return None

        return self._slide_cache[tif_path], self._slide_dimensions[tif_path]

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def _extract_random_tile(
        self, max_attempts: int = 50
    ) -> Tuple[Optional[Image.Image], str]:
        last_attempt_info = "no attempts made"
        empty_tile_count = 0
        open_error_count = 0

        for _ in range(max_attempts):
            tif_path = self._rng.choice(self.tif_files)
            result = self._get_slide_with_dims(tif_path)
            if result is None:
                open_error_count += 1
                last_attempt_info = f"failed to open {os.path.basename(tif_path)}"
                continue

            slide, dims = result
            max_x = dims[0] - self.tile_size
            max_y = dims[1] - self.tile_size

            if max_x <= 0 or max_y <= 0:
                last_attempt_info = (
                    f"{os.path.basename(tif_path)} too small ({dims[0]}x{dims[1]})"
                )
                continue

            x_coord = self._rng.randint(0, max_x)
            y_coord = self._rng.randint(0, max_y)

            try:
                level = min(self.level, slide.level_count - 1)
                downsample = slide.level_downsamples[level]
                level0_x = int(x_coord * downsample)
                level0_y = int(y_coord * downsample)
                img = slide.read_region(
                    (level0_x, level0_y), level, (self.tile_size, self.tile_size)
                )

                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert("RGB")

                arr = np.array(img)
                near_black_mask = (
                    (arr[:, :, 0] < 4) & (
                        arr[:, :, 1] < 4) & (arr[:, :, 2] < 4)
                )
                arr[near_black_mask] = [255, 255, 255]
                img = Image.fromarray(arr)

                if has_content(img):
                    return img, "success"

                empty_tile_count += 1
                last_attempt_info = (
                    f"{os.path.basename(tif_path)} at ({x_coord},{y_coord}) was empty "
                    "(black/white/uniform)"
                )
            except Exception as exc:
                last_attempt_info = (
                    f"error reading {os.path.basename(tif_path)} at "
                    f"({x_coord},{y_coord}): {exc}"
                )

        debug_info = (
            f"Failed after {max_attempts} attempts. Empty tiles: {empty_tile_count}, "
            f"open errors: {open_error_count}. Last: {last_attempt_info}"
        )
        return None, debug_info

    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        if not self.color_jitter:
            return img

        strength = self.color_jitter_strength
        brightness = self._rng.uniform(1.0 - strength, 1.0 + strength)
        contrast = self._rng.uniform(1.0 - strength, 1.0 + strength)
        saturation = self._rng.uniform(1.0 - strength, 1.0 + strength)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        try:
            flip_h = Image.Transpose.FLIP_LEFT_RIGHT
            flip_v = Image.Transpose.FLIP_TOP_BOTTOM
        except AttributeError:
            flip_h = Image.FLIP_LEFT_RIGHT
            flip_v = Image.FLIP_TOP_BOTTOM

        if self._rng.random() > 0.5:
            img = img.transpose(flip_h)
        if self._rng.random() > 0.5:
            img = img.transpose(flip_v)

        rotations = self._rng.randint(0, 3)
        if rotations > 0:
            img = img.rotate(rotations * 90, expand=False)

        return self._apply_color_jitter(img)

    def __getitem__(self, idx: int) -> np.ndarray:
        total_attempts = 0
        max_total_attempts = 500
        img = None
        debug_info = ""

        while total_attempts < max_total_attempts:
            img, debug_info = self._extract_random_tile()
            total_attempts += 50
            if img is not None:
                break
            if total_attempts % 100 == 0:
                print(
                    f"Warning: Struggled to find valid tile after {total_attempts} "
                    f"attempts. {debug_info}"
                )

        if img is None:
            raise RuntimeError(
                f"Could not find a valid tile after {max_total_attempts} attempts. "
                f"Last error: {debug_info}. Check that your TIF files contain "
                "non-empty regions."
            )

        img = self._apply_augmentations(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return arr

    def generator(self) -> Generator[np.ndarray, None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __del__(self) -> None:
        for slide in self._slide_cache.values():
            try:
                slide.close()
            except Exception:
                pass


def create_dataset(
    dataset: OpenSlideTileDataset,
    batch_size: int,
    img_size: int,
    img_channels: int,
    shuffle: bool,
    drop_remainder: bool,
    num_workers: int,
) -> tf.data.Dataset:
    output_signature = tf.TensorSpec(
        shape=(img_size, img_size, img_channels), dtype=tf.float32
    )
    host_cpu_count = max(1, os.cpu_count() or 1)
    # TensorFlow's Python-backed OpenSlide pipeline benefits from more host threads
    # than the PyTorch DataLoader worker count it is mirroring.
    loader_thread_count = max(num_workers, min(
        host_cpu_count, num_workers * 2))
    worker_count = max(1, min(loader_thread_count, len(dataset)))

    def worker_generator(worker_index: np.integer) -> Generator[np.ndarray, None, None]:
        worker_dataset = dataset.clone(seed_offset=int(worker_index))
        yield from worker_dataset.generator()

    if worker_count == 1:
        tf_dataset = tf.data.Dataset.from_generator(
            dataset.generator,
            output_signature=output_signature,
        )
    else:
        worker_ids = tf.data.Dataset.range(worker_count)
        tf_dataset = worker_ids.interleave(
            lambda worker_id: tf.data.Dataset.from_generator(
                worker_generator,
                args=(worker_id,),
                output_signature=output_signature,
            ),
            cycle_length=worker_count,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        ).take(len(dataset))

    if shuffle:
        tf_dataset = tf_dataset.shuffle(
            min(len(dataset), max(batch_size * 16, 256)))

    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)
    options = tf.data.Options()
    options.threading.private_threadpool_size = loader_thread_count
    options.deterministic = not shuffle
    tf_dataset = tf_dataset.with_options(options)
    return tf_dataset.prefetch(max(2 * batch_size, worker_count))


# =============================================================================
# Entrypoint
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE on image data with TensorFlow")

    parser.add_argument("--data-root", type=str, required=True,
                        help="Directory containing .tif/.svs files")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Image/tile size (default: 256)")
    parser.add_argument("--img-channels", type=int,
                        default=3, help="Number of image channels")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Dataset worker thread count")
    parser.add_argument("--tiles-per-epoch", type=int,
                        default=10000, help="Number of tiles per epoch")
    parser.add_argument("--level", type=int, default=0,
                        help="OpenSlide pyramid level (0=highest resolution)")

    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base channel count (default: 32 for ~8M params)")
    parser.add_argument("--latent-channels", type=int,
                        default=32, help="Number of latent channels")
    parser.add_argument("--channel-multipliers", type=str,
                        default="1,2,4", help="Channel multipliers (comma-separated)")
    parser.add_argument("--num-res-blocks", type=int,
                        default=2, help="Residual blocks per stage")
    parser.add_argument("--use-attention-at", type=str, default="32",
                        help="Spatial sizes for attention (comma-separated)")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float,
                        default=0.01, help="Weight decay")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Maximum KL weight (beta-VAE)")
    parser.add_argument("--kl-warmup-steps", type=int,
                        default=8000, help="Steps for KL warmup")
    parser.add_argument("--max-grad-norm", type=float,
                        default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--recon-loss-type", type=str, default="l1",
                        choices=["l1", "l2"], help="Reconstruction loss type")

    parser.add_argument("--use-amp", action="store_true",
                        default=True, help="Use mixed precision training")
    parser.add_argument("--no-amp", action="store_false",
                        dest="use_amp", help="Disable mixed precision training")

    parser.add_argument("--log-dir", type=str,
                        default="runs_vae_tf", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints_vae_tf", help="Checkpoint directory")
    parser.add_argument("--log-interval", type=int,
                        default=100, help="Steps between scalar logging")
    parser.add_argument("--save-interval", type=int,
                        default=5, help="Epochs between checkpoints")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def configure_devices(device_arg: str) -> bool:
    if device_arg.lower() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    use_gpu = bool(gpus) and device_arg.lower() != "cpu"
    if use_gpu:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    return use_gpu


def save_metadata(
    checkpoint_dir: str,
    name: str,
    epoch: int,
    global_step: int,
    train_metrics: Optional[Dict[str, float]],
    val_metrics: Optional[Dict[str, float]],
    config: VAEConfig,
    kl_scheduler: Any,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": asdict(config),
        "kl_scheduler": kl_scheduler.state_dict(),
    }
    with open(os.path.join(checkpoint_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def create_optimizer(config: VAEConfig) -> tf.keras.optimizers.Optimizer:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        beta_1=config.betas[0],
        beta_2=config.betas[1],
        epsilon=1e-8,
        amsgrad=False,
        weight_decay=config.weight_decay,
        global_clipnorm=config.max_grad_norm,
    )

    if config.use_amp:
        return mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer


def main() -> None:
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to run vae_tf.py. Install tensorflow first."
        ) from TF_IMPORT_ERROR

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    use_gpu = configure_devices(args.device)
    print(f"Using device: {'GPU' if use_gpu else 'CPU'}")
    if use_gpu:
        print(f"GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")

    channel_multipliers = tuple(int(x)
                                for x in args.channel_multipliers.split(",") if x)
    use_attention_at = tuple(int(x)
                             for x in args.use_attention_at.split(",") if x)

    config = VAEConfig(
        img_channels=args.img_channels,
        img_size=args.img_size,
        base_channels=args.base_channels,
        channel_multipliers=channel_multipliers,
        latent_channels=args.latent_channels,
        num_res_blocks_per_stage=args.num_res_blocks,
        use_attention_at=use_attention_at,
        beta=args.beta,
        kl_warmup_steps=args.kl_warmup_steps,
        recon_loss_type=args.recon_loss_type,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        use_amp=args.use_amp and use_gpu,
    )
    config.validate()

    if config.use_amp:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")

    print("\nVAE Configuration:")
    print(
        f"  Image size: {config.img_size}x{config.img_size}x{config.img_channels}")
    print(
        f"  Latent size: {config.latent_size}x{config.latent_size}x{config.latent_channels}")
    print(f"  Base channels: {config.base_channels}")
    print(f"  Channel multipliers: {config.channel_multipliers}")
    print(f"  Attention at: {config.use_attention_at}")
    print(f"  Beta (max KL weight): {config.beta}")
    print(f"  KL warmup steps: {config.kl_warmup_steps}")
    print(f"  Recon loss type: {config.recon_loss_type}")
    print(f"  Mixed precision: {config.use_amp}")
    print()

    model = VAE(config=config)
    dummy_input = tf.zeros(
        [1, config.img_size, config.img_size, config.img_channels], dtype=tf.float32
    )
    model(dummy_input, training=False)

    num_params = int(np.sum([np.prod(var.shape) for var in model.variables]))
    num_trainable = int(np.sum([np.prod(var.shape)
                        for var in model.trainable_variables]))
    print(f"Model parameters: {num_params:,} ({num_trainable:,} trainable)")

    optimizer = create_optimizer(config)
    kl_scheduler = CyclicKLScheduler(
        beta=config.beta, cycle_steps=config.kl_warmup_steps, ratio=0.5)

    print(f"\nCreating dataset from TIF files in: {args.data_root}")
    train_dataset_obj = OpenSlideTileDataset(
        data_root=args.data_root,
        tile_size=config.img_size,
        tiles_per_epoch=args.tiles_per_epoch,
        level=args.level,
        color_jitter=True,
        color_jitter_strength=0.05,
        seed=args.seed,
    )
    val_dataset_obj = OpenSlideTileDataset(
        data_root=args.data_root,
        tile_size=config.img_size,
        tiles_per_epoch=max(1, args.tiles_per_epoch // 10),
        level=args.level,
        color_jitter=False,
        seed=args.seed + 1,
    )

    train_dataset = create_dataset(
        train_dataset_obj,
        batch_size=args.batch_size,
        img_size=config.img_size,
        img_channels=config.img_channels,
        shuffle=True,
        drop_remainder=True,
        num_workers=args.num_workers,
    )
    val_dataset = create_dataset(
        val_dataset_obj,
        batch_size=args.batch_size,
        img_size=config.img_size,
        img_channels=config.img_channels,
        shuffle=False,
        drop_remainder=False,
        num_workers=args.num_workers,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, run_name)
    writer = tf.summary.create_file_writer(log_path)
    print(f"TensorBoard logs: {log_path}")

    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=tf.Variable(0, dtype=tf.int64),
        epoch=tf.Variable(0, dtype=tf.int64),
    )
    periodic_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=os.path.join(args.checkpoint_dir, "periodic"),
        max_to_keep=5,
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=os.path.join(args.checkpoint_dir, "best"),
        max_to_keep=1,
        checkpoint_name="checkpoint_best",
    )

    global_step = 0
    best_val_loss = float("inf")
    steps_per_epoch = args.tiles_per_epoch // args.batch_size
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()

    for epoch in range(args.epochs):
        train_metrics, global_step = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            epoch=epoch,
            kl_scheduler=kl_scheduler,
            global_step=global_step,
            max_grad_norm=config.max_grad_norm,
            writer=writer,
            log_interval=args.log_interval,
        )

        current_kl_weight = kl_scheduler(global_step)
        val_metrics = evaluate(
            model=model, dataset=val_dataset, kl_weight=current_kl_weight)

        with writer.as_default():
            tf.summary.scalar("epoch/train_loss",
                              train_metrics["loss"], step=epoch)
            tf.summary.scalar("epoch/train_recon_loss",
                              train_metrics["recon_loss"], step=epoch)
            tf.summary.scalar("epoch/train_kl_loss",
                              train_metrics["kl_loss"], step=epoch)
            tf.summary.scalar("epoch/val_loss",
                              val_metrics["loss"], step=epoch)
            tf.summary.scalar("epoch/val_recon_loss",
                              val_metrics["recon_loss"], step=epoch)
            tf.summary.scalar("epoch/val_kl_loss",
                              val_metrics["kl_loss"], step=epoch)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"KL weight: {current_kl_weight:.4f}"
        )

        checkpoint.global_step.assign(global_step)
        checkpoint.epoch.assign(epoch)

        if (epoch + 1) % args.save_interval == 0 or val_metrics["loss"] < best_val_loss:
            if (epoch + 1) % args.save_interval == 0:
                save_path = periodic_manager.save(checkpoint_number=epoch + 1)
                print(f"Saved checkpoint: {save_path}")
                save_metadata(
                    args.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}",
                    epoch,
                    global_step,
                    train_metrics,
                    val_metrics,
                    config,
                    kl_scheduler,
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_path = best_manager.save(checkpoint_number=epoch + 1)
                print(f"Saved best checkpoint: {save_path}")
                save_metadata(
                    args.checkpoint_dir,
                    "checkpoint_best",
                    epoch,
                    global_step,
                    train_metrics,
                    val_metrics,
                    config,
                    kl_scheduler,
                )

    final_dir = os.path.join(args.checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_path = checkpoint.save(os.path.join(final_dir, "checkpoint_final"))
    print(f"Saved final checkpoint: {final_path}")
    save_metadata(
        args.checkpoint_dir,
        "checkpoint_final",
        args.epochs - 1,
        global_step,
        None,
        None,
        config,
        kl_scheduler,
    )

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
