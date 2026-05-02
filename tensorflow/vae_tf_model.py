"""TensorFlow VAE encoder, decoder, and top-level model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from vae_tf_config import VAEConfig
from vae_tf_layers import (
    DecoderStage,
    EncoderStage,
    get_activation,
    get_norm_layer,
    make_conv2d,
)
from vae_tf_losses import kl_divergence, reconstruction_loss
from vae_tf_runtime import tf


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
                next_channels = base_channels * channel_multipliers_rev[index + 1]
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

        recon_loss = reconstruction_loss(x, x_recon, loss_type=self.recon_loss_type)
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