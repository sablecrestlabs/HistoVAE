"""PyTorch VAE encoder, decoder, and top-level model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .config import VAEConfig
from .layers import (
    DecoderStage,
    EncoderStage,
    get_activation,
    get_norm_layer,
)
from .losses import kl_divergence, reconstruction_loss
from .runtime import nn, torch


class Encoder(nn.Module):
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
        self.use_attention_at = set(use_attention_at)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        stages = []
        in_ch = base_channels
        for index, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            stages.append(
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
        self.stages = nn.ModuleList(stages)

        final_ch = base_channels * channel_multipliers[-1]
        self.norm_out = get_norm_layer(final_ch, norm_type, norm_num_groups)
        self.act_out = get_activation(activation)
        self.conv_out = nn.Conv2d(
            final_ch, 2 * latent_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        current_res = x.shape[-1]
        for stage in self.stages:
            h, current_res = stage(h, current_res, self.use_attention_at)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
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
        self.conv_in = nn.Conv2d(latent_channels, first_ch, kernel_size=3, padding=1)

        stages = []
        in_ch = first_ch
        for index, mult in enumerate(channel_multipliers_rev):
            out_ch = base_channels * mult
            next_channels = None
            if index < len(channel_multipliers_rev) - 1:
                next_channels = base_channels * channel_multipliers_rev[index + 1]
            stages.append(
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
        self.stages = nn.ModuleList(stages)

        final_ch = base_channels * channel_multipliers_rev[-1]
        self.norm_out = get_norm_layer(final_ch, norm_type, norm_num_groups)
        self.act_out = get_activation(activation)
        self.conv_out = nn.Conv2d(final_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        current_res = z.shape[-1]
        for stage in self.stages:
            h, current_res = stage(h, current_res, self.use_attention_at)

        h = self.norm_out(h)
        h = self.act_out(h)
        x_recon = self.conv_out(h)
        return torch.tanh(x_recon)


class VAE(nn.Module):
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: torch.Tensor,
        kl_weight: float = 1.0,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = torch.clamp(x, min=-1.0, max=1.0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, training=self.training)
        z = torch.clamp(z, min=-10.0, max=10.0)
        x_recon = self.decode(z)
        x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)

        recon_loss = reconstruction_loss(x, x_recon, loss_type=self.recon_loss_type)
        kl_loss = kl_divergence(mu, logvar)
        recon_loss = torch.clamp(recon_loss, min=0.0, max=100.0)
        kl_loss = torch.clamp(kl_loss, min=0.0, max=1000.0)
        loss = recon_loss + kl_weight * kl_loss

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

    def sample(
        self,
        num_samples: int,
        device: torch.device,
        latent_size: Optional[int] = None,
    ) -> torch.Tensor:
        if latent_size is None:
            latent_size = self.img_size // (2 ** len(self.encoder.stages))
        z = torch.randn(
            num_samples,
            self.latent_channels,
            latent_size,
            latent_size,
            device=device,
        )
        with torch.no_grad():
            return self.decode(z)

    def reconstruct(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, training=not deterministic)
        return self.decode(z)
