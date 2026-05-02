"""Configuration objects for the TensorFlow VAE implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple


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