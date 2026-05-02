#!/usr/bin/env python3
"""Run a small forward/backward PyTorch VAE smoke test."""

from __future__ import annotations

from .cli import create_optimizer
from .config import VAEConfig
from .model import VAE
from .runtime import torch


def main() -> None:
    config = VAEConfig(
        img_size=32,
        img_channels=3,
        base_channels=16,
        channel_multipliers=(1, 2),
        downsample_factor=4,
        latent_channels=8,
        num_res_blocks_per_stage=1,
        use_attention_at=(),
        use_amp=False,
    )
    config.validate()

    model = VAE(config=config)
    optimizer = create_optimizer(config, model)
    inputs = torch.randn(2, config.img_channels, config.img_size, config.img_size)
    outputs = model(inputs, kl_weight=0.1, return_latent=True)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()

    non_null_gradients = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    if non_null_gradients == 0:
        raise RuntimeError("Smoke test failed: no gradients were produced.")

    print("PyTorch smoke test passed")
    print(
        {
            "loss": float(loss.item()),
            "recon_shape": tuple(outputs["x_recon"].shape),
            "latent_shape": tuple(outputs["z"].shape),
            "non_null_gradients": non_null_gradients,
        }
    )


if __name__ == "__main__":
    main()
