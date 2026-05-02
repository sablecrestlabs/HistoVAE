#!/usr/bin/env python3
"""Run a small forward/backward TensorFlow VAE smoke test."""

from __future__ import annotations

from .cli import create_optimizer
from .config import VAEConfig
from .model import VAE
from .runtime import TF_AVAILABLE, TF_IMPORT_ERROR, mixed_precision, tf
from .training import run_train_step


def main() -> None:
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to run the smoke test. Install tensorflow first."
        ) from TF_IMPORT_ERROR

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

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
    mixed_precision.set_global_policy("float32")

    model = VAE(config=config)
    optimizer = create_optimizer(config)
    inputs = tf.random.normal([2, config.img_size, config.img_size, config.img_channels])

    outputs = model(inputs, training=False)
    train_outputs, loss, gradients = run_train_step(
        model,
        inputs,
        tf.convert_to_tensor(0.1, dtype=tf.float32),
        optimizer,
    )
    non_null_gradients = sum(gradient is not None for gradient in gradients)

    if non_null_gradients == 0:
        raise RuntimeError("Smoke test failed: no gradients were produced.")

    print("TensorFlow smoke test passed")
    print(
        {
            "loss": float(loss.numpy()),
            "recon_shape": tuple(outputs["x_recon"].shape),
            "latent_shape": tuple(train_outputs["z"].shape),
            "non_null_gradients": non_null_gradients,
        }
    )


if __name__ == "__main__":
    main()