"""Losses and scheduling utilities for TensorFlow VAE training."""

from __future__ import annotations

from typing import Any, Dict

from .runtime import tf


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