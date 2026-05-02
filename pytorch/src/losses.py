"""Losses and KL scheduling for PyTorch VAE training."""

from __future__ import annotations

from typing import Any, Dict

from .runtime import F, torch


def kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor, free_nats: float = 0.5
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    mu = torch.clamp(mu, min=-10.0, max=10.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.sum(dim=[1, 2, 3])
    kl = torch.clamp(kl, min=free_nats)
    kl = torch.nan_to_num(kl, nan=free_nats, posinf=1e3, neginf=free_nats)
    return kl.mean()


def reconstruction_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(x_recon, x, reduction="mean")
    if loss_type == "l2":
        return F.mse_loss(x_recon, x, reduction="mean")
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.beta = state_dict["beta"]
        self.warmup_steps = state_dict["warmup_steps"]


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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.beta = state_dict["beta"]
        self.cycle_steps = state_dict["cycle_steps"]
        self.ratio = state_dict["ratio"]
        self.warmup_steps = int(self.cycle_steps * self.ratio)
