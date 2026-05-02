"""Training and evaluation helpers for the PyTorch VAE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .runtime import TORCHVISION_AVAILABLE, autocast, make_grid, torch


def check_for_nan(loss: torch.Tensor, name: str = "loss") -> bool:
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"WARNING: {name} contains NaN or Inf!")
        return True
    return False


def has_nonfinite_gradients(model: torch.nn.Module) -> bool:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            grad_min = parameter.grad.min().item()
            grad_max = parameter.grad.max().item()
            grad_mean = parameter.grad.mean().item()
            print(
                f"Non-finite gradients detected in '{name}': "
                f"min={grad_min:.3e}, max={grad_max:.3e}, mean={grad_mean:.3e}"
            )
            return True
    return False


def log_images(
    writer: Any,
    x: torch.Tensor,
    x_recon: torch.Tensor,
    step: int,
    prefix: str = "train",
    num_images: int = 4,
) -> None:
    if not TORCHVISION_AVAILABLE:
        return
    x = x[:num_images].cpu()
    x_recon = x_recon[:num_images].cpu()
    x = torch.clamp((x + 1.0) / 2.0, 0, 1)
    x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0, 1)
    grid_orig = make_grid(x, nrow=num_images, normalize=False)
    grid_recon = make_grid(x_recon, nrow=num_images, normalize=False)
    combined = torch.cat([grid_orig, grid_recon], dim=1)
    writer.add_image(f"{prefix}/orig_vs_recon", combined, step)


def train_epoch(
    model: Any,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    kl_scheduler: Any,
    global_step: int,
    scaler: Optional[Any] = None,
    max_grad_norm: Optional[float] = None,
    writer: Optional[Any] = None,
    log_interval: int = 100,
    image_log_interval: int = 1000,
) -> Tuple[Dict[str, float], int]:
    del epoch
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        kl_weight = kl_scheduler(global_step)
        optimizer.zero_grad()

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input data")
            continue

        if scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x, kl_weight=kl_weight, return_latent=True)
                loss = outputs["loss"]

            if check_for_nan(loss, "loss"):
                print(f"Skipping batch {batch_idx} due to NaN loss (forward)")
                optimizer.zero_grad(set_to_none=True)
                scaler.update(scaler.get_scale() * 0.5)
                continue

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, kl_weight=kl_weight, return_latent=True)
            loss = outputs["loss"]

            if check_for_nan(loss, "loss"):
                print(f"Skipping batch {batch_idx} due to NaN loss")
                continue

            loss.backward()

            if has_nonfinite_gradients(model):
                print(f"Skipping batch {batch_idx} due to non-finite gradients")
                optimizer.zero_grad(set_to_none=True)
                continue

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        total_loss += loss.item()
        total_recon_loss += outputs["recon_loss"].item()
        total_kl_loss += outputs["kl_loss"].item()
        num_batches += 1

        if writer is not None and global_step % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/recon_loss", outputs["recon_loss"].item(), global_step)
            writer.add_scalar("train/kl_loss", outputs["kl_loss"].item(), global_step)
            writer.add_scalar("train/kl_weight", kl_weight, global_step)
            if "mu" in outputs and "logvar" in outputs:
                writer.add_histogram("train/mu", outputs["mu"].detach(), global_step)
                writer.add_histogram("train/logvar", outputs["logvar"].detach(), global_step)

        if writer is not None and global_step % image_log_interval == 0:
            log_images(writer, x, outputs["x_recon"], global_step, prefix="train")

        global_step += 1

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }
    return metrics, global_step


@torch.no_grad()
def evaluate(
    model: Any,
    dataloader: Any,
    device: torch.device,
    kl_weight: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        outputs = model(x, kl_weight=kl_weight, return_latent=False)
        total_loss += outputs["loss"].item()
        total_recon_loss += outputs["recon_loss"].item()
        total_kl_loss += outputs["kl_loss"].item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }