"""CLI and orchestration for PyTorch VAE training."""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .config import VAEConfig
from .data import OpenSlideTileDataset
from .losses import CyclicKLScheduler
from .model import VAE
from .runtime import AMP_AVAILABLE, GradScaler, SummaryWriter, TENSORBOARD_AVAILABLE, torch
from .training import evaluate, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on image data")
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing .tif/.svs files")
    parser.add_argument("--img-size", type=int, default=256, help="Image/tile size (default: 256)")
    parser.add_argument("--img-channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=12, help="Number of data loader workers")
    parser.add_argument("--tiles-per-epoch", type=int, default=10000, help="Number of tiles per epoch")
    parser.add_argument("--level", type=int, default=0, help="OpenSlide pyramid level (0=highest resolution)")

    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count (default: 32 for ~8M params)")
    parser.add_argument("--latent-channels", type=int, default=32, help="Number of latent channels")
    parser.add_argument("--channel-multipliers", type=str, default="1,2,4", help="Channel multipliers (comma-separated)")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Residual blocks per stage")
    parser.add_argument("--use-attention-at", type=str, default="32", help="Spatial sizes for attention (comma-separated)")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta", type=float, default=0.3, help="Maximum KL weight (beta-VAE)")
    parser.add_argument("--kl-warmup-steps", type=int, default=8000, help="Steps for KL warmup")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--recon-loss-type", type=str, default="l1", choices=["l1", "l2"], help="Reconstruction loss type")

    parser.add_argument("--use-amp", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="Disable mixed precision training")

    parser.add_argument("--log-dir", type=str, default="runs_vae_pytorch", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_vae", help="Checkpoint directory")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between logging")
    parser.add_argument("--save-interval", type=int, default=5, help="Epochs between checkpoints")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def create_optimizer(config: VAEConfig, model: VAE) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def build_checkpoint(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    kl_scheduler: Any,
    config: VAEConfig,
    epoch: int,
    global_step: int,
    train_metrics: Optional[Dict[str, float]] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "kl_scheduler_state_dict": kl_scheduler.state_dict(),
        "config": {
            "img_channels": config.img_channels,
            "img_size": config.img_size,
            "base_channels": config.base_channels,
            "channel_multipliers": config.channel_multipliers,
            "latent_channels": config.latent_channels,
            "num_res_blocks_per_stage": config.num_res_blocks_per_stage,
            "use_attention_at": config.use_attention_at,
            "beta": config.beta,
        },
    }
    if train_metrics is not None:
        checkpoint["train_metrics"] = train_metrics
    if val_metrics is not None:
        checkpoint["val_metrics"] = val_metrics
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    channel_multipliers = tuple(int(x) for x in args.channel_multipliers.split(",") if x)
    use_attention_at = tuple(int(x) for x in args.use_attention_at.split(",") if x)

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
        use_amp=args.use_amp and AMP_AVAILABLE,
    )
    config.validate()

    print("\nVAE Configuration:")
    print(f"  Image size: {config.img_size}x{config.img_size}x{config.img_channels}")
    print(f"  Latent size: {config.latent_size}x{config.latent_size}x{config.latent_channels}")
    print(f"  Base channels: {config.base_channels}")
    print(f"  Channel multipliers: {config.channel_multipliers}")
    print(f"  Attention at: {config.use_attention_at}")
    print(f"  Beta (max KL weight): {config.beta}")
    print(f"  KL warmup steps: {config.kl_warmup_steps}")
    print(f"  Recon loss type: {config.recon_loss_type}")
    print(f"  Mixed precision: {config.use_amp}")
    print()

    model = VAE(config=config).to(device)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    num_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_trainable:,} trainable)")

    optimizer = create_optimizer(config, model)
    kl_scheduler = CyclicKLScheduler(beta=config.beta, cycle_steps=config.kl_warmup_steps, ratio=0.5)

    scaler = None
    if config.use_amp and AMP_AVAILABLE:
        scaler = GradScaler()
        print("Using mixed precision training (AMP)")

    print(f"\nCreating dataset from TIF files in: {args.data_root}")
    train_dataset = OpenSlideTileDataset(
        data_root=args.data_root,
        tile_size=config.img_size,
        tiles_per_epoch=args.tiles_per_epoch,
        level=args.level,
        color_jitter=True,
        color_jitter_strength=0.05,
    )
    val_dataset = OpenSlideTileDataset(
        data_root=args.data_root,
        tile_size=config.img_size,
        tiles_per_epoch=args.tiles_per_epoch // 10,
        level=args.level,
        color_jitter=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    writer = None
    if TENSORBOARD_AVAILABLE:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, run_name)
        writer = SummaryWriter(log_dir=log_path)
        print(f"TensorBoard logs: {log_path}")

    global_step = 0
    best_val_loss = float("inf")
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Steps per epoch: {len(train_loader)}")
    print()

    for epoch in range(args.epochs):
        train_metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            kl_scheduler=kl_scheduler,
            global_step=global_step,
            scaler=scaler,
            max_grad_norm=config.max_grad_norm,
            writer=writer,
            log_interval=args.log_interval,
        )

        current_kl_weight = kl_scheduler(global_step)
        val_metrics = evaluate(model=model, dataloader=val_loader, device=device, kl_weight=current_kl_weight)

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            writer.add_scalar("epoch/train_recon_loss", train_metrics["recon_loss"], epoch)
            writer.add_scalar("epoch/train_kl_loss", train_metrics["kl_loss"], epoch)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("epoch/val_recon_loss", val_metrics["recon_loss"], epoch)
            writer.add_scalar("epoch/val_kl_loss", val_metrics["kl_loss"], epoch)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"KL weight: {current_kl_weight:.4f}"
        )

        if (epoch + 1) % args.save_interval == 0 or val_metrics["loss"] < best_val_loss:
            checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                kl_scheduler=kl_scheduler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                scaler=scaler,
            )

            if (epoch + 1) % args.save_interval == 0:
                save_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(checkpoint, save_path)
                print(f"Saved checkpoint: {save_path}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
                torch.save(checkpoint, save_path)
                print(f"Saved best checkpoint: {save_path}")

    checkpoint = build_checkpoint(
        model=model,
        optimizer=optimizer,
        kl_scheduler=kl_scheduler,
        config=config,
        epoch=args.epochs - 1,
        global_step=global_step,
        scaler=scaler,
    )
    save_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved final checkpoint: {save_path}")

    if writer is not None:
        writer.close()

    print("\nTraining complete!")