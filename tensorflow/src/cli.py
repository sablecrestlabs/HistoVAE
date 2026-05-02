"""CLI and orchestration for TensorFlow VAE training."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .config import VAEConfig
from .runtime import TF_AVAILABLE, TF_IMPORT_ERROR, mixed_precision, tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE on image data with TensorFlow"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory containing .tif/.svs files",
    )
    parser.add_argument(
        "--img-size", type=int, default=256, help="Image/tile size (default: 256)"
    )
    parser.add_argument(
        "--img-channels", type=int, default=3, help="Number of image channels"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=12, help="Dataset worker thread count"
    )
    parser.add_argument(
        "--tiles-per-epoch", type=int, default=10000, help="Number of tiles per epoch"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OpenSlide pyramid level (0=highest resolution)",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base channel count (default: 32 for ~8M params)",
    )
    parser.add_argument(
        "--latent-channels", type=int, default=32, help="Number of latent channels"
    )
    parser.add_argument(
        "--channel-multipliers",
        type=str,
        default="1,2,4",
        help="Channel multipliers (comma-separated)",
    )
    parser.add_argument(
        "--num-res-blocks", type=int, default=2, help="Residual blocks per stage"
    )
    parser.add_argument(
        "--use-attention-at",
        type=str,
        default="32",
        help="Spatial sizes for attention (comma-separated)",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--beta", type=float, default=0.3, help="Maximum KL weight (beta-VAE)"
    )
    parser.add_argument(
        "--kl-warmup-steps", type=int, default=8000, help="Steps for KL warmup"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm (0 to disable)",
    )
    parser.add_argument(
        "--recon-loss-type",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Reconstruction loss type",
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="use_amp",
        help="Disable mixed precision training",
    )

    parser.add_argument(
        "--log-dir", type=str, default="runs_vae", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_vae",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Steps between scalar logging"
    )
    parser.add_argument(
        "--save-interval", type=int, default=5, help="Epochs between checkpoints"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
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
    with open(
        os.path.join(checkpoint_dir, f"{name}.json"), "w", encoding="utf-8"
    ) as handle:
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
            "TensorFlow is required to run cli.py. Install tensorflow first."
        ) from TF_IMPORT_ERROR

    from .data import OpenSlideTileDataset, create_dataset
    from .losses import CyclicKLScheduler
    from .model import VAE
    from .training import evaluate, train_epoch

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    use_gpu = configure_devices(args.device)
    print(f"Using device: {'GPU' if use_gpu else 'CPU'}")
    if use_gpu:
        print(f"GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")
    use_xla = use_gpu

    channel_multipliers = tuple(
        int(x) for x in args.channel_multipliers.split(",") if x
    )
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
        use_amp=args.use_amp and use_gpu,
    )
    config.validate()

    if config.use_amp:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")

    print("\nVAE Configuration:")
    print(f"  Image size: {config.img_size}x{config.img_size}x{config.img_channels}")
    print(
        f"  Latent size: {config.latent_size}x{config.latent_size}x{config.latent_channels}"
    )
    print(f"  Base channels: {config.base_channels}")
    print(f"  Channel multipliers: {config.channel_multipliers}")
    print(f"  Attention at: {config.use_attention_at}")
    print(f"  Beta (max KL weight): {config.beta}")
    print(f"  KL warmup steps: {config.kl_warmup_steps}")
    print(f"  Recon loss type: {config.recon_loss_type}")
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  XLA JIT: {use_xla}")
    print()

    model = VAE(config=config)
    dummy_input = tf.zeros(
        [1, config.img_size, config.img_size, config.img_channels], dtype=tf.float32
    )
    model(dummy_input, training=False)

    num_params = int(np.sum([np.prod(var.shape) for var in model.variables]))
    num_trainable = int(
        np.sum([np.prod(var.shape) for var in model.trainable_variables])
    )
    print(f"Model parameters: {num_params:,} ({num_trainable:,} trainable)")

    optimizer = create_optimizer(config)
    kl_scheduler = CyclicKLScheduler(
        beta=config.beta, cycle_steps=config.kl_warmup_steps, ratio=0.5
    )

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
    val_steps = math.ceil(len(val_dataset_obj) / args.batch_size)
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()

    try:
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
                total_batches=steps_per_epoch,
                progress_desc=f"Train {epoch + 1}/{args.epochs}",
                use_xla=use_xla,
            )

            current_kl_weight = kl_scheduler(global_step)
            val_metrics = evaluate(
                model=model,
                dataset=val_dataset,
                kl_weight=current_kl_weight,
                total_batches=val_steps,
                progress_desc=f"Val {epoch + 1}/{args.epochs}",
                use_xla=use_xla,
            )

            with writer.as_default():
                tf.summary.scalar("epoch/train_loss", train_metrics["loss"], step=epoch)
                tf.summary.scalar(
                    "epoch/train_recon_loss", train_metrics["recon_loss"], step=epoch
                )
                tf.summary.scalar(
                    "epoch/train_kl_loss", train_metrics["kl_loss"], step=epoch
                )
                tf.summary.scalar("epoch/val_loss", val_metrics["loss"], step=epoch)
                tf.summary.scalar(
                    "epoch/val_recon_loss", val_metrics["recon_loss"], step=epoch
                )
                tf.summary.scalar(
                    "epoch/val_kl_loss", val_metrics["kl_loss"], step=epoch
                )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"KL weight: {current_kl_weight:.4f}"
            )

            checkpoint.global_step.assign(global_step)
            checkpoint.epoch.assign(epoch)

            if (epoch + 1) % args.save_interval == 0 or val_metrics[
                "loss"
            ] < best_val_loss:
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
    finally:
        if hasattr(train_dataset, "close"):
            train_dataset.close()
        if hasattr(val_dataset, "close"):
            val_dataset.close()

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
