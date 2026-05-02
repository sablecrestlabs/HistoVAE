"""Training and evaluation helpers for the TensorFlow VAE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from vae_tf_runtime import tf


def check_for_nan(loss: tf.Tensor, name: str = "loss") -> bool:
    if bool(tf.reduce_any(~tf.math.is_finite(loss)).numpy()):
        print(f"WARNING: {name} contains NaN or Inf!")
        return True
    return False


def has_nonfinite_gradients(gradients: Any, variables: Any) -> bool:
    for grad, variable in zip(gradients, variables):
        if grad is None:
            continue
        if bool(tf.reduce_any(~tf.math.is_finite(grad)).numpy()):
            grad_min = float(tf.reduce_min(grad).numpy())
            grad_max = float(tf.reduce_max(grad).numpy())
            grad_mean = float(tf.reduce_mean(grad).numpy())
            print(
                f"Non-finite gradients detected in '{variable.name}': "
                f"min={grad_min:.3e}, max={grad_max:.3e}, mean={grad_mean:.3e}"
            )
            return True
    return False


def log_images(
    writer: tf.summary.SummaryWriter,
    x: tf.Tensor,
    x_recon: tf.Tensor,
    step: int,
    prefix: str = "train",
    num_images: int = 4,
) -> None:
    x = tf.cast(tf.clip_by_value((x[:num_images] + 1.0) / 2.0, 0.0, 1.0), tf.float32)
    x_recon = tf.cast(
        tf.clip_by_value((x_recon[:num_images] + 1.0) / 2.0, 0.0, 1.0),
        tf.float32,
    )
    combined = tf.concat([x, x_recon], axis=2)
    with writer.as_default():
        tf.summary.image(f"{prefix}/orig_vs_recon", combined, step=step, max_outputs=num_images)


@tf.function(reduce_retracing=True)
def run_train_step(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Any]:
    with tf.GradientTape() as tape:
        outputs = model(x, kl_weight=kl_weight, return_latent=True, training=True)
        loss = tf.cast(outputs["loss"], tf.float32)
        gradient_loss = optimizer.scale_loss(loss) if hasattr(optimizer, "scale_loss") else loss

    gradients = tape.gradient(gradient_loss, model.trainable_variables)
    return outputs, loss, gradients


@tf.function(reduce_retracing=True)
def run_eval_step(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    return model(x, kl_weight=kl_weight, return_latent=False, training=False)


def train_epoch(
    model: Any,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    epoch: int,
    kl_scheduler: Any,
    global_step: int,
    max_grad_norm: Optional[float] = None,
    writer: Optional[tf.summary.SummaryWriter] = None,
    log_interval: int = 100,
    image_log_interval: int = 1000,
) -> Tuple[Dict[str, float], int]:
    del epoch, max_grad_norm

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, x in enumerate(dataset):
        kl_weight = kl_scheduler(global_step)

        if bool(tf.reduce_any(~tf.math.is_finite(x)).numpy()):
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input data")
            continue

        outputs, loss, gradients = run_train_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
            optimizer,
        )

        if check_for_nan(loss, "loss"):
            print(f"Skipping batch {batch_idx} due to NaN loss")
            continue

        grad_var_pairs = [
            (grad, variable)
            for grad, variable in zip(gradients, model.trainable_variables)
            if grad is not None
        ]
        if not grad_var_pairs:
            print(f"Skipping batch {batch_idx} due to missing gradients")
            continue

        gradients, variables = zip(*grad_var_pairs)
        gradients = list(gradients)
        variables = list(variables)

        if has_nonfinite_gradients(gradients, variables):
            print(f"Skipping batch {batch_idx} due to non-finite gradients")
            continue

        optimizer.apply_gradients(zip(gradients, variables))

        loss_value = float(loss.numpy())
        recon_value = float(tf.cast(outputs["recon_loss"], tf.float32).numpy())
        kl_value = float(tf.cast(outputs["kl_loss"], tf.float32).numpy())

        total_loss += loss_value
        total_recon_loss += recon_value
        total_kl_loss += kl_value
        num_batches += 1

        if writer is not None and global_step % log_interval == 0:
            with writer.as_default():
                tf.summary.scalar("train/loss", loss_value, step=global_step)
                tf.summary.scalar("train/recon_loss", recon_value, step=global_step)
                tf.summary.scalar("train/kl_loss", kl_value, step=global_step)
                tf.summary.scalar("train/kl_weight", kl_weight, step=global_step)
                tf.summary.histogram("train/mu", outputs["mu"], step=global_step)
                tf.summary.histogram("train/logvar", outputs["logvar"], step=global_step)

        if writer is not None and global_step % image_log_interval == 0:
            log_images(writer, x, outputs["x_recon"], global_step, prefix="train")

        global_step += 1

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }
    return metrics, global_step


def evaluate(
    model: Any,
    dataset: tf.data.Dataset,
    kl_weight: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    for batch_idx, x in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break

        outputs = run_eval_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
        )
        total_loss += float(tf.cast(outputs["loss"], tf.float32).numpy())
        total_recon_loss += float(tf.cast(outputs["recon_loss"], tf.float32).numpy())
        total_kl_loss += float(tf.cast(outputs["kl_loss"], tf.float32).numpy())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }