"""Training and evaluation helpers for the TensorFlow VAE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .runtime import tf

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _create_progress(total: Optional[int], desc: str) -> Any:
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False)


def check_for_nan(loss: tf.Tensor, name: str = "loss") -> bool:
    if bool(tf.reduce_any(~tf.math.is_finite(loss)).numpy()):
        print(f"WARNING: {name} contains NaN or Inf!")
        return True
    return False


def _scale_loss_if_needed(optimizer: Any, loss: tf.Tensor) -> tf.Tensor:
    if hasattr(optimizer, "get_scaled_loss"):
        return optimizer.get_scaled_loss(loss)
    if hasattr(optimizer, "scale_loss"):
        return optimizer.scale_loss(loss)
    return loss


def _manually_unscale_gradient(grad: Any, scale: tf.Tensor) -> Any:
    if grad is None:
        return None
    if isinstance(grad, tf.IndexedSlices):
        return tf.IndexedSlices(
            values=tf.cast(grad.values, tf.float32) / scale,
            indices=grad.indices,
            dense_shape=grad.dense_shape,
        )
    return tf.cast(grad, tf.float32) / scale


def _unscale_gradients_if_needed(optimizer: Any, gradients: Any) -> Any:
    if hasattr(optimizer, "get_unscaled_gradients"):
        return optimizer.get_unscaled_gradients(gradients)

    loss_scale_factor = getattr(optimizer, "loss_scale_factor", None)
    if loss_scale_factor is None:
        return gradients

    scale = tf.cast(loss_scale_factor, tf.float32)
    return [_manually_unscale_gradient(grad, scale) for grad in gradients]


def _gradient_values(grad: Any) -> Any:
    if isinstance(grad, tf.IndexedSlices):
        return grad.values
    return grad


def _gradients_are_finite(gradients: Any) -> tf.Tensor:
    finite_flags = [
        tf.reduce_all(tf.math.is_finite(tf.cast(_gradient_values(grad), tf.float32)))
        for grad in gradients
        if grad is not None
    ]
    if not finite_flags:
        return tf.constant(False)
    return tf.reduce_all(tf.stack(finite_flags))


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

    image_count = tf.shape(x)[0]
    grid_orig = tf.reshape(tf.transpose(x, perm=[1, 0, 2, 3]), [tf.shape(x)[1], image_count * tf.shape(x)[2], tf.shape(x)[3]])
    grid_recon = tf.reshape(
        tf.transpose(x_recon, perm=[1, 0, 2, 3]),
        [tf.shape(x_recon)[1], image_count * tf.shape(x_recon)[2], tf.shape(x_recon)[3]],
    )
    combined = tf.expand_dims(tf.concat([grid_orig, grid_recon], axis=0), axis=0)

    with writer.as_default():
        tf.summary.image(f"{prefix}/orig_vs_recon", combined, step=step, max_outputs=1)


def run_train_step(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Any]:
    with tf.GradientTape() as tape:
        outputs = model(x, kl_weight=kl_weight, return_latent=True, training=True)
        loss = tf.cast(outputs["loss"], tf.float32)
        gradient_loss = _scale_loss_if_needed(optimizer, loss)

    gradients = tape.gradient(gradient_loss, model.trainable_variables)
    gradients = _unscale_gradients_if_needed(optimizer, gradients)
    return outputs, loss, gradients


def _run_train_step_compiled_impl(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    outputs, loss, gradients = run_train_step(model, x, kl_weight, optimizer)
    grad_var_pairs = [
        (grad, variable)
        for grad, variable in zip(gradients, model.trainable_variables)
        if grad is not None
    ]
    has_gradients = tf.constant(bool(grad_var_pairs))
    finite_loss = tf.reduce_all(tf.math.is_finite(loss))
    finite_gradients = _gradients_are_finite([grad for grad, _ in grad_var_pairs])

    def apply_gradients() -> tf.Tensor:
        optimizer.apply_gradients(grad_var_pairs)
        return tf.constant(True)

    did_apply = tf.cond(
        tf.logical_and(finite_loss, tf.logical_and(has_gradients, finite_gradients)),
        apply_gradients,
        lambda: tf.constant(False),
    )
    return outputs, loss, has_gradients, finite_gradients, did_apply


@tf.function(reduce_retracing=True)
def run_train_step_compiled(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _run_train_step_compiled_impl(model, x, kl_weight, optimizer)


@tf.function(reduce_retracing=True, jit_compile=True)
def run_train_step_xla(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _run_train_step_compiled_impl(model, x, kl_weight, optimizer)


def _run_eval_step_impl(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    return model(x, kl_weight=kl_weight, return_latent=False, training=False)


@tf.function(reduce_retracing=True)
def run_eval_step(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    return _run_eval_step_impl(model, x, kl_weight)


@tf.function(reduce_retracing=True, jit_compile=True)
def run_eval_step_xla(
    model: Any,
    x: tf.Tensor,
    kl_weight: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    return _run_eval_step_impl(model, x, kl_weight)


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
    total_batches: Optional[int] = None,
    progress_desc: Optional[str] = None,
    use_xla: bool = False,
) -> Tuple[Dict[str, float], int]:
    del max_grad_norm

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    progress = _create_progress(total_batches, progress_desc or f"Train {epoch + 1}")
    compiled_train_step = run_train_step_xla if use_xla else run_train_step_compiled

    for batch_idx, x in enumerate(dataset):
        if progress is not None:
            progress.update(1)

        kl_weight = kl_scheduler(global_step)

        if bool(tf.reduce_any(~tf.math.is_finite(x)).numpy()):
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input data")
            continue

        outputs, loss, has_gradients, finite_gradients, did_apply = compiled_train_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
            optimizer,
        )

        if check_for_nan(loss, "loss"):
            print(f"Skipping batch {batch_idx} due to NaN loss")
            continue

        if not bool(has_gradients.numpy()):
            print(f"Skipping batch {batch_idx} due to missing gradients")
            continue

        if not bool(finite_gradients.numpy()):
            print(f"Skipping batch {batch_idx} due to non-finite gradients")
            continue

        if not bool(did_apply.numpy()):
            print(f"Skipping batch {batch_idx} because the compiled train step did not apply gradients")
            continue

        loss_value = float(loss.numpy())
        recon_value = float(tf.cast(outputs["recon_loss"], tf.float32).numpy())
        kl_value = float(tf.cast(outputs["kl_loss"], tf.float32).numpy())

        total_loss += loss_value
        total_recon_loss += recon_value
        total_kl_loss += kl_value
        num_batches += 1

        if progress is not None:
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                recon=f"{recon_value:.4f}",
                kl=f"{kl_value:.4f}",
            )

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

    if progress is not None:
        progress.close()

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
    total_batches: Optional[int] = None,
    progress_desc: str = "Validation",
    use_xla: bool = False,
) -> Dict[str, float]:
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    progress = _create_progress(total_batches, progress_desc)
    compiled_eval_step = run_eval_step_xla if use_xla else run_eval_step

    for batch_idx, x in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if progress is not None:
            progress.update(1)

        outputs = compiled_eval_step(
            model,
            x,
            tf.convert_to_tensor(kl_weight, dtype=tf.float32),
        )
        loss_value = float(tf.cast(outputs["loss"], tf.float32).numpy())
        recon_value = float(tf.cast(outputs["recon_loss"], tf.float32).numpy())
        kl_value = float(tf.cast(outputs["kl_loss"], tf.float32).numpy())
        total_loss += loss_value
        total_recon_loss += recon_value
        total_kl_loss += kl_value
        num_batches += 1

        if progress is not None:
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                recon=f"{recon_value:.4f}",
                kl=f"{kl_value:.4f}",
            )

    if progress is not None:
        progress.close()

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon_loss / max(num_batches, 1),
        "kl_loss": total_kl_loss / max(num_batches, 1),
    }