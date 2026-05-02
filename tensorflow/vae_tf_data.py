"""OpenSlide-backed tile dataset helpers for the TensorFlow VAE."""

from __future__ import annotations

import glob
import os
import random
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from vae_tf_runtime import (
    Image,
    ImageEnhance,
    OPENSLIDE_AVAILABLE,
    PIL_AVAILABLE,
    openslide,
    tf,
)


def has_content(
    img: Image.Image,
    min_std: float = 5.0,
    min_mean: float = 10.0,
    max_mean: float = 245.0,
) -> bool:
    arr = np.array(img, dtype=np.float32)
    if arr.std() < min_std:
        return False
    if arr.mean() < min_mean:
        return False
    if arr.mean() > max_mean:
        return False
    return True


class OpenSlideTileDataset:
    """Random tile dataset backed by OpenSlide for tf.data generators."""

    def __init__(
        self,
        data_root: str,
        tile_size: int = 256,
        tiles_per_epoch: int = 10000,
        level: int = 0,
        color_jitter: bool = False,
        color_jitter_strength: float = 0.05,
        seed: Optional[int] = None,
    ):
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python is required. Install with: pip install openslide-python"
            )
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        self.data_root = data_root
        self.tile_size = tile_size
        self.tiles_per_epoch = tiles_per_epoch
        self.level = level
        self.color_jitter = color_jitter
        self.color_jitter_strength = color_jitter_strength
        self.seed = seed
        self._rng = random.Random(seed)

        self.tif_files = glob.glob(os.path.join(data_root, "*.tif"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.TIF"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.svs"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.SVS"))
        self.tif_files += glob.glob(os.path.join(data_root, "**", "*.tif"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root, "**", "*.TIF"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root, "**", "*.svs"), recursive=True)
        self.tif_files += glob.glob(os.path.join(data_root, "**", "*.SVS"), recursive=True)
        self.tif_files = list(set(self.tif_files))

        if not self.tif_files:
            raise ValueError(f"No .tif/.svs files found in {data_root}")

        print(f"Found {len(self.tif_files)} TIF/SVS files in {data_root}")
        self._slide_cache: Dict[str, Any] = {}
        self._slide_dimensions: Dict[str, Tuple[int, int]] = {}
        self._invalid_slides = set()

    def clone(self, seed_offset: int = 0) -> "OpenSlideTileDataset":
        clone_seed = None if self.seed is None else self.seed + seed_offset + 1
        return OpenSlideTileDataset(
            data_root=self.data_root,
            tile_size=self.tile_size,
            tiles_per_epoch=self.tiles_per_epoch,
            level=self.level,
            color_jitter=self.color_jitter,
            color_jitter_strength=self.color_jitter_strength,
            seed=clone_seed,
        )

    def _get_slide_with_dims(
        self, tif_path: str
    ) -> Optional[Tuple[Any, Tuple[int, int]]]:
        if tif_path in self._invalid_slides:
            return None

        if tif_path not in self._slide_cache:
            try:
                slide = openslide.OpenSlide(tif_path)
                level = min(self.level, slide.level_count - 1)
                dims = slide.level_dimensions[level]

                if dims[0] < self.tile_size or dims[1] < self.tile_size:
                    slide.close()
                    self._invalid_slides.add(tif_path)
                    return None

                self._slide_cache[tif_path] = slide
                self._slide_dimensions[tif_path] = dims
            except Exception as exc:
                print(f"Warning: Could not open {tif_path}: {exc}")
                self._invalid_slides.add(tif_path)
                return None

        return self._slide_cache[tif_path], self._slide_dimensions[tif_path]

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def _extract_random_tile(
        self, max_attempts: int = 50
    ) -> Tuple[Optional[Image.Image], str]:
        last_attempt_info = "no attempts made"
        empty_tile_count = 0
        open_error_count = 0

        for _ in range(max_attempts):
            tif_path = self._rng.choice(self.tif_files)
            result = self._get_slide_with_dims(tif_path)
            if result is None:
                open_error_count += 1
                last_attempt_info = f"failed to open {os.path.basename(tif_path)}"
                continue

            slide, dims = result
            max_x = dims[0] - self.tile_size
            max_y = dims[1] - self.tile_size

            if max_x <= 0 or max_y <= 0:
                last_attempt_info = (
                    f"{os.path.basename(tif_path)} too small ({dims[0]}x{dims[1]})"
                )
                continue

            x_coord = self._rng.randint(0, max_x)
            y_coord = self._rng.randint(0, max_y)

            try:
                level = min(self.level, slide.level_count - 1)
                downsample = slide.level_downsamples[level]
                level0_x = int(x_coord * downsample)
                level0_y = int(y_coord * downsample)
                img = slide.read_region(
                    (level0_x, level0_y), level, (self.tile_size, self.tile_size)
                )

                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert("RGB")

                arr = np.array(img)
                near_black_mask = (
                    (arr[:, :, 0] < 4)
                    & (arr[:, :, 1] < 4)
                    & (arr[:, :, 2] < 4)
                )
                arr[near_black_mask] = [255, 255, 255]
                img = Image.fromarray(arr)

                if has_content(img):
                    return img, "success"

                empty_tile_count += 1
                last_attempt_info = (
                    f"{os.path.basename(tif_path)} at ({x_coord},{y_coord}) was empty "
                    "(black/white/uniform)"
                )
            except Exception as exc:
                last_attempt_info = (
                    f"error reading {os.path.basename(tif_path)} at "
                    f"({x_coord},{y_coord}): {exc}"
                )

        debug_info = (
            f"Failed after {max_attempts} attempts. Empty tiles: {empty_tile_count}, "
            f"open errors: {open_error_count}. Last: {last_attempt_info}"
        )
        return None, debug_info

    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        if not self.color_jitter:
            return img

        strength = self.color_jitter_strength
        brightness = self._rng.uniform(1.0 - strength, 1.0 + strength)
        contrast = self._rng.uniform(1.0 - strength, 1.0 + strength)
        saturation = self._rng.uniform(1.0 - strength, 1.0 + strength)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        try:
            flip_h = Image.Transpose.FLIP_LEFT_RIGHT
            flip_v = Image.Transpose.FLIP_TOP_BOTTOM
        except AttributeError:
            flip_h = Image.FLIP_LEFT_RIGHT
            flip_v = Image.FLIP_TOP_BOTTOM

        if self._rng.random() > 0.5:
            img = img.transpose(flip_h)
        if self._rng.random() > 0.5:
            img = img.transpose(flip_v)

        rotations = self._rng.randint(0, 3)
        if rotations > 0:
            img = img.rotate(rotations * 90, expand=False)

        return self._apply_color_jitter(img)

    def __getitem__(self, idx: int) -> np.ndarray:
        total_attempts = 0
        max_total_attempts = 500
        img = None
        debug_info = ""

        while total_attempts < max_total_attempts:
            img, debug_info = self._extract_random_tile()
            total_attempts += 50
            if img is not None:
                break
            if total_attempts % 100 == 0:
                print(
                    f"Warning: Struggled to find valid tile after {total_attempts} "
                    f"attempts. {debug_info}"
                )

        if img is None:
            raise RuntimeError(
                f"Could not find a valid tile after {max_total_attempts} attempts. "
                f"Last error: {debug_info}. Check that your TIF files contain "
                "non-empty regions."
            )

        img = self._apply_augmentations(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return arr

    def generator(self) -> Generator[np.ndarray, None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __del__(self) -> None:
        for slide in self._slide_cache.values():
            try:
                slide.close()
            except Exception:
                pass


def create_dataset(
    dataset: OpenSlideTileDataset,
    batch_size: int,
    img_size: int,
    img_channels: int,
    shuffle: bool,
    drop_remainder: bool,
    num_workers: int,
) -> tf.data.Dataset:
    output_signature = tf.TensorSpec(
        shape=(img_size, img_size, img_channels), dtype=tf.float32
    )
    host_cpu_count = max(1, os.cpu_count() or 1)
    loader_thread_count = max(num_workers, min(host_cpu_count, num_workers * 2))
    worker_count = max(1, min(loader_thread_count, len(dataset)))

    def worker_generator(worker_index: np.integer) -> Generator[np.ndarray, None, None]:
        worker_dataset = dataset.clone(seed_offset=int(worker_index))
        yield from worker_dataset.generator()

    if worker_count == 1:
        tf_dataset = tf.data.Dataset.from_generator(
            dataset.generator,
            output_signature=output_signature,
        )
    else:
        worker_ids = tf.data.Dataset.range(worker_count)
        tf_dataset = worker_ids.interleave(
            lambda worker_id: tf.data.Dataset.from_generator(
                worker_generator,
                args=(worker_id,),
                output_signature=output_signature,
            ),
            cycle_length=worker_count,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        ).take(len(dataset))

    if shuffle:
        tf_dataset = tf_dataset.shuffle(min(len(dataset), max(batch_size * 16, 256)))

    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)
    options = tf.data.Options()
    options.threading.private_threadpool_size = loader_thread_count
    options.deterministic = not shuffle
    tf_dataset = tf_dataset.with_options(options)
    return tf_dataset.prefetch(max(2 * batch_size, worker_count))