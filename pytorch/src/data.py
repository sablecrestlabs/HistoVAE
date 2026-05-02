"""OpenSlide-backed dataset helpers for PyTorch VAE training."""

from __future__ import annotations

import glob
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .runtime import (
    OPENSLIDE_AVAILABLE,
    PIL_AVAILABLE,
    TORCHVISION_AVAILABLE,
    Image,
    openslide,
    torch,
    transforms,
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


class OpenSlideTileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        tile_size: int = 256,
        tiles_per_epoch: int = 10000,
        level: int = 0,
        color_jitter: bool = False,
        color_jitter_strength: float = 0.05,
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

        self.tif_files = glob.glob(os.path.join(data_root, "*.tif"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.TIF"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.svs"))
        self.tif_files += glob.glob(os.path.join(data_root, "*.SVS"))
        self.tif_files += glob.glob(
            os.path.join(data_root, "**", "*.tif"), recursive=True
        )
        self.tif_files += glob.glob(
            os.path.join(data_root, "**", "*.TIF"), recursive=True
        )
        self.tif_files += glob.glob(
            os.path.join(data_root, "**", "*.svs"), recursive=True
        )
        self.tif_files += glob.glob(
            os.path.join(data_root, "**", "*.SVS"), recursive=True
        )
        self.tif_files = list(set(self.tif_files))

        if not self.tif_files:
            raise ValueError(f"No .tif/.svs files found in {data_root}")

        print(f"Found {len(self.tif_files)} TIF/SVS files in {data_root}")
        self._slide_cache: Dict[str, Any] = {}
        self._slide_dimensions: Dict[str, Tuple[int, int]] = {}
        self._invalid_slides: set[str] = set()

        self.to_tensor = transforms.ToTensor() if TORCHVISION_AVAILABLE else None
        if color_jitter and TORCHVISION_AVAILABLE:
            self.jitter = transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=color_jitter_strength * 0.5,
            )
        else:
            self.jitter = None

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
            tif_path = random.choice(self.tif_files)
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

            x_coord = random.randint(0, max_x)
            y_coord = random.randint(0, max_y)

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
                    (arr[:, :, 0] < 4) & (arr[:, :, 1] < 4) & (arr[:, :, 2] < 4)
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
                last_attempt_info = f"error reading {os.path.basename(tif_path)} at ({x_coord},{y_coord}): {exc}"

        debug_info = (
            f"Failed after {max_attempts} attempts. Empty tiles: {empty_tile_count}, "
            f"open errors: {open_error_count}. Last: {last_attempt_info}"
        )
        return None, debug_info

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        try:
            flip_h = Image.Transpose.FLIP_LEFT_RIGHT
            flip_v = Image.Transpose.FLIP_TOP_BOTTOM
        except AttributeError:
            flip_h = Image.FLIP_LEFT_RIGHT
            flip_v = Image.FLIP_TOP_BOTTOM

        if random.random() > 0.5:
            img = img.transpose(flip_h)
        if random.random() > 0.5:
            img = img.transpose(flip_v)

        rotations = random.randint(0, 3)
        if rotations > 0:
            img = img.rotate(rotations * 90, expand=False)

        if self.jitter is not None:
            img = self.jitter(img)
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        del idx
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
                    f"Warning: Struggled to find valid tile after {total_attempts} attempts. {debug_info}"
                )

        if img is None:
            raise RuntimeError(
                f"Could not find a valid tile after {max_total_attempts} attempts. "
                f"Last error: {debug_info}. Check that your TIF files contain non-empty regions."
            )

        img = self._apply_augmentations(img)
        if self.to_tensor is not None:
            tensor = self.to_tensor(img)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return tensor * 2.0 - 1.0

    def __del__(self) -> None:
        for slide in self._slide_cache.values():
            try:
                slide.close()
            except Exception:
                pass
