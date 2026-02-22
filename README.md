![Banner](HistoVAE_banner.png)

# HistoVAE

[![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)](#status)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#requirements)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![OpenSlide](https://img.shields.io/badge/WSI-OpenSlide-informational)](https://openslide.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue)](#license)

Fast-converging convolutional **Variational Autoencoder (VAE)** for **whole-slide image (WSI)** `.tif`/`.svs` files.

This repo trains directly on random WSI tiles via OpenSlide and is designed to converge quickly on histology tile distributions.

## What’s in this repo

- A single, self-contained training script: [vae.py](vae.py)
- Optional convenience scripts:
  - [train_vae.sh](train_vae.sh) (example invocation)
  - [tensorboard.sh](tensorboard.sh) (runs TensorBoard via Docker)
- Example weights: [pretrained/HistoVAE_trained.pt](pretrained/vae_trained.pt)

### Model/training highlights

Implemented in [vae.py](vae.py):

- Convolutional VAE with **spatial latents** (not flattened)
- **Cyclic KL annealing** to reduce posterior collapse
- **Mixed precision (AMP)** support
- TensorBoard logging (loss curves + image reconstructions)
- OpenSlide-backed dataset that samples random tiles and filters empty/background tiles

## Quickstart

### Requirements

- Python 3.9+
- A working OpenSlide install (system library) + `openslide-python`

On Ubuntu/Debian, you typically need:

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0
```

On macOS (Homebrew):

```bash
brew install openslide
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

Point `--data-root` at a directory containing WSI `.tif` / `.svs` files (case insensitive, recursively searched):

```bash
python vae.py --data-root /path/to/wsi_tifs
```

Common knobs:

```bash
python vae.py \
  --data-root /path/to/wsi_tifs \
  --img-size 256 \
  --batch-size 8 \
  --tiles-per-epoch 10000 \
  --level 0 \
  --epochs 50 \
  --beta 0.3 \
  --kl-warmup-steps 8000
```

By default, tiles are normalized from `[0, 1]` to `[-1, 1]` before being fed to the model.

### Monitor with TensorBoard

Training logs go under `runs_vae/<timestamp>/` by default.

If you have Docker, you can run:

```bash
# note: arguments are optional
./tensorboard.sh runs_vae 6006
```

Then open `http://localhost:6006`.

## Data format

`vae.py` uses `OpenSlideTileDataset`, which:

- Recursively scans `--data-root` for `.tif` and `.svs` files (case-insensitive)
- Randomly samples tile coordinates at a chosen OpenSlide pyramid `--level`
- Converts OpenSlide RGBA output to RGB on a white background
- Filters near-empty tiles (very low variance / mostly black / mostly white)
- Applies simple augmentations (random flips, rotations, optional light color jitter)

If you have tiles already extracted as PNG/JPEG, you’ll need to swap the dataset to a standard image-folder dataset.

## Outputs

- Checkpoints (default `--checkpoint-dir checkpoints_vae`):
  - `checkpoint_epoch_<N>.pt` (periodic)
  - `checkpoint_best.pt` (best validation loss)
  - `checkpoint_final.pt`
- TensorBoard logs (default `--log-dir runs_vae`):
  - Scalar losses (train/val)
  - Image grids of original vs reconstruction

## Loading a checkpoint (example)

Checkpoints saved by training are dictionaries with at least `model_state_dict`.

```python
import torch

from vae import VAE, VAEConfig

ckpt = torch.load("checkpoints_vae/checkpoint_best.pt", map_location="cpu")

# Training saves a small config subset in ckpt["config"].
cfg = ckpt.get("config", {})
config = VAEConfig(
    img_channels=cfg.get("img_channels", 3),
    img_size=cfg.get("img_size", 256),
    base_channels=cfg.get("base_channels", 32),
    channel_multipliers=tuple(cfg.get("channel_multipliers", (1, 2, 4))),
    latent_channels=cfg.get("latent_channels", 32),
)

model = VAE(config=config)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
```

## License

Dual-licensed under **MIT** and **Apache 2.0**.

- [LICENSE-MIT](LICENSE-MIT)
- [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)
