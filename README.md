![Banner](HistoVAE_banner.png)

# HistoVAE

[![CI Status](https://github.com/sablecrestlabs/HistoVAE/actions/workflows/python.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sablecrestlabs%2Fhistovae-blue?logo=docker)](https://hub.docker.com/r/sablecrestlabs/histovae)
[![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)](#status)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#requirements)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900)](#run-with-docker-gpu)
[![OpenSlide](https://img.shields.io/badge/WSI-OpenSlide-informational)](https://openslide.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue)](#license)

Fast-converging convolutional **Variational Autoencoder (VAE)** for **whole-slide image (WSI)** `.tif`/`.svs` files.

This repo trains directly on random WSI tiles via OpenSlide and is designed to converge quickly on histology tile distributions.

Trained on a single RTX 5090 with default settings, this implementation demonstrates accurate half-resolution reconstruction within about 90 seconds. Within ~15 minutes, the reconstructed tiles are only distinguishable by differences in noise patterns. This makes it suitable for real-time, human-in-the-loop workflows.

## What’s in this repo

- Repo-root scripts: [train_vae_pytorch.sh](train_vae_pytorch.sh), [train_vae_tf.sh](train_vae_tf.sh), [tensorboard.sh](tensorboard.sh)
- PyTorch framework directory: [pytorch/requirements.txt](pytorch/requirements.txt)
- PyTorch container scripts: [pytorch/build_docker.sh](pytorch/build_docker.sh), [pytorch/run_docker.sh](pytorch/run_docker.sh)
- PyTorch source package: [pytorch/src/cli.py](pytorch/src/cli.py), [pytorch/src/model.py](pytorch/src/model.py), [pytorch/src/data.py](pytorch/src/data.py), [pytorch/src/layers.py](pytorch/src/layers.py), [pytorch/src/losses.py](pytorch/src/losses.py), [pytorch/src/training.py](pytorch/src/training.py), [pytorch/src/config.py](pytorch/src/config.py), [pytorch/src/runtime.py](pytorch/src/runtime.py), [pytorch/src/smoke_test.py](pytorch/src/smoke_test.py), [pytorch/src/create_onnx.py](pytorch/src/create_onnx.py), [pytorch/src/validate_ort_cuda.py](pytorch/src/validate_ort_cuda.py)
- TensorFlow framework directory: [tensorflow/requirements.txt](tensorflow/requirements.txt)
- TensorFlow container scripts: [tensorflow/build_docker.sh](tensorflow/build_docker.sh), [tensorflow/run_docker.sh](tensorflow/run_docker.sh)
- TensorFlow source package: [tensorflow/src/cli.py](tensorflow/src/cli.py), [tensorflow/src/model.py](tensorflow/src/model.py), [tensorflow/src/data.py](tensorflow/src/data.py), [tensorflow/src/layers.py](tensorflow/src/layers.py), [tensorflow/src/losses.py](tensorflow/src/losses.py), [tensorflow/src/training.py](tensorflow/src/training.py), [tensorflow/src/config.py](tensorflow/src/config.py), [tensorflow/src/runtime.py](tensorflow/src/runtime.py), [tensorflow/src/smoke_test.py](tensorflow/src/smoke_test.py)
- Optional convenience scripts:
  - [train_vae_pytorch.sh](train_vae_pytorch.sh) (PyTorch launcher)
  - [train_vae_tf.sh](train_vae_tf.sh) (TensorFlow launcher)
  - [tensorboard.sh](tensorboard.sh) (runs TensorBoard for both backends via Docker)
- Example weights: [pretrained/vae_trained.pt](pretrained/vae_trained.pt) trained on [CAMELYON17](https://camelyon17.grand-challenge.org/)

### Model/training highlights

Implemented in [pytorch/src](pytorch/src):

- Convolutional VAE with **spatial latents** (not flattened)
- **Cyclic KL annealing** to reduce posterior collapse
- **Mixed precision (AMP)** support
- TensorBoard logging (loss curves + image reconstructions)
- Terminal batch progress bars for training and validation
- OpenSlide-backed dataset that samples random tiles and filters empty/background tiles

## Quickstart

## Run with Docker (GPU)

This repo now keeps framework-specific Dockerfiles in [pytorch/Dockerfile](pytorch/Dockerfile) and [tensorflow/Dockerfile](tensorflow/Dockerfile). To use the GPU, you’ll need:

The GitHub Docker publish workflow continues to publish the PyTorch image to `sablecrestlabs/histovae:latest`, and tagged releases additionally publish `sablecrestlabs/histovae:<tag>`.

- NVIDIA drivers installed on the host
- Docker + NVIDIA Container Toolkit (so `--gpus all` works)

### Pull

```bash
docker pull sablecrestlabs/histovae:latest
```

### Build

```bash
./pytorch/build_docker.sh histovae-pytorch
./tensorflow/build_docker.sh histovae-tensorflow
```

PyTorch build defaults are set in [pytorch/Dockerfile](pytorch/Dockerfile) (`PYTORCH_VERSION=2.10.0`, `CUDA_VERSION=13.0`). You can override them:

```bash
./pytorch/build_docker.sh histovae-pytorch \
  --build-arg PYTORCH_VERSION=2.10.0 \
  --build-arg CUDA_VERSION=13.0
```

The PyTorch image sets `CUDA_VERSION` inside the container as an environment variable as well.

TensorFlow uses NVIDIA's TensorFlow NGC image as its base so the container ships with an NVIDIA-validated GPU stack rather than falling back to a generic pip wheel that JIT-compiles PTX for newer GPUs at startup. The default tag is `25.02-tf2-py3`, and you can override it:

```bash
./tensorflow/build_docker.sh histovae-tensorflow \
  --build-arg NVIDIA_TENSORFLOW_TAG=25.02-tf2-py3
```

The TensorFlow image and wrapper also default to disabling TensorFlow's XLA device exposure and Triton GEMM path inside the container. On RTX 5090 / Blackwell-class GPUs this avoids the repeated `+ptx85` feature warnings some NGC TensorFlow 2.17 builds emit at runtime. To opt back in for debugging or performance experiments, set `HISTOVAE_TF_XLA_FLAGS` and/or `HISTOVAE_XLA_FLAGS` before running [tensorflow/run_docker.sh](tensorflow/run_docker.sh), or pass replacement `TF_XLA_FLAGS` / `XLA_FLAGS` directly to `docker run`.

### Train (mount host data directory)

Mount your WSI directory from the host into `/data` in the container:

```bash
docker run --rm --gpus all \
  -v /host/path/to/wsi_files:/data:ro \
  -v "$PWD/pytorch/runs_vae:/workspace/pytorch/runs_vae" \
  -v "$PWD/pytorch/checkpoints_vae:/workspace/pytorch/checkpoints_vae" \
  histovae-pytorch \
  --data-root /data \
  --device cuda
```

Or use the wrapper, which mirrors the local train script interface:

```bash
./pytorch/run_docker.sh /path/to/wsi_files --device cuda
```

For TensorFlow:

```bash
docker run --rm --gpus all \
  -v /host/path/to/wsi_files:/data:ro \
  -v "$PWD/tensorflow/runs_vae:/workspace/tensorflow/runs_vae" \
  -v "$PWD/tensorflow/checkpoints_vae_tf:/workspace/tensorflow/checkpoints_vae_tf" \
  histovae-tensorflow \
  --data-root /data \
  --device cuda
```

Or use the wrapper:

```bash
./tensorflow/run_docker.sh /path/to/wsi_files --device cuda
```

If you want a shell instead of running training, override the entrypoint:

```bash
docker run --rm -it --gpus all \
  -v /host/path/to/wsi:/data:ro \
  --entrypoint bash \
  histovae-pytorch
```

## Run on bare metal

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
pip install -r pytorch/requirements.txt
```

For the TensorFlow port specifically, install [tensorflow/requirements.txt](tensorflow/requirements.txt) in the active environment.

### Train

Point `--data-root` at a directory containing WSI `.tif` / `.svs` files (case insensitive, recursively searched):

```bash
cd pytorch
python -m src.cli --data-root /path/to/wsi_files
```

Or use the repo-root launcher:

```bash
./train_vae_pytorch.sh /path/to/wsi_files
```

Common knobs:

```bash
cd pytorch
python -m src.cli \
  --data-root /path/to/wsi_files \
  --img-size 256 \
  --batch-size 8 \
  --tiles-per-epoch 10000 \
  --level 0 \
  --epochs 50 \
  --beta 0.3 \
  --kl-warmup-steps 8000
```

By default, tiles are normalized from `[0, 1]` to `[-1, 1]` before being fed to the model.

### Train with TensorFlow

The TensorFlow port keeps its implementation under [tensorflow/src](tensorflow/src) while the repo-root launcher delegates into that directory.

```bash
cd tensorflow
python -m src.cli --data-root /path/to/wsi_files
```

Or use the repo-root launcher:

```bash
./train_vae_tf.sh /path/to/wsi_files
```

### PyTorch Smoke Test

Run a quick forward/backward verification of the PyTorch implementation with:

```bash
cd pytorch
python -m src.smoke_test
```

### TensorFlow Smoke Test

Run a quick forward/backward verification of the TensorFlow implementation with:

```bash
cd tensorflow
python -m src.smoke_test
```

### Supported Formats

HistoVAE relies on [OpenSlide](https://openslide.org/) for slide access, so the formats it can open are the formats OpenSlide supports on the host system. Supported formats include:

- `.svs`
- `.tif`
- `.dcm`
- `.ndpi`
- `.vms`
- `.vmu`
- `.scn`
- `.mrxs`
- `.tiff`
- `.svslide`
- `.bif`
- `.czi`

### Monitor with TensorBoard

PyTorch logs go under `pytorch/runs_vae/<timestamp>/` by default, and TensorFlow logs go under `tensorflow/runs_vae/<timestamp>/` by default.

If you have Docker, you can run:

```bash
# optional args: logdir spec, port
./tensorboard.sh
```

The default TensorBoard view exposes both backends in one UI as `pytorch` and `tensorflow`. To override that, pass a custom `--logdir_spec`-style first argument such as `custom:/workspace/pytorch/runs_vae`.

Then open `http://localhost:6006`.

## Data format

[pytorch/src/data.py](pytorch/src/data.py) provides `OpenSlideTileDataset`, which:

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

from src.config import VAEConfig
from src.model import VAE

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
