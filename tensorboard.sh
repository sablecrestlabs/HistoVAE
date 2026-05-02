#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p pytorch/runs_vae
mkdir -p tensorflow/runs_vae

LOGDIR_SPEC="${1:-pytorch:/workspace/pytorch/runs_vae,tensorflow:/workspace/tensorflow/runs_vae}"
PORT="${2:-6006}"

docker run --rm -it \
  -p "${PORT}:6006" \
  -v "$PWD:/workspace:ro" \
  hkube/tensorboard \
  tensorboard --logdir_spec="${LOGDIR_SPEC}" --bind_all