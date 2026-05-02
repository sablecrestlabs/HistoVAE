#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="$HOME/Datasets/CAMELYON17/images"
if [[ $# -gt 0 && "$1" != -* ]]; then
	DATA_ROOT="$1"
	shift
fi

IMAGE_NAME="${HISTOVAE_IMAGE_NAME:-histovae-tensorflow}"
TF_CPP_MIN_LOG_LEVEL_VALUE="${HISTOVAE_TF_CPP_MIN_LOG_LEVEL:-1}"
TF_XLA_FLAGS_VALUE="${HISTOVAE_TF_XLA_FLAGS:---tf_xla_enable_xla_devices=false}"
XLA_FLAGS_VALUE="${HISTOVAE_XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"
GPU_ARGS=()
if [[ "${HISTOVAE_DISABLE_GPU:-0}" != "1" ]]; then
	GPU_ARGS+=(--gpus all)
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
	TTY_ARGS+=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" "${GPU_ARGS[@]}" \
	-e TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL_VALUE" \
	-e TF_XLA_FLAGS="$TF_XLA_FLAGS_VALUE" \
	-e XLA_FLAGS="$XLA_FLAGS_VALUE" \
	-v "$DATA_ROOT:/data:ro" \
	-v "$SCRIPT_DIR/runs_vae:/workspace/tensorflow/runs_vae" \
	-v "$SCRIPT_DIR/checkpoints_vae:/workspace/tensorflow/checkpoints_vae" \
	"$IMAGE_NAME" \
	--data-root /data "$@"