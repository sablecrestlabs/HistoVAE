#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_ROOT="$HOME/Datasets/CAMELYON17/images"
if [[ $# -gt 0 && "$1" != -* ]]; then
	DATA_ROOT="$1"
	shift
fi

IMAGE_NAME="${HISTOVAE_IMAGE_NAME:-histovae-pytorch}"
GPU_ARGS=()
if [[ "${HISTOVAE_DISABLE_GPU:-0}" != "1" ]]; then
	GPU_ARGS+=(--gpus all)
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
	TTY_ARGS+=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" "${GPU_ARGS[@]}" \
	-v "$DATA_ROOT:/data:ro" \
	-v "$SCRIPT_DIR/runs_vae:/workspace/pytorch/runs_vae" \
	-v "$SCRIPT_DIR/checkpoints_vae:/workspace/pytorch/checkpoints_vae" \
	"$IMAGE_NAME" \
	--data-root /data "$@"