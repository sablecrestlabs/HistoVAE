#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

MODE="${HISTOVAE_ONNX_MODE:-reconstruct}"
ONNX_PATH="${HISTOVAE_ONNX_PATH:-pretrained/HistoVae_${MODE}.onnx}"
BATCH_SIZE="${HISTOVAE_ONNX_BATCH_SIZE:-16}"
DYNAMIC_BATCH="${HISTOVAE_ONNX_DYNAMIC_BATCH:-0}"

create_args=(
	--dynamo
	--mode "$MODE"
	--out "$ONNX_PATH"
)
validate_args=(
	--dynamo
	--mode "$MODE"
	--onnx "$ONNX_PATH"
	--batch-size "$BATCH_SIZE"
)

if [[ "$DYNAMIC_BATCH" == "1" ]]; then
	create_args+=(--dynamic-batch)
	validate_args+=(--dynamic-batch)
else
	create_args+=(--static-batch --static-batch-size "$BATCH_SIZE")
fi

python create_onnx.py "${create_args[@]}"
python validate_ort_cuda.py "${validate_args[@]}"