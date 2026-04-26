#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python create_onnx.py --dynamo
python validate_ort_cuda.py --dynamo --dynamic-batch