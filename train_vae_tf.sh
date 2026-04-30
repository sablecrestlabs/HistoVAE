#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:-$HOME/Datasets/CAMELYON17/images}"
source .venv/bin/activate
python vae_tf.py --data-root "$DATA_ROOT"