#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:-$HOME/Repositories/camelyon/CAMELYON17/images}"
source .venv/bin/activate
python pytorch/vae_pytorch.py --data-root $DATA_ROOT