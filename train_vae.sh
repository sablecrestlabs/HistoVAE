#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:-$HOME/Repositories/camelyon/CAMELYON177/images}"
source .venv/bin/activate
python vae.py --data-root $DATA_ROOT