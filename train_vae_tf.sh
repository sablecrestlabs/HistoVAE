#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="$HOME/Datasets/CAMELYON17/images"
if [[ $# -gt 0 && "$1" != -* ]]; then
	DATA_ROOT="$1"
	shift
fi

source ./.venv/bin/activate
cd tensorflow
python -m src.cli --data-root "$DATA_ROOT" "$@"