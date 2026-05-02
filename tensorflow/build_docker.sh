#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${HISTOVAE_IMAGE_NAME:-histovae-tensorflow}"
if [[ $# -gt 0 && "$1" != -* ]]; then
	IMAGE_NAME="$1"
	shift
fi

docker build \
	-f "$SCRIPT_DIR/Dockerfile" \
	-t "$IMAGE_NAME" \
	"$@" \
	"$REPO_ROOT"