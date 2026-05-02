#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
	echo "Usage: ./build_and_run_docker.sh [run_docker args]"
	echo "Builds the default image, then runs it with the provided run_docker arguments."
	exit 0
fi

"$SCRIPT_DIR/build_docker.sh"
"$SCRIPT_DIR/run_docker.sh" "$@"