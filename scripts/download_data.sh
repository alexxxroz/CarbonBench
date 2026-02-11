#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
REPO_ID="alexroz/CarbonBench"

if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Install it with: pip install huggingface_hub"
    exit 1
fi

echo "Downloading CarbonBench data to ${DATA_DIR} ..."
huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$DATA_DIR"
echo "Done."
