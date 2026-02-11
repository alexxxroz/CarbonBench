#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
git clone https://huggingface.co/datasets/alexroz/CarbonBench "$DATA_DIR"
