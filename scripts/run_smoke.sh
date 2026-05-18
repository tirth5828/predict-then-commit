#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoint.pt
