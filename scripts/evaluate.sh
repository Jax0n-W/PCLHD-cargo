#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a Stage 1 or Stage 2 checkpoint path}"
DATA_DIR="${DATA_DIR:-./datasets/CARGO}"
GPU="${GPU:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-./evaluation.json}"

CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/evaluate_cargo.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --protocols all aa gg ag g2ag \
  --output-json "$OUTPUT_JSON"
