#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./datasets/CARGO}"
LOGS_DIR="${LOGS_DIR:-./logs/cargo_smoke}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/train_cargo.py \
  --data-dir "$DATA_DIR" \
  --logs-dir "$LOGS_DIR" \
  --stage1-log-name cargo_pclhd_channel_off_smoke_s1 \
  --stage2-log-name cargo_pclhd_cmhard_channel_off_smoke_s2 \
  --epochs 1 --iters 2 --skip-evaluation --smoke-max-ids 200 \
  --aerial-eps 0.40 --ground-eps 0.40 --all-eps 0.40 \
  --cluster-collapse-ratio 0.50 \
  --batch-size 64 --test-batch 64 --num-instances 16 \
  --workers 8 --print-freq 1 --debug-nonfinite
