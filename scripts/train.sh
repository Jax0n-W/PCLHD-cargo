#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./datasets/CARGO}"
LOGS_DIR="${LOGS_DIR:-./logs/cargo_baseline}"
GPU="${GPU:-0}"
SEED="${SEED:-1}"
AERIAL_EPS="${AERIAL_EPS:-0.40}"
GROUND_EPS="${GROUND_EPS:-0.40}"
ALL_EPS="${ALL_EPS:-0.40}"

CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/train_cargo.py \
  --data-dir "$DATA_DIR" \
  --logs-dir "$LOGS_DIR" \
  --stage1-log-name "cargo_pclhd_channel_off_s1_seed${SEED}" \
  --stage2-log-name "cargo_pclhd_cmhard_channel_off_s2_seed${SEED}" \
  --epochs 50 --iters 400 --eval-step 5 \
  --aerial-eps "$AERIAL_EPS" \
  --ground-eps "$GROUND_EPS" \
  --all-eps "$ALL_EPS" \
  --cluster-collapse-ratio 0.50 \
  --k1 30 --k2 6 --batch-size 64 --test-batch 64 \
  --height 288 --width 144 --num-instances 16 \
  --lr 0.00035 --weight-decay 0.0005 --step-size 20 \
  --temp 0.05 --momentum 0.2 --seed "$SEED" \
  --workers 8 --print-freq 50 --debug-nonfinite
