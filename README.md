# CARGO PCLHD-CMhard Baseline

This repository contains a focused, source-only implementation of a two-stage
PCLHD baseline for the CARGO aerial-ground person re-identification dataset.
It is intended for reproducible baseline and ablation experiments; datasets,
pretrained weights, checkpoints, logs, and reported result files are not
included.

## Frozen experiment configuration

| Component | Setting |
|---|---|
| Stage 1 memory | `CM` |
| Stage 2 memory / hard loss | `CMhard` |
| ChannelAdapGray / ChannelExchange | disabled |
| Dynamic AGVA | disabled |
| Three-domain matching | disabled |
| EMA loss in backward | disabled |
| EMA encoder update | enabled, momentum 0.999 |
| ALL-memory second optimizer step | enabled |
| Backbone | CARGO-specific stable AGW |
| Checkpoint policy | fixed final epoch |

The stable AGW keeps the original topology and evaluates GeM through an
algebraically equivalent rescaling that avoids FP32 cubic overflow. The code
also supports independent aerial, ground, and ALL DBSCAN thresholds,
non-finite diagnostics, cluster-health logging, and an optional label-free
cluster-collapse guard.

## Repository structure

```text
cargo_baseline/
  train_cargo.py              public training entry
  _train_cargo_engine.py      two-stage training implementation
  inspect_cargo.py            read-only dataset audit
  diagnose_eps.py             label-free DBSCAN threshold scan
  evaluate_cargo.py           checkpoint evaluation entry
  cargo_evaluation.py         CARGO protocol implementation
clustercontrast/              minimal PCLHD dependencies
scripts/                      smoke, training, and evaluation launchers
tests/                        static release-contract tests
```

## Installation

Linux, Python 3.10, CUDA, and a CUDA-compatible PyTorch/FAISS combination are
recommended.

```bash
conda env create -f environment.yml
conda activate cargo-pclhd
```

Alternatively, install PyTorch and FAISS for your CUDA version first, then run:

```bash
pip install -r requirements.txt
```

The model initializes three ResNet-50 branches from ImageNet-pretrained
weights. PyTorch may download these weights on the first run.

## Dataset preparation

The CARGO dataset must be obtained separately. Expected layout:

```text
CARGO/
  train/
    Cam1/ ... Cam13/
  query/
    Cam1/ ... Cam13/
  gallery/
    Cam1/ ... Cam13/
```

Cameras 1--5 are treated as aerial and cameras 6--13 as ground. File names
must follow:

```text
Cam<camera>_<day|night>_<person_id>_<index>.jpg
```

Audit an extraction before training:

```bash
python cargo_baseline/inspect_cargo.py \
  --data-dir /path/to/CARGO \
  --output cargo_protocol_report.json
```

Training identities are parsed only to describe and relabel the source split;
the optimization loop replaces them with DBSCAN pseudo labels.

## Low-cost smoke test

```bash
DATA_DIR=/path/to/CARGO GPU=0 bash scripts/smoke_test.sh
```

This keeps 200 training identities, runs one epoch with two iterations, and
skips evaluation. It checks execution only and must not be reported as a ReID
result.

## Select DBSCAN thresholds

Do not select thresholds from test Rank-1 or mAP. With an available Stage 1
checkpoint, scan cluster/noise/size behavior without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/diagnose_eps.py \
  --data-dir /path/to/CARGO \
  --checkpoint /path/to/stage1/checkpoint.pth.tar \
  --eps-values 0.25 0.30 0.35 0.40 0.45 0.50 \
  --output ./logs/cargo_eps_scan.json
```

Add `--include-all` only when the larger ALL-distance matrix is required.
Replace the example thresholds below with values fixed from the label-free
diagnostic. In the development run, using 0.60 merged many identities into
large pseudo clusters and is therefore not a safe universal default.

## Formal training

```bash
DATA_DIR=/path/to/CARGO \
LOGS_DIR=./logs/cargo_baseline \
GPU=0 SEED=1 \
AERIAL_EPS=0.40 GROUND_EPS=0.40 ALL_EPS=0.40 \
bash scripts/train.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/train_cargo.py \
  --data-dir /path/to/CARGO \
  --logs-dir ./logs/cargo_baseline \
  --stage1-log-name cargo_pclhd_channel_off_s1_seed1 \
  --stage2-log-name cargo_pclhd_cmhard_channel_off_s2_seed1 \
  --epochs 50 --iters 400 --eval-step 5 \
  --aerial-eps 0.40 --ground-eps 0.40 --all-eps 0.40 \
  --cluster-collapse-ratio 0.50 \
  --k1 30 --k2 6 --batch-size 64 --test-batch 64 \
  --height 288 --width 144 --num-instances 16 \
  --lr 0.00035 --weight-decay 0.0005 --step-size 20 \
  --temp 0.05 --momentum 0.2 --seed 1 \
  --workers 8 --print-freq 50 --debug-nonfinite
```

Stage 2 automatically loads the fixed-last Stage 1 checkpoint. Each stage
writes `checkpoint.pth.tar` at its final epoch. The code intentionally does
not select `model_best` from test Rank-1 or mAP.

To run Stage 2 alone:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python cargo_baseline/train_cargo.py \
  --stage2-only \
  --stage1-checkpoint /path/to/stage1/checkpoint.pth.tar \
  --data-dir /path/to/CARGO \
  --logs-dir ./logs/cargo_stage2 \
  --stage2-log-name cargo_pclhd_cmhard_channel_off_s2_seed1 \
  --epochs 50 --iters 400 \
  --aerial-eps 0.40 --ground-eps 0.40 --all-eps 0.40 \
  --seed 1 --debug-nonfinite
```

## Evaluation

```bash
CHECKPOINT=/path/to/checkpoint.pth.tar \
DATA_DIR=/path/to/CARGO \
GPU=0 OUTPUT_JSON=./evaluation.json \
bash scripts/evaluate.sh
```

The evaluator extracts original and horizontally flipped features, averages
them, performs L2 normalization, and ranks by inner product. It reports
Rank-1/5/10/20, mAP, and mINP for:

- `all`: full query and gallery;
- `aa`: aerial query to aerial gallery;
- `gg`: ground query to ground gallery;
- `ag`: official two-domain aerial-ground setting;
- `g2ag`: custom ground-query to complete aerial+ground gallery setting.

Physical camera IDs are preserved for `g2ag`, and same-person/same-camera
items are removed. The custom protocol must be identified as such in papers;
do not present it as an official CARGO protocol without an external protocol
definition.

## Diagnose a legacy Stage1 checkpoint

The read-only diagnostic entry validates the protocol with an identity oracle,
strictly checks an old `arch=agw` checkpoint against the original source tree,
and inspects feature collapse and legacy direct-cubic GeM numerics on a fixed
test subset. It runs the legacy model in an isolated process, never falls back
to the stable AGW, and does not start training or download weights.

```bash
python cargo_baseline/diagnose_stage1.py \
  --legacy-code-dir /path/to/original/PCLHD_repro_release \
  --checkpoint /path/to/stage1/checkpoint.pth.tar \
  --data-dir /path/to/CARGO \
  --device cuda:0 \
  --output-dir ./reports/cargo_stage1_diagnosis
```

See `cargo_baseline/diagnostics/README.md` for the server command, output
contract, optional log parser, fixed-subset policy, and initialization
comparison requirements.

## Verification before publishing

```bash
python -m unittest discover -s tests -v
python verify_release.py
```

`verify_release.py` compiles every Python source file, rejects private absolute
paths and checkpoint artifacts, and writes `MANIFEST_SHA256.json`.

## Reproducibility notes

- Report the exact dataset release, data split, seed, thresholds, checkpoint
  epoch, GPU/software environment, and whether each evaluation protocol is
  official or custom.
- Run multiple seeds and report mean and standard deviation before making
  statistical claims.
- Do not use test results to select DBSCAN thresholds, epochs, or checkpoints.
- The full ALL Jaccard distance can be memory intensive; monitor host memory
  before starting formal training.

## License and citation

The retained upstream code is distributed under the MIT license in `LICENSE`.
See `THIRD_PARTY_NOTICES.md` for attribution and dataset exclusions. Cite the
original PCLHD work and CARGO dataset paper when using this release. Add the
citation for your own method or paper to this section before publishing the
repository.
