# CARGO Stage1 read-only diagnostics

This tool diagnoses an existing legacy `arch=agw` Stage1 checkpoint. It does
not train, alter the checkpoint, or write into the CARGO dataset.

The real model is constructed in a separate Python process. The legacy source
checkout is placed first on `sys.path`, implicit pretrained-weight downloads
are disabled, and the checkpoint must pass `strict=True`. The worker never
falls back to `agw_cargo_stable`. This is necessary because the old AGW uses
direct cubic GeM (`mean(x ** 3) ** (1/3)`), whereas the public CARGO training
entry now exposes a numerically rescaled stable implementation with the same
topology. Topological similarity is not treated as proof of checkpoint
compatibility.

## Dependencies

- Python 3.10 or a version compatible with the original environment
- PyTorch matching the checkpoint environment
- NumPy
- Pillow
- scikit-learn is not required by this diagnosis entry
- CUDA is required only when `--device cuda:N` is selected

No package or weight is downloaded automatically.

## Linux command

```bash
cd /path/to/PCLHD-cargo

python cargo_baseline/diagnose_stage1.py \
  --legacy-code-dir /path/to/original/PCLHD_repro_release \
  --checkpoint /path/to/stage1/checkpoint.pth.tar \
  --data-dir /path/to/CARGO \
  --device cuda:0 \
  --output-dir ./reports/cargo_stage1_diagnosis
```

To parse the old Stage1 log as well, append:

```bash
  --stage1-log /path/to/stage1/1log.txt
```

The initialization comparison is deliberately `SKIPPED` unless
`--imagenet-weights` is supplied. The supplied file must be an exact full
legacy-AGW initialization state that passes `strict=True`; random weights are
never reported as ImageNet initialization.

Use `--stages oracle log` or another subset to run CPU/read-only stages
independently. The default is `--stages all`. Real inference defaults to at
most 32 valid queries and 256 gallery images per AA/GG/AG protocol. Sampling
is deterministic (`--seed 1`) and forces one legal positive for every selected
query before adding distractors.

## Stage1 follow-up diagnosis

Run the source-consistency audit, historical-artifact inventory, and
train-split ImageNet/Stage1 comparison with:

```bash
python cargo_baseline/diagnose_stage1_followup.py \
  --legacy-code-dir /path/to/original/PCLHD_repro_release \
  --checkpoint /path/to/stage1/checkpoint.pth.tar \
  --imagenet-weights /path/to/imagenet_agw_init.pth.tar \
  --data-dir /path/to/CARGO \
  --device cuda:0 \
  --mode audit sample \
  --output-dir ./reports/cargo_stage1_followup
```

The default run creates `pipeline_consistency.json`,
`train_feature_comparison.json`, `pseudo_artifact_inventory.json`, and
`diagnosis_summary.md`. Static source findings use file/line evidence from the
server-side legacy checkout; missing source is reported as `BLOCKED` or
`UNKNOWN`, never inferred from this release.

The full-cluster mode is intentionally double-gated and has no DBSCAN/Jaccard
defaults:

```bash
python cargo_baseline/diagnose_stage1_followup.py \
  --legacy-code-dir /path/to/original/PCLHD_repro_release \
  --checkpoint /path/to/stage1/checkpoint.pth.tar \
  --data-dir /path/to/CARGO --device cuda:0 \
  --mode audit full-cluster --enable-full-reclustering \
  --aerial-eps 0.40 --ground-eps 0.40 \
  --k1 30 --k2 6 --min-samples 4 \
  --output-dir ./reports/cargo_stage1_full_cluster
```

Replace the example values with the configuration verified from the historical
run. The worker processes aerial and ground domains sequentially, checks
available host memory, does not scan eps, and never builds the Stage2 ALL
matrix. `pseudo_label_quality.json` is explicitly marked as checkpoint
reclustering rather than historical epoch labels.
