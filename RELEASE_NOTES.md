# Release notes

## v1.0.0 — 2026-10-08

Initial public, source-only CARGO baseline release.

### Included

- two-stage PCLHD training entry for CARGO;
- Stage 1 `CM` and Stage 2 `CMhard` configuration;
- channel augmentation, dynamic AGVA, and three-domain matching disabled;
- no-EMA-loss Stage 2 with the EMA model update retained;
- ALL-memory second optimizer step;
- overflow-safe GeM implementation without changing the AGW topology;
- independent aerial/ground/ALL DBSCAN thresholds;
- non-finite checks, cluster-health logs, and optional collapse guard;
- CARGO data audit, threshold diagnosis, standalone evaluation, launch scripts,
  release-contract tests, and SHA-256 manifest generation.

### Not included

- CARGO images or annotations;
- ImageNet weights;
- trained checkpoints or formal result files;
- DATM/D-AGVA/tri-view modules;
- private experiment logs or machine-specific paths.

### Validation status

- all Python files pass source compilation;
- package import and command-line parser checks pass in the packaging
  environment;
- four static release-contract tests pass;
- no new full GPU training was run specifically for packaging.

Users should first audit their CARGO extraction and execute the low-cost smoke
test before launching the 50+50 epoch experiment.
