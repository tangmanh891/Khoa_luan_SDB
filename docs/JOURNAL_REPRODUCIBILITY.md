# Journal Reproducibility Workflow

This workflow separates data sampling, model training, validation-only protocol
selection, test evaluation, and artifact packaging.

## Required inputs

- `artifacts/models/ckpt_0_200_0.pth`: original AutoShot checkpoint.
- `artifacts/experiments/ablation_full/resolved_meta.pickle`: resolved train,
  validation, and SHOT test metadata.
- SHOT, BBC, and ClipShots videos and ground truth under the paths used by
  `scripts/run_journal_study.py`.

The original checkpoint is intentionally not tracked in Git. The current
workspace does not contain it, so the three-seed retraining stage cannot be
executed from the repository alone.

## Three-seed study

Validate the commands without training:

```powershell
python scripts/run_journal_study.py --dry-run
```

Run training seeds 42, 43, and 44 with a shared data seed of 42:

```powershell
python scripts/run_journal_study.py
```

Each seed writes:

- `training_data_manifest.json`: exact selected video IDs, source dataset,
  per-video sample counts, metadata hash, and base-checkpoint hash.
- `run_manifest.json`: training configuration, checkpoint hash, validation
  logit-key hash, and elapsed training time.
- `frozen_protocol.json`: five validation folds, search space, selected
  temperature, Gaussian sigma, threshold, and input hashes.
- Test logits and result JSON files for SHOT, BBC, and ClipShots.

The shared feature cache is accepted only when its complete schema-v2 identity
matches. Changing metadata, selected videos, sampling parameters, data seed, or
base checkpoint forces a rebuild.

## Analysis and paper

```powershell
python scripts/analyze_paper_results.py
python scripts/sync_experimental_results.py --write
.\scripts\build_paper.ps1 -Release
```

The checked-in seed-42 protocol manifest is:

```text
reports/source_results/journal_frozen_protocol_seed42.json
```

It selects `T=0.661786`, `sigma=2.0`, and `threshold=0.10` without test
feedback.

## Runtime and release artifacts

Measure decoding, inference, post-processing, end-to-end FPS, and peak GPU
memory:

```powershell
python scripts/benchmark_journal_runtime.py `
  --checkpoint artifacts/experiments/journal_study/seed_42/checkpoint.pth `
  --video path/to/example.mp4 `
  --temperature 0.661786 --sigma 2 --threshold 0.10 `
  --output reports/journal_runtime_seed42.json
```

Build the checksum manifest and release ZIP:

```powershell
python scripts/package_journal_artifacts.py
```

Do not describe the three-seed study as complete until
`artifacts/experiments/journal_study/journal_results.json` contains all three
seeds for all three test datasets.
