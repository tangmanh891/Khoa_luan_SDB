# Experiment Rerun Implementation Summary

This note records the current implementation status for the experiment rerun
plan intended to strengthen the paper.

## Implemented

- Added `scripts/regenerate_deploy_study.py` to regenerate or reuse
  deploy-checkpoint logits.
- Added strict split-coverage validation before bootstrap, calibration, per-video
  CSV, and ClipShots error-analysis artifacts are written.
- Added per-dataset overrides for videos and logits caches:
  `--shot-videos`, `--bbc-videos`, `--clipshots-videos`,
  `--shot-logits-cache`, `--bbc-logits-cache`, and
  `--clipshots-logits-cache`.
- Added `--coverage-only` to audit prerequisites without launching expensive
  video inference.
- Added per-video `--out-logits` checkpoint/resume support in
  `autoshotv2.eval` so interrupted video inference can continue from the
  existing logits file.
- Updated `docs/JOURNAL_REPRODUCIBILITY.md` with the deploy-regeneration
  workflow and current data-source coverage.
- Regenerated deploy-checkpoint logits for SHOT, BBC, and ClipShots from the
  local video sources.

## Current Coverage Audit

Default local sources:

| Dataset | Video source | Matched GT videos | Status |
|---|---|---:|---|
| SHOT | `data/ShotData` | 200/200 | Ready |
| BBC | `data/BBCDataset` | 11/11 | Ready |
| ClipShots | `data/ClipShots/videos/test` | 500/500 | Ready |

Additional SHOT checks:

| Source | Matched GT videos | Status |
|---|---:|---|
| `data/ShotData/video_download/video_download` | 131/200 | Incomplete |
| `data/ShotData/original_videos/original_videos` | 33/200 | Incomplete |

## Regenerated Deploy Results

| Dataset | F1 | Precision | Recall | Videos |
|---|---:|---:|---:|---:|
| SHOT | 0.8546 | 0.8554 | 0.8537 | 200 |
| BBC | 0.9656 | 0.9750 | 0.9564 | 11 |
| ClipShots | 0.7529 | 0.6661 | 0.8657 | 500 |

The combined analysis is written to:

- `reports/deploy_regen_analysis_results.json`
- `reports/deploy_regen_analysis_summary.md`
- `reports/deploy_regen_per_video.csv`
- `reports/deploy_regen_clipshots_top_errors.csv`

## Maintenance Commands

Audit local videos and regenerated logits:

```powershell
python scripts/regenerate_deploy_study.py --coverage-only --datasets shot,bbc,clipshots
```

Regenerate deploy logits from complete local videos, recomputing from scratch:

```powershell
python scripts/regenerate_deploy_study.py `
  --datasets shot,bbc,clipshots `
  --device cuda `
  --force-inference
```
