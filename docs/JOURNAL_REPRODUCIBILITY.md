# Journal Reproducibility Workflow

This workflow separates data sampling, model training, validation-only protocol
selection, test evaluation, and artifact packaging.

## Required inputs

- `artifacts/models/ckpt_0_200_0.pth`: original AutoShot checkpoint.
- `artifacts/experiments/ablation_full/resolved_meta.pickle`: resolved train,
  validation, and SHOT test metadata.
- SHOT, BBC, and ClipShots videos and ground truth under the paths used by
  `scripts/run_journal_study.py`.

The current workspace includes the original AutoShot checkpoint and the
deployment checkpoint under `artifacts/models/`. If these files are absent in
a fresh checkout, restore them from the release artifact before rerunning
training or deploy inference.

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

## Deploy-checkpoint regeneration

Before rerunning the deployed checkpoint, audit whether local videos or cached
logits cover the exact evaluation ground-truth split:

```powershell
python scripts/regenerate_deploy_study.py --coverage-only --datasets shot,bbc,clipshots
```

The script writes:

- `reports/deploy_regen_coverage_report.json`
- `reports/deploy_regen_coverage_report.md`

Do not run or report regenerated deploy results unless the relevant source
matches every ground-truth video. In the current workspace, the default SHOT
source under `data/ShotData` covers the 200-video SHOT test split, while
`data/ShotDataset` covers only 33/200. The BBC and ClipShots default video
sources also cover their test splits. Older SHOT subdirectories remain
incomplete (`data/ShotData/video_download/video_download` covers 131/200 and
`data/ShotData/original_videos/original_videos` covers 33/200).

To regenerate from a valid video source:

```powershell
python scripts/regenerate_deploy_study.py --datasets shot,bbc,clipshots --device cuda
```

During video inference, `autoshotv2.eval` checkpoints `--out-logits` after
each video and resumes from that file by default. Use
`--force-inference` from `scripts/regenerate_deploy_study.py` only when you
intend to recompute a dataset from scratch.

To analyze an already regenerated logits cache without rerunning video
inference:

```powershell
python scripts/regenerate_deploy_study.py `
  --datasets shot `
  --shot-logits-cache path\to\eval_cache_shot_clipshots\shot_test_logits.pkl
```

The script fails by default if logits do not cover the full ground-truth split.
Use `--allow-partial` only for debugging, never for paper tables.

The acceptance condition for replacing the current deploy result-JSON tier is:

- coverage audit reports SHOT 200/200, BBC 11/11, and ClipShots 500/500;
- `reports/deploy_regen_analysis_results.json` exists and lists zero missing
  predictions for every reported dataset;
- `reports/deploy_regen_analysis_summary.md` reports the deploy metrics,
  bootstrap intervals, calibration metrics, and ClipShots transition analysis
  from deploy-checkpoint logits.

The current workspace satisfies these conditions with regenerated deploy
logits in `artifacts/experiments/deploy_regen/`.

When running from a video directory, `scripts/regenerate_deploy_study.py`
passes `--filter-to-gt` to the evaluator so that only files whose stems appear
in the selected ground-truth split are processed. This matters for SHOT because
`data/ShotData` contains 854 videos while the journal test split contains 200.

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

If deploy-regeneration reports exist, they are included in the checksum
manifest and release ZIP alongside the controlled journal-study artifacts.

Do not describe the three-seed study as complete until
`artifacts/experiments/journal_study/journal_results.json` contains all three
seeds for all three test datasets.
