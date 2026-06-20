# Deploy Regeneration Coverage Audit

This report checks whether local videos and cached logits cover each evaluation ground-truth split before rerunning deploy inference.

## Video Sources

| Dataset | Path | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | `data/ShotData` | 200 | 200 | 0 | 654 |
| BBC | `data/BBCDataset` | 11 | 11 | 0 | 0 |
| ClipShots | `data/ClipShots/videos/test` | 500 | 500 | 0 | 0 |

## Cached Logits

| Dataset | Source | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | out-dir: `artifacts/experiments/deploy_regen/shot_test_logits.pkl` | 200 | 200 | 0 | 0 |
| BBC | out-dir: `artifacts/experiments/deploy_regen/bbc_test_logits.pkl` | 11 | 11 | 0 | 0 |
| ClipShots | out-dir: `artifacts/experiments/deploy_regen/clipshots_test_logits.pkl` | 500 | 500 | 0 | 0 |
