# Deploy Regeneration Coverage Audit

This report checks whether local videos and cached logits cover each evaluation ground-truth split before rerunning deploy inference.

## Video Sources

| Dataset | Path | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | `data\ShotData` | 200 | 200 | 0 | 654 |

## Cached Logits

| Dataset | Source | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | out-dir: `artifacts/experiments/deploy_regen/shot_test_logits.pkl` | 0 | 200 | 200 | 0 |
