# Deploy Regeneration Coverage Audit

This report checks whether local videos and cached logits cover each evaluation ground-truth split before rerunning deploy inference.

## Video Sources

| Dataset | Path | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | `data\ShotData\video_download\video_download` | 131 | 200 | 69 | 355 |

## Cached Logits

| Dataset | Source | Matched | GT | Missing | Extra |
|---|---|---:|---:|---:|---:|
| SHOT | out-dir: `artifacts/experiments/deploy_regen/shot_test_logits.pkl` | 33 | 200 | 167 | 167 |
