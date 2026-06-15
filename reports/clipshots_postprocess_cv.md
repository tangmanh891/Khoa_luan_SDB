# ClipShots Postprocess Cross-Validation

This report tunes post-processing on cached deploy-checkpoint logits using video-level cross-validation. Each fold selects parameters on the other folds and evaluates on held-out videos.

The main deploy protocol is unchanged; full-set best values are oracle diagnostics and should not be used as headline claims.

## Summary

| Setting | F1 | Precision | Recall | TP | FP | FN | Delta F1 vs deploy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed deploy threshold | 0.7529 | 0.6661 | 0.8657 | 6241 | 3129 | 968 | +0.0000 |
| 5-fold CV selected | 0.7715 | 0.7270 | 0.8218 | 5924 | 2224 | 1285 | +0.0186 |
| Locked most-selected CV param on full set | 0.7736 | 0.7471 | 0.8022 | 5783 | 1958 | 1426 | +0.0208 |
| Full-set oracle | 0.7736 | 0.7471 | 0.8022 | 5783 | 1958 | 1426 | +0.0208 |

Conclusion: The cross-validated ClipShots-specific post-process improves over the fixed deploy threshold by more than 0.005 F1. It is a candidate for a clearly labeled dataset-specific setting.

## Selected Fold Parameters

| Fold | Held-out videos | Mode | T | Sigma | Threshold | Min distance | Prominence | Held-out F1 | Precision | Recall |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | segment | 0.5000 | 2.50 | 0.150 | 0 | 0.000 | 0.7973 | 0.7524 | 0.8479 |
| 2 | 100 | segment | 0.5000 | 2.50 | 0.150 | 0 | 0.000 | 0.7144 | 0.7498 | 0.6822 |
| 3 | 100 | segment | 0.5000 | 2.50 | 0.138 | 0 | 0.000 | 0.7893 | 0.7185 | 0.8754 |
| 4 | 100 | segment | 0.5000 | 2.50 | 0.137 | 0 | 0.000 | 0.7576 | 0.6716 | 0.8689 |
| 5 | 100 | segment | 0.5000 | 2.50 | 0.138 | 0 | 0.000 | 0.7983 | 0.7489 | 0.8547 |

## Top Full-Set Parameters

| Rank | Mode | T | Sigma | Threshold | Min distance | Prominence | F1 | Precision | Recall | TP | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | segment | 0.5000 | 2.50 | 0.150 | 0 | 0.000 | 0.7736 | 0.7471 | 0.8022 | 5783 | 1958 | 1426 |
| 2 | segment | 0.5000 | 2.50 | 0.138 | 0 | 0.000 | 0.7735 | 0.7217 | 0.8334 | 6008 | 2317 | 1201 |
| 3 | segment | 0.5000 | 2.50 | 0.137 | 0 | 0.000 | 0.7732 | 0.7199 | 0.8349 | 6019 | 2342 | 1190 |
| 4 | segment | 0.5000 | 2.50 | 0.139 | 0 | 0.000 | 0.7731 | 0.7227 | 0.8310 | 5991 | 2299 | 1218 |
| 5 | segment | 0.5000 | 2.50 | 0.136 | 0 | 0.000 | 0.7729 | 0.7182 | 0.8366 | 6031 | 2366 | 1178 |
| 6 | segment | 0.5000 | 2.50 | 0.135 | 0 | 0.000 | 0.7727 | 0.7166 | 0.8384 | 6044 | 2390 | 1165 |
| 7 | segment | 0.5000 | 2.50 | 0.134 | 0 | 0.000 | 0.7725 | 0.7144 | 0.8409 | 6062 | 2424 | 1147 |
| 8 | peak | 0.5000 | 2.50 | 0.138 | 16 | 0.000 | 0.7681 | 0.7290 | 0.8116 | 5851 | 2175 | 1358 |
| 9 | peak | 0.5000 | 2.50 | 0.137 | 16 | 0.000 | 0.7677 | 0.7270 | 0.8132 | 5862 | 2201 | 1347 |
| 10 | peak | 0.5000 | 2.50 | 0.139 | 16 | 0.000 | 0.7675 | 0.7298 | 0.8093 | 5834 | 2160 | 1375 |
| 11 | peak | 0.5000 | 2.50 | 0.135 | 16 | 0.000 | 0.7673 | 0.7235 | 0.8169 | 5889 | 2251 | 1320 |
| 12 | peak | 0.5000 | 2.50 | 0.136 | 16 | 0.000 | 0.7673 | 0.7251 | 0.8147 | 5873 | 2227 | 1336 |
| 13 | peak | 0.5000 | 2.50 | 0.134 | 16 | 0.000 | 0.7672 | 0.7213 | 0.8194 | 5907 | 2282 | 1302 |
| 14 | peak | 0.5000 | 2.50 | 0.150 | 0 | 0.000 | 0.7672 | 0.7345 | 0.8030 | 5789 | 2093 | 1420 |
| 15 | segment | 0.5000 | 2.00 | 0.150 | 0 | 0.000 | 0.7669 | 0.6980 | 0.8510 | 6135 | 2655 | 1074 |
| 16 | segment | 0.5000 | 2.50 | 0.121 | 0 | 0.000 | 0.7668 | 0.6927 | 0.8588 | 6191 | 2747 | 1018 |
| 17 | peak | 0.5000 | 2.50 | 0.150 | 16 | 0.000 | 0.7665 | 0.7521 | 0.7815 | 5634 | 1857 | 1575 |
| 18 | peak | 0.5000 | 2.50 | 0.138 | 0 | 0.000 | 0.7663 | 0.7082 | 0.8348 | 6018 | 2480 | 1191 |
| 19 | peak | 0.5000 | 2.50 | 0.139 | 0 | 0.000 | 0.7658 | 0.7092 | 0.8323 | 6000 | 2460 | 1209 |
| 20 | segment | 0.5000 | 2.50 | 0.118 | 0 | 0.000 | 0.7658 | 0.6889 | 0.8620 | 6214 | 2806 | 995 |

## Protocol

- Logits: `artifacts/experiments/deploy_regen/clipshots_test_logits.pkl`
- Ground truth: `artifacts/experiments/journal_study/shared/clipshots_test_gt.pkl`
- Folds: 5 video-level folds with seed 42
- Grid size: 630 parameter settings
