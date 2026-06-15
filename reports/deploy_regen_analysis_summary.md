# Deploy Checkpoint Regeneration Analysis

This report is generated from regenerated or reused deploy-checkpoint logits.

## Coverage

| Dataset | Matched | GT | Missing | Extra |
|---|---:|---:|---:|---:|
| SHOT | 200 | 200 | 0 | 0 |
| BBC | 11 | 11 | 0 | 0 |
| ClipShots | 500 | 500 | 0 | 0 |

## Deploy Metrics With Bootstrap CI

| Dataset | F1 | 95% CI | Precision | Recall | Videos |
|---|---:|---:|---:|---:|---:|
| SHOT | 0.8546 | [0.8335, 0.8739] | 0.8554 | 0.8537 | 200 |
| BBC | 0.9656 | [0.9454, 0.9788] | 0.9750 | 0.9564 | 11 |
| ClipShots | 0.7529 | [0.7089, 0.7932] | 0.6661 | 0.8657 | 500 |

## Test Calibration After Deploy Temperature

| Dataset | NLL | Brier | ECE | Adaptive ECE | Balanced ECE |
|---|---:|---:|---:|---:|---:|
| SHOT | 0.021143 | 0.004701 | 0.002427 | 0.002666 | 0.130019 |
| BBC | 0.012375 | 0.001602 | 0.000896 | 0.001648 | 0.081124 |
| ClipShots | 0.007115 | 0.001500 | 0.002214 | 0.002215 | 0.155122 |

## ClipShots Recall by Transition Type

| Type | GT | Matched | Missed | Recall |
|---|---:|---:|---:|---:|
| Cut | 4907 | 4430 | 477 | 0.9028 |
| Gradual | 2302 | 1811 | 491 | 0.7867 |
