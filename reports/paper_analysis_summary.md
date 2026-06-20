# AutoShotV2 Paper Analysis

Controlled headline: A1 BCE one-hot logits with B4 temperature scaling and Gaussian smoothing.

## Bootstrap 95% confidence intervals

| Dataset | F1 | 95% CI | Precision | Recall | Videos |
|---|---:|---:|---:|---:|---:|
| SHOT | 0.8540 | [0.8329, 0.8733] | 0.8617 | 0.8466 | 200 |
| BBC | 0.9570 | [0.9310, 0.9717] | 0.9733 | 0.9412 | 11 |
| CLIPSHOTS | 0.7441 | [0.6997, 0.7841] | 0.6498 | 0.8703 | 500 |

## Validation calibration diagnostics

| Variant | NLL | Brier | ECE | Adaptive ECE | Balanced ECE |
|---|---:|---:|---:|---:|---:|
| Before scaling | 0.013006 | 0.002194 | 0.007149 | 0.007149 | 0.080506 |
| Temperature scaled | 0.009591 | 0.001911 | 0.002214 | 0.002243 | 0.070405 |

The calibration diagnostics use the same validation set that selected the temperature and are
therefore diagnostic rather than an untouched-test estimate.

## Paired F1 delta, selected method minus A1

| Dataset | Delta F1 | 95% CI | Excludes zero |
|---|---:|---:|:---:|
| SHOT | 0.0162 | [0.0093, 0.0236] | yes |
| BBC | -0.0000 | [-0.0033, 0.0041] | no |
| CLIPSHOTS | 0.0457 | [0.0267, 0.0670] | yes |
