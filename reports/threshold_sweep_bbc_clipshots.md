# BBC and ClipShots Threshold Sweep

This report sweeps decision thresholds on cached deploy-checkpoint logits. Dataset-specific best thresholds are test-set oracle values and should not replace the fixed deploy threshold in headline reporting.

## Best Thresholds

| Dataset | Sweep | Best threshold | F1 | Precision | Recall | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BBC | coarse | 0.100 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 |
| BBC | fine | 0.089 | 0.9672 | 0.9736 | 0.9608 | 4654 | 126 | 190 |
| ClipShots | coarse | 0.150 | 0.7556 | 0.7126 | 0.8041 | 5797 | 2338 | 1412 |
| ClipShots | fine | 0.135 | 0.7587 | 0.7000 | 0.8281 | 5970 | 2559 | 1239 |

## Deploy Threshold Comparison

| Dataset | Deploy threshold | Deploy F1 | Fine best threshold | Fine best F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|
| BBC | 0.100 | 0.9656 | 0.089 | 0.9672 | +0.0016 |
| ClipShots | 0.100 | 0.7529 | 0.135 | 0.7587 | +0.0058 |

## Fine Sweep Top Thresholds

### BBC

| Rank | Threshold | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.089 | 0.9672 | 0.9736 | 0.9608 |
| 2 | 0.090 | 0.9672 | 0.9738 | 0.9606 |
| 3 | 0.091 | 0.9669 | 0.9740 | 0.9600 |
| 4 | 0.088 | 0.9669 | 0.9730 | 0.9608 |
| 5 | 0.083 | 0.9668 | 0.9718 | 0.9618 |
| 6 | 0.087 | 0.9668 | 0.9724 | 0.9612 |
| 7 | 0.081 | 0.9667 | 0.9710 | 0.9624 |
| 8 | 0.085 | 0.9667 | 0.9722 | 0.9612 |
| 9 | 0.086 | 0.9667 | 0.9722 | 0.9612 |
| 10 | 0.080 | 0.9666 | 0.9708 | 0.9624 |

### ClipShots

| Rank | Threshold | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.135 | 0.7587 | 0.7000 | 0.8281 |
| 2 | 0.137 | 0.7586 | 0.7020 | 0.8251 |
| 3 | 0.136 | 0.7585 | 0.7007 | 0.8266 |
| 4 | 0.118 | 0.7584 | 0.6852 | 0.8492 |
| 5 | 0.121 | 0.7584 | 0.6870 | 0.8463 |
| 6 | 0.138 | 0.7583 | 0.7024 | 0.8240 |
| 7 | 0.139 | 0.7582 | 0.7031 | 0.8227 |
| 8 | 0.120 | 0.7582 | 0.6864 | 0.8469 |
| 9 | 0.122 | 0.7582 | 0.6879 | 0.8446 |
| 10 | 0.134 | 0.7582 | 0.6985 | 0.8291 |
