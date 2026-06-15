# Bảng Tổng Hợp Kết Quả Thực Nghiệm AutoShotV2

File này được sinh từ `reports/experimental_results.json` bằng
`python scripts/sync_experimental_results.py --write`. Không sửa trực tiếp.

Các nhóm dùng protocol khác nhau; chỉ so sánh trực tiếp trong cùng nhóm.

## Bảng Tất Cả Thực Nghiệm

| Nhóm | Thực nghiệm / protocol | SHOT | BBC | ClipShots | Ghi chú |
|---|---|---:|---:|---:|---|
| Deploy checkpoint | `phase2_deploy_threshold` - Phase2 deploy checkpoint, deploy threshold | 0.8546 | 0.9656 | 0.7529 | Regenerated deploy-checkpoint logits; T=0.3878, sigma=2.0, threshold=0.10. |
| Deploy checkpoint | `phase2_best_sweep` - Phase2 deploy checkpoint, best sweep | 0.8546 | 0.9656 | 0.7556 | ClipShots tốt nhất tại threshold 0.15; SHOT/BBC trùng deploy. |
| Ablation Phase 2 | `A0_autoshot_original` - A0 -- AutoShot gốc | 0.8405 | 0.9554 | 0.7649 | AutoShot gốc, best threshold từng dataset. |
| Ablation Phase 2 | `A1_phase2_bce_onehot` - A1 -- BCE + one-hot | 0.8378 | 0.9570 | 0.6983 | Control Phase2 tối giản. |
| Ablation Phase 2 | `A2_focal_only` - A2 -- Focal only | 0.8378 | 0.9567 | 0.6967 | Chỉ thêm Focal Loss. |
| Ablation Phase 2 | `A3_manyhot_only` - A3 -- Many-hot only | 0.8375 | 0.9563 | 0.7005 | Chỉ thêm nhãn many-hot. |
| Ablation Phase 2 | `P1_gaussian_only` - P1 -- Gaussian only | 0.8432 | 0.9436 | 0.7519 | Tăng ClipShots nhưng giảm BBC. |
| Ablation Phase 2 | `P2_temperature_only` - P2 -- Temperature only | 0.8378 | 0.9570 | 0.6983 | Gần như trùng control. |
| Ablation Phase 2 | `B1_focal_manyhot` - B1 -- Focal + many-hot | 0.8384 | 0.9559 | 0.7006 | Tác động train-time nhỏ. |
| Ablation Phase 2 | `B4_temperature_gaussian` - B4 -- Temperature + Gaussian | 0.8540 | 0.9570 | 0.7441 | Cấu hình hậu xử lý được ưu tiên. |
| Ablation Phase 2 | `B5_full_candidate` - B5 -- Full candidate | 0.8542 | 0.9551 | 0.7409 | Focal + many-hot + hậu xử lý. |
| Calibration CV | `calibration_cv_A0_autoshot_baseline` - A0 -- AutoShot baseline, 5-fold CV | 0.8443 | 0.9538 | 0.7881 | Honest 5-fold CV; baseline mạnh nhất trên ClipShots. |
| Calibration ceiling | `calibration_ceiling_A0_autoshot_baseline` - A0 -- AutoShot baseline, tune trên test | 0.8475 | 0.9561 | 0.7896 | Mức trần lạc quan; không dùng làm deploy honest. |
| Calibration CV | `calibration_cv_A1_phase2_control` - A1 -- Phase2 control, 5-fold CV | 0.8540 | 0.9631 | 0.7575 | Honest 5-fold CV; Phase2 mạnh trên SHOT/BBC. |
| Calibration ceiling | `calibration_ceiling_A1_phase2_control` - A1 -- Phase2 control, tune trên test | 0.8540 | 0.9631 | 0.7599 | Mức trần lạc quan; không dùng làm deploy honest. |
| Calibration CV | `calibration_cv_B5_phase2_full` - B5 -- Phase2 đầy đủ, 5-fold CV | 0.8539 | 0.9584 | 0.7555 | Honest 5-fold CV; Phase2 mạnh trên SHOT/BBC. |
| Calibration ceiling | `calibration_ceiling_B5_phase2_full` - B5 -- Phase2 đầy đủ, tune trên test | 0.8550 | 0.9603 | 0.7607 | Mức trần lạc quan; không dùng làm deploy honest. |

## Kết Quả Checkpoint Deploy

| Dataset | Chế độ | Threshold | F1 | Precision | Recall | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SHOT | deploy | 0.1000 | 0.8546 | 0.8554 | 0.8537 | 2148 | 363 | 368 |
| BBC | deploy | 0.1000 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 |
| ClipShots | deploy | 0.1000 | 0.7529 | 0.6661 | 0.8657 | 6241 | 3129 | 968 |
| SHOT | best_sweep | 0.1000 | 0.8546 | 0.8554 | 0.8537 | 2148 | 363 | 368 |
| BBC | best_sweep | 0.1000 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 |
| ClipShots | best_sweep | 0.1500 | 0.7556 | 0.7126 | 0.8041 | 5797 | 2338 | 1412 |

## Nguồn Và Mức Tái Lập

- `result_json`: tái lập ở mức đọc lại JSON đã chốt hoặc chạy lại từ checkpoint/dataset.
- `logits`: có thể tính lại metric từ logits và ground truth khi artifact tương ứng có sẵn.
- `literature` và `legacy_result`: chỉ dùng trong bảng so sánh, không được coi là run mới.
