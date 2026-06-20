# Artifacts Manifest

Git track **source-only**. Mọi artifact dưới đây phải lấy từ bản bundle local hoặc nguồn phân phối
riêng (Kaggle dataset / Drive zip). Manifest này ghi rõ artifact nào cần để tái lập từng con số.

Quy ước cột "In Git": ❌ = không track (artifact), ✅ = track (source/report).

## 1. Bảng kết quả deploy trong README (checkpoint deploy)

Số: SHOT 0.8545 | ClipShots 0.7557 (best) / 0.7530 (deploy) | BBC 0.9656.

| Artifact | Vai trò | In Git | Có trong bundle? |
|---|---|:--:|:--:|
| `reports/deploy_results/inference_results.json` | Kết quả SHOT (đã chốt) | ✅ | ✅ |
| `reports/deploy_results/clipshot_test_inference_results.json` | Kết quả ClipShots | ✅ | ✅ |
| `reports/deploy_results/bbc_shot_inference_results.json` | Kết quả BBC | ✅ | ✅ |
| `ckpt_phase2_shot_clipshots_best.pth` | Checkpoint deploy | ❌ | có (local) |
| `eval_cache_shot_clipshots/…_logits.pkl` (run deploy) | Logits sinh ra bảng | ❌ | **KHÔNG** |

> ⚠️ Logits của run deploy **không kèm trong bundle**. Vì vậy bảng deploy chỉ tái lập được tới mức
> file result JSON, **không** chạy lại được từ logits. Nếu cần reproduce đầy đủ, phải khôi phục
> `eval_cache_shot_clipshots/` của run deploy (hoặc chạy lại inference từ `ckpt_phase2_shot_clipshots_best.pth`
> trên video — cần GPU + ffmpeg).

## 2. Baseline A0 (AutoShot gốc)

Số: SHOT 0.8405 | ClipShots 0.7649 | BBC 0.9554 (best threshold/dataset).

| Artifact | Vai trò | In Git |
|---|---|:--:|
| `artifacts/experiments/published_sweeps/eval_cache_shot_clipshots/shot_test_logits.pkl` | Probabilities SHOT | ❌ |
| `artifacts/experiments/published_sweeps/eval_cache_clipshots/clipshot_test_logits.pkl` | Probabilities ClipShots | ❌ |
| `artifacts/experiments/published_sweeps/eval_cache_bbc/bbc_test_logits.pkl` | Probabilities BBC | ❌ |
| `artifacts/experiments/published_sweeps/gt_scenes_dict_baseline_v2.pickle` | GT SHOT | ❌ |
| `artifacts/experiments/published_sweeps/clipshots_test_gt_scenes.pickle` | GT ClipShots | ❌ |
| `artifacts/experiments/published_sweeps/bbc_shots_gt_scenes.pickle` | GT BBC | ❌ |

Reproduce: `python -m autoshotv2.postprocess_calibration --reproduce` (các dòng "A0 baseline").

## 3. Ablation B4 / B5 (Phase2 có kiểm soát)

Số: SHOT 0.8540/0.8542 | ClipShots 0.7441/0.7409 | BBC 0.9570/0.9551.

| Artifact | Vai trò | In Git |
|---|---|:--:|
| `artifacts/experiments/ablation_full/A1_phase2_bce_onehot/eval_cache/*_logits.pkl` | Logits control A1 (cho B4) | ❌ |
| `artifacts/experiments/ablation_full/B5_full_candidate/eval_cache/*_logits.pkl` | Logits B5 | ❌ |
| GT: như mục 2 (`artifacts/experiments/published_sweeps/*.pickle`) | GT 3 dataset | ❌ |
| `artifacts/experiments/ablation_full/ablation_results.{csv,json}` | Bảng ablation đầy đủ | ❌ |

Reproduce nhanh: `python -m autoshotv2.postprocess_calibration --reproduce` (dòng B4/B5).
Reproduce đầy đủ matrix: `python scripts/run_ablation_experiments.py …` (xem README mục Ablation).

## 4. Calibration cross-validation (reports/postprocess_calibration_summary.md)

| Artifact | Vai trò | In Git |
|---|---|:--:|
| Logits A0 (mục 2) + A1/B5 (mục 3) + GT | Input calibration | ❌ |
| `reports/postprocess_calibration_results.json` | Kết quả CV (đã chạy) | ✅ |
| `reports/postprocess_calibration_summary.md` | Bảng tóm tắt | ✅ |

Reproduce: `python -m autoshotv2.postprocess_calibration` (cần các logit cache + GT ở mục 2–3).

## 5. Train Phase2 (sinh checkpoint deploy)

| Artifact | Vai trò | In Git |
|---|---|:--:|
| `shot_clipshots_trainval*.pickle` | Metadata train/val | ❌ |
| `shot_clipshots_phase2_sample_cache.pkl` (+ `.parts/`) | Feature/sample cache | ❌ |
| `phase2_shot_clipshots_resume.pt`, `phase2_shot_clipshots_checkpoints/` | Resume + epoch ckpt | ❌ |
| `ckpt_0_200_0.pth` | Checkpoint AutoShot gốc (backbone) | ❌ |

Reproduce: chạy `research/notebooks/` trên Kaggle (cần GPU). Xem `docs/KAGGLE_AUTOSHOT_PHASE2.md`.

## Tóm tắt: cần gì để chạy lại không cần GPU

Chỉ cần 2 nhóm logit cache + GT là chạy được toàn bộ calibration/ablation-eval trên CPU:

- `artifacts/experiments/published_sweeps/` (A0 + GT 3 dataset)
- `artifacts/experiments/ablation_full/{A1_phase2_bce_onehot,B5_full_candidate}/eval_cache/`

Sau đó: `python -m autoshotv2.postprocess_calibration --reproduce` rồi `python -m autoshotv2.postprocess_calibration`.
