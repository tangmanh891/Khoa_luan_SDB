# AutoShotV2-HeatMap — Source archive

Source code + scripts cho cải tiến **heatmap soft labels + asymmetric label smoothing**
trên AutoShotV2 (shot boundary detection).

> **Lưu ý về vị trí:** Thư mục `src/autoshotv2/heatmap/` là bản lưu source cho biến thể
> HeatMap. Hai file `phase2_data.py` và `train_phase2.py` ở đây **là bản đã sửa cho heatmap**
> (thêm `make_heatmap_labels`, `LabelSmoothingBCE`, cờ `--heatmap-sigma/--label-smoothing/--amp`),
> khác với `src/autoshotv2/phase2_data.py` và `src/autoshotv2/train_phase2.py` của gói chính.
> Thư mục này **không phải** package Python (không có `__init__.py`) và không được import bởi
> `autoshotv2`; nó chỉ là source tham chiếu để tái lập biến thể HeatMap. Không thay thế các
> module đã được pin của gói chính bằng các file ở đây (sẽ đổi cache config và phá test
> provenance).

## Cải tiến
- **Heatmap boundary labels** (`phase2_data.make_heatmap_labels`): thay nhãn cứng 0/1 bằng
  nhãn mềm Gaussian (sigma=3.0) quanh frame boundary → bắt gradual transition tốt hơn.
- **Asymmetric label smoothing** (eps=0.1, chỉ trên nhãn dương): giảm overconfidence.
- **AMP fp16** (`train_phase2`): mixed precision trên T4.
- Loại 200 video GT chính thức khỏi train (fix data leakage) qua
  `scripts/prepare_shot_clipshots_trainval_flat.py --official-gt`.

## Cấu hình model (ckpt_phase2_best.pth, val_f1=0.8631)
focal gamma=2.0 alpha=0.6, heatmap-sigma=3.0, label-smoothing=0.1 asymmetric,
manyhot-weight=0.3, lr=7e-6; deploy temperature=0.268, sigma=2.0, threshold=0.15.

## Kết quả (chuẩn gốc AutoShot, mAP_f1_p_fix_r, max-F1 sweep)
| Tập dữ liệu | F1 | Precision | Recall |
|---|---|---|---|
| SHOT (200 video) | 0.8557 | 0.8324 | 0.8804 |
| ClipShots test (500 video) | 0.7560 | 0.6727 | 0.8629 |

So với AutoShotV2 (BCE one-hot): recall trên SHOT tăng 0.8466 → 0.8804 (+3.4đ),
đánh đổi bằng precision giảm — phù hợp khi bỏ sót transition tốn kém.

## Tái lập
- Train (Kaggle T4): `autoshotv2_heatmap_train.ipynb`
- Eval SHOT local: `python eval_local_200.py`
- Eval ClipShots local: `python eval_local_clipshots.py`
- Eval ClipShots Kaggle: `clipshots_eval/eval_clipshots_kaggle.ipynb`

Checkpoint `ckpt_phase2_best.pth` (55MB) không kèm trong archive này (chỉ source);
lấy từ output train Kaggle hoặc `apps/web/backend/models/autoshotv2_heatmap.pth`.
