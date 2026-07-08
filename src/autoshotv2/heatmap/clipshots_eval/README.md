# ClipShots test evaluation (Kaggle)

Eval checkpoint `ckpt_phase2_best.pth` trên **ClipShots test set** theo chuẩn gốc AutoShot.

## Files

- `eval_clipshots_kaggle.ipynb` — notebook chạy trên Kaggle (inference + eval).
- `build_clipshots_test_gt.py` — script độc lập build GT scenes dict từ `annotations/test.json`.

## Cách chạy trên Kaggle

1. Tạo notebook mới, **Add Data** 3 dataset:
   - `heatmap` (source code autoshotv2)
   - `dataset-clipshots` (ClipShots: annotations + videos)
   - `result-heatmap1` (checkpoint đã train: `ckpt_phase2_best.pth`)
2. Import `eval_clipshots_kaggle.ipynb`.
3. Bật GPU (T4) → Run All.

Notebook tự:
- tìm source / ClipShots root / checkpoint trong `/kaggle/input`,
- build GT scenes dict cho ClipShots test (`transitions_to_scenes`),
- chạy inference trên ClipShots test videos (resume được, logits lưu tăng dần),
- tính **F1 max-sweep** theo `mAP_f1_p_fix_r` (chuẩn tác giả).

## GT scenes dict

ClipShots cho transitions trực tiếp trong `annotations/test.json`
(`{filename: {transitions:[[s,e],...], frame_num: N}}`). GT scenes = phần bù của
transitions, dựng bằng `autoshotv2.phase2_data.transitions_to_scenes` — đúng logic
dùng khi train, nên metric nhất quán với tập Shot.
