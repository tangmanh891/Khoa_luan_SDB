# AutoShotV2 IEEE Paper

Thư mục này chứa snapshot độc lập của bài báo AutoShotV2 theo định dạng
IEEEtran. Source có thể đóng gói trực tiếp để đưa lên Overleaf.

## Dữ liệu thực nghiệm

Không sửa trực tiếp số liệu trong `generated/`. Các macro và hàng bảng được
sinh từ manifest chung:

```powershell
python scripts/sync_experimental_results.py --write
python scripts/sync_experimental_results.py --check
```

Nguồn chính là các JSON trong `reports/`. Breakdown Cut/Gradual được lưu tại
`reports/source_results/clipshots_transition_breakdown.json` cùng thông tin
protocol và provenance.

Quy trình journal, protocol validation-only, ba seed và đóng gói artifact được
mô tả tại `docs/JOURNAL_REPRODUCIBILITY.md`.

## Build

Yêu cầu MiKTeX hoặc TeX Live có `pdflatex` và package `IEEEtran`.

```powershell
.\scripts\build_paper.ps1
```

PDF tạm nằm tại `publications/paper/build/main.pdf`. Để cập nhật bản phát hành chuẩn:

```powershell
.\scripts\build_paper.ps1 -Release
```

Artifact chuẩn: `publications/paper/releases/AutoShotV2_Paper.pdf`.

## Quy ước kết quả

Paper báo cáo theo **hai track**:

- **Track 1 — deploy checkpoint (fixed-threshold deployment analysis)**: cấu hình cố định
  T*=0.3878, sigma=2.0, threshold=0.10 → F1 0.8545 SHOT / 0.9656 BBC /
  0.7529 ClipShots (per-dataset operating point báo cáo riêng). Tier result-JSON:
  chỉ còn result JSON đã checksum, không còn test logits.
- **Track 2 — controlled replication (B4)**: head BCE train lại theo
  frozen validation-only protocol (T=0.6618) với đầy đủ logit cache.
  Mọi CI, paired bootstrap, calibration, ablation và seed study neo
  trên track này (macro `\PaperBFour*`).

