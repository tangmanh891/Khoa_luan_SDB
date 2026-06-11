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

- SHOT và BBC dùng cấu hình deploy cố định.
- ClipShots deploy dùng threshold 0.10; best sweep 0.15 được báo cáo riêng.
- Ablation Phase 2 và EMA toàn model là hai pipeline khác nhau.
- EMA không phải thành phần của checkpoint deploy.
