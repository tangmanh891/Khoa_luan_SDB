# AutoShotV2 Thesis

Thư mục này chứa snapshot khóa luận LaTeX và slide bảo vệ đã được nhập vào
monorepo ngày 6/6/2026.

## Cấu Trúc

- `main.tex`, `Chapter*/`, `Appendix/`, `Title/`, `References/`: source LaTeX.
- `images/`: ảnh được dùng bởi khóa luận và slide.
- `generated/`: macro, hàng bảng và dữ liệu slide sinh từ kết quả thực nghiệm.
- `slides/build_slides.py`: generator PowerPoint 16:9, 27 slide.
- `build/`: artifact build cục bộ, không track.
- `releases/`: một PDF khóa luận và một PowerPoint bảo vệ chuẩn được track.

## Nguồn Số Liệu

Không sửa trực tiếp các file trong `generated/`. Chạy từ thư mục gốc:

```powershell
python scripts/sync_experimental_results.py --write
python scripts/sync_experimental_results.py --check
```

Script đọc JSON deploy, ablation, calibration và
`reports/literature_results.json`, sau đó tạo:

- `reports/experimental_results.json`
- `reports/experimental_results_summary.md`
- `publications/thesis/generated/experiment_macros.tex`
- `publications/thesis/generated/experiment_tables.tex`
- `publications/thesis/generated/slide_results.json`

## Build

Yêu cầu:

- MiKTeX với `pdflatex` và `biber`.
- Python 3.10 trở lên.
- Dependency trong `publications/thesis/requirements.txt`.

```powershell
pip install -r publications/thesis/requirements.txt

# Build vào publications/thesis/build/
.\scripts\build_thesis.ps1 -Target all

# Build và cập nhật hai file chuẩn trong publications/thesis/releases/
.\scripts\build_thesis.ps1 -Target all -Release
```

Quy trình PDF dùng `pdflatex`, `biber`, rồi `pdflatex` hai lần. Không dùng
`latexmk` vì cài đặt MiKTeX hiện tại cần Perl.

## Chính Sách Release

Chỉ hai file nhị phân sau được giữ trong Git:

- `releases/AutoShotV2_Thesis.pdf`
- `releases/AutoShotV2_Defense.pptx`

Các file PDF/PPTX cũ, artifact LaTeX và nội dung trong `build/` không được đưa
vào Git.
