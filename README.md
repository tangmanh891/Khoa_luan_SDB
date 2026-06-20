# AutoShotV2 Phase2 Report Bundle

Repo này chứa **source code** cho AutoShotV2 (train, inference/evaluation, ablation, calibration) cùng các report kết quả. Theo `.gitignore`, **git chỉ track source + report**; các artifact nặng (checkpoint, cache, logit cache, dataset) **không nằm trong git** mà được phân phối riêng (Kaggle dataset / Drive zip). Xem `docs/ARTIFACTS_MANIFEST.md` để biết artifact nào cần cho việc tái lập từng con số. Source được ghép từ `archive1/autoshotv2`, `archive2/autoshotv2` và `archive2/inference`.

> Lưu ý: thư mục làm việc local có thể chứa cả artifact (~8.7 GB) như một "bundle", nhưng bản trên Git
> là **source-only**. Đừng giả định người clone có sẵn checkpoint/cache — họ cần lấy artifact theo manifest.

## Mục Tiêu

AutoShotV2 dùng để phát hiện ranh giới cảnh quay trong video ngắn. Bài toán là Shot Boundary Detection: với một video đầu vào, mô hình dự đoán tại frame nào có chuyển cảnh, sau đó chuyển chuỗi dự đoán thành danh sách scene `[start_frame, end_frame]`.

Bundle này tập trung vào phase2 fine-tuning/evaluation:

- Dùng checkpoint AutoShot gốc `ckpt_0_200_0.pth` làm backbone.
- Train thêm classification head trên dữ liệu Shot train + ClipShots train.
- Cache feature/logits để resume được trên Kaggle và tránh chạy lại từ đầu.
- Tối ưu post-process bằng temperature scaling, Gaussian smoothing và threshold sweep.
- Đánh giá trên SHOT test, ClipShots test và BBC.

## Mô Hình

Model chính nằm trong:

- `src/autoshotv2/model/supernet.py`
- `src/autoshotv2/model/linear.py`
- `src/autoshotv2/train_phase2.py`

Kiến trúc dựa trên AutoShot/TransNetV2-style supernet, kết hợp 3D ConvNet và Transformer blocks để xử lý chuỗi frame video. Input video được decode bằng `ffmpeg`, resize về `48x27`, chia thành batch 100 frame với padding đầu/cuối, sau đó mô hình trả về logits theo frame.

Phase2 bổ sung head phân loại:

- `ClassificationHead`: một hidden layer `1024` chiều, dropout và hai output heads.
- Loss chính: focal loss cho boundary/one-hot targets.
- Optimizer: `ManualAdam`, viết thủ công để tránh lỗi import `torch._dynamo/onnx` trong một số môi trường Kaggle.
- Checkpoint cuối: `ckpt_phase2_shot_clipshots_best.pth`.

## Cải Tiến Trong Bản Này

- Train phase2 trên tập kết hợp `Shot train + ClipShots train` thay vì chỉ dùng checkpoint gốc.
- Có resume state: `phase2_shot_clipshots_resume.pt`.
- Có checkpoint từng epoch trong `phase2_shot_clipshots_checkpoints/`.
- Có sample cache lớn `shot_clipshots_phase2_sample_cache.pkl` và thư mục `.parts/` để resume khi Kaggle bị ngắt.
- Có eval cache `eval_cache_shot_clipshots/` để không phải chạy inference lại khi chỉ cần đánh giá.
- Có notebook Kaggle với cấu hình P100/T4, timeout an toàn và cơ chế resume.
- Có inference folder riêng cho SHOT, ClipShots và BBC.
- Có smoke test artifacts để kiểm tra nhanh pipeline.

## Dataset

### SHOT / AutoShot Dataset

SHOT là dataset video ngắn được giới thiệu trong paper AutoShot CVPRW 2023. Dataset gốc có 853 video ngắn và 11,606 annotation shot, trong đó test split có 200 video với 2,716 boundary annotation chất lượng cao.

Trong bundle này:

- `DatasetShot/train` được dùng cho train/validation.
- `DatasetShot/test` được giữ riêng để final test.
- Ground truth SHOT test đã có trong `gt_scenes_dict_baseline_v2.pickle`.

### ClipShots

ClipShots được dùng để tăng độ đa dạng dữ liệu khi train phase2:

- `ClipShots/train` được đưa vào training pool.
- `ClipShots/test` không đưa vào train, dùng cho evaluation riêng.
- Ground truth inference nằm trong `inference/clipshots_test_gt_scenes.pickle`.

### BBC

BBC được dùng để kiểm tra khả năng generalize trên dataset khác:

- Ground truth nằm trong `inference/bbc_shots_gt_scenes.pickle`.
- Kết quả đã chạy nằm trong `inference/bbc_shot_inference_results.json`.

## Cấu Trúc Quan Trọng

Các phần **được Git track** (source-only):

Source nằm trong package `autoshotv2` (cài bằng `pip install -e .`). Các phần **được Git track**:

```text
.
├── pyproject.toml                              # Cài package: pip install -e .
├── requirements.txt
├── src/autoshotv2/
│   ├── train_phase2.py                         # Train phase2 head (python -m autoshotv2.train_phase2)
│   ├── prepare_metadata.py                     # Chuẩn bị metadata train/val
│   ├── eval.py                                 # Inference/eval (python -m autoshotv2.eval)
│   ├── ablation.py                             # Runner ablation có kiểm soát
│   ├── postprocess_calibration.py             # Calibration hậu xử lý (K-fold CV, không train lại)
│   ├── utils.py                                # Evaluation + scene conversion utils
│   ├── model/
│   │   ├── supernet.py                         # Model definition (TransNetV2Supernet)
│   │   └── linear.py                           # Linear_ layer dùng cho head
├── scripts/
│   └── run_ablation_experiments.py            # Thin CLI -> autoshotv2.ablation
├── docs/
│   ├── MERGE_NOTES.md                          # Nguồn gốc bundle
│   ├── KAGGLE_AUTOSHOT_PHASE2.md               # Ghi chú chạy Kaggle
│   ├── INFERENCE.md                            # Ghi chú inference/eval
│   └── ARTIFACTS_MANIFEST.md                   # Artifact nào tái lập số nào
├── apps/
│   └── web/                                    # FastAPI backend + React/Vite frontend
├── artifacts/
│   ├── models/                                 # Checkpoint deploy cục bộ
│   └── experiments/                            # Cache và output thực nghiệm ngoài Git
├── data/                                       # Dataset ngoài Git
├── research/
│   └── notebooks/                              # 4 notebook Kaggle (train/resume/smoke)
├── reports/                                    # Ablation + calibration analysis
│   └── deploy_results/                         # Result JSON deploy (SHOT/ClipShots/BBC)
├── publications/
│   ├── paper/                                  # Bài báo IEEE, source + PDF phát hành
│   └── thesis/                                 # Khóa luận, slide và bản phát hành
└── tests/                                      # Smoke tests (reproduce + edge-case)
```

Các nhóm **artifact KHÔNG vào Git** (chỉ có ở bundle local / phân phối riêng), liệt kê trong manifest:
`ckpt_*.pth`, `*_resume.pt`, `*_checkpoints/`, `*_sample_cache.pkl(.parts/)`, `eval_cache_*/`,
`artifacts/experiments/`, `artifacts/models/`, `data/`.

## Kết Quả Đã Chạy

> **Lưu ý về hai pipeline khác nhau (đọc trước khi so số).** Repo có hai bộ số không
> được so sánh trực tiếp:
>
> 1. **Checkpoint deploy** (`ckpt_phase2_shot_clipshots_best.pth`, train trên *full* Shot+ClipShots,
>    `temperature=0.38780`) — đây là bảng "Kết Quả Đã Chạy" ngay dưới.
> 2. **Nghiên cứu ablation có kiểm soát** (control `A1_phase2_bce_onehot` + post-process, chạy trên
>    metadata subset "local", `temperature=0.661785`) — nằm trong
>    `reports/ablation_no_ema_summary_report.md`.
>
> Hai pipeline dùng training data và temperature khác nhau nên F1 không trùng; tuy vậy SHOT gần như
> bằng nhau (0.8545 deploy vs 0.8540 ablation B4) là tín hiệu nhất quán tốt.

Bảng dưới đây là **checkpoint deploy**, đọc trực tiếp từ các file result JSON trong bundle:
`reports/deploy_results/inference_results.json` (SHOT),
`reports/deploy_results/clipshot_test_inference_results.json` (ClipShots),
`reports/deploy_results/bbc_shot_inference_results.json` (BBC).

| Dataset | Videos | Threshold | F1 | Precision | Recall | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHOT test | 200 | 0.10 | 0.8545 | 0.8557 | 0.8533 | 2147 | 362 | 369 |
| ClipShots test | 500 | 0.15 best sweep | 0.7557 | 0.7127 | 0.8041 | 5797 | 2337 | 1412 |
| ClipShots deploy | 500 | 0.10 | 0.7530 | 0.6662 | 0.8659 | 6242 | 3128 | 967 |
| BBC | 11 | 0.10 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 |

Post-process chính (checkpoint deploy):

- temperature: `0.38780461844336944`
- Gaussian sigma: `2.0`
- deploy threshold: `0.1`

> **Mức độ tái lập (hai mức khác nhau — đọc cùng `docs/ARTIFACTS_MANIFEST.md`).** Hai bảng số trong
> repo tái lập tới hai mức khác nhau:
>
> 1. **Bảng deploy (SHOT/ClipShots/BBC ở trên):** chỉ tái lập tới **mức file result JSON** trong
>    `reports/deploy_results/`. Logits của run deploy (`eval_cache_shot_clipshots/`) **không kèm bundle**,
>    nên bảng này không chạy lại được từ logits — chỉ đọc lại JSON đã chốt, hoặc chạy lại inference từ
>    checkpoint deploy + video (cần GPU + ffmpeg).
> 2. **Bảng A0/B4/B5 (calibration/ablation bên dưới):** tái lập tới **mức logits** —
>    `python -m autoshotv2.postprocess_calibration --reproduce` tính lại F1 từ logit cache có sẵn và khớp
>    số (sai số `< 2e-3`).
>
> Nói cách khác: con số có sức nặng nhất (bảng deploy) chỉ verify được ở mức JSON; phần verify được từ
> đầu (logits → F1) là A0/B4/B5 — và các kết luận chính (vai trò post-process, gap ClipShots) đều dựa
> trên nhóm số tái lập-từ-logits này.

Các logits cache sinh ra bảng này (`eval_cache_shot_clipshots/` của run deploy) bị loại khỏi Git theo
`.gitignore` và **không kèm trong bundle**; chỉ các file result JSON ở trên được giữ lại. Bộ logits có
trong bundle để chạy lại post-process là `artifacts/experiments/published_sweeps/`
(baseline AutoShot A0) và `artifacts/experiments/ablation_full/*/eval_cache/`
(các run Phase2 A1–B5). Xem mục "Calibration Hậu Xử Lý" bên dưới.

Train result trong `phase2_shot_clipshots_results.pkl`:

- Validation no-temperature F1: `0.7986`
- Validation with temperature F1: `0.8400`
- Sample count: `365,963`
- One-hot positive rate: `0.0729`
- Boundary positive rate: `0.2169`

## Calibration Hậu Xử Lý (Không Train Lại)

`postprocess_calibration.py` hiệu chỉnh các knob hậu xử lý (temperature, Gaussian sigma, threshold)
**theo từng dataset** trên logits đã cache, **không train lại**. Để tránh "tune trên test", knob được
chọn bằng **5-fold cross-validation** (chọn trên fold calib, đo trên fold giữ lại, micro-average tp/fp/fn).

```powershell
python -m autoshotv2.postprocess_calibration --reproduce   # xác nhận harness khớp số A0/B4/B5 trong bundle
python -m autoshotv2.postprocess_calibration               # chạy CV, sinh reports/postprocess_calibration_*.{json,md}
```

Kết quả CV (F1 trung thực — xem `reports/postprocess_calibration_summary.md`):

| Model | SHOT | ClipShots | BBC |
|---|---:|---:|---:|
| A0 baseline (AutoShot gốc) | 0.8443 | **0.7881** | 0.9538 |
| A1 Phase2 control | **0.8540** | 0.7575 | **0.9631** |
| B5 Phase2 full | 0.8539 | 0.7555 | 0.9584 |

Ba kết luận chính, có kiểm chứng:

- **Calibration là một win rẻ:** ngay cả baseline A0 cũng tăng đáng kể nhờ calibration per-dataset
  (ClipShots 0.7649 → 0.7881, SHOT 0.8405 → 0.8443). Knob được chọn ổn định qua các fold (sigma=2.0
  gần như luôn được chọn → khớp kết luận ablation rằng Gaussian smoothing là thành phần tác động lớn nhất).
- **Phase2 giúp SHOT và BBC** (0.854 / 0.96 so với A0 0.844 / 0.954) nhưng **kém A0 trên ClipShots**
  (0.757 so với 0.788), và **calibration không bù được khoảng cách này**. Đây là bằng chứng định lượng
  cho nhận định "domain generalization sang ClipShots chưa được giải quyết bằng fine-tuning hiện tại".
- **Optimism gap nhỏ** (ceiling tune-trên-test rất sát số CV, ví dụ ClipShots A0 0.7896 vs 0.7881)
  → calibration ổn định, không overfit.

Lưu ý: đây là calibration trên A0 + các run Phase2 ablation (A1/B5) có cache trong bundle, không phải
checkpoint deploy (logits của nó không kèm theo). Mục đích là phương pháp luận calibration + trả lời
câu hỏi A0-vs-Phase2 trên ClipShots, không phải thay thế bảng deploy ở trên.

## Cài Đặt

Cài package ở chế độ editable (khuyến nghị — để `python -m autoshotv2.*` chạy được ở mọi nơi):

```powershell
pip install -r requirements.txt
pip install -e .
```

Ngoài Python packages, cần có `ffmpeg` command-line tool nếu chạy inference trực tiếp từ video.

## Chuẩn Bị Metadata Train

```powershell
python -m autoshotv2.prepare_metadata `
  --shot-root <DatasetShot> `
  --clipshots-root <ClipShots> `
  --out shot_clipshots_trainval.pickle `
  --val-ratio 0.10 `
  --max-val-videos 200
```

Output chính:

- `shot_clipshots_trainval.pickle`

## Train Phase2

Chạy train đầy đủ:

```powershell
python -m autoshotv2.train_phase2 `
  --meta shot_clipshots_trainval.pickle `
  --base-ckpt artifacts/models/ckpt_0_200_0.pth `
  --sample-cache shot_clipshots_phase2_sample_cache.pkl `
  --resume-state artifacts/models/training/phase2_shot_clipshots_resume.pt `
  --checkpoint-dir artifacts/models/training/phase2_shot_clipshots_checkpoints `
  --out-ckpt artifacts/models/training/ckpt_phase2_shot_clipshots_best.pth `
  --results phase2_shot_clipshots_results.pkl `
  --eval-cache-dir eval_cache_shot_clipshots `
  --epochs 20 `
  --batch-size 512 `
  --max-samples-per-video 160 `
  --save-every-videos 25 `
  --save-every-epochs 1 `
  --log-every-batches 100 `
  --stop-after-minutes 660 `
  --no-resume
```

Ghi chú Kaggle chi tiết nằm trong `docs/KAGGLE_AUTOSHOT_PHASE2.md`.

## Inference Và Evaluation

> **Lưu ý:** checkpoint deploy `ckpt_phase2_shot_clipshots_best.pth` **không nằm trong bundle**
> (xem `docs/ARTIFACTS_MANIFEST.md`). `eval` vẫn chạy được từ logit cache mà không cần checkpoint —
> nó chỉ cảnh báo rồi dùng post-process từ tham số CLI. Logit cache deploy mặc định cũng không kèm;
> dùng cache có sẵn trong `artifacts/experiments/published_sweeps/` hoặc
> `artifacts/experiments/ablation_full/`. Ví dụ chạy được ngay:
>
> ```powershell
> python -m autoshotv2.eval `
>   --logits-cache artifacts/experiments/ablation_full/A1_phase2_bce_onehot/eval_cache/shot_test_logits.pkl `
>   --gt artifacts/experiments/published_sweeps/gt_scenes_dict_baseline_v2.pickle `
>   --temperature 0.661785970550883 --sigma 2.0 --threshold 0.1
> ```

Chạy lại evaluation SHOT từ cached logits (cần checkpoint + cache deploy có ở local):

```powershell
python -m autoshotv2.eval
```

Chạy inference từ video mới:

```powershell
python -m autoshotv2.eval `
  --checkpoint ckpt_phase2_shot_clipshots_best.pth `
  --videos-dir <folder_video> `
  --out-logits eval_cache_custom/inference_logits.pkl `
  --results custom_inference_results.json
```

Nếu có ground truth để eval, truyền thêm:

```powershell
python -m autoshotv2.eval `
  --checkpoint ckpt_phase2_shot_clipshots_best.pth `
  --videos-dir <folder_video> `
  --gt <gt_scenes.pickle> `
  --results custom_eval_results.json
```

Tên file video phải khớp key ground truth. Ví dụ `31602670982.mp4` tương ứng key `31602670982`.

## Ablation Tách Biệt Thành Phần

Runner ablation được tách riêng khỏi workflow reproduce chính để so sánh từng thành phần train-time và post-process:

```powershell
python scripts/run_ablation_experiments.py `
  --meta shot_clipshots_trainval_local.pickle `
  --base-ckpt <ckpt_0_200_0.pth> `
  --out-dir artifacts/experiments/ablation_runs `
  --datasets all `
  --seed 42 `
  --epochs 20 `
  --reuse-sample-cache shot_clipshots_phase2_sample_cache_local.pkl
```

Smoke test nhanh:

```powershell
python scripts/run_ablation_experiments.py `
  --meta shot_clipshots_trainval_local.pickle `
  --base-ckpt <ckpt_0_200_0.pth> `
  --out-dir artifacts/experiments/ablation_smoke `
  --datasets shot `
  --epochs 1 `
  --max-train-videos 2 `
  --max-val-videos 1 `
  --max-test-videos 1 `
  --thresholds 0.05,0.1,0.15 `
  --device cpu
```

Output chính:

- `ablation_results.csv` / `ablation_results.json`
- `ablation_summary.md`
- `figures/component_delta_f1.png`
- `figures/dataset_tradeoff.png`
- `figures/precision_recall_delta.png`
- `resolved_meta.pickle` nếu runner cần relocate video path local từ metadata cũ

Các option ablation mới trong train script:

- `--loss bce|focal`
- `--temperature-mode off|auto`

## Artifact Và Dung Lượng

**Git track source-only.** Các artifact nặng (~8.7 GB) không nằm trong Git và được phân phối riêng:

- `shot_clipshots_phase2_sample_cache.pkl` (+ `.parts/`)
- `phase2_shot_clipshots_checkpoints/`, checkpoint `.pth`/`.pt`
- cached logits trong `eval_cache_*/` và `artifacts/experiments/`
- dataset trong `data/`

Chi tiết artifact nào cần để tái lập từng con số nằm trong `docs/ARTIFACTS_MANIFEST.md`. Để chạy lại
calibration/eval mà không train lại, chỉ cần các logit cache liệt kê trong manifest (xem mục
"Calibration Hậu Xử Lý").

## Tests

Smoke test tối thiểu trong `tests/` (chạy được không cần pytest):

```powershell
python tests/test_smoke.py
```

Bao gồm: tái lập 9 mốc A0/B4/B5 từ logit cache (`python -m autoshotv2.postprocess_calibration --reproduce`)
và kiểm tra edge-case `predictions_to_scenes([])`. Test calibration tự bỏ qua nếu không có logit cache.

## Ứng Dụng Web

Monorepo bao gồm web MVP trong `apps/web/`: React/Vite frontend, FastAPI backend,
MongoDB, Cloudinary hoặc local storage, và export JSON/CSV/TXT. Backend gọi
trực tiếp runtime trong package `autoshotv2`; không duy trì bản sao model riêng.

Đặt checkpoint ở `artifacts/models/deploy.pth`, sau đó chạy bản CPU:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Nếu checkpoint chưa có, chế độ `auto` dùng OpenCV baseline. Để chạy NVIDIA GPU:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Hướng dẫn chi tiết: `docs/WEB.md`. Biến môi trường và triển khai:
`docs/WEB_DEPLOY.md`.

## Khóa Luận Và Slide

Source LaTeX, generator slide và hai bản phát hành chuẩn nằm trong
`publications/thesis/`.
Số liệu thực nghiệm cho khóa luận và slide được hợp nhất trong
`reports/experimental_results.json`, sinh từ các JSON kết quả gốc:

```powershell
python scripts/sync_experimental_results.py --write
python scripts/sync_experimental_results.py --check
```

Build PDF và PowerPoint:

```powershell
pip install -r publications/thesis/requirements.txt
.\scripts\build_thesis.ps1 -Target all
```

Thêm `-Release` để cập nhật `publications/thesis/releases/AutoShotV2_Thesis.pdf`
và `publications/thesis/releases/AutoShotV2_Defense.pptx`. Chi tiết xem
`publications/thesis/README.md`.

## Bài Báo IEEE

Source bài báo tự chứa nằm trong `publications/paper/` và dùng chung
`reports/experimental_results.json` với khóa luận. Các bảng kết quả, ablation
và macro headline được sinh bằng cùng lệnh đồng bộ:

```powershell
python scripts/sync_experimental_results.py --write
python scripts/sync_experimental_results.py --check
```

Build và phát hành PDF:

```powershell
.\scripts\build_paper.ps1
.\scripts\build_paper.ps1 -Release
```

Artifact chuẩn: `publications/paper/releases/AutoShotV2_Paper.pdf`. Chi tiết
protocol ClipShots xem `publications/paper/README.md`.

## Nguồn Gốc AutoShot

Repo/paper gốc:

- AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection, CVPRW 2023
- Paper/source: https://github.com/wentaozhu/AutoShot

Citation:

```bibtex
@inproceedings{zhuautoshot,
  title={AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection},
  author={Zhu, Wentao and Huang, Yufang and Xie, Xiufeng and Liu, Wenxian and Deng, Jincan and Zhang, Debing and Wang, Zhangyang and Liu, Ji},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2023}
}
```
