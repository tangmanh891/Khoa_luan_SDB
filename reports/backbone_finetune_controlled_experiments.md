# Bao cao phan tich fine-tune backbone co kiem soat

Ngay lap: 2026-06-19

## 1. Muc tieu

Muc tieu cua loat thuc nghiem nay la kiem tra viec mo khoa mot phan nho backbone co giup tang F1 tren ClipShots hay khong, dong thoi theo doi BBC nhu guardrail generalization. Cac cau hinh duoc uu tien theo huong bao thu:

- `fc1_0`: chi mo khoa lop gan head nhat.
- Giam so batch/video hoac so video fine-tune de han che drift.
- Giam learning rate de kiem tra co giu BBC tot hon khong.
- Chi xem `layer5` va cac lop sau hon nhu ablation, khong dung lam huong chinh neu BBC giam manh.

## 2. Protocol

- Base checkpoint: `artifacts/models/ckpt_0_200_0.pth`.
- Metadata train/val: `shot_clipshots_trainval.pickle`.
- Tap train: SHOT train sau khi loai overlap voi SHOT test + ClipShots train.
- Validation: 200 video tu metadata train/val de chon temperature va threshold deploy.
- Test: BBC 11 video va ClipShots 500 video.
- Hau xu ly khi eval: temperature scaling + Gaussian smoothing `sigma=2.0`.
- `fixed deploy`: dung temperature va threshold duoc chon tu validation.
- `best sweep`: sweep threshold tren test; chi dung de phan tich tiem nang theo dataset, khong dung lam ket qua deploy chinh.

## 3. Ket qua tong hop

| Cau hinh | BBC fixed F1 | BBC best F1 @thr | ClipShots fixed F1 | ClipShots best F1 @thr | Nhan xet |
|---|---:|---:|---:|---:|---|
| Baseline deploy | 0.9656 | 0.9656 @ 0.10 | 0.7529 | 0.7556 @ 0.15 | Moc so sanh chinh. |
| `fc1_0_v200_b4` | 0.9107 | 0.9385 @ 0.02 | 0.7746 | 0.7746 @ 0.06 | ClipShots tang, BBC giam manh. |
| `layer5_v200_b4` | 0.9084 | 0.9406 @ 0.02 | 0.7738 | 0.7738 @ 0.06 | Mo sau hon khong tot hon `fc1_0`, BBC giam manh. |
| `fc1_0_v200_b2` | 0.9357 | 0.9361 @ 0.02 | 0.7800 | 0.7897 @ 0.10 | Giam batch/video giup can bang hon. |
| `fc1_0_v100_b2` | 0.9341 | 0.9461 @ 0.06 | 0.7962 | 0.7962 @ 0.10 | ClipShots tot nhat hien tai; BBC van giam so voi baseline. |
| `fc1_0_v200_b2_lr5e-6` | 0.9310 | 0.9441 @ 0.06 | chua xong | chua xong | BBC khong cai thien; ClipShots moi cache 306/500 video. |

## 4. Ket qua chi tiet tren BBC

| Cau hinh | Deploy threshold | F1 | Precision | Recall | TP | FP | FN | Best threshold | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline deploy | 0.10 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 | 0.10 | 0.9656 |
| `fc1_0_v200_b4` | 0.06 | 0.9107 | 0.9851 | 0.8468 | 4102 | 62 | 742 | 0.02 | 0.9385 |
| `layer5_v200_b4` | 0.06 | 0.9084 | 0.9843 | 0.8433 | 4085 | 65 | 759 | 0.02 | 0.9406 |
| `fc1_0_v200_b2` | 0.06 | 0.9357 | 0.9767 | 0.8980 | 4350 | 104 | 494 | 0.02 | 0.9361 |
| `fc1_0_v100_b2` | 0.10 | 0.9341 | 0.9811 | 0.8914 | 4318 | 83 | 526 | 0.06 | 0.9461 |
| `fc1_0_v200_b2_lr5e-6` | 0.10 | 0.9310 | 0.9808 | 0.8860 | 4292 | 84 | 552 | 0.06 | 0.9441 |

Nhan xet BBC:

- Baseline van manh nhat ro rang: F1 0.9656.
- Mo `layer5` hoac train `fc1_0` voi 4 batch/video lam recall BBC giam manh, tu 0.9564 xuong khoang 0.843-0.847. Day la dau hieu over-adaptation vao SHOT/ClipShots.
- Giam tu 4 batch/video xuong 2 batch/video giup BBC hoi phuc dang ke: `fc1_0_v200_b2` dat 0.9357.
- `fc1_0_v100_b2` khong giu BBC fixed tot hon `v200_b2`, nhung best sweep BBC cao hon vi threshold 0.06 phu hop hon threshold deploy 0.10.
- Giam learning rate xuong `5e-6` khong giai quyet duoc drift BBC; F1 fixed chi dat 0.9310.

## 5. Ket qua chi tiet tren ClipShots

| Cau hinh | Deploy threshold | F1 | Precision | Recall | TP | FP | FN | Best threshold | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline deploy | 0.10 | 0.7529 | 0.6661 | 0.8657 | 6241 | 3129 | 968 | 0.15 | 0.7556 |
| `fc1_0_v200_b4` | 0.06 | 0.7746 | 0.7515 | 0.7991 | 5761 | 1905 | 1448 | 0.06 | 0.7746 |
| `layer5_v200_b4` | 0.06 | 0.7738 | 0.7458 | 0.8040 | 5796 | 1976 | 1413 | 0.06 | 0.7738 |
| `fc1_0_v200_b2` | 0.06 | 0.7800 | 0.7159 | 0.8566 | 6175 | 2450 | 1034 | 0.10 | 0.7897 |
| `fc1_0_v100_b2` | 0.10 | 0.7962 | 0.7558 | 0.8412 | 6064 | 1959 | 1145 | 0.10 | 0.7962 |
| `fc1_0_v200_b2_lr5e-6` | 0.10 | chua xong | chua xong | chua xong | chua xong | chua xong | chua xong | chua xong | chua xong |

Nhan xet ClipShots:

- Tat ca cau hinh fine-tune hoan tat deu tang F1 tren ClipShots so voi baseline deploy.
- `fc1_0_v100_b2` la cau hinh tot nhat hien tai tren ClipShots: F1 0.7962, tang +0.0433 so voi baseline deploy 0.7529 va tang +0.0406 so voi baseline best 0.7556.
- `fc1_0_v200_b2` cho thay threshold rieng theo dataset co tac dong ro: fixed threshold 0.06 dat 0.7800, nhung best threshold 0.10 dat 0.7897.
- `v100_b2` co threshold validation la 0.10 va day cung la best threshold tren ClipShots, nen fixed deploy va best sweep trung nhau.
- Precision tren ClipShots tang manh o cac cau hinh fine-tune, dac biet `v100_b2` tang tu 0.6661 len 0.7558. Recall giam nhe so voi baseline, nhung giam false positive du de F1 tang.

## 6. Phan tich theo cau hinh

### `fc1_0_v200_b4`

Mo khoa `fc1_0` voi 200 video va 4 batch/video lam ClipShots tang tu 0.7529 len 0.7746. Tuy nhien BBC giam tu 0.9656 xuong 0.9107. Muc giam nay qua lon de chon lam huong chinh. Nguyen nhan kha nang cao la model thich nghi qua manh voi mien SHOT/ClipShots, lam mat kha nang tong quat tren BBC.

### `layer5_v200_b4`

`layer5` khong cai thien hon `fc1_0`. ClipShots dat 0.7738, thap hon nhe `fc1_0_v200_b4`; BBC dat 0.9084, cung thap hon. Ket qua nay ung ho gia thuyet: mo backbone sau hon lam overfit/generalization loss manh hon, khong nen dung lam huong chinh.

### `fc1_0_v200_b2`

Giam so batch/video tu 4 xuong 2 la thay doi co ich. BBC tang lai tu 0.9107 len 0.9357, trong khi ClipShots van tang len 0.7800 fixed va 0.7897 best sweep. Day la cau hinh can bang hon so voi `b4`.

### `fc1_0_v100_b2`

Giam so video fine-tune xuong 100 khong giu BBC fixed tot hon `v200_b2`, nhung lai cho ClipShots tot nhat: 0.7962. Dieu nay cho thay so luong update it hon co the giam mot so false positive tren ClipShots va threshold 0.10 phu hop hon. Tuy nhien, BBC van thap hon baseline 0.0315 F1, nen neu bao cao trong paper can trinh bay nhu trade-off, khong nen noi la cai thien dong thoi tren moi dataset.

### `fc1_0_v200_b2_lr5e-6`

Giam learning rate xuong `5e-6` khong giu BBC tot hon. BBC fixed chi dat 0.9310, thap hon `v200_b2` va `v100_b2`. ClipShots eval chua hoan tat; cache hien co 306/500 video tai:

`artifacts/experiments/backbone_finetune/fc1_0_seed42_v200_b2_lr5e-6/deploy_eval_valcal/clipshots_test_logits.pkl`

Do do chua duoc dung de ket luan ClipShots.

## 7. Ket luan

Ket qua ung ho huong fine-tune backbone co kiem soat, nhung chi nen mo khoa phan rat gan head. Mo sau hon (`layer5`) hoac tang so update (`b4`) lam BBC giam manh, cho thay model drift khoi phan bo goc.

Ung vien tot nhat neu muc tieu la tang ClipShots:

- `fc1_0_seed42_v100_b2`
- ClipShots fixed/best F1: 0.7962
- BBC fixed F1: 0.9341

Ung vien bao thu hon neu muon giu BBC nhinh hon mot chut:

- `fc1_0_seed42_v200_b2`
- ClipShots fixed F1: 0.7800, best F1: 0.7897
- BBC fixed F1: 0.9357

Khong nen chon `layer5_v200_b4` hoac `fc1_0_v200_b4` lam huong chinh vi BBC giam qua manh. `lr=5e-6` cung chua cho dau hieu tot tren BBC.

## 8. Khuyen nghi cho buoc tiep theo

1. Dung `fc1_0_v100_b2` lam candidate ClipShots chinh trong phan phan tich, nhung bao cao ro rang la co trade-off BBC.
2. Bao cao hai dong ket qua:
   - fixed deploy threshold: cong bang, threshold chon tu validation.
   - best threshold: chi la phan tich tiem nang theo dataset.
3. Neu can mot cau hinh can bang hon, dung `fc1_0_v200_b2` thay vi `v100_b2`.
4. Khong chay `layer4_layer5` nhu huong chinh. Neu can cho ablation, chi chay ban nho de minh hoa rang mo sau hon lam generalization kem.
5. Chi tiep tuc `lr=3e-6` neu thuc su can dong thuc nghiem learning-rate; ket qua `5e-6` da cho thay giam LR khong tu dong giu BBC.

## 9. Artifact chinh

- `artifacts/experiments/backbone_finetune/fc1_0_seed42_v100_b2/`
- `artifacts/experiments/backbone_finetune/fc1_0_seed42_v200_b2/`
- `artifacts/experiments/backbone_finetune/fc1_0_seed42_v200_b2_lr5e-6/`
- `artifacts/experiments/backbone_finetune/fc1_0_seed42_v200_b4/`
- `artifacts/experiments/backbone_finetune/layer5_seed42_v200_b4/`
- Baseline deploy: `reports/deploy_regen_analysis_results.json`
