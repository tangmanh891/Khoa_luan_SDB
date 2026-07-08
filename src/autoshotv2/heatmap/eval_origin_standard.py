"""Evaluate ckpt_phase2_best.pth theo chuan goc AutoShot (mAP_f1_p_fix_r).

Dung shot_test_logits.pkl (da co san) + gt_scenes_dict_baseline_v2.pickle
de tinh F1 dung cach goc - khong can chay lai inference.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Them utils.py cua AutoShot_origin vao path
ORIGIN_DIR = Path(__file__).parent.parent / 'AutoShot_origin'
sys.path.insert(0, str(ORIGIN_DIR))
from utils import mAP_f1_p_fix_r, predictions_to_scenes

# ── Load du lieu ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

print('Loading shot_test_logits.pkl ...')
with open(HERE / 'shot_test_logits.pkl', 'rb') as f:
    logits_data = pickle.load(f)
logits_dict = logits_data['logits']  # {shot_test:<name>: ndarray (N,1)}

print('Loading gt_scenes_dict_baseline_v2.pickle ...')
with open(ORIGIN_DIR / 'gt_scenes_dict_baseline_v2.pickle', 'rb') as f:
    gt_scenes_dict = pickle.load(f)  # {name: ndarray [[start,end],...]}

print(f'GT videos   : {len(gt_scenes_dict)}')
print(f'Logit videos: {len(logits_dict)}')

# ── Load postprocess config tu ckpt ──────────────────────────────────────────
import torch
ckpt = torch.load(HERE / 'ckpt_phase2_best.pth', map_location='cpu')
temperature = float(ckpt.get('temperature', 1.0))
sigma       = float(ckpt.get('sigma', 0.0))
threshold   = float(ckpt.get('threshold', 0.5))
print(f'\nPostprocess: temperature={temperature:.3f}  sigma={sigma:.2f}  threshold={threshold:.3f}')

# ── Chuyen logits -> one-hot predictions theo format goc ─────────────────────
from scipy.special import expit  # sigmoid
from scipy.ndimage import gaussian_filter1d

pred_dict = {}
matched = 0
missing_in_gt = []
missing_logits = []

for key, logits in logits_dict.items():
    # key dang "shot_test:12580139534" -> name = "12580139534"
    name = key.split(':', 1)[1]

    if name not in gt_scenes_dict:
        missing_in_gt.append(name)
        continue

    # logits shape (N, 1) -> (N,)
    l = logits.squeeze(-1).astype(np.float32)

    # temperature scaling
    l = l / temperature

    # sigmoid
    probs = expit(l)

    # gaussian smoothing
    if sigma > 0:
        probs = gaussian_filter1d(probs, sigma=sigma)

    pred_dict[name] = probs
    matched += 1

# Kiem tra video trong GT nhung khong co logits
for name in gt_scenes_dict:
    if name not in pred_dict:
        missing_logits.append(name)

print(f'\nMatched     : {matched}')
if missing_in_gt:
    print(f'Logits co nhung GT khong co ({len(missing_in_gt)}): {missing_in_gt[:5]}')
if missing_logits:
    print(f'GT co nhung logits khong co ({len(missing_logits)}): {missing_logits[:5]}')

# Chi giu cac video co ca logits lan GT
pred_dict_filtered = {k: v for k, v in pred_dict.items() if k in gt_scenes_dict}
gt_filtered        = {k: v for k, v in gt_scenes_dict.items() if k in pred_dict_filtered}

print(f'\nSo video dung de eval: {len(pred_dict_filtered)}')

# ── Evaluate ──────────────────────────────────────────────────────────────────
print('\n=== KET QUA THEO CHUAN GOC AutoShot ===')

# fixed_r=-1: tim threshold toi uu (max F1)
mAP, f1, precision, recall, thr, miou = mAP_f1_p_fix_r(
    pred_dict_filtered, gt_filtered, fixed_r=-1
)
print(f'Max F1:  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  thr={thr:.4f}  mIoU={miou:.4f}')

# So sanh voi baseline AutoShot origin
print('\n=== SO SANH ===')
print(f'TransNetV2 baseline (paper): F1=0.7993  P=0.9042  R=0.7162')
print(f'AutoShot supernet (paper)  : F1=0.8405  P=0.8473  R=0.8339')
print(f'AutoShotV2 (heatmap+smooth): F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}')
