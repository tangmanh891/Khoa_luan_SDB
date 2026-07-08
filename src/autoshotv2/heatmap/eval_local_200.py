"""Eval ckpt_phase2_best.pth tren 200 video official GT, dung chuan goc AutoShot.

Chay inference truc tiep tren video (Shot/{ads_game_videos,original_videos,video_download})
roi tinh F1 bang mAP_f1_p_fix_r (copy nguyen tu AutoShot_origin/utils.py, khong can
module 'ffmpeg' cua python).

Cach chay:
    cd autoshotv2_src
    python eval_local_200.py

Inference chay tren CPU (may khong co CUDA) nen co the mat 15-40 phut cho 200 video.
Logits duoc luu tang dan vao eval_local/official_200_logits.pkl -> chay lai se resume.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ── 1. ffmpeg vao PATH (cai qua winget) ──────────────────────────────────────
FFMPEG_DIR = (Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Packages"
              / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
              / "ffmpeg-8.1.2-full_build" / "bin")
if FFMPEG_DIR.exists():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

# ── 2. import package autoshotv2 ──────────────────────────────────────────────
sys.path.insert(0, str(HERE / "src"))
import torch  # noqa: E402
from autoshotv2 import runtime  # noqa: E402
from autoshotv2.eval import run_video_inference  # noqa: E402
from autoshotv2.common import clean_key  # noqa: E402

# ── 3. Ham eval chuan goc AutoShot (copy nguyen tu utils.py) ──────────────────
def predictions_to_scenes(predictions):
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2):
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)
    i = j = tp = fp = fn = 0
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1; i += 1
        elif i == len(gt_trans):
            fp += 1; j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1; j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1; i += 1
        else:
            i += 1; j += 1; tp += 1
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = (p * r * 2) / (p + r) if p + r else 0
    return p, r, f1, (tp, fp, fn)


def mAP_f1_p_fix_r_maxf1(one_hot_pred, gt_scenes):
    """Nhanh fixed_r<=0 cua mAP_f1_p_fix_r (skip_map_miou=True): sweep threshold, lay max F1."""
    thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.255,
                           0.26, 0.265, 0.27, 0.275, 0.28, 0.2833, 0.2867, 0.29, 0.292, 0.294,
                           0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308, 0.31, 0.3133, 0.3167,
                           0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38, 0.39,
                           0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    n = len(thresholds)
    precision = np.zeros(n); recall = np.zeros(n); f1 = np.zeros(n)
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    for k in range(n):
        for name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > thresholds[k]).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[name], pred_scenes)
            tp[k] += tp_; fp[k] += fp_; fn[k] += fn_
        precision[k] = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0
        recall[k] = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0
        f1[k] = (precision[k] * recall[k] * 2) / (precision[k] + recall[k]) if precision[k] + recall[k] else 0
    b = int(np.argmax(f1))
    return f1[b], precision[b], recall[b], thresholds[b]


def main():
    CKPT = HERE / "ckpt_phase2_best.pth"
    SHOT_ROOT = HERE / "Shot"
    GT_PATH = HERE / "gt_scenes_dict_baseline_v2.pickle"
    OUT_LOGITS = HERE / "eval_local" / "official_200_logits.pkl"

    print(f"Checkpoint : {CKPT}")
    print(f"Videos     : {SHOT_ROOT}")
    print(f"GT         : {GT_PATH}")
    print(f"Device     : cpu (cuda={torch.cuda.is_available()})")

    with open(GT_PATH, "rb") as f:
        gt = pickle.load(f)
    include_keys = {clean_key(k) for k in gt}
    print(f"GT videos  : {len(include_keys)}")

    cfg = runtime.load_checkpoint_config(CKPT)
    temperature = float(cfg.get("temperature", runtime.DEFAULT_TEMPERATURE))
    sigma = float(cfg.get("sigma", runtime.DEFAULT_SIGMA))
    print(f"Postprocess: temperature={temperature:.5f} sigma={sigma:.2f}\n")

    # ── Inference (resume-able) ──────────────────────────────────────────────
    logits = run_video_inference(
        CKPT, SHOT_ROOT, OUT_LOGITS, device="cpu",
        include_keys=include_keys, resume=True,
    )

    # ── logits -> probs (1D) ─────────────────────────────────────────────────
    pred = {}
    for key, arr in logits.items():
        probs = runtime.logits_to_probabilities(arr, temperature=temperature, sigma=sigma)
        pred[clean_key(key)] = np.asarray(probs).squeeze()

    gt_clean = {clean_key(k): np.asarray(v) for k, v in gt.items()}
    common = sorted(set(pred) & set(gt_clean))
    pred_c = {k: pred[k] for k in common}
    gt_c = {k: gt_clean[k] for k in common}
    print(f"\nVideos eval: {len(common)}/200")

    f1, p, r, thr = mAP_f1_p_fix_r_maxf1(pred_c, gt_c)
    print("\n=== KET QUA TREN 200 VIDEO OFFICIAL (CHUAN TAC GIA) ===")
    print(f"AutoShotV2 heatmap+smoothing: F1={f1:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.4f}")
    print("\n=== SO SANH VOI PAPER ===")
    print(f"TransNetV2 baseline       : F1=0.7993  P=0.9042  R=0.7162")
    print(f"AutoShot supernet (paper) : F1=0.8405  P=0.8473  R=0.8339")
    print(f"AutoShotV2 (cua chung ta) : F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

    # Luu ket qua
    out = HERE / "eval_local" / "official_200_result.txt"
    out.write_text(
        f"F1={f1:.4f} P={p:.4f} R={r:.4f} thr={thr:.4f}  (200 video, chuan tac gia)\n",
        encoding="utf-8",
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
