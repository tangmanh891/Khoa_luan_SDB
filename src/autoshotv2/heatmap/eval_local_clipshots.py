"""Eval ckpt_phase2_best.pth tren ClipShots test set, chuan goc AutoShot.

Chay inference truc tiep tren ClipShots/videos/test/*.mp4, build GT scenes tu
annotations/test.json (transitions -> scenes), roi tinh F1 bang mAP_f1_p_fix_r
(max-F1 sweep, copy nguyen tu AutoShot_origin/utils.py).

Cach chay:
    cd autoshotv2_src
    python eval_local_clipshots.py

CPU inference 500 video co the mat 1-2 gio. Logits luu tang dan vao
eval_local/clipshots_test_logits.pkl -> chay lai se resume.
"""

import os
import sys
import json
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
from autoshotv2.phase2_data import transitions_to_scenes  # noqa: E402


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
    return tp, fp, fn


def mAP_f1_maxf1(one_hot_pred, gt_scenes):
    thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.255,
                           0.26, 0.265, 0.27, 0.275, 0.28, 0.2833, 0.2867, 0.29, 0.292, 0.294,
                           0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308, 0.31, 0.3133, 0.3167,
                           0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38, 0.39,
                           0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    best = (0.0, 0.0, 0.0, 0.0)
    for thr in thresholds:
        tp = fp = fn = 0
        for name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > thr).astype(np.uint8))
            tp_, fp_, fn_ = evaluate_scenes(gt_scenes[name], pred_scenes)
            tp += tp_; fp += fp_; fn += fn_
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = (p * r * 2) / (p + r) if p + r else 0
        if f1 > best[0]:
            best = (f1, p, r, thr)
    return best


def build_clipshots_test_gt(clipshots_root, video_dir):
    with open(clipshots_root / "annotations" / "test.json", encoding="utf-8") as f:
        annotations = json.load(f)
    list_path = clipshots_root / "video_lists" / "test.txt"
    if list_path.exists():
        with open(list_path, encoding="utf-8") as f:
            listed = [ln.strip() for ln in f if ln.strip()]
    else:
        listed = list(annotations.keys())
    # bo sung file co trong json/video nhung khong nam trong list
    for fn in annotations:
        if fn not in listed:
            listed.append(fn)

    gt = {}
    skip = {"no_ann": 0, "no_video": 0, "no_frames": 0}
    for fn in listed:
        a = annotations.get(fn)
        if a is None:
            skip["no_ann"] += 1; continue
        if not (video_dir / fn).exists():
            skip["no_video"] += 1; continue
        if not a.get("frame_num"):
            skip["no_frames"] += 1; continue
        stem = Path(fn).stem
        gt[stem] = transitions_to_scenes(a.get("transitions", []), int(float(a["frame_num"])))
    return gt, skip


def main():
    CKPT = HERE / "ckpt_phase2_best.pth"
    CLIP_ROOT = HERE / "ClipShots"
    VIDEO_DIR = CLIP_ROOT / "videos" / "test"
    OUT_LOGITS = HERE / "eval_local" / "clipshots_test_logits.pkl"

    print(f"Checkpoint : {CKPT}")
    print(f"Videos     : {VIDEO_DIR}")
    print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    gt, skip = build_clipshots_test_gt(CLIP_ROOT, VIDEO_DIR)
    print(f"ClipShots test GT: {len(gt)} videos  "
          f"(skip no_ann={skip['no_ann']}, no_video={skip['no_video']}, no_frames={skip['no_frames']})")
    include_keys = {clean_key(k) for k in gt}

    cfg = runtime.load_checkpoint_config(CKPT)
    temperature = float(cfg.get("temperature", runtime.DEFAULT_TEMPERATURE))
    sigma = float(cfg.get("sigma", runtime.DEFAULT_SIGMA))
    print(f"Postprocess: temperature={temperature:.5f} sigma={sigma:.2f}\n")

    logits = run_video_inference(
        CKPT, VIDEO_DIR, OUT_LOGITS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        include_keys=include_keys, resume=True,
    )

    pred = {}
    for key, arr in logits.items():
        probs = runtime.logits_to_probabilities(arr, temperature=temperature, sigma=sigma)
        pred[clean_key(key)] = np.asarray(probs).squeeze()

    gt_clean = {clean_key(k): np.asarray(v) for k, v in gt.items()}
    common = sorted(set(pred) & set(gt_clean))
    pred_c = {k: pred[k] for k in common}
    gt_c = {k: gt_clean[k] for k in common}
    print(f"\nVideos eval: {len(common)}/{len(gt)}")

    f1, p, r, thr = mAP_f1_maxf1(pred_c, gt_c)
    print("\n=== KET QUA CLIPSHOTS TEST (CHUAN TAC GIA AutoShot) ===")
    print(f"AutoShotV2 heatmap+smoothing: F1={f1:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.4f}")
    print(f"(eval tren {len(common)} video, max-F1 sweep threshold)")

    out = HERE / "eval_local" / "clipshots_test_result.txt"
    out.write_text(
        f"ClipShots test: F1={f1:.4f} P={p:.4f} R={r:.4f} thr={thr:.4f}  (n={len(common)})\n",
        encoding="utf-8",
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
