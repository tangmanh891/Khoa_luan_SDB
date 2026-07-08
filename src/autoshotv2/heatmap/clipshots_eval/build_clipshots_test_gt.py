"""Build clipshots_test_gt_scenes.pickle tu ClipShots annotations/test.json.

ClipShots (cau truc goc tren Kaggle):
  ClipShots/
  ├── annotations/test.json   {filename: {transitions:[[s,e],...], frame_num: N}}
  ├── video_lists/test.txt    danh sach filename test
  └── videos/test/*.mp4

GT scenes dict cho eval: {stem: ndarray [[scene_start, scene_end], ...]}
dung transitions_to_scenes (giong pipeline train) de chuyen transitions -> scenes.

Cach chay (vd tren Kaggle):
    python build_clipshots_test_gt.py \
        --clipshots-root /kaggle/input/.../ClipShots \
        --out /kaggle/working/clipshots_test_gt_scenes.pickle
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def transitions_to_scenes(transitions, n_frames):
    """Copy nguyen tu autoshotv2.phase2_data (de script chay doc lap)."""
    transitions = np.asarray(transitions, dtype=np.int32)
    if n_frames <= 0:
        return np.asarray([[0, 0]], dtype=np.int32)
    if transitions.size == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)
    transitions = transitions.reshape(-1, 2)
    transitions = transitions[np.argsort(transitions[:, 0])]
    transitions = np.clip(transitions, 0, n_frames - 1)
    transitions = transitions[transitions[:, 0] <= transitions[:, 1]]
    if len(transitions) == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)
    scenes = [[0, int(transitions[0, 0])]]
    for i in range(1, len(transitions)):
        scenes.append([int(transitions[i - 1, 1]), int(transitions[i, 0])])
    scenes.append([int(transitions[-1, 1]), n_frames - 1])
    arr = np.asarray(scenes, dtype=np.int32)
    arr = np.clip(arr, 0, n_frames - 1)
    arr = arr[arr[:, 0] <= arr[:, 1]]
    if len(arr) == 0:
        arr = np.asarray([[0, n_frames - 1]], dtype=np.int32)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clipshots-root", required=True,
                    help="Thu muc ClipShots (chua annotations/, video_lists/, videos/)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.clipshots_root)
    ann_path = root / "annotations" / f"{args.split}.json"
    list_path = root / "video_lists" / f"{args.split}.txt"
    video_dir = root / "videos" / args.split

    with ann_path.open(encoding="utf-8") as f:
        annotations = json.load(f)
    with list_path.open(encoding="utf-8") as f:
        listed = [ln.strip() for ln in f if ln.strip()]

    gt_scenes = {}
    skipped_no_ann = []
    skipped_no_video = []
    skipped_no_frames = []
    for filename in listed:
        ann = annotations.get(filename)
        if ann is None:
            skipped_no_ann.append(filename)
            continue
        if not (video_dir / filename).exists():
            skipped_no_video.append(filename)
            continue
        n_frames = ann.get("frame_num")
        if not n_frames:
            skipped_no_frames.append(filename)
            continue
        # Key theo stem de khop voi video_path.stem trong run_video_inference
        stem = Path(filename).stem
        scenes = transitions_to_scenes(ann.get("transitions", []), int(float(n_frames)))
        gt_scenes[stem] = scenes

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(gt_scenes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"ClipShots {args.split} GT scenes -> {args.out}")
    print(f"  videos in list      : {len(listed)}")
    print(f"  GT scenes built     : {len(gt_scenes)}")
    if skipped_no_ann:
        print(f"  skip (no annotation): {len(skipped_no_ann)}")
    if skipped_no_video:
        print(f"  skip (no video file): {len(skipped_no_video)}")
    if skipped_no_frames:
        print(f"  skip (no frame_num) : {len(skipped_no_frames)}")


if __name__ == "__main__":
    main()
