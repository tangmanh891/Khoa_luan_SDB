import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.regenerate_deploy_study as regen
from autoshotv2 import eval as eval_mod
from autoshotv2.eval import filter_video_files_for_keys, run_video_inference


def test_parse_datasets_defaults_and_rejects_unknown():
    assert regen.parse_datasets("") == list(regen.DATASETS)
    assert regen.parse_datasets("shot, BBC") == ["shot", "bbc"]

    with pytest.raises(ValueError, match="Unknown dataset"):
        regen.parse_datasets("shot,unknown")


def test_coverage_summary_and_full_coverage_guard():
    coverage = regen.coverage_summary(
        {"video_a": np.zeros(2), "extra": np.zeros(2)},
        {"video_a": np.zeros(2), "missing": np.zeros(2)},
    )

    assert coverage["predicted_videos"] == 2
    assert coverage["gt_videos"] == 2
    assert coverage["matched_videos"] == 1
    assert coverage["missing_prediction_keys"] == ["missing"]
    assert coverage["extra_prediction_keys"] == ["extra"]

    with pytest.raises(RuntimeError, match="SHOT logits do not cover"):
        regen.require_full_coverage("shot", coverage, allow_partial=False)
    regen.require_full_coverage("shot", coverage, allow_partial=True)
    regen.require_full_coverage(
        "shot",
        regen.coverage_summary({"video_a": np.zeros(2)}, {"video_a": np.zeros(2)}),
        allow_partial=False,
    )


def test_resource_audit_reports_video_and_logits_coverage(tmp_path, monkeypatch):
    ground_truth_path = tmp_path / "shot_gt.pkl"
    with ground_truth_path.open("wb") as handle:
        pickle.dump(
            {
                "video_a": np.zeros((2, 2), dtype=np.int64),
                "video_b": np.zeros((2, 2), dtype=np.int64),
            },
            handle,
        )

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video_a.mp4").write_bytes(b"not a real video")
    (videos_dir / "extra.mov").write_bytes(b"not a real video")

    logits_path = tmp_path / "shot_logits.pkl"
    with logits_path.open("wb") as handle:
        pickle.dump(
            {
                "logits": {
                    "video_a": np.zeros((3, 1), dtype=np.float32),
                    "extra": np.zeros((3, 1), dtype=np.float32),
                }
            },
            handle,
        )

    monkeypatch.setattr(
        regen,
        "DATASETS",
        {
            "shot": {
                "label": "SHOT",
                "videos": videos_dir,
                "gt": ground_truth_path,
                "logits": "shot_test_logits.pkl",
                "results": "results_shot.json",
            }
        },
    )
    args = SimpleNamespace(out_dir=tmp_path / "out", shot_videos=None, shot_logits_cache=logits_path)
    output_json = tmp_path / "coverage.json"
    output_md = tmp_path / "coverage.md"

    payload = regen.write_resource_audit(args, ["shot"], output_json, output_md)

    assert payload["datasets"]["shot"]["video_source"]["matched_videos"] == 1
    assert payload["datasets"]["shot"]["video_source"]["missing_count"] == 1
    assert payload["datasets"]["shot"]["override_logits"]["matched_videos"] == 1
    assert payload["datasets"]["shot"]["override_logits"]["missing_count"] == 1
    assert json.loads(output_json.read_text(encoding="utf-8")) == payload
    assert "Deploy Regeneration Coverage Audit" in output_md.read_text(encoding="utf-8")


def test_video_filter_keeps_only_ground_truth_keys(tmp_path):
    videos = []
    for name in ("video_b.mp4", "extra.mp4", "video_a.mov"):
        path = tmp_path / name
        path.write_bytes(b"not a real video")
        videos.append(path)

    filtered = filter_video_files_for_keys(videos, {"video_a", "video_b"})

    assert [path.stem for path in filtered] == ["video_a", "video_b"]
    with pytest.raises(FileNotFoundError, match="missing 1 ground-truth videos"):
        filter_video_files_for_keys(videos, {"video_a", "missing"})


def test_video_inference_resumes_existing_logits(tmp_path, monkeypatch):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video_a.mp4").write_bytes(b"not a real video")
    (videos_dir / "video_b.mp4").write_bytes(b"not a real video")
    out_logits = tmp_path / "logits.pkl"
    existing = np.array([1.0], dtype=np.float32)
    with out_logits.open("wb") as handle:
        pickle.dump({"logits": {"video_a": existing}}, handle)

    calls = []
    monkeypatch.setattr(eval_mod, "load_model", lambda checkpoint_path, device: object())

    def fake_predict(model, video_path, device):
        calls.append(video_path.stem)
        return np.array([2.0], dtype=np.float32)

    monkeypatch.setattr(eval_mod, "predict_video_logits", fake_predict)

    logits = run_video_inference(
        tmp_path / "checkpoint.pth",
        videos_dir,
        out_logits,
        "cpu",
        include_keys={"video_a", "video_b"},
    )

    assert calls == ["video_b"]
    assert sorted(logits) == ["video_a", "video_b"]
    with out_logits.open("rb") as handle:
        saved = pickle.load(handle)["logits"]
    assert sorted(saved) == ["video_a", "video_b"]
