import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np

import scripts.run_journal_study as study


def test_training_command_uses_frozen_protocol_arguments(tmp_path):
    args = SimpleNamespace(
        meta=tmp_path / "meta.pickle",
        base_ckpt=tmp_path / "base.pth",
        epochs=3,
        batch_size=256,
        data_seed=42,
        max_train_videos=400,
        max_val_videos=50,
        force=False,
    )
    run_dir = tmp_path / "seed_43"
    command = study.training_command(
        args, 43, tmp_path / "shared", run_dir, run_dir / "checkpoint.pth"
    )

    assert command[:3] == [sys.executable, "-m", "autoshotv2.train_phase2"]
    pairs = dict(zip(command[3:-1:2], command[4::2]))
    # Frozen training configuration — must never vary across seeds.
    assert pairs["--loss"] == "bce"
    assert pairs["--manyhot-weight"] == "0"
    assert pairs["--boundary-window"] == "0"
    assert pairs["--temperature-mode"] == "off"
    assert pairs["--sigma"] == "0"
    assert pairs["--seed"] == "43"
    assert pairs["--data-seed"] == "42"
    assert pairs["--sample-cache"] == str(tmp_path / "shared" / "sample_cache.pkl")
    assert command[-1] == "--skip-test-eval"

    args.force = True
    forced = study.training_command(
        args, 43, tmp_path / "shared", run_dir, run_dir / "checkpoint.pth"
    )
    assert forced[-2:] == ["--no-resume", "--no-eval-cache"]


def test_materialize_ground_truth_from_canonical_sources(tmp_path, monkeypatch):
    shot_json = tmp_path / "shot.json"
    shot_json.write_text(
        json.dumps({"scenes": {"folder/video.mp4": [[0, 4], [5, 9]]}}),
        encoding="utf-8",
    )

    bbc_annotations = tmp_path / "bbc"
    bbc_annotations.mkdir()
    np.savetxt(
        bbc_annotations / "01_episode.txt",
        np.asarray([[0, 3], [4, 8]], dtype=np.int64),
        fmt="%d",
    )

    clipshots_annotations = tmp_path / "clipshots.json"
    clipshots_annotations.write_text(
        json.dumps(
            {
                "folder/clip.mp4": {
                    "transitions": [[4, 5]],
                }
            }
        ),
        encoding="utf-8",
    )
    reference_logits = tmp_path / "clipshots_logits.pkl"
    with reference_logits.open("wb") as handle:
        pickle.dump(
            {"logits": {"clip": np.zeros((10, 1), dtype=np.float32)}},
            handle,
        )

    monkeypatch.setattr(study, "SHOT_GT_JSON", shot_json)
    monkeypatch.setattr(study, "BBC_ANNOTATIONS", bbc_annotations)
    monkeypatch.setattr(study, "CLIPSHOTS_ANNOTATIONS", clipshots_annotations)
    monkeypatch.setattr(study, "CLIPSHOTS_REFERENCE_LOGITS", reference_logits)

    args = SimpleNamespace(
        shot_gt=tmp_path / "outputs" / "shot.pkl",
        bbc_gt=tmp_path / "other" / "bbc.pkl",
        clipshots_gt=tmp_path / "third" / "clipshots.pkl",
    )
    study.materialize_ground_truth(args)

    with args.shot_gt.open("rb") as handle:
        shot = pickle.load(handle)
    with args.bbc_gt.open("rb") as handle:
        bbc = pickle.load(handle)
    with args.clipshots_gt.open("rb") as handle:
        clipshots = pickle.load(handle)

    assert shot["video"].tolist() == [[0, 4], [5, 9]]
    assert bbc["bbc_01"].tolist() == [[0, 3], [4, 8]]
    assert clipshots["clip"].tolist() == [[0, 4], [5, 9]]
    assert (args.shot_gt.parent / "ground_truth_manifest.json").is_file()
