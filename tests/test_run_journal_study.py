import json
import pickle
from types import SimpleNamespace

import numpy as np

import scripts.run_journal_study as study


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
