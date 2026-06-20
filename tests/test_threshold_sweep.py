import numpy as np
import pytest

import scripts.sweep_dataset_thresholds as sweep


def test_parse_datasets_defaults_and_rejects_unknown():
    assert sweep.parse_datasets("") == ["bbc", "clipshots"]
    assert sweep.parse_datasets("BBC, clipshots") == ["bbc", "clipshots"]

    with pytest.raises(ValueError, match="Unknown dataset"):
        sweep.parse_datasets("shot")


def test_fine_thresholds_are_inclusive_and_validated():
    assert sweep.fine_thresholds(0.02, 0.023, 0.001) == [0.02, 0.021, 0.022, 0.023]

    with pytest.raises(ValueError, match="fine-step"):
        sweep.fine_thresholds(0.02, 0.20, 0)
    with pytest.raises(ValueError, match="fine-max"):
        sweep.fine_thresholds(0.20, 0.02, 0.001)


def test_sweep_thresholds_selects_best_f1():
    predictions = {
        "true_boundary": np.array([[0.0], [0.4], [0.4], [0.0], [0.0]], dtype=np.float32),
        "no_boundary": np.array([[0.0], [0.2], [0.2], [0.0], [0.0]], dtype=np.float32),
    }
    ground_truth = {
        "true_boundary": np.array([[0, 1], [3, 4]], dtype=np.int64),
        "no_boundary": np.array([[0, 4]], dtype=np.int64),
    }

    rows = sweep.sweep_thresholds(predictions, ground_truth, [0.1, 0.3, 0.5])
    best = sweep.best_threshold(rows)

    assert best["threshold"] == 0.3
    assert best["f1"] == 1.0
    assert best["tp"] == 1
    assert best["fp"] == 0
    assert best["fn"] == 0


def test_require_full_coverage_rejects_missing_predictions():
    coverage = {
        "matched_videos": 1,
        "gt_videos": 2,
        "missing_prediction_count": 1,
        "missing_prediction_keys": ["missing"],
    }

    with pytest.raises(RuntimeError, match="BBC predictions do not cover"):
        sweep.require_full_coverage("bbc", coverage, allow_partial=False)
    sweep.require_full_coverage("bbc", coverage, allow_partial=True)
