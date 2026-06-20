import numpy as np
import pytest

from autoshotv2.paper_analysis import (
    benchmark_postprocess,
    bootstrap_confidence_intervals,
    calibration_metrics,
    match_transition_intervals,
    match_transition_pairs,
    paired_bootstrap_delta,
    transition_kind_mask,
)


def test_benchmark_postprocess_rejects_empty_logits():
    with pytest.raises(ValueError, match="No logits frames"):
        benchmark_postprocess({})
    with pytest.raises(ValueError, match="No logits frames"):
        benchmark_postprocess({"shot": {}})


def test_bootstrap_confidence_intervals_are_deterministic():
    stats = np.asarray([[3, 1, 2], [4, 2, 1], [2, 0, 3]], dtype=np.int64)
    first = bootstrap_confidence_intervals(stats, n_samples=500, seed=42)
    second = bootstrap_confidence_intervals(stats, n_samples=500, seed=42)

    assert first == second
    assert first["point_counts"] == {"tp": 9, "fp": 3, "fn": 6}
    assert first["metrics"]["f1"]["ci95_low"] <= first["metrics"]["f1"]["value"]
    assert first["metrics"]["f1"]["ci95_high"] >= first["metrics"]["f1"]["value"]


def test_calibration_metrics_include_probability_one_in_last_bin():
    logits = np.asarray([-100.0, 0.0, 100.0])
    labels = np.asarray([0.0, 0.0, 1.0])
    result = calibration_metrics(logits, labels, temperature=1.0, n_bins=2)

    assert result["frames"] == 3
    assert sum(item["count"] for item in result["bins"]) == 3
    assert result["bins"][-1]["count"] == 2
    assert 0.0 <= result["ece"] <= 1.0
    assert sum(item["count"] for item in result["adaptive_bins"]) == 3
    assert 0.0 <= result["adaptive_ece"] <= 1.0
    assert 0.0 <= result["class_balanced_ece"] <= 1.0


def test_paired_bootstrap_delta_detects_identical_and_improved_counts():
    baseline = np.asarray([[3, 2, 2], [4, 2, 1], [2, 1, 3]], dtype=np.int64)
    identical = paired_bootstrap_delta(baseline, baseline, n_samples=500, seed=42)
    improved = paired_bootstrap_delta(
        baseline,
        np.asarray([[4, 1, 1], [5, 1, 0], [3, 0, 2]], dtype=np.int64),
        n_samples=500,
        seed=42,
    )

    assert identical["delta"] == 0.0
    assert identical["excludes_zero"] is False
    assert improved["delta"] > 0.0
    assert improved["ci95_low"] > 0.0


def test_transition_matching_respects_type_and_tolerance():
    transitions = np.asarray([[10, 11], [20, 25]], dtype=np.int64)
    assert transition_kind_mask(transitions, "cut").tolist() == [True, False]
    assert transition_kind_mask(transitions, "gradual").tolist() == [False, True]

    assert match_transition_intervals(
        np.asarray([[10, 11]]),
        np.asarray([[12, 13]]),
        tolerance=2,
    ) == (1, 0, 0)
    assert match_transition_intervals(
        np.asarray([[10, 11]]),
        np.asarray([[14, 15]]),
        tolerance=2,
    ) == (0, 1, 1)
    assert match_transition_pairs(
        np.asarray([[10, 11], [20, 25]]),
        np.asarray([[12, 13], [21, 22]]),
        tolerance=2,
    ) == [(0, 0), (1, 1)]
