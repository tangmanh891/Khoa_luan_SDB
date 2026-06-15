import numpy as np

import scripts.clipshots_postprocess_cv as cv


def test_parse_lists_and_modes():
    assert cv.parse_float_list("0.2, 0.1, 0.1") == [0.1, 0.2]
    assert cv.parse_int_list("8,4,8") == [4, 8]
    assert cv.parse_modes("") == ["segment", "peak"]
    assert cv.parse_modes("PEAK") == ["peak"]


def test_peak_mask_applies_nms_to_nearby_lower_peak():
    scores = np.array([0.0, 0.8, 0.0, 0.7, 0.0], dtype=np.float32)

    without_nms = cv.peak_mask(scores, threshold=0.1, min_distance=0)
    with_nms = cv.peak_mask(scores, threshold=0.1, min_distance=2)

    assert without_nms.tolist() == [0, 1, 0, 1, 0]
    assert with_nms.tolist() == [0, 1, 0, 0, 0]


def test_peak_mask_applies_immediate_neighbor_prominence():
    scores = np.array([0.0, 0.20, 0.19, 0.0, 0.50, 0.10], dtype=np.float32)

    mask = cv.peak_mask(scores, threshold=0.1, prominence=0.05)

    assert mask.tolist() == [0, 0, 0, 0, 1, 0]


def test_select_best_param_prefers_precision_then_lower_threshold_on_ties():
    low_threshold = cv.PostprocessParam("segment", 1.0, 0.0, 0.1)
    high_threshold = cv.PostprocessParam("segment", 1.0, 0.0, 0.2)
    high_precision = cv.PostprocessParam("peak", 1.0, 0.0, 0.3)

    counts_by_param = {
        low_threshold: {"a": (5, 1, 1)},
        high_threshold: {"a": (5, 1, 1)},
        high_precision: {"a": (5, 0, 2)},
    }

    selected, metrics = cv.select_best_param(
        [high_threshold, low_threshold, high_precision],
        counts_by_param,
        ["a"],
    )

    assert selected == high_precision
    assert metrics["precision"] == 1.0

    selected_tie, _ = cv.select_best_param(
        [high_threshold, low_threshold],
        counts_by_param,
        ["a"],
    )

    assert selected_tie == low_threshold


def test_cross_validate_selects_on_calibration_and_reports_heldout_counts():
    simple = cv.PostprocessParam("segment", 1.0, 0.0, 0.1)
    strict = cv.PostprocessParam("segment", 1.0, 0.0, 0.2)
    counts_by_param = {
        simple: {
            "a": (1, 0, 0),
            "b": (1, 0, 0),
            "c": (1, 2, 0),
            "d": (1, 2, 0),
        },
        strict: {
            "a": (0, 0, 1),
            "b": (0, 0, 1),
            "c": (1, 0, 0),
            "d": (1, 0, 0),
        },
    }
    folds = [["a", "b"], ["c", "d"]]

    result = cv.cross_validate([simple, strict], counts_by_param, folds)

    assert result["folds"][0]["selected_param"]["threshold"] == 0.2
    assert result["folds"][1]["selected_param"]["threshold"] == 0.1
    assert result["aggregate"]["tp"] == 2
    assert result["aggregate"]["fp"] == 4
    assert result["aggregate"]["fn"] == 2
