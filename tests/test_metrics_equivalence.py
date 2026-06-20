"""Bit-exact equivalence between the repo's F1/precision/recall implementations.

Written before consolidating them into autoshotv2.common: proves the scalar
formula in paper_analysis.classification_metrics, postprocess_calibration.f1_pr,
the inline math in train_phase2.evaluate_fixed, and the vectorized form nested
in paper_analysis.paired_bootstrap_delta all agree to the last bit, so the
consolidation cannot change any reported number.

utils.py computes F1 as (p * r * 2) / (p + r) — a different operation order
that can differ in the final bit — and is deliberately NOT covered here nor
consolidated.
"""

import numpy as np

from autoshotv2.paper_analysis import classification_metrics, paired_bootstrap_delta
from autoshotv2.postprocess_calibration import f1_pr


def evaluate_fixed_inline(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # Verbatim copy of the tail of train_phase2.evaluate_fixed (lines 898-900).
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1, precision, recall


def vectorized_f1(values: np.ndarray) -> np.ndarray:
    # Verbatim copy of f1_from_counts nested in paper_analysis.paired_bootstrap_delta.
    tp = values[..., 0].astype(np.float64)
    fp = values[..., 1].astype(np.float64)
    fn = values[..., 2].astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )


def count_triples() -> np.ndarray:
    rng = np.random.default_rng(1234)
    triples = rng.integers(0, 500, size=(200, 3))
    edge_cases = np.array(
        [
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 5],
            [0, 5, 5],
            [5, 0, 0],
            [1, 0, 999],
            [999, 1, 0],
        ]
    )
    return np.vstack([triples, edge_cases])


def test_scalar_implementations_are_bit_identical():
    for tp, fp, fn in count_triples().tolist():
        metrics = classification_metrics(tp, fp, fn)
        f1_a, p_a, r_a = f1_pr(tp, fp, fn)
        f1_b, p_b, r_b = evaluate_fixed_inline(tp, fp, fn)

        assert metrics["f1"] == f1_a == f1_b
        assert metrics["precision"] == p_a == p_b
        assert metrics["recall"] == r_a == r_b


def test_vectorized_f1_matches_scalar_exactly():
    triples = count_triples()
    vector = vectorized_f1(triples)
    for row, expected in zip(triples.tolist(), vector.tolist()):
        assert classification_metrics(*row)["f1"] == expected


def test_paired_bootstrap_delta_uses_the_same_f1():
    # Identical per-video counts must give a point delta of exactly 0.0.
    stats = count_triples()[:20]
    result = paired_bootstrap_delta(stats, stats, n_samples=64, seed=7)
    assert result["delta"] == 0.0
