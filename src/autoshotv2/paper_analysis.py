"""Reproducible analysis artifacts for the AutoShotV2 paper.

The analysis uses the controlled A1 logits with the selected B4 post-process
configuration. It produces video-level bootstrap intervals, validation-set
calibration diagnostics, a ClipShots transition-type breakdown, and a
reliability diagram without retraining the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoshotv2.ablation import load_logits, scores_from_cache
from autoshotv2.eval import evaluate_scenes, predictions_to_scenes
from autoshotv2.train_phase2 import transitions_to_scenes
from autoshotv2.utils import scenes2zero_one_representation


ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = ROOT / "artifacts" / "experiments" / "ablation_full"
A1_CACHE = ABLATION_ROOT / "A1_phase2_bce_onehot" / "eval_cache"
META_PATH = ABLATION_ROOT / "resolved_meta.pickle"
SHOT_GT_PATH = ROOT / "reports" / "source_results" / "shot_test_ground_truth.json"
FROZEN_PROTOCOL_PATH = (
    ROOT
    / "reports"
    / "source_results"
    / "journal_frozen_protocol_seed42.json"
)
CLIPSHOTS_ANNOTATIONS = ROOT / "data" / "ClipShots" / "annotations" / "test.json"
BBC_ANNOTATIONS = ROOT / "data" / "BBCDataset" / "annotations" / "shots"

B4_TEMPERATURE = 0.661785970550883
B4_SIGMA = 2.0
B4_THRESHOLD = 0.1
A1_THRESHOLD = 0.5
MATCH_TOLERANCE = 2
BOOTSTRAP_SAMPLES = 10_000
SEED = 42
ECE_BINS = 15

SHOT_GT_URL = (
    "https://raw.githubusercontent.com/wentaozhu/AutoShot/main/"
    "gt_scenes_dict_baseline_v2.pickle"
)


def normalize_key(value: str) -> str:
    return Path(str(value).split(":", 1)[-1]).stem


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def bootstrap_confidence_intervals(
    stats: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Micro-average sampled video rows and return percentile intervals."""
    values = np.asarray(stats, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("stats must have shape (n_videos, 3)")

    point = classification_metrics(*values.sum(axis=0).tolist())
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(n_samples, dtype=np.float64) for name in ("f1", "precision", "recall")}

    offset = 0
    batch_size = min(512, n_samples)
    while offset < n_samples:
        current = min(batch_size, n_samples - offset)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        aggregate = values[indices].sum(axis=1)
        tp = aggregate[:, 0].astype(np.float64)
        fp = aggregate[:, 1].astype(np.float64)
        fn = aggregate[:, 2].astype(np.float64)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        distributions["precision"][offset : offset + current] = precision
        distributions["recall"][offset : offset + current] = recall
        distributions["f1"][offset : offset + current] = f1
        offset += current

    intervals = {}
    for name, distribution in distributions.items():
        low, high = np.percentile(distribution, [2.5, 97.5])
        intervals[name] = {
            "value": float(point[name]),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return {
        "n_videos": int(len(values)),
        "bootstrap_samples": int(n_samples),
        "seed": int(seed),
        "sampling_unit": "video",
        "point_counts": {key: point[key] for key in ("tp", "fp", "fn")},
        "metrics": intervals,
    }


def paired_bootstrap_delta(
    baseline_stats: np.ndarray,
    method_stats: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_stats, dtype=np.int64)
    method = np.asarray(method_stats, dtype=np.int64)
    if baseline.shape != method.shape or baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("paired stats must have equal shape (n_videos, 3)")
    if len(baseline) == 0:
        raise ValueError("paired stats must be non-empty")

    def f1_from_counts(values: np.ndarray) -> np.ndarray:
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

    point = float(
        f1_from_counts(method.sum(axis=0))
        - f1_from_counts(baseline.sum(axis=0))
    )
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_samples, dtype=np.float64)
    offset = 0
    batch_size = min(512, n_samples)
    while offset < n_samples:
        current = min(batch_size, n_samples - offset)
        indices = rng.integers(0, len(baseline), size=(current, len(baseline)))
        baseline_sample = baseline[indices].sum(axis=1)
        method_sample = method[indices].sum(axis=1)
        distribution[offset : offset + current] = (
            f1_from_counts(method_sample) - f1_from_counts(baseline_sample)
        )
        offset += current
    low, high = np.percentile(distribution, [2.5, 97.5])
    return {
        "metric": "f1",
        "sampling_unit": "paired video",
        "n_videos": int(len(baseline)),
        "bootstrap_samples": int(n_samples),
        "seed": int(seed),
        "delta": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def _calibration_bins(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_ids: np.ndarray,
    n_bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    counts = np.bincount(bin_ids, minlength=n_bins)
    confidence_sum = np.bincount(
        bin_ids,
        weights=probabilities,
        minlength=n_bins,
    )
    accuracy_sum = np.bincount(
        bin_ids,
        weights=targets,
        minlength=n_bins,
    )
    bins = []
    ece = 0.0
    for index in range(n_bins):
        count = int(counts[index])
        confidence = float(confidence_sum[index] / count) if count else 0.0
        accuracy = float(accuracy_sum[index] / count) if count else 0.0
        ece += (count / len(probabilities)) * abs(accuracy - confidence)
        bins.append(
            {
                "index": index,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), bins


def calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    n_bins: int = ECE_BINS,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.shape != targets.shape or len(values) == 0:
        raise ValueError("logits and labels must be non-empty arrays with equal shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = np.clip(values / temperature, -50.0, 50.0)
    probabilities = 1.0 / (1.0 + np.exp(-scaled))
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    nll = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    brier = np.mean((probabilities - targets) ** 2)

    bin_ids = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    ece, bins = _calibration_bins(probabilities, targets, bin_ids, n_bins)
    for index, item in enumerate(bins):
        item["lower"] = index / n_bins
        item["upper"] = (index + 1) / n_bins

    order = np.argsort(probabilities, kind="stable")
    adaptive_ids = np.empty(len(probabilities), dtype=np.int64)
    for index, indices in enumerate(np.array_split(order, n_bins)):
        adaptive_ids[indices] = index
    adaptive_ece, adaptive_bins = _calibration_bins(
        probabilities,
        targets,
        adaptive_ids,
        n_bins,
    )
    for item in adaptive_bins:
        mask = adaptive_ids == item["index"]
        item["lower"] = float(probabilities[mask].min()) if np.any(mask) else 0.0
        item["upper"] = float(probabilities[mask].max()) if np.any(mask) else 0.0

    class_ece = {}
    for label, name in ((0.0, "negative"), (1.0, "positive")):
        mask = targets == label
        if not np.any(mask):
            class_ece[name] = None
            continue
        class_bin_ids = np.minimum(
            (probabilities[mask] * n_bins).astype(np.int64),
            n_bins - 1,
        )
        value, _ = _calibration_bins(
            probabilities[mask],
            targets[mask],
            class_bin_ids,
            n_bins,
        )
        class_ece[name] = value
    class_balanced_ece = float(
        np.mean([value for value in class_ece.values() if value is not None])
    )
    return {
        "temperature": float(temperature),
        "frames": int(len(probabilities)),
        "positive_rate": float(targets.mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "adaptive_ece": adaptive_ece,
        "class_ece": class_ece,
        "class_balanced_ece": class_balanced_ece,
        "n_bins": int(n_bins),
        "bins": bins,
        "adaptive_bins": adaptive_bins,
    }


def scene_transitions(scenes: np.ndarray) -> np.ndarray:
    values = np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
    if len(values) < 2:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([values[:-1, 1], values[1:, 0]], axis=1)


def transition_kind_mask(transitions: np.ndarray, kind: str) -> np.ndarray:
    widths = np.asarray(transitions)[:, 1] - np.asarray(transitions)[:, 0]
    if kind == "cut":
        return widths <= 1
    if kind == "gradual":
        return widths > 1
    raise ValueError(f"Unknown transition kind: {kind}")


def match_transition_intervals(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    tolerance: int = MATCH_TOLERANCE,
) -> tuple[int, int, int]:
    """Two-pointer interval matching equivalent to the canonical scene matcher."""
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1, 2)
    shift = tolerance / 2.0
    adjustment = np.array([0.5 - shift, -0.5 + shift], dtype=np.float64)
    gt = gt + adjustment
    pred = pred + adjustment

    i = j = tp = fp = fn = 0
    while i < len(gt) or j < len(pred):
        if j == len(pred):
            fn += 1
            i += 1
        elif i == len(gt):
            fp += 1
            j += 1
        elif pred[j, 1] < gt[i, 0]:
            fp += 1
            j += 1
        elif pred[j, 0] > gt[i, 1]:
            fn += 1
            i += 1
        else:
            tp += 1
            i += 1
            j += 1
    return tp, fp, fn


def match_transition_pairs(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    tolerance: int = MATCH_TOLERANCE,
) -> list[tuple[int, int]]:
    """Return matched (ground-truth index, prediction index) pairs."""
    gt_raw = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    pred_raw = np.asarray(predictions, dtype=np.float64).reshape(-1, 2)
    shift = tolerance / 2.0
    adjustment = np.array([0.5 - shift, -0.5 + shift], dtype=np.float64)
    gt = gt_raw + adjustment
    pred = pred_raw + adjustment

    pairs = []
    i = j = 0
    while i < len(gt) and j < len(pred):
        if pred[j, 1] < gt[i, 0]:
            j += 1
        elif pred[j, 0] > gt[i, 1]:
            i += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    return pairs


def transition_type_breakdown(
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    threshold: float = B4_THRESHOLD,
    tolerance: int = MATCH_TOLERANCE,
) -> list[dict[str, Any]]:
    counts = {
        kind: {"ground_truth": 0, "matched": 0}
        for kind in ("cut", "gradual")
    }
    for key in sorted(set(probabilities) & set(ground_truth)):
        binary = (np.asarray(probabilities[key]).reshape(-1) > threshold).astype(np.uint8)
        pred_transitions = scene_transitions(predictions_to_scenes(binary))
        gt_transitions = scene_transitions(ground_truth[key])
        gt_kinds = np.where(transition_kind_mask(gt_transitions, "cut"), "cut", "gradual")
        pairs = match_transition_pairs(gt_transitions, pred_transitions, tolerance)
        for kind in counts:
            counts[kind]["ground_truth"] += int(np.count_nonzero(gt_kinds == kind))
        for gt_index, _ in pairs:
            counts[str(gt_kinds[gt_index])]["matched"] += 1

    rows = []
    for kind, label in (("cut", "Cut"), ("gradual", "Gradual")):
        total = counts[kind]["ground_truth"]
        matched = counts[kind]["matched"]
        rows.append(
            {
                "id": kind,
                "label": label,
                "ground_truth": total,
                "matched": matched,
                "missed": total - matched,
                "recall": matched / total if total else 0.0,
            }
        )
    return rows


def import_shot_ground_truth(pickle_path: Path, output_path: Path = SHOT_GT_PATH) -> dict[str, Any]:
    raw = pickle_path.read_bytes()
    with pickle_path.open("rb") as handle:
        scenes = pickle.load(handle)
    normalized = {
        normalize_key(key): np.asarray(value, dtype=np.int64).reshape(-1, 2).tolist()
        for key, value in scenes.items()
    }
    payload = {
        "schema_version": 1,
        "dataset": "shot",
        "split": "test",
        "videos": len(normalized),
        "source": {
            "repository": "https://github.com/wentaozhu/AutoShot",
            "url": SHOT_GT_URL,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "scenes": dict(sorted(normalized.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_shot_ground_truth(path: Path = SHOT_GT_PATH) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload["scenes"].items()}


def load_clipshots_ground_truth(
    logits: dict[str, np.ndarray],
    annotation_path: Path = CLIPSHOTS_ANNOTATIONS,
) -> dict[str, np.ndarray]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    normalized = {normalize_key(key): value for key, value in annotations.items()}
    result = {}
    for key, value in logits.items():
        if key not in normalized:
            continue
        transitions = np.asarray(normalized[key]["transitions"], dtype=np.int64).reshape(-1, 2)
        transitions = np.sort(transitions, axis=1)
        result[key] = transitions_to_scenes(transitions, len(value))
    return result


def load_bbc_ground_truth(annotation_dir: Path = BBC_ANNOTATIONS) -> dict[str, np.ndarray]:
    result = {}
    for path in sorted(annotation_dir.glob("*.txt")):
        episode = int(path.name[:2])
        result[f"bbc_{episode:02d}"] = np.loadtxt(path, dtype=np.int64).reshape(-1, 2)
    return result


def load_normalized_logits(path: Path) -> dict[str, np.ndarray]:
    return {normalize_key(key): np.asarray(value) for key, value in load_logits(path).items()}


def per_video_boundary_stats(
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    threshold: float = B4_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    keys = sorted(set(probabilities) & set(ground_truth))
    rows = np.empty((len(keys), 3), dtype=np.int64)
    for index, key in enumerate(keys):
        binary = (np.asarray(probabilities[key]).reshape(-1) > threshold).astype(np.uint8)
        rows[index] = evaluate_scenes(ground_truth[key], predictions_to_scenes(binary))
    return keys, rows


def validation_logits_and_labels() -> tuple[np.ndarray, np.ndarray, int]:
    with META_PATH.open("rb") as handle:
        metadata = pickle.load(handle)
    logits = load_logits(A1_CACHE / "combined_val_logits.pkl")
    all_logits = []
    all_labels = []
    matched = 0
    for key, value in logits.items():
        if key not in metadata["entries"]:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        entry = metadata["entries"][key]
        scenes = transitions_to_scenes(entry["transitions"], len(arr))
        labels, _ = scenes2zero_one_representation(scenes, len(arr))
        all_logits.append(arr)
        all_labels.append(labels.astype(np.float32))
        matched += 1
    if not all_logits:
        raise RuntimeError("Validation logits do not overlap resolved metadata")
    return np.concatenate(all_logits), np.concatenate(all_labels), matched


def logits_and_labels(
    logits: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    all_logits = []
    all_labels = []
    matched = 0
    for key in sorted(set(logits) & set(ground_truth)):
        values = np.asarray(logits[key], dtype=np.float32).reshape(-1)
        labels = np.zeros(len(values), dtype=np.float32)
        transitions = scene_transitions(ground_truth[key])
        if len(transitions):
            indices = transitions[:, 0] + (
                transitions[:, 1] - transitions[:, 0]
            ) // 2
            indices = np.clip(indices, 0, len(values) - 1)
            labels[indices.astype(np.int64)] = 1.0
        all_logits.append(values)
        all_labels.append(labels)
        matched += 1
    if not all_logits:
        raise RuntimeError("Logits do not overlap ground truth")
    return np.concatenate(all_logits), np.concatenate(all_labels), matched


def reliability_figure(
    before: dict[str, Any],
    after: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=180)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="Perfect calibration")
    for payload, label, color, marker in (
        (before, "Before scaling", "#1f77b4", "o"),
        (after, "Temperature scaled", "#d62728", "s"),
    ):
        populated = [item for item in payload["adaptive_bins"] if item["count"]]
        ax.plot(
            [item["confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (equal-mass bins)")
    ax.set_ylabel("Empirical positive rate")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def benchmark_postprocess(logits_by_dataset: dict[str, dict[str, np.ndarray]], rounds: int = 3) -> dict[str, Any]:
    frames = sum(len(value) for logits in logits_by_dataset.values() for value in logits.values())
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for logits in logits_by_dataset.values():
            scores_from_cache(logits, B4_TEMPERATURE, B4_SIGMA, "logits")
        timings.append(time.perf_counter() - start)
    median_seconds = float(np.median(timings))
    return {
        "scope": "temperature scaling and Gaussian smoothing on cached logits",
        "device": "CPU",
        "frames": int(frames),
        "rounds": int(rounds),
        "median_seconds": median_seconds,
        "milliseconds_per_million_frames": median_seconds * 1_000_000_000.0 / frames,
    }


def write_per_video_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "video", "tp", "fp", "fn"))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# AutoShotV2 Paper Analysis",
        "",
        "Controlled headline: A1 BCE one-hot logits with B4 temperature scaling and Gaussian smoothing.",
        "",
        "## Bootstrap 95% confidence intervals",
        "",
        "| Dataset | F1 | 95% CI | Precision | Recall | Videos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("shot", "bbc", "clipshots"):
        item = result["bootstrap"][dataset]
        metrics = item["metrics"]
        lines.append(
            f"| {dataset.upper()} | {metrics['f1']['value']:.4f} | "
            f"[{metrics['f1']['ci95_low']:.4f}, {metrics['f1']['ci95_high']:.4f}] | "
            f"{metrics['precision']['value']:.4f} | {metrics['recall']['value']:.4f} | "
            f"{item['n_videos']} |"
        )
    calibration = result["calibration"]
    lines += [
        "",
        "## Validation calibration diagnostics",
        "",
        "| Variant | NLL | Brier | ECE | Adaptive ECE | Balanced ECE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Before scaling | {calibration['before']['nll']:.6f} | "
        f"{calibration['before']['brier']:.6f} | {calibration['before']['ece']:.6f} | "
        f"{calibration['before']['adaptive_ece']:.6f} | "
        f"{calibration['before']['class_balanced_ece']:.6f} |",
        f"| Temperature scaled | {calibration['after_temperature']['nll']:.6f} | "
        f"{calibration['after_temperature']['brier']:.6f} | "
        f"{calibration['after_temperature']['ece']:.6f} | "
        f"{calibration['after_temperature']['adaptive_ece']:.6f} | "
        f"{calibration['after_temperature']['class_balanced_ece']:.6f} |",
        "",
        "The calibration diagnostics use the same validation set that selected the temperature and are",
        "therefore diagnostic rather than an untouched-test estimate.",
        "",
        "## Paired F1 delta, selected method minus A1",
        "",
        "| Dataset | Delta F1 | 95% CI | Excludes zero |",
        "|---|---:|---:|:---:|",
    ]
    for dataset in ("shot", "bbc", "clipshots"):
        item = result["paired_delta_vs_a1"][dataset]
        lines.append(
            f"| {dataset.upper()} | {item['delta']:.4f} | "
            f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | "
            f"{'yes' if item['excludes_zero'] else 'no'} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    frozen_protocol = json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))
    selected_protocol = frozen_protocol["selected"]
    expected = (B4_TEMPERATURE, B4_SIGMA, B4_THRESHOLD)
    actual = (
        float(selected_protocol["temperature"]),
        float(selected_protocol["sigma"]),
        float(selected_protocol["threshold"]),
    )
    if not np.allclose(expected, actual, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "Analysis constants do not match the frozen validation-only protocol: "
            f"expected={expected}, manifest={actual}"
        )
    logits_by_dataset = {
        "shot": load_normalized_logits(A1_CACHE / "shot_test_logits.pkl"),
        "bbc": load_normalized_logits(A1_CACHE / "bbc_test_logits.pkl"),
        "clipshots": load_normalized_logits(A1_CACHE / "clipshots_test_logits.pkl"),
    }
    ground_truth = {
        "shot": load_shot_ground_truth(),
        "bbc": load_bbc_ground_truth(),
        "clipshots": load_clipshots_ground_truth(logits_by_dataset["clipshots"]),
    }

    bootstrap = {}
    per_video_rows = []
    probabilities_by_dataset = {}
    baseline_stats_by_dataset = {}
    selected_stats_by_dataset = {}
    for dataset in ("shot", "bbc", "clipshots"):
        baseline_probabilities = scores_from_cache(
            logits_by_dataset[dataset],
            temperature=1.0,
            sigma=0.0,
            input_kind="logits",
        )
        baseline_probabilities = {
            normalize_key(key): value
            for key, value in baseline_probabilities.items()
        }
        probabilities = scores_from_cache(
            logits_by_dataset[dataset],
            temperature=B4_TEMPERATURE,
            sigma=B4_SIGMA,
            input_kind="logits",
        )
        probabilities = {normalize_key(key): value for key, value in probabilities.items()}
        probabilities_by_dataset[dataset] = probabilities
        baseline_keys, baseline_stats = per_video_boundary_stats(
            baseline_probabilities,
            ground_truth[dataset],
            threshold=A1_THRESHOLD,
        )
        keys, stats = per_video_boundary_stats(probabilities, ground_truth[dataset])
        if baseline_keys != keys:
            raise RuntimeError(f"Paired analysis key mismatch for {dataset}")
        baseline_stats_by_dataset[dataset] = baseline_stats
        selected_stats_by_dataset[dataset] = stats
        bootstrap[dataset] = bootstrap_confidence_intervals(stats)
        for key, (tp, fp, fn) in zip(keys, stats.tolist()):
            per_video_rows.append(
                {"dataset": dataset, "video": key, "tp": tp, "fp": fp, "fn": fn}
            )

    validation_logits, validation_labels, validation_videos = validation_logits_and_labels()
    before = calibration_metrics(validation_logits, validation_labels, temperature=1.0)
    after = calibration_metrics(
        validation_logits,
        validation_labels,
        temperature=B4_TEMPERATURE,
    )
    clipshots_breakdown = transition_type_breakdown(
        probabilities_by_dataset["clipshots"],
        ground_truth["clipshots"],
    )

    head_total = (4864 * 1024 + 1024) + 2 * (1024 + 1)
    head_primary = (4864 * 1024 + 1024) + (1024 + 1)
    test_calibration = {}
    for dataset in ("shot", "bbc", "clipshots"):
        dataset_logits, dataset_labels, dataset_videos = logits_and_labels(
            logits_by_dataset[dataset],
            ground_truth[dataset],
        )
        test_calibration[dataset] = {
            "videos": dataset_videos,
            "before": calibration_metrics(
                dataset_logits,
                dataset_labels,
                temperature=1.0,
            ),
            "after_temperature": calibration_metrics(
                dataset_logits,
                dataset_labels,
                temperature=B4_TEMPERATURE,
            ),
        }
    result = {
        "schema_version": 1,
        "method": "A1 BCE one-hot + B4 temperature scaling and Gaussian smoothing",
        "protocol": {
            "temperature": B4_TEMPERATURE,
            "sigma": B4_SIGMA,
            "threshold": B4_THRESHOLD,
            "matching_tolerance_frames": MATCH_TOLERANCE,
            "seed": SEED,
        },
        "sources": {
            "logits": str(A1_CACHE.relative_to(ROOT)).replace("\\", "/"),
            "shot_ground_truth": str(SHOT_GT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "clipshots_annotations": str(CLIPSHOTS_ANNOTATIONS.relative_to(ROOT)).replace("\\", "/"),
            "bbc_annotations": str(BBC_ANNOTATIONS.relative_to(ROOT)).replace("\\", "/"),
            "frozen_protocol": str(FROZEN_PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "selection": {
            "scope": frozen_protocol["selection_scope"],
            "metric": frozen_protocol["selection_metric"],
            "folds": frozen_protocol["n_folds"],
            "validation_keys_hash": frozen_protocol["validation_keys_hash"],
            "cross_validated_macro_source_f1": selected_protocol[
                "cross_validated_macro_source_f1"
            ],
        },
        "bootstrap": bootstrap,
        "paired_delta_vs_a1": {
            dataset: paired_bootstrap_delta(
                baseline_stats_by_dataset[dataset],
                selected_stats_by_dataset[dataset],
            )
            for dataset in ("shot", "bbc", "clipshots")
        },
        "calibration": {
            "split": "combined validation",
            "videos": validation_videos,
            "bins": ECE_BINS,
            "note": "Diagnostic: the validation set also selected the temperature.",
            "before": before,
            "after_temperature": after,
        },
        "test_calibration": test_calibration,
        "clipshots_transition_breakdown": {
            "classification": "Ground truth is Cut if transition width <= 1 frame; gradual otherwise.",
            "matching": "Class-agnostic predictions are matched once with the canonical +/-2 frame interval matcher; recall is then grouped by ground-truth type.",
            "metric_note": "Type-specific precision and F1 are undefined because the detector does not predict a transition type.",
            "transition_types": clipshots_breakdown,
        },
        "efficiency": {
            "feature_dimension": 4864,
            "hidden_dimension": 1024,
            "trainable_head_parameters": head_total,
            "deployed_primary_path_parameters": head_primary,
            "frozen_backbone_parameters": 14_299_202,
            "training_samples": 33_174,
            "training_videos": 400,
            "epochs": 20,
            "training_seed": SEED,
            "training_elapsed_time": None,
            "training_time_note": "Elapsed training time was not retained in the run artifact.",
            "postprocess_benchmark": benchmark_postprocess(logits_by_dataset),
        },
    }

    output_json = ROOT / "reports" / "paper_analysis_results.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_per_video_csv(ROOT / "reports" / "paper_analysis_per_video.csv", per_video_rows)
    write_summary(ROOT / "reports" / "paper_analysis_summary.md", result)
    reliability_figure(
        before,
        after,
        ROOT / "publications" / "paper" / "images" / "reliability_diagram.png",
    )

    breakdown_payload = {
        "schema_version": 2,
        "dataset": "clipshots",
        "split": "test",
        "videos": len(probabilities_by_dataset["clipshots"]),
        "protocol": result["protocol"],
        "source_kind": "recomputed_from_logits_and_annotations",
        "sources": [
            result["sources"]["logits"],
            result["sources"]["clipshots_annotations"],
        ],
        "classification": result["clipshots_transition_breakdown"]["classification"],
        "matching": result["clipshots_transition_breakdown"]["matching"],
        "metric_note": result["clipshots_transition_breakdown"]["metric_note"],
        "transition_types": clipshots_breakdown,
    }
    breakdown_path = ROOT / "reports" / "source_results" / "clipshots_transition_breakdown.json"
    breakdown_path.write_text(json.dumps(breakdown_payload, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import-shot-gt-pickle",
        type=Path,
        help="Import the official AutoShot ground-truth pickle into a tracked JSON source artifact.",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Stop after importing SHOT ground truth.",
    )
    args = parser.parse_args()

    if args.import_shot_gt_pickle:
        payload = import_shot_ground_truth(args.import_shot_gt_pickle)
        print(f"SHOT ground truth -> {SHOT_GT_PATH} ({payload['videos']} videos)")
    if args.import_only:
        return
    if not SHOT_GT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {SHOT_GT_PATH}. Import the official file with "
            "--import-shot-gt-pickle before running the analysis."
        )
    result = run_analysis()
    print(
        "Paper analysis complete: "
        + ", ".join(
            f"{dataset} F1={result['bootstrap'][dataset]['metrics']['f1']['value']:.4f}"
            for dataset in ("shot", "bbc", "clipshots")
        )
    )


if __name__ == "__main__":
    main()
