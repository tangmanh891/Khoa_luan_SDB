"""Cross-validate ClipShots-specific post-processing on cached deploy logits."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoshotv2 import runtime
from autoshotv2.common import classification_metrics, load_logits, normalize_key
from autoshotv2.eval import evaluate_scenes, logits_to_predictions, predictions_to_scenes

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOGITS = ROOT / "artifacts" / "experiments" / "deploy_regen" / "clipshots_test_logits.pkl"
DEFAULT_GT = ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "clipshots_test_gt.pkl"
DEFAULT_OUTPUT_JSON = ROOT / "reports" / "clipshots_postprocess_cv.json"
DEFAULT_OUTPUT_CSV = ROOT / "reports" / "clipshots_postprocess_cv.csv"
DEFAULT_OUTPUT_MD = ROOT / "reports" / "clipshots_postprocess_cv.md"

DEFAULT_THRESHOLDS = "0.100,0.118,0.121,0.134,0.135,0.136,0.137,0.138,0.139,0.150"
DEFAULT_TEMPERATURES = f"0.35,{runtime.DEFAULT_TEMPERATURE:.17g},0.5"
DEFAULT_SIGMAS = "1.5,2.0,2.5"
DEFAULT_MIN_DISTANCES = "0,16,24"
DEFAULT_PROMINENCES = "0,0.001"
DEFAULT_MODES = "segment,peak"


@dataclass(frozen=True, order=True)
class PostprocessParam:
    mode: str
    temperature: float
    sigma: float
    threshold: float
    min_distance: int = 0
    prominence: float = 0.0

    def compact_key(self) -> str:
        return (
            f"{self.mode}|T={self.temperature:.8g}|sigma={self.sigma:.8g}|"
            f"thr={self.threshold:.6g}|dist={self.min_distance}|prom={self.prominence:.6g}"
        )


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    return sorted(set(values))


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    return sorted(set(values))


def parse_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(modes) - {"segment", "peak"})
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Expected segment and/or peak")
    return modes or ["segment", "peak"]


def load_ground_truth(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return {normalize_key(key): np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def load_score_cache(logits_path: Path, temperature: float, sigma: float) -> dict[str, np.ndarray]:
    logits = load_logits(logits_path)
    predictions = logits_to_predictions(logits, temperature=temperature, sigma=sigma)
    return {
        normalize_key(key): np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in predictions.items()
    }


def segment_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores).reshape(-1) > float(threshold)).astype(np.uint8)


def peak_mask(
    scores: np.ndarray,
    threshold: float,
    min_distance: int = 0,
    prominence: float = 0.0,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    mask = np.zeros(len(values), dtype=np.uint8)
    if len(values) < 3:
        return mask

    centers = np.arange(1, len(values) - 1)
    left = values[:-2]
    current = values[1:-1]
    right = values[2:]
    candidates = centers[(current > float(threshold)) & (current >= left) & (current > right)]

    if prominence > 0 and len(candidates):
        neighbor_max = np.maximum(values[candidates - 1], values[candidates + 1])
        candidates = candidates[(values[candidates] - neighbor_max) >= float(prominence)]

    if min_distance > 0 and len(candidates) > 1:
        ordered = candidates[np.argsort(values[candidates])[::-1]]
        keep: list[int] = []
        suppressed = np.zeros(len(values), dtype=bool)
        for index in ordered:
            if suppressed[index]:
                continue
            keep.append(int(index))
            start = max(0, int(index) - int(min_distance))
            stop = min(len(values), int(index) + int(min_distance) + 1)
            suppressed[start:stop] = True
        candidates = np.asarray(sorted(keep), dtype=np.int64)

    mask[candidates] = 1
    return mask


def build_param_grid(
    modes: list[str],
    temperatures: list[float],
    sigmas: list[float],
    thresholds: list[float],
    min_distances: list[int],
    prominences: list[float],
) -> list[PostprocessParam]:
    params: list[PostprocessParam] = []
    for temperature in temperatures:
        for sigma in sigmas:
            for threshold in thresholds:
                if "segment" in modes:
                    params.append(
                        PostprocessParam(
                            mode="segment",
                            temperature=float(temperature),
                            sigma=float(sigma),
                            threshold=float(threshold),
                        )
                    )
                if "peak" in modes:
                    for min_distance in min_distances:
                        for prominence in prominences:
                            params.append(
                                PostprocessParam(
                                    mode="peak",
                                    temperature=float(temperature),
                                    sigma=float(sigma),
                                    threshold=float(threshold),
                                    min_distance=int(min_distance),
                                    prominence=float(prominence),
                                )
                            )
    return params


def evaluate_param_counts(scores: np.ndarray, gt_scenes: np.ndarray, param: PostprocessParam) -> tuple[int, int, int]:
    if param.mode == "segment":
        binary = segment_mask(scores, param.threshold)
    elif param.mode == "peak":
        binary = peak_mask(scores, param.threshold, param.min_distance, param.prominence)
    else:
        raise ValueError(f"Unsupported postprocess mode: {param.mode}")

    pred_scenes = predictions_to_scenes(binary)
    tp, fp, fn = evaluate_scenes(np.asarray(gt_scenes), pred_scenes)
    return int(tp), int(fp), int(fn)


def add_counts(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def aggregate_counts(
    counts_by_param: dict[PostprocessParam, dict[str, tuple[int, int, int]]],
    param: PostprocessParam,
    keys: list[str],
) -> tuple[int, int, int]:
    total = (0, 0, 0)
    param_counts = counts_by_param[param]
    for key in keys:
        total = add_counts(total, param_counts[key])
    return total


def metrics_from_counts(counts: tuple[int, int, int]) -> dict[str, float | int]:
    return classification_metrics(*counts)


def selection_key(metrics: dict[str, float | int], param: PostprocessParam) -> tuple[float, float, float, int, float, int, float]:
    complexity = 0 if param.mode == "segment" else 1
    return (
        round(float(metrics["f1"]), 12),
        round(float(metrics["precision"]), 12),
        round(float(metrics["recall"]), 12),
        -complexity,
        -float(param.threshold),
        -int(param.min_distance),
        -float(param.prominence),
    )


def select_best_param(
    params: list[PostprocessParam],
    counts_by_param: dict[PostprocessParam, dict[str, tuple[int, int, int]]],
    keys: list[str],
) -> tuple[PostprocessParam, dict[str, float | int]]:
    if not params:
        raise ValueError("Parameter grid is empty")
    if not keys:
        raise ValueError("Cannot select parameters from an empty split")

    best_param = params[0]
    best_metrics = metrics_from_counts(aggregate_counts(counts_by_param, best_param, keys))
    best_key = selection_key(best_metrics, best_param)
    for param in params[1:]:
        metrics = metrics_from_counts(aggregate_counts(counts_by_param, param, keys))
        key = selection_key(metrics, param)
        if key > best_key:
            best_param = param
            best_metrics = metrics
            best_key = key
    return best_param, best_metrics


def make_video_folds(keys: list[str], fold_count: int, seed: int) -> list[list[str]]:
    if fold_count < 2:
        raise ValueError("--folds must be at least 2")
    if fold_count > len(keys):
        raise ValueError("--folds cannot exceed the number of matched videos")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(keys), dtype=object)
    rng.shuffle(shuffled)
    return [sorted(str(item) for item in fold) for fold in np.array_split(shuffled, fold_count)]


def cross_validate(
    params: list[PostprocessParam],
    counts_by_param: dict[PostprocessParam, dict[str, tuple[int, int, int]]],
    folds: list[list[str]],
) -> dict[str, Any]:
    all_keys = [key for fold in folds for key in fold]
    heldout_total = (0, 0, 0)
    fold_results: list[dict[str, Any]] = []

    for fold_index, heldout_keys in enumerate(folds, 1):
        heldout_set = set(heldout_keys)
        train_keys = sorted(key for key in all_keys if key not in heldout_set)
        selected_param, calibration_metrics = select_best_param(params, counts_by_param, train_keys)
        heldout_counts = aggregate_counts(counts_by_param, selected_param, heldout_keys)
        heldout_metrics = metrics_from_counts(heldout_counts)
        heldout_total = add_counts(heldout_total, heldout_counts)
        fold_results.append(
            {
                "fold": fold_index,
                "calibration_videos": len(train_keys),
                "heldout_videos": len(heldout_keys),
                "selected_param": asdict(selected_param),
                "selected_key": selected_param.compact_key(),
                "calibration": calibration_metrics,
                "heldout": heldout_metrics,
            }
        )

    return {
        "fold_count": len(folds),
        "aggregate": metrics_from_counts(heldout_total),
        "folds": fold_results,
    }


def precompute_counts(
    logits_path: Path,
    ground_truth: dict[str, np.ndarray],
    keys: list[str],
    params: list[PostprocessParam],
) -> dict[PostprocessParam, dict[str, tuple[int, int, int]]]:
    grouped: dict[tuple[float, float], list[PostprocessParam]] = defaultdict(list)
    for param in params:
        grouped[(param.temperature, param.sigma)].append(param)

    counts_by_param: dict[PostprocessParam, dict[str, tuple[int, int, int]]] = {}
    for group_index, ((temperature, sigma), group_params) in enumerate(sorted(grouped.items()), 1):
        print(
            f"precompute {group_index}/{len(grouped)}: "
            f"T={temperature:.6g} sigma={sigma:.6g} params={len(group_params)}",
            flush=True,
        )
        scores = load_score_cache(logits_path, temperature=temperature, sigma=sigma)
        for param in group_params:
            counts_by_param[param] = {
                key: evaluate_param_counts(scores[key], ground_truth[key], param)
                for key in keys
            }
    return counts_by_param


def summarize_coverage(scores_keys: set[str], gt_keys: set[str]) -> dict[str, Any]:
    missing = sorted(gt_keys - scores_keys)
    extra = sorted(scores_keys - gt_keys)
    return {
        "predicted_videos": len(scores_keys),
        "gt_videos": len(gt_keys),
        "matched_videos": len(scores_keys & gt_keys),
        "missing_prediction_count": len(missing),
        "extra_prediction_count": len(extra),
        "missing_prediction_keys": missing[:20],
        "extra_prediction_keys": extra[:20],
    }


def rank_full_set_params(
    params: list[PostprocessParam],
    counts_by_param: dict[PostprocessParam, dict[str, tuple[int, int, int]]],
    keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for param in params:
        counts = aggregate_counts(counts_by_param, param, keys)
        rows.append({"param": param, "metrics": metrics_from_counts(counts)})
    rows.sort(key=lambda row: selection_key(row["metrics"], row["param"]), reverse=True)
    return [
        {
            "rank": index,
            **asdict(row["param"]),
            **row["metrics"],
            "key": row["param"].compact_key(),
        }
        for index, row in enumerate(rows, 1)
    ]


def find_param(params: list[PostprocessParam], target: PostprocessParam) -> PostprocessParam | None:
    for param in params:
        if param == target:
            return param
    return None


def locked_param_from_folds(fold_results: list[dict[str, Any]], ranked_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_keys = [row["selected_key"] for row in fold_results]
    counts = Counter(selected_keys)
    rank_by_key = {row["key"]: row for row in ranked_rows}
    best_key = max(
        counts,
        key=lambda key: (counts[key], -int(rank_by_key.get(key, {"rank": 10**9})["rank"])),
    )
    return {
        "selection_count": counts[best_key],
        **rank_by_key[best_key],
    }


def write_csv(path: Path, ranked_rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "mode",
        "temperature",
        "sigma",
        "threshold",
        "min_distance",
        "prominence",
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in ranked_rows)


def metric_row(label: str, metrics: dict[str, Any], delta_base: float | None = None) -> str:
    delta = "" if delta_base is None else f" | {float(metrics['f1']) - delta_base:+.4f}"
    return (
        f"| {label} | {float(metrics['f1']):.4f} | {float(metrics['precision']):.4f} | "
        f"{float(metrics['recall']):.4f} | {int(metrics['tp'])} | {int(metrics['fp'])} | "
        f"{int(metrics['fn'])}{delta} |"
    )


def write_markdown(path: Path, payload: dict[str, Any], top_k: int) -> None:
    deploy = payload["references"]["deploy_threshold"]
    cv_selected = payload["cv_selected"]["aggregate"]
    locked = payload["references"]["locked_cv_param"]
    oracle = payload["references"]["full_set_oracle"]
    deploy_f1 = float(deploy["f1"])
    cv_delta = float(cv_selected["f1"]) - deploy_f1
    if cv_delta > 0.005:
        conclusion = (
            "The cross-validated ClipShots-specific post-process improves over the fixed deploy "
            "threshold by more than 0.005 F1. It is a candidate for a clearly labeled dataset-specific setting."
        )
    else:
        conclusion = (
            "The cross-validated ClipShots-specific post-process does not show a robust gain over the fixed "
            "deploy threshold. Keep the main deploy result unchanged and treat the full-set oracle as analysis only."
        )

    lines = [
        "# ClipShots Postprocess Cross-Validation",
        "",
        "This report tunes post-processing on cached deploy-checkpoint logits using video-level cross-validation. "
        "Each fold selects parameters on the other folds and evaluates on held-out videos.",
        "",
        "The main deploy protocol is unchanged; full-set best values are oracle diagnostics and should not be used as headline claims.",
        "",
        "## Summary",
        "",
        "| Setting | F1 | Precision | Recall | TP | FP | FN | Delta F1 vs deploy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("Fixed deploy threshold", deploy, deploy_f1),
        metric_row("5-fold CV selected", cv_selected, deploy_f1),
        metric_row("Locked most-selected CV param on full set", locked, deploy_f1),
        metric_row("Full-set oracle", oracle, deploy_f1),
        "",
        f"Conclusion: {conclusion}",
        "",
        "## Selected Fold Parameters",
        "",
        "| Fold | Held-out videos | Mode | T | Sigma | Threshold | Min distance | Prominence | Held-out F1 | Precision | Recall |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in payload["cv_selected"]["folds"]:
        param = fold["selected_param"]
        heldout = fold["heldout"]
        lines.append(
            f"| {fold['fold']} | {fold['heldout_videos']} | {param['mode']} | "
            f"{float(param['temperature']):.4f} | {float(param['sigma']):.2f} | "
            f"{float(param['threshold']):.3f} | {int(param['min_distance'])} | "
            f"{float(param['prominence']):.3f} | {float(heldout['f1']):.4f} | "
            f"{float(heldout['precision']):.4f} | {float(heldout['recall']):.4f} |"
        )

    lines += [
        "",
        "## Top Full-Set Parameters",
        "",
        "| Rank | Mode | T | Sigma | Threshold | Min distance | Prominence | F1 | Precision | Recall | TP | FP | FN |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["top_full_set_params"][:top_k]:
        lines.append(
            f"| {row['rank']} | {row['mode']} | {float(row['temperature']):.4f} | "
            f"{float(row['sigma']):.2f} | {float(row['threshold']):.3f} | "
            f"{int(row['min_distance'])} | {float(row['prominence']):.3f} | "
            f"{float(row['f1']):.4f} | {float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} |"
        )

    lines += [
        "",
        "## Protocol",
        "",
        f"- Logits: `{payload['sources']['logits']}`",
        f"- Ground truth: `{payload['sources']['ground_truth']}`",
        f"- Folds: {payload['protocol']['folds']} video-level folds with seed {payload['protocol']['seed']}",
        f"- Grid size: {payload['protocol']['param_count']} parameter settings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = parse_float_list(args.thresholds)
    temperatures = parse_float_list(args.temperatures)
    sigmas = parse_float_list(args.sigmas)
    min_distances = parse_int_list(args.min_distances)
    prominences = parse_float_list(args.prominences)
    modes = parse_modes(args.modes)
    params = build_param_grid(modes, temperatures, sigmas, thresholds, min_distances, prominences)
    if not params:
        raise ValueError("Parameter grid is empty")

    ground_truth = load_ground_truth(args.gt)
    sample_scores = load_score_cache(args.logits, temperatures[0], sigmas[0])
    coverage = summarize_coverage(set(sample_scores), set(ground_truth))
    if coverage["missing_prediction_count"] and not args.allow_partial:
        raise RuntimeError(
            "ClipShots logits do not cover the ground-truth split: "
            f"{coverage['matched_videos']}/{coverage['gt_videos']} matched. "
            f"Missing sample: {coverage['missing_prediction_keys']}"
        )
    keys = sorted(set(sample_scores) & set(ground_truth))
    folds = make_video_folds(keys, args.folds, args.seed)

    counts_by_param = precompute_counts(args.logits, ground_truth, keys, params)
    ranked_rows = rank_full_set_params(params, counts_by_param, keys)
    cv_selected = cross_validate(params, counts_by_param, folds)

    deploy_param = find_param(
        params,
        PostprocessParam(
            mode="segment",
            temperature=runtime.DEFAULT_TEMPERATURE,
            sigma=runtime.DEFAULT_SIGMA,
            threshold=runtime.DEFAULT_THRESHOLD,
        ),
    )
    if deploy_param is None:
        deploy_param = PostprocessParam(
            mode="segment",
            temperature=runtime.DEFAULT_TEMPERATURE,
            sigma=runtime.DEFAULT_SIGMA,
            threshold=runtime.DEFAULT_THRESHOLD,
        )
        deploy_scores = load_score_cache(args.logits, deploy_param.temperature, deploy_param.sigma)
        deploy_counts = {
            key: evaluate_param_counts(deploy_scores[key], ground_truth[key], deploy_param)
            for key in keys
        }
        counts_by_param[deploy_param] = deploy_counts

    deploy_reference = metrics_from_counts(aggregate_counts(counts_by_param, deploy_param, keys))
    deploy_reference.update(asdict(deploy_param))

    locked_param = locked_param_from_folds(cv_selected["folds"], ranked_rows)
    full_set_oracle = ranked_rows[0]

    payload = {
        "schema_version": 1,
        "method": "ClipShots-specific video-level cross-validation of post-processing on cached deploy logits",
        "sources": {
            "logits": str(Path(args.logits).resolve().relative_to(ROOT)).replace("\\", "/"),
            "ground_truth": str(Path(args.gt).resolve().relative_to(ROOT)).replace("\\", "/"),
        },
        "coverage": coverage,
        "protocol": {
            "folds": args.folds,
            "seed": args.seed,
            "modes": modes,
            "thresholds": thresholds,
            "temperatures": temperatures,
            "sigmas": sigmas,
            "min_distances": min_distances,
            "prominences": prominences,
            "param_count": len(params),
            "selection_tie_break": "F1, precision, recall, lower complexity, lower threshold, lower min_distance, lower prominence",
        },
        "references": {
            "deploy_threshold": deploy_reference,
            "locked_cv_param": locked_param,
            "full_set_oracle": full_set_oracle,
        },
        "cv_selected": cv_selected,
        "top_full_set_params": ranked_rows[: args.top_k],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, ranked_rows)
    write_markdown(args.output_md, payload, args.top_k)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--temperatures", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS)
    parser.add_argument("--min-distances", default=DEFAULT_MIN_DISTANCES)
    parser.add_argument("--prominences", default=DEFAULT_PROMINENCES)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = run(args)
    deploy = payload["references"]["deploy_threshold"]
    cv_selected = payload["cv_selected"]["aggregate"]
    oracle = payload["references"]["full_set_oracle"]
    print(
        "ClipShots postprocess CV complete: "
        f"deploy F1={deploy['f1']:.4f}, "
        f"CV-selected F1={cv_selected['f1']:.4f}, "
        f"full-set oracle F1={oracle['f1']:.4f}"
    )
    print(f"report saved -> {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
