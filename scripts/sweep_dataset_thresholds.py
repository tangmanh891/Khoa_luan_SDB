"""Sweep decision thresholds for cached deploy logits."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from autoshotv2 import runtime
from autoshotv2.common import load_logits, normalize_key
from autoshotv2.eval import DEFAULT_THRESHOLDS, eval_at_threshold, logits_to_predictions

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_JSON = ROOT / "reports" / "threshold_sweep_bbc_clipshots.json"
DEFAULT_OUTPUT_CSV = ROOT / "reports" / "threshold_sweep_bbc_clipshots.csv"
DEFAULT_OUTPUT_MD = ROOT / "reports" / "threshold_sweep_bbc_clipshots.md"

DATASETS = {
    "bbc": {
        "label": "BBC",
        "gt": ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "bbc_test_gt.pkl",
        "logits": ROOT / "artifacts" / "experiments" / "deploy_regen" / "bbc_test_logits.pkl",
    },
    "clipshots": {
        "label": "ClipShots",
        "gt": ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "clipshots_test_gt.pkl",
        "logits": ROOT / "artifacts" / "experiments" / "deploy_regen" / "clipshots_test_logits.pkl",
    },
}


def parse_datasets(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Expected one of {sorted(DATASETS)}")
    return names or list(DATASETS)


def fine_thresholds(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--fine-step must be positive")
    if max_value < min_value:
        raise ValueError("--fine-max must be greater than or equal to --fine-min")
    count = int(np.floor((max_value - min_value) / step + 1e-9))
    values = [round(min_value + index * step, 6) for index in range(count + 1)]
    if not np.isclose(values[-1], max_value):
        values.append(round(max_value, 6))
    return sorted(set(values))


def load_ground_truth(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {normalize_key(key): np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def load_predictions(logits_path: Path, temperature: float, sigma: float) -> dict[str, np.ndarray]:
    logits = {normalize_key(key): np.asarray(value) for key, value in load_logits(logits_path).items()}
    return {normalize_key(key): value for key, value in logits_to_predictions(logits, temperature, sigma).items()}


def coverage_summary(predictions: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]) -> dict[str, Any]:
    predicted = set(predictions)
    expected = set(ground_truth)
    missing = sorted(expected - predicted)
    extra = sorted(predicted - expected)
    return {
        "predicted_videos": len(predicted),
        "gt_videos": len(expected),
        "matched_videos": len(predicted & expected),
        "missing_prediction_count": len(missing),
        "extra_prediction_count": len(extra),
        "missing_prediction_keys": missing[:20],
        "extra_prediction_keys": extra[:20],
    }


def require_full_coverage(dataset: str, coverage: dict[str, Any], allow_partial: bool) -> None:
    if allow_partial or coverage["missing_prediction_count"] == 0:
        return
    raise RuntimeError(
        f"{DATASETS[dataset]['label']} predictions do not cover the ground-truth split: "
        f"{coverage['matched_videos']}/{coverage['gt_videos']} matched. "
        f"Missing sample: {coverage['missing_prediction_keys']}"
    )


def sweep_thresholds(
    predictions: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    thresholds: list[float] | np.ndarray,
) -> list[dict[str, Any]]:
    common_keys = sorted(set(predictions) & set(ground_truth))
    common_pred = {key: predictions[key] for key in common_keys}
    common_gt = {key: ground_truth[key] for key in common_keys}
    rows = [eval_at_threshold(common_pred, common_gt, float(threshold)) for threshold in sorted(thresholds)]
    return sorted(rows, key=lambda item: item["threshold"])


def best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda item: item["f1"])


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: (-item["f1"], item["threshold"]))
    return [{**row, "rank": index} for index, row in enumerate(ranked, 1)]


def analyze_dataset(
    dataset: str,
    fine_grid: list[float],
    deploy_threshold: float,
    temperature: float,
    sigma: float,
    top_k: int,
    allow_partial: bool,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    predictions = load_predictions(Path(spec["logits"]), temperature, sigma)
    ground_truth = load_ground_truth(Path(spec["gt"]))
    coverage = coverage_summary(predictions, ground_truth)
    require_full_coverage(dataset, coverage, allow_partial)

    coarse_rows = sweep_thresholds(predictions, ground_truth, DEFAULT_THRESHOLDS)
    fine_rows = sweep_thresholds(predictions, ground_truth, fine_grid)
    deploy = eval_at_threshold(
        {key: predictions[key] for key in sorted(set(predictions) & set(ground_truth))},
        {key: ground_truth[key] for key in sorted(set(predictions) & set(ground_truth))},
        deploy_threshold,
    )
    return {
        "label": spec["label"],
        "sources": {
            "logits": str(Path(spec["logits"]).relative_to(ROOT)).replace("\\", "/"),
            "ground_truth": str(Path(spec["gt"]).relative_to(ROOT)).replace("\\", "/"),
        },
        "coverage": coverage,
        "deploy_threshold": deploy,
        "coarse": {
            "threshold_count": len(coarse_rows),
            "best": best_threshold(coarse_rows),
            "top_thresholds": rank_rows(coarse_rows)[:top_k],
            "sweep": coarse_rows,
        },
        "fine": {
            "threshold_count": len(fine_rows),
            "best": best_threshold(fine_rows),
            "top_thresholds": rank_rows(fine_rows)[:top_k],
            "sweep": fine_rows,
        },
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = ["dataset", "sweep", "rank", "threshold", "f1", "precision", "recall", "tp", "fp", "fn"]
    rows = []
    for dataset, result in payload["datasets"].items():
        for sweep_name in ("coarse", "fine"):
            for row in rank_rows(result[sweep_name]["sweep"]):
                rows.append({"dataset": dataset, "sweep": sweep_name, **{field: row[field] for field in fields[2:]}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BBC and ClipShots Threshold Sweep",
        "",
        "This report sweeps decision thresholds on cached deploy-checkpoint logits. "
        "Dataset-specific best thresholds are test-set oracle values and should not replace the fixed deploy threshold in headline reporting.",
        "",
        "## Best Thresholds",
        "",
        "| Dataset | Sweep | Best threshold | F1 | Precision | Recall | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        label = result["label"]
        for sweep_name in ("coarse", "fine"):
            best = result[sweep_name]["best"]
            lines.append(
                f"| {label} | {sweep_name} | {best['threshold']:.3f} | {best['f1']:.4f} | "
                f"{best['precision']:.4f} | {best['recall']:.4f} | "
                f"{best['tp']} | {best['fp']} | {best['fn']} |"
            )

    lines += [
        "",
        "## Deploy Threshold Comparison",
        "",
        "| Dataset | Deploy threshold | Deploy F1 | Fine best threshold | Fine best F1 | Delta F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"].values():
        deploy = result["deploy_threshold"]
        best = result["fine"]["best"]
        lines.append(
            f"| {result['label']} | {deploy['threshold']:.3f} | {deploy['f1']:.4f} | "
            f"{best['threshold']:.3f} | {best['f1']:.4f} | {best['f1'] - deploy['f1']:+.4f} |"
        )

    lines += [
        "",
        "## Fine Sweep Top Thresholds",
    ]
    for result in payload["datasets"].values():
        lines += [
            "",
            f"### {result['label']}",
            "",
            "| Rank | Threshold | F1 | Precision | Recall |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in result["fine"]["top_thresholds"]:
            lines.append(
                f"| {row['rank']} | {row['threshold']:.3f} | {row['f1']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="bbc,clipshots")
    parser.add_argument("--temperature", type=float, default=runtime.DEFAULT_TEMPERATURE)
    parser.add_argument("--sigma", type=float, default=runtime.DEFAULT_SIGMA)
    parser.add_argument("--deploy-threshold", type=float, default=runtime.DEFAULT_THRESHOLD)
    parser.add_argument("--fine-min", type=float, default=0.02)
    parser.add_argument("--fine-max", type=float, default=0.20)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    fine_grid = fine_thresholds(args.fine_min, args.fine_max, args.fine_step)
    payload = {
        "schema_version": 1,
        "method": "Dataset-specific test-set threshold sweep on deploy-checkpoint logits",
        "protocol": {
            "temperature": args.temperature,
            "sigma": args.sigma,
            "deploy_threshold": args.deploy_threshold,
            "coarse_thresholds": [float(value) for value in DEFAULT_THRESHOLDS],
            "fine_min": args.fine_min,
            "fine_max": args.fine_max,
            "fine_step": args.fine_step,
        },
        "datasets": {},
    }
    for dataset in datasets:
        payload["datasets"][dataset] = analyze_dataset(
            dataset,
            fine_grid,
            args.deploy_threshold,
            args.temperature,
            args.sigma,
            args.top_k,
            args.allow_partial,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, payload)
    write_markdown(args.output_md, payload)

    print(
        "Threshold sweep complete: "
        + ", ".join(
            f"{result['label']} best={result['fine']['best']['threshold']:.3f} "
            f"F1={result['fine']['best']['f1']:.4f}"
            for result in payload["datasets"].values()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
