"""Regenerate and analyze deploy-checkpoint logits for the paper.

This script closes the reproducibility gap between the headline deploy
checkpoint and the controlled analysis track. It can run inference from the
tracked local deploy checkpoint, reuse existing deploy logits, and then write
bootstrap, calibration, and ClipShots error-analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autoshotv2 import runtime
from autoshotv2.common import classification_metrics, load_logits, normalize_key, scores_from_cache
from autoshotv2.paper_analysis import (
    BOOTSTRAP_SAMPLES,
    MATCH_TOLERANCE,
    SEED,
    bootstrap_confidence_intervals,
    calibration_metrics,
    logits_and_labels,
    per_video_boundary_stats,
    transition_type_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "experiments" / "deploy_regen"
DEFAULT_REPORT_JSON = ROOT / "reports" / "deploy_regen_analysis_results.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "deploy_regen_analysis_summary.md"
DEFAULT_PER_VIDEO_CSV = ROOT / "reports" / "deploy_regen_per_video.csv"
DEFAULT_TOP_ERRORS_CSV = ROOT / "reports" / "deploy_regen_clipshots_top_errors.csv"
DEFAULT_COVERAGE_JSON = ROOT / "reports" / "deploy_regen_coverage_report.json"
DEFAULT_COVERAGE_MD = ROOT / "reports" / "deploy_regen_coverage_report.md"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

DEPLOY_TEMPERATURE = runtime.DEFAULT_TEMPERATURE
DEPLOY_SIGMA = runtime.DEFAULT_SIGMA
DEPLOY_THRESHOLD = runtime.DEFAULT_THRESHOLD

DATASETS = {
    "shot": {
        "label": "SHOT",
        "videos": ROOT / "data" / "ShotData",
        "gt": ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "shot_test_gt.pkl",
        "logits": "shot_test_logits.pkl",
        "results": "results_shot.json",
    },
    "bbc": {
        "label": "BBC",
        "videos": ROOT / "data" / "BBCDataset",
        "gt": ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "bbc_test_gt.pkl",
        "logits": "bbc_test_logits.pkl",
        "results": "results_bbc.json",
    },
    "clipshots": {
        "label": "ClipShots",
        "videos": ROOT / "data" / "ClipShots" / "videos" / "test",
        "gt": ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "clipshots_test_gt.pkl",
        "logits": "clipshots_test_logits.pkl",
        "results": "results_clipshots.json",
    },
}


def run(command: list[str], dry_run: bool = False) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def parse_datasets(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Expected one of {sorted(DATASETS)}")
    return names or list(DATASETS)


def ensure_ground_truth(datasets: list[str], dry_run: bool = False) -> None:
    missing = [name for name in datasets if not DATASETS[name]["gt"].is_file()]
    if not missing:
        return
    run([sys.executable, str(ROOT / "scripts" / "run_journal_study.py"), "--prepare-only"], dry_run=dry_run)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def dataset_videos(args: argparse.Namespace, dataset: str) -> Path:
    override = getattr(args, f"{dataset}_videos")
    return Path(override) if override else Path(DATASETS[dataset]["videos"])


def dataset_logits_cache(args: argparse.Namespace, dataset: str) -> Path | None:
    override = getattr(args, f"{dataset}_logits_cache")
    return Path(override) if override else None


def count_matching_videos(videos_dir: Path, gt_path: Path) -> tuple[int, int, list[str]]:
    ground_truth = load_ground_truth(gt_path)
    available = {
        path.stem
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    wanted = set(ground_truth)
    missing = sorted(wanted - available)
    return len(wanted & available), len(wanted), missing[:10]


def video_source_audit(videos_dir: Path, gt_path: Path) -> dict[str, Any]:
    ground_truth = load_ground_truth(gt_path)
    available = {
        path.stem
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    wanted = set(ground_truth)
    missing = sorted(wanted - available)
    extra = sorted(available - wanted)
    return {
        "path": display_path(videos_dir),
        "exists": videos_dir.is_dir(),
        "video_files": len(available),
        "gt_videos": len(wanted),
        "matched_videos": len(wanted & available),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
    }


def logits_source_audit(path: Path | None, gt_path: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return {
            "path": display_path(path),
            "exists": False,
            "predicted_videos": 0,
            "gt_videos": len(load_ground_truth(gt_path)),
            "matched_videos": 0,
            "missing_count": len(load_ground_truth(gt_path)),
            "extra_count": 0,
            "missing_sample": [],
            "extra_sample": [],
        }
    logits = load_normalized_logits(path)
    ground_truth = load_ground_truth(gt_path)
    coverage = coverage_summary(logits, ground_truth)
    return {
        "path": display_path(path),
        "exists": True,
        "predicted_videos": coverage["predicted_videos"],
        "gt_videos": coverage["gt_videos"],
        "matched_videos": coverage["matched_videos"],
        "missing_count": len(coverage["missing_prediction_keys"]),
        "extra_count": len(coverage["extra_prediction_keys"]),
        "missing_sample": coverage["missing_prediction_keys"][:20],
        "extra_sample": coverage["extra_prediction_keys"][:20],
    }


def write_resource_audit(
    args: argparse.Namespace,
    datasets: list[str],
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "method": "Deploy regeneration prerequisite coverage audit",
        "datasets": {},
    }
    for dataset in datasets:
        spec = DATASETS[dataset]
        out_logits = Path(args.out_dir) / str(spec["logits"])
        override_logits = dataset_logits_cache(args, dataset)
        payload["datasets"][dataset] = {
            "label": spec["label"],
            "ground_truth": display_path(Path(spec["gt"])),
            "video_source": video_source_audit(dataset_videos(args, dataset), Path(spec["gt"])),
            "out_dir_logits": logits_source_audit(out_logits, Path(spec["gt"])),
            "override_logits": logits_source_audit(override_logits, Path(spec["gt"])) if override_logits else None,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_resource_audit_summary(output_md, payload)
    return payload


def write_resource_audit_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Deploy Regeneration Coverage Audit",
        "",
        "This report checks whether local videos and cached logits cover each evaluation ground-truth split before rerunning deploy inference.",
        "",
        "## Video Sources",
        "",
        "| Dataset | Path | Matched | GT | Missing | Extra |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in payload["datasets"].values():
        source = item["video_source"]
        lines.append(
            f"| {item['label']} | `{source['path']}` | {source['matched_videos']} | "
            f"{source['gt_videos']} | {source['missing_count']} | {source['extra_count']} |"
        )

    lines += [
        "",
        "## Cached Logits",
        "",
        "| Dataset | Source | Matched | GT | Missing | Extra |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in payload["datasets"].values():
        for label, source in (("out-dir", item["out_dir_logits"]), ("override", item["override_logits"])):
            if source is None:
                continue
            lines.append(
                f"| {item['label']} | {label}: `{source['path']}` | {source['matched_videos']} | "
                f"{source['gt_videos']} | {source['missing_count']} | {source['extra_count']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_video_source(dataset: str, videos_dir: Path) -> None:
    spec = DATASETS[dataset]
    overlap, total, missing_sample = count_matching_videos(videos_dir, Path(spec["gt"]))
    if overlap == total:
        return
    label = spec["label"]
    sample = ", ".join(missing_sample) if missing_sample else "none"
    raise RuntimeError(
        f"{label} video source does not cover the ground-truth split: "
        f"{overlap}/{total} videos found under {videos_dir}. "
        f"Missing sample: {sample}. "
        f"Pass --{dataset}-videos or --{dataset}-logits-cache with the correct split."
    )


def logits_cache_has_full_coverage(logits_path: Path, gt_path: Path) -> bool:
    audit = logits_source_audit(logits_path, gt_path)
    return bool(
        audit
        and audit["exists"]
        and audit["matched_videos"] == audit["gt_videos"]
        and audit["missing_count"] == 0
    )


def evaluate_dataset(
    dataset: str,
    checkpoint: Path,
    out_dir: Path,
    videos_dir: Path,
    logits_cache: Path | None,
    device: str,
    force_inference: bool,
    dry_run: bool,
) -> None:
    spec = DATASETS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_path = out_dir / str(spec["logits"])
    results_path = out_dir / str(spec["results"])
    command = [
        sys.executable,
        "-m",
        "autoshotv2.eval",
        "--checkpoint",
        str(checkpoint),
        "--gt",
        str(spec["gt"]),
        "--results",
        str(results_path),
        "--temperature",
        str(DEPLOY_TEMPERATURE),
        "--sigma",
        str(DEPLOY_SIGMA),
        "--threshold",
        str(DEPLOY_THRESHOLD),
        "--device",
        device,
    ]
    gt_path = Path(spec["gt"])
    cache_path = None
    if logits_cache is not None and not force_inference:
        cache_path = logits_cache
    elif (
        logits_path.is_file()
        and not force_inference
        and logits_cache_has_full_coverage(logits_path, gt_path)
    ):
        cache_path = logits_path

    if cache_path is not None:
        command.extend(["--logits-cache", str(cache_path)])
    else:
        if not dry_run:
            validate_video_source(dataset, videos_dir)
        command.extend(["--videos-dir", str(videos_dir), "--out-logits", str(logits_path), "--filter-to-gt"])
        if force_inference:
            command.append("--no-resume")
    run(command, dry_run=dry_run)


def load_ground_truth(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {normalize_key(key): np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def load_normalized_logits(path: Path) -> dict[str, np.ndarray]:
    return {normalize_key(key): np.asarray(value) for key, value in load_logits(path).items()}


def resolve_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def coverage_summary(logits: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]) -> dict[str, Any]:
    predicted = set(logits)
    expected = set(ground_truth)
    missing = sorted(expected - predicted)
    extra = sorted(predicted - expected)
    return {
        "predicted_videos": len(predicted),
        "gt_videos": len(expected),
        "matched_videos": len(predicted & expected),
        "missing_prediction_keys": missing,
        "extra_prediction_keys": extra,
    }


def require_full_coverage(dataset: str, coverage: dict[str, Any], allow_partial: bool) -> None:
    if allow_partial or not coverage["missing_prediction_keys"]:
        return
    label = DATASETS[dataset]["label"]
    missing_sample = ", ".join(coverage["missing_prediction_keys"][:10])
    extra_sample = ", ".join(coverage["extra_prediction_keys"][:10])
    raise RuntimeError(
        f"{label} logits do not cover the ground-truth split: "
        f"{coverage['matched_videos']}/{coverage['gt_videos']} matched videos, "
        f"{len(coverage['missing_prediction_keys'])} missing, "
        f"{len(coverage['extra_prediction_keys'])} extra. "
        f"Missing sample: {missing_sample or 'none'}. "
        f"Extra sample: {extra_sample or 'none'}. "
        "Use a matching logits cache/video source, or pass --allow-partial only for debugging."
    )


def per_video_rows(
    dataset: str,
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    keys, stats = per_video_boundary_stats(probabilities, ground_truth, DEPLOY_THRESHOLD)
    rows = []
    for key, (tp, fp, fn) in zip(keys, stats.tolist()):
        metrics = classification_metrics(tp, fp, fn)
        rows.append(
            {
                "dataset": dataset,
                "video": key,
                "frames": int(len(probabilities[key])),
                "gt_boundaries": int(tp + fn),
                "pred_boundaries": int(tp + fp),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            }
        )
    return rows, stats


def bucket_name(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for index, edge in enumerate(edges):
        if value <= edge:
            return labels[index]
    return labels[-1]


def clipshots_error_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_rows = [row for row in rows if row["dataset"] == "clipshots"]
    top_fp = sorted(clip_rows, key=lambda row: (row["fp"], row["pred_boundaries"]), reverse=True)[:20]
    top_fn = sorted(clip_rows, key=lambda row: (row["fn"], row["gt_boundaries"]), reverse=True)[:20]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        tp = sum(int(row["tp"]) for row in selected)
        fp = sum(int(row["fp"]) for row in selected)
        fn = sum(int(row["fn"]) for row in selected)
        return {"videos": len(selected), **classification_metrics(tp, fp, fn)}

    length_groups: dict[str, list[dict[str, Any]]] = {}
    density_groups: dict[str, list[dict[str, Any]]] = {}
    for row in clip_rows:
        length = int(row["frames"])
        density = float(row["gt_boundaries"]) / max(1.0, length / 1000.0)
        length_groups.setdefault(
            bucket_name(length, (6_000, 12_000, 24_000), ("<=6k", "6k-12k", "12k-24k", ">24k")),
            [],
        ).append(row)
        density_groups.setdefault(
            bucket_name(density, (10.0, 25.0, 50.0), ("<=10/1k", "10-25/1k", "25-50/1k", ">50/1k")),
            [],
        ).append(row)
    return {
        "top_false_positive_videos": top_fp,
        "top_false_negative_videos": top_fn,
        "length_buckets": {key: aggregate(value) for key, value in sorted(length_groups.items())},
        "boundary_density_buckets": {key: aggregate(value) for key, value in sorted(density_groups.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    out_dir: Path,
    datasets: list[str],
    output_json: Path,
    summary_md: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    logits_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    ground_truth_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    probabilities_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    bootstrap: dict[str, Any] = {}
    test_calibration: dict[str, Any] = {}
    result_metrics: dict[str, Any] = {}
    coverage_by_dataset: dict[str, Any] = {}
    logits_sources: dict[str, str] = {}
    all_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        spec = DATASETS[dataset]
        logits_path = out_dir / str(spec["logits"])
        results_path = out_dir / str(spec["results"])
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing deploy results for {dataset}: {results_path}")
        result_metrics[dataset] = json.loads(results_path.read_text(encoding="utf-8"))
        if not logits_path.is_file():
            fallback = resolve_existing_path(result_metrics[dataset].get("logits_source"))
            if fallback is None:
                raise FileNotFoundError(
                    f"Missing deploy logits for {dataset}: {logits_path}. "
                    f"Result logits_source is not readable: {result_metrics[dataset].get('logits_source')}"
                )
            logits_path = fallback
        logits_sources[dataset] = display_path(logits_path)
        logits = load_normalized_logits(logits_path)
        ground_truth = load_ground_truth(spec["gt"])
        coverage = coverage_summary(logits, ground_truth)
        require_full_coverage(dataset, coverage, allow_partial)
        coverage_by_dataset[dataset] = {
            **coverage,
            "missing_prediction_count": len(coverage["missing_prediction_keys"]),
            "extra_prediction_count": len(coverage["extra_prediction_keys"]),
            "missing_prediction_keys": coverage["missing_prediction_keys"][:20],
            "extra_prediction_keys": coverage["extra_prediction_keys"][:20],
        }
        common_keys = sorted(set(logits) & set(ground_truth))
        logits = {key: logits[key] for key in common_keys}
        ground_truth = {key: ground_truth[key] for key in common_keys}
        probabilities = {
            normalize_key(key): value
            for key, value in scores_from_cache(
                logits,
                temperature=DEPLOY_TEMPERATURE,
                sigma=DEPLOY_SIGMA,
                input_kind="logits",
            ).items()
        }
        rows, stats = per_video_rows(dataset, probabilities, ground_truth)
        all_rows.extend(rows)
        logits_by_dataset[dataset] = logits
        ground_truth_by_dataset[dataset] = ground_truth
        probabilities_by_dataset[dataset] = probabilities
        bootstrap[dataset] = bootstrap_confidence_intervals(stats)

        values, labels, videos = logits_and_labels(logits, ground_truth)
        test_calibration[dataset] = {
            "videos": videos,
            "before": calibration_metrics(values, labels, temperature=1.0),
            "after_temperature": calibration_metrics(values, labels, temperature=DEPLOY_TEMPERATURE),
        }

    clipshots_breakdown = None
    clipshots_errors = None
    if "clipshots" in datasets:
        clipshots_breakdown = transition_type_breakdown(
            probabilities_by_dataset["clipshots"],
            ground_truth_by_dataset["clipshots"],
            threshold=DEPLOY_THRESHOLD,
        )
        clipshots_errors = clipshots_error_breakdown(all_rows)

    payload = {
        "schema_version": 1,
        "method": "Deploy checkpoint regenerated logits",
        "sources": {
            "checkpoint": "artifacts/models/deploy.pth",
            "logits_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
            "logits": logits_sources,
        },
        "protocol": {
            "temperature": DEPLOY_TEMPERATURE,
            "sigma": DEPLOY_SIGMA,
            "threshold": DEPLOY_THRESHOLD,
            "matching_tolerance_frames": MATCH_TOLERANCE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SEED,
        },
        "datasets": datasets,
        "coverage": coverage_by_dataset,
        "results": result_metrics,
        "bootstrap": bootstrap,
        "test_calibration": test_calibration,
        "clipshots_transition_breakdown": clipshots_breakdown,
        "clipshots_error_analysis": clipshots_errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write_csv(
        DEFAULT_PER_VIDEO_CSV,
        all_rows,
        [
            "dataset",
            "video",
            "frames",
            "gt_boundaries",
            "pred_boundaries",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    if clipshots_errors:
        write_csv(
            DEFAULT_TOP_ERRORS_CSV,
            clipshots_errors["top_false_positive_videos"],
            [
                "dataset",
                "video",
                "frames",
                "gt_boundaries",
                "pred_boundaries",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ],
        )
    write_summary(summary_md, payload)
    return payload


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Deploy Checkpoint Regeneration Analysis",
        "",
        "This report is generated from regenerated or reused deploy-checkpoint logits.",
        "",
        "## Coverage",
        "",
        "| Dataset | Matched | GT | Missing | Extra |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in payload["datasets"]:
        label = DATASETS[dataset]["label"]
        item = payload["coverage"][dataset]
        lines.append(
            f"| {label} | {item['matched_videos']} | {item['gt_videos']} | "
            f"{item['missing_prediction_count']} | {item['extra_prediction_count']} |"
        )
    lines += [
        "",
        "## Deploy Metrics With Bootstrap CI",
        "",
        "| Dataset | F1 | 95% CI | Precision | Recall | Videos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in payload["datasets"]:
        label = DATASETS[dataset]["label"]
        item = payload["bootstrap"][dataset]
        metrics = item["metrics"]
        lines.append(
            f"| {label} | {metrics['f1']['value']:.4f} | "
            f"[{metrics['f1']['ci95_low']:.4f}, {metrics['f1']['ci95_high']:.4f}] | "
            f"{metrics['precision']['value']:.4f} | {metrics['recall']['value']:.4f} | "
            f"{item['n_videos']} |"
        )
    lines += [
        "",
        "## Test Calibration After Deploy Temperature",
        "",
        "| Dataset | NLL | Brier | ECE | Adaptive ECE | Balanced ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in payload["datasets"]:
        label = DATASETS[dataset]["label"]
        metrics = payload["test_calibration"][dataset]["after_temperature"]
        lines.append(
            f"| {label} | {metrics['nll']:.6f} | {metrics['brier']:.6f} | "
            f"{metrics['ece']:.6f} | {metrics['adaptive_ece']:.6f} | "
            f"{metrics['class_balanced_ece']:.6f} |"
        )
    if payload.get("clipshots_transition_breakdown"):
        lines += [
            "",
            "## ClipShots Recall by Transition Type",
            "",
            "| Type | GT | Matched | Missed | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in payload["clipshots_transition_breakdown"]:
            lines.append(
                f"| {row['label']} | {row['ground_truth']} | {row['matched']} | "
                f"{row['missed']} | {row['recall']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "models" / "deploy.pth")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", default="shot,bbc,clipshots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--coverage-json", type=Path, default=DEFAULT_COVERAGE_JSON)
    parser.add_argument("--coverage-md", type=Path, default=DEFAULT_COVERAGE_MD)
    parser.add_argument("--shot-videos", type=Path, default=None)
    parser.add_argument("--bbc-videos", type=Path, default=None)
    parser.add_argument("--clipshots-videos", type=Path, default=None)
    parser.add_argument("--shot-logits-cache", type=Path, default=None)
    parser.add_argument("--bbc-logits-cache", type=Path, default=None)
    parser.add_argument("--clipshots-logits-cache", type=Path, default=None)
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    ensure_ground_truth(datasets, dry_run=args.dry_run)
    if args.coverage_only:
        if args.dry_run:
            return
        audit = write_resource_audit(args, datasets, args.coverage_json, args.coverage_md)
        print(
            "Deploy regeneration coverage audit complete: "
            + ", ".join(
                f"{item['label']} videos={item['video_source']['matched_videos']}/{item['video_source']['gt_videos']}"
                for item in audit["datasets"].values()
            )
        )
        return
    if not args.skip_inference:
        for dataset in datasets:
            evaluate_dataset(
                dataset=dataset,
                checkpoint=args.checkpoint,
                out_dir=args.out_dir,
                videos_dir=dataset_videos(args, dataset),
                logits_cache=dataset_logits_cache(args, dataset),
                device=args.device,
                force_inference=args.force_inference,
                dry_run=args.dry_run,
            )
    if args.dry_run:
        return
    payload = analyze(args.out_dir, datasets, args.output_json, args.summary_md, allow_partial=args.allow_partial)
    print(
        "Deploy regeneration analysis complete: "
        + ", ".join(
            f"{DATASETS[dataset]['label']} F1={payload['bootstrap'][dataset]['metrics']['f1']['value']:.4f}"
            for dataset in datasets
        )
    )


if __name__ == "__main__":
    main()
