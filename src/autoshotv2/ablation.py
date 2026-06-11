import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoshotv2.ablation_report import add_deltas, flatten_metric, make_figures, write_csv, write_summary  # noqa: F401

# clean_key/load_logits/scores_from_cache are re-exported here for backward
# compatibility; canonical home is autoshotv2.common.
from autoshotv2.common import (
    build_train_phase2_command,
    clean_key,
    load_logits,
    load_pickle_payload,
    scores_from_cache,
)
from autoshotv2.eval import DEFAULT_THRESHOLDS, eval_at_threshold
from autoshotv2.train_phase2 import (
    build_sample_cache_config,
    evaluate_best,
    find_temperature,
    gt_for_logits,
    load_metadata,
    logits_to_pred_dict,
    select_training_keys,
    sha256_file,
)

DATASETS = ("shot", "clipshots", "bbc")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    description: str
    kind: str
    source_experiment: str | None = None
    loss: str = "bce"
    manyhot_weight: float = 0.0
    boundary_window: int = 0
    use_ema: bool = False
    ema_decay: float = 0.999
    sigma: float = 0.0
    temperature_mode: str = "off"
    control_id: str | None = "A1_phase2_bce_onehot"


EXPERIMENTS: dict[str, Experiment] = {
    "A0_autoshot_original": Experiment(
        "A0_autoshot_original",
        "Original AutoShot checkpoint without Phase2 head training.",
        "baseline",
        control_id=None,
    ),
    "A1_phase2_bce_onehot": Experiment(
        "A1_phase2_bce_onehot",
        "Minimal Phase2 control: BCE, one-hot only, no EMA.",
        "train",
        loss="bce",
        manyhot_weight=0.0,
        boundary_window=0,
        sigma=0.0,
        temperature_mode="off",
        control_id=None,
    ),
    "A2_focal_only": Experiment(
        "A2_focal_only",
        "Focal loss effect with one-hot labels only.",
        "train",
        loss="focal",
        manyhot_weight=0.0,
        boundary_window=0,
    ),
    "A3_manyhot_only": Experiment(
        "A3_manyhot_only",
        "Many-hot auxiliary target effect with BCE.",
        "train",
        loss="bce",
        manyhot_weight=0.3,
        boundary_window=1,
    ),
    "A4_ema_only": Experiment(
        "A4_ema_only",
        "EMA effect with BCE and one-hot labels only.",
        "train",
        loss="bce",
        manyhot_weight=0.0,
        boundary_window=0,
        use_ema=True,
    ),
    "P1_gaussian_only": Experiment(
        "P1_gaussian_only",
        "Gaussian smoothing effect on A1 logits.",
        "postprocess",
        source_experiment="A1_phase2_bce_onehot",
        sigma=2.0,
        temperature_mode="off",
    ),
    "P2_temperature_only": Experiment(
        "P2_temperature_only",
        "Temperature scaling effect on A1 logits.",
        "postprocess",
        source_experiment="A1_phase2_bce_onehot",
        sigma=0.0,
        temperature_mode="auto",
    ),
    "B1_focal_manyhot": Experiment(
        "B1_focal_manyhot",
        "Focal loss and many-hot auxiliary target combined.",
        "train",
        loss="focal",
        manyhot_weight=0.3,
        boundary_window=1,
    ),
    "B2_focal_ema": Experiment(
        "B2_focal_ema",
        "Focal loss with EMA.",
        "train",
        loss="focal",
        manyhot_weight=0.0,
        boundary_window=0,
        use_ema=True,
    ),
    "B3_manyhot_ema": Experiment(
        "B3_manyhot_ema",
        "Many-hot auxiliary target with EMA.",
        "train",
        loss="bce",
        manyhot_weight=0.3,
        boundary_window=1,
        use_ema=True,
    ),
    "B4_temperature_gaussian": Experiment(
        "B4_temperature_gaussian",
        "Temperature scaling and Gaussian smoothing on A1 logits.",
        "postprocess",
        source_experiment="A1_phase2_bce_onehot",
        sigma=2.0,
        temperature_mode="auto",
    ),
    "B5_full_candidate": Experiment(
        "B5_full_candidate",
        "Focal loss, many-hot target, temperature scaling, and Gaussian smoothing.",
        "train",
        loss="focal",
        manyhot_weight=0.3,
        boundary_window=1,
        sigma=2.0,
        temperature_mode="auto",
    ),
}


DEFAULT_EXPERIMENT_IDS = (
    "A0_autoshot_original",
    "A1_phase2_bce_onehot",
    "A2_focal_only",
    "A3_manyhot_only",
    "P1_gaussian_only",
    "P2_temperature_only",
    "B1_focal_manyhot",
    "B4_temperature_gaussian",
    "B5_full_candidate",
)


def parse_dataset_list(value: str) -> list[str]:
    if value == "all":
        return list(DATASETS)
    datasets = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    return datasets


def parse_experiment_list(value: str) -> list[str]:
    if value == "all":
        return list(DEFAULT_EXPERIMENT_IDS)
    ids = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(ids) - set(EXPERIMENTS))
    if missing:
        raise ValueError(f"Unknown experiments: {missing}")
    ordered = [exp_id for exp_id in EXPERIMENTS if exp_id in ids]
    if any(exp.kind == "postprocess" for exp in (EXPERIMENTS[exp_id] for exp_id in ordered)):
        if "A1_phase2_bce_onehot" not in ordered:
            ordered.insert(0, "A1_phase2_bce_onehot")
    return ordered


def parse_thresholds(value: str) -> list[float]:
    if not value:
        return [float(item) for item in DEFAULT_THRESHOLDS]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_command(cmd: list[str], cwd: Path, continue_on_error: bool) -> tuple[bool, str]:
    print(" ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        message = f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        if continue_on_error:
            return False, message
        raise RuntimeError(message)
    return True, ""


def logits_overlap_gt(logits_path: Path, gt_path: Path) -> bool:
    try:
        logits = load_logits(logits_path)
        with gt_path.open("rb") as f:
            gt = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, ValueError) as exc:
        print(f"WARNING: cannot read {logits_path} or {gt_path}: {exc}", flush=True)
        return False
    pred_keys = {clean_key(key) for key in logits}
    return bool(pred_keys & set(gt))


def prepare_filtered_videos(videos_dir: Path, gt_path: Path, out_dir: Path) -> Path:
    with gt_path.open("rb") as f:
        gt = pickle.load(f)
    wanted_names = {Path(str(key)).stem: str(key) for key in gt}
    needs_gt_suffix_names = any(
        Path(str(key)).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        for key in gt
    )
    source_by_stem = {
        path.stem: path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    }
    available = sorted(set(wanted_names) & set(source_by_stem))
    if not available or (len(source_by_stem) <= len(available) and not needs_gt_suffix_names):
        return videos_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in available:
        src = source_by_stem[stem]
        gt_name = wanted_names[stem]
        dst_name = (
            f"{gt_name}{src.suffix}"
            if Path(gt_name).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
            else src.name
        )
        dst = out_dir / dst_name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
    return out_dir


def prepare_subset_gt(gt_path: Path, out_path: Path, max_videos: int) -> Path:
    if max_videos <= 0:
        return gt_path
    with gt_path.open("rb") as f:
        gt = pickle.load(f)
    keys = sorted(gt)[:max_videos]
    subset = {key: gt[key] for key in keys}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def relocated_video_path(entry: dict[str, Any], repo_dir: Path) -> Path | None:
    source_name = str(entry.get("source_name", ""))
    source_split = str(entry.get("source_split", ""))
    dataset = str(entry.get("dataset", ""))
    if dataset == "clipshots":
        candidate = repo_dir / "data" / "ClipShots" / "videos" / source_split / f"{source_name}.mp4"
        return candidate if candidate.exists() else None
    if dataset == "shot":
        candidates = [
            repo_dir / "data" / "ShotDataset" / f"{source_name}.mp4",
            repo_dir / "data" / "ShotDataset" / "test" / "videos" / f"{source_name}.mp4",
            repo_dir / "data" / "ShotDataset" / "train" / "videos" / f"{source_name}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def relocate_metadata_paths(meta: dict[str, Any], repo_dir: Path, out_path: Path) -> tuple[Path, dict[str, int]]:
    stats = {
        "entries_total": 0,
        "entries_missing_before": 0,
        "entries_relocated": 0,
        "entries_missing_after": 0,
        "shot_test_total": 0,
        "shot_test_missing_before": 0,
        "shot_test_relocated": 0,
        "shot_test_missing_after": 0,
    }
    relocated = pickle.loads(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    for section, prefix in (("entries", "entries"), ("shot_test_entries", "shot_test")):
        for entry in relocated.get(section, {}).values():
            stats[f"{prefix}_total"] += 1
            path = Path(entry["video_path"])
            if path.exists():
                continue
            stats[f"{prefix}_missing_before"] += 1
            candidate = relocated_video_path(entry, repo_dir)
            if candidate is not None:
                entry["video_path"] = str(candidate)
                stats[f"{prefix}_relocated"] += 1
            if not Path(entry["video_path"]).exists():
                stats[f"{prefix}_missing_after"] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(relocated, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path, stats


def evaluate_cached_scores(
    logits_path: Path,
    gt_path: Path,
    temperature: float,
    sigma: float,
    threshold: float,
    thresholds: list[float],
    input_kind: str,
    results_path: Path,
) -> dict[str, Any]:
    scores = load_logits(logits_path)
    with gt_path.open("rb") as f:
        gt = pickle.load(f)
    pred = scores_from_cache(scores, temperature=temperature, sigma=sigma, input_kind=input_kind)
    common_keys = sorted(set(pred) & set(gt))
    common_pred = {key: pred[key] for key in common_keys}
    common_gt = {key: gt[key] for key in common_keys}
    sweep = [eval_at_threshold(common_pred, common_gt, thr) for thr in thresholds]
    best = max(sweep, key=lambda item: item["f1"])
    deploy = eval_at_threshold(common_pred, common_gt, threshold)
    result = {
        "checkpoint": "",
        "logits_source": str(logits_path),
        "postprocess": {
            "temperature": temperature,
            "sigma": sigma,
            "threshold": threshold,
            "input_kind": input_kind,
        },
        "videos_evaluated": len(common_pred),
        "gt_videos": len(gt),
        "missing_prediction_keys": sorted(set(gt) - set(pred)),
        "extra_prediction_keys": sorted(set(pred) - set(gt)),
        "best_sweep": best,
        "deploy": deploy,
        "top_thresholds": sorted(sweep, key=lambda item: item["f1"], reverse=True)[:8],
    }
    write_json(results_path, result)
    return result


def sample_cache_matches(
    cache_path: Path,
    meta_path: Path,
    train_keys: list[str],
    base_ckpt_hash: str,
    args: argparse.Namespace,
    exp: Experiment,
) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = load_pickle_payload(cache_path)
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, ValueError) as exc:
        print(f"WARNING: sample cache {cache_path} unreadable; it will be rebuilt: {exc}", flush=True)
        return False
    selected_keys = select_training_keys(
        train_keys,
        args.data_seed,
        args.max_train_videos,
    )
    cache_args = argparse.Namespace(**vars(args))
    cache_args.boundary_window = exp.boundary_window
    expected = build_sample_cache_config(
        str(meta_path),
        selected_keys,
        base_ckpt_hash,
        cache_args,
    )
    return cached.get("config") == expected


def train_run(
    exp: Experiment,
    run_dir: Path,
    repo_dir: Path,
    meta_path: Path,
    base_ckpt: Path,
    sample_cache: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    ckpt_path = run_dir / "checkpoint.pth"
    if ckpt_path.exists() and not args.force:
        return True, ""

    run_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "--meta": meta_path,
        "--base-ckpt": base_ckpt,
        "--sample-cache": sample_cache,
        "--resume-state": run_dir / "resume.pt",
        "--checkpoint-dir": run_dir / "checkpoints",
        "--out-ckpt": ckpt_path,
        "--results": run_dir / "train_results.pkl",
        "--eval-cache-dir": run_dir / "eval_cache",
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--loss": exp.loss,
        "--manyhot-weight": exp.manyhot_weight,
        "--boundary-window": exp.boundary_window,
        "--sigma": exp.sigma,
        "--temperature-mode": exp.temperature_mode,
        "--max-samples-per-video": args.max_samples_per_video,
        "--max-total-samples": args.max_total_samples,
        "--neg-per-pos": args.neg_per_pos,
        "--min-neg-per-video": args.min_neg_per_video,
        "--seed": args.seed,
        "--data-seed": args.data_seed,
        "--max-train-videos": args.max_train_videos,
        "--max-val-videos": args.max_val_videos,
        "--max-test-videos": args.max_test_videos,
        "--save-every-videos": args.save_every_videos,
        "--save-every-epochs": args.save_every_epochs,
        "--log-every-batches": args.log_every_batches,
        "--stop-after-minutes": args.stop_after_minutes,
        "--max-cache-video-frames": args.max_cache_video_frames,
        "--max-cache-video-seconds": args.max_cache_video_seconds,
        "--data-manifest": run_dir / "training_data_manifest.json",
        "--run-manifest": run_dir / "run_manifest.json",
    }
    extra: list[object] = []
    if exp.use_ema:
        extra.extend(["--use-ema", "--ema-decay", exp.ema_decay])
    if args.no_eval_cache:
        extra.append("--no-eval-cache")
    if args.skip_test_eval:
        extra.append("--skip-test-eval")
    if args.rebuild_sample_cache:
        extra.append("--rebuild-sample-cache")
    if not args.resume_training:
        extra.append("--no-resume")
    cmd = build_train_phase2_command(options, extra)
    return run_command(cmd, repo_dir, args.continue_on_error)


def tune_postprocess(
    run_dir: Path,
    meta: dict[str, Any],
    max_val_videos: int,
    exp: Experiment,
) -> dict[str, Any]:
    val_logits_path = run_dir / "eval_cache" / "combined_val_logits.pkl"
    if not val_logits_path.exists():
        return {
            "temperature": 1.0,
            "threshold": 0.1,
            "val_metric": None,
            "status": "missing_validation_logits",
        }
    logits = load_logits(val_logits_path)
    val_keys = list(meta["val_keys"])
    if max_val_videos > 0:
        val_keys = val_keys[:max_val_videos]
    logits = {key: value for key, value in logits.items() if key in set(val_keys)}
    gt = gt_for_logits(meta["entries"], logits)
    temperature = find_temperature(logits, gt) if exp.temperature_mode == "auto" else 1.0
    pred = logits_to_pred_dict(logits, temperature=temperature, sigma=exp.sigma)
    metric = evaluate_best(pred, gt)
    return {
        "temperature": float(temperature),
        "threshold": float(metric["threshold"]),
        "val_metric": metric,
        "status": "ok",
    }


def resource_candidates(repo_dir: Path, args: argparse.Namespace) -> dict[str, dict[str, Path | None]]:
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else repo_dir / "artifacts" / "experiments" / "published_sweeps"
    )
    return {
        "shot": {
            "gt": Path(args.shot_gt) if args.shot_gt else artifact_root / "gt_scenes_dict_baseline_v2.pickle",
            "videos": Path(args.shot_videos)
            if args.shot_videos
            else repo_dir / "data" / "ShotDataset",
            "logits": Path(args.shot_logits)
            if args.shot_logits
            else artifact_root / "eval_cache_shot_clipshots" / "shot_test_logits.pkl",
        },
        "clipshots": {
            "gt": Path(args.clipshots_gt) if args.clipshots_gt else artifact_root / "clipshots_test_gt_scenes.pickle",
            "videos": Path(args.clipshots_videos)
            if args.clipshots_videos
            else repo_dir / "data" / "ClipShots" / "videos" / "test",
            "logits": Path(args.clipshots_logits)
            if args.clipshots_logits
            else artifact_root / "eval_cache_clipshots" / "clipshot_test_logits.pkl",
        },
        "bbc": {
            "gt": Path(args.bbc_gt) if args.bbc_gt else artifact_root / "bbc_shots_gt_scenes.pickle",
            "videos": Path(args.bbc_videos)
            if args.bbc_videos
            else repo_dir / "data" / "BBCDataset",
            "logits": Path(args.bbc_logits)
            if args.bbc_logits
            else artifact_root / "eval_cache_bbc" / "bbc_test_logits.pkl",
        },
    }


def evaluate_dataset(
    exp: Experiment,
    exp_run_dir: Path,
    source_run_dir: Path,
    checkpoint: Path,
    dataset: str,
    resources: dict[str, dict[str, Path | None]],
    postprocess: dict[str, Any],
    repo_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gt = resources[dataset]["gt"]
    videos = resources[dataset]["videos"]
    if gt is None or not gt.exists():
        return {"status": "missing_gt", "dataset": dataset, "error": str(gt)}
    gt = prepare_subset_gt(gt, exp_run_dir / "_eval_gt" / f"{dataset}_gt.pickle", args.max_eval_videos)

    logits_name = {
        "shot": "shot_test_logits.pkl",
        "clipshots": "clipshots_test_logits.pkl",
        "bbc": "bbc_test_logits.pkl",
    }[dataset]
    source_logits = source_run_dir / "eval_cache" / logits_name
    source_input_kind = "logits"
    out_logits = source_logits
    if dataset == "shot":
        default_train_logits = source_run_dir / "eval_cache" / "shot_test_logits.pkl"
        if default_train_logits.exists():
            source_logits = default_train_logits
            out_logits = default_train_logits
    if exp.kind == "baseline":
        out_logits = exp_run_dir / "eval_cache" / logits_name
        resource_logits = resources[dataset].get("logits")
        if isinstance(resource_logits, Path) and resource_logits.exists():
            source_logits = resource_logits
            source_input_kind = args.artifact_input_kind
        else:
            source_logits = out_logits

    results_path = exp_run_dir / f"results_{dataset}.json"
    if results_path.exists() and not args.force:
        try:
            with results_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if "deploy" in payload and not payload.get("missing_prediction_keys"):
                return {"status": "ok", "dataset": dataset, "result": payload}
        except (OSError, json.JSONDecodeError):
            pass

    cmd = [
        sys.executable,
        "-m",
        "autoshotv2.eval",
        "--checkpoint",
        str(checkpoint),
        "--gt",
        str(gt),
        "--results",
        str(results_path),
        "--temperature",
        str(postprocess["temperature"]),
        "--sigma",
        str(exp.sigma),
        "--threshold",
        str(postprocess["threshold"]),
        "--device",
        args.device,
    ]
    source_logits_usable = source_logits.exists() and logits_overlap_gt(source_logits, gt)
    if source_logits_usable:
        if source_input_kind == "probabilities":
            result = evaluate_cached_scores(
                source_logits,
                gt,
                temperature=float(postprocess["temperature"]),
                sigma=float(exp.sigma),
                threshold=float(postprocess["threshold"]),
                thresholds=args._thresholds,
                input_kind=source_input_kind,
                results_path=results_path,
            )
            return {"status": "ok", "dataset": dataset, "result": result}
        cmd.extend(["--logits-cache", str(source_logits)])
    elif videos is not None and videos.exists():
        videos = prepare_filtered_videos(videos, gt, exp_run_dir / "_eval_videos" / dataset)
        out_logits.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--videos-dir", str(videos), "--out-logits", str(out_logits)])
    else:
        return {
            "status": "missing_logits_or_videos",
            "dataset": dataset,
            "error": f"logits={source_logits}; source_logits_usable={source_logits_usable}; videos={videos}",
        }

    ok, error = run_command(cmd, repo_dir, args.continue_on_error)
    if not ok:
        return {"status": "failed", "dataset": dataset, "error": error}
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {"status": "ok", "dataset": dataset, "result": payload}


def copy_or_select_sample_cache(
    exp: Experiment,
    run_dir: Path,
    meta_path: Path,
    train_keys: list[str],
    base_ckpt_hash: str,
    args: argparse.Namespace,
) -> Path:
    if args.reuse_sample_cache:
        cache_path = Path(args.reuse_sample_cache)
        if sample_cache_matches(cache_path, meta_path, train_keys, base_ckpt_hash, args, exp):
            return cache_path
    return run_dir / "sample_cache.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled AutoShotV2 ablation experiments.")
    parser.add_argument("--meta", default="shot_clipshots_trainval_local.pickle")
    parser.add_argument("--base-ckpt", default="ckpt_0_200_0.pth")
    parser.add_argument("--out-dir", default="artifacts/experiments/ablation_runs")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--artifact-input-kind", choices=["logits", "probabilities"], default="logits")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--no-relocate-missing-paths", action="store_true")
    parser.add_argument("--reuse-sample-cache", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples-per-video", type=int, default=160)
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--min-neg-per-video", type=int, default=32)
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=200)
    parser.add_argument("--max-test-videos", type=int, default=0)
    parser.add_argument("--max-eval-videos", type=int, default=0)
    parser.add_argument("--max-cache-video-frames", type=int, default=180000)
    parser.add_argument("--max-cache-video-seconds", type=float, default=7200.0)
    parser.add_argument("--save-every-videos", type=int, default=25)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--stop-after-minutes", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shot-gt", default="")
    parser.add_argument("--clipshots-gt", default="")
    parser.add_argument("--bbc-gt", default="")
    parser.add_argument("--shot-videos", default="")
    parser.add_argument("--clipshots-videos", default="")
    parser.add_argument("--bbc-videos", default="")
    parser.add_argument("--shot-logits", default="")
    parser.add_argument("--clipshots-logits", default="")
    parser.add_argument("--bbc-logits", default="")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--rebuild-sample-cache", action="store_true")
    parser.add_argument("--no-eval-cache", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]  # src/autoshotv2/ablation.py -> repo root
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta).resolve()
    base_ckpt = Path(args.base_ckpt).resolve()
    datasets = parse_dataset_list(args.datasets)
    experiment_ids = parse_experiment_list(args.experiments)
    args._thresholds = parse_thresholds(args.thresholds)
    resources = resource_candidates(repo_dir, args)
    meta = load_metadata(str(meta_path))
    relocation_stats = None
    if not args.no_relocate_missing_paths:
        relocated_meta_path, relocation_stats = relocate_metadata_paths(
            meta, repo_dir, out_dir / "resolved_meta.pickle"
        )
        meta_path = relocated_meta_path.resolve()
        meta = load_metadata(str(meta_path))
    train_keys = list(meta["train_keys"])
    base_ckpt_hash = sha256_file(str(base_ckpt)) if base_ckpt.exists() else ""
    rows: list[dict[str, Any]] = []

    write_json(
        out_dir / "ablation_run_config.json",
        {
            "args": vars(args),
            "datasets": datasets,
            "experiments": experiment_ids,
            "resources": {k: {kk: str(vv) if vv else "" for kk, vv in v.items()} for k, v in resources.items()},
            "resolved_meta": str(meta_path),
            "relocation_stats": relocation_stats,
        },
    )

    for exp_id in experiment_ids:
        exp = EXPERIMENTS[exp_id]
        run_dir = out_dir / exp.experiment_id
        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        source_run_dir = out_dir / (exp.source_experiment or exp.experiment_id)
        checkpoint = base_ckpt if exp.kind == "baseline" else run_dir / "checkpoint.pth"

        write_json(run_dir / "run_config.json", {"experiment": exp.__dict__, "datasets": datasets})

        if exp.kind == "train":
            sample_cache = copy_or_select_sample_cache(exp, run_dir, meta_path, train_keys, base_ckpt_hash, args)
            ok, error = train_run(exp, run_dir, repo_dir, meta_path, base_ckpt, sample_cache, args)
            if not ok:
                for dataset in datasets:
                    rows.append(
                        flatten_metric(
                            exp, dataset, {"status": "failed", "error": error}, {"temperature": 1.0, "threshold": 0.1}
                        )
                    )
                continue
        elif exp.kind == "postprocess" and not (source_run_dir / "checkpoint.pth").exists():
            error = f"missing source experiment checkpoint: {source_run_dir / 'checkpoint.pth'}"
            for dataset in datasets:
                rows.append(
                    flatten_metric(
                        exp,
                        dataset,
                        {"status": "missing_source", "error": error},
                        {"temperature": 1.0, "threshold": 0.1},
                    )
                )
            continue

        if exp.kind == "postprocess":
            checkpoint = source_run_dir / "checkpoint.pth"
        postprocess = (
            {"temperature": 1.0, "threshold": 0.1, "val_metric": None, "status": "baseline_default"}
            if exp.kind == "baseline"
            else tune_postprocess(
                source_run_dir if exp.kind == "postprocess" else run_dir, meta, args.max_val_videos, exp
            )
        )
        write_json(run_dir / "postprocess_config.json", postprocess)

        for dataset in datasets:
            eval_payload = evaluate_dataset(
                exp,
                run_dir,
                source_run_dir if exp.kind == "postprocess" else run_dir,
                checkpoint,
                dataset,
                resources,
                postprocess,
                repo_dir,
                args,
            )
            rows.append(flatten_metric(exp, dataset, eval_payload, postprocess))

    add_deltas(rows)
    write_csv(out_dir / "ablation_results.csv", rows)
    write_json(out_dir / "ablation_results.json", rows)
    make_figures(out_dir, rows)
    write_summary(out_dir, rows)
    print(f"Ablation outputs saved -> {out_dir}")


if __name__ == "__main__":
    main()
