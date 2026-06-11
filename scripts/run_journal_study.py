"""Run the three-seed validation-frozen AutoShotV2 journal study."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autoshotv2.common import build_train_phase2_command, load_logits, normalize_key
from autoshotv2.journal_protocol import load_frozen_protocol
from autoshotv2.train_phase2 import transitions_to_scenes

ROOT = Path(__file__).resolve().parents[1]
SHOT_GT_JSON = ROOT / "reports" / "source_results" / "shot_test_ground_truth.json"
BBC_ANNOTATIONS = ROOT / "data" / "BBCDataset" / "annotations" / "shots"
CLIPSHOTS_ANNOTATIONS = ROOT / "data" / "ClipShots" / "annotations" / "test.json"
CLIPSHOTS_REFERENCE_LOGITS = (
    ROOT
    / "artifacts"
    / "experiments"
    / "ablation_full"
    / "A1_phase2_bce_onehot"
    / "eval_cache"
    / "clipshots_test_logits.pkl"
)


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def dataset_resources(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    return {
        "shot": (args.shot_gt, args.shot_videos),
        "bbc": (args.bbc_gt, args.bbc_videos),
        "clipshots": (args.clipshots_gt, args.clipshots_videos),
    }


def training_command(
    args: argparse.Namespace,
    seed: int,
    shared: Path,
    run_dir: Path,
    checkpoint: Path,
) -> list[str]:
    """Frozen training configuration for the journal study (do not vary)."""
    options = {
        "--meta": args.meta,
        "--base-ckpt": args.base_ckpt,
        "--out-ckpt": checkpoint,
        "--sample-cache": shared / "sample_cache.pkl",
        "--results": run_dir / "train_results.pkl",
        "--eval-cache-dir": run_dir / "eval_cache",
        "--resume-state": run_dir / "resume.pt",
        "--checkpoint-dir": run_dir / "checkpoints",
        "--data-manifest": run_dir / "training_data_manifest.json",
        "--run-manifest": run_dir / "run_manifest.json",
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--loss": "bce",
        "--manyhot-weight": "0",
        "--boundary-window": "0",
        "--temperature-mode": "off",
        "--sigma": "0",
        "--seed": seed,
        "--data-seed": args.data_seed,
        "--max-train-videos": args.max_train_videos,
        "--max-val-videos": args.max_val_videos,
    }
    extra = ["--skip-test-eval"]
    if args.force:
        extra.extend(["--no-resume", "--no-eval-cache"])
    return build_train_phase2_command(options, extra)


def materialize_ground_truth(args: argparse.Namespace) -> None:
    sources = {
        "SHOT JSON ground truth": SHOT_GT_JSON,
        "BBC annotation directory": BBC_ANNOTATIONS,
        "ClipShots annotations": CLIPSHOTS_ANNOTATIONS,
        "ClipShots reference logits": CLIPSHOTS_REFERENCE_LOGITS,
    }
    missing = [
        f"{label}: {path}"
        for label, path in sources.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot materialize journal ground truth:\n- "
            + "\n- ".join(missing)
        )

    for output in (args.shot_gt, args.bbc_gt, args.clipshots_gt):
        output.parent.mkdir(parents=True, exist_ok=True)
    if not args.shot_gt.is_file():
        payload = json.loads(SHOT_GT_JSON.read_text(encoding="utf-8"))
        scenes = {
            normalize_key(key): np.asarray(value, dtype=np.int64).reshape(-1, 2)
            for key, value in payload["scenes"].items()
        }
        with args.shot_gt.open("wb") as handle:
            pickle.dump(scenes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.bbc_gt.is_file():
        scenes = {}
        for path in sorted(BBC_ANNOTATIONS.glob("*.txt")):
            episode = int(path.name[:2])
            scenes[f"bbc_{episode:02d}"] = np.loadtxt(
                path,
                dtype=np.int64,
            ).reshape(-1, 2)
        with args.bbc_gt.open("wb") as handle:
            pickle.dump(scenes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.clipshots_gt.is_file():
        annotations = json.loads(CLIPSHOTS_ANNOTATIONS.read_text(encoding="utf-8"))
        normalized_annotations = {
            normalize_key(key): value
            for key, value in annotations.items()
        }
        reference_logits = {
            normalize_key(key): np.asarray(value)
            for key, value in load_logits(CLIPSHOTS_REFERENCE_LOGITS).items()
        }
        scenes = {}
        for key, logits in reference_logits.items():
            if key not in normalized_annotations:
                continue
            transitions = np.asarray(
                normalized_annotations[key]["transitions"],
                dtype=np.int64,
            ).reshape(-1, 2)
            scenes[key] = transitions_to_scenes(
                np.sort(transitions, axis=1),
                len(logits),
            )
        with args.clipshots_gt.open("wb") as handle:
            pickle.dump(scenes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "schema_version": 1,
        "sources": {label: str(path) for label, path in sources.items()},
        "outputs": {
            "shot": str(args.shot_gt),
            "bbc": str(args.bbc_gt),
            "clipshots": str(args.clipshots_gt),
        },
    }
    (args.shot_gt.parent / "ground_truth_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def aggregate_results(out_dir: Path, seeds: list[int], data_seed: int) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        run_dir = out_dir / f"seed_{seed}"
        for dataset in ("shot", "bbc", "clipshots"):
            path = run_dir / f"results_{dataset}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            by_dataset.setdefault(dataset, []).append(
                {"seed": seed, **payload["deploy"]}
            )

    summary = {}
    for dataset, rows in by_dataset.items():
        summary[dataset] = {
            "runs": rows,
            "n_seeds": len(rows),
            "mean": {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("f1", "precision", "recall")
            },
            "sample_std": {
                metric: (
                    float(np.std([row[metric] for row in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0
                )
                for metric in ("f1", "precision", "recall")
            },
        }
    payload = {
        "schema_version": 1,
        "study": "AutoShotV2 journal reproducibility study",
        "training_seeds": seeds,
        "data_seed": data_seed,
        "datasets": summary,
    }
    (out_dir / "journal_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "ablation_full" / "resolved_meta.pickle",
    )
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=ROOT / "artifacts" / "models" / "ckpt_0_200_0.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "journal_study",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-train-videos", type=int, default=400)
    parser.add_argument("--max-val-videos", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shot-gt",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "shot_test_gt.pkl",
    )
    parser.add_argument(
        "--clipshots-gt",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "clipshots_test_gt.pkl",
    )
    parser.add_argument(
        "--bbc-gt",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "journal_study" / "shared" / "bbc_test_gt.pkl",
    )
    parser.add_argument(
        "--shot-videos",
        type=Path,
        default=ROOT / "data" / "ShotDataset",
    )
    parser.add_argument(
        "--clipshots-videos",
        type=Path,
        default=ROOT / "data" / "ClipShots" / "videos" / "test",
    )
    parser.add_argument(
        "--bbc-videos",
        type=Path,
        default=ROOT / "data" / "BBCDataset",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize and validate evaluation resources without training.",
    )
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not args.dry_run:
        materialize_ground_truth(args)
    required_files = {
        "metadata": args.meta,
        "original AutoShot checkpoint": args.base_ckpt,
        "SHOT ground truth": args.shot_gt,
        "BBC ground truth": args.bbc_gt,
        "ClipShots ground truth": args.clipshots_gt,
    }
    required_directories = {
        "SHOT videos": args.shot_videos,
        "BBC videos": args.bbc_videos,
        "ClipShots videos": args.clipshots_videos,
    }
    if not args.dry_run:
        missing = [
            f"{label}: {path}"
            for label, path in required_files.items()
            if not path.is_file()
        ]
        missing.extend(
            f"{label}: {path}"
            for label, path in required_directories.items()
            if not path.is_dir()
        )
        if missing:
            raise FileNotFoundError(
                "Journal study prerequisites are missing:\n- "
                + "\n- ".join(missing)
            )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.prepare_only:
            print("Journal study resources prepared and validated.")
            return
    shared = args.out_dir / "shared"
    if not args.dry_run:
        shared.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = args.out_dir / f"seed_{seed}"
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "checkpoint.pth"
        validation_logits = run_dir / "eval_cache" / "combined_val_logits.pkl"
        protocol_path = run_dir / "frozen_protocol.json"

        if args.force or not checkpoint.is_file():
            run(training_command(args, seed, shared, run_dir, checkpoint), args.dry_run)

        if args.force or not protocol_path.is_file():
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "select_journal_protocol.py"),
                    "--meta",
                    str(args.meta),
                    "--validation-logits",
                    str(validation_logits),
                    "--output",
                    str(protocol_path),
                    *(["--overwrite"] if args.force else []),
                ],
                args.dry_run,
            )

        if args.dry_run:
            continue
        protocol = load_frozen_protocol(protocol_path)["selected"]
        for dataset, (ground_truth, videos) in dataset_resources(args).items():
            results = run_dir / f"results_{dataset}.json"
            logits = run_dir / "eval_cache" / f"{dataset}_test_logits.pkl"
            if results.is_file() and not args.force:
                continue
            command = [
                sys.executable,
                "-m",
                "autoshotv2.eval",
                "--checkpoint",
                str(checkpoint),
                "--gt",
                str(ground_truth),
                "--results",
                str(results),
                "--temperature",
                str(protocol["temperature"]),
                "--sigma",
                str(protocol["sigma"]),
                "--threshold",
                str(protocol["threshold"]),
                "--device",
                args.device,
            ]
            if logits.is_file() and not args.force:
                command.extend(["--logits-cache", str(logits)])
            else:
                command.extend(["--videos-dir", str(videos), "--out-logits", str(logits)])
            run(command, False)

    if not args.dry_run:
        result = aggregate_results(args.out_dir, seeds, args.data_seed)
        print(
            "Journal study complete: "
            + ", ".join(
                f"{dataset} F1={values['mean']['f1']:.4f}"
                for dataset, values in result["datasets"].items()
            )
        )


if __name__ == "__main__":
    main()
