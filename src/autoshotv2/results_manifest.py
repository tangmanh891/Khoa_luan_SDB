"""Build the consolidated experiment manifest from raw result JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PUBLICATIONS = ROOT / "publications"
THESIS_GENERATED = PUBLICATIONS / "thesis" / "generated"
PAPER_GENERATED = PUBLICATIONS / "paper" / "generated"

DATASET_ORDER = ("shot", "bbc", "clipshots")
DATASET_LABELS = {
    "shot": "SHOT",
    "bbc": "BBC",
    "clipshots": "ClipShots",
}

ABLATION_ORDER = (
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

ABLATION_LABELS = {
    "A0_autoshot_original": "A0 -- AutoShot gốc",
    "A1_phase2_bce_onehot": "A1 -- BCE + one-hot",
    "A2_focal_only": "A2 -- Focal only",
    "A3_manyhot_only": "A3 -- Many-hot only",
    "P1_gaussian_only": "P1 -- Gaussian only",
    "P2_temperature_only": "P2 -- Temperature only",
    "B1_focal_manyhot": "B1 -- Focal + many-hot",
    "B4_temperature_gaussian": "B4 -- Temperature + Gaussian",
    "B5_full_candidate": "B5 -- Full candidate no-EMA",
}

ABLATION_NOTES = {
    "A0_autoshot_original": "AutoShot gốc, best threshold từng dataset.",
    "A1_phase2_bce_onehot": "Control Phase2 tối giản.",
    "A2_focal_only": "Chỉ thêm Focal Loss.",
    "A3_manyhot_only": "Chỉ thêm nhãn many-hot.",
    "P1_gaussian_only": "Tăng ClipShots nhưng giảm BBC.",
    "P2_temperature_only": "Gần như trùng control.",
    "B1_focal_manyhot": "Tác động train-time nhỏ.",
    "B4_temperature_gaussian": "Cấu hình hậu xử lý được ưu tiên.",
    "B5_full_candidate": "Focal + many-hot + hậu xử lý.",
}

CALIBRATION_LABELS = {
    "A0_autoshot_baseline": "A0 -- AutoShot baseline",
    "A1_phase2_control": "A1 -- Phase2 control",
    "B5_phase2_full": "B5 -- Phase2 đầy đủ",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_block(block: dict[str, Any], videos: int | None = None) -> dict[str, Any]:
    result = {
        key: block[key]
        for key in ("f1", "precision", "recall", "tp", "fp", "fn", "threshold")
        if key in block
    }
    if videos is not None:
        result["videos"] = videos
    return result


def build_deploy_experiments() -> list[dict[str, Any]]:
    paths = {
        "shot": REPORTS / "deploy_results" / "inference_results.json",
        "bbc": REPORTS / "deploy_results" / "bbc_shot_inference_results.json",
        "clipshots": REPORTS / "deploy_results" / "clipshot_test_inference_results.json",
    }
    payloads = {dataset: load_json(path) for dataset, path in paths.items()}
    experiments = []
    for mode, identifier, label, note in (
        (
            "deploy",
            "phase2_deploy_threshold",
            "Phase2 deploy checkpoint, deploy threshold",
            "T=0.3878, sigma=2.0, threshold=0.10.",
        ),
        (
            "best_sweep",
            "phase2_best_sweep",
            "Phase2 deploy checkpoint, best sweep",
            "ClipShots tốt nhất tại threshold 0.15; SHOT/BBC trùng deploy.",
        ),
    ):
        metrics = {
            dataset: metric_block(payload[mode], payload["videos_evaluated"])
            for dataset, payload in payloads.items()
        }
        experiments.append(
            {
                "id": identifier,
                "group": "Deploy checkpoint",
                "label": label,
                "protocol": mode,
                "reproducibility": "result_json",
                "sources": [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths.values()],
                "metrics": metrics,
                "note": note,
            }
        )
    return experiments


def build_ablation_experiments() -> list[dict[str, Any]]:
    source = REPORTS / "source_results" / "ablation_results.json"
    rows = load_json(source)
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            raise ValueError(f"Ablation row is not successful: {row['experiment_id']}/{row['dataset']}")
        by_id.setdefault(row["experiment_id"], {})[row["dataset"]] = row

    experiments = []
    for experiment_id in ABLATION_ORDER:
        dataset_rows = by_id.get(experiment_id, {})
        missing = set(DATASET_ORDER) - set(dataset_rows)
        if missing:
            raise ValueError(f"{experiment_id} is missing datasets: {sorted(missing)}")
        metrics = {}
        for dataset in DATASET_ORDER:
            row = dataset_rows[dataset]
            metrics[dataset] = metric_block(row)
        experiments.append(
            {
                "id": experiment_id,
                "group": "Ablation no-EMA",
                "label": ABLATION_LABELS[experiment_id],
                "protocol": "controlled_phase2",
                "reproducibility": "logits",
                "sources": ["reports/source_results/ablation_results.json"],
                "metrics": metrics,
                "note": ABLATION_NOTES[experiment_id],
            }
        )
    return experiments


def build_calibration_experiments() -> list[dict[str, Any]]:
    source = REPORTS / "postprocess_calibration_results.json"
    payload = load_json(source)
    rows = {
        (row["model"], row["dataset"]): row
        for row in payload["results"]
    }
    experiments = []
    for model in CALIBRATION_LABELS:
        cv_metrics = {}
        ceiling_metrics = {}
        for dataset in DATASET_ORDER:
            row = rows[(model, dataset)]
            cv_metrics[dataset] = metric_block(row["cv_deploy"], row["n_videos"])
            ceiling_metrics[dataset] = metric_block(row["test_ceiling"], row["n_videos"])

        experiments.append(
            {
                "id": f"calibration_cv_{model}",
                "group": "Calibration CV",
                "label": f"{CALIBRATION_LABELS[model]}, 5-fold CV",
                "protocol": "five_fold_cross_validation",
                "reproducibility": "logits",
                "sources": ["reports/postprocess_calibration_results.json"],
                "metrics": cv_metrics,
                "note": (
                    "Honest 5-fold CV; baseline mạnh nhất trên ClipShots."
                    if model == "A0_autoshot_baseline"
                    else "Honest 5-fold CV; Phase2 mạnh trên SHOT/BBC."
                ),
            }
        )
        experiments.append(
            {
                "id": f"calibration_ceiling_{model}",
                "group": "Calibration ceiling",
                "label": f"{CALIBRATION_LABELS[model]}, tune trên test",
                "protocol": "test_tuned_ceiling",
                "reproducibility": "logits",
                "sources": ["reports/postprocess_calibration_results.json"],
                "metrics": ceiling_metrics,
                "note": "Mức trần lạc quan; không dùng làm deploy honest.",
            }
        )
    return experiments


def ema_metric(dataset: str, payload: dict[str, Any], key: str) -> dict[str, Any]:
    if dataset == "shot":
        model = payload["models"][key]["best_threshold"]
        videos = payload["models"][key]["n_videos"]
        return metric_block(model, videos)
    if dataset == "bbc":
        model = payload["shot_level"][key]
        return {
            "f1": model["f1_best"],
            "precision": model["p_best"],
            "recall": model["r_best"],
            "threshold": model["thr_best"],
            "videos": model["n_videos"],
        }
    model = payload["models"][key]
    return {
        "f1": model["f1_best"],
        "precision": model["p_best"],
        "recall": model["r_best"],
        "threshold": model["thr_best"],
        "videos": model["n_videos"],
    }


def build_ema_experiments() -> list[dict[str, Any]]:
    alpha_paths = {
        "shot": REPORTS / "ema_study" / "results_autoshot_alpha999.json",
        "bbc": REPORTS / "ema_study" / "results_bbc_alpha999.json",
        "clipshots": REPORTS / "ema_study" / "results_clipshots_alpha999.json",
    }
    noema_paths = {
        "shot": REPORTS / "ema_study" / "results_autoshot_noema.json",
        "bbc": REPORTS / "ema_study" / "results_bbc_noema.json",
        "clipshots": REPORTS / "ema_study" / "results_clipshots_noema.json",
    }
    alpha = {dataset: load_json(path) for dataset, path in alpha_paths.items()}
    noema = {dataset: load_json(path) for dataset, path in noema_paths.items()}
    definitions = (
        (
            "ema_phase1_raw",
            "A -- Phase 1 raw",
            alpha,
            {"shot": "autoshot_phase1_raw", "bbc": "A_phase1_raw", "clipshots": "A_phase1_raw"},
            "Baseline raw trong EMA study.",
        ),
        (
            "ema_phase1_gaussian",
            "B -- Phase 1 + Gaussian",
            alpha,
            {
                "shot": "autoshot_phase1_gaussian",
                "bbc": "B_phase1_gauss",
                "clipshots": "B_phase1_gauss",
            },
            "Gaussian hậu xử lý rất mạnh trên ClipShots.",
        ),
        (
            "ema_full_model_alpha999",
            "D-EMA, alpha=0.999",
            alpha,
            {"shot": "autoshot_ema", "bbc": "D_ema", "clipshots": "D_ema"},
            "Fine-tune toàn model + EMA.",
        ),
        (
            "ema_full_model_noema",
            "D-noEMA control",
            noema,
            {"shot": "autoshot_ema", "bbc": "D_ema", "clipshots": "D_ema"},
            "Fine-tune toàn model, tắt EMA.",
        ),
    )
    sources = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (*alpha_paths.values(), *noema_paths.values())
    ]
    experiments = []
    for identifier, label, payloads, keys, note in definitions:
        metrics = {
            dataset: ema_metric(dataset, payloads[dataset], keys[dataset])
            for dataset in DATASET_ORDER
        }
        experiments.append(
            {
                "id": identifier,
                "group": "EMA full fine-tune",
                "label": label,
                "protocol": "full_model_finetune",
                "reproducibility": "result_json",
                "sources": sources,
                "metrics": metrics,
                "note": note,
            }
        )
    return experiments


def build_manifest() -> dict[str, Any]:
    literature_path = REPORTS / "literature_results.json"
    literature = load_json(literature_path)["comparison_models"]
    deployment_config = load_json(
        REPORTS / "deploy_results" / "inference_results.json"
    )["postprocess"]
    clipshots_breakdown = load_json(
        REPORTS / "source_results" / "clipshots_transition_breakdown.json"
    )
    paper_analysis = load_json(REPORTS / "paper_analysis_results.json")
    experiments = (
        build_deploy_experiments()
        + build_ablation_experiments()
        + build_calibration_experiments()
        + build_ema_experiments()
    )
    deploy_best = next(item for item in experiments if item["id"] == "phase2_best_sweep")
    comparison_models = literature + [
        {
            "id": "autoshotv2_deploy",
            "name": "AutoShotV2",
            "source_kind": "result_json",
            "source": "reports/deploy_results/*.json",
            "metrics": {
                dataset: deploy_best["metrics"][dataset]["f1"]
                for dataset in DATASET_ORDER
            },
        }
    ]
    return {
        "schema_version": 1,
        "snapshot_date": "2026-06-07",
        "dataset_order": list(DATASET_ORDER),
        "dataset_labels": DATASET_LABELS,
        "deployment_config": deployment_config,
        "comparison_models": comparison_models,
        "experiments": experiments,
        "supplemental_results": {
            "clipshots_transition_breakdown": clipshots_breakdown,
            "paper_analysis": paper_analysis,
        },
    }
