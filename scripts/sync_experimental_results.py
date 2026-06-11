"""Build the consolidated experiment manifest and publication tables.

The thesis and slide generator consume only the generated consolidated data.
Raw result JSON files remain the audit trail and are checked on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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
    experiments = (
        build_deploy_experiments()
        + build_ablation_experiments()
        + build_calibration_experiments()
        + build_ema_experiments()
    )
    deploy_fixed = next(item for item in experiments if item["id"] == "phase2_deploy_threshold")
    comparison_models = literature + [
        {
            "id": "autoshotv2_deploy",
            "name": "AutoShotV2",
            "source_kind": "result_json",
            "source": "reports/deploy_results/*.json",
            "metrics": {
                "shot": deploy_fixed["metrics"]["shot"]["f1"],
                "bbc": deploy_fixed["metrics"]["bbc"]["f1"],
                "clipshots": deploy_fixed["metrics"]["clipshots"]["f1"],
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
        },
    }


def f4(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def f3(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def experiment_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in manifest["experiments"]}


def comparison_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in manifest["comparison_models"]}


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Bảng Tổng Hợp Kết Quả Thực Nghiệm AutoShotV2",
        "",
        "File này được sinh từ `reports/experimental_results.json` bằng",
        "`python scripts/sync_experimental_results.py --write`. Không sửa trực tiếp.",
        "",
        "Các nhóm dùng protocol khác nhau; chỉ so sánh trực tiếp trong cùng nhóm.",
        "",
        "## Bảng Tất Cả Thực Nghiệm",
        "",
        "| Nhóm | Thực nghiệm / protocol | SHOT | BBC | ClipShots | Ghi chú |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in manifest["experiments"]:
        metrics = item["metrics"]
        lines.append(
            "| {group} | `{identifier}` - {label} | {shot} | {bbc} | {clip} | {note} |".format(
                group=item["group"],
                identifier=item["id"],
                label=item["label"],
                shot=f4(metrics["shot"]["f1"]),
                bbc=f4(metrics["bbc"]["f1"]),
                clip=f4(metrics["clipshots"]["f1"]),
                note=item["note"].replace("|", "\\|"),
            )
        )

    by_id = experiment_map(manifest)
    lines += [
        "",
        "## Kết Quả Checkpoint Deploy",
        "",
        "| Dataset | Chế độ | Threshold | F1 | Precision | Recall | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id in ("phase2_deploy_threshold", "phase2_best_sweep"):
        item = by_id[experiment_id]
        for dataset in DATASET_ORDER:
            metric = item["metrics"][dataset]
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {item['protocol']} | "
                f"{f4(metric.get('threshold'))} | {f4(metric['f1'])} | "
                f"{f4(metric['precision'])} | {f4(metric['recall'])} | "
                f"{metric['tp']} | {metric['fp']} | {metric['fn']} |"
            )

    lines += [
        "",
        "## Nguồn Và Mức Tái Lập",
        "",
        "- `result_json`: tái lập ở mức đọc lại JSON đã chốt hoặc chạy lại từ checkpoint/dataset.",
        "- `logits`: có thể tính lại metric từ logits và ground truth khi artifact tương ứng có sẵn.",
        "- `literature` và `legacy_result`: chỉ dùng trong bảng so sánh, không được coi là run mới.",
        "",
    ]
    return "\n".join(lines)


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def tex_value(value: float | None, bold: bool = False) -> str:
    rendered = "chưa chạy" if value is None else f"{value:.4f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def render_tex_macros(manifest: dict[str, Any]) -> str:
    experiments = experiment_map(manifest)
    comparisons = comparison_map(manifest)
    deploy = experiments["phase2_deploy_threshold"]["metrics"]
    best = experiments["phase2_best_sweep"]["metrics"]
    a0 = experiments["A0_autoshot_original"]["metrics"]
    b4 = experiments["B4_temperature_gaussian"]["metrics"]
    ema = experiments["ema_full_model_alpha999"]["metrics"]
    noema = experiments["ema_full_model_noema"]["metrics"]
    transnet = comparisons["transnetv2_reported"]["metrics"]
    definitions = {
        "ASVTwoShotFOne": f4(deploy["shot"]["f1"]),
        "ASVTwoBBCFOne": f4(deploy["bbc"]["f1"]),
        "ASVTwoClipFOne": f4(deploy["clipshots"]["f1"]),
        "ASVTwoClipDeployFOne": f4(deploy["clipshots"]["f1"]),
        "ASVTwoClipBestFOne": f4(best["clipshots"]["f1"]),
        "ASVTwoShotPrecision": f4(deploy["shot"]["precision"]),
        "ASVTwoShotRecall": f4(deploy["shot"]["recall"]),
        "AutoShotOriginalShotFOne": f4(a0["shot"]["f1"]),
        "BFourShotFOne": f4(b4["shot"]["f1"]),
        "BFourBBCFOne": f4(b4["bbc"]["f1"]),
        "BFourClipFOne": f4(b4["clipshots"]["f1"]),
        "EMAShotFOne": f4(ema["shot"]["f1"]),
        "EMABBCFOne": f4(ema["bbc"]["f1"]),
        "EMAClipFOne": f4(ema["clipshots"]["f1"]),
        "EMAShotDelta": f"{(ema['shot']['f1'] - noema['shot']['f1']) * 100:+.2f}\\%",
        "EMABBCDelta": f"{(ema['bbc']['f1'] - noema['bbc']['f1']) * 100:+.2f}\\%",
        "EMAClipDelta": f"{(ema['clipshots']['f1'] - noema['clipshots']['f1']) * 100:+.2f}\\%",
        "TransNetShotFOne": f4(transnet["shot"]),
        "ASVTwoVsTransNetShotPP": f"{(deploy['shot']['f1'] - transnet['shot']) * 100:.1f}",
        "ASVTwoVsAutoShotShotPP": f"{(deploy['shot']['f1'] - a0['shot']['f1']) * 100:.1f}",
    }
    lines = [
        "% Generated by scripts/sync_experimental_results.py. Do not edit.",
    ]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in definitions.items())
    lines.append("")
    return "\n".join(lines)


def render_tex_tables(manifest: dict[str, Any]) -> str:
    comparisons = comparison_map(manifest)
    experiments = experiment_map(manifest)
    lines = ["% Generated by scripts/sync_experimental_results.py. Do not edit.", ""]

    overview_rows = [
        "AutoShot & 0.8412 & 0.9554 & 0.7648 & Phiên bản gốc \\\\",
        "AutoShotV2 & 0.8545 & 0.9656 & "
        f"{f4(experiments['phase2_deploy_threshold']['metrics']['clipshots']['f1'])} & Phiên bản chính \\\\",
        r"AutoShotV2-SHOT & \textbf{0.8607} & 0.9648 & 0.7281  & Tốt nhất cho SHOT \\",
        r"AutoShotV2-BBC & 0.8545 & \textbf{0.9656} & 0.7530 & Tốt nhất cho BBC \\",
        r"AutoShotV2-ClipShots & 0.841 & 0.9529 & \textbf{0.7706} & Tốt nhất cho ClipShots \\",
    ]
    lines += [r"\newcommand{\ExperimentOverviewRows}{%", *overview_rows, "}", ""]

    rows = [
        "AutoShot được báo cáo & 0.8410 & 0.9710 & 0.7870 \\\\",
        "AutoShot tái thực nghiệm & 0.8412 & 0.9554 & 0.7648 \\\\",
        "TransNetV2 & 0.7993 & 0.9620 & 0.7760 \\\\",
        "TransNetV2 tái thực nghiệm & chưa thực hiện & 0.9593 & 0.7454 \\\\",
        "AutoShotV1 & 0.8480 & chưa thực hiện & chưa thực hiện \\\\",
        f"AutoShotV2 & 0.8545 & 0.9656 & {f4(comparisons['autoshotv2_deploy']['metrics']['clipshots'])} \\\\",
    ]
    lines += [r"\newcommand{\MainComparisonRows}{%", *rows, "}", ""]

    deploy = experiments["phase2_deploy_threshold"]["metrics"]
    best = experiments["phase2_best_sweep"]["metrics"]
    rows = []
    for dataset in ("shot", "bbc", "clipshots"):
        metric = deploy[dataset]
        threshold = 0.10
        rows.append(
            f"{DATASET_LABELS[dataset]} & {threshold:.2f} & "
            f"{metric['f1']:.4f} & {metric['precision']:.4f} & "
            f"{metric['recall']:.4f} \\\\"
        )
    lines += [r"\newcommand{\DeployPRFRows}{%", *rows, "}", ""]

    rows = []
    for identifier in ABLATION_ORDER:
        item = experiments[identifier]
        metrics = item["metrics"]
        values = [
            tex_value(metrics["shot"]["f1"], identifier == "B4_temperature_gaussian"),
            tex_value(metrics["bbc"]["f1"], identifier == "B4_temperature_gaussian"),
            tex_value(metrics["clipshots"]["f1"], identifier == "B4_temperature_gaussian"),
        ]
        rows.append(
            f"{tex_escape(item['label'])} & {values[0]} & {values[1]} & {values[2]} \\\\"
        )
    lines += [r"\newcommand{\AblationRows}{%", *rows, "}", ""]

    ema_order = (
        "ema_phase1_raw",
        "ema_phase1_gaussian",
        "ema_full_model_noema",
        "ema_full_model_alpha999",
    )
    rows = []
    for identifier in ema_order:
        item = experiments[identifier]
        metrics = item["metrics"]
        rows.append(
            f"{tex_escape(item['label'])} & "
            f"{tex_value(metrics['shot']['f1'], identifier == 'ema_phase1_gaussian')} & "
            f"{tex_value(metrics['bbc']['f1'], identifier == 'ema_full_model_alpha999')} & "
            f"{tex_value(metrics['clipshots']['f1'], identifier == 'ema_phase1_gaussian')} \\\\"
        )
    lines += [r"\newcommand{\EMAStudyRows}{%", *rows, "}", ""]

    calibration_rows = []
    for identifier in (
        "calibration_cv_A0_autoshot_baseline",
        "calibration_cv_A1_phase2_control",
        "calibration_cv_B5_phase2_full",
    ):
        item = experiments[identifier]
        metrics = item["metrics"]
        calibration_rows.append(
            f"{tex_escape(item['label'].replace(', 5-fold CV', ''))} & "
            f"{tex_value(metrics['shot']['f1'], identifier.endswith('A1_phase2_control'))} & "
            f"{tex_value(metrics['bbc']['f1'], identifier.endswith('A1_phase2_control'))} & "
            f"{tex_value(metrics['clipshots']['f1'], identifier.endswith('A0_autoshot_baseline'))} \\\\"
        )
    lines += [r"\newcommand{\CalibrationCVRows}{%", *calibration_rows, "}", ""]

    shot = comparisons
    lines += [
        r"\newcommand{\AppendixShotRows}{%",
        "SHOT tập kiểm tra & "
        f"{f4(shot['transnetv2_reported']['metrics']['shot'])} & "
        f"{f4(experiments['A0_autoshot_original']['metrics']['shot']['f1'])} & "
        rf"\textbf{{{f4(experiments['phase2_deploy_threshold']['metrics']['shot']['f1'])}}} \\",
        "}",
        "",
        r"\newcommand{\AppendixClipshotsRows}{%",
        "ClipShots (theo báo cáo) & "
        f"{f4(shot['transnetv2_reported']['metrics']['clipshots'])} & "
        rf"\textbf{{{f4(shot['autoshot_reported']['metrics']['clipshots'])}}} & "
        f"{f4(experiments['phase2_deploy_threshold']['metrics']['clipshots']['f1'])} \\\\",
        "ClipShots (tự chạy lại) & "
        f"{f4(shot['transnetv2_reproduced']['metrics']['clipshots'])} & "
        f"{f4(shot['autoshot_reproduced_legacy']['metrics']['clipshots'])} & "
        f"{f4(experiments['phase2_deploy_threshold']['metrics']['clipshots']['f1'])} \\\\",
        "}",
        "",
    ]

    rows = []
    for item in manifest["experiments"]:
        metrics = item["metrics"]
        rows.append(
            f"{tex_escape(item['group'])} & {tex_escape(item['label'])} & "
            f"{f4(metrics['shot']['f1'])} & {f4(metrics['bbc']['f1'])} & "
            f"{f4(metrics['clipshots']['f1'])} & {tex_escape(item['note'])} \\\\"
        )
    lines += [r"\newcommand{\AllExperimentsRows}{%", *rows, "}", ""]
    return "\n".join(lines)


def render_slide_data(manifest: dict[str, Any]) -> str:
    experiments = experiment_map(manifest)
    comparisons = comparison_map(manifest)
    deploy = experiments["phase2_deploy_threshold"]["metrics"]
    a0 = experiments["A0_autoshot_original"]["metrics"]
    data = {
        "schema_version": 1,
        "comparison_models": manifest["comparison_models"],
        "deploy": deploy,
        "ablation": [
            {
                "id": identifier,
                "label": experiments[identifier]["label"],
                "metrics": experiments[identifier]["metrics"],
            }
            for identifier in ABLATION_ORDER
        ],
        "summary": {
            "shot_f1": deploy["shot"]["f1"],
            "bbc_f1": deploy["bbc"]["f1"],
            "clipshots_f1": deploy["clipshots"]["f1"],
            "shot_delta_vs_transnet_pp": (
                deploy["shot"]["f1"] - comparisons["transnetv2_reported"]["metrics"]["shot"]
            )
            * 100,
            "shot_delta_vs_autoshot_pp": (
                deploy["shot"]["f1"] - a0["shot"]["f1"]
            )
            * 100,
        },
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def render_paper_tex_macros(manifest: dict[str, Any]) -> str:
    experiments = experiment_map(manifest)
    comparisons = comparison_map(manifest)
    deploy = experiments["phase2_deploy_threshold"]["metrics"]
    best = experiments["phase2_best_sweep"]["metrics"]
    autoshot = comparisons["autoshot_reproduced_legacy"]["metrics"]
    transnet = comparisons["transnetv2_reported"]["metrics"]
    deployment_config = manifest["deployment_config"]
    definitions = {
        "PaperTemperature": f"{deployment_config['temperature']:.4f}",
        "PaperGaussianSigma": f"{deployment_config['sigma']:.1f}",
        "PaperDeployThreshold": f"{deploy['shot']['threshold']:.2f}",
        "PaperClipBestThreshold": f"{best['clipshots']['threshold']:.2f}",
        "PaperASVTwoShotFOne": f4(best["shot"]["f1"]),
        "PaperASVTwoBBCFOne": f4(best["bbc"]["f1"]),
        "PaperASVTwoClipDeployFOne": f4(deploy["clipshots"]["f1"]),
        "PaperASVTwoClipBestFOne": f4(best["clipshots"]["f1"]),
        "PaperASVTwoShotPrecision": f4(best["shot"]["precision"]),
        "PaperASVTwoShotRecall": f4(best["shot"]["recall"]),
        "PaperAutoShotShotFOne": f4(autoshot["shot"]),
        "PaperTransNetShotFOne": f4(transnet["shot"]),
        "PaperASVTwoVsAutoShotPP": f"{(best['shot']['f1'] - autoshot['shot']) * 100:.2f}",
        "PaperASVTwoVsTransNetPP": f"{(best['shot']['f1'] - transnet['shot']) * 100:.2f}",
    }
    lines = ["% Generated by scripts/sync_experimental_results.py. Do not edit."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in definitions.items())
    lines.append("")
    return "\n".join(lines)


def paper_metric(value: float | None, bold: bool = False) -> str:
    rendered = "--" if value is None else f"{value:.4f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def render_paper_tex_tables(manifest: dict[str, Any]) -> str:
    experiments = experiment_map(manifest)
    comparisons = comparison_map(manifest)
    lines = ["% Generated by scripts/sync_experimental_results.py. Do not edit.", ""]

    comparison_rows = []
    comparison_definitions = (
        ("autoshot_reported", "AutoShot, reported in original paper"),
        ("autoshot_reproduced_legacy", "AutoShot, reproduced"),
        ("transnetv2_reported", "TransNetV2, reported baseline"),
        ("transnetv2_reproduced", "TransNetV2, reproduced"),
        ("autoshot_v1_gaussian", "AutoShot reproduced + Gaussian smoothing"),
    )
    for identifier, label in comparison_definitions:
        metrics = comparisons[identifier]["metrics"]
        comparison_rows.append(
            f"{label} & {paper_metric(metrics['shot'])} & "
            f"{paper_metric(metrics['bbc'])} & {paper_metric(metrics['clipshots'])} \\\\"
        )

    deploy = experiments["phase2_deploy_threshold"]["metrics"]
    best = experiments["phase2_best_sweep"]["metrics"]
    comparison_rows += [
        "AutoShotV2, fixed deploy threshold & "
        f"{paper_metric(deploy['shot']['f1'], True)} & "
        f"{paper_metric(deploy['bbc']['f1'], True)} & "
        f"{paper_metric(deploy['clipshots']['f1'])} \\\\",
        "AutoShotV2, ClipShots best sweep & -- & -- & "
        f"{paper_metric(best['clipshots']['f1'])} \\\\",
    ]
    lines += [r"\newcommand{\PaperMainResultRows}{%", *comparison_rows, "}", ""]

    paper_ablation_labels = {
        "A0_autoshot_original": "A0 -- AutoShot baseline",
        "A1_phase2_bce_onehot": "A1 -- BCE + one-hot",
        "A2_focal_only": "A2 -- Focal only",
        "A3_manyhot_only": "A3 -- Many-hot only",
        "P1_gaussian_only": "P1 -- Gaussian only",
        "P2_temperature_only": "P2 -- Temperature only",
        "B1_focal_manyhot": "B1 -- Focal + many-hot",
        "B4_temperature_gaussian": "B4 -- Temperature + Gaussian",
        "B5_full_candidate": "B5 -- Full candidate, no EMA",
    }
    ablation_rows = []
    for identifier in ABLATION_ORDER:
        item = experiments[identifier]
        metrics = item["metrics"]
        selected = identifier == "B4_temperature_gaussian"
        label = paper_ablation_labels[identifier]
        if selected:
            label += " (selected)"
        ablation_rows.append(
            f"{tex_escape(label)} & "
            f"{paper_metric(metrics['shot']['f1'], selected)} & "
            f"{paper_metric(metrics['bbc']['f1'], selected)} & "
            f"{paper_metric(metrics['clipshots']['f1'], selected)} \\\\"
        )
    lines += [r"\newcommand{\PaperControlledAblationRows}{%", *ablation_rows, "}", ""]

    breakdown = manifest["supplemental_results"]["clipshots_transition_breakdown"]
    breakdown_rows = []
    for item in breakdown["transition_types"]:
        breakdown_rows.append(
            f"{tex_escape(item['label'])} & {item['precision']:.4f} & "
            f"{item['recall']:.4f} & {item['f1']:.4f} \\\\"
        )
    lines += [r"\newcommand{\PaperClipBreakdownRows}{%", *breakdown_rows, "}", ""]
    return "\n".join(lines)


def expected_outputs() -> dict[Path, str]:
    manifest = build_manifest()
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return {
        REPORTS / "experimental_results.json": manifest_text,
        REPORTS / "experimental_results_summary.md": render_markdown(manifest),
        THESIS_GENERATED / "experiment_macros.tex": render_tex_macros(manifest),
        THESIS_GENERATED / "experiment_tables.tex": render_tex_tables(manifest),
        THESIS_GENERATED / "slide_results.json": render_slide_data(manifest),
        PAPER_GENERATED / "experiment_macros.tex": render_paper_tex_macros(manifest),
        PAPER_GENERATED / "experiment_tables.tex": render_paper_tex_tables(manifest),
    }


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"WROTE {path.relative_to(ROOT)}")


def check_outputs(outputs: dict[Path, str]) -> int:
    failures = 0
    for path, expected in outputs.items():
        if not path.is_file():
            print(f"MISSING {path.relative_to(ROOT)}")
            failures += 1
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            print(f"STALE {path.relative_to(ROOT)}")
            failures += 1
        else:
            print(f"OK {path.relative_to(ROOT)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate consolidated outputs.")
    mode.add_argument("--check", action="store_true", help="Fail when generated outputs are stale.")
    args = parser.parse_args()

    outputs = expected_outputs()
    if args.write:
        write_outputs(outputs)
        return 0
    return 1 if check_outputs(outputs) else 0


if __name__ == "__main__":
    sys.exit(main())
