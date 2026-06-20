"""Reporting/figure side of the ablation study."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from autoshotv2.ablation import Experiment


def flatten_metric(
    exp: Experiment,
    dataset: str,
    eval_payload: dict[str, Any],
    postprocess: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "experiment_id": exp.experiment_id,
        "description": exp.description,
        "kind": exp.kind,
        "control_id": exp.control_id or "",
        "source_experiment": exp.source_experiment or "",
        "dataset": dataset,
        "loss": exp.loss,
        "manyhot_weight": exp.manyhot_weight,
        "boundary_window": exp.boundary_window,
        "temperature_mode": exp.temperature_mode,
        "temperature": postprocess["temperature"],
        "sigma": exp.sigma,
        "threshold": postprocess["threshold"],
        "status": eval_payload["status"],
        "f1": "",
        "precision": "",
        "recall": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "best_f1": "",
        "best_threshold": "",
        "delta_f1": "",
        "delta_precision": "",
        "delta_recall": "",
        "delta_tp": "",
        "delta_fp": "",
        "delta_fn": "",
    }
    if eval_payload["status"] != "ok":
        row["error"] = eval_payload.get("error", "")
        return row

    result = eval_payload["result"]
    deploy = result["deploy"]
    best = result["best_sweep"]
    if exp.kind == "baseline":
        metric = best
        row["threshold"] = metric["threshold"]
    else:
        metric = deploy
    row.update(
        {
            "f1": metric["f1"],
            "precision": metric["precision"],
            "recall": metric["recall"],
            "tp": metric["tp"],
            "fp": metric["fp"],
            "fn": metric["fn"],
            "best_f1": best["f1"],
            "best_threshold": best["threshold"],
            "error": "",
        }
    )
    return row


def add_deltas(rows: list[dict[str, Any]]) -> None:
    by_key = {(row["experiment_id"], row["dataset"]): row for row in rows if row["status"] == "ok"}
    for row in rows:
        control_id = row.get("control_id")
        if row["status"] != "ok" or not control_id:
            continue
        control = by_key.get((control_id, row["dataset"]))
        if not control:
            continue
        for field in ("f1", "precision", "recall"):
            row[f"delta_{field}"] = float(row[field]) - float(control[field])
        for field in ("tp", "fp", "fn"):
            row[f"delta_{field}"] = int(row[field]) - int(control[field])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["status"] == "ok" and row.get("delta_f1") != ""]
    if not completed:
        return

    labels = [f"{row['experiment_id']}\n{row['dataset']}" for row in completed]
    delta_f1 = [float(row["delta_f1"]) for row in completed]
    plt.figure(figsize=(max(10, len(labels) * 0.6), 5))
    plt.bar(labels, delta_f1, color="#4c78a8")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.ylabel("Delta F1 vs control")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "component_delta_f1.png", dpi=180)
    plt.close()

    delta_precision = [float(row["delta_precision"]) for row in completed]
    delta_recall = [float(row["delta_recall"]) for row in completed]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(10, len(labels) * 0.6), 5))
    plt.bar(x - width / 2, delta_precision, width, label="Precision", color="#59a14f")
    plt.bar(x + width / 2, delta_recall, width, label="Recall", color="#f28e2b")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.ylabel("Delta vs control")
    plt.xticks(x, labels, rotation=70, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_delta.png", dpi=180)
    plt.close()

    by_exp: dict[str, dict[str, float]] = {}
    for row in completed:
        by_exp.setdefault(row["experiment_id"], {})[row["dataset"]] = float(row["delta_f1"])
    xs = []
    ys = []
    texts = []
    for exp_id, values in by_exp.items():
        if "shot" in values and "clipshots" in values:
            xs.append(values["shot"])
            ys.append(values["clipshots"])
            texts.append(exp_id)
    if xs:
        plt.figure(figsize=(7, 6))
        plt.scatter(xs, ys, color="#4c78a8")
        for x_val, y_val, text in zip(xs, ys, texts):
            plt.annotate(text, (x_val, y_val), fontsize=8, xytext=(4, 4), textcoords="offset points")
        plt.axhline(0, color="#333333", linewidth=0.8)
        plt.axvline(-0.002, color="#d62728", linewidth=0.8, linestyle="--", label="SHOT guardrail")
        plt.xlabel("SHOT delta F1")
        plt.ylabel("ClipShots delta F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "dataset_tradeoff.png", dpi=180)
        plt.close()


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    lines = [
        "# Ablation Summary",
        "",
        f"- Completed rows: {len(completed)}",
        f"- Incomplete rows: {len(failed)}",
        "- Control for train-time and post-process components: `A1_phase2_bce_onehot`.",
        "- `A0_autoshot_original` uses per-dataset best threshold when logits/videos are available.",
        "",
        "## Best Completed Rows",
        "",
    ]
    for row in sorted(completed, key=lambda item: float(item["f1"]), reverse=True)[:10]:
        lines.append(
            f"- {row['experiment_id']} / {row['dataset']}: "
            f"F1={float(row['f1']):.4f}, P={float(row['precision']):.4f}, R={float(row['recall']):.4f}"
        )
    if failed:
        lines.extend(["", "## Incomplete Rows", ""])
        for row in failed[:20]:
            lines.append(f"- {row['experiment_id']} / {row['dataset']}: {row['status']} {row.get('error', '')}")
    (out_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
