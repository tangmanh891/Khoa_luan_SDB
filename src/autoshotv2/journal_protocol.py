"""Validation-only protocol selection for the AutoShotV2 journal study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from autoshotv2.common import load_logits
from autoshotv2.train_phase2 import (
    evaluate_fixed,
    find_temperature,
    gt_for_logits,
    hash_keys,
    load_metadata,
    logits_to_pred_dict,
)


DEFAULT_SIGMAS = (0.0, 1.0, 2.0, 3.0)
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 0.51, 0.05))


def hash_logits(logits: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(logits):
        value = np.asarray(logits[key], dtype=np.float32)
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def stratified_folds(
    entries: dict[str, dict[str, Any]],
    keys: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    groups: dict[str, list[str]] = {}
    for key in sorted(keys):
        dataset = str(entries[key].get("dataset", "unknown"))
        groups.setdefault(dataset, []).append(key)

    folds = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for dataset in sorted(groups):
        dataset_keys = groups[dataset]
        rng.shuffle(dataset_keys)
        for index, key in enumerate(dataset_keys):
            folds[index % n_folds].append(key)
    for fold in folds:
        fold.sort()
    if any(not fold for fold in folds):
        raise ValueError("Every validation fold must contain at least one video")
    return folds


def _subset(values: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in keys}


def _macro_source_f1(
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    entries: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[float, dict[str, float]]:
    by_source: dict[str, list[str]] = {}
    for key in sorted(set(probabilities) & set(ground_truth)):
        source = str(entries[key].get("dataset", "unknown"))
        by_source.setdefault(source, []).append(key)
    source_f1 = {}
    for source, source_keys in by_source.items():
        selected = set(source_keys)
        result = evaluate_fixed(
            _subset(probabilities, selected),
            _subset(ground_truth, selected),
            threshold,
        )
        source_f1[source] = float(result["f1"])
    return float(np.mean(list(source_f1.values()))), source_f1


def select_validation_protocol(
    logits: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    entries: dict[str, dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    keys = sorted(set(logits) & set(ground_truth) & set(entries))
    folds = stratified_folds(entries, keys, n_folds=n_folds, seed=seed)
    candidate_rows = []
    fold_contexts = []
    for fold_index, held_keys in enumerate(folds):
        held = set(held_keys)
        fit = set(keys) - held
        fold_contexts.append(
            {
                "fold": fold_index,
                "held": held,
                "temperature": find_temperature(
                    _subset(logits, fit),
                    _subset(ground_truth, fit),
                ),
            }
        )

    for sigma in sigmas:
        # Predictions depend only on (fold, sigma); compute once per fold
        # instead of once per (fold, sigma, threshold).
        fold_predictions = [
            logits_to_pred_dict(
                _subset(logits, context["held"]),
                temperature=context["temperature"],
                sigma=sigma,
            )
            for context in fold_contexts
        ]
        for threshold in thresholds:
            fold_scores = []
            for context, held_probabilities in zip(fold_contexts, fold_predictions):
                macro_f1, source_f1 = _macro_source_f1(
                    held_probabilities,
                    _subset(ground_truth, context["held"]),
                    entries,
                    threshold,
                )
                fold_scores.append(
                    {
                        "fold": context["fold"],
                        "temperature": context["temperature"],
                        "macro_source_f1": macro_f1,
                        "source_f1": source_f1,
                    }
                )
            candidate_rows.append(
                {
                    "sigma": sigma,
                    "threshold": threshold,
                    "mean_macro_source_f1": float(
                        np.mean([item["macro_source_f1"] for item in fold_scores])
                    ),
                    "folds": fold_scores,
                }
            )

    selected = max(
        candidate_rows,
        key=lambda item: (
            item["mean_macro_source_f1"],
            -item["sigma"],
            -abs(item["threshold"] - 0.1),
        ),
    )
    final_temperature = find_temperature(logits, ground_truth)
    return {
        "schema_version": 1,
        "status": "frozen",
        "selection_scope": "validation_only",
        "selection_metric": "mean macro-F1 across validation data sources and folds",
        "tie_break": "lower sigma, then threshold nearest 0.10",
        "seed": seed,
        "n_folds": n_folds,
        "validation_videos": len(keys),
        "validation_keys_hash": hash_keys(keys),
        "validation_logits_sha256": hash_logits(_subset(logits, set(keys))),
        "folds": [
            {
                "fold": index,
                "keys_hash": hash_keys(fold),
                "keys": fold,
            }
            for index, fold in enumerate(folds)
        ],
        "search_space": {
            "sigmas": list(sigmas),
            "thresholds": list(thresholds),
        },
        "selected": {
            "temperature": final_temperature,
            "sigma": selected["sigma"],
            "threshold": selected["threshold"],
            "cross_validated_macro_source_f1": selected["mean_macro_source_f1"],
        },
        "candidates": candidate_rows,
    }


def write_frozen_protocol(path: Path, payload: dict[str, Any], overwrite: bool = False) -> None:
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized and not overwrite:
        raise FileExistsError(
            f"Frozen protocol already exists with different content: {path}. "
            "Pass --overwrite only when intentionally starting a new study."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def load_frozen_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("selection_scope") != "validation_only":
        raise ValueError(f"Protocol is not a frozen validation-only manifest: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--validation-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata(str(args.meta))
    logits = load_logits(args.validation_logits)
    ground_truth = gt_for_logits(metadata["entries"], logits)
    payload = select_validation_protocol(
        logits,
        ground_truth,
        metadata["entries"],
        n_folds=args.folds,
        seed=args.seed,
    )
    payload["sources"] = {
        "metadata": str(args.meta),
        "metadata_sha256": hashlib.sha256(args.meta.read_bytes()).hexdigest(),
        "validation_logits": str(args.validation_logits),
        "validation_logits_file_sha256": hashlib.sha256(
            args.validation_logits.read_bytes()
        ).hexdigest(),
    }
    write_frozen_protocol(args.output, payload, overwrite=args.overwrite)
    print(
        f"Frozen protocol -> {args.output}: "
        f"T={payload['selected']['temperature']:.6f}, "
        f"sigma={payload['selected']['sigma']:.2f}, "
        f"threshold={payload['selected']['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
