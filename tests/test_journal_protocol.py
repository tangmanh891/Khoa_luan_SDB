import numpy as np

from autoshotv2.journal_protocol import (
    load_frozen_protocol,
    select_validation_protocol,
    stratified_folds,
    write_frozen_protocol,
)


def synthetic_protocol_data():
    entries = {}
    logits = {}
    ground_truth = {}
    for source in ("shot", "clipshots"):
        for index in range(5):
            key = f"{source}-{index}"
            entries[key] = {"dataset": source}
            values = np.full((20, 1), -4.0, dtype=np.float32)
            values[9:11] = 4.0
            logits[key] = values
            ground_truth[key] = np.asarray([[0, 9], [10, 19]], dtype=np.int32)
    return entries, logits, ground_truth


def test_stratified_folds_are_deterministic_and_cover_all_sources():
    entries, logits, _ = synthetic_protocol_data()
    folds = stratified_folds(entries, list(logits), n_folds=5, seed=42)

    assert folds == stratified_folds(entries, list(logits), n_folds=5, seed=42)
    assert sorted(key for fold in folds for key in fold) == sorted(logits)
    for fold in folds:
        assert {entries[key]["dataset"] for key in fold} == {"shot", "clipshots"}


def test_protocol_selection_is_validation_only_and_freezable(tmp_path):
    entries, logits, ground_truth = synthetic_protocol_data()
    protocol = select_validation_protocol(
        logits,
        ground_truth,
        entries,
        n_folds=5,
        seed=42,
        sigmas=(0.0, 1.0),
        thresholds=(0.1, 0.5),
    )

    assert protocol["status"] == "frozen"
    assert protocol["selection_scope"] == "validation_only"
    assert protocol["selected"]["sigma"] in {0.0, 1.0}
    assert protocol["selected"]["threshold"] in {0.1, 0.5}

    output = tmp_path / "protocol.json"
    write_frozen_protocol(output, protocol)
    assert load_frozen_protocol(output)["validation_keys_hash"] == protocol["validation_keys_hash"]
