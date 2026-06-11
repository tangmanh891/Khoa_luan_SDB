import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_experimental_results.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_experimental_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_manifest_is_complete_and_current():
    sync = load_sync_module()
    expected = sync.expected_outputs()
    assert sync.check_outputs(expected) == 0

    manifest = json.loads(expected[ROOT / "reports" / "experimental_results.json"])
    experiments = {item["id"]: item for item in manifest["experiments"]}
    assert len(experiments) == 17
    assert len(manifest["comparison_models"]) == 6

    for experiment_id in sync.ABLATION_ORDER:
        assert set(experiments[experiment_id]["metrics"]) == {"shot", "bbc", "clipshots"}

    deploy = experiments["phase2_best_sweep"]["metrics"]
    assert round(deploy["shot"]["f1"], 4) == 0.8545
    assert round(deploy["bbc"]["f1"], 4) == 0.9656
    assert round(deploy["clipshots"]["f1"], 4) == 0.7557


def test_thesis_snapshot_and_releases_are_present():
    thesis = ROOT / "publications" / "thesis"
    assert (thesis / "main.tex").is_file()
    assert len(list((thesis / "images").iterdir())) == 18
    assert (thesis / "releases" / "AutoShotV2_Thesis.pdf").stat().st_size > 3_000_000
    assert (thesis / "releases" / "AutoShotV2_Defense.pptx").stat().st_size > 3_000_000


def test_slide_data_matches_manifest():
    manifest = json.loads((ROOT / "reports" / "experimental_results.json").read_text(encoding="utf-8"))
    slide_data = json.loads(
        (ROOT / "publications" / "thesis" / "generated" / "slide_results.json").read_text(
            encoding="utf-8"
        )
    )
    experiments = {item["id"]: item for item in manifest["experiments"]}
    assert slide_data["summary"]["shot_f1"] == experiments["phase2_best_sweep"]["metrics"]["shot"]["f1"]
    assert len(slide_data["ablation"]) == 9


def test_paper_snapshot_generated_tables_and_release_are_present():
    paper = ROOT / "publications" / "paper"
    assert (paper / "main.tex").is_file()
    assert len(list((paper / "sections").glob("*.tex"))) == 8
    assert (paper / "images" / "reliability_diagram.png").stat().st_size > 20_000

    macros = (paper / "generated" / "experiment_macros.tex").read_text(encoding="utf-8")
    tables = (paper / "generated" / "experiment_tables.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\PaperASVTwoShotFOne}{0.8540}" in macros
    assert r"\newcommand{\PaperASVTwoClipFOne}{0.7441}" in macros
    assert r"\PaperProtocolMatchedRows" in tables
    assert r"\PaperConfidenceRows" in tables
    assert r"\PaperControlledAblationRows" in tables
    assert (paper / "releases" / "AutoShotV2_Paper.pdf").stat().st_size > 250_000


def test_clipshots_breakdown_has_explicit_deploy_protocol():
    source = json.loads(
        (ROOT / "reports" / "source_results" / "clipshots_transition_breakdown.json").read_text(
            encoding="utf-8"
        )
    )
    assert source["protocol"]["threshold"] == 0.1
    assert source["videos"] == 500
    assert source["source_kind"] == "recomputed_from_logits_and_annotations"
    assert {item["id"] for item in source["transition_types"]} == {"cut", "gradual"}
    assert sum(item["ground_truth"] for item in source["transition_types"]) == 7209
