# AutoShotV2

Thesis monorepo for shot boundary detection (AutoShot/TransNetV2-based), preparing a journal submission. The user is Vietnamese-speaking — respond in Vietnamese unless asked otherwise; keep code, comments, and commit messages in English.

## Commands

```powershell
pip install -e .                      # editable install (CPU torch: see requirements.txt header)
pytest                                # full suite; artifact/GPU/web tests self-skip when inputs are absent
python tests/test_smoke.py            # smoke tests runnable without pytest
ruff check --fix .                    # lint (run before pushing; CI enforces it)
python scripts/sync_experimental_results.py --check   # verify generated reports/TeX are current (7x OK)
```

CLI entry points: `autoshotv2-train`, `autoshotv2-eval`, `autoshotv2-calibrate`, `autoshotv2-ablation`, `autoshotv2-train-ema`, `autoshotv2-ema-report`. Journal study: `scripts/run_journal_study.py`. Web app: `apps/web` via docker-compose (see `docs/WEB.md`).

## Layout

- `src/autoshotv2/` — package. `common.py` (shared helpers: keys, logits IO, F1, seeds, subprocess argv), `phase2_data.py` (sample cache / data side), `train_phase2.py` (training side), `eval.py`, `runtime.py` (deployed inference), `ablation.py` + `ablation_report.py`, `postprocess_calibration.py`, `journal_protocol.py`, `paper_analysis.py`, `results_manifest.py` + `results_render.py` (feed `scripts/sync_experimental_results.py`). `model/` is the ported TransNetV2 supernet — legacy upstream code, do not refactor.
- `scripts/` — CLI wrappers + journal-study tooling. `tests/` — pytest. `publications/` — LaTeX paper/thesis. `reports/` — tracked result JSONs/summaries. `docs/ARTIFACTS_MANIFEST.md` maps artifacts to reproducible results.
- `data/`, `artifacts/` checkpoints and caches are git-ignored; never `git add` them.

## Load-bearing constraints

- **Metric neutrality**: every number in `reports/` and generated TeX is pinned. After any change run `pytest` AND `python scripts/sync_experimental_results.py --check` (must print 7× OK). `python scripts/analyze_paper_results.py` must regenerate reports byte-identical except the two wall-clock fields `median_seconds` / `milliseconds_per_million_frames` (restore those with `git checkout -- reports/` after diffing — do not commit rerun timings; the committed values are the measured ones).
- **Frozen journal protocol**: `reports/source_results/journal_frozen_protocol_seed42.json` and the training config in `run_journal_study.training_command` are frozen — never change folds, search space, seeds, or hyperparameters. `paper_analysis.run_analysis` validates its constants against the frozen JSON at runtime.
- **Cache identity**: `phase2_data.build_sample_cache_config` is compared with `==`; adding/removing/renaming keys or changing the RNG call order in `build_or_load_sample_cache` silently changes training data. Pinned by `tests/test_train_phase2_provenance.py::test_cache_config_is_pinned_exactly`. Never bump `SAMPLE_CACHE_SCHEMA_VERSION` casually.
- **F1 formula**: `utils.py` keeps `(p * r * 2) / (p + r)` on purpose (different float op order than `common.f1_pr`); do not unify.
- Re-exports in `train_phase2`, `eval`, `ablation` exist because other modules/tests import moved names from their old homes — keep them.
