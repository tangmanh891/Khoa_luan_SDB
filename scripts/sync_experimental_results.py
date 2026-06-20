"""Build the consolidated experiment manifest and publication tables.

The thesis and slide generator consume only the generated consolidated data.
Raw result JSON files remain the audit trail and are checked on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoshotv2.results_manifest import (
    ABLATION_ORDER,  # noqa: F401
    PAPER_GENERATED,
    REPORTS,
    ROOT,
    THESIS_GENERATED,
    build_manifest,
)
from autoshotv2.results_render import (
    render_markdown,
    render_paper_tex_macros,
    render_paper_tex_tables,
    render_tex_macros,
    render_tex_tables,
)


def expected_outputs() -> dict[Path, str]:
    manifest = build_manifest()
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return {
        REPORTS / "experimental_results.json": manifest_text,
        REPORTS / "experimental_results_summary.md": render_markdown(manifest),
        THESIS_GENERATED / "experiment_macros.tex": render_tex_macros(manifest),
        THESIS_GENERATED / "experiment_tables.tex": render_tex_tables(manifest),
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
