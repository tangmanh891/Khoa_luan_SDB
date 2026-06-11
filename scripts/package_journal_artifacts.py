"""Create a checksum manifest and optional ZIP for journal-study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(study_dir: Path) -> list[Path]:
    files = [path for path in study_dir.rglob("*") if path.is_file()]
    supplemental = (
        ROOT / "reports" / "paper_analysis_results.json",
        ROOT / "reports" / "paper_analysis_per_video.csv",
        ROOT / "reports" / "source_results" / "shot_test_ground_truth.json",
    )
    files.extend(path for path in supplemental if path.is_file())
    return sorted(set(files))


def archive_name(path: Path, study_dir: Path) -> str:
    if path.is_relative_to(study_dir):
        return f"journal_study/{path.relative_to(study_dir).as_posix()}"
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "journal_study",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "releases",
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    if not study_dir.is_dir():
        raise FileNotFoundError(f"Journal study directory not found: {study_dir}")
    files = collect_files(study_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": archive_name(path, study_dir),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "artifact": "AutoShotV2 journal reproducibility bundle",
        "files": entries,
    }
    manifest_path = args.output_dir / "AutoShotV2_Journal_Artifacts_SHA256.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest -> {manifest_path} ({len(entries)} files)")

    if args.manifest_only:
        return
    zip_path = args.output_dir / "AutoShotV2_Journal_Artifacts.zip"
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.write(manifest_path, manifest_path.name)
        for path in files:
            archive.write(path, archive_name(path, study_dir))
    print(f"Bundle -> {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
