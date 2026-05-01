from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import ensure_dir, write_json


LINEAGE_MANIFEST_SCHEMA_VERSION = "phase3_dynamic_project_lineage_manifest.v1"
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".tmp",
        "tmp",
        ".venv",
        ".uv-cache",
        ".uv-python",
        ".hf-cache",
        ".lighton-ocr-venv",
        "manim_env",
        "media",
        "videos",
    }
)


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part in IGNORED_DIRS for part in relative_parts):
            continue
        files.append(path)
    return files


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_role(path: Path, *, project_root: Path, phase3_dynamic_root: Path) -> tuple[str, str, str, str | None]:
    resolved = path.resolve()
    phase3_root = project_root / "src" / "epigraph_ph" / "phase3"
    if phase3_dynamic_root in resolved.parents or resolved == phase3_dynamic_root:
        relative = _relative(resolved, phase3_dynamic_root)
        if relative.startswith("src/phase3_dynamic/"):
            planned_new = (project_root / "src" / "epigraph_ph" / "phase3_dynamic" / relative.removeprefix("src/phase3_dynamic/")).as_posix()
        elif relative.startswith("tests/"):
            planned_new = (project_root / "tests" / "phase3_dynamic" / relative.removeprefix("tests/")).as_posix()
        elif relative.startswith("artifacts/"):
            planned_new = (project_root / "artifacts" / "phase3_dynamic" / relative.removeprefix("artifacts/")).as_posix()
        else:
            planned_new = None
        return (
            "phase3_dynamic_active",
            "new_phase3_dynamic_claim_support",
            "active",
            planned_new,
        )
    if phase3_root in resolved.parents or resolved == phase3_root:
        return (
            "phase3_root_quarantined",
            "historical_reference_only",
            "quarantined",
            None,
        )
    artifacts_runs = project_root / "artifacts" / "runs"
    if artifacts_runs in resolved.parents or resolved == artifacts_runs:
        return (
            "mixed_lineage_run_artifact",
            "per_run_contract_check_required",
            "mixed_lineage",
            None,
        )
    if "HIV_Data" in resolved.parts:
        return (
            "raw_hiv_data_source",
            "source_data_support",
            "active",
            None,
        )
    return (
        "project_support_file",
        "supporting_reference",
        "active",
        None,
    )


def build_project_lineage_manifest(
    *,
    project_root: Path,
    phase3_dynamic_root: Path,
    include_roots: list[Path] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    phase3_dynamic_root = phase3_dynamic_root.resolve()
    roots = (
        include_roots
        if include_roots is not None
        else [
            phase3_dynamic_root,
            project_root / "src" / "epigraph_ph" / "phase3",
            project_root / "artifacts" / "scientific_audits",
        ]
    )
    entries: list[dict[str, Any]] = []
    for root in roots:
        for path in _iter_files(Path(root)):
            stat = path.stat()
            artifact_role, claim_supported, quarantine_status, planned_new_path = _artifact_role(
                path,
                project_root=project_root,
                phase3_dynamic_root=phase3_dynamic_root,
            )
            entries.append(
                {
                    "path": path.resolve().as_posix(),
                    "relative_path": _relative(path.resolve(), project_root),
                    "checksum_sha256": _sha256(path),
                    "size_bytes": int(stat.st_size),
                    "modified_at": _utc_from_timestamp(float(stat.st_mtime)),
                    "artifact_role": artifact_role,
                    "claim_supported": claim_supported,
                    "allowed_use_status": claim_supported,
                    "quarantine_status": quarantine_status,
                    "planned_new_path": planned_new_path,
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": LINEAGE_MANIFEST_SCHEMA_VERSION,
        "project_root": project_root.as_posix(),
        "phase3_dynamic_root": phase3_dynamic_root.as_posix(),
        "entry_count": len(entries),
        "roots": [Path(root).resolve().as_posix() for root in roots],
        "summary": {
            "artifact_role_counts": dict(Counter(str(entry["artifact_role"]) for entry in entries)),
            "quarantine_status_counts": dict(Counter(str(entry["quarantine_status"]) for entry in entries)),
        },
        "entries": entries,
    }


def write_project_lineage_manifest(
    *,
    project_root: Path,
    phase3_dynamic_root: Path,
    include_roots: list[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    manifest = build_project_lineage_manifest(
        project_root=project_root,
        phase3_dynamic_root=phase3_dynamic_root,
        include_roots=include_roots,
    )
    target_path = output_path or (
        ensure_dir(project_root / "artifacts" / "scientific_audits")
        / "project_lineage_manifest_20260425.json"
    )
    write_json(target_path, manifest)
    return target_path
