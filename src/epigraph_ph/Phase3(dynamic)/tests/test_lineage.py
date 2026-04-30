from __future__ import annotations

from phase3_dynamic.lineage import build_project_lineage_manifest


def test_lineage_manifest_tracks_active_and_quarantined_phase3_paths(tmp_path) -> None:
    project_root = tmp_path / "EpiGraph_PH"
    phase3_dynamic_root = project_root / "src" / "epigraph_ph" / "Phase3(dynamic)"
    active_src = phase3_dynamic_root / "src" / "phase3_dynamic"
    active_tests = phase3_dynamic_root / "tests"
    legacy_root = project_root / "src" / "epigraph_ph" / "phase3"
    ignored = phase3_dynamic_root / "__pycache__"

    active_src.mkdir(parents=True)
    active_tests.mkdir(parents=True)
    legacy_root.mkdir(parents=True)
    ignored.mkdir(parents=True)

    (active_src / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    (active_tests / "test_model.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (legacy_root / "legacy.py").write_text("LEGACY = 1\n", encoding="utf-8")
    (ignored / "ignored.pyc").write_text("ignored\n", encoding="utf-8")

    manifest = build_project_lineage_manifest(
        project_root=project_root,
        phase3_dynamic_root=phase3_dynamic_root,
        include_roots=[phase3_dynamic_root, legacy_root],
    )

    assert manifest["entry_count"] == 3
    by_relative = {entry["relative_path"]: entry for entry in manifest["entries"]}
    active_entry = by_relative["src/epigraph_ph/Phase3(dynamic)/src/phase3_dynamic/model.py"]
    test_entry = by_relative["src/epigraph_ph/Phase3(dynamic)/tests/test_model.py"]
    legacy_entry = by_relative["src/epigraph_ph/phase3/legacy.py"]

    assert active_entry["artifact_role"] == "phase3_dynamic_active"
    assert active_entry["planned_new_path"].endswith("/src/epigraph_ph/phase3_dynamic/model.py")
    assert test_entry["planned_new_path"].endswith("/tests/phase3_dynamic/test_model.py")
    assert legacy_entry["artifact_role"] == "phase3_root_quarantined"
    assert legacy_entry["quarantine_status"] == "quarantined"

