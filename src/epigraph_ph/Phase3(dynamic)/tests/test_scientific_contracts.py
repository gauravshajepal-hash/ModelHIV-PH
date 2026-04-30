from __future__ import annotations

from phase3_dynamic.scientific_contracts import (
    build_hazard_semantics,
    build_model_contract,
    build_observation_allowed_use_schema,
    build_project_lineage_lock,
    write_project_contract_artifacts,
)


def test_observation_allowed_use_schema_declares_claim_controls() -> None:
    schema = build_observation_allowed_use_schema()

    assert schema["schema_version"] == "epigraph_observation_allowed_use.v1"
    assert "allowed_downstream_roles" in schema
    assert "structured_prior" in schema["allowed_downstream_roles"]
    assert "validated_mechanistic_transition" in schema["always_disallowed_without_extra_validation"]


def test_model_contract_and_semantics_emit_incidence_sidecar() -> None:
    contract = build_model_contract(
        incidence_enabled=True,
        observation_calibration_enabled=True,
        phase2_direct_enabled=False,
        phase2_hidden_enabled=False,
    )
    semantics = build_hazard_semantics(
        incidence_enabled=True,
        observation_calibration_enabled=True,
        phase2_direct_enabled=False,
        phase2_hidden_enabled=False,
        decomposition_enabled=True,
    )

    assert contract["incidence_contract"] == "train_only_effective_population_incidence_to_U_with_evidence_typed_exit_channels"
    assert contract["model_kind"] == "mechanistic_transition_with_latent_inflow_and_observation_calibration"
    assert semantics["latent_incidence_inflow"]["status"] == "emitted"
    assert semantics["state_specific_exit_flows"]["status"] == "emitted"
    assert semantics["care_leakage_channels"]["status"] == "emitted"
    assert semantics["hiv_decomposition_controls"]["status"] == "emitted"


def test_project_lineage_lock_quarantines_root_phase3(tmp_path) -> None:
    project_root = tmp_path / "EpiGraph_PH"
    phase3_dynamic_root = project_root / "src" / "epigraph_ph" / "Phase3(dynamic)"
    phase3_dynamic_root.mkdir(parents=True)

    lock = build_project_lineage_lock(project_root, phase3_dynamic_root)

    assert lock["schema_version"] == "epigraph_project_lineage_lock.v1"
    assert lock["canonical_phase3_dynamics_path"] == phase3_dynamic_root.resolve().as_posix()
    assert any(
        row["status"] == "quarantined_for_new_scientific_claims"
        and row["path"].endswith("/src/epigraph_ph/phase3")
        for row in lock["quarantined_paths"]
    )


def test_write_project_contract_artifacts(tmp_path) -> None:
    project_root = tmp_path / "EpiGraph_PH"
    phase3_dynamic_root = project_root / "src" / "epigraph_ph" / "Phase3(dynamic)"
    phase3_dynamic_root.mkdir(parents=True)

    paths = write_project_contract_artifacts(
        project_root=project_root,
        phase3_dynamic_root=phase3_dynamic_root,
    )

    assert paths["lineage_lock"].exists()
    assert paths["observation_allowed_use_schema"].exists()
    assert paths["observation_role_ledger_schema"].exists()
