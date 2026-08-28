from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe")
ARTIFACT = ROOT / "prediction_portal/managed_models/epoxy_resin/tg/20260825_200526_model_XGBoost_6.joblib"
ARTIFACT_2 = ROOT / "prediction_portal/managed_models/epoxy_resin/tg/20260825_200532_model_XGBoost_6.joblib"


def _feature(**overrides):
    value = {
        "feature_id": "a",
        "name": "temperature",
        "source_type": "manual_input",
        "data_type": "float",
        "unit": "C",
        "required_for_prediction": False,
        "nullable": True,
        "default_policy": "explicit_only",
        "status": "draft",
    }
    value.update(overrides)
    return value


def _registry(features=None, profiles=None, approval=None):
    return {
        "schema_version": 1,
        "registry_version": "2026.08.27",
        "features": features or [],
        "model_profiles": profiles or {},
        "approval": approval or {"status": "draft"},
    }


def test_registry_rejects_duplicate_feature_names():
    from core.feature_registry import validate_registry

    payload = _registry([_feature(feature_id="a", name="same"), _feature(feature_id="b", name="same")])
    report = validate_registry(payload)
    assert report["ok"] is False
    assert any("name" in error for error in report["errors"])


def test_registry_rejects_manual_numeric_default():
    from core.feature_registry import validate_registry

    payload = _registry([_feature(default=0, required_for_prediction=True, nullable=False)])
    report = validate_registry(payload)
    assert report["ok"] is False
    assert any("default" in error for error in report["errors"])


def test_registry_hash_ignores_review_metadata_but_changes_semantics():
    from core.feature_registry import compute_registry_hash

    base = _registry([_feature(status="approved")], approval={"status": "approved", "approved_at": "2026-08-27T00:00:00+08:00"})
    changed_time = _registry([_feature(status="approved")], approval={"status": "approved", "approved_at": "2026-08-28T00:00:00+08:00", "approved_by": "another-user", "change_summary": "same semantics", "review_note": "same semantics"})
    changed_unit = _registry([_feature(status="approved", unit="°C")], approval={"status": "approved", "approved_at": "2026-08-27T00:00:00+08:00"})
    assert compute_registry_hash(base) == compute_registry_hash(changed_time)
    assert compute_registry_hash(base) != compute_registry_hash(changed_unit)


def test_approved_hash_is_recomputed_and_mismatch_is_rejected():
    from core.feature_registry import compute_registry_hash, validate_registry

    payload = _registry(approval={"status": "approved", "approved_hash": "not-the-computed-hash"})
    report = validate_registry(payload, require_approved=True)
    assert report["ok"] is False
    assert report["registry_hash"] == compute_registry_hash(payload)
    assert any("approved_hash" in error for error in report["errors"])


def test_profile_references_and_snapshot_filtering():
    from core.feature_registry import build_registry_snapshot, compute_registry_hash, validate_registry

    features = [_feature(feature_id="a", name="a", status="approved"), _feature(feature_id="b", name="b", status="approved")]
    payload = _registry(features, {"p": {"material_type": "epoxy_resin", "target": "tg", "target_col": "tg_c", "feature_ids": ["a"], "status": "approved"}}, {"status": "approved"})
    payload["approval"]["approved_hash"] = compute_registry_hash(payload)
    assert validate_registry(payload, require_approved=True)["ok"] is True
    snapshot = build_registry_snapshot(payload, "p")
    assert [item["feature_id"] for item in snapshot["features"]] == ["a"]
    assert snapshot["registry_version"] == payload["registry_version"]
    assert snapshot["registry_hash"] == payload["approval"]["approved_hash"]


def test_compact_snapshot_hash_cannot_be_replaced_with_contract_hash():
    from core.feature_registry import build_registry_snapshot, compute_registry_hash
    from core.prediction_portal import validate_publication_artifact, build_prediction_contract
    from core.dataset_manifest import compute_dataset_manifest_hash

    feature = _feature(feature_id="a", name="a", status="approved")
    registry = _registry([feature], {"p": {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"}}, {"status": "approved"})
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    snapshot = build_registry_snapshot(registry, "p")
    tampered = dict(snapshot)
    tampered["features"] = [dict(feature, unit="tampered")]
    import hashlib, json
    tampered["registry_hash"] = "forged-parent-hash"
    tampered["selected_features_hash"] = hashlib.sha256(json.dumps({"profile_id": tampered["profile_id"], "model_profile": tampered["model_profile"], "features": tampered["features"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": object(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=tampered, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])
    report = validate_publication_artifact(artifact, contract, registry_snapshot=tampered, dataset_manifest=manifest)
    assert report["ok"] is False
    assert any("registry" in error.lower() and "hash" in error.lower() for error in report["errors"])


def test_publication_rejects_compact_snapshot_when_payload_is_not_approved():
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_registry import build_registry_snapshot, compute_registry_hash
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    feature = _feature(feature_id="a", name="a", status="approved")
    registry = _registry(
        [feature],
        {"p": {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"}},
        {"status": "draft"},
    )
    snapshot = {
        "schema_version": 1,
        "registry_version": registry["registry_version"],
        "registry_hash": compute_registry_hash(registry),
        "registry_payload": registry,
        "profile_id": "p",
        "model_profile": registry["model_profiles"]["p"],
        "features": [feature],
        "selected_features_hash": "ignored-by-gate",
    }
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": object(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])

    report = validate_publication_artifact(artifact, contract, registry_snapshot=snapshot, dataset_manifest=manifest)

    assert report["ok"] is False
    assert any("approved" in error.lower() for error in report["errors"])


def test_publication_rejects_compact_snapshot_without_payload_even_with_registry_hash():
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    snapshot = {
        "schema_version": 1,
        "registry_version": "v1",
        "registry_hash": "claimed-approved-hash",
        "profile_id": "p",
        "model_profile": {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"},
        "features": [_feature(feature_id="a", name="a", status="approved")],
        "selected_features_hash": "claimed-selected-hash",
    }
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": object(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])

    report = validate_publication_artifact(artifact, contract, registry_snapshot=snapshot, dataset_manifest=manifest)

    assert report["ok"] is False
    assert any("payload" in error.lower() or "approved" in error.lower() for error in report["errors"])


def test_publication_rejects_compact_snapshot_top_level_feature_tampering_with_recomputed_hashes():
    import hashlib
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_registry import build_registry_snapshot, compute_registry_hash
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    feature = _feature(feature_id="a", name="a", status="approved")
    registry = _registry([feature], {"p": {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"}}, {"status": "approved"})
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    snapshot = build_registry_snapshot(registry, "p")
    snapshot["features"] = [dict(snapshot["features"][0], unit="tampered")]
    snapshot["selected_features_hash"] = hashlib.sha256(json.dumps({"profile_id": snapshot["profile_id"], "model_profile": snapshot["model_profile"], "features": snapshot["features"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": object(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])

    report = validate_publication_artifact(artifact, contract, registry_snapshot=snapshot, dataset_manifest=manifest)

    assert report["ok"] is False
    assert any("feature" in error.lower() and ("mismatch" in error.lower() or "一致" in error) for error in report["errors"])


def test_publication_rejects_full_snapshot_with_unapproved_model_profile():
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_registry import compute_registry_hash
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    feature = _feature(feature_id="a", name="a", status="approved")
    registry = _registry([feature], {"p": {"feature_ids": ["a"], "status": "draft", "target_col": "tg_c"}}, {"status": "approved"})
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": object(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=registry, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])

    report = validate_publication_artifact(artifact, contract, registry_snapshot=registry, dataset_manifest=manifest)

    assert report["ok"] is False
    assert any("profile" in error.lower() and "approved" in error.lower() for error in report["errors"])


def test_publication_accepts_full_approved_snapshot_with_declared_registry_hash():
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_registry import compute_registry_hash
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    class Model:
        feature_names_in_ = ["a"]
        n_features_in_ = 1

    feature = _feature(feature_id="a", name="a", status="approved")
    registry = _registry([feature], {"p": {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"}}, {"status": "approved"})
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    registry["registry_hash"] = compute_registry_hash(registry)
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact = {"model": Model(), "pipeline": None, "feature_cols": ["a"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["a"], target_col="tg_c", registry_snapshot=registry, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["a"], effective_feature_cols=["a"], removed_feature_cols=[])

    report = validate_publication_artifact(artifact, contract, registry_snapshot=registry, dataset_manifest=manifest)

    assert report["ok"] is True


def test_blocked_feature_requires_reason():
    from core.feature_registry import validate_registry

    report = validate_registry(_registry([_feature(source_type="unknown", status="blocked")]))
    assert report["ok"] is False
    assert any("blocking" in error.lower() or "reason" in error.lower() for error in report["errors"])


def test_workflow_requires_rule_inputs_and_implementation():
    from core.feature_registry import validate_registry

    feature = _feature(source_type="derived_workflow", default_policy="workflow_only", calculation_rule={})
    report = validate_registry(_registry([feature]))
    assert report["ok"] is False
    assert any("input_fields" in error or "implementation" in error for error in report["errors"])


def test_workflow_requires_unit_and_null_invalid_policies():
    from core.feature_registry import validate_registry

    rule = {
        "input_fields": ["schedule"],
        "implementation": "core.process_features:derive_declared_feature",
        "null_policy": "",
        "invalid_policy": "",
    }
    feature = _feature(
        feature_id="derived",
        name="derived",
        source_type="derived_workflow",
        unit="",
        default_policy="workflow_only",
        calculation_rule=rule,
    )
    report = validate_registry(_registry([feature]))
    assert report["ok"] is False
    assert any("unit" in error for error in report["errors"])
    assert any("null_policy" in error for error in report["errors"])
    assert any("invalid_policy" in error for error in report["errors"])


def test_workflow_rejects_blank_or_non_string_input_fields():
    from core.feature_registry import validate_registry

    feature = _feature(
        source_type="derived_workflow",
        default_policy="workflow_only",
        calculation_rule={
            "input_fields": ["schedule", "  ", 3],
            "implementation": " impl ",
            "null_policy": " reject ",
            "invalid_policy": " reject ",
        },
    )
    report = validate_registry(_registry([feature]))
    assert report["ok"] is False
    assert any("input_fields" in error for error in report["errors"])


def test_profile_status_target_and_blocked_list_are_bidirectionally_validated():
    from core.feature_registry import validate_registry

    blocked = _feature(feature_id="blocked", name="blocked", status="blocked", blocking_reason="unknown")
    approved = _feature(feature_id="approved", name="approved", status="approved")
    base_profile = {"feature_ids": ["blocked", "approved"], "status": "approved", "target_col": "tg_c"}

    missing_status = _registry([blocked, approved], {"p": {**base_profile, "status": ""}})
    assert validate_registry(missing_status)["ok"] is False

    missing_target = _registry([blocked, approved], {"p": {**base_profile, "target_col": ""}})
    assert validate_registry(missing_target)["ok"] is False

    missing_blocked = _registry([blocked, approved], {"p": {**base_profile, "blocked_feature_ids": []}})
    report = validate_registry(missing_blocked)
    assert report["ok"] is False
    assert any("blocked" in error for error in report["errors"])

    duplicate_blocked = _registry([blocked, approved], {"p": {**base_profile, "blocked_feature_ids": ["blocked", "blocked"]}})
    report = validate_registry(duplicate_blocked)
    assert report["ok"] is False
    assert any("duplicate" in error for error in report["errors"])

    malformed_refs = _registry([blocked, approved], {"p": {**base_profile, "feature_ids": [["blocked"]], "blocked_feature_ids": []}})
    report = validate_registry(malformed_refs)
    assert report["ok"] is False
    assert any("non-empty strings" in error for error in report["errors"])

    wrong_blocked = _registry([blocked, approved], {"p": {**base_profile, "blocked_feature_ids": ["approved"]}})
    report = validate_registry(wrong_blocked)
    assert report["ok"] is False
    assert any("blocked" in error for error in report["errors"])


def test_approved_profile_requires_approved_definitions():
    from core.feature_registry import validate_registry

    legacy = _feature(feature_id="legacy", name="legacy", status="legacy_observed", legacy_source="artifact")
    profile = {"feature_ids": ["legacy"], "status": "approved", "target_col": "tg_c"}
    report = validate_registry(_registry([legacy], {"p": profile}))
    assert report["ok"] is False
    assert any("approved" in error for error in report["errors"])


def test_snapshot_rejects_unapproved_registry_and_blocked_profile():
    from core.feature_registry import build_registry_snapshot, compute_registry_hash

    approved_feature = _feature(feature_id="a", name="a", status="approved")
    profile = {"feature_ids": ["a"], "status": "approved", "target_col": "tg_c"}
    registry = _registry([approved_feature], {"p": profile}, {"status": "draft"})
    with pytest.raises(ValueError):
        build_registry_snapshot(registry, "p")

    registry["approval"] = {"status": "approved"}
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    registry["model_profiles"]["p"]["status"] = "blocked"
    with pytest.raises(ValueError):
        build_registry_snapshot(registry, "p")


def test_bootstrap_uses_ordinal_audit_ids_and_preserves_legacy_name():
    from scripts.bootstrap_feature_registry import build_registry

    payload = build_registry(ARTIFACT)
    legacy = [item for item in payload["features"] if item["status"] == "legacy_observed"]
    assert len(legacy) == 504
    assert all(item["feature_id"].startswith("cfrp.tg.legacy_observed.") for item in legacy)
    assert all(item.get("legacy_name") == item["name"] for item in legacy)
    assert all(item["feature_id"].rsplit(".", 1)[-1].isdigit() for item in legacy)


def test_known_gap_id_is_explicit_and_survives_model_column_rename():
    from scripts.bootstrap_feature_registry import GAP_FEATURE_IDS, _gap_definition

    original = _gap_definition("cure_total_time_h")
    assert original["feature_id"] == GAP_FEATURE_IDS["cure_total_time_h"]
    renamed_column_mapping = {"cure_total_time_hours": original["feature_id"]}
    assert renamed_column_mapping["cure_total_time_hours"] == "cfrp.tg.cure_total_time_h"


def test_real_model_column_rename_requires_explicit_mapping_and_preserves_gap_id():
    from scripts.bootstrap_feature_registry import build_registry
    import joblib

    artifact = joblib.load(ARTIFACT)
    renamed = dict(artifact)
    renamed["feature_cols"] = ["renamed_pressure" if name == "curing_pressure_mpa" else name for name in artifact["feature_cols"]]
    renamed_path = ROOT / "prediction_portal" / "renamed_test_artifact.joblib"
    try:
        joblib.dump(renamed, renamed_path)
        with pytest.raises(ValueError, match="explicit column_mapping|required"):
            build_registry(renamed_path)
        payload = build_registry(renamed_path, {"renamed_pressure": "cfrp.tg.curing_pressure_mpa"})
        pressure = next(item for item in payload["features"] if item["feature_id"] == "cfrp.tg.curing_pressure_mpa")
        assert pressure["name"] == "curing_pressure_mpa"
    finally:
        renamed_path.unlink(missing_ok=True)


def test_model_column_mapping_rejects_duplicate_gap_semantics():
    from scripts.bootstrap_feature_registry import build_registry
    import joblib

    artifact = joblib.load(ARTIFACT)
    renamed = dict(artifact)
    renamed["feature_cols"] = list(artifact["feature_cols"])
    renamed["feature_cols"][renamed["feature_cols"].index("curing_pressure_mpa")] = "renamed_pressure"
    renamed["feature_cols"][renamed["feature_cols"].index("eew_g_eq")] = "renamed_pressure_2"
    renamed_path = ROOT / "prediction_portal" / "duplicate_mapping_test_artifact.joblib"
    try:
        joblib.dump(renamed, renamed_path)
        with pytest.raises(ValueError, match="same gap"):
            build_registry(
                renamed_path,
                {
                    "renamed_pressure": "cfrp.tg.curing_pressure_mpa",
                    "renamed_pressure_2": "cfrp.tg.curing_pressure_mpa",
                },
            )
    finally:
        renamed_path.unlink(missing_ok=True)


def test_legacy_audit_ids_are_stable_when_artifact_columns_reordered():
    from scripts.bootstrap_feature_registry import build_registry
    import joblib

    artifact = joblib.load(ARTIFACT)
    reordered = dict(artifact)
    reordered["feature_cols"] = list(reversed(artifact["feature_cols"]))
    reordered_path = ROOT / "prediction_portal" / "reordered_test_artifact.joblib"
    try:
        joblib.dump(reordered, reordered_path)
        first = build_registry(ARTIFACT)
        second = build_registry(reordered_path)
        left = {item["legacy_name"]: item["feature_id"] for item in first["features"] if item["status"] == "legacy_observed"}
        right = {item["legacy_name"]: item["feature_id"] for item in second["features"] if item["status"] == "legacy_observed"}
        assert left == right
    finally:
        reordered_path.unlink(missing_ok=True)


def test_bootstrap_source_counts_and_artifact_bytes_unchanged():
    from scripts.bootstrap_feature_registry import GAPS, build_registry
    import hashlib

    before = ARTIFACT.read_bytes()
    before_2 = ARTIFACT_2.read_bytes()
    hash_before = hashlib.sha256(before).hexdigest()
    hash_before_2 = hashlib.sha256(before_2).hexdigest()
    payload = build_registry(ARTIFACT)
    payload_2 = build_registry(ARTIFACT_2)
    after = ARTIFACT.read_bytes()
    after_2 = ARTIFACT_2.read_bytes()
    assert before == after
    assert before_2 == after_2
    assert hashlib.sha256(after).hexdigest() == hash_before
    assert hashlib.sha256(after_2).hexdigest() == hash_before_2
    assert len(payload_2["features"]) == 532
    counts = {}
    for item in payload["features"]:
        counts[item["source_type"]] = counts.get(item["source_type"], 0) + 1
    assert counts["derived_workflow"] == 18
    assert counts["manual_input"] == 7
    assert counts["molecular_workflow"] == 2
    assert counts["unknown"] == 1
    assert len(GAPS) == 28


def test_bootstrap_output_counts_and_blocked_ratio():
    output = ROOT / "prediction_portal/feature_registry.test.json"
    try:
        result = subprocess.run(
            [str(PYTHON), "scripts/bootstrap_feature_registry.py", "--artifact", str(ARTIFACT), "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "532 model features / 504 workflow features / 28 historical gaps" in result.stdout
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert len(payload["features"]) == 532
        profile = payload["model_profiles"]["epoxy_resin.tg"]
        assert len(profile["feature_ids"]) == 532
        assert len(profile["blocked_feature_ids"]) >= 1
        ratio = next(item for item in payload["features"] if item["name"] == "stoichiometric_ratio_r")
        assert ratio["source_type"] == "unknown"
        assert ratio["status"] == "blocked"
        assert ratio.get("blocking_reason")
    finally:
        output.unlink(missing_ok=True)
