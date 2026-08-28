from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe")
ARTIFACT = ROOT / "prediction_portal/managed_models/epoxy_resin/tg/20260825_200526_model_XGBoost_6.joblib"


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
    renamed = _gap_definition("cure_total_time_h")
    assert original["feature_id"] == GAP_FEATURE_IDS["cure_total_time_h"]
    assert renamed["feature_id"] == "cfrp.tg.cure_total_time_h"
    assert renamed["feature_id"] != "cfrp.tg.cure_total_time_hours"


def test_bootstrap_source_counts_and_artifact_bytes_unchanged():
    from scripts.bootstrap_feature_registry import GAPS, build_registry

    before = ARTIFACT.read_bytes()
    payload = build_registry(ARTIFACT)
    after = ARTIFACT.read_bytes()
    assert before == after
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
