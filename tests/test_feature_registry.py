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
