import json
import pandas as pd


def test_artifact_round_trip_keeps_registry_and_manifest_hash():
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact

    context = {
        "prediction_contract": {"schema_version": 2, "feature_registry_hash": "r1", "dataset_manifest_hash": "m1"},
        "registry_snapshot": {"registry_hash": "r1"},
        "dataset_manifest": {"manifest_hash": "m1"},
        "feature_audit": {"canonical_feature_cols": ["x"], "effective_feature_cols": ["x"], "removed_feature_cols": []},
    }
    artifact = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["x"], model=None, contract_context=context)
    restored = loads_artifact(dumps_artifact(artifact))
    assert restored["extra"]["prediction_contract"]["feature_registry_hash"] == "r1"
    assert restored["extra"]["dataset_manifest"]["manifest_hash"] == "m1"


def test_training_lock_rejects_unregistered_column_before_model_creation(tmp_path):
    from core.feature_registry import compute_registry_hash
    from core.training_contract import lock_training_contract
    registry = {"schema_version": 1, "registry_version": "v1", "features": [{"feature_id": "x", "name": "temperature", "source_type": "manual_input", "unit": "C", "default_policy": "explicit_only", "status": "approved"}], "model_profiles": {"p": {"feature_ids": ["x"], "status": "approved", "target_col": "tg_c"}}, "approval": {"status": "approved"}}
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    path = tmp_path / "registry.json"; path.write_text(json.dumps(registry), encoding="utf-8")
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "x", "raw_columns": ["temperature"], "source_role": "manual_input", "unit": "C"}], "status": "approved"}
    try:
        lock_training_contract(path, manifest, "epoxy_resin", "tg", "tg_c", ["temperature", "not_registered"], pd.DataFrame({"temperature": [100], "not_registered": [1]}), None)
    except ValueError as exc:
        assert "not_registered" in str(exc)
    else:
        raise AssertionError("unregistered feature must block training")


def test_training_result_with_removed_canonical_feature_is_not_publishable():
    from core.training_contract import audit_training_result
    audit = audit_training_result({"canonical_feature_cols": ["x", "y"], "feature_registry_hash": "r1", "dataset_manifest_hash": "m1"}, {"feature_names": ["x"], "feature_mask": [True, False]})
    assert audit["publishable"] is False
    assert audit["removed_feature_cols"] == ["y"]
