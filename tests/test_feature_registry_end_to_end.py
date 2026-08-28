import json
import hashlib


class ContractModel:
    feature_names_in_ = ["pressure"]
    n_features_in_ = 1

    def predict(self, frame):
        return [0.0] * len(frame)


def test_approved_registry_manifest_contract_artifact_round_trip(tmp_path):
    from core.dataset_manifest import compute_dataset_manifest_hash, validate_dataset_manifest
    from core.feature_registry import build_registry_snapshot, compute_registry_hash, validate_registry
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    registry = {
        "schema_version": 1,
        "registry_version": "v1",
        "features": [{"feature_id": "pressure", "name": "pressure", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}],
        "model_profiles": {"p": {"material_type": "epoxy_resin", "target": "tg", "target_col": "tg_c", "feature_ids": ["pressure"], "status": "approved"}},
        "approval": {"status": "approved", "approved_by": "local-user"},
    }
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    assert validate_registry(registry, require_approved=True)["ok"] is True
    snapshot = build_registry_snapshot(registry, "p")
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "pressure", "raw_columns": ["p_raw"], "canonical_name": "pressure", "source_role": "manual_input", "unit": "MPa"}], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    assert validate_dataset_manifest(manifest, registry, frame_columns=["p_raw"], require_approved=True)["ok"] is True
    model = ContractModel()
    artifact = {"model": model, "pipeline": None, "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["pressure"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["pressure"], effective_feature_cols=["pressure"], removed_feature_cols=[], removed_feature_reasons={})
    bundle = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["pressure"], model=model, contract_context={"prediction_contract": contract, "registry_snapshot": snapshot, "dataset_manifest": manifest, "feature_audit": {"canonical_feature_cols": ["pressure"], "effective_feature_cols": ["pressure"], "removed_feature_cols": []}})
    restored = loads_artifact(dumps_artifact(bundle))
    report = validate_publication_artifact(restored, registry_snapshot=snapshot, dataset_manifest=manifest)
    assert report["ok"] is True


def test_activation_rejects_tampered_artifact_and_keeps_previous_active(tmp_path):
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_registry import build_registry_snapshot, compute_registry_hash
    from core.model_io import create_model_artifact, dumps_artifact
    from core.prediction_portal import activate_publication, build_prediction_contract

    registry = {
        "schema_version": 1, "registry_version": "v1",
        "features": [{"feature_id": "pressure", "name": "pressure", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}],
        "model_profiles": {"p": {"target_col": "tg_c", "feature_ids": ["pressure"], "status": "approved"}},
        "approval": {"status": "approved"},
    }
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    snapshot = build_registry_snapshot(registry, "p")
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "pressure", "raw_columns": ["p_raw"], "canonical_name": "pressure", "source_role": "manual_input", "unit": "MPa"}], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    model = ContractModel()
    bare = {"model": model, "pipeline": None, "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=bare, feature_cols=["pressure"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["pressure"], effective_feature_cols=["pressure"], removed_feature_cols=[])
    artifact = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["pressure"], model=model, contract_context={"prediction_contract": contract, "registry_snapshot": snapshot, "dataset_manifest": manifest})
    path = tmp_path / "model.joblib"
    path.write_bytes(dumps_artifact(artifact))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    old = {"id": "old", "version": "v1", "enabled": True, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}, "artifact_path": str(path), "artifact_hash": digest, "contract": contract, "_artifact": artifact}
    config = {"project_root": str(tmp_path), "materials": {"epoxy_resin": {"targets": {"tg": {"models": []}}}}}
    activate_publication(config, material_key="epoxy_resin", target_key="tg", entry=old)
    path.write_bytes(b"tampered")
    forged = dict(old, id="new", version="v2", enabled=True, gate_report={"ok": True, "status": "valid"}, artifact_hash=digest)
    try:
        activate_publication(config, material_key="epoxy_resin", target_key="tg", entry=forged)
    except ValueError as exc:
        assert "hash" in str(exc).lower()
    else:
        raise AssertionError("tampered artifact must be rejected")
    models = config["materials"]["epoxy_resin"]["targets"]["tg"]["models"]
    assert models[0]["version"] == "v1"
    assert models[0]["enabled"] is True
