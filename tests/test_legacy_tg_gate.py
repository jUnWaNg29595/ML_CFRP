import pytest
import hashlib


def test_legacy_tg_artifact_is_never_published_by_missing_status():
    from core.prediction_portal import should_show_publication, select_active_publication, validate_publication_artifact

    class LegacyModel:
        feature_names_in_ = ["x"]
        n_features_in_ = 1

    artifact = {"model": LegacyModel(), "pipeline": None, "feature_cols": ["x"], "target_col": "tg_c", "extra": {}}
    report = validate_publication_artifact(artifact)
    assert report["status"] == "needs_validation"
    assert should_show_publication(report) is False
    assert select_active_publication([{"id": "legacy", "enabled": True}]) is None


def test_publication_entry_defaults_to_disabled_until_gate_passes():
    from core.prediction_portal import make_publication_entry

    entry = make_publication_entry(
        material_key="epoxy_resin", target_key="tg", artifact_path="x.joblib", artifact_hash="h",
        label="Tg", unit="°C", description="legacy", contract={}, metrics={}, version="v1",
        published_at="2026-08-27T00:00:00Z",
    )
    assert entry["publication_status"] == "needs_validation"
    assert entry["enabled"] is False


def test_publication_entry_cannot_claim_published_without_a_valid_gate_report():
    from core.prediction_portal import make_publication_entry

    entry = make_publication_entry(
        material_key="epoxy_resin", target_key="tg", artifact_path="x.joblib", artifact_hash="h",
        label="Tg", unit="°C", description="demo", contract={}, metrics={}, version="v2",
        published_at="2026-08-27T00:00:00Z", publication_status="published", enabled=True,
        gate_report={"ok": False, "status": "invalid", "errors": [{"code": "blocked_feature"}]},
    )
    assert entry["publication_status"] == "needs_validation"
    assert entry["enabled"] is False


def test_failed_activation_keeps_previous_release_enabled():
    from core.prediction_portal import activate_publication

    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": [
        {"id": "old", "version": "v1", "publication_status": "published", "enabled": True}
    ]}}}}}
    try:
        activate_publication(config, material_key="epoxy_resin", target_key="tg", entry={"id": "bad", "version": "v2", "publication_status": "needs_validation", "enabled": False})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid publication must be rejected")
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0]["enabled"] is True


def test_admin_enable_toggle_uses_activation_gate_and_preserves_old_active():
    from UserPrediction import toggle_model_enabled

    old = {"id": "old", "version": "v1", "publication_status": "published", "enabled": True, "gate_report": {"ok": True, "status": "valid"}}
    unvalidated = {"id": "new", "version": "v2", "publication_status": "needs_validation", "enabled": False, "gate_report": {}}
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": [old, unvalidated]}}}}}
    with pytest.raises(ValueError, match="门禁|published|验证|artifact"):
        toggle_model_enabled(config, "epoxy_resin", "tg", unvalidated)
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0]["enabled"] is True
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][1]["enabled"] is False


def test_admin_can_reenable_disabled_published_entry_without_mutating_input(tmp_path, monkeypatch):
    from UserPrediction import toggle_model_enabled
    from core.prediction_portal import compute_contract_hash

    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"fixture")
    contract = {
        "schema_version": 1,
        "feature_cols": ["x"],
        "target_col": "tg_c",
        "source_columns": [],
        "workflow_source_columns": [],
        "workflow_source_fields": [],
        "workflow_present": False,
        "molecular_features_indicated": False,
        "pipeline_present": False,
        "imputer_present": False,
        "scaler_present": False,
        "numeric_ranges": {},
    }
    entry = {
        "id": "old",
        "version": "v1",
        "publication_status": "published",
        "enabled": False,
        "gate_report": {"ok": True, "status": "valid"},
        "artifact_path": str(artifact_path),
        "artifact_hash": hashlib.sha256(b"fixture").hexdigest(),
        "contract": contract,
        "_artifact": {"model": object(), "pipeline": None, "feature_cols": ["x"], "target_col": "tg_c", "extra": {}},
    }
    config = {"project_root": str(tmp_path), "materials": {"epoxy_resin": {"targets": {"tg": {"models": [entry]}}}}}
    monkeypatch.setattr("core.prediction_portal.validate_publication_artifact", lambda *args, **kwargs: {"ok": True, "status": "valid", "errors": []})

    toggle_model_enabled(config, "epoxy_resin", "tg", entry)

    assert entry["enabled"] is False
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0]["enabled"] is True


def test_upsert_model_entry_records_artifact_hash_and_stays_disabled_without_gate(tmp_path, monkeypatch):
    import hashlib
    import UserPrediction
    from core.model_io import create_model_artifact_bytes

    file_bytes = create_model_artifact_bytes(model_name="upload", target_col="tg_c", feature_cols=["pressure"])
    monkeypatch.setattr(UserPrediction, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(UserPrediction, "MODEL_ROOT", tmp_path / "managed_models")
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": []}}}}}
    entry = UserPrediction.upsert_model_entry(config, "epoxy_resin", "tg", "model.joblib", file_bytes, "Upload", "", [], "")
    assert entry["artifact_hash"] == hashlib.sha256(file_bytes).hexdigest()
    assert entry["publication_status"] == "needs_validation"
    assert entry["enabled"] is False
    assert entry["gate_report"] == {}


def test_failed_rollback_keeps_previous_release_enabled_when_artifact_missing():
    from core.prediction_portal import rollback_publication

    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": [
        {"version": "v1", "publication_status": "published", "enabled": True, "gate_report": {"ok": True, "status": "valid"}},
        {"version": "v2", "publication_status": "published", "enabled": False, "gate_report": {"ok": True, "status": "valid"}},
    ]}}}}}
    with pytest.raises(ValueError, match="artifact|contract"):
        rollback_publication(config, material_key="epoxy_resin", target_key="tg", version="v2")
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0]["enabled"] is True


def test_schema_one_contract_is_never_loaded_for_prediction():
    from core.portal_prediction import load_published_portal_model

    class LegacyModel:
        feature_names_in_ = ["x"]
        n_features_in_ = 1

    contract = {
        "schema_version": 1, "feature_cols": ["x"], "target_col": "tg_c",
        "source_columns": [], "workflow_source_columns": [], "workflow_present": False,
        "molecular_features_indicated": False, "pipeline_present": False,
        "imputer_present": False, "scaler_present": False, "numeric_ranges": {},
    }
    artifact = {"model": LegacyModel(), "pipeline": None, "feature_cols": ["x"], "target_col": "tg_c", "extra": {"prediction_contract": contract}}
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": [{
        "id": "legacy", "version": "v1", "enabled": True, "publication_status": "published",
        "target_col": "tg_c", "contract": contract, "_artifact": artifact,
    }]}}}}}
    with pytest.raises(ValueError, match="验证|schema"):
        load_published_portal_model(config, "epoxy_resin", "tg")


def test_publication_diagnostics_have_fixed_fields():
    from core.prediction_portal import validate_publication_artifact

    report = validate_publication_artifact({"model": None, "pipeline": None, "feature_cols": [], "target_col": "tg_c", "extra": {}})
    assert report["diagnostics"]
    assert {"code", "feature", "source", "rule", "message"} <= set(report["diagnostics"][0])


def test_missing_publication_status_is_not_counted_in_portal_statistics():
    from UserPrediction import _material_statistics

    targets, models = _material_statistics({"targets": {"tg": {"models": [{"id": "legacy", "enabled": True}]}}})
    assert targets == 1
    assert models == 0


def test_replaced_artifact_bytes_fail_hash_gate(tmp_path, monkeypatch):
    from core.portal_prediction import load_published_portal_model

    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    entry = {
        "id": "v1", "version": "v1", "enabled": True, "publication_status": "published",
        "gate_report": {"ok": True, "status": "valid"}, "artifact_path": str(artifact_path), "artifact_hash": digest,
        "contract": {"schema_version": 2, "feature_cols": ["x"], "target_col": "tg_c", "numeric_ranges": {}},
    }
    config = {"project_root": str(tmp_path), "materials": {"epoxy_resin": {"targets": {"tg": {"models": [entry]}}}}}
    artifact_path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="hash|替换"):
        load_published_portal_model(config, "epoxy_resin", "tg")


def test_task_snapshot_does_not_persist_api_key_or_full_inputs(tmp_path, monkeypatch):
    from core.portal_tasks import PortalTaskManager

    monkeypatch.setattr(PortalTaskManager, "_run_task", lambda self, task_id: None)
    manager = PortalTaskManager(tmp_path)
    task_id = manager.create_task({"request": {"inputs": {"secret_measurement": "private"}, "confirmed_by_user": True}, "ai_config": {"api_key": "sk-secret"}})
    saved = (tmp_path / "prediction_portal" / "tasks" / f"{task_id}.json").read_text(encoding="utf-8")
    assert "sk-secret" not in saved
    assert "private" not in saved
    assert "request_summary_hash" in saved
    manager.shutdown()
