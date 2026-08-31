import copy
import os
import json

import pandas as pd
import pytest

from core.prediction_portal import (
    activate_publication,
    build_prediction_contract,
    _is_process_running,
    is_port_open,
    portal_process_status,
    start_prediction_portal,
    stop_prediction_portal,
    make_publication_entry,
    portal_health_label,
    rollback_publication,
    select_active_publication,
    should_show_publication,
    validate_publication_artifact,
    publish_imported_entry,
)


class _NamedModel:
    feature_names_in_ = ["resin_xtb_gap", "curing_agent_xtb_gap"]
    n_features_in_ = 2


class _UnfittedPreprocessor:
    def fit(self, values):
        return self

    def transform(self, values):
        return values


class _NoneLearnedPreprocessor:
    statistics_ = None
    n_features_in_ = 2

    def transform(self, values):
        return values


class _PlaceholderPipeline:
    steps = [("model", None)]

    def predict(self, values):
        return values


def test_publication_contract_records_numeric_ranges_and_workflow_sources():
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "target_col": "Tg",
        "metrics": {"r2": 0.91},
        "extra": {},
    }
    training_frame = pd.DataFrame(
        {
            "resin_xtb_gap": [1.0, 2.0, float("nan")],
            "curing_agent_xtb_gap": [3.0, 5.0, float("inf")],
        }
    )
    workflow = {
        "schema_version": 3,
        "workflow_hash": "workflow-123",
        "steps": [
            {
                "step_id": "resin",
                "role": "resin",
                "source_columns": ["resin_smiles_1"],
                "order": 1,
            },
            {
                "step_id": "hardener",
                "role": "hardener",
                "source_columns": ["curing_agent_smiles_1"],
                "order": 2,
            },
        ],
        "merge_order": ["resin", "hardener"],
    }

    contract = build_prediction_contract(
        artifact=artifact,
        feature_cols=artifact["feature_cols"],
        target_col="Tg",
        workflow=workflow,
        training_frame=training_frame,
    )

    assert contract["feature_cols"] == artifact["feature_cols"]
    assert contract["target_col"] == "Tg"
    assert contract["numeric_ranges"]["resin_xtb_gap"] == {"min": 1.0, "max": 2.0}
    assert contract["numeric_ranges"]["curing_agent_xtb_gap"] == {"min": 3.0, "max": 5.0}
    assert contract["source_columns"] == [
        {"column": "resin_smiles_1", "roles": ["resin"]},
        {"column": "curing_agent_smiles_1", "roles": ["hardener"]},
    ]
    assert contract["workflow_hash"] == "workflow-123"
    assert contract["workflow_schema_version"] == 3


def test_v2_contract_rejects_recomputed_registry_hash_or_semantic_definition_drift():
    artifact = {"model": _NamedModel(), "pipeline": None, "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"], "target_col": "Tg", "extra": {}}
    registry = {"registry_version": "v1", "registry_hash": "stale", "features": [
        {"name": "resin_xtb_gap", "feature_id": "r", "source_type": "molecular_workflow", "status": "approved", "unit": "eV"},
        {"name": "curing_agent_xtb_gap", "feature_id": "h", "source_type": "molecular_workflow", "status": "approved", "unit": "eV"},
    ]}
    manifest = {"manifest_hash": "m1"}
    contract = {
        "schema_version": 2, "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "canonical_feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"], "effective_feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"], "removed_feature_cols": [],
        "workflow_feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"], "molecular_workflow_feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"], "derived_feature_cols": [], "manual_input_feature_cols": [],
        "feature_definitions": [dict(item) for item in registry["features"]], "feature_registry_version": "v1", "feature_registry_hash": "stale", "dataset_manifest_hash": "m1", "target_col": "Tg",
        "workflow_source_fields": [], "source_columns": [], "workflow_source_columns": [], "workflow_present": False, "molecular_features_indicated": False, "pipeline_present": False, "imputer_present": False, "scaler_present": False, "numeric_ranges": {}, "contract_hash": "",
    }
    from core.prediction_portal import compute_contract_hash
    contract["contract_hash"] = compute_contract_hash(contract)
    report = validate_publication_artifact(artifact, contract, registry_snapshot=registry, dataset_manifest=manifest)
    assert report["ok"] is False
    assert any("hash" in error.lower() for error in report["errors"])


def test_publication_rejects_missing_molecular_workflow_for_molecular_sources():
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "target_col": "Tg",
        "extra": {},
    }
    contract = {
        "feature_cols": ["resin_xtb_gap"],
        "source_columns": [{"column": "resin_smiles_1", "roles": ["resin"]}],
        "workflow_present": False,
    }

    report = validate_publication_artifact(artifact, contract)

    assert report["ok"] is False
    assert any("workflow" in error.lower() for error in report["errors"])


def test_publication_rejects_workflow_feature_gap_before_activation():
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "target_col": "Tg",
        "extra": {
            "molecular_feature_workflow": {
                "schema_version": 3,
                "workflow_hash": "workflow-123",
                "steps": [],
                "merge_order": [],
                "final_feature_names": ["resin_xtb_gap"],
            }
        },
    }
    contract = _molecular_contract(
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )

    report = validate_publication_artifact(artifact, contract)

    assert report["ok"] is False
    assert any("final_feature_names" in error for error in report["errors"])
    assert any("curing_agent_xtb_gap" in error for error in report["errors"])


def test_legacy_artifact_needs_validation_and_is_not_publishable():
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "extra": {},
        }
    )

    assert report["ok"] is False
    assert report["status"] == "needs_validation"
    assert should_show_publication(report) is False


def test_publication_rejects_missing_model_target_and_exact_contract():
    report = validate_publication_artifact(
        {
            "model": None,
            "pipeline": None,
            "feature_cols": [],
            "target_col": "",
            "extra": {"prediction_contract": {"feature_cols": []}},
        }
    )

    assert report["ok"] is False
    assert any("model" in error.lower() or "pipeline" in error.lower() for error in report["errors"])
    assert any("target" in error.lower() for error in report["errors"])
    assert any("feature" in error.lower() for error in report["errors"])


def test_activate_and_rollback_keep_one_active_version(tmp_path, monkeypatch):
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": []}}}}}
    gate_report = {"ok": True, "status": "valid", "errors": []}
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"fixture")
    contract = {"schema_version": 2, "feature_cols": ["x"], "target_col": "Tg"}
    def release(version):
        return {"id": f"tg-{version}", "version": version, "enabled": True, "publication_status": "published", "gate_report": gate_report,
                "artifact_path": str(artifact_path), "artifact_hash": __import__('hashlib').sha256(b"fixture").hexdigest(),
                "contract": contract, "_artifact": {"model": object(), "feature_cols": ["x"], "target_col": "Tg", "extra": {}}}
    monkeypatch.setattr("core.prediction_portal.validate_publication_artifact", lambda *args, **kwargs: {"ok": True, "status": "valid", "errors": []})
    first, second = release("v1"), release("v2")

    activate_publication(config, material_key="epoxy_resin", target_key="tg", entry=first)
    activate_publication(config, material_key="epoxy_resin", target_key="tg", entry=second)
    rollback_publication(config, material_key="epoxy_resin", target_key="tg", version="v1")

    models = config["materials"]["epoxy_resin"]["targets"]["tg"]["models"]
    assert [item["enabled"] for item in models] == [True, False]


def test_activation_creates_missing_config_path_and_copies_entries():
    config = {}
    entry = make_publication_entry(
        material_key="epoxy_resin",
        target_key="tg",
        artifact_path="managed_models/epoxy_resin/tg/v1.joblib",
        artifact_hash="abc123",
        label="玻璃化温度",
        unit="°C",
        description="外部用户预测",
        contract={"feature_cols": ["resin_xtb_gap"]},
        metrics={"r2": 0.9},
        version="v1",
        published_at="2026-08-19T00:00:00Z",
        publication_status="published",
        enabled=True,
        gate_report={"ok": True, "status": "valid", "errors": []},
    )

    with pytest.raises(ValueError, match="artifact|contract"):
        activate_publication(config, material_key="epoxy_resin", target_key="tg", entry=entry)
    assert config == {}


def test_rollback_rejects_unknown_version():
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": []}}}}}

    with pytest.raises(ValueError, match="version"):
        rollback_publication(config, material_key="epoxy_resin", target_key="tg", version="v9")


def test_publish_imported_entry_stamps_version_and_activates(monkeypatch):
    """Imported artifacts (e.g. downloaded from the training platform) carry no
    release version; publish_imported_entry must re-run the local gate, stamp a
    version, and make it the single active published release."""
    cfg = {"project_root": "."}
    artifact = {
        "model_name": "tg_import",
        "target_col": "tg",
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "model": object(),
        "extra": {"prediction_contract": {"schema_version": 2}},
    }
    entry = {
        "id": "tg_1",
        "label": "tg_import",
        "enabled": False,
        "publication_status": "needs_validation",
        "gate_report": {},
        "artifact_path": "path/to.tgz",
        "artifact_hash": "abc",
        "_artifact": artifact,
    }
    import core.prediction_portal as pp
    monkeypatch.setattr(pp, "validate_publication_artifact",
                        lambda a, c: {"ok": True, "status": "valid", "errors": [], "diagnostics": []})

    pp.publish_imported_entry(cfg, material_key="epoxy", target_key="tg", entry=entry)

    assert entry.get("version"), "导入模型应被补打版本号"
    assert entry["publication_status"] == "published"
    assert entry["enabled"] is True
    assert entry["gate_report"].get("ok") is True

    active = pp.select_active_publication([entry])
    assert active is not None and active is entry


def test_publish_imported_entry_rejects_invalid_gate(monkeypatch):
    import core.prediction_portal as pp
    monkeypatch.setattr(pp, "validate_publication_artifact",
                        lambda a, c: {"ok": False, "status": "invalid",
                                      "errors": ["特征删减无法发布"], "diagnostics": []})
    entry = {
        "id": "tg_1", "enabled": False,
        "publication_status": "needs_validation", "gate_report": {},
        "artifact_path": "path/to.tgz", "artifact_hash": "abc",
        "_artifact": {"extra": {"prediction_contract": {"schema_version": 2}}},
    }
    with pytest.raises(ValueError, match="未通过发布门禁验证"):
        pp.publish_imported_entry({"project_root": "."}, material_key="epoxy",
                                    target_key="tg", entry=entry)
    assert entry.get("version") in (None, "") or entry["publication_status"] != "published"

    assert portal_health_label(True) == "可访问"
    assert portal_health_label(False) == "未启动"
    assert select_active_publication(
            [{"id": "v1", "enabled": False, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}}, {"id": "v2", "enabled": True, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}}]
    )["id"] == "v2"


def test_portal_port_probe_returns_false_for_unused_local_port():
    assert is_port_open("127.0.0.1", 1, timeout=0.05) is False
    assert select_active_publication([]) is None


def test_publication_gate_accepts_only_valid_contract_report():
    assert should_show_publication({"ok": True}) is False
    assert should_show_publication({"ok": True, "status": "valid"}) is True
    assert should_show_publication({"ok": False}) is False
    assert should_show_publication({"ok": True, "status": "needs_validation"}) is False


def test_active_publication_requires_valid_gate_report():
    assert select_active_publication([{"id": "v1", "enabled": True, "publication_status": "published"}]) is None
    assert select_active_publication([{"id": "v1", "enabled": True, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}}])["id"] == "v1"


def _molecular_contract(**overrides):
    contract = {
        "schema_version": 1,
        "feature_cols": ["resin_xtb_gap"],
        "target_col": "Tg",
        "workflow_hash": "workflow-123",
        "workflow_schema_version": 3,
        "source_columns": [
            {"column": "resin_smiles_1", "roles": ["resin"]}
        ],
        "workflow_source_columns": [
            {"column": "resin_smiles_1", "roles": ["resin"]}
        ],
        "workflow_present": True,
        "molecular_features_indicated": True,
        "pipeline_present": False,
        "imputer_present": True,
        "scaler_present": False,
        "numeric_ranges": {"resin_xtb_gap": {"min": 1.0, "max": 2.0}},
    }
    contract.update(overrides)
    return contract


def _numeric_contract(**overrides):
    contract = {
        "schema_version": 1,
        "feature_cols": ["temperature"],
        "target_col": "Tg",
        "workflow_hash": None,
        "workflow_schema_version": None,
        "source_columns": [],
        "workflow_source_columns": [],
        "workflow_present": False,
        "molecular_features_indicated": False,
        "pipeline_present": False,
        "imputer_present": False,
        "scaler_present": False,
        "numeric_ranges": {"temperature": {"min": 80.0, "max": 180.0}},
    }
    contract.update(overrides)
    return contract


def test_validation_reconciles_preprocessing_flags_for_non_molecular_artifacts():
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["temperature"],
        "target_col": "Tg",
        "extra": {},
    }

    report = validate_publication_artifact(
        artifact,
        _numeric_contract(pipeline_present=True, imputer_present=True, scaler_present=True),
    )

    assert report["ok"] is False
    assert sum("与 artifact 不一致" in error for error in report["errors"]) == 3


def test_validation_rejects_explicit_contract_mismatch_with_saved_contract():
    saved_contract = _numeric_contract()
    explicit_contract = _numeric_contract(
        numeric_ranges={"temperature": {"min": 90.0, "max": 180.0}}
    )
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["temperature"],
        "target_col": "Tg",
        "extra": {"prediction_contract": saved_contract},
    }

    report = validate_publication_artifact(artifact, explicit_contract)

    assert report["ok"] is False
    assert any("saved" in error.lower() or "已保存" in error for error in report["errors"])


def test_validation_rejects_none_valued_learned_preprocessor_attribute():
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "imputer": _NoneLearnedPreprocessor(),
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "extra": {},
        },
        _molecular_contract(),
    )

    assert report["ok"] is False
    assert any("imputer" in error.lower() or "preprocessor" in error.lower() for error in report["errors"])


def test_validation_rejects_pipeline_with_placeholder_step():
    report = validate_publication_artifact(
        {
            "model": None,
            "pipeline": _PlaceholderPipeline(),
            "feature_cols": ["temperature"],
            "target_col": "Tg",
            "extra": {},
        },
        _numeric_contract(pipeline_present=True),
    )

    assert report["ok"] is False
    assert any("pipeline" in error.lower() for error in report["errors"])


def test_validation_requires_full_contract_schema_and_consistent_workflow_metadata():
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "extra": {},
        },
        {"feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"]},
    )

    assert report["ok"] is False
    assert any("schema" in error.lower() for error in report["errors"])
    assert any("workflow" in error.lower() for error in report["errors"])


def test_molecular_features_without_workflow_are_not_publishable():
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "imputer": object(),
            "extra": {},
        },
        {
            "schema_version": 1,
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "workflow_hash": None,
            "workflow_schema_version": None,
            "source_columns": [],
            "workflow_source_columns": [],
            "workflow_present": False,
            "molecular_features_indicated": True,
            "pipeline_present": False,
            "imputer_present": False,
            "scaler_present": False,
            "numeric_ranges": {
                "resin_xtb_gap": {"min": 1.0, "max": 2.0},
                "curing_agent_xtb_gap": {"min": 3.0, "max": 5.0},
            },
        },
    )

    assert report["ok"] is False
    assert any("workflow" in error.lower() for error in report["errors"])


def test_validation_rejects_inconsistent_workflow_source_metadata():
    contract = _molecular_contract(
        workflow_source_columns=[
            {"column": "other_smiles", "roles": ["resin"]}
        ]
    )
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "imputer": object(),
            "extra": {},
        },
        contract,
    )

    assert report["ok"] is False
    assert any("source" in error.lower() for error in report["errors"])


def test_validation_rejects_non_usable_preprocessor_placeholder():
    report = validate_publication_artifact(
        {
            "model": _NamedModel(),
            "pipeline": None,
            "imputer": _UnfittedPreprocessor(),
            "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
            "target_col": "Tg",
            "extra": {},
        },
        _molecular_contract(),
    )

    assert report["ok"] is False
    assert any("imputer" in error.lower() or "preprocessor" in error.lower() for error in report["errors"])


def test_validation_rejects_missing_or_mismatched_artifact_workflow_metadata():
    contract = _molecular_contract()
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "imputer": object(),
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "target_col": "Tg",
        "extra": {
            "molecular_feature_workflow": {
                "schema_version": 4,
                "workflow_hash": "different-workflow",
                "steps": [],
            }
        },
    }

    report = validate_publication_artifact(artifact, contract)

    assert report["ok"] is False
    assert any("workflow_hash" in error for error in report["errors"])
    assert any("schema_version" in error for error in report["errors"])
    assert any("source" in error.lower() for error in report["errors"])


def test_rollback_unknown_version_does_not_change_active_release():
    config = {
        "materials": {
            "epoxy_resin": {
                "targets": {
                    "tg": {
                        "models": [
                            {"version": "v1", "enabled": True},
                            {"version": "v2", "enabled": False},
                        ]
                    }
                }
            }
        }
    }
    before = {"version": "v1", "enabled": True}

    with pytest.raises(ValueError, match="version"):
        rollback_publication(config, material_key="epoxy_resin", target_key="tg", version="v9")

    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0] == before
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][1]["enabled"] is False


def test_multiple_enabled_releases_are_rejected_instead_of_first_match():
    with pytest.raises(ValueError, match="multiple|多个"):
        select_active_publication(
            [{"id": "v1", "enabled": True, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}}, {"id": "v2", "enabled": True, "publication_status": "published", "gate_report": {"ok": True, "status": "valid"}}]
        )


def test_rollback_rejects_duplicate_requested_versions_without_mutation():
    config = {
        "materials": {
            "epoxy_resin": {
                "targets": {
                    "tg": {
                        "models": [
                            {"id": "v1", "version": "v1", "enabled": True},
                            {"id": "v2-a", "version": "v2", "enabled": False},
                            {"id": "v2-b", "version": "v2", "enabled": False},
                        ]
                    }
                }
            }
        }
    }
    before = copy.deepcopy(config)

    with pytest.raises(ValueError, match="duplicate|重复"):
        rollback_publication(config, material_key="epoxy_resin", target_key="tg", version="v2")

    assert config == before


def test_start_prediction_portal_launches_user_prediction_and_persists_pid(tmp_path, monkeypatch):
    (tmp_path / "UserPrediction.py").write_text("# test portal", encoding="utf-8")
    calls = []

    class _Process:
        pid = 4321

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return _Process()

    monkeypatch.setattr("core.prediction_portal.is_port_open", lambda *args, **kwargs: False)
    monkeypatch.setattr("core.prediction_portal.subprocess.Popen", fake_popen)

    result = start_prediction_portal(
        project_root=tmp_path,
        python_executable="python-test",
        port=8555,
    )

    assert result["status"] == "starting"
    assert result["pid"] == 4321
    assert calls[0][0][:5] == [
        "python-test",
        "-m",
        "streamlit",
        "run",
        str(tmp_path / "UserPrediction.py"),
    ]
    runtime_file = tmp_path / "prediction_portal" / "portal_runtime.json"
    assert json.loads(runtime_file.read_text(encoding="utf-8"))["pid"] == 4321


def test_stop_prediction_portal_only_stops_the_managed_process(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "prediction_portal"
    runtime_dir.mkdir()
    runtime_file = runtime_dir / "portal_runtime.json"
    runtime_file.write_text(
        json.dumps({"pid": 4321, "port": 8555}),
        encoding="utf-8",
    )
    commands = []

    monkeypatch.setattr("core.prediction_portal._is_process_running", lambda pid: pid == 4321)
    monkeypatch.setattr("core.prediction_portal._is_managed_portal_process", lambda pid, port: True)
    monkeypatch.setattr(
        "core.prediction_portal.subprocess.run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    result = stop_prediction_portal(project_root=tmp_path, port=8555)

    assert result["status"] == "stopped"
    assert commands[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert not runtime_file.exists()


def test_portal_process_status_reports_unmanaged_open_port(tmp_path, monkeypatch):
    monkeypatch.setattr("core.prediction_portal.is_port_open", lambda *args, **kwargs: True)

    result = portal_process_status(project_root=tmp_path, port=8555)

    assert result["status"] == "running"
    assert result["managed"] is False

def test_is_process_running_accepts_current_windows_process():
    assert _is_process_running(os.getpid()) is True
