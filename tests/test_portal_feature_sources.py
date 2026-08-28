import pandas as pd
import pytest


def test_portal_renders_only_manual_input_as_editable_fields():
    from UserPrediction import build_manual_input_fields, build_workflow_source_fields

    registry = {"schema_version": 1, "model_profile": {"status": "approved"}, "features": [
        {"feature_id": "m", "name": "pressure", "label": "固化压力", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "valid_range": {"min": 0, "max": 20}, "status": "approved"},
        {"feature_id": "d", "name": "cure_stage_count", "label": "固化阶段数", "source_type": "derived_workflow", "status": "approved"},
    ]}
    contract = {"schema_version": 2, "manual_input_feature_cols": ["pressure"], "workflow_source_fields": [{"column": "cure_schedule", "roles": ["derived"]}]}
    assert [field["name"] for field in build_manual_input_fields(contract, registry)] == ["pressure"]
    assert [field["name"] for field in build_workflow_source_fields(contract, registry)] == ["cure_schedule"]


def test_missing_manual_input_is_rejected_without_default_fill(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    monkeypatch.setattr(
        "core.portal_prediction.load_published_portal_model",
        lambda *args: {
            "entry": {"id": "v1"},
            "artifact": {"model": object(), "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}},
            "contract": {
                "schema_version": 2,
                "feature_cols": ["pressure"],
                "manual_input_feature_cols": ["pressure"],
                "workflow_feature_cols": [],
                "molecular_workflow_feature_cols": [],
                "derived_feature_cols": [],
                "feature_definitions": [{"feature_id": "p", "name": "pressure", "source_type": "manual_input", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}],
                "target_col": "tg_c",
                "numeric_ranges": {"pressure": {"min": 0, "max": 20}},
                "missing_value_policy": "reject_user_missing",
            },
        },
    )
    errors = validate_prediction_request({"material_type": "epoxy_resin", "target": "tg", "inputs": {}, "confirmed_by_user": True}, {"materials": {}})
    assert any("pressure" in error for error in errors)


def test_manual_fields_require_approved_manual_registry_definition():
    from UserPrediction import build_manual_input_fields

    contract = {"manual_input_feature_cols": ["pressure", "draft_value"]}
    registry = {"schema_version": 1, "model_profile": {"status": "approved"}, "features": [
        {"name": "pressure", "source_type": "manual_input", "status": "approved", "default_policy": "explicit_only"},
        {"name": "draft_value", "source_type": "manual_input", "status": "draft", "default_policy": "explicit_only"},
    ]}
    fields = build_manual_input_fields(contract, registry)
    assert [field["name"] for field in fields] == ["pressure"]


def test_workflow_fields_ignore_legacy_aliases_and_use_canonical_contract_key():
    from UserPrediction import build_workflow_source_fields

    assert build_workflow_source_fields({"source_columns": [{"column": "legacy"}]}) == []
    assert [field["name"] for field in build_workflow_source_fields({"schema_version": 2,
        "workflow_source_fields": [{"column": "cure_schedule", "roles": ["derived"]}],
        "source_columns": [{"column": "legacy"}],
    }, {"schema_version": 1, "model_profile": {"status": "approved"}, "features": []})] == ["cure_schedule"]


def test_workflow_fields_require_valid_v2_registry_snapshot():
    from UserPrediction import build_workflow_source_fields

    contract = {"schema_version": 1, "workflow_source_fields": [{"column": "resin_smiles"}]}
    assert build_workflow_source_fields(contract, {}) == []


def test_nullable_optional_workflow_source_can_be_empty(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    contract = {
        "schema_version": 2,
        "feature_cols": ["resin_x"], "manual_input_feature_cols": [],
        "workflow_feature_cols": ["resin_x"], "molecular_workflow_feature_cols": ["resin_x"],
        "derived_feature_cols": [], "feature_definitions": [],
        "target_col": "tg_c", "numeric_ranges": {},
        "workflow_source_fields": [{"column": "resin_smiles", "roles": ["resin"], "required": True, "nullable": False},
                                    {"column": "curing_agent_smiles", "roles": ["hardener"], "required": False, "nullable": True}],
    }
    artifact = {"model": object(), "feature_cols": ["resin_x"], "target_col": "tg_c", "extra": {}}
    monkeypatch.setattr("core.portal_prediction.load_published_portal_model", lambda *a: {"entry": {}, "artifact": artifact, "contract": contract})
    monkeypatch.setattr("core.portal_prediction._structure_error", lambda value: None)
    monkeypatch.setattr("core.portal_prediction.validate_single_row_source_values", lambda frame, sources: {"ok": True, "missing_columns": [], "empty_columns": []})
    errors = validate_prediction_request({"material_type": "epoxy_resin", "target": "tg", "inputs": {"resin_smiles": "CCO", "curing_agent_smiles": ""}, "confirmed_by_user": True}, {})
    assert not any("curing_agent_smiles" in error and ("不能为空" in error or "empty" in error.lower()) for error in errors)


def test_optional_curating_agent_pandas_na_runs_real_request_validation(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    contract = {
        "schema_version": 2,
        "feature_cols": ["resin_x"], "manual_input_feature_cols": [],
        "workflow_feature_cols": ["resin_x"], "molecular_workflow_feature_cols": ["resin_x"], "derived_feature_cols": [],
        "feature_definitions": [], "target_col": "tg_c", "numeric_ranges": {"resin_x": {"min": 0, "max": 10}},
        "workflow_source_fields": [
            {"column": "resin_smiles", "roles": ["resin"], "required": True, "nullable": False},
            {"column": "curing_agent_smiles", "roles": ["hardener"], "required": False, "nullable": True},
        ],
    }
    artifact = {"model": object(), "feature_cols": ["resin_x"], "target_col": "tg_c", "extra": {}}
    monkeypatch.setattr("core.portal_prediction.load_published_portal_model", lambda *args: {"entry": {}, "artifact": artifact, "contract": contract})
    errors = validate_prediction_request(
        {"material_type": "epoxy_resin", "target": "tg", "inputs": {"resin_smiles": "CCO", "curing_agent_smiles": pd.NA, "resin_x": 1.0}, "confirmed_by_user": True},
        {},
    )
    assert not any("curing_agent_smiles" in error for error in errors)
