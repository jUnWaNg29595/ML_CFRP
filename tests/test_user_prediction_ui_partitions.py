# -*- coding: utf-8 -*-
"""用户门户输入分区与模型信息摘要测试（Agent D 范围，纯函数无 Streamlit 渲染依赖）。"""

from __future__ import annotations

import json
from pathlib import Path

from UserPrediction import (
    MODEL_MANAGEMENT_AUDIT_FILE,
    append_model_management_audit,
    build_input_partition_plan,
    model_contract_summary,
)


def _contract() -> dict:
    definitions = [
        {"name": "固化温度", "source_type": "manual_input", "required_for_prediction": True, "status": "approved"},
        {"name": "测试方法", "source_type": "manual_input", "required_for_prediction": False, "status": "approved"},
        {"name": "MolWt_resin", "source_type": "derived_workflow", "status": "approved"},
        {"name": "fp_bit_1", "source_type": "molecular_workflow", "status": "approved"},
    ]
    return {
        "schema_version": 2,
        "feature_cols": ["固化温度", "测试方法", "MolWt_resin", "fp_bit_1"],
        "canonical_feature_cols": ["固化温度", "测试方法", "MolWt_resin", "fp_bit_1"],
        "manual_input_feature_cols": ["固化温度", "测试方法"],
        "molecular_workflow_feature_cols": ["fp_bit_1"],
        "derived_feature_cols": ["MolWt_resin"],
        "workflow_source_fields": [{"column": "resin_smiles", "roles": ["resin"]}],
        "feature_definitions": definitions,
        "model_profile_id": "epoxy_resin.tg",
        "feature_registry_version": "v1",
        "feature_registry_hash": "a" * 64,
        "workflow_hash": "b" * 64,
    }


def test_partition_groups_required_and_optional_manual():
    plan = build_input_partition_plan(_contract())
    groups = {section["group"]: section for section in plan}
    assert groups["required_manual"]["features"] == ["固化温度"]
    assert groups["optional_manual"]["features"] == ["测试方法"]


def test_partition_molecular_group_has_workflow_source_slot():
    plan = build_input_partition_plan(_contract())
    groups = {section["group"]: section for section in plan}
    assert "molecular" in groups
    assert groups["molecular"]["kind"] == "workflow_source"


def test_partition_computed_group_lists_workflow_and_derived():
    plan = build_input_partition_plan(_contract())
    groups = {section["group"]: section for section in plan}
    computed = groups["computed"]["features"]
    assert "MolWt_resin" in computed
    assert "fp_bit_1" in computed
    assert groups["computed"]["kind"] == "display"
    # 系统计算特征绝不出现在人工输入组
    assert "MolWt_resin" not in groups["required_manual"]["features"]
    assert "MolWt_resin" not in groups["optional_manual"]["features"]


def test_partition_with_empty_contract_returns_default_groups():
    plan = build_input_partition_plan({})
    assert plan
    for section in plan:
        assert section["features"] == [] or section["kind"] == "workflow_source"


def test_partition_with_screening_fixed_input_cols():
    contract = _contract()
    contract["screening_fixed_input_cols"] = ["固化温度"]
    plan = build_input_partition_plan(contract)
    fixed = [section for section in plan if section["group"] == "fixed_inputs"]
    assert fixed and fixed[0]["features"] == ["固化温度"]


def test_model_contract_summary_fields():
    model = {
        "id": "m1",
        "label": "Tg 模型",
        "model_name": "XGBoost",
        "updated_at": "2026-08-30T10:00:00",
        "artifact_hash": "c" * 64,
        "publication_status": "published",
        "gate_report": {"ok": True, "status": "valid"},
        "contract": _contract(),
        "registry_snapshot": {"profile_id": "epoxy_resin.tg"},
    }
    summary = model_contract_summary(model)
    assert summary["model_version"] == "2026-08-30T10:00:00"
    assert summary["model_profile_id"] == "epoxy_resin.tg"
    assert summary["contract_schema_version"] == 2
    assert summary["feature_registry_hash"] == "a" * 8
    assert summary["workflow_hash"] == "b" * 8
    assert summary["artifact_hash"] == "c" * 8
    assert summary["publication_status"] == "已发布"
    assert summary["contract_features"] == 4
    assert summary["manual_features"] == 2


def test_model_contract_summary_handles_empty_model():
    summary = model_contract_summary({})
    assert summary["publication_status"] == "未知"
    assert summary["contract_features"] == 0


def test_append_model_management_audit_writes_jsonl(tmp_path, monkeypatch):
    import UserPrediction as up

    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(up, "MODEL_MANAGEMENT_AUDIT_FILE", audit_file)
    append_model_management_audit("enable", model_id="m1", model_version="v1", detail="启用")
    append_model_management_audit("rollback", model_id="m2", model_version="v2", detail="回滚")
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["action"] == "enable"
    assert record["model_id"] == "m1"
    assert "ts" in record and "reviewer" in record
