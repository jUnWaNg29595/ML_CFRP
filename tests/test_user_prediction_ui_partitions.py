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


def test_partition_groups_manual_fields_by_user_intent():
    """人工输入字段按用户心智模型分组（配方 / 固化制度 / 测试条件）。

    原设计按 required/optional 分组；已改为按语义分组（见 spec §4.4），
    因为用户关心的是「配什么料、怎么固化、按什么标准测」。
    """
    plan = build_input_partition_plan(_contract())
    groups = {section["group"]: section for section in plan}
    # 「固化温度」/「测试方法」均归入测试条件（默认保守分类）
    editable = (
        groups["recipe"]["features"]
        + groups["cure_schedule"]["features"]
        + groups["test_conditions"]["features"]
    )
    assert sorted(editable) == sorted(["固化温度", "测试方法"])


def test_partition_has_five_sections():
    """分区计划固定为 5 组（配方/固化制度/测试条件/推导/计算）。"""
    plan = build_input_partition_plan(_contract())
    assert [section["group"] for section in plan] == [
        "recipe",
        "cure_schedule",
        "test_conditions",
        "derived",
        "computed",
    ]


def test_partition_display_groups_are_readonly():
    """推导与计算分区必须只读，且分别列出 derived / molecular 特征。"""
    plan = build_input_partition_plan(_contract())
    groups = {section["group"]: section for section in plan}
    assert groups["derived"]["kind"] == "display"
    assert groups["computed"]["kind"] == "display"
    assert groups["derived"]["features"] == ["MolWt_resin"]
    assert groups["computed"]["features"] == ["fp_bit_1"]
    # 系统计算特征绝不出现在人工输入组
    editable = (
        groups["recipe"]["features"]
        + groups["cure_schedule"]["features"]
        + groups["test_conditions"]["features"]
    )
    assert "MolWt_resin" not in editable
    assert "fp_bit_1" not in editable


def test_partition_with_empty_contract_returns_default_groups():
    plan = build_input_partition_plan({})
    assert plan
    for section in plan:
        assert section["features"] == []


def test_partition_with_screening_fixed_input_cols():
    contract = _contract()
    contract["screening_fixed_input_cols"] = ["固化温度"]
    plan = build_input_partition_plan(contract)
    fixed = [section for section in plan if section["group"] == "fixed_inputs"]
    assert fixed and fixed[0]["features"] == ["固化温度"]
    assert fixed[0]["kind"] == "display"


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
