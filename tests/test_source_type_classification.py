# -*- coding: utf-8 -*-
"""特征来源分类规则与 manifest 别名归一化测试。"""

from __future__ import annotations

import pytest

from core.feature_registry import (
    classify_feature_source_type,
    normalize_manifest_entry,
    validate_manifest_source_type_consistency,
)


def test_process_parameters_are_manual_input_even_if_numeric():
    """工艺温度/时间/压力是数值字段也必须归类为 manual_input。"""
    assert classify_feature_source_type({"name": "固化温度", "data_type": "numeric"}) == "manual_input"
    assert classify_feature_source_type({"name": "固化时间", "unit": "min"}) == "manual_input"
    assert classify_feature_source_type({"name": "固化压力", "data_type": "float"}) == "manual_input"
    assert classify_feature_source_type({"name": "curing temperature (°C)"}) == "manual_input"
    assert classify_feature_source_type({"name": "cure time (min)"}) == "manual_input"
    assert classify_feature_source_type({"name": "molding pressure"}) == "manual_input"


def test_test_conditions_are_manual_input():
    assert classify_feature_source_type({"name": "测试方法"}) == "manual_input"
    assert classify_feature_source_type({"name": "测试标准"}) == "manual_input"
    assert classify_feature_source_type({"name": "测试条件"}) == "manual_input"
    assert classify_feature_source_type({"name": "test standard"}) == "manual_input"
    assert classify_feature_source_type({"name": "样品状态"}) == "manual_input"


def test_manual_observation_fields_are_manual_input():
    assert classify_feature_source_type({"name": "备注"}) == "manual_input"
    assert classify_feature_source_type({"name": "质量等级"}) == "manual_input"
    assert classify_feature_source_type({"name": "目视观察"}) == "manual_input"
    assert classify_feature_source_type({"name": "异常标记"}) == "manual_input"


def test_molecular_descriptors_are_molecular_workflow():
    assert classify_feature_source_type({"name": "树脂分子量"}) == "molecular_workflow"
    assert classify_feature_source_type({"name": "MolWt_resin"}) == "molecular_workflow"
    assert classify_feature_source_type({"name": "fp_bit_128"}) == "molecular_workflow"
    assert classify_feature_source_type({"name": "原子数"}) == "molecular_workflow"
    assert classify_feature_source_type({"name": "官能团数_epoxy"}) == "molecular_workflow"


def test_stoichiometry_with_calculation_rule_is_derived_workflow():
    rule = {"input_fields": ["resin_smiles", "hardener_smiles", "phr"], "implementation": "core.x"}
    assert classify_feature_source_type({"name": "等当量比", "calculation_rule": rule}) == "derived_workflow"
    assert classify_feature_source_type({"name": "配方统计均值", "calculation_rule": rule}) == "derived_workflow"
    assert classify_feature_source_type({"name": "EEW 推导值", "calculation_rule": rule}) == "derived_workflow"


def test_numeric_process_field_never_becomes_derived():
    """数值型工艺字段不因可计算/与目标相关而变成 derived_workflow。"""
    assert classify_feature_source_type({"name": "固化温度", "data_type": "numeric", "calculation_rule": None}) == "manual_input"
    # 即使 AI 提示"可能可以推导"，仍保持 manual_input
    assert classify_feature_source_type({"name": "固化时间", "extra_hints": None}) == "manual_input"


def test_recipe_raw_inputs_differ_from_derived_features():
    """配方原始输入（SMILES/配比/PHR）与派生特征分类不同。"""
    hints = {}
    assert classify_feature_source_type({"name": "树脂SMILES"}, hints) == "manual_input"
    assert hints.get("recipe_input") is True
    assert classify_feature_source_type({"name": "配比"}) == "manual_input"
    assert classify_feature_source_type({"name": "PHR"}) == "manual_input"
    # 派生特征（带 calculation_rule、非分子关键词名）分类不同
    rule = {"input_fields": ["resin_smiles"]}
    assert classify_feature_source_type({"name": "配方统计均值", "calculation_rule": rule}) == "derived_workflow"


def test_declared_source_type_is_respected():
    """已登记合法 source_type 的特征直接尊重登记语义。"""
    assert classify_feature_source_type({"name": "固化温度", "source_type": "manual_input"}) == "manual_input"
    assert classify_feature_source_type({"name": " MolWt_resin", "source_type": "molecular_workflow"}) == "molecular_workflow"
    assert classify_feature_source_type({"name": "某某", "source_type": "target"}) == "target"


def test_unknown_when_no_evidence():
    assert classify_feature_source_type({"name": "xyzzy_alpha"}) == "unknown"
    assert classify_feature_source_type("plain_string_field") == "unknown"


def test_normalize_manifest_entry_alias_groups():
    entry = {
        "source_type": "manual_input",
        "raw_columns": ["temp_c"],
        "source_fields": ["temp_c"],
        "semantic_feature_id": "f001",
        "units": "°C",
        "accepted_aliases": ["固化温度"],
    }
    normalized = normalize_manifest_entry(entry)
    assert normalized["source_role"] == "manual_input"
    assert normalized["raw_column"] == ["temp_c"]
    assert normalized["source_field"] == ["temp_c"]
    assert normalized["feature_id"] == "f001"
    assert normalized["unit"] == "°C"
    assert normalized["aliases"] == ["固化温度"]
    # 别名键被删除
    assert "source_type" not in normalized
    assert "raw_columns" not in normalized
    assert "semantic_feature_id" not in normalized
    assert "units" not in normalized
    assert "accepted_aliases" not in normalized


def test_normalize_manifest_entry_canonical_name_wins():
    entry = {"source_role": "manual_input", "source_type": "derived_workflow"}
    normalized = normalize_manifest_entry(entry)
    assert normalized["source_role"] == "manual_input"
    assert "source_type" not in normalized
    # 原对象不被修改
    assert entry["source_type"] == "derived_workflow"


def test_normalize_manifest_entry_keeps_unknown_keys():
    entry = {"source_role": "manual_input", "reviewer": "someone", "custom_field": 123}
    normalized = normalize_manifest_entry(entry)
    assert normalized["reviewer"] == "someone"
    assert normalized["custom_field"] == 123


def test_invalid_source_role_does_not_crash():
    """非法 source_role 归一化后不崩溃，交由一致性校验给出中文错误。"""
    entry = {"source_role": "weird_role", "feature_id": "f1"}
    normalized = normalize_manifest_entry(entry)
    assert normalized["source_role"] == "weird_role"  # 保留原值不清除
    errors = validate_manifest_source_type_consistency(
        {"feature_bindings": [normalized]},
        {"features": [{"feature_id": "f1", "name": "固化温度", "source_type": "manual_input"}]},
    )
    assert isinstance(errors, list)


def test_source_type_consistency_detects_manual_as_derived():
    """manual_input 特征被 manifest 声明为 derived → 中文错误。"""
    manifest = {
        "feature_bindings": [
            {"feature_id": "f1", "source_role": "derived_workflow", "raw_columns": ["t"]}
        ]
    }
    registry = {
        "features": [
            {"feature_id": "f1", "name": "固化温度", "source_type": "manual_input"}
        ]
    }
    errors = validate_manifest_source_type_consistency(manifest, registry)
    assert errors
    assert any("manual_input" in error for error in errors)
    assert any("固化温度" in error for error in errors)


def test_source_type_consistency_detects_recipe_input_misdeclared():
    """配方原始输入行声明为 derived_workflow 输出 → 错误。"""
    manifest = {
        "mappings": [
            {"feature_id": "f2", "source_role": "derived_workflow"}
        ]
    }
    registry = {
        "features": [
            {"feature_id": "f2", "name": "树脂SMILES", "source_type": "manual_input"}
        ]
    }
    errors = validate_manifest_source_type_consistency(manifest, registry)
    assert errors


def test_source_type_consistency_accepts_aligned_bindings():
    manifest = {
        "feature_bindings": [
            {"feature_id": "f1", "source_role": "manual_input", "raw_column": "t"},
            {"feature_id": "f2", "source_role": "derived_workflow", "raw_column": "mw"},
        ]
    }
    registry = {
        "features": [
            {"feature_id": "f1", "name": "固化温度", "source_type": "manual_input"},
            {"feature_id": "f2", "name": "MolWt_resin", "source_type": "derived_workflow"},
        ]
    }
    assert validate_manifest_source_type_consistency(manifest, registry) == []


def test_workflow_outputs_can_be_classified_in_batch():
    """同一 workflow 下的多个派生特征批量归类一致。"""
    rule = {"input_fields": ["resin_smiles", "hardener_smiles"], "implementation": "core.x"}
    names = ["等当量比", "配方统计均值", "EEW 推导值", "AHEW 推导值"]
    results = [classify_feature_source_type({"name": n, "calculation_rule": rule}) for n in names]
    assert all(result == "derived_workflow" for result in results)
