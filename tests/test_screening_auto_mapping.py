"""锁定「自动映射」：用户不需要手工确认任何映射关系。

背景：筛选页原先强制用户在「筛选前确认：模型特征来源」面板里逐项确认，
未确认就阻断预测。用户诉求是「不想手动确认映射关系，帮我自动完成」。

本测试锁定：
1. 自动映射**永不产生 pending**（pending 正是让用户被迫手点的状态）。
2. 自动映射必须能通过 validate_mapping 的正式校验。
3. 优先级正确性：候选表真实值 > keep（否则系统算出的配方物理量进不了矩阵）。
4. candidate 源要求整列有限（validate_mapping 的硬约束）。
5. 用户显式给定的统一输入优先于候选表同名列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _catalog(candidate, computed_definitions=None):
    from core.post_feature_mapping import build_post_feature_catalog

    return build_post_feature_catalog(
        candidate, computed_definitions=computed_definitions or {}
    )


# ---------------------------------------------------------------------------
# 不阻塞
# ---------------------------------------------------------------------------
def test_auto_mapping_never_produces_pending():
    """核心诉求：自动映射不得留下任何待确认项。"""
    from core.post_feature_mapping import build_auto_mapping

    candidate = pd.DataFrame(
        {
            "resin_smiles": ["CCO"],
            "resin_1_amount_phr": [100.0],
            "formulation_r_value": [1.13],
            "partial_nan": [np.nan],
        }
    )
    mapping = build_auto_mapping(
        model_feature_cols=[
            "resin_1_amount_phr",
            "formulation_r_value",
            "partial_nan",
            "resin_1_structure_Resin_MACCS_8",
            "totally_unknown",
        ],
        candidate_df=candidate,
        molecular_feature_cols=["resin_1_structure_Resin_MACCS_8"],
    )
    assert mapping["confirmed"] is True
    assert mapping["status"] == "confirmed"
    pending = [
        f
        for f, rule in mapping["rules"].items()
        if rule["source_type"] == "pending"
    ]
    assert not pending, f"自动映射仍留下待确认项：{pending}"


def test_auto_mapping_passes_validation():
    """自动映射必须能通过正式校验（否则仍会阻断用户）。"""
    from core.post_feature_mapping import build_auto_mapping, validate_mapping

    candidate = pd.DataFrame(
        {
            "resin_smiles": ["CCO"],
            "resin_1_amount_phr": [100.0],
            "formulation_r_value": [1.13],
        }
    )
    features = [
        "resin_1_amount_phr",
        "formulation_r_value",
        "resin_1_structure_Resin_MACCS_8",
    ]
    mapping = build_auto_mapping(
        model_feature_cols=features,
        candidate_df=candidate,
        molecular_feature_cols=["resin_1_structure_Resin_MACCS_8"],
    )
    report = validate_mapping(
        mapping,
        model_feature_cols=features,
        candidate_df=candidate,
        catalog=_catalog(candidate),
        missing_input_tolerant=True,
    )
    assert report["ok"], report["errors"]


def test_auto_mapping_without_catalog_fingerprint_conflict():
    """不得写入 catalog_fingerprint（口径不同会误报 catalog changed）。"""
    from core.post_feature_mapping import build_auto_mapping

    mapping = build_auto_mapping(
        model_feature_cols=["a"],
        candidate_df=pd.DataFrame({"a": [1.0]}),
    )
    assert not mapping.get("catalog_fingerprint")


# ---------------------------------------------------------------------------
# 优先级正确性
# ---------------------------------------------------------------------------
def test_candidate_source_beats_keep_for_real_candidate_columns():
    """候选表真实列必须用 candidate 源，而不是 keep。

    为什么关键：post_feature_model_cols 只由 apply_mapping 写入矩阵，
    keep 会保留 base_row 的值。若把系统算出的配方物理量映射成 keep，
    screening_feature_derivation 的计算结果就进不了矩阵，等于白算。
    """
    from core.post_feature_mapping import apply_mapping, build_auto_mapping

    candidate = pd.DataFrame(
        {"formulation_r_value": [1.13, 1.47], "resin_1_amount_phr": [100.0, 100.0]}
    )
    features = ["formulation_r_value", "resin_1_amount_phr"]
    mapping = build_auto_mapping(
        model_feature_cols=features, candidate_df=candidate
    )
    assert mapping["rules"]["formulation_r_value"]["source_type"] == "candidate"

    # 端到端验证：算出来的值必须真的进入矩阵（而不是被 base_row 覆盖）
    base = pd.DataFrame(
        {"formulation_r_value": [9.9, 9.9], "resin_1_amount_phr": [9.9, 9.9]}
    )
    out = apply_mapping(
        base, candidate, mapping, model_feature_cols=features
    )
    assert out["formulation_r_value"].tolist() == pytest.approx([1.13, 1.47])
    assert out["resin_1_amount_phr"].tolist() == pytest.approx([100.0, 100.0])


def test_candidate_source_rejected_when_column_has_nan():
    """含 NaN 的列不能用 candidate（validate_mapping 会直接报错），退回 keep。"""
    from core.post_feature_mapping import build_auto_mapping, validate_mapping

    candidate = pd.DataFrame({"partial": [1.0, np.nan]})
    features = ["partial"]
    mapping = build_auto_mapping(
        model_feature_cols=features, candidate_df=candidate
    )
    assert mapping["rules"]["partial"]["source_type"] == "keep"
    report = validate_mapping(
        mapping,
        model_feature_cols=features,
        candidate_df=candidate,
        catalog=_catalog(candidate),
        missing_input_tolerant=True,
    )
    assert report["ok"], report["errors"]


def test_user_uniform_input_beats_candidate_column():
    """用户显式给定的统一输入优先于候选表同名列。"""
    from core.post_feature_mapping import build_auto_mapping

    candidate = pd.DataFrame({"process_atmosphere": [0.0]})
    mapping = build_auto_mapping(
        model_feature_cols=["process_atmosphere"],
        candidate_df=candidate,
        constant_values={"process_atmosphere": 1.0},
    )
    assert mapping["rules"]["process_atmosphere"]["source_type"] == "candidate"

    # 候选表没有该列时，统一输入生效为常量
    mapping2 = build_auto_mapping(
        model_feature_cols=["process_atmosphere"],
        candidate_df=pd.DataFrame({"other": [1.0]}),
        constant_values={"process_atmosphere": 1.0},
    )
    assert mapping2["rules"]["process_atmosphere"]["source_type"] == "constant"
    assert mapping2["rules"]["process_atmosphere"]["constant_value"] == pytest.approx(
        1.0
    )


def test_molecular_features_use_keep():
    from core.post_feature_mapping import build_auto_mapping

    mapping = build_auto_mapping(
        model_feature_cols=["resin_1_structure_Resin_MACCS_8"],
        candidate_df=pd.DataFrame({"x": [1.0]}),
        molecular_feature_cols=["resin_1_structure_Resin_MACCS_8"],
    )
    rule = mapping["rules"]["resin_1_structure_Resin_MACCS_8"]
    assert rule["source_type"] == "keep"


def test_summary_counts_match_rules():
    from core.post_feature_mapping import auto_mapping_summary, build_auto_mapping

    candidate = pd.DataFrame({"a": [1.0], "b": [2.0]})
    mapping = build_auto_mapping(
        model_feature_cols=["a", "b", "c"], candidate_df=candidate
    )
    counts = auto_mapping_summary(mapping)
    assert sum(counts.values()) == 3
    assert counts.get("candidate") == 2
    assert counts.get("keep") == 1


# ---------------------------------------------------------------------------
# 真实模型回归
# ---------------------------------------------------------------------------
def test_real_model_auto_mapping_has_no_pending_and_validates():
    """真实 TBPFN-0.83（1408 特征）必须能全自动映射并通过校验。"""
    import json
    from pathlib import Path

    from core.computable_features import apply_schedule_derivations
    from core.post_feature_mapping import (
        build_auto_mapping,
        validate_mapping,
    )
    from core.screening_feature_derivation import (
        derive_computable_candidate_columns,
    )
    from core.virtual_screening import (
        POST_FEATURE_COMPUTED_DEFINITIONS as PFD,
        add_candidate_equivalent_metrics,
    )

    config_path = Path("prediction_portal/prediction_config.json")
    if not config_path.exists():
        pytest.skip("未找到门户模型配置")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = [
        m
        for m in (
            config.get("materials", {})
            .get("epoxy_resin", {})
            .get("targets", {})
            .get("tg", {})
            .get("models", [])
        )
        if m.get("enabled") and m.get("feature_cols")
    ]
    if not models:
        pytest.skip("没有启用的 Tg 模型")
    feature_cols = list(models[0]["feature_cols"])

    dgeba = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    frame = pd.DataFrame(
        {
            "resin_smiles": [dgeba],
            "hardener_smiles": ["Nc1ccc(Cc2ccc(N)cc2)cc1"],
            "resin_1_amount_phr": [100.0],
            "curing_agent_1_amount_phr": [33.0],
            "cure_schedule": ["130 °C/2 h + 180 °C/2 h"],
        }
    )
    frame = add_candidate_equivalent_metrics(
        frame, resin_col="resin_smiles", hardener_col="hardener_smiles"
    )
    frame, _ = apply_schedule_derivations(
        frame, feature_cols=feature_cols, schedule_column="cure_schedule"
    )
    frame, _ = derive_computable_candidate_columns(
        frame, feature_cols=feature_cols
    )

    molecular = [
        c
        for c in feature_cols
        if c.startswith(("resin_1_structure", "curing_agent_1_structure"))
    ]
    mapping = build_auto_mapping(
        model_feature_cols=feature_cols,
        candidate_df=frame,
        molecular_feature_cols=molecular,
    )
    assert not [
        f for f, r in mapping["rules"].items() if r["source_type"] == "pending"
    ], "真实模型仍留下待确认项"

    report = validate_mapping(
        mapping,
        model_feature_cols=feature_cols,
        candidate_df=frame,
        catalog=_catalog(frame, PFD),
        missing_input_tolerant=True,
    )
    assert report["ok"], report["errors"][:5]
