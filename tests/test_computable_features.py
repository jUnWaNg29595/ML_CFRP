"""锁定「可计算特征一键识别」的判定口径与派生行为。

背景：虚拟筛选页曾把所有数值型非分子特征都当作「要枚举的工艺特征」，
导致 formulation_r_value / process_total_time_h / resin_1_epoxy_group_count
这类系统本来就能算出来的列，被要求用户手动枚举——既费事又污染筛选。

本测试锁定：
1. 真实模型（TBPFN-0.83）的 1408 个特征里，真正的设计变量必须只有配方 phr
   与测试/工艺条件，其余全部可自动计算。
2. 判定必须保守：不确定的不能谎报为可计算。
3. 固化制度 → process_* 族派生必须与 core.process_features 同口径。
4. apply_schedule_derivations 不得覆盖已有实测值。
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 判定口径
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feature,expected_source",
    [
        # 配方可算：EEW / AHEW / 化学计量比 / 组分统计
        ("formulation_r_value", "formulation"),
        ("formulation_resin_total_eew_g_eq", "formulation"),
        ("formulation_hardener_total_ahew_g_eq", "formulation"),
        ("formulation_resin_hardener_equivalent_ratio", "formulation"),
        ("formulation_epoxy_binder_total_phr", "formulation"),
        ("resin_epoxy_group_total", "formulation"),
        ("curing_agent_active_hydrogen_total", "formulation"),
        ("resin_component_count", "formulation"),
        ("catalyst_component_count", "formulation"),
        ("initiator_present", "formulation"),
        ("stoichiometric_ratio_r", "formulation"),
        ("eew", "formulation"),
        ("ahew", "formulation"),
        # 分子结构可算：逐组分物理量 + 指纹/描述符
        ("resin_1_molecular_weight_g_mol", "molecular"),
        ("resin_3_equivalent_weight_g_eq", "molecular"),
        ("curing_agent_2_active_hydrogen_equivalent_count", "molecular"),
        ("resin_1_structure_Resin_MACCS_8", "molecular"),
        ("resin_1_structure_polymer_char_len", "molecular"),
        # 原始列可派生：固化制度族（两套命名）
        ("cure_stage_count", "derived"),
        ("post_cure_total_time_h", "derived"),
        ("overall_max_cure_temperature_c", "derived"),
        ("process_total_time_h", "derived"),
        ("process_max_temperature_c", "derived"),
        ("process_temperature_time_integral_c_h", "derived"),
        ("process_has_post_cure", "derived"),
        # 需人工给定：配比 phr + 工艺/测试条件
        ("resin_1_amount_phr", "manual"),
        ("curing_agent_3_amount_phr", "manual"),
        ("process_atmosphere", "manual"),
        ("tg_c_test_method", "manual"),
        ("tg_c_test_atmosphere", "manual"),
        ("tg_c_heating_rate_c_min", "manual"),
        ("test_standard_canonical", "manual"),
        ("test_standard_organization", "manual"),
        ("curing_pressure_mpa", "manual"),
    ],
)
def test_source_classification_is_stable(feature, expected_source):
    from core.computable_features import classify_computable_feature

    verdict = classify_computable_feature(feature)
    assert verdict.source == expected_source, (
        f"{feature} 判定为 {verdict.source}（{verdict.evidence}），期望 {expected_source}"
    )


def test_auto_computable_is_exclusive_of_manual():
    """可计算与需人工给定必须互斥，不能同时成立。"""
    from core.computable_features import classify_computable_features

    for verdict in classify_computable_features(
        ["process_total_time_h", "resin_1_amount_phr", "formulation_r_value"]
    ):
        assert not (verdict.auto_computable and verdict.is_design_variable)


def test_unknown_feature_is_not_reported_as_computable():
    """保守性：未登记的陌生特征不得被谎报成可计算。"""
    from core.computable_features import classify_computable_feature

    verdict = classify_computable_feature("some_brand_new_lab_column_xyz")
    assert verdict.source == "unknown"
    assert verdict.auto_computable is False
    assert verdict.is_design_variable is False


def test_registry_definition_takes_priority():
    """registry 显式声明的 source_type 优先于关键词兜底。"""
    from core.computable_features import classify_computable_feature

    verdict = classify_computable_feature(
        "weird_name",
        definition={
            "source_type": "derived_workflow",
            "description": "由固化制度派生",
            "calculation_rule": {
                "implementation": "core.process_features:derive_declared_feature",
                "input_fields": ["cure_schedule"],
            },
        },
    )
    assert verdict.source == "derived"
    assert verdict.requires_columns == ["cure_schedule"]
    assert verdict.implementation == "core.process_features:derive_declared_feature"


def test_workflow_outputs_are_reclassified_as_molecular():
    """即使关键词没命中，属于 workflow 输出的列也必须算可自动计算。"""
    from core.computable_features import split_screening_feature_pool

    split = split_screening_feature_pool(
        ["opaque_col_1"],
        workflow_feature_names=["opaque_col_1"],
    )
    assert [v.feature for v in split["auto"]] == ["opaque_col_1"]
    assert split["design"] == []


# ---------------------------------------------------------------------------
# 真实模型回归：设计变量必须只有 phr 与工艺/测试条件
# ---------------------------------------------------------------------------
def _enabled_tg_model_feature_cols():
    import json
    from pathlib import Path

    config_path = Path("prediction_portal/prediction_config.json")
    if not config_path.exists():
        pytest.skip("未找到门户模型配置")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = (
        config.get("materials", {})
        .get("epoxy_resin", {})
        .get("targets", {})
        .get("tg", {})
    )
    models = [
        model
        for model in (targets.get("models") or [])
        if model.get("enabled") and model.get("feature_cols")
    ]
    if not models:
        pytest.skip("没有启用的 Tg 模型")
    return list(models[0]["feature_cols"])


def test_real_model_enumeration_pool_has_no_computable_features():
    """核心回归：可枚举池里不得残留任何系统能算出来的特征。"""
    from core.computable_features import (
        AUTO_SOURCES,
        classify_computable_features,
        split_screening_feature_pool,
    )

    feature_cols = _enabled_tg_model_feature_cols()
    split = split_screening_feature_pool(feature_cols)
    design_names = {v.feature for v in split["design"]}

    leaked = [
        verdict.feature
        for verdict in split["auto"]
        if verdict.feature in design_names
    ]
    assert not leaked, f"同时被判定为可计算与设计变量：{leaked[:5]}"

    # 逐个复核：枚举池成员绝不能是 auto 来源
    for verdict in classify_computable_features(sorted(design_names)):
        assert verdict.source not in AUTO_SOURCES, (
            f"{verdict.feature} 可自动计算却进了枚举池"
        )
        assert verdict.is_design_variable, f"{verdict.feature} 进了枚举池但不是设计变量"


def test_real_model_design_pool_is_small_and_expected():
    """设计变量必须收敛到「配比 + 工艺/测试条件」，不能是上百个。"""
    from core.computable_features import split_screening_feature_pool

    feature_cols = _enabled_tg_model_feature_cols()
    split = split_screening_feature_pool(feature_cols)
    design = {v.feature for v in split["design"]}

    assert len(design) <= 40, f"设计变量过多（{len(design)}），疑似误枚举"
    assert "resin_1_amount_phr" in design
    assert "tg_c_test_method" in design
    # 这些绝不能出现在枚举池里
    for forbidden in (
        "formulation_r_value",
        "process_total_time_h",
        "process_max_temperature_c",
        "resin_epoxy_group_total",
        "resin_1_molecular_weight_g_mol",
    ):
        assert forbidden not in design, f"{forbidden} 不应出现在枚举池"


# ---------------------------------------------------------------------------
# 固化制度 → process_* 派生
# ---------------------------------------------------------------------------
def test_process_family_derivation_matches_process_features_contract():
    """process_* 族与 core.process_features 的 cure_ 族必须同口径。"""
    from core.computable_features import derive_process_family_from_schedule
    from core.process_features import _schedule_pairs

    schedule = "130 °C/2 h + 180 °C/2 h"
    pairs = _schedule_pairs(schedule)
    derived = derive_process_family_from_schedule(schedule)

    assert derived["process_max_temperature_c"] == pytest.approx(max(t for t, _ in pairs))
    assert derived["process_total_time_h"] == pytest.approx(sum(d for _, d in pairs))
    assert derived["process_temperature_time_integral_c_h"] == pytest.approx(
        sum(t * d for t, d in pairs)
    )
    assert derived["process_final_cure_temperature_c"] == pytest.approx(pairs[-1][0])
    assert derived["process_final_cure_time_h"] == pytest.approx(pairs[-1][1])
    assert derived["process_has_post_cure"] == 0.0


def test_process_family_marks_post_cure_when_schedule_given():
    from core.computable_features import derive_process_family_from_schedule

    derived = derive_process_family_from_schedule(
        "150 °C/2 h", post_cure_schedule="200 °C/1 h"
    )
    assert derived["process_has_post_cure"] == 1.0
    assert derived["process_max_temperature_c"] == pytest.approx(200.0)
    assert derived["process_total_time_h"] == pytest.approx(3.0)


def test_process_family_returns_empty_on_unparseable_schedule():
    """不可解析时返回空 dict，绝不伪造数值。"""
    from core.computable_features import derive_process_family_from_schedule

    assert derive_process_family_from_schedule("") == {}
    assert derive_process_family_from_schedule(None) == {}
    assert derive_process_family_from_schedule("室温/24 h") == {}


def test_apply_schedule_derivations_fills_missing_only():
    """已有实测值优先，只补空位；不覆盖用户/实验数据。"""
    from core.computable_features import apply_schedule_derivations

    frame = pd.DataFrame(
        {
            "cure_schedule": ["130 °C/2 h + 180 °C/2 h", "80 °C/3 h"],
            "process_total_time_h": [99.0, None],
        }
    )
    out, filled = apply_schedule_derivations(
        frame,
        feature_cols=["process_total_time_h", "process_max_temperature_c"],
        schedule_column="cure_schedule",
    )
    # 第 0 行保留实测 99.0，第 1 行补 3.0
    assert out.loc[0, "process_total_time_h"] == pytest.approx(99.0)
    assert out.loc[1, "process_total_time_h"] == pytest.approx(3.0)
    assert out.loc[0, "process_max_temperature_c"] == pytest.approx(180.0)
    assert "process_max_temperature_c" in filled
    # 原表不得被改动
    assert frame.loc[0, "process_total_time_h"] == pytest.approx(99.0)


def test_apply_schedule_derivations_noop_without_schedule_column():
    from core.computable_features import apply_schedule_derivations

    frame = pd.DataFrame({"other": [1.0]})
    out, filled = apply_schedule_derivations(
        frame,
        feature_cols=["process_total_time_h"],
        schedule_column="cure_schedule",
    )
    assert filled == []
    assert "process_total_time_h" not in out.columns


def test_derived_feature_requirements_lists_only_needed_features():
    from core.computable_features import derived_feature_requirements

    requirements = derived_feature_requirements(
        ["process_total_time_h", "resin_1_amount_phr", "cure_stage_count"]
    )
    assert set(requirements) == {"process_total_time_h", "cure_stage_count"}
    for columns in requirements.values():
        assert columns, "派生特征必须声明所需原始列"


def test_resolve_schedule_source_column_prefers_canonical_name():
    from core.computable_features import resolve_schedule_source_column

    assert resolve_schedule_source_column(["a", "cure_schedule"]) == "cure_schedule"
    assert resolve_schedule_source_column(["固化制度"]) == "固化制度"
    assert resolve_schedule_source_column(["nothing_useful"]) is None
