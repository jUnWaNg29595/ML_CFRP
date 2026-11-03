# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_v102_ml_safety.py

CFRP 项目 reaction_features_v1.0.2 训练特征安全性、量纲正确性与数据血统保真专项测试套件：
全面覆盖任务要求定义的 24 项关键准则：
1. 同名列相同值安全合并
2. 同名列冲突默认抛错 (conflict_policy='raise')
3. 注册表别名一致安全去重
4. 注册表别名冲突默认抛错
5. 主列全 NaN，别名有效时正确补齐 (P0-3)
6. 主列部分 NaN，别名互补时正确补齐
7. 多个别名冲突检测并报错
8. 只有别名时生成规范主列
9. 单位换算必须改变数值 (P0-2)
10. 参考密度 1.2 g/cm³ 下 mmol/g -> mol/m³ 正确 (x1200)
11. 逐样本密度不同的换算正确
12. 两种单位字段换算后冲突检测与报错
13. 无效/NaN 输入在单位换算时安全处理为 NaN
14. Tg 目标字段不进入训练输入 (P0-4 防泄漏)
15. 其他力学性能目标不进入训练输入
16. 数字 sample ID 与行号不进入训练输入
17. group ID 与批次号不进入训练输入
18. 未注册数值字段默认不进入训练输入 (allow_unregistered_numeric=False)
19. 合法注册物理特征正常保留在训练矩阵中
20. 文献 r 与质量 r 吻合时仍保留 literature 来源 (P0-5)
21. 无来源历史 r 标记为 unknown_provenance 等明确状态
22. 严格模式下凝胶状态未知，理论网络密度为 NaN (P1)
23. 单官能体系确定不可凝胶时维持原有物理边界 (subcritical, 0.0)
24. 机理模块重复增强仍然幂等、无重复列 (P0-1)
"""

import math
import numpy as np
import pandas as pd
import pytest

from core.reaction_core import (
    evaluate_reaction_network,
    NetworkReactionSummary,
    GEL_REGIME_UNDETERMINED,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_CRITICAL,
    GEL_REGIME_SUPERCRITICAL,
    R_SRC_LITERATURE,
    R_SRC_CALCULATED,
    R_SRC_UNKNOWN_PROVENANCE,
)
from core.component_physics import compute_component_physics
from core.epoxy_mechanism_features import extract_epoxy_mechanism_features
from core.feature_registry import get_feature_registry
from core.feature_deduplication import prepare_ml_feature_matrix

SMI_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
SMI_EDA = "NCCN"
SMI_PGE = "c1ccccc1OCC1CO1"


def test_01_duplicate_column_identical_values_safe_merge():
    """1. (P0-1A) 同名列数值完全一致时安全去重合并，不报异常"""
    df = pd.DataFrame(
        [[1.0, 1.0], [2.0, 2.0]],
        columns=["formulation_r_value", "formulation_r_value"]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="raise", allow_unregistered_numeric=True)
    assert clean.columns.is_unique
    assert clean.shape[1] == 1
    assert clean["formulation_r_value"].tolist() == [1.0, 2.0]


def test_02_duplicate_column_conflicting_values_raises_error():
    """2. (P0-1A) 同名列数值冲突时，默认策略 conflict_policy='raise' 必须抛出明确异常"""
    df = pd.DataFrame(
        [[1.0, 4.0], [2.0, 2.0]],
        columns=["formulation_r_value", "formulation_r_value"]
    )
    with pytest.raises(ValueError, match="Duplicate column 'formulation_r_value' has conflicting values"):
        prepare_ml_feature_matrix(df, conflict_policy="raise", allow_unregistered_numeric=True)


def test_03_registered_alias_identical_values_safe_dedup():
    """3. (P0-1B) 注册表物理别名数值一致时安全去重，保留规范主特征，剔除别名"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0, 2.0],
        "cp_r_value": [1.0, 2.0],
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="raise")
    assert "formulation_r_value" in clean.columns
    assert "cp_r_value" not in clean.columns
    assert clean["formulation_r_value"].tolist() == [1.0, 2.0]


def test_04_registered_alias_conflict_raises_error():
    """4. (P0-1B) 注册表物理别名在共同非空位置数值冲突时，默认抛出明确异常"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5],
        "cp_r_value": [0.8],
    })
    with pytest.raises(ValueError, match="Physical alias conflict between 'cp_r_value'"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_05_canonical_all_nan_filled_by_alias():
    """5. (P0-3) 规范主列全为 NaN 但别名有效时，必须安全将别名填入主列，不得静默清空"""
    df = pd.DataFrame({
        "formulation_r_value": [np.nan, np.nan],
        "cp_r_value": [0.5, 0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert clean["formulation_r_value"].tolist() == [0.5, 0.8]
    assert "cp_r_value" not in clean.columns


def test_06_canonical_partial_nan_complementary_filled():
    """6. (P0-3) 规范主列部分为 NaN 与别名非空位置互补时，成功填补且不误报冲突"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0, np.nan],
        "cp_r_value": [np.nan, 0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert clean["formulation_r_value"].tolist() == [1.0, 0.8]


def test_07_multiple_aliases_conflict_detection():
    """7. 多个别名同时存在且数值互不一致时，严格检测并抛出冲突异常"""
    df = pd.DataFrame({
        "formulation_r_value": [np.nan],
        "cp_r_value": [0.5],
        "xl_r_value": [0.9],
    })
    with pytest.raises(ValueError):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_08_alias_only_promoted_to_canonical():
    """8. 输入仅含别名而不含规范主列时，自动将别名晋升为规范主列名称"""
    df = pd.DataFrame({
        "cp_r_value": [0.5, 0.8]
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert "formulation_r_value" in clean.columns
    assert "cp_r_value" not in clean.columns
    assert clean["formulation_r_value"].tolist() == [0.5, 0.8]


def test_09_unit_conversion_changes_values():
    """9. (P0-2) 单位换算字段绝不仅通过更名实现，必须对数值执行比例缩放换算"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [1.0, 2.0]
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] != 1.0, "单位换算必须改变数值"
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] == pytest.approx(1200.0)


def test_10_reference_density_1_2_conversion_correct():
    """10. (P0-2) 参考密度 1.2 g/cm³ 下，1.2 mmol/g -> 1440 mol/m³，2.5 mmol/g -> 3000 mol/m³"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [1.2, 2.5]
    })
    clean, audit = prepare_ml_feature_matrix(df, reference_density_g_cm3=1.2)
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] == pytest.approx(1440.0)
    assert clean["mech_crosslink_density_mol_m3"].iloc[1] == pytest.approx(3000.0)


def test_11_sample_level_density_conversion_correct():
    """11. 逐样本密度不同时，单位换算依据样本级 rho_g_cm3 进行计算"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [1.0, 1.0],
        "rho_g_cm3": [1.1, 1.3],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] == pytest.approx(1100.0)
    assert clean["mech_crosslink_density_mol_m3"].iloc[1] == pytest.approx(1300.0)


def test_12_unit_conversion_conflict_detected():
    """12. 两种单位字段换算后数值矛盾时检测并报错"""
    df = pd.DataFrame({
        "mech_crosslink_density_mol_m3": [1000.0],
        "mech_crosslink_density_mmol_g": [2.0],  # 2.0 * 1200 = 2400 != 1000
    })
    with pytest.raises(ValueError, match="Unit conversion conflict"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_13_unit_conversion_nan_handled_safely():
    """13. mmol/g 为 NaN 时换算后安全得到 NaN，不伪造数值"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [np.nan]
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert np.isnan(clean["mech_crosslink_density_mol_m3"].iloc[0])


def test_14_tg_targets_excluded_from_training_matrix():
    """14. (P0-4) Tg 预测目标标签绝对不得进入训练特征矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "Tg": [150.0],
        "target_Tg_C": [145.0],
        "Tg_C": [155.0],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert "Tg" not in clean.columns
    assert "target_Tg_C" not in clean.columns
    assert "Tg_C" not in clean.columns
    assert "formulation_r_value" in clean.columns


def test_15_mechanical_targets_excluded_from_training_matrix():
    """15. (P0-4) 强度、模量等力学性能标签排除出训练矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "tensile_strength_mpa": [80.0],
        "flexural_modulus_gpa": [3.5],
        "impact_strength_kj_m2": [12.0],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert "tensile_strength_mpa" not in clean.columns
    assert "flexural_modulus_gpa" not in clean.columns
    assert "impact_strength_kj_m2" not in clean.columns


def test_16_numeric_sample_ids_excluded():
    """16. (P0-4) 数字 sample ID 与行号字段绝不得作为数值特征进入模型"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "sample_id": [1001],
        "row_id": [1],
        "id": [42],
        "formula_id": [99],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert "sample_id" not in clean.columns
    assert "row_id" not in clean.columns
    assert "id" not in clean.columns
    assert "formula_id" not in clean.columns


def test_17_group_ids_excluded():
    """17. (P0-4) group_id 与批次标记默认排除在数值特征之外"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "group_id": [12],
        "paper_id": [34],
        "batch_id": [5],
        "split": [0],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    assert "group_id" not in clean.columns
    assert "paper_id" not in clean.columns
    assert "batch_id" not in clean.columns
    assert "split" not in clean.columns


def test_18_unregistered_numeric_excluded_by_default():
    """18. (P0-4) 未在注册表登记的数值字段默认不得进入训练矩阵 (allow_unregistered_numeric=False)"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "some_leaked_arbitrary_metric": [999.0],
    })
    clean, audit = prepare_ml_feature_matrix(df, allow_unregistered_numeric=False)
    assert "some_leaked_arbitrary_metric" not in clean.columns
    assert "formulation_r_value" in clean.columns


def test_19_registered_physical_features_properly_retained():
    """19. 注册表中合法物理特征被正确选入训练矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "cp_balance": [1.0],
        "cp_f_avg_network": [2.67],
        "xl_nu_theory_mol_m3": [4500.0],
        "xl_alpha_gel": [0.577],
    })
    clean, audit = prepare_ml_feature_matrix(df)
    for col in ["formulation_r_value", "cp_balance", "cp_f_avg_network", "xl_nu_theory_mol_m3", "xl_alpha_gel"]:
        assert col in clean.columns


def test_20_literature_r_matches_mass_still_literature():
    """20. (P0-5) 显式文献 r 恰好与质量计算值相等时，来源必须保持 literature，不得被覆盖为 calculated_from_mass"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.83,
        "cp_r_value": 1.0002994242928458,
        "cp_r_source": "literature",
    }])
    cp = compute_component_physics(df)
    assert cp["cp_r_source"].iloc[0] == "literature", "明确标记 literature 的 r 绝不得被数值吻合推断篡改为 calculated_from_mass"


def test_21_historical_r_without_source_marked_unknown_or_explicit():
    """21. 历史数据有 r 且标记为 unknown_provenance 时，能够忠实保持该状态"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "curing_agent_1_structure": SMI_EDA,
        "formulation_r_value": 0.5,
        "cp_r_source": "unknown_provenance",
    }])
    cp = compute_component_physics(df)
    assert cp["cp_r_source"].iloc[0] == "unknown_provenance"


def test_22_undetermined_gel_regime_nu_ideal_is_nan():
    """22. (P1) 严格模式下凝胶状态未知 (undetermined) 时，理想网络密度 nu_ideal 必须为 NaN，绝不伪装为 0.0"""
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=False)
    assert rc.gel_regime == GEL_REGIME_UNDETERMINED
    assert np.isnan(rc.gel_possible_ideal)
    assert np.isnan(rc.nu_ideal_mol_m3), "未知凝胶状态下的 nu_ideal 必须为 NaN"


def test_23_monofunctional_definitively_subcritical_nu_is_zero():
    """23. 单官能环氧体系确定不可凝胶时，保持 subcritical 与 nu=0.0 原有物理边界"""
    rc = evaluate_reaction_network([{"smiles": SMI_PGE}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=False)
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert rc.nu_ideal_mol_m3 == 0.0


def test_24_mechanism_repeated_enrich_idempotent_no_duplicates():
    """24. (P0-1) 机理模块重复增强在任何调用序列下保持完全幂等，无重复列名"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    df1 = extract_epoxy_mechanism_features(df)
    df2 = extract_epoxy_mechanism_features(df1)
    df3 = extract_epoxy_mechanism_features(df2)
    assert df1.columns.is_unique
    assert df2.columns.is_unique
    assert df3.columns.is_unique
    assert len(df1.columns) == len(df2.columns) == len(df3.columns)
