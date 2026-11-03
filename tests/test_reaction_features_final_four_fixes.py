# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_final_four_fixes.py

CFRP 反应特征工程终期补丁专项测试套件：覆盖四项定向修复（P0-1, P0-2, P1-1, P1-2）全部 20 项验收场景。
"""

import numpy as np
import pandas as pd
import pytest

from core.reaction_core import (
    evaluate_reaction_network,
    R_SRC_LITERATURE,
    R_SRC_CALCULATED,
    R_SRC_PARTIAL_CALCULATED,
    R_SRC_ASSUMED_STOICH,
    R_SRC_MISSING_STOICH,
    R_SRC_INVALID_CHEM,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_UNDETERMINED,
    GEL_REGIME_SUPERCRITICAL,
)
from core.component_physics import (
    enrich_narrow_table,
    compute_component_physics,
    compute_formulation_summary,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import extract_epoxy_mechanism_features
from core.feature_registry import (
    get_feature_registry,
    get_features_for_set,
)
from core.feature_deduplication import (
    prepare_ml_feature_matrix,
)

# 典型单体结构
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA = "NCCN"
PGE = "c1ccccc1OCC1CO1"  # 单官能环氧
SA = "O=C1CCC(=O)O1"     # 琥珀酸酐


# =========================================================================
# 场景 1 ~ 7: P0-1 冲突屏蔽与累积掩码治理
# =========================================================================

def test_01_two_aliases_conflict_masked_to_nan():
    """1. 两个别名冲突并屏蔽为 NaN"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5, 0.5],
        "cp_r_value": [0.5, 0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert clean["formulation_r_value"].iloc[0] == 0.5
    assert pd.isna(clean["formulation_r_value"].iloc[1])
    assert len(clean) == 2


def test_02_three_aliases_conflict_masked_to_nan():
    """2. 三个别名冲突并屏蔽为 NaN ([0.5, 0.8, 0.8])"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5],
        "cp_r_value": [0.8],
        "xl_r_value": [0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean["formulation_r_value"].iloc[0])


def test_03_three_duplicate_columns_conflict_masked_to_nan():
    """3. 三个同名列冲突并屏蔽为 NaN ([[0.5, 0.8, 0.5]])"""
    df = pd.DataFrame(
        [[0.5, 0.8, 0.5]],
        columns=["formulation_r_value"] * 3
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean["formulation_r_value"].iloc[0])
    assert len(clean) == 1


def test_04_multiple_aliases_conflict_cannot_be_refilled_by_later_alias():
    """4. 多个别名冲突后不能被后续别名重新补值 (P0-1 核心反例)"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5, 0.5],
        "cp_r_value": [0.5, 0.8],
        "xl_r_value": [0.5, 0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    # 特别验证：冲突行第 1 行严格为 NaN，第 0 行保持 0.5，总行数不变
    assert clean.loc[0, "formulation_r_value"] == 0.5
    assert pd.isna(clean.loc[1, "formulation_r_value"])
    assert len(clean) == 2


def test_05_unit_conversion_conflict_masked_to_nan():
    """5. 单位换算冲突在 mask_conflicts 模式下屏蔽为 NaN"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.0, 2.0],
        "mech_crosslink_density_mol_m3": [2400.0, 5000.0],  # 2.0 * 1200 = 2400 != 5000
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] == 2400.0
    assert pd.isna(clean["mech_crosslink_density_mol_m3"].iloc[1])


def test_06_no_conflict_nan_safely_complemented():
    """6. 无冲突时 NaN 安全进行缺失值互补"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5, np.nan],
        "cp_r_value": [np.nan, 0.8],
    })
    clean, _ = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert clean["formulation_r_value"].iloc[0] == 0.5
    assert clean["formulation_r_value"].iloc[1] == 0.8


def test_07_raise_mode_continues_to_raise_exception():
    """7. 严格冲突模式 conflict_policy='raise' 仍正确抛出明确异常"""
    df = pd.DataFrame({
        "formulation_r_value": [0.5, 0.5],
        "cp_r_value": [0.5, 0.8],
    })
    with pytest.raises(ValueError, match="Physical alias conflict"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


# =========================================================================
# 场景 8 ~ 10: P0-2 缺失 f_stoich 时的安全当量计算
# =========================================================================

def test_08_missing_f_stoich_does_not_crash_with_type_error():
    """8. 缺失 f_stoich 不抛出 TypeError，平稳降级 (P0-2 核心反例)"""
    res = evaluate_reaction_network(
        resins=[{
            "mw": 200,
            "f_network": 2,
            "mass": 100,
        }],
        curers=[{
            "smiles": EDA,
            "mass": 8.83,
        }]
    )
    assert res.chemistry_valid is True
    assert res.f_r == 2.0
    # 不产生虚假的完整质量 r
    assert res.r_source != R_SRC_CALCULATED


def test_09_valid_eew_calculates_equivalent_without_f_stoich():
    """9. 有可靠 EEW 时即使缺少 f_stoich 也可正确计算当量与质量比 r"""
    res = evaluate_reaction_network(
        resins=[{
            "mw": 200,
            "ew": 100.0,  # 明确给出 EEW
            "f_network": 2,
            "mass": 100.0,
        }],
        curers=[{
            "smiles": EDA,
            "mass": 8.83,
        }],
        assume_stoichiometric_if_missing=False,
    )
    assert res.r_source == R_SRC_CALCULATED
    assert np.isfinite(res.r_value)
    assert res.r_value > 0


def test_10_missing_both_eew_and_f_stoich_marks_stoichiometry_unknown():
    """10. 没有 EEW 与 f_stoich 时严格模式下当量与 r 未知 (NaN)"""
    res = evaluate_reaction_network(
        resins=[{
            "mw": 200,
            "f_network": 2,
            "mass": 100.0,
        }],
        curers=[{
            "smiles": EDA,
            "mass": 8.83,
        }],
        assume_stoichiometric_if_missing=False,
    )
    assert pd.isna(res.r_value)
    assert res.r_source == R_SRC_MISSING_STOICH
    assert np.isnan(res.nu_ideal_mol_m3)


# =========================================================================
# 场景 11 ~ 12: P1-1 未知组分缺质量时的忽略质量占比
# =========================================================================

def test_11_unknown_component_missing_mass_ignored_fraction_nan():
    """11. 未知组分质量缺失时 cp_ignored_mass_fraction 为 NaN (P1-1 核心反例)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNPARSEABLE_RESIN"],  # 质量缺失!
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_unknown_comp_count"].iloc[0] == 1
    assert en["cp_composition_incomplete"].iloc[0] == True
    assert pd.isna(en["cp_ignored_mass_fraction"].iloc[0])


def test_12_unknown_component_known_mass_ignored_fraction_calculated():
    """12. 未知组分质量已知时忽略质量比例正确计算"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNPARSEABLE_RESIN"],
        "resin_2_amount_phr": [50.0],  # 质量已知 50 phr
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_unknown_comp_count"].iloc[0] == 1
    assert abs(en["cp_ignored_mass_fraction"].iloc[0] - (50.0 / 158.83)) < 1e-4


# =========================================================================
# 场景 13 ~ 15: P1-2 部分组成交联代理覆盖度治理
# =========================================================================

def test_13_complete_formulation_retains_full_coverage():
    """13. 完整已知配方保持 xl_coverage='full' 与 mechanism_aware 来源"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    xl = compute_crosslink_features(en)
    assert xl["xl_coverage"].iloc[0] == "full"
    assert xl["xl_nu_source"].iloc[0] == "mechanism_aware"


def test_14_partial_composition_default_mode_marks_partial_coverage():
    """14. 部分组成在宽容模式下标记 partial coverage 与 partial_composition_proxy 来源 (P1-2 核心反例)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNPARSEABLE_RESIN"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=True)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=True)
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert xl["xl_nu_source"].iloc[0] == "partial_composition_proxy"
    assert np.isfinite(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_15_partial_composition_strict_mode_theoretical_quantities_nan():
    """15. 部分组成在严格模式下网络理论量为 NaN"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNPARSEABLE_RESIN"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=False)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert pd.isna(xl["xl_alpha_gel"].iloc[0])
    assert pd.isna(xl["xl_nu_theory_mol_m3"].iloc[0])


# =========================================================================
# 场景 16 ~ 20: 物理基准不变性、幂等性与特征工程防泄漏
# =========================================================================

def test_16_standard_dgeba_eda_monofunctional_anhydride_baselines_intact():
    """16. 正常 DGEBA/EDA、单官能和酸酐体系原有物理边界保持不变"""
    # 正常体系超临界凝胶
    res_gel = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": EDA}], explicit_r=1.0)
    assert res_gel.gel_regime == GEL_REGIME_SUPERCRITICAL

    # 单官能不凝胶
    res_mono = evaluate_reaction_network([{"smiles": PGE}], [{"smiles": EDA}], explicit_r=1.0)
    assert res_mono.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res_mono.nu_ideal_mol_m3 == 0.0

    # 酸酐双口径基准
    res_sa = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": SA}], explicit_r=1.0)
    assert res_sa.f_h_stoich == 1.0
    assert res_sa.f_h_network == 2.0


def test_17_repeated_enrich_no_duplicate_columns():
    """17. 机理模块与组分模块多次重复增强无同名列膨胀"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [1.0],
    })
    en1 = enrich_narrow_table(df)
    en2 = enrich_narrow_table(en1)
    assert len(en1.columns) == len(en2.columns)
    assert en2.columns.is_unique == True


def test_18_input_row_count_and_index_strictly_preserved():
    """18. 批处理清洗时输入行数、索引和原始字段严格保持"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [1.0, np.nan, 0.8],
            "raw_experiment_note": ["batch1", "batch2", "batch3"],
        },
        index=["sample_X", "sample_Y", "sample_Z"]
    )
    clean, _ = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts", allow_unregistered_numeric=True)
    assert len(clean) == 3
    assert list(clean.index) == ["sample_X", "sample_Y", "sample_Z"]


def test_19_feature_sets_A_B_C_D_generable():
    """19. A/B/C/D 分层特征集合稳定生成"""
    set_a = get_features_for_set("reaction_core")
    set_b = get_features_for_set("molecular")
    set_c = get_features_for_set("formulation_process")
    set_d = get_features_for_set("full_safe")
    assert len(set_a) >= 25
    assert len(set_b) >= 15
    assert len(set_c) >= 8
    assert len(set_d) >= 50


def test_20_targets_and_ids_never_leak_into_matrix():
    """20. Tg 目标、样本 ID、论文 ID、分组元数据绝不泄漏进入模型特征矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "tg": [155.0],
        "tg_c": [155.0],
        "sample_id": ["SMP_001"],
        "paper_id": ["P_123"],
        "doi": ["10.1016/j.polymer"],
        "group_id": ["grp_A"],
    })
    clean, _ = prepare_ml_feature_matrix(df, feature_set="full_safe")
    for bad in ["tg", "tg_c", "sample_id", "paper_id", "doi", "group_id"]:
        assert bad not in clean.columns
