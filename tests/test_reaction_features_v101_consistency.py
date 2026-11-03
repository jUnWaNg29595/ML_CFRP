# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_v101_consistency.py

CFRP 项目 reaction_features_v1.0.1 核心修复、数据契约一致性与特征治理验收测试套件：
覆盖任务书第十一节定义的 30 项严密场景：
A. 重复增强 (1-4)
B. r 与来源一致性 (5-12)
C. 凝胶状态严密判定 (13-17)
D. 组分覆盖与数据质量治理 (18-20)
E. 统一特征注册表与训练矩阵去重 (21-26)
F. 工程兼容性与物理尺度不变性 (27-30)
"""

import math
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from core.reaction_core import (
    evaluate_reaction_network,
    compute_molar_average_functionality,
    compute_gel_point_conversion,
    NetworkReactionSummary,
    R_SRC_LITERATURE,
    R_SRC_CALCULATED,
    R_SRC_ASSUMED_STOICH,
    R_SRC_MISSING_STOICH,
    R_SRC_INVALID_CHEM,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_CRITICAL,
    GEL_REGIME_SUPERCRITICAL,
    GEL_REGIME_UNDETERMINED,
    GEL_ASSUMPTION_IDEAL,
    GEL_ASSUMPTION_MECHANISM_APPROX,
)
from core.component_physics import (
    compute_component_physics,
    compute_formulation_summary,
    enrich_narrow_table,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import (
    EpoxyMechanismEngine,
    extract_epoxy_mechanism_features,
)
from core.feature_registry import (
    get_feature_registry,
    get_canonical_feature_names,
    get_duplicate_feature_groups,
    export_registry_reports,
)
from core.feature_deduplication import prepare_ml_feature_matrix

SMI_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"  # f_r = 2
SMI_CCN = "CCN"                                            # 乙胺: f_stoich = 2, f_network = 2
SMI_EDA = "NCCN"                                            # 乙二胺: f_stoich = 4, f_network = 4
SMI_PGE = "c1ccccc1OCC1CO1"                                 # 苯基缩水甘油醚: 单官能环氧 f_r = 1
SMI_SA = "O=C1CCC(=O)O1"                                    # 琥珀酸酐: 单酸酐 f_stoich = 1, f_network = 2


# ===========================================================================
# Group A: 重复增强 (1-4)
# ===========================================================================

def test_01_mechanism_repeated_enrich_no_duplicate_columns():
    """1. 机理模块增强两次无重复列，列名唯一且列数不膨胀 (P0-1)"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    df1 = extract_epoxy_mechanism_features(df)
    df2 = extract_epoxy_mechanism_features(df1)

    assert df1.columns.is_unique, "初次增强后列名必须唯一"
    assert df2.columns.is_unique, "重复增强后列名必须唯一"
    assert len(df2.columns) == len(df1.columns), "重复增强后特征列数不得膨胀翻倍"
    assert not any(c.endswith("_x") or c.endswith("_y") for c in df2.columns)


def test_02_four_modules_sequential_enrich_no_duplicates():
    """2. 四模块流水线连续增强无重复列名"""
    df = pd.DataFrame([{"resin_1_structure": SMI_DGEBA, "curing_agent_1_structure": SMI_EDA}])
    en1 = enrich_narrow_table(df)
    en2 = compute_crosslink_features(en1)
    en3 = extract_epoxy_mechanism_features(pd.concat([en1, en2[[c for c in en2.columns if c not in en1.columns]]], axis=1))

    assert en3.columns.is_unique
    assert not any(c.endswith("_x") or c.endswith("_y") for c in en3.columns)


def test_03_custom_index_preserved_across_repeated_enrich():
    """3. 自定义行索引在重复增强中严格保持"""
    custom_ids = ["SAMPLE_A", "SAMPLE_B"]
    df = pd.DataFrame([
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 1.0},
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 0.5},
    ], index=custom_ids)

    df1 = extract_epoxy_mechanism_features(df)
    df2 = extract_epoxy_mechanism_features(df1)
    assert list(df1.index) == custom_ids
    assert list(df2.index) == custom_ids


def test_04_identical_samples_stable_across_repeated_enrich():
    """4. 同一原始样本重复增强结果完全自洽稳定"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    df1 = extract_epoxy_mechanism_features(df)
    df2 = extract_epoxy_mechanism_features(df1)

    for col in ["mech_stoichiometry_r", "mech_average_functionality", "mech_theoretical_alpha_gel", "mech_theoretical_Mc"]:
        assert pytest.approx(df1[col].iloc[0], abs=1e-4) == df2[col].iloc[0]


# ===========================================================================
# Group B: r 与来源一致性 (5-12)
# ===========================================================================

def test_05_explicit_literature_r_consistency_across_four_modules():
    """5. (P0-2) cp_r_value=0.5 且标记 literature 时，四模块解析结果严格一致"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "curing_agent_1_structure": SMI_EDA,
        "cp_r_value": 0.5,
        "cp_r_source": "literature",
    }])

    cp = compute_component_physics(df)
    xl = compute_crosslink_features(df, cp=cp)
    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df)
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], explicit_r=0.5, existing_r_source="literature")

    assert pytest.approx(cp["cp_r_value"].iloc[0], abs=1e-4) == 0.5
    assert cp["cp_r_source"].iloc[0] == "literature"

    assert pytest.approx(xl["xl_r_value"].iloc[0], abs=1e-4) == 0.5
    assert xl["xl_r_source"].iloc[0] == "literature"

    assert pytest.approx(mech["mech_stoichiometry_r"].iloc[0], abs=1e-4) == 0.5
    assert rc.r_value == 0.5
    assert rc.r_source == "literature"


def test_06_strict_mode_preserves_true_literature_r():
    """6. 严格模式下对于真实文献 r=0.5 不得将其置为 NaN，正常保留"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "curing_agent_1_structure": SMI_EDA,
        "cp_r_value": 0.5,
        "cp_r_source": "literature",
    }])
    cp_str = compute_component_physics(df, assume_stoichiometric_if_missing=False)
    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)

    assert cp_str["cp_r_value"].iloc[0] == 0.5
    assert cp_str["cp_r_source"].iloc[0] == "literature"
    assert xl_str["xl_r_value"].iloc[0] == 0.5
    assert xl_str["xl_r_source"].iloc[0] == "literature"


def test_07_complete_mass_calculated_r_source_retention():
    """7. 完整质量计算得到的 r，其来源严格保持为 calculated_from_mass，不被误认"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 17.65,
    }])
    cp = compute_component_physics(df)
    assert pytest.approx(cp["cp_r_value"].iloc[0], abs=1e-2) == 2.0
    assert cp["cp_r_source"].iloc[0] == "calculated_from_mass"
    assert cp["rxn_r_assumed"].iloc[0] == 0.0


def test_08_historical_assumed_r_overridden_by_new_mass():
    """8. (P0-4) 历史人工假设 r=1.0 在新补充完整质量时，必须被重新计算并升级为 calculated_from_mass"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA,
        "curing_agent_1_phr": 17.65,
        "formulation_r_value": 1.0,
        "rxn_r_assumed": 1.0,
        "cp_r_source": "assumed_stoichiometric",
    }])
    # 默认模式下重算
    cp = compute_component_physics(df, assume_stoichiometric_if_missing=True)
    assert pytest.approx(cp["cp_r_value"].iloc[0], abs=1e-2) == 2.0, "新质量到达必须覆盖历史假定值"
    assert cp["cp_r_source"].iloc[0] == "calculated_from_mass", "必须升级为 calculated_from_mass"
    assert cp["rxn_r_assumed"].iloc[0] == 0.0

    # 严格模式下重算
    cp_str = compute_component_physics(df, assume_stoichiometric_if_missing=False)
    assert pytest.approx(cp_str["cp_r_value"].iloc[0], abs=1e-2) == 2.0
    assert cp_str["cp_r_source"].iloc[0] == "calculated_from_mass"
    assert cp_str["rxn_r_assumed"].iloc[0] == 0.0


def test_09_literature_r_conflicts_with_mass_prioritizes_literature():
    """9. 文献明确 r=0.5 与质量推导 r=2.0 冲突时，优先保留文献 r 并标记冲突"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 17.65,
        "formulation_r_value": 0.5,
    }])
    cp = compute_component_physics(df)
    assert cp["cp_r_value"].iloc[0] == 0.5
    assert cp["cp_r_source"].iloc[0] == "literature"
    assert pytest.approx(cp["cp_r_from_mass"].iloc[0], abs=1e-2) == 2.0
    assert cp["cp_r_conflict"].iloc[0] == 1.0


def test_10_multiple_r_columns_conflict_explicitly_tagged():
    """10. 多列 r 存在时，明确按照规范文献列优先并记录推导差值"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.83,
        "formulation_r_value": 0.5,
    }])
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["formulation_r_value"].iloc[0] == 0.5
    assert en["cp_r_source"].iloc[0] == "literature"
    assert en["cp_r_conflict"].iloc[0] == 1.0


def test_11_enrich_save_reload_reparse_source_no_drift():
    """11. (P0-3) 默认增强 -> 保存 CSV/Parquet -> 重载 -> 再次解析，来源标签零漂移"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    en = enrich_narrow_table(df, keep_metadata=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p = os.path.join(tmpdir, "en.csv")
        parq_p = os.path.join(tmpdir, "en.parquet")
        en.to_csv(csv_p, index=False)
        en.to_parquet(parq_p, index=False)

        re_csv = enrich_narrow_table(pd.read_csv(csv_p), keep_metadata=True)
        re_parq = enrich_narrow_table(pd.read_parquet(parq_p), keep_metadata=True)

        assert re_csv["cp_r_source"].iloc[0] == "assumed_stoichiometric"
        assert re_parq["cp_r_source"].iloc[0] == "assumed_stoichiometric"


def test_12_historical_r_without_source_not_blindly_literature():
    """12. 历史数据无来源标记时，若为推导值不盲目宣称为文献实测"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.83,
        "formulation_r_value": 1.000299,
        "cp_r_from_mass": 1.000299,
    }])
    cp = compute_component_physics(df)
    assert cp["cp_r_source"].iloc[0] == "calculated_from_mass", "与质量完全相符的历史 r 应当判定为 calculated_from_mass"


# ===========================================================================
# Group C: 凝胶状态严密判定 (13-17)
# ===========================================================================

def test_13_strict_mode_missing_r_gel_regime_undetermined():
    """13. (P1-1) 严格模式下，反应结构有效但配比 r 未知时，凝胶状态必须判定为 undetermined，凝胶可行性为 NaN"""
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=False)
    assert rc.gel_regime == GEL_REGIME_UNDETERMINED, "配比未知且具备分支能力时必须为 undetermined，绝不能冒充 subcritical"
    assert np.isnan(rc.gel_possible_ideal)
    assert np.isnan(rc.alpha_gel)


def test_14_monofunctional_decisively_subcritical_even_without_r():
    """14. 单官能环氧体系结构上绝不可能凝胶，即使缺失 r 也可确定判定为 subcritical / gel_possible_ideal=0.0"""
    rc = evaluate_reaction_network([{"smiles": SMI_PGE}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=False)
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)


def test_15_a2_b2_critical_boundary():
    """15. A2-B2 r=1.0 临界边界保持正常：只能线性增长，判定为 critical，不可凝胶"""
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_CCN}], explicit_r=1.0)
    assert rc.gel_regime == GEL_REGIME_CRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)
    assert rc.nu_ideal_mol_m3 == 0.0


def test_16_a2_b4_supercritical_gelation():
    """16. A2-B4 r=1.0 超临界凝胶正常"""
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], explicit_r=1.0)
    assert rc.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert rc.gel_possible_ideal == 1.0
    assert pytest.approx(rc.alpha_gel, abs=1e-4) == 1.0 / math.sqrt(3.0)
    assert rc.nu_ideal_mol_m3 > 0


def test_17_a2_b4_r01_subcritical_non_gelling():
    """17. A2-B4 r=0.1 转化率物理不可达，判定为 subcritical，xl_nu_source='non_gelling'"""
    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], explicit_r=0.1)
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)

    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 0.1}])
    xl = compute_crosslink_features(df)
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])


# ===========================================================================
# Group D: 组分覆盖与数据质量治理 (18-20)
# ===========================================================================

def test_18_unknown_resin_component_marked_incomplete():
    """18. (P1-2) 配方混入未知树脂组分时，必须标记 composition_incomplete=True，记录 unknown_component_count"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA, "mass": 100.0}, {"smiles": "UNRECOGNIZED_TRASH", "mass": 50.0}],
        [{"smiles": SMI_EDA, "mass": 8.83}],
    )
    assert rc.unknown_component_count == 1
    assert rc.composition_incomplete is True
    assert rc.has_composition is False, "存在未识别质量组分时，配方组成不得标记为完整"
    assert rc.ignored_mass_fraction > 0.20


def test_19_nonreactive_additives_mass_accounted():
    """19. 质量统计能够准确计算被忽略未反应组分的质量占比"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA, "mass": 100.0}, {"smiles": "INERT_FILLER", "mass": 100.0}],
        [{"smiles": SMI_EDA, "mass": 8.83}],
    )
    assert rc.ignored_component_count == 1
    assert pytest.approx(rc.ignored_mass_fraction, abs=1e-2) == 100.0 / 208.83


def test_20_composition_mass_fractions_audited():
    """20. 完整已知配方的 ignored_mass_fraction 严格为 0.0，composition_incomplete 为 False"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA, "mass": 100.0}],
        [{"smiles": SMI_EDA, "mass": 8.83}],
    )
    assert rc.unknown_component_count == 0
    assert rc.ignored_mass_fraction == 0.0
    assert rc.composition_incomplete is False
    assert rc.has_composition is True


# ===========================================================================
# Group E: 统一特征注册表与训练矩阵去重 (21-26)
# ===========================================================================

def test_21_ml_feature_matrix_clean_no_duplicate_columns():
    """21. (P1-3) 经 prepare_ml_feature_matrix 处理后的训练矩阵绝对不存在同名重复列"""
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 1.0, "cp_r_value": 1.0, "xl_r_value": 1.0,
        "cp_balance": 1.0, "xl_balance": 1.0,
    }])
    clean_df, summary = prepare_ml_feature_matrix(df)
    assert clean_df.columns.is_unique, "清洗后特征矩阵列名必须严格唯一"


def test_22_exact_registered_aliases_deduplicated():
    """22. 注册表中的完全物理别名在 ML 矩阵中自动排除，仅保留规范特征"""
    df = pd.DataFrame([{
        "formulation_r_value": 1.0,
        "cp_r_value": 1.0,
        "xl_r_value": 1.0,
        "cp_balance": 1.0,
        "xl_balance": 1.0,
        "cp_f_avg_network": 2.67,
        "xl_f_avg": 2.67,
    }])
    clean_df, summary = prepare_ml_feature_matrix(df)
    assert "formulation_r_value" in clean_df.columns
    assert "cp_r_value" not in clean_df.columns
    assert "xl_r_value" not in clean_df.columns
    assert "cp_balance" in clean_df.columns
    assert "xl_balance" not in clean_df.columns
    assert "cp_f_avg_network" in clean_df.columns
    assert "xl_f_avg" not in clean_df.columns


def test_23_distinct_physical_models_preserved():
    """23. 具有不同物理模型定义的近似特征 (如机制网络链密度 vs Flory交联点数密度) 不被粗暴合并误删"""
    df = pd.DataFrame([{
        "xl_nu_theory_mol_m3": 4500.0,
        "xl_nu_junction_mol_m3": 3200.0,
        "mech_crosslink_density_mol_m3": 4100.0,
    }])
    clean_df, _ = prepare_ml_feature_matrix(df)
    assert "xl_nu_theory_mol_m3" in clean_df.columns
    assert "xl_nu_junction_mol_m3" in clean_df.columns
    assert "mech_crosslink_density_mol_m3" in clean_df.columns


def test_24_fmo_energy_statistics_preserved():
    """24. 多组分能级差分项中的 min, max, weighted 具有明确物理区分度，全部保留供特征消融"""
    df = pd.DataFrame([{
        "mech_fmo_delta_e_fwd_min": 3.5,
        "mech_fmo_delta_e_fwd_max": 4.8,
        "mech_fmo_delta_e_fwd_weighted": 4.1,
    }])
    clean_df, _ = prepare_ml_feature_matrix(df)
    assert "mech_fmo_delta_e_fwd_min" in clean_df.columns
    assert "mech_fmo_delta_e_fwd_max" in clean_df.columns
    assert "mech_fmo_delta_e_fwd_weighted" in clean_df.columns


def test_25_dedup_actions_and_reasons_auditable():
    """25. 特征清洗返回完整的审计摘要，每个剔除动作均记录明确原因"""
    df = pd.DataFrame([{
        "formulation_r_value": 1.0, "cp_r_value": 1.0,
        "mech_crosslink_density_mol_m3": 4000.0, "mech_crosslink_density_mmol_g": 3.33,
    }])
    clean_df, summary = prepare_ml_feature_matrix(df)
    assert isinstance(summary, pd.DataFrame)
    reasons = summary["reason"].tolist()
    assert any("exact_alias" in r for r in reasons)
    assert any("unit_conversion" in r for r in reasons)


def test_26_registry_reports_exported_successfully():
    """26. 成功导出 feature_registry.csv, feature_duplicate_groups.csv, feature_selection_summary.csv"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p1, p2, p3 = export_registry_reports(tmpdir)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        assert os.path.exists(p3)

        df1 = pd.read_csv(p1)
        assert "feature_name" in df1.columns
        assert len(df1) > 20


# ===========================================================================
# Group F: 工程兼容性与物理尺度不变性 (27-30)
# ===========================================================================

def test_27_legacy_api_and_key_columns_backward_compatible():
    """27. 既有核心公共特征及公共 API 入口保持 100% 兼容"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.83,
    }])
    en = enrich_narrow_table(df)
    xl = compute_crosslink_features(en)
    assert "formulation_r_value" in en.columns
    assert "cp_f_avg_network" in en.columns
    assert "xl_nu_theory_mol_m3" in xl.columns
    assert "xl_alpha_gel" in xl.columns


def test_28_default_mode_permits_r1_assumption():
    """28. 默认实用模式下允许缺失配比自动假设 r=1.0 并标记 assumed_stoichiometric"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    assert en["formulation_r_value"].iloc[0] == 1.0
    assert en["rxn_r_assumed"].iloc[0] == 1.0


def test_29_strict_mode_prohibits_silent_fallbacks():
    """29. 严格科研模式下彻底禁止未知配比的隐式回退，必须为 NaN 与 missing_stoichiometry"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(en["formulation_r_value"].iloc[0])
    assert en["rxn_r_assumed"].iloc[0] == 0.0


def test_30_mass_x10_scale_invariance():
    """30. 配方质量整体放大 10 倍，归一化物理特征 (r, f_avg, 凝胶点, 交联密度) 尺度严格不变"""
    df1 = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.827,
    }], index=["base"])

    df2 = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 1000.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 88.27,
    }], index=["scaled_x10"])

    xl1 = compute_crosslink_features(df1)
    xl2 = compute_crosslink_features(df2)

    assert pytest.approx(xl1["xl_r_value"].iloc[0], abs=1e-4) == xl2["xl_r_value"].iloc[0]
    assert pytest.approx(xl1["xl_f_avg"].iloc[0], abs=1e-4) == xl2["xl_f_avg"].iloc[0]
    assert pytest.approx(xl1["xl_alpha_gel"].iloc[0], abs=1e-4) == xl2["xl_alpha_gel"].iloc[0]
    assert pytest.approx(xl1["xl_nu_theory_mol_m3"].iloc[0], abs=1e-4) == xl2["xl_nu_theory_mol_m3"].iloc[0]
