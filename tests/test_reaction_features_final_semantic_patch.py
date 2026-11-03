# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_final_semantic_patch.py

CFRP 反应特征工程最终定向语义修复与含噪声文献容错专项回归套件。
严格覆盖 Section XI 要求的全部 30 项终期语义一致性与容错场景。
"""

import tempfile
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
    R_SRC_UNKNOWN_PROVENANCE,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_UNDETERMINED,
    GEL_REGIME_SUPERCRITICAL,
    GEL_REGIME_NOT_APPLICABLE,
)
from core.component_physics import (
    enrich_narrow_table,
    compute_component_physics,
    compute_formulation_summary,
    resolve_component,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import (
    EpoxyMechanismEngine,
    extract_epoxy_mechanism_features,
)
from core.feature_registry import (
    get_feature_registry,
    get_canonical_feature_names,
    get_features_for_set,
)
from core.feature_deduplication import (
    prepare_ml_feature_matrix,
    MLFeaturePipeline,
)

# 经典单体 SMILES
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA = "NCCN"
DDM = "Nc1ccc(Cc2ccc(N)cc2)cc1"
MONO_EPOXY = "CC(C)COC1CO1"  # 单官能异丁基缩水甘油醚
NON_EPOXY_ETHANOL = "CCO"     # 乙醇 (非环氧树脂)
NON_EPOXY_TOLUENE = "Cc1ccccc1" # 甲苯 (惰性非反应物)


# =========================================================================
# 一、化学模型适用性 (场景 1 ~ 7)
# =========================================================================

def test_01_dgeba_eda_generates_standard_physics():
    """1. DGEBA/EDA 正常生成完整物理特征"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA, "mass": 100.0}],
        [{"smiles": EDA, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.chemistry_valid is True
    assert res.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert res.gel_possible_ideal == 1.0
    assert np.isfinite(res.nu_ideal_mol_m3)
    assert res.nu_ideal_mol_m3 > 1000.0


def test_02_cco_nccn_does_not_output_positive_epoxy_density():
    """2. CCO/NCCN 非环氧体系不输出确定环氧网络密度，标记为 not_applicable (P0-1)"""
    res = evaluate_reaction_network(
        [{"smiles": NON_EPOXY_ETHANOL}],
        [{"smiles": EDA}],
    )
    assert res.chemistry_valid is False
    assert res.gel_regime == GEL_REGIME_NOT_APPLICABLE
    assert np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.nu_ideal_mol_m3)


def test_03_bad_nccn_does_not_output_zero_crosslink_density():
    """3. BAD/NCCN 无法解析体系不输出确定零交联密度，标记为 undetermined / NaN (P0-1)"""
    res = evaluate_reaction_network(
        [{"smiles": "CORRUPT_SMILES_BAD"}],
        [{"smiles": EDA}],
    )
    assert res.chemistry_valid is False
    assert res.gel_regime == GEL_REGIME_UNDETERMINED
    assert np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.nu_ideal_mol_m3)


def test_04_non_epoxy_valid_structure_retains_descriptors():
    """4. 非环氧但结构有效的样本保留分子描述符 (MW, TPSA, 芳环等)"""
    df = pd.DataFrame({
        "resin_1_structure": [NON_EPOXY_TOLUENE],
        "curing_agent_1_structure": [EDA],
    })
    en1 = enrich_narrow_table(df, keep_metadata=True)
    engine = EpoxyMechanismEngine(verbose=False)
    en2 = engine.enrich_dataframe(en1)
    assert len(en2) == 1
    assert en2["resin_1_has_structure"].iloc[0] == True
    # 结构有效，分子量与芳环数正常保留
    assert en2["resin_1_mw_resolved"].iloc[0] > 90.0
    assert en2["mech_weighted_aromatic_rings"].iloc[0] > 0.0


def test_05_broken_smiles_retains_process_and_formulation():
    """5. SMILES 错误依然保留文献工艺与配方字段 (phr, 固化温度, 时间, Tg等)"""
    df = pd.DataFrame({
        "resin_1_structure": ["%CORRUPTED%"],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "process_max_temperature_c": [150.0],
        "process_total_time_h": [4.0],
        "tg_c": [165.0],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert len(en) == 1
    assert en["resin_1_amount_phr"].iloc[0] == 100.0
    assert en["process_max_temperature_c"].iloc[0] == 150.0
    assert en["process_total_time_h"].iloc[0] == 4.0
    assert en["tg_c"].iloc[0] == 165.0


def test_06_monofunctional_epoxy_correctly_judged_subcritical():
    """6. 单官能且模型适用的环氧体系正确判为不可凝胶 (subcritical / 0.0)"""
    res = evaluate_reaction_network(
        [{"smiles": MONO_EPOXY, "mass": 100.0}],
        [{"smiles": EDA, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.chemistry_valid is True
    assert res.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res.gel_possible_ideal == 0.0
    assert res.nu_ideal_mol_m3 == 0.0


def test_07_distinguish_model_inapplicable_from_subcritical():
    """7. 模型不适用与确定亚临界能够明确区分"""
    res_inapplicable = evaluate_reaction_network([{"smiles": NON_EPOXY_ETHANOL}], [{"smiles": EDA}])
    res_subcritical = evaluate_reaction_network([{"smiles": MONO_EPOXY}], [{"smiles": EDA}], explicit_r=1.0)
    assert res_inapplicable.gel_regime == GEL_REGIME_NOT_APPLICABLE
    assert np.isnan(res_inapplicable.gel_possible_ideal)
    assert res_subcritical.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res_subcritical.gel_possible_ideal == 0.0


# =========================================================================
# 二、结构与官能度冲突 (场景 8 ~ 11)
# =========================================================================

def test_08_smiles_functionality_conflict_detected():
    """8. 已解析非环氧 SMILES 与外部环氧官能度冲突被准确检测 (P0-2)"""
    res = evaluate_reaction_network(
        [{"smiles": NON_EPOXY_ETHANOL, "mw": 200, "f_stoich": 2, "f_network": 2, "mass": 100}],
        [{"smiles": EDA, "mass": 8.83}],
    )
    assert res.structure_functionality_conflict is True
    assert res.chemistry_valid is False
    assert res.gel_regime == "structure_functionality_conflict"


def test_09_conflict_original_fields_preserved():
    """9. 冲突原始字段在增强表格中得到完整保留，不强制删除"""
    df = pd.DataFrame({
        "resin_1_structure": [NON_EPOXY_ETHANOL],
        "resin_1_epoxy_group_count": [2.0],
        "curing_agent_1_structure": [EDA],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert len(en) == 1
    assert en["structure_functionality_conflict"].iloc[0] == True
    assert en["resin_1_structure"].iloc[0] == NON_EPOXY_ETHANOL
    assert en["resin_1_epoxy_group_count"].iloc[0] == 2.0


def test_10_conflict_does_not_generate_fake_high_confidence_gel():
    """10. 来源不明的结构-官能度冲突不生成高置信度凝胶值 (gel_possible=NaN, nu=NaN)"""
    res = evaluate_reaction_network(
        [{"smiles": NON_EPOXY_ETHANOL, "mw": 200, "f_stoich": 2, "f_network": 2, "mass": 100}],
        [{"smiles": EDA, "mass": 8.83}],
    )
    assert np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.nu_ideal_mol_m3)


def test_11_conflict_components_retains_independent_descriptors():
    """11. 结构-官能度冲突样本不删除其独立的分子量与工艺参数"""
    df = pd.DataFrame({
        "resin_1_structure": [NON_EPOXY_ETHANOL],
        "resin_1_epoxy_group_count": [2.0],
        "curing_agent_1_structure": [EDA],
        "process_max_temperature_c": [120.0],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["process_max_temperature_c"].iloc[0] == 120.0
    assert en["resin_1_has_structure"].iloc[0] == True


# =========================================================================
# 三、密度 (场景 12 ~ 17)
# =========================================================================

def test_12_new_valid_density_overrides_historical_reference_cache():
    """12. 新有效密度必须覆盖历史参考计算缓存 (P0-3)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [1.0],
    })
    en1 = extract_epoxy_mechanism_features(df)
    assert en1["rho_used_for_mech_nu_g_cm3"].iloc[0] == 1.2
    initial_nu = en1["mech_crosslink_density_mol_m3"].iloc[0]

    # 新增真实密度 1.4
    en1["rho_g_cm3"] = 1.4
    en2 = extract_epoxy_mechanism_features(en1)
    assert en2["rho_used_for_mech_nu_g_cm3"].iloc[0] == 1.4
    # 体积交联密度按 1.4 / 1.2 比例更新
    assert pytest.approx(en2["mech_crosslink_density_mol_m3"].iloc[0], abs=1e-1) == initial_nu * (1.4 / 1.2)


def test_13_repeated_enrich_density_idempotent():
    """13. 重复增强的密度与网络特征保持幂等性"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [1.0],
        "rho_g_cm3": [1.3],
    })
    en1 = extract_epoxy_mechanism_features(df)
    en2 = extract_epoxy_mechanism_features(en1)
    assert en1["mech_crosslink_density_mol_m3"].iloc[0] == en2["mech_crosslink_density_mol_m3"].iloc[0]
    assert len(en1.columns) == len(en2.columns)


def test_14_disable_fallback_density_clears_old_cached_fallback():
    """14. 不允许回退时旧参考计算值不能绕过策略，清除为 NaN (P0-4)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "mech_crosslink_density_mol_m3": [3000.0],  # 旧的 1.2 回退值
        "rho_g_cm3": [-1.0],                         # 无效密度
        "rho_used_for_mech_nu_g_cm3": [1.2],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert clean["mech_crosslink_density_mol_m3"].isna().all()


def test_15_independent_valid_primary_density_not_dropped():
    """15. 具备真实有效输入密度的行正常换算并保留"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [1.4],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert pytest.approx(clean["mech_crosslink_density_mol_m3"].iloc[0]) == 3500.0


def test_16_invalid_density_does_not_generate_negative_or_inf():
    """16. 非法密度不生成负值、零值或无穷大交联密度"""
    for bad_rho in [-5.0, 0.0, np.inf, "corrupt"]:
        df = pd.DataFrame({
            "mech_crosslink_density_mmol_g": [2.0],
            "rho_g_cm3": [bad_rho],
        })
        clean, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
        assert clean["mech_crosslink_density_mol_m3"].isna().all()


def test_17_mass_and_volume_crosslink_densities_share_same_baseline():
    """17. 质量比与体积比交联密度采用相同生成基准，无伪别名冲突"""
    engine = EpoxyMechanismEngine(verbose=False)
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [1.0],
        "rho_g_cm3": [1.35],
    })
    en = engine.enrich_dataframe(df)
    clean, _ = prepare_ml_feature_matrix(en, conflict_policy="raise")
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] > 0


# =========================================================================
# 四、r 来源与组成 (场景 18 ~ 24)
# =========================================================================

def test_18_historical_unknown_r_updated_by_new_real_mass():
    """18. 历史 unknown r 遇到新完整质量更新为 calculated_from_mass (P1-2)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "cp_r_value": [0.7],
        "cp_r_source": ["unknown_provenance"],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_CALCULATED
    assert pytest.approx(en["formulation_r_value"].iloc[0], abs=1e-3) == 1.0


def test_19_historical_partial_composition_r_does_not_upgrade_to_literature():
    """19. 历史部分组成 r 重新解析不升级为 literature (P0-1)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [0.55],
        "cp_r_source": [R_SRC_PARTIAL_CALCULATED],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_PARTIAL_CALCULATED


def test_20_explicit_literature_r_priority_and_conflict_flag():
    """20. 显式文献 r 与质量推导冲突时优先保留文献 r 并标记冲突"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],  # 质量推导 r=1.0
        "formulation_r_value": [0.5],          # 文献实测 r=0.5
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["formulation_r_value"].iloc[0] == 0.5
    assert en["cp_r_source"].iloc[0] == R_SRC_LITERATURE
    assert en["cp_r_conflict"].iloc[0] == 1.0


def test_21_unknown_component_missing_mass_ignored_fraction_nan():
    """21. 未知组分质量缺失时忽略质量比例为 NaN (P1-1)"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA, "mass": 100.0}, {"smiles": "UNKNOWN_RESIN"}],
        [{"smiles": EDA, "mass": 8.83}],
    )
    assert res.unknown_component_count == 1
    assert res.composition_incomplete is True
    assert np.isnan(res.ignored_mass_fraction)


def test_22_unknown_component_known_mass_ignored_fraction_calculated():
    """22. 未知组分质量已知时忽略质量比例正确计算"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA, "mass": 100.0}, {"smiles": "UNKNOWN_RESIN", "mass": 50.0}],
        [{"smiles": EDA, "mass": 8.83}],
    )
    assert res.unknown_component_count == 1
    assert pytest.approx(res.ignored_mass_fraction, abs=1e-2) == 50.0 / 158.83


def test_23_default_mode_assumes_r1_for_compatible_epoxy():
    """23. 默认模式保留合法环氧体系 r=1.0 假定"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=True)
    assert en["formulation_r_value"].iloc[0] == 1.0
    assert en["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH


def test_24_strict_mode_unknown_r_remains_nan():
    """24. 严格模式未知 r 不隐藏回退，保持 NaN"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=False)
    assert pd.isna(en["formulation_r_value"].iloc[0])
    assert en["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH


# =========================================================================
# 五、ML 容错与去重 (场景 25 ~ 30)
# =========================================================================

def test_25_single_conflict_row_raises_in_raise_mode():
    """25. 两行数据中一行冲突，raise 模式正常抛错 (P1-4)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.0, 2.0],
        "mech_crosslink_density_mol_m3": [2400.0, 5000.0],
        "formulation_r_value": [1.0, 0.8],
    })
    with pytest.raises(ValueError, match="Unit conversion conflict"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_26_mask_conflicts_policy_masks_only_conflict_cell():
    """26. mask_conflicts 模式只屏蔽冲突行相应字段为 NaN，其他行正常输出 (P1-4)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.0, 2.0],
        "mech_crosslink_density_mol_m3": [2400.0, 5000.0],  # row 1 冲突
        "formulation_r_value": [1.0, 0.8],
    })
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] == 2400.0
    assert pd.isna(clean["mech_crosslink_density_mol_m3"].iloc[1])  # 冲突行变 NaN
    assert clean["formulation_r_value"].iloc[1] == 0.8              # 无关特征正常保留


def test_27_mask_conflicts_preserves_length_and_index():
    """27. mask_conflicts 模式保证数据总行数、索引不变 (P1-4)"""
    df = pd.DataFrame(
        {
            "mech_crosslink_density_mmol_g": [2.0, 2.0],
            "mech_crosslink_density_mol_m3": [2400.0, 5000.0],
            "formulation_r_value": [1.0, 0.8],
        },
        index=["sample_A", "sample_B"],
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert list(clean.index) == ["sample_A", "sample_B"]
    assert len(clean) == 2


def test_28_target_and_id_columns_never_leak():
    """28. 目标、ID 和分组字段不泄漏入特征矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "tg_c": [150.0],
        "sample_id": ["S100"],
        "doi": ["10.1016/j.composites"],
    })
    clean, _ = prepare_ml_feature_matrix(df, feature_set="full_safe")
    for bad in ["tg_c", "sample_id", "doi"]:
        assert bad not in clean.columns


def test_29_feature_sets_A_B_C_D_stable():
    """29. A/B/C/D 特征集合稳定生成，无分类变量交叉污染"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "resin_1_molecular_weight_g_mol": [340.0],
        "process_max_temperature_c": [120.0],
        "cp_r_source": ["literature"],
        "process_has_post_cure": [1.0],
    })
    cA, _ = prepare_ml_feature_matrix(df, feature_set="reaction_core", encode_categoricals=True)
    cB, _ = prepare_ml_feature_matrix(df, feature_set="molecular", encode_categoricals=True)
    cC, _ = prepare_ml_feature_matrix(df, feature_set="formulation_process", encode_categoricals=True)
    assert "cp_r_source_literature" in cA.columns
    assert "process_has_post_cure_1.0" not in cA.columns
    assert "cp_r_source_literature" not in cB.columns
    assert "process_has_post_cure_1.0" in cC.columns
    assert "cp_r_source_literature" not in cC.columns


def test_30_heterogeneous_noise_batch_does_not_crash():
    """30. 混合非法 SMILES、缺质量和缺密度批次不会整批崩溃"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA, "%BAD%", NON_EPOXY_ETHANOL],
        "curing_agent_1_structure": [EDA, EDA, EDA],
        "resin_1_amount_phr": [100.0, np.nan, 100.0],
        "curing_agent_1_amount_phr": [8.83, 8.83, 8.83],
        "rho_g_cm3": [1.2, -1.0, 1.4],
        "formulation_r_value": [1.0, np.nan, 1.0],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    clean, _ = prepare_ml_feature_matrix(en, conflict_policy="mask_conflicts", allow_fallback_density=True)
    assert len(clean) == 3
    assert clean["formulation_r_value"].iloc[0] == 1.0
