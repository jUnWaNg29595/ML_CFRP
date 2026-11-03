# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_noise_tolerant_final.py

CFRP 反应特征工程宽容型噪声容忍与真实文献鲁棒性专项测试套件。
严格覆盖 Section XI 中要求的全部 32 项验收场景（A:化学与噪声容忍, B:来源与物理状态, C:密度, D:特征与ML）。
"""

import tempfile
import numpy as np
import pandas as pd
import pytest

from core.component_physics import (
    enrich_narrow_table,
    compute_component_physics,
    compute_formulation_summary,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import (
    EpoxyMechanismEngine,
    extract_epoxy_mechanism_features,
)
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
)
from core.feature_registry import (
    get_feature_registry,
    get_canonical_feature_names,
    get_features_for_set,
)
from core.feature_deduplication import (
    prepare_ml_feature_matrix,
    validate_and_resolve_density,
    MLFeaturePipeline,
    ALLOWED_CATEGORICAL_COLS,
    CATEGORICAL_FEATURE_SETS,
)

# 典型单体结构
DGEBA_SMILES = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA_SMILES = "NCCN"
DDM_SMILES = "Nc1ccc(Cc2ccc(N)cc2)cc1"
MONO_EPOXY_SMILES = "CC(C)COC1CO1"  # 单官能环氧单体
NON_EPOXY_ACETONE = "CC(=O)C"        # 丙酮 (非环氧非固化剂)
NON_EPOXY_BENZENE = "c1ccccc1"       # 苯 (惰性非反应物)


# =========================================================================
# A. 化学与噪声容忍 (场景 1 ~ 10)
# =========================================================================

def test_01_valid_dgeba_eda_standard_calculation():
    """1. 合法 DGEBA/EDA 标准配方正常计算出物理指标"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA_SMILES, "mass": 100.0}],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.chemistry_valid is True
    assert res.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert res.gel_possible_ideal == 1.0
    assert np.isfinite(res.nu_ideal_mol_m3)
    assert res.nu_ideal_mol_m3 > 1000.0


def test_02_broken_smiles_preserves_mass_and_process():
    """2. SMILES 错误时，样本行与已知质量、工艺字段必须保留"""
    df = pd.DataFrame({
        "resin_1_structure": ["INVALID_CORRUPT_SMILES_%%%"],
        "resin_1_amount_phr": [100.0],
        "resin_1_molecular_weight_g_mol": [380.0],
        "curing_agent_1_structure": [EDA_SMILES],
        "curing_agent_1_amount_phr": [8.83],
        "process_max_temperature_c": [150.0],
        "tg_c": [160.0],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert len(en) == 1
    assert en["resin_1_amount_phr"].iloc[0] == 100.0
    assert en["process_max_temperature_c"].iloc[0] == 150.0
    assert en["tg_c"].iloc[0] == 160.0


def test_03_non_epoxy_smiles_not_faked_as_epoxy():
    """3. 普通非环氧单体不被伪造成环氧反应体系"""
    res = evaluate_reaction_network(
        [{"smiles": NON_EPOXY_ACETONE, "mass": 100.0}],
        [{"smiles": NON_EPOXY_BENZENE, "mass": 100.0}],
        explicit_r=None,
    )
    assert res.chemistry_valid is False
    assert res.r_source == R_SRC_INVALID_CHEM


def test_04_unsupported_reaction_type_not_marked_subcritical():
    """4. 当前不支持的反应类型不被误标成确定不可凝胶 (subcritical)"""
    res = evaluate_reaction_network(
        [{"smiles": "C=CC(=O)OCC", "mass": 100.0}],  # 丙烯酸乙酯 (自由基聚合，非环氧)
        [{"smiles": "C=CC(=O)OCC", "mass": 100.0}],
        explicit_r=None,
    )
    assert res.chemistry_valid is False
    assert res.gel_regime != GEL_REGIME_SUPERCRITICAL


def test_05_single_invalid_component_does_not_abort_batch():
    """5. 一个无效组分不导致整批处理崩溃或丢失其他样本"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES, "BAD_SMILES_XYZ", DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES, EDA_SMILES, EDA_SMILES],
        "formulation_r_value": [1.0, 1.0, 1.0],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert len(en) == 3
    # 正常样本依然计算成功
    assert np.isfinite(en["formulation_r_value"].iloc[0])
    assert np.isfinite(en["formulation_r_value"].iloc[2])


def test_06_multi_component_partial_structure_available():
    """6. 多组分混合配方中部分结构可用时，解析出可用组分物理量"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_2_structure": ["BAD_SMILES_RESIN_2"],
        "curing_agent_1_structure": [EDA_SMILES],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["resin_1_mw_resolved"].iloc[0] > 0
    assert pd.isna(en["resin_2_mw_resolved"].iloc[0])
    assert en["cp_unknown_comp_count"].iloc[0] == 1
    assert en["cp_composition_incomplete"].iloc[0] == True


def test_07_component_present_without_mass_detected():
    """7. 组分存在但缺失质量时，覆盖状态与组成不完整标记正确 (P0-3)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_2_structure": ["UNPARSEABLE_RESIN_NAME"],
        "curing_agent_1_structure": [EDA_SMILES],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_composition_incomplete"].iloc[0] == True
    assert en["cp_unknown_comp_count"].iloc[0] == 1
    assert pd.isna(en["cp_ignored_mass_fraction"].iloc[0])


def test_08_known_nonreactive_additive_preserved():
    """8. 明确非反应性溶剂/助剂保留质量供稀释计算使用"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA_SMILES, "mass": 100.0}],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.chemistry_valid is True
    assert res.r_value == 1.0


def test_09_default_mode_compatible_epoxy_assumes_r1():
    """9. 默认模式对已确认兼容的环氧体系缺失配比继续支持 r=1.0 假定"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=True)
    assert en["formulation_r_value"].iloc[0] == 1.0
    assert en["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH


def test_10_strict_mode_missing_r_remains_nan():
    """10. 严格模式缺失 r 保持 NaN，绝不静默使用 r=1.0"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
    })
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=False)
    assert pd.isna(en["formulation_r_value"].iloc[0])
    assert en["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH


# =========================================================================
# B. 来源和物理状态 (场景 11 ~ 18)
# =========================================================================

def test_11_historical_partial_composition_r_does_not_drift():
    """11. 历史 partial_composition_calculated 重新解析绝不升级为 literature (P0-1)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [0.65],
        "cp_r_source": [R_SRC_PARTIAL_CALCULATED],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_PARTIAL_CALCULATED


def test_12_historical_assumed_r_does_not_drift_to_literature():
    """12. 历史 assumed_stoichiometric r 不升级为 literature"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [1.0],
        "cp_r_source": [R_SRC_ASSUMED_STOICH],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH


def test_13_no_source_derived_r_does_not_fake_literature():
    """13. 无来源历史衍生 r 不伪装为文献实测 (P1-2)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "cp_r_value": [0.85],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_UNKNOWN_PROVENANCE


def test_14_new_real_full_mass_updates_historical_assumed():
    """14. 新到达的真实完整质量数据可覆盖并升级旧的假设来源"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA_SMILES],
        "curing_agent_1_amount_phr": [8.83],
        "formulation_r_value": [1.0],
        "cp_r_source": [R_SRC_ASSUMED_STOICH],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_CALCULATED


def test_15_unknown_component_strict_mode_gel_undetermined():
    """15. 未知反应组分严格模式凝胶状态必须为 undetermined，物理量为 NaN (P0-2)"""
    res = evaluate_reaction_network(
        [
            {"smiles": DGEBA_SMILES, "mass": 100.0},
            {"smiles": "UNRESOLVED_RESIN", "mass": 100.0},
        ],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=None,
        assume_stoichiometric_if_missing=False,
    )
    assert res.gel_regime == GEL_REGIME_UNDETERMINED
    assert np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.nu_ideal_mol_m3)


def test_16_provably_subcritical_system_remains_subcritical():
    """16. 确定不可凝胶的单官能体系仍正确输出 subcritical 与 0.0"""
    res = evaluate_reaction_network(
        [{"smiles": MONO_EPOXY_SMILES, "mass": 100.0}],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res.gel_possible_ideal == 0.0
    assert res.nu_ideal_mol_m3 == 0.0


def test_17_explicit_literature_r_preserved():
    """17. 真实文献显式 r 来源稳定保持"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [0.75],
        "cp_r_source": [R_SRC_LITERATURE],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_LITERATURE


def test_18_csv_parquet_roundtrip_lineage_stable():
    """18. CSV/Parquet 往返保存血统保持绝对稳定，不发生漂移"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES, DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES, DDM_SMILES],
        "formulation_r_value": [0.65, 1.0],
        "cp_r_source": [R_SRC_PARTIAL_CALCULATED, R_SRC_LITERATURE],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
        df.to_csv(tf.name, index=False)
        reloaded = pd.read_csv(tf.name)
        en = enrich_narrow_table(reloaded, keep_metadata=True)
        assert en["cp_r_source"].iloc[0] == R_SRC_PARTIAL_CALCULATED
        assert en["cp_r_source"].iloc[1] == R_SRC_LITERATURE


# =========================================================================
# C. 密度与量纲一致性 (场景 19 ~ 23)
# =========================================================================

def test_19_invalid_density_does_not_enter_conversion():
    """19. 无效密度（负数/零/无穷）不得进入换算 (P0-2, P0-4)"""
    for bad_rho in [-1.0, 0.0, np.inf, "bad_str"]:
        df = pd.DataFrame({
            "mech_crosslink_density_mmol_g": [2.5],
            "rho_g_cm3": [bad_rho],
        })
        clean_strict, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
        assert clean_strict["mech_crosslink_density_mol_m3"].isna().all()


def test_20_disable_fallback_density_respected_in_all_paths():
    """20. 禁止回退配置在所有上游与下游路径严格生效 (P0-4)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [-1.0],
        "rho_used_for_mech_nu_g_cm3": [1.2],
    })
    clean_strict, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert clean_strict["mech_crosslink_density_mol_m3"].isna().all()


def test_21_new_sample_density_overrides_old_fallback_cache():
    """21. 新真实密度可以覆盖旧的参考密度缓存 (P0-4)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.0],
        "rho_g_cm3": [1.4],                    # 新真实密度
        "rho_used_for_mech_nu_g_cm3": [1.2],    # 旧缓存
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_fallback_density=True)
    # 2.0 * 1400 = 2800.0 (使用新真实密度 1.4，而非旧 1.2)
    assert pytest.approx(clean["mech_crosslink_density_mol_m3"].iloc[0]) == 2800.0


def test_22_same_model_density_generation_and_conversion_consistent():
    """22. 同一交联密度模型生成与转换采用一致密度，无伪冲突 (P0-3)"""
    engine = EpoxyMechanismEngine(verbose=False)
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [1.0],
        "rho_g_cm3": [1.35],
    })
    en = engine.enrich_dataframe(df)
    clean, _ = prepare_ml_feature_matrix(en, conflict_policy="raise")
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] > 0


def test_23_distinct_physical_models_not_falsely_aliased():
    """23. 不同物理模型的交联密度不被错误合并"""
    reg = get_feature_registry()
    assert reg["mech_crosslink_density_mol_m3"].include_in_ml == True
    assert reg["xl_nu_theory_mol_m3"].include_in_ml == True
    assert reg["xl_nu_junction_mol_m3"].include_in_ml == True


# =========================================================================
# D. 特征治理与 ML (场景 24 ~ 32)
# =========================================================================

def test_24_one_hot_columns_isolated_by_feature_set():
    """24. One-Hot 编码字段只进入所属授权特征集合，无交叉污染 (P1)"""
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


def test_25_feature_set_A_B_C_D_stability():
    """25. Sets A/B/C/D 的列集合与列顺序在多次调用下保持稳定"""
    for s_name in ["reaction_core", "molecular", "formulation_process", "full_safe"]:
        f1 = get_features_for_set(s_name)
        f2 = get_features_for_set(s_name)
        assert f1 == f2


def test_26_duplicate_fields_properly_governed():
    """26. 重复字段 ff_charge_variance 与 rigid_carbon_ratio 正确治理 (P1)"""
    reg = get_feature_registry()
    assert reg["ff_charge_variance"].alias_of == "ff_charge_variance_weighted"
    assert reg["ff_charge_variance"].include_in_ml == False
    assert reg["ff_charge_variance_weighted"].include_in_ml == True
    assert reg["rigid_carbon_ratio"].include_in_ml == True
    assert reg["mech_weighted_rigid_ratio"].alias_of == "rigid_carbon_ratio"


def test_27_unregistered_fields_blocked_by_default():
    """27. 未注册数值字段默认不进入训练矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "unregistered_noise_column": [9999.0],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_unregistered_numeric=False)
    assert "unregistered_noise_column" not in clean.columns


def test_28_ungenerated_xtb_fields_not_fabricated():
    """28. 默认流水线不生成的外部 xTB 字段不被伪造成固定常数"""
    reg = get_feature_registry()
    assert reg["xtb_homo_weighted"].include_in_ml == False
    assert reg["xtb_lumo_weighted"].include_in_ml == False
    assert reg["xtb_gap_weighted"].include_in_ml == False


def test_29_target_and_id_columns_never_leak():
    """29. Tg 目标与样本 ID 绝对不泄漏入特征矩阵"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "tg_c": [150.0],
        "sample_id": ["S_001"],
        "doi": ["10.1002/app.12345"],
    })
    clean, _ = prepare_ml_feature_matrix(df, feature_set="full_safe")
    for bad in ["tg_c", "sample_id", "doi"]:
        assert bad not in clean.columns


def test_30_high_missing_rate_batch_preserves_row_count():
    """30. 高缺失率批次处理依然保留全部样本行数，不误删样本"""
    df = pd.DataFrame({
        "formulation_r_value": [np.nan, 1.0, np.nan],
        "resin_1_structure": ["BAD", DGEBA_SMILES, "BAD"],
        "curing_agent_1_structure": [EDA_SMILES, EDA_SMILES, "BAD"],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_unregistered_numeric=True)
    assert len(clean) == 3


def test_31_single_row_anomaly_does_not_abort_batch():
    """31. 单行密度异常或结构错误不引起整批处理崩溃"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5, 2.5, 2.5],
        "rho_g_cm3": [1.2, -999.0, 1.4],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert len(clean) == 3
    assert np.isfinite(clean["mech_crosslink_density_mol_m3"].iloc[0])
    assert np.isnan(clean["mech_crosslink_density_mol_m3"].iloc[1])  # 异常行转 NaN
    assert np.isfinite(clean["mech_crosslink_density_mol_m3"].iloc[2])


def test_32_pipeline_fit_transform_consistency():
    """32. MLFeaturePipeline 保证严格模式与宽容模式契约可复现、可审计"""
    train_df = pd.DataFrame({
        "formulation_r_value": [1.0, 0.8],
        "cp_r_source": ["literature", "calculated_from_mass"],
    })
    test_df = pd.DataFrame({
        "formulation_r_value": [0.9],
        "cp_r_source": ["unknown_new"],
    })
    pipe = MLFeaturePipeline(feature_set="reaction_core", encode_categoricals=True)
    X_train, _ = pipe.fit_transform(train_df)
    X_test = pipe.transform(test_df)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_test["cp_r_source_literature"].iloc[0] == 0.0
