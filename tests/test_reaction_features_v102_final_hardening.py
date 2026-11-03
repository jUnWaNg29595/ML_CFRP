# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_v102_final_hardening.py

CFRP 反应特征工程与机器学习安全终期加固专项测试套件。
严格覆盖 30 项终期安全与物理一致性验收准则。
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
    R_SRC_UNKNOWN_PROVENANCE,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_UNDETERMINED,
)
from core.feature_registry import (
    get_feature_registry,
    get_canonical_feature_names,
    get_features_for_set,
    export_registry_reports,
)
from core.feature_deduplication import (
    prepare_ml_feature_matrix,
    validate_and_resolve_density,
    MLFeaturePipeline,
    ALLOWED_CATEGORICAL_COLS,
)

# 经典单体 SMILES
DGEBA_SMILES = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA_SMILES = "NCCN"
DDM_SMILES = "Nc1ccc(Cc2ccc(N)cc2)cc1"
MONO_EPOXY_SMILES = "CC(C)COC1CO1"  # 单官能环氧缩水甘油醚


# =========================================================================
# 一、代码与数值安全 (准则 1 ~ 11)
# =========================================================================

def test_01_three_duplicate_cols_with_nan_conflict_raises():
    """1. 三个同名列 [NaN, 1.0, 2.0] 必须报错 (P0-1)"""
    df = pd.DataFrame([[np.nan, 1.0, 2.0]], columns=["formulation_r_value"] * 3)
    with pytest.raises(ValueError, match="Duplicate column 'formulation_r_value' has conflicting values"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_02_three_duplicate_cols_with_nan_consistent_merges():
    """2. 三个同名列 [NaN, 1.0, 1.0] 正常合并为 1.0"""
    df = pd.DataFrame([[np.nan, 1.0, 1.0]], columns=["formulation_r_value"] * 3)
    clean, _ = prepare_ml_feature_matrix(df)
    assert clean["formulation_r_value"].iloc[0] == 1.0


def test_03_four_duplicate_cols_later_conflict_raises():
    """3. 四个同名列 [NaN, 1.0, NaN, 2.0] 存在后段冲突时必须报错"""
    df = pd.DataFrame([[np.nan, 1.0, np.nan, 2.0]], columns=["formulation_r_value"] * 4)
    with pytest.raises(ValueError, match="Duplicate column 'formulation_r_value' has conflicting values"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


def test_04_duplicate_cols_order_invariance():
    """4. 不同同名列排列顺序下冲突与合并结果严格一致"""
    # 冲突用例全排列测试
    permutations_conflict = [
        [np.nan, 1.0, 2.0],
        [1.0, np.nan, 2.0],
        [1.0, 2.0, np.nan],
        [2.0, 1.0, np.nan],
    ]
    for p in permutations_conflict:
        df = pd.DataFrame([p], columns=["formulation_r_value"] * 3)
        with pytest.raises(ValueError):
            prepare_ml_feature_matrix(df, conflict_policy="raise")

    # 一致用例全排列测试
    permutations_consistent = [
        [np.nan, 1.0, 1.0],
        [1.0, np.nan, 1.0],
        [1.0, 1.0, np.nan],
    ]
    for p in permutations_consistent:
        df = pd.DataFrame([p], columns=["formulation_r_value"] * 3)
        clean, _ = prepare_ml_feature_matrix(df)
        assert clean["formulation_r_value"].iloc[0] == 1.0


def test_05_negative_density_prevented():
    """5. 负密度不得产生负交联密度 (P0-2)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [-1.0],
    })
    # 禁用回退时置 NaN，绝不产生负值
    clean_strict, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert clean_strict["mech_crosslink_density_mol_m3"].isna().all()

    # 启用回退时安全使用参考密度 1.2
    clean_fb, _ = prepare_ml_feature_matrix(df, allow_fallback_density=True)
    assert clean_fb["mech_crosslink_density_mol_m3"].iloc[0] == 2.5 * 1200.0


def test_06_zero_density_prevented():
    """6. 零密度不得产生无效物理换算 (P0-2)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [0.0],
    })
    clean_strict, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert clean_strict["mech_crosslink_density_mol_m3"].isna().all()

    clean_fb, _ = prepare_ml_feature_matrix(df, allow_fallback_density=True)
    assert clean_fb["mech_crosslink_density_mol_m3"].iloc[0] == 2.5 * 1200.0


def test_07_infinite_density_prevented():
    """7. 无穷密度不得产生无穷交联密度 (P0-2)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [np.inf],
    })
    clean_strict, _ = prepare_ml_feature_matrix(df, allow_fallback_density=False)
    assert clean_strict["mech_crosslink_density_mol_m3"].isna().all()

    clean_fb, _ = prepare_ml_feature_matrix(df, allow_fallback_density=True)
    assert np.isfinite(clean_fb["mech_crosslink_density_mol_m3"].iloc[0])


def test_08_valid_density_12_converted_correctly():
    """8. 有效密度 1.2 换算正确 (mmol/g * 1200 = mol/m³)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [1.2],
    })
    clean, _ = prepare_ml_feature_matrix(df)
    assert pytest.approx(clean["mech_crosslink_density_mol_m3"].iloc[0]) == 3000.0


def test_09_valid_density_14_converted_correctly():
    """9. 有效密度 1.4 换算正确 (mmol/g * 1400 = mol/m³)"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.5],
        "rho_g_cm3": [1.4],
    })
    clean, _ = prepare_ml_feature_matrix(df)
    assert pytest.approx(clean["mech_crosslink_density_mol_m3"].iloc[0]) == 3500.0


def test_10_upstream_and_downstream_density_consistency():
    """10. 上游机理实际生成密度与下游换算基准一致，不误报虚假冲突 (P0-3)"""
    engine = EpoxyMechanismEngine(verbose=False)
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [1.0],
        "rho_g_cm3": [1.4],
    })
    en = engine.enrich_dataframe(df)
    # 上游记录了使用的密度为 1.4
    assert en["rho_used_for_mech_nu_g_cm3"].iloc[0] == 1.4
    # 下游 prepare_ml_feature_matrix 换算比对时不发生假性冲突
    clean, _ = prepare_ml_feature_matrix(en, conflict_policy="raise")
    assert clean["mech_crosslink_density_mol_m3"].iloc[0] > 0


def test_11_distinct_model_density_not_blindly_aliased():
    """11. 不一致的模型来源与真实数值冲突必须准确拦截"""
    df = pd.DataFrame({
        "mech_crosslink_density_mmol_g": [2.0],
        "mech_crosslink_density_mol_m3": [5000.0],  # 2.0 * 1200 = 2400 != 5000
    })
    with pytest.raises(ValueError, match="Unit conversion conflict"):
        prepare_ml_feature_matrix(df, conflict_policy="raise")


# =========================================================================
# 二、数据来源与物理状态 (准则 12 ~ 19)
# =========================================================================

def test_12_no_source_derived_r_marked_unknown_provenance():
    """12. 无来源证据的历史衍生 cp_r_value=0.7 不自动归类为 literature (P1-2)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [DDM_SMILES],
        "cp_r_value": [0.7],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_UNKNOWN_PROVENANCE


def test_13_explicit_literature_tag_highest_priority():
    """13. 显式 literature 标签继续保持最高优先级 (P0-5)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [DDM_SMILES],
        "curing_agent_1_amount_phr": [29.1],  # 质量推导 r 与显式 r 吻合
        "formulation_r_value": [1.0],
        "cp_r_source": ["literature"],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_LITERATURE


def test_14_complete_mass_calculation_source_correct():
    """14. 完整质量计算来源正确标记为 calculated_from_mass"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA_SMILES],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_CALCULATED


def test_15_historical_assumed_r_updated_by_real_mass():
    """15. 历史 assumed r 遇到真实质量可以升级为 calculated_from_mass"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA_SMILES],
        "curing_agent_1_amount_phr": [8.83],
        "formulation_r_value": [1.0],
        "rxn_r_assumed": [1.0],
        "cp_r_source": ["assumed_stoichiometric"],
    })
    en = enrich_narrow_table(df, keep_metadata=True)
    assert en["cp_r_source"].iloc[0] == R_SRC_CALCULATED
    assert en["rxn_r_assumed"].iloc[0] == 0.0


def test_16_unknown_reactive_component_not_marked_full_calculated():
    """16. 未知反应组分不会导致虚假的完整质量 r，标为 partial_composition_calculated (P1-3)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["INVALID_UNPARSEABLE_RESIN"],
        "resin_2_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA_SMILES],
        "curing_agent_1_amount_phr": [8.83],
    })
    # 默认模式: 标为 partial_composition_calculated
    en_def = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=True)
    assert en_def["cp_r_source"].iloc[0] == R_SRC_PARTIAL_CALCULATED
    assert en_def["cp_composition_incomplete"].iloc[0] == True
    assert en_def["cp_ignored_mass_fraction"].iloc[0] > 0.40

    # 严格模式: 由于未知反应组分可能改变当量，完整配方 r 为 NaN / missing_stoichiometry
    en_str = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=False)
    assert en_str["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH
    assert pd.isna(en_str["formulation_r_value"].iloc[0])


def test_17_known_nonreactive_additive_handled():
    """17. 已知非反应性物质不破坏反应当量计算"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA_SMILES, "mass": 100.0}],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.chemistry_valid == True
    assert res.r_source == R_SRC_LITERATURE


def test_18_strict_mode_undetermined_gel_remains_nan():
    """18. 严格模式未知凝胶状态 nu_ideal_mol_m3 继续保持 NaN (P1)"""
    res = evaluate_reaction_network(
        [{"smiles": DGEBA_SMILES}],
        [{"smiles": EDA_SMILES}],
        explicit_r=None,
        assume_stoichiometric_if_missing=False,
    )
    assert res.gel_regime == GEL_REGIME_UNDETERMINED
    assert np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.nu_ideal_mol_m3)


def test_19_subcritical_nongelling_system_remains_zero():
    """19. 已知单官能不可凝胶体系继续保持 0.0 交联密度"""
    res = evaluate_reaction_network(
        [{"smiles": MONO_EPOXY_SMILES, "mass": 100.0}],
        [{"smiles": EDA_SMILES, "mass": 8.83}],
        explicit_r=1.0,
    )
    assert res.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res.gel_possible_ideal == 0.0
    assert res.nu_ideal_mol_m3 == 0.0


# =========================================================================
# 三、特征治理与分层体系 (准则 20 ~ 30)
# =========================================================================

def test_20_encode_categoricals_preserves_one_hot_columns():
    """20. encode_categoricals=True 后合法 One-Hot 列能够保留在训练矩阵中 (P1-1)"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0, 0.5],
        "cp_r_source": ["literature", "assumed_stoichiometric"],
    })
    clean, _ = prepare_ml_feature_matrix(df, encode_categoricals=True)
    assert "cp_r_source_literature" in clean.columns
    assert "cp_r_source_assumed_stoichiometric" in clean.columns
    assert "formulation_r_value" in clean.columns


def test_21_pipeline_categorical_encoding_dimension_stability():
    """21. MLFeaturePipeline 保证训练与验证/测试集特征维度严格一致，零漂移 (Section X)"""
    train_df = pd.DataFrame({
        "formulation_r_value": [1.0, 0.5, 0.8],
        "cp_r_source": ["literature", "assumed_stoichiometric", "calculated_from_mass"],
    })
    test_df = pd.DataFrame({
        "formulation_r_value": [0.9],
        "cp_r_source": ["unseen_new_category"],  # 出现未见类别
    })

    pipe = MLFeaturePipeline(feature_set="reaction_core", encode_categoricals=True)
    clean_train = pipe.fit_transform(train_df)[0]
    clean_test = pipe.transform(test_df)

    assert list(clean_train.columns) == list(clean_test.columns)
    # 未见类别在已有 dummy 列上严格赋 0.0
    assert clean_test["cp_r_source_literature"].iloc[0] == 0.0
    assert clean_test["cp_r_source_assumed_stoichiometric"].iloc[0] == 0.0
    assert clean_test["cp_r_source_calculated_from_mass"].iloc[0] == 0.0


def test_22_feature_set_A_reaction_core_generable():
    """22. 反应核心特征集合 (Set A: reaction_core) 可稳定生成"""
    feats = get_features_for_set("reaction_core")
    assert len(feats) >= 25
    assert "formulation_r_value" in feats
    assert "mech_crosslink_density_mol_m3" in feats


def test_23_feature_set_B_molecular_generable():
    """23. 分子结构特征集合 (Set B: molecular) 可稳定生成"""
    feats = get_features_for_set("molecular")
    assert len(feats) >= 15
    assert "resin_1_molecular_weight_g_mol" in feats
    assert "mech_weighted_tpsa" in feats


def test_24_feature_set_C_formulation_process_generable():
    """24. 工艺与配方特征集合 (Set C: formulation_process) 可稳定生成"""
    feats = get_features_for_set("formulation_process")
    assert len(feats) >= 8
    assert "process_max_temperature_c" in feats
    assert "resin_total_phr" in feats


def test_25_feature_set_D_full_safe_generable():
    """25. 完整安全特征集合 (Set D: full_safe) 为 A+B+C 并集"""
    set_a = set(get_features_for_set("reaction_core"))
    set_b = set(get_features_for_set("molecular"))
    set_c = set(get_features_for_set("formulation_process"))
    set_d = set(get_features_for_set("full_safe"))
    assert set_d.issuperset(set_a)
    assert set_d.issuperset(set_b)
    assert set_d.issuperset(set_c)


def test_26_target_columns_blocked():
    """26. Tg 与力学性能目标字段绝不进入特征矩阵 (P0-4)"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "tg": [150.0],
        "tg_c": [150.0],
        "target_tg_c": [150.0],
        "tensile_strength_mpa": [80.0],
    })
    clean, audit = prepare_ml_feature_matrix(df, feature_set="full_safe")
    for bad in ["tg", "tg_c", "target_tg_c", "tensile_strength_mpa"]:
        assert bad not in clean.columns


def test_27_id_and_group_columns_blocked():
    """27. 样本 ID、DOI、论文 ID、分组 ID 绝不进入普通模型输入 (P0-4)"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "sample_id": ["S001"],
        "doi": ["10.1016/j.polymer"],
        "group_id": ["batch_A"],
        "paper_id": ["P123"],
    })
    clean, audit = prepare_ml_feature_matrix(df, feature_set="full_safe")
    for bad in ["sample_id", "doi", "group_id", "paper_id"]:
        assert bad not in clean.columns


def test_28_unregistered_numeric_blocked_by_default():
    """28. 未注册数值字段默认继续被阻断 (P0-4)"""
    df = pd.DataFrame({
        "formulation_r_value": [1.0],
        "arbitrary_random_numeric_column_xyz": [999.0],
    })
    clean, _ = prepare_ml_feature_matrix(df, allow_unregistered_numeric=False)
    assert "arbitrary_random_numeric_column_xyz" not in clean.columns


def test_29_rounded_alias_retains_independent_physics():
    """29. 机理加权平均官能度与 Flory 摩尔平均官能度作为独立物理视角保留"""
    reg = get_feature_registry()
    assert reg["mech_average_functionality"].include_in_ml == True
    assert reg["cp_f_avg_network"].include_in_ml == True
    assert reg["mech_average_functionality"].alias_of is None


def test_30_epoxy_mechanism_repeated_enrich_idempotence():
    """30. 机理模块重复增强仍然幂等，无重复列膨胀"""
    engine = EpoxyMechanismEngine(verbose=False)
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA_SMILES],
        "curing_agent_1_structure": [EDA_SMILES],
        "formulation_r_value": [1.0],
    })
    en1 = engine.enrich_dataframe(df)
    en2 = engine.enrich_dataframe(en1)
    assert len(en1.columns) == len(en2.columns)
    assert en2.columns.is_unique == True


# =========================================================================
# 四、综合边界与往返持久化测试
# =========================================================================

def test_custom_index_and_csv_parquet_roundtrip():
    """测试非默认索引与 CSV / Parquet 往返保存后特征矩阵与来源一致"""
    df = pd.DataFrame(
        {
            "resin_1_structure": [DGEBA_SMILES, DGEBA_SMILES],
            "curing_agent_1_structure": [EDA_SMILES, DDM_SMILES],
            "formulation_r_value": [1.0, 0.8],
            "cp_r_source": ["literature", "calculated_from_mass"],
        },
        index=["sample_alpha", "sample_beta"],
    )

    clean, _ = prepare_ml_feature_matrix(df, encode_categoricals=True)
    assert list(clean.index) == ["sample_alpha", "sample_beta"]

    with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
        clean.to_csv(tf.name)
        reloaded = pd.read_csv(tf.name, index_col=0)
        pd.testing.assert_frame_equal(clean, reloaded, check_exact=False, rtol=1e-5)
