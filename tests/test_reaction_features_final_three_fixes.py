# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_final_three_fixes.py

CFRP 反应特征工程终期遗留缺陷定向修复专项测试套件。
严格覆盖以下三类问题（P0-1, P0-2, P1）的 20 项以上真实接口验收场景：
1. reaction_core 缺少 f_stoich 时的全部异常分支防御 (Issue A, Issue B, EW 反推, 酸酐双口径等)
2. crosslink_physics 严格模式下三种调用路径 (直接严格、传cp严格、先默认增强再严格) 的 r 及全部派生量一致性
3. mask_conflicts 模式在重复索引、非连续索引、多别名下的精准位置屏蔽 (避免过度屏蔽)
"""

import math
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
    GEL_REGIME_NOT_APPLICABLE,
)
from core.component_physics import (
    enrich_narrow_table,
    compute_component_physics,
    compute_formulation_summary,
)
from core.crosslink_physics import compute_crosslink_features
from core.feature_registry import (
    get_feature_registry,
    get_features_for_set,
)
from core.feature_deduplication import (
    prepare_ml_feature_matrix,
    MLFeaturePipeline,
)

# 典型单体结构
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA = "NCCN"
PGE = "c1ccccc1OCC1CO1"  # 单官能环氧
SA = "O=C1CCC(=O)O1"     # 琥珀酸酐


# =========================================================================
# 一、P0-1: reaction_core 缺少 f_stoich 关键分支防御 (场景 1 ~ 7)
# =========================================================================

def test_01_missing_f_stoich_with_valid_ew_curer_issue_a():
    """1. (Issue A) 固化剂具有有效 AHEW 但缺少 f_stoich，正常计算，不抛 TypeError"""
    res = evaluate_reaction_network(
        resins=[{"smiles": DGEBA, "mass": 100.0}],
        curers=[{"mw": 60.0, "f_network": 4.0, "ew": 15.0, "mass": 8.83}]
    )
    assert res.chemistry_valid is True
    assert res.f_h_stoich == 4.0  # 由 mw / ew = 60 / 15 推导出 4.0
    assert np.isfinite(res.r_value)
    assert res.gel_regime == GEL_REGIME_SUPERCRITICAL


def test_02_missing_mass_virtual_reconstruction_issue_b():
    """2. (Issue B) 树脂缺少质量且缺少 f_stoich 但具有有效 EEW，虚拟重构不抛 TypeError"""
    res = evaluate_reaction_network(
        resins=[{"mw": 200.0, "f_network": 2.0, "ew": 100.0}],
        curers=[{"smiles": EDA, "mass": 8.83}],
        assume_stoichiometric_if_missing=True,
    )
    assert res.chemistry_valid is True
    assert res.f_r == 2.0
    assert res.r_value == 1.0


def test_03_missing_both_ew_and_f_stoich_safe_degradation():
    """3. 缺少 f_stoich 且无有效 EW 时平稳降级，严格模式理论量为 NaN，绝不崩溃"""
    res = evaluate_reaction_network(
        resins=[{"mw": 200.0, "f_network": 2.0}],  # 无 ew 也无 f_stoich
        curers=[{"smiles": EDA, "mass": 8.83}],
        assume_stoichiometric_if_missing=False,
    )
    assert pd.isna(res.r_value)
    assert res.r_source == R_SRC_MISSING_STOICH
    assert np.isnan(res.nu_ideal_mol_m3)


def test_04_anhydride_f_stoich_and_f_network_dual_standard_preserved():
    """4. 酸酐体系中 f_stoich=1.0 与 f_network=2.0 双口径严格区分，不被错误互换"""
    res = evaluate_reaction_network(
        resins=[{"smiles": DGEBA}],
        curers=[{"smiles": SA}],
        explicit_r=1.0,
    )
    assert res.f_h_stoich == 1.0
    assert res.f_h_network == 2.0
    assert res.chemistry_valid is True


def test_05_curer_missing_f_stoich_strict_mode():
    """5. 固化剂仅缺少 f_stoich 且无 EW 时，严格模式下当量未知并安全置 NaN"""
    res = evaluate_reaction_network(
        resins=[{"smiles": DGEBA, "mass": 100.0}],
        curers=[{"mw": 120.0, "f_network": 4.0, "mass": 10.0}],  # 无 f_stoich 无 ew
        assume_stoichiometric_if_missing=False,
    )
    assert pd.isna(res.r_value)
    assert np.isnan(res.f_h_stoich)
    assert res.f_h_network == 4.0


def test_06_both_sides_missing_f_stoich_with_valid_ews():
    """6. 树脂与固化剂双端均缺少 f_stoich 但均有有效 EW 时，当量与 r 计算完全正常"""
    res = evaluate_reaction_network(
        resins=[{"mw": 380.0, "ew": 190.0, "f_network": 2.0, "mass": 100.0}],
        curers=[{"mw": 60.0, "ew": 15.0, "f_network": 4.0, "mass": 10.0}],
        assume_stoichiometric_if_missing=False,
    )
    assert res.chemistry_valid is True
    assert res.r_source == R_SRC_CALCULATED
    expected_r = (10.0 / 15.0) / (100.0 / 190.0)
    assert pytest.approx(res.r_value, rel=1e-3) == expected_r


def test_07_invalid_f_stoich_strings_and_zeros_safely_handled():
    """7. 非法 f_stoich (0, 'invalid', np.nan) 安全过滤，不抛出异常"""
    for bad_f in [0, 0.0, -1.0, "invalid_f", np.nan, None]:
        res = evaluate_reaction_network(
            resins=[{"mw": 200.0, "f_network": 2.0, "f_stoich": bad_f, "ew": 100.0, "mass": 100.0}],
            curers=[{"smiles": EDA, "mass": 8.83}],
            assume_stoichiometric_if_missing=True,
        )
        assert np.isfinite(res.r_value)


# =========================================================================
# 二、P0-2: crosslink_physics 严格模式三路径一致性 (场景 8 ~ 14)
# =========================================================================

@pytest.fixture
def sample_incomplete_df():
    return pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNRESOLVED_RESIN"],  # 未解析树脂，质量未知
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })


def test_08_crosslink_strict_mode_three_paths_semantic_consistency(sample_incomplete_df):
    """8. (P0-2 核心) 严格模式下直接调用、传 cp、先默认增强的三种调用路径 r 及全部派生量 100% 一致"""
    # 路径 1: 直接严格计算
    xl1 = compute_crosslink_features(sample_incomplete_df, assume_stoichiometric_if_missing=False)

    # 路径 2: 传入默认 cp 后调用严格模式
    cp = compute_component_physics(sample_incomplete_df)
    xl2 = compute_crosslink_features(sample_incomplete_df, cp=cp, assume_stoichiometric_if_missing=False)

    # 路径 3: 先默认增强，再严格计算
    en = enrich_narrow_table(sample_incomplete_df, assume_stoichiometric_if_missing=True)
    xl3 = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)

    for xl in [xl1, xl2, xl3]:
        assert pd.isna(xl["xl_r_value"].iloc[0]), "r_value 必须为 NaN"
        assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry", "r_source 必须为 missing_stoichiometry"
        assert pd.isna(xl["xl_balance"].iloc[0]), "balance 必须为 NaN"
        assert pd.isna(xl["xl_alpha_max"].iloc[0]), "alpha_max 必须为 NaN"
        assert pd.isna(xl["xl_alpha_gel"].iloc[0]), "alpha_gel 必须为 NaN"
        assert pd.isna(xl["xl_nu_theory_mol_m3"].iloc[0]), "nu_theory 必须为 NaN"
        assert pd.isna(xl["xl_nu_junction_mol_m3"].iloc[0]), "nu_junction 必须为 NaN"
        assert xl["xl_coverage"].iloc[0] == "partial", "xl_coverage 必须为 partial"


def test_09_crosslink_tolerant_mode_preserves_partial_composition_proxy(sample_incomplete_df):
    """9. 宽容模式下部分组成代理正常保留并显式标记来源与覆盖度"""
    en = enrich_narrow_table(sample_incomplete_df, assume_stoichiometric_if_missing=True)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=True)
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert xl["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl["xl_nu_source"].iloc[0] == "partial_composition_proxy"
    assert np.isfinite(xl["xl_r_value"].iloc[0])
    assert np.isfinite(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_10_explicit_literature_r_not_masked_in_strict_mode():
    """10. 若存在独立可靠的显式文献 r，即使组成不完整，严格模式依然保留文献 r"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_RESIN"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "formulation_r_value": [0.75],
        "formulation_r_source": ["literature"],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert xl["xl_r_value"].iloc[0] == 0.75
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_coverage"].iloc[0] == "partial"


def test_11_clean_full_composition_preserves_full_coverage():
    """11. 干净完整的 DGEBA+EDA 配方无论何种模式均保持 xl_coverage='full'"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    xl_def = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert xl_def["xl_coverage"].iloc[0] == "full"
    assert xl_str["xl_coverage"].iloc[0] == "full"
    assert xl_def["xl_nu_source"].iloc[0] == "mechanism_aware"


def test_12_csv_parquet_roundtrip_strict_mode_consistency(sample_incomplete_df):
    """12. CSV 保存与重载后严格模式与直接严格计算完全一致"""
    en = enrich_narrow_table(sample_incomplete_df, assume_stoichiometric_if_missing=True)
    with pytest.MonkeyPatch.context() as m:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
            en.to_csv(tf.name, index=False)
            reloaded = pd.read_csv(tf.name)
            xl_reloaded = compute_crosslink_features(reloaded, assume_stoichiometric_if_missing=False)
            assert pd.isna(xl_reloaded["xl_r_value"].iloc[0])
            assert xl_reloaded["xl_r_source"].iloc[0] == "missing_stoichiometry"
            assert pd.isna(xl_reloaded["xl_balance"].iloc[0])


def test_13_historical_assumed_r_not_leaked_into_strict_mode():
    """13. 历史 assumed r 在严格模式下统一重置为 NaN，不发生来源与数值泄漏"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "formulation_r_value": [1.0],
        "cp_r_source": ["assumed_stoichiometric"],
        "rxn_r_assumed": [1.0],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"


def test_14_derived_parameters_synchronized_to_nan_in_strict_mode(sample_incomplete_df):
    """14. 严格模式下 W_g, epoxy_conc 等全部依赖整体 r 的派生量同步置 NaN"""
    xl = compute_crosslink_features(sample_incomplete_df, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl["xl_W_g_per_epoxy"].iloc[0])
    assert pd.isna(xl["xl_epoxy_conc_mol_m3"].iloc[0])
    assert pd.isna(xl["xl_Mc_g_mol"].iloc[0])


# =========================================================================
# 三、P1: mask_conflicts 在重复索引下的精准位置屏蔽 (场景 15 ~ 20)
# =========================================================================

def test_15_duplicate_string_index_masks_only_conflicted_row_position():
    """15. (P1 核心) 重复字符串索引 ['x', 'x'] 下只屏蔽冲突行位置 0，保留位置 1 的合法值"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [0.5, 0.5],
            "cp_r_value": [0.8, 0.5]
        },
        index=["x", "x"]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean.iloc[0]["formulation_r_value"])
    assert clean.iloc[1]["formulation_r_value"] == 0.5
    assert clean.index.tolist() == ["x", "x"]
    assert len(clean) == 2


def test_16_duplicate_integer_index_masks_only_conflicted_position():
    """16. 重复整数索引 [0, 0] 下精准按行位置屏蔽"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [0.5, 0.5],
            "cp_r_value": [0.8, 0.5]
        },
        index=[0, 0]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean.iloc[0]["formulation_r_value"])
    assert clean.iloc[1]["formulation_r_value"] == 0.5
    assert clean.index.tolist() == [0, 0]


def test_17_audit_records_contain_row_position_and_row_index():
    """17. 审计记录包含准确的 row_position (0..N-1) 与原始 row_index"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [0.5, 0.5],
            "cp_r_value": [0.8, 0.5]
        },
        index=["sample_A", "sample_A"]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert "row_position" in audit.columns
    assert "row_index" in audit.columns
    assert audit["row_position"].iloc[0] == 0
    assert audit["row_index"].iloc[0] == "sample_A"


def test_18_three_aliases_conflict_in_duplicate_index_cannot_be_refilled():
    """18. 重复索引场景下三个别名冲突后不能被后续候选字段回填"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [0.5, 0.5],
            "cp_r_value": [0.8, 0.5],
            "xl_r_value": [0.8, 0.5],
        },
        index=["key", "key"]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean.iloc[0]["formulation_r_value"])
    assert clean.iloc[1]["formulation_r_value"] == 0.5


def test_19_duplicate_columns_conflict_in_duplicate_index():
    """19. 重复索引场景下三个同名列冲突精准按行位置屏蔽"""
    df = pd.DataFrame(
        [[0.5, 0.8, 0.5], [0.5, 0.5, 0.5]],
        columns=["formulation_r_value"] * 3,
        index=["id_1", "id_1"]
    )
    clean, audit = prepare_ml_feature_matrix(df, conflict_policy="mask_conflicts")
    assert pd.isna(clean.iloc[0]["formulation_r_value"])
    assert clean.iloc[1]["formulation_r_value"] == 0.5
    assert clean.index.tolist() == ["id_1", "id_1"]


def test_20_pipeline_fit_transform_on_duplicate_index():
    """20. MLFeaturePipeline 在包含重复索引与冲突的数据集上契约稳定"""
    df = pd.DataFrame(
        {
            "formulation_r_value": [0.5, 0.5],
            "cp_r_value": [0.8, 0.5],
            "cp_r_source": ["literature", "literature"],
        },
        index=["x", "x"]
    )
    pipe = MLFeaturePipeline(feature_set="reaction_core", conflict_policy="mask_conflicts")
    clean, _ = pipe.fit_transform(df)
    assert pd.isna(clean.iloc[0]["formulation_r_value"])
    assert clean.iloc[1]["formulation_r_value"] == 0.5
    assert clean.index.tolist() == ["x", "x"]
