# -*- coding: utf-8 -*-
"""
tests/test_crosslink_r_provenance_final.py

CFRP 反应特征工程终期补丁专项测试套件：
严格覆盖 Section VI 要求的 18 项 r 来源优先级、文献证据校验与严格模式派生量一致性测试。
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

# 经典单体结构
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA = "NCCN"
PGE = "c1ccccc1OCC1CO1"  # 单官能环氧
SA = "O=C1CCC(=O)O1"     # 琥珀酸酐


# =========================================================================
# 一、P0-1: 真实质量与文献 r 优先于历史 assumed 标记 (场景 1 ~ 5)
# =========================================================================

def test_01_complete_mass_without_assumed_tag():
    """1. 完整 DGEBA/EDA 质量，无历史 assumed 标记，严格模式正常计算出完整质量 r"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isfinite(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"
    assert np.isfinite(xl["xl_balance"].iloc[0])


def test_02_complete_mass_with_historical_assumed_tag():
    """2. 完整 DGEBA/EDA 质量，但原始表存在旧 rxn_r_assumed=1.0，新质量优先，不被清空 (P0-1 反例A)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "rxn_r_assumed": [1.0],
        "cp_r_source": ["assumed_stoichiometric"],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isfinite(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"
    assert np.isfinite(xl["xl_balance"].iloc[0])


def test_03_cp_calculated_from_mass_overrides_raw_assumed_source():
    """3. cp_r_source=calculated_from_mass 明确优先于原始表的历史 assumed 标记"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "rxn_r_assumed": [1.0],
        "cp_r_source": ["assumed_stoichiometric"],
    })
    cp = pd.DataFrame({
        "cp_r_value": [1.000299],
        "cp_r_source": ["calculated_from_mass"],
        "cp_composition_incomplete": [False],
    })
    from core.cp_cache_integrity import stamp_cp_cache
    cp = stamp_cp_cache(cp, df, assume_stoichiometric_if_missing=False)
    xl = compute_crosslink_features(df, cp=cp, assume_stoichiometric_if_missing=False)
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert np.isfinite(xl["xl_balance"].iloc[0])


def test_04_explicit_literature_r_without_assumed():
    """4. 明确文献 r=0.75，没有历史 assumed，严格模式正常保留文献 r"""
    df = pd.DataFrame({
        "formulation_r_value": [0.75],
        "formulation_r_source": ["literature"],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert xl["xl_r_value"].iloc[0] == 0.75
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_balance"].iloc[0] == 0.75


def test_05_explicit_literature_r_overrides_historical_assumed_tag():
    """5. 明确文献 r=0.75，同时存在旧 rxn_r_assumed=1.0，新文献事实优先 (P0-1 反例B)"""
    df = pd.DataFrame({
        "formulation_r_value": [0.75],
        "formulation_r_source": ["literature"],
        "rxn_r_assumed": [1.0],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert xl["xl_r_value"].iloc[0] == 0.75
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_balance"].iloc[0] == 0.75


# =========================================================================
# 二、P0-2: literature 来源必须与真实数值绑定校验 (场景 6 ~ 10)
# =========================================================================

def test_06_isolated_literature_tag_without_numerical_value():
    """6. 只有 r_source=literature 但没有 r 数值，严格模式输出 NaN (P0-2)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "r_source": ["literature"],  # 孤立标签，无数值!
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert pd.isna(xl["xl_balance"].iloc[0])


def test_07_isolated_literature_tag_with_partial_cp_does_not_bypass_strict():
    """7. 存在部分组成 cp r 时，孤立的 literature 标签不能放行绕过严格模式 (P0-2 核心反例)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNRESOLVED_RESIN"],  # 未解析组分
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "formulation_r_source": ["literature"],          # 孤立标签!
    })
    # 先默认增强产生 cp 部分组成估计
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    # 严格模式计算
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert pd.isna(xl["xl_balance"].iloc[0])
    assert xl["xl_coverage"].iloc[0] == "partial"


def test_08_partial_cp_r_without_literature_source():
    """8. 部分组成 cp r 在无文献来源时，严格模式输出 NaN，宽容模式保留代理"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)

    xl_strict = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl_strict["xl_r_value"].iloc[0])
    assert xl_strict["xl_r_source"].iloc[0] == "missing_stoichiometry"

    xl_tolerant = compute_crosslink_features(en, assume_stoichiometric_if_missing=True)
    assert np.isfinite(xl_tolerant["xl_r_value"].iloc[0])
    assert xl_tolerant["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl_tolerant["xl_coverage"].iloc[0] == "partial"


def test_09_historical_assumed_r_without_new_evidence():
    """9. 历史 assumed r 无新证据时，严格模式输出 NaN"""
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


def test_10_historical_unknown_provenance_r():
    """10. 历史 unknown_provenance r 在严格模式下不当作真实值，输出 NaN"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "cp_r_value": [0.7],
        "cp_r_source": ["unknown_provenance"],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert pd.isna(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"


# =========================================================================
# 三、文献 r 与质量计算并存及冲突处理 (场景 11 ~ 13)
# =========================================================================

def test_11_real_literature_r_with_incomplete_composition():
    """11. 真实文献 r 与部分组成配方共存时，严格模式依然保留文献 r (P0-2 保证)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD_UNRESOLVED_RESIN"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
        "formulation_r_value": [0.75],
        "formulation_r_source": ["literature"],
    })
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert xl["xl_r_value"].iloc[0] == 0.75
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert xl["xl_balance"].iloc[0] == 0.75


def test_12_literature_r_conflicts_with_mass_calculation():
    """12. 真实文献 r 与完整质量计算 r 冲突时，优先保留文献 r"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],  # 质量推导 r ≈ 1.0
        "formulation_r_value": [0.5],          # 文献实测 r = 0.5
        "formulation_r_source": ["literature"],
    })
    en = enrich_narrow_table(df)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert xl["xl_r_value"].iloc[0] == 0.5
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_balance"].iloc[0] == 0.5


def test_13_three_calling_paths_strict_mode_consistency():
    """13. 三种调用路径 (直接、传 cp、先默认增强再严格) 严格模式输出物理语义 100% 一致"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })

    # 路径 1: 直接严格计算
    xl1 = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)

    # 路径 2: 传入默认 cp 后严格计算
    cp = compute_component_physics(df)
    xl2 = compute_crosslink_features(df, cp=cp, assume_stoichiometric_if_missing=False)

    # 路径 3: 先默认增强，再严格计算
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    xl3 = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)

    for res in (xl1, xl2, xl3):
        assert pd.isna(res["xl_r_value"].iloc[0])
        assert res["xl_r_source"].iloc[0] == "missing_stoichiometry"
        assert pd.isna(res["xl_balance"].iloc[0])
        assert pd.isna(res["xl_alpha_gel"].iloc[0])
        assert pd.isna(res["xl_nu_theory_mol_m3"].iloc[0])
        assert res["xl_coverage"].iloc[0] == "partial"


# =========================================================================
# 四、持久化、幂等性与物理派生量联动 (场景 14 ~ 18)
# =========================================================================

def test_14_csv_and_parquet_roundtrip_strict_mode_lineage():
    """14. CSV 和 Parquet 往返保存后严格模式物理量与来源标签保持稳定"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)

    with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
        en.to_csv(tf.name, index=False)
        reloaded_csv = pd.read_csv(tf.name)
        xl_csv = compute_crosslink_features(reloaded_csv, assume_stoichiometric_if_missing=False)
        assert pd.isna(xl_csv["xl_r_value"].iloc[0])
        assert xl_csv["xl_r_source"].iloc[0] == "missing_stoichiometry"

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tf:
        en.to_parquet(tf.name, index=False)
        reloaded_pq = pd.read_parquet(tf.name)
        xl_pq = compute_crosslink_features(reloaded_pq, assume_stoichiometric_if_missing=False)
        assert pd.isna(xl_pq["xl_r_value"].iloc[0])
        assert xl_pq["xl_r_source"].iloc[0] == "missing_stoichiometry"


def test_15_repeated_strict_calculations_idempotent():
    """15. 连续两次执行严格计算，结果完全幂等"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    xl_1 = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    xl_2 = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    pd.testing.assert_frame_equal(xl_1, xl_2)


def test_16_tolerant_mode_preserves_partial_composition_proxy():
    """16. 默认宽容模式保留带来源标记的部分组成代理"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    en = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=True)
    assert np.isfinite(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert xl["xl_nu_source"].iloc[0] == "partial_composition_proxy"


def test_17_standard_baselines_remain_intact():
    """17. 正常 DGEBA/EDA、酸酐、单官能及不适用化学体系的原有物理基准保持不变"""
    # 正常 DGEBA + EDA (r=1.0)
    res_dgeba = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": EDA}], explicit_r=1.0)
    assert res_dgeba.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert res_dgeba.gel_possible_ideal == 1.0

    # 单官能环氧体系 (PGE + EDA)
    res_mono = evaluate_reaction_network([{"smiles": PGE}], [{"smiles": EDA}], explicit_r=1.0)
    assert res_mono.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res_mono.nu_ideal_mol_m3 == 0.0

    # 酸酐双口径 (SA)
    res_sa = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": SA}], explicit_r=1.0)
    assert res_sa.f_h_stoich == 1.0
    assert res_sa.f_h_network == 2.0

    # 非环氧不适用体系
    res_inapp = evaluate_reaction_network([{"smiles": "CCO"}], [{"smiles": EDA}])
    assert res_inapp.gel_regime == GEL_REGIME_NOT_APPLICABLE
    assert np.isnan(res_inapp.nu_ideal_mol_m3)


def test_18_r_and_all_derived_parameters_strict_synchronization():
    """18. r 与 balance、alpha_max、W_g、epoxy_conc、凝胶点、交联密度在严格模式下严格同步为 NaN (Section V)"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "resin_1_amount_phr": [100.0],
        "resin_2_structure": ["BAD"],
        "curing_agent_1_structure": [EDA],
        "curing_agent_1_amount_phr": [8.83],
    })
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)

    # 验证 r 为 NaN 时，所有依赖 r 的派生量必须 100% 同步为 NaN
    assert pd.isna(xl["xl_r_value"].iloc[0])
    assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert pd.isna(xl["xl_balance"].iloc[0])
    assert pd.isna(xl["xl_alpha_max"].iloc[0])
    assert pd.isna(xl["xl_W_g_per_epoxy"].iloc[0])
    assert pd.isna(xl["xl_epoxy_conc_mol_m3"].iloc[0])
    assert pd.isna(xl["xl_alpha_gel"].iloc[0])
    assert pd.isna(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert pd.isna(xl["xl_nu_junction_mol_m3"].iloc[0])
    assert pd.isna(xl["xl_Mc_g_mol"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "missing_stoichiometry"
    assert xl["xl_coverage"].iloc[0] == "partial"
