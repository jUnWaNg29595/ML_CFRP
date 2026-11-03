# -*- coding: utf-8 -*-
"""
tests/test_r_provenance_freeze_gate.py

CFRP 反应特征工程最终收尾验收与冻结门禁测试套件：
全面覆盖 Section IV 要求的 15 项真实接口测试：
1. 历史 xl_r_source=literature 但没有文献 r
2. 真实文献 r 与质量估计完全相同
3. 真实文献 r 与质量估计非常接近
4. 历史 partial r 与孤立 literature 标签共存
5. 历史 unknown r 与孤立 literature 标签共存
6. 历史 assumed r 被新真实文献值替代
7. 历史 assumed r 被完整质量计算值替代
8. 默认增强后未知来源不能升级为 literature
9. keep_metadata=True 和 False 的物理语义一致
10. CSV 与 Parquet 往返后来源稳定
11. 重复增强后数值和来源稳定
12. 修改固化剂质量后，旧 cp 不得静默控制当前结果
13. 更新真实文献 r 后，旧 W 与浓度缓存必须失效
14. DGEBA/EDA 正常基准不变
15. 酸酐、单官能、不完整组分及错误 SMILES 样本保持安全降级
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from core.reaction_core import evaluate_reaction_network
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
# 一、P0-1: 历史 xl_r_source 不能跨字段证明当前 r 是文献值 (1 ~ 5)
# =========================================================================

def test_01_historical_xl_r_source_literature_without_lit_r():
    """1. (P0-1) 部分组成配方残留历史 xl_r_source='literature'，无独立文献数值时，三条调用路径严格模式均为 NaN"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_amount_phr": 100.0,
        "resin_2_structure": "BAD",  # 质量缺失
        "curing_agent_1_structure": EDA,
        "curing_agent_1_amount_phr": 8.83,
        "xl_r_source": "literature",
    }])

    # 路径 1: 直接严格计算
    xl1 = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    # 路径 2: 默认增强后严格计算
    en_def = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)
    xl2 = compute_crosslink_features(en_def, assume_stoichiometric_if_missing=False)
    # 路径 3: keep_metadata=True 增强后严格计算
    en_meta = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=True)
    xl3 = compute_crosslink_features(en_meta, assume_stoichiometric_if_missing=False)

    for res in (xl1, xl2, xl3):
        assert len(res) == 1
        assert np.isnan(res["xl_r_value"].iloc[0])
        assert res["xl_r_source"].iloc[0] == "missing_stoichiometry"
        assert np.isnan(res["xl_balance"].iloc[0])
        assert np.isnan(res["xl_alpha_max"].iloc[0])
        assert np.isnan(res["xl_alpha_gel"].iloc[0])
        assert np.isnan(res["xl_nu_theory_mol_m3"].iloc[0])
        assert res["xl_coverage"].iloc[0] == "partial"


def test_02_real_literature_r_equals_mass_estimate():
    """2. (P0-1) 真实文献 r 与质量估计完全相等 (1.000299424)，严格模式必须正确保留并标记 literature"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "BAD",  # 组成不完整
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": 1.000299424,
        "formulation_r_source": "literature",
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl) == 1
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299424)
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert np.isclose(xl["xl_balance"].iloc[0], 1.0 / 1.000299424)
    assert xl["xl_coverage"].iloc[0] == "partial"
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_03_real_literature_r_close_to_estimate():
    """3. (P0-1) 真实文献 r 与估计值非常接近 (1.000349424)，严格模式正确保留并标记 literature"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "BAD",
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": 1.000349424,
        "formulation_r_source": "literature",
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl) == 1
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000349424)
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert np.isclose(xl["xl_balance"].iloc[0], 1.0 / 1.000349424)
    assert xl["xl_coverage"].iloc[0] == "partial"


def test_04_historical_partial_r_with_isolated_literature_tag():
    """4. (P0-1 & P0-2) 历史 partial r 与孤立 literature 标签共存时，严格模式输出 NaN，宽容模式保留代理"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": EDA,
        "formulation_r_value": np.nan,
        "formulation_r_source": "literature",
        "cp_r_value": 0.84,
        "cp_r_source": "partial_composition_calculated",
        "cp_composition_incomplete": True,
    }])

    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl_str) == 1
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert np.isnan(xl_str["xl_balance"].iloc[0])
    assert np.isnan(xl_str["xl_nu_theory_mol_m3"].iloc[0])

    xl_tol = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_tol["xl_r_value"].iloc[0], 0.84)
    assert xl_tol["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl_tol["xl_nu_source"].iloc[0] == "partial_composition_proxy"


def test_05_historical_unknown_r_with_isolated_literature_tag():
    """5. (P0-1 & P0-2) 历史 unknown r 与孤立 literature 标签共存时，严格模式输出 NaN，禁止升级为 literature"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": EDA,
        "formulation_r_value": np.nan,
        "formulation_r_source": "literature",
        "cp_r_value": 0.84,
        "cp_r_source": "unknown_provenance",
    }])

    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl_str) == 1
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"

    xl_tol = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_tol["xl_r_value"].iloc[0], 0.84)
    assert xl_tol["xl_r_source"].iloc[0] == "unknown_provenance"


# =========================================================================
# 二、P0-1 优先级: 新真实数据覆盖历史 assumed 标记 (6 ~ 7)
# =========================================================================

def test_06_historical_assumed_r_replaced_by_new_literature_r():
    """6. 历史 rxn_r_assumed=1.0 残留，新提供 formulation_r_value=0.75 与 literature，文献值优先保留"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": EDA,
        "formulation_r_value": 0.75,
        "formulation_r_source": "literature",
        "rxn_r_assumed": 1.0,
        "cp_r_source": "assumed_stoichiometric",
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl) == 1
    assert np.isclose(xl["xl_r_value"].iloc[0], 0.75)
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert np.isclose(xl["xl_balance"].iloc[0], 0.75)


def test_07_historical_assumed_r_replaced_by_complete_mass():
    """7. 历史 rxn_r_assumed=1.0 残留，新提供完整配方质量 (DGEBA 100, EDA 8.83)，完整质量计算优先保留"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "rxn_r_assumed": 1.0,
        "cp_r_source": "assumed_stoichiometric",
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert len(xl) == 1
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"


# =========================================================================
# 三、P0-2: 默认增强后未知来源不得升级为 literature (8 ~ 11)
# =========================================================================

def test_08_default_enrichment_unknown_source_not_upgraded_to_literature():
    """8. (P0-2) 输入仅有 cp_r_value=0.84 且无质量/文献，默认增强后严格计算必须拒绝，宽容模式标记 unknown"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "cp_r_value": [0.84],
    })

    en = enrich_narrow_table(df)
    xl_str = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert len(xl_str) == 1
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert np.isnan(xl_str["xl_balance"].iloc[0])
    assert np.isnan(xl_str["xl_nu_theory_mol_m3"].iloc[0])

    xl_tol = compute_crosslink_features(en, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_tol["xl_r_value"].iloc[0], 0.84)
    assert xl_tol["xl_r_source"].iloc[0] == "unknown_provenance"


def test_09_keep_metadata_true_and_false_physical_semantics_consistent():
    """9. (P0-2) keep_metadata=True 和 False 模式下，未知来源 r 的物理语义与来源标签 100% 一致"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "cp_r_value": [0.84],
    })

    en_false = enrich_narrow_table(df, keep_metadata=False)
    en_true = enrich_narrow_table(df, keep_metadata=True)

    xl_false_str = compute_crosslink_features(en_false, assume_stoichiometric_if_missing=False)
    xl_true_str = compute_crosslink_features(en_true, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_false_str["xl_r_value"].iloc[0])
    assert np.isnan(xl_true_str["xl_r_value"].iloc[0])
    assert xl_false_str["xl_r_source"].iloc[0] == xl_true_str["xl_r_source"].iloc[0] == "missing_stoichiometry"

    xl_false_tol = compute_crosslink_features(en_false, assume_stoichiometric_if_missing=True)
    xl_true_tol = compute_crosslink_features(en_true, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_false_tol["xl_r_value"].iloc[0], xl_true_tol["xl_r_value"].iloc[0])
    assert xl_false_tol["xl_r_source"].iloc[0] == xl_true_tol["xl_r_source"].iloc[0] == "unknown_provenance"


def test_10_csv_and_parquet_roundtrip_provenance_stability():
    """10. (P0-2) 默认增强后保存至 CSV 与 Parquet 并重载，严格模式物理来源保持绝对稳定"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "cp_r_value": [0.84],
    })
    en = enrich_narrow_table(df)

    with tempfile.TemporaryDirectory() as td:
        p_csv = os.path.join(td, "test.csv")
        p_parq = os.path.join(td, "test.parquet")
        en.to_csv(p_csv, index=False)
        en.to_parquet(p_parq, index=False)

        re_csv = pd.read_csv(p_csv)
        re_parq = pd.read_parquet(p_parq)

    xl_csv_str = compute_crosslink_features(re_csv, assume_stoichiometric_if_missing=False)
    xl_parq_str = compute_crosslink_features(re_parq, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_csv_str["xl_r_value"].iloc[0])
    assert np.isnan(xl_parq_str["xl_r_value"].iloc[0])
    assert xl_csv_str["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert xl_parq_str["xl_r_source"].iloc[0] == "missing_stoichiometry"


def test_11_repeated_enrichment_value_and_source_stability():
    """11. (P0-2) 连续增强两次，r 数值与来源标签幂等稳定"""
    df = pd.DataFrame({
        "resin_1_structure": [DGEBA],
        "curing_agent_1_structure": [EDA],
        "cp_r_value": [0.84],
    })
    en1 = enrich_narrow_table(df)
    en2 = enrich_narrow_table(en1)

    xl_str1 = compute_crosslink_features(en1, assume_stoichiometric_if_missing=False)
    xl_str2 = compute_crosslink_features(en2, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str1["xl_r_value"].iloc[0])
    assert np.isnan(xl_str2["xl_r_value"].iloc[0])
    assert xl_str1["xl_r_source"].iloc[0] == xl_str2["xl_r_source"].iloc[0] == "missing_stoichiometry"


# =========================================================================
# 四、P1: cp 缓存一致性契约与输入版本校验 (12 ~ 13)
# =========================================================================

def test_12_curer_mass_modified_old_cp_does_not_control_result():
    """12. (P1) 固化剂质量由 8.83 phr 修改为 6.0 phr 后，传入旧预计算 cp 不得静默控制当前交联特征"""
    df_init = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])
    en_init = enrich_narrow_table(df_init, keep_metadata=True)

    df_mod = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 6.0,
    }])

    xl_direct = compute_crosslink_features(df_mod, assume_stoichiometric_if_missing=False)
    xl_with_cp = compute_crosslink_features(df_mod, cp=en_init, assume_stoichiometric_if_missing=False)

    assert len(xl_direct) == len(xl_with_cp) == 1
    assert np.isclose(xl_direct["xl_r_value"].iloc[0], xl_with_cp["xl_r_value"].iloc[0])
    assert np.isclose(xl_direct["xl_r_value"].iloc[0], 0.679705, rtol=1e-4)
    assert np.isclose(xl_direct["xl_W_g_per_epoxy"].iloc[0], xl_with_cp["xl_W_g_per_epoxy"].iloc[0])
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], xl_with_cp["xl_nu_theory_mol_m3"].iloc[0])
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], 4988.30, rtol=1e-3)


def test_13_literature_r_updated_old_w_and_epoxy_conc_invalidated():
    """13. (P1) 更新真实文献 r=0.75 后，旧 W_g 与浓度缓存必须失效并重算"""
    df_init = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])
    cp_old = compute_component_physics(df_init)

    df_upd = df_init.copy()
    df_upd["formulation_r_value"] = 0.75
    df_upd["formulation_r_source"] = "literature"

    xl_direct = compute_crosslink_features(df_upd, assume_stoichiometric_if_missing=False)
    xl_with_cp = compute_crosslink_features(df_upd, cp=cp_old, assume_stoichiometric_if_missing=False)

    assert len(xl_direct) == len(xl_with_cp) == 1
    assert np.isclose(xl_direct["xl_r_value"].iloc[0], 0.75)
    assert np.isclose(xl_with_cp["xl_r_value"].iloc[0], 0.75)
    assert np.isclose(xl_direct["xl_W_g_per_epoxy"].iloc[0], xl_with_cp["xl_W_g_per_epoxy"].iloc[0])
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], xl_with_cp["xl_nu_theory_mol_m3"].iloc[0])
    assert np.isclose(xl_direct["xl_W_g_per_epoxy"].iloc[0], 181.47825)
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], 4959.27, rtol=1e-3)


# =========================================================================
# 五、基准物理量与安全降级测试 (14 ~ 15)
# =========================================================================

def test_14_standard_dgeba_eda_baselines_intact():
    """14. 标准 DGEBA/EDA 正常基准完好无损 (f_r=2, f_h=4, r=1.000299, alpha_gel=0.577, nu=4858.59)"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])

    xl = compute_crosslink_features(df, rho_g_cm3=1.2, assume_stoichiometric_if_missing=False)
    assert len(xl) == 1
    assert np.isclose(xl["xl_f_r"].iloc[0], 2.0)
    assert np.isclose(xl["xl_f_h"].iloc[0], 4.0)
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert np.isclose(xl["xl_alpha_gel"].iloc[0], 1.0 / np.sqrt(3.0), rtol=1e-3)
    assert np.isclose(xl["xl_nu_theory_mol_m3"].iloc[0], 4858.59, rtol=1e-2)
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"


def test_15_anhydride_monofunctional_and_invalid_smiles_safe_degradation():
    """15. 酸酐双口径、单官能 PGE 不凝胶及错误 SMILES 样本保持安全降级，不丢行"""
    # 酸酐
    rc_sa = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": SA}], explicit_r=1.0)
    assert rc_sa.f_h_stoich == 1.0
    assert rc_sa.f_h_network == 2.0

    # 单官能体系
    df_pge = pd.DataFrame([{
        "resin_1_structure": PGE,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 10.0,
    }])
    xl_pge = compute_crosslink_features(df_pge, assume_stoichiometric_if_missing=False)
    assert len(xl_pge) == 1
    assert np.isnan(xl_pge["xl_nu_theory_mol_m3"].iloc[0])
    assert xl_pge["xl_nu_source"].iloc[0] == "non_gelling"

    # 含噪声错误 SMILES 样本不丢行
    df_noisy = pd.DataFrame([
        {"sample_id": "1", "resin_1_structure": DGEBA, "curing_agent_1_structure": EDA, "formulation_r_value": 0.8},
        {"sample_id": "2", "resin_1_structure": "INVALID_RESIN", "curing_agent_1_structure": EDA, "formulation_r_value": np.nan},
    ])
    en_noisy = enrich_narrow_table(df_noisy, keep_metadata=True)
    xl_noisy = compute_crosslink_features(en_noisy, assume_stoichiometric_if_missing=False)
    assert len(en_noisy) == 2
    assert len(xl_noisy) == 2
    assert xl_noisy["xl_r_value"].iloc[0] == 0.8
    assert xl_noisy["xl_r_source"].iloc[0] == "literature"
    assert np.isnan(xl_noisy["xl_r_value"].iloc[1])
    assert xl_noisy["xl_r_source"].iloc[1] == "missing_stoichiometry"
