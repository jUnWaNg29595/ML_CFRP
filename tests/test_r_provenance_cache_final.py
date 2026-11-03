# -*- coding: utf-8 -*-
"""
tests/test_r_provenance_cache_final.py

CFRP 反应特征工程终期验收测试套件：
全面覆盖独立审核指出的 20 项专项回归测试：
1. 真实文献 r 与部分组成质量估计完全相等
2. 真实文献 r 与估计值差异小于 1e-4
3. 历史 partial_composition_calculated 且无 cp_r_from_mass
4. 历史 unknown_provenance 且无 cp_r_from_mass
5. 只有 literature 标签但无有效文献 r 数值
6. 新完整质量覆盖历史 assumed
7. 新真实文献 r 覆盖历史 assumed
8. 新 r 覆盖旧 cp 缓存后，W 和交联密度一致
9. 更新前后重复增强不残留旧 r 派生量
10. 部分组成 r 补入 formulation_r_value 后来源同步
11. 原始文献 r 已存在时，不被增强覆盖
12. keep_metadata=False 仍保留必要来源血统
13. CSV 保存重载后来源稳定
14. Parquet 保存重载后来源稳定
15. 严格模式三种调用路径一致
16. 宽容模式保留部分组成代理
17. DGEBA/EDA 正常物理数值不变
18. 酸酐与单官能基准不变
19. 含错误 SMILES 的样本不因单个字段无效而丢行
20. A/B/C/D 特征集合数量与原有规范保持兼容
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
from core.epoxy_mechanism_features import extract_epoxy_mechanism_features
from core.feature_registry import get_features_for_set
from core.feature_deduplication import prepare_ml_feature_matrix

# 经典单体结构
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA = "NCCN"
PGE = "c1ccccc1OCC1CO1"  # 单官能环氧
SA = "O=C1CCC(=O)O1"     # 琥珀酸酐


# =========================================================================
# 一、P0-1: 真实文献 r 即使与部分组成质量估计相等，也必须严格保留 (1 ~ 2)
# =========================================================================

def test_01_real_literature_r_equals_partial_composition_estimate():
    """1. 真实文献 r 与部分组成质量估计完全相等 (1.000299424)，严格模式必须正确保留并标记 literature"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # 缺少质量，组成不完整
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
    assert xl["xl_coverage"].iloc[0] == "partial"  # 存在未识别组分，覆盖度标为 partial
    # 严格模式下，组成不完整时理论交联密度 nu 必须为 NaN
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_02_real_literature_r_close_to_estimate_diff_less_than_1e4():
    """2. 真实文献 r 与估计值微小差异小于 1e-4 (如 1.000349424)，严格模式不可遭错误屏蔽"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
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


# =========================================================================
# 二、P0-2: 旧数据缺少 cp_r_from_mass 时，禁止历史 r 升级为 literature (3 ~ 5)
# =========================================================================

def test_03_historical_partial_without_cp_r_from_mass():
    """3. 历史数据仅有 cp_r_source='partial_composition_calculated' 且无 cp_r_from_mass，严格模式置 NaN"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": EDA,
        "formulation_r_value": 0.84,
        "cp_r_source": "partial_composition_calculated",
        "cp_composition_incomplete": True,
    }])

    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert np.isnan(xl_str["xl_balance"].iloc[0])
    assert np.isnan(xl_str["xl_nu_theory_mol_m3"].iloc[0])

    # 宽容模式可以保留作为代理
    xl_def = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_def["xl_r_value"].iloc[0], 0.84)
    assert xl_def["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl_def["xl_nu_source"].iloc[0] == "partial_composition_proxy"


def test_04_historical_unknown_without_cp_r_from_mass():
    """4. 历史数据标记 cp_r_source='unknown_provenance' 且无 cp_r_from_mass，严格模式禁止静默升级为 literature"""
    df = pd.DataFrame({
        "formulation_r_value": [0.84],
        "cp_r_source": ["unknown_provenance"],
    })

    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"

    xl_def = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl_def["xl_r_value"].iloc[0], 0.84)
    assert xl_def["xl_r_source"].iloc[0] == "unknown_provenance"


def test_05_isolated_literature_tag_without_numerical_value():
    """5. 仅有 formulation_r_source='literature' 标签但 formulation_r_value 为 NaN，严格模式禁止放行部分组成"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # 缺质量
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": np.nan,
        "formulation_r_source": "literature",
    }])

    xl_str = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"
    assert np.isnan(xl_str["xl_nu_theory_mol_m3"].iloc[0])


# =========================================================================
# 三、P0-1 优先级: 新真实数据覆盖历史 assumed 标记 (6 ~ 7)
# =========================================================================

def test_06_new_complete_mass_overrides_historical_assumed():
    """6. 新完整质量 (DGEBA 100 phr, EDA 8.83 phr) 成功推算 r，历史 rxn_r_assumed=1.0 被完全覆盖"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "rxn_r_assumed": 1.0,
        "cp_r_source": "assumed_stoichiometric",
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"


def test_07_new_real_literature_r_overrides_historical_assumed():
    """7. 新明确文献 r (formulation_r_value=0.75, literature) 覆盖历史 rxn_r_assumed=1.0"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": EDA,
        "formulation_r_value": 0.75,
        "formulation_r_source": "literature",
        "rxn_r_assumed": 1.0,
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isclose(xl["xl_r_value"].iloc[0], 0.75)
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert np.isclose(xl["xl_balance"].iloc[0], 0.75)


# =========================================================================
# 四、P0-3: r 更新后旧 cp 缓存派生量同步更新与失效 (8 ~ 9)
# =========================================================================

def test_08_r_update_invalidates_old_cp_w_and_epoxy_conc():
    """8. 完整 DGEBA/EDA 配方更新文献 r=0.75 后，传入旧预计算 cp 与直接计算结果 100% 一致"""
    df_init = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])
    cp_old = compute_component_physics(df_init)

    df_updated = df_init.copy()
    df_updated["formulation_r_value"] = 0.75
    df_updated["formulation_r_source"] = "literature"

    xl_direct = compute_crosslink_features(df_updated, assume_stoichiometric_if_missing=False)
    xl_with_cp = compute_crosslink_features(df_updated, cp=cp_old, assume_stoichiometric_if_missing=False)

    assert np.isclose(xl_direct["xl_r_value"].iloc[0], 0.75)
    assert np.isclose(xl_with_cp["xl_r_value"].iloc[0], 0.75)
    assert np.isclose(xl_direct["xl_W_g_per_epoxy"].iloc[0], xl_with_cp["xl_W_g_per_epoxy"].iloc[0])
    assert np.isclose(xl_direct["xl_epoxy_conc_mol_m3"].iloc[0], xl_with_cp["xl_epoxy_conc_mol_m3"].iloc[0])
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], xl_with_cp["xl_nu_theory_mol_m3"].iloc[0])
    assert np.isclose(xl_direct["xl_nu_theory_mol_m3"].iloc[0], 4959.27, rtol=1e-3)


def test_09_repeated_enrichment_does_not_leak_stale_r_derivatives():
    """9. 更新前后重复调用 enrich_narrow_table 不残留旧 r 派生量"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])
    en1 = enrich_narrow_table(df, keep_metadata=True)
    assert np.isclose(en1["formulation_r_value"].iloc[0], 1.000299, rtol=1e-4)

    # 外部更新为文献值 0.75
    en1["formulation_r_value"] = 0.75
    en1["formulation_r_source"] = "literature"
    en2 = enrich_narrow_table(en1, keep_metadata=True)

    assert np.isclose(en2["formulation_r_value"].iloc[0], 0.75)
    assert en2["formulation_r_source"].iloc[0] == "literature"
    assert np.isclose(en2["cp_W_g_per_epoxy"].iloc[0], 170.2095 + 0.75 * 15.025)


# =========================================================================
# 五、P1: enrich_narrow_table 填补 r 时来源同步更新 (10 ~ 12)
# =========================================================================

def test_10_partial_composition_r_filled_synchronizes_source():
    """10. 原始 formulation_r_value 为空但带有孤立 literature 标签，补入部分组成 r 时来源同步为 partial"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # 缺质量
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": np.nan,
        "formulation_r_source": "literature",
    }])

    en = enrich_narrow_table(df, keep_metadata=True)
    assert np.isclose(en["formulation_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert en["formulation_r_source"].iloc[0] == "partial_composition_calculated"
    assert en["cp_r_source"].iloc[0] == "partial_composition_calculated"


def test_11_original_literature_r_not_overwritten_by_enrichment():
    """11. 原始文献 r=0.75 已存在时，enrich_narrow_table 严格保留原数值与来源"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": 0.75,
        "formulation_r_source": "literature",
    }])

    en = enrich_narrow_table(df, keep_metadata=True)
    assert np.isclose(en["formulation_r_value"].iloc[0], 0.75)
    assert en["formulation_r_source"].iloc[0] == "literature"


def test_12_keep_metadata_false_preserves_lineage_and_prevents_string_columns():
    """12. keep_metadata=False 时，原表存在的 formulation_r_source 正确更新且不产生未许可字符串列"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # 缺质量
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": np.nan,
        "formulation_r_source": "literature",
    }])

    en = enrich_narrow_table(df, keep_metadata=False)
    # 原表存在的 formulation_r_source 来源已修正
    assert en["formulation_r_source"].iloc[0] == "partial_composition_calculated"
    # 新增列中不得包含未许可的字符串列
    new_cols = [c for c in en.columns if c not in df.columns]
    bad = [c for c in new_cols if not pd.api.types.is_numeric_dtype(en[c]) and not pd.api.types.is_bool_dtype(en[c])]
    assert not bad, f"keep_metadata=False 新增列违背纯数值/布尔约束: {bad}"


# =========================================================================
# 六、持久化往返与多路径一致性 (13 ~ 16)
# =========================================================================

def test_13_csv_roundtrip_provenance_stability():
    """13. CSV 保存并重载后，严格模式下来源判定保持绝对稳定"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": 0.84,
        "cp_r_source": "partial_composition_calculated",
        "cp_composition_incomplete": True,
    }])

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.csv")
        df.to_csv(p, index=False)
        reloaded = pd.read_csv(p)

    xl_str = compute_crosslink_features(reloaded, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"


def test_14_parquet_roundtrip_provenance_stability():
    """14. Parquet 保存并重载后，严格模式下来源判定保持绝对稳定"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
        "formulation_r_value": 0.84,
        "cp_r_source": "partial_composition_calculated",
        "cp_composition_incomplete": True,
    }])

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.parquet")
        df.to_parquet(p, index=False)
        reloaded = pd.read_parquet(p)

    xl_str = compute_crosslink_features(reloaded, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_str["xl_r_value"].iloc[0])
    assert xl_str["xl_r_source"].iloc[0] == "missing_stoichiometry"


def test_15_three_calling_paths_strict_mode_consistency():
    """15. 组成不完整样本在严格模式下三种调用路径物理输出 100% 一致"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",  # 缺质量
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])

    # 路径 1: 直接严格计算
    xl1 = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    # 路径 2: 先生成默认 cp 再严格计算
    cp = compute_component_physics(df)
    xl2 = compute_crosslink_features(df, cp=cp, assume_stoichiometric_if_missing=False)
    # 路径 3: 先默认增强再严格计算
    en = enrich_narrow_table(df, keep_metadata=True)
    xl3 = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)

    for xl in (xl1, xl2, xl3):
        assert np.isnan(xl["xl_r_value"].iloc[0])
        assert xl["xl_r_source"].iloc[0] == "missing_stoichiometry"
        assert np.isnan(xl["xl_balance"].iloc[0])
        assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
        assert xl["xl_coverage"].iloc[0] == "partial"


def test_16_tolerant_mode_preserves_partial_composition_proxy():
    """16. 宽容模式下部分组成估计正常保留为代理特征，并显式标记来源"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "resin_2_structure": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert xl["xl_r_source"].iloc[0] == "partial_composition_calculated"
    assert xl["xl_nu_source"].iloc[0] == "partial_composition_proxy"
    assert xl["xl_coverage"].iloc[0] == "partial"


# =========================================================================
# 七、既有化学物理与数据治理基准验证 (17 ~ 20)
# =========================================================================

def test_17_standard_dgeba_eda_physics_intact():
    """17. 标准 DGEBA/EDA 理论物理量数值与公式基准完好无损"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 8.83,
    }])
    xl = compute_crosslink_features(df, rho_g_cm3=1.2, assume_stoichiometric_if_missing=False)
    assert np.isclose(xl["xl_f_r"].iloc[0], 2.0)
    assert np.isclose(xl["xl_f_h"].iloc[0], 4.0)
    assert np.isclose(xl["xl_r_value"].iloc[0], 1.000299, rtol=1e-4)
    assert np.isclose(xl["xl_alpha_gel"].iloc[0], 1.0 / np.sqrt(3.0), rtol=1e-3)
    assert np.isclose(xl["xl_nu_theory_mol_m3"].iloc[0], 4858.59, rtol=1e-2)
    assert xl["xl_r_source"].iloc[0] == "calculated_from_mass"
    assert xl["xl_coverage"].iloc[0] == "full"


def test_18_anhydride_and_monofunctional_baselines_intact():
    """18. 酸酐双口径 (f_stoich=1, f_net=2) 与单官能 PGE 不凝胶基准完好无损"""
    # 酸酐
    rc_sa = evaluate_reaction_network([{"smiles": DGEBA}], [{"smiles": SA}], explicit_r=1.0)
    assert rc_sa.f_h_stoich == 1.0
    assert rc_sa.f_h_network == 2.0

    # 单官能 PGE 体系
    df_pge = pd.DataFrame([{
        "resin_1_structure": PGE,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA,
        "curing_agent_1_phr": 10.0,
    }])
    xl_pge = compute_crosslink_features(df_pge, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_pge["xl_nu_theory_mol_m3"].iloc[0])
    assert xl_pge["xl_nu_source"].iloc[0] == "non_gelling"


def test_19_invalid_smiles_does_not_drop_sample():
    """19. 包含错误 SMILES 的含噪声文献样本不丢行，合理标记并在严格模式下保留元数据"""
    df = pd.DataFrame([
        {"sample_id": "S1", "resin_1_structure": DGEBA, "curing_agent_1_structure": EDA, "formulation_r_value": 0.9},
        {"sample_id": "S2", "resin_1_structure": "INVALID_RESIN_SMILES", "curing_agent_1_structure": EDA, "formulation_r_value": 1.0},
        {"sample_id": "S3", "resin_1_structure": DGEBA, "curing_agent_1_structure": "NOT_A_SMILES", "formulation_r_value": np.nan},
    ])
    en = enrich_narrow_table(df, keep_metadata=True, assume_stoichiometric_if_missing=False)
    xl = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert len(en) == 3
    assert len(xl) == 3
    assert xl["xl_r_value"].iloc[0] == 0.9
    assert xl["xl_r_source"].iloc[0] == "literature"
    assert xl["xl_r_source"].iloc[1] == "literature"  # S2 有明确文献 r=1.0，独立文献值不因结构解析失败而丢弃


def test_20_abcd_feature_set_counts_compatible():
    """20. A/B/C/D 分层特征集合维度 (Set A=28, Set B=18, Set C=10, Set D=56) 严格兼容"""
    set_a = get_features_for_set("reaction_core")
    set_b = get_features_for_set("molecular")
    set_c = get_features_for_set("formulation_process")
    set_d = get_features_for_set("full_safe")

    assert len(set_a) == 28, f"Set A 期望 28 维，实际 {len(set_a)}"
    assert len(set_b) == 18, f"Set B 期望 18 维，实际 {len(set_b)}"
    assert len(set_c) == 10, f"Set C 期望 10 维，实际 {len(set_c)}"
    assert len(set_d) == 56, f"Set D 期望 56 维，实际 {len(set_d)}"

    # 构造包含 Set D 规范特征列以及泄漏目标列的测试数据
    dummy_data = {f: [1.0] for f in set_d}
    dummy_data["tg_c"] = [150.0]  # 目标列必须被阻断
    dummy_data["tensile_strength_mpa"] = [80.0]  # 力学标签必须被阻断
    df_test = pd.DataFrame(dummy_data)

    clean_d, audit = prepare_ml_feature_matrix(df_test, feature_set="full_safe")
    assert clean_d.shape[1] == 56
    assert "tg_c" not in clean_d.columns
    assert "tensile_strength_mpa" not in clean_d.columns
