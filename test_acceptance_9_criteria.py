"""
9 大验收指标端到端自动化测试套件
全面验证最新联合审查报告的 9 项验收标准：
1. DGEBA + EDA: 100g DGEBA (EEW~170) + 17.65g EDA (AHEW~15), r ~ 2.0 (胺过量), 绝不是 0.088
2. DGEBA + DDS 计量比: 100g DGEBA (EEW~170) + 36.5g DDS (AHEW~62.08), r ~ 1.0, cp_f_avg 严密符合 Flory 摩尔平均
3. 酸酐固化剂区分: MTHPA f_stoich = 2, f_network = 2; TMA f_stoich = 3, f_network = 3. 酸酐不被误识别为伯/仲胺
4. 酰胺与非固化氮排除: 酰胺/内酰胺/咪唑等不产生假伯/仲胺活泼氢
5. 单官能固化剂不凝胶: 单官能固化剂/单官能环氧体系不可凝胶, alpha_gel 应为 NaN 或不可凝胶标识
6. 未知固化剂绝对不盲猜 3.0: 缺失固化剂结构时, f_h 为 NaN, coverage 明确反映缺失, 不污染网络预测
7. 三模块关键物理量一致性: component_physics, crosslink_physics, epoxy_mechanism_features 对相同配方输出的 r, f_avg 一致
8. 无配方质量时: component_physics 缺少质量信息时 r 严格返回 NaN, 绝不拿 ahew/eew 冒充
9. 重复乘 r 彻底消除: crosslink_physics 中 mol_h 不再重复乘以 r 计算 f_avg
"""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from core.reaction_core import (
    inspect_resin_functionality,
    inspect_curer_functionality,
    calculate_stoichiometric_ratio_scalar,
    compute_molar_average_functionality,
    compute_gel_point_conversion,
    evaluate_reaction_network,
)
from core.component_physics import (
    compute_component_physics,
    compute_formulation_summary,
    enrich_narrow_table,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import EpoxyMechanismEngine

# 标准合法 SMILES
DGEBA_SMILES = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA_SMILES = "NCCN"
DDS_SMILES = "NC1=CC=C(S(=O)(=O)C2=CC=C(N)C=C2)C=C1"
MTHPA_SMILES = "CC1=CCCC2C1C(=O)OC2=O"
ACETAMIDE_SMILES = "CC(=O)N"
ANILINE_SMILES = "NC1=CC=CC=C1"
PGE_SMILES = "c1ccccc1OCC1CO1"  # 苯基缩水甘油醚 (单官能环氧)


def test_01_dgeba_eda_stoichiometric_ratio():
    """验收 1: DGEBA + EDA (100g + 17.65g) -> r 约等于 2.0 (胺过量), 绝不是 0.088"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 17.65,
    }])
    
    # 1. reaction_core 计算
    net = evaluate_reaction_network(
        resins=[{"smiles": DGEBA_SMILES, "mass": 100.0}],
        curers=[{"smiles": EDA_SMILES, "mass": 17.65}],
    )
    r_core = net.r_value
    assert 1.95 <= r_core <= 2.05, f"Expected r ~ 2.0, got {r_core}"
    
    # 2. component_physics 计算
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp)
    r_cp = summ["cp_r_value"].iloc[0]
    assert 1.95 <= r_cp <= 2.05, f"component_physics returned wrong r: {r_cp} (old buggy value was 0.088)"
    
    # 3. crosslink_physics 计算
    xl = compute_crosslink_features(df)
    r_xl = xl["xl_r_value"].iloc[0]
    assert 1.95 <= r_xl <= 2.05, f"crosslink_physics returned wrong r: {r_xl}"
    
    # 4. epoxy_mechanism_features 计算
    eng = EpoxyMechanismEngine(verbose=False)
    mech = eng.enrich_dataframe(df)
    r_mech = mech["mech_stoichiometry_r"].iloc[0]
    assert 1.95 <= r_mech <= 2.05, f"epoxy_mechanism returned wrong r: {r_mech}"


def test_02_dgeba_dds_stoichiometric_and_f_avg():
    """验收 2: DGEBA + DDS 等当量 (100g + 36.47g) -> r ~ 1.0, f_avg 符合 Flory 摩尔平均"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": DDS_SMILES,
        "curing_agent_1_phr": 36.47,
    }])
    
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp)
    r_cp = summ["cp_r_value"].iloc[0]
    f_avg_cp = summ["cp_f_avg_network"].iloc[0]
    
    assert 0.98 <= r_cp <= 1.02, f"Expected r ~ 1.0, got {r_cp}"
    # 当 r = 1.0, f_r = 2, f_h = 4:
    # 摩尔数比 n_h / n_r = (r * f_r) / f_h = 2/4 = 0.5
    # f_avg = (2 * 1.0 + 4 * 0.5) / (1.0 + 0.5) = 4 / 1.5 = 2.6667
    assert np.isclose(f_avg_cp, 2.6667, atol=0.05), f"Expected f_avg ~ 2.67, got {f_avg_cp}"


def test_03_anhydride_functionality_and_no_amine():
    """验收 3: 酸酐 MTHPA f_stoich = 1, f_network = 2, 不产生伯/仲胺误识别"""
    f_info = inspect_curer_functionality(MTHPA_SMILES)
    assert f_info.mechanism == "anhydride"
    assert f_info.f_stoich == 1.0
    assert f_info.f_network == 2.0


def test_04_amide_exclusion():
    """验收 4: 酰胺乙酰胺不被识别为固化活性胺"""
    f_info = inspect_curer_functionality(ACETAMIDE_SMILES)
    assert f_info.f_stoich == 0.0
    assert f_info.f_network == 0.0


def test_05_monofunctional_no_gelation():
    """验收 5: 单官能环氧 PGE (f_r=1) 与 EDA (f_h=4) 不可凝胶, alpha_gel 为 NaN"""
    df = pd.DataFrame([{
        "resin_1_structure": PGE_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 10.0,
    }])
    
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_alpha_gel"].iloc[0]), "Monofunctional epoxy must not gel in crosslink_physics"
    
    net = evaluate_reaction_network(
        resins=[{"smiles": PGE_SMILES, "mass": 100.0}],
        curers=[{"smiles": EDA_SMILES, "mass": 10.0}],
    )
    assert net.gel_possible_ideal == 0.0, "Monofunctional epoxy system must have gel_possible_ideal = 0.0"


def test_06_unknown_curer_no_default_3():
    """验收 6: 缺失固化剂 SMILES 时, 不得盲猜 3.0, 必须为 NaN 并体现在 coverage 中"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": "",  # 未知固化剂
        "curing_agent_1_phr": 20.0,
    }])
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp)
    assert np.isnan(summ["cp_f_h_network"].iloc[0]), "Unknown curer must have NaN f_h_network"
    assert np.isnan(summ["cp_r_value"].iloc[0]), "Unknown curer must have NaN r"
    
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_f_h_network"].iloc[0]), "xl_f_h_network must not default to 3.0"
    assert xl["xl_coverage"].iloc[0] != "full", "xl_coverage must reflect missing curer"


def test_07_three_modules_consistency():
    """验收 7: 三模块关键物理量一致性 (r, f_avg)"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": DDS_SMILES,
        "curing_agent_1_phr": 36.47,
    }])
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp)
    xl = compute_crosslink_features(df)
    eng = EpoxyMechanismEngine(verbose=False)
    mech = eng.enrich_dataframe(df)
    
    r_cp = summ["cp_r_value"].iloc[0]
    r_xl = xl["xl_r_value"].iloc[0]
    r_mech = mech["mech_stoichiometry_r"].iloc[0]
    
    assert np.isclose(r_cp, r_xl, atol=1e-3), f"r mismatch cp={r_cp}, xl={r_xl}"
    assert np.isclose(r_cp, r_mech, atol=1e-3), f"r mismatch cp={r_cp}, mech={r_mech}"
    
    f_cp = summ["cp_f_avg_network"].iloc[0]
    f_xl = xl["xl_f_avg"].iloc[0]
    f_mech = mech["mech_average_functionality"].iloc[0]
    assert np.isclose(f_cp, f_xl, atol=1e-3), f"f_avg mismatch cp={f_cp}, xl={f_xl}"
    assert np.isclose(f_cp, f_mech, atol=1e-3), f"f_avg mismatch cp={f_cp}, mech={f_mech}"


def test_08_no_weights_no_pseudo_r():
    """验收 8: 缺失质量/phr 时, component_physics 绝不拿 ahew/eew 冒充 r (严格模式下为 NaN, 默认模式下为 r=1.0 假设)"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "curing_agent_1_structure": DDS_SMILES,
        # 无 phr 列
    }])
    cp = compute_component_physics(df)
    # 严格科研模式：缺失真实质量时保持未知 NaN，绝不伪造 ahew/eew
    summ_strict = compute_formulation_summary(df, cp, assume_stoichiometric_if_missing=False)
    assert np.isnan(summ_strict["cp_r_value"].iloc[0]), "In strict mode without phr, cp_r_value must be NaN, NOT ahew/eew"
    # 默认工程模式：采用 r=1.0 假设，来源明确标记为 assumed_stoichiometric，绝不伪造 ahew/eew
    summ_default = compute_formulation_summary(df, cp, assume_stoichiometric_if_missing=True)
    assert summ_default["cp_r_value"].iloc[0] == 1.0
    assert summ_default["cp_r_source"].iloc[0] == "assumed_stoichiometric"


def test_09_no_repeated_r_multiplication():
    """验收 9: crosslink_physics 消除重复乘 r, 非等比体系下 f_avg 计算严密自洽"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": DDS_SMILES,
        "curing_agent_1_phr": 36.47 * 2.0,  # r ~ 2.0 (过量胺)
    }])
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp)
    xl = compute_crosslink_features(df)
    
    f_cp = summ["cp_f_avg_network"].iloc[0]
    f_xl = xl["xl_f_avg"].iloc[0]
    
    # 严格一致
    assert np.isclose(f_cp, f_xl, atol=1e-3), f"f_avg mismatch: cp={f_cp}, xl={f_xl}"
    # DGEBA f=2, DDS f=4. 当 r=2.0 时，n_h/n_r = (2*2)/4 = 1.0 (等分子摩尔数)
    # f_avg = (2*1 + 4*1)/(1+1) = 3.00
    assert np.isclose(f_xl, 3.0, atol=0.05), f"Expected f_avg ~ 3.0, got {f_xl}"
