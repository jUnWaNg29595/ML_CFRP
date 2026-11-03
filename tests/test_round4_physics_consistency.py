"""CFRP 系统第四轮物理一致性与高缺失率兼容定向回归测试集。
覆盖审查报告全部 16 个核心验收场景：
1. enrich_narrow_table() 单行正常执行。
2. 重复执行特征增强后，无重复列。
3. 多组分部分用量缺失时使用假设 r=1。
4. 完全无配比时四模块默认 r、平均官能度一致。
5. 显式关闭假设模式时，行为符合严格模式。
6. r=0.1 不可凝胶时，不得产生正的理想有效网络密度。
7. r=0.5、1、2 的凝胶判断正确。
8. 文献 r 与质量计算 r 冲突时，来源与口径清晰。
9. 酸酐当量重正确。
10. 酸酐机制标签及近似特征口径一致。
11. 单官能环氧不凝胶。
12. 未知反应组分不被默默伪造成有效网络。
13. 自定义索引完全保留。
14. 质量整体乘 10 后，归一化特征不变。
15. 含多行数据且每行缺失模式不同，不发生跨行污染或错位。
16. 传入已计算 cp 与自动计算 cp，两条路径结果一致。
"""

import numpy as np
import pandas as pd
import pytest

from core.reaction_core import evaluate_reaction_network
from core.component_physics import (
    compute_component_physics,
    compute_formulation_summary,
    enrich_narrow_table,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import extract_epoxy_mechanism_features, EpoxyMechanismEngine

DGEBA_SMILES = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
EDA_SMILES = "NCCN"
PGE_SMILES = "c1ccccc1OCC2CO2"  # 单官能环氧
SA_SMILES = "O=C1CCC(=O)O1"     # 琥珀酸酐


def test_01_enrich_narrow_table_single_row():
    """1. enrich_narrow_table() 单行正常执行无崩溃、无 TypeError"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 8.83,
    }])
    out = enrich_narrow_table(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert out.columns.is_unique


def test_02_repeated_enrich_no_duplicate_columns():
    """2. 重复执行特征增强后，无重复列，无 _x/_y 后缀"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 8.83,
    }])
    out1 = enrich_narrow_table(df)
    out2 = enrich_narrow_table(out1)
    assert out2.columns.is_unique
    assert len(out2.columns) == len(out1.columns)
    for col in out2.columns:
        assert not col.endswith("_x") and not col.endswith("_y")


def test_03_multicomponent_partial_missing_mass():
    """3. 多组分部分用量缺失时使用假设 r=1，绝不能标记为 calculated_from_mass"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "resin_2_structure": PGE_SMILES,
        # resin_2_phr 缺失!
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 8.83,
    }])
    cp = compute_component_physics(df)
    summ = compute_formulation_summary(df, cp, assume_stoichiometric_if_missing=True)
    assert summ["cp_r_value"].iloc[0] == 1.0
    assert summ["cp_r_source"].iloc[0] == "assumed_stoichiometric"
    assert summ["rxn_r_assumed"].iloc[0] == 1.0
    assert summ["cp_coverage"].iloc[0] != "full"


def test_04_all_missing_four_modules_consistent():
    """4. 完全无配比时四模块默认 r=1.0、平均官能度 f_avg≈2.667 一致"""
    df = pd.DataFrame([{"resin_1_smiles": DGEBA_SMILES, "curing_agent_1_smiles": EDA_SMILES}])
    rc = evaluate_reaction_network([{"smiles": DGEBA_SMILES}], [{"smiles": EDA_SMILES}])
    cp = compute_formulation_summary(df)
    xl = compute_crosslink_features(df)
    mech = extract_epoxy_mechanism_features(df)

    assert abs(rc.r_value - 1.0) < 1e-4
    assert abs(cp["cp_r_value"].iloc[0] - 1.0) < 1e-4
    assert abs(xl["xl_r_value"].iloc[0] - 1.0) < 1e-4
    assert abs(mech["mech_stoichiometry_r"].iloc[0] - 1.0) < 1e-4

    assert abs(rc.f_avg - 2.6667) < 0.05
    assert abs(cp["cp_f_avg_network"].iloc[0] - 2.6667) < 0.05
    assert abs(xl["xl_f_avg"].iloc[0] - 2.6667) < 0.05
    assert abs(mech["mech_average_functionality"].iloc[0] - 2.6667) < 0.05


def test_05_strict_mode_when_disabled():
    """5. 显式关闭假设模式时，行为符合严格模式，缺失处保持 NaN"""
    df = pd.DataFrame([{"resin_1_smiles": DGEBA_SMILES, "curing_agent_1_smiles": EDA_SMILES}])
    summ = compute_formulation_summary(df, assume_stoichiometric_if_missing=False)
    assert pd.isna(summ["cp_r_value"].iloc[0])
    assert pd.isna(summ["cp_f_avg_network"].iloc[0])
    assert summ["cp_r_source"].iloc[0] in ("invalid_chemistry", "missing_stoichiometry")


def test_06_non_gelling_r01_no_positive_nu():
    """6. r=0.1 不可凝胶时，不得产生正的理想有效网络密度"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "curing_agent_1_structure": EDA_SMILES,
        "formulation_r_value": 0.1,
    }])
    rc = evaluate_reaction_network([{"smiles": DGEBA_SMILES}], [{"smiles": EDA_SMILES}], explicit_r=0.1)
    xl = compute_crosslink_features(df)

    assert rc.gel_possible_ideal == 0.0
    assert pd.isna(xl["xl_alpha_gel"].iloc[0])
    assert pd.isna(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_07_gelling_cases_r_05_1_2():
    """7. r=0.5、1.0、2.0 凝胶判断正确"""
    for r_val in [0.5, 1.0, 2.0]:
        df = pd.DataFrame([{
            "resin_1_structure": DGEBA_SMILES,
            "curing_agent_1_structure": EDA_SMILES,
            "formulation_r_value": r_val,
        }])
        rc = evaluate_reaction_network([{"smiles": DGEBA_SMILES}], [{"smiles": EDA_SMILES}], explicit_r=r_val)
        xl = compute_crosslink_features(df)
        assert rc.gel_possible_ideal == 1.0
        assert np.isfinite(xl["xl_alpha_gel"].iloc[0])
        assert xl["xl_nu_theory_mol_m3"].iloc[0] > 0


def test_08_lit_vs_mass_conflict_handling():
    """8. 文献 r 与质量计算 r 冲突时，优先保留文献 r，记录推算 r 与冲突标记"""
    df = pd.DataFrame([{
        "resin_1_structure": DGEBA_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 17.65,
        "formulation_r_value": 0.5,
    }])
    cp = compute_formulation_summary(df)
    assert cp["cp_r_value"].iloc[0] == 0.5
    assert cp["cp_r_source"].iloc[0] == "literature"
    assert abs(cp["cp_r_from_mass"].iloc[0] - 2.0) < 0.1
    assert cp["cp_r_conflict"].iloc[0] == 1.0


def test_09_anhydride_equivalent_weight():
    """9. 琥珀酸酐当量重正确 (MW=AHEW≈100.073 g/eq)"""
    eng = EpoxyMechanismEngine(verbose=False)
    sa_props = eng.calc_single_molecule_properties(SA_SMILES, is_resin=False)
    assert abs(sa_props["ew"] - 100.073) < 0.1
    assert sa_props["f_stoich"] == 1.0
    assert sa_props["functionality"] == 2.0


def test_10_anhydride_mechanism_tag_consistent():
    """10. 酸酐机制标签及近似特征口径一致"""
    df = pd.DataFrame([{"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": SA_SMILES}])
    rc = evaluate_reaction_network([{"smiles": DGEBA_SMILES}], [{"smiles": SA_SMILES}])
    cp = compute_formulation_summary(df)
    xl = compute_crosslink_features(df)
    assert rc.gel_assumption == "mechanism_approximate"
    assert cp["cp_mechanism"].iloc[0] == "anhydride"
    assert xl["xl_mechanism"].iloc[0] == "anhydride"


def test_11_monofunctional_epoxy_no_gel():
    """11. 单官能环氧 (PGE) + 多官能胺不可凝胶，无正有效交联密度"""
    df = pd.DataFrame([{
        "resin_1_structure": PGE_SMILES,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": EDA_SMILES,
        "curing_agent_1_phr": 10.0,
    }])
    rc = evaluate_reaction_network([{"smiles": PGE_SMILES, "mass": 100.0}], [{"smiles": EDA_SMILES, "mass": 10.0}])
    xl = compute_crosslink_features(df)
    assert rc.gel_possible_ideal == 0.0
    assert pd.isna(rc.Mc_ideal)
    assert pd.isna(xl["xl_alpha_gel"].iloc[0])
    assert pd.isna(xl["xl_nu_theory_mol_m3"].iloc[0])


def test_12_unknown_curer_handling():
    """12. 未知反应组分不被默默伪造成有效网络"""
    rc = evaluate_reaction_network([{"smiles": DGEBA_SMILES, "mass": 100.0}], [{"smiles": "UNKNOWN_CURER", "mass": 10.0}])
    assert rc.chemistry_valid is False
    assert rc.r_source == "invalid_chemistry"
    assert pd.isna(rc.r_value)


def test_13_custom_index_preserved():
    """13. 自定义索引完全保留"""
    custom_idx = ["Sample_A", "Sample_B"]
    df = pd.DataFrame([
        {"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": EDA_SMILES},
        {"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": EDA_SMILES, "formulation_r_value": 0.8},
    ], index=custom_idx)
    cp = compute_formulation_summary(df)
    xl = compute_crosslink_features(df)
    assert list(cp.index) == custom_idx
    assert list(xl.index) == custom_idx


def test_14_scale_invariance_mass_x10():
    """14. 质量整体乘 10 后，归一化特征不变"""
    df1 = pd.DataFrame([{"resin_1_structure": DGEBA_SMILES, "resin_1_phr": 100.0, "curing_agent_1_structure": EDA_SMILES, "curing_agent_1_phr": 8.83}])
    df2 = pd.DataFrame([{"resin_1_structure": DGEBA_SMILES, "resin_1_phr": 1000.0, "curing_agent_1_structure": EDA_SMILES, "curing_agent_1_phr": 88.3}])
    cp1 = compute_formulation_summary(df1)
    cp2 = compute_formulation_summary(df2)
    assert abs(cp1["cp_r_value"].iloc[0] - cp2["cp_r_value"].iloc[0]) < 1e-4
    assert abs(cp1["cp_f_avg_network"].iloc[0] - cp2["cp_f_avg_network"].iloc[0]) < 1e-4


def test_15_mixed_missing_rows_no_cross_contamination():
    """15. 含多行数据且每行缺失模式不同，不发生跨行污染或错位"""
    df = pd.DataFrame([
        {"resin_1_structure": DGEBA_SMILES, "resin_1_phr": 100.0, "curing_agent_1_structure": EDA_SMILES, "curing_agent_1_phr": 8.83}, # 完整质量
        {"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": EDA_SMILES}, # 完全缺失配比
        {"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": EDA_SMILES, "formulation_r_value": 0.7}, # 文献 r
    ])
    summ = compute_formulation_summary(df)
    assert summ["cp_r_source"].iloc[0] == "calculated_from_mass"
    assert summ["cp_r_source"].iloc[1] == "assumed_stoichiometric"
    assert summ["cp_r_source"].iloc[2] == "literature"
    assert abs(summ["cp_r_value"].iloc[2] - 0.7) < 1e-4


def test_16_passed_cp_vs_auto_calculated_cp():
    """16. 传入已计算 cp 与自动计算 cp，两条路径结果一致"""
    df = pd.DataFrame([{"resin_1_structure": DGEBA_SMILES, "curing_agent_1_structure": EDA_SMILES}])
    comp = compute_component_physics(df)
    xl_passed = compute_crosslink_features(df, cp=comp)
    xl_auto = compute_crosslink_features(df)
    assert abs(xl_passed["xl_f_avg"].iloc[0] - xl_auto["xl_f_avg"].iloc[0]) < 1e-4
    assert abs(xl_passed["xl_r_value"].iloc[0] - xl_auto["xl_r_value"].iloc[0]) < 1e-4
