# -*- coding: utf-8 -*-
"""
test_acceptance_12_criteria.py

CFRP 反应网络与交联物理特征提取——12项端到端验收测试标准：
全面覆盖 HANDOVER_ROUND2_ACC_12.md 定义的 12 项场景：
1. DGEBA + EDA (100g : 8.83g) => r 约 1.0, 来源 calculated_from_mass, rxn_r_assumed = 0.0
2. DGEBA + EDA (100g : 17.65g) => r 约 2.0 (胺过量), 不被 r=1 覆盖, 来源 calculated_from_mass
3. 仅 SMILES 无用量 (DGEBA + EDA) => r = 1.0, 来源 assumed_stoichiometric, rxn_r_assumed = 1.0
4. 单侧用量缺失 (树脂 100g, 固化剂缺失) => r = 1.0, 来源 assumed_stoichiometric, 禁止计算出 0 或 50
5. 文献明确指定 r = 0.5 => 保留 r = 0.5, 来源 literature, 不强行配平为 1.0
6. 单官能环氧 + 四官能胺 (r=1.0) => 理想网络不可凝胶: gel_possible_ideal == 0.0, Mc_ideal == NaN
7. 酸酐固化剂 (MTHPA) => AHEW 采用 f_stoich = 1.0; gel_assumption == 'mechanism_approximate'
8. 多官能度混合环氧体系 => 凝胶依据分支因子发散判据 kappa_A * kappa_B * min(r, 1/r) >= 1.0, 不强求 f_avg > 2.0
9. 有效配方混入未知组分 => 明确记录失配/不支持状态 (composition_assumption 或 coverage 标记)
10. 相同样本跨四个模块计算 => reaction_core, component_physics, crosslink_physics, epoxy_mechanism 对 r, f_avg, 凝胶特征严格一致
11. 配方质量整体乘 10 (等比放大) => 归一化特征 (r, f_avg, 临界转化率, 理论交联特征) 数值完全不变
12. 自定义 DataFrame 索引 (非默认 RangeIndex) => 输出行数与 index 完全保真, 不重置不乱序
"""

import math
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
    R_SRC_INVALID_CHEM,
    COMP_ASSUMPTION_NONE,
    COMP_ASSUMPTION_EQUIMOLAR,
    COMP_ASSUMPTION_RESIN_ONLY,
    GEL_ASSUMPTION_IDEAL,
    GEL_ASSUMPTION_MECHANISM_APPROX,
)
from core.component_physics import compute_component_physics
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import extract_epoxy_mechanism_features

SMI_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
SMI_EDA = "NCCN"
SMI_PGE = "c1ccccc1OCC2CO2"
SMI_MTHPA = "CC1=CCC2C(=O)OC(=O)C2C1"
SMI_TGDDM = "C1OC1CN(c2ccc(Cc3ccc(N(CC4CO4)CC5CO5)cc3)cc2)CC6CO6"


def test_criterion_1_stoichiometric_known_mass():
    resins = [{"smiles": SMI_DGEBA, "mass": 100.0}]
    curers = [{"smiles": SMI_EDA, "mass": 8.827}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert pytest.approx(res.r_value, abs=1e-2) == 1.0
    assert res.r_source == R_SRC_CALCULATED
    assert res.rxn_r_assumed == 0.0
    assert res.composition_assumption == COMP_ASSUMPTION_NONE
    assert res.gel_possible_ideal == 1.0


def test_criterion_2_excess_amine_known_mass():
    resins = [{"smiles": SMI_DGEBA, "mass": 100.0}]
    curers = [{"smiles": SMI_EDA, "mass": 17.654}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert pytest.approx(res.r_value, abs=1e-2) == 2.0
    assert res.r_source == R_SRC_CALCULATED
    assert res.rxn_r_assumed == 0.0
    assert res.balance == pytest.approx(0.5, abs=1e-2)


def test_criterion_3_missing_all_masses():
    resins = [{"smiles": SMI_DGEBA}]
    curers = [{"smiles": SMI_EDA}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert res.r_value == 1.0
    assert res.r_source == R_SRC_ASSUMED_STOICH
    assert res.rxn_r_assumed == 1.0
    assert res.composition_assumption == COMP_ASSUMPTION_EQUIMOLAR
    assert res.gel_possible_ideal == 1.0
    assert np.isfinite(res.Mn) and res.Mn > 0
    assert np.isfinite(res.f_avg) and res.f_avg > 0


def test_criterion_4_one_sided_mass_missing():
    resins = [{"smiles": SMI_DGEBA, "mass": 100.0}]
    curers = [{"smiles": SMI_EDA}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert res.r_value == 1.0
    assert res.r_source == R_SRC_ASSUMED_STOICH
    assert res.rxn_r_assumed == 1.0
    assert res.composition_assumption == COMP_ASSUMPTION_RESIN_ONLY
    assert res.r_value != 0.0
    assert res.r_value < 10.0


def test_criterion_5_literature_explicit_r():
    resins = [{"smiles": SMI_DGEBA}]
    curers = [{"smiles": SMI_EDA}]
    res = evaluate_reaction_network(resins, curers, explicit_r=0.5)

    assert res.chemistry_valid is True
    assert res.r_value == 0.5
    assert res.r_source == R_SRC_LITERATURE
    assert res.rxn_r_assumed == 0.0
    assert res.balance == pytest.approx(0.5, abs=1e-4)


def test_criterion_6_monofunctional_epoxy_no_gel():
    resins = [{"smiles": SMI_PGE}]
    curers = [{"smiles": SMI_EDA}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert res.r_value == 1.0
    assert res.kappa_A == pytest.approx(0.0, abs=1e-5)
    assert res.gel_possible_ideal == 0.0
    assert np.isnan(res.Mc_ideal)
    assert res.nu_ideal_mol_m3 == 0.0


def test_criterion_7_anhydride_stoichiometry_and_assumption():
    resins = [{"smiles": SMI_DGEBA, "mass": 100.0}]
    curers = [{"smiles": SMI_MTHPA, "mass": 97.6}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is True
    assert res.f_h_stoich == pytest.approx(1.0, abs=1e-2)
    assert res.f_h_network == pytest.approx(2.0, abs=1e-2)
    assert pytest.approx(res.r_value, abs=1e-2) == 1.0
    assert res.gel_assumption == GEL_ASSUMPTION_MECHANISM_APPROX


def test_criterion_8_mixed_functionality_branch_divergence():
    resins = [{"smiles": SMI_TGDDM, "mass": 100.0}]
    curers = [{"smiles": SMI_EDA, "mass": 14.2}]
    res = evaluate_reaction_network(resins, curers)
    assert res.gel_possible_ideal == 1.0
    assert res.kappa_A * res.kappa_B * res.balance >= 1.0


def test_criterion_9_unknown_component_handling():
    resins = [{"smiles": SMI_DGEBA, "mass": 100.0}]
    curers = [{"smiles": "UNKNOWN_GARBAGE_STR", "mass": 10.0}]
    res = evaluate_reaction_network(resins, curers)

    assert res.chemistry_valid is False
    assert res.r_source == R_SRC_INVALID_CHEM
    assert np.isnan(res.r_value)
    assert res.gel_possible_ideal == 0.0 or np.isnan(res.gel_possible_ideal)


def test_criterion_10_consistency_across_modules():
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_smiles": SMI_EDA,
        "curing_agent_1_phr": 8.827,
    }], index=["sample_A"])

    res = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA, "mass": 100.0}],
        [{"smiles": SMI_EDA, "mass": 8.827}]
    )

    df_cp = compute_component_physics(df)
    df_xl = compute_crosslink_features(df, cp=df_cp)
    df_mech = extract_epoxy_mechanism_features(df)

    r_rc = res.r_value
    r_cp = df_cp.loc["sample_A", "cp_r_value"]
    r_xl = df_xl.loc["sample_A", "xl_r_value"]
    r_mech = df_mech.loc["sample_A", "mech_stoichiometry_r"]

    assert pytest.approx(r_rc, abs=1e-2) == 1.0
    assert pytest.approx(r_cp, abs=1e-2) == r_rc
    assert pytest.approx(r_xl, abs=1e-2) == r_rc
    assert pytest.approx(r_mech, abs=1e-2) == r_rc

    gel_rc = res.gel_possible_ideal
    gel_mech = df_mech.loc["sample_A", "mech_gel_possible_ideal"]
    assert gel_rc == 1.0
    assert gel_mech == 1.0


def test_criterion_11_scale_invariance():
    df1 = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_smiles": SMI_EDA,
        "curing_agent_1_phr": 8.827,
    }], index=["base"])

    df2 = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "resin_1_phr": 1000.0,
        "curing_agent_1_smiles": SMI_EDA,
        "curing_agent_1_phr": 88.27,
    }], index=["scaled_x10"])

    df_cp1 = compute_component_physics(df1)
    df_cp2 = compute_component_physics(df2)

    assert pytest.approx(df_cp1.loc["base", "cp_r_value"], abs=1e-4) == df_cp2.loc["scaled_x10", "cp_r_value"]
    assert pytest.approx(df_cp1.loc["base", "cp_f_avg_network"], abs=1e-4) == df_cp2.loc["scaled_x10", "cp_f_avg_network"]
    assert pytest.approx(df_cp1.loc["base", "cp_balance"], abs=1e-4) == df_cp2.loc["scaled_x10", "cp_balance"]

    df_mech1 = extract_epoxy_mechanism_features(df1)
    df_mech2 = extract_epoxy_mechanism_features(df2)
    assert pytest.approx(df_mech1.loc["base", "mech_theoretical_alpha_gel"], abs=1e-4) == df_mech2.loc["scaled_x10", "mech_theoretical_alpha_gel"]


def test_criterion_12_custom_index_preservation():
    custom_indices = ["CFRP_EXP_001", "CFRP_EXP_002", "CFRP_EXP_003"]
    df = pd.DataFrame([
        {"resin_1_smiles": SMI_DGEBA, "resin_1_phr": 100.0, "curing_agent_1_smiles": SMI_EDA, "curing_agent_1_phr": 8.83},
        {"resin_1_smiles": SMI_DGEBA, "resin_1_phr": 100.0, "curing_agent_1_smiles": SMI_EDA, "curing_agent_1_phr": 17.66},
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA},
    ], index=custom_indices)

    df_cp = compute_component_physics(df)
    assert list(df_cp.index) == custom_indices
    assert len(df_cp) == 3

    df_xl = compute_crosslink_features(df, cp=df_cp)
    assert list(df_xl.index) == custom_indices
    assert len(df_xl) == 3

    df_mech = extract_epoxy_mechanism_features(df)
    assert list(df_mech.index) == custom_indices
    assert len(df_mech) == 3
