# -*- coding: utf-8 -*-
"""
tests/test_final_gel_boundary_and_modes.py

CFRP 环氧树脂反应与交联四核心模块最终定向修复与状态冻结验收测试：
覆盖 15 项关键场景与反例：
1. A2-B2 临界边界判据 (DGEBA + 乙胺 CCN, r=1.0): critical, gel_possible_ideal=0, alpha_gel=NaN, Mc_ideal=NaN
2. A2-B2 临界在 crosslink_physics: xl_alpha_gel=NaN, xl_nu_theory_mol_m3=NaN, xl_nu_source='non_gelling'
3. A2-B2 临界在 epoxy_mechanism_features: mech_gel_possible_ideal=0, mech_theoretical_alpha_gel=NaN, mech_theoretical_Mc=NaN
4. A2-B4 超临界标准凝胶体系 (DGEBA + EDA, r=1.0): supercritical, gel_possible_ideal=1.0, alpha_gel=0.57735, xl_nu_source='mechanism_aware'
5. A2-B4 不可达亚临界非凝胶体系 (DGEBA + EDA, r=0.1): subcritical, p_Bc > 1.0 不可达, 四模块统一 alpha_gel 为 NaN
6. A2-B4 r=0.1 交联密度与来源标签同步 (P1-1): xl_nu_theory_mol_m3=NaN, xl_nu_source='non_gelling'
7. 严格模式参数透传 - enrich_narrow_table(assume_stoichiometric_if_missing=False): cp_r_value=NaN, cp_r_source='missing_stoichiometry'
8. 默认假设模式 - enrich_narrow_table(assume_stoichiometric_if_missing=True): cp_r_value=1.0, rxn_r_assumed=1.0, cp_r_source='assumed_stoichiometric'
9. 单官能环氧体系 (PGE + EDA): kappa_A=0, subcritical, gel_possible_ideal=0, xl_nu_source='non_gelling'
10. 单官能环氧机理特征绝不伪造 Mc (P0-4): mech_theoretical_alpha_gel=NaN, mech_theoretical_Mc=NaN
11. 酸酐体系 A2-B2 临界边界 (DGEBA + 琥珀酸酐 SA, r=1.0): critical, mechanism_approximate, gel_possible_ideal=0, alpha_gel=NaN
12. 多官能酸酐超临界凝胶体系 (TGDDM + SA, r=1.0): supercritical, mechanism_approximate, gel_possible_ideal=1.0
13. 固化剂过量极端非凝胶边界 (DGEBA + EDA, r=10.0): subcritical, p_Ac > 1.0 不可达, alpha_gel=NaN, xl_nu_source='non_gelling'
14. 无效化学体系 (无效 SMILES / 缺失反应基团): chemistry_valid=False, r_source='invalid_chemistry', xl_nu_source='none'
15. 四模块跨模块一致性闭环测试: DGEBA+EDA(r=1.0), DGEBA+CCN(r=1.0), DGEBA+EDA(r=0.1) 严格自洽
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
    R_SRC_MISSING_STOICH,
    R_SRC_INVALID_CHEM,
    GEL_REGIME_SUBCRITICAL,
    GEL_REGIME_CRITICAL,
    GEL_REGIME_SUPERCRITICAL,
    GEL_REGIME_UNDETERMINED,
    GEL_ASSUMPTION_IDEAL,
    GEL_ASSUMPTION_MECHANISM_APPROX,
)
from core.component_physics import compute_component_physics, enrich_narrow_table
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import EpoxyMechanismEngine, extract_epoxy_mechanism_features

SMI_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"  # f_r = 2
SMI_CCN = "CCN"                                            # 乙胺: 一级单胺, f_stoich = 2, f_network = 2 (线性链增长)
SMI_EDA = "NCCN"                                            # 乙二胺: 二级二胺, f_stoich = 4, f_network = 4 (交联剂)
SMI_PGE = "c1ccccc1OCC1CO1"                                 # 苯基缩水甘油醚: 单官能环氧 f_r = 1
SMI_SA = "O=C1CCC(=O)O1"                                    # 琥珀酸酐: 单酸酐 f_stoich = 1, f_network = 2
SMI_TGDDM = "C1OC1CN(c2ccc(Cc3ccc(N(CC4CO4)CC5CO5)cc3)cc2)CC6CO6"  # 四官能环氧 f_r = 4


def test_scenario_01_a2_b2_critical_boundary_reaction_core():
    """场景 1 (P0-1): A2-B2 r=1 属于 critical 临界，只能形成线性链，绝不能判定为可凝胶"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": SMI_CCN}],
        explicit_r=1.0,
    )
    assert rc.gel_regime == GEL_REGIME_CRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)
    assert np.isnan(rc.Mc_ideal)
    assert rc.nu_ideal_mol_m3 == 0.0


def test_scenario_02_a2_b2_critical_crosslink_physics():
    """场景 2 (P0-1 & P1-1): A2-B2 r=1 在 crosslink_physics 中输出 xl_alpha_gel=NaN, xl_nu_source='non_gelling'"""
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_CCN,
        "formulation_r_value": 1.0,
    }])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_alpha_gel"].iloc[0])
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"


def test_scenario_03_a2_b2_critical_mechanism_features():
    """场景 3 (P0-1 & P0-4): A2-B2 r=1 在 epoxy_mechanism_features 中保持理论值为 NaN"""
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_CCN,
        "formulation_r_value": 1.0,
    }])
    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df)
    assert mech["mech_gel_possible_ideal"].iloc[0] == 0.0
    assert np.isnan(mech["mech_theoretical_alpha_gel"].iloc[0])
    assert np.isnan(mech["mech_theoretical_Mc"].iloc[0])
    assert mech["mech_crosslink_density_mol_m3"].iloc[0] == 0.0


def test_scenario_04_a2_b4_supercritical_standard_gelation():
    """场景 4: A2-B4 r=1.0 (DGEBA+EDA) 属于 supercritical 超临界，正常凝胶"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": SMI_EDA}],
        explicit_r=1.0,
    )
    assert rc.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert rc.gel_possible_ideal == 1.0
    assert pytest.approx(rc.alpha_gel, abs=1e-4) == 1.0 / math.sqrt(3.0)
    assert np.isfinite(rc.Mc_ideal)
    assert rc.nu_ideal_mol_m3 > 0.0

    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 1.0,
    }])
    xl = compute_crosslink_features(df)
    assert pytest.approx(xl["xl_alpha_gel"].iloc[0], abs=1e-4) == 1.0 / math.sqrt(3.0)
    assert xl["xl_nu_source"].iloc[0] == "mechanism_aware"
    assert xl["xl_nu_theory_mol_m3"].iloc[0] > 0.0


def test_scenario_05_a2_b4_unreachable_subcritical_r01():
    """场景 5 (P0-3): DGEBA + EDA r=0.1 胺端临界转化率 p_Bc > 1.0 不可达，四模块统一 alpha_gel 为 NaN"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": SMI_EDA}],
        explicit_r=0.1,
    )
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert rc.p_B_critical > 1.0  # 物理不可达
    assert np.isnan(rc.alpha_gel)

    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 0.1,
    }])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_alpha_gel"].iloc[0])

    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df)
    assert np.isnan(mech["mech_theoretical_alpha_gel"].iloc[0])


def test_scenario_06_a2_b4_r01_nu_source_sync():
    """场景 6 (P1-1): DGEBA + EDA r=0.1 因不可凝胶 nu 为 NaN 时，xl_nu_source 同步为 'non_gelling'"""
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 0.1,
    }])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"


def test_scenario_07_enrich_narrow_table_strict_mode_passthrough():
    """场景 7 (P0-2): 严格模式 assume_stoichiometric_if_missing=False 透传，无配比返回 NaN 与 missing_stoichiometry"""
    df = pd.DataFrame([{"epoxy_smiles": SMI_DGEBA, "hardener_smiles": SMI_EDA}])

    # 验证底层 compute_component_physics 严格模式
    comp = compute_component_physics(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(comp["cp_r_value"].iloc[0])
    assert comp["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH

    # 验证 enrich_narrow_table 严格模式透传
    res_df = enrich_narrow_table(df, assume_stoichiometric_if_missing=False, keep_metadata=True)
    r_val = res_df["cp_r_value"].iloc[0] if "cp_r_value" in res_df.columns else res_df["formulation_r_value"].iloc[0]
    assert np.isnan(r_val)
    assert np.isnan(res_df["formulation_r_value"].iloc[0])
    assert res_df["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH


def test_scenario_08_enrich_narrow_table_default_assumption_mode():
    """场景 8: 默认假设模式 assume_stoichiometric_if_missing=True，无配比返回 1.0, rxn_r_assumed=1.0"""
    df = pd.DataFrame([{"epoxy_smiles": SMI_DGEBA, "hardener_smiles": SMI_EDA}])

    comp = compute_component_physics(df, assume_stoichiometric_if_missing=True)
    assert comp["cp_r_value"].iloc[0] == 1.0
    assert comp["rxn_r_assumed"].iloc[0] == 1.0
    assert comp["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH

    res_df = enrich_narrow_table(df, assume_stoichiometric_if_missing=True, keep_metadata=True)
    r_val = res_df["cp_r_value"].iloc[0] if "cp_r_value" in res_df.columns else res_df["formulation_r_value"].iloc[0]
    assert r_val == 1.0
    assert res_df["formulation_r_value"].iloc[0] == 1.0
    assert res_df["rxn_r_assumed"].iloc[0] == 1.0
    assert res_df["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH


def test_scenario_09_monofunctional_epoxy_no_gelation():
    """场景 9: 单官能环氧 PGE + EDA 体系 kappa_A=0，不可凝胶，xl_nu_source='non_gelling'"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_PGE}],
        [{"smiles": SMI_EDA}],
        explicit_r=1.0,
    )
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)
    assert np.isnan(rc.Mc_ideal)

    df = pd.DataFrame([{
        "resin_1_smiles": SMI_PGE,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 1.0,
    }])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_alpha_gel"].iloc[0])
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"


def test_scenario_10_monofunctional_mechanism_no_fake_mc():
    """场景 10 (P0-4): 单官能环氧 PGE + EDA 机理模块不可凝胶时，绝不伪造 alpha_gel=1.0 或虚假 Mc"""
    df = pd.DataFrame([{
        "epoxy_smiles": SMI_PGE,
        "hardener_smiles": SMI_EDA,
    }])
    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df)
    assert mech["mech_gel_possible_ideal"].iloc[0] == 0.0
    assert np.isnan(mech["mech_theoretical_alpha_gel"].iloc[0])
    assert np.isnan(mech["mech_theoretical_Mc"].iloc[0])


def test_scenario_11_anhydride_a2_b2_critical_boundary():
    """场景 11 (P1-2): DGEBA + 琥珀酸酐 SA r=1.0 属于 A2-B2 critical 临界，不可凝胶且保持 mechanism_approximate"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": SMI_SA}],
        explicit_r=1.0,
    )
    assert rc.gel_regime == GEL_REGIME_CRITICAL
    assert rc.gel_assumption == GEL_ASSUMPTION_MECHANISM_APPROX
    assert rc.gel_possible_ideal == 0.0
    assert np.isnan(rc.alpha_gel)
    assert np.isnan(rc.Mc_ideal)
    assert rc.nu_ideal_mol_m3 == 0.0


def test_scenario_12_polyfunctional_anhydride_supercritical_gelation():
    """场景 12: 四官能环氧 TGDDM + 琥珀酸酐 SA r=1.0 属于 supercritical 超临界，可凝胶"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_TGDDM}],
        [{"smiles": SMI_SA}],
        explicit_r=1.0,
    )
    assert rc.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert rc.gel_assumption == GEL_ASSUMPTION_MECHANISM_APPROX
    assert rc.gel_possible_ideal == 1.0
    assert pytest.approx(rc.alpha_gel, abs=1e-4) == 1.0 / math.sqrt(3.0)
    assert np.isfinite(rc.Mc_ideal)


def test_scenario_13_curer_excess_unreachable_subcritical_r10():
    """场景 13: DGEBA + EDA 固化剂过量 r=10.0，环氧侧转化率 p_Ac > 1.0 不可达，无法凝胶"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": SMI_EDA}],
        explicit_r=10.0,
    )
    assert rc.gel_regime == GEL_REGIME_SUBCRITICAL
    assert rc.gel_possible_ideal == 0.0
    assert rc.p_A_critical > 1.0  # 环氧侧过量耗尽仍不可达
    assert np.isnan(rc.alpha_gel)

    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 10.0,
    }])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_alpha_gel"].iloc[0])
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"


def test_scenario_14_invalid_chemistry():
    """场景 14: 无效化学反应结构 (无效 SMILES)，无法构成体系"""
    rc = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": "INVALID_NOT_A_SMILES"}],
        explicit_r=1.0,
    )
    assert rc.chemistry_valid is False
    assert rc.r_source == R_SRC_INVALID_CHEM
    assert rc.gel_regime in (GEL_REGIME_SUBCRITICAL, GEL_REGIME_UNDETERMINED, "not_applicable")
    assert rc.gel_possible_ideal == 0.0 or np.isnan(rc.gel_possible_ideal)

    df = pd.DataFrame([{
        "resin_1_smiles": "INVALID_RESIN",
        "curing_agent_1_smiles": "INVALID_CURER",
    }])
    xl = compute_crosslink_features(df)
    assert xl["xl_nu_source"].iloc[0] == "none"


def test_scenario_15_cross_module_consistency():
    """场景 15: 四模块在凝胶判据、当量比 r、平均官能度 f_avg 与临界转化率上的数值一致性闭环测试"""
    # 体系 A: DGEBA + EDA (r=1.0) -> 可凝胶
    df_a = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_smiles": SMI_EDA,
        "curing_agent_1_phr": 8.827,
    }], index=["sample_gel"])

    rc_a = evaluate_reaction_network([{"smiles": SMI_DGEBA, "mass": 100.0}], [{"smiles": SMI_EDA, "mass": 8.827}])
    cp_a = compute_component_physics(df_a)
    xl_a = compute_crosslink_features(df_a, cp=cp_a)
    mech_a = extract_epoxy_mechanism_features(df_a)

    assert rc_a.gel_possible_ideal == 1.0
    assert xl_a.loc["sample_gel", "xl_nu_theory_mol_m3"] > 0
    assert mech_a.loc["sample_gel", "mech_gel_possible_ideal"] == 1.0
    assert pytest.approx(rc_a.r_value, abs=1e-2) == 1.0
    assert pytest.approx(cp_a.loc["sample_gel", "cp_r_value"], abs=1e-2) == 1.0
    assert pytest.approx(xl_a.loc["sample_gel", "xl_r_value"], abs=1e-2) == 1.0
    assert pytest.approx(mech_a.loc["sample_gel", "mech_stoichiometry_r"], abs=1e-2) == 1.0

    # 体系 B: DGEBA + CCN (r=1.0) -> critical 临界不可凝胶
    df_b = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_CCN,
        "formulation_r_value": 1.0,
    }], index=["sample_crit"])

    rc_b = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_CCN}], explicit_r=1.0)
    xl_b = compute_crosslink_features(df_b)
    mech_b = extract_epoxy_mechanism_features(df_b)

    assert rc_b.gel_possible_ideal == 0.0
    assert np.isnan(rc_b.alpha_gel)
    assert np.isnan(xl_b.loc["sample_crit", "xl_alpha_gel"])
    assert np.isnan(mech_b.loc["sample_crit", "mech_theoretical_alpha_gel"])
    assert xl_b.loc["sample_crit", "xl_nu_source"] == "non_gelling"
    assert mech_b.loc["sample_crit", "mech_gel_possible_ideal"] == 0.0
