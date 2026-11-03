# -*- coding: utf-8 -*-
"""
tests/test_reaction_features_v1_freeze.py

CFRP 系统 reaction_features_v1 状态冻结前最终回归验证测试套件：
全面覆盖任务要求定义的 22 项严密测试准则：
1. 默认模式 r=1 (assumed_stoichiometric, rxn_r_assumed=1.0)
2. 严格模式 r=NaN (missing_stoichiometry, rxn_r_assumed=0.0)
3. 严格模式下 xl_alpha_gel 与 xl_nu_theory_mol_m3 不得隐藏使用 r=1
4. 已计算 cp 与自动计算 cp 两种路径计算结果 100% 一致
5. 假设 r=1 经 CSV/Parquet 序列化保存并重载后来源不变
6. 假设值再次解析后不能升级为 literature
7. 严格模式重新解析包含历史假设的数据时，不得误认为原始文献 r，必须识别为缺失
8. 真实文献 r=0.5 的来源跨保存与多次解析始终保持为 literature
9. 完整质量计算 r 的来源跨保存与多次解析始终保持为 calculated_from_mass，不得升级为 literature
10. mw 输入包含 None, np.nan, pd.NA, '', 'nan', 0, -1 等异常值时不崩溃且能自愈补齐
11. f_network, f_stoich 包含 None, np.nan 等缺失值时不崩溃且能自愈补齐
12. 合法 SMILES 能自动补齐有效缺失的物理参数 (MW, EEW, AHEW, 官能度)
13. 无效 SMILES 不得伪造官能度，输出 invalid_chemistry
14. A2–B2 临界边界保持正确 (critical, gel_possible_ideal=0.0, alpha_gel=NaN)
15. A2–B4 (r=1.0) 超临界凝胶判断保持正确 (supercritical, gel_possible_ideal=1.0)
16. A2–B4 (r=0.1) 亚临界不可达凝胶判断保持正确 (subcritical, non_gelling)
17. 单官能环氧体系保持不可凝胶 (non_gelling, nu=NaN, Mc=NaN)
18. 酸酐固化剂当量重计算保持 f_stoich=1.0 与 f_network=2.0
19. 自定义 DataFrame 索引在所有模块中完全保持
20. 多行缺失模式不同的 DataFrame 不发生行间交叉污染
21. 重复执行特征增强操作 (enrich_narrow_table) 无重复列、无列名膨胀
22. 体系质量整体放大 10 倍，归一化物理特征严格不变
"""

import math
import os
import tempfile
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
    GEL_ASSUMPTION_IDEAL,
    GEL_ASSUMPTION_MECHANISM_APPROX,
)
from core.component_physics import (
    compute_component_physics,
    compute_formulation_summary,
    enrich_narrow_table,
)
from core.crosslink_physics import compute_crosslink_features
from core.epoxy_mechanism_features import (
    EpoxyMechanismEngine,
    extract_epoxy_mechanism_features,
)

SMI_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"  # f_r = 2
SMI_CCN = "CCN"                                            # 乙胺: f_stoich = 2, f_network = 2 (线性增长)
SMI_EDA = "NCCN"                                            # 乙二胺: f_stoich = 4, f_network = 4 (交联剂)
SMI_PGE = "c1ccccc1OCC1CO1"                                 # 苯基缩水甘油醚: 单官能环氧 f_r = 1
SMI_SA = "O=C1CCC(=O)O1"                                    # 琥珀酸酐: 单酸酐 f_stoich = 1, f_network = 2
SMI_MTHPA = "CC1=CCC2C(=O)OC(=O)C2C1"                       # 甲基四氢苯酐: 单酸酐 f_stoich = 1, f_network = 2
SMI_TGDDM = "C1OC1CN(c2ccc(Cc3ccc(N(CC4CO4)CC5CO5)cc3)cc2)CC6CO6"  # 四官能环氧 f_r = 4


def test_01_default_mode_assumed_r1():
    """1. 默认工程模式下，仅有结构缺失配比时假设 r=1.0，并正确标记 assumed_stoichiometric"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])

    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=True)
    assert rc.r_value == 1.0
    assert rc.r_source == R_SRC_ASSUMED_STOICH
    assert rc.rxn_r_assumed == 1.0

    cp = compute_component_physics(df, assume_stoichiometric_if_missing=True)
    assert cp["cp_r_value"].iloc[0] == 1.0
    assert cp["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH
    assert cp["rxn_r_assumed"].iloc[0] == 1.0

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=True)
    assert xl["xl_r_value"].iloc[0] == 1.0
    assert xl["xl_nu_source"].iloc[0] == "mechanism_aware"
    assert xl["xl_nu_theory_mol_m3"].iloc[0] > 0

    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df, assume_stoichiometric_if_missing=True)
    assert mech["mech_stoichiometry_r"].iloc[0] == 1.0
    assert mech["mech_gel_possible_ideal"].iloc[0] == 1.0


def test_02_strict_mode_missing_r_nan():
    """2. 显式严格模式下，缺失配比必须返回 r=NaN，标记 missing_stoichiometry，rxn_r_assumed=0.0"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])

    rc = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], assume_stoichiometric_if_missing=False)
    assert np.isnan(rc.r_value)
    assert rc.r_source == R_SRC_MISSING_STOICH
    assert rc.rxn_r_assumed == 0.0

    cp = compute_component_physics(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(cp["cp_r_value"].iloc[0])
    assert cp["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH
    assert cp["rxn_r_assumed"].iloc[0] == 0.0

    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl["xl_r_value"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "missing_stoichiometry"

    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(mech["mech_stoichiometry_r"].iloc[0])
    assert np.isnan(mech["mech_gel_possible_ideal"].iloc[0])


def test_03_strict_mode_no_hidden_r1():
    """3. (P0-1) 严格模式下，下游交联与机理特征绝不得暗中隐式使用 r=1.0 计算物理量"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    xl = compute_crosslink_features(df, assume_stoichiometric_if_missing=False)

    assert np.isnan(xl["xl_r_value"].iloc[0]), "xl_r_value 必须为 NaN"
    assert np.isnan(xl["xl_f_avg"].iloc[0]), "缺失 r 时全体系 f_avg 必须为 NaN"
    assert np.isnan(xl["xl_alpha_gel"].iloc[0]), "严格模式缺失 r 时凝胶点转化率必须为 NaN"
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0]), "严格模式缺失 r 时有效交联密度必须为 NaN"
    assert np.isnan(xl["xl_Mc_g_mol"].iloc[0]), "严格模式缺失 r 时理论 Mc 必须为 NaN"
    assert xl["xl_nu_source"].iloc[0] not in ("mechanism_aware", "junction", "aggregate")

    engine = EpoxyMechanismEngine(verbose=False)
    mech = engine.enrich_dataframe(df, assume_stoichiometric_if_missing=False)
    assert np.isnan(mech["mech_theoretical_alpha_gel"].iloc[0])
    assert np.isnan(mech["mech_theoretical_Mc"].iloc[0])


def test_04_precalculated_cp_vs_auto_calculated_cp_equivalence():
    """4. 传入预计算 cp DataFrame 与内部自动计算 cp 两种路径输出完全一致"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA,
        "curing_agent_1_phr": 8.83,
    }])
    # 路径 A: 内部自动计算
    xl_auto = compute_crosslink_features(df, cp=None)

    # 路径 B: 外部显式传入 cp
    cp_ext = compute_component_physics(df)
    xl_ext = compute_crosslink_features(df, cp=cp_ext)

    for col in ("xl_r_value", "xl_f_avg", "xl_alpha_gel", "xl_nu_theory_mol_m3", "xl_Mc_g_mol"):
        assert pytest.approx(xl_auto[col].iloc[0], abs=1e-4) == xl_ext[col].iloc[0]
    assert xl_auto["xl_nu_source"].iloc[0] == xl_ext["xl_nu_source"].iloc[0]


def test_05_assumed_r_persistence_across_storage():
    """5. (P0-2 测试 A) 假设来源持久化：原始无 r -> 默认增强 -> 保存 CSV/Parquet -> 重载 -> 再次解析，全过程保持 assumed"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    enriched = enrich_narrow_table(df, assume_stoichiometric_if_missing=True, keep_metadata=True)

    assert enriched["formulation_r_value"].iloc[0] == 1.0
    assert enriched["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH
    assert enriched["rxn_r_assumed"].iloc[0] == 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test_assumed.csv")
        parquet_file = os.path.join(tmpdir, "test_assumed.parquet")
        enriched.to_csv(csv_file, index=False)
        enriched.to_parquet(parquet_file, index=False)

        # CSV 重新加载测试
        loaded_csv = pd.read_csv(csv_file)
        re_csv = enrich_narrow_table(loaded_csv, assume_stoichiometric_if_missing=True, keep_metadata=True)
        assert re_csv["formulation_r_value"].iloc[0] == 1.0
        assert re_csv["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH
        assert re_csv["rxn_r_assumed"].iloc[0] == 1.0

        # Parquet 重新加载测试
        loaded_parquet = pd.read_parquet(parquet_file)
        re_parq = enrich_narrow_table(loaded_parquet, assume_stoichiometric_if_missing=True, keep_metadata=True)
        assert re_parq["formulation_r_value"].iloc[0] == 1.0
        assert re_parq["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH
        assert re_parq["rxn_r_assumed"].iloc[0] == 1.0


def test_06_assumed_r_cannot_upgrade_to_literature():
    """6. (P0-2) 假设值在再次解析时，绝不能被错误升级为 literature"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    enriched = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)

    # 再次进入底层组件物理计算
    cp_re = compute_component_physics(enriched, assume_stoichiometric_if_missing=True)
    assert cp_re["cp_r_source"].iloc[0] == R_SRC_ASSUMED_STOICH, "假设值绝不得升级为 literature"
    assert cp_re["rxn_r_assumed"].iloc[0] == 1.0


def test_07_strict_mode_reparsing_assumed_data_recognizes_missing_r():
    """7. (P0-2 测试 C) 对曾经采用人工假设的增强表在严格模式下重新解析，必须将该 r 识别为缺失，不得误当成文献值"""
    df = pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}])
    enriched = enrich_narrow_table(df, assume_stoichiometric_if_missing=True)

    # 在严格模式下重新评估曾经假设过的表
    cp_strict = compute_component_physics(enriched, assume_stoichiometric_if_missing=False)
    assert np.isnan(cp_strict["cp_r_value"].iloc[0]), "严格模式下曾经的假设值必须重置为 NaN"
    assert cp_strict["cp_r_source"].iloc[0] == R_SRC_MISSING_STOICH
    assert cp_strict["rxn_r_assumed"].iloc[0] == 0.0

    xl_strict = compute_crosslink_features(enriched, assume_stoichiometric_if_missing=False)
    assert np.isnan(xl_strict["xl_r_value"].iloc[0])
    assert np.isnan(xl_strict["xl_nu_theory_mol_m3"].iloc[0])
    assert xl_strict["xl_nu_source"].iloc[0] == "missing_stoichiometry"


def test_08_real_literature_r_persistence():
    """8. (P0-2 测试 D) 真实文献 r=0.5 跨保存与重载始终保持 literature 来源与真实数值"""
    df = pd.DataFrame([{
        "resin_1_smiles": SMI_DGEBA,
        "curing_agent_1_smiles": SMI_EDA,
        "formulation_r_value": 0.5,
    }])
    enriched = enrich_narrow_table(df, keep_metadata=True)
    assert enriched["formulation_r_value"].iloc[0] == 0.5
    assert enriched["cp_r_source"].iloc[0] == R_SRC_LITERATURE
    assert enriched["rxn_r_assumed"].iloc[0] == 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "lit_test.csv")
        enriched.to_csv(csv_file, index=False)
        re_loaded = enrich_narrow_table(pd.read_csv(csv_file), keep_metadata=True)
        assert re_loaded["formulation_r_value"].iloc[0] == 0.5
        assert re_loaded["cp_r_source"].iloc[0] == R_SRC_LITERATURE
        assert re_loaded["rxn_r_assumed"].iloc[0] == 0.0


def test_09_complete_mass_calculation_persistence():
    """9. (P0-2 测试 E) 完整用量推导出的 r=2.0 跨保存与重载始终保持 calculated_from_mass，不得升级为 literature"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA,
        "curing_agent_1_phr": 17.65,
    }])
    enriched = enrich_narrow_table(df, keep_metadata=True)
    assert pytest.approx(enriched["formulation_r_value"].iloc[0], abs=1e-2) == 2.0
    assert enriched["cp_r_source"].iloc[0] == R_SRC_CALCULATED
    assert enriched["rxn_r_assumed"].iloc[0] == 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "mass_test.csv")
        enriched.to_csv(csv_file, index=False)
        re_loaded = enrich_narrow_table(pd.read_csv(csv_file), keep_metadata=True)
        assert pytest.approx(re_loaded["formulation_r_value"].iloc[0], abs=1e-2) == 2.0
        assert re_loaded["cp_r_source"].iloc[0] == R_SRC_CALCULATED, "纯质量推算值绝不得升级为 literature"
        assert re_loaded["rxn_r_assumed"].iloc[0] == 0.0


def test_10_none_nan_mw_inputs_no_crash():
    """10. (P0-3) mw 输入为 None, np.nan, pd.NA, '', 'nan', 0, -1 等异常值时不触发 TypeError 且能 RDKit 自愈补齐"""
    invalid_mws = [None, np.nan, pd.NA, "", "nan", "None", 0, -1, 0.0]
    for mw_val in invalid_mws:
        res = evaluate_reaction_network(
            [{"smiles": SMI_DGEBA, "mw": mw_val}],
            [{"smiles": SMI_EDA, "mw": mw_val}],
        )
        assert res.chemistry_valid is True, f"mw={mw_val} 应当能通过 RDKit 成功自愈"
        assert res.f_r == 2.0
        assert res.f_h_stoich == 4.0
        assert res.gel_possible_ideal == 1.0


def test_11_none_nan_functionality_inputs_no_crash():
    """11. (P0-3) f_network 与 f_stoich 包含 None, np.nan, pd.NA 时不崩溃并能依据结构安全补齐"""
    res = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA, "f_network": None, "f_stoich": np.nan}],
        [{"smiles": SMI_EDA, "f_network": np.nan, "f_stoich": None}],
    )
    assert res.chemistry_valid is True
    assert res.f_r == 2.0
    assert res.f_h_stoich == 4.0
    assert res.f_h_network == 4.0
    assert res.gel_possible_ideal == 1.0


def test_12_valid_smiles_recovers_missing_physical_parameters():
    """12. 仅提供合法 SMILES 字符串时，体系自动补齐 MW, 当量重及官能度"""
    comp = compute_component_physics(pd.DataFrame([{"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA}]))
    assert pytest.approx(comp["resin_1_mw_resolved"].iloc[0], abs=1.0) == 340.42
    assert pytest.approx(comp["resin_1_ew_resolved"].iloc[0], abs=1.0) == 170.21
    assert comp["resin_1_f_stoich"].iloc[0] == 2.0
    assert comp["resin_1_f_network"].iloc[0] == 2.0

    assert pytest.approx(comp["curing_agent_1_mw_resolved"].iloc[0], abs=1.0) == 60.10
    assert pytest.approx(comp["curing_agent_1_ew_resolved"].iloc[0], abs=1.0) == 15.02
    assert comp["curing_agent_1_f_stoich"].iloc[0] == 4.0
    assert comp["curing_agent_1_f_network"].iloc[0] == 4.0


def test_13_invalid_smiles_no_fake_functionality():
    """13. 完全无效的 SMILES 不得伪造官能度，安全标记为 invalid_chemistry"""
    res = evaluate_reaction_network(
        [{"smiles": SMI_DGEBA}],
        [{"smiles": "TOTALLY_INVALID_SMILES_STRING"}],
    )
    assert res.chemistry_valid is False
    assert res.r_source == R_SRC_INVALID_CHEM
    assert res.gel_possible_ideal == 0.0 or np.isnan(res.gel_possible_ideal)
    assert np.isnan(res.r_value)


def test_14_a2_b2_critical_boundary_preserved():
    """14. A2–B2 临界边界保持正确：DGEBA + 乙胺 CCN (r=1.0) 判定为 critical，不可形成三维网络"""
    res = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_CCN}], explicit_r=1.0)
    assert res.gel_regime == GEL_REGIME_CRITICAL
    assert res.gel_possible_ideal == 0.0
    assert np.isnan(res.alpha_gel)
    assert np.isnan(res.Mc_ideal)
    assert res.nu_ideal_mol_m3 == 0.0


def test_15_a2_b4_supercritical_gelation_preserved():
    """15. A2–B4 (DGEBA + EDA, r=1.0) 超临界凝胶判断保持正确"""
    res = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], explicit_r=1.0)
    assert res.gel_regime == GEL_REGIME_SUPERCRITICAL
    assert res.gel_possible_ideal == 1.0
    assert pytest.approx(res.alpha_gel, abs=1e-4) == 1.0 / math.sqrt(3.0)
    assert res.nu_ideal_mol_m3 > 0


def test_16_a2_b4_subcritical_unreachable_r01_preserved():
    """16. A2–B4 (DGEBA + EDA, r=0.1) 亚临界不可达凝胶判断保持正确"""
    res = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_EDA}], explicit_r=0.1)
    assert res.gel_regime == GEL_REGIME_SUBCRITICAL
    assert res.gel_possible_ideal == 0.0
    assert res.p_B_critical > 1.0  # 物理不可达
    assert np.isnan(res.alpha_gel)


def test_17_monofunctional_epoxy_no_gel_preserved():
    """17. 单官能环氧体系 (PGE + EDA) 保持不可凝胶，无正有效网络密度"""
    res = evaluate_reaction_network([{"smiles": SMI_PGE}], [{"smiles": SMI_EDA}], explicit_r=1.0)
    assert res.gel_possible_ideal == 0.0
    assert np.isnan(res.alpha_gel)
    assert np.isnan(res.Mc_ideal)

    df = pd.DataFrame([{"resin_1_smiles": SMI_PGE, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 1.0}])
    xl = compute_crosslink_features(df)
    assert np.isnan(xl["xl_nu_theory_mol_m3"].iloc[0])
    assert xl["xl_nu_source"].iloc[0] == "non_gelling"


def test_18_anhydride_stoichiometry_and_ew_preserved():
    """18. 酸酐固化剂当量重计算保持 f_stoich=1.0 与 f_network=2.0，不折半"""
    res = evaluate_reaction_network([{"smiles": SMI_DGEBA}], [{"smiles": SMI_MTHPA}], explicit_r=1.0)
    assert res.f_h_stoich == pytest.approx(1.0, abs=1e-2)
    assert res.f_h_network == pytest.approx(2.0, abs=1e-2)
    assert res.gel_assumption == GEL_ASSUMPTION_MECHANISM_APPROX


def test_19_custom_dataframe_index_preserved():
    """19. 自定义 DataFrame 索引在所有处理环节保持完全一致"""
    custom_ids = ["CFRP_EXP_SAMPLE_001", "CFRP_EXP_SAMPLE_002"]
    df = pd.DataFrame([
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 1.0},
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 0.5},
    ], index=custom_ids)

    cp = compute_component_physics(df)
    assert list(cp.index) == custom_ids

    xl = compute_crosslink_features(df, cp=cp)
    assert list(xl.index) == custom_ids

    mech = extract_epoxy_mechanism_features(df)
    assert list(mech.index) == custom_ids


def test_20_mixed_missing_rows_no_cross_contamination():
    """20. 多行缺失模式不同的 DataFrame 在批量计算中不发生错位或交叉污染"""
    df = pd.DataFrame([
        {"resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0, "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.83},  # row 0: 完整质量
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA},                                                            # row 1: 纯缺失
        {"resin_1_smiles": SMI_DGEBA, "curing_agent_1_smiles": SMI_EDA, "formulation_r_value": 0.5},                                # row 2: 显式文献
    ])
    enriched = enrich_narrow_table(df, assume_stoichiometric_if_missing=True, keep_metadata=True)

    assert enriched["cp_r_source"].iloc[0] == R_SRC_CALCULATED
    assert enriched["cp_r_source"].iloc[1] == R_SRC_ASSUMED_STOICH
    assert enriched["cp_r_source"].iloc[2] == R_SRC_LITERATURE

    assert enriched["rxn_r_assumed"].iloc[0] == 0.0
    assert enriched["rxn_r_assumed"].iloc[1] == 1.0
    assert enriched["rxn_r_assumed"].iloc[2] == 0.0


def test_21_repeated_enrich_no_duplicate_columns():
    """21. 重复执行特征增强操作无重复列名、无 _x/_y 产生，列数稳定"""
    df = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA,
        "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA,
        "curing_agent_1_phr": 8.83,
    }])
    out1 = enrich_narrow_table(df)
    out2 = enrich_narrow_table(out1)
    assert out2.columns.is_unique
    assert len(out2.columns) == len(out1.columns)
    assert not any(c.endswith("_x") or c.endswith("_y") for c in out2.columns)


def test_22_scale_invariance_mass_x10():
    """22. 体系质量整体放大 10 倍，归一化物理特征 (r, f_avg, 凝胶点, 交联密度) 严格不变"""
    df1 = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 100.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 8.827,
    }], index=["base"])

    df2 = pd.DataFrame([{
        "resin_1_structure": SMI_DGEBA, "resin_1_phr": 1000.0,
        "curing_agent_1_structure": SMI_EDA, "curing_agent_1_phr": 88.27,
    }], index=["scaled_x10"])

    xl1 = compute_crosslink_features(df1)
    xl2 = compute_crosslink_features(df2)

    assert pytest.approx(xl1["xl_r_value"].iloc[0], abs=1e-4) == xl2["xl_r_value"].iloc[0]
    assert pytest.approx(xl1["xl_f_avg"].iloc[0], abs=1e-4) == xl2["xl_f_avg"].iloc[0]
    assert pytest.approx(xl1["xl_alpha_gel"].iloc[0], abs=1e-4) == xl2["xl_alpha_gel"].iloc[0]
    assert pytest.approx(xl1["xl_nu_theory_mol_m3"].iloc[0], abs=1e-4) == xl2["xl_nu_theory_mol_m3"].iloc[0]
