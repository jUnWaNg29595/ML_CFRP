# -*- coding: utf-8 -*-
"""“添加剂感知”特征方案回归测试。

覆盖四项能力：
P1 角色平票修复（见 tests/test_molecular_feature_workflow.py 相邻职责，本文件只测提取侧）
P2 反应型添加剂分诊进组分列表（环氧→树脂侧；胺/酸/酚等→固化剂侧）
P3 惰性添加剂加权描述符 + Fox 稀释项 + 缺 PHR 典型掺量与 20% 封顶
P4 促进剂修正转化率估算
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("rdkit")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reaction_simulator import (
    EpoxyReactionSimulator,
    MulticomponentCrosslinkedFeatureExtractor,
    _extract_multicomponent_chunk,
    _collect_additive_components,
    _apply_additive_weights,
    _classify_additive_component,
)

DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
DDS = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
DILUENT = "Cc1ccccc1OCC1CO1"                # 邻甲酚缩水甘油醚（含环氧）
DMP30 = "CN(C)c1cc(CN(C)C)cc(CN(C)C)c1O"    # 叔胺促进剂类
DOPO = "O=[PH]1Oc2ccccc2-c2ccccc21"         # 磷系阻燃剂（惰性）
DIACID = "OC(=O)CCCCCCCC(=O)O"              # 二酸（增韧剂代表）


@pytest.fixture(scope="module")
def sim():
    return EpoxyReactionSimulator(verbose=False)


@pytest.mark.parametrize(
    "smiles,exp_side,exp_cls,exp_acc",
    [
        (DILUENT, "resin", "reactive_diluent", False),
        (DIACID, "curer", "reactive_curer", False),
        ("NCCN", "curer", "reactive_curer", False),
        (DOPO, "inert", "flame_retardant", False),
        (DMP30, "inert", "accelerator", True),
        ("CCn1cc[n+](C)c1.F[B-](F)(F)F", "inert", "accelerator", True),
        ("[Li+].[O-][Cl+3]([O-])([O-])[O-]", "inert", "accelerator", True),
        ("c1ccccc1", "inert", "inert", False),
    ],
)
def test_additive_classification(sim, smiles, exp_side, exp_cls, exp_acc):
    side, cls, acc = _classify_additive_component(smiles, sim)
    assert (side, cls, acc) == (exp_side, exp_cls, exp_acc)


def test_default_phr_and_cap(sim):
    """缺 PHR 时用类别典型掺量，且封顶不超过单侧的 25% 基数（=20% 占比）。"""
    row = pd.Series({f"small_additive_{i}_structure": s for i, s in
                     enumerate([DILUENT, DMP30, DOPO], start=1)})
    info = _collect_additive_components(row, None, list(row.index), sim)
    assert info["weight_source"] == "default"
    assert info["accelerator_present"] is True
    r_add, c_add, feats = _apply_additive_weights(info, resin_base_phr=100.0, curer_base_phr=30.0)
    assert feats["additive_n_reactive_resin"] == 1.0
    assert feats["additive_n_inert"] == 2.0
    # 稀释剂默认 10phr < 上限 25phr，不应触发封顶
    assert all(w <= 25.0 + 1e-6 for _, w in r_add)
    # Fox 稀释项与惰性加权 MW 应有值
    assert feats["additive_fox_dilution_term"] > 0
    assert feats["additive_inert_weighted_mw"] > 0


def test_cap_prevents_weight_explosion(sim):
    """无 PHR 超量场景（多个高典型掺量添加剂）应触发封顶。"""
    row = pd.Series({
        "small_additive_1_structure": DIACID,
        "small_additive_2_structure": DIACID,
        "small_additive_3_structure": DIACID,
    })
    info = _collect_additive_components(row, None, list(row.index), sim)
    r_add, c_add, feats = _apply_additive_weights(info, resin_base_phr=100.0, curer_base_phr=30.0)
    total_curer_add = sum(w for _, w in c_add)
    assert total_curer_add <= 0.25 * 30.0 + 1e-6
    assert feats["additive_weight_capped"] == 1.0


def test_accelerator_boost_conversion(sim):
    """P4：促进剂存在时转化率估算上调。"""
    base_r = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    thpa = "CC1CCC2C(=O)OC(=O)C2C1"
    off = sim.estimate_conversion(base_r, thpa, 1.0, 150.0, 2.0, accelerator_present=False)
    on = sim.estimate_conversion(base_r, thpa, 1.0, 150.0, 2.0, accelerator_present=True)
    assert on > off
    m_off = sim.estimate_conversion_multicomponent([(base_r, 1.0)], [(thpa, 1.0)], 1.0, 150.0, 2.0,
                                                  accelerator_present=False)
    m_on = sim.estimate_conversion_multicomponent([(base_r, 1.0)], [(thpa, 1.0)], 1.0, 150.0, 2.0,
                                                 accelerator_present=True)
    assert m_on > m_off


@pytest.fixture(scope="module")
def additive_df():
    return pd.DataFrame({
        "resin_1_structure": [DGEBA] * 4,
        "curing_agent_1_structure": [DDS] * 4,
        "small_additive_1_structure": [DILUENT, DOPO, None, DIACID],
        "small_additive_2_structure": [DMP30, None, None, None],
        "small_additive_1_amount_phr": [10.0, 7.0, None, 12.0],
        "small_additive_2_amount_phr": [2.0, None, None, None],
        "resin_1_amount_phr": [100.0] * 4,
        "curing_agent_1_amount_phr": [30.0] * 4,
    })


def test_end_to_end_main_loop(additive_df):
    ext = MulticomponentCrosslinkedFeatureExtractor(verbose=False)
    res = ext.batch_extract_features_from_dataframe(
        additive_df, resin_cols=["resin_1_structure"], curer_cols=["curing_agent_1_structure"],
        max_components=1, stoichiometry_col=None, auto_estimate_conversion=True,
        reaction_method="weighted", prefix="crosslink", n_jobs=1,
    )
    assert len(res) == 4
    assert res.loc[0, "crosslink_additive_n_reactive_resin"] == 1.0
    assert res.loc[0, "crosslink_additive_accelerator_present"] == 1.0
    assert res.loc[1, "crosslink_additive_fox_dilution_term"] > 0
    assert res.loc[3, "crosslink_additive_n_reactive_curer"] == 1.0
    # 促进剂行转化率应高于无添加剂行
    assert res.loc[0, "crosslink_estimated_conversion"] > res.loc[2, "crosslink_estimated_conversion"]


def test_worker_path_matches_main_loop(additive_df):
    """并行 worker（子进程路径）与主循环产出相同的添加剂特征。"""
    ext = MulticomponentCrosslinkedFeatureExtractor(verbose=False)
    main = ext.batch_extract_features_from_dataframe(
        additive_df, resin_cols=["resin_1_structure"], curer_cols=["curing_agent_1_structure"],
        max_components=1, stoichiometry_col=None, auto_estimate_conversion=True,
        reaction_method="weighted", prefix="crosslink", n_jobs=1,
    )
    worker_rows = _extract_multicomponent_chunk(
        additive_df, None, ["resin_1_structure"], ["curing_agent_1_structure"],
        "resin_smiles", "curing_agent_smiles",
        None, "stoichiometric_ratio_r_cleaned", None, None, None,
        150.0, 2.0, True, "weighted", "crosslink", False,
    )
    worker = pd.DataFrame(worker_rows)
    cols = [c for c in main.columns if "additive" in c] + ["crosslink_estimated_conversion"]
    assert cols and all(c in worker.columns for c in cols)
    pd.testing.assert_frame_equal(
        main[cols].fillna(-1).round(6), worker[cols].fillna(-1).round(6),
        check_dtype=False,
    )
