import pytest
import numpy as np
import pandas as pd
from core.epoxy_mechanism_features import EpoxyMechanismEngine


def test_high_tg_adaptive_mechanism_features():
    """测试基于活化能自适应加权的高 Tg 22 维物理特征体系"""
    engine = EpoxyMechanismEngine()

    # 构建一个包含端炔基的芳香配方
    df = pd.DataFrame([{
        "resin_1_structure": "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1",
        "curing_agent_1_structure": "C#CC1=CC=C(N)C=C1",
        "formulation_r_value": 1.0,
    }])

    out = engine.enrich_dataframe(df, include_ff_features=False, include_xtb_features=False)

    # 验证指南要求的 22 个严格连续特征
    required_features = [
        # 1. 拓扑与广义交联网络维
        "gen_Mc_adaptive",
        "gen_crosslink_density",
        "stoich_defect_ratio",
        "triazine_node_density",
        # 2. 量子化学动力学与反应概率维
        "prob_primary_epoxy_amine",
        "prob_alkyne_thermal",
        "xtb_reaction_gap_primary",
        "xtb_chemical_hardness_weighted",
        "xtb_chemical_potential_diff",
        # 3. 链段构象阻力与杂化刚度维
        "rigid_carbon_ratio",
        "ff_torsion_density",
        "ff_oop_density",
        "flexibility_index",
        # 4. 3D 几何形貌与自由体积维
        "geom_pbf",
        "geom_asphericity",
        "geom_compactness",
        "geom_pseudo_ffv",
        # 5. 凝聚态内聚能（CED）与产物生成维
        "post_hbond_donor_density",
        "ff_charge_variance",
        # 6. 非线性物理耦合项
        "cross_rigidity_crosslink",
        "cross_torsion_crosslink",
    ]

    for col in required_features:
        assert col in out.columns, f"缺少必要特征: {col}"
        val = out[col].iloc[0]
        assert not pd.isna(val), f"特征 {col} 存在 NaN"

    # 关键物理合理性约束校验
    # 刚性碳比率应处于 [0, 1] 之间，且对于双酚A + 苯胺衍生物应当 > 0.7
    assert 0.7 <= out["rigid_carbon_ratio"].iloc[0] <= 1.0

    # 偏离平面度 PBF 应 > 0
    assert out["geom_pbf"].iloc[0] > 0.0

    # 反应概率应严格在 (0, 1) 概率区间内
    assert 0.0 < out["prob_primary_epoxy_amine"].iloc[0] < 1.0
    assert 0.0 < out["prob_alkyne_thermal"].iloc[0] < 1.0

    # 自由体积 FFV 应落在高分子物理正常区间 [0.1, 0.4]
    assert 0.1 <= out["geom_pseudo_ffv"].iloc[0] <= 0.4
