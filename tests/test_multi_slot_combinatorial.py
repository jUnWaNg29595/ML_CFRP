import pytest
import numpy as np
import pandas as pd
from core.epoxy_mechanism_features import EpoxyMechanismEngine
from core.reaction_simulator import MulticomponentCrosslinkedFeatureExtractor


def test_multi_slot_and_small_molecules():
    """测试多槽位（多树脂 + 多固化剂 + 小分子促进剂/稀释剂）自适应全覆盖"""
    engine = EpoxyMechanismEngine()

    # 复杂工业级配方：
    # - 树脂1: 主环氧 DGEBA
    # - 树脂2: 刚性耐热环氧 TGDDM
    # - 固化剂1: 主芳香胺 DDS
    # - 固化剂2: 芳香二胺 DDM
    # - 促进剂槽位: 微量咪唑小分子 (2-E-4-MI)
    df_multi = pd.DataFrame([
        {
            "resin_1_structure": "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1",
            "resin_2_structure": "C1OC1CN(CC2CO2)C3=CC=C(CC4=CC=C(N(CC5CO5)CC6CO6)C=C4)C=C3",
            "curing_agent_1_structure": "NC1=CC=C(S(=O)(=O)C2=CC=C(N)C=C2)C=C1",
            "curing_agent_2_structure": "NC1=CC=C(CC2=CC=C(N)C=C2)C=C1",
            "accelerator_1_structure": "CCC1=NC=C(C)N1", # 2-乙基-4-甲基咪唑小分子
            "resin_1_amount_phr": 70.0,
            "resin_2_amount_phr": 30.0,
            "curing_agent_1_amount_phr": 25.0,
            "curing_agent_2_amount_phr": 10.0,
            "accelerator_1_phr": 2.0,
        },
        # 对比行：同批次中仅有单树脂 + 单固化剂的样本（空槽位安全忽略）
        {
            "resin_1_structure": "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1",
            "resin_2_structure": None,
            "curing_agent_1_structure": "NC1=CC=C(S(=O)(=O)C2=CC=C(N)C=C2)C=C1",
            "curing_agent_2_structure": None,
            "accelerator_1_structure": None,
            "resin_1_amount_phr": 100.0,
            "resin_2_amount_phr": None,
            "curing_agent_1_amount_phr": 30.0,
            "curing_agent_2_amount_phr": None,
            "accelerator_1_phr": None,
        }
    ])

    out = engine.enrich_dataframe(df_multi, include_ff_features=False, include_xtb_features=False)

    # 1. 验证没有因为空槽位产生任何 NaN
    mech_cols = [c for c in out.columns if c.startswith("mech_") or c.startswith("geom_") or c.startswith("cross_") or c.startswith("gen_")]
    assert len(mech_cols) >= 30
    for col in mech_cols:
        assert not out[col].isna().any(), f"列 {col} 存在 NaN 空缺！"

    # 2. 物理合理性检验：
    # 第1行加入了四官能刚性树脂 TGDDM，其刚性碳比例与交联密度应该显著高于纯 DGEBA 的第2行
    assert out["rigid_carbon_ratio"].iloc[0] > 0.70
    assert out["gen_crosslink_density"].iloc[0] > out["gen_crosslink_density"].iloc[1]


def test_combinatorial_kinetic_simulation_multi_slot():
    """测试组合反应模拟在多槽位与小分子促进剂下的动力学加权与执行"""
    sim = MulticomponentCrosslinkedFeatureExtractor(verbose=False)

    resin_components = [
        ("CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1", 80.0), # DGEBA
        ("C1OC1COC1=CC=CC=C1OCC1CO1", 20.0), # 间苯二酚二缩水甘油醚
    ]
    curer_components = [
        ("NC1=CC=C(CC2=CC=C(N)C=C2)C=C1", 25.0), # DDM
        ("CCC1=NC=C(C)N1", 2.0), # 咪唑小分子促进剂
    ]

    res = sim.extract_multicomponent_features(
        resin_components=resin_components,
        curer_components=curer_components,
        stoichiometry_r=1.0,
        reaction_method="combinatorial",
    )

    assert "product_smiles" in res
    assert res["product_smiles"] is not None
    assert len(res["product_smiles"]) > 0
    assert res.get("n_combinations", 0) > 0
