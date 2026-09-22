# -*- coding: utf-8 -*-
"""配方物理量特征的识别与补齐回归测试。

背景：dsc 系列模型的 44 个特征（cp_* / *_mw_resolved / *_ew_resolved /
*_f_network / *_epoxy_group_count / formulation_* 等）训练时来自
component_physics.enrich_narrow_table，不在 workflow 产物里。
修复前它们被 looks_molecular 误判为分子特征，resolver 步骤 D 会对它们
启动全提取引擎（RDKit→MACCS→Morgan→FGD→环氧→Mordred→3D，
一轮 >2 分钟）且永远落空——用户看到的就是“模型补齐数据不按工作流干活”。

修复后：
  - is_physics_feature 识别全部物理量特征（含大写 W 的 cp_W_g_per_epoxy）
  - resolve() 的 A0 步骤用 component_physics 分层补齐（与训练侧同口径）
  - 步骤 D 重后端对物理量特征永不启动
"""
import numpy as np
import pandas as pd
import pytest

from core.auto_feature_resolver import (
    AutoFeatureResolver,
    _physics_alias,
    is_physics_feature,
    looks_molecular,
)

DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
DDM = "Nc1ccc(Cc2ccc(N)cc2)cc1"

#: dsc初始温度/dsc放热峰 模型契约里真实出现的物理量特征（回放后落入 resolver 的）
MODEL_PHYSICS_FEATURES = [
    "cp_W_g_per_epoxy", "cp_ahew", "cp_balance", "cp_dilution", "cp_eew",
    "cp_epoxy_conc_mol_m3", "cp_f_avg_network", "cp_f_h_network", "cp_f_r",
    "cp_r_value",
    "curing_agent_1_active_hydrogen_equivalent_count",
    "curing_agent_1_ew_resolved", "curing_agent_1_f_network",
    "curing_agent_1_molecular_weight_g_mol", "curing_agent_1_mw_resolved",
    "curing_agent_2_active_hydrogen_equivalent_count",
    "curing_agent_2_ew_resolved", "curing_agent_2_f_network",
    "curing_agent_2_molecular_weight_g_mol", "curing_agent_2_mw_resolved",
    "curing_agent_3_ew_resolved", "curing_agent_3_f_network",
    "curing_agent_3_molecular_weight_g_mol", "curing_agent_3_mw_resolved",
    "curing_agent_active_hydrogen_total",
    "formulation_hardener_total_ahew_g_eq", "formulation_r_value",
    "formulation_resin_total_eew_g_eq",
    "resin_1_epoxy_group_count", "resin_1_ew_resolved", "resin_1_f_network",
    "resin_1_molecular_weight_g_mol", "resin_1_mw_resolved",
    "resin_2_epoxy_group_count", "resin_2_ew_resolved", "resin_2_f_network",
    "resin_2_molecular_weight_g_mol", "resin_2_mw_resolved",
    "resin_3_epoxy_group_count", "resin_3_f_network", "resin_3_mw_resolved",
]


def test_all_model_physics_features_recognized():
    """模型契约里真实的 41 个物理量特征必须全部识别（含大写 W）。"""
    missing = [f for f in MODEL_PHYSICS_FEATURES if not is_physics_feature(f)]
    assert not missing, f"未识别的物理量特征: {missing}"


def test_non_physics_features_not_misclassified():
    assert not is_physics_feature("resin_Resin_MACCS_3")
    assert not is_physics_feature("resin_1_structure_xtb_homo")
    assert not is_physics_feature("cure_stage_1_temperature_c")
    assert not is_physics_feature("resin_1_structure")
    assert not is_physics_feature("")


def test_looks_molecular_excludes_physics():
    assert looks_molecular("cp_eew") is False
    assert looks_molecular("cp_W_g_per_epoxy") is False
    assert looks_molecular("resin_1_mw_resolved") is False
    assert looks_molecular("resin_2_f_network") is False
    # 真分子特征不受影响
    assert looks_molecular("resin_Resin_MACCS_3") is True
    assert looks_molecular("resin_1_structure_xtb_homo") is True


def test_physics_alias_mapping():
    assert _physics_alias("formulation_resin_total_eew_g_eq") == "cp_eew"
    assert _physics_alias("formulation_hardener_total_ahew_g_eq") == "cp_ahew"
    assert _physics_alias("formulation_r_value") == "cp_r_value"
    assert _physics_alias("formulation_resin_hardener_equivalent_ratio") == "cp_r_value"
    assert _physics_alias("resin_2_molecular_weight_g_mol") == "resin_2_mw_resolved"
    assert _physics_alias("curing_agent_1_equivalent_weight_g_eq") == "curing_agent_1_ew_resolved"
    assert _physics_alias("resin_1_epoxy_group_count") == "resin_1_f_stoich"
    assert _physics_alias("curing_agent_1_active_hydrogen_equivalent_count") == (
        "curing_agent_1_f_stoich"
    )
    # cp_* 在物理量帧里同名直取，无别名
    assert _physics_alias("cp_balance") is None


def _workspace_df(n=3):
    return pd.DataFrame({
        "resin_1_structure": [DGEBA] * n,
        "curing_agent_1_structure": [DDM] * n,
    })


def test_resolve_fills_physics_without_heavy_backend():
    """核心回归：物理量特征由 component_physics 补齐，重后端零调用。"""
    df = _workspace_df()
    todo = ["cp_eew", "cp_ahew", "cp_r_value", "cp_W_g_per_epoxy",
            "cp_f_h_network", "resin_1_mw_resolved", "curing_agent_1_f_network",
            "resin_1_epoxy_group_count", "formulation_resin_total_eew_g_eq",
            "formulation_r_value"]
    resolver = AutoFeatureResolver(master_tables=None, verbose=False)
    out, report = resolver.resolve(df, todo)

    # 全部填上
    for f in todo:
        assert f in out.columns and out[f].notna().any(), f"{f} 未填充"
        assert "配方物理量" in str(report["computed"].get(f)), f"{f} 不是物理量路径填充"

    # 数值与 component_physics 口径一致
    assert out["cp_eew"].iloc[0] == pytest.approx(170.21, abs=0.01)      # DGEBA EEW
    assert out["resin_1_mw_resolved"].iloc[0] == pytest.approx(340.42, abs=0.01)
    assert out["curing_agent_1_f_network"].iloc[0] == pytest.approx(4.0)  # DDM 4 活泼氢
    assert out["cp_W_g_per_epoxy"].iloc[0] == pytest.approx(
        out["cp_eew"].iloc[0] + out["cp_r_value"].iloc[0] * out["cp_ahew"].iloc[0]
    )

    # 重后端（RDKit/MACCS/Morgan/FGD/环氧/Mordred/3D）从未启动
    assert not resolver.backend._method_cache
    assert not resolver.backend._cache
    assert not report["from_extractor"]


def test_resolve_physics_graceful_without_structure():
    """无结构列时物理量特征按未解析处理（交给 imputer/手工映射），不启动重后端。"""
    df = pd.DataFrame({"process_temp_c": [120.0, 130.0]})
    resolver = AutoFeatureResolver(master_tables=None, verbose=False)
    out, report = resolver.resolve(df, ["cp_eew", "cp_r_value"])
    assert "cp_eew" in report["unresolved"]
    assert not resolver.backend._method_cache
    assert not resolver.backend._cache
