# -*- coding: utf-8 -*-
"""
core/feature_registry.py

CFRP 环氧树脂反应与交联统一特征注册表与元数据管理中心。
管理四模块输出特征的物理含义、物理单位、计算来源、假设级别、是否为模型输入及别名关系。
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


@dataclass
class FeatureMeta:
    """特征元数据规范定义"""
    feature_name: str           # 原始特征字段名
    canonical_name: str         # 统一规范特征名
    category: str               # 类别: stoichiometry / functionality / network_physics / kinetics / thermodynamics / geometry / electronic / molecular_structure / charge_electrostatic / quantum_xtb / formulation / curing_process / composite
    unit: str                   # 物理单位: dimensionless, g/mol, mol/m3, mmol/g, eV, etc.
    definition: str             # 物理化学定义说明
    calculation_source: str     # 计算模块来源: component_physics / crosslink_physics / epoxy_mechanism / reaction_core / literature
    assumption_level: str       # 假设级别: theoretical_exact / semi_empirical / proxy / empirical
    is_proxy: bool              # 是否为经验代理量
    include_in_ml: bool         # 规范 ML 训练矩阵中是否默认选入
    alias_of: Optional[str] = None      # 若为某规范特征的别名/重复列，指向规范特征名
    relationship: str = "exact"         # 别名关系: "exact" (同一量纲物理别名) 或 "unit_conversion" (确定性单位换算)
    conversion_factor: Optional[float] = None # 单位换算乘数 (如 mmol/g -> mol/m3 乘数为 1200.0)
    dedup_reason: Optional[str] = None  # 排除原因: exact_alias / unit_conversion / string_metadata / redundant
    feature_set: str = "reaction_core"  # 所属特征集合: "reaction_core" (Set A) | "molecular" (Set B) | "formulation_process" (Set C) | "composite"


# 核心特征规范清单
FEATURE_REGISTRY_ENTRIES: List[FeatureMeta] = [
    # ---- 1. 化学计量与当量平衡 (stoichiometry) ----
    FeatureMeta(
        feature_name="formulation_r_value",
        canonical_name="formulation_r_value",
        category="stoichiometry",
        unit="dimensionless",
        definition="环氧-固化剂反应当量比 r = N_B / N_A",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="cp_r_value",
        canonical_name="formulation_r_value",
        category="stoichiometry",
        unit="dimensionless",
        definition="逐组分物理汇总推导的化学计量比 r（与 formulation_r_value 完全等价）",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="formulation_r_value",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="xl_r_value",
        canonical_name="formulation_r_value",
        category="stoichiometry",
        unit="dimensionless",
        definition="交联物理模块使用的反应当量比 r（与 formulation_r_value 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="formulation_r_value",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="mech_stoichiometry_r",
        canonical_name="formulation_r_value",
        category="stoichiometry",
        unit="dimensionless",
        definition="机理特征模块使用的反应当量比 r（与 formulation_r_value 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="formulation_r_value",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="cp_balance",
        canonical_name="cp_balance",
        category="stoichiometry",
        unit="dimensionless",
        definition="化学计量对称平衡度 min(r, 1/r)",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_balance",
        canonical_name="cp_balance",
        category="stoichiometry",
        unit="dimensionless",
        definition="交联物理模块对称平衡度（与 cp_balance 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_balance",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="xl_alpha_max",
        canonical_name="cp_balance",
        category="stoichiometry",
        unit="dimensionless",
        definition="极限反应程度 min(1, r, 1/r)（在 r>0 时与 balance 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_balance",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="mech_theoretical_alpha_max",
        canonical_name="cp_balance",
        category="stoichiometry",
        unit="dimensionless",
        definition="机理模块理论最大反应程度（与 cp_balance 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_balance",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="rxn_r_assumed",
        canonical_name="rxn_r_assumed",
        category="stoichiometry",
        unit="flag",
        definition="是否基于等当量配平假定推导 r (1.0=假设, 0.0=实测/计算)",
        calculation_source="reaction_core",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="cp_r_conflict",
        canonical_name="cp_r_conflict",
        category="stoichiometry",
        unit="flag",
        definition="文献显式 r 与全组分质量推算 r 差异 >5% 冲突标记",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),

    # ---- 2. 官能度与网络拓扑因子 (functionality) ----
    FeatureMeta(
        feature_name="cp_f_avg_network",
        canonical_name="cp_f_avg_network",
        category="functionality",
        unit="eq/mol",
        definition="Flory 摩尔统计平均官能度 (n_R*f_R + n_H*f_H_net) / (n_R + n_H)",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_f_avg",
        canonical_name="cp_f_avg_network",
        category="functionality",
        unit="eq/mol",
        definition="交联物理模块摩尔平均官能度（与 cp_f_avg_network 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_f_avg_network",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="mech_average_functionality",
        canonical_name="mech_average_functionality",
        category="functionality",
        unit="eq/mol",
        definition="机理模块基于单体分子片段加权统计的平均官能度 (与逐组分物理口径独立)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="cp_f_r",
        canonical_name="cp_f_r",
        category="functionality",
        unit="eq/mol",
        definition="环氧树脂端摩尔加权平均官能度",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="cp_f_h_network",
        canonical_name="cp_f_h_network",
        category="functionality",
        unit="eq/mol",
        definition="固化剂端摩尔加权平均网络支化官能度（酸酐 f_net=2.0）",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_f_h_network",
        canonical_name="cp_f_h_network",
        category="functionality",
        unit="eq/mol",
        definition="交联物理固化剂网络官能度（与 cp_f_h_network 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_f_h_network",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="xl_f_h",
        canonical_name="cp_f_h_network",
        category="functionality",
        unit="eq/mol",
        definition="固化剂官能度简写（与 cp_f_h_network 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_f_h_network",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="cp_f_h_stoich",
        canonical_name="cp_f_h_stoich",
        category="functionality",
        unit="eq/mol",
        definition="固化剂化学计量官能度（酸酐 f_stoich=1.0）",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),

    # ---- 3. 配方有效质量与基团浓度 ----
    FeatureMeta(
        feature_name="cp_W_g_per_epoxy",
        canonical_name="cp_W_g_per_epoxy",
        category="network_physics",
        unit="g/mol_epoxy",
        definition="每 mol 环氧基对应的配方总质量 EEW + r*AHEW",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_W_g_per_epoxy",
        canonical_name="cp_W_g_per_epoxy",
        category="network_physics",
        unit="g/mol_epoxy",
        definition="交联物理每环氧基质量（与 cp_W_g_per_epoxy 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_W_g_per_epoxy",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="cp_epoxy_conc_mol_m3",
        canonical_name="cp_epoxy_conc_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="环氧基空间摩尔浓度 rho / W",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_epoxy_conc_mol_m3",
        canonical_name="cp_epoxy_conc_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="交联物理环氧基摩尔浓度（与 cp_epoxy_conc_mol_m3 完全等价）",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=False,
        alias_of="cp_epoxy_conc_mol_m3",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="cp_dilution",
        canonical_name="cp_dilution",
        category="network_physics",
        unit="dimensionless",
        definition="树脂在粘料中的稀释质量比",
        calculation_source="component_physics",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=True,
    ),

    # ---- 4. 凝胶点与交联网络密度 (network_physics) ----
    FeatureMeta(
        feature_name="xl_alpha_gel",
        canonical_name="xl_alpha_gel",
        category="network_physics",
        unit="dimensionless",
        definition="Flory-Stockmayer 凝胶点临界转化率 1/sqrt(kappa_A*kappa_B)",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_theoretical_alpha_gel",
        canonical_name="mech_theoretical_alpha_gel",
        category="network_physics",
        unit="dimensionless",
        definition="机理模块基于单体基团图直接计算的理论凝胶转化率 (与交联物理口径独立)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_nu_theory_mol_m3",
        canonical_name="xl_nu_theory_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="机制感知有效网络链密度 (epoxy_conc * (f_h-1)/f_h * balance * dilution)",
        calculation_source="crosslink_physics",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="xl_nu_junction_mol_m3",
        canonical_name="xl_nu_junction_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="Flory 交联点数密度理论模型",
        calculation_source="crosslink_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_crosslink_density_mol_m3",
        canonical_name="mech_crosslink_density_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="机理模块理想网络链密度 (rho*1e6 / Mc)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_crosslink_density",
        canonical_name="mech_crosslink_density_mol_m3",
        category="network_physics",
        unit="mol/m3",
        definition="机理模块交联密度简写（与 mech_crosslink_density_mol_m3 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        alias_of="mech_crosslink_density_mol_m3",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="mech_crosslink_density_proxy",
        canonical_name="mech_crosslink_density_mol_m3",
        category="network_physics",
        unit="mmol/g",
        definition="机理模块交联密度代理值（单位为 mmol/g，可换算为 mol/m3）",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=False,
        alias_of="mech_crosslink_density_mol_m3",
        relationship="unit_conversion",
        conversion_factor=1200.0,
        dedup_reason="unit_conversion",
    ),
    FeatureMeta(
        feature_name="mech_crosslink_density_mmol_g",
        canonical_name="mech_crosslink_density_mol_m3",
        category="network_physics",
        unit="mmol/g",
        definition="以 mmol/g 为单位的交联密度（换算关系: nu_mol_m3 = nu_mmol_g * rho * 1000）",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=False,
        alias_of="mech_crosslink_density_mol_m3",
        relationship="unit_conversion",
        conversion_factor=1200.0,
        dedup_reason="unit_conversion",
    ),
    FeatureMeta(
        feature_name="xl_Mc_g_mol",
        canonical_name="xl_Mc_g_mol",
        category="network_physics",
        unit="g/mol",
        definition="交联点间平均分子量 Mc = rho*1e6 / nu",
        calculation_source="crosslink_physics",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_theoretical_Mc",
        canonical_name="mech_theoretical_Mc",
        category="network_physics",
        unit="g/mol",
        definition="机理模块理论交联点间分子量 Mn / (f_avg - 2)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_Mc_proxy",
        canonical_name="mech_Mc_proxy",
        category="network_physics",
        unit="g/mol",
        definition="机理模块经验代理 Mc (Mn / max(f_avg-2, 0.1))",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
    ),

    # ---- 5. 自由体积与刚柔性指标 ----
    FeatureMeta(
        feature_name="mech_proxy_ffv",
        canonical_name="mech_proxy_ffv",
        category="geometry",
        unit="dimensionless",
        definition="基于基团拓扑与分子体积估计的分数自由体积代理 FFV",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="geom_pseudo_ffv",
        canonical_name="mech_proxy_ffv",
        category="geometry",
        unit="dimensionless",
        definition="几何模块伪自由体积代理（与 mech_proxy_ffv 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=False,
        alias_of="mech_proxy_ffv",
        dedup_reason="exact_alias",
    ),
    FeatureMeta(
        feature_name="cross_flexibility_index",
        canonical_name="cross_flexibility_index",
        category="geometry",
        unit="dimensionless",
        definition="柔顺度指数: (N_rot / (N_arom + 1)) * (Mc / 100)",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="cross_torsion_density_index",
        canonical_name="cross_torsion_density_index",
        category="geometry",
        unit="dimensionless",
        definition="微观扭转能密度与宏观交联密度的耦合指数",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
    ),

    # ---- 6. 电子轨道与反应驱动力 (electronic) ----
    FeatureMeta(
        feature_name="mech_fmo_delta_e_fwd_primary",
        canonical_name="mech_fmo_delta_e_fwd_primary",
        category="electronic",
        unit="eV",
        definition="主反应对前线轨道正向能差 LUMO(resin) - HOMO(curer)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_fmo_delta_e_rev_primary",
        canonical_name="mech_fmo_delta_e_rev_primary",
        category="electronic",
        unit="eV",
        definition="主反应对前线轨道逆向能差 LUMO(curer) - HOMO(resin)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_fmo_delta_e_fwd_weighted",
        canonical_name="mech_fmo_delta_e_fwd_weighted",
        category="electronic",
        unit="eV",
        definition="多组分摩尔权重加权正向能差",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_fmo_delta_e_fwd_min",
        canonical_name="mech_fmo_delta_e_fwd_min",
        category="electronic",
        unit="eV",
        definition="体系最低正向反应能垒驱动力",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
    ),
    FeatureMeta(
        feature_name="mech_fmo_delta_e_fwd_max",
        canonical_name="mech_fmo_delta_e_fwd_max",
        category="electronic",
        unit="eV",
        definition="体系最高正向反应能垒",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="reaction_core",
    ),

    # =========================================================================
    # ---- 7. 分子结构与化学特征 (Feature Set B: molecular) ----
    # =========================================================================
    FeatureMeta(
        feature_name="resin_1_molecular_weight_g_mol",
        canonical_name="resin_1_molecular_weight_g_mol",
        category="molecular_structure",
        unit="g/mol",
        definition="主树脂文献报告分子量 MW_R1 (可包含高聚物实际分子量)",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="resin_1_mw_resolved",
        canonical_name="resin_1_mw_resolved",
        category="molecular_structure",
        unit="g/mol",
        definition="基于结构 SMILES 解析的主树脂单体重复单元分子量",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="curing_agent_1_molecular_weight_g_mol",
        canonical_name="curing_agent_1_molecular_weight_g_mol",
        category="molecular_structure",
        unit="g/mol",
        definition="主固化剂文献报告分子量 MW_H1",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="curing_agent_1_mw_resolved",
        canonical_name="curing_agent_1_mw_resolved",
        category="molecular_structure",
        unit="g/mol",
        definition="基于结构 SMILES 解析的主固化剂单体分子量",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="formulation_resin_total_eew_g_eq",
        canonical_name="formulation_resin_total_eew_g_eq",
        category="molecular_structure",
        unit="g/eq",
        definition="配方总环氧当量重 EEW",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="formulation_hardener_total_ahew_g_eq",
        canonical_name="formulation_hardener_total_ahew_g_eq",
        category="molecular_structure",
        unit="g/eq",
        definition="配方总活泼氢当量重 AHEW",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="mech_weighted_formula_mw",
        canonical_name="mech_weighted_formula_mw",
        category="molecular_structure",
        unit="g/mol",
        definition="配方摩尔数均分子量 Mn",
        calculation_source="epoxy_mechanism",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="mech_weighted_tpsa",
        canonical_name="mech_weighted_tpsa",
        category="molecular_structure",
        unit="Å²",
        definition="体系加权拓扑极性表面积 TPSA",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="mech_weighted_rotatable_bonds",
        canonical_name="mech_weighted_rotatable_bonds",
        category="molecular_structure",
        unit="count",
        definition="体系加权可旋转键数",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="rigid_carbon_ratio",
        canonical_name="rigid_carbon_ratio",
        category="molecular_structure",
        unit="dimensionless",
        definition="体系刚性碳原子占比加权平均 (sp+sp2碳与总碳比值)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="mech_weighted_rigid_ratio",
        canonical_name="rigid_carbon_ratio",
        category="molecular_structure",
        unit="dimensionless",
        definition="刚性碳原子占比加权平均（与 rigid_carbon_ratio 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        alias_of="rigid_carbon_ratio",
        dedup_reason="exact_alias",
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="mech_weighted_aromatic_rings",
        canonical_name="mech_weighted_aromatic_rings",
        category="molecular_structure",
        unit="count",
        definition="芳香环数量加权平均",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="ff_torsion_density",
        canonical_name="ff_torsion_density",
        category="molecular_structure",
        unit="dimensionless",
        definition="分子力场构象扭转能密度代理",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="ff_charge_variance_weighted",
        canonical_name="ff_charge_variance_weighted",
        category="charge_electrostatic",
        unit="dimensionless",
        definition="体系加权 Gasteiger 电荷方差",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="ff_charge_variance",
        canonical_name="ff_charge_variance_weighted",
        category="charge_electrostatic",
        unit="dimensionless",
        definition="单体 Gasteiger 电荷方差（与 ff_charge_variance_weighted 完全重复）",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=False,
        alias_of="ff_charge_variance_weighted",
        dedup_reason="exact_alias",
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="geom_vdw_volume_weighted",
        canonical_name="geom_vdw_volume_weighted",
        category="geometry",
        unit="Å³",
        definition="加权范德华体积",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="geom_rg_weighted",
        canonical_name="geom_rg_weighted",
        category="geometry",
        unit="Å",
        definition="加权回转半径 Rg",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="geom_asphericity_weighted",
        canonical_name="geom_asphericity_weighted",
        category="geometry",
        unit="dimensionless",
        definition="加权分子非球面度",
        calculation_source="epoxy_mechanism",
        assumption_level="empirical",
        is_proxy=True,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_reaction_gap_primary",
        canonical_name="mech_fmo_delta_e_fwd_primary",
        category="quantum_xtb",
        unit="eV",
        definition="主反应对前线轨道正向反应能隙（与 mech_fmo_delta_e_fwd_primary 完全等价）",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        alias_of="mech_fmo_delta_e_fwd_primary",
        dedup_reason="exact_alias",
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_chemical_hardness_weighted",
        canonical_name="xtb_chemical_hardness_weighted",
        category="quantum_xtb",
        unit="eV",
        definition="体系加权化学硬度 eta",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_chemical_potential_diff",
        canonical_name="xtb_chemical_potential_diff",
        category="quantum_xtb",
        unit="eV",
        definition="体系反应物间化学势差值",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_homo_weighted",
        canonical_name="xtb_homo_weighted",
        category="quantum_xtb",
        unit="eV",
        definition="需外部 xTB 二进制计算支持的可选特征 (默认流水线未启用)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_lumo_weighted",
        canonical_name="xtb_lumo_weighted",
        category="quantum_xtb",
        unit="eV",
        definition="需外部 xTB 二进制计算支持的可选特征 (默认流水线未启用)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        feature_set="molecular",
    ),
    FeatureMeta(
        feature_name="xtb_gap_weighted",
        canonical_name="xtb_gap_weighted",
        category="quantum_xtb",
        unit="eV",
        definition="需外部 xTB 二进制计算支持的可选特征 (默认流水线未启用)",
        calculation_source="epoxy_mechanism",
        assumption_level="semi_empirical",
        is_proxy=False,
        include_in_ml=False,
        feature_set="molecular",
    ),

    # =========================================================================
    # ---- 8. 配方用量与固化工艺特征 (Feature Set C: formulation_process) ----
    # =========================================================================
    FeatureMeta(
        feature_name="resin_1_amount_phr",
        canonical_name="resin_1_amount_phr",
        category="formulation",
        unit="phr",
        definition="主树脂相对用量 phr",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="curing_agent_1_amount_phr",
        canonical_name="curing_agent_1_amount_phr",
        category="formulation",
        unit="phr",
        definition="主固化剂相对用量 phr",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="resin_total_phr",
        canonical_name="resin_total_phr",
        category="formulation",
        unit="phr",
        definition="配方树脂总用量 phr (基准100)",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="curing_agent_total_phr",
        canonical_name="curing_agent_total_phr",
        category="formulation",
        unit="phr",
        definition="配方固化剂总用量 phr",
        calculation_source="component_physics",
        assumption_level="theoretical_exact",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_max_temperature_c",
        canonical_name="process_max_temperature_c",
        category="curing_process",
        unit="°C",
        definition="固化工艺最高温度",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_temperature_time_integral_c_h",
        canonical_name="process_temperature_time_integral_c_h",
        category="curing_process",
        unit="°C·h",
        definition="温度-时间工艺历程积分 ∫T(t)dt",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_final_cure_temperature_c",
        canonical_name="process_final_cure_temperature_c",
        category="curing_process",
        unit="°C",
        definition="最终固化阶段保温温度",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_total_time_h",
        canonical_name="process_total_time_h",
        category="curing_process",
        unit="h",
        definition="总固化时长 (含升温与各阶段保温)",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_final_cure_time_h",
        canonical_name="process_final_cure_time_h",
        category="curing_process",
        unit="h",
        definition="最终固化阶段保温时间",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
    FeatureMeta(
        feature_name="process_has_post_cure",
        canonical_name="process_has_post_cure",
        category="curing_process",
        unit="flag",
        definition="是否存在后固化阶段 (0/1)",
        calculation_source="literature",
        assumption_level="empirical",
        is_proxy=False,
        include_in_ml=True,
        feature_set="formulation_process",
    ),
]


def get_feature_registry() -> Dict[str, FeatureMeta]:
    """获取字典形式的统一特征注册表"""
    return {entry.feature_name: entry for entry in FEATURE_REGISTRY_ENTRIES}


def get_canonical_feature_names() -> List[str]:
    """获取所有选入规范 ML 矩阵的特征名列表"""
    return [e.feature_name for e in FEATURE_REGISTRY_ENTRIES if e.include_in_ml]


def get_duplicate_feature_groups() -> Dict[str, List[str]]:
    """获取完全同义/别名特征分组"""
    groups: Dict[str, List[str]] = {}
    for e in FEATURE_REGISTRY_ENTRIES:
        if e.alias_of:
            groups.setdefault(e.alias_of, []).append(e.feature_name)
    return groups


def export_registry_reports(output_dir: str = "reports") -> Tuple[str, str, str]:
    """生成并导出三份特征治理与审计 CSV 报表文件"""
    os.makedirs(output_dir, exist_ok=True)
    reg_df = pd.DataFrame([asdict(e) for e in FEATURE_REGISTRY_ENTRIES])
    reg_path = os.path.join(output_dir, "feature_registry.csv")
    reg_df.to_csv(reg_path, index=False, encoding="utf-8-sig")

    dup_rows = []
    for canon, aliases in get_duplicate_feature_groups().items():
        for a in aliases:
            meta = next(e for e in FEATURE_REGISTRY_ENTRIES if e.feature_name == a)
            dup_rows.append({
                "canonical_feature": canon,
                "duplicate_or_alias_feature": a,
                "dedup_reason": meta.dedup_reason or "exact_alias",
                "category": meta.category,
                "unit": meta.unit,
            })
    dup_df = pd.DataFrame(dup_rows)
    dup_path = os.path.join(output_dir, "feature_duplicate_groups.csv")
    dup_df.to_csv(dup_path, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "total_registered_features", "count": len(FEATURE_REGISTRY_ENTRIES)},
        {"metric": "canonical_ml_features", "count": len(get_canonical_feature_names())},
        {"metric": "exact_aliases_excluded", "count": len(dup_df[dup_df["dedup_reason"] == "exact_alias"])},
        {"metric": "unit_conversions_excluded", "count": len(dup_df[dup_df["dedup_reason"] == "unit_conversion"])},
        {"metric": "features_set_A_reaction_core", "count": len(get_features_for_set("reaction_core"))},
        {"metric": "features_set_B_molecular", "count": len(get_features_for_set("molecular"))},
        {"metric": "features_set_C_formulation_process", "count": len(get_features_for_set("formulation_process"))},
        {"metric": "features_set_D_full_safe", "count": len(get_features_for_set("full_safe"))},
    ]
    sum_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(output_dir, "feature_selection_summary.csv")
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    return reg_path, dup_path, sum_path


FEATURE_SETS: Dict[str, str] = {
    "reaction_core": "Set A: 反应物理与交联网络核心特征 (约 28 维)",
    "molecular": "Set B: 分子结构、分子量、极性、电荷与量子化学特征",
    "formulation_process": "Set C: 配方用量与固化工艺历程特征",
    "full_safe": "Set D: 完整安全候选特征集合 (A + B + C)",
}


def get_features_for_set(feature_set: str = "reaction_core") -> List[str]:
    """根据特征集合代号获取对应的规范特征列表。

    参数:
        feature_set:
            - 'reaction_core' / 'A': 反应与交联物理核心特征 (Set A)
            - 'molecular' / 'B': 分子结构与化学特征 (Set B)
            - 'formulation_process' / 'C': 配方与固化工艺特征 (Set C)
            - 'A+B' / 'reaction_core+molecular': Set A + Set B
            - 'B+C' / 'molecular+formulation_process': Set B + Set C
            - 'A+C' / 'reaction_core+formulation_process': Set A + Set C
            - 'full_safe' / 'D' / 'all' / 'A+B+C': 完整安全候选特征 (A + B + C)
    """
    fs = str(feature_set).strip().lower()
    if fs in ("reaction_core", "a"):
        target_sets = {"reaction_core"}
    elif fs in ("molecular", "b"):
        target_sets = {"molecular"}
    elif fs in ("formulation_process", "c"):
        target_sets = {"formulation_process"}
    elif fs in ("a+b", "b+a", "reaction_core+molecular"):
        target_sets = {"reaction_core", "molecular"}
    elif fs in ("b+c", "c+b", "molecular+formulation_process"):
        target_sets = {"molecular", "formulation_process"}
    elif fs in ("a+c", "c+a", "reaction_core+formulation_process"):
        target_sets = {"reaction_core", "formulation_process"}
    elif fs in ("full_safe", "d", "all", "a+b+c"):
        target_sets = {"reaction_core", "molecular", "formulation_process", "composite"}
    else:
        target_sets = {fs}

    return [e.feature_name for e in FEATURE_REGISTRY_ENTRIES if e.include_in_ml and e.feature_set in target_sets]


# =========================================================================
# 二、语义化特征注册表、校验、Hash 与契约快照辅助函数 (Training Contract / Portal / UI)
# =========================================================================

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

SOURCE_TYPES = {"molecular_workflow", "derived_workflow", "manual_input", "target", "metadata", "unknown"}
DEFAULT_POLICIES = {"forbidden", "explicit_only", "workflow_only"}
FEATURE_STATUSES = {"draft", "approved", "legacy_observed", "deprecated", "blocked"}
PROFILE_STATUSES = {"draft", "approved", "deprecated", "blocked"}
_REVIEW_KEYS = {"approved_hash", "approved_at", "approved_by", "reviewer", "reviewed_at", "review_note", "review_notes", "change_summary", "change_summaries", "review_metadata"}


def _without_mutable_approval_metadata(payload: Any) -> Any:
    value = copy.deepcopy(payload)
    if isinstance(value, dict) and isinstance(value.get("approval"), dict):
        for key in _REVIEW_KEYS:
            value["approval"].pop(key, None)
    return value


def compute_registry_hash(payload: Mapping[str, Any]) -> str:
    clean = _without_mutable_approval_metadata(payload)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_registry(payload: Mapping[str, Any], require_approved: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return {"ok": False, "errors": ["registry must be a JSON object"], "warnings": [], "registry_hash": compute_registry_hash({})}
    registry_hash = compute_registry_hash(payload)
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(payload.get("registry_version"), str) or not payload.get("registry_version"):
        errors.append("registry_version is required")
    features = payload.get("features")
    if not isinstance(features, list):
        errors.append("features must be a list")
        features = []
    profiles = payload.get("model_profiles")
    if not isinstance(profiles, Mapping):
        errors.append("model_profiles must be an object")
        profiles = {}
    approval = payload.get("approval")
    if not isinstance(approval, Mapping):
        errors.append("approval must be an object")
        approval = {}
    ids: set[str] = set()
    names: set[str] = set()
    definitions: dict[str, Mapping[str, Any]] = {}
    for index, feature in enumerate(features):
        prefix = f"features[{index}]"
        if not isinstance(feature, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        feature_id, name = feature.get("feature_id"), feature.get("name")
        if not isinstance(feature_id, str) or not feature_id.strip():
            errors.append(f"{prefix}.feature_id is required")
        elif feature_id in ids:
            errors.append(f"duplicate feature_id: {feature_id}")
        else:
            ids.add(feature_id)
            definitions[feature_id] = feature
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name is required")
        elif name in names:
            errors.append(f"duplicate feature name: {name}")
        else:
            names.add(name)
        source_type, policy, status = feature.get("source_type"), feature.get("default_policy"), feature.get("status")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{prefix}.source_type is invalid: {source_type}")
        if policy not in DEFAULT_POLICIES:
            errors.append(f"{prefix}.default_policy is invalid: {policy}")
        if status not in FEATURE_STATUSES:
            errors.append(f"{prefix}.status is invalid: {status}")
        unit = feature.get("unit")
        if "unit" in feature and unit is not None:
            if not isinstance(unit, str) or not unit.strip():
                errors.append(f"{prefix}.unit is required")
        if "required_for_prediction" in feature and not isinstance(feature.get("required_for_prediction"), bool):
            errors.append(f"{prefix}.required_for_prediction must be boolean")
        if "nullable" in feature and not isinstance(feature.get("nullable"), bool):
            errors.append(f"{prefix}.nullable must be boolean")
        if feature.get("required_for_prediction") is True and feature.get("nullable") is True:
            errors.append(f"{prefix} required_for_prediction cannot be nullable")
        if source_type == "manual_input" and isinstance(feature.get("default"), (int, float)) and not isinstance(feature.get("default"), bool):
            errors.append(f"{prefix}.default numeric values are forbidden for manual_input")
        if source_type in {"derived_workflow", "molecular_workflow"}:
            rule = feature.get("calculation_rule")
            if not isinstance(rule, Mapping):
                errors.append(f"{prefix}.calculation_rule is required for workflow feature")
            else:
                if not isinstance(rule.get("input_fields"), list) or not rule.get("input_fields"):
                    errors.append(f"{prefix}.calculation_rule.input_fields is required")
                elif any(not isinstance(field, str) or not field.strip() for field in rule["input_fields"]):
                    errors.append(f"{prefix}.calculation_rule.input_fields must contain non-empty strings")
                if not isinstance(rule.get("implementation"), str) or not rule.get("implementation").strip():
                    errors.append(f"{prefix}.calculation_rule.implementation is required")
                for policy_name in ("null_policy", "invalid_policy"):
                    if policy_name in rule:
                        if not isinstance(rule.get(policy_name), str) or not rule.get(policy_name).strip():
                            errors.append(f"{prefix}.calculation_rule.{policy_name} is required")
        if source_type == "unknown" and status != "blocked":
            errors.append(f"{prefix} unknown feature must be blocked")
        if status == "blocked" and not any(feature.get(key) for key in ("blocking_reason", "block_reason", "reason")):
            errors.append(f"{prefix} blocked feature requires a blocking reason")
        if status == "legacy_observed" and not any(feature.get(key) for key in ("legacy_source", "source_artifact", "source_dataset")):
            errors.append(f"{prefix} legacy_observed feature requires a legacy source")
        valid_range = feature.get("valid_range")
        if valid_range is not None:
            if not isinstance(valid_range, Mapping):
                errors.append(f"{prefix}.valid_range must be an object")
            else:
                lower, upper = valid_range.get("min"), valid_range.get("max")
                for label, bound in (("min", lower), ("max", upper)):
                    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound)):
                        errors.append(f"{prefix}.valid_range.{label} must be finite")
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
                    errors.append(f"{prefix}.valid_range min cannot exceed max")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            errors.append(f"model profile {profile_id} must be an object")
            continue
        refs = profile.get("feature_ids")
        if not isinstance(refs, list):
            errors.append(f"model profile {profile_id}.feature_ids must be a list")
            refs = []
        seen_refs: set[str] = set()
        for feature_id in refs:
            if not isinstance(feature_id, str) or not feature_id.strip():
                errors.append(f"model profile {profile_id}.feature_ids entries must be non-empty strings")
                continue
            if feature_id not in definitions:
                errors.append(f"model profile {profile_id} references unknown feature_id: {feature_id}")
            if feature_id in seen_refs:
                errors.append(f"model profile {profile_id} references duplicate feature_id: {feature_id}")
            seen_refs.add(feature_id)
        profile_status = profile.get("status")
        if profile_status not in PROFILE_STATUSES:
            errors.append(f"model profile {profile_id}.status is invalid: {profile_status}")
        target_col = profile.get("target_col")
        if not isinstance(target_col, str) or not target_col.strip():
            errors.append(f"model profile {profile_id}.target_col is required")
        if profile_status == "approved":
            for feature_id in refs:
                if not isinstance(feature_id, str):
                    continue
                definition = definitions.get(feature_id)
                if definition is not None and definition.get("status") != "approved":
                    errors.append(f"model profile {profile_id} approved profile requires approved feature: {feature_id}")
        blocked = profile.get("blocked_feature_ids", [])
        if blocked and not isinstance(blocked, list):
            errors.append(f"model profile {profile_id}.blocked_feature_ids must be a list")
            blocked = []
        if isinstance(blocked, list):
            if any(not isinstance(feature_id, str) or not feature_id.strip() for feature_id in blocked):
                errors.append(f"model profile {profile_id}.blocked_feature_ids entries must be non-empty strings")
            if len(blocked) != len(set(item for item in blocked if isinstance(item, str))):
                errors.append(f"model profile {profile_id}.blocked_feature_ids contains duplicate entries")
            blocked_set = {item for item in blocked if isinstance(item, str)}
            actual_blocked = {
                feature_id for feature_id in refs
                if isinstance(feature_id, str)
                if definitions.get(feature_id, {}).get("status") == "blocked"
            }
            for feature_id in blocked:
                if feature_id not in refs:
                    errors.append(f"model profile {profile_id}.blocked_feature_ids references feature outside profile: {feature_id}")
                elif definitions.get(feature_id, {}).get("status") != "blocked":
                    errors.append(f"model profile {profile_id}.blocked_feature_ids requires blocked feature: {feature_id}")
            if blocked_set != actual_blocked:
                errors.append(f"model profile {profile_id}.blocked_feature_ids must match blocked definitions")
    approval_status = approval.get("status")
    if approval_status not in {"draft", "approved", "deprecated"}:
        errors.append(f"approval.status is invalid: {approval_status}")
    if require_approved:
        if approval_status != "approved":
            errors.append("registry approval status is not approved")
        if approval.get("approved_hash") != registry_hash:
            errors.append("approval.approved_hash does not match computed registry_hash")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "registry_hash": registry_hash}


def load_registry(path: str | Path, require_approved: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = validate_registry(payload, require_approved=require_approved)
    if not report["ok"]:
        raise ValueError("invalid feature registry: " + "; ".join(report["errors"]))
    return payload


def get_model_profile(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = registry.get("model_profiles", {})[profile_id]
    except (KeyError, TypeError):
        raise KeyError(f"unknown model profile: {profile_id}") from None
    if not isinstance(profile, Mapping):
        raise ValueError(f"model profile is not an object: {profile_id}")
    return copy.deepcopy(dict(profile))


def build_registry_snapshot(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    report = validate_registry(registry, require_approved=True)
    if not report["ok"]:
        raise ValueError("cannot build snapshot from unapproved registry: " + "; ".join(report["errors"]))
    profile = get_model_profile(registry, profile_id)
    if profile.get("status") != "approved":
        raise ValueError(f"model profile is not approved: {profile_id}")
    definitions = {item.get("feature_id"): item for item in registry.get("features", []) if isinstance(item, Mapping)}
    selected = []
    for feature_id in profile.get("feature_ids", []):
        if feature_id not in definitions:
            raise KeyError(f"model profile references unknown feature_id: {feature_id}")
        if definitions[feature_id].get("status") != "approved":
            raise ValueError(f"model profile references non-approved feature: {feature_id}")
        selected.append(copy.deepcopy(dict(definitions[feature_id])))
    snapshot = {
        "schema_version": registry.get("schema_version"),
        "registry_version": registry.get("registry_version"),
        "registry_hash": compute_registry_hash(registry),
        "registry_payload": copy.deepcopy(dict(registry)),
        "profile_id": profile_id,
        "model_profile": profile,
        "features": selected,
    }
    selected_payload = {"profile_id": profile_id, "model_profile": profile, "features": selected}
    snapshot["selected_features_hash"] = hashlib.sha256(
        json.dumps(selected_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return snapshot


def register_new_feature(
    registry: Mapping[str, Any],
    feature_definition: Mapping[str, Any],
    *,
    reviewer: str,
    target_profile_id: str | None = None,
    review_note: str = "",
) -> dict[str, Any]:
    """Explicitly register a newly approved semantic feature into registry draft."""
    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ValueError("reviewer is required to register a new feature")
    feature_id = str(feature_definition.get("feature_id") or "").strip()
    name = str(feature_definition.get("name") or feature_id).strip()
    if not feature_id:
        raise ValueError("feature_id cannot be empty")
    if not name:
        raise ValueError("feature name cannot be empty")
    unit = str(feature_definition.get("unit") or "unknown").strip()
    if not unit:
        unit = "unknown"
    source_type = str(feature_definition.get("source_type") or "manual_input").strip()
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {source_type}")

    updated = copy.deepcopy(dict(registry))
    features = updated.setdefault("features", [])

    status = str(feature_definition.get("status") or "draft").strip().lower()
    if status not in {"draft", "approved"}:
        status = "draft"

    new_entry: dict[str, Any] = {
        "feature_id": feature_id,
        "name": name,
        "label": str(feature_definition.get("label") or name).strip(),
        "source_type": source_type,
        "data_type": str(feature_definition.get("data_type") or "float").strip(),
        "unit": unit,
        "required_for_prediction": bool(feature_definition.get("required_for_prediction", False)),
        "nullable": bool(feature_definition.get("nullable", True)),
        "default_policy": str(feature_definition.get("default_policy") or "explicit_only").strip(),
        "description": str(feature_definition.get("description") or review_note or "人工审核登记的新特征").strip(),
        "status": status,
    }
    if "aliases" in feature_definition:
        aliases = feature_definition["aliases"]
        new_entry["aliases"] = [str(a).strip() for a in (aliases if isinstance(aliases, list) else [aliases]) if str(a).strip()]

    replaced = False
    for i, item in enumerate(features):
        if isinstance(item, Mapping) and item.get("feature_id") == feature_id:
            features[i] = new_entry
            replaced = True
            break
    if not replaced:
        features.append(new_entry)

    if target_profile_id and target_profile_id in updated.get("model_profiles", {}):
        prof = updated["model_profiles"][target_profile_id]
        f_ids = prof.setdefault("feature_ids", [])
        if feature_id not in f_ids:
            f_ids.append(feature_id)

    ver = updated.get("registry_version", "2026.08.27")
    updated["registry_version"] = ver
    return updated


def save_registry_atomic(path: str | Path, registry: Mapping[str, Any]) -> None:
    """Save registry using temporary file and atomic replace."""
    from .feature_mapping_review import save_atomic_json
    target = Path(path).resolve()
    val = validate_registry(registry, require_approved=False)
    if not val["ok"]:
        raise ValueError("无法保存不合法的 registry: " + "; ".join(val["errors"]))
    save_atomic_json(target, dict(registry))


def classify_feature_source_type(feature: Any, hints: dict[str, Any] | None = None) -> str:
    """Classify feature definition into manual_input, molecular_workflow, derived_workflow, target, metadata, unknown."""
    if not isinstance(feature, dict):
        return "unknown"

    decl = feature.get("source_type")
    if decl and decl in SOURCE_TYPES:
        return decl

    name = str(feature.get("name") or feature.get("feature_id") or "").strip()
    name_lower = name.lower()

    manual_keywords = [
        "固化温度", "固化时间", "固化压力", "curing temperature", "cure time", "molding pressure",
        "测试方法", "测试标准", "测试条件", "test standard", "样品状态",
        "备注", "质量等级", "目视观察", "异常标记",
        "树脂smiles", "配比", "phr",
    ]
    if any(k in name_lower for k in manual_keywords) or any(k in name for k in ["固化温度", "固化时间", "固化压力", "测试方法", "测试标准", "测试条件", "样品状态", "备注", "质量等级", "目视观察", "异常标记", "树脂SMILES", "配比", "PHR"]):
        if hints is not None and "smiles" in name_lower:
            hints["recipe_input"] = True
        return "manual_input"

    molecular_keywords = ["树脂分子量", "molwt", "fp_bit", "原子数", "官能团数"]
    if any(k in name_lower for k in molecular_keywords):
        return "molecular_workflow"

    if feature.get("calculation_rule"):
        return "derived_workflow"

    return "unknown"


def normalize_manifest_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize alias keys in a manifest binding entry without mutating original."""
    d = dict(entry)

    if "source_role" not in d and "source_type" in d:
        d["source_role"] = d.pop("source_type")
    else:
        d.pop("source_type", None)

    if "raw_column" not in d and "raw_columns" in d:
        d["raw_column"] = d.pop("raw_columns")
    else:
        d.pop("raw_columns", None)

    if "source_field" not in d and "source_fields" in d:
        d["source_field"] = d.pop("source_fields")
    else:
        d.pop("source_fields", None)

    if "feature_id" not in d and "semantic_feature_id" in d:
        d["feature_id"] = d.pop("semantic_feature_id")
    else:
        d.pop("semantic_feature_id", None)

    if "unit" not in d and "units" in d:
        d["unit"] = d.pop("units")
    else:
        d.pop("units", None)

    if "aliases" not in d and "accepted_aliases" in d:
        d["aliases"] = d.pop("accepted_aliases")
    else:
        d.pop("accepted_aliases", None)

    return d


def validate_manifest_source_type_consistency(manifest: Mapping[str, Any], registry: Mapping[str, Any]) -> list[str]:
    """Validate consistency between manifest bindings and registry definitions."""
    errors: list[str] = []
    bindings = manifest.get("feature_bindings") or manifest.get("mappings") or []
    reg_features = {f.get("feature_id"): f for f in registry.get("features", []) if isinstance(f, Mapping)}

    for item in bindings:
        if not isinstance(item, Mapping):
            continue
        norm = normalize_manifest_entry(item)
        fid = norm.get("feature_id")
        role = norm.get("source_role")
        if fid in reg_features:
            rf = reg_features[fid]
            reg_type = rf.get("source_type")
            name = rf.get("name", fid)
            if role != reg_type:
                errors.append(f"特征 '{name}' ({fid}) 的来源角色声明为 '{role}'，但注册表登记为 '{reg_type}'")
    return errors


