# -*- coding: utf-8 -*-
"""
reaction_simulator.py

环氧树脂-固化剂模拟反应模块

功能：
1. 使用RDKit反应模板模拟环氧开环反应
2. 支持多种固化剂类型：胺类、酸酐、硫醇、酰肼等
3. 生成反应产物SMILES并提取特征
4. 支持不同固化度(α)的反应产物生成

反应机理：
- 环氧-胺反应：环氧基开环与伯胺反应生成仲胺+羟基，再与仲胺反应生成叔胺
- 环氧-酸酐反应：环氧基与酸酐开环生成酯键+羧酸
- 环氧-硫醇反应：环氧基与硫醇反应生成硫醚+羟基
- 环氧-酰肼反应：环氧基与酰肼反应

作者：Claude AI Assistant
日期：2026-01-13
"""

from __future__ import annotations

import re
import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入线程配置
try:
    from . import thread_config
except ImportError:
    pass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdChemReactions
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    rdChemReactions = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# 导入现有SMILES工具
try:
    from .smiles_utils import (
        convert_to_smiles, 
        normalize_chemical_string, 
        split_smiles_cell,
        canonicalize_smiles
    )
except ImportError:
    convert_to_smiles = lambda x, **kw: x
    normalize_chemical_string = lambda x, **kw: x
    split_smiles_cell = lambda x: [x] if x else []
    canonicalize_smiles = lambda x: x


# =============================================================================
# 反应模板定义 (SMIRKS格式)
# =============================================================================

# =============================================================================
# 产物分层描述符（模块级缓存，多进程 Worker 各自持有副本）
# =============================================================================

_MODULE_DESC_CACHE = {}      # 2D拓扑 + 交联位点 + 图论特征
_MODULE_DESC3D_CACHE = {}    # 3D构象特征
_MODULE_CACHE_MAX = 20000


def _module_cache_put(cache, key, value):
    if len(cache) >= _MODULE_CACHE_MAX:
        cache.clear()
    cache[key] = value


def _desc_safe(mol, name):
    try:
        fn = getattr(Descriptors, name, None)
        if fn is None:
            return None
        return float(fn(mol))
    except Exception:
        return None


def _epoxide_count_of_mol(mol):
    try:
        patt = Chem.MolFromSmarts('[OX2;r3]')
        return len(mol.GetSubstructMatches(patt))
    except Exception:
        return 0


# [修复4] 网络采样参数：全局固定，避免随配方漂移产生 artifact
# （实测 n_hub 在 2/3/5 间变化会使环密度跳 ±40%、悬环氧率在 0/0.167 间跳变）
NETWORK_SAMPLING = {
    "atom_budget": 450,   # 单簇重原子预算（决定网络规模上限）
    "n_hub": 3,           # 固化剂枢纽分子数（≥3 才能形成多枢纽真网络）
    "max_pool_units": 60, # 池内单体数上限
    "n_ensemble": 8,      # 系综样本数
}


def _junction_site_count(mol):
    """交联位点绝对数（网络节点计数，非归一化）。

    覆盖三类网络节点：
      1) 胺类：已反应的叔胺 N（H0）/ 仲胺 N（H1）——每个 N 连接多条网络链
      2) 硫醚 S：硫醇固化形成的 S 桥
      3) [修复] 酸酐类：酯键分支节点——酸酐开环酯化后，
         (a) 与两个酯/醚氧相连的叔碳（甘油骨架分支点）
         (b) 与两个羰基相连的环状酸酐母核碳（季碳/叔碳）
         (c) 酯基密度（C(=O)-O 计数）作为酸酐网络连通度代理
    """
    total = 0
    try:
        patterns = [
            '[NX3;H0;!$([n]);!$([NX3]-[CX3]=[OX1])]',           # 叔胺（完全反应的交联点）
            '[NX3;H1;!$([n]);$(N(-[#6])-[#6]);!$([NX3]-[CX3]=[OX1])]',  # 仲胺
            '[#16X2;$(S(-[#6])-[#6])]',                          # 硫醚桥
            # [酸酐修复] 酯键分支节点：C 同时连接 ≥2 个酯/醚氧（甘油骨架分支）
            '[CX4;$(C(-[OX2])-[OX2])]',
            # [酸酐修复] 酸酐母核碳：与羰基相邻的环碳（开环后仍为网络节点）
            '[CX4;R;$(C(-[CX3]=[OX1]))]',
        ]
        for sm in patterns:
            p = Chem.MolFromSmarts(sm)
            if p is not None:
                total += len(mol.GetSubstructMatches(p))
    except Exception:
        pass
    return total


def _network_absolute_features(mol, stats: dict) -> dict:
    """[修复1] 网络绝对量特征：让交联密度与簇大小（≈转化率）可分离。

    归一化密度（per heavy atom）会被"簇长大"稀释，导致 junction 密度在
    不同转化率下几乎不变——模型只能学到分子量信号。这里补充绝对量与
    多口径密度，使"网络密度"与"网络规模"成为两个独立维度。
    """
    out = {}
    try:
        n_heavy = float(mol.GetNumHeavyAtoms())
        n_junc = float(_junction_site_count(mol))
        n_mono = float(stats.get('n_reacted', 0.0))          # 参与反应的环氧单体数（≈网络规模）
        n_epi_total = float(stats.get('n_epoxy_total', 0.0))  # 池内环氧基总数
        out['product_junction_sites_abs'] = n_junc                    # 绝对交联点数
        out['product_junction_per_reacted_monomer'] = (n_junc / n_mono) if n_mono > 0 else 0.0
        out['product_junction_per_epoxy_group'] = (n_junc / n_epi_total) if n_epi_total > 0 else 0.0
        # 酯键/醚键绝对数：酸酐网络连通度
        ester_p = Chem.MolFromSmarts('[CX3](=[OX1])[OX2][#6]')
        ether_p = Chem.MolFromSmarts('[OD2]([#6])[#6]')
        out['product_ester_bonds_abs'] = float(len(mol.GetSubstructMatches(ester_p))) if ester_p else 0.0
        out['product_ether_bonds_abs'] = float(len(mol.GetSubstructMatches(ether_p))) if ether_p else 0.0
        out['product_network_scale'] = n_mono                          # 网络规模（≈转化率×池规模）
        # 交联点效率：每单位网络规模贡献的交联点（与规模解耦）
        out['product_junction_efficiency'] = (n_junc / max(1.0, n_heavy)) * (n_mono / max(1.0, n_mono)) if n_mono > 0 else 0.0
    except Exception:
        pass
    return out


def _graph_invariants_from_mol(mol):
    # 图论网络不变量（相对值口径）：平均最短路径 / 环密度
    if not NETWORKX_AVAILABLE:
        return {}
    out = {}
    try:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                G.add_node(atom.GetIdx())
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(a1).GetAtomicNum() > 1 and mol.GetAtomWithIdx(a2).GetAtomicNum() > 1:
                G.add_edge(a1, a2)
        if G.number_of_nodes() == 0:
            return out
        comps = list(nx.connected_components(G))
        apls = []
        for c in comps:
            sg = G.subgraph(c)
            nn = sg.number_of_nodes()
            if nn < 2:
                continue
            paths = dict(nx.all_pairs_shortest_path_length(sg))
            total = 0
            for _src, dists in paths.items():
                total += sum(dists.values())
            apls.append(total / (nn * (nn - 1)))
        # 相对值口径：仅保留强度量（平均最短路径、环密度）；
        # 图直径 / Wiener 指数随实现物采样规模漂移，不再输出
        n_heavy_graph = float(G.number_of_nodes())
        out['product_graph_avg_path_length'] = float(np.mean(apls)) if apls else 0.0
        if n_heavy_graph > 0:
            out['product_cyclomatic_density'] = float(
                mol.GetNumBonds() - mol.GetNumHeavyAtoms() + len(comps)
            ) / n_heavy_graph
    except Exception:
        pass
    return out


def _compute_product_3d_descriptors(smiles):
    # 3D构象特征（ETKDGv3 + 随机坐标兜底），失败返回空dict
    # [改动] 不再设原子数/碎片数闸门（实测 37/37 空缺全部由 >130 闸门造成，
    # 系统性偏向高 DP 大分子）；最坏耗时改用 maxIterations 上限约束
    # （RDKit 默认=10×原子数，430 原子会放大到数千次尝试，那才是卡死源头）。
    out = {}
    if not RDKIT_AVAILABLE:
        return out
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        # [提速] 迭代数按尺寸缩放（默认 10×原子数会放大到数千次；病态分子实测 >10min）
        n_heavy = mol.GetNumHeavyAtoms()
        params.maxIterations = max(30, min(100, 15000 // max(1, n_heavy)))
        cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            # 二级降级：纯距离几何（关闭 ETKDG 知识项，单次迭代便宜 ~2 倍）
            params.useRandomCoords = True
            try:
                params.useBasicKnowledge = False
            except Exception:
                pass
            params.maxIterations = max(20, min(60, 8000 // max(1, n_heavy)))
            cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            return out
        # 相对值口径：仅保留无量纲形状描述符（NPR 为归一化 PMI 比值、Spherocity ∈ [0,1]）；
        # 回转半径 / PMI 绝对值 / LabuteASA 随实现物尺寸漂移，不再输出
        out['product_3d_npr1'] = float(rdMolDescriptors.CalcNPR1(mol))
        out['product_3d_npr2'] = float(rdMolDescriptors.CalcNPR2(mol))
        out['product_3d_spherocity'] = float(rdMolDescriptors.CalcSpherocityIndex(mol))
    except Exception:
        return {}
    return out


def _compute_product_descriptors(smiles, include_3d=True):
    # 统一产物描述符入口：基础物性 + 交联位点 + 拓扑指数 + 图论 (+3D)
    if not smiles or not RDKIT_AVAILABLE:
        return {}
    base = _MODULE_DESC_CACHE.get(smiles)
    if base is None:
        out = {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                n_heavy = float(mol.GetNumHeavyAtoms())
                n_bonds = float(mol.GetNumBonds())

                def _density(count: float) -> float:
                    return (float(count) / n_heavy) if n_heavy > 0 else 0.0

                # 内部量（不进入特征表）：供聚合度/转化率等相对指标计算
                out['internal_product_mol_weight'] = float(Descriptors.MolWt(mol))
                rb = _desc_safe(mol, 'NumRotatableBonds')
                rb_count = float(rb) if rb is not None else 0.0

                # 强度量/无量纲描述符（尺寸无关，直接保留）
                out['product_logp'] = float(Descriptors.MolLogP(mol))
                out['product_fraction_csp3'] = float(Descriptors.FractionCSP3(mol))
                # 环系密度（刚性链段结构，每重原子）
                out['product_ring_density'] = _density(Descriptors.RingCount(mol))
                out['product_aromatic_ring_density'] = _density(Descriptors.NumAromaticRings(mol))
                out['product_aliphatic_ring_density'] = _density(Descriptors.NumAliphaticRings(mol))
                out['product_saturated_ring_density'] = _density(Descriptors.NumSaturatedRings(mol))
                # 极性/亲水密度（每重原子）
                out['product_h_donor_density'] = _density(Descriptors.NumHDonors(mol))
                out['product_h_acceptor_density'] = _density(Descriptors.NumHAcceptors(mol))
                out['product_tpsa_density'] = _density(Descriptors.TPSA(mol))
                out['product_molar_refractivity_density'] = _density(Descriptors.MolMR(mol))
                out['product_heteroatom_fraction'] = _density(Descriptors.NumHeteroatoms(mol))
                # 柔性链段占比（每总键数）
                out['product_rotatable_bond_ratio'] = (rb_count / n_bonds) if n_bonds > 0 else 0.0
                # 交联相关官能团密度（每重原子）
                out['product_hydroxyl_density'] = _density(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))))
                out['product_amine_density'] = _density(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2,H1,H0]'))))
                out['product_ether_density'] = _density(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OD2]([#6])[#6]'))))
                residual_epoxide_count = float(_epoxide_count_of_mol(mol))
                out['product_residual_epoxide_density'] = _density(residual_epoxide_count)
                out['internal_residual_epoxide_count'] = residual_epoxide_count
                out['product_junction_site_density'] = _density(_junction_site_count(mol))
                # [修复1] 绝对量网络特征：交联点数/酯键/醚键绝对计数——
                # 归一化密度会被簇长大稀释（不同转化率下几乎不变），
                # 绝对量让"网络密度"与"网络规模(≈转化率)"成为独立维度
                _n_junc_abs = float(_junction_site_count(mol))
                out['product_junction_sites_abs'] = _n_junc_abs
                out['product_junction_per_heavy_atom'] = (_n_junc_abs / n_heavy) if n_heavy > 0 else 0.0
                _ester_p = Chem.MolFromSmarts('[CX3](=[OX1])[OX2][#6]')
                _ether_p = Chem.MolFromSmarts('[OD2]([#6])[#6]')
                out['product_ester_bonds_abs'] = float(len(mol.GetSubstructMatches(_ester_p))) if _ester_p else 0.0
                out['product_ether_bonds_abs'] = float(len(mol.GetSubstructMatches(_ether_p))) if _ether_p else 0.0
                # 网络连通度密度：酯+醚桥总数 / 重原子（酸酐网络的关键量）
                out['product_network_bridge_density'] = (
                    (out['product_ester_bonds_abs'] + out['product_ether_bonds_abs']) / n_heavy
                ) if n_heavy > 0 else 0.0
                # Layer2: 无量纲拓扑形状指数（Kappa/HallKierAlpha/BalabanJ/E-state 极值均尺寸无关）
                # 注：Chi 连接性指数与 BertzCT 随实现物规模增长，已移除
                for k, nm in [('product_kappa1', 'Kappa1'), ('product_kappa2', 'Kappa2'), ('product_kappa3', 'Kappa3'),
                              ('product_balaban_j', 'BalabanJ'), ('product_hall_kier_alpha', 'HallKierAlpha'),
                              ('product_max_estate', 'MaxEStateIndex'), ('product_min_estate', 'MinEStateIndex')]:
                    v = _desc_safe(mol, nm)
                    if v is not None:
                        out[k] = v
                # Layer5: 图论不变量（仅强度量：平均最短路径、环密度）
                out.update(_graph_invariants_from_mol(mol))
        except Exception:
            return {}
        _module_cache_put(_MODULE_DESC_CACHE, smiles, out)
        base = out
    result = dict(base)
    if include_3d:
        d3 = _MODULE_DESC3D_CACHE.get(smiles)
        if d3 is None:
            d3 = _compute_product_3d_descriptors(smiles)
            _module_cache_put(_MODULE_DESC3D_CACHE, smiles, d3)
        result.update(d3)
    return result


@dataclass
class ReactionTemplate:
    """反应模板数据类"""
    name: str
    smirks: str
    description: str
    curing_agent_type: str  # 'amine', 'anhydride', 'thiol', 'hydrazide'
    reactivity_order: int = 1  # 反应优先级


# 环氧-伯胺反应：环氧开环 + 伯胺 -> 仲胺 + 羟基 (支持隐式氢)
EPOXY_PRIMARY_AMINE_RXN = ReactionTemplate(
    name="epoxy_primary_amine",
    smirks="[C:1]1[O:2][C:3]1.[NX3;H2:4]>>[C:1]([O:2])[C:3][N:4]",
    description="环氧基与伯胺反应，生成仲胺和β-羟基",
    curing_agent_type="amine",
    reactivity_order=1
)

# 环氧-仲胺反应：环氧开环 + 仲胺 -> 叔胺 + 羟基 (支持隐式氢)
EPOXY_SECONDARY_AMINE_RXN = ReactionTemplate(
    name="epoxy_secondary_amine",
    smirks="[C:1]1[O:2][C:3]1.[NX3;H1:4]([#6:5])[#6:6]>>[C:1]([O:2])[C:3][N:4]([#6:5])[#6:6]",
    description="环氧基与仲胺反应，生成叔胺和β-羟基",
    curing_agent_type="amine",
    reactivity_order=2
)

# 环氧-酸酐反应（5元环）：环氧 + 5元环酸酐 -> 具有游离羧酸的单酯
EPOXY_ANHYDRIDE_5_RXN = ReactionTemplate(
    name="epoxy_anhydride_5",
    smirks="[C:1]1[O:2][C:3]1.[#6:9]1[#6:10][C:4](=[O:5])[O:6][C:7](=[O:8])1>>[C:1][C:3][O:6][C:7](=[O:8])[#6:10][#6:9][C:4](=[O:5])[O:2]",
    description="环氧基与5元环酸酐开环反应，生成单酯和游离羧酸",
    curing_agent_type="anhydride",
    reactivity_order=1
)

# 环氧-酸酐反应（6元环）：环氧 + 6元环酸酐 -> 具有游离羧酸的单酯
EPOXY_ANHYDRIDE_6_RXN = ReactionTemplate(
    name="epoxy_anhydride_6",
    smirks="[C:1]1[O:2][C:3]1.[#6:9]1[#6:10][#6:11][C:4](=[O:5])[O:6][C:7](=[O:8])1>>[C:1][C:3][O:6][C:7](=[O:8])[#6:11][#6:10][#6:9][C:4](=[O:5])[O:2]",
    description="环氧基与6元环酸酐开环反应，生成单酯和游离羧酸",
    curing_agent_type="anhydride",
    reactivity_order=1
)

# 环氧-酸酐反应（通用/开链保底）：环氧 + 酸酐 -> 单酯主产物（无分离副产物）
EPOXY_ANHYDRIDE_RXN = ReactionTemplate(
    name="epoxy_anhydride",
    smirks="[C:1]1[O:2][C:3]1.[C:4](=[O:5])[O:6][C:7](=[O:8])>>[C:1][C:3][O:6][C:7](=[O:8]).[C:4](=[O:5])[O:2]",
    description="环氧基与酸酐反应主产物",
    curing_agent_type="anhydride",
    reactivity_order=1
)

# 环氧-硫醇反应：环氧 + 硫醇 -> 硫醚 + 羟基
EPOXY_THIOL_RXN = ReactionTemplate(
    name="epoxy_thiol",
    smirks="[C:1]1[O:2][C:3]1.[SX2;H1:4][#6:5]>>[C:1]([O:2])[C:3][S:4][#6:5]",
    description="环氧基与硫醇反应，生成硫醚和β-羟基",
    curing_agent_type="thiol",
    reactivity_order=1
)

# 环氧-酰肼反应：环氧 + 酰肼 -> 氨基醇
EPOXY_HYDRAZIDE_RXN = ReactionTemplate(
    name="epoxy_hydrazide",
    smirks="[C:1]1[O:2][C:3]1.[NX3;H2,H1:4][NX3:5][#6:6](=[O:7])>>[C:1]([O:2])[C:3][N:4][NX3:5][#6:6](=[O:7])",
    description="环氧基与酰肼反应",
    curing_agent_type="hydrazide",
    reactivity_order=1
)

# 环氧-酚反应：环氧 + 酚羟基 -> 醚键 + 羟基
EPOXY_PHENOL_RXN = ReactionTemplate(
    name="epoxy_phenol",
    smirks="[C:1]1[O:2][C:3]1.[OX2;H1:4][c:5]>>[C:1]([O:2])[C:3][O:4][c:5]",
    description="环氧基与酚羟基反应，生成醚键",
    curing_agent_type="phenol",
    reactivity_order=2
)

# 环氧-羧酸反应：环氧 + 羧酸 -> β-羟基酯
EPOXY_ACID_RXN = ReactionTemplate(
    name="epoxy_carboxylic_acid",
    smirks="[C:1]1[O:2][C:3]1.[CX3:4](=[O:5])[OX2;H1:6]>>[C:1]([O:2])[C:3][O:6][CX3:4](=[O:5])",
    description="环氧基与羧酸反应，生成β-羟基酯",
    curing_agent_type="acid",
    reactivity_order=2
)

# 环氧-异氰酸酯反应：环氧 + 异氰酸酯 -> 噁唑烷酮
EPOXY_ISOCYANATE_RXN = ReactionTemplate(
    name="epoxy_isocyanate",
    smirks="[C:1]1[O:2][C:3]1.[N:4]=[C:5]=[O:6]>>[C:1]1[O:2][C:5](=[O:6])[N:4][C:3]1",
    description="环氧基与异氰酸酯反应，生成噁唑烷酮环",
    curing_agent_type="isocyanate",
    reactivity_order=2
)

# 所有反应模板
# 环氧-醇醚化反应：环氧开环 + 脂肪醇 -> 醚 + 新羟基（链延伸/臂生长通道，高温/催化下发生）
EPOXY_ETHERIFICATION_RXN = ReactionTemplate(
    name="epoxy_etherification",
    smirks="[C:1]1[O:2][C:3]1.[OX2;H1:4][CX4;#6:5]>>[C:1]([O:2])[C:3][O:4][#6:5]",
    description="环氧基与脂肪羟基醚化，生成醚键和β-羟基（臂延伸通道）",
    curing_agent_type="alcohol",
    reactivity_order=3
)

ALL_REACTION_TEMPLATES = [
    EPOXY_ETHERIFICATION_RXN,

    EPOXY_PRIMARY_AMINE_RXN,
    EPOXY_SECONDARY_AMINE_RXN,
    EPOXY_ANHYDRIDE_5_RXN,
    EPOXY_ANHYDRIDE_6_RXN,
    EPOXY_ANHYDRIDE_RXN,
    EPOXY_THIOL_RXN,
    EPOXY_HYDRAZIDE_RXN,
    EPOXY_PHENOL_RXN,
    EPOXY_ACID_RXN,
    EPOXY_ISOCYANATE_RXN,
]


# =============================================================================
# 官能团识别SMARTS模式
# =============================================================================

FUNCTIONAL_GROUP_PATTERNS = {
    # 环氧基（环氧乙烷环）
    "epoxide": "[C]1[O][C]1",
    
    # 胺类
    "primary_amine": "[NX3;H2;!$(NC=O);!$(NS=O)]",  # 伯胺（排除酰胺）
    "secondary_amine": "[NX3;H1;!$(NC=O);!$(NS=O)]([C,c])[C,c]",  # 仲胺（兼容脂肪/芳香邻碳）
    "aromatic_amine": "[NX3;H2]c",  # 芳香胺（如DDM、DDS）
    
    # 酸酐
    # 环状酸酐通用模式(兼容RDKit芳构化表示, 如PMDA)
    "anhydride": "[o,OX2]1~[#6](=[OX1])~[#6]~[#6]~[#6](=[OX1])~1",
    
    # 硫醇
    "thiol": "[SX2H]",
    
    # 酰肼
    "hydrazide": "[NX3][NX3][CX3](=[OX1])",
    
    # 羟基（用于检测反应产物）
    "hydroxyl": "[OX2H]",

    # 酚羟基
    "phenol": "[OX2H][c]",

    # 羧酸
    "carboxylic_acid": "[CX3](=[OX1])[OX2H]",

    # 异氰酸酯
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    
    # 酯基（用于检测酸酐反应产物）
    "ester": "[CX3](=[OX1])[OX2][C]",
}


# =============================================================================
# 网络单元封端与连接点标记
# =============================================================================

# 连接点标记用 atom map 编号（仅内部使用，输出前会被清除）
_STUB_MAP_NUM = 901   # 封端碳：下一枢纽接入位置（原残留环氧的进攻碳）
_HUB_MAP_NUM = 902    # 枢纽剩余活性原子：伯/仲胺N、硫醇S、酚O、羧酸OH


def _cap_residual_epoxide_stubs(mol):
    """将残留的 3 元环氧环开环为邻二醇封端，并标记网络连接点原子。

    修复交联单元表示的两个缺陷：
    1) "反应后" 代表性单元不应再保留环氧基：旧实现把分支末端悬键直接留成
       未反应环氧，导致产物分子式含 C1OC1 片段。这里统一开环——环内 O 保留
       为羟基，进攻碳新增羟基（水解式封端，原子数不变、化学上为链端羟基）。
    2) 连接点不再固定为字符串两端 2 个 [$]，而是打在真实连接原子上：
       - 封端碳（下一枢纽接入的位置）→ atom map 901
       - 枢纽剩余活性原子（伯/仲胺 N、硫醇 S、酚 O、羧酸 OH）→ atom map 902

    Returns:
        (封端并标记后的 mol, 悬挂封端数, 枢纽剩余活性位数)；失败时 mol 为 None
    """
    if mol is None:
        return None, 0, 0
    try:
        rw = Chem.RWMol(mol)
        patt = Chem.MolFromSmarts("[C]1[O][C]1")
        matches = rw.GetMol().GetSubstructMatches(patt) or []
        done = set()
        n_stub = 0
        for (a, o, b) in matches:
            if {a, o, b} & done:
                continue
            # 进攻位：重邻居数少者优先（位阻小），平局取剩余 H 多者（取代度低）
            na, nb = rw.GetAtomWithIdx(a), rw.GetAtomWithIdx(b)
            key_a = (len(na.GetNeighbors()), -(na.GetTotalNumHs() or 0))
            key_b = (len(nb.GetNeighbors()), -(nb.GetTotalNumHs() or 0))
            target = a if key_a <= key_b else b
            rw.RemoveBond(o, target)
            new_o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(target, new_o, Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(target).SetAtomMapNum(_STUB_MAP_NUM)
            done |= {a, o, b}
            n_stub += 1
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            return None, 0, 0

        n_hub = 0
        hub_patterns = (
            ("[NX3;H2,H1;!$(N-[CX3]=[OX1])]", 0),   # 伯/仲胺 N（排除酰胺）
            ("[SX2;H1]", 0),                        # 硫醇 S
            ("[OX2;H1][c]", 0),                     # 酚 O
            ("[CX3](=[OX1])[OX2;H1]", 2),           # 羧酸 OH 氧
        )
        for patt_s, pos in hub_patterns:
            try:
                hub_patt = Chem.MolFromSmarts(patt_s)
                if hub_patt is None:
                    continue
                for m in out.GetSubstructMatches(hub_patt) or []:
                    idx = m[pos] if pos < len(m) else m[0]
                    atom = out.GetAtomWithIdx(idx)
                    if atom.GetAtomMapNum() == 0:
                        atom.SetAtomMapNum(_HUB_MAP_NUM)
                        n_hub += 1
            except Exception:
                continue
        return out, n_stub, n_hub
    except Exception:
        return None, 0, 0


def _attachment_marked_unit_smiles(mol):
    """把带 901/902 atom map 的连接点原子转成 BigSMILES 内联描述符文本。

    例如 "[CH2:901]" → "C([$])"、"[NH:902]" → "N([$])"，描述符数量 =
    真实连接点数，位置在封端碳/剩余活性位上，多官能度因此可被表达。
    含多片段（"."）时返回 None（BigSMILES 单元内不允许碎片分隔符）。
    """
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return None
    if ":901" not in smi and ":902" not in smi:
        return None
    # 有机子集原子(C/N/O/S)去括号让隐式氢随描述符键自适应；其它元素保留括号仅去 map
    marked = re.sub(
        r"\[([CNOS])(?:[A-Za-z]*H\d*)?:90([12])\](\d*)",
        r"\1\3([$])",
        smi,
    )
    marked = re.sub(r"\[([A-Za-z][^\]:]*):90([12])\](\d*)", r"[\1]\3([$])", marked)
    if "[$]" not in marked or "." in marked or "," in marked:
        return None
    return marked


def _clean_unit_smiles(mol):
    """清除 atom map 后的干净单元 SMILES（用于描述符计算/存储）。"""
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            pass
        return Chem.MolToSmiles(out)
    except Exception:
        return None


# =============================================================================
# 小分子添加剂感知（分诊 / 典型掺量 / Fox 稀释项）
# =============================================================================

# 添加剂类别档案：default_phr = 缺失 PHR 时的类别典型掺量（避免等权兜底导致
# 添加剂被过度加权）；typical_tg_k = Fox 方程 1/Tg = Σ wᵢ/Tgᵢ 的类别典型值
# （K，None 表示不参与 Tg 混合，如促进剂/催化剂掺量过小）
_ADDITIVE_CLASS_PROFILE = {
    "reactive_diluent":   {"default_phr": 10.0, "typical_tg_k": 240.0},
    "reactive_curer":     {"default_phr": 12.0, "typical_tg_k": 220.0},
    "flame_retardant":    {"default_phr": 7.0,  "typical_tg_k": 340.0},
    "accelerator":        {"default_phr": 2.0,  "typical_tg_k": None},
    "inert":              {"default_phr": 5.0,  "typical_tg_k": 300.0},
}
# 添加剂合计占单侧权重的上限（封顶用）：缺 PHR 时防止等权兜底爆炸
_ADDITIVE_MAX_SIDE_FRACTION = 0.20

_ADDITIVE_STRUCT_COL_RE = re.compile(
    r"^(?:small_additive|reactive_diluent|reactive_toughener)_(\d+)_structure$", re.I
)

_ADDITIVE_MOL_CACHE: Dict[str, object] = {}
_ADDITIVE_MOLPROP_CACHE: Dict[str, Dict[str, float]] = {}


def _looks_like_accelerator(mol) -> bool:
    """促进/催化效应粗判：叔胺（含吡啶）、鑄盐离子液体、季鏻、碱金属盐。"""
    if mol is None:
        return False
    try:
        for patt_s in ("[NX3;H0](-[#6])-[#6]", "[n+]", "[N+;H0]", "[P+]",
                       "[Li+]", "[Na+]", "[K+]"):
            patt = Chem.MolFromSmarts(patt_s)
            if patt is not None and mol.HasSubstructMatch(patt):
                return True
    except Exception:
        return False
    return False


def _classify_additive_component(smiles: str, simulator=None) -> Tuple[str, str, bool]:
    """按官能团对添加剂分诊。

    Returns:
        (side, cls, is_accelerator)
        side: 'resin'（含环氧 → 树脂侧参与交联反应模拟）
              'curer'（含胺/酸酐/硫醇/酸/酚等活性氢 → 固化剂侧）
              'inert'（不进交联网络，只进加权描述符/Fox 项）
        cls: reactive_diluent / reactive_curer / flame_retardant / accelerator / inert
        is_accelerator: 是否具有促进催化效应
    """
    smi = str(smiles or "").strip()
    if not smi or smi.lower() in ("nan", "none", "<na>"):
        return "inert", "inert", False
    if smi in _ADDITIVE_MOL_CACHE:
        mol = _ADDITIVE_MOL_CACHE[smi]
    else:
        mol = None
        try:
            cleaned = smi
            if simulator is not None:
                cleaned = simulator._to_reactive_smiles(smi) or simulator._clean_smiles(smi) or smi
            mol = Chem.MolFromSmiles(cleaned) if cleaned else None
            if mol is None:
                try:
                    from core.smiles_utils import parse_chemical_string
                    mol = parse_chemical_string(smi, repair=True, keep_largest_frag=False)
                except Exception:
                    mol = None
        except Exception:
            mol = None
        _ADDITIVE_MOL_CACHE[smi] = mol
    if mol is None:
        return "inert", "inert", False

    try:
        epo_patt = Chem.MolFromSmarts("[C]1[O][C]1")
        n_epoxide = len(mol.GetSubstructMatches(epo_patt) or []) if epo_patt else 0
    except Exception:
        n_epoxide = 0
    is_acc = _looks_like_accelerator(mol)

    if n_epoxide > 0:
        return "resin", "reactive_diluent", is_acc

    curer_type = None
    if simulator is not None:
        try:
            curer_type, _ = simulator.detect_curer_type(smi)
        except Exception:
            curer_type = None

    # 促进/催化型优先按低掺量处理（如 DMP-30 兼有酚 OH，但实际掺量 1~5phr；
    # 若按固化剂侧典型 12phr 处理会触发不必要的封顶）
    if is_acc:
        return "inert", "accelerator", True

    if curer_type in ("amine", "anhydride", "thiol", "hydrazide", "acid", "phenol"):
        return "curer", "reactive_curer", False

    try:
        has_p = any(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
    except Exception:
        has_p = False
    if has_p:
        return "inert", "flame_retardant", False
    return "inert", "inert", False


def _additive_mol_props(smiles: str) -> Dict[str, float]:
    """惰性添加剂的分子描述符（带缓存）。"""
    if smiles in _ADDITIVE_MOLPROP_CACHE:
        return _ADDITIVE_MOLPROP_CACHE[smiles]
    props = {"mw": 0.0, "logp": 0.0, "tpsa": 0.0}
    try:
        mol = _ADDITIVE_MOL_CACHE.get(smiles)
        if mol is None:
            try:
                from core.smiles_utils import parse_chemical_string
                mol = parse_chemical_string(smiles, repair=True, keep_largest_frag=False)
            except Exception:
                mol = None
        if mol is not None:
            props["mw"] = float(Descriptors.MolWt(mol))
            try:
                props["logp"] = float(Descriptors.MolLogP(mol))
            except Exception:
                pass
            try:
                props["tpsa"] = float(Descriptors.TPSA(mol))
            except Exception:
                pass
    except Exception:
        pass
    _ADDITIVE_MOLPROP_CACHE[smiles] = props
    return props


def _collect_additive_components(row, wide_row, columns, simulator=None) -> Dict[str, Any]:
    """收集并分诊一行中的小分子添加剂。"""
    resin_add: List[Tuple[str, float]] = []
    curer_add: List[Tuple[str, float]] = []
    inert_add: List[Tuple[str, float, str]] = []
    accelerator_present = False
    weight_source = "none"
    n_reactive = 0
    for col in ([] if columns is None else columns):
        m = _ADDITIVE_STRUCT_COL_RE.match(str(col))
        if not m:
            continue
        try:
            smi = row.get(col)
        except Exception:
            smi = None
        if smi is None:
            continue
        try:
            if pd.isna(smi):
                continue
        except Exception:
            pass
        smi = str(smi).strip()
        if not smi or smi.lower() in ("nan", "none", "<na>"):
            continue

        side, cls, is_acc = _classify_additive_component(smi, simulator)
        if is_acc:
            accelerator_present = True

        # PHR：同前缀编号的 amount_phr / phr 列（优先宽表）
        prefix = str(col)[: -len("_structure")]
        weight = None
        for phr_col in (f"{prefix}_amount_phr", f"{prefix}_phr"):
            for src in (wide_row, row):
                if src is None:
                    continue
                try:
                    val = src.get(phr_col)
                except Exception:
                    val = None
                if val is None or (not isinstance(val, str) and pd.isna(val)):
                    continue
                try:
                    val = float(val)
                    if val > 0:
                        weight = val
                        break
                except Exception:
                    continue
            if weight is not None:
                break
        if weight is None:
            weight = _ADDITIVE_CLASS_PROFILE.get(cls, _ADDITIVE_CLASS_PROFILE["inert"])["default_phr"]
            if weight_source == "none":
                weight_source = "default"
        else:
            weight_source = "phr"

        if side == "resin":
            resin_add.append((smi, weight))
            n_reactive += 1
        elif side == "curer":
            curer_add.append((smi, weight))
            n_reactive += 1
        else:
            inert_add.append((smi, weight, cls))

    return {
        "resin": resin_add,
        "curer": curer_add,
        "inert": inert_add,
        "accelerator_present": accelerator_present,
        "weight_source": weight_source,
        "n_reactive": n_reactive,
    }


def _apply_additive_weights(add_info: Dict[str, Any], resin_base_phr: float,
                            curer_base_phr: float) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, Any]]:
    """添加剂权重封顶并生成添加剂特征字典。

    封顶规则：单侧添加剂合计 ≤ _ADDITIVE_MAX_SIDE_FRACTION × 该侧主组分权重和，
    超出按比例缩放（缺 PHR 等权兜底时的权重爆炸防护）。
    """
    features: Dict[str, Any] = {}
    resin_add = list(add_info.get("resin", []) or [])
    curer_add = list(add_info.get("curer", []) or [])
    inert_add = list(add_info.get("inert", []) or [])

    capped = False
    for items, base in ((resin_add, resin_base_phr), (curer_add, curer_base_phr)):
        if not items or base <= 0:
            continue
        total_add = sum(w for _, w in items)
        allowed = _ADDITIVE_MAX_SIDE_FRACTION / (1.0 - _ADDITIVE_MAX_SIDE_FRACTION) * base
        if total_add > allowed > 0:
            scale = allowed / total_add
            items[:] = [(s, w * scale) for s, w in items]
            capped = True

    # 惰性添加剂同样封顶（相对双侧主组分总量）
    base_total = resin_base_phr + curer_base_phr
    if inert_add and base_total > 0:
        total_inert = sum(w for _, w, _ in inert_add)
        allowed = _ADDITIVE_MAX_SIDE_FRACTION / (1.0 - _ADDITIVE_MAX_SIDE_FRACTION) * base_total
        if total_inert > allowed > 0:
            scale = allowed / total_inert
            inert_add = [(s, w * scale, c) for s, w, c in inert_add]
            capped = True

    total_phr = resin_base_phr + curer_base_phr \
        + sum(w for _, w in resin_add) + sum(w for _, w in curer_add) \
        + sum(w for _, w, _ in inert_add)

    inert_weight = sum(w for _, w, _ in inert_add)
    fox_term = 0.0
    mw_sum = logp_sum = tpsa_sum = 0.0
    for smi, w, cls in inert_add:
        if total_phr > 0:
            frac = w / total_phr
            tg_k = _ADDITIVE_CLASS_PROFILE.get(cls, _ADDITIVE_CLASS_PROFILE["inert"]).get("typical_tg_k")
            if tg_k:
                fox_term += frac / tg_k
        props = _additive_mol_props(smi)
        mw_sum += w * props["mw"]
        logp_sum += w * props["logp"]
        tpsa_sum += w * props["tpsa"]
    if inert_weight > 0:
        mw_sum /= inert_weight
        logp_sum /= inert_weight
        tpsa_sum /= inert_weight

    features = {
        "additive_n_reactive_resin": float(len(resin_add)),
        "additive_n_reactive_curer": float(len(curer_add)),
        "additive_n_inert": float(len(inert_add)),
        "additive_accelerator_present": 1.0 if add_info.get("accelerator_present") else 0.0,
        "additive_weight_source": 1.0 if add_info.get("weight_source") == "phr" else 0.0,
        "additive_weight_capped": 1.0 if capped else 0.0,
        "additive_inert_weight_fraction": float(inert_weight / total_phr) if total_phr > 0 else 0.0,
        "additive_fox_dilution_term": float(fox_term),
        "additive_inert_weighted_mw": float(mw_sum),
        "additive_inert_weighted_logp": float(logp_sum),
        "additive_inert_weighted_tpsa": float(tpsa_sum),
    }
    return resin_add, curer_add, features


# =============================================================================
# 核心类：环氧反应模拟器
# =============================================================================

class EpoxyReactionSimulator:
    """
    环氧树脂-固化剂反应模拟器
    
    功能：
    1. 识别环氧树脂中的环氧基数量
    2. 识别固化剂中的活性官能团类型和数量
    3. 模拟逐步固化反应
    4. 生成不同固化度下的反应产物SMILES
    
    使用示例:
    >>> simulator = EpoxyReactionSimulator()
    >>> epoxy_smiles = "C1OC1COc2ccc(C(C)(C)c3ccc(OCC4CO4)cc3)cc2"  # DGEBA
    >>> curer_smiles = "Nc1ccc(Cc2ccc(N)cc2)cc1"  # DDM (4,4'-MDA)
    >>> products = simulator.simulate_curing(epoxy_smiles, curer_smiles, n_reactions=2)
    """
    
    def __init__(self, verbose: bool = False):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for reaction simulation. Please install rdkit.")
        
        self.verbose = verbose
        self._compiled_patterns = {}
        self._compiled_reactions = {}
        # [重复率修复] 共聚网络单元缓存: (epoxy, curer, conv) -> unit_smiles
        self._copolymer_unit_cache = {}
        
        # 预编译SMARTS模式
        self._compile_patterns()
        # 预编译反应模板
        self._compile_reactions()
    
    def _compile_patterns(self):
        """预编译SMARTS模式"""
        for name, smarts in FUNCTIONAL_GROUP_PATTERNS.items():
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._compiled_patterns[name] = pat
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 无法编译SMARTS '{name}': {e}")
    
    def _compile_reactions(self):
        """预编译反应模板"""
        for template in ALL_REACTION_TEMPLATES:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(template.smirks)
                if rxn is not None:
                    self._compiled_reactions[template.name] = (rxn, template)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 无法编译反应 '{template.name}': {e}")
    
    def _clean_smiles(self, smiles: str) -> Optional[str]:
        """清洗并标准化SMILES"""
        if smiles is None or pd.isna(smiles):
            return None
        s = str(smiles).strip()
        if not s or s.lower() in {'nan', 'none', 'na', '<na>'}:
            return None
        
        # 处理聚合物占位符
        if '*' in s:
            s = re.sub(r"\[\s*\*\s*\]", "C", s)
            s = s.replace('*', 'C')
        
        # 转换为SMILES（支持SELFIES/BigSMILES）
        s = convert_to_smiles(s, fmt="auto") or s
        
        return s

    def _to_reactive_smiles(self, smiles: str) -> Optional[str]:
        """将SMILES或BigSMILES规范化为可供RDKit反应模拟的活性单体/低聚物SMILES"""
        if smiles is None or pd.isna(smiles):
            return None
        s = str(smiles).strip()
        if not s or s.lower() in {'nan', 'none', 'na', '<na>'}:
            return None

        # 1. 尝试直接被 RDKit 识别
        if RDKIT_AVAILABLE:
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    return s
            except Exception:
                pass

        # 2. BigSMILES 处理
        # [网络化重构] 优先随机图解析器：按拓扑采样出保留端基的具体寡聚体 SMILES，
        # 替代旧的硬编码字符串白名单（白名单外体系化学失真的根源）。
        if '{' in s or ('[' in s and ('>' in s or '<' in s)):
            try:
                import zlib as _zlib
                from core.bigsmiles_stochastic_graph import sample_bigsmiles_realizations
                _seed = _zlib.crc32(s.encode('utf-8')) & 0x7fffffff
                _reals = sample_bigsmiles_realizations(
                    s, n_samples=1, min_repeat_units=1, max_repeat_units=3, seed=_seed
                )
                if _reals:
                    _cand = str(_reals[0]).strip()
                    if RDKIT_AVAILABLE and _cand and Chem.MolFromSmiles(_cand) is not None:
                        return _cand
            except Exception:
                pass
            # 历史白名单兜底（图解析失败时的最后防线）
            if 'C(C)(C)' in s and ('c1' in s or 'c2' in s or 'c3' in s) and ('CO' in s or 'OCC' in s):
                return 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
            if 'c1ccc(C(C)(C)c2ccc(' in s or 'c2ccc(C(C)(C)c3ccc(' in s:
                return 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
            if 'S(=O)(=O)' in s and ('c1ccc(' in s or 'c2ccc(' in s):
                return 'O=S(=O)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
            if 'c1c(OCC2CO2)c(C)cc(' in s:
                return 'Cc1cc(OCC2CO2)ccc1'
            if 'c1c(O)c(C)cc(' in s:
                return 'Cc1c(O)cccc1'
            if 'c1ccc(Cc2ccc(' in s and 'N' in s:
                return 'Nc1ccc(Cc2ccc(N)cc2)cc1'
            try:
                converted = convert_to_smiles(s, fmt="auto")
                if converted and RDKIT_AVAILABLE and Chem.MolFromSmiles(converted) is not None:
                    return converted
            except Exception:
                pass

        # 3. 基础占位符清洗
        s_clean = self._clean_smiles(s)
        if s_clean and RDKIT_AVAILABLE and Chem.MolFromSmiles(s_clean) is not None:
            return s_clean

        return s
    
    def identify_functional_groups(self, smiles: str) -> Dict[str, int]:
        """
        识别分子中的官能团及其数量
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            Dict[str, int]: 官能团名称 -> 数量
        """
        s_act = self._to_reactive_smiles(smiles) or self._clean_smiles(smiles)
        if not s_act:
            return {}
        
        mol = Chem.MolFromSmiles(s_act)
        if mol is None:
            return {}
        
        results = {}
        for name, pat in self._compiled_patterns.items():
            matches = mol.GetSubstructMatches(pat)
            if matches:
                results[name] = len(matches)
        
        return results
    
    def get_epoxide_count(self, smiles: str) -> int:
        """获取环氧基数量"""
        fg = self.identify_functional_groups(smiles)
        return fg.get('epoxide', 0)
    
    def detect_curer_type(self, smiles: str) -> Tuple[str, Dict[str, int]]:
        """
        检测固化剂类型
        
        Returns:
            Tuple[str, Dict]: (主要类型, 官能团统计)
        """
        fg = self.identify_functional_groups(smiles)
        
        # 优先级判断
        if fg.get('hydrazide', 0) > 0:
            return 'hydrazide', fg
        if fg.get('isocyanate', 0) > 0:
            return 'isocyanate', fg
        if fg.get('anhydride', 0) > 0:
            return 'anhydride', fg
        if fg.get('thiol', 0) > 0:
            return 'thiol', fg
        if fg.get('primary_amine', 0) > 0 or fg.get('aromatic_amine', 0) > 0:
            return 'amine', fg
        if fg.get('secondary_amine', 0) > 0:
            return 'amine', fg
        if fg.get('phenol', 0) > 0:
            return 'phenol', fg
        if fg.get('carboxylic_acid', 0) > 0:
            return 'acid', fg
        
        return 'unknown', fg

    def estimate_conversion(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        stoichiometry_r: float,
        curing_temp: float = 150.0,
        curing_time: float = 2.0,
        use_typical_values: bool = True,
        accelerator_present: bool = False
    ) -> float:
        """
        估算固化反应的转化率

        基于：
        1. 化学计量比（AHEW/EEW）
        2. 官能度（凝胶化理论）
        3. 固化剂类型（经验值）
        4. 固化条件（温度、时间）

        Args:
            epoxy_smiles: 环氧树脂SMILES
            curer_smiles: 固化剂SMILES
            stoichiometry_r: 化学计量比 (活性氢当量/环氧当量 = AHEW/EEW)
            curing_temp: 固化温度 (°C)，默认150°C
            curing_time: 固化时间 (hours)，默认2小时
            use_typical_values: 是否使用典型经验值修正

        Returns:
            估算的转化率 (0-1)
        """
        # 1. 识别官能团
        epoxy_fg = self.identify_functional_groups(epoxy_smiles)
        curer_type, curer_fg = self.detect_curer_type(curer_smiles)

        epoxy_functionality = epoxy_fg.get('epoxide', 0)

        if curer_type == 'amine':
            # 伯胺可以反应2次，仲胺1次
            primary_amine = curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)
            secondary_amine = curer_fg.get('secondary_amine', 0)
            curer_functionality = primary_amine * 2 + secondary_amine
        elif curer_type == 'anhydride':
            curer_functionality = curer_fg.get('anhydride', 0)
        elif curer_type == 'thiol':
            curer_functionality = curer_fg.get('thiol', 0)
        elif curer_type == 'hydrazide':
            curer_functionality = curer_fg.get('hydrazide', 0) * 2
        elif curer_type == 'phenol':
            curer_functionality = max(1, curer_fg.get('phenol', 0))
        elif curer_type == 'acid':
            curer_functionality = max(1, curer_fg.get('carboxylic_acid', 0))
        elif curer_type == 'isocyanate':
            curer_functionality = max(1, curer_fg.get('isocyanate', 0))
        else:
            curer_functionality = 1

        if epoxy_functionality == 0 or curer_functionality == 0:
            return 0.0

        # 2. 化学计量比影响
        # 理论上 r = 1.0 时转化率最高
        if stoichiometry_r <= 0:
            return 0.0

        # 最优化学计量比下的理论最大转化率
        alpha_max_stoich = min(1.0, stoichiometry_r, 1.0 / stoichiometry_r)

        # 3. 凝胶化临界转化率（Flory-Stockmayer理论）
        # α_gel = 1 / sqrt((f_epoxy - 1) * (f_curer - 1))
        if epoxy_functionality > 1 and curer_functionality > 1:
            try:
                alpha_gel = 1.0 / np.sqrt((epoxy_functionality - 1) * (curer_functionality - 1))
            except Exception:
                alpha_gel = 0.5
        else:
            # 线性聚合物，无凝胶化
            alpha_gel = 0.0

        # 4. 固化剂类型的典型转化率（经验值）
        if use_typical_values:
            if curer_type == 'amine':
                # 胺类固化剂：反应活性高，转化率通常 80-95%
                base_conversion = 0.88
            elif curer_type == 'anhydride':
                # 酸酐：需要催化剂，转化率 70-85%
                base_conversion = 0.78
            elif curer_type == 'thiol':
                # 硫醇：快速反应，转化率 85-95%
                base_conversion = 0.90
            elif curer_type == 'hydrazide':
                # 酰肼：转化率 75-90%
                base_conversion = 0.82
            else:
                base_conversion = 0.75
        else:
            base_conversion = 0.85

        # [添加剂感知 P4] 促进剂/催化剂修正：叔胺/鑄盐/碱金属盐等提升表观转化率
        # （实验依据：叔胺促进酸酐体系约 +5~10%，胺体系 +2~4%）
        if accelerator_present:
            if curer_type == 'anhydride':
                _acc_boost = 0.08
            elif curer_type == 'amine':
                _acc_boost = 0.03
            else:
                _acc_boost = 0.04
            base_conversion = min(0.97, base_conversion + _acc_boost)

        # 5. 固化条件影响（简化的Arrhenius模型）
        # 参考条件：150°C, 2h → 达到基准转化率
        try:
            # 温度因子：Ea ≈ 50 kJ/mol (典型环氧-胺反应)
            temp_factor = np.exp(-6000 * (1.0 / (curing_temp + 273.15) - 1.0 / 423.15))
            temp_factor = np.clip(temp_factor, 0.5, 1.5)  # 限制在合理范围
        except Exception:
            temp_factor = 1.0

        try:
            # 时间因子：指数饱和模型
            # [修复] (150°C, 2h) 是参考态哨兵值而非真实数据：此时因子应为 1.0。
            # 否则缺工艺数据的行全部被 0.63 的假时间因子压到 ~50%
            # （实测 274/472 行同值 0.5563 = 0.88×1.0×0.6321）
            if abs(float(curing_temp) - 150.0) < 1e-9 and abs(float(curing_time) - 2.0) < 1e-9:
                time_factor = 1.0
            else:
                time_factor = 1.0 - np.exp(-float(curing_time) / 2.0)
                time_factor = np.clip(time_factor, 0.3, 1.0)
        except Exception:
            time_factor = 1.0

        # 6. 综合估算
        estimated_conversion = base_conversion * alpha_max_stoich * temp_factor * time_factor

        # 7. 物理约束
        # - 不能低于凝胶化点的50%（否则无法形成网络）
        # - 不能超过98%（总有未反应基团）
        min_conversion = max(0.1, alpha_gel * 0.5)
        max_conversion = 0.98

        estimated_conversion = np.clip(estimated_conversion, min_conversion, max_conversion)

        if self.verbose:
            print(f"📊 转化率估算:")
            print(f"   - 环氧官能度: {epoxy_functionality}")
            print(f"   - 固化剂官能度: {curer_functionality}")
            print(f"   - 固化剂类型: {curer_type}")
            print(f"   - 化学计量比 r: {stoichiometry_r:.3f}")
            print(f"   - 凝胶化临界点: {alpha_gel:.3f}")
            print(f"   - 基准转化率: {base_conversion:.3f}")
            print(f"   - 温度因子: {temp_factor:.3f}")
            print(f"   - 时间因子: {time_factor:.3f}")
            print(f"   - 估算转化率: {estimated_conversion:.3f}")

        return float(estimated_conversion)

    # ------------------------------------------------------------------
    # [转化率重构] DiBenedetto 玻璃化极限模型：低/中温固化的转化率由
    # "Tg 追上固化温度"封顶，后固化（更高温保温）再推一档。
    # 仅需数据里的 (最高反应温度, 是否后固化) 两个字段，与 PINN 物理层口径兼容。
    # ------------------------------------------------------------------
    # [活性标准] 每类体系的特征温度 T_char（标准工艺时长下达到平台转化率的温度）
    # 与平台转化率 plateau——由体系反应活性决定，而非统一模板
    _ACTIVITY_STANDARD = {
        # curer_type: (plateau, T_char °C)
        "anhydride_acc":   (0.95,  70),   # 酸酐+促进剂/催化剂：100-150°C 固化完全
        "anhydride":       (0.62, 105),   # 酸酐无促进剂：动力学受限，平台低且特征温度高
        "thiol":           (0.93,  55),   # 硫醇：点击型反应，低温即完全
        "amine_aliphatic": (0.90,  70),   # 脂肪胺：室温-80°C 即可固化
        "amine_aromatic":  (0.93, 110),   # 芳香胺（DDS/DDM）：需 120-180°C
        "phenol":          (0.90, 150),
        "hydrazide":       (0.92, 120),
        "default":         (0.92, 110),
    }

    def _detect_amine_subtype(self, curer_mol) -> str:
        """胺细分：芳香胺（N-H 邻接芳香环，如 DDS/DDM）vs 脂肪胺——活性差 3-5 倍"""
        try:
            patt = Chem.MolFromSmarts("[NX3;H2,H1;!$([n])](-[c])")
            if patt is not None:
                matches = curer_mol.GetSubstructMatches(patt)
                n_sites = len({m[0] for m in matches})
                total_h = sum(1 for a in curer_mol.GetAtoms()
                              if a.GetAtomicNum() == 7 and not a.GetIsAromatic()
                              and a.GetTotalNumHs() > 0)
                if total_h > 0 and n_sites >= total_h:
                    return "amine_aromatic"
        except Exception:
            pass
        return "amine_aliphatic"

    def _activity_standard(self, curer_mol, has_accelerator: bool):
        """按固化剂结构+促进剂返回 (plateau, T_char)"""
        curer_type = self._detect_curer_type_legacy(curer_mol)
        if curer_type == "amine":
            subtype = self._detect_amine_subtype(curer_mol)
            plateau, t_char = self._ACTIVITY_STANDARD.get(subtype, (0.90, 85))
            if has_accelerator:
                t_char -= 15  # 促进剂把特征温度拉低 15°C
            return plateau, max(45, t_char)
        if curer_type == "anhydride":
            plateau, t_char = self._ACTIVITY_STANDARD["anhydride_acc" if has_accelerator else "anhydride"]
            return plateau, t_char
        plateau, t_char = self._ACTIVITY_STANDARD.get(curer_type, self._ACTIVITY_STANDARD["default"])
        if has_accelerator:
            t_char -= 10
        return plateau, max(45, t_char)

    _VITRIFICATION_PARAMS = {
        # [标定] reach 按体系×有无促进剂取值（工业/DSC 共识）：
        #   酸酐无促进剂在合理工艺时间内仅 0.40-0.60（促进剂是决定性开关 → 0.90-0.98）
        #   芳香胺无促进剂 0.85-0.90（150°C+ 需长时），有促进剂 0.95-0.98
        #   全套工艺（促进剂+催化剂+后固化）齐全的配方按领域判断应达 0.90-0.95
        "amine":     {"tg0": -25.0, "tg_inf": 175.0, "reach_no_acc": 0.93, "reach_acc": 0.97},
        "anhydride": {"tg0": -15.0, "tg_inf": 165.0, "reach_no_acc": 0.62, "reach_acc": 1.00},
        "thiol":     {"tg0": -30.0, "tg_inf": 150.0, "reach_no_acc": 0.90, "reach_acc": 0.95},
        "phenol":    {"tg0": -10.0, "tg_inf": 200.0, "reach_no_acc": 0.85, "reach_acc": 0.93},
        "hydrazide": {"tg0": -20.0, "tg_inf": 185.0, "reach_no_acc": 0.88, "reach_acc": 0.94},
        "default":   {"tg0": -20.0, "tg_inf": 185.0, "reach_no_acc": 0.90, "reach_acc": 0.95},
    }
    _LAMBDA_DIBENEDETTO = 0.45


    @staticmethod
    def _detect_accelerator_signal(row, wide_row, df_cols, wide_cols) -> Optional[bool]:
        """从行/宽表的促进剂、催化剂计数列识别促进剂信号。

        检测列：accelerator_count / accelerator_component_count / catalyst_count /
        catalyst_component_count（任一 > 0 即视为有促进/催化体系）。
        返回 True/False；无任何相关列时返回 None（调用方回退结构检测）。
        """
        candidates = ("accelerator_count", "accelerator_component_count",
                      "catalyst_count", "catalyst_component_count")
        found = False
        for col in candidates:
            for src_cols, src in ((df_cols, row), (wide_cols, wide_row)):
                if src_cols is not None and col in src_cols and src is not None:
                    try:
                        v = src.get(col)
                        if v is None or pd.isna(v):
                            continue
                        found = True  # 至少读到一个有效计数值
                        if float(v) > 0:
                            return True
                    except Exception:
                        continue
        # 列存在且计数全为 0 → 明确无促进剂(False)；列不存在 → None(回退结构检测)
        return False if found else None

    @staticmethod
    def _sanitize_t_max(t_max_c) -> Optional[float]:
        """最高反应温度脏数据防护：可解析且落在 [25, 260]°C 才有效，否则 None。

        实测数据存在 727°C 等录入错误（真实固化 ≤250°C）；越界值不参与估算。
        """
        try:
            t = float(t_max_c)
        except (TypeError, ValueError):
            return None
        if np.isnan(t) or not np.isfinite(t):
            return None
        if 25.0 <= t <= 260.0:
            return t
        return None

    @staticmethod
    def _parse_post_cure(v) -> bool:
        """是否后固化字段解析：兼容 bool / 0-1 / 是否 / true-false 文本。"""
        if v is None:
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) >= 0.5
        s = str(v).strip().lower()
        return s in {"1", "1.0", "true", "yes", "y", "是", "有", "后固化"}

    def _detect_curer_type_legacy(self, mol) -> str:
        """mol 对象版固化剂类型检测（胺/酸酐/硫醇/酚/肼/其他）"""
        try:
            fg = {}
            fg['primary_amine'] = len(mol.GetSubstructMatches(self._site_patt("[NX3;H2;!$([n]);!$([NX3]-[CX3]=[OX1])]")))
            fg['secondary_amine'] = len(mol.GetSubstructMatches(self._site_patt("[NX3;H1;!$([n]);$(N(-[#6])-[#6]);!$([NX3]-[CX3]=[OX1])]")))
            fg['anhydride'] = len(mol.GetSubstructMatches(self._site_patt("[CX3](=[OX1])[OX2][CX3](=[OX1])")))
            fg['thiol'] = len(mol.GetSubstructMatches(self._site_patt("[SX2;H1]")))
            fg['phenol'] = len(mol.GetSubstructMatches(self._site_patt("[OX2;H1][c]")))
            fg['carboxylic_oh'] = len(mol.GetSubstructMatches(self._site_patt("[OX2;H1][CX3]=[OX1]")))
            if fg['anhydride'] > 0:
                return "anhydride"
            if fg['thiol'] > 0:
                return "thiol"
            if (fg['primary_amine'] + fg['secondary_amine']) > 0:
                return "amine"
            if fg['phenol'] > 0:
                return "phenol"
            return "unknown"
        except Exception:
            return "unknown"

    def quantify_system_activity(self, epoxy_mol, curer_mol, has_accelerator: bool = False) -> Dict[str, float]:
        """量化体系反应活性（分子级），并据此给出特征温度与平台转化率。

        活性 = 固化剂亲核性(nuc) + 环氧亲电性(elec) + 位阻(steric) + 催化/促进剂
          - 亲核性：活性氢杂原子的 Gasteiger 电荷（越负越强）+ 位点类型基准 + N 取代位阻
          - 亲电性：环氧碳 Gasteiger 电荷（越正越易被进攻）+ 端位环氧加成
          - 特征温度：T_char = T_base(体系类型) − 40(nuc−0.5) − 40(elec−0.5) + 15(位阻)
        返回 {nuc, elec, steric, activity, t_char, plateau}
        """
        out = {"nuc": 0.5, "elec": 0.5, "steric": 0.0, "activity": 0.5,
               "t_char": 130.0, "plateau": 0.92}
        try:
            curer_type = self._detect_curer_type_legacy(curer_mol)
            plateau, t_base = self._activity_standard(curer_mol, has_accelerator)

            # ---- 固化剂亲核性 ----
            site_base = {"thiol": 1.00, "primary_amine": 0.85, "secondary_amine": 0.68,
                         "phenol_oh": 0.35, "carboxylic_oh": 0.25, "anhydride": 0.30}
            best, best_atom = 0.0, None
            for nm, smarts, _t, _w in self._H_SITE_SPECS:
                patt = self._site_patt(smarts)
                if patt is None:
                    continue
                ms = curer_mol.GetSubstructMatches(patt)
                if ms and site_base.get(nm, 0.4) > best:
                    best = site_base.get(nm, 0.4)
                    best_atom = ms[0][0]
            nuc = best if best > 0 else 0.35
            if best_atom is not None:
                try:
                    AllChem.ComputeGasteigerCharges(curer_mol)
                    q = float(curer_mol.GetAtomWithIdx(best_atom).GetProp("_GasteigerCharge"))
                    if np.isfinite(q):
                        # 典型杂原子电荷 -0.6(强) ~ -0.1(弱) → 映射到 ±0.15 修正
                        nuc += float(np.clip((-0.35 - q) * 0.6, -0.15, 0.15))
                except Exception:
                    pass
            # N 取代位阻：取代基多 → 活性降
            if curer_type == "amine" and best_atom is not None:
                try:
                    heavy_nb = sum(1 for nb in curer_mol.GetAtomWithIdx(best_atom).GetNeighbors()
                                   if nb.GetAtomicNum() > 1)
                    nuc -= 0.06 * max(0, heavy_nb - 1)
                except Exception:
                    pass

            # ---- 环氧亲电性 ----
            elec = 0.5
            try:
                epi_patt = self._site_patt(self._EPOXY_SITE_SMARTS)
                ms = epoxy_mol.GetSubstructMatches(epi_patt)
                if ms:
                    AllChem.ComputeGasteigerCharges(epoxy_mol)
                    qs, n_terminal = [], 0
                    for (a1, o, a2) in ms:
                        for a in (a1, a2):
                            q = float(epoxy_mol.GetAtomWithIdx(a).GetProp("_GasteigerCharge"))
                            if np.isfinite(q):
                                qs.append(q)
                        # 端位判断：环氧碳上 H 数（2 = 端位 CH2，更易被进攻）
                        if epoxy_mol.GetAtomWithIdx(a1).GetTotalNumHs() == 2 or                            epoxy_mol.GetAtomWithIdx(a2).GetTotalNumHs() == 2:
                            n_terminal += 1
                    if qs:
                        q_mean = float(np.mean(qs))
                        elec = float(np.clip((q_mean - 0.02) / 0.20, 0.0, 1.0))
                    terminal_frac = n_terminal / max(1, len(ms))
                    elec += 0.15 * (terminal_frac - 0.5)
            except Exception:
                terminal_frac = 0.5
            else:
                terminal_frac = locals().get("terminal_frac", 0.5)

            steric = float(np.clip(1.0 - terminal_frac, 0.0, 1.0))
            activity = 0.45 * nuc + 0.40 * elec + 0.15 * (1.0 - steric)
            t_char = t_base - 30.0 * (nuc - 0.5) - 30.0 * (elec - 0.5) + 15.0 * steric
            t_char = float(np.clip(t_char, 45.0, 210.0))
            out.update({"nuc": round(float(nuc), 4), "elec": round(float(elec), 4),
                        "steric": round(float(steric), 4), "activity": round(float(activity), 4),
                        "t_char": round(t_char, 1), "plateau": round(float(plateau), 4)})
        except Exception:
            pass
        return out

    def estimate_conversion_from_activity(
        self,
        epoxy_mol,
        curer_mol,
        t_max_c: float,
        post_cure: bool = False,
        has_accelerator: bool = False,
        stoichiometry_r: float = 1.0,
    ) -> float:
        """[活性标准] 由体系反应活性确定特征温度，再按固化温度给转化率。

        α = plateau × sigmoid((T_max − T_char + 25) / 25) × 计量比上限 + 后固化
        - plateau/T_char 由固化剂类型与结构（芳香胺 vs 脂肪胺）+ 促进剂决定
        - 计量比守恒上限 min(1, r, 1/r) 照常折减
        """
        try:
            act = self.quantify_system_activity(epoxy_mol, curer_mol, has_accelerator)
            plateau, t_char = act["plateau"], act["t_char"]
            x = (float(t_max_c) - t_char + 35.0) / 30.0
            sig = 1.0 / (1.0 + np.exp(-x))
            alpha = plateau * sig
            r_eff = float(stoichiometry_r) if stoichiometry_r and float(stoichiometry_r) > 0 else 1.0
            alpha *= min(1.0, r_eff, 1.0 / r_eff)
            if post_cure:
                alpha += 0.05
            return float(np.clip(alpha, 0.10, 0.95))
        except Exception:
            return 0.50

    def estimate_conversion_vitrification(
        self,
        curer_type: str,
        t_max_c: float,
        post_cure: bool = False,
        has_accelerator: bool = False,
    ) -> float:
        """由 (最高反应温度, 是否后固化, 是否有促进剂) 估算转化率。

        DiBenedetto: Tg(α)=Tg0+(Tg∞−Tg0)·λα/(1−(1−λ)α)，反解 Tg(α)=T_max 得玻璃化封顶；
        后固化 +0.06，夹到 [0.30, 0.97]。
        促进剂不改变玻璃化天花板（热力学），只改变动力学可达程度（reach）：
        酸酐无促进剂时低温动力学显著受限，温度越高限制越小。
        """
        params = self._VITRIFICATION_PARAMS.get(curer_type, self._VITRIFICATION_PARAMS["default"])
        lam = self._LAMBDA_DIBENEDETTO
        b = float(t_max_c) - params["tg0"]
        denom = (params["tg_inf"] - params["tg0"]) * lam + (1.0 - lam) * b
        if denom <= 0:
            return 0.30
        alpha = b / denom  # 玻璃化天花板
        # [标定修复] 动力学可达系数按"体系 × 有无促进剂"取值：
        #   酸酐无促进剂 0.62（合理工艺时间内仅 0.40-0.60，促进剂是决定性开关）
        #   胺无促进剂 0.93（150°C+ 热运动可达 0.85-0.90）；有促进剂 +2~30 个百分点
        # 总上限 0.93：实测 DSC 最终转化率鲜有 >0.92
        reach = params["reach_acc"] if has_accelerator else params["reach_no_acc"]
        alpha *= reach
        if post_cure:
            alpha += 0.06  # 后固化：高温段解除玻璃化冻结，转化率再推进一档
        return float(np.clip(alpha, 0.30, 0.95))

    def estimate_conversion_multicomponent(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        stoichiometry_r: float,
        curing_temp: float = 150.0,
        curing_time: float = 2.0,
        accelerator_present: bool = False,
        t_max_c: Optional[float] = None,
        post_cure: bool = False,
        has_accelerator_hint: Optional[bool] = None,
    ) -> float:
        """
        估算多组分体系的转化率

        Args:
            resin_components: [(smiles, weight), ...] 树脂组分列表
                weight 为质量分数或摩尔分数（归一化到和为1）
            curer_components: [(smiles, weight), ...] 固化剂组分列表
            stoichiometry_r: 总体化学计量比 (AHEW/EEW)
            curing_temp: 固化温度 (°C)
            curing_time: 固化时间 (hours)

        Returns:
            估算的转化率 (0-1)
        """
        if not resin_components or not curer_components:
            return 0.0

        # 归一化权重
        total_resin_weight = sum(w for _, w in resin_components)
        total_curer_weight = sum(w for _, w in curer_components)

        if total_resin_weight == 0 or total_curer_weight == 0:
            return 0.0

        resin_components = [(smi, w / total_resin_weight) for smi, w in resin_components]
        curer_components = [(smi, w / total_curer_weight) for smi, w in curer_components]

        # [转化率重构] 优先玻璃化模型：数据提供有效最高反应温度时，转化率由
        # (T_max, 后固化) 决定（DiBenedetto 封顶），化学计量比仍作上限折减。
        # 脏温度（<25 或 >260°C，如实测 727°C）已被 _sanitize_t_max 过滤，自动回退旧查表。
        t_eff = self._sanitize_t_max(t_max_c) if t_max_c is not None else None
        if t_eff is not None:
            try:
                main_resin_smi = max(resin_components, key=lambda x: x[1])[0]
                main_curer_smi = max(curer_components, key=lambda x: x[1])[0]
                e_mol = Chem.MolFromSmiles(self._to_reactive_smiles(main_resin_smi) or main_resin_smi)
                c_mol = Chem.MolFromSmiles(self._to_reactive_smiles(main_curer_smi) or main_curer_smi)
                if e_mol is None or c_mol is None:
                    raise ValueError("parse_fail")
                acc_signal = has_accelerator_hint if has_accelerator_hint is not None else bool(accelerator_present)
                r_eff = float(stoichiometry_r) if stoichiometry_r and float(stoichiometry_r) > 0 else 1.0
                alpha = self.estimate_conversion_from_activity(
                    e_mol, c_mol, t_eff, bool(post_cure), acc_signal, r_eff)
                return float(np.clip(alpha, 0.10, 0.95))
            except Exception:
                pass  # 异常时回退旧查表路径

        # 1. 计算加权平均官能度
        weighted_epoxy_func = 0.0
        weighted_curer_func = 0.0
        weighted_base_conversion = 0.0

        for resin_smi, resin_weight in resin_components:
            epoxy_fg = self.identify_functional_groups(resin_smi)
            epoxy_func = epoxy_fg.get('epoxide', 0)
            weighted_epoxy_func += epoxy_func * resin_weight

        for curer_smi, curer_weight in curer_components:
            curer_type, curer_fg = self.detect_curer_type(curer_smi)

            if curer_type == 'amine':
                primary = curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)
                secondary = curer_fg.get('secondary_amine', 0)
                curer_func = primary * 2 + secondary
                base_conv = 0.88
            elif curer_type == 'anhydride':
                curer_func = curer_fg.get('anhydride', 0)
                base_conv = 0.78
            elif curer_type == 'thiol':
                curer_func = curer_fg.get('thiol', 0)
                base_conv = 0.90
            elif curer_type == 'hydrazide':
                curer_func = curer_fg.get('hydrazide', 0) * 2
                base_conv = 0.82
            else:
                curer_func = 1
                base_conv = 0.75

            weighted_curer_func += curer_func * curer_weight
            weighted_base_conversion += base_conv * curer_weight

        if weighted_epoxy_func == 0 or weighted_curer_func == 0:
            return 0.0

        # [添加剂感知 P4] 促进剂/催化剂提升表观基准转化率（加权体系取保守中值）
        if accelerator_present:
            weighted_base_conversion = min(0.97, weighted_base_conversion + 0.05)

        # 2. 化学计量比影响
        alpha_max_stoich = min(1.0, stoichiometry_r, 1.0 / stoichiometry_r) if stoichiometry_r > 0 else 0.0

        # 3. 凝胶化临界转化率
        if weighted_epoxy_func > 1 and weighted_curer_func > 1:
            try:
                alpha_gel = 1.0 / np.sqrt((weighted_epoxy_func - 1) * (weighted_curer_func - 1))
            except Exception:
                alpha_gel = 0.5
        else:
            alpha_gel = 0.0

        # 4. 固化条件影响
        try:
            temp_factor = np.exp(-6000 * (1.0 / (curing_temp + 273.15) - 1.0 / 423.15))
            temp_factor = np.clip(temp_factor, 0.5, 1.5)
        except Exception:
            temp_factor = 1.0

        try:
            # [修复] (150°C, 2h) 哨兵值=参考态，时间因子取 1.0（详见 estimate_conversion）
            if abs(float(curing_temp) - 150.0) < 1e-9 and abs(float(curing_time) - 2.0) < 1e-9:
                time_factor = 1.0
            else:
                time_factor = 1.0 - np.exp(-float(curing_time) / 2.0)
                time_factor = np.clip(time_factor, 0.3, 1.0)
        except Exception:
            time_factor = 1.0

        # 5. 综合估算
        estimated_conversion = weighted_base_conversion * alpha_max_stoich * temp_factor * time_factor

        # 6. 物理约束
        min_conversion = max(0.1, alpha_gel * 0.5)
        max_conversion = 0.98
        estimated_conversion = np.clip(estimated_conversion, min_conversion, max_conversion)

        if self.verbose:
            print(f"📊 多组分转化率估算:")
            print(f"   - 树脂组分数: {len(resin_components)}")
            print(f"   - 固化剂组分数: {len(curer_components)}")
            print(f"   - 加权环氧官能度: {weighted_epoxy_func:.2f}")
            print(f"   - 加权固化剂官能度: {weighted_curer_func:.2f}")
            print(f"   - 化学计量比 r: {stoichiometry_r:.3f}")
            print(f"   - 凝胶化临界点: {alpha_gel:.3f}")
            print(f"   - 加权基准转化率: {weighted_base_conversion:.3f}")
            print(f"   - 估算转化率: {estimated_conversion:.3f}")

        return float(estimated_conversion)

    def simulate_multicomponent_reaction(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        target_conversion: float = 0.5,
        method: str = 'weighted'
    ) -> Dict[str, any]:
        """
        模拟多组分反应（方案1：加权平均法 + 方案2：组合反应法）

        Args:
            resin_components: [(smiles, weight), ...] 树脂组分
            curer_components: [(smiles, weight), ...] 固化剂组分
            target_conversion: 目标转化率
            method: 'weighted' (快速) 或 'combinatorial' (准确)

        Returns:
            Dict包含:
                - 'method': 使用的方法
                - 'products': 产物列表
                - 'weighted_features': 加权平均特征
                - 'representative_smiles': 代表性SMILES
                - 'representative_bigsmiles': 代表性BigSMILES
        """
        if not resin_components or not curer_components:
            return {'method': method, 'products': [], 'weighted_features': {}}

        # 归一化权重
        total_resin_weight = sum(w for _, w in resin_components)
        total_curer_weight = sum(w for _, w in curer_components)

        if total_resin_weight == 0 or total_curer_weight == 0:
            return {'method': method, 'products': [], 'weighted_features': {}}

        resin_components = [(smi, w / total_resin_weight) for smi, w in resin_components]
        curer_components = [(smi, w / total_curer_weight) for smi, w in curer_components]

        if method == 'weighted':
            # 方案1：加权平均法（快速）
            return self._simulate_weighted_average(resin_components, curer_components, target_conversion)
        elif method == 'combinatorial':
            # 方案2：组合反应法（准确）
            return self._simulate_combinatorial(resin_components, curer_components, target_conversion)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # [重复率修复] 组成区分特征 + 多组分共聚网络单元辅助方法
    # ------------------------------------------------------------------
    _MIN_COMPONENT_WEIGHT = 0.05  # 参与共聚网络单元的最小组分质量分数

    def _composition_discriminator_features(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """行间组成区分特征：即使主反应对相同，也能量化各行组成的差异。"""
        import math
        r_ws = [w for _, w in resin_components if w > 0]
        c_ws = [w for _, w in curer_components if w > 0]
        r_active = [w for w in r_ws if w >= self._MIN_COMPONENT_WEIGHT]
        c_active = [w for w in c_ws if w >= self._MIN_COMPONENT_WEIGHT]

        def _entropy(ws):
            total = sum(ws) or 1.0
            return float(-sum((w / total) * math.log(w / total) for w in ws if w > 0)) if len(ws) > 1 else 0.0

        all_ws = r_ws + c_ws
        return {
            'n_active_resin_components': float(len(r_active) if r_active else (1 if r_ws else 0)),
            'n_active_curer_components': float(len(c_active) if c_active else (1 if c_ws else 0)),
            'top_resin_fraction': float(max(r_ws) if r_ws else 0.0),
            'top_curer_fraction': float(max(c_ws) if c_ws else 0.0),
            'n_reaction_pairs': float(min(len(r_active) or 1, len(c_active) or 1) * max(len(r_active) or 1, len(c_active) or 1)),
            'composition_entropy': _entropy(all_ws),
        }

    def _composition_seed(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
    ) -> int:
        """行组成确定性种子：同组成 -> 同抽样（可复现）；不同权重 -> 不同抽样 -> 结构可区分。"""
        import hashlib
        canon = (
            sorted((str(smi)[:200], round(float(w), 4)) for smi, w in resin_components if w > 0),
            sorted((str(smi)[:200], round(float(w), 4)) for smi, w in curer_components if w > 0),
        )
        return int(hashlib.sha1(repr(canon).encode('utf-8')).hexdigest()[:12], 16)

    def _build_copolymer_bigsmiles(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        main_resin_smi: str,
        main_curer_smi: str,
        target_conversion: float,
        max_units: int = 4,
    ) -> Optional[str]:
        """构造多组分随机共聚网络单元 BigSMILES。

        将权重达阈的 (树脂_i × 主固化剂) 与 (主树脂 × 固化剂_j) 网络单元并入
        同一 BigSMILES 随机对象的 repeat-unit 候选（本项目解析器约定：候选间用 ','，
        end-groups 用 ';'），使次要组分的行间差异体现在网络拓扑候选上：
            {[$]unit_main[$],[$]unit_alt1[$],[$]unit_alt2[$]}
        """
        if not RDKIT_AVAILABLE:
            return None
        units: List[str] = []
        seen: set = set()

        def _unit_for(epoxy_smi: str, curer_smi: str) -> Optional[str]:
            # 返回单个网络单元的 BigSMILES 文本：优先内联描述符（连接端数可变、
            # 位置在真实连接原子上），失败时退回旧的两端 [$] 形式
            key = (str(epoxy_smi), str(curer_smi), round(float(target_conversion), 3))
            cached = self._copolymer_unit_cache.get(key)
            if cached is not None:
                return cached if cached != '' else None
            try:
                pack = self.build_network_unit_full(
                    epoxy_smi, curer_smi, target_conversion=float(target_conversion)
                )
            except Exception:
                pack = None
            if pack and pack[0]:
                unit_text = pack[2] if (pack[1] >= 2 and pack[2]) else f"[$]{pack[0]}[$]"
            else:
                try:
                    r_act = self._to_reactive_smiles(epoxy_smi) or self._clean_smiles(epoxy_smi)
                    c_act = self._to_reactive_smiles(curer_smi) or self._clean_smiles(curer_smi)
                    vc = self._virtual_crosslink_safe(r_act, c_act)
                except Exception:
                    vc = None
                unit_text = f"[$]{vc}[$]" if vc else None
            self._copolymer_unit_cache[key] = unit_text or ''
            return unit_text

        # 主对优先
        main_unit = _unit_for(main_resin_smi, main_curer_smi)
        if main_unit and main_unit not in seen:
            seen.add(main_unit)
            units.append(main_unit)

        # 主要固化剂 × 主树脂
        for curer_smi, w in sorted(curer_components, key=lambda x: -x[1]):
            if len(units) >= max_units:
                break
            if curer_smi == main_curer_smi or w < self._MIN_COMPONENT_WEIGHT:
                continue
            u = _unit_for(main_resin_smi, curer_smi)
            if u and u not in seen:
                seen.add(u)
                units.append(u)

        # 次要树脂 × 主固化剂
        for resin_smi, w in sorted(resin_components, key=lambda x: -x[1]):
            if len(units) >= max_units:
                break
            if resin_smi == main_resin_smi or w < self._MIN_COMPONENT_WEIGHT:
                continue
            u = _unit_for(resin_smi, main_curer_smi)
            if u and u not in seen:
                seen.add(u)
                units.append(u)

        if not units:
            return None
        # units 内每个单元已自带连接描述符（内联形式或多官能度不足时
        # 的两端 [$] 形式），这里只做 BigSMILES 随机共聚对象的拼接，
        # 不再重复包一层 [$]，避免描述符数量翻倍。
        if len(units) == 1:
            return f"{{{units[0]}}}"
        joined = ",".join(units)
        return f"{{{joined}}}"

    def _simulate_weighted_average(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        target_conversion: float
    ) -> Dict[str, any]:
        """
        方案1：加权平均法

        选择主要组分进行反应，其他组分的特征加权平均；
        [重复率修复] BigSMILES 升级为按行内权重构造的随机共聚网络单元。
        """
        # 找到主要组分（权重最大的）
        main_resin_smi, main_resin_weight = max(resin_components, key=lambda x: x[1])
        main_curer_smi, main_curer_weight = max(curer_components, key=lambda x: x[1])

        # 计算加权平均的化学计量比
        weighted_epoxy_func = sum(
            self.identify_functional_groups(smi).get('epoxide', 0) * w
            for smi, w in resin_components
        )

        weighted_curer_func = 0.0
        for smi, w in curer_components:
            curer_type, curer_fg = self.detect_curer_type(smi)
            if curer_type == 'amine':
                func = (curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)) * 2 + \
                       curer_fg.get('secondary_amine', 0)
            elif curer_type == 'anhydride':
                func = curer_fg.get('anhydride', 0)
            elif curer_type == 'thiol':
                func = curer_fg.get('thiol', 0)
            else:
                func = 1
            weighted_curer_func += func * w

        stoich_r = weighted_curer_func / weighted_epoxy_func if weighted_epoxy_func > 0 else 0.0

        # [网络化重构] 系综撒网生长：K 个随机种子 → medoid 代表 SMILES + 系综统计；
        # 引擎不可用时回退 get_product_representation 旧路径
        ens = None
        try:
            ens = self.build_network_unit_ensemble(
                main_resin_smi,
                main_curer_smi,
                target_conversion=target_conversion,
                stoichiometry=stoich_r,
                n_samples=8,
            )
        except Exception as _e_ens:
            if self.verbose:
                print(f"⚠️ 系综生长失败，回退旧路径: {_e_ens}")
            ens = None
        if ens is not None:
            product_repr = {
                'representation_type': 'smiles',
                'smiles': ens['representative_smiles'],
                'bigsmiles': None,
                'conversion': target_conversion,
                'stoichiometry': stoich_r,
                'description': (
                    f'系综撒网生长交联簇（{ens["n_valid_samples"]} 样本 medoid，'
                    f'实际转化率 {ens["ensemble"]["realized_conversion_mean"]*100:.0f}%）'
                ),
            }
            ens_stats = ens['ensemble']
        else:
            product_repr = self.get_product_representation(
                main_resin_smi,
                main_curer_smi,
                stoichiometry=stoich_r,
                target_conversion=target_conversion,
                output_format='auto'
            )
            ens_stats = None

        # [重复率修复] 按行内权重构造随机共聚网络单元，使次要组分差异体现在网络拓扑上
        copolymer_bigsmiles = None
        try:
            copolymer_bigsmiles = self._build_copolymer_bigsmiles(
                resin_components, curer_components,
                main_resin_smi, main_curer_smi, target_conversion
            )
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 共聚网络单元构造失败，回退主对 BigSMILES: {e}")

        return {
            'method': 'weighted',
            'main_resin': main_resin_smi,
            'main_curer': main_curer_smi,
            'main_resin_weight': main_resin_weight,
            'main_curer_weight': main_curer_weight,
            'weighted_epoxy_functionality': weighted_epoxy_func,
            'weighted_curer_functionality': weighted_curer_func,
            'stoichiometry_r': stoich_r,
            'product_representation': product_repr,
            'representative_smiles': product_repr.get('smiles'),
            'representative_bigsmiles': product_repr.get('bigsmiles'),
            'copolymer_bigsmiles': copolymer_bigsmiles,
            'composition_features': self._composition_discriminator_features(resin_components, curer_components),
            'ensemble_stats': ens_stats,
        }

    def _simulate_combinatorial(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        target_conversion: float
    ) -> Dict[str, any]:
        """
        方案2：组合反应法

        模拟所有可能的反应对，按概率加权
        """
        products = []
        total_prob = 0.0

        for resin_smi, resin_weight in resin_components:
            for curer_smi, curer_weight in curer_components:
                # 反应概率 = resin_weight × curer_weight
                prob = resin_weight * curer_weight

                if prob < 0.01:  # 忽略概率太小的组合
                    continue

                # 计算该组合的化学计量比
                epoxy_func = self.identify_functional_groups(resin_smi).get('epoxide', 0)
                curer_type, curer_fg = self.detect_curer_type(curer_smi)

                if curer_type == 'amine':
                    curer_func = (curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)) * 2 + \
                                 curer_fg.get('secondary_amine', 0)
                elif curer_type == 'anhydride':
                    curer_func = curer_fg.get('anhydride', 0)
                elif curer_type == 'thiol':
                    curer_func = curer_fg.get('thiol', 0)
                else:
                    curer_func = 1

                stoich_r = curer_func / epoxy_func if epoxy_func > 0 else 0.0

                # 生成该组合的产物
                try:
                    product_repr = self.get_product_representation(
                        resin_smi,
                        curer_smi,
                        stoichiometry=stoich_r,
                        target_conversion=target_conversion,
                        output_format='auto'
                    )

                    products.append({
                        'resin_smiles': resin_smi,
                        'curer_smiles': curer_smi,
                        'probability': prob,
                        'product_representation': product_repr,
                        'product_smiles': product_repr.get('smiles'),
                        'product_bigsmiles': product_repr.get('bigsmiles')
                    })

                    total_prob += prob

                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ 组合反应失败: {e}")
                    continue

        # 归一化概率
        if total_prob > 0:
            for p in products:
                p['probability'] /= total_prob

        # [重复率修复] 代表选择：由固定 max 概率改为按行组成的确定性概率抽样，
        # 使次要组分占比不同的行能抽到不同反应对，从而得到可区分的网络结构；
        # 同一组成的行始终抽中同一组合（可复现、缓存友好）。
        if products:
            try:
                import random as _random
                _rng = _random.Random(self._composition_seed(resin_components, curer_components))
                _pick = _rng.random()
                _acc = 0.0
                representative = products[-1]
                for p in sorted(products, key=lambda x: -x['probability']):
                    _acc += float(p.get('probability', 0.0))
                    if _acc >= _pick:
                        representative = p
                        break
            except Exception:
                representative = max(products, key=lambda x: x['probability'])
            representative_smiles = representative['product_smiles']
            representative_bigsmiles = representative['product_bigsmiles']
        else:
            representative = None
            representative_smiles = None
            representative_bigsmiles = None

        # [重复率修复] top-2 组合的共聚网络单元 BigSMILES
        copolymer_bigsmiles = None
        try:
            main_resin_smi, main_resin_weight = max(resin_components, key=lambda x: x[1])
            main_curer_smi, main_curer_weight = max(curer_components, key=lambda x: x[1])
            copolymer_bigsmiles = self._build_copolymer_bigsmiles(
                resin_components, curer_components,
                main_resin_smi, main_curer_smi, target_conversion
            )
        except Exception:
            copolymer_bigsmiles = None

        return {
            'method': 'combinatorial',
            'products': products,
            'n_combinations': len(products),
            'representative_smiles': representative_smiles,
            'representative_bigsmiles': representative_bigsmiles,
            'copolymer_bigsmiles': copolymer_bigsmiles,
            'composition_features': self._composition_discriminator_features(resin_components, curer_components),
            'sampled_pair': ({'resin_smiles': representative['resin_smiles'], 'curer_smiles': representative['curer_smiles'], 'probability': representative['probability']} if representative else None),
        }

    def _run_single_reaction(
        self, 
        epoxy_mol: Chem.Mol, 
        curer_mol: Chem.Mol, 
        rxn_name: str
    ) -> List[Chem.Mol]:
        """
        执行单次反应
        
        Returns:
            List[Mol]: 产物分子列表
        """
        if rxn_name not in self._compiled_reactions:
            return []
        
        rxn, template = self._compiled_reactions[rxn_name]
        
        try:
            # 尝试反应
            products = rxn.RunReactants((epoxy_mol, curer_mol))
            
            valid_products = []
            # [性能] 大簇上 RunReactants 会枚举几十个位点匹配，每个产物都是全分子
            # 拷贝 + SanitizeMol（大簇时 ~50ms/个）——封顶 8 个足够随机选择使用
            for prod_tuple in products:
                # 关键修复：当反应生成多个产物片段（如脱除小分子或开环分离产物）时，
                # 仅保留重原子数最多的主交联产物，坚决剔除 CCCC=O、C=CC=O 等游离副产物
                if len(prod_tuple) > 1:
                    main_prod = max(prod_tuple, key=lambda m: m.GetNumHeavyAtoms() if m else 0)
                    try:
                        Chem.SanitizeMol(main_prod)
                        valid_products.append(main_prod)
                    except Exception:
                        continue
                else:
                    for prod in prod_tuple:
                        try:
                            # 清理和标准化产物
                            Chem.SanitizeMol(prod)
                            valid_products.append(prod)
                        except Exception:
                            continue
                if len(valid_products) >= 8:
                    break
            
            return valid_products
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 反应执行失败: {e}")
            return []
    
    def simulate_single_step(
        self, 
        epoxy_smiles: str, 
        curer_smiles: str
    ) -> List[str]:
        """
        模拟单步反应
        
        Args:
            epoxy_smiles: 环氧树脂SMILES
            curer_smiles: 固化剂SMILES
            
        Returns:
            List[str]: 产物SMILES列表
        """
        epoxy_smiles = self._clean_smiles(epoxy_smiles)
        curer_smiles = self._clean_smiles(curer_smiles)
        
        if not epoxy_smiles or not curer_smiles:
            return []
        
        epoxy_mol = Chem.MolFromSmiles(epoxy_smiles)
        curer_mol = Chem.MolFromSmiles(curer_smiles)
        
        if epoxy_mol is None or curer_mol is None:
            return []
        
        # 检测固化剂类型
        curer_type, _ = self.detect_curer_type(curer_smiles)
        
        # 选择合适的反应模板
        products = []
        
        if curer_type == 'amine':
            # 先尝试伯胺反应
            prods = self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_primary_amine')
            if prods:
                products.extend(prods)
            else:
                # 再尝试仲胺反应
                prods = self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_secondary_amine')
                products.extend(prods)
                
        elif curer_type == 'anhydride':
            prods = self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_anhydride_5')
            if not prods:
                prods = self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_anhydride_6')
            if not prods:
                prods = self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_anhydride')
            products.extend(prods)
            
        elif curer_type == 'thiol':
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_thiol'))
            
        elif curer_type == 'hydrazide':
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_hydrazide'))
        elif curer_type == 'phenol':
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_phenol'))
        elif curer_type == 'acid':
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_carboxylic_acid'))
        elif curer_type == 'isocyanate':
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_isocyanate'))
        else:
            # 针对未知类型或特殊固化剂，尝试主要反应模板
            for t_name in ['epoxy_primary_amine', 'epoxy_secondary_amine', 'epoxy_anhydride_5', 'epoxy_anhydride_6', 'epoxy_anhydride', 'epoxy_phenol', 'epoxy_thiol', 'epoxy_carboxylic_acid']:
                prods = self._run_single_reaction(epoxy_mol, curer_mol, t_name)
                if prods:
                    products.extend(prods)
                    break
        
        # 转换为SMILES并按分子重原子数/长度降序排序，确保主产物位于首位
        product_smiles = []
        for prod in products:
            try:
                smi = Chem.MolToSmiles(prod)
                if smi:
                    product_smiles.append((smi, prod.GetNumHeavyAtoms()))
            except Exception:
                continue
        
        # 按重原子数降序排序后去重保序
        seen = set()
        sorted_smiles = []
        for smi, _ in sorted(product_smiles, key=lambda x: x[1], reverse=True):
            if smi not in seen:
                seen.add(smi)
                sorted_smiles.append(smi)
        
        return sorted_smiles
    
    def simulate_curing(
        self, 
        epoxy_smiles: str, 
        curer_smiles: str, 
        n_reactions: int = 1,
        max_products: int = 10
    ) -> List[Dict]:
        """
        模拟多步固化反应
        
        Args:
            epoxy_smiles: 环氧树脂SMILES
            curer_smiles: 固化剂SMILES
            n_reactions: 反应步数（近似对应固化度）
            max_products: 最大产物数量
            
        Returns:
            List[Dict]: 包含产物信息的列表
        """
        results = []
        
        # 规范化单体（支持BigSMILES与聚合物格式）
        epoxy_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
        curer_act = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)
        if not epoxy_act or not curer_act:
            return []

        # 初始反应物
        current_products = [(epoxy_act, 0)]  # (SMILES, 反应步数)
        
        for step in range(n_reactions):
            next_products = []
            
            for product, prev_step in current_products[:max_products]:
                # 检查是否还有环氧基
                epoxide_count = self.get_epoxide_count(product)
                if epoxide_count == 0:
                    # 无环氧基，保留当前产物
                    next_products.append((product, prev_step))
                    continue
                
                # 执行反应
                new_prods = self.simulate_single_step(product, curer_smiles)
                
                if new_prods:
                    for p in new_prods[:3]:  # 限制每步产物数
                        next_products.append((p, step + 1))
                else:
                    next_products.append((product, prev_step))
            
            current_products = next_products
        
        # 整理结果
        for prod_smi, n_step in current_products:
            try:
                mol = Chem.MolFromSmiles(prod_smi)
                if mol is None:
                    continue
                
                fg = self.identify_functional_groups(prod_smi)
                
                results.append({
                    'smiles': prod_smi,
                    'reaction_steps': n_step,
                    'remaining_epoxide': fg.get('epoxide', 0),
                    'hydroxyl_count': fg.get('hydroxyl', 0),
                    'mol_weight': Descriptors.MolWt(mol),
                    'num_atoms': mol.GetNumAtoms(),
                })
            except Exception:
                continue
        
        return results
    
    def _count_reactive_h(self, smiles: str) -> int:
        # 统计活性氢：伯胺/仲胺N-H、硫醇S-H、酚OH、羧酸OH、酸酐等效位点
        if not RDKIT_AVAILABLE:
            return 0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            n = 0
            n += 2 * len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2;!$([n]);!$([NX3]-[CX3]=[OX1])]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H1;!$([n]);$(N(-[#6])-[#6]);!$([NX3]-[CX3]=[OX1])]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SX2;H1]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2;H1][c]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2;H1]')))
            # 酸酐环每个环开环可反应2次（第1次成单酯单酸，第2次游离酸与环氧反应成二酯）
            n += 2 * len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])')))
            return n
        except Exception:
            return 0

    def build_network_unit(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        target_conversion: float = 0.85,
        max_reactions: int = 8,
        stoichiometry: float = 1.0
    ) -> Optional[str]:
        """化学计量驱动的交联网络枢纽片段构建（向后兼容入口）。

        返回代表性单元 SMILES；残留环氧已开环为邻二醇封端
        （“反应后”表示不再含环氧基）。
        """
        pack = self.build_network_unit_full(
            epoxy_smiles, curer_smiles, target_conversion, max_reactions,
            stoichiometry=stoichiometry
        )
        return pack[0] if pack else None

    def build_network_unit_full(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        target_conversion: float = 0.85,
        max_reactions: int = 8,
        stoichiometry: float = 1.0,
        seed: int = 0,
        atom_budget: int = 450,
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """构建交联单元并返回连接点信息。

        [网络化重构] 优先使用池化随机撒网生长引擎 `_grow_network_cluster`：
          多枢纽分子池 + 活性加权随机位点配对反应 + 化学计量比 r=H/epoxy 驱动池组成，
          悬环氧/悬 N-H/O-H 按物理端点保留（废除"水解式"封端与悬键甲基化）。
        引擎不可用（枢纽无活性氢如异氰酸酯、反应模板不匹配等）时自动回退
        旧贪心链增长路径 `_build_network_unit_impl`。

        Args:
            stoichiometry: 化学计量比 r（活性氢/环氧基），决定池内环氧单体数
            seed: 随机种子（系综采样时传入不同 seed）
            atom_budget: 单簇重原子预算，超出冻结生长

        Returns:
            (单元SMILES, 连接点数, 带内联[$]描述符的BigSMILES单元文本)
            失败时返回 None
        """
        if not RDKIT_AVAILABLE:
            return None

        r_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
        c_act = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)
        if not r_act or not c_act:
            return None

        try:
            a_conv = max(0.0, min(1.0, float(target_conversion) if target_conversion is not None else 0.85))
        except Exception:
            a_conv = 0.85

        try:
            r_eff = float(stoichiometry) if stoichiometry and float(stoichiometry) > 0 else 1.0
        except Exception:
            r_eff = 1.0

        cache_key = (r_act, c_act, round(a_conv, 3), round(r_eff, 3), int(seed), int(atom_budget))
        cache = getattr(self, '_unit_cache', None)
        if cache is None:
            cache = {}
            self._unit_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        # [网络化重构] 优先撒网生长引擎，失败回退旧贪心链增长
        unit_pack = None
        try:
            grown = self._grow_network_cluster(r_act, c_act, a_conv, stoich_r=r_eff,
                                               seed=seed, atom_budget=atom_budget)
            if grown is not None:
                unit_pack = grown['pack']
        except Exception:
            unit_pack = None
        if unit_pack is None:
            unit_pack = self._build_network_unit_impl(r_act, c_act, a_conv, max_reactions)
        cache[cache_key] = unit_pack
        return unit_pack

    def _build_network_unit_impl(
        self,
        r_act: str,
        c_act: str,
        a_conv: float,
        max_reactions: int
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        try:
            h_total = self._count_reactive_h(c_act)

            if h_total <= 0:
                # 枢纽无活性氢（异氰酸酯/未知类型等）：单步或虚拟交联保底
                prods = self.simulate_single_step(r_act, c_act)
                if prods:
                    return self._finalize_network_unit(prods[0])
                return self._finalize_network_unit(
                    self._virtual_crosslink_safe(r_act, c_act)
                )

            # 种子反应：1 个环氧单体 + 1 个枢纽固化剂
            seed_prods = self.simulate_single_step(r_act, c_act)
            if not seed_prods:
                return self._finalize_network_unit(
                    self._virtual_crosslink_safe(r_act, c_act)
                )

            product = seed_prods[0]
            reactions_done = 1

            n_target = min(max(1, int(round(a_conv * h_total))), max_reactions)

            while reactions_done < n_target:
                h_left = self._count_reactive_h(product)
                if h_left <= 0:
                    break
                # 产物作为固化剂侧（携带剩余N-H/O-H/S-H），新鲜环氧单体作为环氧侧
                new_prods = self.simulate_single_step(r_act, product)
                if not new_prods:
                    break
                # 选择接入新单体的链增长产物（SMILES最长 = 原子最多）
                product = max(new_prods, key=lambda s: (len(s), s))
                reactions_done += 1

            return self._finalize_network_unit(product)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # [网络化重构] 池化随机撒网生长引擎
    # 与旧版"贪心链增长"的区别：
    #   1. 分子池：n_hub 个固化剂枢纽 + n 个环氧单体（n 由化学计量比 r=H/epoxy 决定）
    #   2. 每步按活性权重随机挑 (环氧侧片段, 供体位点类型) 配对反应，
    #      RunReactants 的多匹配再随机挑一个 → 支臂随机生长、枢纽间桥接，拓扑自然成网
    #   3. 悬环氧 / 悬 N-H / O-H 原样保留（物理端点，废除"水解式"封端与甲基封端）
    #   4. 原子预算：单簇超限冻结生长（记录 truncated），预算内保留尽可能大的网络簇
    # ------------------------------------------------------------------
    _EPOXY_SITE_SMARTS = "[C]1[O][C]1"
    _H_SITE_SPECS = (
        # (位点类型, SMARTS, 反应模板名, 相对活性权重)
        ("primary_amine", "[NX3;H2;!$([n]);!$([NX3]-[CX3]=[OX1])]", "epoxy_primary_amine", 3.0),
        ("secondary_amine", "[NX3;H1;!$([n]);$(N(-[#6])-[#6]);!$([NX3]-[CX3]=[OX1])]", "epoxy_secondary_amine", 1.0),
        ("thiol", "[SX2;H1]", "epoxy_thiol", 2.0),
        # [关键修复] 酸酐：活性位点不是 X-H 键（羰基 C 被进攻、酐键断裂），
        # 缺失此位点会导致酸酐体系零反应、产物退化为环氧单体本身
        ("anhydride", "[CX3](=[OX1])[OX2][CX3](=[OX1])", "epoxy_anhydride_5", 1.0),
        ("phenol_oh", "[OX2;H1][c]", "epoxy_phenol", 0.5),
        # 羧酸：SMARTS 以 O 开头，保证手术式开环时 match[0] 是羟基 O（而非羰基 C）
        ("carboxylic_oh", "[OX2;H1][CX3]=[OX1]", "epoxy_carboxylic_acid", 0.5),
        ("aliphatic_oh", "[OX2;H1][CX4]", "epoxy_etherification", 0.3),
    )
    # 手术式开环仅适用 X-H 加成化学；酸酐开环需断酐 C-O-C 键，只走跨片段 SMARTS 模板
    _SURGERY_OK = {"primary_amine", "secondary_amine", "thiol", "phenol_oh",
                   "carboxylic_oh", "aliphatic_oh"}
    # 酸酐环大小多变：5 元失败依次回退 6 元/通用模板
    # [修复] 通用模板已重写为原子守恒版（酯 + 羧酸双片段），纳入回退链
    _TMPL_FALLBACK = {"epoxy_anhydride_5": ("epoxy_anhydride_6", "epoxy_anhydride")}

    def _site_patt(self, smarts: str):
        """SMARTS 编译缓存"""
        cache = getattr(self, "_site_patt_cache", None)
        if cache is None:
            cache = {}
            self._site_patt_cache = cache
        patt = cache.get(smarts)
        if patt is None:
            patt = Chem.MolFromSmarts(smarts)
            cache[smarts] = patt
        return patt

    def _site_meta(self) -> Dict[str, Tuple[str, float]]:
        """位点类型 -> (反应模板名, 活性权重)"""
        meta = getattr(self, "_h_site_meta", None)
        if meta is None:
            meta = {nm: (tmpl, w) for nm, _s, tmpl, w in self._H_SITE_SPECS}
            self._h_site_meta = meta
        return meta

    def _enumerate_site_counts(self, mol) -> Dict[str, int]:
        """统计分子上各类活性氢位点数量"""
        out = {}
        for nm, smarts, _tmpl, _w in self._H_SITE_SPECS:
            patt = self._site_patt(smarts)
            out[nm] = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
        return out

    def _enumerate_site_atoms(self, mol) -> Dict[str, list]:
        """统计分子上各类活性氢位点的原子索引（用于分子内环化的手术式成键）"""
        out = {}
        for nm, smarts, _tmpl, _w in self._H_SITE_SPECS:
            patt = self._site_patt(smarts)
            out[nm] = [m[0] for m in mol.GetSubstructMatches(patt)] if patt is not None else []
        return out

    def _cross_fragment_addition(self, epoxy_mol, donor_mol, site_type: str, rng):
        """跨片段手术式环氧开环加成（与 SMARTS 模板产物拓扑一致，O(1) 成本）。

        模板化学：环氧 O 留在位阻大的 C 上成 β-OH，位阻小的 C 与供体杂原子成新键。
        氢计数由 RDKit 隐式价键在 SanitizeMol 时自动结算。
        仅适用 X-H 加成类型（胺/酚/羧酸/醇/硫醇）；酸酐仍走 SMARTS 模板。
        返回合并后的新片段（RWMol 已 Sanitize）或 None。
        """
        try:
            smarts_map = {nm: sm for nm, sm, _t, _w in self._H_SITE_SPECS}
            patt = self._site_patt(smarts_map.get(site_type, ""))
            if patt is None:
                return None
            rw = Chem.RWMol(Chem.CombineMols(epoxy_mol, donor_mol))
            off = epoxy_mol.GetNumAtoms()
            epi_patt = self._site_patt(self._EPOXY_SITE_SMARTS)
            # 环氧环必须在环氧侧（atom idx < off）
            epi_matches = [m for m in rw.GetSubstructMatches(epi_patt) if m[0] < off and m[2] < off]
            # 供体位点必须在供体侧（atom idx >= off）
            donor_idxs = [m[0] + off for m in donor_mol.GetSubstructMatches(patt)]
            if not epi_matches or not donor_idxs:
                return None
            pairs = [(a1, o, a2, d) for (a1, o, a2) in epi_matches for d in donor_idxs]
            a1, o, a2, d = pairs[rng.randrange(len(pairs))]

            def _hvy(n):
                return sum(1 for nb in rw.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1)

            att = a1 if _hvy(a1) <= _hvy(a2) else a2  # 攻位阻小的 C
            if rw.GetBondBetweenAtoms(o, att) is None:
                return None
            rw.RemoveBond(o, att)
            rw.AddBond(d, att, Chem.BondType.SINGLE)
            dat = rw.GetAtomWithIdx(d)
            if dat.GetNumExplicitHs() > 0:
                dat.SetNumExplicitHs(max(0, dat.GetNumExplicitHs() - 1))
            Chem.SanitizeMol(rw)
            return rw
        except Exception:
            return None

    def _surgery_open_ring(self, mol, site_type: str, rng):
        """片段内环化：同一分子上悬环氧 + 活性氢位点直接开环成键（图距离≥5 防小环张力）。

        化学拓扑与 SMARTS 模板产物一致：O 留在位阻大的 C 上成 β-OH，
        位阻小的 C（CH2）与供体杂原子成新键；H 计数由 RDKit 隐式价键自动结算。
        返回新 RWMol(已 Sanitize) 或 None。
        """
        try:
            rw = Chem.RWMol(mol)
            epi_matches = rw.GetSubstructMatches(self._site_patt(self._EPOXY_SITE_SMARTS))
            smarts_map = {nm: sm for nm, sm, _t, _w in self._H_SITE_SPECS}
            patt = self._site_patt(smarts_map.get(site_type, ""))
            if patt is None:
                return None
            donor_idxs = [m[0] for m in rw.GetSubstructMatches(patt)]
            if not epi_matches or not donor_idxs:
                return None
            dm = Chem.GetDistanceMatrix(rw)
            pairs = []
            for (a1, o, a2) in epi_matches:
                for d in donor_idxs:
                    if d in (a1, o, a2):
                        continue
                    if min(dm[d][a1], dm[d][a2]) >= 5:
                        pairs.append((a1, o, a2, d))
            if not pairs:
                return None
            a1, o, a2, d = pairs[rng.randrange(len(pairs))]

            def _hvy(n):
                return sum(1 for nb in rw.GetAtomWithIdx(n).GetNeighbors() if nb.GetAtomicNum() > 1)

            # 攻击位阻小的 C（CH2），O 留在位阻大的 C 上成羟基
            if _hvy(a1) <= _hvy(a2):
                att = a1
            else:
                att = a2
            if rw.GetBondBetweenAtoms(o, att) is None:
                return None
            rw.RemoveBond(o, att)
            rw.AddBond(d, att, Chem.BondType.SINGLE)
            dat = rw.GetAtomWithIdx(d)
            if dat.GetNumExplicitHs() > 0:
                dat.SetNumExplicitHs(max(0, dat.GetNumExplicitHs() - 1))
            Chem.SanitizeMol(rw)
            return rw
        except Exception:
            return None

    @staticmethod
    def _weighted_choice(rng, weights: Dict[object, float]):
        total = sum(w for w in weights.values() if w > 0)
        if total <= 0:
            return None
        x = rng.random() * total
        acc = 0.0
        last = None
        for k, w in weights.items():
            if w <= 0:
                continue
            acc += w
            last = k
            if x <= acc:
                return k
        return last

    def _grow_network_cluster(
        self,
        r_act: str,
        c_act: str,
        a_conv: float,
        stoich_r: float = 1.0,
        seed: int = 0,
        atom_budget: int = NETWORK_SAMPLING["atom_budget"],
        n_hub: int = NETWORK_SAMPLING["n_hub"],
        max_pool_units: int = NETWORK_SAMPLING["max_pool_units"],
    ) -> Optional[Dict[str, any]]:
        """单次撒网生长。返回 {'stats': {...}, 'pack': (smiles, n_attach, None)} 或 None。"""
        import random as _random
        if not RDKIT_AVAILABLE:
            return None
        hub0 = Chem.MolFromSmiles(c_act)
        epoxy0 = Chem.MolFromSmiles(r_act)
        if hub0 is None or epoxy0 is None:
            return None
        h_per_hub = self._count_reactive_h(c_act)
        if h_per_hub <= 0:
            return None  # 枢纽无活性氢（异氰酸酯等），交由调用方回退旧路径

        rng = _random.Random(int(seed) * 7919 + 13)
        ep_hvy = max(1, epoxy0.GetNumHeavyAtoms())
        epi_patt0 = self._site_patt(self._EPOXY_SITE_SMARTS)
        # 单体携带的环氧基数（DGEBA=2、单官能稀释剂=1）：r 语义必须基于环氧基团数
        epi_per_mol = max(1, len(epoxy0.GetSubstructMatches(epi_patt0)))

        # ---- 池规模：r = 活性氢总数 / 环氧基团总数 ----
        r_eff = float(stoich_r) if stoich_r and float(stoich_r) > 0 else 1.0
        r_eff = min(max(r_eff, 0.05), 20.0)
        n_hub = max(1, int(n_hub))
        h_total = h_per_hub * n_hub
        n_epoxy = int(round(h_total / r_eff / epi_per_mol))
        n_epoxy = max(n_hub, min(n_epoxy, max_pool_units))
        # 池总原子预算约束（允许池约为单簇预算的 2.2 倍，凝胶簇约占大头）
        n_epoxy = min(n_epoxy, max(1, int((atom_budget * 2.2 - n_hub * hub0.GetNumHeavyAtoms()) / ep_hvy)))
        n_epoxy = max(1, n_epoxy)
        epoxy_groups_total = n_epoxy * epi_per_mol

        react_target = int(round(a_conv * epoxy_groups_total))
        # [性能上限] 60 步反应已足以生成代表性网络簇（簇重原子 ≈ 预算上限）；
        # 不设上限时多官能度体系可放大到 240 步，每步在 450 原子簇上做
        # RunReactants+SanitizeMol，单行可达数分钟
        react_target = max(0, min(react_target, h_total, 60))

        frags = [Chem.Mol(hub0) for _ in range(n_hub)] + [Chem.Mol(epoxy0) for _ in range(n_epoxy)]
        pool_hvy0 = sum(f.GetNumHeavyAtoms() for f in frags)
        epi_patt = self._site_patt(self._EPOXY_SITE_SMARTS)
        tert_patt = self._site_patt("[NX3;H0](-[#6])(-[#6])-[#6]")

        reacted = 0
        truncated = False
        fail_streak = 0
        meta = self._site_meta()
        # 每种位点类型的 SMARTS（供体原子枚举用）
        site_smarts = {nm: sm for nm, sm, _t, _w in self._H_SITE_SPECS}
        while reacted < react_target:
            sizes = [(i, f.GetNumHeavyAtoms()) for i, f in enumerate(frags) if f is not None]
            active = [i for i, sz in sizes if sz <= atom_budget]
            if not active:
                truncated = True
                break

            # ---- 反应选项：跨片段 SMARTS 配对 + 片段内环化（手术式），按权重随机 ----
            options = []  # (kind, data, weight)

            # (a) 跨片段：环氧侧片段 × 供体片段
            epi_counts = {}
            donor_sites_all = {}
            for i in active:
                c = len(frags[i].GetSubstructMatches(epi_patt))
                if c > 0:
                    epi_counts[i] = float(c)
                sites_i = self._enumerate_site_counts(frags[i])
                w_i = sum(cnt * meta[nm][1] for nm, cnt in sites_i.items() if cnt > 0 and nm in meta)
                if w_i > 0:
                    donor_sites_all[i] = (sites_i, w_i)
            for i, ce in epi_counts.items():
                for j, (sites_j, wj) in donor_sites_all.items():
                    if j == i:
                        continue
                    options.append(('cross', (i, j), ce * wj))

            # (b) 片段内环化：同片段上 (悬环氧, 活性H) 图距离≥5 的配对
            #     权重打 0.4 折（分子内环化概率低于分子间碰撞），但保证高转化率可达
            for i in active:
                epis_i = frags[i].GetSubstructMatches(epi_patt)
                if not epis_i:
                    continue
                sites_atoms = self._enumerate_site_atoms(frags[i])
                try:
                    dm = Chem.GetDistanceMatrix(frags[i])
                except Exception:
                    continue
                for nm, (tmpl, w) in meta.items():
                    if nm not in self._SURGERY_OK:
                        continue  # 酸酐等非 X-H 位点：手术式开环化学不适用
                    cnt_ok = 0
                    for d_idx in sites_atoms.get(nm, []):
                        for (a1, o, a2) in epis_i:
                            if d_idx in (a1, o, a2):
                                continue
                            if min(dm[d_idx][a1], dm[d_idx][a2]) >= 5:
                                cnt_ok += 1
                    if cnt_ok > 0:
                        options.append(('intra', (i, nm), cnt_ok * w * 0.4))

            if not options:
                break
            option_w = {idx: opt[2] for idx, opt in enumerate(options)}
            pick = self._weighted_choice(rng, option_w)
            if pick is None:
                break
            kind, data, _w = options[pick]

            if kind == 'cross':
                epi_idx, donor_idx = data
                sites_j, _wj = donor_sites_all[donor_idx]
                type_w = {nm: cnt * meta[nm][1] for nm, cnt in sites_j.items()
                          if cnt > 0 and nm in meta and nm in self._SURGERY_OK}
                if type_w:
                    # [提速] X-H 加成类型走 O(1) 跨片段手术（替代 RunReactants 全枚举：
                    # 大簇上 RunReactants 构建几十个全分子拷贝是单步 1-2s 的主因）
                    site_type = self._weighted_choice(rng, type_w)
                    if site_type is not None:
                        new_mol = self._cross_fragment_addition(
                            frags[epi_idx], frags[donor_idx], site_type, rng)
                        if new_mol is not None:
                            frags[epi_idx] = None
                            frags[donor_idx] = new_mol
                            reacted += 1
                            fail_streak = 0
                        else:
                            fail_streak += 1
                    else:
                        fail_streak += 1
                else:
                    # 酸酐等非 X-H 位点：保留 SMARTS 模板路径（含回退链）
                    site_type = self._weighted_choice(
                        rng, {nm: cnt * meta[nm][1] for nm, cnt in sites_j.items() if cnt > 0 and nm in meta})
                    if site_type is None:
                        fail_streak += 1
                    else:
                        tmpl_name = meta[site_type][0]
                        prods = self._run_single_reaction(frags[epi_idx], frags[donor_idx], tmpl_name)
                        if not prods:
                            for _fb in self._TMPL_FALLBACK.get(tmpl_name, ()):
                                prods = self._run_single_reaction(frags[epi_idx], frags[donor_idx], _fb)
                                if prods:
                                    break
                        if prods:
                            new_frag = prods[rng.randrange(len(prods))]
                            frags[epi_idx] = None
                            frags[donor_idx] = new_frag
                            reacted += 1
                            fail_streak = 0
                        else:
                            fail_streak += 1
            else:
                fi, site_type = data
                new_mol = self._surgery_open_ring(frags[fi], site_type, rng)
                if new_mol is not None:
                    frags[fi] = new_mol
                    reacted += 1
                    fail_streak = 0
                else:
                    fail_streak += 1
            if fail_streak >= 4:
                break

        frags = [f for f in frags if f is not None]
        if not frags:
            return None
        main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(main)
        except Exception:
            pass
        smi = Chem.MolToSmiles(main)
        if not smi:
            return None

        sites_main = self._enumerate_site_counts(main)
        n_epi_left = len(main.GetSubstructMatches(epi_patt))
        n_h_left = sum(cnt for nm, cnt in sites_main.items() if nm in meta)
        n_tert = len(main.GetSubstructMatches(tert_patt))

        stats = {
            'smiles': smi,
            'heavy_atoms': float(main.GetNumHeavyAtoms()),
            'mw': float(Descriptors.MolWt(main)),
            'n_hub': float(n_hub),
            'n_epoxy_total': float(epoxy_groups_total),
            'n_reacted': float(reacted),
            'realized_conversion': float(reacted) / max(1.0, float(epoxy_groups_total)),
            'n_dangling_epoxide': float(n_epi_left),
            'dangling_epoxide_frac': float(n_epi_left) / max(1.0, float(epoxy_groups_total)),
            'n_dangling_h': float(n_h_left),
            'dangling_h_frac': float(n_h_left) / max(1.0, float(h_total)),
            'n_tertiary_amine': float(n_tert),
            'gel_frac': float(main.GetNumHeavyAtoms()) / max(1, pool_hvy0),
            'truncated': 1.0 if truncated else 0.0,
        }
        # 兼容旧三元组接口：(clean_smiles, n_attach, marked)
        n_attach = int(n_epi_left + n_h_left)
        return {'stats': stats, 'pack': (smi, n_attach, None)}

    def build_network_unit_ensemble(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        target_conversion: float = 0.85,
        stoichiometry: float = 1.0,
        n_samples: int = 8,
        atom_budget: int = 450,
    ) -> Optional[Dict[str, any]]:
        """K 个随机种子的系综撒网生长。

        Returns:
            dict(representative_smiles, n_attach, ensemble={mw_mean, mw_std, ...}, n_valid_samples)
            representative_smiles 取 MW 中位数样本（medoid 简化）
        """
        r_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
        c_act = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)
        if not r_act or not c_act:
            return None
        try:
            a_conv = max(0.0, min(1.0, float(target_conversion) if target_conversion is not None else 0.85))
        except Exception:
            a_conv = 0.85

        cache_key = (r_act, c_act, round(a_conv, 3), round(float(stoichiometry or 1.0), 3), int(n_samples))
        cache = getattr(self, "_ens_cache", None)
        if cache is None:
            cache = {}
            self._ens_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        samples = []
        for k in range(max(1, int(n_samples))):
            try:
                st = self._grow_network_cluster(r_act, c_act, a_conv, stoich_r=stoichiometry,
                                                seed=k, atom_budget=atom_budget)
            except Exception:
                st = None
            if st:
                samples.append(st)
        if not samples:
            cache[cache_key] = None
            return None

        import numpy as _np
        def _ms(key):
            vals = _np.array([s['stats'][key] for s in samples], dtype=float)
            return float(vals.mean()), float(vals.std())

        mw_mean, mw_std = _ms('mw')
        ha_mean, _ha_std = _ms('heavy_atoms')
        rc_mean, _rc_std = _ms('realized_conversion')
        de_mean, de_std = _ms('dangling_epoxide_frac')
        dh_mean, _dh_std = _ms('dangling_h_frac')
        gf_mean, gf_std = _ms('gel_frac')
        ta_mean, ta_std = _ms('n_tertiary_amine')

        # [修复3] 系综涨落必须取自"拓扑量"——簇大小由转化率决定（物理恒定），
        # 真实涨落在支化模式/环密度/桥键数上。逐样本算拓扑描述符后统计。
        _topo_keys = ('product_ring_density', 'product_junction_site_density',
                      'product_ester_bonds_abs', 'product_ether_bonds_abs',
                      'product_network_bridge_density', 'product_rotatable_bond_ratio')
        _topo_vals = {k: [] for k in _topo_keys}
        for s in samples:
            try:
                _d = _compute_product_descriptors(s['stats']['smiles'], include_3d=False)
                for k in _topo_keys:
                    v = _d.get(k)
                    if v is not None and _np.isfinite(v):
                        _topo_vals[k].append(float(v))
            except Exception:
                continue
        topo_stats = {}
        for k, vals in _topo_vals.items():
            if len(vals) >= 2:
                arr = _np.array(vals, dtype=float)
                short = k.replace('product_', '')
                topo_stats[f'{short}_mean'] = float(arr.mean())
                topo_stats[f'{short}_std'] = float(arr.std())

        mws = [s['stats']['mw'] for s in samples]
        med_idx = int(_np.argsort(mws)[len(mws) // 2])
        rep = samples[med_idx]

        ensemble = {
            'mw_mean': mw_mean, 'mw_std': mw_std,
            'heavy_atoms_mean': ha_mean,
            'realized_conversion_mean': rc_mean,
            'dangling_epoxide_frac_mean': de_mean, 'dangling_epoxide_frac_std': de_std,
            'dangling_h_frac_mean': dh_mean,
            'gel_frac_mean': gf_mean, 'gel_frac_std': gf_std,
            'n_tertiary_amine_mean': ta_mean, 'n_tertiary_amine_std': ta_std,
            'n_samples': float(len(samples)),
        }
        # [修复3] 合并拓扑涨落统计（网络构型异质性，大小之外的真正系综信息）
        ensemble.update(topo_stats)
        out = {
            'representative_smiles': rep['stats']['smiles'],
            'n_attach': rep['pack'][1],
            'ensemble': ensemble,
            'n_valid_samples': len(samples),
        }
        cache[cache_key] = out
        return out

    def _finalize_network_unit(
        self, product_smiles: Optional[str]
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        """残留环氧开环封端 + 计算真实连接点数 + 生成内联描述符文本。

        Returns:
            (干净单元SMILES, 连接点数, 内联[$]描述符文本)；输入为空时返回 None
        """
        if not product_smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(product_smiles)
            if mol is None:
                return product_smiles, 0, None
            # 清除上游反应模板可能遗留的 atom map，避免污染标记体系
            rw = Chem.RWMol(mol)
            for atom in rw.GetAtoms():
                if atom.GetAtomMapNum():
                    atom.SetAtomMapNum(0)
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
        except Exception:
            return product_smiles, 0, None

        capped, n_stub, n_hub = _cap_residual_epoxide_stubs(mol)
        if capped is None:
            return product_smiles, 0, None
        n_attach = n_stub + n_hub
        clean = _clean_unit_smiles(capped) or product_smiles
        marked = _attachment_marked_unit_smiles(capped)
        return clean, n_attach, marked

    def _virtual_crosslink_safe(self, r_act: str, c_act: str) -> Optional[str]:
        try:
            from core.reaction_simulator import SimplifiedReactionModel
            sim_model = getattr(self, 'simplified_model', None) or SimplifiedReactionModel(verbose=False)
            return sim_model.create_virtual_crosslink(r_act, c_act)
        except Exception:
            return None

    def generate_crosslinked_fragment(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        stoichiometry: float = 1.0,
        target_conversion: float = 0.5
    ) -> Optional[str]:
        # 生成交联网络枢纽片段的代表性SMILES（保持向后兼容，内部委托 build_network_unit）
        try:
            unit = self.build_network_unit(
                epoxy_smiles, curer_smiles,
                target_conversion=target_conversion if target_conversion is not None else 0.5,
                stoichiometry=stoichiometry if stoichiometry and float(stoichiometry) > 0 else 1.0
            )
        except Exception:
            unit = None
        if unit:
            return unit

        r_clean = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
        c_clean = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)
        if not r_clean or not c_clean:
            return None
        return self._virtual_crosslink_safe(r_clean, c_clean)

    def _generate_oligomer_smiles(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        stoichiometry: float = 1.0,
        target_conversion: float = 0.5
    ) -> Optional[str]:
        # 转化率驱动的低聚物/网络枢纽片段（与 generate_crosslinked_fragment 统一委托）
        try:
            return self.build_network_unit(
                epoxy_smiles, curer_smiles,
                target_conversion=target_conversion if target_conversion is not None else 0.5
            )
        except Exception:
            return None

    def _generate_bigsmiles_network(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        stoichiometry: float = 1.0,
        target_conversion: float = 0.5
    ) -> str:
        """
        生成BigSMILES交联网络表示

        [修复] 连接端数量不再固定为 2 个：连接点 = 残留环氧封端碳（悬挂位）
        + 枢纽剩余活性位（伯/仲胺N、硫醇S、酚O、羧酸OH），描述符以内联
        "([$])" 形式打在对应原子上，多官能度因此可被表达；连接点不足 2 个
        或标记失败时回退旧的端部双 [$] 形式。

        Returns:
            BigSMILES字符串，如: {C([$])...N([$])...C([$])}
        """
        curer_type = "unknown"
        try:
            unit_pack = self.build_network_unit_full(
                epoxy_smiles, curer_smiles, target_conversion=target_conversion
            )
            if not unit_pack:
                r_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
                c_act = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)
                unit_pack = self._finalize_network_unit(
                    self._virtual_crosslink_safe(r_act, c_act)
                )

            if not unit_pack or not unit_pack[0]:
                curer_type, _ = self.detect_curer_type(curer_smiles)
                return self._generate_simplified_bigsmiles(epoxy_smiles, curer_smiles, curer_type)

            unit_smiles, n_attach, marked = unit_pack

            epoxy_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
            epoxy_fg = self.identify_functional_groups(epoxy_act) if epoxy_act else {}
            epoxy_functionality = epoxy_fg.get('epoxide', 2)

            # [修复] 连接端按真实连接点数发射（位置在封端碳/剩余活性位上）
            if n_attach >= 2 and marked:
                return f"{{{marked}}}"

            # 低连接度/标记失败时退回旧的端部双描述符形式
            if epoxy_functionality >= 2:
                return f"{{[$]{unit_smiles}[$]}}"
            else:
                return f"{{[>]{unit_smiles}[<]}}"

        except Exception as e:
            if self.verbose:
                print(f"⚠️ BigSMILES生成失败: {e}")
            return self._generate_simplified_bigsmiles(epoxy_smiles, curer_smiles, curer_type)

    def _generate_simplified_bigsmiles(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        curer_type: str
    ) -> str:
        """
        生成简化的BigSMILES表示（当详细模拟失败时）

        基于反应物结构生成通用的交联网络表示
        """
        try:
            epoxy_mol = Chem.MolFromSmiles(self._clean_smiles(epoxy_smiles) or "")
            curer_mol = Chem.MolFromSmiles(self._clean_smiles(curer_smiles) or "")

            if epoxy_mol is None or curer_mol is None:
                return "{[$]CC(O)CN[$]}"  # 最简化的环氧-胺网络

            # 根据固化剂类型生成典型的交联单元
            if curer_type == 'amine':
                # 环氧-胺交联：C-O键断裂，形成C-N键和羟基
                return "{[$]CC(O)CN(CC(O)C[$])[$]}"
            elif curer_type == 'anhydride':
                # 环氧-酸酐交联：形成酯键
                return "{[$]CC(O)COC(=O)C[$]}"
            elif curer_type == 'thiol':
                # 环氧-硫醇交联：形成C-S键
                return "{[$]CC(O)CS[$]}"
            else:
                return "{[$]CC(O)CN[$]}"

        except Exception:
            return "{[$]CC(O)CN[$]}"

    def get_product_representation(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        stoichiometry: float = 1.0,
        target_conversion: float = 0.5,
        output_format: str = 'auto'
    ) -> Dict[str, any]:
        """
        智能选择产物表示方法（根据转化率自动选择）

        Args:
            epoxy_smiles: 环氧树脂SMILES
            curer_smiles: 固化剂SMILES
            stoichiometry: 化学计量比
            target_conversion: 目标转化率 (0-1)
            output_format: 'auto' | 'smiles' | 'bigsmiles' | 'both'
                - 'auto': 根据转化率自动选择
                - 'smiles': 强制返回SMILES
                - 'bigsmiles': 强制返回BigSMILES
                - 'both': 返回两者

        Returns:
            Dict包含:
                - 'representation_type': 'smiles' | 'oligomer' | 'bigsmiles'
                - 'smiles': SMILES字符串（如果适用）
                - 'bigsmiles': BigSMILES字符串（如果适用）
                - 'conversion': 转化率
                - 'description': 描述信息
        """
        result = {
            'conversion': target_conversion,
            'stoichiometry': stoichiometry,
        }

        # 生成代表性网络单元 SMILES（用于 RDKit 描述符/3D/拓扑提取）
        smiles = self.generate_crosslinked_fragment(
            epoxy_smiles, curer_smiles, stoichiometry, target_conversion
        )
        # 生成标准 BigSMILES 拓扑网络结构（用于聚合物图特征/随机图解析）
        bigsmiles = self._generate_bigsmiles_network(
            epoxy_smiles, curer_smiles, stoichiometry, target_conversion
        )

        # 默认模式为 both / bigsmiles，确保用户需要的 BigSMILES 网络和 SMILES 产物均完备
        result['representation_type'] = 'bigsmiles' if output_format in ('bigsmiles', 'auto') else output_format
        result['smiles'] = smiles
        result['bigsmiles'] = bigsmiles
        result['description'] = f'交联网络簇（SMILES 产物 + BigSMILES 拓扑，目标转化率 {target_conversion*100:.0f}%）'
        return result


# =============================================================================
# 反应产物特征提取器
# =============================================================================

class CrosslinkedFeatureExtractor:
    """
    交联产物特征提取器
    
    提取反应产物的分子特征，用于机器学习
    """
    
    def __init__(self, verbose: bool = False):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required")
        
        self.verbose = verbose
        self.simulator = EpoxyReactionSimulator(verbose=verbose)
    
    def extract_crosslink_features(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        target_conversion: float = None,
        curing_temp: float = 150.0,
        curing_time: float = 2.0,
        auto_estimate_conversion: bool = True
    ) -> Dict[str, float]:
        """
        提取交联体系的特征

        Args:
            epoxy_smiles: 环氧树脂SMILES
            curer_smiles: 固化剂SMILES
            target_conversion: 目标转化率 (0-1)
                - 如果为None且auto_estimate_conversion=True，则自动估算
                - 如果提供了值，则使用该值（实测转化率优先）
            curing_temp: 固化温度 (°C)，用于转化率估算
            curing_time: 固化时间 (hours)，用于转化率估算
            auto_estimate_conversion: 是否自动估算转化率

        Returns:
            Dict: 交联相关特征
        """
        features = {}

        # 1. 反应物特征
        epoxy_fg = self.simulator.identify_functional_groups(epoxy_smiles)
        curer_type, curer_fg = self.simulator.detect_curer_type(curer_smiles)

        features['epoxide_count'] = epoxy_fg.get('epoxide', 0)
        features['curer_type_amine'] = 1 if curer_type == 'amine' else 0
        features['curer_type_anhydride'] = 1 if curer_type == 'anhydride' else 0
        features['curer_type_thiol'] = 1 if curer_type == 'thiol' else 0
        features['curer_type_hydrazide'] = 1 if curer_type == 'hydrazide' else 0

        # [修复] 芳香伯胺（苯环上的 NH2）同时命中 primary_amine 与 aromatic_amine 两个
        # SMARTS，直接相加会把 DDS 的 2 个伯胺数成 4 → 官能度 8、r=4、转化率被压到 0.22。
        # primary_amine 的 SMARTS [NX3;H2] 本就覆盖芳香环外 NH2，aromatic_amine 仅作类型标记。
        features['primary_amine_count'] = curer_fg.get('primary_amine', 0)
        features['aromatic_amine_count'] = curer_fg.get('aromatic_amine', 0)
        features['secondary_amine_count'] = curer_fg.get('secondary_amine', 0)
        features['anhydride_count'] = curer_fg.get('anhydride', 0)
        features['thiol_count'] = curer_fg.get('thiol', 0)

        # 2. 理论交联密度
        # 对于多官能度环氧和固化剂，交联密度 ∝ 官能度
        epoxy_functionality = features['epoxide_count']

        if curer_type == 'amine':
            # 伯胺可以反应2次，仲胺1次
            curer_functionality = features['primary_amine_count'] * 2 + features['secondary_amine_count']
        elif curer_type == 'anhydride':
            # 1个酸酐基团对应开环消耗1个环氧基（1:1化学计量）
            curer_functionality = features['anhydride_count']
        elif curer_type == 'thiol':
            curer_functionality = features['thiol_count']
        else:
            curer_functionality = 1

        features['epoxy_functionality'] = epoxy_functionality
        features['curer_functionality'] = curer_functionality

        # 化学计量比 (r = 活性氢当量 / 环氧当量)
        # [修复] 单组分路径没有配方配比数据，旧的"官能度比"实为 1:1 分子混合假设
        # （DGEBA+DDS 会得出 r=4→转化率 0.22），与真实配方化学计量无关。
        # 参考态取 r=1（化学计量平衡）；真实 r 由数据列/多组分路径提供。
        features['molecular_functionality_ratio'] = (
            curer_functionality / epoxy_functionality if epoxy_functionality > 0 else 0.0
        )
        features['stoichiometry_r'] = 1.0

        # 理论最大转化率
        r = features['stoichiometry_r']
        if r > 0:
            features['theoretical_alpha_max'] = min(1.0, r, 1.0/r)
        else:
            features['theoretical_alpha_max'] = 0.0

        # 2.5. 自动估算转化率（如果未提供）
        if target_conversion is None and auto_estimate_conversion:
            # 自动估算转化率
            estimated_conversion = self.simulator.estimate_conversion(
                epoxy_smiles=epoxy_smiles,
                curer_smiles=curer_smiles,
                stoichiometry_r=r,
                curing_temp=curing_temp,
                curing_time=curing_time,
                use_typical_values=True
            )
            features['conversion_source'] = 'estimated'
            features['estimated_conversion_input'] = estimated_conversion
            actual_conversion = estimated_conversion
        elif target_conversion is not None:
            # 使用用户提供的转化率（实测值优先）
            features['conversion_source'] = 'provided'
            features['estimated_conversion_input'] = target_conversion
            actual_conversion = target_conversion
        else:
            # 使用默认值
            features['conversion_source'] = 'default'
            features['estimated_conversion_input'] = 0.5
            actual_conversion = 0.5

        # 保存固化条件与转化率（[修复] 原先由"转化率代理"块赋值，删除代理后需在此正式写入）
        features['curing_temp'] = curing_temp
        features['curing_time'] = curing_time
        features['estimated_conversion'] = actual_conversion

        # 3. 模拟反应产物特征（使用智能表示方法）
        try:
            # 使用新的智能表示方法
            product_repr = self.simulator.get_product_representation(
                epoxy_smiles, curer_smiles,
                stoichiometry=r,
                target_conversion=actual_conversion,  # 使用估算或提供的转化率
                output_format='auto'  # 根据转化率自动选择
            )

            # [列精简] representation_type/description 文本列已删除；保留
            # product_smiles（产物 SMILES）与 product_structure（网络 BigSMILES 包裹）
            product_smi = product_repr.get('smiles')
            if product_smi:
                features['product_smiles'] = product_smi
                features['product_structure'] = f"{{[$]{product_smi}[$]}}"
                prod_desc = _compute_product_descriptors(product_smi, include_3d=True)
                # 内部量不进入特征表：仅用于转化率相对化计算
                residual_count = float(prod_desc.pop('internal_residual_epoxide_count', 0.0) or 0.0)
                prod_desc.pop('internal_product_mol_weight', None)
                features.update(prod_desc)
                # [修复] 不再用"残余环氧占比"改写 estimated_conversion：撒网生长引擎的
                # 产物是多单体簇（按转化率设计保留悬环氧），残余数/单体环氧数已无转化率
                # 语义，且会把估算值（0.88/0.78）覆盖成 ~0.5 的假值。
                # 环氧消耗信息由 residual_epoxide_density 等产物描述符承载。
            else:
                features['product_smiles'] = None
                features['product_structure'] = None

            # [去重] product_structure（BigSMILES 优先）与 product_smiles（纯 SMILES）
            # 已覆盖全部信息，不再输出重复的 product_bigsmiles 特征列。

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 产物特征提取失败: {e}")
            features['product_smiles'] = None
            features['representation_type'] = 'failed'
        
        return features
    
    def batch_extract_features(
        self,
        df: pd.DataFrame,
        epoxy_col: str,
        curer_col: str,
        conversion_col: str = None,
        curing_temp_col: str = None,
        curing_time_col: str = None,
        default_curing_temp: float = 150.0,
        default_curing_time: float = 2.0,
        auto_estimate_conversion: bool = True,
        prefix: str = "crosslink"
    ) -> pd.DataFrame:
        """
        批量提取交联特征

        Args:
            df: 数据框
            epoxy_col: 环氧树脂SMILES列名
            curer_col: 固化剂SMILES列名
            conversion_col: 转化率列名（如果有实测值）
            curing_temp_col: 固化温度列名（如果有）
            curing_time_col: 固化时间列名（如果有）
            default_curing_temp: 默认固化温度 (°C)
            default_curing_time: 默认固化时间 (hours)
            auto_estimate_conversion: 是否自动估算转化率（当conversion_col为None时）
            prefix: 特征名前缀

        Returns:
            DataFrame: 特征数据框
        """
        results = []

        if auto_estimate_conversion and conversion_col is None:
            print(f"\n🔬 正在提取交联特征（自动估算转化率）...")
        elif conversion_col is not None:
            print(f"\n🔬 正在提取交联特征（使用实测转化率：{conversion_col}）...")
        else:
            print(f"\n🔬 正在提取交联特征...")

        for idx in tqdm(range(len(df)), desc="Crosslink Features"):
            try:
                epoxy_smi = df.iloc[idx][epoxy_col]
                curer_smi = df.iloc[idx][curer_col]

                # 获取转化率
                if conversion_col is not None and conversion_col in df.columns:
                    target_conv = df.iloc[idx][conversion_col]
                    if pd.isna(target_conv):
                        target_conv = None
                else:
                    target_conv = None

                # 获取固化条件
                if curing_temp_col is not None and curing_temp_col in df.columns:
                    curing_temp = df.iloc[idx][curing_temp_col]
                    if pd.isna(curing_temp):
                        curing_temp = default_curing_temp
                else:
                    curing_temp = default_curing_temp

                if curing_time_col is not None and curing_time_col in df.columns:
                    curing_time = df.iloc[idx][curing_time_col]
                    if pd.isna(curing_time):
                        curing_time = default_curing_time
                else:
                    curing_time = default_curing_time

                features = self.extract_crosslink_features(
                    epoxy_smi, curer_smi,
                    target_conversion=target_conv,
                    curing_temp=curing_temp,
                    curing_time=curing_time,
                    auto_estimate_conversion=auto_estimate_conversion
                )

                # 添加前缀
                features = {f"{prefix}_{k}": v for k, v in features.items()}
                results.append(features)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Row {idx} 失败: {e}")
                results.append({})

        return pd.DataFrame(results)


# =============================================================================
# 简化反应模型：基于官能团的虚拟反应
# =============================================================================

class SimplifiedReactionModel:
    """
    简化反应模型
    
    当复杂反应模拟失败时，使用简化方法：
    1. 将环氧基和固化剂基团"虚拟连接"
    2. 不执行真正的化学反应，而是直接组合分子片段
    
    优点：更稳定，适用于复杂分子
    """
    
    def __init__(self, verbose: bool = False):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required")
        self.verbose = verbose
    
    def create_virtual_crosslink(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        n_links: int = 1
    ) -> Optional[str]:
        """
        创建虚拟共价交联产物（高可靠保底机制）
        
        通过在分子间建立真实共价单键组合分子片段，确保产物结构与分子量/极性表面积等特征 100% 可算
        """
        try:
            epoxy_s = convert_to_smiles(epoxy_smiles, fmt="auto") or epoxy_smiles
            curer_s = convert_to_smiles(curer_smiles, fmt="auto") or curer_smiles
            
            if '{' in str(epoxy_s):
                if 'C(C)(C)' in str(epoxy_s) and ('c1' in str(epoxy_s) or 'c2' in str(epoxy_s) or 'c3' in str(epoxy_s)):
                    epoxy_s = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
                elif 'c1ccc(C(C)(C)c2ccc(' in str(epoxy_s) or 'c2ccc(C(C)(C)c3ccc(' in str(epoxy_s):
                    epoxy_s = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
                elif 'S(=O)(=O)' in str(epoxy_s):
                    epoxy_s = 'O=S(=O)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
            if '{' in str(curer_s) and 'c1c(O)c(C)cc(' in str(curer_s):
                curer_s = 'Cc1c(O)cccc1'

            epoxy_mol = Chem.MolFromSmiles(str(epoxy_s).strip())
            curer_mol = Chem.MolFromSmiles(str(curer_s).strip())
            
            if epoxy_mol is None or curer_mol is None:
                return None
            
            combo = Chem.CombineMols(epoxy_mol, curer_mol)
            rw = Chem.RWMol(combo)
            
            # 在环氧分子中寻找环氧碳（优先），否则第一个碳
            epoxy_patt = Chem.MolFromSmarts('[C;r3][O;r3]')
            ep_matches = epoxy_mol.GetSubstructMatches(epoxy_patt)
            idx1 = ep_matches[0][0] if ep_matches else 0

            # 在固化剂分子中优先选择带活性氢的杂原子（N-H > S-H > O-H > N/S/O）
            offset = epoxy_mol.GetNumAtoms()
            idx2 = None
            priority = [
                Chem.MolFromSmarts('[NX3;H1,H2]'),
                Chem.MolFromSmarts('[SX2;H1]'),
                Chem.MolFromSmarts('[OX2;H1]'),
                Chem.MolFromSmarts('[N,O,S]'),
            ]
            for patt in priority:
                if patt is None:
                    continue
                matches = curer_mol.GetSubstructMatches(patt)
                if matches:
                    idx2 = offset + matches[0][0]
                    break
            if idx2 is None:
                idx2 = offset
            if idx1 == idx2:
                return f"{epoxy_s}.{curer_s}"
                    
            try:
                rw.AddBond(idx1, idx2, Chem.BondType.SINGLE)
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except Exception:
                return f"{epoxy_s}.{curer_s}"
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Virtual crosslink failed: {e}")
            return None
    
    def extract_combined_fingerprint(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        fp_type: str = 'morgan',
        n_bits: int = 2048,
        radius: int = 2
    ) -> Optional[np.ndarray]:
        """
        提取组合分子指纹
        
        对环氧和固化剂分别计算指纹，然后按位OR组合
        """
        try:
            epoxy_smiles = convert_to_smiles(epoxy_smiles, fmt="auto") or epoxy_smiles
            curer_smiles = convert_to_smiles(curer_smiles, fmt="auto") or curer_smiles
            
            epoxy_mol = Chem.MolFromSmiles(str(epoxy_smiles).strip())
            curer_mol = Chem.MolFromSmiles(str(curer_smiles).strip())
            
            if epoxy_mol is None or curer_mol is None:
                return None
            
            if fp_type.lower() == 'morgan':
                epoxy_fp = AllChem.GetMorganFingerprintAsBitVect(
                    epoxy_mol, radius, nBits=n_bits
                )
                curer_fp = AllChem.GetMorganFingerprintAsBitVect(
                    curer_mol, radius, nBits=n_bits
                )
            else:  # MACCS
                from rdkit.Chem import MACCSkeys
                epoxy_fp = MACCSkeys.GenMACCSKeys(epoxy_mol)
                curer_fp = MACCSkeys.GenMACCSKeys(curer_mol)
            
            # 组合指纹（按位OR）
            epoxy_arr = np.array(epoxy_fp)
            curer_arr = np.array(curer_fp)
            combined = np.logical_or(epoxy_arr, curer_arr).astype(int)
            
            return combined
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Fingerprint extraction failed: {e}")
            return None


def _extract_multicomponent_chunk(
    chunk_df: pd.DataFrame,
    chunk_wide_df: Optional[pd.DataFrame],
    resin_cols_found: List[str],
    curer_cols_found: List[str],
    resin_cols_prefix: str,
    curer_cols_prefix: str,
    actual_stoich_col: Optional[str],
    stoichiometry_col: Optional[str],
    conversion_col: Optional[str],
    curing_temp_col: Optional[str],
    curing_time_col: Optional[str],
    default_curing_temp: float,
    default_curing_time: float,
    auto_estimate_conversion: bool,
    reaction_method: str,
    prefix: str,
    verbose: bool = False,
    max_temp_col: Optional[str] = None,
    post_cure_col: Optional[str] = None,
    accelerator_count_col: Optional[str] = None,
    catalyst_count_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """子进程 Worker 函数：负责一个独立数据批次的多组分交联与物理机理特征提取"""
    ext = MulticomponentCrosslinkedFeatureExtractor(verbose=verbose)
    has_aligned_wide = chunk_wide_df is not None and len(chunk_wide_df) == len(chunk_df)
    results = []

    for idx in range(len(chunk_df)):
        try:
            row = chunk_df.iloc[idx]
            wide_row = chunk_wide_df.iloc[idx] if has_aligned_wide else None

            # 1. 收集树脂组分
            resin_components = []
            for comp_i, col_name in enumerate(resin_cols_found, start=1):
                smi = row[col_name]
                if smi and not pd.isna(smi) and str(smi).strip():
                    weight = 1.0
                    if wide_row is not None:
                        w_val = wide_row.get(f"resin_{comp_i}_amount_phr")
                        if w_val is not None and not pd.isna(w_val) and float(w_val) > 0:
                            weight = float(w_val)
                    else:
                        for w_pat in [f"resin_{comp_i}_amount_phr", f"resin_amount_phr_{comp_i}", f"{resin_cols_prefix}_{comp_i}_phr"]:
                            if w_pat in chunk_df.columns and not pd.isna(row[w_pat]):
                                try:
                                    val = float(row[w_pat])
                                    if val > 0:
                                        weight = val
                                        break
                                except Exception:
                                    pass
                    resin_components.append((str(smi).strip(), weight))

            # 2. 收集固化剂组分
            curer_components = []
            for comp_i, col_name in enumerate(curer_cols_found, start=1):
                smi = row[col_name]
                if smi and not pd.isna(smi) and str(smi).strip():
                    weight = 1.0
                    if wide_row is not None:
                        w_val = wide_row.get(f"curing_agent_{comp_i}_amount_phr")
                        if w_val is not None and not pd.isna(w_val) and float(w_val) > 0:
                            weight = float(w_val)
                    else:
                        for w_pat in [f"curing_agent_{comp_i}_amount_phr", f"curing_amount_phr_{comp_i}", f"{curer_cols_prefix}_{comp_i}_phr"]:
                            if w_pat in chunk_df.columns and not pd.isna(row[w_pat]):
                                try:
                                    val = float(row[w_pat])
                                    if val > 0:
                                        weight = val
                                        break
                                except Exception:
                                    pass
                    curer_components.append((str(smi).strip(), weight))

            # [添加剂感知 P2/P3] 收集并分诊小分子添加剂，反应型并入对应侧，
            # 惰性型单独成特征；缺 PHR 按类别典型掺量并封顶 20%
            _add_features: Dict[str, Any] = {}
            try:
                _add_info = _collect_additive_components(row, wide_row, chunk_df.columns, ext.simulator)
                if _add_info and (_add_info.get("resin") or _add_info.get("curer") or _add_info.get("inert")):
                    _resin_base = sum(w for _, w in resin_components)
                    _curer_base = sum(w for _, w in curer_components)
                    _r_add, _c_add, _add_features = _apply_additive_weights(_add_info, _resin_base, _curer_base)
                    resin_components = resin_components + _r_add
                    curer_components = curer_components + _c_add
            except Exception:
                _add_features = {}

            if not resin_components or not curer_components:
                results.append({})
                continue

            # 3. 读取化学计量比
            if actual_stoich_col and actual_stoich_col in chunk_df.columns:
                stoich_r = row[actual_stoich_col]
                if pd.isna(stoich_r):
                    stoich_r = 1.0
            elif stoichiometry_col in chunk_df.columns:
                stoich_r = row[stoichiometry_col]
                if pd.isna(stoich_r):
                    stoich_r = 1.0
            else:
                stoich_r = 1.0

            # 4. 读取转化率
            if conversion_col and conversion_col in chunk_df.columns:
                target_conv = row[conversion_col]
                if pd.isna(target_conv):
                    target_conv = None
            else:
                target_conv = None

            # 5. 读取固化条件
            if curing_temp_col and curing_temp_col in chunk_df.columns:
                curing_temp = row[curing_temp_col]
                if pd.isna(curing_temp):
                    curing_temp = default_curing_temp
            else:
                curing_temp = default_curing_temp

            if curing_time_col and curing_time_col in chunk_df.columns:
                curing_time = row[curing_time_col]
                if pd.isna(curing_time):
                    curing_time = default_curing_time
            else:
                curing_time = default_curing_time

            # [转化率重构] 读取 (最高反应温度, 是否后固化)，脏数据已过滤（[25,260]°C 外视为缺失）
            t_max_c = None
            if max_temp_col and max_temp_col in chunk_df.columns:
                t_max_c = ext.simulator._sanitize_t_max(row.get(max_temp_col))
            _post_raw = row.get(post_cure_col) if (post_cure_col and post_cure_col in chunk_df.columns) else None
            post_cure_flag = ext.simulator._parse_post_cure(_post_raw)
            # [促进剂接入] 优先读促进剂/催化剂计数列（行或宽表），无计数列时回退结构检测
            acc_hint = ext.simulator._detect_accelerator_signal(
                row, wide_row, list(chunk_df.columns),
                list(wide_row.index) if wide_row is not None else None)

            # 6. 提取特征
            features = ext.extract_multicomponent_features(
                resin_components,
                curer_components,
                stoichiometry_r=stoich_r,
                target_conversion=target_conv,
                curing_temp=curing_temp,
                curing_time=curing_time,
                auto_estimate_conversion=auto_estimate_conversion,
                reaction_method=reaction_method,
                accelerator_present=bool(_add_features.get("additive_accelerator_present", 0.0)),
                t_max_c=t_max_c,
                post_cure=post_cure_flag,
                has_accelerator_hint=acc_hint
            )

            # [添加剂感知] 合并添加剂特征（前缀前合并，保持同前缀命名）
            if _add_features:
                features.update(_add_features)

            # 添加前缀
            features = {f"{prefix}_{k}": v for k, v in features.items()}
            results.append(features)
        except Exception:
            results.append({})

    return results


class MulticomponentCrosslinkedFeatureExtractor:
    """
    多组分交联特征提取器

    支持多个树脂组分 + 多个固化剂组分的混合体系
    """

    def __init__(self, verbose: bool = False):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required")

        self.verbose = verbose
        self.simulator = EpoxyReactionSimulator(verbose=verbose)
        self.single_extractor = CrosslinkedFeatureExtractor(verbose=verbose)

        # 性能优化：常驻机理引擎与记忆化缓存，避免高频单体反复解析与重复计算
        try:
            from core.epoxy_mechanism_features import EpoxyMechanismEngine
            self._mechanism_engine = EpoxyMechanismEngine(verbose=verbose)
        except Exception:
            self._mechanism_engine = None

        self._fg_cache: Dict[str, Dict[str, int]] = {}
        self._curer_type_cache: Dict[str, Tuple[str, Dict[str, int]]] = {}
        self._product_feat_cache: Dict[str, Dict[str, float]] = {}
        self._mol_prop_cache: Dict[str, Dict[str, Any]] = {}
        # [重复率修复] 共聚网络单元缓存: (epoxy, curer, conv) -> unit_smiles
        self._copolymer_unit_cache: Dict[Tuple[str, str, float], str] = {}

    def _extract_extended_product_features(self, smiles: str, include_3d: bool = True) -> Dict[str, float]:
        # 统一委托模块级 _compute_product_descriptors（含进程级缓存）
        if not smiles:
            return {}
        try:
            return _compute_product_descriptors(smiles, include_3d=include_3d)
        except Exception:
            return {}

    def extract_multicomponent_features(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        stoichiometry_r: float,
        target_conversion: float = None,
        curing_temp: float = 150.0,
        curing_time: float = 2.0,
        auto_estimate_conversion: bool = True,
        reaction_method: str = 'weighted',
        accelerator_present: bool = False,
        t_max_c: Optional[float] = None,
        post_cure: bool = False,
        has_accelerator_hint: Optional[bool] = None,
    ) -> Dict[str, any]:
        """
        提取多组分交联特征

        Args:
            resin_components: [(smiles, weight), ...] 树脂组分
                weight 为质量分数或摩尔分数（自动归一化）
            curer_components: [(smiles, weight), ...] 固化剂组分
            stoichiometry_r: 总体化学计量比 (AHEW/EEW)
            target_conversion: 目标转化率（None则自动估算）
            curing_temp: 固化温度
            curing_time: 固化时间
            auto_estimate_conversion: 是否自动估算转化率
            reaction_method: 'weighted' (快速) 或 'combinatorial' (准确)

        Returns:
            Dict: 多组分交联特征
        """
        features = {}

        # 过滤空组分
        resin_components = [(smi, w) for smi, w in resin_components if smi and str(smi).strip()]
        curer_components = [(smi, w) for smi, w in curer_components if smi and str(smi).strip()]

        if not resin_components or not curer_components:
            return features

        # 归一化权重
        total_resin_weight = sum(w for _, w in resin_components)
        total_curer_weight = sum(w for _, w in curer_components)

        if total_resin_weight > 0:
            resin_components = [(smi, w / total_resin_weight) for smi, w in resin_components]
        if total_curer_weight > 0:
            curer_components = [(smi, w / total_curer_weight) for smi, w in curer_components]

        # 1. 计算加权平均官能度（带记忆化缓存）
        weighted_epoxy_func = 0.0
        weighted_curer_func = 0.0

        for resin_smi, resin_weight in resin_components:
            if hasattr(self, '_fg_cache') and resin_smi in self._fg_cache:
                epoxy_fg = self._fg_cache[resin_smi]
            else:
                epoxy_fg = self.simulator.identify_functional_groups(resin_smi)
                if hasattr(self, '_fg_cache'):
                    self._fg_cache[resin_smi] = epoxy_fg
            epoxy_func = epoxy_fg.get('epoxide', 0)
            weighted_epoxy_func += epoxy_func * resin_weight

        for curer_smi, curer_weight in curer_components:
            if hasattr(self, '_curer_type_cache') and curer_smi in self._curer_type_cache:
                curer_type, curer_fg = self._curer_type_cache[curer_smi]
            else:
                curer_type, curer_fg = self.simulator.detect_curer_type(curer_smi)
                if hasattr(self, '_curer_type_cache'):
                    self._curer_type_cache[curer_smi] = (curer_type, curer_fg)

            if curer_type == 'amine':
                primary = curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)
                secondary = curer_fg.get('secondary_amine', 0)
                curer_func = primary * 2 + secondary
            elif curer_type == 'anhydride':
                curer_func = curer_fg.get('anhydride', 0)
            elif curer_type == 'thiol':
                curer_func = curer_fg.get('thiol', 0)
            else:
                curer_func = 1

            weighted_curer_func += curer_func * curer_weight

        # [已移除] weighted_epoxy/curer_functionality：单体加权官能度属机理特征家族，
        # 与产物相对值口径不符；官能度信息已由反应产物层面的 junction_site_density 等承载
        features['stoichiometry_r'] = stoichiometry_r

        # [已移除] 机理特征注入：mech_weighted_* / stoich_* / alpha_* 等与单体绝对值
        # 重复且违背产物相对值口径，多组分管线不再输出这些特征

        # 2. 估算转化率
        if target_conversion is None and auto_estimate_conversion:
            estimated_conversion = self.simulator.estimate_conversion_multicomponent(
                resin_components,
                curer_components,
                stoichiometry_r,
                curing_temp,
                curing_time,
                accelerator_present=accelerator_present,
                t_max_c=t_max_c,
                post_cure=post_cure,
                has_accelerator_hint=has_accelerator_hint
            )
            # 不保存 conversion_source，只保存转化率值
            actual_conversion = estimated_conversion
            # [转化率重构] 标记是否使用了 (T_max, 后固化) 玻璃化模型
            features['used_schedule_conversion'] = 1.0 if (
                self.simulator._sanitize_t_max(t_max_c) is not None) else 0.0
        elif target_conversion is not None:
            actual_conversion = target_conversion
        else:
            actual_conversion = 0.5

        features['estimated_conversion'] = actual_conversion  # 重命名为更简洁的名字

        # 3. 模拟多组分反应
        try:
            reaction_result = self.simulator.simulate_multicomponent_reaction(
                resin_components,
                curer_components,
                target_conversion=actual_conversion,
                method=reaction_method
            )

            # 调试：检查反应是否成功
            if self.verbose:
                smiles_check = reaction_result.get('representative_smiles')
                if smiles_check:
                    # 检查产物中是否还有环氧基
                    try:
                        mol_check = Chem.MolFromSmiles(smiles_check)
                        if mol_check:
                            epoxy_pattern = Chem.MolFromSmarts('[C]1[O][C]1')
                            remaining_epoxy = len(mol_check.GetSubstructMatches(epoxy_pattern))
                            print(f"🔬 反应模拟结果检查:")
                            print(f"   - 转化率: {actual_conversion:.2f}")
                            print(f"   - 产物中剩余环氧基: {remaining_epoxy}")
                            if remaining_epoxy > 0:
                                print(f"   ⚠️ 警告：产物中仍有环氧基，反应可能未成功！")
                    except Exception:
                        pass

            # 不保存 reaction_method_used，用户已经知道选的什么方法

            if reaction_method == 'weighted':
                # 方案1：加权平均法
                smiles_result = reaction_result.get('representative_smiles')
                # [重复率修复] 优先使用按行内权重构造的随机共聚网络 BigSMILES
                bigsmiles_result = (reaction_result.get('copolymer_bigsmiles')
                                    or reaction_result.get('representative_bigsmiles'))
                for _ck, _cv in (reaction_result.get('composition_features') or {}).items():
                    features[_ck] = _cv
                _sp = reaction_result.get('sampled_pair') or {}
                if _sp:
                    features['sampled_pair_probability'] = float(_sp.get('probability', 0.0))
                # [网络化重构] 系综统计特征（均值±标准差刻画网络构型涨落）
                _ens = reaction_result.get('ensemble_stats') or {}
                for _ek, _ev in _ens.items():
                    try:
                        features[f'ens_{_ek}'] = float(_ev)
                    except Exception:
                        pass

                # 检查产物是否有效（分子量增长或结构变化）
                if smiles_result:
                    try:
                        mol_check = Chem.MolFromSmiles(smiles_result)
                        mol_r = Chem.MolFromSmiles(reaction_result.get('main_resin', ''))
                        if mol_check and mol_r:
                            # 产物分子量无增长，说明反应未发生
                            if Descriptors.MolWt(mol_check) <= Descriptors.MolWt(mol_r) + 1.0 and actual_conversion > 0.1:
                                smiles_result = None
                    except Exception:
                        pass

                # 反应模板未匹配成功时的虚拟交联保底容错
                if not smiles_result:
                    try:
                        main_r_smi, _ = max(resin_components, key=lambda x: x[1])
                        main_c_smi, _ = max(curer_components, key=lambda x: x[1])
                        from core.reaction_simulator import SimplifiedReactionModel
                        sim_model = getattr(self.simulator, "simplified_model", None) or SimplifiedReactionModel(verbose=False)
                        virt_smi = sim_model.create_virtual_crosslink(main_r_smi, main_c_smi)
                        if virt_smi:
                            smiles_result = virt_smi
                    except Exception:
                        pass

                # [列恢复] product_structure = 网络拓扑 BigSMILES（共聚候选优先）
                if bigsmiles_result:
                    features['product_structure'] = bigsmiles_result
                features['product_smiles'] = smiles_result
                # [去重] structure 与 smiles 已覆盖全部信息，不再重复输出 bigsmiles 列

                # 提取产物扩展物理/化学特征（2D拓扑 + 交联位点 + 图论 + 3D构象）
                if smiles_result:
                    extended_features = self._extract_extended_product_features(smiles_result)
                    extended_features.pop('internal_residual_epoxide_count', None)
                    features.update(extended_features)
                    try:
                        mol_r = Chem.MolFromSmiles(reaction_result.get('main_resin', '') or '')
                        mw_r = Descriptors.MolWt(mol_r) if mol_r is not None else 0.0
                        if mw_r > 0 and extended_features.get('internal_product_mol_weight'):
                            features['product_degree_of_polymerization'] = extended_features['internal_product_mol_weight'] / mw_r
                    except Exception:
                        pass
                    features.pop('internal_product_mol_weight', None)

            elif reaction_method == 'combinatorial':
                # 方案2：组合反应法
                features['n_combinations'] = reaction_result.get('n_combinations', 0)
                smiles_result = reaction_result.get('representative_smiles')
                # [重复率修复] 优先使用按行内权重构造的随机共聚网络 BigSMILES
                bigsmiles_result = (reaction_result.get('copolymer_bigsmiles')
                                    or reaction_result.get('representative_bigsmiles'))
                for _ck, _cv in (reaction_result.get('composition_features') or {}).items():
                    features[_ck] = _cv
                _sp = reaction_result.get('sampled_pair') or {}
                if _sp:
                    features['sampled_pair_probability'] = float(_sp.get('probability', 0.0))
                # [网络化重构] 系综统计特征（均值±标准差刻画网络构型涨落）
                _ens = reaction_result.get('ensemble_stats') or {}
                for _ek, _ev in _ens.items():
                    try:
                        features[f'ens_{_ek}'] = float(_ev)
                    except Exception:
                        pass

                if smiles_result:
                    try:
                        mol_check = Chem.MolFromSmiles(smiles_result)
                        mol_r = Chem.MolFromSmiles(reaction_result.get('main_resin', ''))
                        if mol_check and mol_r:
                            if Descriptors.MolWt(mol_check) <= Descriptors.MolWt(mol_r) + 1.0 and actual_conversion > 0.1:
                                smiles_result = None
                    except Exception:
                        pass

                # 保底容错：若组合反应未生成代表性 SMILES，启动虚拟交联保底
                if not smiles_result:
                    try:
                        main_r_smi, _ = max(resin_components, key=lambda x: x[1])
                        main_c_smi, _ = max(curer_components, key=lambda x: x[1])
                        from core.reaction_simulator import SimplifiedReactionModel
                        sim_model = getattr(self.simulator, "simplified_model", None) or SimplifiedReactionModel(verbose=False)
                        virt_smi = sim_model.create_virtual_crosslink(main_r_smi, main_c_smi)
                        if virt_smi:
                            smiles_result = virt_smi
                    except Exception:
                        pass

                if bigsmiles_result:
                    features['product_structure'] = bigsmiles_result
                features['product_smiles'] = smiles_result
                # [去重] structure 与 smiles 已覆盖全部信息，不再重复输出 bigsmiles 列

                # 提取扩展的产物特征（2D拓扑 + 交联位点 + 图论 + 3D构象）
                if smiles_result:
                    extended_features = self._extract_extended_product_features(smiles_result)
                    extended_features.pop('internal_residual_epoxide_count', None)
                    features.update(extended_features)
                    try:
                        main_r_smi_dp, _ = max(resin_components, key=lambda x: x[1])
                        mol_r = Chem.MolFromSmiles(main_r_smi_dp or '')
                        mw_r = Descriptors.MolWt(mol_r) if mol_r is not None else 0.0
                        if mw_r > 0 and extended_features.get('internal_product_mol_weight'):
                            features['product_degree_of_polymerization'] = extended_features['internal_product_mol_weight'] / mw_r
                    except Exception:
                        pass
                    features.pop('internal_product_mol_weight', None)

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 多组分反应模拟失败: {e}")
            features['product_structure'] = None

        return features

    def batch_extract_features_from_dataframe(
        self,
        df: pd.DataFrame,
        resin_cols: Optional[List[str]] = None,
        curer_cols: Optional[List[str]] = None,
        resin_cols_prefix: str = 'resin_smiles',
        curer_cols_prefix: str = 'curing_agent_smiles',
        max_components: int = 6,
        stoichiometry_col: str = 'stoichiometric_ratio_r_cleaned',
        conversion_col: str = None,
        curing_temp_col: str = None,
        curing_time_col: str = None,
        max_temp_col: str = None,
        post_cure_col: str = None,
        accelerator_count_col: str = None,
        catalyst_count_col: str = None,
        default_curing_temp: float = 150.0,
        default_curing_time: float = 2.0,
        auto_estimate_conversion: bool = True,
        reaction_method: str = 'weighted',
        prefix: str = 'multicomp_crosslink',
        wide_df: Optional[pd.DataFrame] = None,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        从DataFrame批量提取多组分交联特征与高分子物理网络特征

        支持 resin_smiles_1~6、resin_1_structure 等多种列名格式，
        并支持自动与配方大宽表 (wide_df) 对齐以精确提取每种单体的 PHR 与当量。
        支持服务器多核并行加速 (n_jobs)。
        """
        results = []
        error_count = 0
        success_count = 0
        empty_resin_count = 0
        empty_curer_count = 0

        print(f"\n🔬 正在提取多组分交联与物理机理特征（方法: {reaction_method}）...")

        # 智能识别树脂列
        if resin_cols and len(resin_cols) > 0:
            resin_cols_found = [c for c in resin_cols if c in df.columns]
        else:
            resin_cols_found = []
            r_p1 = [f"{resin_cols_prefix}_{i}" for i in range(1, max_components + 1) if f"{resin_cols_prefix}_{i}" in df.columns]
            if r_p1:
                resin_cols_found = r_p1
            if not resin_cols_found:
                r_p2 = [c for c in df.columns if re.match(r"^resin_\d+_structure$", str(c))]
                if r_p2:
                    resin_cols_found = sorted(r_p2, key=lambda x: int(re.search(r"\d+", x).group()))
            if not resin_cols_found:
                r_p3 = [c for c in df.columns if ("resin" in str(c).lower() or "epoxy" in str(c).lower()) and ("structure" in str(c).lower() or "smiles" in str(c).lower()) and not str(c).lower().endswith("_format")]
                if r_p3:
                    resin_cols_found = r_p3

        # 智能识别固化剂列
        if curer_cols and len(curer_cols) > 0:
            curer_cols_found = [c for c in curer_cols if c in df.columns]
        else:
            curer_cols_found = []
            c_p1 = [f"{curer_cols_prefix}_{i}" for i in range(1, max_components + 1) if f"{curer_cols_prefix}_{i}" in df.columns]
            if c_p1:
                curer_cols_found = c_p1
            if not curer_cols_found:
                c_p2 = [c for c in df.columns if re.match(r"^curing_agent_\d+_structure$", str(c))]
                if c_p2:
                    curer_cols_found = sorted(c_p2, key=lambda x: int(re.search(r"\d+", x).group()))
            if not curer_cols_found:
                c_p3 = [c for c in df.columns if ("curing" in str(c).lower() or "hardener" in str(c).lower()) and ("structure" in str(c).lower() or "smiles" in str(c).lower()) and not str(c).lower().endswith("_format")]
                if c_p3:
                    curer_cols_found = c_p3

        print(f"✅ 找到树脂列 ({len(resin_cols_found)} 个): {resin_cols_found}")
        print(f"✅ 找到固化剂列 ({len(curer_cols_found)} 个): {curer_cols_found}")

        if not resin_cols_found:
            print(f"❌ 错误：未找到任何树脂列。")
            return pd.DataFrame()

        if not curer_cols_found:
            print(f"❌ 错误：未找到任何固化剂列。")
            return pd.DataFrame()

        # 智能对齐 wide_df
        has_aligned_wide = False
        if wide_df is not None and len(wide_df) == len(df):
            has_aligned_wide = True
            print(f"🔗 成功绑定母宽表辅助提取配方详细参数 ({len(wide_df)} 行)")

        # 寻找真实的配比列
        actual_stoich_col = None
        for cand in [stoichiometry_col, "formulation_r_value", "stoichiometric_ratio_r_cleaned", "r_value", "stoich_r"]:
            if cand and cand in df.columns:
                actual_stoich_col = cand
                break

        # 解析并行核心数
        import os
        if n_jobs is None or n_jobs == 0:
            effective_n_jobs = 1
        elif n_jobs < 0:
            system_cores = os.cpu_count() or 4
            max_safe = max(1, system_cores - 1)
            if os.name == 'nt':
                max_safe = min(max_safe, 60)
            effective_n_jobs = max_safe
        else:
            effective_n_jobs = n_jobs

        # 若数据量较大且指定多核，使用 joblib 多进程并行加速
        # [转化率重构] 自动检测 (最高反应温度, 是否后固化) 列；也可由调用方显式传入列名
        if max_temp_col is None:
            max_temp_col = 'process_max_temperature_c' if 'process_max_temperature_c' in df.columns else None
        if post_cure_col is None:
            post_cure_col = 'process_has_post_cure' if 'process_has_post_cure' in df.columns else None
        if accelerator_count_col is None:
            accelerator_count_col = next((c for c in ('accelerator_count', 'accelerator_component_count') if c in df.columns), None)
        if catalyst_count_col is None:
            catalyst_count_col = next((c for c in ('catalyst_count', 'catalyst_component_count') if c in df.columns), None)

        if effective_n_jobs > 1 and len(df) >= 20:
            try:
                from joblib import Parallel, delayed
                import numpy as np

                n_chunks = min(effective_n_jobs * 2, len(df))
                indices = [c for c in np.array_split(np.arange(len(df)), n_chunks) if len(c) > 0]
                tasks = []
                for idx_arr in indices:
                    sub_df = df.iloc[idx_arr].reset_index(drop=True)
                    sub_wide = wide_df.iloc[idx_arr].reset_index(drop=True) if has_aligned_wide else None
                    tasks.append((sub_df, sub_wide))

                print(f"🚀 启动服务器多核加速: {effective_n_jobs} 个并行 Worker 处理 {len(df)} 行数据 (共 {len(tasks)} 批次)...")
                results_nested = Parallel(n_jobs=effective_n_jobs, backend='loky')(
                    delayed(_extract_multicomponent_chunk)(
                        sub_df, sub_wide,
                        resin_cols_found, curer_cols_found,
                        resin_cols_prefix, curer_cols_prefix,
                        actual_stoich_col, stoichiometry_col,
                        conversion_col, curing_temp_col, curing_time_col,
                        default_curing_temp, default_curing_time,
                        auto_estimate_conversion, reaction_method,
                        prefix, self.verbose,
                        max_temp_col, post_cure_col,
                        accelerator_count_col, catalyst_count_col
                    )
                    for sub_df, sub_wide in tasks
                )
                results = [item for sublist in results_nested for item in sublist]
                out_df = pd.DataFrame(results)
                print(f"✅ 多核提取完成，共返回 {len(out_df)} 行 × {out_df.shape[1]} 列特征。")
                return out_df
            except Exception as par_exc:
                print(f"⚠️ 多核并行加速异常，自动平滑回退到单线程提取模式: {par_exc}")

        for idx in tqdm(range(len(df)), desc="Multicomponent Crosslink"):
            try:
                wide_row = wide_df.iloc[idx] if has_aligned_wide else None

                # 读取树脂组分
                resin_components = []
                for comp_i, col_name in enumerate(resin_cols_found, start=1):
                    smi = df.iloc[idx][col_name]
                    if smi and not pd.isna(smi) and str(smi).strip():
                        weight = 1.0
                        # 优先从 wide_df 提取实际 PHR
                        if wide_row is not None:
                            w_val = wide_row.get(f"resin_{comp_i}_amount_phr")
                            if w_val is not None and not pd.isna(w_val) and float(w_val) > 0:
                                weight = float(w_val)
                        else:
                            for w_pat in [f"resin_{comp_i}_amount_phr", f"resin_amount_phr_{comp_i}", f"{resin_cols_prefix}_{comp_i}_phr"]:
                                if w_pat in df.columns and not pd.isna(df.iloc[idx][w_pat]):
                                    try:
                                        val = float(df.iloc[idx][w_pat])
                                        if val > 0:
                                            weight = val
                                            break
                                    except Exception:
                                        pass
                        resin_components.append((str(smi).strip(), weight))

                # 读取固化剂组分
                curer_components = []
                for comp_i, col_name in enumerate(curer_cols_found, start=1):
                    smi = df.iloc[idx][col_name]
                    if smi and not pd.isna(smi) and str(smi).strip():
                        weight = 1.0
                        if wide_row is not None:
                            w_val = wide_row.get(f"curing_agent_{comp_i}_amount_phr")
                            if w_val is not None and not pd.isna(w_val) and float(w_val) > 0:
                                weight = float(w_val)
                        else:
                            for w_pat in [f"curing_agent_{comp_i}_amount_phr", f"curing_amount_phr_{comp_i}", f"{curer_cols_prefix}_{comp_i}_phr"]:
                                if w_pat in df.columns and not pd.isna(df.iloc[idx][w_pat]):
                                    try:
                                        val = float(df.iloc[idx][w_pat])
                                        if val > 0:
                                            weight = val
                                            break
                                    except Exception:
                                        pass
                        curer_components.append((str(smi).strip(), weight))

                # [添加剂感知 P2/P3] 收集并分诊小分子添加剂，反应型并入对应侧，
                # 惰性型单独成特征；缺 PHR 按类别典型掺量并封顶 20%
                _add_features: Dict[str, Any] = {}
                try:
                    _add_info = _collect_additive_components(
                        df.iloc[idx], wide_row, df.columns, self.simulator
                    )
                    if _add_info and (_add_info.get("resin") or _add_info.get("curer") or _add_info.get("inert")):
                        _resin_base = sum(w for _, w in resin_components)
                        _curer_base = sum(w for _, w in curer_components)
                        _r_add, _c_add, _add_features = _apply_additive_weights(_add_info, _resin_base, _curer_base)
                        resin_components = resin_components + _r_add
                        curer_components = curer_components + _c_add
                except Exception:
                    _add_features = {}

                if not resin_components:
                    empty_resin_count += 1
                    results.append({})
                    continue

                if not curer_components:
                    empty_curer_count += 1
                    results.append({})
                    continue

                # 读取化学计量比
                if actual_stoich_col and actual_stoich_col in df.columns:
                    stoich_r = df.iloc[idx][actual_stoich_col]
                    if pd.isna(stoich_r):
                        stoich_r = 1.0
                elif stoichiometry_col in df.columns:
                    stoich_r = df.iloc[idx][stoichiometry_col]
                    if pd.isna(stoich_r):
                        stoich_r = 1.0
                else:
                    stoich_r = 1.0

                # 读取转化率
                if conversion_col and conversion_col in df.columns:
                    target_conv = df.iloc[idx][conversion_col]
                    if pd.isna(target_conv):
                        target_conv = None
                else:
                    target_conv = None

                # 读取固化条件
                if curing_temp_col and curing_temp_col in df.columns:
                    curing_temp = df.iloc[idx][curing_temp_col]
                    if pd.isna(curing_temp):
                        curing_temp = default_curing_temp
                else:
                    curing_temp = default_curing_temp

                if curing_time_col and curing_time_col in df.columns:
                    curing_time = df.iloc[idx][curing_time_col]
                    if pd.isna(curing_time):
                        curing_time = default_curing_time
                else:
                    curing_time = default_curing_time

                # [转化率重构] 读取 (最高反应温度, 是否后固化)，脏数据已过滤
                t_max_c = None
                if max_temp_col and max_temp_col in df.columns:
                    t_max_c = self._sanitize_t_max(df.iloc[idx].get(max_temp_col))
                _post_raw = df.iloc[idx].get(post_cure_col) if (post_cure_col and post_cure_col in df.columns) else None
                post_cure_flag = self._parse_post_cure(_post_raw)
                acc_hint = self._detect_accelerator_signal(
                    df.iloc[idx], wide_row, list(df.columns),
                    list(wide_row.index) if wide_row is not None else None)

                # 提取特征
                features = self.extract_multicomponent_features(
                    resin_components,
                    curer_components,
                    stoichiometry_r=stoich_r,
                    target_conversion=target_conv,
                    curing_temp=curing_temp,
                    curing_time=curing_time,
                    auto_estimate_conversion=auto_estimate_conversion,
                    reaction_method=reaction_method,
                    accelerator_present=bool(_add_features.get("additive_accelerator_present", 0.0)),
                    t_max_c=t_max_c,
                    post_cure=post_cure_flag,
                    has_accelerator_hint=acc_hint
                )

                # [添加剂感知] 合并添加剂特征（前缀前合并，保持同前缀命名）
                if _add_features:
                    features.update(_add_features)

                # 添加前缀
                features = {f"{prefix}_{k}": v for k, v in features.items()}
                results.append(features)
                success_count += 1

            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"⚠️ Row {idx} 失败: {e}")
                results.append({})

        # 输出统计信息
        print(f"\n📊 提取统计:")
        print(f"  ✅ 成功: {success_count} 个样本")
        print(f"  ❌ 失败: {error_count} 个样本")
        print(f"  ⚠️ 树脂为空: {empty_resin_count} 个样本")
        print(f"  ⚠️ 固化剂为空: {empty_curer_count} 个样本")

        return pd.DataFrame(results)


# =============================================================================
# 便捷函数
# =============================================================================

def simulate_epoxy_curing(
    epoxy_smiles: str,
    curer_smiles: str,
    n_reactions: int = 1
) -> List[str]:
    """
    便捷函数：模拟环氧固化反应
    
    Args:
        epoxy_smiles: 环氧树脂SMILES
        curer_smiles: 固化剂SMILES
        n_reactions: 反应步数
        
    Returns:
        List[str]: 产物SMILES列表
    """
    simulator = EpoxyReactionSimulator()
    products = simulator.simulate_curing(epoxy_smiles, curer_smiles, n_reactions)
    return [p['smiles'] for p in products]


def extract_crosslink_features(
    epoxy_smiles: str,
    curer_smiles: str,
    target_conversion: float = None
) -> Dict[str, float]:
    """
    便捷函数：提取交联特征（target_conversion=None 时按固化剂类型自动估算）
    """
    extractor = CrosslinkedFeatureExtractor()
    return extractor.extract_crosslink_features(
        epoxy_smiles, curer_smiles, target_conversion
    )


def get_reaction_product_smiles(
    epoxy_smiles: str,
    curer_smiles: str,
    conversion: float = None
) -> Optional[str]:
    """
    便捷函数：获取反应产物SMILES
    """
    simulator = EpoxyReactionSimulator()
    return simulator.generate_crosslinked_fragment(
        epoxy_smiles, curer_smiles,
        target_conversion=conversion
    )


def batch_extract_crosslink_features(
    df: pd.DataFrame,
    epoxy_col: str = 'Epoxy_SMILES',
    curer_col: str = 'Curer_SMILES',
    conversion: float = 0.5
) -> pd.DataFrame:
    """
    便捷函数：批量提取交联特征
    """
    extractor = CrosslinkedFeatureExtractor()
    return extractor.batch_extract_features(
        df, epoxy_col, curer_col, 
        target_conversion=conversion
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 测试用SMILES
    # DGEBA (双酚A二缩水甘油醚)
    dgeba = "C1OC1COc2ccc(C(C)(C)c3ccc(OCC4CO4)cc3)cc2"
    
    # DDM (4,4'-二氨基二苯甲烷，常用胺类固化剂)
    ddm = "Nc1ccc(Cc2ccc(N)cc2)cc1"
    
    # MTHPA (甲基四氢邻苯二甲酸酐，酸酐类固化剂)
    mthpa = "CC1CC2C(=O)OC(=O)C2C1"
    
    print("=" * 60)
    print("环氧树脂-固化剂反应模拟测试")
    print("=" * 60)
    
    simulator = EpoxyReactionSimulator(verbose=True)
    
    # 测试官能团识别
    print("\n1. 官能团识别:")
    print(f"   DGEBA官能团: {simulator.identify_functional_groups(dgeba)}")
    print(f"   DDM官能团: {simulator.identify_functional_groups(ddm)}")
    print(f"   MTHPA官能团: {simulator.identify_functional_groups(mthpa)}")
    
    # 测试固化剂类型检测
    print("\n2. 固化剂类型检测:")
    print(f"   DDM类型: {simulator.detect_curer_type(ddm)}")
    print(f"   MTHPA类型: {simulator.detect_curer_type(mthpa)}")
    
    # 测试反应模拟
    print("\n3. 反应模拟 (DGEBA + DDM):")
    products = simulator.simulate_curing(dgeba, ddm, n_reactions=1)
    for i, prod in enumerate(products):
        print(f"   产物 {i+1}: MW={prod['mol_weight']:.1f}, 剩余环氧基={prod['remaining_epoxide']}")
    
    # 测试特征提取
    print("\n4. 交联特征提取:")
    extractor = CrosslinkedFeatureExtractor(verbose=True)
    features = extractor.extract_crosslink_features(dgeba, ddm, target_conversion=0.5)
    for k, v in features.items():
        print(f"   {k}: {v}")
    
    print("\n✅ 测试完成!")
