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


def _junction_site_count(mol):
    # 交联位点：已反应的胺N / 硫醚S（排除酰胺N与芳香n）
    total = 0
    try:
        p1 = Chem.MolFromSmarts('[NX3;H0;!$([n]);!$([NX3]-[CX3]=[OX1])]')
        p2 = Chem.MolFromSmarts('[NX3;H1;!$([n]);$(N(-[#6])-[#6]);!$([NX3]-[CX3]=[OX1])]')
        p3 = Chem.MolFromSmarts('[#16X2;$(S(-[#6])-[#6])]')
        for p in (p1, p2, p3):
            if p is not None:
                total += len(mol.GetSubstructMatches(p))
    except Exception:
        pass
    return total


def _graph_invariants_from_mol(mol):
    # 图论网络不变量：图直径 / 平均最短路径 / Wiener指数 / 环数
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
        diameters = []
        apls = []
        wiener = 0
        for c in comps:
            sg = G.subgraph(c)
            nn = sg.number_of_nodes()
            if nn < 2:
                continue
            try:
                diameters.append(nx.diameter(sg))
            except Exception:
                pass
            paths = dict(nx.all_pairs_shortest_path_length(sg))
            total = 0
            for _src, dists in paths.items():
                total += sum(dists.values())
            wiener += total
            apls.append(total / (nn * (nn - 1)))
        out['product_graph_diameter'] = float(max(diameters)) if diameters else 0.0
        out['product_graph_avg_path_length'] = float(np.mean(apls)) if apls else 0.0
        out['product_wiener_index'] = float(wiener)
        out['product_cyclomatic_number'] = float(mol.GetNumBonds() - mol.GetNumHeavyAtoms() + len(comps))
    except Exception:
        pass
    return out


def _compute_product_3d_descriptors(smiles):
    # 3D构象特征（ETKDGv3 + 随机坐标兜底），失败返回空dict
    out = {}
    if not RDKIT_AVAILABLE:
        return out
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out
        n_frags = len(Chem.GetMolFrags(mol))
        if mol.GetNumHeavyAtoms() > 130 or n_frags > 6:
            return out
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            return out
        out['product_3d_radius_of_gyration'] = float(rdMolDescriptors.CalcRadiusOfGyration(mol))
        out['product_3d_pmi1'] = float(rdMolDescriptors.CalcPMI1(mol))
        out['product_3d_pmi2'] = float(rdMolDescriptors.CalcPMI2(mol))
        out['product_3d_pmi3'] = float(rdMolDescriptors.CalcPMI3(mol))
        out['product_3d_npr1'] = float(rdMolDescriptors.CalcNPR1(mol))
        out['product_3d_npr2'] = float(rdMolDescriptors.CalcNPR2(mol))
        out['product_3d_spherocity'] = float(rdMolDescriptors.CalcSpherocityIndex(mol))
        out['product_3d_labute_asa'] = float(rdMolDescriptors.CalcLabuteASA(mol))
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
                out['product_mol_weight'] = float(Descriptors.MolWt(mol))
                out['product_num_atoms'] = float(mol.GetNumAtoms())
                out['product_num_heavy_atoms'] = float(mol.GetNumHeavyAtoms())
                rb = _desc_safe(mol, 'NumRotatableBonds')
                out['product_num_rotatable_bonds'] = float(rb) if rb is not None else 0.0
                out['product_tpsa'] = float(Descriptors.TPSA(mol))
                out['product_logp'] = float(Descriptors.MolLogP(mol))
                out['product_molar_refractivity'] = float(Descriptors.MolMR(mol))
                out['product_num_h_donors'] = float(Descriptors.NumHDonors(mol))
                out['product_num_h_acceptors'] = float(Descriptors.NumHAcceptors(mol))
                out['product_num_rings'] = float(Descriptors.RingCount(mol))
                out['product_num_aromatic_rings'] = float(Descriptors.NumAromaticRings(mol))
                out['product_num_aliphatic_rings'] = float(Descriptors.NumAliphaticRings(mol))
                out['product_fraction_csp3'] = float(Descriptors.FractionCsp3(mol))
                out['product_num_saturated_rings'] = float(Descriptors.NumSaturatedRings(mol))
                out['product_num_heteroatoms'] = float(Descriptors.NumHeteroatoms(mol))
                # 官能团（交联相关）
                out['product_num_hydroxyl'] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))))
                out['product_num_amine'] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2,H1,H0]'))))
                out['product_num_ether'] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OD2]([#6])[#6]'))))
                # Layer1: 固化专用结构特征
                out['product_residual_epoxide'] = float(_epoxide_count_of_mol(mol))
                out['product_num_junction_sites'] = float(_junction_site_count(mol))
                # Layer2: 拓扑指数
                for k, nm in [('product_kappa1', 'Kappa1'), ('product_kappa2', 'Kappa2'), ('product_kappa3', 'Kappa3'),
                              ('product_chi0n', 'Chi0n'), ('product_chi1n', 'Chi1n'), ('product_chi2n', 'Chi2n'),
                              ('product_chi3n', 'Chi3n'), ('product_chi4n', 'Chi4n'),
                              ('product_balaban_j', 'BalabanJ'), ('product_bertz_ct', 'BertzCT'),
                              ('product_hall_kier_alpha', 'HallKierAlpha'),
                              ('product_max_estate', 'MaxEStateIndex'), ('product_min_estate', 'MinEStateIndex')]:
                    v = _desc_safe(mol, nm)
                    if v is not None:
                        out[k] = v
                # Layer5: 图论不变量
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

# 环氧-酸酐反应：环氧 + 酸酐 -> 酯键
EPOXY_ANHYDRIDE_RXN = ReactionTemplate(
    name="epoxy_anhydride",
    smirks="[C:1]1[O:2][C:3]1.[C:4](=[O:5])[O:6][C:7](=[O:8])>>[C:1]([O:2][C:4](=[O:5]))[C:3][O:6].[O:8]=[C:7]",
    description="环氧基与酸酐反应，生成酯键",
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
ALL_REACTION_TEMPLATES = [
    EPOXY_PRIMARY_AMINE_RXN,
    EPOXY_SECONDARY_AMINE_RXN,
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
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    
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

        # 2. BigSMILES 特征解包与降级（针对高分子化学文献中的低聚物表达形式）
        if '{' in s or ('[' in s and ('>' in s or '<' in s)):
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
        use_typical_values: bool = True
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
            curer_functionality = curer_fg.get('anhydride', 0) * 2
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
            time_factor = 1.0 - np.exp(-curing_time / 2.0)
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

    def estimate_conversion_multicomponent(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        stoichiometry_r: float,
        curing_temp: float = 150.0,
        curing_time: float = 2.0
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
                curer_func = curer_fg.get('anhydride', 0) * 2
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
            time_factor = 1.0 - np.exp(-curing_time / 2.0)
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

    def _simulate_weighted_average(
        self,
        resin_components: List[Tuple[str, float]],
        curer_components: List[Tuple[str, float]],
        target_conversion: float
    ) -> Dict[str, any]:
        """
        方案1：加权平均法

        选择主要组分进行反应，其他组分的特征加权平均
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
                func = curer_fg.get('anhydride', 0) * 2
            elif curer_type == 'thiol':
                func = curer_fg.get('thiol', 0)
            else:
                func = 1
            weighted_curer_func += func * w

        stoich_r = weighted_curer_func / weighted_epoxy_func if weighted_epoxy_func > 0 else 0.0

        # 使用主要组分生成产物
        product_repr = self.get_product_representation(
            main_resin_smi,
            main_curer_smi,
            stoichiometry=stoich_r,
            target_conversion=target_conversion,
            output_format='auto'
        )

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
            'representative_bigsmiles': product_repr.get('bigsmiles')
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
                    curer_func = curer_fg.get('anhydride', 0) * 2
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

        # 选择概率最大的作为代表
        if products:
            representative = max(products, key=lambda x: x['probability'])
            representative_smiles = representative['product_smiles']
            representative_bigsmiles = representative['product_bigsmiles']
        else:
            representative_smiles = None
            representative_bigsmiles = None

        return {
            'method': 'combinatorial',
            'products': products,
            'n_combinations': len(products),
            'representative_smiles': representative_smiles,
            'representative_bigsmiles': representative_bigsmiles
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
            for prod_tuple in products:
                for prod in prod_tuple:
                    try:
                        # 清理和标准化产物
                        Chem.SanitizeMol(prod)
                        valid_products.append(prod)
                    except Exception:
                        continue
            
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
            products.extend(self._run_single_reaction(epoxy_mol, curer_mol, 'epoxy_anhydride'))
            
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
            for t_name in ['epoxy_primary_amine', 'epoxy_secondary_amine', 'epoxy_anhydride', 'epoxy_phenol', 'epoxy_thiol', 'epoxy_carboxylic_acid']:
                prods = self._run_single_reaction(epoxy_mol, curer_mol, t_name)
                if prods:
                    products.extend(prods)
                    break
        
        # 转换为SMILES
        product_smiles = []
        for prod in products:
            try:
                smi = Chem.MolToSmiles(prod)
                if smi:
                    product_smiles.append(smi)
            except Exception:
                continue
        
        return list(set(product_smiles))  # 去重
    
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
        # 统计活性氢：伯胺/仲胺N-H、硫醇S-H、酚OH、羧酸OH
        if not RDKIT_AVAILABLE:
            return 0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            n = 0
            n += 2 * len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2;!$([n]);!$([NX3]-[CX3]=[OX1])]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H1;!$([n]);!$([NX3]-[CX3]=[OX1])]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SX2;H1]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2;H1][c]')))
            n += len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2;H1]')))
            return n
        except Exception:
            return 0

    def build_network_unit(
        self,
        epoxy_smiles: str,
        curer_smiles: str,
        target_conversion: float = 0.85,
        max_reactions: int = 8
    ) -> Optional[str]:
        # 化学计量驱动的交联网络枢纽片段构建（带进程内缓存）
        # 以 1 个固化剂分子为交联枢纽，按目标转化率迭代开环接入新鲜环氧单体：
        #   反应步数 n = round(alpha x 枢纽活性氢数)
        #   每步优先消耗枢纽上活性最高位点（伯胺H > 仲胺H > 酚OH/羧酸OH > 硫醇H）
        #   分支末端保留未反应环氧基（代表网络中指向相邻枢纽的悬键）
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

        cache_key = (r_act, c_act, round(a_conv, 3))
        cache = getattr(self, '_unit_cache', None)
        if cache is None:
            cache = {}
            self._unit_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        unit = self._build_network_unit_impl(r_act, c_act, a_conv, max_reactions)
        cache[cache_key] = unit
        return unit

    def _build_network_unit_impl(
        self,
        r_act: str,
        c_act: str,
        a_conv: float,
        max_reactions: int
    ) -> Optional[str]:
        try:
            h_total = self._count_reactive_h(c_act)

            if h_total <= 0:
                # 枢纽无活性氢（异氰酸酯/未知类型等）：单步或虚拟交联保底
                prods = self.simulate_single_step(r_act, c_act)
                if prods:
                    return prods[0]
                return self._virtual_crosslink_safe(r_act, c_act)

            # 种子反应：1 个环氧单体 + 1 个枢纽固化剂
            seed_prods = self.simulate_single_step(r_act, c_act)
            if not seed_prods:
                return self._virtual_crosslink_safe(r_act, c_act)

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

            return product
        except Exception:
            return None

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
                target_conversion=target_conversion if target_conversion is not None else 0.5
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
        stoichiometry: float = 1.0
    ) -> str:
        """
        生成BigSMILES交联网络表示

        用于高转化率（>70%），表示三维交联网络拓扑

        Returns:
            BigSMILES字符串，格式如: {[$]CC(O)CN(CC(O)C[$])CC(O)C[$]}
        """
        epoxy_act = self._to_reactive_smiles(epoxy_smiles) or self._clean_smiles(epoxy_smiles)
        curer_act = self._to_reactive_smiles(curer_smiles) or self._clean_smiles(curer_smiles)

        # 识别官能团
        epoxy_fg = self.identify_functional_groups(epoxy_act)
        curer_type, curer_fg = self.detect_curer_type(curer_act)

        epoxy_functionality = epoxy_fg.get('epoxide', 0)

        # 生成一个代表性的反应单元
        products = self.simulate_curing(
            epoxy_act,
            curer_act,
            n_reactions=1,  # 只需要一步反应的产物
            max_products=3
        )

        if not products:
            # 如果反应失败，生成简化的BigSMILES
            return self._generate_simplified_bigsmiles(epoxy_smiles, curer_smiles, curer_type)

        # 获取反应单元的SMILES
        unit_smiles = products[0]['smiles']

        try:
            unit_mol = Chem.MolFromSmiles(unit_smiles)
            if unit_mol is None:
                return self._generate_simplified_bigsmiles(epoxy_smiles, curer_smiles, curer_type)

            # 识别可能的交联点（剩余的环氧基或羟基）
            unit_fg = self.identify_functional_groups(unit_smiles)
            remaining_epoxy = unit_fg.get('epoxide', 0)
            hydroxyl_count = unit_fg.get('hydroxyl', 0)

            # 构建BigSMILES
            # 策略：在反应单元周围添加连接点标记 [$]
            if epoxy_functionality >= 2:
                # 多官能度环氧：形成交联网络
                # 格式: {[$]反应单元[$]}，表示可以从多个位置交联
                bigsmiles = f"{{[$]{unit_smiles}[$]}}"
            else:
                # 单官能度：线性聚合物
                bigsmiles = f"{{[>]{unit_smiles}[<]}}"

            return bigsmiles

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

        # 根据output_format决定表示方法
        if output_format == 'smiles':
            # 强制SMILES
            smiles = self.generate_crosslinked_fragment(
                epoxy_smiles, curer_smiles, stoichiometry, target_conversion
            )
            result['representation_type'] = 'smiles'
            result['smiles'] = smiles
            result['bigsmiles'] = None
            result['description'] = '低聚物片段（SMILES）'

        elif output_format == 'bigsmiles':
            # 强制BigSMILES
            bigsmiles = self._generate_bigsmiles_network(
                epoxy_smiles, curer_smiles, stoichiometry
            )
            result['representation_type'] = 'bigsmiles'
            result['smiles'] = None
            result['bigsmiles'] = bigsmiles
            result['description'] = '交联网络（BigSMILES）'

        elif output_format == 'both':
            # 返回两者
            smiles = self.generate_crosslinked_fragment(
                epoxy_smiles, curer_smiles, stoichiometry, target_conversion
            )
            bigsmiles = self._generate_bigsmiles_network(
                epoxy_smiles, curer_smiles, stoichiometry
            )
            result['representation_type'] = 'both'
            result['smiles'] = smiles
            result['bigsmiles'] = bigsmiles
            result['description'] = '低聚物片段 + 交联网络'

        else:  # output_format == 'auto'
            # 根据转化率自动选择
            if target_conversion < 0.3:
                # 低转化率：返回SMILES
                smiles = self.generate_crosslinked_fragment(
                    epoxy_smiles, curer_smiles, stoichiometry, target_conversion
                )
                result['representation_type'] = 'smiles'
                result['smiles'] = smiles
                result['bigsmiles'] = None
                result['description'] = f'低转化率（{target_conversion*100:.0f}%）- 低聚物片段'

            elif target_conversion < 0.7:
                # 中等转化率：返回带端基的SMILES
                smiles = self._generate_oligomer_smiles(
                    epoxy_smiles, curer_smiles, stoichiometry, target_conversion
                )
                result['representation_type'] = 'oligomer'
                result['smiles'] = smiles
                result['bigsmiles'] = None
                result['description'] = f'中等转化率（{target_conversion*100:.0f}%）- 部分交联低聚物'

            else:
                # 高转化率：生成BigSMILES + SMILES片段
                smiles = self.generate_crosslinked_fragment(
                    epoxy_smiles, curer_smiles, stoichiometry, target_conversion
                )
                bigsmiles = self._generate_bigsmiles_network(
                    epoxy_smiles, curer_smiles, stoichiometry
                )
                result['representation_type'] = 'bigsmiles'
                result['smiles'] = smiles  # 保留片段用于特征提取
                result['bigsmiles'] = bigsmiles
                result['description'] = f'高转化率（{target_conversion*100:.0f}%）- 交联网络'

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

        features['primary_amine_count'] = curer_fg.get('primary_amine', 0) + curer_fg.get('aromatic_amine', 0)
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
            curer_functionality = features['anhydride_count'] * 2  # 酸酐开环后有2个反应位点
        elif curer_type == 'thiol':
            curer_functionality = features['thiol_count']
        else:
            curer_functionality = 1

        features['epoxy_functionality'] = epoxy_functionality
        features['curer_functionality'] = curer_functionality

        # 化学计量比 (r = 活性氢当量 / 环氧当量)
        if epoxy_functionality > 0:
            features['stoichiometry_r'] = curer_functionality / epoxy_functionality
        else:
            features['stoichiometry_r'] = 0.0
        
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

        # 保存固化条件
        features['curing_temp'] = curing_temp
        features['curing_time'] = curing_time

        # 3. 模拟反应产物特征（使用智能表示方法）
        try:
            # 使用新的智能表示方法
            product_repr = self.simulator.get_product_representation(
                epoxy_smiles, curer_smiles,
                stoichiometry=r,
                target_conversion=actual_conversion,  # 使用估算或提供的转化率
                output_format='auto'  # 根据转化率自动选择
            )

            # 保存表示类型和描述
            features['representation_type'] = product_repr.get('representation_type', 'unknown')
            features['representation_description'] = product_repr.get('description', '')

            # 保存SMILES（如果有）并提取完整分层产物描述符
            product_smi = product_repr.get('smiles')
            if product_smi:
                features['product_smiles'] = product_smi
                prod_desc = _compute_product_descriptors(product_smi, include_3d=True)
                features.update(prod_desc)
                features.setdefault('product_remaining_epoxide', prod_desc.get('product_residual_epoxide', 0.0))
                features.setdefault('product_hydroxyl_count', prod_desc.get('product_num_hydroxyl', 0.0))
                # 转化率代理（枢纽活性氢消耗近似，夹取到[0,1]）
                if features['epoxide_count'] > 0:
                    consumed = max(0.0, features['epoxide_count'] - prod_desc.get('product_residual_epoxide', 0.0))
                    features['estimated_conversion'] = min(1.0, consumed / features['epoxide_count'])
                else:
                    features['estimated_conversion'] = 0.0
            else:
                features['product_smiles'] = None

            # 保存BigSMILES（如果有）
            product_bigsmiles = product_repr.get('bigsmiles')
            if product_bigsmiles:
                features['product_bigsmiles'] = product_bigsmiles
            else:
                features['product_bigsmiles'] = None

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 产物特征提取失败: {e}")
            features['product_smiles'] = None
            features['product_bigsmiles'] = None
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
    verbose: bool = False
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

            # 6. 提取特征
            features = ext.extract_multicomponent_features(
                resin_components,
                curer_components,
                stoichiometry_r=stoich_r,
                target_conversion=target_conv,
                curing_temp=curing_temp,
                curing_time=curing_time,
                auto_estimate_conversion=auto_estimate_conversion,
                reaction_method=reaction_method
            )

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
        reaction_method: str = 'weighted'
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
                curer_func = curer_fg.get('anhydride', 0) * 2
            elif curer_type == 'thiol':
                curer_func = curer_fg.get('thiol', 0)
            else:
                curer_func = 1

            weighted_curer_func += curer_func * curer_weight

        features['weighted_epoxy_functionality'] = weighted_epoxy_func
        features['weighted_curer_functionality'] = weighted_curer_func
        features['stoichiometry_r'] = stoichiometry_r

        # 1.5 注入高分子物理交联网络与多组分动力学特征（复用常驻引擎与分子缓存）
        try:
            if not hasattr(self, '_mechanism_engine') or self._mechanism_engine is None:
                from core.epoxy_mechanism_features import EpoxyMechanismEngine
                self._mechanism_engine = EpoxyMechanismEngine(verbose=self.verbose)
            _engine = self._mechanism_engine
            _res_items = []
            for smi, w in resin_components:
                c_key = f"R|{smi}"
                if hasattr(self, '_mol_prop_cache') and c_key in self._mol_prop_cache:
                    p = dict(self._mol_prop_cache[c_key])
                else:
                    p = _engine.calc_single_molecule_properties(smi, is_resin=True)
                    if hasattr(self, '_mol_prop_cache'):
                        self._mol_prop_cache[c_key] = p
                p['weight'] = w
                _res_items.append(p)
            _cur_items = []
            for smi, w in curer_components:
                c_key = f"H|{smi}"
                if hasattr(self, '_mol_prop_cache') and c_key in self._mol_prop_cache:
                    p = dict(self._mol_prop_cache[c_key])
                else:
                    p = _engine.calc_single_molecule_properties(smi, is_resin=False)
                    if hasattr(self, '_mol_prop_cache'):
                        self._mol_prop_cache[c_key] = p
                p['weight'] = w
                _cur_items.append(p)
            _mech = _engine.compute_formulation_mechanism_features(
                _res_items, _cur_items, given_r_value=stoichiometry_r
            )
            for k, v in _mech.items():
                clean_k = k.replace('mech_', '')
                features[clean_k] = v
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 机理特征注入跳过: {e}")

        # 2. 估算转化率
        if target_conversion is None and auto_estimate_conversion:
            estimated_conversion = self.simulator.estimate_conversion_multicomponent(
                resin_components,
                curer_components,
                stoichiometry_r,
                curing_temp,
                curing_time
            )
            # 不保存 conversion_source，只保存转化率值
            actual_conversion = estimated_conversion
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
                bigsmiles_result = reaction_result.get('representative_bigsmiles')

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

                # 产物结构字符串：首选 BigSMILES (若有)，否则 SMILES
                features['product_structure'] = bigsmiles_result or smiles_result

                # 提取产物扩展物理/化学特征（2D拓扑 + 交联位点 + 图论 + 3D构象）
                if smiles_result:
                    extended_features = self._extract_extended_product_features(smiles_result)
                    features.update(extended_features)
                    try:
                        mol_r = Chem.MolFromSmiles(reaction_result.get('main_resin', '') or '')
                        mw_r = Descriptors.MolWt(mol_r) if mol_r is not None else 0.0
                        if mw_r > 0 and extended_features.get('product_mol_weight'):
                            features['product_degree_of_polymerization'] = extended_features['product_mol_weight'] / mw_r
                    except Exception:
                        pass

            elif reaction_method == 'combinatorial':
                # 方案2：组合反应法
                features['n_combinations'] = reaction_result.get('n_combinations', 0)
                smiles_result = reaction_result.get('representative_smiles')
                bigsmiles_result = reaction_result.get('representative_bigsmiles')

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

                features['product_structure'] = bigsmiles_result or smiles_result

                # 提取扩展的产物特征（2D拓扑 + 交联位点 + 图论 + 3D构象）
                if smiles_result:
                    extended_features = self._extract_extended_product_features(smiles_result)
                    features.update(extended_features)
                    try:
                        main_r_smi_dp, _ = max(resin_components, key=lambda x: x[1])
                        mol_r = Chem.MolFromSmiles(main_r_smi_dp or '')
                        mw_r = Descriptors.MolWt(mol_r) if mol_r is not None else 0.0
                        if mw_r > 0 and extended_features.get('product_mol_weight'):
                            features['product_degree_of_polymerization'] = extended_features['product_mol_weight'] / mw_r
                    except Exception:
                        pass

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
                        prefix, self.verbose
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

                # 提取特征
                features = self.extract_multicomponent_features(
                    resin_components,
                    curer_components,
                    stoichiometry_r=stoich_r,
                    target_conversion=target_conv,
                    curing_temp=curing_temp,
                    curing_time=curing_time,
                    auto_estimate_conversion=auto_estimate_conversion,
                    reaction_method=reaction_method
                )

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
    target_conversion: float = 0.5
) -> Dict[str, float]:
    """
    便捷函数：提取交联特征
    """
    extractor = CrosslinkedFeatureExtractor()
    return extractor.extract_crosslink_features(
        epoxy_smiles, curer_smiles, target_conversion
    )


def get_reaction_product_smiles(
    epoxy_smiles: str,
    curer_smiles: str,
    conversion: float = 0.5
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
