# -*- coding: utf-8 -*-
"""自动分子特征解析（Auto Molecular Feature Resolver）

在"外部模型特征补齐"流程中，外部模型需要的特征可能有三种来源：

    1. 工作区数据里已有列            → 直接用（exact / 别名 / 归一化）
    2. 能从工作区的结构列**现场提取**  → 自动提取（复用虚拟筛选的提取引擎）
    3. 工作区既没有、也算不出来        → 去**总表**（参考数据集）按结构查

本模块负责 2 和 3，让用户不必手工映射。

提取后端（与"虚拟分子筛选"完全一致，通过 core.virtual_screening.extract_features_from_config）：
    - RDKit 标准/并行/内存优化版
    - 分子指纹（MACCS / Morgan / RDKit FP …）
    - Mordred（1800+ 描述符）
    - 3D 构象描述符（含 Coulomb 矩阵）
    - Transformer Embedding（ChemBERTa 等）
    - ML 力场（torchANI）
    - 快速力场（UFF/MMFF）
    - xTB（半经验量子化学）
    - FGD（官能团描述符）
    - 环氧树脂领域特征（含交联反应模拟）
    - 图神经网络（GNN 特征化）

优先级策略：
    轻量方法优先（RDKit → 指纹 → FGD → Mordred → 3D → 力场 → xTB → 嵌入 → GNN），
    按"能否产出目标特征名"逐个尝试，命中即停。也可由用户显式指定方法。
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .external_feature_augmenter import clean_smiles, is_missing, normalize_name

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 提取方法注册表：与虚拟筛选的 method 字符串保持一致
#   每项：(显示名, method 关键字, params, 成本等级)
# ---------------------------------------------------------------------------
EXTRACTION_METHODS: List[Dict[str, Any]] = [
    {
        "key": "rdkit",
        "label": "RDKit 描述符（标准版）",
        "method": "RDKit 标准版",
        "params": {},
        "cost": 1,
        "desc": "200+ 基础描述符：MW/LogP/TPSA/环数/官能团…",
    },
    {
        "key": "fingerprint",
        "label": "分子指纹（MACCS）",
        "method": "分子指纹",
        "params": {"fp_type": "MACCS", "fp_bits": 167, "fp_radius": 2},
        "cost": 1,
        "desc": "MACCS 167 位结构键指纹",
    },
    {
        "key": "morgan",
        "label": "分子指纹（Morgan/ECFP）",
        "method": "分子指纹",
        "params": {"fp_type": "Morgan", "fp_bits": 2048, "fp_radius": 2},
        "cost": 1,
        "desc": "Morgan 圆形指纹，适合相似性/活性预测",
    },
    {
        "key": "fgd",
        "label": "FGD 官能团描述符",
        "method": "FGD",
        "params": {"fgd_multi_label": True, "fgd_keep_largest_frag": True},
        "cost": 1,
        "desc": "官能团计数向量",
    },
    {
        "key": "mordred",
        "label": "Mordred（1800+ 描述符）",
        "method": "Mordred",
        "params": {
            "mordred_batch_size": 500,
            "mordred_ignore_3d": True,
            # [Windows 稳定性] Mordred 默认在 Windows 上开 min(cpu-2, 16) 个进程，
            # 在 Streamlit 内会触发 WinError 1450（系统资源不足）。单进程稳定。
            "mordred_n_jobs": 1,
        },
        "cost": 3,
        "desc": "全量 2D 分子描述符，覆盖最广",
    },
    {
        "key": "rdkit3d",
        "label": "3D 构象描述符",
        "method": "3D构象",
        "params": {"rdkit3d_coulomb_top_k": 10},
        "cost": 4,
        "desc": "需要 3D 嵌入，含 Coulomb 矩阵特征",
    },
    {
        "key": "quick_ff",
        "label": "快速力场（UFF/MMFF）",
        "method": "快速力场",
        "params": {"ff_mode": "auto", "ff_minimize": True, "ff_max_iters": 200},
        "cost": 4,
        "desc": "力场优化后的能量/几何特征",
    },
    {
        "key": "epoxy",
        "label": "环氧树脂领域特征",
        "method": "环氧树脂",
        "params": {"enable_reaction_simulation": False},
        "cost": 5,
        "desc": "EEW/AHEW/化学计量/交联网络（需要树脂+固化剂）",
    },
    {
        "key": "ani",
        "label": "ML 力场（torchANI）",
        "method": "ML力场",
        "params": {"ani_batch_size": 256},
        "cost": 6,
        "desc": "神经网络势能面特征（需 GPU 更快）",
    },
    {
        "key": "xtb",
        "label": "xTB 半经验量子化学",
        "method": "xTB",
        "params": {"xtb_method": "gfn2", "xtb_run_mode": "sp"},
        "cost": 8,
        "desc": "量子化学级别电子结构特征",
    },
    {
        "key": "transformer",
        "label": "Transformer 嵌入（ChemBERTa）",
        "method": "Transformer Embedding",
        "params": {"lm_model_name": "seyonec/ChemBERTa-zinc-base-v1", "lm_pooling": "cls"},
        "cost": 6,
        "desc": "预训练语言模型嵌入向量",
    },
    {
        "key": "gnn",
        "label": "图神经网络特征化",
        "method": "图神经网络",
        "params": {},
        "cost": 5,
        "desc": "分子图张量化（需 torch_geometric）",
    },
]

#: 默认尝试顺序（成本从低到高）
DEFAULT_METHOD_ORDER = ["rdkit", "fingerprint", "morgan", "fgd", "epoxy", "mordred", "rdkit3d", "quick_ff"]


#: 跨库特征别名：不同提取器对同一物理量的命名差异
#:   键 = 期望/常见名称（归一化后），值 = 提取器实际列名（归一化后）
#:   注意：键值都是 normalize_name() 之后的字符串（会自动单数化、去 num 前缀）
_FEATURE_ALIASES_CROSS_LIBRARY: Dict[str, str] = {
    # ---- 环计数（Mordred: nARing/nRing；RDKit: NumAromaticRings/RingCount）----
    "naromring": "naring",       # nAromRings / NumAromaticRings → nARing
    "arring": "naring",
    "ringcount": "nring",
    "numring": "nring",
    "ring": "nring",
    "naliring": "nalring",
    "aliring": "nalring",
    "numaliphaticring": "nalring",
    "numaromaticring": "naring",
    "naromatom": "naromatom",
    "narombond": "narombond",
    # ---- 尺寸 ----
    "heavy": "nheavyatom",
    "hac": "nheavyatom",
    "heavyatom": "nheavyatom",
    "heavyatomcount": "nheavyatom",
    "numheavyatom": "nheavyatom",
    "numatom": "natom",
    "atomcount": "natom",
    "numbond": "nbond",
    "bondcount": "nbond",
    "numrotatablebond": "nrot",
    "rotatablebond": "nrot",
    "rotb": "nrot",
    "numheteroatom": "nhetero",
    "heteroatom": "nhetero",
    "numstereocenter": "nstereo",
    "numamidebond": "namidebond",
    # ---- 疏水 / 极性 ----
    "logp": "slogp",
    "clogp": "slogp",
    "xlogp": "slogp",
    "mollogp": "slogp",
    "molarlogp": "slogp",
    "molarrefractivity": "mrmol",
    "molrefractivity": "mrmol",
    "molecularweight": "mw",
    "molwt": "mw",
    "molweight": "mw",
    "exactmw": "amw",
    "fsp3": "fcsp3",
    "fractioncsp3": "fcsp3",
    # ---- 氢键：Mordred 默认无 nHBDon/nHBAcc，仅当提取器确实提供时才映射 ----
    "nhbd": "nhbdon",
    "hbd": "nhbdon",
    "numhbd": "nhbdon",
    "hbonddonor": "nhbdon",
    "nhba": "nhbacc",
    "hba": "nhbacc",
    "numhba": "nhbacc",
    "hbondacceptor": "nhbacc",
    # RDKit 实际列名
    "numhdonor": "numhdonor",
    "numhaccept": "numhaccept",
    "numhacceptors": "numhacceptors",
    "nhohcount": "nhohcount",
    "nnhcount": "nnhcount",
}


def _token_set(name: Any) -> frozenset:
    """把列名拆成有意义的词集合（去停用词、单数化），用于词集匹配。"""
    key = normalize_name(name)
    stop = {"num", "n", "count", "total", "number", "of", "the"}
    tokens = set()
    for part in key.split("_"):
        if not part or part in stop:
            continue
        if len(part) > 3 and part.endswith("s"):
            part = part[:-1]
        tokens.add(part)
    return frozenset(tokens)


def _singular(key: str) -> str:
    """把整个字符串末尾的复数 s 去掉（处理 naromrings → naromring）。

    normalize_name 只在有下划线分隔时才单数化（如 num_aromatic_rings），
    但 Mordred 的驼峰名（nAromRings）归一化后是无下划线的 naromrings，
    需要单独处理，否则别名表键对不上。
    """
    if len(key) > 3 and key.endswith("s") and not key.endswith(("ss", "us", "is")):
        return key[:-1]
    return key


#: 别名表的双向索引：载入时全部归一化，避免键/值格式不一致
#:   用 _norm_key 同时注册单数/复数两种写法
_ALIAS_LOOKUP: Dict[str, str] = {}


def _norm_key(name: Any) -> str:
    """别名表专用归一化：小写 + 去非字母数字 + 末尾单数化（保留 num 前缀）。"""
    key = re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    return _singular(key)


for _alias_raw, _canonical_raw in _FEATURE_ALIASES_CROSS_LIBRARY.items():
    _a = _norm_key(_alias_raw)
    _c = _norm_key(_canonical_raw)
    if _a and _c:
        _ALIAS_LOOKUP[_a] = _c
        _ALIAS_LOOKUP.setdefault(_a + "s", _c)

#: 反向索引：标准名 → 别名（用于 feature 是标准名、列是别名的情况）
_ALIAS_REVERSE: Dict[str, List[str]] = {}
for _a, _c in _ALIAS_LOOKUP.items():
    _ALIAS_REVERSE.setdefault(_c, []).append(_a)


#: 已知的“陷阱词”：这些特征名不能被宽松子串命中，因为会被截断匹配
#: 例：nHBDon 会被 substring 命中 Mordred 的 nH（氢原子总数），语义完全不同
_SUBSTRING_GUARD = ("nhbdon", "nhbacc", "nhbd", "nhba", "hbd", "hba", "hdonor", "haccept")


# ---------------------------------------------------------------------------
# SMARTS 库（轻量路径：不依赖重后端时的直接计算）
# ---------------------------------------------------------------------------
SMARTS_LIBRARY: Dict[str, str] = {
    "epoxide": "C1OC1", "epoxy": "C1OC1", "oxirane": "C1OC1",
    "hydroxyl": "[OX2H]", "oh": "[OX2H]",
    "primary_amine": "[NX3;H2;!$(N-C=O);!$(N-S);!$(N=*)]",
    "secondary_amine": "[NX3;H1;!$(N-C=O);!$(N-S);!$(N=*)]",
    "tertiary_amine": "[NX3;H0;!$(N-C=O);!$(N-S);!$(N=*)]",
    "amine": "[NX3;!$(N-C=O);!$(N-S);!$(N=*)]",
    "aromatic_amine": "[NX3;!$(N-C=O);!$(N-S);!$(N=*)]-c",
    "anhydride": "C(=O)OC(=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "amide": "[NX3][CX3]=[OX1]",
    "ether": "[OD2]([#6])[#6]",
    "thiol": "[SX2H]",
    "sulfide": "[SD2]([#6])[#6]",
    "sulfone": "S(=O)(=O)",
    "nitrile": "C#N",
    "isocyanate": "N=C=O",
    "halogen": "[F,Cl,Br,I]",
    "fluorine": "[F]", "chlorine": "[Cl]", "bromine": "[Br]",
    "silicon": "[Si]", "phosphorus": "[P]", "boron": "[B]",
    "aromatic_ring": "c1ccccc1", "benzene_ring": "c1ccccc1",
    "vinyl": "C=C", "alkyne": "C#C", "imine": "C=N",
    "urea": "NC(=O)N", "urethane": "NC(=O)O", "carbonate": "OC(=O)O",
    "nitro": "[N+](=O)[O-]",
    "heteroatom": "[!#6;!#1]",
    "ring5": "[R]1[R][R][R][R]1", "ring6": "[R]1[R][R][R][R][R]1",
    "branch_carbon": "[CX4;D4]",
}


def _count_smarts(mol: "Chem.Mol", smarts: str) -> int:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(pattern))
    except Exception:
        return 0


def _largest_fragment(mol: "Chem.Mol") -> "Chem.Mol":
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            return max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        pass
    return mol


def _m_desc(attr: str, fn=None):
    def _inner(mol):
        try:
            return float(fn(mol) if fn else getattr(Descriptors, attr)(mol))
        except Exception:
            return np.nan
    return _inner


def _rd(attr: str, fn=None):
    def _inner(mol):
        try:
            return float(fn(mol) if fn else getattr(rdMolDescriptors, attr)(mol))
        except Exception:
            return np.nan
    return _inner


EXACT_CALCULATORS: Dict[str, Any] = {
    "mw": _m_desc("MolWt"), "molwt": _m_desc("MolWt"), "molecular_weight": _m_desc("MolWt"),
    "mol_weight": _m_desc("MolWt"), "molecularweight": _m_desc("MolWt"),
    "exact_mw": _m_desc("ExactMolWt"), "exact_mol_wt": _m_desc("ExactMolWt"),
    "heavy": lambda m: float(m.GetNumHeavyAtoms()),
    "heavyatom": lambda m: float(m.GetNumHeavyAtoms()),
    "heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "heavyatomcount": lambda m: float(m.GetNumHeavyAtoms()),
    "num_heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "hac": lambda m: float(m.GetNumHeavyAtoms()),
    "num_atoms": lambda m: float(m.GetNumAtoms()),
    "num_bonds": lambda m: float(m.GetNumBonds()),
    "logp": _m_desc("MolLogP"), "clogp": _m_desc("MolLogP"), "xlogp": _m_desc("MolLogP"),
    "tpsa": _m_desc("TPSA"), "polar_surface_area": _m_desc("TPSA"),
    "mr": _m_desc("MolMR"), "molar_refractivity": _m_desc("MolMR"),
    "hbd": _rd("CalcNumHBD"), "num_hbd": _rd("CalcNumHBD"), "h_bond_donors": _rd("CalcNumHBD"),
    "hba": _rd("CalcNumHBA"), "num_hba": _rd("CalcNumHBA"), "h_bond_acceptors": _rd("CalcNumHBA"),
    "rotb": _m_desc("NumRotatableBonds"), "rotatable_bonds": _m_desc("NumRotatableBonds"),
    "num_rotatable_bonds": _m_desc("NumRotatableBonds"),
    "rings": _rd("CalcNumRings"), "num_rings": _rd("CalcNumRings"), "ring_count": _rd("CalcNumRings"),
    "arrings": _rd("CalcNumAromaticRings"), "aromatic_rings": _rd("CalcNumAromaticRings"),
    "num_aromatic_rings": _rd("CalcNumAromaticRings"),
    "alirings": _rd("CalcNumAliphaticRings"), "aliphatic_rings": _rd("CalcNumAliphaticRings"),
    "fsp3": _rd("CalcFractionCSP3"), "fraction_csp3": _rd("CalcFractionCSP3"),
    "num_heteroatoms": _rd("CalcNumHeteroatoms"), "heteroatoms": _rd("CalcNumHeteroatoms"),
    "num_stereocenters": _rd("CalcNumAtomStereoCenters"),
    "num_saturated_rings": _rd("CalcNumSaturatedRings"),
    "num_aromatic_carbocycles": _rd("CalcNumAromaticCarbocycles"),
    "num_aromatic_heterocycles": _rd("CalcNumAromaticHeterocycles"),
    "num_aliphatic_carbocycles": _rd("CalcNumAliphaticCarbocycles"),
    "num_aliphatic_heterocycles": _rd("CalcNumAliphaticHeterocycles"),
    "num_amide_bonds": _rd("CalcNumAmideBonds"),
}

ELEMENT_CALCULATORS: Dict[str, str] = {
    "n": "N", "nitrogen": "N", "n_count": "N", "num_n": "N", "num_nitrogen": "N",
    "o": "O", "oxygen": "O", "o_count": "O", "num_o": "O", "num_oxygen": "O",
    "s": "S", "sulfur": "S", "s_count": "S", "num_s": "S", "num_sulfur": "S",
    "c": "C", "carbon": "C", "c_count": "C", "num_c": "C",
    "f": "F", "cl": "Cl", "br": "Br", "i": "I", "p": "P", "si": "Si", "b": "B",
    "hal": "HAL", "halogen": "HAL", "halogens": "HAL", "num_halogen": "HAL",
}


def _calc_element(mol, symbol: str) -> float:
    if symbol == "HAL":
        return float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")))
    return float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol))


def _count_active_hydrogen(mol) -> int:
    """活泼氢数（可与环氧反应的 N-H / O-H / S-H 总数），排除酰胺/磺酰胺氮。"""
    total = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ("N", "O", "S"):
            continue
        n_h = atom.GetTotalNumHs()
        if n_h <= 0:
            continue
        if symbol == "N":
            unreactive = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "S":
                    unreactive = True
                    break
                if neighbor.GetSymbol() == "C":
                    for bond in neighbor.GetBonds():
                        other = bond.GetOtherAtom(neighbor)
                        if other.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2:
                            unreactive = True
                            break
                if unreactive:
                    break
            if unreactive:
                continue
        total += n_h
    return total


# ---------------------------------------------------------------------------
# 轻量路径：单分子特征直接计算（不启动重后端）
# ---------------------------------------------------------------------------
def compute_single_molecule_feature(smiles: Any, feature: str) -> float:
    """从 SMILES 计算单个分子级特征。算不出返回 NaN。"""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles.strip():
        return np.nan
    text = clean_smiles(smiles)
    if not text:
        return np.nan
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return np.nan
    mol = _largest_fragment(mol)

    key = normalize_name(feature)

    if key in ELEMENT_CALCULATORS:
        return _calc_element(mol, ELEMENT_CALCULATORS[key])
    if key in EXACT_CALCULATORS:
        return EXACT_CALCULATORS[key](mol)

    stripped = re.sub(
        r"^(resin|curing_agent|hardener|small_additive|additive|initiator|accelerator|catalyst|"
        r"reactive_diluent|reactive_toughener|filler|other_component|component|mol)_?\d*_?",
        "", key,
    ).strip("_")
    if stripped != key and stripped in EXACT_CALCULATORS:
        return EXACT_CALCULATORS[stripped](mol)
    if stripped != key and stripped in ELEMENT_CALCULATORS:
        return _calc_element(mol, ELEMENT_CALCULATORS[stripped])

    tokens = set(key.split("_")) | set(stripped.split("_"))
    best = None
    for token, smarts in SMARTS_LIBRARY.items():
        if token in tokens or key.endswith("_" + token) or key == token:
            if best is None or len(token) > len(best[0]):
                best = (token, smarts)
    if best is not None:
        return float(_count_smarts(mol, best[1]))

    for alias, target in (
        (("heavy", "hac", "heavyatom"), "hac"),
        (("molecular_weight", "molwt", "mol_weight", "molar_mass"), "mw"),
        (("active_hydrogen", "ahew", "amine_hydrogen"), "active_hydrogen"),
        (("epoxy", "epoxide", "oxirane", "glycidyl"), "epoxide"),
        (("carboxyl", "acid"), "carboxylic_acid"),
    ):
        if any(a in key or a in stripped for a in alias):
            if target == "active_hydrogen":
                return float(_count_active_hydrogen(mol))
            if target in EXACT_CALCULATORS:
                return EXACT_CALCULATORS[target](mol)
            if target in SMARTS_LIBRARY:
                return float(_count_smarts(mol, SMARTS_LIBRARY[target]))
    return np.nan


def compute_formulation_feature(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """配方级特征：EEW / AHEW / 官能度 / 化学计量比。需要树脂与固化剂结构列。"""
    if not RDKIT_AVAILABLE:
        return None
    resin_col = _find_column(df, ("resin_1_structure", "resin_1_smiles", "resin_smiles", "resin_1_bigsmiles"))
    curer_col = _find_column(df, ("curing_agent_1_structure", "curing_agent_1_smiles", "curing_agent_smiles"))
    if resin_col is None and curer_col is None:
        return None

    key = normalize_name(feature)
    cache: Dict[str, Tuple[float, float, float, float]] = {}

    def props(smiles: Any) -> Tuple[float, float, float, float]:
        text = clean_smiles(smiles) if isinstance(smiles, str) else None
        if text in cache:
            return cache[text]
        if not text:
            out = (np.nan, np.nan, np.nan, np.nan)
        else:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                out = (np.nan, np.nan, np.nan, np.nan)
            else:
                mol = _largest_fragment(mol)
                out = (
                    float(Descriptors.MolWt(mol)),
                    float(_count_smarts(mol, "C1OC1")),
                    float(_count_smarts(mol, "C(=O)OC(=O)")),
                    float(_count_active_hydrogen(mol)),
                )
        cache[text] = out
        return out

    resin = df[resin_col] if resin_col else pd.Series([None] * len(df), index=df.index)
    curer = df[curer_col] if curer_col else pd.Series([None] * len(df), index=df.index)

    if key in ("eew", "epoxy_equivalent_weight", "resin_eew", "formulation_resin_total_eew_g_eq"):
        return pd.Series([(lambda t: t[0] / t[1] if t[1] > 0 else np.nan)(props(s)) for s in resin], index=df.index)

    if key in ("ahew", "amine_hydrogen_equivalent_weight", "resin_ahew", "formulation_hardener_total_ahew_g_eq"):
        def _ahew(s):
            mw, _, n_anh, n_ah = props(s)
            f = n_ah if n_ah > 0 else (2 * n_anh if n_anh > 0 else np.nan)
            return mw / f if f and f > 0 else np.nan
        return pd.Series([_ahew(s) for s in curer], index=df.index)

    if key in ("resin_functionality", "epoxy_functionality", "resin_epoxy_group_count", "resin_epoxy_group_total"):
        return pd.Series([props(s)[1] for s in resin], index=df.index)

    if key in ("hardener_functionality", "curer_functionality", "curing_agent_active_hydrogen_total",
               "curing_agent_active_hydrogen_equivalent_count"):
        def _f(s):
            _, _, n_anh, n_ah = props(s)
            return n_ah if n_ah > 0 else (2 * n_anh if n_anh > 0 else np.nan)
        return pd.Series([_f(s) for s in curer], index=df.index)

    if key in ("formulation_r_value", "formulation_resin_hardener_equivalent_ratio", "stoichiometric_ratio_r"):
        def _r(sr, sc):
            mwr, nepr, _, _ = props(sr)
            mwc, _, nanh, nah = props(sc)
            fc = nah if nah > 0 else (2 * nanh if nanh > 0 else np.nan)
            if nepr > 0 and fc and fc > 0:
                return (mwr / nepr) / (mwc / fc)
            return np.nan
        return pd.Series([_r(a, b) for a, b in zip(resin, curer)], index=df.index)

    return None


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    columns = list(df.columns)
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        hit = lower.get(cand.strip().lower())
        if hit:
            return hit
    norm = {normalize_name(c): c for c in columns}
    for cand in candidates:
        hit = norm.get(normalize_name(cand))
        if hit:
            return hit
    return None


def detect_structure_columns(df: pd.DataFrame) -> List[str]:
    """自动识别结构列（列名含 structure/smiles/bigsmiles，或内容像 SMILES）。"""
    cols: List[str] = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ("structure", "smiles", "bigsmiles")):
            cols.append(c)
    if cols:
        return cols
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(30)
        if len(sample) == 0:
            continue
        looks = sum(1 for v in sample if re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.*{}<>]{2,}", v or ""))
        if looks >= max(3, int(len(sample) * 0.6)):
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# 提取引擎：复用虚拟筛选的 extract_features_from_config
# ---------------------------------------------------------------------------
class ExtractionBackend:
    """按方法尝试提取特征；优先轻量路径，必要时调用虚拟筛选的重后端。"""

    def __init__(
        self,
        methods: Optional[Sequence[str]] = None,
        *,
        device: Any = None,
        max_rows: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        参数:
            methods:  显式指定方法 key 列表（见 EXTRACTION_METHODS）；None 用 DEFAULT_METHOD_ORDER
            device:   torch 设备（Transformer/ANI/GNN 用）
            max_rows: 重后端提取的行数上限（控制耗时）
            verbose:  打印细节
        """
        self.method_keys = list(methods) if methods else list(DEFAULT_METHOD_ORDER)
        self.device = device
        self.max_rows = max_rows
        self.verbose = verbose
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._method_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    def method_spec(self, key: str) -> Optional[Dict[str, Any]]:
        for spec in EXTRACTION_METHODS:
            if spec["key"] == key:
                return spec
        return None

    def extract(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        *,
        hardener_col: Optional[str] = None,
        method_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """用指定结构列提取特征矩阵（缓存）。

        参数:
            method_key: 指定单个方法；None 则按顺序尝试直到有输出
        返回可能为空 DataFrame。
        """
        cache_key = (smiles_col, hardener_col or "", method_key or "")
        if cache_key in self._cache:
            return self._cache[cache_key]

        values = df[smiles_col]
        if self.max_rows is not None and len(values) > self.max_rows:
            values = values.head(self.max_rows)
        smiles_list = [clean_smiles(v) if isinstance(v, str) else None for v in values]
        hardener_list = None
        if hardener_col and hardener_col in df.columns:
            hv = df[hardener_col]
            if self.max_rows is not None and len(hv) > self.max_rows:
                hv = hv.head(self.max_rows)
            hardener_list = [clean_smiles(v) if isinstance(v, str) else None for v in hv]

        keys = [method_key] if method_key else list(self.method_keys)
        for key in keys:
            spec = self.method_spec(key)
            if spec is None:
                continue
            frame = self._try_method(spec, smiles_list, hardener_list)
            if frame is not None and not frame.empty:
                frame = frame.reindex(df.index)
                self._cache[cache_key] = frame
                return frame
        self._cache[cache_key] = pd.DataFrame(index=df.index)
        return pd.DataFrame(index=df.index)

    def _try_method(
        self, spec: Dict[str, Any], smiles_list: List[Optional[str]], hardener_list: Optional[List[Optional[str]]]
    ) -> Optional[pd.DataFrame]:
        try:
            from .virtual_screening import extract_features_from_config
        except Exception:
            return None

        cfg = {
            "method": spec["method"],
            "params": dict(spec.get("params") or {}),
            "prefix": "",
        }
        try:
            frame, error = extract_features_from_config(
                smiles_list, hardener_list, cfg, device=self.device
            )
        except Exception as exc:
            if self.verbose:
                print(f"[提取] {spec['label']} 失败: {exc}")
            return None
        if error or frame is None or frame.empty:
            if self.verbose and error:
                print(f"[提取] {spec['label']} 无结果: {error}")
            return None
        return frame

    def find_feature(self, df: pd.DataFrame, smiles_col: str, feature: str, *,
                     hardener_col: Optional[str] = None) -> Optional[pd.Series]:
        """在所有配置的方法里找目标特征列（名称归一化后匹配）。

        逐个方法尝试，直到找到能产出该特征的提取器——不能只搜第一个成功的方法，
        否则 Mordred/指纹等方法的独有特征会永远找不到。
        """
        values = df[smiles_col]
        if self.max_rows is not None and len(values) > self.max_rows:
            values = values.head(self.max_rows)
        smiles_list = [clean_smiles(v) if isinstance(v, str) else None for v in values]
        hardener_list = None
        if hardener_col and hardener_col in df.columns:
            hv = df[hardener_col]
            if self.max_rows is not None and len(hv) > self.max_rows:
                hv = hv.head(self.max_rows)
            hardener_list = [clean_smiles(v) if isinstance(v, str) else None for v in hv]

        for key in self.method_keys:
            spec = self.method_spec(key)
            if spec is None:
                continue
            cache_key = (smiles_col, hardener_col or "", key)
            if cache_key in self._method_cache:
                frame = self._method_cache[cache_key]
            else:
                frame = self._try_method(spec, smiles_list, hardener_list)
                if frame is not None and not frame.empty:
                    frame = frame.reindex(df.index)
                else:
                    frame = pd.DataFrame(index=df.index)
                self._method_cache[cache_key] = frame
            if frame.empty:
                continue
            series = self._match_column(frame, feature)
            if series is not None and series.notna().any():
                return series
        return None

    @staticmethod
    def _match_column(frame: pd.DataFrame, feature: str) -> Optional[pd.Series]:
        """在特征矩阵里按名称匹配目标列。

        匹配顺序（从严到宽）：
            1. 归一化完全相等
            2. 跨库别名表（nAromRings ↔ nARing、hbd ↔ nHBDon …）
            3. 后缀/前缀包含
            4. 关键词集合相等（分词后集合相同）
            5. 宽松子串（仅在无其他候选时）
        """
        columns = list(frame.columns)
        target = normalize_name(feature)
        target_tokens = _token_set(feature)

        # 1) 归一化完全相等
        for col in columns:
            if normalize_name(col) == target:
                return frame[col]

        # 2) 跨库别名表（键值已归一化）
        t_key = _norm_key(feature)
        for col in columns:
            c_key = _norm_key(col)
            canonical = _ALIAS_LOOKUP.get(t_key)
            if canonical and c_key == canonical:
                return frame[col]
            if t_key in _ALIAS_REVERSE.get(c_key, ()):
                return frame[col]
            # feature 本身是标准名，列名是它的别名
            for canon, aliases in _ALIAS_REVERSE.items():
                if t_key == canon and c_key in aliases:
                    return frame[col]

        # 3) 后缀包含
        for col in columns:
            col_key = normalize_name(col)
            if col_key.endswith("_" + target) or target.endswith("_" + col_key):
                return frame[col]

        # 4) 分词集合相等（忽略语序，如 num_aromatic_rings ↔ aromatic_ring_num）
        if target_tokens:
            for col in columns:
                if _token_set(col) == target_tokens:
                    return frame[col]

        # 5) 宽松子串（要求足够长且词边界合理，避免误命中）
        #    陷阱：nHBDon 会被截断命中 Mordred 的 nH（氢原子总数），必须拦住
        if len(target) >= 6 and not any(g in target for g in _SUBSTRING_GUARD):
            for col in columns:
                col_key = normalize_name(col)
                if target in col_key or col_key in target:
                    return frame[col]
        return None

    @staticmethod
    def explain_match(frame: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """解释匹配结果（调试/UI 用）：返回命中的列名与策略。"""
        if frame is None or frame.empty:
            return {"matched": None, "strategy": "empty_frame"}
        columns = list(frame.columns)
        target = normalize_name(feature)
        target_tokens = _token_set(feature)
        t_key = _norm_key(feature)

        for col in columns:
            if normalize_name(col) == target:
                return {"matched": col, "strategy": "normalized_equal"}
        for col in columns:
            c_key = _norm_key(col)
            canonical = _ALIAS_LOOKUP.get(t_key)
            if canonical and c_key == canonical:
                return {"matched": col, "strategy": f"alias({t_key}->{canonical})"}
            if t_key in _ALIAS_REVERSE.get(c_key, ()):
                return {"matched": col, "strategy": f"alias_reverse({c_key}<-{t_key})"}
            for canon, aliases in _ALIAS_REVERSE.items():
                if t_key == canon and c_key in aliases:
                    return {"matched": col, "strategy": f"alias_reverse({canon}<-{c_key})"}
        for col in columns:
            col_key = normalize_name(col)
            if col_key.endswith("_" + target) or target.endswith("_" + col_key):
                return {"matched": col, "strategy": "suffix"}
        if target_tokens:
            for col in columns:
                if _token_set(col) == target_tokens:
                    return {"matched": col, "strategy": "token_set"}
        if len(target) >= 6 and not any(g in target for g in _SUBSTRING_GUARD):
            for col in columns:
                col_key = normalize_name(col)
                if target in col_key or col_key in target:
                    return {"matched": col, "strategy": "substring"}
        return {"matched": None, "strategy": "no_match", "target_normalized": target}

    @staticmethod
    def list_available(frame: pd.DataFrame, keyword: str = "") -> List[str]:
        """列出特征矩阵里包含关键词的列（调试/UI 用）。"""
        if frame is None or frame.empty:
            return []
        if not keyword:
            return list(frame.columns)
        key = normalize_name(keyword)
        return [c for c in frame.columns if key in normalize_name(c)]

    def available_features(self, df: pd.DataFrame, smiles_col: str, *, hardener_col: Optional[str] = None) -> List[str]:
        """汇总所有方法能提供的特征列（取并集）。"""
        names: List[str] = []
        for key in self.method_keys:
            spec = self.method_spec(key)
            if spec is None:
                continue
            try:
                frame = self.extract(df, smiles_col, hardener_col=hardener_col, method_key=key)
            except Exception:
                continue
            if frame.empty:
                continue
            for col in frame.columns:
                if col not in names:
                    names.append(col)
        return names


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class AutoFeatureResolver:
    """自动为外部模型补齐所需特征：现场提取 → 总表查询 → 兜底。"""

    def __init__(
        self,
        *,
        master_tables: Optional[Sequence[str | pd.DataFrame]] = None,
        structure_columns: Optional[Sequence[str]] = None,
        extraction_methods: Optional[Sequence[str]] = None,
        device: Any = None,
        max_rows_for_extraction: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        参数:
            master_tables:          总表（文件路径或 DataFrame），含结构列 + 已算好的特征列
            structure_columns:      指定工作区里的结构列；None 则自动识别
            extraction_methods:     提取方法 key 列表（见 EXTRACTION_METHODS）
            device:                 torch 设备
            max_rows_for_extraction: 重后端提取的行数上限
            verbose:                打印细节
        """
        self.master_tables: List[Tuple[str, pd.DataFrame]] = []
        for item in (master_tables or []):
            if isinstance(item, pd.DataFrame):
                self.master_tables.append(("<DataFrame>", item))
            elif isinstance(item, (str, os.PathLike)):
                path = str(item)
                if os.path.exists(path):
                    try:
                        self.master_tables.append((os.path.basename(path), pd.read_csv(path, low_memory=False)))
                    except Exception:
                        pass
        self.structure_columns = list(structure_columns) if structure_columns else None
        self.backend = ExtractionBackend(
            extraction_methods, device=device, max_rows=max_rows_for_extraction, verbose=verbose
        )
        self.verbose = verbose
        self._master_index: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

    # -- 总表索引 -----------------------------------------------------------
    def _build_master_index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if self._master_index is not None:
            return self._master_index
        index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for _table_name, frame in self.master_tables:
            struct_cols = detect_structure_columns(frame)
            if not struct_cols:
                continue
            feature_cols = [c for c in frame.columns if c not in struct_cols]
            for struct_col in struct_cols:
                for row_idx, raw in enumerate(frame[struct_col].tolist()):
                    key = self._structure_key(raw)
                    if not key:
                        continue
                    bucket = index.setdefault(key, {})
                    for fc in feature_cols:
                        fname = normalize_name(fc)
                        if fname in bucket:
                            continue
                        value = frame.at[row_idx, fc] if row_idx in frame.index else None
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            continue
                        bucket[fname] = value
        self._master_index = index
        return index

    @staticmethod
    def _structure_key(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        text = clean_smiles(value) or value.strip()
        return re.sub(r"\s+", "", text).lower()

    def lookup_master(self, smiles: Any, feature: str) -> Any:
        key = self._structure_key(smiles)
        if not key:
            return np.nan
        bucket = self._build_master_index().get(key)
        if not bucket:
            return np.nan
        return bucket.get(normalize_name(feature), np.nan)

    # -- 结构列排序 ---------------------------------------------------------
    @staticmethod
    def rank_structure_columns(feature: str, struct_cols: Sequence[str], df: pd.DataFrame) -> List[str]:
        """按与特征名的前缀契合度给结构列排序（最契合的在前）。"""
        key = normalize_name(feature)
        scored: List[Tuple[int, int, str]] = []
        for idx, col in enumerate(struct_cols):
            col_key = normalize_name(col)
            prefix = re.sub(r"_(structure|smiles|bigsmiles)$", "", col_key)
            score = 0
            if prefix and (key == prefix or key.startswith(prefix + "_")):
                score = 1000
            elif prefix:
                p_tokens = set(prefix.split("_"))
                k_tokens = set(key.split("_"))
                if p_tokens & k_tokens:
                    score = 100 * len(p_tokens & k_tokens)
            if score == 0:
                if "resin" in key and "resin" in col_key:
                    score = 10
                elif ("curing" in key or "hardener" in key) and ("curing" in col_key or "hardener" in col_key):
                    score = 10
            scored.append((-score, idx, col))
        scored.sort()
        return [col for _neg, _idx, col in scored]

    # -- 主流程 -------------------------------------------------------------
    def resolve(
        self,
        df: pd.DataFrame,
        required_features: Sequence[str],
        *,
        already_resolved: Optional[Dict[str, str]] = None,
        max_rows_for_compute: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """为 required_features 补齐列。返回 (df_with_new_columns, report)。"""
        result = df.copy()
        already = {str(k): str(v) for k, v in (already_resolved or {}).items()}
        struct_cols = self.structure_columns or detect_structure_columns(result)

        primary_struct = None
        for cand in ("resin_1_structure", "resin_1_smiles", "resin_smiles"):
            hit = _find_column(result, (cand,))
            if hit:
                primary_struct = hit
                break
        if primary_struct is None and struct_cols:
            primary_struct = struct_cols[0]

        report: Dict[str, Any] = {
            "computed": {}, "from_extractor": {}, "from_master": {}, "unresolved": [],
            "columns_added": [], "structure_column": primary_struct,
            "n_master_tables": len(self.master_tables),
        }

        for feature in required_features:
            if feature in already or feature in result.columns:
                continue

            # --- A) 配方级组合量 ---
            series = None
            try:
                series = compute_formulation_feature(result, feature)
            except Exception:
                series = None
            if series is not None and series.notna().any():
                result[feature] = series
                report["computed"][feature] = "配方级计算（树脂+固化剂）"
                report["columns_added"].append(feature)
                continue

            # --- B) 提取引擎（多方法，含重后端）---
            extracted = self._extract_feature(result, feature, struct_cols)
            if extracted is not None and extracted.notna().any():
                result[feature] = extracted
                report["from_extractor"][feature] = "提取引擎"
                report["columns_added"].append(feature)
                continue

            # --- C) 总表查询 ---
            if self.master_tables and primary_struct is not None:
                from_master = pd.Series(
                    [self.lookup_master(v, feature) for v in result[primary_struct]], index=result.index
                )
                if from_master.notna().any():
                    result[feature] = from_master
                    report["from_master"][feature] = "总表结构匹配"
                    report["columns_added"].append(feature)
                    continue

            report["unresolved"].append(feature)

        return result, report

    def _extract_feature(
        self, df: pd.DataFrame, feature: str, struct_cols: Sequence[str], n_limit: Optional[int] = None
    ) -> Optional[pd.Series]:
        """先试轻量单分子计算，再试提取引擎（含重后端）。"""
        ordered = self.rank_structure_columns(feature, struct_cols, df)
        limit = n_limit or self.backend.max_rows

        # 轻量路径
        for col in ordered:
            if col not in df.columns:
                continue
            values = df[col].head(limit) if limit else df[col]
            try:
                computed = pd.Series(
                    [compute_single_molecule_feature(v, feature) for v in values], index=values.index
                )
            except Exception:
                continue
            if computed.notna().any():
                return computed.reindex(df.index)

        # 重后端路径：按前缀选最合适的主结构列
        hardener = _find_column(df, ("curing_agent_1_structure", "curing_agent_1_smiles", "curing_agent_smiles"))
        for col in ordered:
            if col not in df.columns:
                continue
            try:
                series = self.backend.find_feature(df, col, feature, hardener_col=hardener)
            except Exception:
                series = None
            if series is not None and series.notna().any():
                return series.reindex(df.index)
        return None

    # -- 诊断 ---------------------------------------------------------------
    def diagnose(self, df: pd.DataFrame, required_features: Sequence[str]) -> pd.DataFrame:
        """不实际写入，先看每个特征能从哪来（供 UI 预览）。"""
        struct_cols = self.structure_columns or detect_structure_columns(df)
        rows: List[Dict[str, Any]] = []
        for feature in required_features:
            if feature in df.columns:
                rows.append({"feature": feature, "source": "工作区已有列", "detail": feature, "available": True})
                continue
            series = None
            try:
                series = compute_formulation_feature(df, feature)
            except Exception:
                series = None
            if series is not None and series.notna().any():
                rows.append({"feature": feature, "source": "配方级计算",
                             "detail": f"覆盖率 {series.notna().mean()*100:.0f}%", "available": True})
                continue

            probe_df = df.head(min(200, len(df)))
            ordered = self.rank_structure_columns(feature, struct_cols, df)
            hit = False
            for col in ordered:
                if col not in probe_df.columns:
                    continue
                try:
                    probe = pd.Series(
                        [compute_single_molecule_feature(v, feature) for v in probe_df[col]], index=probe_df.index
                    )
                except Exception:
                    continue
                if probe.notna().any():
                    rows.append({"feature": feature, "source": "RDKit 直接计算",
                                 "detail": f"用 {col}（抽样 {probe.notna().sum()}/{len(probe)} 成功）",
                                 "available": True})
                    hit = True
                    break
            if hit:
                continue

            # 重后端探测
            for col in ordered:
                if col not in probe_df.columns:
                    continue
                try:
                    probe = self.backend.find_feature(probe_df, col, feature)
                except Exception:
                    probe = None
                if probe is not None and probe.notna().any():
                    rows.append({"feature": feature, "source": "提取引擎",
                                 "detail": f"用 {col}（{probe.notna().sum()}/{len(probe)} 成功）", "available": True})
                    hit = True
                    break
            if hit:
                continue

            if self.master_tables and struct_cols:
                first = df[struct_cols[0]].iloc[0] if len(df) else None
                value = self.lookup_master(first, feature)
                if not (isinstance(value, float) and np.isnan(value)):
                    rows.append({"feature": feature, "source": "总表查询", "detail": "结构命中", "available": True})
                    continue

            rows.append({"feature": feature, "source": "❌ 无法获取", "detail": "需手工映射", "available": False})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def auto_resolve_features(
    df: pd.DataFrame,
    required_features: Sequence[str],
    *,
    master_tables: Optional[Sequence[str | pd.DataFrame]] = None,
    already_resolved: Optional[Dict[str, str]] = None,
    extraction_methods: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """一行调用：自动提取 + 总表查询，补齐外部模型所需特征。"""
    resolver = AutoFeatureResolver(master_tables=master_tables, extraction_methods=extraction_methods)
    return resolver.resolve(df, required_features, already_resolved=already_resolved)


def list_extraction_methods() -> List[Dict[str, Any]]:
    """列出可用提取方法（供 UI 选择）。"""
    return [
        {"key": s["key"], "label": s["label"], "cost": s["cost"], "desc": s["desc"]}
        for s in EXTRACTION_METHODS
    ]
