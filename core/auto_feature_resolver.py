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


#: 空值占位符（总表里常见的“未测/无”）
_BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown", "未测", "无", "?"}


def _is_blank(value: Any) -> bool:
    """判定是否为缺失值（NaN / None / 空串 / 常见占位符）。热路径，尽量快。"""
    if value is None:
        return True
    if type(value) is float:
        return value != value  # NaN 自比较
    if isinstance(value, str):
        return value.strip().lower() in _BLANK_TOKENS
    if isinstance(value, float):
        return bool(np.isnan(value))
    if isinstance(value, (int, np.integer)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _collapse_values(values: Sequence[Any]) -> Any:
    """把同结构下的多个观测值合并为一个代表值。

    规则（针对总表里“同一结构多行”的现实）：
        - 全部相等（含浮点容差）→ 直接返回
        - 数值型 → 取中位数（比众数稳健，对连续量友好）
        - 分类型 → 取众数（出现最多者；并列时取首个）
    这样比旧实现的 first-wins（随机取一行）可靠得多。
    """
    cleaned = [v for v in values if not _is_blank(v)]
    if not cleaned:
        return np.nan
    if len(cleaned) == 1:
        return cleaned[0]

    numeric: List[float] = []
    all_numeric = True
    for v in cleaned:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            all_numeric = False
            break
    if all_numeric:
        arr = np.asarray(numeric, dtype=float)
        if np.allclose(arr, arr[0], rtol=1e-9, atol=1e-12):
            return float(arr[0])
        return float(np.median(arr))

    # 分类型：众数
    counts: Dict[Any, int] = {}
    for v in cleaned:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


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


#: 非分子来源的特征模式：这些量不可能从 SMILES 算出，必须来自工作区数据/总表。
_NON_MOLECULAR_PATTERNS = (
    r"(^|_)amount_phr$", r"(^|_)phr$", r"(^|_)parts_per_hundred",
    r"^process_", r"_process_", r"(^|_)temperature(_c)?$", r"(^|_)time(_h|_min|_s)?$",
    r"(^|_)atmosphere$", r"(^|_)pressure", r"(^|_)post_cure", r"(^|_)cure_schedule",
    r"(^|_)heating_rate", r"(^|_)cooling_rate", r"(^|_)ramp", r"(^|_)dwell",
    r"^test_", r"_test_", r"(^|_)standard(_canonical|_organization|_count)?$",
    r"(^|_)method$", r"(^|_)instrument$", r"(^|_)lab$", r"(^|_)source$",
    r"(^|_)component_count$", r"(^|_)present$", r"(^|_)has_",
    r"(^|_)stoich", r"(^|_)r_value$", r"(^|_)ratio$",
    r"(^|_)total_phr$", r"(^|_)equivalent_group_total$",
    r"_id$", r"(^|_)paper", r"(^|_)doi", r"(^|_)year$", r"(^|_)author",
    r"(^|_)yield$", r"(^|_)conversion(_pct|_percent)?$",
)
_NON_MOLECULAR_RE = re.compile("|".join(_NON_MOLECULAR_PATTERNS))


def looks_molecular(feature: str) -> bool:
    """判断特征名是否可能由分子结构计算得到。

    用途：过滤掉工艺/测试/配方计数类列，不对它们启动分子提取后端。
    保守策略：只拦明显非分子的（命中 _NON_MOLECULAR_RE 且不含结构/描述符关键词）。
    """
    key = normalize_name(feature)
    if not key:
        return False
    # 先拦配方级聚合/工艺类：这些含 phr / component_count / equivalent_*_total
    # 等词，虽与化学沾边，但必须从配方表取，不能从单个分子算。
    aggregate_patterns = (
        r"(^|_)component_count$", r"(^|_)total_phr$", r"(^|_)amount_phr$",
        r"(^|_)equivalent_group_total$", r"(^|_)equivalent_group_count$",
        r"(^|_)equivalent_weight_g_eq$", r"(^|_)stoich", r"(^|_)r_value$",
        r"(^|_)total_phr", r"(^|_)sum_phr",
    )
    if any(re.search(p, key) for p in aggregate_patterns):
        return False
    # 明确含分子语义的，直接放行
    molecular_hints = (
        "structure", "smiles", "bigsmiles", "selfies", "maccs", "morgan", "fp_",
        "molecular_weight", "molwt", "mw", "logp", "tpsa", "hbd", "hba", "ring",
        "heavy", "hac", "rotb", "rotatable", "atom", "bond", "element", "count_c",
        "epoxy", "epoxide", "oxirane", "amine", "hydroxyl", "carboxyl", "ester",
        "amide", "ether", "aromatic", "aliphatic", "fragment", "xtb", "ff_", "energy",
        "homo", "lumo", "gap", "dipole", "charge", "polar", "refractivity", "tpsa",
        "active_hydrogen", "eew", "ahew", "functionality",
    )
    if any(h in key for h in molecular_hints):
        return True
    if _NON_MOLECULAR_RE.search(key):
        return False
    # 其余保守放行（宁可多试一次，也不要漏算真特征）
    return True


# ---------------------------------------------------------------------------
# 轻量路径：单分子特征直接计算（不启动重后端）
# ---------------------------------------------------------------------------
_MOL_CACHE: Dict[str, Any] = {}


def _parse_molecule_cached(smiles: Any) -> Any:
    """带缓存的分子解析（含最大片段提取），避免同一 SMILES 重复解析。"""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles.strip():
        return None
    text = clean_smiles(smiles)
    if not text:
        return None
    if text in _MOL_CACHE:
        return _MOL_CACHE[text]
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        mol = _largest_fragment(mol)
    if len(_MOL_CACHE) < 20000:
        _MOL_CACHE[text] = mol
    return mol


def _compute_feature_from_mol(mol: Any, feature: str) -> Optional[float]:
    """从已解析的分子算单个特征（与 compute_single_molecule_feature 同逻辑）。

    拆出来是为了批量路径：同一分子只解析一次，多个特征共用一个 mol 对象。
    """
    if mol is None:
        return None
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
    return None


def _feature_belongs_to_column(feature: str, col: str, df: pd.DataFrame) -> bool:
    """判断特征名是否应由该结构列提供（按组分前缀契合度）。

    关键：固化剂特征不能用树脂结构算。例如 curing_agent_1_molecular_weight_g_mol
    必须用 curing_agent_1_structure，不能用 resin_1_structure。

    同时允许同角色回退：resin_3_epoxy_group_count 可由 resin_1_structure 算
    （当工作区没有 resin_3_structure 或该行为空时）。
    """
    fkey = normalize_name(feature)
    ckey = normalize_name(col)

    # 精确前缀：curing_agent_1_structure → curing_agent_1
    prefix = re.sub(r"_(structure|smiles|bigsmiles|selfies|inchi|iupac)$", "", ckey)
    if fkey.startswith(prefix + "_"):
        return True

    # 同角色回退：只归给该角色的第一个结构列（避免重复计算）
    f_role = _ROLE_RE.match(fkey)
    c_role = _ROLE_RE.match(ckey)
    if f_role and c_role and f_role.group(1) == c_role.group(1):
        role = f_role.group(1)
        same_role_cols = _same_role_structure_cols(df, role)
        return bool(same_role_cols) and col == same_role_cols[0]

    # 无前缀的通用分子特征（mw/logp/epoxy…）：归给第一个树脂结构列
    if not f_role:
        resin_cols = _same_role_structure_cols(df, "resin")
        if resin_cols:
            return col == resin_cols[0]
    return False


#: 组分角色前缀（用于归属判定，模块级预编译）
_ROLE_RE = re.compile(
    r"^(resin|curing_agent|hardener|small_additive|additive|initiator|accelerator|"
    r"catalyst|reactive_diluent|reactive_toughener|filler|other_component)"
)
_STRUCT_PREFIX_RE = re.compile(r"_(structure|smiles|bigsmiles|selfies|inchi)$")
_STRUCT_SUFFIX_RE = re.compile(r"_(structure|smiles|bigsmiles|selfies)$")
#: 每列是否结构列、其角色、归一化名 —— 缓存（按列名 + 角色）
_COL_META_CACHE: Dict[Tuple[str, str], List[str]] = {}


def _same_role_structure_cols(df: pd.DataFrame, role: str) -> List[str]:
    """取 df 里属于该角色的结构列（结果缓存，避免每次重算 normalize_name）。"""
    cache_key = ("||".join(map(str, df.columns)), role)
    cached = _COL_META_CACHE.get(cache_key)
    if cached is not None:
        return cached
    out = [
        c for c in df.columns
        if normalize_name(c).startswith(role) and _STRUCT_SUFFIX_RE.search(normalize_name(c))
    ]
    if len(_COL_META_CACHE) > 256:
        _COL_META_CACHE.clear()
    _COL_META_CACHE[cache_key] = out
    return out


def compute_single_molecule_feature(smiles: Any, feature: str) -> float:
    """从 SMILES 计算单个分子级特征。算不出返回 NaN。"""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles.strip():
        return np.nan
    mol = _parse_molecule_cached(smiles)
    if mol is None:
        return np.nan
    value = _compute_feature_from_mol(mol, feature)
    return np.nan if value is None else float(value)


_FORMULATION_PROP_CACHE: Dict[str, Tuple[float, float, float, float]] = {}


def compute_formulation_feature(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """配方级特征：EEW / AHEW / 官能度 / 化学计量比。需要树脂与固化剂结构列。"""
    if not RDKIT_AVAILABLE:
        return None
    resin_col = _find_column(df, ("resin_1_structure", "resin_1_smiles", "resin_smiles", "resin_1_bigsmiles"))
    curer_col = _find_column(df, ("curing_agent_1_structure", "curing_agent_1_smiles", "curing_agent_smiles"))
    if resin_col is None and curer_col is None:
        return None

    key = normalize_name(feature)
    cache = _FORMULATION_PROP_CACHE  # 模块级缓存：跨特征调用复用，避免重复解析同一分子

    def props(smiles: Any) -> Tuple[float, float, float, float]:
        text = clean_smiles(smiles) if isinstance(smiles, str) else None
        if text in cache:
            return cache[text]
        if not text:
            out = (np.nan, np.nan, np.nan, np.nan)
        else:
            mol = _parse_molecule_cached(text)
            if mol is None:
                out = (np.nan, np.nan, np.nan, np.nan)
            else:
                out = (
                    float(Descriptors.MolWt(mol)),
                    float(_count_smarts(mol, "C1OC1")),
                    float(_count_smarts(mol, "C(=O)OC(=O)")),
                    float(_count_active_hydrogen(mol)),
                )
        if len(cache) < 50000:
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
        self._master_paths: Dict[str, List[str]] = {}
        for item in (master_tables or []):
            if isinstance(item, pd.DataFrame):
                self.master_tables.append(("<DataFrame>", item))
            elif isinstance(item, (str, os.PathLike)):
                path = str(item)
                if os.path.exists(path):
                    try:
                        self.master_tables.append((os.path.basename(path), pd.read_csv(path, low_memory=False)))
                        self._master_paths.setdefault(os.path.basename(path), []).append(path)
                    except Exception:
                        pass
        self.structure_columns = list(structure_columns) if structure_columns else None
        self.backend = ExtractionBackend(
            extraction_methods, device=device, max_rows=max_rows_for_extraction, verbose=verbose
        )
        self.verbose = verbose
        self._master_index: Optional[Dict[str, Dict[str, Dict[str, List[Any]]]]] = None
        self._composite_index: Optional[Dict[str, Dict[str, List[Any]]]] = None
        self._fingerprint_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._fingerprint_cols: Optional[List[str]] = None
        self._derived_index: Optional[Dict[str, pd.DataFrame]] = None

    # -- 整行指纹匹配 -------------------------------------------------------
    # 场景：工作区数据就是从某张总表（如 ml_dataset/ml_wide_samples.csv）导出的
    # 行子集，但丢了 ID 列。此时用“两表共有的非目标列”组成整行指纹，
    # 能把工作区行精确对回总表行，从而取到那些“同结构多值”的工艺/测试列
    # （tg_c_test_method、process_max_temperature_c…）——单靠结构键无法区分。
    def _row_fingerprint(self, df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
        """把若干列拼成归一化行指纹（缺失→<NA>，浮点统一 6 位有效数字）。"""
        frames = []
        for c in cols:
            s = df[c]
            if s.dtype.kind in "fiub":
                frames.append(s.map(lambda v: "<NA>" if pd.isna(v) else f"{float(v):.6g}"))
            else:
                frames.append(
                    s.map(lambda v: "<NA>" if _is_blank(v) else re.sub(r"\s+", " ", str(v).strip()))
                )
        if not frames:
            return [""] * len(df)
        joined = frames[0].astype(str)
        for f in frames[1:]:
            joined = joined + "\x1f" + f.astype(str)
        return joined.tolist()

    def _pick_fingerprint_columns(self, workspace_cols: Sequence[str]) -> List[str]:
        """选两表共有的“信息量大”的列做指纹。

        排除：
            - 目标/性能列（tg_c、模量…）—— 它们本来就是要查的东西
            - 纯结构列 —— 单独用不够区分（同结构多行），但可以留作辅助
            - ID 列 —— 工作区通常已丢失
        优先保留：配方量、工艺参数、测试条件、分子描述符等“定位用”列。
        """
        ws_set = set(map(str, workspace_cols))
        out: List[str] = []
        for _tname, frame in self.master_tables:
            for c in frame.columns:
                c = str(c)
                if c not in ws_set or c in out:
                    continue
                k = normalize_name(c)
                # 排除 ID 与性能目标
                if k.endswith("_id") or k in ("source_id", "record_id", "paper_id",
                                               "sample_id", "formulation_id", "process_id"):
                    continue
                if re.search(r"(^|_)(tg|td\d+|tmax|tensile|flexural|compressive|shear|impact|"
                             r"modulus|strength|strain|elongation|charpy|izod|gic|kic|cte|"
                             r"crosslink_density|tan_delta|storage_modulus|gel_time|"
                             r"degree_of_cure|activation_energy|cure_reaction|dsc_cure|"
                             r"char_yield|lap_shear|fracture)", k):
                    continue
                out.append(c)
        return out

    def _build_fingerprint_index(self, workspace: pd.DataFrame) -> None:
        """用两表共有列建整行指纹索引（每个总表各自一份）。"""
        if self._fingerprint_index is not None:
            return
        cols = self._pick_fingerprint_columns(list(workspace.columns))
        self._fingerprint_cols = cols
        index: Dict[str, Dict[str, Any]] = {}
        for tname, frame in self.master_tables:
            usable = [c for c in cols if c in frame.columns]
            if not usable:
                continue
            keys = self._row_fingerprint(frame, usable)
            bucket: Dict[str, Any] = {}
            for key, pos in zip(keys, range(len(frame))):
                bucket.setdefault(key, []).append(pos)
            index[tname] = {"cols": usable, "bucket": bucket, "frame": frame}
        self._fingerprint_index = index

    # -- 关系型总表 join（处理 test_standard_* 这类需跨表取的特征）----------
    def _build_derived_index(self, workspace: pd.DataFrame) -> None:
        """预先把关系型总表里“需要 join 才能得到”的列，物化到主总表上。

        场景：ml_dataset 是关系型库，`test_standard_canonical` 只存在于
        ml_performance_standards.csv，需要 三级 join：
            wide_samples → performance_all(performance_row_id) → standards
        工作区丢 ID 列，所以先靠整行指纹定位到 wide 行，再用宽表自带 ID
        沿链 join 出目标列，物化回宽表副本，后续查表就和普通列一样了。

        通用做法：扫描同目录下的 ml_performance_*.csv，找出含目标列名
        （带/不带 test_ 前缀）的表，用共享 ID 列尝试 join。
        """
        if self._derived_index is not None:
            return
        derived: Dict[str, pd.DataFrame] = {}
        for tname, frame in self.master_tables:
            # 只对含 ID 列的主总表做 join
            id_cols = [c for c in frame.columns if c.endswith("_id")]
            if not id_cols:
                continue
            extra = self._join_related_tables(frame, tname)
            if extra is not None and not extra.empty:
                derived[tname] = extra
        self._derived_index = derived

    def _join_related_tables(
        self, frame: pd.DataFrame, tname: str
    ) -> Optional[pd.DataFrame]:
        """从同目录的关系型表里 join 出额外列（与 frame 行对齐）。

        支持两跳 join：如 test_standard_canonical 需
            wide_samples →(record_id+性能值)→ performance_all →(performance_row_id)→ standards
        """
        import glob

        base_dir = None
        for item in (self._master_paths or {}).get(tname, []):
            base_dir = os.path.dirname(item)
            break
        if base_dir is None:
            return None
        candidates = sorted(glob.glob(os.path.join(base_dir, "ml_performance_*.csv")))
        if not candidates:
            return None

        out = pd.DataFrame(index=frame.index)
        id_cols = [c for c in frame.columns if c.endswith("_id")]
        frames: Dict[str, pd.DataFrame] = {}
        for path in candidates:
            try:
                frames[os.path.basename(path)] = pd.read_csv(path, low_memory=False)
            except Exception:
                continue

        # ---- 第一跳：直接 join ----
        for name, rel in frames.items():
            shared_ids = [c for c in id_cols if c in rel.columns]
            if not shared_ids:
                continue
            new_cols = [c for c in rel.columns if c not in frame.columns and c not in out.columns]
            if not new_cols:
                continue
            merged = self._safe_left_merge(frame, rel, shared_ids, new_cols)
            if merged is None:
                continue
            for c in new_cols:
                if c in merged.columns and merged[c].notna().any():
                    out[c] = merged[c].values

        # ---- 第二跳：用第一跳得到的 performance_row_id 再去 join ----
        #  （standards 只共享 source_id，必须靠 perf_all 的 performance_row_id 桥接）
        bridge_col = None
        for c in ("performance_row_id", "test_condition_id"):
            if c in out.columns and out[c].notna().any():
                bridge_col = c
                break
        if bridge_col is not None:
            bridge = pd.DataFrame({bridge_col: out[bridge_col].values}, index=frame.index)
            for name, rel in frames.items():
                if bridge_col not in rel.columns:
                    continue
                new_cols = [
                    c for c in rel.columns
                    if c not in frame.columns and c not in out.columns and c != bridge_col
                ]
                if not new_cols:
                    continue
                merged = self._safe_left_merge(bridge, rel, [bridge_col], new_cols)
                if merged is None:
                    continue
                for c in new_cols:
                    if c in merged.columns and merged[c].notna().any():
                        out[c] = merged[c].values
        return out if not out.empty else None

    @staticmethod
    def _safe_left_merge(
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: Sequence[str],
        new_cols: Sequence[str],
    ) -> Optional[pd.DataFrame]:
        """一对多安全 join：结果行数与 left 一致（一对多时取首个非空）。

        直接 merge 会因右表同键多行而膨胀，这里改为按键映射取首值。
        """
        try:
            sub = right[list(on) + list(new_cols)].copy()
        except KeyError:
            return None
        # 构造键 → 行 映射（首次出现优先，但优先非空）
        keys = sub[list(on)].astype(str).agg("\x1f".join, axis=1)
        sub = sub.assign(_k=keys)
        # 排序：非空优先
        for c in new_cols:
            sub[c] = sub[c]
        sub = sub.drop_duplicates(subset=["_k"], keep="first")
        mapper = sub.set_index("_k")
        left_keys = left[list(on)].astype(str).agg("\x1f".join, axis=1)
        result = pd.DataFrame(index=left.index)
        joined = mapper.reindex(left_keys.values)
        joined.index = left.index
        for c in new_cols:
            if c in joined.columns:
                result[c] = joined[c]
        return result if not result.empty else None

    def _lookup_from_hits(
        self,
        feature: str,
        row_pos: int,
        fp_hits: Dict[str, List[Optional[List[int]]]],
    ) -> Tuple[Any, str]:
        """用预计算的行号命中结果取值（避免逐特征重算指纹）。"""
        fkey = normalize_name(feature)
        for tname, info in (self._fingerprint_index or {}).items():
            positions = fp_hits.get(tname, [None] * (row_pos + 1))[row_pos]
            if not positions:
                continue
            frame = info["frame"]
            # 1) 主表直接有该列
            if feature in frame.columns:
                vals = [frame.at[p, feature] for p in positions]
                vals = [v for v in vals if not _is_blank(v)]
                if vals:
                    return _collapse_values(vals), f"整行指纹（{tname}）"
            # 2) 关系表 join 出来的列（含名称变体与计数）
            derived = (self._derived_index or {}).get(tname)
            if derived is not None:
                actual = self._match_derived_column(derived, fkey)
                if actual is not None:
                    vals = [derived.iloc[p][actual] for p in positions]
                    vals = [v for v in vals if not _is_blank(v)]
                    if vals:
                        return _collapse_values(vals), f"关系表join（{tname}）"
                elif fkey.endswith("_count"):
                    base = fkey[: -len("_count")]
                    cnts = []
                    for p in positions:
                        vs = [
                            derived.iloc[p][c] for c in derived.columns
                            if normalize_name(c).startswith(base) and not _is_blank(derived.iloc[p][c])
                        ]
                        cnts.append(len(set(map(str, vs))))
                    if cnts:
                        return _collapse_values(cnts), f"派生计数（{tname}）"
        return np.nan, "指纹未匹配"

    def _lookup_derived(
        self, row: pd.Series, feature: str
    ) -> Tuple[Any, str]:
        """先靠整行指纹定位主总表行号，再从 join 出来的派生列取值。

        支持名称变体：模型要 test_standard_canonical，关系表里叫 standard_canonical。
        支持聚合列：test_standard_count ← 同组内 standard_* 非空去重计数。
        """
        if self._derived_index is None or self._fingerprint_index is None:
            return np.nan, "未建派生索引"
        fkey = normalize_name(feature)
        for tname, info in self._fingerprint_index.items():
            derived = self._derived_index.get(tname)
            if derived is None:
                continue
            actual = self._match_derived_column(derived, fkey)
            is_count = actual is None and fkey.endswith("_count")
            if actual is None and not is_count:
                continue
            cols = info["cols"]
            try:
                key = self._row_fingerprint(pd.DataFrame([row[cols].to_dict()]), cols)[0]
            except Exception:
                continue
            positions = info["bucket"].get(key)
            if not positions:
                continue
            if is_count:
                base = fkey[: -len("_count")]
                cnts = []
                for p in positions:
                    vals = [
                        derived.iloc[p][c] for c in derived.columns
                        if normalize_name(c).startswith(base) and not _is_blank(derived.iloc[p][c])
                    ]
                    cnts.append(len(set(map(str, vals))))
                if cnts:
                    return _collapse_values(cnts), f"派生计数（{tname}）"
                continue
            values = [derived.iloc[p][actual] for p in positions]
            values = [v for v in values if not _is_blank(v)]
            if values:
                return _collapse_values(values), f"关系表 join（{tname}，{len(positions)}行）"
        return np.nan, "派生列未匹配"

    @staticmethod
    def _match_derived_column(derived: pd.DataFrame, fkey: str) -> Optional[str]:
        """在派生列里按名称变体匹配（处理 test_ 前缀差异）。"""
        # 1) 完全相等
        for c in derived.columns:
            if normalize_name(c) == fkey:
                return c
        # 2) 去掉/加上 test_ 前缀
        variants = set()
        if fkey.startswith("test_"):
            variants.add(fkey[len("test_"):])
        else:
            variants.add("test_" + fkey)
        # 3) 去 measurement_/tg_c_ 等修饰前缀
        for pre in ("test_", "measurement_"):
            if fkey.startswith(pre):
                variants.add(fkey[len(pre):])
        for c in derived.columns:
            ck = normalize_name(c)
            if ck in variants:
                return c
            # 尾部包含（如 standard_canonical vs test_standard_canonical）
            for v in variants:
                if ck.endswith("_" + v) or v.endswith("_" + ck):
                    return c
        return None

    def lookup_by_fingerprint(
        self, row: pd.Series, feature: str
    ) -> Tuple[Any, str]:
        """用整行指纹查总表。返回 (值, 说明)。

        这能解决“同结构多值”的工艺/测试列：同一树脂+固化剂下不同测试条件
        能通过配方量/工艺参数区分开。
        """
        if self._fingerprint_index is None:
            return np.nan, "未建指纹索引"
        fkey = normalize_name(feature)
        for tname, info in self._fingerprint_index.items():
            frame = info["frame"]
            if feature not in frame.columns:
                continue
            cols = info["cols"]
            try:
                key = self._row_fingerprint(
                    pd.DataFrame([row[cols].to_dict()]), cols
                )[0]
            except Exception:
                continue
            positions = info["bucket"].get(key)
            if not positions:
                continue
            values = [frame.at[p, feature] for p in positions]
            values = [v for v in values if not _is_blank(v)]
            if values:
                return _collapse_values(values), f"整行指纹（{tname}，{len(positions)}行匹配）"
        return np.nan, "指纹未匹配"

    # -- 总表索引 -----------------------------------------------------------
    #: 超过此列数/行数的总表不再建“单键/复合键”全量索引（改用按需逐特征扫描）。
    #: 大宽表（如 ml_wide_samples 10749×1248）建索引要 1200 万次 dict 操作（≈45s），
    #: 而指纹匹配已能精确命中，这些降级索引几乎用不到。
    _INDEX_SIZE_LIMIT = 400

    def _build_master_index(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """构建总表索引（按组分列作用域隔离，避免跨组分污染）。

        旧实现的三个缺陷：
          1. 所有结构列混入同一个 bucket → 用树脂 SMILES 能查到固化剂的 MW
          2. 同结构多行时 first-wins → 工艺列（温度/phr）随机取一个
          3. resolve() 只用 primary_struct（resin_1_structure）作键

        新结构：
          index[结构列名][归一化结构键][归一化特征名] = 值列表
          值列表保留全部观测，查询时按特征类型决策（见 lookup_master）。

        性能：超宽表跳过（返回空 dict），由 lookup_master 的按需扫描兜底。
        """
        if self._master_index is not None:
            return self._master_index
        index: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
        for _table_name, frame in self.master_tables:
            struct_cols = detect_structure_columns(frame)
            if not struct_cols:
                continue
            feature_cols = [c for c in frame.columns if c not in struct_cols]
            if not feature_cols:
                continue
            # 超宽表：不建全量索引（改用 _scan_master_column）
            if len(feature_cols) > self._INDEX_SIZE_LIMIT or len(frame) * len(feature_cols) > 3_000_000:
                continue
            # 预归一化特征名（避免逐格 normalize_name）
            norm_names = {fc: normalize_name(fc) for fc in feature_cols}
            # 预先按组分契合度分组（每个结构列一次，不逐行重算）
            scoped_by_struct: Dict[str, List[str]] = {
                sc: [fc for fc in feature_cols if self._structure_scope(fc, sc) > 0]
                for sc in struct_cols
            }            # 用 numpy 取列值（比 frame.at[row, col] 快数百倍）
            col_arrays: Dict[str, Any] = {fc: frame[fc].to_numpy() for fc in feature_cols}
            for struct_col in struct_cols:
                scoped = scoped_by_struct.get(struct_col) or []
                if not scoped:
                    continue
                col_scope = index.setdefault(struct_col, {})
                keys = [self._structure_key(v) for v in frame[struct_col].tolist()]
                arrays = [(norm_names[fc], col_arrays[fc]) for fc in scoped]
                for pos, key in enumerate(keys):
                    if not key:
                        continue
                    bucket = col_scope.setdefault(key, {})
                    for fname, arr in arrays:
                        value = arr[pos]
                        if _is_blank(value):
                            continue
                        bucket.setdefault(fname, []).append(value)
        self._master_index = index
        return index

    @staticmethod
    def _structure_key(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        text = clean_smiles(value) or value.strip()
        return re.sub(r"\s+", "", text).lower()

    @staticmethod
    def _structure_scope(feature: str, struct_col: str) -> int:
        """特征与结构列的组分契合度（越大越契合）。

        固化剂特征必须用固化剂结构查（curing_agent_1_* ← curing_agent_1_structure），
        否则会拿到同一行里碰巧配对的另一个组分的值。
        """
        fkey = normalize_name(feature)
        ckey = normalize_name(struct_col)
        prefix = _STRUCT_PREFIX_RE.sub("", ckey)
        if prefix and (fkey == prefix or fkey.startswith(prefix + "_")):
            return 1000
        # 组分角色层面的兜底（resin_2_* 也可用 resin_1_structure，若没有 resin_2_structure）
        f_role = _ROLE_RE.match(fkey)
        c_role = _ROLE_RE.match(ckey)
        if f_role and c_role and f_role.group(1) == c_role.group(1):
            return 100
        if not f_role:
            # 无前缀的通用分子特征（mw/logp/epoxy…）：任何结构列都行，弱契合
            return 10
        return 0

    #: 同结构必然多值、查表不可靠的特征（工艺/配方量）——查表命中也不用
    _UNSTABLE_LOOKUP_PATTERNS = (
        r"(^|_)amount_phr$", r"(^|_)phr$", r"^process_", r"_process_",
        r"(^|_)temperature", r"(^|_)time(_h|_min|_s)?$", r"(^|_)atmosphere$",
        r"(^|_)pressure", r"(^|_)post_cure", r"(^|_)heating_rate", r"(^|_)cooling_rate",
        r"(^|_)ramp", r"(^|_)dwell", r"(^|_)cure_schedule", r"(^|_)cycle",
        r"^test_", r"_test_", r"(^|_)standard", r"(^|_)method$", r"(^|_)instrument$",
        r"(^|_)yield$", r"(^|_)conversion", r"(^|_)ratio$", r"(^|_)r_value$",
        r"(^|_)stoich", r"(^|_)density$", r"(^|_)modulus", r"(^|_)strength$",
        r"(^|_)tg_c?$", r"(^|_)tensile", r"(^|_)flexural", r"(^|_)elongation",
    )
    #: 预编译（热路径：每特征都要判）
    _UNSTABLE_RE = re.compile("|".join(_UNSTABLE_LOOKUP_PATTERNS))

    @classmethod
    def is_unstable_feature(cls, feature: str) -> bool:
        """是否为“同结构必然多值”的特征（工艺/配方/测试/性能类）。"""
        return bool(cls._UNSTABLE_RE.search(normalize_name(feature)))

    def lookup_master(
        self,
        smiles: Any,
        feature: str,
        *,
        struct_col: Optional[str] = None,
        allow_unstable: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """按结构查总表。返回标量（多值时取众数/中位数）。

        参数:
            struct_col:     指定用哪个结构列作键（None 则自动选最契合的）
            allow_unstable: 是否允许查“同结构必然多值”的特征（工艺/配方量）。
            context:        上下文结构值 {结构列名: 值}，用于复合键查询
                            （树脂+固化剂组合能唯一定位一行，比单结构准得多）

        返回: 标量值或 np.nan
        """
        fkey = normalize_name(feature)
        unstable = self.is_unstable_feature(feature)
        if unstable and not allow_unstable:
            return np.nan

        # 1) 优先复合键（多结构列组合）—— 对工艺/配方量尤其关键
        if context:
            value = self._lookup_composite(context, feature)
            if not _is_blank(value):
                return value
            if unstable and not allow_unstable:
                return np.nan

        # 2) 单结构键
        key = self._structure_key(smiles)
        if not key:
            return np.nan
        index = self._build_master_index()
        scope = None
        if struct_col is not None:
            scope = index.get(struct_col)
        if scope is None:
            # 超宽表没建索引：按需扫描目标列（只扫一列，比建全量索引快得多）
            scanned = self._scan_master_column(key, feature)
            if not _is_blank(scanned):
                return scanned
            best = None
            for col_name, col_scope in index.items():
                score = self._structure_scope(feature, col_name)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, col_scope)
            if best is None:
                return np.nan
            scope = best[1]
        bucket = scope.get(key)
        if not bucket:
            return np.nan
        values = bucket.get(fkey)
        if not values:
            return np.nan
        return _collapse_values(values)

    def _lookup_composite(
        self, context: Dict[str, Any], feature: str
    ) -> Any:
        """用多个结构列的组合键查表（比单结构键精确得多）。

        例：树脂+固化剂组合能把 50 行工艺候选缩到几行，工艺量也能取到接近真值。
        """
        index = self._build_composite_index()
        if not index:
            return np.nan
        parts = []
        for col, raw in context.items():
            key = self._structure_key(raw)
            if key:
                parts.append(f"{normalize_name(col)}={key}")
        if not parts:
            return np.nan
        # 用“可用结构列的子集”做键：优先用全部，不行再退到部分
        parts.sort()
        for take in range(len(parts), 0, -1):
            combo = "||".join(parts[:take])
            bucket = index.get(combo)
            if not bucket:
                continue
            values = bucket.get(normalize_name(feature))
            if values:
                return _collapse_values(values)
        return np.nan

    def _build_composite_index(self) -> Dict[str, Dict[str, List[Any]]]:
        """构建复合键索引：结构列组合 → 特征名 → 值列表。

        注意：这是“指纹匹配”的降级方案（工作区不是总表行子集时才需要），
        因此只为“不稳定特征”（工艺/配方量）建索引，避免在大宽表上白耗时间。
        """
        if self._composite_index is not None:
            return self._composite_index
        from itertools import combinations

        index: Dict[str, Dict[str, List[Any]]] = {}
        for _table_name, frame in self.master_tables:
            struct_cols = detect_structure_columns(frame)
            if len(struct_cols) < 2:
                continue
            # 只取不稳定特征（工艺/配方量）——它们才需要复合键区分
            feature_cols = [
                c for c in frame.columns
                if c not in struct_cols
                and self.is_unstable_feature(c)
            ]
            if not feature_cols:
                continue
            # 超宽表：不建复合键索引（指纹匹配已能精确定位）
            if len(frame.columns) > self._INDEX_SIZE_LIMIT:
                continue
            norm_names = {fc: normalize_name(fc) for fc in feature_cols}
            col_arrays = {fc: frame[fc].to_numpy() for fc in feature_cols}
            # 每行的结构键
            struct_keys = {
                sc: [self._structure_key(v) for v in frame[sc].tolist()] for sc in struct_cols
            }
            n_rows = len(frame)
            for take in range(2, min(4, len(struct_cols)) + 1):
                for combo_cols in combinations(struct_cols, take):
                    key_lists = [struct_keys[c] for c in combo_cols]
                    label = sorted(normalize_name(c) for c in combo_cols)
                    for pos in range(n_rows):
                        parts = []
                        ok = True
                        for lbl, kl in zip(label, key_lists):
                            k = kl[pos]
                            if not k:
                                ok = False
                                break
                            parts.append(f"{lbl}={k}")
                        if not ok:
                            continue
                        combo = "||".join(parts)
                        bucket = index.setdefault(combo, {})
                        for fname, arr in ((norm_names[fc], col_arrays[fc]) for fc in feature_cols):
                            value = arr[pos]
                            if _is_blank(value):
                                continue
                            bucket.setdefault(fname, []).append(value)
        self._composite_index = index
        return index

    def _scan_master_column(self, key: str, feature: str) -> Any:
        """按需扫描：只在总表里找该特征列，按结构键取值（用于超宽表）。"""
        fkey = normalize_name(feature)
        for _tname, frame in self.master_tables:
            actual = None
            for c in frame.columns:
                if normalize_name(c) == fkey:
                    actual = c
                    break
            if actual is None:
                continue
            struct_cols = detect_structure_columns(frame)
            if not struct_cols:
                continue
            # 选组分最契合的结构列
            scored = sorted(
                struct_cols, key=lambda sc: -self._structure_scope(feature, sc)
            )
            arr = frame[actual].to_numpy()
            for sc in scored:
                if self._structure_scope(feature, sc) <= 0:
                    break
                keys = [self._structure_key(v) for v in frame[sc].tolist()]
                vals = [arr[i] for i, k in enumerate(keys) if k == key and not _is_blank(arr[i])]
                if vals:
                    return _collapse_values(vals)
        return np.nan

    def lookup_master_multi(
        self,
        smiles: Any,
        features: Sequence[str],
        *,
        struct_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """一次查多个特征（复用结构键与作用域选择）。"""
        return {
            f: self.lookup_master(smiles, f, struct_col=struct_col) for f in features
        }

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

        # ---- 批量预提取：把同一结构列上的所有特征一次算完，避免逐特征重复计算 ----
        pending = [
            f for f in required_features
            if f not in already and f not in result.columns and looks_molecular(f)
        ]
        skipped_non_molecular = [
            f for f in required_features
            if f not in already and f not in result.columns and not looks_molecular(f)
        ]
        batched = self._batch_compute(result, pending, struct_cols, max_rows_for_compute) if pending else {}

        # 复合键上下文（所有结构列的值）只构建一次，避免逐特征重建（曾占 22s）
        ctx_cols = [c for c in struct_cols if c in result.columns]
        if self.master_tables and ctx_cols:
            contexts = [
                {c: row[i] for i, c in enumerate(ctx_cols)}
                for row in result[ctx_cols].itertuples(index=False, name=None)
            ]
        else:
            contexts = [None] * len(result)

        # 整行指纹索引（最高优先级）：工作区是总表行子集时能精确定位
        if self.master_tables:
            self._build_fingerprint_index(result)
            self._build_derived_index(result)
        fp_cols = self._fingerprint_cols or []
        # 关键性能优化：工作区每行的指纹只算一次（否则 2064 特征 × N 行 = 巨量重算）
        ws_fp_keys: List[str] = []
        if fp_cols:
            ws_fp_keys = self._row_fingerprint(result, fp_cols)
        # 每行的“总表行号”缓存：{表名: [行号列表 or None]}，供所有特征复用
        fp_hits: Dict[str, List[Optional[List[int]]]] = {}
        for tname, info in (self._fingerprint_index or {}).items():
            usable = info["cols"]
            if usable == fp_cols:
                fp_hits[tname] = [info["bucket"].get(k) for k in ws_fp_keys]
            else:
                # 指纹列不完全一致：按该表自己的列重算一次
                sub_keys = self._row_fingerprint(result, usable)
                fp_hits[tname] = [info["bucket"].get(k) for k in sub_keys]

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

            # --- B) 批量预提取命中 ---
            hit = batched.get(feature)
            if hit is not None and hit.notna().any():
                result[feature] = hit
                report["from_extractor"][feature] = "批量提取"
                report["columns_added"].append(feature)
                continue

            # --- C) 总表查询 ---
            #     C1 整行指纹（最准，能区分同结构不同工艺）
            #     C2 复合键（树脂+固化剂组合）
            #     C3 单结构键（最低置信）
            #     工作区没有该列时，总表查到的值比 NaN 有用。
            if self.master_tables:
                values: List[Any] = []
                sources: List[str] = []
                if fp_cols:
                    for pos, idx in enumerate(result.index):
                        v, how = self._lookup_from_hits(feature, pos, fp_hits)
                        values.append(v)
                        sources.append(how)
                else:
                    values = [np.nan] * len(result)
                    sources = [""] * len(result)

                from_master = pd.Series(values, index=result.index)
                # 指纹没命中的行，退到复合键 / 单结构键（仅当确实有缺失行时才做，
                # 避免在指纹全覆盖时白建昂贵的索引）
                if from_master.isna().any():
                    lookup_col = self._pick_lookup_column(feature, struct_cols, result)
                    if lookup_col is not None:
                        fkey = normalize_name(feature)
                        unstable = self.is_unstable_feature(feature)
                        for pos, (idx, v, ctx) in enumerate(
                            zip(result.index, result[lookup_col], contexts)
                        ):
                            if not pd.isna(from_master.iloc[pos]):
                                continue
                            hv = self._lookup_composite(ctx, feature) if ctx else np.nan
                            if not _is_blank(hv):
                                from_master.iloc[pos] = hv
                                sources[pos] = "复合键"
                            else:
                                sv = self.lookup_master(
                                    v, feature, struct_col=None, allow_unstable=unstable
                                )
                                if not _is_blank(sv):
                                    from_master.iloc[pos] = sv
                                    sources[pos] = "单结构"

                if from_master.notna().any():
                    result[feature] = from_master
                    from collections import Counter
                    _cnt = Counter(s.split("（")[0] for s in sources if s)
                    tag = "、".join(f"{k}×{v}" for k, v in _cnt.most_common(3)) or "总表"
                    report["from_master"][feature] = f"总表查询（{tag}）"
                    if self.is_unstable_feature(feature):
                        n_fp = sum(1 for s in sources if s.startswith("整行指纹"))
                        if n_fp < len(result):
                            report.setdefault("low_confidence", {})[feature] = (
                                f"{len(result) - n_fp}/{len(result)} 行未通过整行指纹定位，"
                                "已退到结构键（取中位数/众数）；建议用工作区真实值"
                            )
                    report["columns_added"].append(feature)
                    continue

            # --- D) 提取引擎（仅对可能来自分子的特征，避免无谓重后端调用）---
            if looks_molecular(feature):
                extracted = self._extract_feature(result, feature, struct_cols)
                if extracted is not None and extracted.notna().any():
                    result[feature] = extracted
                    report["from_extractor"][feature] = "提取引擎"
                    report["columns_added"].append(feature)
                    continue

            report["unresolved"].append(feature)

        report["skipped_non_molecular"] = skipped_non_molecular
        return result, report

    def _pick_lookup_column(
        self, feature: str, struct_cols: Sequence[str], df: pd.DataFrame
    ) -> Optional[str]:
        """选一个用于查总表的结构列（工作区里存在、且与特征组分最契合）。"""
        available = [c for c in struct_cols if c in df.columns]
        if not available:
            return None
        scored = sorted(
            available, key=lambda c: -self._structure_scope(feature, c)
        )
        return scored[0] if scored else None

    def _batch_compute(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        struct_cols: Sequence[str],
        max_rows: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """批量计算：按结构列分组，每个结构列的每行只解析一次分子，
        再把该行能提供的全部特征一次算出。

        这避免了逐特征重算（1400 个特征 × N 行 = 巨量重复 RDKit 调用）。
        """
        out: Dict[str, pd.Series] = {}
        remaining = list(features)
        limit = max_rows or self.backend.max_rows

        for col in struct_cols:
            if not remaining:
                break
            if col not in df.columns:
                continue
            values = df[col].head(limit) if limit else df[col]
            if not values.notna().any():
                continue
            # 该结构列能覆盖哪些特征（按特征名前缀判断）
            covered = [f for f in remaining if _feature_belongs_to_column(f, col, df)]
            if not covered:
                continue
            # 一次解析每行分子，然后逐个特征取值
            mols: List[Any] = []
            for v in values:
                try:
                    mols.append(_parse_molecule_cached(v))
                except Exception:
                    mols.append(None)
            # 逐行 × 逐特征：避免 pandas .loc 逐格赋值（那会走 Series 索引，极慢）
            for feature in covered:
                col_values: List[Any] = []
                any_ok = False
                for mol in mols:
                    if mol is None:
                        col_values.append(np.nan)
                        continue
                    try:
                        val = _compute_feature_from_mol(mol, feature)
                    except Exception:
                        val = None
                    if val is None:
                        col_values.append(np.nan)
                    else:
                        col_values.append(float(val))
                        any_ok = True
                if any_ok:
                    series = pd.Series(col_values, index=values.index, dtype=float)
                    out[feature] = series.reindex(df.index)
                    remaining.remove(feature)
        return out

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
