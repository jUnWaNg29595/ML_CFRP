# -*- coding: utf-8 -*-
"""
高分子物理指数工厂 (Polymer Physics Index Factory)
=================================================
从**结构 SMILES + 配方当量**计算**零标签**的高分子物理量，作为 PINN 的物理输入。

设计原则
--------
- 不依赖任何实测辅助标签（唯一例外: 交联密度 ν 由 CrosslinkNuEncoder 提供）
- 全部指标可由分子结构 + 配方直接算出，无数据泄漏
- 缺失/不可解析时返回 NaN，交由模型插补链处理（不中断）

准入依据（在 ml_dataset 上实测的 Spearman，见 scripts/bench_physics_baselines*.py）
------------------------------------------------------------------------------
    phys_csp3      sp3 碳分数           → tg_c  -0.507   ✅ 硬基座
    phys_f_ar      芳香碳分数           → tg_c   0.478   ✅ 硬基座
    phys_bde_mean  平均键解离能 BDE      → tg_c   0.453   ✅ 硬基座
                                        → td5_c  0.403   ✅ 硬基座
    phys_flex      柔性指数(可旋转键/重原子) → tg_c -0.413  ✅ 硬基座
    phys_delta     溶解度参数 δ (Fedors) → 弱(delta 变异系数仅 6.6%，区分力有限)
    phys_bde_min   最小 BDE              → tg_c  -0.332   ⚠️ 软约束
    phys_ced       内聚能密度 CED         → 弱 (与 delta 同源)

已验证的下游增益（PINN 加入 5 个指数作为输入，λ_ν=0.3, physics_weight=0.25）：
    tg_c                 0.6842 → 0.7649  (+0.081)
    tensile_modulus_gpa  0.6326 → 0.6609  (+0.028)
    td5_c                0.4684 → 0.4993  (+0.031)
    tensile_strength_mpa 0.5477 → 0.5719  (+0.024)

已实测**弃用**的公式（避免后人重复踩坑）
----------------------------------------
    Flory 交联密度公式        → 实测 ν: Spearman  0.088  ❌
    反应模拟网络节点密度       → 实测 ν: Spearman  0.145  ❌
    橡胶弹性 3νRT → 玻璃态模量 → 尺度差 94~139 倍, 秩 0.04~0.24 ❌（仅适用橡胶支路）
    CED/δ → 玻璃态模量        → 0.02~0.20（CED 变异太小）    ❌
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量与表
PHYSICS_PREFIX = "phys_"

# 默认注入 PINN 的物理指数（实测准入 ≥0.4 的核心 5 项）
DEFAULT_PHYSICS_FEATURES: tuple = (
    "csp3",        # sp3 碳分数（链柔性）
    "f_ar",        # 芳香碳分数（链刚性）
    "flex",        # 可旋转键/重原子（链柔性）
    "bde_mean",    # 平均键解离能（键刚度 / 热稳定性）
    "delta",       # 溶解度参数（内聚能）
)

# 全部可提取指数 → 中文标签（UI/导出用）
PHYSICS_FEATURE_LABELS: dict = {
    "csp3": "sp3 碳分数",
    "f_ar": "芳香碳分数",
    "flex": "可旋转键/重原子",
    "bde_mean": "平均键解离能 (kJ/mol)",
    "bde_min": "最小键解离能 (kJ/mol)",
    "bde_weak_frac": "弱键占比 (<340 kJ/mol)",
    "delta": "溶解度参数 δ (MPa^0.5)",
    "ced": "内聚能密度 CED (MPa)",
    "n_arom_ring": "芳香环数",
    "rho_vdw": "堆砌密度 (g/cm³, vdW 近似)",
    "mw": "分子量 (g/mol)",
}

# 全部可提取指数（顺序即输出顺序）
ALL_PHYSICS_FEATURES: tuple = tuple(PHYSICS_FEATURE_LABELS.keys())

# 结构列（按角色）——与窄表/宽表命名保持一致
_STRUCTURE_ROLE_PATTERNS: tuple = (
    ("resin", r"^resin_(\d+)_structure$"),
    ("curing_agent", r"^curing_agent_(\d+)_structure$"),
    ("small_additive", r"^small_additive_(\d+)_structure$"),
    ("reactive_diluent", r"^reactive_diluent_(\d+)_structure$"),
    ("toughener", r"^reactive_toughener_(\d+)_structure$"),
)

# 组分用量列（用于加权，缺失时按角色默认权重）
_AMOUNT_PATTERNS = {
    "resin": (r"^resin_(\d+)_amount_phr$", r"^resin_total_phr$"),
    "curing_agent": (r"^curing_agent_(\d+)_amount_phr$", r"^curing_agent_total_phr$"),
    "small_additive": (r"^small_additive_(\d+)_amount_phr$", None),
    "reactive_diluent": (r"^reactive_diluent_(\d+)_amount_phr$", None),
    "toughener": (r"^reactive_toughener_amount_phr$", r"^reactive_toughener_total_phr$"),
}

# Fedors 基团贡献: Δe (cal/mol), Δv (cm³/mol) —— 文献近似值 (Fedors 1974)
# 量纲校验：算出的 δ 中位数 19.8 MPa^0.5（环氧文献 19~23），CED 393 MPa（文献 350~550）
_FEDORS: Dict[str, tuple] = {
    "CH3": (1125, 33.5), "CH2": (1180, 16.1), "CH": (820, -1.0), "C": (350, -19.2),
    "phenyl": (7630, 71.4), "OH": (7120, 10.0), "O_ether": (1000, 3.8),
    "C=O": (4150, 10.8), "COO": (4300, 18.0), "NH2": (3000, 19.2),
    "NH": (2000, 4.5), "N_tert": (1000, -9.0), "ring": (250, 16.0),
    "S": (3400, 12.0), "Si": (1400, 15.0), "Cl": (2760, 24.0), "F": (1000, 18.0),
}

# 键解离能 BDE (kJ/mol) —— 常见文献值；(元素1, 元素2, 是否芳香键)
_BDE: Dict[tuple, float] = {
    ("C", "C", False): 350.0,   # sp3–sp3
    ("C", "C", True): 480.0,    # 芳环内 / 芳–芳
    ("C", "N", False): 305.0,   # C–N 胺键（环氧-胺网络弱键）
    ("C", "N", True): 380.0,    # 芳–N
    ("C", "O", False): 345.0,   # 醚 / 醇 C–O
    ("C", "O", True): 400.0,    # 芳–O
    ("C", "S", False): 275.0,   # C–S（弱）
    ("S", "S", False): 240.0,   # S–S（最弱）
    ("C", "F", False): 485.0,
    ("C", "Cl", False): 335.0,
    ("C", "Si", False): 320.0,
    ("Si", "O", False): 460.0,
    ("C", "H", False): 415.0,
    ("N", "H", False): 390.0,
    ("O", "H", False): 460.0,
}
_BDE_DEFAULT = 360.0

# 元素范德华半径 (Å)，用于堆砌密度代理
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "F": 1.47,
        "Si": 2.10, "Cl": 1.75, "Br": 1.85, "P": 1.80}
_VDW_DEFAULT = 1.70

_BIGSMILES_MARKERS = re.compile(r"\{|\[<[^\]]*\]|\[>[^\]]*\]|\}")

_SMILES_CACHE: Dict[str, Optional[Dict[str, float]]] = {}
_CACHE_LIMIT = 20000


# ---------------------------------------------------------------- 工具
def clean_structure_string(text: Any) -> str:
    """去除 BigSMILES 的连接标记，尽量还原可解析 SMILES。"""
    return _BIGSMILES_MARKERS.sub("", str(text)).strip()


def _rdkit():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def _fedors(mol) -> tuple:
    """Fedors 基团贡献 → (CED [MPa], δ [MPa^0.5])"""
    Chem = _rdkit()
    de = dv = 0.0
    n_arom_ring = 0
    try:
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                n_arom_ring += 1
    except Exception:
        pass

    counts = {k: 0 for k in _FEDORS}
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s == "C":
            if a.GetIsAromatic():
                continue
            nh = a.GetTotalNumHs()
            nc = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "C")
            if nh >= 3 or nc + nh <= 1:
                counts["CH3"] += 1
            elif nh == 2:
                counts["CH2"] += 1
            elif nh == 1:
                counts["CH"] += 1
            else:
                counts["C"] += 1
        elif s == "O":
            counts["O_ether"] += 1
        elif s == "N":
            nh = a.GetTotalNumHs()
            counts["NH2" if nh >= 2 else ("NH" if nh == 1 else "N_tert")] += 1
        elif s == "S":
            counts["S"] += 1
        elif s == "Si":
            counts["Si"] += 1
        elif s == "Cl":
            counts["Cl"] += 1
        elif s == "F":
            counts["F"] += 1

    p_oh = Chem.MolFromSmarts("[OX2H]")
    p_coo = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    p_co = Chem.MolFromSmarts("[CX3]=[OX1]")
    n_oh = len(mol.GetSubstructMatches(p_oh)) if p_oh else 0
    n_coo = len(mol.GetSubstructMatches(p_coo)) if p_coo else 0
    n_co = len(mol.GetSubstructMatches(p_co)) if p_co else 0
    counts["O_ether"] = max(0, counts["O_ether"] - n_oh - n_co)
    counts["OH"] += n_oh
    counts["COO"] += n_coo
    counts["C=O"] += max(0, n_co - n_coo)
    counts["phenyl"] += n_arom_ring
    counts["ring"] += mol.GetRingInfo().NumRings()

    for k, c in counts.items():
        if c:
            de += c * _FEDORS[k][0]
            dv += c * _FEDORS[k][1]
    if dv <= 1.0:
        return (np.nan, np.nan)
    ced = de / dv * 4.184          # cal/cm³ → J/cm³ = MPa
    return (float(ced), float(np.sqrt(max(ced, 1e-6))))


def _bde_stats(mol) -> tuple:
    """键解离能统计 → (min, mean, 弱键占比)"""
    vals: List[float] = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        if "H" in (s1, s2):
            continue
        arom = bool(a1.GetIsAromatic() and a2.GetIsAromatic())
        v = (_BDE.get((s1, s2, arom)) or _BDE.get((s2, s1, arom))
             or _BDE.get((s1, s2, False)) or _BDE.get((s2, s1, False))
             or _BDE_DEFAULT)
        vals.append(float(v))
    if not vals:
        return (np.nan, np.nan, np.nan)
    v = np.asarray(vals, dtype=float)
    return (float(v.min()), float(v.mean()), float((v < 340).mean()))


def molecule_physics_indices(smiles: Any) -> Optional[Dict[str, float]]:
    """单个分子的物理指数（带缓存）。不可解析返回 None。"""
    key = str(smiles)
    if key in _SMILES_CACHE:
        return _SMILES_CACHE[key]
    Chem = _rdkit()
    from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD

    mol = None
    if isinstance(smiles, str) and smiles.strip():
        try:
            mol = Chem.MolFromSmiles(clean_structure_string(smiles))
        except Exception:
            mol = None
    if mol is None:
        if len(_SMILES_CACHE) < _CACHE_LIMIT:
            _SMILES_CACHE[key] = None
        return None

    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ced, delta = _fedors(mol)
    bmin, bmean, bweak = _bde_stats(mol)
    mw = float(Descriptors.MolWt(mol))
    vdw_vol = sum(4.0 / 3.0 * np.pi * _VDW.get(a.GetSymbol(), _VDW_DEFAULT) ** 3
                  for a in mol.GetAtoms())
    molar_vol = vdw_vol * 0.6022                      # cm³/mol (vdW 近似)
    out = {
        "f_ar": n_arom_atoms / n_heavy,
        "csp3": float(Descriptors.FractionCSP3(mol)),
        "flex": float(Descriptors.NumRotatableBonds(mol)) / n_heavy,
        "n_arom_ring": float(rdMD.CalcNumAromaticRings(mol)),
        "bde_min": bmin,
        "bde_mean": bmean,
        "bde_weak_frac": bweak,
        "ced": ced,
        "delta": delta,
        "rho_vdw": (mw / molar_vol) if molar_vol > 0 else np.nan,
        "mw": mw,
    }
    if len(_SMILES_CACHE) < _CACHE_LIMIT:
        _SMILES_CACHE[key] = out
    return out


def detect_structure_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """探测各角色的结构列（按序号排序）。"""
    roles: Dict[str, List[str]] = {}
    for role, pat in _STRUCTURE_ROLE_PATTERNS:
        rx = re.compile(pat)
        cols = [(int(m.group(1)), c) for c in df.columns if (m := rx.match(str(c)))]
        if cols:
            roles[role] = [c for _, c in sorted(cols)]
    return roles


def _component_weights(df: pd.DataFrame, role: str, cols: Sequence[str]) -> np.ndarray:
    """组分权重：优先逐组分 amount_phr（X_structure → X_amount_phr），
    其次角色总 phr，最后等权。"""
    n = len(df)
    w = np.ones((n, len(cols)), dtype=float)
    hit = False
    for j, c in enumerate(cols):
        cand = str(c).replace("_structure", "_amount_phr")
        if cand != str(c) and cand in df.columns:
            col = pd.to_numeric(df[cand], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(col) & (col > 0)
            if ok.any():
                w[ok, j] = col[ok]
                w[~ok, j] = 0.0
                hit = True
    if not hit:
        _, total_pat = _AMOUNT_PATTERNS.get(role, (None, None))
        if total_pat and total_pat in df.columns:
            tot = pd.to_numeric(df[total_pat], errors="coerce").to_numpy(dtype=float)
            tot = np.where(np.isfinite(tot) & (tot > 0), tot, 0.0)
            w = np.repeat(tot.reshape(-1, 1), len(cols), axis=1)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return w / s


def compute_polymer_physics_indices(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    structure_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    计算物理指数表（列名 phys_<name>，与 df 同索引）。

    Args:
        df: 含结构 SMILES 列的原始表（窄表或融合表）
        features: 需要输出的指数名；None → 输出全部
        structure_columns: 显式指定结构列（覆盖自动探测；全部按一组处理）

    Returns:
        DataFrame（若无结构列可解析，返回空 DataFrame）
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(index=getattr(df, "index", None))

    if structure_columns:
        cols = [c for c in structure_columns if c in df.columns]
        roles = {"custom": cols} if cols else {}
    else:
        roles = detect_structure_columns(df)
    if not roles:
        return pd.DataFrame(index=df.index)

    want = list(features) if features else None
    acc_num: Dict[str, np.ndarray] = {}
    acc_den = np.zeros(len(df), dtype=float)

    for role, cols in roles.items():
        w = _component_weights(df, role, cols)
        for j, col in enumerate(cols):
            for i, smi in enumerate(df[col].to_numpy()):
                d = molecule_physics_indices(smi)
                if not d:
                    continue
                ww = w[i, j]
                if ww <= 0:
                    continue
                for k, v in d.items():
                    if want is not None and k not in want:
                        continue
                    if not np.isfinite(v):
                        continue
                    acc_num.setdefault(k, np.zeros(len(df), dtype=float))[i] += v * ww
                acc_den[i] += ww

    if not acc_num:
        return pd.DataFrame(index=df.index)

    den = np.where(acc_den > 0, acc_den, np.nan)
    out = {}
    for k, arr in acc_num.items():
        out[f"{PHYSICS_PREFIX}{k}"] = arr / den
    res = pd.DataFrame(out, index=df.index)
    # 权重贡献过少的行置 NaN（避免用单个组分代表整配方）
    if len(res):
        arr = res.to_numpy(dtype=float, copy=True)
        arr[(acc_den <= 0.35)] = np.nan
        res = pd.DataFrame(arr, index=df.index, columns=res.columns)
    return res


def augment_polymer_physics(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    enabled: bool = True,
) -> pd.DataFrame:
    """
    向数据表追加物理指数列（幂等：已存在 phys_ 列时跳过）。

    这是 PINN 自动调用入口——用户无需手工计算或归一化。
    """
    if not enabled or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df
    if not detect_structure_columns(df):
        return df
    want = tuple(features) if features else DEFAULT_PHYSICS_FEATURES
    existing = [f"{PHYSICS_PREFIX}{f}" for f in want]
    if all(c in df.columns for c in existing):
        return df
    try:
        idx = compute_polymer_physics_indices(df, features=want)
    except Exception:
        return df
    if idx is None or len(idx.columns) == 0:
        return df
    out = df.copy()
    for c in idx.columns:
        out[c] = idx[c].to_numpy()
    return out


__all__ = [
    "PHYSICS_PREFIX",
    "DEFAULT_PHYSICS_FEATURES",
    "PHYSICS_FEATURE_LABELS",
    "ALL_PHYSICS_FEATURES",
    "clean_structure_string",
    "molecule_physics_indices",
    "detect_structure_columns",
    "compute_polymer_physics_indices",
    "augment_polymer_physics",
]
