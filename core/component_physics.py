# -*- coding: utf-8 -*-
"""
core/component_physics.py

逐组分物理量解析器：为窄表补齐各组分（树脂/固化剂）的
    分子量 MW (g/mol) / 当量重 EEW|AHEW (g/eq) / 官能度 f (eq/mol)

背景与设计约束
--------------
窄表 (ml_qspr_model_*.csv) 原本只有按组分角色聚合的列
（resin_epoxy_group_total / curing_agent_active_hydrogen_total），
**没有逐组分 MW**。这直接导致 core/crosslink_physics.compute_crosslink_features
的 hybrid 分支（依赖 resin_{i}_molecular_weight_g_mol 等）恒不可达，
f_avg 永远 NaN，理论 ν 全部退化为 aggregate 粗口径。

实测原始覆盖率（1552 行交联密度窄表）：
    resin_1_molecular_weight_g_mol       17.6%
    curing_agent_1_molecular_weight_g_mol 52.6%
    resin_1_equivalent_weight_g_eq        72.6%
    curing_agent_1_equivalent_weight_g_eq 66.0%

分层补齐策略（信任层级，高信任优先）
------------------------------------
    L1 文献值     宽表自带 *_molecular_weight_g_mol / *_equivalent_weight_g_eq
    L2 当量×官能度  MW = EEW × f（实测合法性：树脂侧 (MW/f)/EEW 中位 1.000、
                  86.5% 落在 0.9–1.1；固化剂侧中位 1.000、99.4% 落在 0.9–1.1）
    L3 结构直算    RDKit MolWt（仅普通 SMILES）

BigSMILES 特殊处理（关键正确性约束）
-----------------------------------
bigsmiles_to_smiles 返回的是**采样代理**，其重复单元数 n 与采样长度是人为
设定的，因此代理的 MolWt **没有物理意义**。实测证据：
    - 转换后 MW 与文献 EEW 的自洽率仅 3.5%（要求 0.9–1.1）
    - convMW / origMW 中位数 = 0.196（差 5 倍）
    - 同一分子量 465.12 的体系，代理给出 466.79 但 f 从 2 变成 3
但代理的**官能度 f 是可靠的**（单分子端基上的环氧/活性氢个数不随 n 改变）。
因此：BigSMILES 只采信 f，MW 必须走 L2 (EEW×f)；无文献 EEW 时标记代理不可信。

注意：core/smiles_utils.parse_chemical_string 在 keep_largest_frag=True 时
会把多片段合并成一个假分子（DGEBA+novolac 被并成单个连通的 7 环氧分子），
其 MolWt 同样不可用。本模块统一使用显式 Chem.MolFromSmiles 逐片段处理。

两套官能度口径（物理正确性关键）
--------------------------------
酸酐固化剂的 f 存在两种口径，**不可混用**：
    f_stoich  ：化学计量口径。1 个酸酐基团消耗 1 个环氧基 → f=1
    f_network ：网络支化口径。开环酯化后酸酐桥接 2 条链 → f_network=2
用 f_stoich=1 代入 Flory 的 (f-2) 项会得到负交联密度。
实测：ν_chem 口径全机制 Spearman +0.248，但酸酐子集仅 +0.112。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    RDKIT_AVAILABLE = False

try:
    from .smiles_utils import bigsmiles_to_smiles, detect_chem_string_format
except Exception:  # pragma: no cover
    bigsmiles_to_smiles = None

    def detect_chem_string_format(_s):  # type: ignore
        return "unknown"


# ---------------------------------------------------------------------------
# SMARTS
# ---------------------------------------------------------------------------
_EPOXIDE_PAT = "[C]1[O][C]1"
_PRI_AMINE_PAT = "[NX3;H2]"
_SEC_AMINE_PAT = "[NX3;H1]"
# 环状酸酐通用模式（兼容 PMDA/BTDA 型稠环芳酐的芳构化感知）
_ANHYDRIDE_PAT = "[o,OX2]1~[#6](=[OX1])~[#6]~[#6]~[#6](=[OX1])~1"
_THIOL_PAT = "[SX2;H1]"
_ISOCYANATE_PAT = "[NX2]=[CX2]=[OX1]"
_PHENOL_OH_PAT = "[OX2H]-c"
_ALCOHOL_OH_PAT = "[OX2H]-[CX4]"
_CARBOXYL_PAT = "[OX2H]-[CX3]=[OX1]"

#: 官能度物理上限（防御解析异常，如多片段堆叠产生的 f=25）
F_MAX = 8.0

#: MW 合理区间 (g/mol)。超出说明解析或单位有问题
MW_BOUNDS = (20.0, 20000.0)
#: 当量重合理区间 (g/eq)
EW_BOUNDS = (5.0, 5000.0)

#: MW 来源标记
SRC_LITERATURE = "literature"
SRC_EQUIV_FUNC = "equivalent_x_functionality"
SRC_STRUCTURE = "structure"
SRC_UNRESOLVED = "unresolved"

#: 固化剂机制
MECH_AMINE = "amine"
MECH_ANHYDRIDE = "anhydride"
MECH_THIOL = "thiol"
MECH_PHENOL = "phenol"
MECH_ISOCYANATE = "isocyanate"
MECH_CARBOXYL = "carboxyl"
MECH_ALCOHOL = "polyol"
MECH_UNKNOWN = "unknown"


def _smarts(pattern: str):
    if not RDKIT_AVAILABLE:
        return None
    return Chem.MolFromSmarts(pattern)


_PATTERNS: Dict[str, Any] = {}


def _pat(pattern: str):
    if pattern not in _PATTERNS:
        _PATTERNS[pattern] = _smarts(pattern)
    return _PATTERNS[pattern]


def _count(mol, pattern: str) -> int:
    p = _pat(pattern)
    if mol is None or p is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(p))
    except Exception:
        return 0


def _count_unique_center(mol, pattern: str) -> int:
    """以唯一中心原子计数（酸酐等对称环需防双向重复匹配）"""
    p = _pat(pattern)
    if mol is None or p is None:
        return 0
    try:
        return len({m[0] for m in mol.GetSubstructMatches(p)})
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 分子解析（显式逐片段，不合并）
# ---------------------------------------------------------------------------
def _clean(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s


def parse_mol(text: Any) -> Optional[Any]:
    """解析单个组分结构字符串。

    普通 SMILES 直接解析；BigSMILES 走 bigsmiles_to_smiles 转换链。
    **不做多片段合并**——合并会伪造出非物理的连通大分子。
    """
    if not RDKIT_AVAILABLE:
        return None
    s = _clean(text)
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return mol
    except Exception:
        pass
    if "{" in s and bigsmiles_to_smiles is not None:
        try:
            converted = bigsmiles_to_smiles(s)
            if converted:
                return Chem.MolFromSmiles(converted)
        except Exception:
            pass
    return None


def is_bigsmiles(text: Any) -> bool:
    s = _clean(text)
    if not s:
        return False
    if "{" in s or "}" in s:
        return True
    try:
        return detect_chem_string_format(s) == "bigsmiles"
    except Exception:
        return False


def has_multiple_fragments(text: Any) -> bool:
    """普通 SMILES 是否含多个片段（'.' 分隔）。

    多片段单元格（如某配方把 DGEBA 与 novolac 写在同格）无法得到单一 MW，
    其结构直算 MW 不可用。
    """
    s = _clean(text)
    if not s or is_bigsmiles(s):
        return False
    return len(s.split(".")) > 1


# ---------------------------------------------------------------------------
# 官能度
# ---------------------------------------------------------------------------
@dataclass
class Functionality:
    """双口径官能度"""
    f_stoich: float       # 化学计量口径（每分子消耗的环氧/活性氢当量数）
    f_network: float      # 网络支化口径（每分子引入的网络分支数）
    mechanism: str
    n_groups: int         # 检出的反应性基团数

    @property
    def usable_stoich(self) -> bool:
        return self.f_stoich > 0


def _as_mol(text: Any, mol: Optional[Any] = None) -> Optional[Any]:
    """解析入参：允许直接传入已解析的 RDKit Mol 对象。"""
    if mol is not None:
        return mol
    if RDKIT_AVAILABLE and isinstance(text, Chem.Mol):
        return text
    return parse_mol(text)


def resin_functionality(text: Any, mol: Optional[Any] = None) -> Functionality:
    """树脂环氧官能度。环氧基是双官能支化点，两口径一致。"""
    mol = _as_mol(text, mol)
    n = _count(mol, _EPOXIDE_PAT)
    if n <= 0:
        return Functionality(0.0, 0.0, MECH_UNKNOWN, 0)
    f = float(min(n, F_MAX))
    return Functionality(f, f, MECH_AMINE, n)


def curer_functionality(text: Any, mol: Optional[Any] = None) -> Functionality:
    """固化剂官能度（双口径）。

    f_stoich  : 每分子消耗的环氧当量数
    f_network : 每分子在网络中形成的分支数

    胺     ：伯胺 2 氢 → 消耗 2 环氧，并与 2 条链成键 → 两口径都是 2
    酸酐   ：1 酸酐消耗 1 环氧(f_stoich=1)，开环酯化后桥接 2 条链(f_network=2)
    硫醇   ：1:1 消耗，形成 1 个硫醚桥 → 两口径都是 1
    酚羟基 ：1:1 消耗，形成 1 个醚桥 → 两口径都是 1
    异氰酸酯：与羟基/胺 1:1 → 两口径都是 1
    羧基   ：1:1 消耗 → 两口径都是 1
    醇羟基 ：1:1（与 NCO 或醚化）→ 两口径都是 1
    """
    mol = _as_mol(text, mol)
    if mol is None:
        return Functionality(0.0, 0.0, MECH_UNKNOWN, 0)

    n_nco = _count(mol, _ISOCYANATE_PAT)
    if n_nco > 0:
        f = float(min(n_nco, F_MAX))
        return Functionality(f, f, MECH_ISOCYANATE, n_nco)

    n_pri = _count(mol, _PRI_AMINE_PAT)
    n_sec = _count(mol, _SEC_AMINE_PAT)
    if n_pri + n_sec > 0:
        # 伯胺贡献 2 个活性氢；仲胺 1 个
        n_h = n_pri * 2 + n_sec
        f = float(min(n_h, F_MAX))
        return Functionality(f, f, MECH_AMINE, n_h)

    n_anh = _count_unique_center(mol, _ANHYDRIDE_PAT)
    if n_anh > 0:
        # 关键：两套口径分离
        f_st = float(min(n_anh, F_MAX))            # 1 酸酐 : 1 环氧
        f_net = float(min(2.0 * n_anh, F_MAX))     # 开环后桥接 2 条链
        return Functionality(f_st, f_net, MECH_ANHYDRIDE, n_anh)

    n_sh = _count(mol, _THIOL_PAT)
    if n_sh > 0:
        f = float(min(n_sh, F_MAX))
        return Functionality(f, f, MECH_THIOL, n_sh)

    n_ph = _count(mol, _PHENOL_OH_PAT)
    if n_ph > 0:
        f = float(min(n_ph, F_MAX))
        return Functionality(f, f, MECH_PHENOL, n_ph)

    n_cooh = _count(mol, _CARBOXYL_PAT)
    if n_cooh > 0:
        f = float(min(n_cooh, F_MAX))
        return Functionality(f, f, MECH_CARBOXYL, n_cooh)

    n_oh = _count(mol, _ALCOHOL_OH_PAT)
    if n_oh > 0:
        f = float(min(n_oh, F_MAX))
        return Functionality(f, f, MECH_ALCOHOL, n_oh)

    return Functionality(0.0, 0.0, MECH_UNKNOWN, 0)


# ---------------------------------------------------------------------------
# 单组分物理量（分层补齐）
# ---------------------------------------------------------------------------
@dataclass
class ComponentPhysics:
    mw: float = np.nan
    equivalent_weight: float = np.nan     # EEW (树脂) 或 AHEW (固化剂)
    f_stoich: float = 0.0
    f_network: float = 0.0
    mechanism: str = MECH_UNKNOWN
    mw_source: str = SRC_UNRESOLVED
    ew_source: str = SRC_UNRESOLVED
    f_source: str = SRC_UNRESOLVED
    bigsmiles_proxy: bool = False          # 结构来自 BigSMILES 采样代理
    multi_fragment: bool = False

    @property
    def resolved(self) -> bool:
        return np.isfinite(self.mw) and self.mw > 0

    @property
    def trust(self) -> str:
        """MW 可信度标记，供下游决定是否使用。"""
        if self.mw_source == SRC_LITERATURE:
            return "high"
        if self.mw_source == SRC_EQUIV_FUNC:
            return "high" if self.ew_source == SRC_LITERATURE else "medium"
        if self.mw_source == SRC_STRUCTURE:
            return "medium"
        return "none"


def _num(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return np.nan


def resolve_component(
    structure: Any,
    *,
    is_resin: bool,
    mw_literature: Any = None,
    ew_literature: Any = None,
    f_literature: Any = None,
) -> ComponentPhysics:
    """按信任层级解析单组分物理量。

    层级：L1 文献 MW → L2 当量×官能度 → L3 结构直算（仅普通 SMILES）
    """
    out = ComponentPhysics()
    text = _clean(structure)
    if not text:
        return out

    out.bigsmiles_proxy = is_bigsmiles(text)
    out.multi_fragment = has_multiple_fragments(text)

    mol = parse_mol(text)
    func = (
        resin_functionality(text, mol=mol)
        if is_resin
        else curer_functionality(text, mol=mol)
    )
    if func.f_stoich <= 0 and f_literature is not None:
        fl = _num(f_literature)
        if np.isfinite(fl):
            func = Functionality(fl, fl, MECH_UNKNOWN, int(fl))
            out.f_source = SRC_LITERATURE
    out.f_stoich = func.f_stoich
    out.f_network = func.f_network
    out.mechanism = func.mechanism
    if out.f_source == SRC_UNRESOLVED and func.f_stoich > 0:
        out.f_source = SRC_STRUCTURE

    mw_lit = _num(mw_literature)
    ew_lit = _num(ew_literature)
    if np.isfinite(mw_lit) and not (MW_BOUNDS[0] <= mw_lit <= MW_BOUNDS[1]):
        mw_lit = np.nan
    if np.isfinite(ew_lit) and not (EW_BOUNDS[0] <= ew_lit <= EW_BOUNDS[1]):
        ew_lit = np.nan

    # ---- L1 文献分子量 ----
    if np.isfinite(mw_lit):
        out.mw = mw_lit
        out.mw_source = SRC_LITERATURE
    # ---- L2 当量 × 官能度 ----
    elif np.isfinite(ew_lit) and func.f_stoich > 0:
        out.mw = ew_lit * func.f_stoich
        out.mw_source = SRC_EQUIV_FUNC
    # ---- L3 结构直算（BigSMILES 代理与多片段排除） ----
    elif mol is not None and not out.bigsmiles_proxy and not out.multi_fragment:
        try:
            mw = float(Descriptors.MolWt(mol))
            if MW_BOUNDS[0] <= mw <= MW_BOUNDS[1]:
                out.mw = mw
                out.mw_source = SRC_STRUCTURE
        except Exception:
            pass

    # ---- 当量重：文献优先，否则由 MW 与 f_stoich 反推 ----
    if np.isfinite(ew_lit):
        out.equivalent_weight = ew_lit
        out.ew_source = SRC_LITERATURE
    elif np.isfinite(out.mw) and func.f_stoich > 0:
        out.equivalent_weight = out.mw / func.f_stoich
        out.ew_source = SRC_EQUIV_FUNC if out.mw_source != SRC_UNRESOLVED else SRC_UNRESOLVED
        if np.isfinite(out.equivalent_weight) and not (
            EW_BOUNDS[0] <= out.equivalent_weight <= EW_BOUNDS[1]
        ):
            out.equivalent_weight = np.nan
            out.ew_source = SRC_UNRESOLVED

    return out


# ---------------------------------------------------------------------------
# 配方级：逐组分解析 + 摩尔加权
# ---------------------------------------------------------------------------
#: 逐组分列的候选命名（宽表/窄表可能用不同前缀）
_MW_SUFFIX = "molecular_weight_g_mol"
_EW_SUFFIX = "equivalent_weight_g_eq"
_PHR_SUFFIX = "amount_phr"
_EPOXY_F_SUFFIX = "epoxy_group_count"
_ACTIVE_H_SUFFIX = "active_hydrogen_equivalent_count"
_EQUIV_F_SUFFIX = "equivalent_group_count"


def _pick(df: pd.DataFrame, names: Tuple[str, ...]) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


#: 窄表原有的配方级当量重列（这些是实测/文献值，反映真实低聚物分布，优先级高于结构直算）
EXISTING_EEW_COLS = (
    "formulation_resin_total_eew_g_eq",
    "formulation_resin_eew_g_eq",
    "resin_total_eew_g_eq",
)
EXISTING_AHEW_COLS = (
    "formulation_hardener_total_ahew_g_eq",
    "formulation_hardener_ahew_g_eq",
    "curing_agent_total_ahew_g_eq",
)


def _existing_series(df: pd.DataFrame, names: Tuple[str, ...]) -> pd.Series:
    """取第一列存在的配方级当量重（不合并多列，避免语义混淆）。"""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _col(df: pd.DataFrame, side: str, idx: int, suffix: str) -> Optional[pd.Series]:
    """取逐组分列，兼容 role_1_x / role_x 两种命名。"""
    return _pick(df, (f"{side}_{idx}_{suffix}", f"{side}_{suffix}_{idx}"))


def _func_col(df: pd.DataFrame, side: str, idx: int, is_resin: bool) -> Optional[pd.Series]:
    if is_resin:
        return _pick(df, (
            f"{side}_{idx}_{_EPOXY_F_SUFFIX}",
            f"{side}_{idx}_{_EQUIV_F_SUFFIX}",
        ))
    return _pick(df, (
        f"{side}_{idx}_{_ACTIVE_H_SUFFIX}",
        f"{side}_{idx}_{_EQUIV_F_SUFFIX}",
        f"{side}_{idx}_{_EPOXY_F_SUFFIX}",
    ))


@dataclass
class FormulationPhysics:
    """配方级逐组分物理量 + 摩尔加权汇总（ν 计算所需的一切）"""
    frame: pd.DataFrame = None            # 逐组分明细（每行一个配方）

    def __post_init__(self):
        if self.frame is None:
            self.frame = pd.DataFrame()


def compute_component_physics(
    df: pd.DataFrame,
    *,
    max_components: int = 3,
    resin_prefix: str = "resin",
    curer_prefix: str = "curing_agent",
) -> pd.DataFrame:
    """逐行解析各组分物理量，返回与 df 同索引的明细表。

    输出列（每组分 i = 1..max_components）：
        {prefix}_{i}_structure_used    解析所用结构（原始）
        {prefix}_{i}_mw_resolved       补齐后 MW (g/mol)
        {prefix}_{i}_ew_resolved       补齐后 EEW/AHEW (g/eq)
        {prefix}_{i}_f_stoich          化学计量官能度
        {prefix}_{i}_f_network         网络支化官能度
        {prefix}_{i}_mechanism         固化剂机制（树脂侧为空）
        {prefix}_{i}_mw_source         literature|equivalent_x_functionality|structure|unresolved
        {prefix}_{i}_mw_trust          high|medium|none
        {prefix}_{i}_bigsmiles_proxy   是否 BigSMILES 采样代理
        {prefix}_{i}_multi_fragment    是否多片段单元格
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(index=getattr(df, "index", None))

    out: Dict[str, Any] = {}
    n = len(df)

    for side, is_resin in ((resin_prefix, True), (curer_prefix, False)):
        for i in range(1, max_components + 1):
            struct_col = _pick(df, (f"{side}_{i}_structure", f"{side}_{i}_smiles"))
            if struct_col is None:
                continue
            mw_col = _col(df, side, i, _MW_SUFFIX)
            ew_col = _col(df, side, i, _EW_SUFFIX)
            f_col = _func_col(df, side, i, is_resin)

            structs = struct_col if isinstance(struct_col, pd.Series) else pd.Series(struct_col, index=df.index)
            mws = mw_col if isinstance(mw_col, pd.Series) else pd.Series(np.nan, index=df.index)
            ews = ew_col if isinstance(ew_col, pd.Series) else pd.Series(np.nan, index=df.index)
            fs = f_col if isinstance(f_col, pd.Series) else pd.Series(np.nan, index=df.index)

            rows = [
                resolve_component(
                    structs.iloc[k],
                    is_resin=is_resin,
                    mw_literature=mws.iloc[k],
                    ew_literature=ews.iloc[k],
                    f_literature=fs.iloc[k],
                )
                for k in range(n)
            ]

            p = f"{side}_{i}_"
            out[p + "mw_resolved"] = np.array([r.mw for r in rows], dtype=float)
            out[p + "ew_resolved"] = np.array([r.equivalent_weight for r in rows], dtype=float)
            out[p + "f_stoich"] = np.array([r.f_stoich for r in rows], dtype=float)
            out[p + "f_network"] = np.array([r.f_network for r in rows], dtype=float)
            out[p + "mechanism"] = np.array([r.mechanism for r in rows], dtype=object)
            out[p + "mw_source"] = np.array([r.mw_source for r in rows], dtype=object)
            out[p + "ew_source"] = np.array([r.ew_source for r in rows], dtype=object)
            out[p + "f_source"] = np.array([r.f_source for r in rows], dtype=object)
            out[p + "mw_trust"] = np.array([r.trust for r in rows], dtype=object)
            out[p + "bigsmiles_proxy"] = np.array([r.bigsmiles_proxy for r in rows], dtype=bool)
            out[p + "multi_fragment"] = np.array([r.multi_fragment for r in rows], dtype=bool)
            out[p + "has_structure"] = np.array([bool(_clean(s)) for s in structs], dtype=bool)

    res = pd.DataFrame(out, index=df.index)
    return res


def _mole_weighted(
    comp: pd.DataFrame,
    df: pd.DataFrame,
    side: str,
    idx_list: List[int],
    *,
    use_network_functionality: bool,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """摩尔分数加权平均官能度。

    Flory 的 f_avg 是**摩尔加权**（不是质量/phr 加权）。
    返回 (f_avg, mol_total, covered_mask)

    每组分摩尔数 = phr / MW（有 phr 时），否则等摩尔权重 1/MW。
    """
    n = len(df)
    mol_sum = np.zeros(n)
    fmol_sum = np.zeros(n)
    covered = np.zeros(n, dtype=bool)

    for i in idx_list:
        p = f"{side}_{i}_"
        if p + "mw_resolved" not in comp.columns:
            continue
        mw = pd.to_numeric(comp[p + "mw_resolved"], errors="coerce").to_numpy(dtype=float)
        fkey = "f_network" if use_network_functionality else "f_stoich"
        f = pd.to_numeric(comp[p + fkey], errors="coerce").to_numpy(dtype=float)
        has = comp[p + "has_structure"].to_numpy(dtype=bool)

        phr_col = _col(df, side, i, _PHR_SUFFIX)
        if phr_col is not None:
            phr = pd.to_numeric(phr_col, errors="coerce").to_numpy(dtype=float)
            phr = np.where(np.isfinite(phr) & (phr > 0), phr, np.nan)
        else:
            phr = np.full(n, np.nan)

        ok = has & np.isfinite(mw) & (mw > 0) & np.isfinite(f) & (f > 0)
        if not ok.any():
            continue
        # 无 phr 时退化为等摩尔（用 1 作分子数权重）
        mol = np.where(ok, np.where(np.isfinite(phr), phr / mw, 1.0 / mw), 0.0)
        mol_sum += mol
        fmol_sum += np.where(ok, mol * f, 0.0)
        covered |= ok

    with np.errstate(divide="ignore", invalid="ignore"):
        f_avg = np.where(mol_sum > 0, fmol_sum / mol_sum, np.nan)
    return (
        pd.Series(f_avg, index=df.index),
        pd.Series(mol_sum, index=df.index),
        pd.Series(covered, index=df.index),
    )


def compute_formulation_summary(
    df: pd.DataFrame,
    comp: Optional[pd.DataFrame] = None,
    *,
    max_components: int = 3,
    rho_g_cm3: float = 1.2,
) -> pd.DataFrame:
    """在逐组分明细之上计算配方级汇总量（供 ν 公式消费）。

    输出列（前缀 cp_）：
        cp_r_value            化学计量比 r（文献列优先，否则当量比推算）
        cp_balance            min(r, 1/r)
        cp_f_r                树脂摩尔加权官能度（网络口径）
        cp_f_h_stoich         固化剂摩尔加权官能度（化学计量口径）
        cp_f_h_network        固化剂摩尔加权官能度（网络支化口径）
        cp_f_avg_network      网络支化口径平均官能度
        cp_W_g_per_epoxy      每 mol 环氧基的配方质量 (g)  = EEW + r·AHEW
        cp_eew / cp_ahew      配方有效当量重
        cp_epoxy_conc_mol_m3  环氧基摩尔浓度 (mol/m³) = ρ/W
        cp_dilution           树脂占粘料比例（活性稀释剂折算）
        cp_mechanism          主固化机制
        cp_coverage           配方物理量覆盖等级
    """
    if comp is None:
        comp = compute_component_physics(df, max_components=max_components)
    out = pd.DataFrame(index=df.index)
    n = len(df)
    RHO = float(rho_g_cm3) * 1.0e6  # g/m³

    res_idx = [i for i in range(1, max_components + 1) if f"resin_{i}_mw_resolved" in comp.columns]
    cur_idx = [i for i in range(1, max_components + 1) if f"curing_agent_{i}_mw_resolved" in comp.columns]

    f_r, mol_r, cov_r = _mole_weighted(comp, df, "resin", res_idx, use_network_functionality=True)
    f_h_st, mol_h, cov_h = _mole_weighted(comp, df, "curing_agent", cur_idx, use_network_functionality=False)
    f_h_net, _, _ = _mole_weighted(comp, df, "curing_agent", cur_idx, use_network_functionality=True)

    # ---- 配方有效当量重（质量加权调和平均，等效于 phr 加权） ----
    def _eff_ew(side: str, idx_list: List[int]) -> pd.Series:
        num = np.zeros(n)
        den = np.zeros(n)
        for i in idx_list:
            p = f"{side}_{i}_"
            if p + "ew_resolved" not in comp.columns:
                continue
            ew = pd.to_numeric(comp[p + "ew_resolved"], errors="coerce").to_numpy(dtype=float)
            has = comp[p + "has_structure"].to_numpy(dtype=bool)
            phr_col = _col(df, side, i, _PHR_SUFFIX)
            if phr_col is not None:
                phr = pd.to_numeric(phr_col, errors="coerce").to_numpy(dtype=float)
            else:
                phr = np.full(n, np.nan)
            w = np.where(np.isfinite(phr) & (phr > 0), phr, 1.0)
            ok = has & np.isfinite(ew) & (ew > 0)
            num += np.where(ok, w, 0.0)
            den += np.where(ok, w / np.where(ok, ew, 1.0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)

    eew = _eff_ew("resin", res_idx)
    ahew = _eff_ew("curing_agent", cur_idx)

    # ---- 化学计量比 r ----
    r_val = pd.Series(np.nan, index=df.index, dtype=float)
    for c in (
        "formulation_r_value",
        "formulation_resin_hardener_equivalent_ratio",
        "crosslink_stoichiometry_r",
        "stoichiometric_ratio_r_cleaned",
        "r_value",
    ):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            v = v.where(np.isfinite(v) & (v > 0))
            r_val = r_val.fillna(v)
    # 当量比推算兜底
    #
    # ⚠️⚠️ 语义警示（请勿在别处复用本兜底口径）⚠️⚠️
    # 下面的 derived = ahew / eew 是**当量重比**，不是环氧/固化剂**化学计量比 r**。
    # 两者数值与含义均不同，**不可互换**：
    #   - 正确 r = (固化剂phr / AHEW) / (树脂phr / EEW)
    #   - 本兜底  = AHEW / EEW
    # 实测（ml_qspr_selected.csv, n=3237）：
    #   - 正确公式与全表 formulation_r_value 相关系数 0.9749，中位绝对误差 0.00017
    #   - 本兜底口径与全表 formulation_r_value 相关系数仅 -0.0886
    # 具体例：DGEBA/DDS 100:33 时正确 r≈0.905，而本兜底≈0.365。
    #
    # 因此：**禁止**用 cp_r_value 填充 formulation_r_value。
    # 门户侧请用 core/portal_formulation_inputs.py 的 derive_r_value()。
    # 本兜底仅在原表 r 列全缺失时作为内部临时估值使用（受下方 clip 约束），
    # 且已有 cp_r_value 与 formulation_r_value 去重逻辑（见本文件末尾）。
    derived = ahew / eew
    derived = derived.where(np.isfinite(derived) & (derived > 0))
    r_val = r_val.fillna(derived).clip(0.05, 20.0)

    balance = np.minimum(r_val, 1.0 / r_val)

    # ---- 每 mol 环氧基的配方质量 ----
    W = eew + r_val * ahew
    epoxy_conc = pd.Series(np.where(np.isfinite(W) & (W > 0), RHO / W, np.nan), index=df.index)

    # ---- 稀释度：树脂占粘料总质量比 ----
    resin_phr = _pick(df, ("resin_total_phr",))
    binder_phr = _pick(df, ("formulation_epoxy_binder_total_phr",))
    if resin_phr is not None and binder_phr is not None:
        a = pd.to_numeric(resin_phr, errors="coerce")
        b = pd.to_numeric(binder_phr, errors="coerce")
        dilution = (a / b.replace(0.0, np.nan)).clip(0.2, 1.0).fillna(1.0)
    else:
        dilution = pd.Series(1.0, index=df.index)

    # ---- 主机制 ----
    mech = pd.Series(MECH_UNKNOWN, index=df.index, dtype=object)
    for i in cur_idx:
        p = f"curing_agent_{i}_"
        if p + "mechanism" in comp.columns:
            m = comp[p + "mechanism"].astype(str)
            m = m.where(m != MECH_UNKNOWN)
            mech = mech.where(mech != MECH_UNKNOWN, m)
    mech = mech.fillna(MECH_UNKNOWN)

    f_avg_network = (f_r + r_val * f_h_net) / (1.0 + r_val)
    f_avg_network = f_avg_network.where(np.isfinite(f_avg_network))

    out["cp_r_value"] = r_val
    out["cp_balance"] = balance
    out["cp_f_r"] = f_r
    out["cp_f_h_stoich"] = f_h_st
    out["cp_f_h_network"] = f_h_net
    out["cp_f_avg_network"] = f_avg_network
    out["cp_W_g_per_epoxy"] = W

    # ---- 配方级当量重：原表实测值优先，缺失处用补齐值**原位填补** ----
    #
    # [重要] 不另起 cp_eew/cp_ahew 列，而是写回原表列名。
    # 理由：cp_eew 与原表 formulation_resin_total_eew_g_eq 在 756 条重叠样本上
    # **100% 相同**，两列共存 = 完全重复的特征（用户明确要求避免）。
    #
    # 为什么“原表优先”而不是“补齐覆盖”：
    # 实测（n=352 两套都有值的受控子集，仅替换 EEW/AHEW，目标=实测 ν）：
    #     原表 formulation_resin_total_eew_g_eq   spearman=+0.236
    #     补齐（结构 MW/f）                      spearman=+0.080
    # 原因：同一 DGEBA 结构在原表中有 105 个不同 EEW（134~288，中位 196），
    # 这是实测/文献值，反映真实低聚物分布（n=0~0.15 同系物混合物）；
    # 而结构直算恒为单体值 170.21，丢失了低聚物分布信息。
    #
    # 但原表覆盖仅 25%（EEW）/ 21%（AHEW），缺失时补齐值本身有效
    # （仅补齐可用的 535 条样本 spearman=+0.243）。
    # 因此：原表值优先，仅填补空位 → 覆盖提升到 94%/86%，且不降低质量。
    _eew_existing = _existing_series(df, EXISTING_EEW_COLS)
    _ahew_existing = _existing_series(df, EXISTING_AHEW_COLS)

    out["cp_eew"] = _eew_existing.where(_eew_existing.notna(), eew)
    out["cp_ahew"] = _ahew_existing.where(_ahew_existing.notna(), ahew)
    out["cp_eew_source"] = np.where(
        _eew_existing.notna(), SRC_LITERATURE,
        np.where(np.isfinite(eew), SRC_EQUIV_FUNC, SRC_UNRESOLVED),
    )
    out["cp_ahew_source"] = np.where(
        _ahew_existing.notna(), SRC_LITERATURE,
        np.where(np.isfinite(ahew), SRC_EQUIV_FUNC, SRC_UNRESOLVED),
    )
    # 用“原表优先”的口径重算 W 与环氧浓度，避免下游拿到两套不一致的值
    W = out["cp_eew"] + r_val * out["cp_ahew"]
    W = W.where(np.isfinite(W) & (W > 0))
    out["cp_W_g_per_epoxy"] = W
    epoxy_conc = pd.Series(np.where(np.isfinite(W) & (W > 0), RHO / W, np.nan), index=df.index)
    out["cp_epoxy_conc_mol_m3"] = epoxy_conc
    out["cp_dilution"] = dilution
    out["cp_mechanism"] = mech
    out["cp_mol_resin"] = mol_r
    out["cp_mol_curer"] = mol_h

    # ---- 覆盖等级 ----
    cov = np.where(
        cov_r & cov_h & np.isfinite(W) & (W > 0), "full",
        np.where(np.isfinite(epoxy_conc), "partial", "none"),
    )
    out["cp_coverage"] = cov
    return out


#: 不进表的元数据列：这些是来源/机制/覆盖度标记（字符串），
#: 只用于内部降级决策与诊断，不应作为特征写入数据集。
#: 保留在 DataFrame 里会让它们进入特征白名单，污染训练矩阵。
_METADATA_SUFFIXES = (
    "_mw_source", "_ew_source", "_f_source", "_mw_trust",
    "_mechanism",
)
_METADATA_EXACT = ("cp_mechanism", "cp_coverage", "cp_eew_source", "cp_ahew_source")


def is_metadata_column(name: str) -> bool:
    """是否为“只供内部使用、不应写入数据集”的元数据列。"""
    n = str(name)
    if n in _METADATA_EXACT:
        return True
    if n.endswith("_source"):
        return True
    return any(n.endswith(s) for s in _METADATA_SUFFIXES)


def enrich_narrow_table(
    df: pd.DataFrame,
    *,
    max_components: int = 3,
    rho_g_cm3: float = 1.2,
    keep_metadata: bool = False,
) -> pd.DataFrame:
    """窄表增强入口：逐组分物理量 + 配方级汇总，一次补齐。

    返回 df 的副本，追加 cp_ / 逐组分 *_resolved 列。

    参数：
        keep_metadata: 是否保留字符串型元数据列（*_mw_source / *_mechanism /
            cp_coverage 等）。**默认 False**——这些列只用于内部降级决策与
            诊断，不应作为特征写入数据集；需要调试时可置 True。
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df.copy()
    comp = compute_component_physics(df, max_components=max_components)
    summary = compute_formulation_summary(df, comp, max_components=max_components, rho_g_cm3=rho_g_cm3)
    out = pd.concat([df.copy(), comp, summary], axis=1)

    # ---- 当量重去重：把填补结果写回**原表列名**，删掉重复的 cp_ 列 ----
    #
    # cp_eew 与原表 formulation_resin_total_eew_g_eq 在重叠样本上 100% 相同，
    # 两列共存就是完全重复的特征。这里保留原列名（下游 _EEW_COL / 特征白名单
    # / auto_feature_resolver 都按原名引用），只把空位用补齐值填上。
    for _orig_names, _cp_col in ((EXISTING_EEW_COLS, "cp_eew"), (EXISTING_AHEW_COLS, "cp_ahew")):
        if _cp_col not in out.columns:
            continue
        _filled = pd.to_numeric(out[_cp_col], errors="coerce")
        _target = next((c for c in _orig_names if c in out.columns), None)
        if _target is None:
            # 原表完全没有该列：保留 cp_ 列作为唯一载体，改名为规范原名
            out = out.rename(columns={_cp_col: _orig_names[0]})
        else:
            _old = pd.to_numeric(out[_target], errors="coerce")
            out[_target] = _old.where(_old.notna(), _filled)
            out = out.drop(columns=[_cp_col])

    # ---- r 值去重：cp_r_value 与 formulation_r_value 在 958 条重叠样本上 100% 相同 ----
    _R_ORIG_NAMES = (
        "formulation_r_value",
        "formulation_resin_hardener_equivalent_ratio",
        "crosslink_stoichiometry_r",
        "stoichiometric_ratio_r_cleaned",
    )
    if "cp_r_value" in out.columns:
        _r_filled = pd.to_numeric(out["cp_r_value"], errors="coerce")
        _r_target = next((c for c in _R_ORIG_NAMES if c in out.columns), None)
        if _r_target is None:
            out = out.rename(columns={"cp_r_value": _R_ORIG_NAMES[0]})
        else:
            _r_old = pd.to_numeric(out[_r_target], errors="coerce")
            out[_r_target] = _r_old.where(_r_old.notna(), _r_filled)
            out = out.drop(columns=["cp_r_value"])

    if not keep_metadata:
        drop = [c for c in out.columns if is_metadata_column(c) and c not in df.columns]
        if drop:
            out = out.drop(columns=drop)
    return out
