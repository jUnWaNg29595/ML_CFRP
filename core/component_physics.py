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
# 统一从 reaction_core 导入 SMARTS 与规范物理参数
# ---------------------------------------------------------------------------
from .reaction_core import (
    EPOXIDE_PAT as _EPOXIDE_PAT,
    PRI_AMINE_PAT as _PRI_AMINE_PAT,
    SEC_AMINE_PAT as _SEC_AMINE_PAT,
    ANHYDRIDE_PAT as _ANHYDRIDE_PAT,
    THIOL_PAT as _THIOL_PAT,
    ISOCYANATE_PAT as _ISOCYANATE_PAT,
    PHENOL_OH_PAT as _PHENOL_OH_PAT,
    ALCOHOL_OH_PAT as _ALCOHOL_OH_PAT,
    CARBOXYL_PAT as _CARBOXYL_PAT,
    inspect_resin_functionality,
    inspect_curer_functionality,
    count_substruct_matches as _count,
    count_unique_center_matches as _count_unique_center,
)
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

#: 当量比来源常量
R_SRC_LITERATURE = "literature"
R_SRC_CALCULATED = "calculated_from_mass"
R_SRC_ASSUMED_STOICH = "assumed_stoichiometric"
R_SRC_MISSING_STOICH = "missing_stoichiometry"
R_SRC_INVALID_CHEM = "invalid_chemistry"




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
    """树脂环氧官能度（统一对接 reaction_core）。"""
    target = _as_mol(text, mol)
    info = inspect_resin_functionality(target)
    return Functionality(info.f_stoich, info.f_network, info.mechanism, info.n_groups)


def curer_functionality(text: Any, mol: Optional[Any] = None) -> Functionality:
    """固化剂官能度（双口径，统一对接 reaction_core）。"""
    target = _as_mol(text, mol)
    info = inspect_curer_functionality(target)
    return Functionality(info.f_stoich, info.f_network, info.mechanism, info.n_groups)


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
    structure_functionality_conflict: bool = False  # 结构解析活性基团与外部声明官能度冲突 (P0-2)

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
        if np.isfinite(fl) and fl > 0:
            if mol is not None and not out.bigsmiles_proxy and not out.multi_fragment:
                # 结构成功解析但毫无对应环氧/固化位点，而外部声明存在正官能度 (P0-2)
                out.structure_functionality_conflict = True
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
    res: Optional[pd.Series] = None
    for n in names:
        if n in df.columns:
            s = df[n].copy()
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                cleaned = s.astype(str).str.strip().str.lower()
                s = s.where(~cleaned.isin(["", "nan", "none", "null", "<na>"]), np.nan)
            if res is None:
                res = s
            else:
                res = res.combine_first(s)
    return res


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


def _col(df: pd.DataFrame, side: str, idx: int, suffix: Union[str, Tuple[str, ...], List[str]]) -> Optional[pd.Series]:
    """取逐组分列，兼容 role_1_x / role_x 两种命名及 curing_agent / hardener 前缀别名。"""
    prefixes = [side]
    if side in ("curing_agent", "curingagent"):
        prefixes.extend(["hardener", "curer"])
    elif side in ("hardener", "curer"):
        prefixes.extend(["curing_agent", "curingagent"])
    elif side == "resin":
        prefixes.extend(["epoxy"])

    suffixes = [suffix] if isinstance(suffix, str) else list(suffix)

    candidates: List[str] = []
    for p in prefixes:
        for s_raw in suffixes:
            s = s_raw.lstrip("_")
            candidates.extend([f"{p}_{idx}_{s}", f"{p}_{s}_{idx}"])
            if idx == 1:
                candidates.append(f"{p}_{s}")
            if s in (_PHR_SUFFIX, "phr", "amount"):
                candidates.extend([
                    f"{p}_{idx}_phr",
                    f"{p}_{idx}_weight_phr",
                    f"{p}_{idx}_amount",
                    f"{p}_{idx}_mass",
                    f"{p}_{idx}_mass_g",
                    f"{p}_{idx}_weight",
                    f"{p}_{idx}_wt_ratio",
                ])
                if idx == 1:
                    candidates.extend([
                        f"{p}_phr",
                        f"{p}_amount",
                        f"{p}_mass",
                        f"{p}_weight",
                    ])
            elif s in ("structure", "smiles", "name"):
                candidates.extend([
                    f"{p}_{idx}_structure",
                    f"{p}_{idx}_smiles",
                    f"{p}_{idx}_name",
                    f"{p}_{idx}_chemical_name",
                ])
                if idx == 1:
                    candidates.extend([
                        f"{p}_structure",
                        f"{p}_smiles",
                        f"{p}_name",
                    ])
    return _pick(df, tuple(candidates))


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
    assume_stoichiometric_if_missing: bool = True,
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
            prefixes = [side]
            if side in ("curing_agent", "curingagent"):
                prefixes.extend(["hardener", "curer"])
            elif side in ("hardener", "curer"):
                prefixes.extend(["curing_agent", "curingagent"])
            elif side == "resin":
                prefixes.extend(["epoxy"])

            cand = []
            for p in prefixes:
                cand.extend([f"{p}_{i}_structure", f"{p}_{i}_smiles", f"{p}_structure_{i}", f"{p}_smiles_{i}"])
                if i == 1:
                    cand.extend([f"{p}_structure", f"{p}_smiles"])
            struct_col = _pick(df, tuple(cand))
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
            out[p + "struct_func_conflict"] = np.array([r.structure_functionality_conflict for r in rows], dtype=bool)
            out[p + "has_structure"] = np.array([bool(_clean(s)) for s in structs], dtype=bool)

    res = pd.DataFrame(out, index=df.index)
    try:
        summary = compute_formulation_summary(
            df,
            res,
            max_components=max_components,
            assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
        )
        cols_to_add = [c for c in summary.columns if c not in res.columns]
        if cols_to_add:
            res = pd.concat([res, summary[cols_to_add]], axis=1)
    except Exception:
        pass
    # 只记录源输入证据；不影响任何物理公式或来源优先级。
    from .cp_cache_integrity import stamp_cp_cache
    res = stamp_cp_cache(
        res, df, max_components=max_components,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    )
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
    assume_stoichiometric_if_missing: bool = True,
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

    # ---- 配方有效当量重与配方总当量数 ----
    def _eff_ew_and_equivs(side: str, idx_list: List[int]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """返回 (eff_ew, total_phr, total_equivs, has_all_real_mass, total_input_phr, unresolved_count, ignored_phr, unresolved_missing_m)"""
        num = np.zeros(n)
        den = np.zeros(n)
        active_comps = np.zeros(n, dtype=int)
        mass_comps = np.zeros(n, dtype=int)
        total_input_phr = np.zeros(n)
        unresolved_count = np.zeros(n, dtype=int)
        ignored_phr = np.zeros(n)
        unresolved_missing_m = np.zeros(n, dtype=int)
        for i in idx_list:
            p = f"{side}_{i}_"
            phr_col = _col(df, side, i, _PHR_SUFFIX)
            if phr_col is not None:
                phr = pd.to_numeric(phr_col, errors="coerce").to_numpy(dtype=float)
            else:
                phr = np.full(n, np.nan)
            valid_m = np.isfinite(phr) & (phr > 0)

            # 检查是否有结构输入 (P0-3)
            smi_series = _col(df, side, i, ("structure", "smiles", "name"))
            has_smi_entry = (
                smi_series.notna()
                & (smi_series.astype(str).str.strip().ne(""))
                & (smi_series.astype(str).str.lower().ne("none"))
                & (smi_series.astype(str).str.lower().ne("nan"))
            ) if smi_series is not None else np.zeros(n, dtype=bool)

            ew = pd.to_numeric(comp[p + "ew_resolved"], errors="coerce").to_numpy(dtype=float) if (p + "ew_resolved" in comp.columns) else np.full(n, np.nan)
            has = comp[p + "has_structure"].to_numpy(dtype=bool) if (p + "has_structure" in comp.columns) else np.zeros(n, dtype=bool)

            ok = has & np.isfinite(ew) & (ew > 0)
            active_comps += ok.astype(int)
            mass_comps += (ok & valid_m).astype(int)
            w = np.where(valid_m, phr, 1.0)
            num += np.where(ok, w, 0.0)
            den += np.where(ok, w / np.where(ok, ew, 1.0), 0.0)

            # 统计输入了质量或结构但未能解析成有效化学当量的未识别组分 (P1-3 & P1-1)
            comp_present = valid_m | has_smi_entry
            is_unresolved = comp_present & (~ok)
            unresolved_count += is_unresolved.astype(int)
            unresolved_missing_m += (is_unresolved & (~valid_m)).astype(int)
            ignored_phr += np.where(is_unresolved & valid_m, phr, 0.0)
            total_input_phr += np.where(valid_m, phr, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            eff_ew = np.where(den > 0, num / den, np.nan)
        # 严格要求：该侧所有有效组分必须均具有真实质量，任一缺失都不能标记为 has_all_real_mass (P0-2)
        has_all_real_mass = (active_comps > 0) & (active_comps == mass_comps)
        return (
            pd.Series(eff_ew, index=df.index),
            pd.Series(num, index=df.index),
            pd.Series(den, index=df.index),
            pd.Series(has_all_real_mass, index=df.index),
            pd.Series(total_input_phr, index=df.index),
            pd.Series(unresolved_count, index=df.index),
            pd.Series(ignored_phr, index=df.index),
            pd.Series(unresolved_missing_m, index=df.index),
        )

    eew, phr_r, equiv_r, has_mass_r, in_phr_r, unres_r, ign_phr_r, unres_m_r = _eff_ew_and_equivs("resin", res_idx)
    ahew, phr_h, equiv_h, has_mass_h, in_phr_h, unres_h, ign_phr_h, unres_m_h = _eff_ew_and_equivs("curing_agent", cur_idx)

    # 计算忽略质量占比与组成不完整标记 (P1-1: 未知组分缺质量时必须为 NaN，不可误报为 0)
    total_unres_missing_m = unres_m_r + unres_m_h
    total_in_phr = in_phr_r + in_phr_h
    total_ign_phr = ign_phr_r + ign_phr_h
    with np.errstate(divide="ignore", invalid="ignore"):
        ign_ratio = np.where(total_in_phr > 0, total_ign_phr / total_in_phr, 0.0)

    ignored_mass_fraction = np.where(
        total_unres_missing_m > 0,
        np.nan,
        np.where((unres_r + unres_h) > 0, ign_ratio, 0.0)
    )
    ignored_mass_fraction = pd.Series(ignored_mass_fraction, index=df.index)
    comp_incomplete = ((unres_r + unres_h) > 0) | (ignored_mass_fraction.isna()) | (ignored_mass_fraction > 1e-4)

    # ---- 化学计量比 r ----
    _R_LIT_CANDIDATES = (
        "formulation_r_value",
        "formulation_resin_hardener_equivalent_ratio",
        "crosslink_stoichiometry_r",
        "stoichiometric_ratio_r_cleaned",
        "r_value",
        "r_actual",
    )
    _R_DERIVED_CANDIDATES = (
        "cp_r_value",
        "rxn_r_actual",
        "rxn_r_effective",
        "xl_r_value",
    )
    has_lit_col_val = pd.Series(False, index=df.index)
    has_derived_col_val = pd.Series(False, index=df.index)
    r_val = pd.Series(np.nan, index=df.index, dtype=float)

    for c in _R_LIT_CANDIDATES:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            v_valid = v.where(np.isfinite(v) & (v > 0))
            has_lit_col_val = has_lit_col_val | v_valid.notna()
            r_val = r_val.fillna(v_valid)

    for c in _R_DERIVED_CANDIDATES:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            v_valid = v.where(np.isfinite(v) & (v > 0))
            has_derived_col_val = has_derived_col_val | v_valid.notna()
            r_val = r_val.fillna(v_valid)

    # 检测输入数据中是否存在历史来源标记与假定标记 (防来源漂移 P0-2)
    prior_r_source = pd.Series(None, index=df.index, dtype=object)
    for sc in ("formulation_r_source", "r_source", "cp_r_source", "rxn_r_source", "xl_r_source"):
        if sc in df.columns:
            s_val = df[sc].astype(str)
            valid_s = s_val.isin([
                "literature",
                "calculated_from_mass",
                "partial_composition_calculated",
                "assumed_stoichiometric",
                "missing_stoichiometry",
                "invalid_chemistry",
                "unknown_provenance",
            ])
            # 若 sc 为 formulation_r_source/r_source 但原始文献列无值，不能为派生列背书 (P0-1 & P0-2)
            if sc in ("formulation_r_source", "r_source") and (~has_lit_col_val).any():
                s_val = s_val.where(has_lit_col_val, None)
            prior_r_source = prior_r_source.where(prior_r_source.notna(), s_val.where(valid_s, None))

    # P1-2: 无来源证据的历史衍生字段，不得自动认定为 literature，标记为 unknown_provenance
    is_derived_only_without_src = prior_r_source.isna() & has_derived_col_val & (~has_lit_col_val)
    prior_r_source = prior_r_source.where(~is_derived_only_without_src, "unknown_provenance")

    # 继承布尔来源标志 (P0-2: 解决 keep_metadata=False 下来源血统丢失问题)
    if "cp_r_unknown" in df.columns:
        is_unk_flag = df["cp_r_unknown"].astype(bool)
        prior_r_source = prior_r_source.where(~is_unk_flag, "unknown_provenance")
    if "cp_r_partial" in df.columns:
        is_part_flag = df["cp_r_partial"].astype(bool)
        prior_r_source = prior_r_source.where(~is_part_flag, "partial_composition_calculated")

    has_explicit_lit_tag = (prior_r_source == "literature")
    for sc in ("formulation_r_source", "r_source"):
        if sc in df.columns:
            has_explicit_lit_tag = has_explicit_lit_tag | (df[sc].astype(str) == "literature")

    prior_assumed = pd.Series(False, index=df.index, dtype=bool)
    if "rxn_r_assumed" in df.columns:
        assumed_num = pd.to_numeric(df["rxn_r_assumed"], errors="coerce").fillna(0.0)
        prior_assumed = (assumed_num > 0.5)
    prior_assumed = prior_assumed | (prior_r_source == "assumed_stoichiometric")

    is_prior_assumed = (prior_r_source == "assumed_stoichiometric")
    is_prior_partial = (prior_r_source == "partial_composition_calculated")
    is_prior_calculated = (prior_r_source == "calculated_from_mass")
    is_prior_unknown = (prior_r_source == "unknown_provenance")
    is_unknown_provenance = is_prior_unknown

    r_source = pd.Series("invalid_chemistry", index=df.index, dtype=object)
    rxn_r_assumed = pd.Series(0.0, index=df.index, dtype=float)

    # 1. 质量推算 r (必须两端全部具备有效质量与当量)
    can_derive = (
        has_mass_r
        & has_mass_h
        & (equiv_r > 0)
        & (equiv_h > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        r_from_mass = np.where(can_derive, equiv_h / equiv_r, np.nan)
    r_from_mass = pd.Series(r_from_mass, index=df.index)
    r_from_mass = r_from_mass.where(np.isfinite(r_from_mass) & (r_from_mass > 0))

    # 若输入表曾记录 cp_r_from_mass，且当前无新质量推导，复用历史 cp_r_from_mass
    if "cp_r_from_mass" in df.columns:
        hist_rfm = pd.to_numeric(df["cp_r_from_mass"], errors="coerce")
        hist_rfm = hist_rfm.where(np.isfinite(hist_rfm) & (hist_rfm > 0))
        r_from_mass = r_from_mass.combine_first(hist_rfm)

    has_valid_chem = (f_r > 0) & (f_h_st > 0)

    # 检查是否为历史质量推导 (P0-3 来源防丢失 & P0-5 显式文献优先):
    rfm_diff = (r_val - r_from_mass).abs()
    matches_rfm = (~has_explicit_lit_tag) & r_from_mass.notna() & r_val.notna() & (rfm_diff < 1e-4)
    is_prior_mass = is_prior_calculated | matches_rfm

    # 真实文献值判定 (P0-1, P0-2, P0-4, P0-5, P1-2):
    # 显式文献标签只有在有对应数值时才生效 (数值与来源成对判断 P0-1, P0-2):
    # - formulation_r_source == "literature" 只能为 formulation_r_value 等原始文献列数值 (has_lit_col_val) 背书！
    # - 仅当原始文献列有数值 (has_lit_col_val)，或者 cp_r_source 本身就是 "literature" 时，显式文献才成立！
    is_lit_num_paired = (
        (has_explicit_lit_tag & has_lit_col_val)
        | ((df["cp_r_source"].astype(str) == "literature") if "cp_r_source" in df.columns else False)
    )

    is_lit_case = is_lit_num_paired | (
        prior_r_source.isna()
        & has_lit_col_val
        & (~prior_assumed)
        & (~is_prior_mass)
    )
    is_lit_case = is_lit_case & r_val.notna() & (r_val > 0)
    # 严格禁止任何已有非 literature 标签升级为 literature (P0-1)；但若成对显式指定 literature，文献事实优先！
    is_lit_case = is_lit_case & (is_lit_num_paired | (~(is_prior_assumed | is_prior_partial | is_prior_calculated | is_prior_unknown)))
    has_lit_r = is_lit_case

    # 质量计算成功情况 (区分完整质量与部分组成 P0-1, P1-3):
    has_new_full_mass_calc = (~is_lit_case) & can_derive & (~comp_incomplete)
    has_new_partial_mass_calc = (~is_lit_case) & can_derive & comp_incomplete

    # A. 真实文献值
    r_source = r_source.where(~is_lit_case, "literature")

    # B. 完整配方质量计算值 (新计算覆盖或保持历史完整质量计算)
    has_full_mass_source = (~is_lit_case) & (has_new_full_mass_calc | (is_prior_calculated & (~has_new_partial_mass_calc)))
    r_val = r_val.where(~has_new_full_mass_calc, r_from_mass)
    r_source = r_source.where(~has_full_mass_source, "calculated_from_mass")

    # B1. 部分组成质量计算值 (P0-1 保持历史部分组成，或新生成部分组成)
    has_partial_mass_source = (~is_lit_case) & (~has_full_mass_source) & (has_new_partial_mass_calc | is_prior_partial)
    if assume_stoichiometric_if_missing:
        r_val = r_val.where(~has_new_partial_mass_calc, r_from_mass)
        r_source = r_source.where(~has_partial_mass_source, "partial_composition_calculated")
    else:
        # 严格模式下，部分组成不能作为已验证的全配方 r
        r_val = r_val.where(~has_partial_mass_source, np.nan)
        r_source = r_source.where(~has_partial_mass_source, "missing_stoichiometry")

    # B2. 明确保持 unknown_provenance 状态 (P0-5 & P1-2)
    has_unknown_source = (~is_lit_case) & (~has_full_mass_source) & (~has_partial_mass_source) & is_unknown_provenance
    r_source = r_source.where(~has_unknown_source, "unknown_provenance")

    # C. 人工假设或缺失
    has_determined_source = is_lit_case | has_full_mass_source | has_partial_mass_source | has_unknown_source
    if assume_stoichiometric_if_missing:
        needs_assumption = (~has_determined_source) & has_valid_chem
        r_val = r_val.where(~needs_assumption, 1.0)
        r_source = r_source.where(~needs_assumption, "assumed_stoichiometric")
        rxn_r_assumed = rxn_r_assumed.where(~needs_assumption, 1.0)
    else:
        is_missing_stoich = (~has_determined_source) & has_valid_chem
        r_val = r_val.where(~is_missing_stoich, np.nan)
        r_source = r_source.where(~is_missing_stoich, "missing_stoichiometry")
        rxn_r_assumed = rxn_r_assumed.where(~is_missing_stoich, 0.0)

    # 汇总结构与外部声明官能度冲突标记 (P0-2)
    conflict_cols = [c for c in comp.columns if c.endswith("_struct_func_conflict")]
    if conflict_cols:
        any_struct_func_conflict = comp[conflict_cols].any(axis=1)
        out["structure_functionality_conflict"] = any_struct_func_conflict
        # 若存在不可调和的结构-官能度冲突，化学反应体系标记为不适用/冲突 (P0-2)
        r_source = r_source.where(~any_struct_func_conflict, "invalid_chemistry")
    else:
        out["structure_functionality_conflict"] = pd.Series(False, index=df.index)

    r_val = r_val.where(np.isfinite(r_val) & (r_val > 0), np.nan)

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

    # ---- 摩尔平均官能度 f_avg ----
    # 严密高分子统计网络理论（Flory 摩尔统计平均）：
    # 总体摩尔平均官能度 = (n_R * f_R + n_H * f_H,net) / (n_R + n_H)
    # 绝不能使用旧版 (f_r + r * f_h) / (1 + r)，后者错误地将当量比当成了分子摩尔比！
    # 严格判断真实摩尔数：当且仅当两端质量全部具备时，直接加权 f_avg_direct 真实有效
    has_real_mols = has_mass_r & has_mass_h & (mol_r > 0) & (mol_h > 0)

    f_avg_direct = (f_r * mol_r + f_h_net * mol_h) / (mol_r + mol_h).replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r_val * (f_r / f_h_st.replace(0.0, np.nan))
        f_avg_from_r = (f_r + ratio * f_h_net) / (1.0 + ratio)

    # 若缺少真实质量但由假设模式得到了 r_val (如 r=1.0)，必须采用严格当量关系推导的 f_avg_from_r！
    f_avg_network = np.where(has_real_mols, f_avg_direct, f_avg_from_r)
    f_avg_network = pd.Series(f_avg_network, index=df.index)
    f_avg_network = f_avg_network.where(np.isfinite(f_avg_network) & (f_avg_network > 0))

    # 保存 r_from_mass 供文献 r 与质量推算 r 冲突核查 (P1-1)
    out["cp_r_value"] = r_val
    out["cp_r_source"] = r_source
    out["rxn_r_assumed"] = rxn_r_assumed
    out["cp_r_from_mass"] = r_from_mass
    # 冲突标记: 文献有显式 r 且质量推算 r 也存在，且相对差异大于 5%
    r_diff = (r_val - r_from_mass).abs() / r_val.clip(lower=1e-4)
    out["cp_r_conflict"] = np.where(has_lit_r & r_from_mass.notna() & (r_diff > 0.05), 1.0, 0.0)
    out["cp_composition_incomplete"] = comp_incomplete
    out["cp_ignored_mass_fraction"] = ignored_mass_fraction
    out["cp_unknown_comp_count"] = unres_r + unres_h
    out["cp_r_unknown"] = (r_source == "unknown_provenance")
    out["cp_r_partial"] = (r_source == "partial_composition_calculated")
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
    has_complete_mass = has_mass_r & has_mass_h
    cov = np.where(
        has_complete_mass & cov_r & cov_h & np.isfinite(W) & (W > 0), "full",
        np.where(cov_r & cov_h & np.isfinite(epoxy_conc), "partial", "none"),
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
_METADATA_EXACT = (
    "cp_mechanism", "cp_coverage", "cp_eew_source", "cp_ahew_source", "cp_r_source",
    "cp_cache_input_hash", "cp_cache_assume_stoich", "cp_cache_max_components",
)


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
    assume_stoichiometric_if_missing: bool = True,
) -> pd.DataFrame:
    """窄表增强入口：逐组分物理量 + 配方级汇总，一次补齐。

    返回 df 的副本，追加 cp_ / 逐组分 *_resolved 列。

    参数：
        keep_metadata: 是否保留字符串型元数据列（*_mw_source / *_mechanism /
            cp_coverage 等）。**默认 False**——这些列只用于内部降级决策与
            诊断，不应作为特征写入数据集；需要调试时可置 True。
        assume_stoichiometric_if_missing: 当质量缺失时是否默认假设 r=1.0（默认 True）。
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df.copy()
    comp = compute_component_physics(
        df,
        max_components=max_components,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    )
    summary = compute_formulation_summary(
        df,
        comp,
        max_components=max_components,
        rho_g_cm3=rho_g_cm3,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    )
    out = df.copy()
    for c in comp.columns:
        out[c] = comp[c]
    for c in summary.columns:
        out[c] = summary[c]

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
            _r_target = _R_ORIG_NAMES[0]
            _was_na = pd.Series(True, index=out.index)
        else:
            _r_old = pd.to_numeric(out[_r_target], errors="coerce")
            _was_na = _r_old.isna()
            out[_r_target] = _r_old.where(_r_old.notna(), _r_filled)
            out = out.drop(columns=["cp_r_value"])

        # 同步来源 (P1 & P0-2):
        # 当原表 r 字段由 cp_r_value 补齐（_was_na 为 True）时，来源必须同步为真实的 cp_r_source！
        # 绝不能让填入的计算值/部分组成值继承原表的孤立 literature 标签！
        if "cp_r_source" in out.columns and _was_na.any():
            _src_from_cp = out["cp_r_source"].astype(str)
            for _sc_col in ("formulation_r_source", "r_source", "xl_r_source"):
                if _sc_col in out.columns:
                    out[_sc_col] = out[_sc_col].where(~_was_na, _src_from_cp)
                elif keep_metadata:
                    out[_sc_col] = _src_from_cp

    if not keep_metadata:
        drop = [c for c in out.columns if is_metadata_column(c) and c not in df.columns]
        if drop:
            out = out.drop(columns=drop)
    return out
