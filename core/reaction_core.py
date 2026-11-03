"""
core/reaction_core.py

CFRP 材料体系统一化学反应与统计交联核心：
收敛并统一以下核心物理计算：
1. 严密反应官能团识别与 SMARTS 规则（排除酰胺/磺酰胺、支持芳香酸酐、酚羟基等）
2. 双口径官能度模型 (化学计量口径 f_stoich vs 网络支化口径 f_network)
3. 严密化学计量当量比 r 计算（真实配方质量与当量计算，拒绝 ahew/eew 冒充）
4. Flory 摩尔统计平均官能度 f_avg 与数均分子量 Mn
5. Flory-Stockmayer 凝胶分支因子 (kappa_A, kappa_B) 与临界凝胶转化率 (p_Ac, p_Bc)
6. 理想交联网络可行性判定 (gel_possible_ideal) 与物理交联指标
7. 细粒度数据质量与覆盖度等级 (coverage, stoichiometry_source)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from .smiles_utils import bigsmiles_to_smiles, detect_chem_string_format
except Exception:
    bigsmiles_to_smiles = None

    def detect_chem_string_format(_s):  # type: ignore
        return "unknown"


# ---------------------------------------------------------------------------
# 规范反应 SMARTS 模式定义
# 严格排除非反应性酰胺与磺酰胺；严密支持芳香与脂环酸酐
# ---------------------------------------------------------------------------
EPOXIDE_PAT = "[C]1[O][C]1"
PRI_AMINE_PAT = "[NX3;H2;!$(NC=O);!$(NS(=O)=O)]"
SEC_AMINE_PAT = "[NX3;H1;!$(NC=O);!$(NS(=O)=O)]"
ANHYDRIDE_PAT = "[o,OX2]1~[#6](=[OX1])~[#6]~[#6]~[#6](=[OX1])~1"
THIOL_PAT = "[SX2;H1]"
ISOCYANATE_PAT = "[NX2]=[CX2]=[OX1]"
PHENOL_OH_PAT = "[OX2H]-c"
ALCOHOL_OH_PAT = "[OX2H]-[CX4]"
CARBOXYL_PAT = "[OX2H]-[CX3]=[OX1]"
AROMATIC_RING_PAT = "a1aaaaa1"

# 物理与工程上下界
F_MAX = 8.0
MW_BOUNDS = (20.0, 20000.0)
EW_BOUNDS = (5.0, 5000.0)
DEFAULT_RHO_G_CM3 = 1.2

# 机制标识常量
MECH_AMINE = "amine"
MECH_ANHYDRIDE = "anhydride"
MECH_THIOL = "thiol"
MECH_PHENOL = "phenol"
MECH_ISOCYANATE = "isocyanate"
MECH_CARBOXYL = "carboxyl"
MECH_ALCOHOL = "polyol"
MECH_UNKNOWN = "unknown"

# 四级当量比来源常量
R_SRC_LITERATURE = "literature"
R_SRC_CALCULATED = "calculated_from_mass"
R_SRC_PARTIAL_CALCULATED = "partial_composition_calculated"
R_SRC_ASSUMED_STOICH = "assumed_stoichiometric"
R_SRC_MISSING_STOICH = "missing_stoichiometry"
R_SRC_INVALID_CHEM = "invalid_chemistry"
R_SRC_UNKNOWN_PROVENANCE = "unknown_provenance"

# 兼容常量
R_SRC_FORMULATION = "calculated_from_mass"
R_SRC_UNKNOWN = "invalid_chemistry"

# 配方质量假设常量
COMP_ASSUMPTION_NONE = "none"
COMP_ASSUMPTION_EQUIMOLAR = "assumed_equimolar"
COMP_ASSUMPTION_RESIN_ONLY = "assumed_resin_only_reconstructed"
COMP_ASSUMPTION_HARDENER_ONLY = "assumed_hardener_only_reconstructed"

# 凝胶假设常量
GEL_ASSUMPTION_IDEAL = "ideal_statistical"
GEL_ASSUMPTION_MECHANISM_APPROX = "mechanism_approximate"

# 凝胶状态机制常量
GEL_REGIME_SUBCRITICAL = "subcritical"
GEL_REGIME_CRITICAL = "critical"
GEL_REGIME_SUPERCRITICAL = "supercritical"
GEL_REGIME_UNDETERMINED = "undetermined"
GEL_REGIME_NOT_APPLICABLE = "not_applicable"

# 模型适用性状态常量 (P0-1 & P0-2)
MODEL_STATUS_SUPPORTED = "supported"
MODEL_STATUS_UNDETERMINED = "undetermined"
MODEL_STATUS_NOT_APPLICABLE = "not_applicable"
MODEL_STATUS_STRUCTURE_UNRESOLVED = "structure_unresolved"
MODEL_STATUS_CONFLICT = "structure_functionality_conflict"

# 统一 r 与来源解析的候选列名 (P0-2)
R_CANDIDATE_COLS = (
    "formulation_r_value",
    "formulation_resin_hardener_equivalent_ratio",
    "crosslink_stoichiometry_r",
    "stoichiometric_ratio_r_cleaned",
    "cp_r_value",
    "r_value",
    "r_actual",
    "rxn_r_actual",
    "rxn_r_effective",
)

R_SOURCE_CANDIDATE_COLS = (
    "cp_r_source",
    "formulation_r_source",
    "r_source",
    "rxn_r_source",
    "xl_r_source",
)


def safe_positive_float(val: Any) -> Optional[float]:
    """安全解析正浮点数。若为 None, NaN, pd.NA, '', 'nan', <=0 等无效值返回 None。"""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if not s or s in ("nan", "none", "null", "<na>"):
            return None
        try:
            v = float(s)
            return v if (np.isfinite(v) and v > 0) else None
        except (ValueError, TypeError):
            return None
    try:
        if pd.isna(val) or not np.isfinite(val):
            return None
        v = float(val)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


# 编译模式缓存
_COMPILED_PATTERNS: Dict[str, Any] = {}


def get_compiled_smarts(pattern: str) -> Optional[Any]:
    if not RDKIT_AVAILABLE:
        return None
    if pattern not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[pattern] = Chem.MolFromSmarts(pattern)
    return _COMPILED_PATTERNS[pattern]


def count_substruct_matches(mol: Any, pattern: str) -> int:
    pat = get_compiled_smarts(pattern)
    if mol is None or pat is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(pat))
    except Exception:
        return 0


def count_unique_center_matches(mol: Any, pattern: str) -> int:
    """以唯一中心原子计数，避免对称酸酐等双向重复计数"""
    pat = get_compiled_smarts(pattern)
    if mol is None or pat is None:
        return 0
    try:
        matches = mol.GetSubstructMatches(pat)
        return len({m[0] for m in matches})
    except Exception:
        return 0


def clean_structure_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s


def parse_molecule_safe(text: Any) -> Optional[Any]:
    """严密解析化学分子，支持普通 SMILES 及 BigSMILES 重复单元提取"""
    if not RDKIT_AVAILABLE:
        return None
    s = clean_structure_text(text)
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return mol
    except Exception:
        pass

    if "{" in s and "}" in s:
        if bigsmiles_to_smiles is not None:
            try:
                converted = bigsmiles_to_smiles(s)
                if converted:
                    mol = Chem.MolFromSmiles(converted)
                    if mol is not None:
                        return mol
            except Exception:
                pass
        # 正则降级提取
        try:
            stripped = re.sub(r'\{.*?\}', '', s).replace("..", ".").strip(".")
            if stripped:
                mol = Chem.MolFromSmiles(stripped)
                if mol is not None:
                    return mol
        except Exception:
            pass
        try:
            match = re.search(r'\{.*?[<>\d]*\](.*?)(?:\[[<>\d]*\]).*?\}', s)
            if match:
                mol = Chem.MolFromSmiles(match.group(1).strip())
                if mol is not None:
                    return mol
        except Exception:
            pass

    return None


@dataclass
class FunctionalGroupInfo:
    """组分官能度信息（双口径）"""
    f_stoich: float      # 化学计量口径（每分子消耗的当量数）
    f_network: float     # 网络支化口径（每分子进入交联网络的分支度）
    mechanism: str       # 反应机制
    n_groups: int        # 检测到的特征反应基团数
    is_valid: bool       # 官能度是否有效且非未知


def inspect_resin_functionality(text_or_mol: Any, fallback_mol: Any = None) -> FunctionalGroupInfo:
    """分析树脂官能度：环氧基作为支化反应点，两口径一致"""
    target = fallback_mol if fallback_mol is not None and hasattr(fallback_mol, "GetSubstructMatches") else text_or_mol
    mol = target if hasattr(target, "GetSubstructMatches") else parse_molecule_safe(target)
    if mol is None:
        return FunctionalGroupInfo(0.0, 0.0, MECH_UNKNOWN, 0, False)
    n = count_substruct_matches(mol, EPOXIDE_PAT)
    if n <= 0:
        return FunctionalGroupInfo(0.0, 0.0, MECH_UNKNOWN, 0, False)
    f = float(min(n, F_MAX))
    return FunctionalGroupInfo(f, f, MECH_AMINE, n, True)


def inspect_curer_functionality(text_or_mol: Any, fallback_mol: Any = None) -> FunctionalGroupInfo:
    """分析固化剂官能度（严格区分化学计量口径与网络支化口径）"""
    target = fallback_mol if fallback_mol is not None and hasattr(fallback_mol, "GetSubstructMatches") else text_or_mol
    mol = target if hasattr(target, "GetSubstructMatches") else parse_molecule_safe(target)
    if mol is None:
        return FunctionalGroupInfo(0.0, 0.0, MECH_UNKNOWN, 0, False)

    # 1. 异氰酸酯
    n_nco = count_substruct_matches(mol, ISOCYANATE_PAT)
    if n_nco > 0:
        f = float(min(n_nco, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_ISOCYANATE, n_nco, True)

    # 2. 胺类（排除酰胺、磺酰胺）
    n_pri = count_substruct_matches(mol, PRI_AMINE_PAT)
    n_sec = count_substruct_matches(mol, SEC_AMINE_PAT)
    if n_pri + n_sec > 0:
        n_h = n_pri * 2 + n_sec
        f = float(min(n_h, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_AMINE, n_h, True)

    # 3. 酸酐类（开环酯化：1酸酐消耗1环氧 f_stoich=1，桥接2条链 f_network=2）
    n_anh = count_unique_center_matches(mol, ANHYDRIDE_PAT)
    if n_anh > 0:
        f_st = float(min(n_anh, F_MAX))
        f_net = float(min(2.0 * n_anh, F_MAX))
        return FunctionalGroupInfo(f_st, f_net, MECH_ANHYDRIDE, n_anh, True)

    # 4. 硫醇
    n_sh = count_substruct_matches(mol, THIOL_PAT)
    if n_sh > 0:
        f = float(min(n_sh, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_THIOL, n_sh, True)

    # 5. 酚羟基
    n_ph = count_substruct_matches(mol, PHENOL_OH_PAT)
    if n_ph > 0:
        f = float(min(n_ph, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_PHENOL, n_ph, True)

    # 6. 羧基
    n_cooh = count_substruct_matches(mol, CARBOXYL_PAT)
    if n_cooh > 0:
        f = float(min(n_cooh, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_CARBOXYL, n_cooh, True)

    # 7. 醇羟基
    n_oh = count_substruct_matches(mol, ALCOHOL_OH_PAT)
    if n_oh > 0:
        f = float(min(n_oh, F_MAX))
        return FunctionalGroupInfo(f, f, MECH_ALCOHOL, n_oh, True)

    return FunctionalGroupInfo(0.0, 0.0, MECH_UNKNOWN, 0, False)


# ---------------------------------------------------------------------------
# 反应核心物理计算函数（无力场严密统计高分子物理）
# ---------------------------------------------------------------------------
def calculate_stoichiometric_ratio_scalar(
    resin_mass: float,
    hardener_mass: float,
    eew: float,
    ahew: float,
) -> float:
    """根据真实质量与有效当量计算单组分或等效组分的固化剂/环氧反应当量比 r。
    若任意必要输入无效或 <= 0，返回 NaN，绝不使用 ahew / eew 兜底冒充。
    """
    if not (
        np.isfinite(resin_mass) and resin_mass > 0
        and np.isfinite(hardener_mass) and hardener_mass > 0
        and np.isfinite(eew) and eew > 0
        and np.isfinite(ahew) and ahew > 0
    ):
        return np.nan
    epoxy_equiv = resin_mass / eew
    hardener_equiv = hardener_mass / ahew
    if epoxy_equiv <= 0:
        return np.nan
    return float(hardener_equiv / epoxy_equiv)


@dataclass
class NetworkReactionSummary:
    """配方级严密高分子反应与交联统计网络结果"""
    r_value: float              # 反应当量比 r = N_B / N_A
    r_source: str               # 来源: literature / calculated_from_mass / assumed_stoichiometric / invalid_chemistry
    rxn_r_assumed: float        # 是否基于假设推算 r (1.0 为假设, 0.0 为文献/真实计算)
    composition_assumption: str # 配方质量重构假设: none / assumed_equimolar / assumed_resin_only_reconstructed / etc.
    gel_assumption: str         # 凝胶假设: ideal_statistical / mechanism_approximate
    balance: float              # 对称平衡度 min(r, 1/r)
    f_r: float                  # 树脂摩尔加权平均官能度
    f_h_stoich: float           # 固化剂摩尔加权平均官能度（化学计量口径）
    f_h_network: float          # 固化剂摩尔加权平均官能度（网络支化口径）
    f_avg: float                # 全配方分子数摩尔平均官能度 (n_R*f_R + n_H*f_H) / (n_R + n_H)
    Mn: float                   # 配方数均分子量 (g/mol)
    kappa_A: float              # 环氧侧 Flory 分支指数
    kappa_B: float              # 固化剂侧 Flory 分支指数
    p_A_critical: float         # 环氧侧临界凝胶转化率
    p_B_critical: float         # 固化剂侧临界凝胶转化率
    alpha_gel: float            # 体系临界凝胶转化率
    alpha_max: float            # 理论最大反应程度 min(1, r, 1/r)
    gel_possible_ideal: float   # 统计理论下能否凝胶 (1.0 或 0.0)
    gel_regime: str             # 凝胶状态机制: subcritical / critical / supercritical
    Mc_ideal: float             # 理想交联点间分子量 Mn / (f_avg - 2)
    nu_ideal_mol_m3: float      # 理想有效网络链密度 (mol/m³)
    chemistry_valid: bool       # 化学反应体系是否完全合法有效
    has_composition: bool       # 是否包含完整配方质量/PHR 信息
    unknown_component_count: int = 0      # 无法识别/无效结构的组分数量 (P1-2)
    ignored_component_count: int = 0      # 未参与反应计算的组分数量
    ignored_mass_fraction: float = 0.0    # 未参与反应计算的组分质量占比 (0.0 ~ 1.0)
    composition_incomplete: bool = False  # 配方组成是否存在未识别/缺失缺口
    reaction_model_status: str = "supported"  # 反应模型状态: supported / undetermined / not_applicable / structure_unresolved / structure_functionality_conflict
    structure_functionality_conflict: bool = False  # SMILES 结构与外部声明官能度冲突标记 (P0-2)


def evaluate_reaction_network(
    resins: List[Dict[str, Any]],
    curers: List[Dict[str, Any]],
    explicit_r: Optional[float] = None,
    rho_g_cm3: float = DEFAULT_RHO_G_CM3,
    assume_stoichiometric_if_missing: bool = True,
    existing_r_source: Optional[str] = None,
    rxn_r_assumed: Optional[float] = None,
) -> NetworkReactionSummary:
    """严密评估多组分体系的高分子交联反应网络。

    支持四级当量比 r 来源优先级：
    1. literature: 显式给出 explicit_r > 0 (rxn_r_assumed = 0.0)
    2. calculated_from_mass: 具备完整组分质量与当量 (rxn_r_assumed = 0.0)
    3. assumed_stoichiometric: 质量缺失时执行等当量配平重构 (r=1.0, rxn_r_assumed=1.0)
    4. missing_stoichiometry: 严格模式下结构有效但配比缺失 (r=NaN, rxn_r_assumed=0.0)
    5. invalid_chemistry: 无法构成有效反应体系 (r=NaN, rxn_r_assumed = 0.0)
    """
    def _enrich_comp(comp: Dict[str, Any], is_resin: bool) -> Dict[str, Any]:
        c = dict(comp)
        if "mass" not in c and "phr" in c:
            c["mass"] = c["phr"]

        # 安全清洗并规范化已传入字段（处理 None, NaN, pd.NA 等）
        c["mw"] = safe_positive_float(c.get("mw"))
        c["f_stoich"] = safe_positive_float(c.get("f_stoich"))
        c["f_network"] = safe_positive_float(c.get("f_network"))
        c["ew"] = safe_positive_float(c.get("ew"))
        c["mass"] = safe_positive_float(c.get("mass"))

        smi = c.get("smiles") or c.get("structure")
        struct_parsed = False
        struct_has_func = False
        struct_func_conflict = False
        if smi and RDKIT_AVAILABLE:
            mol = parse_molecule_safe(smi)
            if mol is not None:
                struct_parsed = True
                if c["mw"] is None:
                    try:
                        mw_calc = float(rdMolDescriptors.CalcExactMolWt(mol))
                        if mw_calc > 0:
                            c["mw"] = mw_calc
                    except Exception:
                        pass
                info = inspect_resin_functionality(mol) if is_resin else inspect_curer_functionality(mol)
                struct_has_func = (info.f_stoich > 0)
                ext_f_st = safe_positive_float(c.get("f_stoich"))
                ext_f_net = safe_positive_float(c.get("f_network"))

                # 结构与外部官能度矛盾检测 (P0-2):
                # 结构解析成功但完全不具备相应反应活性基团 (struct_has_func == False)，而外部却声明了正的官能度
                if (not struct_has_func) and (ext_f_st is not None or ext_f_net is not None):
                    struct_func_conflict = True

                if c["f_stoich"] is None and info.f_stoich > 0:
                    c["f_stoich"] = info.f_stoich
                if c["f_network"] is None and info.f_network > 0:
                    c["f_network"] = info.f_network
                if c.get("mechanism") is None or c.get("mechanism") == MECH_UNKNOWN:
                    c["mechanism"] = info.mechanism
                # 严格按照化学计量口径计算当量重与官能度互推（若已知 MW 和其中一个，严密推导另一个 P0-1）:
                if c["ew"] is None and c["mw"] is not None and c["f_stoich"] is not None and c["f_stoich"] > 0:
                    c["ew"] = c["mw"] / c["f_stoich"]
                elif c["f_stoich"] is None and c["mw"] is not None and c["ew"] is not None and c["ew"] > 0:
                    if c.get("mechanism") == MECH_ANHYDRIDE:
                        c["f_stoich"] = 1.0
                    else:
                        c["f_stoich"] = c["mw"] / c["ew"]
                    c["f_stoich_derived_from_ew"] = True
            else:
                c["unresolvable_structure"] = True

        # 若未提供 SMILES 但传入了数值，执行当量重与官能度安全互推 (P0-1)
        if c["ew"] is None and c["mw"] is not None and c["f_stoich"] is not None and c["f_stoich"] > 0:
            c["ew"] = c["mw"] / c["f_stoich"]
        elif c["f_stoich"] is None and c["mw"] is not None and c["ew"] is not None and c["ew"] > 0:
            if c.get("mechanism") == MECH_ANHYDRIDE:
                c["f_stoich"] = 1.0
            else:
                c["f_stoich"] = c["mw"] / c["ew"]
            c["f_stoich_derived_from_ew"] = True

        c["struct_parsed"] = struct_parsed
        c["struct_has_func"] = struct_has_func
        c["structure_functionality_conflict"] = struct_func_conflict
        return c

    valid_resins_in = [_enrich_comp(r, True) for r in resins]
    valid_curers_in = [_enrich_comp(c, False) for c in curers]

    # 检测是否存在结构与外部官能度矛盾 (P0-2)
    has_struct_func_conflict = any(
        x.get("structure_functionality_conflict", False) for x in valid_resins_in + valid_curers_in
    )

    # 计算组分已知端的加权平均官能度（若有 mass 按摩尔加权，否则按单体均分 P0-1）
    def _calc_known_avg_f(comps: List[Dict[str, Any]], key: str) -> float:
        if not comps:
            return np.nan
        vals = [safe_positive_float(c.get(key)) for c in comps]
        if any(v is None for v in vals):
            return np.nan
        mols = [
            (float(c["mass"]) / float(c["mw"])) if (c.get("mass") is not None and c.get("mw") is not None and float(c["mw"]) > 0)
            else 1.0
            for c in comps
        ]
        tot_m = sum(mols)
        return sum(m * v for m, v in zip(mols, vals)) / tot_m if tot_m > 0 else np.nan

    # 1. 验证组分合法性与覆盖质量治理 (P1-2)
    valid_r = [
        r for r in valid_resins_in
        if r["mw"] is not None and r["f_network"] is not None and not r.get("structure_functionality_conflict", False)
    ]
    valid_h = [
        c for c in valid_curers_in
        if c["mw"] is not None and c["f_network"] is not None and not c.get("structure_functionality_conflict", False)
    ]

    unknown_r = [r for r in valid_resins_in if r not in valid_r]
    unknown_h = [c for c in valid_curers_in if c not in valid_h]
    unknown_comp_count = len(unknown_r) + len(unknown_h)

    input_mass_all = sum(safe_positive_float(x.get("mass", x.get("phr"))) or 0.0 for x in resins + curers)
    valid_mass_all = sum(safe_positive_float(x.get("mass", x.get("phr"))) or 0.0 for x in valid_r + valid_h)
    ignored_mass = max(input_mass_all - valid_mass_all, 0.0)

    # P1-1: 未知组分质量缺失时，不能将忽略质量占比误报为 0.0
    unknown_comps_have_no_mass = any(
        safe_positive_float(x.get("mass", x.get("phr"))) is None for x in unknown_r + unknown_h
    )
    if unknown_comp_count > 0 and unknown_comps_have_no_mass:
        ignored_mass_fraction = np.nan
    elif input_mass_all > 0:
        ignored_mass_fraction = float(ignored_mass / input_mass_all)
    else:
        ignored_mass_fraction = np.nan if unknown_comp_count > 0 else 0.0

    comp_incomplete = (unknown_comp_count > 0) or (
        pd.isna(ignored_mass_fraction) or ignored_mass_fraction > 1e-4
    )

    # 判断是否存在酸酐固化剂
    has_anhydride = any(c.get("mechanism") == MECH_ANHYDRIDE for c in valid_h)
    gel_assumption = GEL_ASSUMPTION_MECHANISM_APPROX if has_anhydride else GEL_ASSUMPTION_IDEAL

    exp_r = safe_positive_float(explicit_r)

    if not valid_r or not valid_h or has_struct_func_conflict:
        # 4级: invalid_chemistry (无法构成有效环氧-固化剂反应体系 P0-1 & P0-2)
        has_unresolved_struct = any(
            x.get("unresolvable_structure", False) for x in valid_resins_in + valid_curers_in
        )
        if has_struct_func_conflict:
            rxn_model_status = MODEL_STATUS_CONFLICT
            gel_reg = MODEL_STATUS_CONFLICT
        elif has_unresolved_struct:
            rxn_model_status = MODEL_STATUS_STRUCTURE_UNRESOLVED
            gel_reg = GEL_REGIME_UNDETERMINED
        else:
            rxn_model_status = MODEL_STATUS_NOT_APPLICABLE
            gel_reg = GEL_REGIME_NOT_APPLICABLE

        return NetworkReactionSummary(
            r_value=exp_r if exp_r is not None else np.nan,
            r_source=R_SRC_INVALID_CHEM,
            rxn_r_assumed=0.0,
            composition_assumption=COMP_ASSUMPTION_NONE,
            gel_assumption=gel_assumption,
            balance=np.nan,
            f_r=np.nan,
            f_h_stoich=np.nan,
            f_h_network=np.nan,
            f_avg=np.nan,
            Mn=np.nan,
            kappa_A=0.0,
            kappa_B=0.0,
            p_A_critical=np.nan,
            p_B_critical=np.nan,
            alpha_gel=np.nan,
            alpha_max=np.nan,
            gel_possible_ideal=np.nan,
            gel_regime=gel_reg,
            Mc_ideal=np.nan,
            nu_ideal_mol_m3=np.nan,
            chemistry_valid=False,
            has_composition=False,
            unknown_component_count=unknown_comp_count,
            ignored_component_count=unknown_comp_count,
            ignored_mass_fraction=ignored_mass_fraction,
            composition_incomplete=comp_incomplete,
            reaction_model_status=rxn_model_status,
            structure_functionality_conflict=has_struct_func_conflict,
        )

    # 2. 检查配方质量信息：必须全部有效反应组分均具有明确相对质量/phr，且组成完整
    all_r_have_mass = all(r["mass"] is not None for r in valid_r)
    all_h_have_mass = all(c["mass"] is not None for c in valid_h)
    has_comp = all_r_have_mass and all_h_have_mass and not comp_incomplete
    any_r_has_mass = any(r["mass"] is not None for r in valid_r)
    any_h_has_mass = any(c["mass"] is not None for c in valid_h)

    # 3. 计算实际给出的当量数（仅在 all_r_have_mass and all_h_have_mass 成立时）
    eq_A = 0.0
    eq_B = 0.0
    all_r_have_ew = True
    all_h_have_ew = True
    if all_r_have_mass and all_h_have_mass:
        for r in valid_r:
            m = float(r["mass"])
            ew = safe_positive_float(r.get("ew"))
            if ew is None or ew <= 0:
                mw_r = safe_positive_float(r.get("mw"))
                fs_r = safe_positive_float(r.get("f_stoich"))
                if mw_r is not None and fs_r is not None and fs_r > 0:
                    ew = mw_r / fs_r
                else:
                    all_r_have_ew = False
                    ew = None
            if ew is not None and ew > 0:
                eq_A += m / ew
            else:
                all_r_have_ew = False

        for c in valid_h:
            m = float(c["mass"])
            ew = safe_positive_float(c.get("ew"))
            if ew is None or ew <= 0:
                mw_c = safe_positive_float(c.get("mw"))
                fs_c = safe_positive_float(c.get("f_stoich"))
                if mw_c is not None and fs_c is not None and fs_c > 0:
                    ew = mw_c / fs_c
                else:
                    all_h_have_ew = False
                    ew = None
            if ew is not None and ew > 0:
                eq_B += m / ew
            else:
                all_h_have_ew = False

    # 四级优先级判定 r (防来源漂移 P0-2 & 历史假设让位真实质量 P0-4):
    r_calc = np.nan
    r_source = R_SRC_INVALID_CHEM
    rxn_r_assumed_val = 0.0
    comp_assumption = COMP_ASSUMPTION_NONE

    is_prior_assumed = (existing_r_source == R_SRC_ASSUMED_STOICH) or (rxn_r_assumed is not None and rxn_r_assumed > 0.5)
    is_prior_calculated = (existing_r_source == R_SRC_CALCULATED)
    is_prior_partial = (existing_r_source == R_SRC_PARTIAL_CALCULATED)
    is_prior_unknown = (existing_r_source == R_SRC_UNKNOWN_PROVENANCE)

    # 具备完整质量时计算质量当量比 (P0-2: 必须确保 ew 解析成功)
    can_calc_mass = (
        all_r_have_mass
        and all_h_have_mass
        and all_r_have_ew
        and all_h_have_ew
        and eq_A > 0
        and eq_B > 0
    )
    r_mass_val = float(eq_B / eq_A) if can_calc_mass else np.nan

    # 真实文献值优先：显式 r 且非历史假定、非纯质量推算、非部分组成推算、非未知来源
    has_true_lit_r = (
        (exp_r is not None)
        and (not is_prior_assumed)
        and (not is_prior_calculated)
        and (not is_prior_partial)
        and (not is_prior_unknown)
    )

    if has_true_lit_r:
        # 优先级 1: 真实文献 r (最高优先级)
        r_calc = exp_r
        r_source = R_SRC_LITERATURE
        rxn_r_assumed_val = 0.0
        if not has_comp:
            comp_assumption = COMP_ASSUMPTION_RESIN_ONLY if all_r_have_mass else COMP_ASSUMPTION_EQUIMOLAR
    elif can_calc_mass:
        # 优先级 2: 当前质量计算 (区分完整配方与部分组成 P1-3 & P0-2)
        if comp_incomplete:
            if not assume_stoichiometric_if_missing:
                # 严格模式下，由于未知反应组分可能改变当量，完整配方 r 无法确定 -> 立即返回 undetermined / NaN (P0-2)
                can_struct_branch = all(r["f_network"] > 1.001 for r in valid_r) and all(c["f_network"] > 1.001 for c in valid_h)
                gel_regime_missing = GEL_REGIME_UNDETERMINED if can_struct_branch else GEL_REGIME_SUBCRITICAL
                gel_possible_missing = np.nan if can_struct_branch else 0.0

                f_r_known = _calc_known_avg_f(valid_r, "f_network")
                f_h_net_known = _calc_known_avg_f(valid_h, "f_network")
                f_h_st_known = _calc_known_avg_f(valid_h, "f_stoich")

                return NetworkReactionSummary(
                    r_value=np.nan,
                    r_source=R_SRC_MISSING_STOICH,
                    rxn_r_assumed=0.0,
                    composition_assumption=COMP_ASSUMPTION_NONE,
                    gel_assumption=gel_assumption,
                    balance=np.nan,
                    f_r=f_r_known,
                    f_h_stoich=f_h_st_known,
                    f_h_network=f_h_net_known,
                    f_avg=np.nan,
                    Mn=np.nan,
                    kappa_A=0.0,
                    kappa_B=0.0,
                    p_A_critical=np.nan,
                    p_B_critical=np.nan,
                    alpha_gel=np.nan,
                    alpha_max=np.nan,
                    gel_possible_ideal=gel_possible_missing,
                    gel_regime=gel_regime_missing,
                    Mc_ideal=np.nan,
                    nu_ideal_mol_m3=np.nan if np.isnan(gel_possible_missing) else 0.0,
                    chemistry_valid=True,
                    has_composition=False,
                    unknown_component_count=unknown_comp_count,
                    ignored_component_count=unknown_comp_count,
                    ignored_mass_fraction=ignored_mass_fraction,
                    composition_incomplete=True,
                )
            else:
                # 默认模式下，计算已识别组分的子集当量比，但标记为部分组成计算
                r_calc = r_mass_val
                r_source = R_SRC_PARTIAL_CALCULATED
                rxn_r_assumed_val = 0.0
                comp_assumption = COMP_ASSUMPTION_NONE
        else:
            r_calc = r_mass_val
            r_source = R_SRC_CALCULATED
            rxn_r_assumed_val = 0.0
            comp_assumption = COMP_ASSUMPTION_NONE
    elif is_prior_calculated and exp_r is not None:
        # 历史质量计算结果保持
        r_calc = exp_r
        r_source = R_SRC_CALCULATED
        rxn_r_assumed_val = 0.0
        comp_assumption = COMP_ASSUMPTION_NONE
    elif is_prior_partial and exp_r is not None:
        # 历史部分组成计算结果保持 (P0-1 来源防漂移)
        r_calc = exp_r
        r_source = R_SRC_PARTIAL_CALCULATED
        rxn_r_assumed_val = 0.0
        comp_assumption = COMP_ASSUMPTION_NONE
    elif is_prior_unknown and exp_r is not None:
        # 历史未知来源衍生结果保持 (P0-1 来源防漂移)
        r_calc = exp_r
        r_source = R_SRC_UNKNOWN_PROVENANCE
        rxn_r_assumed_val = 0.0
        comp_assumption = COMP_ASSUMPTION_NONE
    elif assume_stoichiometric_if_missing:
        # 优先级 3: 默认模式假定 r=1.0
        r_calc = 1.0
        r_source = R_SRC_ASSUMED_STOICH
        rxn_r_assumed_val = 1.0
        if all_r_have_mass and not any_h_has_mass:
            comp_assumption = COMP_ASSUMPTION_RESIN_ONLY
        elif not any_r_has_mass and all_h_have_mass:
            comp_assumption = COMP_ASSUMPTION_HARDENER_ONLY
        else:
            comp_assumption = COMP_ASSUMPTION_EQUIMOLAR
    else:
        # 优先级 4: 严格模式下缺失真实 r 与真实质量 (P1-1 凝胶状态修正: 区分 undetermined 与 subcritical)
        # 若两端结构上均多官能，r 缺失导致无法判定是否凝胶 -> undetermined / gel_possible_ideal = NaN
        can_struct_branch = all(r["f_network"] > 1.001 for r in valid_r) and all(c["f_network"] > 1.001 for c in valid_h)
        if can_struct_branch:
            gel_regime_missing = GEL_REGIME_UNDETERMINED
            gel_possible_missing = np.nan
        else:
            gel_regime_missing = GEL_REGIME_SUBCRITICAL
            gel_possible_missing = 0.0

        f_r_known = _calc_known_avg_f(valid_r, "f_network")
        f_h_net_known = _calc_known_avg_f(valid_h, "f_network")
        f_h_st_known = _calc_known_avg_f(valid_h, "f_stoich")

        return NetworkReactionSummary(
            r_value=np.nan,
            r_source=R_SRC_MISSING_STOICH if (valid_r and valid_h) else R_SRC_INVALID_CHEM,
            rxn_r_assumed=0.0,
            composition_assumption=COMP_ASSUMPTION_NONE,
            gel_assumption=gel_assumption,
            balance=np.nan,
            f_r=f_r_known,
            f_h_stoich=f_h_st_known,
            f_h_network=f_h_net_known,
            f_avg=np.nan,
            Mn=np.nan,
            kappa_A=0.0,
            kappa_B=0.0,
            p_A_critical=np.nan,
            p_B_critical=np.nan,
            alpha_gel=np.nan,
            alpha_max=np.nan,
            gel_possible_ideal=gel_possible_missing,
            gel_regime=gel_regime_missing,
            Mc_ideal=np.nan,
            nu_ideal_mol_m3=np.nan if np.isnan(gel_possible_missing) else 0.0,
            chemistry_valid=True,
            has_composition=has_comp,
            unknown_component_count=unknown_comp_count,
            ignored_component_count=unknown_comp_count,
            ignored_mass_fraction=ignored_mass_fraction,
            composition_incomplete=comp_incomplete,
        )

    # 4. 基于真实质量或自洽虚拟质量推算各组分摩尔数 n_i（彻底杜绝 KeyError: 'mass' 与 TypeError: NoneType P0-1）
    def _get_ew(x: Dict[str, Any]) -> Optional[float]:
        ew_v = safe_positive_float(x.get("ew"))
        if ew_v is not None and ew_v > 0:
            return ew_v
        mw_v = safe_positive_float(x.get("mw"))
        fs_v = safe_positive_float(x.get("f_stoich"))
        if mw_v is not None and fs_v is not None and fs_v > 0:
            return mw_v / fs_v
        return None

    if not has_comp:
        # 统一按照高分子等当量基准重构：
        # 树脂侧总当量 eq_A_base，固化剂侧总当量 target_eq_B = eq_A_base * r_calc
        # 树脂侧：若全部有 mass，按其 mass；否则按 100g 均分/等当量分配
        if all_r_have_mass:
            mass_r = [float(r["mass"]) for r in valid_r]
        else:
            mass_r = [100.0 / len(valid_r) for _ in valid_r]
        mols_r = [m / float(r["mw"]) for m, r in zip(mass_r, valid_r)]

        eq_A_list = []
        for m, r in zip(mass_r, valid_r):
            ew_r = _get_ew(r)
            if ew_r is not None and ew_r > 0:
                eq_A_list.append(m / ew_r)
            else:
                eq_A_list.append(None)

        if all(eq is not None for eq in eq_A_list):
            eq_A_base = sum(eq_A_list)
            target_eq_B = eq_A_base * r_calc
            eq_per_h = target_eq_B / len(valid_h)

            mass_h = []
            mols_h = []
            for c in valid_h:
                ew_c = _get_ew(c)
                if ew_c is not None and ew_c > 0:
                    m_c = eq_per_h * ew_c
                else:
                    m_c = float(c["mass"]) if c.get("mass") is not None else 10.0
                mass_h.append(m_c)
                mols_h.append(m_c / float(c["mw"]))
        else:
            # 树脂侧当量信息不足以完成虚拟配方等当量重构
            mass_h = [float(c["mass"]) if c.get("mass") is not None else 10.0 for c in valid_h]
            mols_h = [m / float(c["mw"]) for m, c in zip(mass_h, valid_h)]

        mass_r_tot = sum(mass_r)
        mass_h_tot = sum(mass_h)
    else:
        mass_r_tot = sum(float(r["mass"]) for r in valid_r)
        mass_h_tot = sum(float(c["mass"]) for c in valid_h)
        mols_r = [float(r["mass"]) / float(r["mw"]) for r in valid_r]
        mols_h = [float(c["mass"]) / float(c["mw"]) for c in valid_h]

    tot_mol_r = sum(mols_r) if mols_r else 0.0
    tot_mol_h = sum(mols_h) if mols_h else 0.0
    tot_sys_mol = tot_mol_r + tot_mol_h
    tot_sys_mass = mass_r_tot + mass_h_tot

    f_r_avg = sum(n * r["f_network"] for n, r in zip(mols_r, valid_r)) / tot_mol_r if tot_mol_r > 0 else np.nan

    # f_h_st_avg: 安全计算平均化学计量官能度 (若任一固化剂缺少 f_stoich 输出 NaN，绝不抛 TypeError P0-1)
    if tot_mol_h > 0 and all(safe_positive_float(c.get("f_stoich")) is not None for c in valid_h):
        f_h_st_avg = sum(n * float(c["f_stoich"]) for n, c in zip(mols_h, valid_h)) / tot_mol_h
    else:
        f_h_st_avg = np.nan

    f_h_net_avg = sum(n * c["f_network"] for n, c in zip(mols_h, valid_h)) / tot_mol_h if tot_mol_h > 0 else np.nan

    # 全体系分子数摩尔平均官能度 (严格物理公式: n_R*f_R + n_H*f_H) / (n_R + n_H)
    f_avg = (tot_mol_r * f_r_avg + tot_mol_h * f_h_net_avg) / tot_sys_mol if tot_sys_mol > 0 else np.nan
    Mn = tot_sys_mass / tot_sys_mol if tot_sys_mol > 0 else np.nan

    balance = min(r_calc, 1.0 / r_calc)
    alpha_max = min(1.0, r_calc, 1.0 / r_calc)
    # 5. Flory-Stockmayer 凝胶分支因子 kappa_A, kappa_B
    # 分支因子定义:
    # kappa_A = sum(n_i * f_i * (f_i - 1)) / sum(n_i * f_i)
    # kappa_B = sum(n_j * g_j * (g_j - 1)) / sum(n_j * g_j)
    sum_n_fa = sum(n * r["f_network"] for n, r in zip(mols_r, valid_r))
    sum_n_fa_term = sum(n * r["f_network"] * max(r["f_network"] - 1.0, 0.0) for n, r in zip(mols_r, valid_r))
    kappa_A = (sum_n_fa_term / sum_n_fa) if sum_n_fa > 0 else 0.0

    sum_n_fb = sum(n * c["f_network"] for n, c in zip(mols_h, valid_h))
    sum_n_fb_term = sum(n * c["f_network"] * max(c["f_network"] - 1.0, 0.0) for n, c in zip(mols_h, valid_h))
    kappa_B = (sum_n_fb_term / sum_n_fb) if sum_n_fb > 0 else 0.0

    # 临界转化率与凝胶判断:
    can_branch = (kappa_A > 1e-5 and kappa_B > 1e-5)
    if can_branch:
        prod_kappa = kappa_A * kappa_B
        p_A_crit = math.sqrt(r_calc / prod_kappa)
        p_B_crit = math.sqrt(1.0 / (r_calc * prod_kappa))
        prod_branch = prod_kappa * balance

        # P0-1: 区分 subcritical / critical / supercritical
        if prod_branch < (1.0 - 1e-4):
            gel_regime = GEL_REGIME_SUBCRITICAL
        elif prod_branch <= (1.0 + 1e-4):
            gel_regime = GEL_REGIME_CRITICAL
        else:
            gel_regime = GEL_REGIME_SUPERCRITICAL

        # P0-3: 物料平衡下检查两侧临界转化率必须 p_Ac <= 1.0 + 1e-4 且 p_Bc <= 1.0 + 1e-4
        p_reachable = (p_A_crit <= (1.0 + 1e-4)) and (p_B_crit <= (1.0 + 1e-4))
        can_gel = (gel_regime == GEL_REGIME_SUPERCRITICAL) and p_reachable

        if can_gel:
            alpha_gel = 1.0 / math.sqrt(prod_kappa)
            gel_possible_ideal = 1.0
        else:
            alpha_gel = np.nan
            gel_possible_ideal = 0.0
    else:
        # 单官能体系 (如 PGE: kappa_A = 0)
        p_A_crit = np.nan
        p_B_crit = np.nan
        alpha_gel = np.nan
        gel_regime = GEL_REGIME_SUBCRITICAL
        gel_possible_ideal = 0.0

    # 6. 理想交联指标
    if gel_possible_ideal > 0.5:
        if f_avg > 2.0001:
            Mc_ideal = Mn / (f_avg - 2.0)
            Mc_ideal = float(np.clip(Mc_ideal, 40.0, 20000.0))
            nu_ideal_mol_m3 = float((rho_g_cm3 * 1.0e6) / Mc_ideal)
        else:
            eff_f = max(f_r_avg, f_h_net_avg)
            if eff_f > 2.0001:
                Mc_ideal = Mn / (eff_f - 2.0)
                Mc_ideal = float(np.clip(Mc_ideal, 40.0, 20000.0))
                nu_ideal_mol_m3 = float((rho_g_cm3 * 1.0e6) / Mc_ideal)
            else:
                Mc_ideal = np.nan
                nu_ideal_mol_m3 = 0.0
    else:
        Mc_ideal = np.nan
        nu_ideal_mol_m3 = 0.0

    return NetworkReactionSummary(
        r_value=r_calc,
        r_source=r_source,
        rxn_r_assumed=rxn_r_assumed_val,
        composition_assumption=comp_assumption,
        gel_assumption=gel_assumption,
        balance=balance,
        f_r=f_r_avg,
        f_h_stoich=f_h_st_avg,
        f_h_network=f_h_net_avg,
        f_avg=f_avg,
        Mn=Mn,
        kappa_A=kappa_A,
        kappa_B=kappa_B,
        p_A_critical=p_A_crit,
        p_B_critical=p_B_crit,
        alpha_gel=alpha_gel,
        alpha_max=alpha_max,
        gel_possible_ideal=gel_possible_ideal,
        gel_regime=gel_regime,
        Mc_ideal=Mc_ideal,
        nu_ideal_mol_m3=nu_ideal_mol_m3,
        chemistry_valid=True,
        has_composition=has_comp,
        unknown_component_count=unknown_comp_count,
        ignored_component_count=unknown_comp_count,
        ignored_mass_fraction=ignored_mass_fraction,
        composition_incomplete=comp_incomplete,
    )


def compute_molar_average_functionality(
    mol_r: Any,
    f_r: Any,
    mol_h: Any,
    f_h: Any,
    r_val: Any = None,
) -> Any:
    """高分子全配方分子数摩尔平均官能度 f_avg。
    
    严格物理定义: f_avg = (n_r * f_r + n_h * f_h) / (n_r + n_h)
    1. 若提供了有效摩尔数 n_r, n_h 且 n_r + n_h > 0，直接使用摩尔加权；
    2. 若摩尔数缺失但已知当量比 r = (n_h * f_h) / (n_r * f_r)，
       严密推导式为: f_avg = (f_r * f_h * (1 + r)) / (f_h + r * f_r)；
    3. 严禁使用粗暴的 (f_r + r * f_h) / (1 + r)。
    """
    is_series = isinstance(mol_r, pd.Series) or isinstance(f_r, pd.Series) or isinstance(r_val, pd.Series)
    if is_series:
        mr = pd.to_numeric(mol_r, errors="coerce") if isinstance(mol_r, pd.Series) else pd.Series(mol_r, index=f_r.index if isinstance(f_r, pd.Series) else r_val.index)
        fr = pd.to_numeric(f_r, errors="coerce") if isinstance(f_r, pd.Series) else pd.Series(f_r, index=mr.index)
        mh = pd.to_numeric(mol_h, errors="coerce") if isinstance(mol_h, pd.Series) else pd.Series(mol_h, index=mr.index)
        fh = pd.to_numeric(f_h, errors="coerce") if isinstance(f_h, pd.Series) else pd.Series(f_h, index=mr.index)
        r_s = pd.to_numeric(r_val, errors="coerce") if isinstance(r_val, pd.Series) else (pd.Series(r_val, index=mr.index) if r_val is not None else pd.Series(np.nan, index=mr.index))
        
        has_mols = mr.notna() & mh.notna() & (mr > 0) & ((mr + mh) > 0)
        res = pd.Series(np.nan, index=mr.index, dtype=float)
        
        # 分支 1: 有摩尔数
        with np.errstate(divide="ignore", invalid="ignore"):
            res_mols = (mr * fr + mh * fh) / (mr + mh)
            res = res.where(~has_mols, res_mols)
            
            # 分支 2: 无摩尔数但有 r
            has_r_only = (~has_mols) & r_s.notna() & (r_s > 0) & fr.notna() & fh.notna() & (fr > 0) & (fh > 0)
            res_r = (fr * fh * (1.0 + r_s)) / (fh + r_s * fr)
            res = res.where(~has_r_only, res_r)
            
        return res.clip(1.0, 8.0)
    else:
        # 标量
        mr = float(mol_r) if mol_r is not None and not pd.isna(mol_r) else np.nan
        fr = float(f_r) if f_r is not None and not pd.isna(f_r) else np.nan
        mh = float(mol_h) if mol_h is not None and not pd.isna(mol_h) else np.nan
        fh = float(f_h) if f_h is not None and not pd.isna(f_h) else np.nan
        r_v = float(r_val) if r_val is not None and not pd.isna(r_val) else np.nan
        
        if np.isfinite(mr) and np.isfinite(mh) and mr > 0 and (mr + mh) > 0 and np.isfinite(fr) and np.isfinite(fh):
            return float(np.clip((mr * fr + mh * fh) / (mr + mh), 1.0, 8.0))
        elif np.isfinite(r_v) and r_v > 0 and np.isfinite(fr) and np.isfinite(fh) and fr > 0 and fh > 0:
            denom = fh + r_v * fr
            if denom > 0:
                return float(np.clip((fr * fh * (1.0 + r_v)) / denom, 1.0, 8.0))
        return np.nan


def compute_gel_point_conversion(
    f_r: Any,
    f_h: Any,
    r_val: Any = None,
    *,
    assume_stoichiometric_if_missing: bool = False,
) -> Any:
    """Flory-Stockmayer 凝胶点临界转化率 alpha_gel 与凝胶可行性。
    
    理论基础: 分支发散因子 (f_r - 1) * (f_h - 1) * min(r, 1/r) > 1.0 + 1e-4 (supercritical)。
    - 若任一端为单官能 (f_r <= 1.0 或 f_h <= 1.0)、临界/亚临界或转化率不可达，体系无法形成超临界三维网络，返回 NaN！
    - 若 r_val 缺失且不采用假设模式，必须返回 NaN，绝不能暗中假设 r=1.0！
    """
    is_series = isinstance(f_r, pd.Series) or isinstance(f_h, pd.Series) or isinstance(r_val, pd.Series)
    if is_series:
        fr = pd.to_numeric(f_r, errors="coerce") if isinstance(f_r, pd.Series) else pd.Series(f_r)
        fh = pd.to_numeric(f_h, errors="coerce") if isinstance(f_h, pd.Series) else pd.Series(f_h, index=fr.index)
        
        if r_val is not None:
            r_s = pd.to_numeric(r_val, errors="coerce") if isinstance(r_val, pd.Series) else pd.Series(r_val, index=fr.index)
        else:
            r_s = pd.Series(np.nan, index=fr.index, dtype=float)
            
        if assume_stoichiometric_if_missing:
            r_s = r_s.fillna(1.0)
            
        balance = np.minimum(r_s, 1.0 / r_s)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            branch_prod = (fr - 1.0) * (fh - 1.0)
            can_gel = (
                fr.notna() & fh.notna() & (fr > 1.001) & (fh > 1.001) 
                & r_s.notna() & (r_s > 0)
                & ((branch_prod * balance) > (1.0 + 1e-4))
            )
            alpha_gel_val = 1.0 / np.sqrt(branch_prod.clip(lower=1e-8))
            res = pd.Series(np.nan, index=fr.index, dtype=float)
            res = res.where(~can_gel, alpha_gel_val.clip(0.0, 1.0))
        return res
    else:
        fr = float(f_r) if f_r is not None and not pd.isna(f_r) else np.nan
        fh = float(f_h) if f_h is not None and not pd.isna(f_h) else np.nan
        
        if r_val is not None and not pd.isna(r_val):
            r_v = float(r_val) if float(r_val) > 0 else np.nan
        elif assume_stoichiometric_if_missing:
            r_v = 1.0
        else:
            r_v = np.nan
        
        if np.isfinite(fr) and np.isfinite(fh) and fr > 1.001 and fh > 1.001 and np.isfinite(r_v) and r_v > 0:
            branch_prod = (fr - 1.0) * (fh - 1.0)
            balance = min(r_v, 1.0 / r_v)
            if (branch_prod * balance) > (1.0 + 1e-4):
                val = 1.0 / math.sqrt(branch_prod)
                return float(min(max(val, 0.0), 1.0))
        return np.nan
