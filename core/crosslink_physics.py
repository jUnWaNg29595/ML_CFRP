# -*- coding: utf-8 -*-
"""
crosslink_physics.py

环氧树脂交联密度物理层（单位统一为 **mol/m³**）

口径声明（唯一权威定义）
------------------------
本模块所有交联密度量纲一律为 **mol/m³**（SI 体积摩尔浓度），
换算关系：  ν [mol/m³] = ρ [g/m³] / Mc [g/mol]，ρ 默认 1.2e6 g/m³ (1.2 g/cm³)

历史遗留口径（已废弃，勿混用）：
    core/epoxy_mechanism_features.py 的 mech_crosslink_density_proxy = 1000/Mc
    是 **mmol/g**（每克树脂的交联点毫摩尔数），与 mol/m³ 相差 1000 倍。
    两者都叫"交联密度"，混用会造成 1000 倍量纲错误。

物理模型（按机制分派，实测标定）
--------------------------------
交联密度 = 环氧基摩尔浓度 × 网络连通度因子

    [ep]  = ρ / W          W = EEW + r·AHEW   每 mol 环氧基的配方质量 (g)
    ν     = [ep] · (f_h,net − 1) / f_h,net · bal

其中 bal = min(r, 1/r) 为化学计量平衡因子。实测表明 bal 的适用性依机制而变：
    胺类  ：过量胺本身起链终止作用 → 不需要 bal（实测 0.229 vs 0.157）
    酸酐类：化学计量失衡直接减少酯化桥接 → 需要 bal（实测 0.328 vs 0.056）

因此采用**机制感知**形式（5 折交叉验证中 5/5 折均被选中，全样本 Spearman 0.242）：
    胺类   ν = [ep] · (f_h,net − 1) / f_h,net
    酸酐类 ν = [ep] · (f_h,net − 1) / f_h,net · bal

官能度双口径（物理正确性关键）
------------------------------
酸酐固化剂的 f 有两种口径，**不可混用**：
    f_h,stoich ：化学计量口径，1 酸酐 : 1 环氧 → f=1
    f_h,net    ：网络支化口径，开环酯化后桥接 2 条链 → f=2
用 f_stoich=1 代入 (f−2) 型 Flory 公式会得到**负交联密度**
（实测酸酐子集 ν_junction 口径 Spearman = −0.067，完全失效）。
本模块统一使用 f_h,net。

数据来源与信任层级
------------------
逐组分 MW/EEW/AHEW/f 由 core.component_physics 分层补齐：
    L1 文献值 → L2 当量×官能度 → L3 结构直算（BigSMILES 采样代理与多片段被拒绝）

输出列（前缀 xl_）
    xl_r_value, xl_balance, xl_dilution,
    xl_f_r, xl_f_h, xl_f_h_stoich, xl_f_h_network, xl_f_avg,
    xl_alpha_max, xl_alpha_gel, xl_alpha_cure_est,
    xl_epoxy_conc_mol_m3, xl_W_g_per_epoxy,
    xl_Mc_g_mol, xl_nu_theory_mol_m3, xl_nu_junction_mol_m3,
    xl_mechanism, xl_nu_source ∈ {mechanism_aware, junction, aggregate, none},
    xl_coverage ∈ {full, partial, none}
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: 默认环氧网络密度（g/cm³）：ν = ρ/Mc 的 mol/m³ 换算基准
DEFAULT_RHO_G_CM3 = 1.2

#: 交联密度合理区间 (mol/m³)，对应 Mc ≈ 120–6000 g/mol
NU_BOUNDS_MOL_M3 = (100.0, 1.0e4)

_R_COL_CANDIDATES = (
    "formulation_r_value",
    "formulation_resin_hardener_equivalent_ratio",
    "crosslink_stoichiometry_r",
    "stoichiometric_ratio_r_cleaned",
    "r_value",
)
_EEW_COL = "formulation_resin_total_eew_g_eq"
_AHEW_COL = "formulation_hardener_total_ahew_g_eq"
_BINDER_PHR_COL = "formulation_epoxy_binder_total_phr"
_RESIN_PHR_COL = "resin_total_phr"
_INTEGRAL_COL = "process_temperature_time_integral_c_h"
_MAXT_COL = "process_max_temperature_c"
_POST_CURE_COL = "process_has_post_cure"

#: 固化机理 → 活性氢官能度默认值（仅在无法从结构解析时使用）
_FH_BY_MECHANISM = {
    "amine": 4.0,
    "amine_addition": 4.0,
    "anhydride_esterification": 2.0,   # 网络支化口径
    "phenolic_hydroxyl_epoxy": 2.6,
    "cationic_ring_opening": 2.0,
    "anionic_homopolymerization": 2.0,
    "thiol_epoxy_click": 2.0,
    "radical_photo_cure": 2.0,
}
_FH_DEFAULT = 3.0

#: 需要化学计量平衡因子的机制
_BALANCE_MECHANISMS = {"anhydride", "anhydride_esterification", "carboxyl"}


# ---------------------------------------------------------------------------
# 基础工具（保持向后兼容的公开名）
# ---------------------------------------------------------------------------
def _parse_num(s) -> pd.Series:
    """把带单位/混合文本的数值列安全转为 float（非数值 → NaN）。"""
    if isinstance(s, pd.DataFrame):  # 重复列名防御：取第一列
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return _parse_num(df[col])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _stoich_r(df: pd.DataFrame) -> pd.Series:
    for col in _R_COL_CANDIDATES:
        if col in df.columns:
            r = _get(df, col)
            return r.where(np.isfinite(r) & (r > 0), np.nan)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _estimate_alpha_cure(df: pd.DataFrame) -> pd.Series:
    """工艺固化度估算：优先温度-时间积分，缺失回退最高固化温度。"""
    alpha = pd.Series(np.nan, index=df.index, dtype=float)
    integral = _get(df, _INTEGRAL_COL)
    if integral.notna().any():
        a = 1.0 - np.exp(-np.clip(integral, 0.0, None) / 2500.0)
        alpha = a.where(integral.notna() & (integral > 0), alpha)
    tmax = _get(df, _MAXT_COL)
    if tmax.notna().any():
        alpha = alpha.fillna(np.clip((tmax - 80.0) / 120.0, 0.30, 0.95))
    post = _get(df, _POST_CURE_COL).fillna(0.0)
    alpha = alpha + 0.05 * post.clip(0.0, 1.0)
    return alpha.clip(0.30, 1.0)


def _molar_weighted_functionality(
    df: pd.DataFrame,
    side: str,  # "resin" | "curing_agent"
    func_col_names: tuple,
) -> tuple:
    """逐组分**摩尔分数**加权官能度（Flory 定义）。返回 (f, mol_total, f_is_curated)。

    注意：Flory 的 f_avg 是摩尔加权，不是 phr（质量分数）加权。
    每组分摩尔数 = phr / MW；无 phr 时退化为等质量基准 1/MW。
    """
    idx = df.index
    mol_sum = pd.Series(0.0, index=idx)
    fmol_sum = pd.Series(0.0, index=idx)
    curated_any = pd.Series(False, index=idx)

    for i in (1, 2, 3):
        mw = _get(df, f"{side}_{i}_molecular_weight_g_mol")
        # 优先用补齐后的分子量列
        mw = mw.fillna(_get(df, f"{side}_{i}_mw_resolved"))
        phr = _get(df, f"{side}_{i}_amount_phr").fillna(0.0)
        f_curated = pd.Series(np.nan, index=idx, dtype=float)
        for fc in func_col_names:
            f_curated = f_curated.fillna(_get(df, f"{side}_{i}_{fc}"))
        ok = mw.notna() & (mw > 0) & (phr > 0)
        if not ok.any():
            continue
        mol = (phr / mw).where(ok, 0.0)
        mol_sum = mol_sum + mol
        fmask = ok & f_curated.notna()
        fmol_sum = fmol_sum + (mol * f_curated).where(fmask, 0.0)
        curated_any = curated_any | fmask

    with np.errstate(divide="ignore", invalid="ignore"):
        f = fmol_sum / mol_sum.replace(0.0, np.nan)
    f = f.clip(1.0, 8.0)
    return f, mol_sum, curated_any


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def compute_crosslink_features(
    df: pd.DataFrame,
    rho_g_cm3: float = DEFAULT_RHO_G_CM3,
    *,
    use_component_physics: bool = True,
) -> pd.DataFrame:
    """计算交联密度理论特征（xl_ 前缀列，单位统一 mol/m³）。

    参数
    ----
    df : 窄表或宽表。若含逐组分结构列，会经 component_physics 分层补齐 MW/EEW/AHEW/f。
    rho_g_cm3 : 环氧网络密度，ν 与之线性相关。
    use_component_physics : 是否启用逐组分物理量补齐（默认开启）。
        关闭时退化为仅用聚合列的老口径（aggregate）。

    空输入返回空表。
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame()

    idx = df.index
    out = pd.DataFrame(index=idx, dtype=float)
    RHO = float(rho_g_cm3) * 1.0e6  # g/m³

    # ---- 逐组分物理量补齐（可选） ----
    cp: Optional[pd.DataFrame] = None
    if use_component_physics:
        try:
            from .component_physics import (
                compute_component_physics,
                compute_formulation_summary,
            )

            cp = compute_component_physics(df)
            if len(cp) == len(df):
                cp = pd.concat(
                    [cp, compute_formulation_summary(df, cp, rho_g_cm3=rho_g_cm3)], axis=1
                )
        except Exception:
            cp = None

    # ---- 基础量 ----
    r = _stoich_r(df)
    if cp is not None and "cp_r_value" in cp.columns:
        cp_r = _parse_num(cp["cp_r_value"])
        r = r.where(r.notna(), cp_r)
    r = r.where(np.isfinite(r) & (r > 0))

    eew = _get(df, _EEW_COL)
    ahew = _get(df, _AHEW_COL)
    if cp is not None:
        if "cp_eew" in cp.columns:
            cp_eew = _parse_num(cp["cp_eew"])
            eew = cp_eew.where(cp_eew.notna(), eew)
        if "cp_ahew" in cp.columns:
            cp_ahew = _parse_num(cp["cp_ahew"])
            ahew = cp_ahew.where(cp_ahew.notna(), ahew)

    resin_phr = _get(df, _RESIN_PHR_COL).fillna(100.0)
    binder_phr = _get(df, _BINDER_PHR_COL)
    dilution = (resin_phr / binder_phr.replace(0.0, np.nan)).clip(0.2, 1.0).fillna(1.0)
    balance = np.minimum(r.fillna(1.0), 1.0 / r.fillna(1.0))
    alpha_cure = _estimate_alpha_cure(df)

    # ---- 机制 ----
    mech = pd.Series("unknown", index=idx, dtype=object)
    raw_mech = df.get("curing_mechanism")
    if raw_mech is not None and not isinstance(raw_mech, pd.DataFrame):
        mech = raw_mech.astype(str).str.lower()
    if cp is not None and "cp_mechanism" in cp.columns:
        cp_mech = cp["cp_mechanism"].astype(str).str.lower()
        mech = mech.where(mech != "unknown", cp_mech)
    mech = mech.fillna("unknown")

    # ---- 官能度（信任层级） ----
    # 关键：固化剂的两套口径必须严格分离，不能让文献的化学计量列
    # (active_hydrogen_equivalent_count) 污染网络支化口径。
    # 实测反例：酸酐文献列 f=1，若覆盖 f_network=2 会让 (f-1)/f 从 0.5 变 0
    # 并把 ν 的 Spearman 从 0.242 拉到 0.192。
    # 因此 cp（component_physics 结构解析结果）优先，聚合列仅作兜底。

    f_r, mol_r, f_r_curated = _molar_weighted_functionality(
        df, "resin", ("epoxy_group_count", "equivalent_group_count")
    )
    if cp is not None and "cp_f_r" in cp.columns:
        cp_f_r = _parse_num(cp["cp_f_r"])
        f_r = cp_f_r.where(cp_f_r.notna(), f_r)
        # 结构解析得到的 f_r 是可靠官能度，应允许 alpha_gel 计算。
        # 历史 bug：f_r_curated 仅在逐组分文献列存在时为 True，而窄表恰好没有
        # 那些列 → xl_alpha_gel 恒为全空。
        f_r_curated = f_r_curated | cp_f_r.notna()
        mol_r = mol_r.where(
            mol_r > 0, _parse_num(cp.get("cp_mol_resin", pd.Series(0.0, index=idx)))
        )

    # 固化剂：网络支化口径（优先结构解析结果）
    f_h_net, mol_h, f_h_curated = _molar_weighted_functionality(
        df, "curing_agent", ("f_network", "active_hydrogen_equivalent_count", "equivalent_group_count")
    )
    if cp is not None and "cp_f_h_network" in cp.columns:
        cp_f_net = _parse_num(cp["cp_f_h_network"])
        f_h_net = cp_f_net.where(cp_f_net.notna(), f_h_net)
        f_h_curated = f_h_curated | cp_f_net.notna()
        mol_h = mol_h.where(
            mol_h > 0, _parse_num(cp.get("cp_mol_curer", pd.Series(0.0, index=idx)))
        )

    # 固化剂：化学计量口径（仅作记录/诊断，不参与 ν）
    f_h_st, _, _ = _molar_weighted_functionality(
        df, "curing_agent", ("f_stoich", "active_hydrogen_equivalent_count", "equivalent_group_count")
    )
    if cp is not None and "cp_f_h_stoich" in cp.columns:
        cp_f_st = _parse_num(cp["cp_f_h_stoich"])
        f_h_st = cp_f_st.where(cp_f_st.notna(), f_h_st)

    # 机制默认仅填补仍缺失者
    f_h_default = mech.map(_FH_BY_MECHANISM).fillna(_FH_DEFAULT)
    f_h_net = f_h_net.where(f_h_net.notna() & (f_h_net > 0), f_h_default)
    f_h_st = f_h_st.where(f_h_st.notna() & (f_h_st > 0), f_h_net)

    mol_h_s = mol_h * r.fillna(1.0)
    mol_total = (mol_r + mol_h_s).replace(0.0, np.nan)
    f_avg = ((f_r * mol_r + f_h_net * mol_h_s) / mol_total).clip(2.0, 8.0)
    f_avg = f_avg.where(mol_r > 0)  # 必须有树脂官能度信息

    # ---- 环氧基摩尔浓度与每环氧基配方质量 ----
    W_g = eew + r.fillna(1.0) * ahew
    if cp is not None and "cp_W_g_per_epoxy" in cp.columns:
        cp_w = _parse_num(cp["cp_W_g_per_epoxy"])
        W_g = cp_w.where(cp_w.notna(), W_g)
    epoxy_conc = pd.Series(
        np.where(np.isfinite(W_g) & (W_g > 0), RHO / W_g, np.nan), index=idx
    )
    if cp is not None and "cp_epoxy_conc_mol_m3" in cp.columns:
        cp_ep = _parse_num(cp["cp_epoxy_conc_mol_m3"])
        epoxy_conc = cp_ep.where(cp_ep.notna(), epoxy_conc)

    # ---- 理论 ν（机制感知） ----
    conn = (f_h_net - 1.0) / f_h_net
    nu_mech = epoxy_conc * conn
    needs_bal = mech.isin(_BALANCE_MECHANISMS)
    nu_mech = nu_mech.where(~needs_bal, nu_mech * balance)
    nu_mech = nu_mech * dilution
    nu_mech = nu_mech.where(np.isfinite(nu_mech) & (nu_mech > 0))

    # ---- junction 口径（Flory 交联点数密度） ----
    with np.errstate(divide="ignore", invalid="ignore"):
        n_r = (1.0 / f_r).replace([np.inf, -np.inf], np.nan)
        n_h = (r.fillna(1.0) / f_h_net).replace([np.inf, -np.inf], np.nan)
        nu_junction = RHO * (n_r * (f_r - 2.0).clip(lower=0.0)
                             + n_h * (f_h_net - 2.0).clip(lower=0.0)) / W_g.replace(0.0, np.nan)
    nu_junction = nu_junction.where(np.isfinite(nu_junction) & (nu_junction > 0))

    # ---- 老口径兜底：聚合列（无逐组分信息时） ----
    eq_density_eew = (1.0e6 / eew).replace([np.inf, -np.inf], np.nan)
    eq_density_ahew = (1.0e6 / ahew).replace([np.inf, -np.inf], np.nan) * np.minimum(
        1.0, 1.0 / r.fillna(1.0)
    )
    eq_density = eq_density_eew.fillna(eq_density_ahew)
    nu_agg = 0.5 * eq_density * balance * dilution
    nu_agg = nu_agg.where(np.isfinite(nu_agg) & (nu_agg > 0))

    # ---- 口径选择与来源标记 ----
    nu = nu_mech.where(nu_mech.notna(), nu_junction)
    nu = nu.where(nu.notna(), nu_agg)

    source = np.where(
        nu_mech.notna(), "mechanism_aware",
        np.where(nu_junction.notna(), "junction",
                 np.where(nu_agg.notna(), "aggregate", "none")),
    )

    if cp is not None and "cp_coverage" in cp.columns:
        coverage = cp["cp_coverage"].astype(str).values
    else:
        coverage = np.where(epoxy_conc.notna() & f_avg.notna(), "full",
                            np.where(epoxy_conc.notna(), "partial", "none"))

    # ---- 派生量 ----
    with np.errstate(all="ignore"):
        alpha_gel = (1.0 / np.sqrt(((f_r - 1.0) * (f_h_net - 1.0)).clip(lower=1e-5))).clip(0.0, 1.0)
    alpha_gel = alpha_gel.where(f_r_curated & (mol_r > 0) & (f_h_net > 1.0))
    alpha_max = np.minimum(1.0, np.minimum(r.fillna(1.0), 1.0 / r.fillna(1.0)))
    mc = ((1.0e6 * rho_g_cm3) / nu.replace(0.0, np.nan)).clip(50.0, 5000.0)

    out["xl_r_value"] = r
    out["xl_balance"] = balance
    out["xl_dilution"] = dilution
    out["xl_f_r"] = f_r
    out["xl_f_h"] = f_h_net
    out["xl_f_h_stoich"] = f_h_st
    out["xl_f_h_network"] = f_h_net
    out["xl_f_avg"] = f_avg
    out["xl_alpha_max"] = alpha_max
    out["xl_alpha_gel"] = alpha_gel
    out["xl_alpha_cure_est"] = alpha_cure
    out["xl_epoxy_conc_mol_m3"] = epoxy_conc
    out["xl_W_g_per_epoxy"] = W_g
    out["xl_Mc_g_mol"] = mc
    out["xl_nu_theory_mol_m3"] = nu
    out["xl_nu_junction_mol_m3"] = nu_junction
    out["xl_mechanism"] = mech.values
    out["xl_nu_source"] = source
    out["xl_coverage"] = coverage
    return out


def measured_nu_series(df: pd.DataFrame, nu_column: Optional[str] = None) -> tuple:
    """提取实测交联密度列（自动探测 'crosslink_density_mol_m3'）。返回 (列名, Series[float])。"""
    col = nu_column or "crosslink_density_mol_m3"
    if col not in df.columns:
        return None, pd.Series(np.nan, index=df.index, dtype=float)
    return col, _parse_num(df[col]).astype(float)


def nu_mol_per_m3_to_mmol_per_g(nu_mol_m3: pd.Series, rho_g_cm3: float = DEFAULT_RHO_G_CM3) -> pd.Series:
    """mol/m³ → mmol/g 显式换算（用于与旧口径 epoxy_mechanism_features 对接）。

    ν [mmol/g] = ν [mol/m³] / (ρ [g/cm³] × 1e6 [cm³/m³]) × 1e3 [mmol/mol]
               = ν [mol/m³] / (ρ × 1e3)
    """
    return nu_mol_m3 / (float(rho_g_cm3) * 1.0e3)


def mmol_per_g_to_nu_mol_per_m3(nu_mmol_g: pd.Series, rho_g_cm3: float = DEFAULT_RHO_G_CM3) -> pd.Series:
    """mmol/g → mol/m³ 显式换算（旧口径迁移用）。"""
    return nu_mmol_g * float(rho_g_cm3) * 1.0e3
