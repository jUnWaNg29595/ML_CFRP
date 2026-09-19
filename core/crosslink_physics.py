# -*- coding: utf-8 -*-
"""
crosslink_physics.py

环氧树脂交联密度物理层（纯 pandas/numpy，无 RDKit 依赖）

设计定位：ν 理论值是 PINN ν 潜变量的**物理基线**（log ν = nu_head(z) + log ν_theory），
强信号来自实测交联密度监督（按配方哈希关联，各目标表覆盖 31–48%）。
理论基线只需秩上合理、覆盖尽量广即可，残差由网络学习。

公式（信任层级）：
1. f_r（环氧官能度）：逐组分 epoxy_group_count 摩尔加权（高信任，仅用 curated 列）
2. f_h（活性氢官能度）：逐组分 active_hydrogen_equivalent_count 摩尔加权；
   缺失时按固化机理默认（amine=4.0 / anhydride=2.2 / phenolic=2.6 / cationic=2.0 / 其他=3.0）
3. ν_hybrid = (1e6/EEW) · balance · (f_avg−2)/f_avg · dilution   [f_avg 可用时]
   ν_agg    = 0.5 · eq_density · balance · dilution              [仅聚合列时]
   eq_density = 1e6/EEW（优先）或 1e6/AHEW·min(1,1/r)
4. Mc = 1e6·ρ / ν（clip 50–5000）；α_max = min(1, r, 1/r)；
   α_gel = 1/√((f_r−1)(f_h−1))（仅真实官能度可用时）

输出列（前缀 xl_）：xl_r_value, xl_balance, xl_dilution, xl_f_r, xl_f_h, xl_f_avg,
    xl_alpha_max, xl_alpha_gel, xl_alpha_cure_est, xl_Mc_g_mol,
    xl_nu_theory_mol_m3, xl_nu_source ∈ {hybrid, aggregate, none}
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# 默认环氧网络密度（g/cm³）：ν = ρ/Mc 的 mol/m³ 换算
DEFAULT_RHO_G_CM3 = 1.2

_R_COL_CANDIDATES = ("formulation_r_value", "formulation_resin_hardener_equivalent_ratio")
_EEW_COL = "formulation_resin_total_eew_g_eq"
_AHEW_COL = "formulation_hardener_total_ahew_g_eq"
_BINDER_PHR_COL = "formulation_epoxy_binder_total_phr"
_RESIN_PHR_COL = "resin_total_phr"
_INTEGRAL_COL = "process_temperature_time_integral_c_h"
_MAXT_COL = "process_max_temperature_c"
_POST_CURE_COL = "process_has_post_cure"

# 固化机理 → 活性氢官能度默认值
_FH_BY_MECHANISM = {
    "amine": 4.0,
    "anhydride_esterification": 2.2,
    "phenolic_hydroxyl_epoxy": 2.6,
    "cationic_ring_opening": 2.0,
}
_FH_DEFAULT = 3.0


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
    """逐组分摩尔分数加权官能度。返回 (f, mol_total, f_is_curated)。"""
    idx = df.index
    mol_sum = pd.Series(0.0, index=idx)
    fmol_sum = pd.Series(0.0, index=idx)
    curated_any = pd.Series(False, index=idx)

    for i in (1, 2, 3):
        mw = _get(df, f"{side}_{i}_molecular_weight_g_mol")
        phr = _get(df, f"{side}_{i}_amount_phr").fillna(0.0)
        f_curated = pd.Series(np.nan, index=idx, dtype=float)
        for fc in func_col_names:
            v = _get(df, f"{side}_{i}_{fc}")
            f_curated = f_curated.fillna(v)
        ok = mw.notna() & (mw > 0) & (phr > 0)
        if not ok.any():
            continue
        mol = (phr / mw).where(ok, 0.0)
        mol_sum = mol_sum + mol
        fmask = ok & f_curated.notna()
        fmol_sum = fmol_sum + (mol * f_curated).where(fmask, 0.0)
        curated_any = curated_any | fmask

    f = fmol_sum / mol_sum.replace(0.0, np.nan)
    f = f.clip(1.0, 8.0)
    return f, mol_sum, curated_any


def compute_crosslink_features(df: pd.DataFrame, rho_g_cm3: float = DEFAULT_RHO_G_CM3) -> pd.DataFrame:
    """计算交联密度理论特征（xl_ 前缀列）。空输入返回空表。"""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame()

    idx = df.index
    out = pd.DataFrame(index=idx, dtype=float)

    # ---- 基础量 ----
    r = _stoich_r(df)
    eew = _get(df, _EEW_COL)
    ahew = _get(df, _AHEW_COL)
    resin_phr = _get(df, _RESIN_PHR_COL).fillna(100.0)
    binder_phr = _get(df, _BINDER_PHR_COL)
    dilution = (resin_phr / binder_phr.replace(0.0, np.nan)).clip(0.2, 1.0).fillna(1.0)
    balance = np.minimum(r.fillna(1.0), 1.0 / r.fillna(1.0))
    alpha_cure = _estimate_alpha_cure(df)

    # ---- 官能度（信任层级） ----
    f_r, mol_r, f_r_curated = _molar_weighted_functionality(
        df, "resin", ("epoxy_group_count", "equivalent_group_count")
    )
    f_h_raw, mol_h, f_h_curated = _molar_weighted_functionality(
        df, "curing_agent", ("active_hydrogen_equivalent_count", "equivalent_group_count")
    )
    mech = df.get("curing_mechanism")
    if mech is None or isinstance(mech, pd.DataFrame):
        mech = pd.Series("unknown", index=idx)
    f_h_default = mech.astype(str).str.lower().map(_FH_BY_MECHANISM).fillna(_FH_DEFAULT)
    f_h = f_h_raw.where(f_h_raw.notna() & f_h_curated, f_h_default)

    # 摩尔分数加权平均官能度（固化剂侧按 r 折算）
    mol_h_s = mol_h * r.fillna(1.0)
    mol_total = (mol_r + mol_h_s).replace(0.0, np.nan)
    f_avg = ((f_r * mol_r + f_h * mol_h_s) / mol_total).clip(2.0, 8.0)
    f_avg = f_avg.where(mol_r > 0)  # 必须有树脂官能度信息

    # ---- 理论 ν（信任层级） ----
    eq_density_eew = (1.0e6 / eew).replace([np.inf, -np.inf], np.nan)
    eq_density_ahew = (1.0e6 / ahew).replace([np.inf, -np.inf], np.nan) * np.minimum(1.0, 1.0 / r.fillna(1.0))
    eq_density = eq_density_eew.fillna(eq_density_ahew)

    nu_hybrid = eq_density * balance * ((f_avg - 2.0) / f_avg) * dilution
    nu_agg = 0.5 * eq_density * balance * dilution

    nu = nu_hybrid.where(f_avg.notna(), nu_agg)
    nu = nu.where(nu > 0)
    source = np.where(f_avg.notna() & nu_hybrid.notna(), "hybrid",
                      np.where(nu_agg.notna(), "aggregate", "none"))

    # ---- 派生量 ----
    with np.errstate(all="ignore"):
        alpha_gel = (1.0 / np.sqrt(((f_r - 1.0) * (f_h - 1.0)).clip(lower=1e-5))).clip(0.0, 1.0)
    alpha_gel = alpha_gel.where(f_r_curated & (mol_r > 0))  # 仅真实官能度
    alpha_max = np.minimum(1.0, np.minimum(r.fillna(1.0), 1.0 / r.fillna(1.0)))
    mc = ((1.0e6 * rho_g_cm3) / nu.replace(0.0, np.nan)).clip(50.0, 5000.0)

    out["xl_r_value"] = r
    out["xl_balance"] = balance
    out["xl_dilution"] = dilution
    out["xl_f_r"] = f_r
    out["xl_f_h"] = f_h
    out["xl_f_avg"] = f_avg
    out["xl_alpha_max"] = alpha_max
    out["xl_alpha_gel"] = alpha_gel
    out["xl_alpha_cure_est"] = alpha_cure
    out["xl_Mc_g_mol"] = mc
    out["xl_nu_theory_mol_m3"] = nu
    out["xl_nu_source"] = source
    return out


def measured_nu_series(df: pd.DataFrame, nu_column: Optional[str] = None) -> tuple:
    """提取实测交联密度列（自动探测 'crosslink_density_mol_m3'）。返回 (列名, Series[float])。"""
    col = nu_column or "crosslink_density_mol_m3"
    if col not in df.columns:
        return None, pd.Series(np.nan, index=df.index, dtype=float)
    return col, _parse_num(df[col]).astype(float)
