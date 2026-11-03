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

from .reaction_core import (
    DEFAULT_RHO_G_CM3,
    compute_molar_average_functionality,
    compute_gel_point_conversion,
)

#: 默认环氧网络密度（g/cm³）：ν = ρ/Mc 的 mol/m³ 换算基准
# DEFAULT_RHO_G_CM3 继承自 reaction_core (1.2)

#: 交联密度合理区间 (mol/m³)，对应 Mc ≈ 120–6000 g/mol
NU_BOUNDS_MOL_M3 = (100.0, 1.0e4)

_R_COL_CANDIDATES = (
    "formulation_r_value",
    "formulation_resin_hardener_equivalent_ratio",
    "crosslink_stoichiometry_r",
    "stoichiometric_ratio_r_cleaned",
    "r_value",
    "cp_r_value",
    "r_actual",
    "rxn_r_actual",
    "rxn_r_effective",
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
# 审查报告修复：移除未知固化剂默认 3.0，缺失时严格设为 NaN，禁止凭空猜测
_FH_DEFAULT = np.nan

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
    r = pd.Series(np.nan, index=df.index, dtype=float)
    for col in _R_COL_CANDIDATES:
        if col in df.columns:
            candidate = _get(df, col)
            candidate = candidate.where(np.isfinite(candidate) & (candidate > 0), np.nan)
            r = r.combine_first(candidate)
    return r


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
    cp: Optional[pd.DataFrame] = None,
    assume_stoichiometric_if_missing: bool = True,
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
    if cp is None and use_component_physics:
        try:
            from .component_physics import (
                compute_component_physics,
                compute_formulation_summary,
            )

            cp_raw = compute_component_physics(df, assume_stoichiometric_if_missing=assume_stoichiometric_if_missing)
            if cp_raw is not None and not cp_raw.empty:
                summ = compute_formulation_summary(
                    df,
                    cp_raw,
                    rho_g_cm3=rho_g_cm3,
                    assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
                )
                cols_to_add = [c for c in summ.columns if c not in cp_raw.columns]
                cp = pd.concat([cp_raw, summ[cols_to_add]], axis=1) if cols_to_add else cp_raw
        except Exception:
            cp = None
    elif cp is not None and use_component_physics:
        # cp 是外部预计算值，必须验证其生成时的真实原始输入。
        # 旧 cp 缺少签名时一律视为不可信，并基于当前 df 重建。
        # 若用户只改了 EEW/AHEW/SMILES/f/极小 phr，输入指纹也会变化。
        from .cp_cache_integrity import invalid_cp_cache_mask
        invalid = invalid_cp_cache_mask(
            df, cp, max_components=3,
            assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
        )
        must_refresh = bool(invalid.any()) or (
            "cp_r_value" not in cp.columns or "cp_f_avg_network" not in cp.columns
        )
        if must_refresh:
            # Correctness-first: refresh the batch atomically. Unchanged cp
            # batches bypass RDKit entirely; changed batches never reuse old
            # topology, functionality or chemistry. No swallowed exceptions.
            from .component_physics import (
                compute_component_physics,
                compute_formulation_summary,
            )
            cp_fresh = compute_component_physics(
                df,
                assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
            )
            summ_fresh = compute_formulation_summary(
                df, cp_fresh, rho_g_cm3=rho_g_cm3,
                assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
            )
            cols_to_add = [c for c in summ_fresh.columns if c not in cp_fresh.columns]
            cp = (
                pd.concat([cp_fresh, summ_fresh[cols_to_add]], axis=1)
                if cols_to_add else cp_fresh
            )
    # Never mix an unverified cp with current input facts.
    # -----------------------------------------------------------------------
    # 1. 组分与配方完整度判定 (P0-2 & P1-2)
    # -----------------------------------------------------------------------
    is_partial_composition = pd.Series(False, index=idx, dtype=bool)
    if cp is not None:
        if "cp_composition_incomplete" in cp.columns:
            is_partial_composition = is_partial_composition | cp["cp_composition_incomplete"].astype(bool)
        if "cp_r_source" in cp.columns:
            is_partial_composition = is_partial_composition | (cp["cp_r_source"].astype(str) == "partial_composition_calculated")
        if "cp_unknown_comp_count" in cp.columns:
            is_partial_composition = is_partial_composition | (pd.to_numeric(cp["cp_unknown_comp_count"], errors="coerce").fillna(0) > 0)
    if "cp_composition_incomplete" in df.columns:
        is_partial_composition = is_partial_composition | df["cp_composition_incomplete"].astype(bool)
    if "cp_r_source" in df.columns:
        is_partial_composition = is_partial_composition | (df["cp_r_source"].astype(str) == "partial_composition_calculated")
    if "cp_unknown_comp_count" in df.columns:
        is_partial_composition = is_partial_composition | (pd.to_numeric(df["cp_unknown_comp_count"], errors="coerce").fillna(0) > 0)

    # -----------------------------------------------------------------------
    # 2. 证据优先级解析当前有效 r 与来源 (P0-1, P0-2, 统一来源规则)
    # -----------------------------------------------------------------------
    # 检查 df 与 cp 中针对 r 的显式来源证据
    # 关键 (P0-1): 历史 xl_r_source 是交联物理输出，绝不能跨字段自动证明 formulation_r_value 为 literature！
    is_lit_in_df = pd.Series(False, index=idx, dtype=bool)
    for sc in ("formulation_r_source", "r_source"):
        if sc in df.columns:
            is_lit_in_df = is_lit_in_df | (df[sc].astype(str) == "literature")

    is_lit_in_cp = pd.Series(False, index=idx, dtype=bool)
    if cp is not None and "cp_r_source" in cp.columns:
        is_lit_in_cp = (cp["cp_r_source"].astype(str) == "literature")
    elif "cp_r_source" in df.columns:
        is_lit_in_cp = (df["cp_r_source"].astype(str) == "literature")

    has_explicit_lit_tag = is_lit_in_df | is_lit_in_cp

    # 检查明确的非文献来源标签 (P0-2)
    is_tagged_partial = pd.Series(False, index=idx, dtype=bool)
    is_tagged_unknown = pd.Series(False, index=idx, dtype=bool)
    is_tagged_assumed = pd.Series(False, index=idx, dtype=bool)
    is_tagged_calc = pd.Series(False, index=idx, dtype=bool)

    for sc in ("formulation_r_source", "r_source", "cp_r_source"):
        if sc in df.columns:
            s = df[sc].astype(str)
            is_tagged_partial = is_tagged_partial | (s == "partial_composition_calculated")
            is_tagged_unknown = is_tagged_unknown | (s == "unknown_provenance")
            is_tagged_assumed = is_tagged_assumed | (s == "assumed_stoichiometric")
            is_tagged_calc = is_tagged_calc | (s == "calculated_from_mass")
    if cp is not None and "cp_r_source" in cp.columns:
        s = cp["cp_r_source"].astype(str)
        is_tagged_partial = is_tagged_partial | (s == "partial_composition_calculated")
        is_tagged_unknown = is_tagged_unknown | (s == "unknown_provenance")
        is_tagged_assumed = is_tagged_assumed | (s == "assumed_stoichiometric")
        is_tagged_calc = is_tagged_calc | (s == "calculated_from_mass")

    # 包含布尔标志 (即使 keep_metadata=False 导致字符串来源被裁剪，布尔标志仍完整保留血统 P0-2)
    if "cp_r_unknown" in df.columns:
        is_tagged_unknown = is_tagged_unknown | df["cp_r_unknown"].astype(bool)
    if cp is not None and "cp_r_unknown" in cp.columns:
        is_tagged_unknown = is_tagged_unknown | cp["cp_r_unknown"].astype(bool)

    if "cp_r_partial" in df.columns:
        is_tagged_partial = is_tagged_partial | df["cp_r_partial"].astype(bool)
    if cp is not None and "cp_r_partial" in cp.columns:
        is_tagged_partial = is_tagged_partial | cp["cp_r_partial"].astype(bool)

    prior_assumed_tag = is_tagged_assumed.copy()
    if "rxn_r_assumed" in df.columns:
        assumed_num = pd.to_numeric(df["rxn_r_assumed"], errors="coerce").fillna(0.0)
        prior_assumed_tag = prior_assumed_tag | (assumed_num > 0.5)
    if cp is not None and "rxn_r_assumed" in cp.columns:
        assumed_num_cp = pd.to_numeric(cp["rxn_r_assumed"], errors="coerce").fillna(0.0)
        prior_assumed_tag = prior_assumed_tag | (assumed_num_cp > 0.5)

    # A. 提取独立、明确的原始文献 r 数值 (必须数值与文献证据配对绑定 P0-2)
    raw_lit_r = pd.Series(np.nan, index=idx, dtype=float)
    for col in ("formulation_r_value", "formulation_resin_hardener_equivalent_ratio",
                "crosslink_stoichiometry_r", "stoichiometric_ratio_r_cleaned", "r_value", "r_actual"):
        if col in df.columns:
            val = _parse_num(df[col])
            val = val.where(np.isfinite(val) & (val > 0), np.nan)
            raw_lit_r = raw_lit_r.combine_first(val)

    # 包含带有显式 literature 来源证据的 cp_r_value (P0-1 & P0-2)
    # 必须 cp_r_value 自身的来源也是 literature，才可采信为文献值！绝不能被孤立的 formulation_r_source 挂接 (P0-2)
    if "cp_r_value" in df.columns and "cp_r_source" in df.columns:
        v_cp = _parse_num(df["cp_r_value"])
        v_cp = v_cp.where(np.isfinite(v_cp) & (v_cp > 0), np.nan)
        is_cp_lit = (df["cp_r_source"].astype(str) == "literature")
        raw_lit_r = raw_lit_r.combine_first(v_cp.where(is_cp_lit, np.nan))
    if cp is not None and "cp_r_value" in cp.columns and "cp_r_source" in cp.columns:
        v_cp = _parse_num(cp["cp_r_value"])
        v_cp = v_cp.where(np.isfinite(v_cp) & (v_cp > 0), np.nan)
        is_lit_in_cp = (cp["cp_r_source"].astype(str) == "literature")
        raw_lit_r = raw_lit_r.combine_first(v_cp.where(is_lit_in_cp, np.nan))

    # 获取质量推导 r 供审计参考
    cp_rfm = pd.Series(np.nan, index=idx, dtype=float)
    if cp is not None and "cp_r_from_mass" in cp.columns:
        cp_rfm = _parse_num(cp["cp_r_from_mass"])
    elif "cp_r_from_mass" in df.columns:
        cp_rfm = _parse_num(df["cp_r_from_mass"])

    # 真正的合法独立文献 r 判定 (P0-1 & P0-2):
    # 1. 原始文献列必须有实际有限正数 (raw_lit_r.notna() & raw_lit_r > 0)；
    # 2. 若显式标记为 literature: 只要数值有效，文献事实成立！
    #    (P0-1: 彻底删除以数值相等或接近来判断历史克隆的错误逻辑，真实文献 r 即使与 cp_rfm 完全相同也必须保留！
    #     P0-1 反例B: 即使存在旧的 rxn_r_assumed 缓存，新文献事实优先！)
    # 3. 若未显式标记 literature:
    #    - 配方组成必须完整 (~is_partial_composition)；若配方不完整且无显式文献证据，不可冒充全配方文献值！
    #    - 且未带有任何明确的非文献来源标签 (partial_composition_calculated, unknown_provenance, assumed_stoichiometric, calculated_from_mass)；
    # 4. 严格禁止任何已有 partial_composition_calculated 或 unknown_provenance 升级为 literature！(P0-2)
    # 5. 孤立的 literature 标签但原始文献列中无有效数值，不得放行！(P0-2)
    is_valid_lit_r = (
        raw_lit_r.notna()
        & (raw_lit_r > 0)
        & (
            has_explicit_lit_tag
            | (
                (~is_partial_composition)
                & (~(is_tagged_partial | is_tagged_unknown | prior_assumed_tag | is_tagged_calc))
            )
        )
    )

    # B. 检查当前完整、有效的配方质量计算 r (Priority 2, 优先于历史 assumed 标记 P0-1 反例A)
    calc_mass_r = pd.Series(np.nan, index=idx, dtype=float)
    is_valid_calc_mass_r = pd.Series(False, index=idx, dtype=bool)
    if cp is not None and "cp_r_value" in cp.columns and "cp_r_source" in cp.columns:
        cp_src = cp["cp_r_source"].astype(str)
        cp_val = _parse_num(cp["cp_r_value"])
        is_calc_src = (cp_src == "calculated_from_mass") & (~is_partial_composition)
        is_valid_calc_mass_r = is_calc_src & cp_val.notna() & (cp_val > 0)
        calc_mass_r = cp_val.where(is_valid_calc_mass_r, np.nan)
    elif "cp_r_source" in df.columns and "cp_r_value" in df.columns:
        df_src = df["cp_r_source"].astype(str)
        df_val = _parse_num(df["cp_r_value"])
        is_calc_src = (df_src == "calculated_from_mass") & (~is_partial_composition)
        is_valid_calc_mass_r = is_calc_src & df_val.notna() & (df_val > 0)
        calc_mass_r = df_val.where(is_valid_calc_mass_r, np.nan)

    # C. 检查部分组成计算值 (Priority 3)
    partial_comp_r = pd.Series(np.nan, index=idx, dtype=float)
    is_partial_r = pd.Series(False, index=idx, dtype=bool)
    if cp is not None and "cp_r_value" in cp.columns:
        cp_val = _parse_num(cp["cp_r_value"])
        is_partial_r = is_partial_r | (is_partial_composition & cp_val.notna() & (cp_val > 0))
        partial_comp_r = partial_comp_r.combine_first(cp_val.where(is_partial_r, np.nan))
    if "cp_r_value" in df.columns:
        df_val = _parse_num(df["cp_r_value"])
        is_partial_r = is_partial_r | (is_partial_composition & df_val.notna() & (df_val > 0))
        partial_comp_r = partial_comp_r.combine_first(df_val.where(is_partial_r, np.nan))
    if "formulation_r_value" in df.columns:
        v = _parse_num(df["formulation_r_value"])
        is_part_tagged = is_tagged_partial & v.notna() & (v > 0)
        is_partial_r = is_partial_r | is_part_tagged
        partial_comp_r = partial_comp_r.combine_first(v.where(is_part_tagged, np.nan))

    # D. 检查历史未知来源值 (Priority 5)
    unknown_prov_r = pd.Series(np.nan, index=idx, dtype=float)
    is_unknown_r = pd.Series(False, index=idx, dtype=bool)
    for col in ("formulation_r_value", "cp_r_value", "r_value"):
        if col in df.columns:
            v = _parse_num(df[col])
            is_unk = is_tagged_unknown & v.notna() & (v > 0)
            is_unknown_r = is_unknown_r | is_unk
            unknown_prov_r = unknown_prov_r.combine_first(v.where(is_unk, np.nan))
    if cp is not None and "cp_r_value" in cp.columns:
        v = _parse_num(cp["cp_r_value"])
        is_unk = is_tagged_unknown & v.notna() & (v > 0)
        is_unknown_r = is_unknown_r | is_unk
        unknown_prov_r = unknown_prov_r.combine_first(v.where(is_unk, np.nan))

    # ---- 确定当前有效 r 与来源标签 (四级来源优先级统一判定) ----
    resolved_r = pd.Series(np.nan, index=idx, dtype=float)
    resolved_src = pd.Series("none", index=idx, dtype=object)

    # 1. 真实文献 r 优先 (Priority 1)
    resolved_r = resolved_r.where(~is_valid_lit_r, raw_lit_r)
    resolved_src = resolved_src.where(~is_valid_lit_r, "literature")

    # 2. 当前完整质量计算 r (Priority 2, 优先于历史 assumed 标记 P0-1 反例A)
    has_p2 = (~is_valid_lit_r) & is_valid_calc_mass_r
    resolved_r = resolved_r.where(~has_p2, calc_mass_r)
    resolved_src = resolved_src.where(~has_p2, "calculated_from_mass")

    # 3. 部分组成估计 r (Priority 3)
    has_p3 = (~is_valid_lit_r) & (~has_p2) & is_partial_r
    if assume_stoichiometric_if_missing:
        resolved_r = resolved_r.where(~has_p3, partial_comp_r)
        resolved_src = resolved_src.where(~has_p3, "partial_composition_calculated")
    else:
        # 严格模式下，部分组成不能作为有效完整配方 r (P0-2)
        resolved_r = resolved_r.where(~has_p3, np.nan)
        resolved_src = resolved_src.where(~has_p3, "missing_stoichiometry")

    # 4. 历史未知来源衍生 r (Priority 5)
    has_p5 = (~is_valid_lit_r) & (~has_p2) & (~has_p3) & is_unknown_r
    if assume_stoichiometric_if_missing:
        resolved_r = resolved_r.where(~has_p5, unknown_prov_r)
        resolved_src = resolved_src.where(~has_p5, "unknown_provenance")
    else:
        resolved_r = resolved_r.where(~has_p5, np.nan)
        resolved_src = resolved_src.where(~has_p5, "missing_stoichiometry")

    # 5. 历史假设 r 或无 r 回退 (Priority 4 & 6)
    has_p4_6 = (~is_valid_lit_r) & (~has_p2) & (~has_p3) & (~has_p5)
    if assume_stoichiometric_if_missing:
        # 宽容模式：若前序无有效值，允许等当量假设 1.0
        resolved_r = resolved_r.where(~has_p4_6, 1.0)
        resolved_src = resolved_src.where(~has_p4_6, "assumed_stoichiometric")
    else:
        # 严格模式：前序无有效值，一律置为 NaN
        resolved_r = resolved_r.where(~has_p4_6, np.nan)
        resolved_src = resolved_src.where(~has_p4_6, "missing_stoichiometry")

    r = resolved_r
    r = r.where(np.isfinite(r) & (r > 0), np.nan)

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
    balance = np.minimum(r, 1.0 / r)
    alpha_cure = _estimate_alpha_cure(df)

    # ---- 机制 ----
    mech = pd.Series("unknown", index=idx, dtype=object)
    raw_mech = df.get("curing_mechanism")
    if raw_mech is not None and not isinstance(raw_mech, pd.DataFrame):
        mech = raw_mech.astype(str).str.lower()
    if cp is not None and "cp_mechanism" in cp.columns:
        cp_mech_col = cp["cp_mechanism"].iloc[:, 0] if isinstance(cp["cp_mechanism"], pd.DataFrame) else cp["cp_mechanism"]
        cp_mech = cp_mech_col.astype(str).str.lower()
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

    # 机制默认仅填补已知机制仍缺失者（未知机制保持 NaN，绝不盲目填 3.0）
    f_h_default = mech.map(_FH_BY_MECHANISM)
    f_h_net = f_h_net.where(f_h_net.notna() & (f_h_net > 0), f_h_default)
    f_h_st = f_h_st.where(f_h_st.notna() & (f_h_st > 0), f_h_net)

    # -----------------------------------------------------------------------
    # P0-3 缓存失效与一致性判定:
    # 检查传入的 cp 缓存是否与当前有效 r 处于同一基准 (相同 r)
    # 若当前 r 更新，所有依赖 r 的派生量 (f_avg, W_g, epoxy_conc 等) 必须基于当前 r 重新计算！
    # -----------------------------------------------------------------------
    cp_r = pd.Series(np.nan, index=idx, dtype=float)
    if cp is not None and "cp_r_value" in cp.columns:
        cp_r = _parse_num(cp["cp_r_value"])
    is_r_same_as_cp = cp_r.notna() & r.notna() & ((r - cp_r).abs() < 1e-4)

    # ---- 配方平均官能度 f_avg (优先复用 cp_f_avg_network 或 cp_f_avg，避免别名读取失败回退到错误计算) ----
    f_avg = pd.Series(np.nan, index=idx, dtype=float)
    if cp is not None:
        if "cp_f_avg_network" in cp.columns:
            cp_f_avg = _parse_num(cp["cp_f_avg_network"])
            f_avg = cp_f_avg.where(cp_f_avg.notna(), np.nan)
        elif "cp_f_avg" in cp.columns:
            cp_f_avg = _parse_num(cp["cp_f_avg"])
            f_avg = cp_f_avg.where(cp_f_avg.notna(), np.nan)

    # 严格判断真实摩尔数：仅当两端具备真实质量 (has_mass_r & has_mass_h) 时，直接摩尔加权才有效
    has_real_mass_r = any(c in df.columns for c in ("resin_1_phr", "resin_1_mass", "resin_phr", "resin_mass"))
    has_real_mass_h = any(c in df.columns for c in ("curing_agent_1_phr", "curing_agent_1_mass", "curing_agent_phr", "hardener_phr", "hardener_mass"))
    has_real_mols = bool(has_real_mass_r and has_real_mass_h) & (mol_r > 0) & (mol_h > 0)

    # 当 has_real_mols 为 False 时，f_avg 是从当量比 r 推算的；
    # 当前有效 r 与 cp 缓存生成时的 r 不一致时，旧 cp_f_avg 缓存失效，不可保留！
    f_avg = f_avg.where(has_real_mols | is_r_same_as_cp, np.nan)

    f_avg_calc = compute_molar_average_functionality(
        mol_r=mol_r.where(has_real_mols, np.nan),
        f_r=f_r,
        mol_h=mol_h.where(has_real_mols, np.nan),
        f_h=f_h_net,
        r_val=r,
    )
    f_avg = f_avg.combine_first(f_avg_calc)
    f_avg = f_avg.where(has_real_mols | r.notna(), np.nan)
    # ---- 环氧基摩尔浓度与每环氧基配方质量 (P0-3 缓存失效与重算) ----
    # 物理严密公式: W_g = EEW + r * AHEW
    W_g = eew + r * ahew
    if cp is not None and "cp_W_g_per_epoxy" in cp.columns:
        cp_w = _parse_num(cp["cp_W_g_per_epoxy"])
        # 仅当当前 r 与 cp 缓存生成时的 r 完全一致时，才允许复用 cp_W_g_per_epoxy 兜底缺失的 eew/ahew
        W_g = W_g.where(W_g.notna(), cp_w.where(is_r_same_as_cp, np.nan))
    # 当前有效 r 缺失时，W_g 必须为 NaN！
    W_g = W_g.where(r.notna(), np.nan)

    # epoxy_conc = RHO / W_g
    epoxy_conc = pd.Series(
        np.where(np.isfinite(W_g) & (W_g > 0), RHO / W_g, np.nan), index=idx
    )
    # 不使用历史 cp_epoxy_conc 兜底：它依赖生成时的 rho，且没有可验证的
    # 密度基准。W 缺失时保守 NaN，避免混用旧密度与当前配方。
    epoxy_conc = epoxy_conc.where(r.notna(), np.nan)

    # ---- 理论 ν（机制感知） ----
    # 物理凝胶判据：若环氧端为单官能 (f_r < 1.05)，体系无法形成共价交联网络
    is_monofunctional = (f_r < 1.05) & f_r.notna()
    conn = (f_h_net - 1.0) / f_h_net
    nu_mech = epoxy_conc * conn
    needs_bal = mech.isin(_BALANCE_MECHANISMS)
    nu_mech = nu_mech.where(~needs_bal, nu_mech * balance)
    nu_mech = nu_mech * dilution
    nu_mech = nu_mech.where(~is_monofunctional, np.nan)
    nu_mech = nu_mech.where(r.notna(), np.nan)
    nu_mech = nu_mech.where(np.isfinite(nu_mech) & (nu_mech > 0))

    # ---- junction 口径（Flory 交联点数密度） ----
    with np.errstate(divide="ignore", invalid="ignore"):
        n_r = (1.0 / f_r).replace([np.inf, -np.inf], np.nan)
        n_h = (r / f_h_net).replace([np.inf, -np.inf], np.nan)
        nu_junction = RHO * (n_r * (f_r - 2.0).clip(lower=0.0)
                             + n_h * (f_h_net - 2.0).clip(lower=0.0)) / W_g.replace(0.0, np.nan)
    nu_junction = nu_junction.where(r.notna(), np.nan)
    nu_junction = nu_junction.where(np.isfinite(nu_junction) & (nu_junction > 0))

    # ---- 老口径兜底：聚合列（无逐组分信息时） ----
    eq_density_eew = (1.0e6 / eew).replace([np.inf, -np.inf], np.nan)
    eq_density_ahew = (1.0e6 / ahew).replace([np.inf, -np.inf], np.nan) * np.minimum(
        1.0, 1.0 / r
    )
    eq_density = eq_density_eew.fillna(eq_density_ahew)
    nu_agg = 0.5 * eq_density * balance * dilution
    nu_agg = nu_agg.where(r.notna(), np.nan)
    nu_agg = nu_agg.where(np.isfinite(nu_agg) & (nu_agg > 0))

    # ---- 口径选择与来源标记 ----
    # 物理原则 (P1)：若体系单官能或不可凝胶 (f_r <= 1.0 或 f_h_net <= 1.0 或 alpha_gel 为 NaN)，
    # 无法形成三维网络，有效交联密度 nu 必须为 NaN / 0.0，绝不能通过经验 fallback 赋正值！
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

    # P1-2 覆盖度治理: 存在未识别/不完整反应组分时，绝不允许输出 "full"，降级为 "partial"
    coverage = np.where(is_partial_composition, "partial", coverage)

    # ---- 派生量 ----
    # 临界凝胶转化率: Flory-Stockmayer 凝胶理论 (alpha_gel = 1 / sqrt(r * (f_r - 1) * (f_h_net - 1)))
    # 若单官能 (f_r <= 1 或 f_h_net <= 1)，体系无法共价交联，返回 NaN
    alpha_gel = compute_gel_point_conversion(f_r, f_h_net, r, assume_stoichiometric_if_missing=False)
    alpha_gel = alpha_gel.where(f_r_curated & (f_r > 1.05) & (f_h_net > 1.05) & r.notna(), np.nan)
    # 物理原则 (P1)：若体系单官能、不可凝胶或 r 缺失 (f_r <= 1.0 或 f_h_net <= 1.0 或 alpha_gel 为 NaN 或 r 为 NaN)，
    # 无法形成三维网络，有效交联密度 nu 必须为 NaN / 0.0，绝不能通过经验 fallback 赋正值！
    cannot_gel = (f_r <= 1.001) | (f_h_net <= 1.001) | alpha_gel.isna() | r.isna()
    nu = nu.where(~cannot_gel, np.nan)

    # P1-2: 严格模式下，组成不完整且真实配比不可知时，网络理论量置为 NaN
    if not assume_stoichiometric_if_missing:
        nu = nu.where(~is_partial_composition, np.nan)
        nu_junction = nu_junction.where(~is_partial_composition, np.nan)
        alpha_gel = alpha_gel.where(~is_partial_composition, np.nan)

    # P1-1 & P1-2: 来源标签同步与部分组成代理标记
    has_valid_components = f_r.notna() & (f_r > 0) & f_h_net.notna() & (f_h_net > 0)
    source = np.where(
        r.isna(),
        np.where(has_valid_components, "missing_stoichiometry", "none"),
        np.where(
            cannot_gel & has_valid_components, "non_gelling",
            np.where(
                is_partial_composition,
                np.where(nu.notna(), "partial_composition_proxy", np.where(has_valid_components, "undetermined", "none")),
                np.where(nu.notna(), source, "none")
            )
        )
    )

    alpha_max = np.minimum(1.0, np.minimum(r, 1.0 / r))
    alpha_max = alpha_max.where(r.notna(), np.nan)
    balance = balance.where(r.notna(), np.nan)
    W_g = W_g.where(r.notna(), np.nan)
    epoxy_conc = epoxy_conc.where(r.notna(), np.nan)
    mc = ((1.0e6 * rho_g_cm3) / nu.replace(0.0, np.nan)).clip(50.0, 5000.0)
    mc = mc.where(nu.notna() & r.notna(), np.nan)

    # 确定 r 来源标签 (P0-1 & P0-2)
    # 直接绑定解析阶段确立的来源证据，并在严格模式下同步状态
    r_src = resolved_src
    if not assume_stoichiometric_if_missing:
        r_src = r_src.where(r.notna() | is_valid_lit_r, "missing_stoichiometry")

    out["xl_r_value"] = r
    out["xl_r_source"] = r_src
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
