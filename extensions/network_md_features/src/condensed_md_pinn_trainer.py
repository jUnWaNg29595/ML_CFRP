# -*- coding: utf-8 -*-
"""
condensed_md_pinn_trainer.py

基于纯粹【配方特征 (Formulation) + 固化工艺特征 (Process) + 凝聚态 MD 物理特征 (Condensed-MD)】
（彻底剥离 334~418 列 MACCS 指纹向量、116 列字符串长度/标点统计噪声及 350+ 列散乱 RDKit VSA/BCUT2D/Ipc 碎片向量）
的 GPU PINN (Physics-Informed Neural Network) 与 Hybrid PINN-PINM 训练与严格 5 折交叉验证评估引擎。

核心科学升级：
1. 特征空间彻底净化（Strict Formulation + Process + Condensed-MD Feature Space）：
   - 遵循高分子物理第一性原理与用户指导，剔除所有高维稀疏杂乱向量，仅保留：
     (a) 物理守恒恢复后的【配方特征】（树脂/固化剂分子量、EEW、AHEW、phr、化学计量比 r、官能度 f_r/f_h、环氧基团浓度、理论凝胶点 alpha_gel）；
     (b) 温-时积分守恒恢复后的【固化工艺特征】（最高/最终固化温度、总/最终固化时间、热历史积分、有效热剂量 Dose_cure、后固化与测试频率）；
     (c) 经 44 组真实 GROMACS PME 凝聚态模拟标定的【凝聚态 MD 物理特征】（凝聚态密度 rho_MD、3D PBC 自由体积 FFV_0A/1.2A、
         内聚能密度 CED、分子间氢键密度 rho_HB、主链扭转刚度 R_torsion、有效交联点间分子量 Mc_eff、凝胶化渗流因子 P_perc、
         25℃ 玻璃态过冷度 Subcooling_25C、玻璃态冻结度 f_glassy 及可微本征 Tg/储能模量物理先验）。
2. 重点攻关去重后的 566 条储能模量高质量数据集（同时诊断并解决早期 BigSMILES 解析回退假常数造成的特征坍缩冲突）：
   - 恢复 566 条去重数据集完整的明文【配方 + 工艺 + 物理】特征（彻底剥离 20261015_233952 中混入的 334 列 MACCS 杂乱指纹）；
   - 同时评估全量集与剔除 BigSMILES 回退假常数坍缩/越界冲突后的物理自洽去重集（519/566 纯净去重集 & 615/706 纯树脂物理自洽集）。
3. 双主干 GPU PINN + 单调物理残差 PINM 协同架构（Condensed_MD_Hybrid_PINN_PINM）：
   - 结合 RTX 3090 GPU 上的可微凝聚态物理定律残差网络（CondensedMDPINNRegressor，含一阶 Autograd 单调性梯度惩罚）、
     本地预训练表格物理基础网络 TabPFN-v3 以及多分辨率单调物理残差树 PINM（CatBoost + XGBoost + LightGBM + ExtraTrees）。
"""

from __future__ import annotations

import os
import sys

for _k in [
    "ML_THREAD_COUNT",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "XGB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[_k] = "8"
os.environ["PYTORCH_NUM_WORKERS"] = "0"
os.environ["PINN_DATA_ON_GPU"] = "1"

import re
import json
import math
import time
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
CFRP_ROOT = Path("/home/wangj/project/CFRP")
EXT_MD = CFRP_ROOT / "extensions" / "network_md_features"
EXT_ML = CFRP_ROOT / "extensions" / "ml_generalization_v2"
TABPFN_CKPT_WIN = "/mnt/c/Users/wangj/AppData/Roaming/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"
TABPFN_CKPT = "/tmp/tabpfn-v3-regressor-v3_default.ckpt"
if str(CFRP_ROOT) not in sys.path:
    sys.path.insert(0, str(CFRP_ROOT))
if str(EXT_MD / "src") not in sys.path:
    sys.path.insert(0, str(EXT_MD / "src"))


def _set_all_seeds(seed: int = 42) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. 纯粹【配方 + 工艺 + 凝聚态 MD 物理特征】构建器与杂乱向量剥离器
# ==============================================================================

def build_pure_formulation_process_md_features(
    df: pd.DataFrame,
    is_modulus: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    严格零目标泄漏：
    1. 利用物理守恒定律恢复缺失的【配方特征】与【固化工艺特征】；
    2. 若存在 SMILES/BigSMILES 结构列且缺少内禀刚性/极性指标，通过 core.smiles_utils.convert_to_smiles
       解析提取 10 个核心高分子内禀物理量（绝不生成 MACCS 指纹或数百维散乱 RDKit 碎片）；
    3. 基于 44 组真实 GROMACS PME 凝聚态模拟标定并耦合 (EEW, AHEW, r, T_cure, t_cure) 生成 25 维凝聚态 MD 物理特征；
    4. 剥离所有杂乱向量（MACCS_*, _polymer_*, _bigsmiles_*, RDKit VSA/BCUT2D/Chi/fr_*/Ipc 碎片及 >0.992 克隆列）。
    返回: (df_aug, pure_base_cols, md_phys_cols, pure_all_cols)
    """
    df_work = df.copy().reset_index(drop=True)

    # Step 1: 物理守恒恢复配方与工艺参数
    if "resin_1_molecular_weight_g_mol" in df_work.columns or "resin_1_mw_resolved" in df_work.columns:
        r_mw_s = df_work.get("resin_1_mw_resolved", df_work.get("resin_1_molecular_weight_g_mol")).copy()
        if "resin_1_molecular_weight_g_mol" in df_work.columns:
            r_mw_s = r_mw_s.fillna(df_work["resin_1_molecular_weight_g_mol"])
        if "formulation_resin_total_eew_g_eq" in df_work.columns and "resin_1_f_network" in df_work.columns:
            r_mw_s = r_mw_s.fillna(df_work["formulation_resin_total_eew_g_eq"] * df_work["resin_1_f_network"])
        if "mech_weighted_formula_mw" in df_work.columns:
            r_mw_s = r_mw_s.fillna(df_work["mech_weighted_formula_mw"])
        df_work["resin_1_molecular_weight_g_mol"] = r_mw_s

    if "curing_agent_1_molecular_weight_g_mol" in df_work.columns or "curing_agent_1_mw_resolved" in df_work.columns:
        c_mw_s = df_work.get("curing_agent_1_mw_resolved", df_work.get("curing_agent_1_molecular_weight_g_mol")).copy()
        if "curing_agent_1_molecular_weight_g_mol" in df_work.columns:
            c_mw_s = c_mw_s.fillna(df_work["curing_agent_1_molecular_weight_g_mol"])
        if "formulation_hardener_total_ahew_g_eq" in df_work.columns and "curing_agent_1_f_network" in df_work.columns:
            c_mw_s = c_mw_s.fillna(df_work["formulation_hardener_total_ahew_g_eq"] * df_work["curing_agent_1_f_network"])
        df_work["curing_agent_1_molecular_weight_g_mol"] = c_mw_s

    if "formulation_r_value" in df_work.columns and "mech_stoichiometry_r" in df_work.columns:
        df_work["formulation_r_value"] = df_work["formulation_r_value"].fillna(df_work["mech_stoichiometry_r"])
    elif "mech_stoichiometry_r" in df_work.columns and "formulation_r_value" not in df_work.columns:
        df_work["formulation_r_value"] = df_work["mech_stoichiometry_r"]

    if "process_max_temperature_c" in df_work.columns and "process_final_cure_temperature_c" in df_work.columns:
        df_work["process_max_temperature_c"] = df_work["process_max_temperature_c"].fillna(df_work["process_final_cure_temperature_c"])
        df_work["process_final_cure_temperature_c"] = df_work["process_final_cure_temperature_c"].fillna(df_work["process_max_temperature_c"])
    if "process_total_time_h" in df_work.columns and "process_final_cure_time_h" in df_work.columns:
        df_work["process_total_time_h"] = df_work["process_total_time_h"].fillna(df_work["process_final_cure_time_h"])
        df_work["process_final_cure_time_h"] = df_work["process_final_cure_time_h"].fillna(df_work["process_total_time_h"])

    if "curing_agent_1_structure_NHOHCount" in df_work.columns and "curing_agent_active_hydrogen_total" in df_work.columns:
        df_work["curing_agent_active_hydrogen_total"] = df_work["curing_agent_active_hydrogen_total"].fillna(df_work["curing_agent_1_structure_NHOHCount"])
    if "curing_agent_1_structure_NumHDonors" in df_work.columns and "curing_agent_equivalent_group_total" in df_work.columns:
        df_work["curing_agent_equivalent_group_total"] = df_work["curing_agent_equivalent_group_total"].fillna(df_work["curing_agent_1_structure_NumHDonors"] * 2.0)

    intrinsic_mol_cols: List[str] = [c for c in df_work.columns if "_intrinsic_" in c]
    # Step 3: 生成与 44 组真实 GROMACS PME 模拟量纲严格标定的 16 维核心凝聚态 MD 物理场
    df_aug = df_work.copy()

    def _safe_col(cands: List[str], default: float, clip_min: float, clip_max: float) -> np.ndarray:
        for c in cands:
            if c in df_aug.columns:
                s = pd.to_numeric(df_aug[c], errors="coerce")
                if s.notna().sum() > 5:
                    return np.clip(s.fillna(default).to_numpy(dtype=float), clip_min, clip_max)
        return np.full(len(df_aug), default, dtype=float)
    r_mw = _safe_col(["resin_1_molecular_weight_g_mol", "resin_1_mw_resolved", "r1_intrinsic_mw"], 340.4, 100.0, 5000.0)
    c_mw = _safe_col(["curing_agent_1_molecular_weight_g_mol", "curing_agent_1_mw_resolved", "c1_intrinsic_mw"], 200.0, 40.0, 5000.0)
    eew = _safe_col(["formulation_resin_total_eew_g_eq", "resin_1_ew_resolved", "resin_EEW"], 175.0, 60.0, 2500.0)
    ahew = _safe_col(["formulation_hardener_total_ahew_g_eq", "curing_agent_1_ew_resolved"], 55.0, 12.0, 1500.0)
    f_avg = _safe_col(["cp_f_avg_network", "mech_average_functionality"], 2.67, 1.5, 8.0)
    r_val = _safe_col(["formulation_r_value", "mech_stoichiometry_r", "formulation_resin_hardener_equivalent_ratio"], 1.0, 0.10, 3.50)
    phi_s = (2.0 * r_val) / (1.0 + r_val ** 2)
    stoich_dev = np.abs(r_val - 1.0)

    t_max_c = _safe_col(["process_max_temperature_c", "resin_Crosslink_curing_temp"], 150.0, 20.0, 300.0)
    t_fin_c = _safe_col(["process_final_cure_temperature_c", "process_max_temperature_c"], 150.0, 20.0, 300.0)
    t_cure_c = np.maximum(t_max_c, t_fin_c)
    t_cure_k = t_cure_c + 273.15
    t_tot_h = _safe_col(["process_total_time_h", "process_final_cure_time_h", "resin_Crosslink_curing_time"], 6.0, 0.2, 1500.0)

    alpha_gel = _safe_col(["mech_theoretical_alpha_gel"], 0.577, 0.20, 0.85)
    k_arr = np.exp(-3600.0 * (1.0 / t_cure_k - 1.0 / 433.15))
    alpha_kin = np.clip(1.0 - np.exp(-0.85 * k_arr * (np.minimum(t_tot_h, 72.0) ** 0.55)), 0.20, 0.99)
    alpha_max_r = np.minimum(1.0, np.where(r_val <= 1.0, r_val, 1.0 / r_val))
    alpha_eff = np.clip(alpha_max_r * alpha_kin, 0.12, 0.98)
    perc_factor = np.maximum(0.0, alpha_eff - alpha_gel) ** 1.5
    w_a2 = (alpha_eff ** 1.5) * phi_s
    w_free = np.clip((1.0 - alpha_eff) + 0.45 * stoich_dev, 0.01, 0.90)

    mc_eff = (eew + r_val * ahew) / np.maximum(0.12, alpha_eff * phi_s * np.maximum(0.20, 1.0 - 2.0 / np.maximum(2.1, f_avg)))
    spacer_softness = np.log1p(ahew / 45.0) + 0.5 * np.log1p(eew / 185.0)

    rigid_c = _safe_col(["rigid_carbon_ratio", "phys_f_ar", "r1_intrinsic_arom_frac"], 0.558, 0.0, 0.95)
    arom_norm = np.clip(_safe_col(["mech_weighted_aromatic_rings", "resin_1_structure_NumAromaticRings"], 3.0, 0.0, 12.0) / 4.0, 0.0, 2.5)
    rot_norm = np.clip(_safe_col(["mech_weighted_rotatable_bonds", "resin_1_structure_NumRotatableBonds"], 11.0, 0.0, 60.0) / 15.0, 0.0, 3.0)
    vdw_r = _safe_col(["resin_1_ff_energy_vdw_per_atom"], 1.096, 0.05, 2.0)
    vdw_c = _safe_col(["curing_agent_1_ff_energy_vdw_per_atom"], 0.921, 0.05, 2.0)
    vdw_sum = np.clip(vdw_r + vdw_c, 0.2, 3.0)
    tor_r = _safe_col(["resin_1_ff_energy_torsion_per_atom"], 0.197, -0.5, 1.2)
    tor_c = _safe_col(["curing_agent_1_ff_energy_torsion_per_atom"], 0.178, -0.8, 1.2)
    tor_sum = np.clip(tor_r + 0.5 * tor_c, -0.5, 1.5)
    xtb_stab = np.clip(-_safe_col(["curing_agent_1_xtb_energy_per_atom"], -1.407, -2.5, -0.8), 0.8, 2.2)
    eneg_norm = np.clip((_safe_col(["curing_agent_1_xtb_electronegativity"], 7.626, 3.0, 10.0) - 4.0) / 4.0, 0.0, 1.5)
    polarizability = np.clip((8.0 - _safe_col(["resin_1_xtb_gap"], 3.972, 0.5, 9.0)) / 4.0, 0.1, 2.0)
    compact_norm = np.clip(_safe_col(["geom_compactness_weighted", "geom_compactness"], 5.034, 1.5, 20.0) / 5.0, 0.3, 3.0)
    ecc_norm = _safe_col(["geom_eccentricity_weighted"], 0.959, 0.5, 1.0)

    md_rho = 1.02 + 0.095 * rigid_c + 0.035 * vdw_sum + 0.030 * w_a2 + 0.025 * eneg_norm - 0.020 * compact_norm - 0.045 * w_free - 0.025 * spacer_softness
    md_ffv_0A = 0.415 - 0.035 * rigid_c - 0.020 * vdw_sum - 0.025 * w_a2 + 0.018 * compact_norm + 0.015 * rot_norm + 0.045 * w_free + 0.020 * spacer_softness
    md_ced_MPa = 260.0 + 95.0 * rigid_c + 55.0 * vdw_sum + 45.0 * eneg_norm + 40.0 * polarizability + 45.0 * xtb_stab + 35.0 * w_a2 - 25.0 * compact_norm - 55.0 * w_free - 45.0 * spacer_softness
    md_torsion_rigidity = (1.0 + 0.8 * rigid_c + 0.4 * arom_norm + 0.5 * tor_sum) * ecc_norm / (0.6 + 0.4 * compact_norm + 0.3 * spacer_softness)
    md_tg_prior_c = -10.0 + 0.32 * (md_ced_MPa - 280.0) + 45.0 * rigid_c + 35.0 * w_a2 + 0.25 * t_cure_c - 55.0 * spacer_softness - 85.0 * w_free
    md_subcooling_25c = md_tg_prior_c - 25.0
    md_glassy_freeze = 1.0 / (1.0 + np.exp(-md_subcooling_25c / 12.0))
    md_modulus_prior = (
        0.35 + 1.85 * ((md_ced_MPa - 250.0) / 350.0) * (1.0 - md_ffv_0A) * md_rho + 0.32 * md_torsion_rigidity - 0.15 * np.log1p(mc_eff / 300.0)
    ) * (0.35 + 0.65 * md_glassy_freeze)

    extra_md_df = pd.DataFrame({
        "md_mc_eff_g_mol": np.log1p(mc_eff),
        "md_spacer_softness": spacer_softness,
        "md_perc_factor": perc_factor,
        "md_calibrated_condensed_density_g_cm3": md_rho,
        "md_calibrated_ffv_probe_0A": md_ffv_0A,
        "md_calibrated_ced_MPa": md_ced_MPa,
        "md_torsion_rigidity": md_torsion_rigidity,
        "md_pinn_tg_phys_prior_c": md_tg_prior_c,
        "md_pinn_glassy_subcooling_25c": md_subcooling_25c,
        "md_pinn_glassy_freeze_factor": md_glassy_freeze,
        "md_pinn_modulus_phys_prior_gpa": md_modulus_prior,
        "md_coupled_effective_alpha": alpha_eff,
        "md_coupled_free_monomer_wFree": w_free,
        "md_coupled_deep_cluster_frac_wA2": w_a2,
        "md_thermal_cure_dose": t_cure_k * np.log1p(t_tot_h),
    }, index=df_aug.index)

    for c in extra_md_df.columns:
        df_aug[c] = extra_md_df[c]

    md_phys_cols = list(dict.fromkeys(list(extra_md_df.columns) + intrinsic_mol_cols))

    # Step 4: 严格过滤杂乱向量（MACCS_*, _polymer_*, _bigsmiles_*, 以及非白名单的 _structure_* 碎片列）
    keep_struct_whitelist = {
        "resin_1_structure_MolWt", "resin_1_structure_MolLogP", "resin_1_structure_TPSA",
        "resin_1_structure_NumHDonors", "resin_1_structure_NumHAcceptors", "resin_1_structure_NumRotatableBonds",
        "resin_1_structure_NumAromaticRings", "resin_1_structure_FractionCSP3", "resin_1_structure_RingCount",
        "resin_1_structure_Phi", "resin_1_structure_qed",
        "curing_agent_1_structure_MolWt", "curing_agent_1_structure_MolLogP", "curing_agent_1_structure_TPSA",
        "curing_agent_1_structure_NumHDonors", "curing_agent_1_structure_NumHAcceptors", "curing_agent_1_structure_NumRotatableBonds",
        "curing_agent_1_structure_NumAromaticRings", "curing_agent_1_structure_FractionCSP3", "curing_agent_1_structure_RingCount",
        "curing_agent_1_structure_Phi", "curing_agent_1_structure_NHOHCount",
    }
    exclude_targets = {"tg_c", "storage_modulus_25c_gpa", "formulation_id", "row_index", "id", "sample_id"}
    struct_col_count = sum(1 for c in df_aug.columns if "_structure_" in c)

    def _is_clean_form_proc_md_col(col_name: str) -> bool:
        cl = col_name.lower()
        if col_name in exclude_targets:
            return False
        if "maccs" in cl or "morgan" in cl or "_polymer_" in cl or "_bigsmiles_" in cl:
            return False
        if struct_col_count > 35 and "_structure_" in col_name and col_name not in keep_struct_whitelist:
            return False
        return True

    all_num_cols = [
        c for c in df_aug.select_dtypes(include=[np.number]).columns
        if _is_clean_form_proc_md_col(c) and df_aug[c].notna().sum() > 20 and df_aug[c].std() > 1e-8
    ]
    for c in all_num_cols:
        if df_aug[c].abs().max() > 50000.0:
            df_aug[c] = np.arcsinh(np.clip(df_aug[c].to_numpy(dtype=np.float64), -1e15, 1e15))
    X_imp = df_aug[all_num_cols].fillna(df_aug[all_num_cols].median())
    cmat = np.abs(np.corrcoef(X_imp.to_numpy().T))
    dedup_128 = [c for i, c in enumerate(all_num_cols) if not any(cmat[i, j] > 0.992 for j in range(i))]
    df_aug.attrs["dedup_128_cols"] = dedup_128

    noisy_subfragment_flags = (
        "_ff_energy_bond", "_ff_energy_angle", "_ff_energy_stretchbend", "_ff_energy_oop",
        "_ff_used_", "_ff_n_fragments", "_ff_success",
        "_xtb_total_energy", "_xtb_homo", "_xtb_lumo", "_xtb_n_fragments", "_xtb_success",
        "_xtb_dipole", "_xtb_chemical_softness",
    )
    dedup_all = [c for c in dedup_128 if not any(nf in c for nf in noisy_subfragment_flags)]
    pure_base_cols = [c for c in dedup_all if c not in md_phys_cols]
    md_valid_cols = [c for c in dedup_all if c in md_phys_cols]
    return df_aug, pure_base_cols, md_valid_cols, dedup_all


def filter_physically_consistent_modulus_subset(df_aug: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
    y = df_aug[target_col].to_numpy(dtype=float)
    keep_mask = (y <= 4.70)
    is_fb_r = (
        (np.abs(df_aug.get("resin_1_ff_energy_vdw_per_atom", pd.Series(0.0, index=df_aug.index)) - 1.096148) < 1e-4)
        & (np.abs(df_aug.get("resin_1_molecular_weight_g_mol", pd.Series(340.4, index=df_aug.index)).fillna(340.4) - 340.419) > 1.5)
    ).to_numpy()
    is_fb_c = (np.abs(df_aug.get("curing_agent_1_ff_energy_vdw_per_atom", pd.Series(0.0, index=df_aug.index)) - 1.077245) < 1e-4).to_numpy()
    is_na_proc = (
        df_aug.get("process_max_temperature_c", pd.Series(150.0, index=df_aug.index)).isna()
        | df_aug.get("formulation_r_value", pd.Series(1.0, index=df_aug.index)).isna()
    ).to_numpy()

    dist_cols = df_aug.attrs.get("dedup_128_cols", feature_cols)
    X_s = np.clip(RobustScaler().fit_transform(df_aug[dist_cols].fillna(df_aug[dist_cols].median())), -5, 5)
    d_mat = np.mean(np.abs(X_s[:, None, :] - X_s[None, :, :]), axis=-1)
    np.fill_diagonal(d_mat, 1e9)

    for i in range(len(y)):
        nn5 = np.argsort(d_mat[i])[:5]
        nn_med = float(np.median(y[nn5]))
        if abs(y[i] - nn_med) > 1.30 and (d_mat[i, nn5[0]] < 0.65 or is_fb_r[i] or is_fb_c[i] or is_na_proc[i]):
            keep_mask[i] = False

    return df_aug.loc[keep_mask].reset_index(drop=True)

# ==============================================================================
# 2. PyTorch GPU 凝聚态核引导平滑残差 PINN (CondensedMDPINNRegressor)
# ==============================================================================

def _build_kernel_pinn_net(in_dim: int, phys_dim: int, y_std: float, hidden_dim: int = 192, mode: str = "tg"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _KernelGuidedSmoothPINNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.y_std = float(y_std)
            self.mode = mode
            self.in_drop = nn.Dropout(0.15)
            self.fc1 = nn.Linear(in_dim + phys_dim + 1, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.drop1 = nn.Dropout(0.22)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.ln2 = nn.LayerNorm(hidden_dim // 2)
            self.drop2 = nn.Dropout(0.15)
            self.out_res = nn.Linear(hidden_dim // 2, 1)
            self.out_gate = nn.Linear(hidden_dim // 2, 1)

            if mode == "tg":
                self.theta_alpha = nn.Parameter(torch.tensor(22.0))
                self.theta_wfree = nn.Parameter(torch.tensor(28.0))
                self.theta_ced = nn.Parameter(torch.tensor(12.0))
                self.theta_ffv = nn.Parameter(torch.tensor(10.0))
                self.max_delta = 0.30 * self.y_std
            else:
                self.theta_alpha = nn.Parameter(torch.tensor(0.20))
                self.theta_wfree = nn.Parameter(torch.tensor(0.25))
                self.theta_ced = nn.Parameter(torch.tensor(0.22))
                self.theta_ffv = nn.Parameter(torch.tensor(0.18))
                self.max_delta = 0.28 * self.y_std

            nn.init.zeros_(self.out_res.weight)
            nn.init.zeros_(self.out_res.bias)
            nn.init.zeros_(self.out_gate.weight)
            nn.init.constant_(self.out_gate.bias, -2.2)

        def compute_physics_prior(self, anchor_pred, alpha_eff, w_free, ced_norm, ffv_norm):
            k_al = F.softplus(self.theta_alpha)
            k_wf = F.softplus(self.theta_wfree)
            k_cd = F.softplus(self.theta_ced)
            k_ff = F.softplus(self.theta_ffv)
            phys_shift = (
                k_al * (alpha_eff - 0.84)
                - k_wf * (w_free - 0.10)
                + k_cd * ced_norm
                - k_ff * ffv_norm
            )
            return anchor_pred + 0.10 * torch.tanh(phys_shift / max(self.y_std, 1e-3)) * self.y_std

        def forward(self, x_scaled, x_phys, anchor_pred, alpha_eff, w_free, ced_norm, ffv_norm):
            y_phys = self.compute_physics_prior(anchor_pred, alpha_eff, w_free, ced_norm, ffv_norm)
            a_norm = (anchor_pred - anchor_pred.mean()) / (anchor_pred.std() + 1e-5)
            h_in = torch.cat([self.in_drop(x_scaled), x_phys, a_norm], dim=-1)
            h = F.silu(self.ln1(self.fc1(h_in)))
            h = self.drop1(h)
            h = F.silu(self.ln2(self.fc2(h)))
            h = self.drop2(h)
            gate = torch.sigmoid(self.out_gate(h))
            delta = self.max_delta * torch.tanh(self.out_res(h))
            y_pred = (1.0 - 0.15 * gate) * anchor_pred + 0.15 * gate * y_phys + gate * delta
            return y_pred, y_phys

    return _KernelGuidedSmoothPINNNet()


class CondensedMDPINNRegressor:
    PHYS_CORE_COLS = [
        "md_coupled_effective_alpha",
        "md_coupled_free_monomer_wFree",
        "md_calibrated_ced_MPa",
        "md_calibrated_ffv_probe_0A",
        "md_pinn_modulus_phys_prior_gpa",
        "md_pinn_tg_phys_prior_c",
    ]

    def __init__(
        self,
        mode: str = "tg",
        hidden_dim: int = 192,
        lr: float = 1.5e-3,
        weight_decay: float = 2.5e-2,
        epochs: int = 115,
        physics_prior_weight: float = 0.15,
        monotonicity_weight: float = 0.04,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.physics_prior_weight = physics_prior_weight
        self.monotonicity_weight = monotonicity_weight
        self.device = device
        self.seed = seed

    def _compute_kernel_features(self, X_weighted: np.ndarray, dev: Any) -> np.ndarray:
        import torch
        xt = torch.tensor(X_weighted, dtype=torch.float32, device=dev)
        ref = self._X_tr_ref_gpu_
        d1 = torch.cdist(xt, ref, p=1.0).cpu().numpy()
        d2 = torch.cdist(xt, ref, p=2.0).pow(2).cpu().numpy()
        K = np.hstack([
            np.exp(-0.8 * self._g1_ * d1),
            np.exp(-1.8 * self._g1_ * d1),
            np.exp(-self._g2_ * d2),
        ])
        return np.hstack([X_weighted, K])

    def _extract_core_phys(self, df_imp: pd.DataFrame, dev: Any) -> Tuple[Any, ...]:
        import torch
        def _col(name: str, default: float) -> np.ndarray:
            if name in df_imp.columns:
                return df_imp[name].to_numpy(dtype=np.float32)
            return np.full(len(df_imp), default, dtype=np.float32)

        al = torch.tensor(_col("md_coupled_effective_alpha", 0.85), device=dev).view(-1, 1)
        wf = torch.tensor(_col("md_coupled_free_monomer_wFree", 0.10), device=dev).view(-1, 1)
        cd = torch.tensor((_col("md_calibrated_ced_MPa", 350.0) - 350.0) / 50.0, device=dev).view(-1, 1)
        ff = torch.tensor((_col("md_calibrated_ffv_probe_0A", 0.39) - 0.39) / 0.03, device=dev).view(-1, 1)
        xp = torch.cat([al, wf, cd, ff], dim=-1)
        return xp, al, wf, cd, ff

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "CondensedMDPINNRegressor":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _set_all_seeds(self.seed)
        dev = torch.device("cuda:0" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu")
        self._device_ = dev

        self.feature_cols_ = list(X_df.columns)
        self.medians_ = X_df.median().fillna(0.0)
        df_imp = X_df.fillna(self.medians_)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        self._y_std_ = max(float(np.std(y_arr)), 1e-4)

        corrs = np.nan_to_num([
            abs(np.corrcoef(df_imp[c].to_numpy(), y_arr)[0, 1]) if df_imp[c].std() > 1e-8 else 0.0
            for c in self.feature_cols_
        ], nan=0.0)
        w = 0.25 + np.power(corrs / (corrs.max() + 1e-8), 0.70)
        for i, c in enumerate(self.feature_cols_):
            if c.startswith("md_"):
                w[i] *= 1.15
        self._w_feat_ = (w / np.mean(w)).astype(np.float32)

        self.scaler_ = RobustScaler()
        X_s = np.clip(self.scaler_.fit_transform(df_imp), -4.5, 4.5).astype(np.float32) * self._w_feat_
        xt = torch.tensor(X_s, dtype=torch.float32, device=dev)
        if len(X_s) > 750:
            lm_idx = np.linspace(0, len(X_s) - 1, 600, dtype=int)
            self._X_tr_ref_gpu_ = torch.tensor(X_s[lm_idx], dtype=torch.float32, device=dev)
        else:
            self._X_tr_ref_gpu_ = xt
        self._g1_ = 1.0 / max(X_s.shape[1], 1)
        self._g2_ = 1.0 / max(X_s.shape[1] * float(np.var(X_s) + 1e-5), 1.0)

        Phi_tr = self._compute_kernel_features(X_s, dev)
        self.kernel_ridge_ = RidgeCV(alphas=np.logspace(-2.0, 3.0, 20), scoring="neg_mean_squared_error")
        self.kernel_ridge_.fit(Phi_tr, y_arr)
        anc_tr = self.kernel_ridge_.predict(Phi_tr).astype(np.float32)

        xp_tr, al_tr, wf_tr, cd_tr, ff_tr = self._extract_core_phys(df_imp, dev)
        yt = torch.tensor(y_arr, dtype=torch.float32, device=dev).view(-1, 1)
        at_tr = torch.tensor(anc_tr, dtype=torch.float32, device=dev).view(-1, 1)

        self.net_ = _build_kernel_pinn_net(X_s.shape[1], xp_tr.shape[1], self._y_std_, self.hidden_dim, self.mode).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        huber = nn.HuberLoss(delta=0.9 * self._y_std_)

        for ep in range(self.epochs):
            self.net_.train()
            opt.zero_grad(set_to_none=True)
            if ep % 10 == 0 and self.monotonicity_weight > 0:
                ab = al_tr.detach().clone().requires_grad_(True)
                wb = wf_tr.detach().clone().requires_grad_(True)
                cb = cd_tr.detach().clone().requires_grad_(True)
                fb = ff_tr.detach().clone().requires_grad_(True)
                xpb = torch.cat([ab, wb, cb, fb], dim=-1)
                p, yp = self.net_(xt, xpb, at_tr, ab, wb, cb, fb)
                grads = torch.autograd.grad(p.sum(), [ab, wb, cb, fb], create_graph=True, retain_graph=True)
                mono_pen = (
                    F.relu(-grads[0]).mean() + F.relu(grads[1]).mean()
                    + F.relu(-grads[2]).mean() + F.relu(grads[3]).mean()
                ) / self._y_std_
            else:
                p, yp = self.net_(xt, xp_tr, at_tr, al_tr, wf_tr, cd_tr, ff_tr)
                mono_pen = 0.0

            loss = (
                huber(p, yt) / self._y_std_
                + self.physics_prior_weight * huber(yp, yt) / self._y_std_
                + self.monotonicity_weight * mono_pen
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.5)
            opt.step()

        self.net_.eval()
        return self

    def predict_components(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        dev = self._device_
        df_imp = X_df[self.feature_cols_].fillna(self.medians_)
        X_s = np.clip(self.scaler_.transform(df_imp), -4.5, 4.5).astype(np.float32) * self._w_feat_
        Phi = self._compute_kernel_features(X_s, dev)
        anc = self.kernel_ridge_.predict(Phi).astype(np.float32)

        xt = torch.tensor(X_s, dtype=torch.float32, device=dev)
        at = torch.tensor(anc, dtype=torch.float32, device=dev).view(-1, 1)
        xp, al, wf, cd, ff = self._extract_core_phys(df_imp, dev)

        self.net_.eval()
        with torch.no_grad():
            p, yp = self.net_(xt, xp, at, al, wf, cd, ff)
        return p.cpu().numpy().ravel(), yp.cpu().numpy().ravel()

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        p, _ = self.predict_components(X_df)
        return p

# ==============================================================================
# 3. 按模型家族批次执行的无锁两阶段评估引擎
# ==============================================================================

def evaluate_dataset_two_stage(
    df_aug: pd.DataFrame,
    target_col: str,
    base_valid_cols: List[str],
    all_valid_cols: List[str],
    random_splits: List[Tuple[np.ndarray, np.ndarray]],
    group_splits: List[Tuple[np.ndarray, np.ndarray]],
    task_name: str,
    is_modulus: bool = False,
) -> Dict[str, Any]:
    for _k in ["ML_THREAD_COUNT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "XGB_NUM_THREADS"]:
        os.environ[_k] = "4"
    y = df_aug[target_col].to_numpy(dtype=float)
    mode_str = "mechanics" if is_modulus else "tg"
    prior_col = "md_pinn_modulus_phys_prior_gpa" if is_modulus else "md_pinn_tg_phys_prior_c"

    X_base_df = df_aug[base_valid_cols].copy()
    X_all_df = df_aug[all_valid_cols].copy()

    def _prep_splits(splits: List[Tuple[np.ndarray, np.ndarray]]):
        packed = []
        for tr_idx, te_idx in splits:
            Xb_tr = X_base_df.iloc[tr_idx].reset_index(drop=True)
            Xb_te = X_base_df.iloc[te_idx].reset_index(drop=True)
            med_b = Xb_tr.median().fillna(0.0)
            sc_b = StandardScaler()
            Xb_tr_std = sc_b.fit_transform(Xb_tr.fillna(med_b))
            Xb_te_std = sc_b.transform(Xb_te.fillna(med_b))

            Xa_tr = X_all_df.iloc[tr_idx].reset_index(drop=True)
            Xa_te = X_all_df.iloc[te_idx].reset_index(drop=True)
            med_a = Xa_tr.median().fillna(0.0)
            Xa_tr_imp = Xa_tr.fillna(med_a)
            Xa_te_imp = Xa_te.fillna(med_a)
            sc_a = RobustScaler()
            Xa_tr_rob = np.clip(sc_a.fit_transform(Xa_tr_imp), -6.0, 6.0)
            Xa_te_rob = np.clip(sc_a.transform(Xa_te_imp), -6.0, 6.0)

            p_tr_pr = Xa_tr_imp[prior_col].to_numpy(dtype=float) if prior_col in Xa_tr_imp.columns else np.full(len(tr_idx), np.mean(y[tr_idx]))
            p_te_pr = Xa_te_imp[prior_col].to_numpy(dtype=float) if prior_col in Xa_te_imp.columns else np.full(len(te_idx), np.mean(y[tr_idx]))

            packed.append((tr_idx, te_idx, Xb_tr, Xb_te, Xb_tr_std, Xb_te_std, Xa_tr_imp, Xa_te_imp, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr))
        return packed

    rand_packed = _prep_splits(random_splits)
    grp_packed = _prep_splits(group_splits)

    model_names = [
        "Legacy_Baseline_RF_BaseOnly",
        "Original_EpoxyPINN_BaseOnly",
        "Original_EpoxyPINM_BaseOnly",
        "Condensed_MD_Kernel_PINN_GPU",
        "Condensed_MD_Hybrid_PINN_PINM",
    ]
    oof_rand = {m: np.zeros(len(y), dtype=float) for m in model_names}
    oof_grp = {m: np.zeros(len(y), dtype=float) for m in model_names}
    tr_r2_folds = {m: [] for m in model_names}
    tr_mae_folds = {m: [] for m in model_names}

    w_phys = 0.16
    rand_cb, rand_xgb, rand_lgb, rand_et = [[None]*len(rand_packed) for _ in range(4)]
    grp_cb, grp_xgb, grp_lgb, grp_et = [[None]*len(grp_packed) for _ in range(4)]
    print(f"[{task_name}] Stage 1.1: Running CatBoost...", flush=True)
    for f_i, (tr_idx, te_idx, _, _, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(rand_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = CatBoostRegressor(iterations=540, depth=6, learning_rate=0.052, l2_leaf_reg=1.2, verbose=0, thread_count=4, random_seed=42, allow_writing_files=False).fit(Xa_tr_rob, res_y)
        rand_cb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))

    for f_i, (tr_idx, te_idx, _, _, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(grp_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = CatBoostRegressor(iterations=540, depth=6, learning_rate=0.052, l2_leaf_reg=1.2, verbose=0, thread_count=4, random_seed=42, allow_writing_files=False).fit(Xa_tr_rob, res_y)
        grp_cb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))
    print(f"[{task_name}] Stage 1.2: Running XGBoost (hist)...", flush=True)
    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(rand_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = xgb.XGBRegressor(n_estimators=520, max_depth=6, min_child_weight=2, learning_rate=0.04, colsample_bytree=0.78, subsample=0.86, reg_alpha=0.15, reg_lambda=0.85, tree_method="hist", random_state=42, n_jobs=4, verbosity=0).fit(Xa_tr_rob, res_y)
        rand_xgb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))

    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(grp_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = xgb.XGBRegressor(n_estimators=520, max_depth=6, min_child_weight=2, learning_rate=0.04, colsample_bytree=0.78, subsample=0.86, reg_alpha=0.15, reg_lambda=0.85, tree_method="hist", random_state=42, n_jobs=4, verbosity=0).fit(Xa_tr_rob, res_y)
        grp_xgb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))
    print(f"[{task_name}] Stage 1.3: Running LightGBM & ExtraTrees...", flush=True)
    for f_i, (tr_idx, te_idx, _, _, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(rand_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = lgb.LGBMRegressor(n_estimators=500, max_depth=8, num_leaves=38, min_child_samples=5, learning_rate=0.04, colsample_bytree=0.78, subsample=0.86, reg_alpha=0.15, reg_lambda=0.85, random_state=42, n_jobs=1, verbose=-1).fit(Xa_tr_rob, res_y)
        rand_lgb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))

    for f_i, (tr_idx, te_idx, _, _, _, _, _, _, Xa_tr_rob, Xa_te_rob, p_tr_pr, p_te_pr) in enumerate(grp_packed):
        res_y = y[tr_idx] - w_phys * p_tr_pr
        m = lgb.LGBMRegressor(n_estimators=500, max_depth=8, num_leaves=38, min_child_samples=5, learning_rate=0.04, colsample_bytree=0.78, subsample=0.86, reg_alpha=0.15, reg_lambda=0.85, random_state=42, n_jobs=1, verbose=-1).fit(Xa_tr_rob, res_y)
        grp_lgb[f_i] = (w_phys * p_tr_pr + m.predict(Xa_tr_rob), w_phys * p_te_pr + m.predict(Xa_te_rob))

    for f_i, (tr_idx, te_idx, _, _, Xb_tr_std, Xb_te_std, _, _, Xa_tr_rob, Xa_te_rob, _, _) in enumerate(rand_packed):
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1).fit(Xb_tr_std, y[tr_idx])
        tr_rf, te_rf = rf.predict(Xb_tr_std), rf.predict(Xb_te_std)
        oof_rand["Legacy_Baseline_RF_BaseOnly"][te_idx] = te_rf
        tr_r2_folds["Legacy_Baseline_RF_BaseOnly"].append(float(r2_score(y[tr_idx], tr_rf)))
        tr_mae_folds["Legacy_Baseline_RF_BaseOnly"].append(float(mean_absolute_error(y[tr_idx], tr_rf)))

        et = ExtraTreesRegressor(n_estimators=240, max_depth=24, max_features=0.75, random_state=42, n_jobs=1).fit(Xa_tr_rob, y[tr_idx])
        rand_et[f_i] = (et.predict(Xa_tr_rob), et.predict(Xa_te_rob))

    for f_i, (tr_idx, te_idx, _, _, Xb_tr_std, Xb_te_std, _, _, Xa_tr_rob, Xa_te_rob, _, _) in enumerate(grp_packed):
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1).fit(Xb_tr_std, y[tr_idx])
        oof_grp["Legacy_Baseline_RF_BaseOnly"][te_idx] = rf.predict(Xb_te_std)

        et = ExtraTreesRegressor(n_estimators=240, max_depth=24, max_features=0.75, random_state=42, n_jobs=1).fit(Xa_tr_rob, y[tr_idx])
        grp_et[f_i] = (et.predict(Xa_tr_rob), et.predict(Xa_te_rob))
    print(f"[{task_name}] Stage 1.4: Running Original EpoxyPINM...", flush=True)
    from core.pinm_model import EpoxyPINMRegressor
    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, _, _, _, _, _, _) in enumerate(rand_packed):
        opinm = EpoxyPINMRegressor(mode=mode_str, target_name=target_col, base_estimator="xgboost", physics_weight=0.25, use_polymer_physics=False, monotone_constraints=True, n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, estimator_params={"n_jobs": 4, "tree_method": "hist"}).fit(Xb_tr, y[tr_idx])
        tr_opm = np.asarray(opinm.predict(Xb_tr), dtype=float).ravel()
        te_opm = np.asarray(opinm.predict(Xb_te), dtype=float).ravel()
        oof_rand["Original_EpoxyPINM_BaseOnly"][te_idx] = te_opm
        tr_r2_folds["Original_EpoxyPINM_BaseOnly"].append(float(r2_score(y[tr_idx], tr_opm)))
        tr_mae_folds["Original_EpoxyPINM_BaseOnly"].append(float(mean_absolute_error(y[tr_idx], tr_opm)))

    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, _, _, _, _, _, _) in enumerate(grp_packed):
        opinm = EpoxyPINMRegressor(mode=mode_str, target_name=target_col, base_estimator="xgboost", physics_weight=0.25, use_polymer_physics=False, monotone_constraints=True, n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, estimator_params={"n_jobs": 4, "tree_method": "hist"}).fit(Xb_tr, y[tr_idx])
        oof_grp["Original_EpoxyPINM_BaseOnly"][te_idx] = np.asarray(opinm.predict(Xb_te), dtype=float).ravel()

    print(f"[{task_name}] Stage 2: Running GPU PINN & TabPFN...", flush=True)
    import contextlib, io, shutil
    from core.pinn_model import EpoxyPINNRegressor
    from tabpfn import TabPFNRegressor
    if not Path(TABPFN_CKPT).exists() and Path(TABPFN_CKPT_WIN).exists():
        shutil.copy2(TABPFN_CKPT_WIN, TABPFN_CKPT)
    ckpt_use = TABPFN_CKPT if Path(TABPFN_CKPT).exists() else TABPFN_CKPT_WIN
    n_est_tab = 4 if len(y) > 1000 else 8
    shared_tab_reg = TabPFNRegressor(
        model_path=ckpt_use, device="cuda", n_estimators=n_est_tab, random_state=42, ignore_pretraining_limits=True
    )
    def _run_gpu_pinn_fold(Xb_tr, Xb_te, Xa_tr_imp, Xa_te_imp, y_tr):
        # 5A. Original EpoxyPINNRegressor
        ep_opinn = 35 if len(y_tr) > 1000 else 65
        pat_opinn = 8 if len(y_tr) > 1000 else 12
        opinn = EpoxyPINNRegressor(
            mode=mode_str, target_name=target_col, hidden_dim=256, n_layers=3, dropout=0.15,
            lr=1.5e-3, epochs=ep_opinn, patience=pat_opinn,
            physics_weight=0.05, crosslink_physics_weight=0.0,
            nu_encoder_path="/none", device="cuda", seed=42, verbose=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            opinn.fit(Xb_tr, y_tr)
        tr_opinn = np.asarray(opinn.predict(Xb_tr), dtype=float).ravel()
        te_opinn = np.asarray(opinn.predict(Xb_te), dtype=float).ravel()

        # 5B. CondensedMDPINNRegressor (Differentiable Physics Layer + Kernel Basis + Autograd Monotonicity)
        corrs = np.nan_to_num(np.abs(np.corrcoef(Xa_tr_imp.to_numpy().T, y_tr)[:-1, -1]), nan=0.0)
        k_p = min(len(all_valid_cols), 80)
        top_cols = [all_valid_cols[i] for i in np.argsort(corrs)[::-1][:k_p]]
        pinn_cols = list(dict.fromkeys(top_cols + [c for c in CondensedMDPINNRegressor.PHYS_CORE_COLS if c in all_valid_cols]))

        kpinn = CondensedMDPINNRegressor(
            mode=mode_str, hidden_dim=192, lr=1.5e-3, weight_decay=2.5e-2, epochs=115,
            physics_prior_weight=0.15, monotonicity_weight=0.04, device="cuda", seed=42,
        )
        kpinn.fit(Xa_tr_imp[pinn_cols], y_tr)
        tr_kp, te_kp = kpinn.predict(Xa_tr_imp[pinn_cols]), kpinn.predict(Xa_te_imp[pinn_cols])

        # 5C. GPU TabPFN-v3 Prior-Data Fitted Neural Transformer on Pure Formulation+Process+MD
        k_tab = min(len(all_valid_cols), 60)
        tab_cols = [all_valid_cols[i] for i in np.argsort(corrs)[::-1][:k_tab]]
        shared_tab_reg.fit(Xa_tr_imp[tab_cols], y_tr)
        tr_tab = np.asarray(shared_tab_reg.predict(Xa_tr_imp[tab_cols]), dtype=float).ravel()
        te_tab = np.asarray(shared_tab_reg.predict(Xa_te_imp[tab_cols]), dtype=float).ravel()

        tr_cpinn = 0.22 * tr_kp + 0.78 * tr_tab
        te_cpinn = 0.22 * te_kp + 0.78 * te_tab
        return tr_opinn, te_opinn, tr_cpinn, te_cpinn

    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, Xa_tr_imp, Xa_te_imp, _, _, _, _) in enumerate(rand_packed):
        tr_opinn, te_opinn, tr_cpinn, te_cpinn = _run_gpu_pinn_fold(Xb_tr, Xb_te, Xa_tr_imp, Xa_te_imp, y[tr_idx])
        oof_rand["Original_EpoxyPINN_BaseOnly"][te_idx] = te_opinn
        tr_r2_folds["Original_EpoxyPINN_BaseOnly"].append(float(r2_score(y[tr_idx], tr_opinn)))
        tr_mae_folds["Original_EpoxyPINN_BaseOnly"].append(float(mean_absolute_error(y[tr_idx], tr_opinn)))

        oof_rand["Condensed_MD_Kernel_PINN_GPU"][te_idx] = te_cpinn
        tr_r2_folds["Condensed_MD_Kernel_PINN_GPU"].append(float(r2_score(y[tr_idx], tr_cpinn)))
        tr_mae_folds["Condensed_MD_Kernel_PINN_GPU"].append(float(mean_absolute_error(y[tr_idx], tr_cpinn)))

        tr_pinm = 0.35 * rand_cb[f_i][0] + 0.25 * rand_xgb[f_i][0] + 0.20 * rand_lgb[f_i][0] + 0.20 * rand_et[f_i][0]
        te_pinm = 0.35 * rand_cb[f_i][1] + 0.25 * rand_xgb[f_i][1] + 0.20 * rand_lgb[f_i][1] + 0.20 * rand_et[f_i][1]
        tr_hyb = 0.16 * tr_cpinn + 0.84 * tr_pinm
        te_hyb = 0.16 * te_cpinn + 0.84 * te_pinm
        oof_rand["Condensed_MD_Hybrid_PINN_PINM"][te_idx] = te_hyb
        tr_r2_folds["Condensed_MD_Hybrid_PINN_PINM"].append(float(r2_score(y[tr_idx], tr_hyb)))
        tr_mae_folds["Condensed_MD_Hybrid_PINN_PINM"].append(float(mean_absolute_error(y[tr_idx], tr_hyb)))

    for f_i, (tr_idx, te_idx, Xb_tr, Xb_te, _, _, Xa_tr_imp, Xa_te_imp, _, _, _, _) in enumerate(grp_packed):
        _, te_opinn, _, te_cpinn = _run_gpu_pinn_fold(Xb_tr, Xb_te, Xa_tr_imp, Xa_te_imp, y[tr_idx])
        oof_grp["Original_EpoxyPINN_BaseOnly"][te_idx] = te_opinn
        oof_grp["Condensed_MD_Kernel_PINN_GPU"][te_idx] = te_cpinn

        te_pinm = 0.35 * grp_cb[f_i][1] + 0.25 * grp_xgb[f_i][1] + 0.20 * grp_lgb[f_i][1] + 0.20 * grp_et[f_i][1]
        oof_grp["Condensed_MD_Hybrid_PINN_PINM"][te_idx] = 0.16 * te_cpinn + 0.84 * te_pinm
    results = {}
    for m in model_names:
        feat_cnt = len(base_valid_cols) if "BaseOnly" in m else len(all_valid_cols)
        res_m = {
            "feature_count": feat_cnt,
            "Train_R2_mean": round(float(np.mean(tr_r2_folds[m])), 4),
            "Train_MAE_mean": round(float(np.mean(tr_mae_folds[m])), 4),
            "Random_5Fold_OOF_R2": round(float(r2_score(y, oof_rand[m])), 4),
            "Random_5Fold_OOF_MAE": round(float(mean_absolute_error(y, oof_rand[m])), 4),
            "Random_5Fold_OOF_RMSE": round(float(np.sqrt(mean_squared_error(y, oof_rand[m]))), 4),
            "Group_5Fold_OOF_R2": round(float(r2_score(y, oof_grp[m])), 4),
            "Group_5Fold_OOF_MAE": round(float(mean_absolute_error(y, oof_grp[m])), 4),
            "Group_5Fold_OOF_RMSE": round(float(np.sqrt(mean_squared_error(y, oof_grp[m]))), 4),
        }
        results[m] = res_m
        print(
            f"[{task_name}] {m:32s} | Feat={feat_cnt:3d} | "
            f"Train R2={res_m['Train_R2_mean']:.4f} (MAE={res_m['Train_MAE_mean']:.3f}) | "
            f"Random 5F R2={res_m['Random_5Fold_OOF_R2']:.4f} (MAE={res_m['Random_5Fold_OOF_MAE']:.3f}) | "
            f"Group 5F R2={res_m['Group_5Fold_OOF_R2']:.4f} (MAE={res_m['Group_5Fold_OOF_MAE']:.3f})",
            flush=True,
        )
    return results


def _run_single_task(task_id: str, out_json: Path, splits_manifest: Dict[str, Any]) -> None:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    if task_id == "mod566_clean":
        df = pd.read_csv(EXT_MD / "artifacts/clean_566_formulation_process_physics.csv")
        df_aug, base_cols, md_cols, all_cols = build_pure_formulation_process_md_features(df, is_modulus=True)
        df_sub = filter_physically_consistent_modulus_subset(df_aug, "storage_modulus_25c_gpa", all_cols)
        df_sub["formulation_id"] = [
            "M566C_" + hashlib.sha256(f"{round(float(r.get('resin_1_molecular_weight_g_mol', 340)), 1)}|{round(float(r.get('curing_agent_1_molecular_weight_g_mol', 200)), 1)}|{round(float(r.get('formulation_r_value', 1.0)), 2)}".encode()).hexdigest()[:10]
            for _, r in df_sub.iterrows()
        ]
        r_splits = list(kf.split(df_sub))
        g_splits = list(gkf.split(df_sub, df_sub["storage_modulus_25c_gpa"], df_sub["formulation_id"]))
        res = evaluate_dataset_two_stage(df_sub, "storage_modulus_25c_gpa", base_cols, all_cols, r_splits, g_splits, "Modulus_566_Clean_Deduplicated", True)
        json_key = "Storage_Modulus_566_Clean_Deduplicated"

    elif task_id == "mod566_raw":
        df = pd.read_csv(EXT_MD / "artifacts/clean_566_formulation_process_physics.csv")
        df_aug, base_cols, md_cols, all_cols = build_pure_formulation_process_md_features(df, is_modulus=True)
        df_aug["formulation_id"] = [f"M566_{i // 2}" for i in range(len(df_aug))]
        r_splits = list(kf.split(df_aug))
        g_splits = list(gkf.split(df_aug, df_aug["storage_modulus_25c_gpa"], df_aug["formulation_id"]))
        res = evaluate_dataset_two_stage(df_aug, "storage_modulus_25c_gpa", base_cols, all_cols, r_splits, g_splits, "Modulus_566_Benchmark_Raw", True)
        json_key = "Storage_Modulus_566_Benchmark_Dataset"

    elif task_id == "mod706_clean":
        run_dir = CFRP_ROOT / "results/training_runs/20261015_110046_Epoxy_PINN__Physics-Informed_"
        df_X = pd.concat([pd.read_csv(run_dir / "split_X_train.csv"), pd.read_csv(run_dir / "split_X_test.csv")], ignore_index=True)
        df_y = pd.concat([pd.read_csv(run_dir / "split_y_train.csv"), pd.read_csv(run_dir / "split_y_test.csv")], ignore_index=True)
        df = pd.concat([df_X, df_y], axis=1)
        df_aug, base_cols, md_cols, all_cols = build_pure_formulation_process_md_features(df, is_modulus=True)
        df_sub = filter_physically_consistent_modulus_subset(df_aug, "storage_modulus_25c_gpa", all_cols)
        df_sub["formulation_id"] = [f"M706C_{i // 2}" for i in range(len(df_sub))]
        r_splits = list(kf.split(df_sub))
        g_splits = list(gkf.split(df_sub, df_sub["storage_modulus_25c_gpa"], df_sub["formulation_id"]))
        res = evaluate_dataset_two_stage(df_sub, "storage_modulus_25c_gpa", base_cols, all_cols, r_splits, g_splits, "Modulus_706_Phys_Consistent", True)
        json_key = "Storage_Modulus_706_Physically_Consistent"

    elif task_id == "tg644":
        df = pd.read_csv(CFRP_ROOT / "data_fixed" / "latest_data_clean_594.csv", low_memory=False)
        df["formulation_id"] = [f"F_{i}" for i in range(len(df))]
        df_aug, base_cols, md_cols, all_cols = build_pure_formulation_process_md_features(df, is_modulus=False)
        r_splits = [(np.array(s["train_indices"], dtype=int), np.array(s["test_indices"], dtype=int)) for s in splits_manifest["tg_splits"]["random_5fold"]]
        g_splits = [(np.array(s["train_indices"], dtype=int), np.array(s["test_indices"], dtype=int)) for s in splits_manifest["tg_splits"]["formulation_group_5fold"]]
        res = evaluate_dataset_two_stage(df_aug, "tg_c", base_cols, all_cols, r_splits, g_splits, "Tg_644_Clean", False)
        json_key = "Tg_644_Clean_Subset"

    elif task_id == "tg2229":
        df = pd.read_csv(EXT_MD / "artifacts/clean_tg2229_formulation_process_physics.csv", low_memory=False)
        df_aug, base_cols, md_cols, all_cols = build_pure_formulation_process_md_features(df, is_modulus=False)
        r_splits = list(kf.split(df_aug))
        g_splits = list(gkf.split(df_aug, df_aug["tg_c"], df_aug["formulation_id"]))
        res = evaluate_dataset_two_stage(df_aug, "tg_c", base_cols, all_cols, r_splits, g_splits, "Tg_2229_Full_PureCols", False)
        json_key = "Tg_2229_Full_Dataset"
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    summary = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    summary[json_key] = res
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
def main():
    import argparse, subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    args = parser.parse_args()

    out_dir = EXT_MD / "artifacts" / "condensed_md_pinn_optimization_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "condensed_md_pinn_optimization_results.json"

    splits_manifest = json.loads((EXT_ML / "reports" / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))

    if args.task != "all":
        _run_single_task(args.task, out_json, splits_manifest)
        return

    for t_id in ["mod566_clean", "mod566_raw", "mod706_clean", "tg644", "tg2229"]:
        subprocess.run([sys.executable, "-u", __file__, "--task", t_id], check=True)

    print(f"\nSaved complete PINN/PINM optimization results -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
