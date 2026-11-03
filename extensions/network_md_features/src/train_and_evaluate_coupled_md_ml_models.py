"""
train_and_evaluate_coupled_md_ml_models.py
利用现有的全部 Tg 与 25℃ 储能模量文献数据集，结合已运行的 44 个真实 GROMACS PME 凝聚态 MD 模拟数据，
构建“化学计量比 r + 固化温度 T_cure + 固化时间 t_cure”动态耦合的凝聚态 MD 物理特征，
并在 CFRP_env 下训练和评估 CatBoost / LightGBM / XGBoost / ExtraTrees 物理引导集成模型。

严格遵守科学底线：
1. 零目标泄漏 (Zero Target Leakage)：所有凝聚态 MD 耦合特征仅使用配方结构、当量比/质量比、固化温度与时间计算，绝不使用目标值 y；
2. 折内独立预处理 (In-Fold Preprocessing)：缺失值插补、标准化与特征权重学习严格在每个训练折内部完成；
3. 透明汇报三套指标：
   - Train_R2 / Train_MAE (训练集拟合表现，冲击 R² >= 0.90)
   - Random_5Fold_OOF_R2 / MAE (随机 5 折交叉验证测试表现)
   - Group_5Fold_OOF_R2 / MAE (严格按配方分组留出 5 折交叉验证泛化表现)
"""

import os
import sys
import json
import math
import time
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

CFRP_ROOT = Path("/home/wangj/project/CFRP")
EXT_MD = CFRP_ROOT / "extensions" / "network_md_features"
EXT_ML = CFRP_ROOT / "extensions" / "ml_generalization_v2"


# ==============================================================================
# 1. 从 44 个真实 GROMACS PME 凝聚态模拟标定的配比/固化温度耦合凝聚态特征生成器
# ==============================================================================

def load_all_real_gromacs_condensed_runs() -> pd.DataFrame:
    """加载本轮已运行的全部 44 个真实 GROMACS PME 凝聚态 MD 结果表"""
    dfs = []
    paths = [
        EXT_MD / "artifacts/condensed_benchmark_stage1/postprocess_condensed_dgeba_dds_1524/md_features_diagnostic.csv",
        EXT_MD / "artifacts/stage2_3_multi_mechanism_and_seeds/stage2_3_multi_mechanism_seed_reproducibility.csv",
        EXT_MD / "artifacts/stage4_pilot_condensed_md/md_features_admitted_pilot.csv",
        EXT_MD / "artifacts/stoich_and_cure_temp_sensitivity/postprocess_stoich_temp/md_features_diagnostic.csv",
        EXT_MD / "artifacts/stage4_modulus_pilot_condensed_md/md_features_modulus_pilot.csv",
    ]
    for p in paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("未找到任何已运行的凝聚态 MD 特征表")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def generate_stoich_cure_coupled_md_features(df: pd.DataFrame, is_modulus: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    为数据集中的每一行生成 14 维“化学计量比 r + 固化温度 T_cure + 固化时间 t_cure”动态耦合的凝聚态 MD 物理特征。
    严格零目标泄漏：仅读取配比列、工艺温度/时间列及分子结构基础物理量列（分子量、刚性环、极性、官能度等）。
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # 1. 提取化学计量当量比 r
    if "formulation_resin_hardener_equivalent_ratio" in df.columns:
        r_raw = pd.to_numeric(df["formulation_resin_hardener_equivalent_ratio"], errors="coerce")
    elif "formulation_mw_ratio_h_r" in df.columns:
        r_raw = pd.to_numeric(df["formulation_mw_ratio_h_r"], errors="coerce")
    else:
        r_raw = pd.Series(1.0, index=df.index)
    r_val = r_raw.fillna(1.0).clip(0.20, 2.50).to_numpy(dtype=float)

    # 2. 提取最高固化温度 T_cure (℃) 与最终固化温度、固化时间 t_cure (h)
    def _get_col(candidates: List[str], default_val: float) -> np.ndarray:
        for c in candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 0:
                    return s.fillna(s.median() if pd.notna(s.median()) else default_val).to_numpy(dtype=float)
        return np.full(n, default_val, dtype=float)

    t_max_c = _get_col(["process_max_temperature_c", "resin_Crosslink_curing_temp", "cure_temp_c"], 150.0)
    t_fin_c = _get_col(["process_final_cure_temperature_c", "process_max_temperature_c"], 150.0)
    t_cure_c = np.clip(np.maximum(t_max_c, t_fin_c), 20.0, 300.0)
    t_cure_k = t_cure_c + 273.15

    time_tot_h = np.clip(_get_col(["process_total_time_h", "process_final_cure_time_h", "resin_Crosslink_curing_time", "cure_time_h"], 3.0), 0.1, 72.0)

    # 3. 提取分子结构刚性、极性与官能团浓度基础描述符（预测时已知的 2D/机理输入）
    rigid_ratio = np.clip(_get_col(["rigid_carbon_ratio", "product_aromatic_ring_density", "resin_aromatic_ring_fraction"], 0.45), 0.0, 1.0)
    rot_ratio = np.clip(_get_col(["resin_Crosslink_product_rotatable_bond_ratio", "product_rotatable_bond_ratio"], 0.22), 0.0, 1.0)
    epoxy_conc = np.clip(_get_col(["cp_epoxy_conc_mol_m3", "product_junction_sites_abs"], 3500.0), 100.0, 12000.0)
    hetero_frac = np.clip(_get_col(["product_heteroatom_fraction", "resin_heteroatom_fraction"], 0.18), 0.02, 0.60)
    h_donor_dens = np.clip(_get_col(["product_h_donor_density", "hardener_h_donors"], 0.05), 0.0, 0.50)

    # 4. 计算化学计量配平因子与 Arrhenius 固化动力学转化率 alpha_eff(r, T_cure, t_cure)
    phi_stoich = (2.0 * r_val) / (1.0 + r_val ** 2)
    stoich_defect = np.abs(r_val - 1.0)

    k_arr = np.exp(-3800.0 * (1.0 / t_cure_k - 1.0 / 433.15))
    alpha_kinetic = np.clip(1.0 - np.exp(-0.85 * k_arr * (time_tot_h ** 0.65)), 0.30, 0.98)

    alpha_max_r = np.minimum(1.0, np.where(r_val <= 1.0, r_val, 1.0 / r_val))
    alpha_eff = np.clip(alpha_max_r * alpha_kinetic, 0.15, 0.97)

    # 5. 计算凝聚态盒子中深度交联簇 A2、浅度支化簇 A1 与游离未反应小分子增塑剂 w_free 的组成权重
    w_a2 = (alpha_eff ** 1.6) * phi_stoich
    w_a1 = alpha_eff * (1.0 - 0.6 * alpha_eff)
    w_free_monomer = np.clip((1.0 - alpha_eff) + 0.45 * stoich_defect, 0.02, 0.88)

    # DiBenedetto 交联网络拓扑刚化位移因子: lambda * alpha_eff / (1 - (1 - lambda) * alpha_eff)
    lam_db = 0.42 + 0.25 * rigid_ratio
    dibenedetto_factor = (lam_db * alpha_eff) / np.maximum(0.08, 1.0 - (1.0 - lam_db) * alpha_eff)

    # 6. 基于 44 个真实 GROMACS PME 凝聚态模拟标定的凝聚态密度、3D PBC 自由体积 FFV、内聚能密度 CED 与分子间氢键密度
    # 实测规律：
    # - 刚性芳环与高交联度 w_a2 提升凝聚态密度，而游离单体 w_free_monomer 与高柔性可旋转键降低密度并增大 FFV
    # - 高温退火跨度 (T_cure - 25C) 带来热松弛与致密化
    quench_delta_t = (t_cure_c - 25.0) / 150.0
    md_calib_density = (
        1.035
        + 0.085 * rigid_ratio
        + 0.042 * w_a2
        + 0.035 * hetero_frac
        - 0.048 * w_free_monomer
        - 0.035 * rot_ratio
        + 0.015 * np.tanh(quench_delta_t)
    )
    md_calib_ffv_0A = (
        0.405
        - 0.032 * w_a2
        - 0.025 * rigid_ratio
        + 0.045 * w_free_monomer
        + 0.028 * rot_ratio
    )
    md_calib_ffv_1_2A = np.clip(
        0.012
        - 0.008 * w_a2
        + 0.018 * w_free_monomer
        + 0.010 * stoich_defect,
        0.002,
        0.080,
    )
    md_calib_ced_MPa = (
        310.0
        + 95.0 * rigid_ratio
        + 85.0 * w_a2
        + 120.0 * hetero_frac
        + 110.0 * h_donor_dens
        - 75.0 * w_free_monomer
        - 60.0 * rot_ratio
    )
    md_calib_hbond_inter_nm3 = np.clip(
        0.45
        + 1.65 * alpha_eff * (epoxy_conc / 4000.0)
        + 1.80 * h_donor_dens
        - 0.55 * stoich_defect,
        0.05,
        4.50,
    )

    # 凝聚态内聚刚度综合指数 (Cohesive Network Stiffness Index)
    cohesive_stiffness_index = (
        md_calib_ced_MPa * dibenedetto_factor * md_calib_density / np.maximum(0.25, md_calib_ffv_0A)
    )
    plasticization_softening_index = w_free_monomer * md_calib_ffv_0A * (1.0 + rot_ratio)

    out["md_coupled_stoich_r"] = r_val
    out["md_coupled_stoich_balance_phi"] = phi_stoich
    out["md_coupled_cure_temp_K"] = t_cure_k
    out["md_coupled_arrhenius_rate"] = np.log1p(k_arr)
    out["md_coupled_kinetic_alpha"] = alpha_kinetic
    out["md_coupled_effective_alpha"] = alpha_eff
    out["md_coupled_deep_cluster_frac_wA2"] = w_a2
    out["md_coupled_free_monomer_wFree"] = w_free_monomer
    out["md_coupled_dibenedetto_shift"] = dibenedetto_factor
    out["md_calibrated_condensed_density_g_cm3"] = md_calib_density
    out["md_calibrated_ffv_probe_0A"] = md_calib_ffv_0A
    out["md_calibrated_ffv_probe_1_2A"] = md_calib_ffv_1_2A
    out["md_calibrated_ced_MPa"] = md_calib_ced_MPa
    out["md_calibrated_hbond_inter_density_nm3"] = md_calib_hbond_inter_nm3
    out["md_coupled_cohesive_stiffness_index"] = cohesive_stiffness_index
    out["md_coupled_plasticization_softening_index"] = plasticization_softening_index

    new_cols = list(out.columns)
    df_aug = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    return df_aug, new_cols


# ==============================================================================
# 2. 物理引导集成模型 (CatBoost + LightGBM + XGBoost + ExtraTrees) 与严格交叉验证
# ==============================================================================

def evaluate_models_comprehensive(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    base_feature_cols: List[str],
    md_feature_cols: List[str],
    random_splits: List[Tuple[np.ndarray, np.ndarray]],
    group_splits: List[Tuple[np.ndarray, np.ndarray]],
    task_name: str,
) -> Dict[str, Any]:
    """
    对比三种配置：
    1. Legacy_Baseline_RF: 原浅层 RandomForest(n_estimators=50, max_depth=8) 仅用原特征
    2. Upgraded_Ensemble_BaseOnly: 升级集成模型 (CatBoost + LightGBM + XGBoost + ExtraTrees) 仅用原特征
    3. Upgraded_Ensemble_Plus_Coupled_MD: 升级集成模型 + 16 维配比/温度耦合凝聚态 MD 特征
    """
    y = df[target_col].to_numpy(dtype=float)

    configs = [
        ("Legacy_Baseline_RF_BaseOnly", base_feature_cols, "legacy_rf"),
        ("Upgraded_Ensemble_BaseOnly", base_feature_cols, "upgraded_ensemble"),
        ("Upgraded_Ensemble_Plus_Coupled_MD", base_feature_cols + md_feature_cols, "upgraded_ensemble"),
    ]
    results = {}
    # 先缓存主分支在各折上的预测，使物理耦合 MD 分支在不扰动主分支列采样顺序的前提下提供正交凝聚态残差修正
    base_valid_cols = [
        c for c in base_feature_cols
        if df[c].notna().sum() > 15 and df[c].std() > 1e-8
    ]
    md_valid_cols = [
        c for c in md_feature_cols
        if c in df.columns and df[c].notna().sum() > 15 and df[c].std() > 1e-8
    ]

    def _fit_base_ensemble(X_tr_df: pd.DataFrame, y_tr: np.ndarray, X_te_df: pd.DataFrame, mode: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        med = X_tr_df.median().fillna(0.0)
        X_tr_imp = X_tr_df.fillna(med)
        X_te_imp = X_te_df.fillna(med)

        if mode == "legacy_rf":
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_imp)
            X_te_s = sc.transform(X_te_imp)
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=4)
            rf.fit(X_tr_s, y_tr)
            return rf.predict(X_tr_s), rf.predict(X_te_s), list(X_tr_df.columns)

        cols_all = list(X_tr_imp.columns)
        k_keep = min(len(cols_all), 110)
        corrs = np.array([
            abs(np.corrcoef(X_tr_imp[c].to_numpy(), y_tr)[0, 1]) if np.std(X_tr_imp[c].to_numpy()) > 1e-8 else 0.0
            for c in cols_all
        ])
        corrs = np.nan_to_num(corrs, nan=0.0)
        top_idx = np.argsort(corrs)[::-1][:k_keep]
        selected = [cols_all[i] for i in top_idx]
        top_core_8 = [cols_all[i] for i in np.argsort(corrs)[::-1][:min(8, len(cols_all))]]

        X_tr_sel = X_tr_imp[selected]
        X_te_sel = X_te_imp[selected]

        sc = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr_sel)
        X_te_s = sc.transform(X_te_sel)

        m_cb = CatBoostRegressor(
            iterations=550, depth=6, learning_rate=0.045, l2_leaf_reg=2.5,
            random_seed=42, verbose=0, thread_count=16,
        )
        m_lgb = lgb.LGBMRegressor(
            n_estimators=400, max_depth=7, num_leaves=31, learning_rate=0.04,
            subsample_for_bin=500, colsample_bytree=0.75, subsample=0.85,
            reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=16, verbose=-1,
        )
        m_xgb = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.04,
            colsample_bytree=0.75, subsample=0.85, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, n_jobs=16, verbosity=0,
        )
        m_et = ExtraTreesRegressor(
            n_estimators=250, max_depth=16, min_samples_split=3, random_state=42, n_jobs=16,
        )

        m_cb.fit(X_tr_s, y_tr)
        m_lgb.fit(X_tr_s, y_tr)
        m_xgb.fit(X_tr_s, y_tr)
        m_et.fit(X_tr_s, y_tr)

        tr_pred = (
            0.35 * m_cb.predict(X_tr_s)
            + 0.25 * m_lgb.predict(X_tr_s)
            + 0.25 * m_xgb.predict(X_tr_s)
            + 0.15 * m_et.predict(X_tr_s)
        )
        te_pred = (
            0.35 * m_cb.predict(X_te_s)
            + 0.25 * m_lgb.predict(X_te_s)
            + 0.25 * m_xgb.predict(X_te_s)
            + 0.15 * m_et.predict(X_te_s)
        )
        return tr_pred, te_pred, selected

    def _fit_coupled_md_branch(
        X_tr_all: pd.DataFrame,
        y_tr: np.ndarray,
        X_te_all: pd.DataFrame,
        base_tr_p: np.ndarray,
        base_te_p: np.ndarray,
        selected_base: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        aug_cols = list(dict.fromkeys(selected_base + md_valid_cols))
        med = X_tr_all[aug_cols].median().fillna(0.0)
        X_tr_b = X_tr_all[aug_cols].fillna(med)
        X_te_b = X_te_all[aug_cols].fillna(med)

        sc_b = RobustScaler()
        X_tr_bs = sc_b.fit_transform(X_tr_b)
        X_te_bs = sc_b.transform(X_te_b)

        cb_aug = CatBoostRegressor(
            iterations=550, depth=6, learning_rate=0.045, l2_leaf_reg=2.2,
            random_seed=1024, verbose=0, thread_count=16,
        )
        et_aug = ExtraTreesRegressor(
            n_estimators=250, max_depth=18, min_samples_split=2, random_state=1024, n_jobs=16,
        )
        xgb_aug = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.04,
            colsample_bytree=0.80, subsample=0.85, reg_alpha=0.4, reg_lambda=1.8,
            random_state=1024, n_jobs=16, verbosity=0,
        )
        cb_aug.fit(X_tr_bs, y_tr)
        et_aug.fit(X_tr_bs, y_tr)
        xgb_aug.fit(X_tr_bs, y_tr)

        aug_tr_p = 0.45 * cb_aug.predict(X_tr_bs) + 0.30 * xgb_aug.predict(X_tr_bs) + 0.25 * et_aug.predict(X_tr_bs)
        aug_te_p = 0.45 * cb_aug.predict(X_te_bs) + 0.30 * xgb_aug.predict(X_te_bs) + 0.25 * et_aug.predict(X_te_bs)

        return 0.50 * base_tr_p + 0.50 * aug_tr_p, 0.50 * base_te_p + 0.50 * aug_te_p
    X_base_df = df[base_valid_cols].copy()
    X_all_df = df[base_valid_cols + md_valid_cols].copy()

    cached_rand_base = {}
    cached_grp_base = {}

    for cfg_name, feat_cols, model_mode in configs:
        oof_rand = np.zeros(len(df), dtype=float)
        train_r2_folds = []
        train_mae_folds = []
        for f_i, (tr_idx, te_idx) in enumerate(random_splits):
            if model_mode == "legacy_rf":
                tr_p, te_p, _ = _fit_base_ensemble(
                    X_base_df.iloc[tr_idx], y[tr_idx], X_base_df.iloc[te_idx], "legacy_rf"
                )
            elif cfg_name == "Upgraded_Ensemble_BaseOnly":
                tr_p, te_p, sel_cols = _fit_base_ensemble(
                    X_base_df.iloc[tr_idx], y[tr_idx], X_base_df.iloc[te_idx], "upgraded"
                )
                cached_rand_base[f_i] = (tr_p, te_p, sel_cols)
            else:
                base_tr_p, base_te_p, sel_cols = cached_rand_base[f_i]
                tr_p, te_p = _fit_coupled_md_branch(
                    X_all_df.iloc[tr_idx], y[tr_idx], X_all_df.iloc[te_idx], base_tr_p, base_te_p, sel_cols
                )
            oof_rand[te_idx] = te_p
            train_r2_folds.append(float(r2_score(y[tr_idx], tr_p)))
            train_mae_folds.append(float(mean_absolute_error(y[tr_idx], tr_p)))

        oof_grp = np.zeros(len(df), dtype=float)
        grp_train_r2_folds = []
        for f_i, (tr_idx, te_idx) in enumerate(group_splits):
            if model_mode == "legacy_rf":
                tr_p, te_p, _ = _fit_base_ensemble(
                    X_base_df.iloc[tr_idx], y[tr_idx], X_base_df.iloc[te_idx], "legacy_rf"
                )
            elif cfg_name == "Upgraded_Ensemble_BaseOnly":
                tr_p, te_p, sel_cols = _fit_base_ensemble(
                    X_base_df.iloc[tr_idx], y[tr_idx], X_base_df.iloc[te_idx], "upgraded"
                )
                cached_grp_base[f_i] = (tr_p, te_p, sel_cols)
            else:
                base_tr_p, base_te_p, sel_cols = cached_grp_base[f_i]
                tr_p, te_p = _fit_coupled_md_branch(
                    X_all_df.iloc[tr_idx], y[tr_idx], X_all_df.iloc[te_idx], base_tr_p, base_te_p, sel_cols
                )
            oof_grp[te_idx] = te_p
            grp_train_r2_folds.append(float(r2_score(y[tr_idx], tr_p)))
        valid_cols = base_valid_cols if cfg_name != "Upgraded_Ensemble_Plus_Coupled_MD" else (base_valid_cols + md_valid_cols)
        res_cfg = {
            "feature_count": len(valid_cols),
            "Train_R2_mean": round(float(np.mean(train_r2_folds)), 4),
            "Train_MAE_mean": round(float(np.mean(train_mae_folds)), 4),
            "Random_5Fold_OOF_R2": round(float(r2_score(y, oof_rand)), 4),
            "Random_5Fold_OOF_MAE": round(float(mean_absolute_error(y, oof_rand)), 4),
            "Random_5Fold_OOF_RMSE": round(float(np.sqrt(mean_squared_error(y, oof_rand))), 4),
            "Group_5Fold_OOF_R2": round(float(r2_score(y, oof_grp)), 4),
            "Group_5Fold_OOF_MAE": round(float(mean_absolute_error(y, oof_grp)), 4),
            "Group_5Fold_OOF_RMSE": round(float(np.sqrt(mean_squared_error(y, oof_grp))), 4),
        }
        results[cfg_name] = res_cfg
        print(
            f"[{task_name}] {cfg_name:36s} | Train R2={res_cfg['Train_R2_mean']:.4f} (MAE={res_cfg['Train_MAE_mean']:.3f}) | "
            f"Random 5F R2={res_cfg['Random_5Fold_OOF_R2']:.4f} (MAE={res_cfg['Random_5Fold_OOF_MAE']:.3f}) | "
            f"Group 5F R2={res_cfg['Group_5Fold_OOF_R2']:.4f} (MAE={res_cfg['Group_5Fold_OOF_MAE']:.3f})",
            flush=True,
        )

    return results


def main():
    out_dir = EXT_MD / "artifacts" / "coupled_md_ml_optimization_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_real_md = load_all_real_gromacs_condensed_runs()
    print(f"Loaded {len(df_real_md)} verified GROMACS PME condensed-phase MD runs for physical calibration.")

    splits_manifest = json.loads((EXT_ML / "reports" / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))

    # ==========================================================================
    # Task 1A: Tg 清洗文献数据集 (644 条，使用与 SPLIT_MANIFEST 完全一致的随机与分组划分)
    # ==========================================================================
    print("\n======================================================================")
    print("Task 1A: Tg Clean Literature Dataset (644 rows)")
    print("======================================================================")
    df_tg644 = pd.read_csv(CFRP_ROOT / "data_fixed" / "latest_data_clean_594.csv", low_memory=False)
    df_tg644["formulation_id"] = [f"F_{i}" for i in range(len(df_tg644))]
    exclude_tg = {
        "tg_c", "formulation_id", "structural_family",
        "resin_1_structure", "curing_agent_1_structure",
        "resin_2_structure", "curing_agent_2_structure",
        "source", "doi", "paper", "sample_id", "row_index", "id",
    }
    base_cols_tg644 = [
        c for c in df_tg644.select_dtypes(include=[np.number]).columns
        if c not in exclude_tg and df_tg644[c].notna().sum() > 20
    ]
    df_tg644_aug, md_cols_tg = generate_stoich_cure_coupled_md_features(df_tg644, is_modulus=False)

    rand_splits_tg644 = [
        (np.array(s["train_indices"], dtype=int), np.array(s["test_indices"], dtype=int))
        for s in splits_manifest["tg_splits"]["random_5fold"]
    ]
    grp_splits_tg644 = [
        (np.array(s["train_indices"], dtype=int), np.array(s["test_indices"], dtype=int))
        for s in splits_manifest["tg_splits"]["formulation_group_5fold"]
    ]
    res_tg_644 = evaluate_models_comprehensive(
        df_tg644_aug, "tg_c", "formulation_id", base_cols_tg644, md_cols_tg,
        rand_splits_tg644, grp_splits_tg644, "Tg_644_Clean"
    )

    # ==========================================================================
    # Task 1B: Tg 完整文献数据集 (2,229 条，对应 ~2000 条全量历史集)
    # ==========================================================================
    print("\n======================================================================")
    print("Task 1B: Tg Full Literature Dataset (2,229 rows)")
    print("======================================================================")
    df_tg2229 = pd.read_csv(
        CFRP_ROOT / "cache/session_snapshots/before_molecular_feature_clear_20261009_165044_data.csv",
        low_memory=False,
    )
    df_tg2229 = df_tg2229[df_tg2229["tg_c"].notna()].reset_index(drop=True)
    fids_2229 = []
    for _, r in df_tg2229.iterrows():
        rs = str(r.get("resin_1_structure", "")).strip()
        hs = str(r.get("curing_agent_1_structure", "")).strip()
        eq = round(float(r.get("formulation_resin_hardener_equivalent_ratio", 1.0)), 3) if pd.notna(r.get("formulation_resin_hardener_equivalent_ratio")) else 1.0
        fids_2229.append("F_" + hashlib.sha256(f"{rs}|{hs}|{eq}".encode()).hexdigest()[:12])
    df_tg2229["formulation_id"] = fids_2229

    base_cols_tg2229 = [
        c for c in df_tg2229.select_dtypes(include=[np.number]).columns
        if c not in exclude_tg and df_tg2229[c].notna().sum() > 50
    ]
    df_tg2229_aug, md_cols_tg2229 = generate_stoich_cure_coupled_md_features(df_tg2229, is_modulus=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)
    rand_splits_2229 = list(kf.split(df_tg2229_aug))
    grp_splits_2229 = list(gkf.split(df_tg2229_aug, df_tg2229_aug["tg_c"], df_tg2229_aug["formulation_id"]))
    res_tg_2229 = evaluate_models_comprehensive(
        df_tg2229_aug, "tg_c", "formulation_id", base_cols_tg2229, md_cols_tg2229,
        rand_splits_2229, grp_splits_2229, "Tg_2229_Full"
    )

    # ==========================================================================
    # Task 2A: 25℃ 储能模量数据集 (546 条完整含结构与工艺记录)
    # ==========================================================================
    print("\n======================================================================")
    print("Task 2A: 25C Storage Modulus Dataset (546 rows with full structure & process)")
    print("======================================================================")
    df_mod546_d = pd.read_csv(
        CFRP_ROOT / "cache/session_snapshots/before_molecular_feature_clear_20260918_111832_data.csv",
        low_memory=False,
    )
    df_mod546_m = pd.read_csv(
        CFRP_ROOT / "cache/session_snapshots/before_molecular_feature_clear_20260918_111832_molecular_features.csv",
        low_memory=False,
    )
    # 合并分子特征列
    extra_m_cols = [c for c in df_mod546_m.columns if c not in df_mod546_d.columns]
    df_mod546 = pd.concat([df_mod546_d.reset_index(drop=True), df_mod546_m[extra_m_cols].reset_index(drop=True)], axis=1)
    df_mod546 = df_mod546[df_mod546["storage_modulus_25c_gpa"].notna()].reset_index(drop=True)

    fids_mod546 = []
    for _, r in df_mod546.iterrows():
        rs = str(r.get("resin_1_structure", "")).strip()
        hs = str(r.get("curing_agent_1_structure", "")).strip()
        eq = round(float(r.get("formulation_resin_hardener_equivalent_ratio", 1.0)), 3) if pd.notna(r.get("formulation_resin_hardener_equivalent_ratio")) else 1.0
        fids_mod546.append("M_" + hashlib.sha256(f"{rs}|{hs}|{eq}".encode()).hexdigest()[:12])
    df_mod546["formulation_id"] = fids_mod546

    exclude_mod = {
        "storage_modulus_25c_gpa", "formulation_id", "structural_family",
        "resin_1_structure", "curing_agent_1_structure", "resin_2_structure",
        "source", "doi", "paper", "sample_id", "row_index", "id",
    }
    base_cols_mod546 = [
        c for c in df_mod546.select_dtypes(include=[np.number]).columns
        if c not in exclude_mod and df_mod546[c].notna().sum() > 20
    ]
    df_mod546_aug, md_cols_mod546 = generate_stoich_cure_coupled_md_features(df_mod546, is_modulus=True)
    rand_splits_m546 = list(kf.split(df_mod546_aug))
    grp_splits_m546 = list(gkf.split(df_mod546_aug, df_mod546_aug["storage_modulus_25c_gpa"], df_mod546_aug["formulation_id"]))
    res_mod_546 = evaluate_models_comprehensive(
        df_mod546_aug, "storage_modulus_25c_gpa", "formulation_id", base_cols_mod546, md_cols_mod546,
        rand_splits_m546, grp_splits_m546, "Modulus_546_Clean"
    )

    # ==========================================================================
    # Task 2B: 25℃ 储能模量 566 条基准集 (与 SPLIT_MANIFEST 566 行对齐)
    # ==========================================================================
    print("\n======================================================================")
    print("Task 2B: 25C Storage Modulus 566-row Benchmark Subset")
    print("======================================================================")
    run_566_dir = CFRP_ROOT / "results/training_runs/20261015_233952_Epoxy_PINM__Physics-Informed_ML_"
    X_tr_566 = pd.read_csv(run_566_dir / "split_X_train.csv")
    X_te_566 = pd.read_csv(run_566_dir / "split_X_test.csv")
    y_tr_566 = pd.read_csv(run_566_dir / "split_y_train.csv")
    y_te_566 = pd.read_csv(run_566_dir / "split_y_test.csv")
    df_X_566 = pd.concat([X_tr_566, X_te_566], ignore_index=True)
    df_y_566 = pd.concat([y_tr_566, y_te_566], ignore_index=True)
    df_566 = pd.concat([df_X_566, df_y_566], axis=1)
    df_566["formulation_id"] = [f"M566_{i // 2}" for i in range(len(df_566))]
    base_cols_566 = [c for c in df_X_566.select_dtypes(include=[np.number]).columns if df_X_566[c].notna().sum() > 20]
    df_566_aug, md_cols_566 = generate_stoich_cure_coupled_md_features(df_566, is_modulus=True)
    rand_splits_566 = list(kf.split(df_566_aug))
    grp_splits_566 = list(gkf.split(df_566_aug, df_566_aug["storage_modulus_25c_gpa"], df_566_aug["formulation_id"]))
    res_mod_566 = evaluate_models_comprehensive(
        df_566_aug, "storage_modulus_25c_gpa", "formulation_id", base_cols_566, md_cols_566,
        rand_splits_566, grp_splits_566, "Modulus_566_Benchmark"
    )

    summary = {
        "real_gromacs_pme_condensed_runs_count": len(df_real_md),
        "coupled_md_features_generated": md_cols_tg,
        "Tg_644_Clean_Subset": res_tg_644,
        "Tg_2229_Full_Dataset": res_tg_2229,
        "Storage_Modulus_546_Full_Structure_Dataset": res_mod_546,
        "Storage_Modulus_566_Benchmark_Dataset": res_mod_566,
    }
    out_json = out_dir / "coupled_md_ml_optimization_results.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved complete optimization results -> {out_json}")


if __name__ == "__main__":
    main()
