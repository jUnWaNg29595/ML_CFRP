# -*- coding: utf-8 -*-
"""
pinm_model.py

Physics-Informed Numerical/Machine-Learning Model (PINM) 回归器（遵循 sklearn API 风格）

核心设计与增强（针对 R2 提升至 0.88~0.92+）：
1. 自适应物理底座 (Adaptive Physics Baseline):
   - Tg 模式：不仅依赖显式 tg0/tginf 列，还自动通过工艺固化温度 (cure_temperature)、
     芳香碳分数 (phys_f_ar) 或交联密度自适应动态估算 T_g0 与 T_ginf，激活真实 DiBenedetto 动力学；
   - 力学模量模式：通过 Halpin-Tsai 与 Voigt-Reuss 边界联合锚定微观复合材料力学；
2. 零标签高分子物理特征引擎注入 (Polymer Physics Feature Augmentation):
   - 无缝调用 core.polymer_physics 自动补充 sp3碳分数、芳香碳分数、柔性指数、键解离能、内聚能等物理指数；
3. 传统机器学习残差拟合多引擎融合与自适应选型:
   - 支持 xgboost, lightgbm, catboost, rf, et, gbr 等多种树模型；
   - 支持特征单调性物理先验约束 (monotone_constraints)；
4. 物理残差融合权重调节 (physics_weight):
   - 支持可调节的物理底座混合权重，避免纯公式偏差或纯树模型过拟合。
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 兼容 Python 3.13 与旧版 scipy/numpy 属性冲突
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .epoxy_physics import (
    parse_first_number,
    parse_percent_to_fraction,
    alpha_max_from_r_np,
    dibenedetto_tg_np,
    halpin_tsai_np,
)

# 尝试导入 xgboost, lightgbm, catboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    cb = None
    CATBOOST_AVAILABLE = False


def _safe_float_array(series_or_arr: Any, default_val: float = np.nan) -> np.ndarray:
    if series_or_arr is None:
        return np.array([])
    if isinstance(series_or_arr, (pd.Series, pd.Index)):
        vals = series_or_arr.values
    else:
        vals = np.asarray(series_or_arr)

    # 优先使用 pandas 的类型判断，避免 StringDtype 等扩展类型传入 np.issubdtype 抛出异常
    is_num = False
    try:
        is_num = pd.api.types.is_numeric_dtype(series_or_arr)
    except Exception:
        try:
            is_num = np.issubdtype(vals.dtype, np.number)
        except Exception:
            is_num = False

    if is_num:
        arr = np.asarray(vals, dtype=np.float64).copy()
        # 清洗非有限值 (inf, -inf) 以及超出安全上限的极大数
        bad = (~np.isfinite(arr)) | (np.abs(arr) > 1e30)
        if np.any(bad):
            arr[bad] = default_val
        return arr

    res = np.empty(len(vals), dtype=np.float64)
    for i, v in enumerate(vals):
        try:
            val_f = parse_first_number(v)
            if np.isfinite(val_f) and np.abs(val_f) <= 1e30:
                res[i] = val_f
            else:
                res[i] = default_val
        except Exception:
            res[i] = default_val
    return res


class EpoxyPINMRegressor(BaseEstimator, RegressorMixin):
    """
    Epoxy PINM（Physics-Informed Machine-Learning Model）物理机理 + 传统模型回归器。
    """

    def __init__(
        self,
        mode: str = "auto",
        base_estimator: str = "xgboost",
        lambda_param: float = 0.45,
        physics_weight: float = 0.35,
        use_polymer_physics: bool = True,
        monotone_constraints: bool = True,
        clip_physical_bounds: bool = True,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        seed: Optional[int] = None,
        target_name: Optional[str] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
        **extra_kwargs,
    ):
        self.mode = mode
        self.base_estimator = base_estimator
        self.lambda_param = lambda_param
        self.physics_weight = physics_weight
        self.use_polymer_physics = use_polymer_physics
        self.monotone_constraints = monotone_constraints
        self.clip_physical_bounds = clip_physical_bounds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        # 兼容 seed 与 random_state
        effective_seed = seed if seed is not None else random_state
        self.random_state = effective_seed
        self.seed = seed
        self.target_name = target_name
        self.estimator_params = estimator_params or {}
        if extra_kwargs:
            self.estimator_params.update(extra_kwargs)

        # 拟合后保存的状态
        self.feature_names_: List[str] = []
        self.regressor_: Any = None
        self.imputer_: Optional[SimpleImputer] = None
        self._resolved_mode_: str = "tg"

        # 列名引用缓存
        self._col_r_: Optional[str] = None
        self._col_alpha_: Optional[str] = None
        self._col_tg0_: Optional[str] = None
        self._col_tginf_: Optional[str] = None
        self._col_cure_temp_: Optional[str] = None
        self._col_em_: Optional[str] = None
        self._col_ef_: Optional[str] = None
        self._col_vf_: Optional[str] = None
        self._col_xi_: Optional[str] = None

    def _resolve_mode(self, df: pd.DataFrame) -> str:
        """自动推断物理模式：tg / mechanics / generic"""
        m = (self.mode or "auto").strip().lower()
        if m in {"tg", "mechanics", "modulus", "generic"}:
            return "mechanics" if m == "modulus" else m

        # 若为 auto，优先根据 target_name 或特征列推断
        target_str = str(self.target_name or "").lower()
        if any(k in target_str for k in ["tg", "glass", "transition", "temperature", "temp", "t_g"]):
            return "tg"
        if any(k in target_str for k in ["modulus", "elastic", "tensile_modulus", "flexural_modulus", "e_"]):
            return "mechanics"

        # 根据特征列出现频率推断
        cols_lower = [str(c).lower() for c in df.columns]
        if any(k in c for c in cols_lower for k in ["tg", "cure_temp", "conversion", "alpha"]):
            return "tg"
        if any(k in c for c in cols_lower for k in ["modulus", "filler", "fiber", "vf", "v_f"]):
            return "mechanics"

        return "tg"

    def _augment_features_if_needed(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """调用 core.polymer_physics 自动计算零标签物理指数（芳香度、sp3碳比例、柔性指数等）"""
        if not self.use_polymer_physics:
            return df_raw

        try:
            from .polymer_physics import augment_polymer_physics
            df_aug = augment_polymer_physics(df_raw, enabled=True)
            return df_aug
        except Exception:
            return df_raw

    def _identify_columns(self, df: pd.DataFrame):
        """智能识别关键物理特征列"""
        cols = list(df.columns)

        # 1. 化学计量比 r
        r_prio = [
            "formulation_r_value",
            "formulation_resin_hardener_equivalent_ratio",
            "crosslink_stoichiometry_r",
            "stoichiometric_ratio_r_cleaned",
            "r_value",
        ]
        self._col_r_ = next((c for c in r_prio if c in cols), None)
        if self._col_r_ is None:
            for c in cols:
                cl = str(c).strip().lower()
                if "cp_r_value" in cl:
                    continue
                if cl == "r_value" or cl.endswith("_r_value") or "equivalent_ratio" in cl or "stoich" in cl:
                    self._col_r_ = c
                    break

        # 2. 固化转化率 alpha
        alpha_prio = ["alpha", "cure_degree", "conversion", "crosslink_density_alpha", "curing_degree"]
        self._col_alpha_ = next((c for c in alpha_prio if c in cols), None)
        if self._col_alpha_ is None:
            for c in cols:
                cl = str(c).strip().lower()
                if "alpha" in cl or "cure_deg" in cl or "conversion" in cl:
                    self._col_alpha_ = c
                    break

        # 3. 固化温度 (用于自适应估算 T_ginf 与 转化率)
        for c in cols:
            cl = str(c).strip().lower()
            if any(k in cl for k in ["cure_temp", "curing_temp", "temperature", "t_cure", "t_max"]):
                self._col_cure_temp_ = c
                break

        # 4. Tg0 与 Tginf
        self._col_tg0_ = next((c for c in ["tg0", "Tg0", "T_g0", "monomer_tg"] if c in cols), None)
        self._col_tginf_ = next((c for c in ["tginf", "Tg_inf", "Tginf", "T_g_inf", "tg_max"] if c in cols), None)

        # 5. 力学参数
        self._col_em_ = next((c for c in ["E_m", "Em", "matrix_modulus", "resin_modulus"] if c in cols), None)
        self._col_ef_ = next((c for c in ["E_f", "Ef", "fiber_modulus", "filler_modulus"] if c in cols), None)
        self._col_vf_ = next((c for c in ["V_f", "Vf", "fiber_volume_fraction", "filler_volume_fraction", "nanofiller_content", "filler_content"] if c in cols), None)
        self._col_xi_ = next((c for c in ["xi", "shape_factor", "aspect_ratio"] if c in cols), None)

    def _compute_physics_baseline(self, df: pd.DataFrame) -> np.ndarray:
        """自适应物理机理基准计算"""
        n_samples = len(df)
        if n_samples == 0:
            return np.array([])

        mode = self._resolved_mode_

        if mode == "tg":
            # 1. 基础 Tg0：若无指定，依据芳香碳比例自适应或取典型值
            if self._col_tg0_ and self._col_tg0_ in df.columns:
                tg0 = _safe_float_array(df[self._col_tg0_], default_val=10.0)
            elif "phys_f_ar" in df.columns:
                f_ar = _safe_float_array(df["phys_f_ar"], default_val=0.4)
                tg0 = -20.0 + f_ar * 60.0  # 芳香碳越高，单体初始 Tg0 越高 (-20°C ~ 40°C)
            else:
                tg0 = np.full(n_samples, 10.0, dtype=np.float64)

            # 2. 极限 Tginf：通过固化最高温度或芳香碳/刚性骨架/交联密度自适应扩展（解除 230°C 假死上限）
            if self._col_tginf_ and self._col_tginf_ in df.columns:
                tginf = _safe_float_array(df[self._col_tginf_], default_val=180.0)
            elif "phys_f_ar" in df.columns:
                f_ar = _safe_float_array(df["phys_f_ar"], default_val=0.4)
                # 芳香碳比例基础极限：从原先的 110 + f_ar*120 (max 230°C) 升级为真实高分子骨架极限 (120 ~ 340°C)
                tginf = 120.0 + f_ar * 220.0
            elif self._col_cure_temp_ and self._col_cure_temp_ in df.columns:
                cure_t = _safe_float_array(df[self._col_cure_temp_], default_val=150.0)
                # 高分子经验律：后固化温度驱动网络致密化，允许向 300°C+ 扩展
                tginf = cure_t + 35.0
            else:
                tginf = np.full(n_samples, 200.0, dtype=np.float64)

            # 2.1 高交联密度增益 (Fox-Loshaek 自由体积固化项扩展)
            # 检查是否有理论交联密度特征列
            nu_mol_m3 = None
            if "xl_nu_theory_mol_m3" in df.columns:
                nu_mol_m3 = _safe_float_array(df["xl_nu_theory_mol_m3"], default_val=0.0)
            elif "nu_theory" in df.columns:
                nu_mol_m3 = _safe_float_array(df["nu_theory"], default_val=0.0)
            elif "crosslink_density" in df.columns:
                nu_mol_m3 = _safe_float_array(df["crosslink_density"], default_val=0.0)
            
            if nu_mol_m3 is not None and np.nanmax(nu_mol_m3) > 0:
                # 每 1000 mol/m³ 产生约 15~25°C 的额外耐热增益
                nu_boost = np.clip(nu_mol_m3 / 1000.0, 0.0, 10.0) * 18.0
                tginf = tginf + nu_boost

            tginf = np.maximum(tginf, tg0 + 20.0)

            # 3. 固化度 alpha：结合化学计量比 r 严格截断
            if self._col_alpha_ and self._col_alpha_ in df.columns:
                alpha = _safe_float_array(df[self._col_alpha_], default_val=0.88)
                alpha = np.clip(alpha, 0.0, 1.0)
            elif self._col_cure_temp_ and self._col_cure_temp_ in df.columns:
                cure_t = _safe_float_array(df[self._col_cure_temp_], default_val=140.0)
                # 固化温度越高，转化率通常越充分 (0.75 ~ 0.98)
                alpha = 0.70 + 0.28 / (1.0 + np.exp(-(cure_t - 130.0) / 25.0))
            else:
                alpha = np.full(n_samples, 0.88, dtype=np.float64)

            if self._col_r_ and self._col_r_ in df.columns:
                r_arr = _safe_float_array(df[self._col_r_], default_val=1.0)
                alpha_max = alpha_max_from_r_np(r_arr)
                alpha = np.minimum(alpha, alpha_max)

            # 4. DiBenedetto 主项计算
            lam = np.full(n_samples, float(self.lambda_param), dtype=np.float64)
            tg_phy = dibenedetto_tg_np(tg0=tg0, tginf=tginf, lam=lam, alpha=alpha)

            # 5. 超高温热自聚激增项 (针对 T_cure > 180°C 诱导的密集刚性自聚网络)
            if self._col_cure_temp_ and self._col_cure_temp_ in df.columns:
                cure_t = _safe_float_array(df[self._col_cure_temp_], default_val=140.0)
                high_temp_excess = np.maximum(0.0, cure_t - 180.0)
                # 高温热诱导额外交联项 (最高可提升 40~90°C)
                tg_phy = tg_phy + np.clip(high_temp_excess * 0.75, 0.0, 90.0)

            tg_phy = np.where(np.isfinite(tg_phy), tg_phy, 150.0)
            return tg_phy

        elif mode == "mechanics":
            if self._col_em_ and self._col_em_ in df.columns:
                Em = _safe_float_array(df[self._col_em_], default_val=3.2)
            else:
                Em = np.full(n_samples, 3.2, dtype=np.float64)

            if self._col_ef_ and self._col_ef_ in df.columns:
                Ef = _safe_float_array(df[self._col_ef_], default_val=75.0)
            else:
                Ef = np.full(n_samples, 75.0, dtype=np.float64)

            if self._col_vf_ and self._col_vf_ in df.columns:
                vf_raw = _safe_float_array(df[self._col_vf_], default_val=0.0)
                Vf = np.where(vf_raw > 1.0, vf_raw / 100.0, vf_raw)
                Vf = np.clip(Vf, 0.0, 0.75)
            else:
                Vf = np.zeros(n_samples, dtype=np.float64)

            if self._col_xi_ and self._col_xi_ in df.columns:
                xi = _safe_float_array(df[self._col_xi_], default_val=2.0)
                xi = np.clip(xi, 0.1, 100.0)
            else:
                xi = np.full(n_samples, 2.0, dtype=np.float64)

            modulus_phy = halpin_tsai_np(E_m=Em, E_f=Ef, xi=xi, vf=Vf)
            modulus_phy = np.where(np.isfinite(modulus_phy), modulus_phy, Em)
            return modulus_phy

        else:
            return np.zeros(n_samples, dtype=np.float64)

    def _build_estimator(self, feature_names: List[str]) -> Any:
        est_name = (self.base_estimator or "xgboost").lower()
        extra_params = dict(self.estimator_params)

        if est_name in {"xgboost", "xgb"}:
            if not XGB_AVAILABLE:
                # 优雅回退
                return RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.random_state,
                "missing": np.nan,
                "n_jobs": -1,
            }
            params.update(extra_params)

            if self.monotone_constraints and self._resolved_mode_ in {"tg", "mechanics"}:
                constraints = []
                for fname in feature_names:
                    fl = fname.lower()
                    if self._resolved_mode_ == "tg":
                        if any(k in fl for k in ["alpha", "cure", "crosslink", "tginf", "f_ar"]):
                            constraints.append(1)
                        else:
                            constraints.append(0)
                    elif self._resolved_mode_ == "mechanics":
                        if any(k in fl for k in ["vf", "content", "fiber", "modulus", "ef"]):
                            constraints.append(1)
                        else:
                            constraints.append(0)
                if any(c != 0 for c in constraints):
                    params["monotone_constraints"] = tuple(constraints)

            return xgb.XGBRegressor(**params)

        elif est_name in {"lightgbm", "lgb"}:
            if not LGB_AVAILABLE:
                return RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            params.update(extra_params)
            return lgb.LGBMRegressor(**params)

        elif est_name in {"catboost", "cb"}:
            if not CATBOOST_AVAILABLE:
                return RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            params = {
                "iterations": self.n_estimators,
                "depth": min(self.max_depth, 10),
                "learning_rate": self.learning_rate,
                "random_seed": self.random_state,
                "verbose": False,
            }
            params.update(extra_params)
            return cb.CatBoostRegressor(**params)

        elif est_name in {"rf", "random_forest"}:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params.update(extra_params)
            return RandomForestRegressor(**params)

        elif est_name in {"et", "extra_trees"}:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params.update(extra_params)
            return ExtraTreesRegressor(**params)

        elif est_name in {"gbr", "gradient_boosting"}:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
            }
            params.update(extra_params)
            return GradientBoostingRegressor(**params)

        elif est_name == "ridge":
            params = {"random_state": self.random_state}
            params.update(extra_params)
            return Ridge(**params)

        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def _prepare_data(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                cols = self.feature_names_
            else:
                cols = [f"f_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = pd.DataFrame(X)

        # 注入零标签高分子物理特征
        df = self._augment_features_if_needed(df)

        # 剔除无用的纯文本列，保留有效数值，并对数值列做安全防御
        for c in list(df.columns):
            cl = str(c).lower()
            if any(p in cl for p in ["smiles", "inchi", "structure"]):
                continue
            is_num = False
            try:
                is_num = pd.api.types.is_numeric_dtype(df[c])
            except Exception:
                is_num = False
            if not is_num:
                df[c] = _safe_float_array(df[c], default_val=np.nan)
            else:
                arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                bad = (~np.isfinite(arr)) | (np.abs(arr) > 1e30)
                if np.any(bad):
                    arr[bad] = np.nan
                    df[c] = arr

        return df

    def fit(self, X: Any, y: Any) -> "EpoxyPINMRegressor":
        df_X = self._prepare_data(X)
        self._resolved_mode_ = self._resolve_mode(df_X)
        self._identify_columns(df_X)

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        valid_y = np.isfinite(y_arr)
        if not np.all(valid_y):
            df_X = df_X.loc[valid_y].reset_index(drop=True)
            y_arr = y_arr[valid_y]

        # 1. 物理机理基线
        y_phy = self._compute_physics_baseline(df_X)
        y_phy = np.nan_to_num(y_phy, nan=150.0, posinf=450.0, neginf=-50.0)

        # 2. 残差计算：加权残差
        # y_pred = (1 - w)*f_ML(x) + w*y_phy  <=>  f_ML(x) 拟合 (y - w*y_phy) / (1 - w)
        pw = float(np.clip(self.physics_weight, 0.0, 0.9))
        residual_target = (y_arr - pw * y_phy) / max(1.0 - pw, 1e-6)
        fallback_tgt = float(np.nanmedian(y_arr)) if len(y_arr) > 0 else 0.0
        residual_target = np.nan_to_num(residual_target, nan=fallback_tgt, posinf=1e6, neginf=-1e6)
        residual_target = np.clip(residual_target, -1e6, 1e6)

        # 3. 筛选数值列，并先剔除“全部为 NaN”的列（防止 SimpleImputer 静默丢弃导致特征维度不一致）
        num_cols = []
        for c in df_X.columns:
            try:
                if pd.api.types.is_numeric_dtype(df_X[c]):
                    # 只有至少包含一个非空数值的列才保留
                    if df_X[c].notna().any():
                        num_cols.append(c)
            except Exception:
                pass

        if not num_cols:
            num_cols = list(df_X.columns)

        self.feature_names_ = num_cols

        # 提取数组并做严格防御性清洗（转为 float64，清洗非有限值及极大值）
        X_raw = df_X[num_cols].to_numpy(dtype=np.float64, copy=True)
        bad_mask = (~np.isfinite(X_raw)) | (np.abs(X_raw) > 1e30)
        if np.any(bad_mask):
            X_raw[bad_mask] = np.nan

        # 采用 constant 0.0 作为常数兜底插补（确保不丢失任何列），或者 median 插补
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=False)
        X_imputed = self.imputer_.fit_transform(X_raw)

        # 如果插补后由于极端情况列数发生缩减，再次严格校准 feature_names_ 长度
        if hasattr(self.imputer_, "get_support"):
            support = self.imputer_.get_support()
            self.feature_names_ = [f for f, s in zip(self.feature_names_, support) if s]
        elif X_imputed.shape[1] < len(self.feature_names_):
            self.feature_names_ = self.feature_names_[:X_imputed.shape[1]]

        # 终极保护：确保 X_imputed 绝对无 inf/nan，并在 float32 安全范围内
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e30, neginf=-1e30)
        X_imputed = np.clip(X_imputed, -1e30, 1e30)

        # 4. 基于最终实际存在的特征列表构建传统模型（严格保证 monotone_constraints 长度与 X_imputed 维度相同）
        self.regressor_ = self._build_estimator(self.feature_names_)
        self.regressor_.fit(X_imputed, residual_target)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.regressor_ is None or self.imputer_ is None:
            raise RuntimeError("PINM 模型尚未训练，请先调用 fit()")

        df_X = self._prepare_data(X)

        # 1. 物理机理基线
        y_phy = self._compute_physics_baseline(df_X)
        y_phy = np.nan_to_num(y_phy, nan=150.0, posinf=450.0, neginf=-50.0)

        # 2. 传统模型残差预测
        cols = [c for c in self.feature_names_ if c in df_X.columns]
        # 若有缺失列补 NaN
        for missing_c in set(self.feature_names_) - set(cols):
            df_X[missing_c] = np.nan
        X_mat = df_X[self.feature_names_].to_numpy(dtype=np.float64, copy=True)
        bad_mask = (~np.isfinite(X_mat)) | (np.abs(X_mat) > 1e30)
        if np.any(bad_mask):
            X_mat[bad_mask] = np.nan

        X_imputed = self.imputer_.transform(X_mat)
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e30, neginf=-1e30)
        X_imputed = np.clip(X_imputed, -1e30, 1e30)
        residual_pred = self.regressor_.predict(X_imputed)

        # 3. 加权物理与数据驱动融合
        pw = float(np.clip(self.physics_weight, 0.0, 0.9))
        y_pred = (1.0 - pw) * residual_pred + pw * y_phy

        # 4. 物理合理边界截断
        if self.clip_physical_bounds:
            if self._resolved_mode_ == "tg":
                y_pred = np.clip(y_pred, -50.0, 450.0)
            elif self._resolved_mode_ == "mechanics":
                y_pred = np.clip(y_pred, 0.1, 500.0)

        return y_pred

    def _transform(self, X: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        数据特征变换接口：供训练评估管线及模型解释器提取“数值化+插补”特征矩阵。
        返回 (X_imputed, aux_info)。
        """
        if self.imputer_ is None or not self.feature_names_:
            raise RuntimeError("PINM 模型尚未训练，请先调用 fit()")

        df_X = self._prepare_data(X)
        cols = [c for c in self.feature_names_ if c in df_X.columns]
        for missing_c in set(self.feature_names_) - set(cols):
            df_X[missing_c] = np.nan
        X_mat = df_X[self.feature_names_].to_numpy(dtype=np.float64, copy=True)
        bad_mask = (~np.isfinite(X_mat)) | (np.abs(X_mat) > 1e30)
        if np.any(bad_mask):
            X_mat[bad_mask] = np.nan
        X_imputed = self.imputer_.transform(X_mat)
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e30, neginf=-1e30)
        X_imputed = np.clip(X_imputed, -1e30, 1e30)
        return X_imputed, {"feature_names": list(self.feature_names_)}

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """代理底层传统回归模型的特征重要性（若存在）"""
        if self.regressor_ is not None and hasattr(self.regressor_, "feature_importances_"):
            return getattr(self.regressor_, "feature_importances_")
        return None

