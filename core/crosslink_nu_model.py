# -*- coding: utf-8 -*-
"""
crosslink_nu_model.py

阶段 1：配方 → 交联密度（ν）嵌入模型

设计定位（两阶段 PINN 的第一阶）：
- 用全部实测 ν 样本（清理后）独立训练一个 ν 模型；
- 冻结后作为 PINN 的"物理编码器"：为任意配方行输出 ν 基线（log 空间），
  PINN 的 ν 潜变量在此基线上学残差，物理读出头（Fox–Loshaek / ν 单调读出）消费该量；
- 推理时无需输入任何 ν 信息——ν 由本模型从配方特征内部算出。

特征 = 窄表配方/工艺数值列（结构列剔除、低基数文本 one-hot）
     + crosslink_physics 的 xl_* 物理特征（ν_theory / Mc / 官能度等）
模型 = XGBoost（表格数据强基线；冻结使用，无需可微）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import crosslink_physics as xphy

DEFAULT_NU_ENCODER_FILENAME = "crosslink_nu_encoder.joblib"
# ν 合理物理区间（mol/m³）：对应 Mc ≈ 120–6000 g/mol（ρ≈1.2 g/cm³）
DEFAULT_NU_BOUNDS = (100.0, 1.0e4)


def default_encoder_path() -> str:
    """ν 编码器默认查找路径：环境变量 → <cwd>/models → 包上级 models 目录。"""
    env = os.environ.get("PINN_NU_ENCODER", "").strip()
    if env and os.path.exists(env):
        return env
    candidates = [
        Path(os.getcwd()) / "models" / DEFAULT_NU_ENCODER_FILENAME,
        Path(__file__).resolve().parent.parent / "models" / DEFAULT_NU_ENCODER_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


class CrosslinkNuEncoder:
    """配方→ν 嵌入模型。fit 一次、冻结复用；picklable（可嵌入 PINN 工件）。"""

    def __init__(
        self,
        nu_bounds: Tuple[float, float] = DEFAULT_NU_BOUNDS,
        use_physics_features: bool = True,
        xgb_params: Optional[Dict] = None,
    ):
        self.nu_bounds = tuple(nu_bounds)
        self.use_physics_features = bool(use_physics_features)
        self.xgb_params = dict(xgb_params or {})

        # fitted
        self.model_ = None
        self.feature_names_: List[str] = []
        self.cat_levels_: Dict[str, List[str]] = {}
        self.median_: Optional[np.ndarray] = None
        self.nu_log_median_: float = float(np.log(2000.0))
        self.provenance_: Dict = {}

    # ---------------- 特征工程（与 PINN 净化逻辑同口径，独立实现保证可独立部署） ----------------

    def _build_features(self, df_raw: pd.DataFrame, fit_categoricals: bool = False) -> pd.DataFrame:
        from .pinn_model import EpoxyPINNRegressor  # 复用黑名单常量，避免两处漂移

        df = df_raw.copy()
        drop_cols = set()
        onehot_frames: List[pd.DataFrame] = []
        if not fit_categoricals:
            cat_levels = dict(self.cat_levels_)
        else:
            cat_levels = {}

        for c in list(df.columns):
            if not isinstance(c, str) or pd.api.types.is_numeric_dtype(df[c]):
                continue
            cl = c.lower()
            if any(p in cl for p in EpoxyPINNRegressor._TEXT_COLS_ALWAYS_DROP) or "smiles" in cl or "inchi" in cl:
                drop_cols.add(c)
                continue
            if any(p in cl for p in EpoxyPINNRegressor._TEXT_COL_BLACKLIST):
                drop_cols.add(c)
                s = df[c].astype(str)
                nuniq = int(s.nunique(dropna=True))
                if 2 <= nuniq <= 12:
                    if fit_categoricals:
                        levels = sorted(s.dropna().unique().tolist())[:12]
                        cat_levels[c] = levels
                    else:
                        levels = cat_levels.get(c)
                        if not levels:
                            continue
                    cat_idx = pd.Categorical(
                        s.where(s.isin(levels), other="__other__"),
                        categories=list(levels) + ["__other__"],
                    )
                    dummies = pd.get_dummies(cat_idx, prefix=c, dtype=float)
                    dummies.index = df.index  # 防非连续索引行错位
                    onehot_frames.append(dummies)
                continue

        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        if onehot_frames:
            df = pd.concat([df] + onehot_frames, axis=1)
        if fit_categoricals:
            self.cat_levels_ = cat_levels

        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1, how="all")
        return df

    # ---------------- 训练 ----------------

    def fit(self, df_raw: pd.DataFrame, nu_values: np.ndarray) -> "CrosslinkNuEncoder":
        nu = np.asarray(nu_values, dtype=float).copy()
        lo, hi = self.nu_bounds
        valid = np.isfinite(nu) & (nu >= lo) & (nu <= hi)
        df_fit = df_raw.loc[valid].reset_index(drop=True)
        nu_fit = nu[valid]
        if len(nu_fit) < 50:
            raise ValueError(f"清理后有效 ν 样本不足（{len(nu_fit)} < 50），请检查数据/边界 {self.nu_bounds}")

        y_log = np.log(nu_fit)
        self.nu_log_median_ = float(np.median(y_log))

        num = self._build_features(df_fit, fit_categoricals=True)

        # 物理特征：ν_theory / Mc / 官能度等（理论基线作为特征交给模型修正）
        if self.use_physics_features:
            xl = xphy.compute_crosslink_features(df_fit)
            xl_num = xl.select_dtypes(include=[np.number])
            xl_num = xl_num.add_prefix("ph_")
            num = pd.concat([num.reset_index(drop=True), xl_num.reset_index(drop=True)], axis=1)

        # 剔除常数列 + 中位数填充 + 标准化
        keep = num.columns[num.notna().sum() > 0]
        num = num[keep]
        self.median_ = num.median(numeric_only=True).to_numpy(dtype=float)
        std = num.std(numeric_only=True, ddof=0).to_numpy(dtype=float)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        self.feature_names_ = list(num.columns)
        self.std_ = std

        X = num.fillna(pd.Series(self.median_, index=num.columns)).to_numpy(dtype=float)
        X = (X - self.median_) / self.std_

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("CrosslinkNuEncoder 需要 xgboost") from exc

        params = {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        params.update(self.xgb_params or {})
        self.model_ = XGBRegressor(**params)
        self.model_.fit(X, y_log)

        self.provenance_ = {
            "n_train": int(len(nu_fit)),
            "n_dropped_dirty": int((~valid).sum()),
            "nu_bounds": [float(lo), float(hi)],
            "use_physics_features": self.use_physics_features,
            "n_features": int(X.shape[1]),
        }
        return self

    # ---------------- 推理 ----------------

    def predict_log_nu(self, df_raw: pd.DataFrame) -> np.ndarray:
        """返回 log ν（NaN 表示该行无法计算，调用方需回退理论/参考值）。"""
        if self.model_ is None:
            raise RuntimeError("CrosslinkNuEncoder 尚未训练")
        n = len(df_raw)
        if n == 0:
            return np.array([], dtype=float)
        num = self._build_features(df_raw, fit_categoricals=False)

        if self.use_physics_features:
            xl = xphy.compute_crosslink_features(df_raw)
            xl_num = xl.select_dtypes(include=[np.number]).add_prefix("ph_")
            num = pd.concat([num.reset_index(drop=True), xl_num.reset_index(drop=True)], axis=1)

        # 对齐训练列：缺列补 NaN（中位数填充），多余列丢弃
        for c in self.feature_names_:
            if c not in num.columns:
                num[c] = np.nan
        num = num[self.feature_names_]
        X = num.fillna(pd.Series(self.median_, index=self.feature_names_)).to_numpy(dtype=float)
        X = (X - self.median_) / self.std_
        with np.errstate(all="ignore"):
            out = self.model_.predict(X).astype(float)
        out = np.clip(out, float(np.log(30.0)), float(np.log(3.0e4)))
        return out

    def predict_nu(self, df_raw: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_nu(df_raw))

    # ---------------- 持久化 ----------------

    def save(self, path: str) -> str:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str) -> "CrosslinkNuEncoder":
        import joblib

        return joblib.load(path)
