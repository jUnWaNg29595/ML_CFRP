# -*- coding: utf-8 -*-
"""主动学习（Active Learning）模块

目标：
    在“高分子/复合材料”小样本场景下，通过模型不确定性驱动下一批实验/模拟选择，
    以更少的标注成本更快提升模型性能或更快找到高性能配方。

本实现是“轻量、可落地”的版本：
    - 不引入 modAL 等额外依赖；直接基于 scikit-learn 实现。
    - 支持回归任务：
        * Gaussian Process（原生输出方差）
        * RandomForest/ExtraTrees（用树集合预测分布估计不确定性）
    - 支持常用采集函数（acquisition function）：
        * 最大不确定性（uncertainty sampling）
        * UCB（Upper Confidence Bound）
        * EI（Expected Improvement）

你可以把这个模块与本项目的“数据处理→分子特征→特征选择→模型训练”流程组合成闭环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from scipy.stats import norm


AcqKind = Literal["uncertainty", "ucb", "ei"]
ModelKind = Literal["gpr", "rf", "etr"]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.values
    if isinstance(x, pd.Series):
        return x.values.reshape(-1, 1)
    return np.asarray(x)


def _safe_ravel(y) -> np.ndarray:
    y = _to_numpy(y)
    return y.reshape(-1)


def _ensure_2d(X) -> np.ndarray:
    X = _to_numpy(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _rf_predict_mu_sigma(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """用树集合的分布估计不确定性。"""
    if not hasattr(model, "estimators_"):
        mu = model.predict(X)
        sigma = np.zeros_like(mu, dtype=float)
        return mu, sigma

    preds = []
    for est in model.estimators_:
        try:
            preds.append(est.predict(X))
        except Exception:
            pass
    if not preds:
        mu = model.predict(X)
        sigma = np.zeros_like(mu, dtype=float)
        return mu, sigma

    P = np.stack(preds, axis=0)
    mu = P.mean(axis=0)
    sigma = P.std(axis=0)
    return mu, sigma


def acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    kind: AcqKind,
    y_best: Optional[float] = None,
    minimize: bool = False,
    xi: float = 0.01,
    kappa: float = 2.0,
) -> np.ndarray:
    """计算采集函数分数，分数越大越优先采样。"""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    sigma = np.maximum(sigma, 1e-12)

    if kind == "uncertainty":
        return sigma

    if kind == "ucb":
        # maximize: mu + kappa*sigma
        # minimize: -(mu - kappa*sigma)  等价于“越小越好”
        if minimize:
            return -(mu - kappa * sigma)
        return mu + kappa * sigma

    if kind == "ei":
        if y_best is None:
            raise ValueError("EI 需要提供 y_best")

        if minimize:
            # improvement = y_best - mu - xi
            imp = (float(y_best) - mu) - float(xi)
        else:
            # improvement = mu - y_best - xi
            imp = (mu - float(y_best)) - float(xi)

        Z = imp / sigma
        return imp * norm.cdf(Z) + sigma * norm.pdf(Z)

    raise ValueError(f"未知 kind: {kind}")


@dataclass
class ActiveLearningResult:
    """主动学习推荐结果。"""

    recommended_indices: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    acq: np.ndarray


class ActiveLearningEngine:
    """主动学习引擎（回归）。"""

    def __init__(
        self,
        *,
        model_kind: ModelKind = "gpr",
        model_params: Optional[Dict] = None,
        random_state: int = 42,
        imputer_strategy: str = "median",
        use_scaler: bool = True,
    ):
        self.model_kind = model_kind
        self.model_params = model_params or {}
        self.random_state = int(random_state)
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.scaler = StandardScaler() if use_scaler else None
        self.model = None

    def _build_model(self):
        if self.model_kind == "gpr":
            # 对小样本较稳：常数核*RBF + WhiteKernel
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

            params = {
                "kernel": kernel,
                "alpha": 0.0,
                "normalize_y": True,
                "random_state": self.random_state,
            }
            params.update(self.model_params)
            return GaussianProcessRegressor(**params)

        if self.model_kind == "rf":
            params = {
                "n_estimators": 300,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params.update(self.model_params)
            return RandomForestRegressor(**params)

        if self.model_kind == "etr":
            params = {
                "n_estimators": 500,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params.update(self.model_params)
            return ExtraTreesRegressor(**params)

        raise ValueError(f"未知 model_kind: {self.model_kind}")

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        X2 = self.imputer.fit_transform(X)
        if self.scaler is not None:
            X2 = self.scaler.fit_transform(X2)
        return X2

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        X2 = self.imputer.transform(X)
        if self.scaler is not None:
            X2 = self.scaler.transform(X2)
        return X2

    def fit(self, X_labeled, y_labeled):
        X = _ensure_2d(X_labeled)
        y = _safe_ravel(y_labeled)

        # 移除 y 中 NaN
        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]
        if X.shape[0] < 3:
            raise ValueError("主动学习至少需要 3 个已标注样本")

        Xp = self._preprocess_fit(X)
        self.model = self._build_model()
        self.model.fit(Xp, y)
        return self

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("ActiveLearningEngine 还未 fit")

        Xp = self._preprocess(X)

        if self.model_kind == "gpr":
            mu, sigma = self.model.predict(Xp, return_std=True)
            return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)

        # 树模型：用树集合 std 估计
        mu, sigma = _rf_predict_mu_sigma(self.model, Xp)
        return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)

    def recommend(
        self,
        X_pool,
        *,
        batch_size: int = 10,
        kind: AcqKind = "uncertainty",
        y_best: Optional[float] = None,
        minimize: bool = False,
        xi: float = 0.01,
        kappa: float = 2.0,
    ) -> ActiveLearningResult:
        Xp = _ensure_2d(X_pool)
        mu, sigma = self.predict(Xp)
        acq = acquisition(mu, sigma, kind=kind, y_best=y_best, minimize=minimize, xi=xi, kappa=kappa)

        batch_size = int(max(1, batch_size))
        idx = np.argsort(-acq)[:batch_size]
        return ActiveLearningResult(
            recommended_indices=idx,
            mu=mu,
            sigma=sigma,
            acq=acq,
        )


def recommend_from_dataframes(
    *,
    df_labeled: pd.DataFrame,
    df_pool: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_kind: ModelKind = "gpr",
    model_params: Optional[Dict] = None,
    acq_kind: AcqKind = "uncertainty",
    batch_size: int = 10,
    minimize: bool = False,
    xi: float = 0.01,
    kappa: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """基于 DataFrame 的便捷接口。

    返回：df_pool 的一个子集，并附加列：al_mu/al_sigma/al_acq。
    """
    if df_labeled is None or df_labeled.empty:
        raise ValueError("df_labeled 为空")
    if df_pool is None or df_pool.empty:
        raise ValueError("df_pool 为空")
    if target_col not in df_labeled.columns:
        raise ValueError(f"df_labeled 不包含目标列: {target_col}")

    missing_cols = [c for c in feature_cols if c not in df_labeled.columns or c not in df_pool.columns]
    if missing_cols:
        raise ValueError(f"特征列在数据中缺失: {missing_cols}")

    X_l = df_labeled[feature_cols]
    y_l = df_labeled[target_col]
    X_p = df_pool[feature_cols]

    # y_best
    y_arr = pd.to_numeric(y_l, errors="coerce").values
    y_arr = y_arr[np.isfinite(y_arr)]
    if y_arr.size == 0:
        raise ValueError("df_labeled 的目标列全是缺失/非数值")
    y_best = float(np.min(y_arr) if minimize else np.max(y_arr))

    engine = ActiveLearningEngine(
        model_kind=model_kind,
        model_params=model_params,
        random_state=random_state,
    )
    engine.fit(X_l, y_l)
    res = engine.recommend(
        X_p,
        batch_size=batch_size,
        kind=acq_kind,
        y_best=y_best,
        minimize=minimize,
        xi=xi,
        kappa=kappa,
    )

    out = df_pool.iloc[res.recommended_indices].copy()
    out["al_mu"] = res.mu[res.recommended_indices]
    out["al_sigma"] = res.sigma[res.recommended_indices]
    out["al_acq"] = res.acq[res.recommended_indices]
    out = out.sort_values("al_acq", ascending=False)
    return out
