# -*- coding: utf-8 -*-
"""
tests/test_pinm_model.py
针对升级版 Epoxy PINM (Physics-Informed Machine Learning) 模型的物理基线、残差拟合及调优测试
"""

import numpy as np

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from core.pinm_model import EpoxyPINMRegressor


def test_pinm_tg_fit_and_predict():
    np.random.seed(42)
    n = 100
    r_val = np.random.uniform(0.7, 1.3, size=n)
    cure_temp = np.random.uniform(120.0, 190.0, size=n)
    alpha = np.random.uniform(0.7, 0.98, size=n)
    tg0 = np.full(n, 20.0)
    tginf = np.full(n, 185.0)
    
    # 真实目标由物理 DiBenedetto 加上配方非线性扰动生成
    lam = 0.45
    denom = 1.0 - (1.0 - lam) * alpha
    y_true = tg0 + (tginf - tg0) * (lam * alpha) / denom + (cure_temp - 150.0) * 0.15 + np.random.normal(0, 1.0, size=n)

    df = pd.DataFrame({
        "formulation_r_value": r_val,
        "cure_temperature": cure_temp,
        "alpha": alpha,
        "tg0": tg0,
        "tginf": tginf,
        "f_1": np.random.randn(n),
        "f_2": np.random.randn(n)
    })

    # 使用升级后的 PINM 进行训练
    model = EpoxyPINMRegressor(
        mode="tg",
        base_estimator="rf",
        physics_weight=0.4,
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    model.fit(df, y_true)

    preds = model.predict(df)
    assert len(preds) == n
    assert np.all(np.isfinite(preds))
    
    r2 = r2_score(y_true, preds)
    # 升级后在多特征下拟合优度应显著提升
    assert r2 > 0.90, f"R2 score should be > 0.90, got {r2}"


def test_pinm_mechanics_fit_and_predict():
    np.random.seed(42)
    n = 80
    Em = np.full(n, 3.4)
    Ef = np.full(n, 240.0)
    Vf = np.random.uniform(0.1, 0.65, size=n)
    xi = np.full(n, 2.0)

    # Halpin-Tsai 基准 + 填料非线性强化
    ratio = Ef / Em
    eta = (ratio - 1.0) / (ratio + xi)
    y_true = Em * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf) + Vf**2 * 8.0 + np.random.normal(0, 0.3, size=n)

    df = pd.DataFrame({
        "E_m": Em,
        "E_f": Ef,
        "V_f": Vf,
        "xi": xi,
        "extra_dim": np.random.randn(n)
    })

    model = EpoxyPINMRegressor(
        mode="mechanics",
        base_estimator="rf",
        physics_weight=0.5,
        n_estimators=80,
        random_state=42
    )
    model.fit(df, y_true)

    preds = model.predict(df)
    assert len(preds) == n
    assert np.all(preds > 0)
    
    r2 = r2_score(y_true, preds)
    assert r2 > 0.92, f"R2 score should be > 0.92, got {r2}"


def test_pinm_trainer_integration():
    from core.model_trainer import EnhancedModelTrainer
    trainer = EnhancedModelTrainer()
    
    cat = trainer.get_available_models(include_unavailable=True)
    assert "Epoxy PINM (Physics-Informed ML)" in cat

    avail = trainer.get_available_models(include_unavailable=False)
    assert "Epoxy PINM (Physics-Informed ML)" in avail

    # 尝试实例化模型，分别测试传入 seed 和 random_state
    model = trainer._get_model(
        "Epoxy PINM (Physics-Informed ML)",
        random_state=42,
        seed=42,
        mode="tg",
        base_estimator="rf",
        physics_weight=0.3
    )
    assert isinstance(model, EpoxyPINMRegressor)
    assert model.physics_weight == 0.3
    assert model.random_state == 42

    # 端到端 train_model 与 _transform 流程测试
    n = 30
    df = pd.DataFrame({
        "formulation_r_value": np.random.uniform(0.8, 1.2, size=n),
        "f_1": np.random.randn(n),
        "f_2": np.random.randn(n),
    })
    y = np.random.uniform(100.0, 200.0, size=n)

    res = trainer.train_model(
        df,
        y,
        model_name="Epoxy PINM (Physics-Informed ML)",
        test_size=0.2,
        random_state=42,
        mode="general",
    )
    assert "model" in res
    assert "X_train" in res
    assert "X_test" in res
    assert isinstance(res["X_train"], pd.DataFrame)
    assert len(res["X_train"]) == int(n * 0.8)
    assert res["pipeline"] is not None
    preds = res["pipeline"].predict(df)
    assert len(preds) == n



def test_pinm_inf_and_extreme_values_with_xgboost():
    """回归测试：输入包含 inf, -inf 及超过 float32 边界的大数时，PINM 与 XGBoost 训练不崩溃"""
    from core.model_trainer import EnhancedModelTrainer

    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "formulation_r_value": np.random.uniform(0.8, 1.2, size=n),
        "alpha": np.random.uniform(0.7, 0.95, size=n),
        "feat_norm": np.random.randn(n),
        "feat_extreme": np.random.randn(n),
    })

    # 注入测试极值
    df.loc[2, "feat_extreme"] = 1e45
    df.loc[5, "feat_extreme"] = -1e45
    df.loc[8, "feat_extreme"] = np.inf
    df.loc[12, "feat_extreme"] = -np.inf

    y = np.random.uniform(120.0, 180.0, size=n)

    # 1. 直接训练 PINM (base_estimator="xgboost")
    model = EpoxyPINMRegressor(
        mode="auto",
        base_estimator="xgboost",
        n_estimators=30,
        random_state=42,
    )
    model.fit(df, y)
    preds = model.predict(df)
    assert len(preds) == n
    assert np.all(np.isfinite(preds))

    # 2. 通过 EnhancedModelTrainer.train_model 训练
    trainer = EnhancedModelTrainer()
    res = trainer.train_model(
        X=df,
        y=y,
        model_name="Epoxy PINM (Physics-Informed ML)",
        base_estimator="xgboost",
        n_estimators=30,
        random_state=42,
    )
    assert "model" in res
    assert "pipeline" in res
    assert np.isfinite(res.get("r2", np.nan))
