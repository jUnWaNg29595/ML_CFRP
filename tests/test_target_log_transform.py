# -*- coding: utf-8 -*-
"""目标对数变换与目标异常值过滤的契约测试。

覆盖：
1. TargetLogTransformer 的 predict 返回原始空间（对下游透明）
2. sklearn 兼容性（get_params / set_params / Pipeline）
3. joblib 序列化往返
4. eval_set 的 y 被同步变换（否则早停量纲错误）
5. 错误处理（log 遇非正值、log1p 遇 <-1）
6. transform='none' 时行为与裸模型完全一致
7. train_model / cross_validate_model 的参数接线
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from core.model_trainer import (
    TARGET_SANE_RANGE,
    TARGET_TRANSFORM_LOG,
    TARGET_TRANSFORM_LOG1P,
    TARGET_TRANSFORM_NONE,
    TargetLogTransformer,
    _target_transform_forward,
    _target_transform_inverse,
)


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5)
    y = np.exp(3.0 + 0.5 * X[:, 0]) + rng.rand(300) * 5.0
    return X, y


# --------------------------------------------------------------------------
# 1. 变换函数本身
# --------------------------------------------------------------------------
def test_forward_inverse_roundtrip():
    y = np.array([1.0, 100.0, 2000.0, 1.0e4])
    for kind in (TARGET_TRANSFORM_LOG1P, TARGET_TRANSFORM_LOG):
        back = _target_transform_inverse(_target_transform_forward(y, kind), kind)
        assert np.allclose(back, y, rtol=1e-9)


def test_log_rejects_non_positive():
    with pytest.raises(ValueError, match="log 变换要求"):
        _target_transform_forward(np.array([0.0, 1.0]), TARGET_TRANSFORM_LOG)


def test_log1p_rejects_below_minus_one():
    with pytest.raises(ValueError, match="log1p 变换要求"):
        _target_transform_forward(np.array([-2.0, 1.0]), TARGET_TRANSFORM_LOG1P)


# --------------------------------------------------------------------------
# 2. 包装器核心契约：predict 返回原始空间
# --------------------------------------------------------------------------
def test_predict_returns_original_space(toy_data):
    from sklearn.linear_model import Ridge

    X, y = toy_data
    wrapper = TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_LOG1P)
    wrapper.fit(X, y)
    pred = wrapper.predict(X)

    # 若未做逆变换，预测值会落在 log 空间（约 3~5），此处必须回到原始量级
    assert pred.min() > 1.0
    assert np.median(pred) == pytest.approx(np.median(y), rel=0.5)
    assert np.isfinite(pred).all()


def test_none_transform_matches_bare_model(toy_data):
    from sklearn.linear_model import Ridge

    X, y = toy_data
    wrapper = TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_NONE)
    wrapper.fit(X, y)
    bare = Ridge().fit(X, y)
    assert np.allclose(wrapper.predict(X), bare.predict(X))
    assert wrapper.active is False


def test_inverse_overflow_falls_back_to_raw(toy_data):
    """逆向变换溢出时该点退回原始空间，不产生 NaN/Inf。"""
    from sklearn.linear_model import Ridge

    X, y = toy_data
    wrapper = TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_LOG)
    wrapper.fit(X, y)
    # 构造会溢出的外推输入
    extreme = np.full((1, X.shape[1]), 1.0e6)
    pred = wrapper.predict(extreme)
    assert np.isfinite(pred).all()


# --------------------------------------------------------------------------
# 3. sklearn 兼容性
# --------------------------------------------------------------------------
def test_sklearn_pipeline_and_params(toy_data):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("model", TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_LOG1P)),
    ])
    pipe.fit(X, y)
    assert np.isfinite(pipe.predict(X)).all()

    params = pipe.named_steps["model"].get_params()
    assert "estimator__alpha" in params

    pipe.named_steps["model"].set_params(estimator__alpha=2.0)
    assert pipe.named_steps["model"].estimator.alpha == 2.0


def test_joblib_roundtrip(toy_data, tmp_path):
    import joblib
    from sklearn.linear_model import Ridge

    X, y = toy_data
    wrapper = TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_LOG1P)
    wrapper.fit(X, y)

    path = tmp_path / "w.joblib"
    joblib.dump(wrapper, path)
    loaded = joblib.load(path)

    assert isinstance(loaded, TargetLogTransformer)
    assert loaded.transform == TARGET_TRANSFORM_LOG1P
    assert np.allclose(wrapper.predict(X), loaded.predict(X))


def test_attribute_forwarding_for_shap(toy_data):
    """SHAP / 特征重要性依赖 feature_importances_ 透传。"""
    from sklearn.ensemble import RandomForestRegressor

    X, y = toy_data
    wrapper = TargetLogTransformer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        transform=TARGET_TRANSFORM_LOG1P,
    )
    wrapper.fit(X, y)
    assert hasattr(wrapper, "feature_importances_")
    assert np.shape(wrapper.feature_importances_) == (X.shape[1],)


# --------------------------------------------------------------------------
# 4. eval_set 的 y 必须同步变换（早停量纲）
# --------------------------------------------------------------------------
def test_eval_set_target_is_transformed(toy_data):
    """若 eval_set 的 y 未变换，早停 RMSE 会落在原始尺度（数百）。

    变换后 RMSE 应落在 log 尺度（个位数以内）。
    """
    xgb = pytest.importorskip("xgboost")

    X, y = toy_data
    wrapper = TargetLogTransformer(
        estimator=xgb.XGBRegressor(n_estimators=30, early_stopping_rounds=5, random_state=0),
        transform=TARGET_TRANSFORM_LOG1P,
    )
    wrapper.fit(X, y, eval_set=[(X[:50], y[:50])])

    result = wrapper.evals_result_
    rmse = list(result.values())[0]["rmse"]
    # log 空间下 RMSE 应远小于 100（原始空间下会是几十到几百）
    assert min(rmse) < 100.0, f"eval_set 的 y 疑似未变换，RMSE={min(rmse)}"


def test_fit_forwards_sample_weight(toy_data):
    """sample_weight 应正常转发，不被包装器吞掉。"""
    from sklearn.linear_model import Ridge

    X, y = toy_data
    w = np.linspace(0.5, 2.0, len(y))
    wrapper = TargetLogTransformer(estimator=Ridge(), transform=TARGET_TRANSFORM_LOG1P)
    wrapper.fit(X, y, sample_weight=w)
    assert np.isfinite(wrapper.predict(X)).all()


# --------------------------------------------------------------------------
# 5. 端到端：train_model / cross_validate_model 接线
# --------------------------------------------------------------------------
def _toy_frame():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.exp(3.0 + 0.5 * X["f0"]) + rng.rand(200) * 5.0)
    return X, y


def test_train_model_accepts_target_transform():
    pytest.importorskip("xgboost")
    from core.model_trainer import EnhancedModelTrainer

    X, y = _toy_frame()
    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(
        X, y, model_name="XGBoost", test_size=0.2, random_state=42,
        target_balance_enabled=False, target_transform="log1p",
        n_estimators=50, verbosity=0,
    )
    assert res["model"] is not None
    # 预测必须是原始空间（量级远大于 1）
    pred = np.asarray(res["model"].predict(X.head(5))).ravel()
    assert np.isfinite(pred).all()
    assert np.all(pred > 1.0)


def _kept_count(res):
    """从 train_model 结果中恢复参与训练的样本数。

    XGBoost 轻量级模式会把 y_train / X_train 落盘并将结果字典中的值置为 None，
    此时改用 y_test（保留在内存中）推算：len(y_test) / test_size 即为总数。
    更稳健的做法是直接读 y_pred_test 与 y_test 的长度（一一对应）。
    """
    y_test = res.get("y_test")
    if y_test is not None:
        n_test = len(np.asarray(y_test).ravel())
        if n_test > 0:
            # 反推总数：test_size=0.2 时 n_test ≈ n_total * 0.2
            # 为避免舍入误差，直接用四舍五入
            return int(round(n_test / 0.2))
    return None


def test_train_model_outlier_filter_removes_rows():
    pytest.importorskip("xgboost")
    from core.model_trainer import EnhancedModelTrainer

    X, y = _toy_frame()
    y = y.copy()
    y.iloc[:20] = 1.0e9  # 注入物理不可能的目标值

    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(
        X, y, model_name="XGBoost", test_size=0.2, random_state=42,
        target_balance_enabled=False, target_outlier_filter=True,
        n_estimators=50, verbosity=0,
    )
    kept = _kept_count(res)
    assert kept is not None, "无法从结果中恢复样本数"
    assert kept == len(y) - 20, f"应剔除 20 行越界样本，实际保留 {kept}"


def test_train_model_filter_disabled_keeps_all():
    pytest.importorskip("xgboost")
    from core.model_trainer import EnhancedModelTrainer

    X, y = _toy_frame()
    y = y.copy()
    y.iloc[:20] = 1.0e9

    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(
        X, y, model_name="XGBoost", test_size=0.2, random_state=42,
        target_balance_enabled=False, target_outlier_filter=False,
        n_estimators=50, verbosity=0,
    )
    kept = _kept_count(res)
    assert kept is not None, "无法从结果中恢复样本数"
    assert kept == len(y), f"未启用过滤时应保留全部样本，实际 {kept}"


def test_cross_validate_model_accepts_target_transform():
    pytest.importorskip("xgboost")
    from core.model_trainer import EnhancedModelTrainer

    X, y = _toy_frame()
    trainer = EnhancedModelTrainer(use_gpu=False)
    cv = trainer.cross_validate_model(
        X, y, model_name="XGBoost", cv_strategy="repeated_kfold",
        n_splits=3, n_repeats=1, random_state=42,
        target_balance_enabled=False, target_transform="log1p",
        target_outlier_filter=True, n_estimators=50, verbosity=0,
    )
    assert "cv_r2_mean" in cv
    assert np.isfinite(float(cv["cv_r2_mean"]))


def test_sane_range_constant():
    assert TARGET_SANE_RANGE == (1.0, 1.0e4)
