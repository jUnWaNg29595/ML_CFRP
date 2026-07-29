import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from core import model_trainer as trainer_module
from core.model_trainer import (
    EnhancedModelTrainer,
    _build_target_balance_info,
    _compute_regression_bin_metrics,
    _weighted_resample_indices,
)


def test_balance_uses_only_train_values_and_upweights_sparse_tail():
    y_train = np.asarray([0.0] * 18 + [1.0] * 8 + [10.0] * 2)

    info = _build_target_balance_info(
        y_train,
        enabled=True,
        n_bins=4,
        max_weight=3.0,
        random_state=7,
    )

    assert info["enabled"] is True
    assert info["method"] == "ready"
    assert len(info["weights"]) == len(y_train)
    assert info["bin_edges"][-1] == pytest.approx(10.0)
    assert float(np.max(info["weights"])) <= 3.0 + 1e-9
    assert np.mean(info["weights"][y_train == 10.0]) > np.mean(
        info["weights"][y_train == 0.0]
    )
    assert info["fallback_reason"] is None


def test_balance_disables_for_small_or_constant_targets():
    small_info = _build_target_balance_info(
        np.arange(10, dtype=float),
        enabled=True,
        n_bins=10,
        max_weight=3.0,
        random_state=42,
    )
    constant_info = _build_target_balance_info(
        np.ones(24, dtype=float),
        enabled=True,
        n_bins=10,
        max_weight=3.0,
        random_state=42,
    )

    assert small_info["method"] == "disabled"
    assert "样本" in small_info["fallback_reason"]
    assert constant_info["method"] == "disabled"
    assert "常量" in constant_info["fallback_reason"]
    assert np.allclose(small_info["weights"], 1.0)
    assert np.allclose(constant_info["weights"], 1.0)


def test_balance_reports_fallback_reason_after_nonfinite_target_cleaning():
    info = _build_target_balance_info(
        np.asarray(
            [
                *np.arange(19, dtype=float),
                np.nan,
                np.inf,
                -np.inf,
            ]
        ),
        enabled=True,
        n_bins=10,
        max_weight=3.0,
        random_state=42,
    )

    assert info["method"] == "disabled"
    assert info["enabled"] is False
    assert info["fallback_reason"] == "有效训练样本少于20，跳过分箱"
    assert np.allclose(info["weights"], 1.0)


def test_weighted_resample_preserves_length_and_is_reproducible():
    weights = np.asarray([0.2, 0.2, 0.2, 3.0], dtype=float)

    first = _weighted_resample_indices(weights, random_state=123)
    second = _weighted_resample_indices(weights, random_state=123)

    assert len(first) == len(weights)
    assert np.array_equal(first, second)
    assert np.sum(first == 3) >= 1


def test_regression_bin_metrics_reports_each_nonempty_interval():
    y_true = np.asarray([0.0, 1.0, 10.0, 11.0])
    y_pred = np.asarray([0.0, 2.0, 8.0, 12.0])

    metrics = _compute_regression_bin_metrics(
        y_true,
        y_pred,
        bin_edges=np.asarray([-np.inf, 5.0, np.inf]),
    )

    assert len(metrics) == 2
    assert metrics[0]["sample_count"] == 2
    assert metrics[1]["sample_count"] == 2
    assert set(metrics[0]) >= {"bin", "sample_count", "r2", "rmse", "mae"}


class WeightedRecorder(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.fit_sample_weight = None
        self.fit_rows = None

    def fit(self, X, y, sample_weight=None):
        self.fit_sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        self.fit_rows = len(y)
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=float)


class PlainRecorder(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.fit_rows = None
        self.fit_y = None
        self.fit_feature_a = None

    def fit(self, X, y):
        self.fit_rows = len(y)
        self.fit_y = np.asarray(y).copy()
        self.fit_feature_a = np.asarray(X)[:, 0].copy()
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=float)


def _training_frame():
    X = pd.DataFrame(
        {
            "feature_a": np.arange(30, dtype=float),
            "feature_b": np.where(np.arange(30) < 27, 0.0, 1.0),
        }
    )
    y = pd.Series(
        [0.0] * 18 + [1.0] * 9 + [10.0] * 3,
        name="target",
    )
    return X, y


def test_weight_compatible_model_receives_sample_weight(monkeypatch):
    X, y = _training_frame()
    model = WeightedRecorder()
    trainer = EnhancedModelTrainer(use_gpu=False)
    monkeypatch.setattr(trainer, "_get_model", lambda model_name, **params: model)

    result = trainer.train_model(
        X,
        y,
        model_name="线性回归",
        test_size=0.2,
        random_state=42,
        target_balance_enabled=True,
        balance_n_bins=4,
        balance_max_weight=3.0,
    )

    assert model.fit_sample_weight is not None
    assert len(model.fit_sample_weight) == model.fit_rows
    assert result["target_balance"]["method"] == "sample_weight"


def test_model_without_sample_weight_uses_fixed_size_weighted_resampling(monkeypatch):
    X, y = _training_frame()
    model = PlainRecorder()
    trainer = EnhancedModelTrainer(use_gpu=False)
    monkeypatch.setattr(trainer, "_get_model", lambda model_name, **params: model)
    monkeypatch.setattr(
        trainer_module,
        "_model_accepts_sample_weight",
        lambda fitted_model: False,
        raising=False,
    )

    result = trainer.train_model(
        X,
        y,
        model_name="决策树",
        test_size=0.2,
        random_state=42,
        target_balance_enabled=True,
        balance_n_bins=4,
        balance_max_weight=3.0,
    )

    assert model.fit_rows == len(result["y_train"])
    assert result["target_balance"]["method"] == "weighted_resample"
    assert result["target_balance"]["resampled_sample_count"] == len(result["y_train"])
    assert model.fit_y is not None
    assert model.fit_feature_a is not None

    train_feature_a = np.asarray(result["X_train_raw"])[:, 0]
    train_y = np.asarray(result["y_train"])
    assert set(model.fit_feature_a).issubset(set(train_feature_a))
    expected_y = dict(zip(train_feature_a.astype(int), train_y))
    assert np.array_equal(
        model.fit_y,
        np.asarray([expected_y[int(row_id)] for row_id in model.fit_feature_a]),
    )
    assert len(np.unique(model.fit_feature_a)) < len(model.fit_feature_a)
    assert np.sum(model.fit_y == 10.0) > np.sum(train_y == 10.0)


def test_xgboost_early_stopping_never_receives_final_test_set(monkeypatch):
    X, y = _training_frame()
    model = WeightedRecorder()
    trainer = EnhancedModelTrainer(use_gpu=False)
    monkeypatch.setattr(trainer, "_get_model", lambda model_name, **params: model)
    monkeypatch.setattr(trainer_module, "XGBOOST_AVAILABLE", True)
    monkeypatch.setattr(trainer_module, "XGBRegressor", WeightedRecorder)

    captured = {}

    def fake_fit(X_fit, y_fit, **kwargs):
        captured["fit_rows"] = len(y_fit)
        captured["fit_y"] = np.asarray(y_fit).copy()
        captured["eval_set"] = kwargs.get("eval_set")
        captured["eval_feature_a"] = [
            np.asarray(valid_X)[:, 0].copy()
            for valid_X, _ in captured["eval_set"]
        ]
        model.value_ = float(np.mean(y_fit))
        return model

    monkeypatch.setattr(model, "fit", fake_fit)

    result = trainer.train_model(
        X,
        y,
        model_name="XGBoost",
        test_size=0.2,
        random_state=42,
        target_balance_enabled=True,
    )

    assert captured["eval_set"]
    final_test_feature_a = set(np.asarray(result["X_test"])[:, 0])
    assert all(
        set(valid_feature_a).isdisjoint(final_test_feature_a)
        for valid_feature_a in captured["eval_feature_a"]
    )
    assert result["target_balance"]["method"] in {"sample_weight", "weighted_resample", "disabled"}


def test_cross_validation_recomputes_balance_per_fold(monkeypatch):
    X, y = _training_frame()
    y = pd.Series(np.arange(len(y), dtype=float), name="target")
    trainer = EnhancedModelTrainer(use_gpu=False)
    calls = []
    feature_a_by_y = dict(zip(y.to_numpy(), X["feature_a"].to_numpy()))

    original_builder = trainer_module._build_target_balance_info

    def recording_builder(y_train, **kwargs):
        call_y = np.asarray(y_train, dtype=float)
        calls.append({feature_a_by_y[value] for value in call_y})
        return original_builder(y_train, **kwargs)

    monkeypatch.setattr(trainer_module, "_build_target_balance_info", recording_builder)
    monkeypatch.setattr(
        trainer,
        "_get_model",
        lambda model_name, **params: WeightedRecorder(),
    )

    result = trainer.cross_validate_model(
        X,
        y,
        model_name="线性回归",
        cv_strategy="stratified_kfold",
        n_splits=3,
        n_repeats=1,
        random_state=42,
        target_balance_enabled=True,
        balance_n_bins=4,
        balance_max_weight=3.0,
    )

    assert len(calls) == 3
    original_feature_a = set(X["feature_a"])
    assert all(call_ids < original_feature_a for call_ids in calls)
    assert all(len(call_ids) < len(original_feature_a) for call_ids in calls)
    assert all(
        calls[index] != calls[other_index]
        for index in range(len(calls))
        for other_index in range(index + 1, len(calls))
    )
    assert len(result["fold_target_balance"]) == 3
