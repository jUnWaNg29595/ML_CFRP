import numpy as np
import pandas as pd
import pytest

from core.optimizer import (
    OptimizationEvaluationConfig,
    build_adaptive_regression_strata,
    prepare_regression_optimization,
)


def _reliable_frame(rows=80):
    index = pd.Index(np.arange(1000, 1000 + rows), name="source_row")
    X = pd.DataFrame(
        {
            "resin_feature": np.arange(rows, dtype=float),
            "process_temperature": np.linspace(80.0, 180.0, rows),
        },
        index=index,
    )
    y = pd.Series(np.repeat(np.arange(10, dtype=float), rows // 10), index=index, name="tg")
    return X, y


def test_preflight_drops_only_invalid_targets_and_preserves_dataframe_columns():
    X, y = _reliable_frame()
    y.iloc[0] = np.nan
    y.iloc[1] = np.inf
    y.iloc[2] = -np.inf
    config = OptimizationEvaluationConfig(cv_folds=4, test_size=0.20, random_state=7)

    preflight = prepare_regression_optimization(X, y, config)

    assert list(preflight.X.columns) == ["resin_feature", "process_temperature"]
    assert preflight.removed_target_rows == 3
    assert preflight.X.index.equals(preflight.y.index)
    assert np.isfinite(preflight.y.to_numpy(dtype=float)).all()
    assert set(preflight.outer_train_indices).isdisjoint(preflight.outer_test_indices)
    assert len(preflight.outer_train_indices) + len(preflight.outer_test_indices) == len(preflight.y)


def test_adaptive_strata_reduces_bins_until_every_bin_supports_test_and_cv():
    y = np.asarray([0.0] * 12 + [1.0] * 12 + [2.0] * 12 + [3.0] * 12)

    labels, actual_bins = build_adaptive_regression_strata(
        y,
        cv_folds=4,
        test_size=0.20,
        requested_bins=10,
    )

    counts = pd.Series(labels).value_counts()
    assert 2 <= actual_bins <= 10
    assert counts.min() >= 5


def test_preflight_rejects_targets_that_cannot_support_stratified_cv():
    X = pd.DataFrame({"feature": np.arange(8, dtype=float)})
    y = pd.Series(np.arange(8, dtype=float), name="tg")

    with pytest.raises(ValueError, match="无法构建满足独立测试集和 5 折交叉验证"):
        prepare_regression_optimization(
            X,
            y,
            OptimizationEvaluationConfig(cv_folds=5, test_size=0.20),
        )
