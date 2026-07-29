import numpy as np
import pandas as pd
from pathlib import Path

from core.model_trainer import EnhancedModelTrainer


def test_regression_training_excludes_only_missing_targets():
    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "feature_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 6.0], name="target")

    result = EnhancedModelTrainer(use_gpu=False).train_model(
        X,
        y,
        model_name="线性回归",
        test_size=0.4,
        random_state=42,
        drop_missing_rows=True,
    )

    assert len(result["y_train"]) + len(result["y_test"]) == 5


def test_classification_training_excludes_only_missing_targets():
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feature_b": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, np.nan, 1, 0, 1], name="target")

    result = EnhancedModelTrainer(use_gpu=False).train_model(
        X,
        y,
        model_name="逻辑回归分类",
        test_size=0.25,
        random_state=42,
        drop_missing_rows=True,
    )

    assert len(result["y_train"]) + len(result["y_test"]) == 7


def test_regression_cross_validation_excludes_only_missing_targets():
    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], name="target")

    result = EnhancedModelTrainer(use_gpu=False).cross_validate_model(
        X,
        y,
        model_name="线性回归",
        cv_strategy="kfold",
        n_splits=2,
        n_repeats=1,
        random_state=42,
        drop_missing_rows=True,
    )

    assert len(result["oof_true"]) == 7


def test_exported_training_script_keeps_rows_with_missing_features():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    script_start = app_source.index("def load_and_train():")
    script_end = app_source.index('if __name__ == "__main__":', script_start)
    training_script = app_source[script_start:script_end]

    assert "complete_mask" not in training_script
    assert "SimpleImputer(strategy='median')" in training_script
