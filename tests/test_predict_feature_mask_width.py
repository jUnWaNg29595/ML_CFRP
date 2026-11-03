"""锁定 feature_mask 模型的预测宽度契约。

真实线上报错：
    ValueError: X has 338 features, but SimpleImputer is expecting 381 features

对应模型：results/training_runs/20260923_032815_TabPFN/model_TabPFN.joblib
    artifact.feature_cols = 381                       ← 契约（mask 前）
    pipeline: SimpleImputer(381) → FeatureMaskTransformer(381→338)
              → StandardScaler(338) → TabPFNRegressor(338)
    model.feature_names_in_ = 338                     ← mask **后**

根因：predict_with_model 优先取 model.feature_names_in_（mask 后 338），
把已按契约构建的 (n,381) 矩阵裁成 (n,338)，而 pipeline 首步
SimpleImputer 期望 381 → 宽度不符 → 抛错。

正确口径：pipeline 的输入契约宽度 = 其**第一个** step 的 n_features_in_。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 辅助函数本身
# ---------------------------------------------------------------------------
def test_pipeline_expected_input_width_uses_first_step():
    """必须取 pipeline 首步宽度，而不是末端 estimator 的宽度。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from core.virtual_screening import _pipeline_expected_input_width

    pipe = Pipeline(
        [("imputer", SimpleImputer()), ("model", LinearRegression())]
    )
    pipe.fit(np.zeros((10, 7)), np.zeros(10))
    # 首步 imputer 期望 7 列
    assert _pipeline_expected_input_width(pipe) == 7
    assert _pipeline_expected_input_width(None) is None


def test_pipeline_expected_input_width_prefers_first_step_over_last():
    """首步宽度 != 末步宽度时，必须返回首步宽度（mask 场景）。"""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from core.virtual_screening import _pipeline_expected_input_width

    class _Mask(BaseEstimator, TransformerMixin):
        def __init__(self, mask=None):
            self.mask = mask

        def fit(self, X, y=None):
            self.mask = np.asarray(self.mask, dtype=bool)
            self.n_features_in_ = int(np.asarray(X).shape[1])
            return self

        def transform(self, X):
            return np.asarray(X)[:, self.mask]

    mask = np.array([True] * 3 + [False] * 4)  # 7 → 3
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("feature_mask", _Mask(mask)),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(np.zeros((10, 7)), np.zeros(10))
    assert _pipeline_expected_input_width(pipe) == 7, "必须返回 mask 前的宽度"


# ---------------------------------------------------------------------------
# 真实模型回归
# ---------------------------------------------------------------------------
def _load(path):
    from core.model_io import load_model_artifact_bytes

    return load_model_artifact_bytes(open(path, "rb").read())


MASK_MODEL = "results/training_runs/20260923_032815_TabPFN/model_TabPFN.joblib"
PLAIN_MODEL = "results/training_runs/20260923_032605_XGBoost/model_XGBoost.joblib"


@pytest.mark.parametrize("path", [MASK_MODEL, PLAIN_MODEL])
def test_contract_width_matrix_predicts_without_error(path):
    """按契约宽度构建的矩阵必须能直接预测（不得被裁错）。"""
    import os

    if not os.path.exists(path):
        pytest.skip(f"模型不存在: {path}")

    from core.virtual_screening import build_feature_matrix, predict_with_model

    art = _load(path)
    pipeline = art.get("pipeline")
    model = art.get("model")
    feature_cols = [str(c) for c in (art.get("feature_cols") or [])]
    assert feature_cols

    X = build_feature_matrix(
        feature_cols,
        pd.DataFrame(
            np.random.default_rng(0).normal(size=(6, min(20, len(feature_cols)))),
            columns=feature_cols[: min(20, len(feature_cols))],
        ),
        base_row=pd.Series({c: 0.0 for c in feature_cols}),
    )
    assert X.shape[1] == len(feature_cols)
    preds = predict_with_model(model, X, pipeline=pipeline)
    assert np.asarray(preds).shape[0] == 6


def test_mask_model_prediction_matches_direct_pipeline():
    """修复后预测值必须与「直接把契约矩阵喂 pipeline」逐位一致。"""
    import os

    if not os.path.exists(MASK_MODEL):
        pytest.skip("mask 模型不存在")

    from core.virtual_screening import build_feature_matrix, predict_with_model

    art = _load(MASK_MODEL)
    pipeline = art["pipeline"]
    feature_cols = [str(c) for c in art["feature_cols"]]

    X = build_feature_matrix(
        feature_cols,
        pd.DataFrame(
            np.random.default_rng(1).normal(size=(6, 20)), columns=feature_cols[:20]
        ),
        base_row=pd.Series({c: 0.0 for c in feature_cols}),
    )
    reference = pipeline.predict(X)
    got = predict_with_model(art["model"], X, pipeline=pipeline)
    assert np.allclose(
        np.asarray(got), np.asarray(reference), equal_nan=True
    ), "与直接调 pipeline 不一致"


def test_plain_model_prediction_unchanged():
    """非 mask 模型不得因本次修复改变预测值。"""
    import os

    if not os.path.exists(PLAIN_MODEL):
        pytest.skip("普通模型不存在")

    from core.virtual_screening import build_feature_matrix, predict_with_model

    art = _load(PLAIN_MODEL)
    pipeline = art["pipeline"]
    feature_cols = [str(c) for c in art["feature_cols"]]

    X = build_feature_matrix(
        feature_cols,
        pd.DataFrame(
            np.random.default_rng(2).normal(size=(20, 40)), columns=feature_cols[:40]
        ),
        base_row=pd.Series({c: 0.0 for c in feature_cols}),
    )
    reference = pipeline.predict(X)
    got = predict_with_model(art["model"], X, pipeline=pipeline)
    assert np.allclose(np.asarray(got), np.asarray(reference), equal_nan=True)


def test_uncertainty_path_also_uses_correct_width():
    """predict_with_uncertainty_info 同样不得裁错宽度。"""
    import os

    if not os.path.exists(MASK_MODEL):
        pytest.skip("mask 模型不存在")

    from core.virtual_screening import (
        build_feature_matrix,
        predict_with_uncertainty_info,
    )

    art = _load(MASK_MODEL)
    feature_cols = [str(c) for c in art["feature_cols"]]
    X = build_feature_matrix(
        feature_cols,
        pd.DataFrame(
            np.random.default_rng(3).normal(size=(5, 20)), columns=feature_cols[:20]
        ),
        base_row=pd.Series({c: 0.0 for c in feature_cols}),
    )
    mu, std, source = predict_with_uncertainty_info(
        art["model"], X, pipeline=art["pipeline"], mc_samples=5
    )
    assert np.asarray(mu).shape[0] == 5


def test_narrower_matrix_without_contract_raises_instead_of_guessing():
    """矩阵比契约窄且无契约列名时，必须**显式报错**，不得静默补错位置。

    为什么不能靠「尾部补 NaN 列」蒙过去：feature_mask 是按**位置**取列的，
    补错位置会静默产出错误预测（比报错严重得多）。
    """
    import os

    if not os.path.exists(PLAIN_MODEL):
        pytest.skip("普通模型不存在")

    from core.virtual_screening import predict_with_model

    art = _load(PLAIN_MODEL)
    feature_cols = [str(c) for c in art["feature_cols"]]
    X = pd.DataFrame(
        np.random.default_rng(4).normal(size=(4, 100)), columns=feature_cols[:100]
    )
    with pytest.raises(ValueError, match="输入特征矩阵只有"):
        predict_with_model(art["model"], X, pipeline=art["pipeline"])


def test_narrower_matrix_with_contract_columns_aligns_correctly():
    """给定契约列名时，窄矩阵应正确 reindex 补齐（缺失列为 NaN，交 imputer）。"""
    import os

    if not os.path.exists(PLAIN_MODEL):
        pytest.skip("普通模型不存在")

    from core.virtual_screening import predict_with_model

    art = _load(PLAIN_MODEL)
    feature_cols = [str(c) for c in art["feature_cols"]]
    X = pd.DataFrame(
        np.random.default_rng(4).normal(size=(4, 100)), columns=feature_cols[:100]
    )
    preds = predict_with_model(
        art["model"], X, pipeline=art["pipeline"], feature_cols=feature_cols
    )
    assert np.asarray(preds).shape[0] == 4


def test_explicit_feature_cols_wins_over_model_feature_names():
    """显式传入的契约列不得被 model.feature_names_in_（mask 后）覆盖。"""
    import os

    if not os.path.exists(MASK_MODEL):
        pytest.skip("mask 模型不存在")

    from core.virtual_screening import predict_with_model

    art = _load(MASK_MODEL)
    feature_cols = [str(c) for c in art["feature_cols"]]
    model_names = list(np.asarray(getattr(art["model"], "feature_names_in_", [])))
    assert len(model_names) < len(feature_cols), "该模型应为 mask 场景"

    X = pd.DataFrame(
        np.random.default_rng(5).normal(size=(3, len(feature_cols))),
        columns=feature_cols,
    )
    # 若被 model_names 覆盖会抛 SimpleImputer 宽度错误
    preds = predict_with_model(
        art["model"], X, pipeline=art["pipeline"], feature_cols=feature_cols
    )
    assert np.asarray(preds).shape[0] == 3
