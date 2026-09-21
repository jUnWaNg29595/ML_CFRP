# -*- coding: utf-8 -*-
"""特征选择页「模型重要性 → SHAP重要性」回归测试。

对应线上报错：
    AttributeError: 'numpy.ndarray' object has no attribute 'iloc'
    core/feature_selector.py:3034  batches = [X_sample.iloc[i:i+batch_size] ...]

根因：X_test 在部分训练路径（GNN / PINN / 导入模型）中是 ndarray，
旧代码先 `X_test.sample(...)`（ndarray 无 sample）或退化为 `X_test.copy()`
（ndarray 复制仍是 ndarray），随后在 KernelExplainer 分批时调用 `.iloc` 崩溃。
"""

import numpy as np
import pandas as pd
import pytest

from core.feature_selector import (
    _as_feature_frame,
    _expected_shap_feature_count,
    _normalize_shap_values,
    _prepare_shap_feature_frame,
)


class _FakeModelWithNames:
    """模拟已训练估计器：只暴露 feature_names_in_。"""

    def __init__(self, names):
        self.feature_names_in_ = np.asarray(names, dtype=object)


class _FakePipeline:
    def __init__(self, n_features):
        self.n_features_in_ = int(n_features)


def test_prepare_shap_frame_converts_ndarray_to_dataframe():
    """核心回归：ndarray 输入必须产出带 .iloc / .sample 的 DataFrame。"""
    X = np.random.RandomState(0).randn(200, 5)

    frame, names = _prepare_shap_feature_frame(
        X,
        model=None,
        pipeline=None,
        name_candidates=[None],
        feature_mask=None,
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (200, 5)
    # 旧代码在此处崩溃的调用必须可用
    assert len(frame.sample(n=50, random_state=42)) == 50
    assert frame.iloc[0:10].shape == (10, 5)
    assert len(names) == 5


def test_prepare_shap_frame_restores_real_names_for_ndarray():
    real_names = [f"mol_desc_{i:03d}" for i in range(6)]
    X = np.arange(48, dtype=float).reshape(8, 6)

    frame, names = _prepare_shap_feature_frame(
        X,
        model=None,
        pipeline=None,
        name_candidates=[real_names],
        feature_mask=None,
    )

    assert names == real_names
    assert list(frame.columns) == real_names


def test_prepare_shap_frame_prefers_matrix_columns_over_stale_names():
    real_names = ["resin_xtb_gap", "curing_agent_xtb_gap"]
    X = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=real_names)

    frame, names = _prepare_shap_feature_frame(
        X,
        model=None,
        pipeline=None,
        name_candidates=[["stale_a", "stale_b"]],
        feature_mask=None,
    )

    assert names == real_names


def test_prepare_shap_frame_applies_training_feature_mask_for_full_width_matrix():
    """X_test 仍是掩码前的完整宽度时，应按训练期掩码裁剪到模型消费的列数。"""
    original_names = ["a", "b", "c", "d"]
    mask = [True, False, True, False]
    X = np.random.RandomState(1).randn(20, 4)

    frame, names = _prepare_shap_feature_frame(
        X,
        model=_FakeModelWithNames(["a", "c"]),
        pipeline=None,
        name_candidates=[original_names],
        feature_mask=mask,
    )

    assert frame.shape[1] == 2
    assert names == ["a", "c"]


def test_prepare_shap_frame_does_not_double_apply_mask():
    """已经是掩码后的矩阵（宽度 == mask.sum()）不能被重复裁剪。"""
    mask = [True, False, True, False]
    X = np.random.RandomState(2).randn(20, 2)

    frame, names = _prepare_shap_feature_frame(
        X,
        model=_FakeModelWithNames(["a", "c"]),
        pipeline=None,
        name_candidates=[["a", "c"]],
        feature_mask=mask,
    )

    assert frame.shape[1] == 2
    assert names == ["a", "c"]


def test_prepare_shap_frame_falls_back_when_all_candidates_placeholder():
    X = np.arange(12, dtype=float).reshape(3, 4)
    fallback = ["f1", "f2", "f3", "f4"]

    frame, names = _prepare_shap_feature_frame(
        X,
        model=None,
        pipeline=None,
        name_candidates=[None],
        feature_mask=None,
        fallback_names=fallback,
    )

    assert names == fallback


def test_expected_shap_feature_count_reads_pipeline_then_model():
    assert _expected_shap_feature_count(_FakeModelWithNames(["a", "b", "c"])) == 3
    assert _expected_shap_feature_count(_FakeModelWithNames(["a"]), _FakePipeline(7)) == 7
    assert _expected_shap_feature_count(None, None) is None


def test_normalize_shap_values_handles_explanation_list_and_3d():
    class _FakeExplanation:
        def __init__(self, values):
            self.values = values

    values_2d = np.ones((5, 3))
    assert _normalize_shap_values(_FakeExplanation(values_2d)).shape == (5, 3)
    assert _normalize_shap_values([values_2d]).shape == (5, 3)

    # 多输出 3D -> 沿输出维平均
    values_3d = np.stack([values_2d, values_2d * 3], axis=-1)
    assert _normalize_shap_values(values_3d).shape == (5, 3)
    # 单输出 3D -> 取第 0 层
    assert _normalize_shap_values(values_2d[:, :, None]).shape == (5, 3)
    # 1D -> (1, n)
    assert _normalize_shap_values(np.arange(4.0)).shape == (1, 4)


def test_as_feature_frame_coerces_object_dtype_to_numeric():
    frame = _as_feature_frame(np.array([["1.5", "2.5"], ["3.0", "x"]], dtype=object))
    assert frame.shape == (2, 2)
    assert frame.dtypes.unique().tolist() == [np.dtype("float64")]
    assert np.isnan(frame.iloc[1, 1])


def test_ndarray_kernel_batching_path_no_longer_raises():
    """端到端模拟：ndarray X_test 走 KernelExplainer 分批分支不再抛 AttributeError。"""
    X_test = np.random.RandomState(3).randn(300, 4)

    frame, _ = _prepare_shap_feature_frame(
        X_test, model=None, pipeline=None, name_candidates=[None], feature_mask=None,
    )
    max_samples = min(1000, len(frame))
    X_sample = frame.sample(n=int(max_samples), random_state=42) if len(frame) > max_samples else frame.copy()
    X_sample = X_sample.reset_index(drop=True)

    n_jobs = 8
    batch_size = max(1, len(X_sample) // n_jobs)
    batches = [X_sample.iloc[i:i + batch_size] for i in range(0, len(X_sample), batch_size)]

    assert len(batches) > 1
    assert sum(len(b) for b in batches) == len(X_sample)
    assert all(isinstance(b, pd.DataFrame) for b in batches)
