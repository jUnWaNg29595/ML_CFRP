# -*- coding: utf-8 -*-
"""批量置换 SHAP 快速路径（TabPFN 等黑盒模型）的正确性回归测试。"""

import numpy as np
import pandas as pd
import pytest

from core.model_interpreter import (
    BATCHED_PERMUTATION_MODELS,
    EnhancedModelInterpreter,
)


class _LinearBlackBox:
    """线性黑盒模型：单行参考背景下 SHAP 有解析解 φ_i = w_i (x_i − bg_i)。"""

    def __init__(self, w, b=3.0):
        self.w = np.asarray(w, dtype=np.float64)
        self.b = float(b)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


def _build_interpreter(w, n_train=400, n_test=40, M=25, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(M)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, M)), columns=cols)
    X_test = pd.DataFrame(rng.normal(size=(n_test, M)), columns=cols)
    return EnhancedModelInterpreter(
        _LinearBlackBox(w),
        X_train,
        np.zeros(n_train),
        X_test,
        np.zeros(n_test),
        model_name="TabPFN",
        feature_names=cols,
        max_samples=n_test,
        kernel_background=50,
        kernel_nsamples=200,
    )


def test_tabpfn_fast_path_matches_linear_analytic_solution():
    rng = np.random.default_rng(1)
    M = 25
    w = rng.normal(size=M)
    interp = _build_interpreter(w)
    bg_row = interp._resolve_background_row(np.random.default_rng(42))

    shap_values, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=8
    )
    expected = (interp.X_test.to_numpy() - bg_row[None, :]) * w[None, :]
    rel_err = np.max(np.abs(shap_values - expected)) / np.max(np.abs(expected))
    assert rel_err < 0.02


def test_tabpfn_fast_path_satisfies_efficiency_exactly():
    w = np.random.default_rng(2).normal(size=20)
    interp = _build_interpreter(w, M=20)
    shap_values, base_values = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=3
    )
    full_pred = interp.model.predict(interp.X_test.to_numpy())
    residual = np.max(np.abs(full_pred - (shap_values.sum(axis=1) + base_values)))
    assert residual < 1e-6


def test_tabpfn_fast_path_is_invariant_to_chunking_and_batching():
    w = np.random.default_rng(3).normal(size=15)
    interp = _build_interpreter(w, n_test=24, M=15)

    tiny, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=3, chunk_rows=29, batch_rows_cap=50,
        random_state=42,
    )
    huge, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=3, chunk_rows=10**6, batch_rows_cap=10**9,
        random_state=42,
    )
    assert np.max(np.abs(tiny - huge)) < 1e-9


def test_compute_shap_values_uses_fast_path_and_matches_sample_matrix():
    w = np.random.default_rng(4).normal(size=12)
    interp = _build_interpreter(w, n_test=20, M=12)
    values = interp.compute_shap_values()

    assert values is not None
    assert values.shape == (len(interp._X_sample), 12)
    assert interp._base_values is not None and interp._base_values.shape == (len(values),)
    assert np.isfinite(values).all()


def test_non_blackbox_models_do_not_take_fast_path():
    w = np.random.default_rng(5).normal(size=10)
    interp = _build_interpreter(w, M=10)
    interp.model_name = "随机森林"  # 树模型应继续走 TreeExplainer
    assert not interp._should_use_batched_permutation_shap()

    interp.model_name = "XGBoost"
    assert not interp._should_use_batched_permutation_shap()


def test_blackbox_model_registry_contains_tabpfn():
    assert "TabPFN" in BATCHED_PERMUTATION_MODELS


# ---------------------------------------------------------------------------
# [性能优化 2026-09] top-K 特征筛选 + TabPFN 降级预测器的回归测试
# ---------------------------------------------------------------------------

def test_batched_shap_topk_mode_matches_analytic_solution_on_selected_cols():
    """top-K 模式：被选中的列保持解析解 φ_i = w_i (x_i − bg_i)，未选中列记 0。"""
    rng = np.random.default_rng(6)
    M, K = 40, 12
    w = rng.normal(size=M)
    interp = _build_interpreter(w, n_test=15, M=M)
    top_idx = np.arange(K)  # 任意选前 K 列
    shap_values, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=8, top_feature_idx=top_idx
    )
    bg_row = interp._resolve_background_row(np.random.default_rng(42))
    X = interp.X_test.to_numpy()

    assert shap_values.shape == (len(X), M)
    # 未选中列 SHAP 必须严格为 0（在 coalition 中恒为背景）
    assert np.all(shap_values[:, K:] == 0.0)
    # 选中列与解析解一致（独立特征 + 线性模型 ⇒ 无交互项）
    expected = (X[:, :K] - bg_row[None, :K]) * w[None, :K]
    rel_err = np.max(np.abs(shap_values[:, :K] - expected)) / np.max(np.abs(expected))
    assert rel_err < 0.02


def test_batched_shap_topk_consistency_gap_equals_unexplained_contribution():
    """top-K 模式的一致性残差 = 未解释列脱离背景的联合贡献（可精确验证）。"""
    w = np.random.default_rng(7).normal(size=30)
    interp = _build_interpreter(w, n_test=10, M=30)
    bg_row = interp._resolve_background_row(np.random.default_rng(42))
    top_idx = np.arange(10)
    shap_values, base_values = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=4, top_feature_idx=top_idx
    )
    X = interp.X_test.to_numpy()
    full_pred = interp.model.predict(X)
    resid = full_pred - (shap_values.sum(axis=1) + base_values)
    # 未选列固定在背景 → 残差 = Σ_{i∉top} w_i (x_i − bg_i)
    expected_gap = (X[:, 10:] - bg_row[None, 10:]) @ w[10:]
    assert np.max(np.abs(resid - expected_gap)) < 1e-8


def test_batched_shap_topk_invariant_to_chunking():
    """top-K 模式同样对 chunk/batch 参数不变。"""
    w = np.random.default_rng(8).normal(size=18)
    interp = _build_interpreter(w, n_test=12, M=18)
    top_idx = np.arange(10)
    tiny, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=3, chunk_rows=17, batch_rows_cap=40,
        random_state=42, top_feature_idx=top_idx,
    )
    huge, _ = interp._compute_batched_permutation_shap(
        interp.X_test, n_permutations=3, chunk_rows=10**6, batch_rows_cap=10**9,
        random_state=42, top_feature_idx=top_idx,
    )
    assert np.max(np.abs(tiny - huge)) < 1e-9


def test_create_fast_shap_predictor_passthrough_non_tabpfn():
    """非 TabPFN 黑盒不做降级，原样返回模型自身的 predict。"""
    w = np.random.default_rng(9).normal(size=8)
    interp = _build_interpreter(w, M=8)
    assert interp._create_fast_shap_predictor() == interp.model.predict


def test_select_top_k_features_disabled_for_low_dim():
    """特征数低于阈值时不启用 top-K（返回 None）。"""
    w = np.random.default_rng(10).normal(size=20)
    interp = _build_interpreter(w, M=20)
    assert interp._select_top_k_features(20) is None
