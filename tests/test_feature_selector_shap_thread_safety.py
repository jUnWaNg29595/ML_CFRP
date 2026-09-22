# -*- coding: utf-8 -*-
"""特征选择页「模型重要性 → SHAP重要性」并发回归测试。

对应线上报错（第二次）：
    IndexError: index 2304 is out of bounds for axis 0 with size 2304
    shap/explainers/_kernel.py:610  self.maskMatrix[self.nsamplesAdded, :] = m

根因：shap 的 KernelExplainer 把 nsamplesAdded / maskMatrix / synth_data 存在
**实例上**，explain() 每次调用都会重置它们。旧代码用
    explainer = shap.KernelExplainer(...)          # 单实例
    with ThreadPoolExecutor(...) as ex:            # 多线程共享
        ex.map(lambda b: explainer.shap_values(b), batches)
多线程互相覆盖计数器 → maskMatrix 越界。

N = 2*M + 2048（shap 的 nsamples='auto'），M=128 时正好是 2304。
"""

import numpy as np
import pandas as pd
import pytest

from core.feature_selector import (
    _normalize_shap_values,
    _prepare_shap_feature_frame,
)
from core.model_interpreter import EnhancedModelInterpreter


class _BlackBoxModel:
    """模拟 TabPFN / 神经网络等黑盒模型：有 predict，但无 feature_importances_。"""

    def __init__(self, n_features):
        self.n_features = int(n_features)
        self._coef = np.arange(1, self.n_features + 1, dtype=float)
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr @ self._coef[: arr.shape[1]]


def test_shared_kernel_explainer_across_threads_is_unsafe():
    """固化根因：确认「共享单实例 + 多线程」确实会越界，防止有人改回去。

    这是对 shap 行为的刻画测试（characterization test）：
    - 串行复用同一实例是安全的（shap 每次 explain() 会重置计数器）
    - 并发复用同一实例会 IndexError
    """
    shap = pytest.importorskip("shap")
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.RandomState(0)
    M = 16
    X_bg = pd.DataFrame(rng.randn(40, M))
    X_run = pd.DataFrame(rng.randn(32, M))

    def _run(batches, workers):
        expl = shap.KernelExplainer(lambda a: np.asarray(a, dtype=float).sum(axis=1), X_bg)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda b: expl.shap_values(b, silent=True), batches))

    bs = max(1, len(X_run) // 8)
    batches = [X_run.iloc[i:i + bs] for i in range(0, len(X_run), bs)]

    # 串行：安全
    assert len(_run(batches, 1)) == len(batches)

    # 并发共享同一实例：越界（shap 0.49 的行为）
    with pytest.raises(IndexError):
        _run(batches, 8)


def test_interpreter_path_handles_many_features_without_race():
    """修复后的路径：黑盒模型 + 128 特征（复现线上的 M=128）不再越界。"""
    M = 128
    rng = np.random.RandomState(1)
    X = pd.DataFrame(
        rng.randn(60, M),
        columns=[f"mol_desc_{i:03d}" for i in range(M)],
    )
    y = pd.Series(X.to_numpy() @ np.arange(1, M + 1) / 1000.0)

    model = _BlackBoxModel(M)

    interpreter = EnhancedModelInterpreter(
        model,
        X,
        y,
        X,
        y,
        "测试黑盒模型",
        feature_names=list(X.columns),
        max_samples=20,
        kernel_background=20,
        kernel_nsamples=400,
    )
    shap_values = interpreter.compute_shap_values()

    assert shap_values is not None
    shap_values = _normalize_shap_values(shap_values)
    assert shap_values.ndim == 2
    assert shap_values.shape[1] == M
    assert len(interpreter.feature_names) == M

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame(
        {"Feature": list(interpreter.feature_names), "SHAP_Importance": mean_abs}
    ).sort_values("SHAP_Importance", ascending=False)

    assert len(imp_df) == M
    assert imp_df["SHAP_Importance"].notna().all()
    # 真实特征名被保留（不是 Feature_i 占位名）
    assert imp_df["Feature"].str.startswith("mol_desc_").all()


def test_feature_selector_shap_path_end_to_end_on_ndarray():
    """端到端：ndarray 输入 → 帧规范化 → 解释器 → 重要性表（无 AttributeError / IndexError）。"""
    M = 32
    rng = np.random.RandomState(2)
    names = [f"fp_{i}" for i in range(M)]
    X_train_arr = rng.randn(80, M)      # ndarray，无 .iloc / .sample
    X_test_arr = rng.randn(40, M)

    model = _BlackBoxModel(M)
    y_train = np.zeros(len(X_train_arr))
    y_test = np.zeros(len(X_test_arr))

    X_test_frame, resolved = _prepare_shap_feature_frame(
        X_test_arr, model=model, pipeline=None, name_candidates=[names], feature_mask=None,
    )
    X_train_frame, _ = _prepare_shap_feature_frame(
        X_train_arr, model=model, pipeline=None, name_candidates=[names],
        feature_mask=None, fallback_names=resolved,
    )
    assert resolved == names
    assert X_train_frame.shape[1] == X_test_frame.shape[1] == M

    interpreter = EnhancedModelInterpreter(
        model, X_train_frame, y_train, X_test_frame, y_test, "测试黑盒模型",
        feature_names=resolved, max_samples=15, kernel_background=10, kernel_nsamples=200,
    )
    shap_values = _normalize_shap_values(interpreter.compute_shap_values())

    assert shap_values.shape[1] == M
    feature_names = list(interpreter.feature_names)
    assert len(feature_names) == shap_values.shape[1]

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("SHAP_Importance", ascending=False)
    assert len(imp_df) == M


def test_tree_model_uses_tree_path_not_kernel():
    """树模型应走 TreeExplainer，不应落到 KernelExplainer（也就不会有该竞争）。"""
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestRegressor

    M = 12
    rng = np.random.RandomState(3)
    X = pd.DataFrame(rng.randn(120, M), columns=[f"f_{i}" for i in range(M)])
    y = pd.Series(X.to_numpy() @ np.arange(1, M + 1))
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

    interpreter = EnhancedModelInterpreter(
        model, X, y, X, y, "随机森林",
        feature_names=list(X.columns), max_samples=20,
    )
    shap_values = _normalize_shap_values(interpreter.compute_shap_values())

    assert shap_values.shape == (20, M)
    assert list(interpreter.feature_names) == list(X.columns)
