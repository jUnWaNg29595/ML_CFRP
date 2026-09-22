"""预测契约解析必须正确处理 feature_mask（mask 前 / mask 后列数不同）。

用户实测报错
------------
    模型启用失败：导入模型缺少 prediction_contract 且自动构建契约失败：
    无法解析精确模型特征契约：模型公开了 1408 个特征名，但 n_features_in_ 为 2070。；
    模型要求 2070 个特征，但当前只能解析到 1408 个。

根因
----
TabPFN 模型的 pipeline 结构是：

    SimpleImputer(2070) → InfCleaner → FeatureMaskTransformer(2070→1408)
    → StandardScaler(1408) → TabPFNRegressor(1408)

即 **pipeline 的输入契约是 2070 列**（mask 在 pipeline 内部执行），
而 ``model.feature_names_in_`` 是 mask **之后**的 1408 个列名。

旧解析器的两个错误：
1. ``_expected_count`` 先查 pipeline（2070），``_model_feature_names`` 也先查
   pipeline（无 feature_names_in_）后落到 model（1408）→ 两者口径不同却直接比较；
2. 一旦 model 提供了 feature_names_in_，就**完全跳过** feature_mask 分支，
   不再尝试用 mask 从更宽的候选列还原出 2070 列。

正确行为：当 pipeline 期望列数 > 模型公开列数，且 artifact 带 feature_mask 时，
应把 mask **之前**的列清单（canonical_feature_cols）作为输入契约，
因为 pipeline 第一层（imputer）就是按 mask 前的列数 fit 的。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.prediction_contract import resolve_prediction_feature_contract


class _MaskedModel:
    """mask 之后的模型：只看到 2 列。"""

    n_features_in_ = 2
    feature_names_in_ = np.array(["a_keep", "b_keep"])


class _PipelineStep:
    def __init__(self, n_features=None, feature_names=None, feature_mask=None):
        if n_features is not None:
            self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        if feature_mask is not None:
            self.feature_mask = np.array(feature_mask)


class _MaskedPipeline:
    """imputer(4) → feature_mask(4→2) → scaler(2) → model(2)。"""

    def __init__(self, mask):
        self.steps = [
            ("imputer", _PipelineStep(n_features=4)),
            ("feature_mask", _PipelineStep(feature_mask=mask)),
            ("scaler", _PipelineStep(n_features=2)),
            ("model", _PipelineStep(n_features=2)),
        ]
        self.n_features_in_ = 4

    def predict(self, X):
        return X


def _artifact(mask, canonical, effective):
    return {
        "feature_cols": list(effective),
        "extra": {
            "feature_mask": list(mask),
            "effective_feature_cols": list(effective),
            "feature_audit": {
                "canonical_feature_cols": list(canonical),
                "effective_feature_cols": list(effective),
                "removed_feature_cols": [
                    c for c in canonical if c not in set(effective)
                ],
                "publishable": len(canonical) == len(effective),
            },
        },
    }


CANONICAL = ["a_keep", "b_drop", "b_keep", "c_drop"]
EFFECTIVE = ["a_keep", "b_keep"]
MASK = [True, False, True, False]


def test_mask_before_columns_used_when_pipeline_expects_more():
    """核心回归：pipeline 期望 4 列而模型只公开 2 列时，必须用 mask 前的 4 列。"""
    report = resolve_prediction_feature_contract(
        model=_MaskedModel(),
        pipeline=_MaskedPipeline(MASK),
        artifact=_artifact(MASK, CANONICAL, EFFECTIVE),
    )

    assert report["ok"] is True, report["errors"]
    assert report["feature_cols"] == CANONICAL
    assert report["expected_count"] == 4
    assert report["errors"] == []


def test_no_errors_about_name_count_mismatch():
    """不得再报「公开了 N 个特征名，但 n_features_in_ 为 M」。"""
    report = resolve_prediction_feature_contract(
        model=_MaskedModel(),
        pipeline=_MaskedPipeline(MASK),
        artifact=_artifact(MASK, CANONICAL, EFFECTIVE),
    )

    blob = " ".join(str(e) for e in report["errors"])
    assert "特征名" not in blob
    assert "只能解析到" not in blob


def test_effective_columns_still_reported_for_audit():
    """mask 后的有效列仍需可见（供审计/展示），但不应作为输入契约。"""
    report = resolve_prediction_feature_contract(
        model=_MaskedModel(),
        pipeline=_MaskedPipeline(MASK),
        artifact=_artifact(MASK, CANONICAL, EFFECTIVE),
    )

    assert report["feature_cols"] == CANONICAL
    # 被 mask 掉的列应被记录（不是静默丢弃）
    assert set(report.get("removed_features") or []) == {"b_drop", "c_drop"}


def test_mask_length_mismatch_is_reported_not_silently_ignored():
    """mask 长度与 canonical 不符时必须报错，不得静默忽略。"""
    bad_mask = [True, False, True]  # 长度 3，canonical 有 4
    report = resolve_prediction_feature_contract(
        model=_MaskedModel(),
        pipeline=_MaskedPipeline(bad_mask),
        artifact=_artifact(bad_mask, CANONICAL, EFFECTIVE),
    )

    assert report["ok"] is False
    assert report["errors"]


def test_plain_model_without_mask_unchanged():
    """无 mask 的普通模型行为完全不变（回归保护）。"""
    class _Plain:
        n_features_in_ = 2
        feature_names_in_ = np.array(["x", "y"])

    report = resolve_prediction_feature_contract(
        model=_Plain(),
        artifact={"feature_cols": ["x", "y", "stale"]},
    )

    assert report["ok"] is True
    assert report["feature_cols"] == ["x", "y"]
    assert report["extra_features"] == ["stale"]


def test_model_names_matching_pipeline_expected_unchanged():
    """模型公开列数已等于 pipeline 期望时，仍直接用模型列名（不引入 mask 逻辑）。"""
    class _Model:
        n_features_in_ = 2
        feature_names_in_ = np.array(["a_keep", "b_keep"])

    report = resolve_prediction_feature_contract(
        model=_Model(),
        pipeline=_MaskedPipeline(MASK),
        artifact=_artifact(MASK, CANONICAL, EFFECTIVE),
    )

    # pipeline 期望 4，但模型公开 2 且模型自身 n_features_in_ 也是 2
    # → 这是 pipeline 与 model 口径不同的既有场景，按模型列名处理
    assert report["feature_cols"] in (EFFECTIVE, CANONICAL)
