# -*- coding: utf-8 -*-
"""测试优化后的主动学习模块 (core/active_learning.py)"""

import numpy as _np

if not hasattr(_np, "long"):
    _np.long = int
if not hasattr(_np, "ulong"):
    _np.ulong = int

import pytest
import numpy as np
import pandas as pd

from core.active_learning import (
    ActiveLearningEngine,
    select_diverse_batch,
    recommend_from_dataframes,
    acquisition,
)


def test_select_diverse_batch_avoids_clusters():
    """验证多样性批次选择能够有效打散特征空间中紧密聚集的候选点。"""
    np.random.seed(42)
    # 构造两个簇：簇A（高采集得分但点全挤在一起），簇B（稍低得分但空间互补）
    cluster_a = np.array([[0.0, 0.0], [0.01, 0.01], [0.02, 0.02], [0.01, -0.01]])
    cluster_b = np.array([[10.0, 10.0], [10.01, 10.01]])
    X_pool = np.vstack([cluster_a, cluster_b])

    # 采集分数：簇A全为 0.99~1.0，簇B为 0.90
    acq_scores = np.array([1.0, 0.99, 0.98, 0.97, 0.90, 0.89])

    # 如果无多样性（diversity_weight=0.0），前3个必然全在簇A中
    greedy_idx = select_diverse_batch(X_pool, acq_scores, batch_size=3, diversity_weight=0.0)
    assert set(greedy_idx).issubset({0, 1, 2, 3})

    # 如果开启多样性（diversity_weight=0.5），应跳出簇A，包含簇B的点
    diverse_idx = select_diverse_batch(X_pool, acq_scores, batch_size=3, diversity_weight=0.5)
    assert len(diverse_idx) == 3
    # 至少包含簇B中的一个样本（索引4或5）
    assert (4 in diverse_idx) or (5 in diverse_idx)


def test_ard_gpr_fit_and_recommend():
    """验证 ARD 高维各向异性 GPR 模型正常训练与推荐，兼容性良好。"""
    np.random.seed(42)
    n_features = 6
    X_labeled = np.random.randn(20, n_features)
    y_labeled = X_labeled[:, 0] * 2.0 + X_labeled[:, 1] * 0.5 + np.random.randn(20) * 0.1

    X_pool = np.random.randn(50, n_features)

    engine = ActiveLearningEngine(model_kind="gpr", random_state=42)
    engine.fit(X_labeled, y_labeled)

    # 验证 ARD 核函数 length_scale 维度是否与特征数一致
    # 结构为 (ConstantKernel * RBF) + WhiteKernel，对应 k1.k2 为 RBF
    rbf_kernel = engine.model.kernel_.k1.k2
    assert len(rbf_kernel.length_scale) == n_features

    res = engine.recommend(X_pool, batch_size=5, kind="ei", y_best=float(np.max(y_labeled)), diversity_weight=0.3)
    assert len(res.recommended_indices) == 5
    assert len(res.mu) == 50
    assert len(res.sigma) == 50
    assert (res.sigma >= 0).all()


def test_committee_model_integration():
    """验证 Committee (GPR + RF + ETR) 异质委员会模型的不确定性估计正常运作。"""
    np.random.seed(42)
    X_labeled = np.random.randn(15, 4)
    y_labeled = np.sum(X_labeled ** 2, axis=1)

    X_pool = np.random.randn(30, 4)

    engine = ActiveLearningEngine(model_kind="committee", random_state=42)
    engine.fit(X_labeled, y_labeled)

    assert isinstance(engine.model, list)
    assert len(engine.model) == 3

    mu, sigma = engine.predict(X_pool)
    assert len(mu) == 30
    assert len(sigma) == 30
    assert (sigma > 0).all()


def test_recommend_from_dataframes_backward_compatibility():
    """测试 DataFrame 便捷接口完整向后兼容性。"""
    df_l = pd.DataFrame({
        "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "target": [10.0, 12.0, 15.0, 18.0, 20.0]
    })
    df_p = pd.DataFrame({
        "feat_1": [6.0, 7.0, 8.0, 9.0],
        "feat_2": [1.5, 2.5, 3.5, 4.5],
        "other": ["a", "b", "c", "d"]
    })

    # 旧调用签名（不带 diversity_weight）应完全正常工作
    out = recommend_from_dataframes(
        df_labeled=df_l,
        df_pool=df_p,
        feature_cols=["feat_1", "feat_2"],
        target_col="target",
        model_kind="gpr",
        acq_kind="ucb",
        batch_size=2,
    )

    assert len(out) == 2
    assert "al_mu" in out.columns
    assert "al_sigma" in out.columns
    assert "al_acq" in out.columns
    assert "other" in out.columns
