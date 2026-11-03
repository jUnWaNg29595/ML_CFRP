# -*- coding: utf-8 -*-
"""
tests/test_imbalanced_split.py
测试目标变量长尾与不均衡分布下的数据划分增强策略：
1. _tail_balanced_split（稀有高值区间保底抽样配额）
2. _extrapolation_split（高值外推边界划分）
3. analyze_split_target_distribution（各区间覆盖率诊断报表）
"""

import numpy as np
import pandas as pd
import pytest

from core.model_trainer import EnhancedModelTrainer, analyze_split_target_distribution


def test_tail_balanced_split_coverage():
    """验证针对 1-3 极多、4-10 极少的不均衡数据，长尾均衡切分在 4-10 区间的测试集样本数明显优于纯随机切分"""
    np.random.seed(42)
    n_dense = 90
    n_sparse = 10
    n_total = n_dense + n_sparse

    y_dense = np.random.uniform(1.0, 3.0, size=n_dense)
    y_sparse = np.random.uniform(4.0, 10.0, size=n_sparse)
    y = np.concatenate([y_dense, y_sparse])

    X = pd.DataFrame({
        'feat_1': np.random.randn(n_total),
        'feat_2': y * 0.5 + np.random.randn(n_total),
    })

    trainer = EnhancedModelTrainer()
    test_size = 0.20

    # 1. 常规随机切分
    tr_rand, te_rand = trainer._resolve_split(X, y, test_size=test_size, random_state=42, split_strategy='random')
    sparse_in_rand = np.sum(y[te_rand] >= 4.0)

    # 2. 长尾均衡切分
    tr_tail, te_tail = trainer._resolve_split(X, y, test_size=test_size, random_state=42, split_strategy='tail_balanced', n_bins=8)
    sparse_in_tail = np.sum(y[te_tail] >= 4.0)

    # 验证稀疏区间在长尾切分下获得了更多、更有统计代表性的测试集样本
    assert sparse_in_tail >= sparse_in_rand, f"Tail-balanced split should guarantee >= random ({sparse_in_tail} vs {sparse_in_rand})"
    # 验证稀疏区间的测试集采样比例达到预期保底配额 (>= 25%)
    assert sparse_in_tail / n_sparse >= 0.25
    # 验证训练集没有被掏空，至少保留了 50% 稀疏样本供拟合
    assert np.sum(y[tr_tail] >= 4.0) >= n_sparse * 0.5
    # 验证测试集总体规模保持在合理容差内 (18 ~ 23)
    assert 18 <= len(te_tail) <= 23


def test_extrapolation_split():
    """验证外推划分将最高的极端值样本切入测试集"""
    np.random.seed(42)
    y = np.linspace(1.0, 10.0, 100)
    X = pd.DataFrame({'f': np.random.randn(100)})

    trainer = EnhancedModelTrainer()
    tr_ext, te_ext = trainer._resolve_split(X, y, test_size=0.20, random_state=42, split_strategy='extrapolation')

    assert len(te_ext) == 20
    assert len(tr_ext) == 80
    # 测试集中的目标值应该严格大于训练集中的所有目标值
    assert np.min(y[te_ext]) >= np.max(y[tr_ext])


def test_split_diagnostics_structure():
    """验证区间划分诊断报表输出结构完整"""
    y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0])
    train_idx = np.array([0, 1, 2, 4, 6])
    test_idx = np.array([3, 5, 7])

    diag = analyze_split_target_distribution(y, train_idx, test_idx, n_bins=3)
    assert "intervals" in diag
    assert len(diag["intervals"]) == 3
    assert diag["n_train"] == 5
    assert diag["n_test"] == 3
    for it in diag["intervals"]:
        assert "interval" in it
        assert "total_count" in it
        assert "test_ratio_pct" in it
