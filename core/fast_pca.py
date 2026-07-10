#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速PCA降维优化模块
针对大规模数据集的PCA性能优化
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FastPCAOptimizer:
    """快速PCA优化器 - 针对大规模数据集优化"""

    def __init__(self,
                 variance_threshold: float = 0.95,
                 min_components: int = 5,
                 auto_select_method: bool = True,
                 batch_size: Optional[int] = None):
        """
        初始化快速PCA优化器

        Args:
            variance_threshold: 累计解释方差阈值
            min_components: 最小保留主成分数
            auto_select_method: 是否自动选择最优方法
            batch_size: 增量PCA的批次大小（None则自动计算）
        """
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.auto_select_method = auto_select_method
        self.batch_size = batch_size
        self.scaler = None
        self.pca = None
        self.method_used = None

    def _select_method(self, n_samples: int, n_features: int) -> str:
        """
        根据数据规模自动选择最优PCA方法

        Args:
            n_samples: 样本数
            n_features: 特征数

        Returns:
            方法名称: 'standard', 'randomized', 'incremental', 'truncated'
        """
        # 数据规模阈值
        LARGE_SAMPLES = 10000
        LARGE_FEATURES = 1000
        HUGE_FEATURES = 5000

        # 决策树
        if n_features > HUGE_FEATURES:
            # 超大特征数：使用增量PCA或截断SVD
            if n_samples > LARGE_SAMPLES:
                return 'incremental'
            else:
                return 'truncated'
        elif n_features > LARGE_FEATURES:
            # 大特征数：使用随机化PCA
            return 'randomized'
        elif n_samples > LARGE_SAMPLES:
            # 大样本数：使用增量PCA
            return 'incremental'
        else:
            # 中小规模：标准PCA
            return 'standard'

    def fit_transform(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     method: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        执行快速PCA降维

        Args:
            X: 输入数据
            method: 指定方法 ('standard', 'randomized', 'incremental', 'truncated')
                   None则自动选择

        Returns:
            (降维后的数据, 统计信息字典)
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        n_samples, n_features = X_array.shape

        # 自动选择方法
        if method is None and self.auto_select_method:
            method = self._select_method(n_samples, n_features)
        elif method is None:
            method = 'standard'

        self.method_used = method

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # 根据方法执行PCA
        if method == 'standard':
            X_pca, stats = self._standard_pca(X_scaled, n_features)
        elif method == 'randomized':
            X_pca, stats = self._randomized_pca(X_scaled, n_features)
        elif method == 'incremental':
            X_pca, stats = self._incremental_pca(X_scaled, n_features)
        elif method == 'truncated':
            X_pca, stats = self._truncated_svd(X_scaled, n_features)
        else:
            raise ValueError(f"未知方法: {method}")

        stats['method'] = method
        stats['n_samples'] = n_samples
        stats['n_features'] = n_features

        return X_pca, stats

    def _standard_pca(self, X_scaled: np.ndarray, n_features: int) -> Tuple[np.ndarray, dict]:
        """标准PCA"""
        self.pca = PCA(
            n_components=self.variance_threshold,
            random_state=42
        )
        X_pca = self.pca.fit_transform(X_scaled)

        # 确保至少保留min_components个主成分
        if X_pca.shape[1] < self.min_components:
            self.pca = PCA(n_components=self.min_components, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)

        stats = {
            'n_components': X_pca.shape[1],
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'total_variance_explained': self.pca.explained_variance_ratio_.sum()
        }
        return X_pca, stats

    def _randomized_pca(self, X_scaled: np.ndarray, n_features: int) -> Tuple[np.ndarray, dict]:
        """随机化PCA - 适合大特征数"""
        # 估算需要的主成分数
        n_comp_estimate = min(int(n_features * 0.5), 500)

        self.pca = PCA(
            n_components=n_comp_estimate,
            svd_solver='randomized',
            random_state=42,
            iterated_power=3  # 提高精度
        )
        X_pca = self.pca.fit_transform(X_scaled)

        # 根据方差阈值截断
        cum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_keep = max(
            np.searchsorted(cum_var, self.variance_threshold) + 1,
            self.min_components
        )
        n_keep = min(n_keep, X_pca.shape[1])

        X_pca = X_pca[:, :n_keep]

        stats = {
            'n_components': n_keep,
            'explained_variance_ratio': self.pca.explained_variance_ratio_[:n_keep],
            'total_variance_explained': cum_var[n_keep-1]
        }
        return X_pca, stats

    def _incremental_pca(self, X_scaled: np.ndarray, n_features: int) -> Tuple[np.ndarray, dict]:
        """增量PCA - 适合大样本数"""
        n_samples = X_scaled.shape[0]

        # 自动计算批次大小
        if self.batch_size is None:
            batch_size = min(max(n_samples // 10, 100), 5000)
        else:
            batch_size = self.batch_size

        # 估算需要的主成分数
        n_comp_estimate = min(int(n_features * 0.5), 500)

        self.pca = IncrementalPCA(
            n_components=n_comp_estimate,
            batch_size=batch_size
        )

        # 分批拟合
        for i in range(0, n_samples, batch_size):
            batch = X_scaled[i:i+batch_size]
            self.pca.partial_fit(batch)

        # 转换
        X_pca = self.pca.transform(X_scaled)

        # 根据方差阈值截断
        cum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_keep = max(
            np.searchsorted(cum_var, self.variance_threshold) + 1,
            self.min_components
        )
        n_keep = min(n_keep, X_pca.shape[1])

        X_pca = X_pca[:, :n_keep]

        stats = {
            'n_components': n_keep,
            'explained_variance_ratio': self.pca.explained_variance_ratio_[:n_keep],
            'total_variance_explained': cum_var[n_keep-1],
            'batch_size': batch_size
        }
        return X_pca, stats

    def _truncated_svd(self, X_scaled: np.ndarray, n_features: int) -> Tuple[np.ndarray, dict]:
        """截断SVD - 适合超大特征数"""
        # 估算需要的主成分数
        n_comp_estimate = min(int(n_features * 0.3), 300)
        n_comp_estimate = max(n_comp_estimate, self.min_components)

        self.pca = TruncatedSVD(
            n_components=n_comp_estimate,
            random_state=42,
            algorithm='randomized',
            n_iter=5
        )
        X_pca = self.pca.fit_transform(X_scaled)

        # 根据方差阈值截断
        cum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_keep = max(
            np.searchsorted(cum_var, self.variance_threshold) + 1,
            self.min_components
        )
        n_keep = min(n_keep, X_pca.shape[1])

        X_pca = X_pca[:, :n_keep]

        stats = {
            'n_components': n_keep,
            'explained_variance_ratio': self.pca.explained_variance_ratio_[:n_keep],
            'total_variance_explained': cum_var[n_keep-1]
        }
        return X_pca, stats

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """对新数据应用已拟合的PCA转换"""
        if self.pca is None or self.scaler is None:
            raise ValueError("请先调用fit_transform()进行拟合")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)
        X_pca = self.pca.transform(X_scaled)

        # 如果使用了截断，应用相同的截断
        if hasattr(self, '_n_keep'):
            X_pca = X_pca[:, :self._n_keep]

        return X_pca


def fast_pca_transform(X: Union[pd.DataFrame, np.ndarray],
                      variance_threshold: float = 0.95,
                      min_components: int = 5,
                      method: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """
    快速PCA降维 - 便捷函数

    Args:
        X: 输入数据
        variance_threshold: 累计解释方差阈值
        min_components: 最小保留主成分数
        method: 指定方法 (None则自动选择)

    Returns:
        (降维后的数据, 统计信息字典)
    """
    optimizer = FastPCAOptimizer(
        variance_threshold=variance_threshold,
        min_components=min_components,
        auto_select_method=(method is None)
    )

    return optimizer.fit_transform(X, method=method)


if __name__ == "__main__":
    # 测试
    print("快速PCA优化器测试")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    n_samples = 5000
    n_features = 2000

    X_test = np.random.randn(n_samples, n_features)

    print(f"测试数据: {n_samples} 样本 × {n_features} 特征")

    # 测试自动选择
    import time
    start = time.time()
    X_pca, stats = fast_pca_transform(X_test, variance_threshold=0.95)
    elapsed = time.time() - start

    print(f"\n自动选择方法: {stats['method']}")
    print(f"降维后: {X_pca.shape[1]} 个主成分")
    print(f"解释方差: {stats['total_variance_explained']:.2%}")
    print(f"耗时: {elapsed:.2f} 秒")
