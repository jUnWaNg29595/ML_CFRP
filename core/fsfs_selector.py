# -*- coding: utf-8 -*-
"""FSFS (Feature Selection via Feature Similarity) 特征选择算法

核心思想：
1. 选择与目标变量相关性高的特征
2. 去除冗余特征（相互之间高度相似的特征）
3. 平衡特征重要性和多样性
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class FSFSSelector:
    """FSFS特征选择器"""

    def __init__(
        self,
        n_features: int = 10,
        similarity_threshold: float = 0.8,
        importance_metric: str = 'mutual_info',
        similarity_metric: str = 'correlation',
        task_type: str = 'regression',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        参数:
            n_features: 要选择的特征数量
            similarity_threshold: 特征相似度阈值（0-1），超过此值认为特征冗余
            importance_metric: 特征重要性度量方法
                - 'mutual_info': 互信息（默认）
                - 'correlation': 相关系数
                - 'variance': 方差
            similarity_metric: 特征相似度度量方法
                - 'correlation': Pearson相关系数（默认）
                - 'spearman': Spearman秩相关
                - 'cosine': 余弦相似度
            task_type: 任务类型 ('regression' 或 'classification')
            random_state: 随机种子
            n_jobs: 并行计算的CPU核心数（-1表示使用所有核心）
        """
        self.n_features = n_features
        self.similarity_threshold = similarity_threshold
        self.importance_metric = importance_metric
        self.similarity_metric = similarity_metric
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.selected_features_ = None
        self.feature_scores_ = None
        self.similarity_matrix_ = None

    def _calculate_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算特征重要性分数"""
        if self.importance_metric == 'mutual_info':
            if self.task_type == 'regression':
                scores = mutual_info_regression(
                    X, y,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                scores = mutual_info_classif(
                    X, y,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )

        elif self.importance_metric == 'correlation':
            # 向量化计算：一次性计算所有特征与目标的相关系数
            # 使用矩阵运算代替循环
            X_centered = X - X.mean(axis=0)
            y_centered = y - y.mean()

            numerator = np.dot(X_centered.T, y_centered)
            X_std = np.sqrt(np.sum(X_centered**2, axis=0))
            y_std = np.sqrt(np.sum(y_centered**2))

            denominator = X_std * y_std
            # 避免除以零
            denominator = np.where(denominator == 0, 1e-10, denominator)

            scores = np.abs(numerator / denominator)
            scores = np.nan_to_num(scores, 0)

        elif self.importance_metric == 'variance':
            scores = np.var(X, axis=0)

        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")

        return scores

    def _calculate_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算特征间相似度矩阵（优化版）"""
        n_features = X.shape[1]

        # [性能优化] 对于大量特征，使用分块计算
        if n_features > 1000:
            print(f"  [优化] 特征数量较多({n_features})，使用分块计算相似度矩阵...")
            return self._calculate_similarity_matrix_chunked(X)

        similarity_matrix = np.zeros((n_features, n_features))

        if self.similarity_metric == 'correlation':
            # Pearson相关系数
            similarity_matrix = np.corrcoef(X.T)

        elif self.similarity_metric == 'spearman':
            # Spearman秩相关
            from scipy.stats import spearmanr
            similarity_matrix, _ = spearmanr(X, axis=0)

        elif self.similarity_metric == 'cosine':
            # 余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(X.T)

        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # 处理NaN值
        similarity_matrix = np.nan_to_num(similarity_matrix, 0)

        return np.abs(similarity_matrix)

    def _calculate_similarity_matrix_chunked(self, X: np.ndarray, chunk_size: int = 500) -> np.ndarray:
        """
        分块计算相似度矩阵（用于大量特征）

        Args:
            X: 特征矩阵
            chunk_size: 块大小

        Returns:
            相似度矩阵
        """
        n_features = X.shape[1]
        similarity_matrix = np.zeros((n_features, n_features))

        # 分块计算
        for i in range(0, n_features, chunk_size):
            end_i = min(i + chunk_size, n_features)

            if self.similarity_metric == 'correlation':
                # 计算当前块与所有特征的相关系数
                chunk = X[:, i:end_i]
                # 标准化
                chunk_centered = chunk - chunk.mean(axis=0)
                X_centered = X - X.mean(axis=0)

                # 计算相关系数
                numerator = np.dot(chunk_centered.T, X_centered)
                chunk_std = np.sqrt(np.sum(chunk_centered**2, axis=0, keepdims=True)).T
                X_std = np.sqrt(np.sum(X_centered**2, axis=0, keepdims=True))

                denominator = chunk_std @ X_std
                denominator = np.where(denominator == 0, 1e-10, denominator)

                similarity_matrix[i:end_i, :] = np.abs(numerator / denominator)

            elif self.similarity_metric == 'cosine':
                from sklearn.metrics.pairwise import cosine_similarity
                chunk = X[:, i:end_i]
                similarity_matrix[i:end_i, :] = np.abs(cosine_similarity(chunk.T, X.T))

            else:
                # 其他方法回退到完整计算
                if self.similarity_metric == 'correlation':
                    similarity_matrix = np.abs(np.corrcoef(X.T))
                elif self.similarity_metric == 'spearman':
                    from scipy.stats import spearmanr
                    similarity_matrix, _ = spearmanr(X, axis=0)
                    similarity_matrix = np.abs(similarity_matrix)
                break

            print(f"    进度: {end_i}/{n_features} 特征")

        # 处理NaN值
        similarity_matrix = np.nan_to_num(similarity_matrix, 0)

        return similarity_matrix

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], verbose: bool = False):
        """
        拟合FSFS选择器

        参数:
            X: 特征矩阵
            y: 目标变量
            verbose: 是否显示进度信息
        """
        import time
        start_time = time.time()

        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        if verbose:
            print(f"数据规模: {X.shape[0]} 样本 × {X.shape[1]} 特征")

        # 处理缺失值和无穷值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化特征（提高相似度计算的稳定性）
        if verbose:
            print("标准化特征...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # StandardScaler可能会产生NaN（当特征方差为0时），再次处理
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. 计算特征重要性
        if verbose:
            print(f"计算特征重要性 (方法: {self.importance_metric})...")
        importance_start = time.time()
        importance_scores = self._calculate_importance(X_scaled, y)
        self.feature_scores_ = importance_scores
        if verbose:
            print(f"  耗时: {time.time() - importance_start:.2f}秒")

        # 2. 计算特征相似度矩阵
        if verbose:
            print(f"计算特征相似度矩阵 (方法: {self.similarity_metric})...")
        similarity_start = time.time()
        self.similarity_matrix_ = self._calculate_similarity_matrix(X_scaled)
        if verbose:
            print(f"  耗时: {time.time() - similarity_start:.2f}秒")

        # 3. FSFS迭代选择
        if verbose:
            print(f"迭代选择特征 (目标: {self.n_features} 个)...")
        selection_start = time.time()

        selected_indices = []
        candidates = np.argsort(importance_scores)[::-1]  # 按重要性降序排列

        for idx in candidates:
            if len(selected_indices) >= self.n_features:
                break

            # 检查与已选特征的相似度
            is_redundant = False
            for sel_idx in selected_indices:
                if self.similarity_matrix_[idx, sel_idx] > self.similarity_threshold:
                    is_redundant = True
                    break

            # 如果不冗余，则选择该特征
            if not is_redundant:
                selected_indices.append(idx)
                if verbose and len(selected_indices) % 10 == 0:
                    print(f"  已选择 {len(selected_indices)} 个特征...")

        self.selected_features_ = [feature_names[i] for i in selected_indices]
        self.selected_indices_ = selected_indices

        if verbose:
            print(f"  耗时: {time.time() - selection_start:.2f}秒")
            print(f"总耗时: {time.time() - start_time:.2f}秒")
            print(f"最终选择了 {len(selected_indices)} 个特征")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        转换数据，只保留选中的特征

        参数:
            X: 特征矩阵

        返回:
            转换后的特征矩阵
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before transform()")

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.selected_indices_]

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """
        获取特征选择的详细信息

        返回:
            包含特征名称、重要性分数、排名的DataFrame
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before get_feature_info()")

        info = pd.DataFrame({
            'feature': self.selected_features_,
            'importance_score': [self.feature_scores_[i] for i in self.selected_indices_],
            'rank': range(1, len(self.selected_features_) + 1)
        })

        return info.sort_values('importance_score', ascending=False)

    def get_redundancy_info(self, top_n: int = 5) -> pd.DataFrame:
        """
        获取被排除特征的冗余信息

        参数:
            top_n: 显示前N个被排除的特征

        返回:
            包含被排除特征及其最相似特征的DataFrame
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before get_redundancy_info()")

        all_indices = set(range(len(self.feature_scores_)))
        excluded_indices = list(all_indices - set(self.selected_indices_))

        # 按重要性排序被排除的特征
        excluded_sorted = sorted(
            excluded_indices,
            key=lambda x: self.feature_scores_[x],
            reverse=True
        )[:top_n]

        redundancy_info = []
        for idx in excluded_sorted:
            # 找到与该特征最相似的已选特征
            similarities = [self.similarity_matrix_[idx, sel_idx] for sel_idx in self.selected_indices_]
            max_sim_idx = np.argmax(similarities)
            max_similarity = similarities[max_sim_idx]
            similar_feature = self.selected_features_[max_sim_idx]

            redundancy_info.append({
                'excluded_feature_idx': idx,
                'importance_score': self.feature_scores_[idx],
                'similar_to': similar_feature,
                'similarity': max_similarity
            })

        return pd.DataFrame(redundancy_info)


def compare_feature_selection_methods(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10
) -> dict:
    """
    比较不同特征选择方法的结果

    参数:
        X: 特征矩阵
        y: 目标变量
        n_features: 要选择的特征数量

    返回:
        包含不同方法结果的字典
    """
    results = {}

    # 1. FSFS方法
    fsfs = FSFSSelector(n_features=n_features, similarity_threshold=0.8)
    fsfs.fit(X, y)
    results['FSFS'] = {
        'features': fsfs.selected_features_,
        'scores': fsfs.feature_scores_[fsfs.selected_indices_]
    }

    # 2. 基于互信息的简单选择（不考虑冗余）
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    mi_selector = SelectKBest(mutual_info_regression, k=n_features)
    mi_selector.fit(X, y)
    mi_features = X.columns[mi_selector.get_support()].tolist()
    results['Mutual_Info'] = {
        'features': mi_features,
        'scores': mi_selector.scores_[mi_selector.get_support()]
    }

    # 3. 基于方差的选择
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold()
    var_selector.fit(X)
    high_var_features = X.columns[var_selector.get_support()].tolist()[:n_features]
    results['Variance'] = {
        'features': high_var_features,
        'scores': var_selector.variances_[var_selector.get_support()][:n_features]
    }

    return results
