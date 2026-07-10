# -*- coding: utf-8 -*-
"""适用域分析模块"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 统一图表风格
try:
    from .plot_style import apply_global_style
    apply_global_style()
except Exception:
    pass


class ApplicabilityDomainAnalyzer:
    """适用域分析器"""

    def __init__(self, X_train_scaled, n_components=2):
        self.X_train_scaled = X_train_scaled
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        if isinstance(X_train_scaled, pd.DataFrame):
            X_train_scaled = X_train_scaled.values

        # Handle NaN values before PCA
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.X_train_pca = self.pca.fit_transform(X_train_scaled)

        self.has_hull = False
        if len(self.X_train_pca) >= 3:
            try:
                self.hull = ConvexHull(self.X_train_pca)
                self.has_hull = True
            except:
                pass

    def _is_in_hull(self, point_pca):
        if not self.has_hull:
            return True

        try:
            new_hull = ConvexHull(np.concatenate((self.X_train_pca, [point_pca])))
            if np.any(new_hull.vertices == len(self.X_train_pca)):
                return False
            return True
        except:
            return True

    def fit(self):
        """兼容旧接口的fit方法"""
        pass

    def is_within_domain(self, X_new):
        """判断新样本是否在适用域内（已标准化的数据）"""
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values

        X_new_pca = self.pca.transform(X_new)
        is_in = self._is_in_hull(X_new_pca[0])

        # 计算到训练集中心的距离
        center = np.mean(self.X_train_pca, axis=0)
        distance = np.linalg.norm(X_new_pca[0] - center)

        return is_in, distance

    def visualize_domain(self, X_new=None, y_train=None, figsize=(14, 10)):
        """
        可视化适用域（增强版）

        Args:
            X_new: 新样本（已标准化），可选
            y_train: 训练集目标值（用于着色），可选
            figsize: 图表大小

        Returns:
            fig: matplotlib figure对象
        """
        fig = plt.figure(figsize=figsize)

        # 2D可视化
        ax1 = plt.subplot(2, 2, 1)

        # 训练数据散点图
        if y_train is not None:
            scatter = ax1.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1],
                                c=y_train, cmap='viridis', alpha=0.6, s=40,
                                edgecolors='black', linewidths=0.3)
            plt.colorbar(scatter, ax=ax1, label='Target Value')
        else:
            ax1.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1],
                       c='steelblue', alpha=0.5, s=40, label='训练数据',
                       edgecolors='black', linewidths=0.3)

        # 绘制凸包边界
        if self.has_hull:
            for simplex in self.hull.simplices:
                ax1.plot(self.X_train_pca[simplex, 0], self.X_train_pca[simplex, 1],
                        'k-', linewidth=1.5, alpha=0.7)

        # 新样本
        if X_new is not None:
            X_new_pca = self.pca.transform(X_new)
            is_in = self._is_in_hull(X_new_pca[0])
            color = 'green' if is_in else 'red'
            ax1.scatter(X_new_pca[:, 0], X_new_pca[:, 1],
                       c=color, s=300, marker='*', label='新样本',
                       edgecolors='black', linewidth=2, zorder=5)

            status = "在适用域内 ✓" if is_in else "超出适用域 ✗"
            ax1.annotate(status, xy=(X_new_pca[0, 0], X_new_pca[0, 1]),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=11, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        ax1.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax1.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax1.set_title("PCA 适用域分析", fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.4)

        # 解释方差图
        ax2 = plt.subplot(2, 2, 2)
        explained_var = self.pca.explained_variance_ratio_
        ax2.bar(range(1, len(explained_var) + 1), explained_var,
               alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('主成分', fontsize=11)
        ax2.set_ylabel('解释方差比例', fontsize=11)
        ax2.set_title('主成分解释方差', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for i in range(len(explained_var)):
            ax2.text(i+1, explained_var[i], f'{explained_var[i]:.1%}',
                    ha='center', va='bottom', fontsize=9)

        # 密度分布图（PC1）
        ax3 = plt.subplot(2, 2, 3)
        ax3.hist(self.X_train_pca[:, 0], bins=30, alpha=0.7,
                color='steelblue', edgecolor='black', density=True)
        if X_new is not None:
            ax3.axvline(X_new_pca[0, 0], color=color, linestyle='--',
                       linewidth=2, label='新样本')
        ax3.set_xlabel('PC1 值', fontsize=11)
        ax3.set_ylabel('密度', fontsize=11)
        ax3.set_title('PC1 分布', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 密度分布图（PC2）
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(self.X_train_pca[:, 1], bins=30, alpha=0.7,
                color='steelblue', edgecolor='black', density=True)
        if X_new is not None:
            ax4.axvline(X_new_pca[0, 1], color=color, linestyle='--',
                       linewidth=2, label='新样本')
        ax4.set_xlabel('PC2 值', fontsize=11)
        ax4.set_ylabel('密度', fontsize=11)
        ax4.set_title('PC2 分布', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def analyze(self, new_sample_df, scaler):
        """分析新样本是否在适用域内（保留旧接口兼容性）"""
        if isinstance(new_sample_df, pd.DataFrame):
            new_sample = new_sample_df.values
        else:
            new_sample = new_sample_df

        new_sample_scaled = scaler.transform(new_sample)
        new_sample_pca = self.pca.transform(new_sample_scaled)

        is_in_domain = self._is_in_hull(new_sample_pca[0])

        # 使用新的可视化方法
        fig = self.visualize_domain(X_new=new_sample_scaled)

        return is_in_domain, fig
# ============================================================
# [新增] 指纹相似度适用域（Tanimoto）
# ============================================================

def _binarize_fingerprint_matrix(X):
    """将指纹矩阵二值化（>0 视为 1）并转为 uint8"""
    if isinstance(X, pd.DataFrame):
        Xv = X.values
    else:
        Xv = np.asarray(X)
    # NaN -> 0
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    return (Xv > 0).astype(np.uint8)


def _tanimoto_sim_vector(train_bin: np.ndarray, query_bin: np.ndarray):
    """计算 query 与 train 每行的 Tanimoto 相似度向量"""
    # intersection: dot product for binary vectors
    inter = np.dot(train_bin.astype(np.uint16), query_bin.astype(np.uint16))
    a = train_bin.sum(axis=1).astype(np.int32)
    b = int(query_bin.sum())
    union = a + b - inter
    # avoid zero division
    sim = np.zeros(train_bin.shape[0], dtype=float)
    mask = union > 0
    sim[mask] = inter[mask] / union[mask]
    return sim


class TanimotoADAnalyzer:
    """基于 Tanimoto 相似度的适用域分析（适用于 MACCS/Morgan 指纹位向量）"""

    def __init__(self, X_train_fp, threshold: float = 0.25, max_train_samples=None, random_state: int = 42):
        self.threshold = float(threshold)
        self.random_state = int(random_state)
        self.X_train_bin_full = _binarize_fingerprint_matrix(X_train_fp)

        # 可选：采样训练集，避免大数据下计算过慢
        if max_train_samples is not None and self.X_train_bin_full.shape[0] > int(max_train_samples):
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(self.X_train_bin_full.shape[0], size=int(max_train_samples), replace=False)
            self.train_indices_ = np.sort(idx)
            self.X_train_bin = self.X_train_bin_full[self.train_indices_]
        else:
            self.train_indices_ = None
            self.X_train_bin = self.X_train_bin_full

    def analyze_single(self, x_fp_row, top_k: int = 5, threshold=None):
        """分析单个样本，返回 (is_in_domain, sim_max, top_k_df, fig)"""
        thr = self.threshold if threshold is None else float(threshold)

        query_bin = _binarize_fingerprint_matrix(np.asarray(x_fp_row).reshape(1, -1))[0]
        sims = _tanimoto_sim_vector(self.X_train_bin, query_bin)
        sim_max = float(np.max(sims)) if sims.size else 0.0

        # top-k
        k = int(max(1, top_k))
        top_idx = np.argsort(sims)[::-1][:k]
        top_sims = sims[top_idx]

        # 原始训练索引（如发生采样）
        if self.train_indices_ is not None:
            top_train_index = self.train_indices_[top_idx]
        else:
            top_train_index = top_idx

        top_df = pd.DataFrame({
            'train_index': top_train_index,
            'similarity': top_sims
        })

        is_in_domain = sim_max >= thr

        # 可视化：相似度分布直方图 + sim_max
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sims, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(sim_max, linestyle='--', linewidth=2)
        ax.axvline(thr, linestyle=':', linewidth=2)
        ax.set_xlabel('Tanimoto Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Tanimoto Similarity to Training Set')
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        return is_in_domain, sim_max, top_df, fig

    def compute_batch_max_similarity(self, X_query_fp, batch_size: int = 256):
        """批量计算每个 query 的最大相似度（支持分批，避免内存峰值）"""
        Xq = _binarize_fingerprint_matrix(X_query_fp)
        n = Xq.shape[0]
        sim_max = np.zeros(n, dtype=float)

        bs = int(max(1, batch_size))
        for start in range(0, n, bs):
            end = min(n, start + bs)
            for i in range(start, end):
                sims = _tanimoto_sim_vector(self.X_train_bin, Xq[i])
                sim_max[i] = float(np.max(sims)) if sims.size else 0.0
        return sim_max
