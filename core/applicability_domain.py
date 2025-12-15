# -*- coding: utf-8 -*-
"""适用域分析模块"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ApplicabilityDomainAnalyzer:
    """适用域分析器"""

    def __init__(self, X_train_scaled, n_components=2):
        self.X_train_scaled = X_train_scaled
        self.pca = PCA(n_components=n_components)
        
        if isinstance(X_train_scaled, pd.DataFrame):
            X_train_scaled = X_train_scaled.values
        
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

    def analyze(self, new_sample_df, scaler):
        """分析新样本是否在适用域内"""
        if isinstance(new_sample_df, pd.DataFrame):
            new_sample = new_sample_df.values
        else:
            new_sample = new_sample_df

        new_sample_scaled = scaler.transform(new_sample)
        new_sample_pca = self.pca.transform(new_sample_scaled)

        is_in_domain = self._is_in_hull(new_sample_pca[0])

        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1],
                   c='gray', alpha=0.5, label='训练数据', s=30)

        if self.has_hull:
            for simplex in self.hull.simplices:
                ax.plot(self.X_train_pca[simplex, 0], self.X_train_pca[simplex, 1], 'k-', linewidth=1)

        color = 'green' if is_in_domain else 'red'
        ax.scatter(new_sample_pca[:, 0], new_sample_pca[:, 1],
                   c=color, s=200, marker='*', label='新样本', edgecolors='black', linewidth=2)

        status = "在适用域内 ✓" if is_in_domain else "超出适用域 ✗"
        ax.annotate(status, xy=(new_sample_pca[0, 0], new_sample_pca[0, 1]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, color=color, fontweight='bold')

        ax.set_xlabel("主成分 1", fontsize=12)
        ax.set_ylabel("主成分 2", fontsize=12)
        ax.set_title("适用域分析 (PCA)", fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
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
