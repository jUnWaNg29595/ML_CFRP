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
