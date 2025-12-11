# -*- coding: utf-8 -*-
"""可视化模块"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """模型可视化工具"""

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, y_pred_lower=None, y_pred_upper=None):
        """预测值 vs 真实值"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")

        if y_pred_lower is not None and y_pred_upper is not None:
            sorted_idx = np.argsort(y_true)
            y_sorted = np.array(y_true)[sorted_idx] if hasattr(y_true, '__iter__') else y_true
            ax.fill_between(y_sorted, y_pred_lower[sorted_idx], y_pred_upper[sorted_idx],
                            color='gray', alpha=0.2, label='90% CI')

        ax.set_xlabel("真实值", fontsize=12)
        ax.set_ylabel("预测值", fontsize=12)
        ax.set_title(f"{model_name} - 预测性能", fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()

        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        export_df = pd.DataFrame({
            "True": y_true_arr,
            "Predicted": y_pred,
            "Residual": y_true_arr - y_pred
        })

        return fig, export_df

    def plot_residuals(self, y_true, y_pred, model_name):
        """残差分析图"""
        residuals = np.array(y_true) - np.array(y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 残差 vs 预测值
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel("预测值")
        axes[0].set_ylabel("残差")
        axes[0].set_title("残差 vs 预测值")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 残差分布
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel("残差")
        axes[1].set_ylabel("频率")
        axes[1].set_title("残差分布")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, importances, feature_names, model_name, top_n=20):
        """特征重要性图"""
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False)

        top_n = min(top_n, len(importance_df))
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

        ax.barh(range(top_n), top_features['重要性'].values[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['特征'].values[::-1])
        ax.set_xlabel('重要性')
        ax.set_title(f'{model_name} - 特征重要性 (Top {top_n})')

        plt.tight_layout()
        return fig
