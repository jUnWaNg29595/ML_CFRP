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

    def plot_parity_train_test(self, y_train, y_pred_train, y_test, y_pred_test, target_name="Target"):
        """
        绘制风格化的 实验值 vs 预测值 对比图 (仿照上传图片风格)
        """
        # 设置字体和风格
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        # 1. 绘制对角线 (y=x)
        all_min = min(np.min(y_train), np.min(y_test), np.min(y_pred_train), np.min(y_pred_test))
        all_max = max(np.max(y_train), np.max(y_test), np.max(y_pred_train), np.max(y_pred_test))
        buffer = (all_max - all_min) * 0.05
        limit_min, limit_max = all_min - buffer, all_max + buffer

        ax.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', linewidth=1.5, zorder=1)

        # 2. 绘制训练集 (蓝色圆形)
        # 颜色参考图片中的淡蓝色/青色
        ax.scatter(y_train, y_pred_train,
                   c='#56B4E9', label='Train', marker='o', s=25, alpha=0.8, edgecolors='none', zorder=2)

        # 3. 绘制测试集 (红色菱形)
        # 颜色参考图片中的红色
        ax.scatter(y_test, y_pred_test,
                   c='#D55E00', label='Test', marker='d', s=30, alpha=0.9, edgecolors='none', zorder=3)

        # 4. 设置标签和标题
        # 根据目标变量名自动调整单位（这里假设如果是温度会有 /°C）
        if "degree" in target_name.lower() or "temp" in target_name.lower() or "Tg" in target_name:
            unit = "/°C"
        elif "MPa" in target_name:
            unit = "/MPa"
        else:
            unit = ""

        label_name = target_name.split('_')[0] if '_' in target_name else target_name  # 简化名字

        ax.set_xlabel(f"Experimental {label_name}{unit}", fontsize=12, fontfamily='Arial')
        ax.set_ylabel(f"Predicted {label_name}{unit}", fontsize=12, fontfamily='Arial')

        # 5. 设置坐标轴范围和刻度
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.tick_params(labelsize=11)

        # 6. 图例
        ax.legend(loc='lower right', frameon=False, fontsize=12, handletextpad=0.1)

        # 7. 边框调整
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        plt.tight_layout()

        return fig
