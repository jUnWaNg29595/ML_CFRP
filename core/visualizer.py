# -*- coding: utf-8 -*-
"""可视化模块"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# 设置中文字体，防止乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """模型可视化工具"""

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, y_pred_lower=None, y_pred_upper=None):
        """预测值 vs 真实值 (基础版 - 用于仅有测试集的情况)"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # 确保输入是 numpy array
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        # 默认颜色
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5, c='#87CEFA', label='Data')

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")

        if y_pred_lower is not None and y_pred_upper is not None:
            sorted_idx = np.argsort(y_true)
            y_sorted = y_true[sorted_idx]
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

        # 生成导出数据
        export_df = pd.DataFrame({
            "True_Value": y_true,
            "Predicted_Value": y_pred,
            "Residual": y_true - y_pred
        })

        return fig, export_df

    def plot_residuals(self, y_true, y_pred, model_name):
        """残差分析图"""
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 残差 vs 预测值
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors="k", linewidth=0.5, c='#87CEFA')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel("预测值")
        axes[0].set_ylabel("残差")
        axes[0].set_title("残差 vs 预测值")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 残差分布
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#87CEFA')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel("残差")
        axes[1].set_ylabel("频率")
        axes[1].set_title("残差分布")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # 生成导出数据
        export_df = pd.DataFrame({
            "Predicted_Value": y_pred,
            "Residual": residuals
        })

        return fig, export_df

    def plot_feature_importance(self, importances, feature_names, model_name, top_n=20):
        """特征重要性图"""
        # 简单修复长度不一致问题
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        top_n = min(top_n, len(importance_df))
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        ax.barh(range(top_n), top_features['Importance'].values[::-1], color='#87CEFA', edgecolor='k', alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['Feature'].values[::-1], fontsize=10)
        ax.set_xlabel('重要性')
        ax.set_title(f'{model_name} - 特征重要性 (Top {top_n})')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()

        return fig, importance_df

    def plot_parity_train_test(self, y_train, y_pred_train, y_test, y_pred_test, target_name="Target"):
        """
        绘制风格化的 实验值 vs 预测值 对比图
        颜色修复：训练集(天蓝 #87CEFA)，测试集(橙红 #FF4500)
        """
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        y_train = np.array(y_train).ravel()
        y_pred_train = np.array(y_pred_train).ravel()
        y_test = np.array(y_test).ravel()
        y_pred_test = np.array(y_pred_test).ravel()

        # 1. 对角线
        all_min = min(np.min(y_train), np.min(y_test), np.min(y_pred_train), np.min(y_pred_test))
        all_max = max(np.max(y_train), np.max(y_test), np.max(y_pred_train), np.max(y_pred_test))
        buffer = (all_max - all_min) * 0.05
        limit_min, limit_max = all_min - buffer, all_max + buffer

        ax.plot([limit_min, limit_max], [limit_min, limit_max], color='gray', linestyle='--', linewidth=1.5, zorder=1)

        # 2. 训练集 - 天蓝色圆形
        r2_tr = r2_score(y_train, y_pred_train)
        ax.scatter(y_train, y_pred_train,
                   c='#87CEFA', label=f'Train ($R^2$={r2_tr:.3f})', 
                   marker='o', s=30, alpha=0.8, edgecolors='none', zorder=2)

        # 3. 测试集 - 橙红色菱形
        r2_te = r2_score(y_test, y_pred_test)
        ax.scatter(y_test, y_pred_test,
                   c='#FF4500', label=f'Test ($R^2$={r2_te:.3f})', 
                   marker='d', s=40, alpha=0.9, edgecolors='none', zorder=3)

        # 4. 标签
        unit = "/°C" if "Tg" in target_name or "temp" in target_name.lower() else ""
        label_name = target_name.split('_')[0]
        ax.set_xlabel(f"Experimental {label_name}{unit}", fontsize=12)
        ax.set_ylabel(f"Predicted {label_name}{unit}", fontsize=12)

        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.tick_params(labelsize=11)
        ax.legend(loc='lower right', frameon=False, fontsize=11)

        for spine in ax.spines.values(): spine.set_linewidth(1.0)
        plt.tight_layout()

        # 导出
        df_tr = pd.DataFrame({"True": y_train, "Pred": y_pred_train, "Set": "Train"})
        df_te = pd.DataFrame({"True": y_test, "Pred": y_pred_test, "Set": "Test"})
        return fig, pd.concat([df_tr, df_te], ignore_index=True)
