# -*- coding: utf-8 -*-
"""可视化模块（美化版）"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 统一全局图表风格
from .plot_style import (
    apply_global_style, style_axes,
    TRAIN_COLOR, TEST_COLOR, ACCENT_COLOR, PALETTE, BG_COLOR, TEXT_COLOR,
)

apply_global_style()


class Visualizer:
    """模型可视化工具（美化版）"""

    @staticmethod
    def _safe_metrics(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_true.size == 0:
            return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }

    @staticmethod
    def _format_metric_block(label, metrics):
        def _fmt(value):
            return f"{value:.3f}" if math.isfinite(value) else "N/A"

        return (
            f"{label}\n"
            f"R2: {_fmt(metrics['r2'])}\n"
            f"RMSE: {_fmt(metrics['rmse'])}\n"
            f"MAE: {_fmt(metrics['mae'])}"
        )

    def plot_paper_style_parity_train_test(self, y_train, y_pred_train, y_test, y_pred_test, target_name="Target"):
        y_train = np.asarray(y_train).ravel()
        y_pred_train = np.asarray(y_pred_train).ravel()
        y_test = np.asarray(y_test).ravel()
        y_pred_test = np.asarray(y_pred_test).ravel()

        all_values = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
        finite_values = all_values[np.isfinite(all_values)]
        if finite_values.size == 0:
            raise ValueError("No finite values available for plotting")

        all_min = float(np.min(finite_values))
        all_max = float(np.max(finite_values))
        span = all_max - all_min
        buf = span * 0.08 if span > 0 else 1.0
        lim_min, lim_max = all_min - buf, all_max + buf

        fig = plt.figure(figsize=(7.2, 7.2))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[5.2, 0.55],
            height_ratios=[0.55, 5.2],
            wspace=0.01,
            hspace=0.01,
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        bins = min(28, max(12, int(np.sqrt(len(y_train) + len(y_test)))))

        ax_top.hist(y_train, bins=bins, color=TRAIN_COLOR, alpha=0.78)
        ax_top.hist(y_test, bins=bins, color=TEST_COLOR, alpha=0.62)
        ax_top.set_xlim(lim_min, lim_max)
        ax_top.grid(False)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", left=False, labelleft=False)
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        ax_top.set_facecolor("white")

        ax_right.hist(y_pred_train, bins=bins, orientation="horizontal", color=TRAIN_COLOR, alpha=0.78)
        ax_right.hist(y_pred_test, bins=bins, orientation="horizontal", color=TEST_COLOR, alpha=0.62)
        ax_right.set_ylim(lim_min, lim_max)
        ax_right.grid(False)
        ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        for spine in ax_right.spines.values():
            spine.set_visible(False)
        ax_right.set_facecolor("white")

        train_metrics = self._safe_metrics(y_train, y_pred_train)
        test_metrics = self._safe_metrics(y_test, y_pred_test)

        xs = np.linspace(lim_min, lim_max, 200)
        span = lim_max - lim_min
        ax_main.fill_between(xs, xs - span * 0.10, xs + span * 0.10,
                             color="#DDDDDD", alpha=0.3, zorder=0, label="±10% band")
        ax_main.plot(xs, xs, color=ACCENT_COLOR, linestyle="--", linewidth=1.8, zorder=1)
        ax_main.scatter(
            y_train, y_pred_train,
            c=TRAIN_COLOR, marker="o", s=26,
            alpha=0.62, edgecolors="white", linewidth=0.35, zorder=3,
            label="Train"
        )
        ax_main.scatter(
            y_test, y_pred_test,
            c=TEST_COLOR, marker="D", s=30,
            alpha=0.72, edgecolors="white", linewidth=0.40, zorder=4,
            label="Test"
        )

        label_name = target_name.split('_')[0] if target_name else "Target"
        unit = " /°C" if target_name and ("Tg" in target_name or "temp" in target_name.lower()) else ""
        style_axes(
            ax_main,
            title=None,
            xlabel=f"Experimental {label_name}{unit}",
            ylabel=f"Predicted {label_name}{unit}",
        )
        ax_main.set_xlim(lim_min, lim_max)
        ax_main.set_ylim(lim_min, lim_max)
        ax_main.set_aspect("equal")

        metric_box = (
            self._format_metric_block("Train", train_metrics)
            + "\n\n"
            + self._format_metric_block("Test", test_metrics)
        )
        ax_main.legend(loc="lower right", frameon=False, fontsize=11)
        ax_main.text(
            0.04, 0.90, metric_box,
            transform=ax_main.transAxes,
            ha="left", va="top", fontsize=10.0, color=TEXT_COLOR,
            linespacing=1.22,
            bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="#D8D8D8", alpha=0.94)
        )

        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.09, right=0.965, bottom=0.09, top=0.965, wspace=0.01, hspace=0.01)

        df_tr = pd.DataFrame({"True": y_train, "Pred": y_pred_train, "Set": "Train"})
        df_te = pd.DataFrame({"True": y_test, "Pred": y_pred_test, "Set": "Test"})
        export_df = pd.concat([df_tr, df_te], ignore_index=True)
        export_df["Residual"] = export_df["True"] - export_df["Pred"]
        metrics = {"train": train_metrics, "test": test_metrics}
        return fig, export_df, metrics

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, y_pred_lower=None, y_pred_upper=None):
        """预测值 vs 真实值 (基础版 - 用于仅有测试集的情况)"""
        fig, ax = plt.subplots(figsize=(7, 7))

        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        # 对角线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        buf = (max_val - min_val) * 0.05
        ax.plot([min_val - buf, max_val + buf], [min_val - buf, max_val + buf],
                color="#999999", linestyle="--", lw=1.8, zorder=1, label="y = x")

        # 散点
        ax.scatter(y_true, y_pred, alpha=0.7, s=55, c=TRAIN_COLOR,
                   edgecolors="white", linewidth=0.6, zorder=2, label="Data")

        if y_pred_lower is not None and y_pred_upper is not None:
            sorted_idx = np.argsort(y_true)
            y_sorted = y_true[sorted_idx]
            ax.fill_between(y_sorted, y_pred_lower[sorted_idx], y_pred_upper[sorted_idx],
                            color=TRAIN_COLOR, alpha=0.12, label="90% CI")

        style_axes(ax, title=f"{model_name}", xlabel="Experimental", ylabel="Predicted")
        ax.set_xlim(min_val - buf, max_val + buf)
        ax.set_ylim(min_val - buf, max_val + buf)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", frameon=False, fontsize=11)

        r2 = r2_score(y_true, y_pred)
        ax.text(0.95, 0.08, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, fontsize=13,
                ha="right", va="bottom", weight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.9))

        plt.tight_layout()

        export_df = pd.DataFrame({
            "True_Value": y_true, "Predicted_Value": y_pred, "Residual": y_true - y_pred
        })
        return fig, export_df

    def plot_residuals(self, y_true, y_pred, model_name):
        """残差分析图"""
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # 残差 vs 预测值
        axes[0].scatter(y_pred, residuals, alpha=0.65, s=50, c=TRAIN_COLOR,
                        edgecolors="white", linewidth=0.5, zorder=2)
        axes[0].axhline(y=0, color=TEST_COLOR, linestyle="--", lw=2, alpha=0.8, zorder=1)
        # ±1σ 参考带
        std_r = np.std(residuals)
        axes[0].axhspan(-std_r, std_r, color=TRAIN_COLOR, alpha=0.06, zorder=0)
        axes[0].text(0.02, 0.97, f"σ = {std_r:.3f}", transform=axes[0].transAxes,
                     fontsize=10, va="top", color="#666666")
        style_axes(axes[0], title="Residuals vs Predicted", xlabel="Predicted", ylabel="Residual")

        # 残差分布
        axes[1].hist(residuals, bins=30, edgecolor="white", alpha=0.85, color=TRAIN_COLOR, linewidth=0.8)
        axes[1].axvline(x=0, color=TEST_COLOR, linestyle="--", lw=2, alpha=0.8)
        # 标注均值和中位数
        mean_r = np.mean(residuals)
        axes[1].axvline(x=mean_r, color=ACCENT_COLOR, linestyle="-", lw=1.5, alpha=0.7, label=f"Mean={mean_r:.3f}")
        axes[1].legend(loc="upper right", frameon=False, fontsize=10)
        style_axes(axes[1], title="Residual Distribution", xlabel="Residual", ylabel="Count")

        plt.tight_layout(w_pad=3)

        export_df = pd.DataFrame({"Predicted_Value": y_pred, "Residual": residuals})
        return fig, export_df

    def plot_feature_importance(self, importances, feature_names, model_name, top_n=20):
        """特征重要性图"""
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        importance_df = pd.DataFrame({
            'Feature': feature_names, 'Importance': importances
        }).sort_values('Importance', ascending=False)

        top_n = min(top_n, len(importance_df))
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))

        values = top_features['Importance'].values[::-1]
        names = top_features['Feature'].values[::-1]
        # 渐变色：重要性越高颜色越深
        norm_vals = values / (values.max() + 1e-9)
        colors = [plt.cm.Blues(0.35 + 0.6 * v) for v in norm_vals]

        bars = ax.barh(range(top_n), values, color=colors, edgecolor="white", linewidth=0.8, height=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names, fontsize=10)

        # 在条形末端标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color="#555555")

        style_axes(ax, title=f'{model_name} - Feature Importance (Top {top_n})', xlabel='Importance', ylabel=None)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.grid(axis='y', visible=False)

        plt.tight_layout()
        return fig, importance_df

    def plot_parity_train_test(self, y_train, y_pred_train, y_test, y_pred_test, target_name="Target"):
        """
        绘制 实验值 vs 预测值 对比图（训练集 + 测试集）
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        y_train = np.array(y_train).ravel()
        y_pred_train = np.array(y_pred_train).ravel()
        y_test = np.array(y_test).ravel()
        y_pred_test = np.array(y_pred_test).ravel()

        # 对角线
        all_min = min(np.min(y_train), np.min(y_test), np.min(y_pred_train), np.min(y_pred_test))
        all_max = max(np.max(y_train), np.max(y_test), np.max(y_pred_train), np.max(y_pred_test))
        buf = (all_max - all_min) * 0.06
        lim_min, lim_max = all_min - buf, all_max + buf

        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="#999999", linestyle="--", linewidth=1.8, zorder=1)

        # ±10% 误差带
        span = lim_max - lim_min
        xs = np.linspace(lim_min, lim_max, 100)
        ax.fill_between(xs, xs - span * 0.10, xs + span * 0.10,
                        color="#DDDDDD", alpha=0.3, zorder=0, label="±10% band")

        # 训练集
        r2_tr = r2_score(y_train, y_pred_train)
        ax.scatter(y_train, y_pred_train,
                   c=TRAIN_COLOR, label=f'Train ($R^2$={r2_tr:.4f})',
                   marker='o', s=40, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2)

        # 测试集
        r2_te = r2_score(y_test, y_pred_test)
        ax.scatter(y_test, y_pred_test,
                   c=TEST_COLOR, label=f'Test ($R^2$={r2_te:.4f})',
                   marker='D', s=50, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=3)

        # 标签
        unit = " /°C" if "Tg" in target_name or "temp" in target_name.lower() else ""
        label_name = target_name.split('_')[0]
        style_axes(ax, title=None,
                   xlabel=f"Experimental {label_name}{unit}",
                   ylabel=f"Predicted {label_name}{unit}")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")
        ax.legend(loc='upper left', frameon=False, fontsize=11)

        plt.tight_layout()

        df_tr = pd.DataFrame({"True": y_train, "Pred": y_pred_train, "Set": "Train"})
        df_te = pd.DataFrame({"True": y_test, "Pred": y_pred_test, "Set": "Test"})
        return fig, pd.concat([df_tr, df_te], ignore_index=True)
