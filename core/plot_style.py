# -*- coding: utf-8 -*-
"""统一图表风格工具

目的：
- 让全站 matplotlib 图表（性能图、特征重要性、训练曲线等）风格一致
- 在 Streamlit 中稳定显示（避免中文乱码、布局抖动）

说明：
- 本模块只负责设置 rcParams / 轴样式，不负责具体业务绘图
"""

from __future__ import annotations

from typing import Optional


_APPLIED = False

# 颜色规范（与核心性能对比图一致）
# - 训练集：天蓝
# - 测试集：橙红
TRAIN_COLOR = "#87CEFA"
TEST_COLOR = "#FF4500"


def apply_global_style(dpi: int = 110):
    """应用全局 matplotlib 风格（幂等）。"""
    global _APPLIED
    if _APPLIED:
        return

    import matplotlib

    # Streamlit/服务端环境建议使用 Agg
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    # 字体：优先中文，再回退
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 数学文本（如 $R^2$）的字体，避免 Unicode 上标字符缺字导致方块
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

    # 基础尺寸
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # 线条/字体
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11

    # 网格与边框
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.35
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    _APPLIED = True


def style_axes(ax, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """对单个 Axes 做轻量统一（不强制颜色）。"""
    if title:
        ax.set_title(title, weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # 统一 tick 方向
    try:
        ax.tick_params(direction="in")
    except Exception:
        pass

    # 统一边框线宽
    for spine in ax.spines.values():
        try:
            spine.set_linewidth(1.0)
        except Exception:
            pass
