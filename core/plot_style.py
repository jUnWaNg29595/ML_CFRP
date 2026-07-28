# -*- coding: utf-8 -*-
"""统一图表风格工具（美化版）

目的：
- 让全站 matplotlib 图表风格一致、美观
- 在 Streamlit 中稳定显示（避免中文乱码、布局抖动）
"""

from __future__ import annotations
from typing import Optional

_APPLIED = False

# ── 配色方案（科研风格）────────────────────────────────────────
TRAIN_COLOR = "#2563EB"    # 主蓝色
TEST_COLOR  = "#DC2626"    # 红色
OOF_COLOR   = "#059669"    # 绿色
ACCENT_COLOR = "#0891B2"   # 青绿
GRID_COLOR  = "#E2E8F0"    # 浅灰网格
BG_COLOR    = "#FFFFFF"    # 白色背景
TEXT_COLOR  = "#1E293B"    # 深灰文字

# 多色调色板（科研风格）
PALETTE = [
    "#2563EB", "#0891B2", "#059669", "#D97706",
    "#DC2626", "#7C3AED", "#DB2777", "#0284C7",
    "#65A30D", "#64748B",
]


def apply_global_style(dpi: int = 130):
    """应用全局 matplotlib 风格（幂等）。"""
    global _APPLIED
    if _APPLIED:
        return

    import matplotlib
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
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

    # 基础尺寸与背景
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = BG_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR

    # 线条与字体大小
    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["legend.frameon"] = False

    # 网格与边框
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.4
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = TEXT_COLOR

    # 散点图默认样式
    plt.rcParams["scatter.edgecolors"] = "white"
    plt.rcParams["scatter.marker"] = "o"

    _APPLIED = True


def style_axes(ax, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """对单个 Axes 做轻量统一。"""
    if title:
        ax.set_title(title, weight="bold", fontsize=14, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    # 统一 tick 方向
    ax.tick_params(direction="in", length=5, width=1.2)

    # 统一边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
