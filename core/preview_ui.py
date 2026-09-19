# -*- coding: utf-8 -*-
"""宽表预览 UI 助手。

背景（2026-09-18 实测）：
    分子特征提取完成后，工作区常有上千列指纹特征。直接 `st.dataframe(wide_df.head(20))`
    会带来两重代价，且**每次交互 rerun 都要重付一次**（点击任意选择框/复选框都卡）：

    1. Arrow 序列化：2000 列 × 20 行约 91ms；1248 列约 55ms。
    2. 发给浏览器的 delta 负载：2000 列约 801KB；6474×3000 时达 1.2MB。

    修复方式：默认只渲染前 N 列，其余列通过**按需展开**的列选择器查看
    （选择器本身也延迟构建，否则上千个列名同样会拖慢页面）。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

#: 默认预览列数上限。经验取值：40 列约 25KB 负载、序列化 <2ms，且足够看清数据结构。
DEFAULT_PREVIEW_COLUMNS = 40
#: 默认预览行数上限。
DEFAULT_PREVIEW_ROWS = 20


def render_capped_preview(
    df: pd.DataFrame,
    max_columns: int = DEFAULT_PREVIEW_COLUMNS,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    caption_prefix: str = "预览",
    key: str | None = None,
    height: int = 300,
) -> None:
    """渲染宽表预览：默认只展示前 max_columns 列，其余按需选择。

    Args:
        df: 待预览的 DataFrame（不会被修改）。
        max_columns: 默认展示的列数上限。
        max_rows: 展示的行数上限。
        caption_prefix: 说明文案前缀。
        key: 控件 key 前缀（同一页面多次调用必须不同，否则 Streamlit 报重复 key）。
        height: 表格高度（像素）。

    列数不超过 max_columns 时行为与直接 st.dataframe 一致，不产生额外控件。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return

    total_cols = df.shape[1]
    shown_cols = min(int(max_columns), total_cols)
    preview = df.head(int(max_rows))

    if total_cols <= shown_cols:
        st.dataframe(preview, width="stretch", height=height)
        return

    st.caption(
        f"{caption_prefix}：{df.shape[0]} 行 × {total_cols} 列"
        f"（默认只显示前 {shown_cols} 列，避免宽表拖慢页面）"
    )

    custom_key = f"{key}_custom_open" if key else None
    if st.toggle("🔎 自定义要预览的列", key=custom_key):
        selected = st.multiselect(
            "选择列",
            options=list(df.columns),
            default=list(df.columns[:shown_cols]),
            key=f"{key}_cols" if key else None,
            help="支持输入关键字搜索（如 fp_ 或 maccs）。",
        )
        if selected:
            st.dataframe(
                df.loc[:, selected].head(int(max_rows)),
                width="stretch",
                height=height,
            )
        return

    st.dataframe(preview.iloc[:, :shown_cols], width="stretch", height=height)
