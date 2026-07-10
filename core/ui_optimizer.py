"""
UI性能优化模块
解决大数据量下的界面卡顿问题
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
from functools import lru_cache


class DataFrameOptimizer:
    """DataFrame显示优化器 - 虚拟化和分页"""

    DEFAULT_PAGE_SIZE = 100
    MAX_DISPLAY_ROWS = 1000

    @staticmethod
    def render_paginated_dataframe(
        df: pd.DataFrame,
        key: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        height: Optional[int] = None,
        use_container_width: bool = True,
        show_stats: bool = True,
        column_config: Optional[Dict] = None
    ):
        """
        渲染分页的DataFrame，避免一次性加载大量数据

        Args:
            df: 要显示的DataFrame
            key: Streamlit组件的唯一key
            page_size: 每页显示的行数
            height: 表格高度（像素）
            use_container_width: 是否使用容器宽度
            show_stats: 是否显示统计信息
            column_config: 列配置
        """
        if df is None or df.empty:
            st.info("暂无数据")
            return

        total_rows = len(df)

        # 如果数据量小，直接显示
        if total_rows <= page_size:
            st.dataframe(
                df,
                use_container_width=use_container_width,
                height=height,
                column_config=column_config
            )
            if show_stats:
                st.caption(f"共 {total_rows} 行")
            return

        # 分页控制
        total_pages = (total_rows + page_size - 1) // page_size

        col1, col2, col3 = st.columns([2, 3, 2])

        with col1:
            page_num = st.number_input(
                "页码",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.get(f"{key}_page", 1),
                key=f"{key}_page_input",
                help=f"共 {total_pages} 页"
            )

        with col2:
            if show_stats:
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                st.info(f"显示第 {start_idx + 1}-{end_idx} 行，共 {total_rows} 行")

        with col3:
            custom_page_size = st.selectbox(
                "每页行数",
                options=[50, 100, 200, 500],
                index=[50, 100, 200, 500].index(page_size) if page_size in [50, 100, 200, 500] else 1,
                key=f"{key}_page_size"
            )
            if custom_page_size != page_size:
                page_size = custom_page_size
                st.session_state[f"{key}_page"] = 1
                st.rerun()

        # 计算当前页数据
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_df = df.iloc[start_idx:end_idx]

        # 显示数据
        st.dataframe(
            page_df,
            use_container_width=use_container_width,
            height=height,
            column_config=column_config
        )

        # 分页按钮
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ 上一页", disabled=(page_num <= 1), key=f"{key}_prev"):
                st.session_state[f"{key}_page"] = page_num - 1
                st.rerun()

        with col_next:
            if st.button("下一页 ➡️", disabled=(page_num >= total_pages), key=f"{key}_next"):
                st.session_state[f"{key}_page"] = page_num + 1
                st.rerun()

    @staticmethod
    def render_smart_dataframe(
        df: pd.DataFrame,
        key: str,
        max_rows: int = MAX_DISPLAY_ROWS,
        **kwargs
    ):
        """
        智能渲染DataFrame：小数据直接显示，大数据自动分页

        Args:
            df: 要显示的DataFrame
            key: 唯一标识
            max_rows: 超过此行数则启用分页
            **kwargs: 传递给dataframe的其他参数
        """
        if df is None or df.empty:
            st.info("暂无数据")
            return

        if len(df) <= max_rows:
            st.dataframe(df, **kwargs)
        else:
            DataFrameOptimizer.render_paginated_dataframe(
                df, key, **kwargs
            )


class LazyLoader:
    """懒加载管理器 - 按需加载数据"""

    @staticmethod
    def create_expander_with_lazy_content(
        label: str,
        key: str,
        content_func,
        expanded: bool = False
    ):
        """
        创建带懒加载内容的expander

        Args:
            label: expander标签
            key: 唯一标识
            content_func: 内容生成函数（无参数）
            expanded: 是否默认展开
        """
        with st.expander(label, expanded=expanded):
            # 只有在展开时才加载内容
            if st.session_state.get(f"{key}_loaded", False) or expanded:
                content_func()
                st.session_state[f"{key}_loaded"] = True
            else:
                if st.button("点击加载内容", key=f"{key}_load_btn"):
                    st.session_state[f"{key}_loaded"] = True
                    st.rerun()


class CacheOptimizer:
    """缓存优化器 - 改进缓存策略"""

    @staticmethod
    @lru_cache(maxsize=128)
    def get_dataframe_hash(df_id: int, shape: tuple) -> str:
        """生成DataFrame的轻量级哈希"""
        return f"{df_id}_{shape[0]}_{shape[1]}"

    @staticmethod
    def cache_dataframe_operation(func):
        """装饰器：缓存DataFrame操作结果"""
        cache_dict = {}

        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # 使用DataFrame的id和shape作为缓存键
            cache_key = (id(df), df.shape, str(args), str(kwargs))

            if cache_key in cache_dict:
                return cache_dict[cache_key]

            result = func(df, *args, **kwargs)
            cache_dict[cache_key] = result

            # 限制缓存大小
            if len(cache_dict) > 50:
                cache_dict.pop(next(iter(cache_dict)))

            return result

        return wrapper


class BatchProcessor:
    """批处理优化器 - 分批处理大数据"""

    @staticmethod
    def process_in_batches(
        data: Union[pd.DataFrame, List],
        process_func,
        batch_size: int = 1000,
        show_progress: bool = True,
        progress_text: str = "处理中..."
    ):
        """
        分批处理数据，显示进度条

        Args:
            data: 要处理的数据（DataFrame或列表）
            process_func: 处理函数，接收批次数据
            batch_size: 批次大小
            show_progress: 是否显示进度条
            progress_text: 进度条文本

        Returns:
            处理结果列表
        """
        if isinstance(data, pd.DataFrame):
            total = len(data)
            batches = [data.iloc[i:i+batch_size] for i in range(0, total, batch_size)]
        else:
            total = len(data)
            batches = [data[i:i+batch_size] for i in range(0, total, batch_size)]

        results = []

        if show_progress:
            progress_bar = st.progress(0, text=progress_text)

            for i, batch in enumerate(batches):
                result = process_func(batch)
                results.append(result)
                progress_bar.progress((i + 1) / len(batches), text=f"{progress_text} ({i+1}/{len(batches)})")

            progress_bar.empty()
        else:
            for batch in batches:
                result = process_func(batch)
                results.append(result)

        return results


class MemoryOptimizer:
    """内存优化器 - 减少内存占用"""

    @staticmethod
    def optimize_dataframe_dtypes(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        优化DataFrame的数据类型，减少内存占用

        Args:
            df: 要优化的DataFrame
            inplace: 是否原地修改

        Returns:
            优化后的DataFrame
        """
        if not inplace:
            df = df.copy()

        # 优化整数类型
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')

        # 优化浮点类型
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # 优化对象类型（转为category）
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # 如果唯一值比例小于50%，转为category
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        return df

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """获取DataFrame的内存使用情况"""
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        return {
            "总内存": f"{memory_mb:.2f} MB",
            "行数": len(df),
            "列数": len(df.columns),
            "平均每行": f"{memory_mb / len(df) * 1024:.2f} KB"
        }


# 便捷函数
def render_optimized_dataframe(df: pd.DataFrame, key: str, **kwargs):
    """便捷函数：渲染优化的DataFrame"""
    return DataFrameOptimizer.render_smart_dataframe(df, key, **kwargs)


def show_memory_stats(df: pd.DataFrame):
    """显示内存统计信息"""
    stats = MemoryOptimizer.get_memory_usage(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总内存", stats["总内存"])
    col2.metric("行数", stats["行数"])
    col3.metric("列数", stats["列数"])
    col4.metric("平均每行", stats["平均每行"])
