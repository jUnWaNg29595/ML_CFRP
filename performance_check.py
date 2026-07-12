"""
性能诊断工具 - 快速检查 Streamlit 应用的性能问题
"""

import streamlit as st
import pandas as pd
import sys
import gc


def check_performance():
    """性能诊断"""

    st.title("⚡ 性能诊断工具")

    # 1. Session State 检查
    st.header("1. Session State 内存占用")

    total_memory = 0
    large_objects = []

    for key, value in st.session_state.items():
        if isinstance(value, pd.DataFrame):
            memory_mb = value.memory_usage(deep=True).sum() / 1024 / 1024
            total_memory += memory_mb

            if memory_mb > 50:
                large_objects.append({
                    "变量名": key,
                    "类型": "DataFrame",
                    "形状": f"{value.shape[0]} × {value.shape[1]}",
                    "内存(MB)": f"{memory_mb:.1f}"
                })

    st.metric("总内存占用", f"{total_memory:.1f} MB")

    if large_objects:
        st.warning(f"发现 {len(large_objects)} 个大对象")
        st.dataframe(pd.DataFrame(large_objects), use_container_width=True)
    else:
        st.success("✅ 内存占用正常")

    # 2. 数据类型检查
    st.header("2. 数据类型优化建议")

    if 'data' in st.session_state:
        df = st.session_state['data']

        float64_cols = len(df.select_dtypes(include=['float64']).columns)
        int64_cols = len(df.select_dtypes(include=['int64']).columns)

        if float64_cols > 0 or int64_cols > 0:
            st.warning(f"可优化: {float64_cols} 个 float64 列, {int64_cols} 个 int64 列")

            if st.button("一键优化数据类型"):
                optimized_df = df.copy()

                for col in df.select_dtypes(include=['float64']).columns:
                    optimized_df[col] = optimized_df[col].astype('float32')

                for col in df.select_dtypes(include=['int64']).columns:
                    optimized_df[col] = optimized_df[col].astype('int32')

                st.session_state['data'] = optimized_df
                st.success("✅ 优化完成！请刷新页面")
                st.rerun()
        else:
            st.success("✅ 数据类型已优化")

    # 3. 缓存检查
    st.header("3. 缓存状态")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("清理数据缓存"):
            st.cache_data.clear()
            st.success("✅ 数据缓存已清理")

    with col2:
        if st.button("清理资源缓存"):
            st.cache_resource.clear()
            st.success("✅ 资源缓存已清理")

    # 4. 内存清理
    st.header("4. 紧急内存清理")

    if st.button("🚀 执行内存清理", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        st.success("✅ 内存清理完成")


if __name__ == "__main__":
    check_performance()
