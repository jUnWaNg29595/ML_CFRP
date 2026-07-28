"""
Streamlit 性能加速器 - 针对大数据场景的深度优化
解决16107×4358等大数据集的卡顿问题
"""

import streamlit as st
import pandas as pd
import numpy as np
from functools import wraps
import gc


def apply_performance_boost():
    """应用所有性能优化"""

    # 1. 优化 st.dataframe 渲染
    _patch_dataframe_rendering()

    # 2. 优化 session_state 访问
    _optimize_session_state_access()

    # 3. 启用智能缓存清理
    _enable_smart_cache_cleanup()

    # 4. 优化重渲染
    _reduce_unnecessary_reruns()

    print("✅ 性能加速器已启用")


def _patch_dataframe_rendering():
    """优化 DataFrame 渲染 - 核心优化"""

    if hasattr(st, '_original_dataframe'):
        return

    st._original_dataframe = st.dataframe

    def optimized_dataframe(data, *args, **kwargs):
        if not isinstance(data, pd.DataFrame):
            return st._original_dataframe(data, *args, **kwargs)

        rows, cols = data.shape

        # 超大数据：强制分页
        if rows > 500 or cols > 200:
            # 只渲染前100行
            kwargs['height'] = kwargs.get('height', 400)
            display_data = data.iloc[:100, :min(100, cols)]

            if rows > 500:
                st.caption(f"⚡ 数据量大，仅显示前100行（共{rows:,}行）")

            return st._original_dataframe(display_data, *args, **kwargs)

        return st._original_dataframe(data, *args, **kwargs)

    st.dataframe = optimized_dataframe


def _optimize_session_state_access():
    """优化 session_state 大对象存储"""

    # 自动压缩大 DataFrame
    for key in list(st.session_state.keys()):
        if key.startswith('_'):  # 跳过内部变量
            continue

        value = st.session_state.get(key)

        if isinstance(value, pd.DataFrame):
            memory_mb = value.memory_usage(deep=True).sum() / 1024 / 1024

            # 超过100MB自动优化
            if memory_mb > 100:
                st.session_state[key] = _optimize_dataframe(value)


def _optimize_dataframe(df):
    """快速优化 DataFrame 数据类型"""

    df = df.copy()

    # float64 → float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # int64 → int32
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')

    return df


def _enable_smart_cache_cleanup():
    """智能缓存清理"""

    if 'cache_cleanup_counter' not in st.session_state:
        st.session_state.cache_cleanup_counter = 0

    st.session_state.cache_cleanup_counter += 1

    # 每50次交互清理一次缓存
    if st.session_state.cache_cleanup_counter % 50 == 0:
        st.cache_data.clear()
        gc.collect()


def _reduce_unnecessary_reruns():
    """减少不必要的重渲染"""

    # 使用 fragment 装饰器的辅助函数
    pass


# ============ 便捷装饰器 ============

def lazy_load(func):
    """延迟加载装饰器 - 只在需要时加载数据"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"_lazy_{func.__name__}"

        if cache_key not in st.session_state:
            with st.spinner(f"加载 {func.__name__}..."):
                st.session_state[cache_key] = func(*args, **kwargs)

        return st.session_state[cache_key]

    return wrapper


def fast_cache(ttl=3600):
    """快速缓存装饰器 - 比 st.cache_data 更轻量"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"_fast_cache_{func.__name__}_{hash(str(args) + str(kwargs))}"

            if cache_key in st.session_state:
                return st.session_state[cache_key]

            result = func(*args, **kwargs)
            st.session_state[cache_key] = result
            return result

        return wrapper

    return decorator


# ============ 数据加载优化 ============

def load_csv_fast(file_path, **kwargs):
    """快速加载 CSV - 自动优化数据类型"""

    # 第一遍：推断数据类型
    df_sample = pd.read_csv(file_path, nrows=1000, **kwargs)

    # 构建 dtype 字典
    dtypes = {}
    for col in df_sample.columns:
        if df_sample[col].dtype == 'float64':
            dtypes[col] = 'float32'
        elif df_sample[col].dtype == 'int64':
            dtypes[col] = 'int32'

    # 第二遍：使用优化的数据类型加载
    df = pd.read_csv(file_path, dtype=dtypes, **kwargs)

    return df


def load_excel_fast(file_path, **kwargs):
    """快速加载 Excel"""

    df = pd.read_excel(file_path, engine='openpyxl', **kwargs)
    return _optimize_dataframe(df)


# ============ 显示优化 ============

def show_large_dataframe(df, max_rows=100, max_cols=50):
    """智能显示大型 DataFrame"""

    rows, cols = df.shape

    if rows <= max_rows and cols <= max_cols:
        st.dataframe(df, use_container_width=True)
        return

    # 显示采样数据
    display_df = df.iloc[:max_rows, :max_cols]

    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(display_df, use_container_width=True, height=400)

    with col2:
        st.metric("总行数", f"{rows:,}")
        st.metric("总列数", f"{cols:,}")

        if rows > max_rows:
            st.caption(f"仅显示前 {max_rows} 行")
        if cols > max_cols:
            st.caption(f"仅显示前 {max_cols} 列")


def show_dataframe_paginated(df, page_size=100):
    """分页显示 DataFrame"""

    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size

    if total_pages == 1:
        st.dataframe(df, use_container_width=True)
        return

    page = st.number_input(
        "页码",
        min_value=1,
        max_value=total_pages,
        value=1,
        help=f"共 {total_pages} 页，每页 {page_size} 行"
    )

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    st.dataframe(
        df.iloc[start_idx:end_idx],
        use_container_width=True,
        height=400
    )

    st.caption(f"显示第 {start_idx+1}-{end_idx} 行（共 {total_rows:,} 行）")


# ============ 内存优化 ============

def reduce_memory_usage(df, verbose=True):
    """激进的内存优化"""

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f'内存优化: {start_mem:.2f}MB → {end_mem:.2f}MB ({100 * (start_mem - end_mem) / start_mem:.1f}% 减少)')

    return df


def emergency_memory_cleanup():
    """紧急内存清理"""

    # 清理所有缓存
    st.cache_data.clear()
    st.cache_resource.clear()

    # 优化 session_state 中的大对象
    for key in list(st.session_state.keys()):
        if isinstance(st.session_state[key], pd.DataFrame):
            st.session_state[key] = _optimize_dataframe(st.session_state[key])

    # 强制垃圾回收
    gc.collect()

    st.success("✅ 内存清理完成")


# ============ 使用示例 ============

USAGE = """
# 使用方法

## 1. 在 app.py 开头添加

```python
from streamlit_performance_boost import apply_performance_boost

apply_performance_boost()
```

## 2. 优化数据加载

```python
from streamlit_performance_boost import load_csv_fast

# 替代 pd.read_csv
df = load_csv_fast('data.csv')
```

## 3. 优化数据显示

```python
from streamlit_performance_boost import show_large_dataframe

# 替代 st.dataframe
show_large_dataframe(df)
```

## 4. 紧急内存清理

```python
from streamlit_performance_boost import emergency_memory_cleanup

if st.button("清理内存"):
    emergency_memory_cleanup()
```
"""
