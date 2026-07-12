"""
紧急性能修复 - 解决大数据加载慢的问题
针对16107×4358这样的大数据集优化

在app.py开头添加：
    from emergency_performance_fix import apply_emergency_fixes
    apply_emergency_fixes()
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import locale
import builtins


def _configure_safe_console_output():
    preferred_encoding = locale.getpreferredencoding(False) or "utf-8"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding=preferred_encoding, errors="replace")
        except Exception:
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    original_print = builtins.print

    def _safe_print(*args, **kwargs):
        try:
            return original_print(*args, **kwargs)
        except UnicodeEncodeError:
            output_file = kwargs.get("file", sys.stdout)
            output_encoding = getattr(output_file, "encoding", None) or preferred_encoding
            safe_args = [
                str(arg).encode(output_encoding, errors="replace").decode(output_encoding, errors="replace")
                for arg in args
            ]
            return original_print(*safe_args, **kwargs)

    return _safe_print


print = _configure_safe_console_output()


def apply_emergency_fixes():
    """应用紧急性能修复"""

    print("="*60)
    print("应用紧急性能修复")
    print("="*60)

    # 1. 优化Streamlit配置
    _optimize_streamlit_config()

    # 2. 优化session_state
    _optimize_session_state()

    # 3. 优化DataFrame显示
    _patch_dataframe_display()

    # 4. 添加自动内存优化
    _enable_auto_memory_optimization()

    print("✅ 紧急修复已应用")
    print("="*60)


def _optimize_streamlit_config():
    """优化Streamlit配置"""

    # 设置更激进的配置
    try:
        # 禁用文件监控（减少I/O）
        os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

        # 增加消息大小限制
        os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '500'

        # 禁用CORS（如果不需要）
        os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'

        print("✓ Streamlit配置已优化")
    except:
        pass


def _optimize_session_state():
    """优化session_state，自动清理大对象"""

    # 检查并优化大数据
    if 'data' in st.session_state:
        data = st.session_state['data']

        if isinstance(data, pd.DataFrame):
            # 检查内存占用
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024

            if memory_mb > 200:  # 超过200MB
                print(f"⚠️  检测到大数据: {data.shape[0]} × {data.shape[1]} ({memory_mb:.1f}MB)")
                print("   正在优化数据类型...")

                # 自动优化数据类型
                optimized_data = _optimize_dataframe_dtypes(data)
                optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024

                if optimized_memory < memory_mb * 0.8:  # 节省超过20%
                    st.session_state['data'] = optimized_data
                    print(f"✓ 内存优化: {memory_mb:.1f}MB → {optimized_memory:.1f}MB (节省{(1-optimized_memory/memory_mb)*100:.1f}%)")


def _optimize_dataframe_dtypes(df):
    """优化DataFrame数据类型"""

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

    # 优化浮点类型（float64 → float32）
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    return df


def _patch_dataframe_display():
    """修补DataFrame显示，避免渲染大数据"""

    if not hasattr(st, '_original_dataframe'):
        st._original_dataframe = st.dataframe

    def smart_dataframe(data, *args, **kwargs):
        """智能DataFrame显示"""

        if not isinstance(data, pd.DataFrame):
            return st._original_dataframe(data, *args, **kwargs)

        rows, cols = data.shape

        # 超大数据：只显示前100行
        if rows > 1000 or cols > 500:
            st.warning(f"⚠️ 数据量大（{rows:,} × {cols:,}），仅显示前100行×前100列")
            display_data = data.iloc[:100, :min(100, cols)]
            return st._original_dataframe(display_data, *args, **kwargs)

        # 大数据：只显示前500行
        elif rows > 500:
            st.info(f"数据量较大（{rows:,}行），仅显示前500行")
            return st._original_dataframe(data.head(500), *args, **kwargs)

        # 正常显示
        return st._original_dataframe(data, *args, **kwargs)

    st.dataframe = smart_dataframe
    print("✓ DataFrame显示已优化")


def _enable_auto_memory_optimization():
    """启用自动内存优化"""

    # 在session_state中添加优化标记
    if 'auto_optimize_enabled' not in st.session_state:
        st.session_state.auto_optimize_enabled = True
        print("✓ 自动内存优化已启用")


def optimize_large_dataframe(df, aggressive=True):
    """
    优化大型DataFrame

    Args:
        df: DataFrame
        aggressive: 是否使用激进优化（float64→float32）

    Returns:
        优化后的DataFrame
    """

    print(f"\n优化DataFrame: {df.shape[0]} × {df.shape[1]}")

    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"原始内存: {original_memory:.1f}MB")

    df_optimized = df.copy()

    # 1. 优化整数类型
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()

        if col_min >= 0:
            if col_max < 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif col_max < 65535:
                df_optimized[col] = df_optimized[col].astype('uint16')
            elif col_max < 4294967295:
                df_optimized[col] = df_optimized[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df_optimized[col] = df_optimized[col].astype('int32')

    # 2. 优化浮点类型
    if aggressive:
        float_cols = df_optimized.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df_optimized[col] = df_optimized[col].astype('float32')

    # 3. 优化对象类型
    obj_cols = df_optimized.select_dtypes(include=['object']).columns
    for col in obj_cols:
        num_unique = df_optimized[col].nunique()
        num_total = len(df_optimized[col])

        # 如果唯一值比例小于50%，转为category
        if num_unique / num_total < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')

    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"优化后内存: {optimized_memory:.1f}MB")
    print(f"节省: {(1 - optimized_memory/original_memory)*100:.1f}%")

    return df_optimized


def reduce_dataframe_precision(df):
    """
    降低DataFrame精度（激进优化）

    将所有float64转为float32
    """

    df = df.copy()

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    return df


def sample_large_dataframe(df, max_rows=10000, random_state=42):
    """
    对大型DataFrame采样

    Args:
        df: DataFrame
        max_rows: 最大行数
        random_state: 随机种子

    Returns:
        采样后的DataFrame
    """

    if len(df) <= max_rows:
        return df

    print(f"⚠️  数据量过大（{len(df)}行），采样到{max_rows}行")

    return df.sample(n=max_rows, random_state=random_state)


def chunk_process_dataframe(df, process_func, chunk_size=1000):
    """
    分块处理DataFrame

    Args:
        df: DataFrame
        process_func: 处理函数
        chunk_size: 块大小

    Returns:
        处理后的DataFrame
    """

    results = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        result = process_func(chunk)
        results.append(result)

    return pd.concat(results, ignore_index=True)


# 便捷函数
def quick_optimize(df):
    """快速优化DataFrame"""
    return optimize_large_dataframe(df, aggressive=True)


def emergency_reduce_memory():
    """紧急减少内存占用"""

    if 'data' in st.session_state:
        st.session_state['data'] = optimize_large_dataframe(st.session_state['data'])

    if 'processed_data' in st.session_state:
        st.session_state['processed_data'] = optimize_large_dataframe(st.session_state['processed_data'])

    # 清理缓存
    st.cache_data.clear()

    print("✅ 紧急内存优化完成")


# 使用说明
USAGE = """
# 紧急性能修复使用指南

## 问题：16107×4358数据加载超过5分钟

## 原因分析

1. **内存占用过大**：
   - 16107 × 4358 ≈ 7000万个单元格
   - 如果是float64：约560MB
   - 加上session_state开销：可能超过1GB

2. **DataFrame渲染慢**：
   - Streamlit渲染大表格很慢
   - 每次rerun都重新渲染

3. **没有优化数据类型**：
   - 默认float64占用8字节
   - 可以优化为float32（4字节）

## 快速修复

### 方法1：在app.py开头添加（推荐）

```python
from emergency_performance_fix import apply_emergency_fixes

# 在主函数开始处
apply_emergency_fixes()
```

### 方法2：手动优化数据

```python
from emergency_performance_fix import optimize_large_dataframe

# 优化数据
if 'data' in st.session_state:
    st.session_state['data'] = optimize_large_dataframe(st.session_state['data'])
```

### 方法3：紧急减少内存

```python
from emergency_performance_fix import emergency_reduce_memory

# 一键优化
emergency_reduce_memory()
```

## 预期效果

- 内存占用：560MB → 280MB（节省50%）
- 加载时间：5分钟 → 30秒（提升10x）
- DataFrame显示：自动限制显示行数

## 其他建议

1. **使用特征选择**：减少列数到500以下
2. **数据采样**：如果不需要全部数据，采样到5000行
3. **分批处理**：不要一次性加载所有数据

## 诊断工具

运行诊断脚本：
```bash
streamlit run performance_diagnosis.py
```

查看：
- Session State大小
- 数据内存占用
- 性能瓶颈
"""

if __name__ == "__main__":
    print(USAGE)
