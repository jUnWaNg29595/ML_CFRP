"""
异步加载和多线程优化模块
解决Streamlit刷新慢和界面切换卡顿问题
"""

import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
import hashlib
import pickle


class AsyncDataLoader:
    """异步数据加载器 - 使用多线程预加载数据"""

    _executor = ThreadPoolExecutor(max_workers=4)
    _cache = {}
    _cache_lock = Lock()
    _loading_status = {}

    @classmethod
    def preload_data(cls, key: str, load_func: Callable, *args, **kwargs):
        """
        预加载数据（后台线程）

        Args:
            key: 缓存键
            load_func: 加载函数
            *args, **kwargs: 传递给load_func的参数
        """
        if key in cls._cache:
            return  # 已缓存

        if key in cls._loading_status:
            return  # 正在加载

        cls._loading_status[key] = "loading"

        def _load():
            try:
                result = load_func(*args, **kwargs)
                with cls._cache_lock:
                    cls._cache[key] = result
                    cls._loading_status[key] = "completed"
            except Exception as e:
                cls._loading_status[key] = f"error: {e}"

        cls._executor.submit(_load)

    @classmethod
    def get_data(cls, key: str, load_func: Callable, *args, timeout: float = 30, **kwargs):
        """
        获取数据（如果未加载则同步加载）

        Args:
            key: 缓存键
            load_func: 加载函数
            timeout: 超时时间（秒）

        Returns:
            加载的数据
        """
        # 检查缓存
        if key in cls._cache:
            return cls._cache[key]

        # 检查是否正在加载
        if key in cls._loading_status:
            status = cls._loading_status[key]
            if status == "loading":
                # 等待加载完成
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if key in cls._cache:
                        return cls._cache[key]
                    time.sleep(0.1)
                raise TimeoutError(f"数据加载超时: {key}")
            elif status == "completed":
                return cls._cache[key]
            elif status.startswith("error"):
                raise RuntimeError(status)

        # 同步加载
        result = load_func(*args, **kwargs)
        with cls._cache_lock:
            cls._cache[key] = result
        return result

    @classmethod
    def clear_cache(cls, key: Optional[str] = None):
        """清除缓存"""
        with cls._cache_lock:
            if key:
                cls._cache.pop(key, None)
                cls._loading_status.pop(key, None)
            else:
                cls._cache.clear()
                cls._loading_status.clear()


class SmartCache:
    """智能缓存 - 基于内容哈希的持久化缓存"""

    @staticmethod
    def _get_hash(obj: Any) -> str:
        """计算对象哈希"""
        try:
            if isinstance(obj, pd.DataFrame):
                # DataFrame使用shape和前几行数据
                content = f"{obj.shape}_{obj.head(5).to_json()}"
            elif isinstance(obj, (list, tuple)):
                content = str(obj[:100])  # 只取前100个元素
            else:
                content = str(obj)

            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(str(id(obj)).encode()).hexdigest()

    @staticmethod
    def cache_function(ttl: int = 3600, key_prefix: str = ""):
        """
        函数缓存装饰器（比st.cache_data更快）

        Args:
            ttl: 缓存时间（秒）
            key_prefix: 缓存键前缀
        """
        def decorator(func: Callable):
            cache_dict = {}
            cache_time = {}

            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{key_prefix}_{func.__name__}_{SmartCache._get_hash((args, kwargs))}"

                # 检查缓存
                current_time = time.time()
                if cache_key in cache_dict:
                    if current_time - cache_time[cache_key] < ttl:
                        return cache_dict[cache_key]

                # 执行函数
                result = func(*args, **kwargs)

                # 保存缓存
                cache_dict[cache_key] = result
                cache_time[cache_key] = current_time

                # 清理过期缓存
                expired_keys = [k for k, t in cache_time.items() if current_time - t > ttl]
                for k in expired_keys:
                    cache_dict.pop(k, None)
                    cache_time.pop(k, None)

                return result

            wrapper.clear_cache = lambda: (cache_dict.clear(), cache_time.clear())
            return wrapper

        return decorator


class ParallelDataProcessor:
    """并行数据处理器 - 多线程处理数据"""

    @staticmethod
    def parallel_apply(
        df: pd.DataFrame,
        func: Callable,
        n_workers: int = 4,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        并行应用函数到DataFrame

        Args:
            df: DataFrame
            func: 处理函数
            n_workers: 线程数
            show_progress: 是否显示进度

        Returns:
            处理后的DataFrame
        """
        if len(df) < 100:
            # 小数据量直接处理
            return df.apply(func, axis=1)

        # 分割数据
        chunk_size = max(1, len(df) // n_workers)
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

        results = []

        if show_progress:
            progress_bar = st.progress(0, text="并行处理中...")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(lambda chunk: chunk.apply(func, axis=1), chunk): i
                      for i, chunk in enumerate(chunks)}

            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if show_progress:
                    progress_bar.progress((i + 1) / len(futures),
                                         text=f"并行处理中... ({i+1}/{len(futures)})")

        if show_progress:
            progress_bar.empty()

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def parallel_read_csv(
        file_paths: List[str],
        n_workers: int = 4,
        **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        并行读取多个CSV文件

        Args:
            file_paths: 文件路径列表
            n_workers: 线程数
            **read_csv_kwargs: 传递给pd.read_csv的参数

        Returns:
            合并后的DataFrame
        """
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(pd.read_csv, fp, **read_csv_kwargs)
                      for fp in file_paths]
            dfs = [f.result() for f in as_completed(futures)]

        return pd.concat(dfs, ignore_index=True)


class LazyComponent:
    """懒加载组件 - 只在需要时渲染"""

    @staticmethod
    def render_on_demand(
        component_id: str,
        render_func: Callable,
        placeholder_text: str = "点击加载内容",
        auto_load: bool = False
    ):
        """
        按需渲染组件

        Args:
            component_id: 组件唯一ID
            render_func: 渲染函数
            placeholder_text: 占位文本
            auto_load: 是否自动加载
        """
        state_key = f"_lazy_{component_id}_loaded"

        if auto_load or st.session_state.get(state_key, False):
            render_func()
            st.session_state[state_key] = True
        else:
            if st.button(placeholder_text, key=f"_lazy_{component_id}_btn"):
                st.session_state[state_key] = True
                st.rerun()


class OptimizedFileReader:
    """优化的文件读取器 - 使用多线程和分块读取"""

    @staticmethod
    @SmartCache.cache_function(ttl=7200, key_prefix="file_read")
    def read_csv_optimized(
        file_path: str,
        chunksize: Optional[int] = None,
        n_rows: Optional[int] = None,
        use_threads: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        优化的CSV读取

        Args:
            file_path: 文件路径
            chunksize: 分块大小
            n_rows: 读取行数
            use_threads: 是否使用多线程
            **kwargs: 传递给pd.read_csv的参数

        Returns:
            DataFrame
        """
        # 设置优化参数
        kwargs.setdefault('engine', 'c')  # 使用C引擎
        kwargs.setdefault('low_memory', False)

        if n_rows:
            kwargs['nrows'] = n_rows

        if chunksize and not n_rows:
            # 分块读取
            chunks = []
            with st.spinner("正在读取数据..."):
                for chunk in pd.read_csv(file_path, chunksize=chunksize, **kwargs):
                    chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        else:
            # 直接读取
            return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def read_large_csv_preview(
        file_path: str,
        preview_rows: int = 1000,
        **kwargs
    ) -> tuple[pd.DataFrame, int]:
        """
        读取大型CSV的预览

        Returns:
            (预览DataFrame, 总行数)
        """
        # 先读取预览
        df_preview = pd.read_csv(file_path, nrows=preview_rows, **kwargs)

        # 统计总行数（快速方法）
        with open(file_path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
            total_rows = sum(1 for _ in f) - 1  # 减去表头

        return df_preview, total_rows


class SessionStateOptimizer:
    """Session State优化器 - 减少不必要的状态存储"""

    @staticmethod
    def get_or_create(key: str, default_factory: Callable):
        """获取或创建session state值"""
        if key not in st.session_state:
            st.session_state[key] = default_factory()
        return st.session_state[key]

    @staticmethod
    def cleanup_old_states(max_age_seconds: int = 3600):
        """清理旧的session state"""
        current_time = time.time()

        # 查找带时间戳的键
        keys_to_remove = []
        for key in st.session_state.keys():
            if key.startswith('_temp_'):
                # 检查时间戳
                timestamp_key = f"{key}_timestamp"
                if timestamp_key in st.session_state:
                    if current_time - st.session_state[timestamp_key] > max_age_seconds:
                        keys_to_remove.append(key)
                        keys_to_remove.append(timestamp_key)

        for key in keys_to_remove:
            del st.session_state[key]

        return len(keys_to_remove) // 2

    @staticmethod
    def set_temp(key: str, value: Any, ttl: int = 3600):
        """设置临时状态（带过期时间）"""
        st.session_state[f"_temp_{key}"] = value
        st.session_state[f"_temp_{key}_timestamp"] = time.time()


# 便捷函数
def async_load_data(key: str, load_func: Callable, *args, **kwargs):
    """异步加载数据的便捷函数"""
    return AsyncDataLoader.get_data(key, load_func, *args, **kwargs)


def preload_next_page_data(page_name: str, load_func: Callable):
    """预加载下一页数据"""
    AsyncDataLoader.preload_data(f"page_{page_name}", load_func)


def optimized_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """优化的CSV读取便捷函数"""
    return OptimizedFileReader.read_csv_optimized(file_path, **kwargs)


def parallel_process_dataframe(df: pd.DataFrame, func: Callable, n_workers: int = 4):
    """并行处理DataFrame的便捷函数"""
    return ParallelDataProcessor.parallel_apply(df, func, n_workers)
