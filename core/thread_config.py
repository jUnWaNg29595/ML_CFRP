# -*- coding: utf-8 -*-
"""
线程配置模块 - 必须在导入 RDKit 等库之前执行

解决问题：即使 n_jobs=1，RDKit 底层的 OpenMP 也会占用所有 CPU 核心
原因：RDKit 使用 OpenMP 进行分子描述符计算的并行化，默认使用所有可用核心

此模块必须在任何可能导入 RDKit 的模块之前导入！
"""

import os
import multiprocessing

def get_optimal_thread_count():
    """
    计算最优线程数
    - 默认使用 CPU 核心数（允许全部核心用于计算密集型任务）
    - 可通过环境变量 ML_THREAD_COUNT 覆盖
    - 若已设置 OMP_NUM_THREADS，则优先沿用
    - 若设为 -1 则使用所有核心
    """
    if "ML_THREAD_COUNT" in os.environ:
        try:
            val = int(os.environ["ML_THREAD_COUNT"])
            if val == -1:
                return multiprocessing.cpu_count()
            return max(1, val)
        except ValueError:
            pass

    if "OMP_NUM_THREADS" in os.environ:
        try:
            val = int(os.environ["OMP_NUM_THREADS"])
            if val == -1:
                return multiprocessing.cpu_count()
            return max(1, val)
        except ValueError:
            pass
    
    cpu_count = multiprocessing.cpu_count()
    # 默认使用所有核心（移除旧的限制）
    # 用户可通过环境变量 ML_THREAD_COUNT 来限制
    return cpu_count

def configure_thread_limits(num_threads=None, verbose=False):
    """
    配置各种库的线程限制
    
    Parameters:
    -----------
    num_threads : int, optional
        线程数，默认自动计算（使用所有核心）
        -1 表示使用所有核心
    verbose : bool
        是否打印配置信息
    
    Note:
    -----
    此函数必须在导入以下库之前调用：
    - RDKit (使用 OpenMP)
    - NumPy/SciPy (可能使用 OpenBLAS/MKL)
    - scikit-learn (使用 OpenMP)
    - XGBoost (使用 OpenMP)
    """
    if num_threads is None:
        num_threads = get_optimal_thread_count()
    elif num_threads == -1:
        num_threads = multiprocessing.cpu_count()
    
    num_threads = max(1, num_threads)
    num_threads_str = str(num_threads)
    
    # 强制设置（直接赋值而非setdefault，确保生效）
    # OpenMP 线程限制（影响 RDKit, scikit-learn, XGBoost 等）
    os.environ["OMP_NUM_THREADS"] = num_threads_str
    
    # OpenBLAS 线程限制
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads_str
    
    # MKL 线程限制（Intel Math Kernel Library）
    os.environ["MKL_NUM_THREADS"] = num_threads_str
    
    # BLAS 通用设置
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads_str
    
    # XGBoost 特定设置
    os.environ["XGB_NUM_THREADS"] = num_threads_str
    
    # TensorFlow 线程设置（在 TF 导入前设置）
    os.environ["TF_NUM_INTEROP_THREADS"] = num_threads_str
    os.environ["TF_NUM_INTRAOP_THREADS"] = num_threads_str
    
    # 额外的 OpenMP 设置
    os.environ["OMP_DYNAMIC"] = "FALSE"  # 禁用动态线程调整
    os.environ["OMP_PROC_BIND"] = "FALSE"  # 不绑定线程到特定核心

    # 使用新的 OMP_MAX_ACTIVE_LEVELS 替代已弃用的 OMP_NESTED
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 禁用嵌套并行
    
    if verbose:
        print(f"✓ 线程配置已应用: {num_threads} 线程 (CPU 核心数: {multiprocessing.cpu_count()})")
    
    return num_threads

def get_current_thread_config():
    """获取当前线程配置信息（用于调试）"""
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "未设置"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "未设置"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "未设置"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "未设置"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "未设置"),
        "XGB_NUM_THREADS": os.environ.get("XGB_NUM_THREADS", "未设置"),
        "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS", "未设置"),
        "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS", "未设置"),
        "CPU_COUNT": multiprocessing.cpu_count(),
        "CONFIGURED_THREADS": os.environ.get("OMP_NUM_THREADS", str(multiprocessing.cpu_count())),
    }


def set_thread_count(num_threads):
    """
    动态设置线程数（运行时调用）
    
    注意：某些库（如已导入的 NumPy/RDKit）可能不会响应此更改
    建议在程序启动时通过环境变量或 configure_thread_limits() 设置
    """
    return configure_thread_limits(num_threads, verbose=True)

# ============================================
# 模块加载时自动配置线程限制
# ============================================
_CONFIGURED_THREADS = configure_thread_limits()

# 提供一个简单的方式来检查是否已配置
THREAD_CONFIG_APPLIED = True
THREAD_COUNT = _CONFIGURED_THREADS
