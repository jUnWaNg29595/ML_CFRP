# -*- coding: utf-8 -*-
"""GPU 自动恢复模块 - 失败时自动清理显存并恢复原状"""

import gc
import functools
import traceback
from typing import Callable, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def force_cleanup_all_gpus():
    """强制清理所有 GPU 显存"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return

    try:
        # 清理所有 GPU
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        # Python 垃圾回收
        gc.collect()

        # 再次清理
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    except Exception:
        pass


def get_gpu_memory_status():
    """获取所有 GPU 显存状态"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}

    status = {}
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            status[i] = {
                'name': props.name,
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'free': total - allocated
            }
    except Exception:
        pass

    return status


def auto_recovery_wrapper(
    fallback_value: Any = None,
    cleanup_on_error: bool = True,
    verbose: bool = True,
    max_retries: int = 0
):
    """
    自动恢复装饰器 - GPU 操作失败时自动清理显存并恢复

    参数:
        fallback_value: 失败时返回的默认值
        cleanup_on_error: 是否在错误时清理 GPU 显存
        verbose: 是否打印详细信息
        max_retries: 最大重试次数（清理后重试）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 记录初始状态
            initial_status = get_gpu_memory_status() if cleanup_on_error else {}

            for attempt in range(max_retries + 1):
                try:
                    # 执行原函数
                    result = func(*args, **kwargs)
                    return result

                except Exception as e:
                    error_msg = str(e)
                    is_oom = "out of memory" in error_msg.lower() or "cuda" in error_msg.lower()

                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"❌ 操作失败: {func.__name__}")
                        print(f"错误类型: {type(e).__name__}")
                        print(f"错误信息: {error_msg[:200]}")

                        if is_oom:
                            print(f"\n🔍 检测到 GPU 显存问题")
                            current_status = get_gpu_memory_status()
                            for gpu_id, info in current_status.items():
                                print(f"GPU {gpu_id} ({info['name']}): "
                                      f"已用 {info['allocated']:.2f}GB / {info['total']:.2f}GB")

                    # 自动清理 GPU 显存
                    if cleanup_on_error and is_oom:
                        if verbose:
                            print(f"\n🔧 正在自动清理所有 GPU 显存...")

                        force_cleanup_all_gpus()

                        if verbose:
                            print(f"✓ 显存清理完成")
                            after_status = get_gpu_memory_status()
                            for gpu_id, info in after_status.items():
                                before = initial_status.get(gpu_id, {})
                                freed = before.get('allocated', 0) - info['allocated']
                                if freed > 0:
                                    print(f"GPU {gpu_id}: 释放了 {freed:.2f}GB 显存")

                    # 重试逻辑
                    if attempt < max_retries and is_oom:
                        if verbose:
                            print(f"\n🔄 尝试重试 ({attempt + 1}/{max_retries})...")
                        continue

                    # 最终失败，返回 fallback 值
                    if verbose:
                        print(f"\n⚠️ 操作最终失败，返回默认值")
                        print(f"{'='*60}\n")

                    return fallback_value

            # 所有重试都失败
            return fallback_value

        return wrapper
    return decorator


class GPUAutoRecoveryContext:
    """GPU 自动恢复上下文管理器"""

    def __init__(self, cleanup_on_exit: bool = True, cleanup_on_error: bool = True, verbose: bool = False):
        self.cleanup_on_exit = cleanup_on_exit
        self.cleanup_on_error = cleanup_on_error
        self.verbose = verbose
        self.initial_status = {}
        self.error_occurred = False

    def __enter__(self):
        self.initial_status = get_gpu_memory_status()
        if self.verbose:
            print(f"📊 GPU 初始状态:")
            for gpu_id, info in self.initial_status.items():
                print(f"  GPU {gpu_id}: {info['allocated']:.2f}GB / {info['total']:.2f}GB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True

            if self.verbose:
                print(f"\n❌ 上下文中发生错误: {exc_type.__name__}")
                print(f"错误信息: {str(exc_val)[:200]}")

            # 错误时清理
            if self.cleanup_on_error:
                if self.verbose:
                    print(f"🔧 自动清理 GPU 显存...")
                force_cleanup_all_gpus()

        # 正常退出时也可以选择清理
        elif self.cleanup_on_exit:
            force_cleanup_all_gpus()

        if self.verbose:
            final_status = get_gpu_memory_status()
            print(f"\n📊 GPU 最终状态:")
            for gpu_id, info in final_status.items():
                before = self.initial_status.get(gpu_id, {})
                change = info['allocated'] - before.get('allocated', 0)
                symbol = "+" if change > 0 else ""
                print(f"  GPU {gpu_id}: {info['allocated']:.2f}GB / {info['total']:.2f}GB ({symbol}{change:.2f}GB)")

        # 不抑制异常，让调用者知道发生了错误
        return False
