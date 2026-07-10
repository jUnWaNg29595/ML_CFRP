# -*- coding: utf-8 -*-
"""
多 GPU 训练工具模块

提供统一的多 GPU 支持函数，用于 ANN、PINN、GNN 等深度学习模型。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def wrap_model_for_multi_gpu(
    model: nn.Module,
    device: torch.device,
    use_data_parallel: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, bool, int]:
    """
    将模型包装为多 GPU 训练模式（如果可用）

    Parameters
    ----------
    model : nn.Module
        PyTorch 模型
    device : torch.device
        目标设备
    use_data_parallel : bool
        是否启用 DataParallel（默认 True）
    verbose : bool
        是否打印信息

    Returns
    -------
    model : nn.Module
        包装后的模型（可能是 DataParallel）
    is_parallel : bool
        是否使用了 DataParallel
    gpu_count : int
        GPU 数量
    """
    gpu_count = 0
    is_parallel = False

    if device.type == "cuda" and use_data_parallel:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            if verbose:
                print(f"  检测到 {gpu_count} 个 GPU，启用 DataParallel")
            model = nn.DataParallel(model)
            is_parallel = True

    return model, is_parallel, gpu_count


def unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """
    从 DataParallel 中提取原始模型

    Parameters
    ----------
    model : nn.Module
        可能被 DataParallel 包装的模型

    Returns
    -------
    nn.Module
        原始模型
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_model_for_inference(model: nn.Module, device: torch.device) -> nn.Module:
    """
    获取用于推理的模型（单 GPU，不使用 DataParallel）

    Parameters
    ----------
    model : nn.Module
        训练好的模型（可能是 DataParallel）
    device : torch.device
        推理设备

    Returns
    -------
    nn.Module
        推理模型
    """
    if isinstance(model, nn.DataParallel):
        return model.module.to(device)
    return model.to(device)


def print_gpu_info(device: torch.device, gpu_count: int = 0):
    """
    打印 GPU 信息

    Parameters
    ----------
    device : torch.device
        当前设备
    gpu_count : int
        GPU 数量
    """
    if device.type == "cuda":
        try:
            if gpu_count > 1:
                print(f"  设备: {device} (使用 {gpu_count} 个 GPU)")
                for i in range(gpu_count):
                    print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"  设备: {device} ({torch.cuda.get_device_name(0)})")
        except:
            print(f"  设备: {device} ✓")
    else:
        print(f"  设备: {device}")
