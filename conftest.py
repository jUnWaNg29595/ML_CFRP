# -*- coding: utf-8 -*-
"""
Pytest 根目录测试前置配置：
适配 NumPy 2.x 与已安装的老版本 SciPy 之间的兼容性补丁 (np.long / np.ulong)。
"""
import numpy as np

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
