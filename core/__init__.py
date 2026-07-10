# -*- coding: utf-8 -*-
"""核心模块包"""

# ============================================
# 重要：必须首先导入线程配置模块！
# 这会在导入 RDKit 等库之前设置 OpenMP 线程限制
# ============================================
from . import thread_config

# [新增] 后台任务管理器
from . import task_manager

# 反应模拟模块（环氧-固化剂模拟反应）
try:
    from .reaction_simulator import (
        EpoxyReactionSimulator,
        CrosslinkedFeatureExtractor,
        SimplifiedReactionModel,
        simulate_epoxy_curing,
        extract_crosslink_features,
        get_reaction_product_smiles,
        batch_extract_crosslink_features,
    )
    REACTION_SIMULATOR_AVAILABLE = True
except ImportError:
    REACTION_SIMULATOR_AVAILABLE = False

__all__ = [
    'thread_config',
    'task_manager',
    # 反应模拟
    'EpoxyReactionSimulator',
    'CrosslinkedFeatureExtractor', 
    'SimplifiedReactionModel',
    'simulate_epoxy_curing',
    'extract_crosslink_features',
    'get_reaction_product_smiles',
    'batch_extract_crosslink_features',
    'REACTION_SIMULATOR_AVAILABLE',
]
