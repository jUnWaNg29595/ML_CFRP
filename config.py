# -*- coding: utf-8 -*-
"""配置文件 - 更新版（含 TensorFlow Sequential 模型）"""

APP_NAME = "碳纤维复合材料智能预测平台"
VERSION = "1.4.2"  # 版本更新

DATA_DIR = "datasets"
MODEL_DIR = "models"
RESULT_DIR = "results"
CACHE_DIR = "cache"

# 传统模型列表
TRADITIONAL_MODELS = [
    "线性回归", "Ridge回归", "Lasso回归", "ElasticNet",
    "决策树", "随机森林", "Extra Trees", "梯度提升树",
    "AdaBoost", "SVR", "多层感知器"
]

# 高级模型列表（新增 TensorFlow Sequential）
ADVANCED_MODELS = [
    "XGBoost", "LightGBM", "CatBoost",
    "TensorFlow Sequential",  # 新增
    "TabPFN", "AutoGluon"
]

# 深度学习模型列表
DEEP_LEARNING_MODELS = [
    "人工神经网络",        # PyTorch ANN
    "TensorFlow Sequential"  # TensorFlow Sequential
]

# 分子特征库
MOLECULAR_FEATURE_LIBS = ["RDKit", "Mordred"]

# 特征工程操作类型定义
FE_OPERATION_TYPES = {
    "data_load": "数据加载",
    "data_clean": "数据清洗",
    "missing_value": "缺失值处理",
    "outlier_detect": "异常值检测",
    "molecular_feature": "分子特征提取",
    "feature_select": "特征选择",
    "feature_transform": "特征变换",
    "data_augment": "数据增强",
    "data_split": "数据划分",
    "model_train": "模型训练",
    "custom": "自定义操作"
}
