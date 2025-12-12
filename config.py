# -*- coding: utf-8 -*-
"""配置文件"""

APP_NAME = "碳纤维复合材料智能预测平台"
VERSION = "1.2.8"

DATA_DIR = "datasets"
MODEL_DIR = "models"
RESULT_DIR = "results"
CACHE_DIR = "cache"

TRADITIONAL_MODELS = [
    "线性回归", "Ridge回归", "Lasso回归", "ElasticNet",
    "决策树", "随机森林", "Extra Trees", "梯度提升树",
    "AdaBoost", "SVR", "多层感知器"
]

ADVANCED_MODELS = ["XGBoost", "LightGBM", "CatBoost", "TabPFN", "AutoGluon"]

MOLECULAR_FEATURE_LIBS = ["RDKit", "Mordred"]
