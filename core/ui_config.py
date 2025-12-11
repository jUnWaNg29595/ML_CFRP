# -*- coding: utf-8 -*-
"""UI界面配置和默认超参数 - 完整版"""

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_OPTUNA_TRIALS = 50
INVERSE_DESIGN_SPACE = {}

MODEL_PARAMETERS = {
    "线性回归": {},
    "Ridge回归": {"alpha": 1.0},
    "Lasso回归": {"alpha": 1.0},
    "ElasticNet": {"alpha": 1.0, "l1_ratio": 0.5},
    "决策树": {"max_depth": 5, "random_state": 42},
    "随机森林": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    "Extra Trees": {"n_estimators": 100, "random_state": 42},
    "梯度提升树": {"n_estimators": 100, "max_depth": 3, "random_state": 42},
    "AdaBoost": {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
    "SVR": {"kernel": 'rbf', "C": 100.0, "epsilon": 0.1},
    "多层感知器": {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": 42},
    "XGBoost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    "LightGBM": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": -1},
    "CatBoost": {"iterations": 100, "depth": 6, "random_state": 42, "verbose": 0},
    "人工神经网络": {"hidden_layer_sizes": "100,50", "epochs": 100},
    "TabPFN": {},
    "AutoGluon": {}
}

# 完整的手动调参配置
MANUAL_TUNING_PARAMS = {
    "线性回归": [],
    
    "Ridge回归": [
        {'name': 'alpha', 'widget': 'number_input', 'label': '正则化强度 (Alpha)',
         'default': 1.0, 'args': {'min_value': 0.01, 'max_value': 100.0, 'step': 0.01}}
    ],
    
    "Lasso回归": [
        {'name': 'alpha', 'widget': 'number_input', 'label': '正则化强度 (Alpha)',
         'default': 1.0, 'args': {'min_value': 0.01, 'max_value': 100.0, 'step': 0.01}}
    ],
    
    "ElasticNet": [
        {'name': 'alpha', 'widget': 'number_input', 'label': '正则化强度 (Alpha)',
         'default': 1.0, 'args': {'min_value': 0.01, 'max_value': 100.0, 'step': 0.01}},
        {'name': 'l1_ratio', 'widget': 'slider', 'label': 'L1比例',
         'default': 0.5, 'args': {'min_value': 0.0, 'max_value': 1.0, 'step': 0.1}}
    ],
    
    "决策树": [
        {'name': 'max_depth', 'widget': 'slider', 'label': '最大深度',
         'default': 5, 'args': {'min_value': 1, 'max_value': 30, 'step': 1}},
        {'name': 'min_samples_split', 'widget': 'slider', 'label': '最小分裂样本数',
         'default': 2, 'args': {'min_value': 2, 'max_value': 20, 'step': 1}}
    ],
    
    "SVR": [
        {'name': 'C', 'widget': 'number_input', 'label': '正则化参数 (C)',
         'default': 100.0, 'args': {'min_value': 0.01, 'max_value': 1000.0, 'step': 0.1}},
        {'name': 'kernel', 'widget': 'selectbox', 'label': '核函数',
         'default': 'rbf', 'args': {'options': ['rbf', 'linear', 'poly', 'sigmoid']}},
        {'name': 'epsilon', 'widget': 'number_input', 'label': 'Epsilon',
         'default': 0.1, 'args': {'min_value': 0.01, 'max_value': 1.0, 'step': 0.01}}
    ],
    
    "随机森林": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '决策树数量',
         'default': 100, 'args': {'min_value': 10, 'max_value': 1000, 'step': 10}},
        {'name': 'max_depth', 'widget': 'number_input', 'label': '最大深度',
         'default': 10, 'args': {'min_value': 1, 'max_value': 100, 'step': 1}},
        {'name': 'min_samples_split', 'widget': 'slider', 'label': '最小分裂样本数',
         'default': 2, 'args': {'min_value': 2, 'max_value': 20, 'step': 1}},
        {'name': 'min_samples_leaf', 'widget': 'slider', 'label': '最小叶子样本数',
         'default': 1, 'args': {'min_value': 1, 'max_value': 10, 'step': 1}}
    ],
    
    "Extra Trees": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '决策树数量',
         'default': 100, 'args': {'min_value': 10, 'max_value': 1000, 'step': 10}},
        {'name': 'max_depth', 'widget': 'number_input', 'label': '最大深度',
         'default': 10, 'args': {'min_value': 1, 'max_value': 100, 'step': 1}}
    ],
    
    "梯度提升树": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '迭代次数',
         'default': 100, 'args': {'min_value': 10, 'max_value': 500, 'step': 10}},
        {'name': 'max_depth', 'widget': 'slider', 'label': '最大深度',
         'default': 3, 'args': {'min_value': 1, 'max_value': 10, 'step': 1}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 0.1, 'args': {'min_value': 0.01, 'max_value': 1.0, 'step': 0.01}}
    ],
    
    "AdaBoost": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '迭代次数',
         'default': 50, 'args': {'min_value': 10, 'max_value': 500, 'step': 10}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 1.0, 'args': {'min_value': 0.01, 'max_value': 2.0, 'step': 0.01}}
    ],
    
    "多层感知器": [
        {'name': 'hidden_layer_sizes', 'widget': 'text_input', 'label': '隐藏层结构 (如: 100,50)',
         'default': '100,50', 'args': {}},
        {'name': 'max_iter', 'widget': 'slider', 'label': '最大迭代次数',
         'default': 500, 'args': {'min_value': 100, 'max_value': 2000, 'step': 100}}
    ],
    
    "XGBoost": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '迭代次数',
         'default': 100, 'args': {'min_value': 50, 'max_value': 1000, 'step': 50}},
        {'name': 'max_depth', 'widget': 'slider', 'label': '最大深度',
         'default': 3, 'args': {'min_value': 1, 'max_value': 10, 'step': 1}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 0.1, 'args': {'min_value': 0.01, 'max_value': 0.5, 'step': 0.01}},
        {'name': 'subsample', 'widget': 'slider', 'label': '样本采样率',
         'default': 1.0, 'args': {'min_value': 0.5, 'max_value': 1.0, 'step': 0.1}},
        {'name': 'colsample_bytree', 'widget': 'slider', 'label': '特征采样率',
         'default': 1.0, 'args': {'min_value': 0.5, 'max_value': 1.0, 'step': 0.1}}
    ],
    
    "LightGBM": [
        {'name': 'n_estimators', 'widget': 'slider', 'label': '迭代次数',
         'default': 100, 'args': {'min_value': 50, 'max_value': 1000, 'step': 50}},
        {'name': 'max_depth', 'widget': 'slider', 'label': '最大深度',
         'default': 5, 'args': {'min_value': 1, 'max_value': 15, 'step': 1}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 0.1, 'args': {'min_value': 0.01, 'max_value': 0.5, 'step': 0.01}},
        {'name': 'num_leaves', 'widget': 'slider', 'label': '叶子节点数',
         'default': 31, 'args': {'min_value': 10, 'max_value': 150, 'step': 5}}
    ],
    
    "CatBoost": [
        {'name': 'iterations', 'widget': 'slider', 'label': '迭代次数',
         'default': 100, 'args': {'min_value': 50, 'max_value': 1000, 'step': 50}},
        {'name': 'depth', 'widget': 'slider', 'label': '树深度',
         'default': 6, 'args': {'min_value': 1, 'max_value': 10, 'step': 1}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 0.1, 'args': {'min_value': 0.01, 'max_value': 0.5, 'step': 0.01}}
    ],
    
    "人工神经网络": [
        {'name': 'hidden_layer_sizes', 'widget': 'text_input', 'label': '隐藏层结构 (如: 100,50)',
         'default': '100,50', 'args': {}},
        {'name': 'activation', 'widget': 'selectbox', 'label': '激活函数',
         'default': 'relu', 'args': {'options': ['relu', 'leaky_relu', 'tanh', 'elu']}},
        {'name': 'dropout_rate', 'widget': 'slider', 'label': 'Dropout率',
         'default': 0.2, 'args': {'min_value': 0.0, 'max_value': 0.5, 'step': 0.05}},
        {'name': 'learning_rate', 'widget': 'number_input', 'label': '学习率',
         'default': 0.001, 'args': {'min_value': 0.0001, 'max_value': 0.1, 'step': 0.0001}},
        {'name': 'epochs', 'widget': 'slider', 'label': '训练轮数',
         'default': 100, 'args': {'min_value': 10, 'max_value': 500, 'step': 10}},
        {'name': 'batch_size', 'widget': 'slider', 'label': '批大小',
         'default': 32, 'args': {'min_value': 8, 'max_value': 128, 'step': 8}}
    ],
    
    "TabPFN": [],
    "AutoGluon": []
}
