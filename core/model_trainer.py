# -*- coding: utf-8 -*-
"""模型训练模块"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# [修复] 导入自定义 ANN 模型
try:
    from .ann_model import ANNRegressor

    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False


def _safe_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), True
    except ImportError:
        return None, False


XGBRegressor, XGBOOST_AVAILABLE = _safe_import('xgboost', 'XGBRegressor')
LGBMRegressor, LIGHTGBM_AVAILABLE = _safe_import('lightgbm', 'LGBMRegressor')
CatBoostRegressor, CATBOOST_AVAILABLE = _safe_import('catboost', 'CatBoostRegressor')
TabPFNRegressor, TABPFN_AVAILABLE = _safe_import('tabpfn', 'TabPFNRegressor')

try:
    from autogluon.tabular import TabularPredictor

    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class EnhancedModelTrainer:
    """增强版模型训练器"""

    def __init__(self):
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        models = ["线性回归", "Ridge回归", "Lasso回归", "ElasticNet", "决策树",
                  "随机森林", "Extra Trees", "梯度提升树", "AdaBoost", "SVR", "多层感知器"]
        if XGBOOST_AVAILABLE: models.append("XGBoost")
        if LIGHTGBM_AVAILABLE: models.append("LightGBM")
        if CATBOOST_AVAILABLE: models.append("CatBoost")
        if TABPFN_AVAILABLE: models.append("TabPFN")
        if AUTOGLUON_AVAILABLE: models.append("AutoGluon")
        # [修复] 注册人工神经网络
        if ANN_AVAILABLE: models.append("人工神经网络")
        return models

    def get_available_models(self):
        return self.available_models.copy()

    def _get_model(self, name, **params):
        models = {
            "线性回归": LinearRegression,
            "Ridge回归": Ridge,
            "Lasso回归": Lasso,
            "ElasticNet": ElasticNet,
            "决策树": DecisionTreeRegressor,
            "随机森林": RandomForestRegressor,
            "Extra Trees": ExtraTreesRegressor,
            "梯度提升树": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor,
            "SVR": SVR,
            "多层感知器": MLPRegressor,
        }

        if name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBRegressor(**params)
        elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
            params.setdefault('verbose', -1)
            return LGBMRegressor(**params)
        elif name == "CatBoost" and CATBOOST_AVAILABLE:
            params.setdefault('verbose', 0)
            return CatBoostRegressor(**params)
        elif name == "TabPFN" and TABPFN_AVAILABLE:
            return TabPFNRegressor(**params)
        # [修复] 实例化人工神经网络
        elif name == "人工神经网络" and ANN_AVAILABLE:
            return ANNRegressor(**params)
        elif name in models:
            return models[name](**params)
        else:
            raise ValueError(f"Unknown model: {name}")

    def train_model(self, X, y, model_name, test_size=0.2, random_state=42, **params):
        # 数据处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

        # 处理缺失值
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 创建Pipeline
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)

        # 智能注入 random_state 到模型参数
        # 排除不支持 random_state 的模型，防止二次报错
        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]

        model_params = params.copy()
        if model_name not in NO_SEED_MODELS:
            model_params['random_state'] = random_state

        # 训练模型
        start_time = time.time()

        # [修复] 这里原本错误地使用了 **params，导致 random_state 未生效
        # 现在改为使用包含 random_state 的 **model_params
        model = self._get_model(model_name, **model_params)

        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # 预测和评估
        y_pred = model.predict(X_test_scaled)

        return {
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'pipeline': None,
            'X_train': pd.DataFrame(X_train_scaled),
            'X_test': pd.DataFrame(X_test_scaled),
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_time': train_time
        }