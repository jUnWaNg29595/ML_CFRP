# -*- coding: utf-8 -*-
"""模型训练模块"""

import time
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

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


# --- [新增] AutoGluon 适配器 ---
class AutoGluonWrapper(BaseEstimator, RegressorMixin):
    """
    将 AutoGluon 封装为 Scikit-Learn 风格的 Estimator
    使其支持 fit(X, y) 和 predict(X)
    """

    def __init__(self, time_limit=60, presets='medium_quality', **kwargs):
        self.time_limit = time_limit
        self.presets = presets
        self.kwargs = kwargs
        self.predictor = None
        self.label_col = 'target'
        self.save_path = f"AutogluonModels/ag-{int(time.time())}"

    def fit(self, X, y):
        # 1. 转换数据格式：AutoGluon 需要 DataFrame 包含 X 和 y
        if isinstance(X, np.ndarray):
            # 如果是 numpy 数组，赋予默认列名
            train_data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            train_data = pd.DataFrame(X).copy()

        # 记录特征列名，以便预测时对齐
        self.feature_names_ = train_data.columns.tolist()

        # 添加目标列
        train_data[self.label_col] = y

        # 2. 调用 AutoGluon 训练
        # verbosity=0 静默模式，避免刷屏
        self.predictor = TabularPredictor(
            label=self.label_col,
            path=self.save_path,
            verbosity=0
        ).fit(
            train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            **self.kwargs
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise RuntimeError("AutoGluon model not fitted yet.")

        # 确保输入格式与训练时一致
        if isinstance(X, np.ndarray):
            test_data = pd.DataFrame(X, columns=self.feature_names_)
        else:
            test_data = pd.DataFrame(X)
            # 尝试对齐列名
            if test_data.shape[1] == len(self.feature_names_):
                test_data.columns = self.feature_names_

        return self.predictor.predict(test_data).values

    def __del__(self):
        # (可选) 清理临时模型文件，防止磁盘占满
        # try:
        #     shutil.rmtree(self.save_path)
        # except:
        #     pass
        pass


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
        # [新增] AutoGluon 分支
        elif name == "AutoGluon" and AUTOGLUON_AVAILABLE:
            # 默认给一个较短的时间限制，防止网页卡死
            params.setdefault('time_limit', 30)
            return AutoGluonWrapper(**params)
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

        # [关键修复]：强制将 y 转换为数值类型，无法转换的（如字符串）变为 NaN
        y = pd.to_numeric(y, errors='coerce')

        # 处理缺失值
        mask = ~np.isnan(y)
        # 维度检查
        if len(X) != len(y):
            # 尝试截断较长的一方（这通常是上游处理错误，但这里做个防御）
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            mask = mask[:min_len]

        X, y = X[mask], y[mask]

        if len(y) == 0:
            raise ValueError("所有样本的目标变量均无效（NaN），无法训练模型，请检查数据")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 创建Pipeline
        # 注意：AutoGluon 其实不需要 Imputer/Scaler，但为了统一流程我们保留
        # AutoGluonWrapper 会接收处理后的 numpy 数组并转回 DataFrame
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)

        # 智能注入 random_state 到模型参数
        # 排除不支持 random_state 的模型
        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]

        model_params = params.copy()
        if model_name not in NO_SEED_MODELS:
            model_params['random_state'] = random_state

        # 训练模型
        start_time = time.time()

        # 获取模型实例
        model = self._get_model(model_name, **model_params)

        # 训练
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # 预测和评估
        y_pred = model.predict(X_test_scaled)  # 测试集预测
        y_pred_train = model.predict(X_train_scaled)  # 训练集预测

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
            'y_pred_test': y_pred,
            'y_pred_train': y_pred_train,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_time': train_time
        }