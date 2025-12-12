# -*- coding: utf-8 -*-
"""
超参数优化模块
更新内容：
1. optimize 方法增加 progress_callback 参数，支持实时进度条。
2. 保持了之前的自动数据清洗逻辑。
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import warnings

# 抑制 Optuna 的日志输出，只显示进度条
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

from core.model_trainer import EnhancedModelTrainer


class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self):
        self.trainer = EnhancedModelTrainer()

    def get_model_params(self, trial, model_name):
        """定义各模型的参数搜索空间"""
        params = {}

        if model_name == "随机森林":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }

        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1
            }

        elif model_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'verbose': 0
            }

        elif model_name == "SVR":
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            }

        elif model_name in ["Ridge回归", "Lasso回归", "ElasticNet"]:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True)
            }
            if model_name == "ElasticNet":
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)

        elif model_name == "AdaBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
            }

        elif model_name == "梯度提升树":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        return params

    def optimize(self, model_name, X, y, n_trials=50, cv=5, random_state=42, progress_callback=None):
        """
        执行超参数优化
        Args:
            progress_callback: 回调函数，接收一个 0-1 之间的浮点数表示进度
        """

        # 1. 确保输入是 numpy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

        # 2. 移除 y 中的 NaN 值
        mask = ~np.isnan(y)
        if np.sum(~mask) > 0:
            print(f"⚠️ 警告: 检测到目标变量 y 中有 {np.sum(~mask)} 个缺失值，已在优化前自动移除对应样本。")
            X = X[mask]
            y = y[mask]

        # 再次检查是否有无穷大
        mask_inf = ~np.isinf(y)
        if np.sum(~mask_inf) > 0:
            print(f"⚠️ 警告: 检测到目标变量 y 中有 {np.sum(~mask_inf)} 个无穷大值，已移除。")
            X = X[mask_inf]
            y = y[mask_inf]

        def objective(trial):
            # 更新进度条
            if progress_callback:
                progress_callback((trial.number + 1) / n_trials)

            # 获取建议参数
            params = self.get_model_params(trial, model_name)

            try:
                # 调用正确的方法名 _get_model
                base_model = self.trainer._get_model(model_name, **params)

                # 增加 SimpleImputer 处理特征缺失
                pipeline = make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler(),
                    base_model
                )

                # 定义交叉验证策略
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

                # 执行交叉验证
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=cv_strategy,
                    scoring='r2',
                    n_jobs=-1,  # 并行计算
                    error_score='raise'
                )

                return scores.mean()

            except Exception as e:
                print(f"❌ Trial {trial.number} failed: {str(e)}")
                return -float('inf')

        # 创建 Study 对象
        study = optuna.create_study(direction="maximize")

        # 执行优化
        study.optimize(objective, n_trials=n_trials)

        # 确保进度条走完
        if progress_callback:
            progress_callback(1.0)

        return study.best_params, study.best_value, study


class InverseDesigner:
    """反向设计器 (占位，预留未来功能)"""

    def __init__(self):
        pass


def generate_tuning_suggestions(model_name, current_score):
    """生成调参建议"""
    return f"建议增加 {model_name} 的搜索空间或增加迭代次数。"