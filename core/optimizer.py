# -*- coding: utf-8 -*-
"""超参数优化模块"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optuna超参数优化器"""

    def __init__(self):
        self.best_params = None
        self.best_score = None
        self.study = None

    def _objective(self, trial, model_name, X, y, cv):
        from .model_trainer import EnhancedModelTrainer

        if pd.isna(y).any():
            valid_idx = ~pd.isna(y)
            X, y = X[valid_idx], y[valid_idx]

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel()

        trainer = EnhancedModelTrainer()
        params = {}

        if model_name == "随机森林":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': 42
            }
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'random_state': 42, 'verbose': -1
            }
        elif model_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42, 'verbose': 0
            }
        elif model_name == "SVR":
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True)
            }
        elif model_name == "Ridge回归":
            params = {'alpha': trial.suggest_float('alpha', 0.01, 100, log=True)}
        elif model_name == "Lasso回归":
            params = {'alpha': trial.suggest_float('alpha', 0.01, 100, log=True)}
        elif model_name == "ElasticNet":
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
            }
        elif model_name == "AdaBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True)
            }
        elif model_name == "梯度提升树":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42
            }

        try:
            model = trainer._get_model(model_name, **params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
            return scores.mean()
        except:
            return -np.inf

    def optimize(self, model_name, X, y, n_trials=50, cv=5):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel()

        # 处理缺失值
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        # 填充特征缺失值
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(X))

        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(
            lambda trial: self._objective(trial, model_name, X, y, cv),
            n_trials=n_trials, show_progress_bar=True
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params, self.best_score, self.study


class InverseDesigner:
    """逆向设计器"""

    def __init__(self, model, scaler, feature_names, target_name):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_name = target_name

    def design(self, target_value, bounds, n_trials=100):
        best_x = None
        best_diff = float('inf')

        for _ in range(n_trials):
            x = np.array([np.random.uniform(bounds[f][0], bounds[f][1]) for f in self.feature_names])
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            pred = self.model.predict(x_scaled)[0]
            diff = abs(pred - target_value)
            if diff < best_diff:
                best_diff = diff
                best_x = x

        return dict(zip(self.feature_names, best_x)), best_diff


def generate_tuning_suggestions(model_name, current_score):
    """生成调优建议"""
    suggestions = []
    
    if model_name in ["随机森林", "Extra Trees"]:
        suggestions.append("尝试增加n_estimators (200-500)")
        suggestions.append("调整max_depth防止过拟合")
    elif model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        suggestions.append("降低learning_rate并增加n_estimators")
        suggestions.append("调整正则化参数防止过拟合")
    elif model_name == "SVR":
        suggestions.append("尝试不同的kernel")
        suggestions.append("调整C和epsilon参数")
    
    if current_score < 0.7:
        suggestions.append("考虑增加更多特征")
        suggestions.append("检查数据质量和异常值")
    
    return suggestions
