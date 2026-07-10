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
from sklearn.model_selection import cross_val_score, KFold, train_test_split
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

    def get_model_params(self, trial, model_name, fast_mode: bool = False):
        """定义各模型的参数搜索空间"""
        params = {}
        fast_mode = bool(fast_mode)

        if model_name == "随机森林":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "Extra Trees":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "决策树":
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "线性回归":
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'positive': trial.suggest_categorical('positive', [True, False])
            }

        elif model_name == "XGBoost":
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }

        elif model_name == "LightGBM":
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
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
            max_iters = 300 if fast_mode else 500
            params = {
                'iterations': trial.suggest_int('iterations', 50, max_iters),
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
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
            }

        elif model_name == "梯度提升树":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "多层感知器":
            params = {
                'hidden_layer_sizes': trial.suggest_categorical(
                    'hidden_layer_sizes',
                    [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)]
                ),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'max_iter': trial.suggest_int('max_iter', 200 if fast_mode else 300, 600)
            }

        elif model_name == "人工神经网络":
            hidden_opts = ["128,64", "256,128,64", "256,256,128", "512,256,128"]
            if fast_mode:
                hidden_opts = ["128,64", "256,128,64"]
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', hidden_opts),
                'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'epochs': trial.suggest_int('epochs', 60 if fast_mode else 100, 200)
            }

        elif model_name == "TensorFlow Sequential":
            hidden_opts = ["128,64", "128,64,32", "256,128,64", "256,256,128"]
            if fast_mode:
                hidden_opts = ["128,64", "128,64,32"]
            params = {
                'hidden_layers': trial.suggest_categorical('hidden_layers', hidden_opts),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'swish']),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.4),
                'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'epochs': trial.suggest_int('epochs', 60 if fast_mode else 100, 200),
                'early_stopping': True,
                'patience': trial.suggest_int('patience', 10, 30),
                'validation_split': trial.suggest_float('validation_split', 0.1, 0.2)
            }

        elif model_name == "Epoxy PINN (Physics-Informed)":
            max_epochs = 150 if fast_mode else 300
            params = {
                'mode': trial.suggest_categorical('mode', ['auto', 'tg', 'mechanics', 'generic']),
                'hidden_dim': trial.suggest_int('hidden_dim', 128, 768, step=64),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
                'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 0.0, 1e-3),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'epochs': trial.suggest_int('epochs', 60, max_epochs, step=30),
                'patience': trial.suggest_int('patience', 10, 40, step=5),
                'physics_weight': trial.suggest_float('physics_weight', 0.0, 5.0, step=0.5),
            }

        return params

    def optimize(
        self,
        model_name,
        X,
        y,
        n_trials=50,
        cv=5,
        random_state=42,
        progress_callback=None,
        cv_strategy: str = "kfold",
        val_size: float = 0.2,
        max_samples: int | None = None,
        n_jobs: int = -1,
        fast_mode: bool = False,
        use_pruner: bool = True,
        timeout: int | None = None,
        optimization_method: str = "tpe",
    ):
        """
        执行超参数优化
        Args:
            progress_callback: 回调函数，接收一个 0-1 之间的浮点数表示进度
            optimization_method: 优化方法，可选:
                - "tpe": Tree-structured Parzen Estimator (贝叶斯优化，默认)
                - "gp": Gaussian Process (高斯过程贝叶斯优化)
                - "cmaes": CMA-ES (协方差矩阵自适应进化策略)
                - "random": 随机搜索
                - "grid": 网格搜索 (需要较少trials)
        """

        # 0. 记录目标列名（用于 Epoxy PINN auto mode）
        y_name = getattr(y, 'name', None)

        # 1. 可选采样加速
        max_samples = int(max_samples) if max_samples is not None else 0
        if max_samples > 0:
            n_total = len(X)
            if n_total > max_samples:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(n_total, size=max_samples, replace=False)
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    X = X.iloc[idx]
                else:
                    X = X[idx]
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = y.iloc[idx]
                else:
                    y = y[idx]

        # 2. 确保输入是 numpy 数组
        if isinstance(X, pd.DataFrame) and model_name != "Epoxy PINN (Physics-Informed)":
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

        # 3. 移除 y 中的 NaN 值
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
            params = self.get_model_params(trial, model_name, fast_mode=fast_mode)

            # Epoxy PINN: 传入目标列名用于 auto mode
            if model_name == "Epoxy PINN (Physics-Informed)" and y_name is not None:
                params.setdefault("target_name", y_name)


            try:
                # 调用正确的方法名 _get_model
                base_model = self.trainer._get_model(model_name, **params)

                # 增加 SimpleImputer 处理特征缺失
                # Epoxy PINN: 允许原始 DataFrame（含字符串列），前处理由模型内部完成
                if model_name == "Epoxy PINN (Physics-Informed)":
                    pipeline = base_model
                else:
                    pipeline = make_pipeline(
                        SimpleImputer(strategy='median'),
                        StandardScaler(),
                        base_model
                    )

                if str(cv_strategy).lower().startswith("hold"):
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=float(val_size), random_state=random_state
                    )
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_val)
                    return r2_score(y_val, y_pred)

                # 定义交叉验证策略
                cv_obj = KFold(n_splits=int(cv), shuffle=True, random_state=random_state)

                if use_pruner:
                    scores = []
                    for fold_idx, (tr_idx, te_idx) in enumerate(cv_obj.split(X, y)):
                        X_tr = X[tr_idx] if not hasattr(X, "iloc") else X.iloc[tr_idx]
                        X_te = X[te_idx] if not hasattr(X, "iloc") else X.iloc[te_idx]
                        y_tr = y[tr_idx] if not hasattr(y, "iloc") else y.iloc[tr_idx]
                        y_te = y[te_idx] if not hasattr(y, "iloc") else y.iloc[te_idx]

                        pipeline.fit(X_tr, y_tr)
                        y_pred = pipeline.predict(X_te)
                        score = r2_score(y_te, y_pred)
                        scores.append(score)
                        trial.report(float(np.mean(scores)), step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    return float(np.mean(scores))

                # 执行交叉验证
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=cv_obj,
                    scoring='r2',
                    n_jobs=int(n_jobs),  # 并行计算
                    error_score='raise'
                )

                return float(scores.mean())

            except Exception as e:
                print(f"❌ Trial {trial.number} failed: {str(e)}")
                return -float('inf')

        # 创建 Study 对象
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3) if use_pruner else optuna.pruners.NopPruner()

        # 根据优化方法选择采样器
        optimization_method = str(optimization_method).lower()

        if optimization_method == "tpe":
            # TPE: Tree-structured Parzen Estimator (贝叶斯优化)
            sampler = optuna.samplers.TPESampler(seed=int(random_state))
            print("🔍 使用 TPE 贝叶斯优化")

        elif optimization_method == "gp":
            # GP: Gaussian Process (高斯过程贝叶斯优化)
            try:
                sampler = optuna.samplers.GPSampler(seed=int(random_state))
                print("🔍 使用 Gaussian Process 贝叶斯优化")
            except AttributeError:
                print("⚠️ GPSampler 不可用，回退到 TPE")
                sampler = optuna.samplers.TPESampler(seed=int(random_state))

        elif optimization_method == "cmaes":
            # CMA-ES: Covariance Matrix Adaptation Evolution Strategy
            try:
                sampler = optuna.samplers.CmaEsSampler(seed=int(random_state))
                print("🔍 使用 CMA-ES 优化")
            except Exception as e:
                print(f"⚠️ CMA-ES 不可用 ({str(e)})，回退到 TPE")
                print("💡 提示: 安装 cmaes 包以启用 CMA-ES: pip install cmaes")
                sampler = optuna.samplers.TPESampler(seed=int(random_state))

        elif optimization_method == "random":
            # 随机搜索
            sampler = optuna.samplers.RandomSampler(seed=int(random_state))
            print("🔍 使用随机搜索")

        elif optimization_method == "grid":
            # 网格搜索 (需要预定义搜索空间)
            sampler = optuna.samplers.GridSampler(seed=int(random_state))
            print("🔍 使用网格搜索")

        else:
            # 默认使用 TPE
            sampler = optuna.samplers.TPESampler(seed=int(random_state))
            print(f"⚠️ 未知优化方法 '{optimization_method}'，使用默认 TPE")

        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

        # 执行优化
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

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
