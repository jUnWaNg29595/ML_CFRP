# -*- coding: utf-8 -*-
"""TabNet模型封装（sklearn风格）

TabNet是专为表格数据设计的深度学习模型，具有以下特点：
- 可解释性：通过attention机制选择重要特征
- 端到端学习：无需手动特征工程
- 自监督预训练：可利用无标签数据
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    _TabNetRegressor = None


class TabNetRegressor(BaseEstimator, RegressorMixin):
    """sklearn风格的TabNet回归器

    参数说明：
    - n_d: 决策层维度（默认8）
    - n_a: 注意力层维度（默认8）
    - n_steps: 决策步数（默认3-10）
    - gamma: 特征重用系数（默认1.3）
    - n_independent: 独立GLU层数（默认2）
    - n_shared: 共享GLU层数（默认2）
    - lambda_sparse: 稀疏正则化系数（默认1e-3）
    - optimizer_fn: 优化器（默认torch.optim.Adam）
    - optimizer_params: 优化器参数
    - scheduler_fn: 学习率调度器
    - scheduler_params: 调度器参数
    - mask_type: mask类型（'sparsemax'或'entmax'）
    - seed: 随机种子
    """

    def __init__(
        self,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        momentum=0.02,
        clip_value=1.0,
        optimizer_fn=None,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        mask_type='sparsemax',
        seed=42,
        verbose=1,
        device_name='auto',
        # 训练参数
        max_epochs=200,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params or {'lr': 5e-3}  # 降低学习率从2e-2到5e-3
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.seed = seed
        self.verbose = verbose
        self.device_name = device_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size

        self.model_ = None

    @staticmethod
    def is_available():
        return TABNET_AVAILABLE

    def fit(self, X, y, eval_set=None, eval_metric=None):
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet需要安装pytorch-tabnet: pip install pytorch-tabnet")

        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # 数据验证
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("输入数据包含NaN或Inf值，请先进行数据清洗")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("目标变量包含NaN或Inf值，请先进行数据清洗")

        # 创建模型（移除可能为None的参数）
        model_params = {
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'n_independent': self.n_independent,
            'n_shared': self.n_shared,
            'lambda_sparse': self.lambda_sparse,
            'momentum': self.momentum,
            'clip_value': self.clip_value,
            'optimizer_params': self.optimizer_params,
            'mask_type': self.mask_type,
            'seed': self.seed,
            'verbose': self.verbose,
            'device_name': self.device_name,
        }

        # 添加学习率调度器（如果未指定）
        if self.scheduler_fn is None and self.scheduler_params is None:
            import torch
            model_params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
            model_params['scheduler_params'] = {
                'mode': 'min',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1e-5,
                'verbose': True
            }
        else:
            # 只在非None时添加这些参数
            if self.scheduler_fn is not None:
                model_params['scheduler_fn'] = self.scheduler_fn
            if self.scheduler_params is not None:
                model_params['scheduler_params'] = self.scheduler_params

        # 只在非None时添加optimizer_fn
        if self.optimizer_fn is not None:
            model_params['optimizer_fn'] = self.optimizer_fn

        self.model_ = _TabNetRegressor(**model_params)

        # 准备验证集
        if eval_set is None:
            # 自动划分10%作为验证集
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=self.seed
            )
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            # 转换eval_set格式
            if isinstance(eval_set, tuple) and len(eval_set) == 2:
                X_val, y_val = eval_set
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, (pd.Series, pd.DataFrame)):
                    y_val = y_val.values.ravel()
                X_val = np.asarray(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
                eval_set = [(X_val, y_val)]

        # 训练
        self.model_.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_metric=eval_metric or ['rmse'],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )

        # 训练后稳定性检查
        self._validate_training_stability(X_train, y_train)

        return self

    def _validate_training_stability(self, X_sample, y_sample):
        """训练后稳定性检查"""
        import warnings

        # 检查预测是否包含NaN/Inf
        try:
            y_pred = self.model_.predict(X_sample[:min(100, len(X_sample))])

            if np.any(np.isnan(y_pred)):
                warnings.warn(
                    "⚠️ 模型预测包含NaN值！这通常是由于:\n"
                    "1. 学习率过高导致梯度爆炸\n"
                    "2. 数据标准化不当\n"
                    "建议: 降低学习率(当前: {:.5f})或检查数据预处理".format(
                        self.optimizer_params.get('lr', 0.02)
                    ),
                    RuntimeWarning
                )

            if np.any(np.isinf(y_pred)):
                warnings.warn(
                    "⚠️ 模型预测包含Inf值！建议降低学习率或增强正则化",
                    RuntimeWarning
                )

            # 检查训练历史中的异常
            if hasattr(self.model_, 'history'):
                history = self.model_.history
                if 'loss' in history:
                    losses = history['loss']
                    # 检查loss是否全为0（模型崩溃）
                    if len(losses) > 10 and all(l == 0.0 for l in losses[-10:]):
                        warnings.warn(
                            "⚠️ 训练loss全为0，模型完全崩溃！\n"
                            "可能原因:\n"
                            "1. 学习率过高\n"
                            "2. 批次大小不当\n"
                            "3. 数据预处理问题\n"
                            "建议: 降低学习率到1e-3，减小batch_size到128-256",
                            RuntimeWarning
                        )
                    # 检查突然的loss跳变
                    elif len(losses) > 1:
                        for i in range(1, len(losses)):
                            if losses[i] > 0 and losses[i-1] > 0:
                                ratio = abs(losses[i] - losses[i-1]) / losses[i-1]
                                if ratio > 2.0:  # 变化超过200%
                                    warnings.warn(
                                        f"⚠️ Epoch {i}: loss突变 ({losses[i-1]:.2f} → {losses[i]:.2f})\n"
                                        "训练不稳定，建议降低学习率或添加梯度裁剪",
                                        RuntimeWarning
                                    )
                                    break
        except Exception as e:
            warnings.warn(f"稳定性检查失败: {str(e)}", RuntimeWarning)

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("模型未训练，请先调用fit()")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        return self.model_.predict(X).ravel()

    def score(self, X, y):
        y_pred = self.predict(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        return -np.mean((y - y_pred) ** 2)  # 返回负MSE（sklearn约定）

    @property
    def feature_importances_(self):
        """获取特征重要性"""
        if self.model_ is None:
            return None
        return self.model_.feature_importances_
