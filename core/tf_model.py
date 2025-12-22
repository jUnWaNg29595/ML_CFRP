# -*- coding: utf-8 -*-
"""
TensorFlow Sequential (TFS) 模型模块

提供基于 TensorFlow/Keras Sequential API 的回归模型，
兼容 scikit-learn 接口，可无缝集成到现有训练流程中。
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# TensorFlow 导入检查
TENSORFLOW_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers

    # 设置 TensorFlow 日志级别
    tf.get_logger().setLevel('ERROR')

    TENSORFLOW_AVAILABLE = True
except Exception as e:
    # 可能的异常：ImportError、DLL 加载失败、CUDA/驱动不匹配等
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = repr(e)
    tf = None
    keras = None
    layers = None
    callbacks = None
    regularizers = None


class TFSequentialRegressor(BaseEstimator, RegressorMixin):
    """
    TensorFlow Sequential 回归模型
    
    基于 Keras Sequential API 构建的全连接神经网络，
    支持自定义网络结构、正则化、早停等功能。
    
    Parameters
    ----------
    hidden_layers : str
        隐藏层结构，格式为逗号分隔的整数，如 "128,64,32"
    activation : str
        激活函数，可选 'relu', 'leaky_relu', 'elu', 'tanh', 'swish'
    dropout_rate : float
        Dropout 比率，范围 [0, 1)
    l2_reg : float
        L2 正则化系数
    optimizer : str
        优化器，可选 'adam', 'sgd', 'rmsprop', 'adamw'
    learning_rate : float
        学习率
    batch_size : int
        批次大小
    epochs : int
        最大训练轮数
    early_stopping : bool
        是否启用早停
    patience : int
        早停耐心值（验证损失不下降的轮数）
    validation_split : float
        验证集比例
    verbose : int
        日志详细程度，0=静默, 1=进度条, 2=每轮一行
    random_state : int
        随机种子
    """
    
    def __init__(
        self,
        hidden_layers="128,64,32",
        activation="relu",
        dropout_rate=0.2,
        l2_reg=0.001,
        optimizer="adam",
        learning_rate=0.001,
        batch_size=32,
        epochs=200,
        early_stopping=True,
        patience=20,
        validation_split=0.1,
        verbose=0,
        random_state=42,
        external_preprocess: bool = False
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        self.verbose = verbose
        self.random_state = random_state
        # 若训练器/外部 Pipeline 已做 imputer + scaler，则可开启此项避免重复预处理
        self.external_preprocess = external_preprocess
        
        # 内部状态
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.imputer_ = SimpleImputer(strategy='mean')
        self.history_ = None
        self.input_dim_ = None
        
    def _parse_hidden_layers(self):
        """解析隐藏层配置字符串"""
        try:
            return [int(x.strip()) for x in str(self.hidden_layers).split(',') if x.strip()]
        except:
            return [128, 64, 32]
    
    def _get_activation(self):
        """获取激活函数"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        activation_map = {
            'relu': 'relu',
            'leaky_relu': layers.LeakyReLU(alpha=0.1),
            'elu': 'elu',
            'tanh': 'tanh',
            'swish': 'swish',
            'selu': 'selu',
            'gelu': 'gelu'
        }
        return activation_map.get(self.activation, 'relu')
    
    def _get_optimizer(self):
        """获取优化器"""
        if not TENSORFLOW_AVAILABLE:
            return None

        # 兼容不同 TensorFlow/Keras 版本（某些版本可能没有 AdamW）
        opt_name = str(self.optimizer).lower()
        if opt_name == 'adamw' and not hasattr(keras.optimizers, 'AdamW'):
            opt_name = 'adam'

        optimizer_map = {
            'adam': keras.optimizers.Adam(learning_rate=self.learning_rate),
            'sgd': keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            'adamw': keras.optimizers.AdamW(learning_rate=self.learning_rate) if hasattr(keras.optimizers, 'AdamW') else keras.optimizers.Adam(learning_rate=self.learning_rate),
            'nadam': keras.optimizers.Nadam(learning_rate=self.learning_rate)
        }
        return optimizer_map.get(opt_name, keras.optimizers.Adam(learning_rate=self.learning_rate))
    
    def _build_model(self, input_dim):
        """构建 Sequential 模型"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow 未安装，无法使用 TFS 模型")
        
        # 设置随机种子
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        hidden_units = self._parse_hidden_layers()
        activation = self._get_activation()
        
        model = keras.Sequential(name="TFS_Regressor")
        
        # 输入层 + 第一个隐藏层
        model.add(layers.Input(shape=(input_dim,), name="input"))
        
        # 批归一化（可选）
        model.add(layers.BatchNormalization(name="bn_input"))
        
        # 隐藏层
        for i, units in enumerate(hidden_units):
            # 全连接层
            model.add(layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                name=f"dense_{i}"
            ))
            
            # 批归一化
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
            
            # 激活函数
            if isinstance(activation, str):
                model.add(layers.Activation(activation, name=f"act_{i}"))
            else:
                model.add(activation)
            
            # Dropout
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))
        
        # 输出层
        model.add(layers.Dense(1, name="output"))
        
        # 编译模型
        model.compile(
            optimizer=self._get_optimizer(),
            loss='mse',
            # 同时记录 MAE/MSE，便于统一绘制训练曲线
            metrics=['mae', 'mse']
        )
        
        return model
    
    def fit(self, X, y):
        """训练模型"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        
        # 预处理：若外部已处理，则跳过内部 imputer/scaler
        if self.external_preprocess:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            # 缺失值填充
            X = self.imputer_.fit_transform(X)
            # 标准化
            X = self.scaler_.fit_transform(X)
        
        self.input_dim_ = X.shape[1]
        
        # 构建模型
        self.model_ = self._build_model(self.input_dim_)
        
        # 回调函数
        callback_list = []
        
        if self.early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss' if self.validation_split > 0 else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
            callback_list.append(early_stop)
        
        # 学习率调度
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if self.validation_split > 0 else 'loss',
            factor=0.5,
            patience=self.patience // 2,
            min_lr=1e-6,
            verbose=0
        )
        callback_list.append(lr_scheduler)
        
        # 训练：若外部训练器提供了 validation_data（通常是 Test 集），优先使用它
        fit_kwargs = {}
        if hasattr(self, 'validation_data') and self.validation_data is not None:
            try:
                X_val, y_val = self.validation_data
                fit_kwargs['validation_data'] = (np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32).ravel())
            except Exception:
                fit_kwargs = {}

        self.history_ = self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=(self.validation_split if (self.validation_split > 0 and 'validation_data' not in fit_kwargs) else None),
            callbacks=callback_list,
            verbose=self.verbose,
            **fit_kwargs
        )
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.model_ is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X, dtype=np.float32)
        
        if self.external_preprocess:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            # 缺失值填充
            X = self.imputer_.transform(X)
            # 标准化
            X = self.scaler_.transform(X)
        
        # 预测
        predictions = self.model_.predict(X, verbose=0)
        
        return predictions.ravel()
    
    def get_training_history(self):
        """获取训练历史"""
        if self.history_ is None:
            return None
        return {
            # loss= MSE（与 metrics['mse'] 可能重复，但绘图侧会自动兼容）
            'loss': self.history_.history.get('loss', []),
            'val_loss': self.history_.history.get('val_loss', []),
            'mae': self.history_.history.get('mae', []),
            'val_mae': self.history_.history.get('val_mae', []),
            'mse': self.history_.history.get('mse', []),
            'val_mse': self.history_.history.get('val_mse', [])
        }
    
    def summary(self):
        """打印模型结构"""
        if self.model_ is not None:
            return self.model_.summary()
        return None
    
    def get_params(self, deep=True):
        """获取参数（sklearn 兼容）"""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'validation_split': self.validation_split,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'external_preprocess': self.external_preprocess
        }
    
    def set_params(self, **params):
        """设置参数（sklearn 兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# 检查 TensorFlow 是否可用的辅助函数
def check_tensorflow_available():
    """检查 TensorFlow 是否可用"""
    return TENSORFLOW_AVAILABLE


def get_tensorflow_version():
    """获取 TensorFlow 版本"""
    if TENSORFLOW_AVAILABLE:
        return tf.__version__
    return None


# TFS 模型的默认参数配置
TFS_DEFAULT_PARAMS = {
    'hidden_layers': "128,64,32",
    'activation': 'relu',
    'dropout_rate': 0.2,
    'l2_reg': 0.001,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping': True,
    'patience': 20,
    'validation_split': 0.1,
    'verbose': 0,
    'random_state': 42
}

# 手动调参界面配置
TFS_TUNING_PARAMS = [
    {
        'name': 'hidden_layers',
        'widget': 'text_input',
        'label': '隐藏层结构 (逗号分隔)',
        'default': "128,64,32",
        'args': {}
    },
    {
        'name': 'activation',
        'widget': 'selectbox',
        'label': '激活函数',
        'default': 'relu',
        'args': {'options': ['relu', 'leaky_relu', 'elu', 'tanh', 'swish', 'selu', 'gelu']}
    },
    {
        'name': 'dropout_rate',
        'widget': 'slider',
        'label': 'Dropout 比率',
        'default': 0.2,
        'args': {'min_value': 0.0, 'max_value': 0.5, 'step': 0.05}
    },
    {
        'name': 'l2_reg',
        'widget': 'number_input',
        'label': 'L2 正则化系数',
        'default': 0.001,
        'args': {'min_value': 0.0, 'max_value': 0.1, 'step': 0.001, 'format': "%.4f"}
    },
    {
        'name': 'optimizer',
        'widget': 'selectbox',
        'label': '优化器',
        'default': 'adam',
        'args': {'options': ['adam', 'adamw', 'sgd', 'rmsprop', 'nadam']}
    },
    {
        'name': 'learning_rate',
        'widget': 'number_input',
        'label': '学习率',
        'default': 0.001,
        'args': {'min_value': 0.0001, 'max_value': 0.1, 'step': 0.0001, 'format': "%.4f"}
    },
    {
        'name': 'batch_size',
        'widget': 'selectbox',
        'label': '批次大小',
        'default': 32,
        'args': {'options': [8, 16, 32, 64, 128, 256]}
    },
    {
        'name': 'epochs',
        'widget': 'slider',
        'label': '最大训练轮数',
        'default': 200,
        'args': {'min_value': 50, 'max_value': 1000, 'step': 50}
    },
    {
        'name': 'early_stopping',
        'widget': 'checkbox',
        'label': '启用早停',
        'default': True,
        'args': {}
    },
    {
        'name': 'patience',
        'widget': 'slider',
        'label': '早停耐心值',
        'default': 20,
        'args': {'min_value': 5, 'max_value': 50, 'step': 5}
    },
    {
        'name': 'validation_split',
        'widget': 'slider',
        'label': '验证集比例',
        'default': 0.1,
        'args': {'min_value': 0.0, 'max_value': 0.3, 'step': 0.05}
    }
]
