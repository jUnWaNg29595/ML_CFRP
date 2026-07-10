# -*- coding: utf-8 -*-
"""
TensorFlow Sequential (TFS) 模型模块 - 修复版

提供基于 TensorFlow/Keras Sequential API 的回归模型，
兼容 scikit-learn 接口，可无缝集成到现有训练流程中。

修复内容：
- 添加 GPU 回退机制，当 GPU 初始化失败时自动切换到 CPU
- 增强错误处理和日志输出
- 优化内存管理
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import os
import sys
from types import SimpleNamespace

# [修复] 导入任务管理器的取消功能
try:
    from .task_manager import get_keras_cancellation_callback, is_cancelled
    CANCELLATION_AVAILABLE = True
except ImportError:
    CANCELLATION_AVAILABLE = False
    def get_keras_cancellation_callback():
        return None
    def is_cancelled():
        return False

warnings.filterwarnings('ignore')

# ============== 关键修复：GPU 初始化与回退机制 ==============
TENSORFLOW_IMPORT_ERROR = None
GPU_AVAILABLE = False
DEVICE_INFO = "未知"

def _configure_tensorflow():
    """
    配置 TensorFlow，优先使用 GPU，失败时回退到 CPU
    返回: (tf_available, gpu_available, device_info, error_msg)
    """
    global GPU_AVAILABLE, DEVICE_INFO

    # 检查是否在子进程中（避免重复打印）
    import multiprocessing as mp
    is_main_process = mp.current_process().name == 'MainProcess'

    try:
        # 设置 TensorFlow 日志级别（减少噪音）
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # 禁用 oneDNN 自定义操作警告
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        import tensorflow as tf

        # 尝试导入 keras（兼容 TF 2.16+ 独立 Keras 3）
        try:
            from tensorflow import keras
        except ImportError:
            import keras  # Keras 3 独立安装

        try:
            from tensorflow.keras import layers, callbacks, regularizers
        except ImportError:
            from keras import layers, callbacks, regularizers

        tf.get_logger().setLevel('ERROR')

        # 尝试配置 GPU
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # 尝试启用内存增长模式
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        # GPU 可能已被初始化
                        pass

                # 测试 GPU 是否真的可用（通过运行简单计算）
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        _ = tf.matmul(test_tensor, test_tensor)
                    GPU_AVAILABLE = True
                    DEVICE_INFO = f"GPU: {gpus[0].name}"
                    if is_main_process:  # 只在主进程打印
                        print(f"✓ TensorFlow GPU 初始化成功: {DEVICE_INFO}")
                except Exception as gpu_test_error:
                    # GPU 测试失败，回退到 CPU
                    if is_main_process:
                        print(f"⚠ GPU 测试失败，回退到 CPU: {gpu_test_error}")
                    _force_cpu_mode()
                    GPU_AVAILABLE = False
                    DEVICE_INFO = "CPU (GPU 不可用)"
            else:
                GPU_AVAILABLE = False
                DEVICE_INFO = "CPU (未检测到 GPU)"
                # 静默处理，不打印消息（避免干扰用户）

        except Exception as gpu_config_error:
            # GPU 配置失败，强制使用 CPU
            if is_main_process:
                print(f"⚠ GPU 配置失败，强制使用 CPU: {gpu_config_error}")
            _force_cpu_mode()
            GPU_AVAILABLE = False
            DEVICE_INFO = "CPU (GPU 配置失败)"

        return True, GPU_AVAILABLE, DEVICE_INFO, None

    except ImportError as e:
        return False, False, "不可用", f"TensorFlow 未安装: {repr(e)}"
    except Exception as e:
        return False, False, "不可用", repr(e)


def _force_cpu_mode():
    """强制 TensorFlow 使用 CPU 模式"""
    try:
        import tensorflow as tf
        # 方法1：隐藏所有 GPU
        tf.config.set_visible_devices([], 'GPU')
        print("✓ 已强制切换到 CPU 模式")
    except Exception as e:
        # 避免在 TF 已初始化后修改全局 CUDA_VISIBLE_DEVICES，影响其它框架
        print(f"⚠ 无法修改可见 GPU（可能已初始化），将继续使用 CPU 设备上下文: {e}")


# 执行配置
TENSORFLOW_AVAILABLE, GPU_AVAILABLE, DEVICE_INFO, TENSORFLOW_IMPORT_ERROR = _configure_tensorflow()

# 根据配置结果导入 TensorFlow 组件
if TENSORFLOW_AVAILABLE:
    import tensorflow as tf
    # 兼容 Keras 2.x 和 Keras 3.x
    try:
        from tensorflow import keras
    except ImportError:
        import keras  # Keras 3 独立安装
    
    try:
        from tensorflow.keras import layers, callbacks, regularizers
    except ImportError:
        from keras import layers, callbacks, regularizers
else:
    tf = None
    keras = None
    layers = None
    callbacks = None
    regularizers = None


class TFSequentialRegressor(BaseEstimator, RegressorMixin):
    """
    TensorFlow Sequential 回归模型（带 GPU 回退机制）
    
    基于 Keras Sequential API 构建的全连接神经网络，
    支持自定义网络结构、正则化、早停等功能。
    
    当 GPU 初始化失败时，会自动回退到 CPU 训练。
    
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
        优化器，可选 'adam', 'sgd', 'rmsprop', 'adamw', 'nadam', 'ssbroyden', 'ssbfgs'
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
    force_cpu : bool
        是否强制使用 CPU（即使 GPU 可用）
    gradient_clip_norm : float
        梯度裁剪阈值（0 表示关闭）
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
        external_preprocess: bool = False,
        force_cpu: bool = False,
        gradient_clip_norm: float = 1.0
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
        self.external_preprocess = external_preprocess
        self.force_cpu = force_cpu
        self.gradient_clip_norm = gradient_clip_norm
        
        # 内部状态
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.imputer_ = SimpleImputer(strategy='mean')
        self.history_ = None
        self.input_dim_ = None
        self.device_used_ = None  # 记录实际使用的设备
        
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
        """获取优化器（兼容 Keras 2.x 和 Keras 3.x）"""
        if not TENSORFLOW_AVAILABLE:
            return None

        opt_name = str(self.optimizer).lower()
        lr = self.learning_rate
        clipnorm = None
        try:
            clipnorm = float(self.gradient_clip_norm)
        except Exception:
            clipnorm = None
        if clipnorm is not None and clipnorm <= 0:
            clipnorm = None
        
        try:
            # Keras 3.x / TensorFlow 2.16+ 使用 keras.optimizers 直接访问
            # Keras 2.x / TensorFlow <2.16 使用 keras.optimizers.legacy 或直接访问
            
            def _get_adam():
                try:
                    return keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
                except Exception:
                    return keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=clipnorm)
            
            def _get_sgd():
                try:
                    return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=clipnorm)
                except Exception:
                    return keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9, clipnorm=clipnorm)
            
            def _get_rmsprop():
                try:
                    return keras.optimizers.RMSprop(learning_rate=lr, clipnorm=clipnorm)
                except Exception:
                    return keras.optimizers.legacy.RMSprop(learning_rate=lr, clipnorm=clipnorm)
            
            def _get_adamw():
                # AdamW 可能在某些版本中不可用
                try:
                    if hasattr(keras.optimizers, 'AdamW'):
                        return keras.optimizers.AdamW(learning_rate=lr, clipnorm=clipnorm)
                    elif hasattr(keras.optimizers, 'experimental') and hasattr(keras.optimizers.experimental, 'AdamW'):
                        return keras.optimizers.experimental.AdamW(learning_rate=lr, clipnorm=clipnorm)
                except Exception:
                    pass
                # 回退到 Adam
                return _get_adam()
            
            def _get_nadam():
                try:
                    return keras.optimizers.Nadam(learning_rate=lr, clipnorm=clipnorm)
                except Exception:
                    try:
                        return keras.optimizers.legacy.Nadam(learning_rate=lr, clipnorm=clipnorm)
                    except Exception:
                        return _get_adam()

            def _get_ssb_fallback():
                # SSBroyden/SSBFGS 通过 SciPy 训练路径处理，这里回退到 Adam
                print("⚠ SSBroyden/SSBFGS 仅支持 SciPy 拟牛顿训练路径，已回退到 Adam。")
                return _get_adam()
            
            optimizer_map = {
                'adam': _get_adam,
                'sgd': _get_sgd,
                'rmsprop': _get_rmsprop,
                'adamw': _get_adamw,
                'nadam': _get_nadam,
                'ssbroyden': _get_ssb_fallback,
                'ssbfgs': _get_ssb_fallback
            }
            
            get_opt_func = optimizer_map.get(opt_name, _get_adam)
            return get_opt_func()
            
        except Exception as e:
            # 最终回退：使用字符串形式的优化器名称
            print(f"⚠ 优化器创建失败，使用默认 Adam: {e}")
            return 'adam'
    
    def _build_model(self, input_dim, compile_model: bool = True):
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
        
        # 批归一化
        model.add(layers.BatchNormalization(name="bn_input"))
        
        # 隐藏层
        for i, units in enumerate(hidden_units):
            model.add(layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                name=f"dense_{i}"
            ))
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
            
            if isinstance(activation, str):
                model.add(layers.Activation(activation, name=f"act_{i}"))
            else:
                model.add(activation)
            
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))
        
        # 输出层
        model.add(layers.Dense(1, name="output"))
        
        if compile_model:
            # 编译模型
            model.compile(
                optimizer=self._get_optimizer(),
                loss='mse',
                metrics=['mae', 'mse']
            )
        
        return model

    def _auto_adjust_batch_size(self, X) -> None:
        """根据输入维度粗略估算安全 batch_size，避免 GPU/内存 OOM。"""
        try:
            n_samples, input_dim = int(X.shape[0]), int(X.shape[1])
        except Exception:
            return

        hidden_units = self._parse_hidden_layers()
        total_units = input_dim + sum(int(u) for u in hidden_units)
        if total_units <= 0:
            return

        # 估算：每样本激活+梯度的内存占用（保守估计）
        per_sample_bytes = int(total_units * 4 * 6)
        if per_sample_bytes <= 0:
            return

        use_gpu = GPU_AVAILABLE and not self.force_cpu
        budget_mb = 256 if use_gpu else 512
        max_batch = int((budget_mb * 1024 * 1024) / max(per_sample_bytes, 1))
        max_batch = max(8, max_batch)
        if n_samples > 0:
            max_batch = min(max_batch, n_samples)

        try:
            cur_batch = int(self.batch_size)
        except Exception:
            cur_batch = 32

        if max_batch > 0 and cur_batch > max_batch:
            print(f"⚠ 自动降低 batch_size: {cur_batch} -> {max_batch} (防止内存不足)")
            self.batch_size = max_batch

    def _fit_with_quasi_newton(self, X, y, method_name: str):
        """使用 SciPy 拟牛顿/布罗伊登优化器训练（全量批次）。"""
        try:
            from scipy import optimize as sp_opt
        except Exception as exc:
            raise ImportError("SSBroyden/SSBFGS 需要 SciPy (scipy.optimize)") from exc

        X_train = X
        y_train = y
        X_val = None
        y_val = None

        # 优先使用外部 validation_data（训练器会注入测试集）
        if hasattr(self, "validation_data") and self.validation_data is not None:
            try:
                X_val, y_val = self.validation_data
            except Exception:
                X_val, y_val = None, None
        elif self.validation_split and 0 < float(self.validation_split) < 1:
            try:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=float(self.validation_split), random_state=self.random_state
                )
            except Exception:
                X_val, y_val = None, None

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
        if y_val is not None:
            y_val = np.asarray(y_val, dtype=np.float32).ravel()

        # 使用 CPU 避免跨设备变量错误
        with tf.device('/CPU:0'):
            self.model_ = self._build_model(self.input_dim_, compile_model=False)
            X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train_tf = tf.reshape(tf.convert_to_tensor(y_train, dtype=tf.float32), (-1, 1))
            X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32) if X_val is not None else None
            y_val_tf = tf.reshape(tf.convert_to_tensor(y_val, dtype=tf.float32), (-1, 1)) if y_val is not None else None

            mse = tf.keras.losses.MeanSquaredError()
            trainable_vars = self.model_.trainable_variables
            if not trainable_vars:
                raise RuntimeError("模型无可训练参数，无法进行拟牛顿优化")

            shapes = [tuple(v.shape) for v in trainable_vars]
            sizes = [int(np.prod(s)) for s in shapes]

            def _unpack(flat_params):
                arrays = []
                offset = 0
                for size, shape in zip(sizes, shapes):
                    chunk = flat_params[offset:offset + size]
                    arrays.append(chunk.reshape(shape))
                    offset += size
                return arrays

            def _assign(flat_params):
                arrays = _unpack(flat_params)
                for var, arr in zip(trainable_vars, arrays):
                    var.assign(arr)

            last_loss = {"train": None}
            history = {"loss": [], "val_loss": []}
            record_in_grad = (method_name == "ssbroyden")

            def _compute_val_loss():
                if X_val_tf is None or y_val_tf is None:
                    return None
                preds = self.model_(X_val_tf, training=False)
                loss_val = mse(y_val_tf, preds)
                if self.model_.losses:
                    loss_val = loss_val + tf.add_n(self.model_.losses)
                return float(loss_val.numpy())

            def _loss_and_grad(flat_params):
                if is_cancelled():
                    raise KeyboardInterrupt("训练已取消")
                _assign(flat_params)
                with tf.GradientTape() as tape:
                    preds = self.model_(X_train_tf, training=True)
                    loss_val = mse(y_train_tf, preds)
                    if self.model_.losses:
                        loss_val = loss_val + tf.add_n(self.model_.losses)
                grads = tape.gradient(loss_val, trainable_vars)
                grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, trainable_vars)]
                grad_flat = np.concatenate([g.numpy().ravel() for g in grads], axis=0)
                loss_float = float(loss_val.numpy())
                last_loss["train"] = loss_float
                if record_in_grad:
                    history["loss"].append(loss_float)
                    val_loss = _compute_val_loss()
                    if val_loss is not None:
                        history["val_loss"].append(val_loss)
                return loss_float, grad_flat.astype(np.float64)

            def _callback(_xk):
                if is_cancelled():
                    raise KeyboardInterrupt("训练已取消")
                loss_val = last_loss.get("train")
                if loss_val is None:
                    loss_val, _ = _loss_and_grad(_xk)
                history["loss"].append(loss_val)
                val_loss = _compute_val_loss()
                if val_loss is not None:
                    history["val_loss"].append(val_loss)

            init_params = np.concatenate([v.numpy().ravel() for v in trainable_vars], axis=0).astype(np.float64)
            max_iter = int(self.epochs) if self.epochs else 200

            result = None
            if method_name == "ssbroyden":
                try:
                    def _grad_only(flat_params):
                        return _loss_and_grad(flat_params)[1]
                    result = sp_opt.root(
                        _grad_only,
                        init_params,
                        method="broyden1",
                        options={"maxiter": max_iter}
                    )
                except Exception as exc:
                    print(f"⚠ SSBroyden 失败，回退到 BFGS: {exc}")

            if result is None or not getattr(result, "success", True):
                record_in_grad = False
                init_params = np.concatenate([v.numpy().ravel() for v in trainable_vars], axis=0).astype(np.float64)
                try:
                    result = sp_opt.minimize(
                        _loss_and_grad,
                        init_params,
                        jac=True,
                        method="BFGS",
                        callback=_callback,
                        options={"maxiter": max_iter}
                    )
                except Exception as exc:
                    raise RuntimeError(f"拟牛顿优化失败: {exc}") from exc

            final_params = getattr(result, "x", init_params)
            _assign(final_params)

            if not history["loss"] and last_loss.get("train") is not None:
                history["loss"].append(last_loss["train"])
                val_loss = _compute_val_loss()
                if val_loss is not None:
                    history["val_loss"].append(val_loss)

        self.history_ = SimpleNamespace(history=history)
        self.device_used_ = f"CPU ({method_name.upper()})"
        return self
    
    def fit(self, X, y):
        """训练模型（带 GPU 回退机制）"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                f"TensorFlow 未安装或初始化失败，请检查环境。\n"
                f"错误信息: {TENSORFLOW_IMPORT_ERROR}\n"
                f"解决方案:\n"
                f"1. pip install tensorflow --upgrade\n"
                f"2. 检查 CUDA/cuDNN 版本兼容性\n"
                f"3. 尝试使用 CPU 版本: pip install tensorflow-cpu"
            )
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        
        if self.external_preprocess:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            X = self.imputer_.fit_transform(X)
            X = self.scaler_.fit_transform(X)
            X = np.asarray(X, dtype=np.float32)

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        self.input_dim_ = X.shape[1]
        self._auto_adjust_batch_size(X)

        opt_name = str(self.optimizer).lower()
        if opt_name in {"ssbroyden", "ssbfgs"}:
            try:
                return self._fit_with_quasi_newton(X, y, opt_name)
            except Exception as exc:
                print(f"⚠ {opt_name} 拟牛顿训练失败，回退到 Adam: {exc}")
        
        # ============== 关键修复：带回退机制的训练 ==============
        def _do_training(device_context=None):
            """执行实际训练"""
            self.model_ = self._build_model(self.input_dim_)
            
            callback_list = []
            if self.early_stopping:
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss' if self.validation_split > 0 else 'loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0
                )
                callback_list.append(early_stop)
            
            lr_scheduler = callbacks.ReduceLROnPlateau(
                monitor='val_loss' if self.validation_split > 0 else 'loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-6,
                verbose=0
            )
            callback_list.append(lr_scheduler)
            
            # [修复] 添加取消回调
            cancel_callback = get_keras_cancellation_callback()
            if cancel_callback is not None:
                callback_list.append(cancel_callback)
            
            fit_kwargs = {}
            if hasattr(self, 'validation_data') and self.validation_data is not None:
                try:
                    X_val, y_val = self.validation_data
                    fit_kwargs['validation_data'] = (
                        np.asarray(X_val, dtype=np.float32),
                        np.asarray(y_val, dtype=np.float32).ravel()
                    )
                except Exception:
                    pass
            
            self.history_ = self.model_.fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=(self.validation_split if (self.validation_split > 0 and 'validation_data' not in fit_kwargs) else None),
                callbacks=callback_list,
                verbose=self.verbose,
                **fit_kwargs
            )
        
        # 决定使用的设备
        use_gpu = GPU_AVAILABLE and not self.force_cpu
        
        if use_gpu:
            # 尝试 GPU 训练，失败则回退到 CPU
            try:
                with tf.device('/GPU:0'):
                    _do_training()
                self.device_used_ = "GPU"
                print(f"✓ 训练完成 (GPU)")
            except Exception as gpu_error:
                print(f"⚠ GPU 训练失败: {gpu_error}")
                print("→ 自动回退到 CPU 模式...")
                
                # 清理 GPU 状态
                try:
                    tf.keras.backend.clear_session()
                except:
                    pass
                
                # 强制 CPU 模式
                _force_cpu_mode()
                self.force_cpu = True
                
                # 使用 CPU 重试
                try:
                    with tf.device('/CPU:0'):
                        _do_training()
                    self.device_used_ = "CPU (GPU 回退)"
                    print(f"✓ 训练完成 (CPU 回退模式)")
                except Exception as cpu_error:
                    raise RuntimeError(
                        f"训练失败（GPU 和 CPU 均失败）:\n"
                        f"GPU 错误: {gpu_error}\n"
                        f"CPU 错误: {cpu_error}\n"
                        f"建议: 检查 TensorFlow 安装或数据格式"
                    )
        else:
            # 直接使用 CPU
            try:
                with tf.device('/CPU:0'):
                    _do_training()
                self.device_used_ = "CPU"
                print(f"✓ 训练完成 (CPU)")
            except Exception as cpu_error:
                raise RuntimeError(f"CPU 训练失败: {cpu_error}")
        
        return self

    def _select_predict_device(self):
        """选择预测设备，避免 CPU/GPU 变量混用导致的图执行错误。"""
        if self.force_cpu:
            return "/CPU:0"
        if self.device_used_:
            if "CPU" in str(self.device_used_).upper():
                return "/CPU:0"
            if "GPU" in str(self.device_used_).upper():
                return "/GPU:0"
        try:
            for v in self.model_.variables:
                dev = getattr(v, "device", "")
                if not dev:
                    continue
                if "GPU" in dev.upper():
                    return "/GPU:0"
                if "CPU" in dev.upper():
                    return "/CPU:0"
        except Exception:
            pass
        return "/GPU:0" if GPU_AVAILABLE else "/CPU:0"
    
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
            X = self.imputer_.transform(X)
            X = self.scaler_.transform(X)
        
        device = self._select_predict_device()
        try:
            with tf.device(device):
                predictions = self.model_.predict(X, verbose=0)
            return predictions.ravel()
        except Exception as gpu_error:
            if device != "/CPU:0":
                try:
                    with tf.device('/CPU:0'):
                        predictions = self.model_.predict(X, verbose=0)
                    self.device_used_ = "CPU (预测回退)"
                    self.force_cpu = True
                    return predictions.ravel()
                except Exception:
                    raise gpu_error
            raise
    
    def get_training_history(self):
        """获取训练历史"""
        if self.history_ is None:
            return None
        return {
            'loss': self.history_.history.get('loss', []),
            'val_loss': self.history_.history.get('val_loss', []),
            'mae': self.history_.history.get('mae', []),
            'val_mae': self.history_.history.get('val_mae', []),
            'mse': self.history_.history.get('mse', []),
            'val_mse': self.history_.history.get('val_mse', [])
        }
    
    def get_device_info(self):
        """获取设备信息"""
        return {
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'device_info': DEVICE_INFO,
            'device_used': self.device_used_,
            'force_cpu': self.force_cpu
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
            'external_preprocess': self.external_preprocess,
            'force_cpu': self.force_cpu,
            'gradient_clip_norm': self.gradient_clip_norm
        }
    
    def set_params(self, **params):
        """设置参数（sklearn 兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# 辅助函数
def check_tensorflow_available():
    """检查 TensorFlow 是否可用"""
    return TENSORFLOW_AVAILABLE


def get_tensorflow_version():
    """获取 TensorFlow 版本"""
    if TENSORFLOW_AVAILABLE:
        return tf.__version__
    return None


def get_device_status():
    """获取完整的设备状态信息"""
    return {
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'tensorflow_version': get_tensorflow_version(),
        'gpu_available': GPU_AVAILABLE,
        'device_info': DEVICE_INFO,
        'import_error': TENSORFLOW_IMPORT_ERROR
    }


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
    'random_state': 42,
    'force_cpu': False,
    'gradient_clip_norm': 1.0
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
        'args': {'options': ['adam', 'adamw', 'sgd', 'rmsprop', 'nadam', 'ssbfgs', 'ssbroyden']},
        'help': 'ssbfgs/ssbroyden 使用 SciPy 拟牛顿法，适合小数据集/小网络'
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
    },
    {
        'name': 'force_cpu',
        'widget': 'checkbox',
        'label': '强制使用 CPU',
        'default': False,
        'args': {}
    },
    {
        'name': 'gradient_clip_norm',
        'widget': 'number_input',
        'label': '梯度裁剪 (clipnorm, 0=关闭)',
        'default': 1.0,
        'args': {'min_value': 0.0, 'max_value': 5.0, 'step': 0.1, 'format': "%.2f"}
    }
]
