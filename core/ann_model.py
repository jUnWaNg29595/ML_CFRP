# -*- coding: utf-8 -*-
"""人工神经网络模型（支持多 GPU）

说明：
- 该 ANNRegressor 兼容 sklearn API，可被 Pipeline 包装。
- 当 device='auto' 且检测到 CUDA 时，会自动使用 GPU 训练；否则使用 CPU。
- 支持多 GPU 训练（DataParallel）。
- 为了便于 joblib/pickle 保存，训练结束后会把模型权重移回 CPU。
"""

import os
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import numpy as np
import pandas as pd

# [修复] 导入任务管理器的取消功能
try:
    from .task_manager import is_cancelled
    CANCELLATION_AVAILABLE = True
except ImportError:
    CANCELLATION_AVAILABLE = False
    def is_cancelled():
        return False

# [新增] 导入多 GPU 工具
from .multi_gpu_utils import wrap_model_for_multi_gpu, unwrap_data_parallel, get_model_for_inference, print_gpu_info


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_size))
            layers.append(nn.ReLU())
            prev_dim = layer_size
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ANNRegressor(BaseEstimator, RegressorMixin):
    """sklearn 风格的前馈神经网络回归器
    
    支持 GPU 加速和混合精度训练(AMP)。
    """

    def __init__(
        self,
        hidden_layer_sizes_str: str = "256,128,64",
        learning_rate: float = 0.001,
        batch_size: int = 512,  # 增大batch_size以提高GPU利用率
        epochs: int = 100,
        verbose: bool = True,
        random_state: int = 42,
        external_preprocess: bool = False,
        device: str = "auto",
        use_data_parallel: bool = True,  # 新增：是否使用多GPU
    ):
        self.hidden_layer_sizes_str = hidden_layer_sizes_str
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.external_preprocess = external_preprocess
        self.device = device
        self.use_data_parallel = use_data_parallel

        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.train_loss_history = []
        self.test_loss_history = []
        self.validation_data = None  # (X_val, y_val)

        # 记录设备信息（训练时写入）
        self._device_used_ = None
        self._is_data_parallel = False  # 记录是否使用了 DataParallel

    def _select_device(self) -> torch.device:
        dev = (self.device or "auto").lower()
        if dev == "cpu":
            return torch.device("cpu")
        if dev == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # ---- 数据准备 ----
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()

        # 外部 Pipeline 已经做了预处理（imputer+scaler），此处不重复
        if self.external_preprocess:
            X_scaled = np.asarray(X, dtype=np.float32)
            # 防御：把非有限值处理掉，避免训练出现 NaN
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed).astype(np.float32)

        input_dim = X_scaled.shape[1]
        hidden_layer_sizes = list(map(int, self.hidden_layer_sizes_str.split(',')))

        device = self._select_device()
        self._device_used_ = str(device)

        # ---- 构建模型 ----
        self.model = FFN(input_dim, hidden_layer_sizes).to(device)

        # ---- 多GPU支持 ----
        self.model, self._is_data_parallel, gpu_count = wrap_model_for_multi_gpu(
            self.model, device, self.use_data_parallel, verbose=False
        )

        # Tensor（保持在 CPU，batch 时再搬到 GPU，避免 DataLoader + GPU Tensor 的坑）
        X_tensor = torch.from_numpy(X_scaled).float()
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32)).float().view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 动态调整 batch_size：优化 GPU 利用率
        n_samples = X_tensor.shape[0]
        if device.type == "cuda":
            # GPU: 使用更大的 batch_size
            min_batches = 4
            max_batch = max(64, n_samples // min_batches)
            effective_batch_size = min(self.batch_size, max_batch)
            effective_batch_size = max(128, effective_batch_size)  # 至少 128
        else:
            # CPU: 使用较小的 batch_size
            effective_batch_size = min(self.batch_size, max(32, n_samples // 8))
        
        # 根据系统类型配置 num_workers
        # 可通过环境变量 PYTORCH_NUM_WORKERS 覆盖
        if "PYTORCH_NUM_WORKERS" in os.environ:
            try:
                num_workers = max(0, int(os.environ["PYTORCH_NUM_WORKERS"]))
            except ValueError:
                num_workers = 0
        elif platform.system() == 'Windows':
            num_workers = 0
        else:
            cpu_count = os.cpu_count() or 1
            num_workers = min(4, max(2, cpu_count // 4))  # 更保守的默认值
        
        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) if num_workers > 0 else False,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 混合精度训练（AMP）- 提高 GPU 利用率
        use_amp = False
        scaler = None
        if device.type == "cuda":
            try:
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                use_amp = True
            except ImportError:
                pass
        
        # 打印配置信息（始终打印）
        print(f"\n{'='*50}")
        print(f"  [ANN] 训练配置")
        print(f"{'='*50}")
        print_gpu_info(device, gpu_count)
        print(f"  样本数: {n_samples} | Batch大小: {effective_batch_size}")
        print(f"  每epoch迭代: {len(loader)} | 特征维度: {input_dim}")
        print(f"  隐藏层: {hidden_layer_sizes}")
        print(f"  混合精度(AMP): {'已启用 ⚡' if use_amp else '未启用'}")
        if n_samples < 500:
            print(f"\n  ⚠️ 小数据集({n_samples}样本)，GPU利用率低是正常现象")
        print(f"{'='*50}\n")

        # Validation data (optional)
        val_X_tensor = None
        val_y_arr = None
        if self.validation_data is not None:
            val_X, val_y = self.validation_data
            if isinstance(val_X, pd.DataFrame):
                val_X = val_X.values
            if isinstance(val_y, (pd.Series, pd.DataFrame)):
                val_y = np.asarray(val_y).ravel()

            if self.external_preprocess:
                val_X_scaled = np.asarray(val_X, dtype=np.float32)
                val_X_scaled = np.nan_to_num(val_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                val_X_imputed = self.imputer.transform(val_X)
                val_X_scaled = self.scaler.transform(val_X_imputed).astype(np.float32)

            val_X_tensor = torch.from_numpy(val_X_scaled).float()
            val_y_arr = np.asarray(val_y, dtype=np.float32).ravel()

        self.train_loss_history = []
        self.test_loss_history = []

        # ---- 训练 ----
        iterator = range(self.epochs)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Training ANN ({self._device_used_})")

        for epoch in iterator:
            # [修复] 检查是否请求取消
            if is_cancelled():
                print("\n⚠️ 检测到取消请求，正在停止训练...")
                break
                
            self.model.train()
            epoch_loss_accum = torch.tensor(0.0, device=device)
            n_samples_epoch = 0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        output = self.model(batch_X)
                        loss = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # 在GPU上累加，避免每batch同步
                with torch.no_grad():
                    epoch_loss_accum += loss.detach() * len(batch_X)
                n_samples_epoch += len(batch_X)

            # epoch结束时才同步到CPU
            avg_loss = float(epoch_loss_accum.cpu().item()) / max(1, n_samples_epoch)
            self.train_loss_history.append(avg_loss)

            # ---- Validation loss ----
            if val_X_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_te = self.model(val_X_tensor.to(device)).detach().cpu().numpy().ravel()
                    mse_te = float(np.mean((pred_te - val_y_arr) ** 2))
                    self.test_loss_history.append(mse_te)

        # 为了便于序列化，训练结束把模型移回 CPU
        # 如果使用了 DataParallel，需要先提取原始模型
        self.model = unwrap_data_parallel(self.model)
        self._is_data_parallel = False
        self.model.to(torch.device("cpu"))
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet!")

        # 数据准备
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.external_preprocess:
            X_scaled = np.asarray(X, dtype=np.float32)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed).astype(np.float32)

        device = self._select_device()

        # 预测可选在 GPU 上跑；结束后把模型移回 CPU，避免后续导出出现 CUDA 依赖
        self.model.eval()

        # 预测时不使用 DataParallel（单次推理，开销大于收益）
        model_for_pred = get_model_for_inference(self.model, device)

        X_tensor = torch.from_numpy(X_scaled).float().to(device)
        with torch.no_grad():
            pred = model_for_pred(X_tensor).detach().cpu().numpy().ravel()

        # 移回 CPU
        self.model = unwrap_data_parallel(self.model)
        self.model.to(torch.device("cpu"))
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return -np.mean((y - y_pred) ** 2)
