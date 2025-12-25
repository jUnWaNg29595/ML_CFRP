# -*- coding: utf-8 -*-
"""人工神经网络模型"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import numpy as np
import pandas as pd


class FFN(nn.Module):
    """前馈神经网络"""

    def __init__(self, input_dim, hidden_layer_sizes_str, activation, dropout_rate):
        super(FFN, self).__init__()

        try:
            hidden_layer_sizes = [int(s.strip()) for s in hidden_layer_sizes_str.split(',')]
        except:
            hidden_layer_sizes = [100, 50]

        layers = []
        act_fn_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU()
        }
        act_fn = act_fn_map.get(activation, nn.ReLU())

        prev_size = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ANNRegressor(BaseEstimator, RegressorMixin):
    """可自定义的人工神经网络回归器"""

    def __init__(self, hidden_layer_sizes="100,50", activation="relu", dropout_rate=0.2,
                 optimizer_name="Adam", learning_rate=1e-3, epochs=100, batch_size=32,
                 verbose=0, random_state=42, external_preprocess: bool = False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        # 若外部 Pipeline 已做缺失填充 + 标准化，可开启此项避免重复预处理
        self.external_preprocess = external_preprocess

        self.model = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.train_losses = []

        # 训练曲线（与系统其它模型统一：Train/Test 的 MAE/MSE）
        self.train_mse_curve = []
        self.test_mse_curve = []
        self.train_mae_curve = []
        self.test_mae_curve = []

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if isinstance(y, pd.Series):
            y_values = y.values
        elif isinstance(y, pd.DataFrame):
            y_values = y.values.ravel()
        else:
            y_values = np.array(y).ravel()

        y_values = np.asarray(y_values, dtype=np.float32).ravel()

        # 预处理：若外部已处理，则跳过内部 imputer/scaler
        if self.external_preprocess:
            X_scaled = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_values).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]
        self.model = FFN(input_dim, self.hidden_layer_sizes, self.activation, self.dropout_rate)

        criterion = nn.MSELoss()
        optimizer_map = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}
        optimizer_class = optimizer_map.get(self.optimizer_name, torch.optim.Adam)
        optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        self.train_losses = []

        # reset metric curves
        self.train_mse_curve = []
        self.test_mse_curve = []
        self.train_mae_curve = []
        self.test_mae_curve = []

        # optional external validation (usually the held-out test set)
        val_X_tensor = None
        val_y = None
        if hasattr(self, 'validation_data') and self.validation_data is not None:
            try:
                Xv, yv = self.validation_data
                yv = np.asarray(yv).ravel().astype(np.float32)
                if self.external_preprocess:
                    Xv_scaled = np.nan_to_num(np.asarray(Xv), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                else:
                    Xv_scaled = self.scaler.transform(self.imputer.transform(Xv))
                val_X_tensor = torch.FloatTensor(Xv_scaled)
                val_y = yv
            except Exception:
                val_X_tensor = None
                val_y = None

        epoch_iter = range(self.epochs)
        if self.verbose > 0:
            epoch_iter = tqdm(epoch_iter, desc="ANN Training")

        for epoch in epoch_iter:
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.train_losses.append(epoch_loss / len(loader))

            # ---- per-epoch metrics (Train/Test MAE & MSE) ----
            self.model.eval()
            with torch.no_grad():
                pred_tr = self.model(X_tensor).cpu().numpy().ravel()
            self.model.train()

            tr_mse = float(np.mean((pred_tr - y_values) ** 2))
            tr_mae = float(np.mean(np.abs(pred_tr - y_values)))
            self.train_mse_curve.append(tr_mse)
            self.train_mae_curve.append(tr_mae)

            if val_X_tensor is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_te = self.model(val_X_tensor).cpu().numpy().ravel()
                self.model.train()
                te_mse = float(np.mean((pred_te - val_y) ** 2))
                te_mae = float(np.mean(np.abs(pred_te - val_y)))
                self.test_mse_curve.append(te_mse)
                self.test_mae_curve.append(te_mae)

        return self

    def predict(self, X):
        self.model.eval()
        if self.external_preprocess:
            X_scaled = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        return predictions

    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        return r2_score(y, y_pred)
