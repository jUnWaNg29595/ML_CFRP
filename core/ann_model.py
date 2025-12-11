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
                 verbose=0, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        self.model = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.train_losses = []

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if isinstance(y, pd.Series):
            y_values = y.values
        elif isinstance(y, pd.DataFrame):
            y_values = y.values.ravel()
        else:
            y_values = np.array(y).ravel()

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

        return self

    def predict(self, X):
        self.model.eval()
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
