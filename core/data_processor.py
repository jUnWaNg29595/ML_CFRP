# -*- coding: utf-8 -*-
"""数据处理模块"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, latent_dim=16, h_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(h_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_dim // 2), nn.ReLU(),
            nn.Linear(h_dim // 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, input_dim), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class DataEnhancer:
    """数据增强器"""
    def __init__(self, data: pd.DataFrame):
        self.original_data = data
        self.numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    def knn_impute(self, n_neighbors=5):
        data_copy = self.original_data.copy()
        if self.numeric_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            data_copy[self.numeric_cols] = imputer.fit_transform(data_copy[self.numeric_cols])
        return data_copy

    def generate_with_vae(self, n_samples, latent_dim=16, h_dim=128, epochs=100, batch_size=32, lr=1e-3):
        df_numeric = self.original_data[self.numeric_cols].dropna()
        if df_numeric.empty:
            raise ValueError("没有可用数据")

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_numeric)

        data_tensor = torch.FloatTensor(data_scaled)
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

        model = VAE(data_scaled.shape[1], latent_dim, h_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in tqdm(range(epochs), desc="VAE Training"):
            for (data,) in loader:
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss = nn.functional.mse_loss(recon, data, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            generated = model.decoder(z).numpy()

        generated_df = pd.DataFrame(scaler.inverse_transform(generated), columns=self.numeric_cols)

        # PCA可视化
        pca = PCA(n_components=2)
        orig_pca = pca.fit_transform(data_scaled)
        gen_pca = pca.transform(scaler.transform(generated_df))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=orig_pca[:, 0], y=orig_pca[:, 1], mode='markers', name='原始', opacity=0.6))
        fig.add_trace(go.Scatter(x=gen_pca[:, 0], y=gen_pca[:, 1], mode='markers', name='生成', opacity=0.6))
        fig.update_layout(title='PCA: 原始 vs 生成', xaxis_title='PC1', yaxis_title='PC2')

        return generated_df, fig


class SparseDataHandler:
    """稀疏数据处理器"""
    def __init__(self, data: pd.DataFrame, threshold=0.3):
        self.data = data
        self.threshold = threshold
        self.numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    def analyze_sparsity(self):
        return {col: {'non_null_ratio': self.data[col].notna().mean()} for col in self.numeric_cols}


class AdvancedDataCleaner:
    """数据清洗器"""
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.cleaned_data = data.copy()
        self.cleaning_log = []

    def detect_pseudo_numeric_columns(self):
        pseudo = {}
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].dtype == 'object':
                converted = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                orig_count = self.cleaned_data[col].notna().sum()
                conv_count = converted.notna().sum()
                if orig_count > 0 and conv_count / orig_count >= 0.5:
                    pseudo[col] = {'转换成功率': conv_count / orig_count}
        return pseudo

    def fix_pseudo_numeric_columns(self):
        for col in self.detect_pseudo_numeric_columns():
            self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
        return self.cleaned_data

    def handle_missing_values(self, strategy='median', fill_value=None):
        numeric_cols = self.cleaned_data.select_dtypes(include=np.number).columns
        if strategy == 'median':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(self.cleaned_data[numeric_cols].median())
        elif strategy == 'mean':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(self.cleaned_data[numeric_cols].mean())
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.cleaned_data[numeric_cols] = imputer.fit_transform(self.cleaned_data[numeric_cols])
        elif strategy == 'drop_rows':
            self.cleaned_data = self.cleaned_data.dropna()
        elif strategy == 'constant':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(fill_value or 0)
        return self.cleaned_data

    def detect_outliers(self, method='iqr', threshold=1.5):
        outliers = {}
        for col in self.cleaned_data.select_dtypes(include=np.number).columns:
            data = self.cleaned_data[col].dropna()
            if len(data) == 0:
                continue
            if method == 'iqr':
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                count = ((self.cleaned_data[col] < Q1 - threshold * IQR) | (self.cleaned_data[col] > Q3 + threshold * IQR)).sum()
            else:
                count = (np.abs(stats.zscore(data)) > threshold).sum()
            if count > 0:
                outliers[col] = {'异常值数量': int(count)}
        return outliers

    def handle_outliers(self, method='clip', threshold=1.5):
        for col in self.cleaned_data.select_dtypes(include=np.number).columns:
            data = self.cleaned_data[col].dropna()
            if len(data) == 0:
                continue
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            if method == 'clip':
                self.cleaned_data[col] = self.cleaned_data[col].clip(lower, upper)
            elif method == 'replace_median':
                mask = (self.cleaned_data[col] < lower) | (self.cleaned_data[col] > upper)
                self.cleaned_data.loc[mask, col] = data.median()
        return self.cleaned_data

    def remove_duplicates(self):
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        return self.cleaned_data
