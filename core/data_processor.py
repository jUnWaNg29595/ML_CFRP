# -*- coding: utf-8 -*-
"""数据处理模块"""

import pandas as pd
import numpy as np

# [新增] 解析带单位/符号的数值（用于 epoxy/复合材料数据清洗）
try:
    from .epoxy_physics import parse_first_number
except Exception:
    parse_first_number = None
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA
from scipy import stats
import plotly.graph_objects as go

# 导入快速PCA优化器
try:
    from .fast_pca import fast_pca_transform
    FAST_PCA_AVAILABLE = True
except ImportError:
    try:
        from fast_pca import fast_pca_transform
        FAST_PCA_AVAILABLE = True
    except ImportError:
        FAST_PCA_AVAILABLE = False
        fast_pca_transform = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    # 在某些环境（尤其是缺少 CUDA 运行库或内存受限）torch 可能导入失败。
    # 本平台除非使用 VAE 等深度学习增强功能，否则不依赖 torch。
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
try:
    from core.smiles_utils import (
        smart_repair_smiles, 
        canonicalize_smiles,
        aggressive_repair_smiles,
        ultra_repair_smiles,
        hybrid_repair_smiles,
        transformer_repair_smiles,
        batch_transformer_repair,
        clean_smiles_column_with_transformer,
        get_correction_pipeline
    )
    SMILES_TRANSFORMER_AVAILABLE = True
except ImportError:
    # 简单的 fallback，防止导入错误导致整个模块崩溃
    smart_repair_smiles = lambda x, k: x
    canonicalize_smiles = lambda x: x
    aggressive_repair_smiles = None
    ultra_repair_smiles = None
    hybrid_repair_smiles = None
    transformer_repair_smiles = None
    batch_transformer_repair = None
    clean_smiles_column_with_transformer = None
    get_correction_pipeline = None
    SMILES_TRANSFORMER_AVAILABLE = False

if TORCH_AVAILABLE:
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

else:
    class VAE:
        """torch 不可用时的占位 VAE"""

        def __init__(self, *args, **kwargs):
            raise ImportError("torch 不可用，无法使用 VAE 数据增强功能。请安装/修复 torch，或关闭相关功能。")

class DataEnhancer:
    """数据增强器"""

    def __init__(self, data: pd.DataFrame):
        self.original_data = data
        self.numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    def knn_impute(self, n_neighbors=5, columns=None):
        """KNN 智能填充

        Args:
            n_neighbors: K近邻数量
            columns: 要填充的列名列表，None 表示填充所有数值列

        Returns:
            填充后的 DataFrame
        """
        data_copy = self.original_data.copy()

        # 确定要填充的列
        if columns is None:
            cols_to_impute = self.numeric_cols
        else:
            # 验证列名是否存在且为数值列
            cols_to_impute = []
            for col in columns:
                if col not in data_copy.columns:
                    raise ValueError(f"列 '{col}' 不存在")
                if col not in self.numeric_cols:
                    raise ValueError(f"列 '{col}' 不是数值列，无法使用 KNN 填充")
                cols_to_impute.append(col)

        if cols_to_impute:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            data_copy[cols_to_impute] = imputer.fit_transform(data_copy[cols_to_impute])

        return data_copy

    def generate_with_vae(self, n_samples, latent_dim=16, h_dim=128, epochs=100, batch_size=32, lr=1e-3):
        df_numeric = self.original_data[self.numeric_cols].dropna()
        if df_numeric.empty:
            raise ValueError("没有可用数据")

        # 检测GPU可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_numeric)

        data_tensor = torch.FloatTensor(data_scaled)
        loader = DataLoader(
            TensorDataset(data_tensor), 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=(device.type == 'cuda'),
        )

        model = VAE(data_scaled.shape[1], latent_dim, h_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in tqdm(range(epochs), desc=f"VAE Training ({device})"):
            for (data,) in loader:
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss = nn.functional.mse_loss(recon, data, reduction='sum') - 0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp())
                loss.backward()
                optimizer.step()

        model.eval()
        model.to('cpu')  # 移回CPU用于生成
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            generated = model.decoder(z).numpy()

        generated_df = pd.DataFrame(scaler.inverse_transform(generated), columns=self.numeric_cols)

        # PCA可视化 - 使用快速PCA
        if FAST_PCA_AVAILABLE and data_scaled.shape[1] > 100:
            # 大特征数使用快速PCA
            orig_pca, _ = fast_pca_transform(data_scaled, variance_threshold=0.95, min_components=2)
            orig_pca = orig_pca[:, :2]  # 只取前2个主成分
            gen_scaled = scaler.transform(generated_df)
            gen_pca, _ = fast_pca_transform(gen_scaled, variance_threshold=0.95, min_components=2)
            gen_pca = gen_pca[:, :2]
        else:
            # 小特征数使用标准PCA
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
        # [修复] 预先识别并保护 SMILES 列，避免被误处理
        self._smiles_columns = self._detect_smiles_columns()
    
    def _detect_smiles_columns(self) -> list:
        """
        智能检测 SMILES 列（基于列名和内容启发式判断）
        返回可能包含 SMILES 的列名列表
        """
        smiles_cols = []
        for col in self.cleaned_data.columns:
            # 1. 基于列名启发式判断
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['smiles', 'smi', 'molecule', 'structure']):
                smiles_cols.append(col)
                continue
            
            # 2. 基于内容启发式判断（仅检查 object 类型列）
            if self.cleaned_data[col].dtype == 'object':
                # 检查前 20 个非空值是否看起来像 SMILES
                sample = self.cleaned_data[col].dropna().head(20).astype(str)
                if len(sample) > 0:
                    smiles_like_count = 0
                    for val in sample:
                        # SMILES 通常包含这些字符组合
                        if any(c in val for c in ['C', 'c', 'N', 'O', '=', '#', '(', ')', '[', ']']):
                            # 排除明显的非 SMILES（如纯数字、URL等）
                            if not val.replace('.', '').replace('-', '').isdigit() and 'http' not in val.lower():
                                smiles_like_count += 1
                    # 如果超过 50% 的样本看起来像 SMILES
                    if len(sample) > 0 and smiles_like_count / len(sample) > 0.5:
                        smiles_cols.append(col)
        
        return smiles_cols
    
    def get_smiles_columns(self) -> list:
        """获取已识别的 SMILES 列"""
        return self._smiles_columns.copy()
    
    def refresh_smiles_columns(self):
        """重新检测 SMILES 列（在数据结构变化后调用）"""
        self._smiles_columns = self._detect_smiles_columns()
        return self._smiles_columns

    def detect_pseudo_numeric_columns(self):
        """检测"看起来像数字"的 object 列。

        兼容两类情况：
        1) 纯数字字符串：pd.to_numeric 可直接转换
        2) 带单位/符号：如 '80 wt%', '25°C', '1.2e3 MPa' —— 通过正则抽取首个数字转换
        
        [修复] 自动排除 SMILES 列，避免将分子结构误转为数值
        """
        pseudo = {}
        # 获取需要保护的 SMILES 列
        protected_cols = set(self._smiles_columns) if hasattr(self, '_smiles_columns') else set()
        
        for col in self.cleaned_data.columns:
            # [修复] 跳过 SMILES 列
            if col in protected_cols:
                continue
            
            if self.cleaned_data[col].dtype == 'object':
                s = self.cleaned_data[col]
                converted = pd.to_numeric(s, errors='coerce')
                orig_count = s.notna().sum()
                conv_count = converted.notna().sum()
                if orig_count > 0 and conv_count / orig_count >= 0.5:
                    pseudo[col] = {'转换成功率': conv_count / orig_count, 'method': 'direct'}
                    continue

                if parse_first_number is not None and orig_count > 0:
                    sample = s.dropna().astype(str).head(200)
                    if len(sample) == 0:
                        continue
                    parsed = sample.apply(parse_first_number)
                    parsed_ok = np.isfinite(parsed).sum()
                    ratio = parsed_ok / max(1, len(sample))
                    if ratio >= 0.5:
                        pseudo[col] = {'转换成功率': ratio, 'method': 'regex_number'}
        return pseudo

    def fix_pseudo_numeric_columns(self):
        """将 detect_pseudo_numeric_columns 检出的列转为数值。"""
        pseudo = self.detect_pseudo_numeric_columns()
        for col, info in pseudo.items():
            method = info.get('method', 'direct')
            if method == 'direct':
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
            elif method == 'regex_number' and parse_first_number is not None:
                self.cleaned_data[col] = self.cleaned_data[col].apply(parse_first_number)
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
            else:
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
        return self.cleaned_data

    def handle_infinite_values(self, strategy='to_nan', replace_value=None):
        """
        处理无穷大值（inf和-inf）

        Parameters:
        -----------
        strategy : str
            处理策略：
            - 'to_nan': 转换为NaN（默认）
            - 'replace': 替换为指定值
            - 'drop_rows': 删除包含无穷大值的行
            - 'clip': 裁剪到列的最大/最小有限值
        replace_value : float, optional
            当strategy='replace'时使用的替换值

        Returns:
        --------
        pd.DataFrame
            处理后的数据
        """
        numeric_cols = self.cleaned_data.select_dtypes(include=np.number).columns

        if strategy == 'to_nan':
            # 将无穷大值替换为NaN
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
            self.cleaning_log.append(f"将无穷大值转换为NaN")

        elif strategy == 'replace':
            # 替换为指定值
            if replace_value is None:
                replace_value = 0
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].replace([np.inf, -np.inf], replace_value)
            self.cleaning_log.append(f"将无穷大值替换为 {replace_value}")

        elif strategy == 'drop_rows':
            # 删除包含无穷大值的行
            mask = np.isinf(self.cleaned_data[numeric_cols]).any(axis=1)
            rows_before = len(self.cleaned_data)
            self.cleaned_data = self.cleaned_data[~mask].reset_index(drop=True)
            rows_removed = rows_before - len(self.cleaned_data)
            self.cleaning_log.append(f"删除包含无穷大值的 {rows_removed} 行")

        elif strategy == 'clip':
            # 裁剪到列的最大/最小有限值
            for col in numeric_cols:
                finite_values = self.cleaned_data[col][np.isfinite(self.cleaned_data[col])]
                if len(finite_values) > 0:
                    min_val = finite_values.min()
                    max_val = finite_values.max()
                    # 将正无穷替换为最大有限值，负无穷替换为最小有限值
                    self.cleaned_data[col] = self.cleaned_data[col].replace(np.inf, max_val)
                    self.cleaned_data[col] = self.cleaned_data[col].replace(-np.inf, min_val)
            self.cleaning_log.append(f"将无穷大值裁剪到有限值范围")

        return self.cleaned_data

    def handle_missing_values(self, strategy='median', fill_value=None):
        numeric_cols = self.cleaned_data.select_dtypes(include=np.number).columns

        if strategy == 'median':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                self.cleaned_data[numeric_cols].median())

        elif strategy == 'mean':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                self.cleaned_data[numeric_cols].mean())

        elif strategy == 'mode':
            # 使用众数填充（对离散型数值/计数特征更友好）
            try:
                mode_vals = self.cleaned_data[numeric_cols].mode().iloc[0]
                self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(mode_vals)
            except Exception:
                # 如果 mode 计算失败，回退到中位数
                self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                    self.cleaned_data[numeric_cols].median())

        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.cleaned_data[numeric_cols] = imputer.fit_transform(self.cleaned_data[numeric_cols])

        elif strategy == 'drop_rows':
            self.cleaned_data = self.cleaned_data.dropna()

        elif strategy == 'constant':
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(fill_value or 0)

        return self.cleaned_data

    def detect_outliers(self, method='iqr', threshold=1.5, columns=None, min_outlier_ratio=0.01):
        """
        检测异常值

        Parameters:
        -----------
        method : str
            检测方法，'iqr' 或 'zscore'
        threshold : float
            阈值（IQR 倍数或 Z-score 标准差倍数）
        columns : list, optional
            要检测的列，None 表示所有数值列
        min_outlier_ratio : float
            最小异常值比例，低于此比例不报告（默认 1%）
        """
        outliers = {}
        numeric_cols = self.cleaned_data.select_dtypes(include=np.number).columns.tolist()
        if columns is not None:
            numeric_cols = [c for c in columns if c in numeric_cols]
        for col in numeric_cols:
            data = self.cleaned_data[col].dropna()
            if len(data) == 0:
                continue
            if method == 'iqr':
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:  # 避免除零
                    continue
                count = ((self.cleaned_data[col] < Q1 - threshold * IQR) | (
                            self.cleaned_data[col] > Q3 + threshold * IQR)).sum()
            else:  # zscore
                mean_val = data.mean()
                std_val = data.std()
                if std_val == 0:  # 避免除零
                    continue
                z_scores = np.abs((data - mean_val) / std_val)
                count = (z_scores > threshold).sum()

            # 只报告异常值比例超过阈值的列
            outlier_ratio = count / len(data) if len(data) > 0 else 0
            if count > 0 and outlier_ratio >= min_outlier_ratio:
                outliers[col] = {
                    '异常值数量': int(count),
                    '异常值比例': f'{outlier_ratio:.2%}',
                    '总样本数': len(data)
                }
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

    def detect_high_repetition_columns(self, threshold=0.8):
        """检测高重复率列"""
        high_rep_cols = {}
        for col in self.cleaned_data.columns:
            if len(self.cleaned_data) == 0: continue
            # 计算众数出现的频率
            try:
                value_counts = self.cleaned_data[col].value_counts(normalize=True)
                if not value_counts.empty:
                    max_freq = value_counts.iloc[0]
                    if max_freq >= threshold:
                        high_rep_cols[col] = {
                            'most_frequent_value': value_counts.index[0],
                            'frequency': max_freq
                        }
            except:
                pass
        return high_rep_cols

    def reduce_feature_repetition(self, column, target_rate=0.5):
        """降低特定特征的重复率（通过删除众数样本）"""
        if column not in self.cleaned_data.columns:
            return self.cleaned_data

        df = self.cleaned_data
        value_counts = df[column].value_counts()
        if value_counts.empty:
            return df

        most_freq_val = value_counts.index[0]
        current_count = value_counts.iloc[0]
        total_count = len(df)
        other_count = total_count - current_count

        # 目标：让 most_freq_val / (other_count + new_most_freq_count) = target_rate
        # 推导：new_most_freq_count = target_rate * other_count / (1 - target_rate)

        if target_rate >= 1.0 or target_rate <= 0:
            return df

        if other_count == 0:
            # 如果全是重复值，为了达到比例，只能直接采样保留一定比例的行
            return df.sample(frac=target_rate, random_state=42).reset_index(drop=True)

        desired_count = int(target_rate * other_count / (1 - target_rate))

        if desired_count >= current_count:
            # 当前比例已经低于目标，无需操作
            return df

        # 1. 找出众数行的索引
        # 处理 NaN 的情况
        if pd.isna(most_freq_val):
            mask_most_freq = df[column].isna()
        else:
            mask_most_freq = df[column] == most_freq_val

        most_freq_indices = df[mask_most_freq].index

        # 2. 找出非众数行的索引
        other_indices = df[~mask_most_freq].index

        # 3. 对众数行进行随机降采样
        keep_indices = np.random.choice(most_freq_indices, size=desired_count, replace=False)

        # 4. 合并索引并重构数据
        final_indices = np.concatenate([other_indices, keep_indices])
        # 保持原有顺序（可选）
        final_indices.sort()

        self.cleaned_data = df.loc[final_indices].reset_index(drop=True)
        return self.cleaned_data

    def balance_category_counts(self, column, max_samples=None):
        """
        平衡类别计数：强制限制某一列（如SMILES）中每个类别的最大样本数。
        这有助于解决数据集中某些单体重复次数过多，导致模型过拟合的问题。

        Args:
            column: 要平衡的列名
            max_samples: 每个类别的最大样本数。如果为None，则不做处理。
        """
        if column not in self.cleaned_data.columns or max_samples is None:
            return self.cleaned_data

        df = self.cleaned_data

        # 获取该列的所有值计数
        value_counts = df[column].value_counts()

        indices_to_keep = []

        # 遍历每个唯一的类别值
        for val, count in value_counts.items():
            # 处理 NaN 值
            if pd.isna(val):
                mask = df[column].isna()
            else:
                mask = df[column] == val

            # 获取该类别所有的索引
            group_indices = df[mask].index.tolist()

            # 如果该类别的样本数超过最大限制，则随机抽样
            if count > max_samples:
                selected_indices = np.random.choice(group_indices, size=max_samples, replace=False)
                indices_to_keep.extend(selected_indices)
            else:
                # 否则保留所有样本
                indices_to_keep.extend(group_indices)

        # 排序索引以保持原始数据的相对顺序（如果重要）
        indices_to_keep = sorted(indices_to_keep)

        # 更新数据
        self.cleaned_data = df.loc[indices_to_keep].reset_index(drop=True)
        return self.cleaned_data
    def aggregate_by_keys(self, keys, target_col, agg: str = 'median', dropna_target: bool = True):
        """按配方/键聚合重复记录(用于 Tg/力学等性质的稳健建模)

        - 默认对 target_col 做 median 聚合，抗异常值更强
        - 同时生成重复次数与标准差：{prefix}_rep_n / {prefix}_rep_std
        - 其他数值列按 median 聚合(抗异常值)，类别列取众数

        Args:
            keys: 分组键列名列表
            target_col: 需要聚合的目标列名
            agg: 聚合方法：median/mean/min/max
            dropna_target: 是否删除聚合后 target 仍为 NaN 的组

        Returns:
            聚合后的 DataFrame
        """
        df = self.cleaned_data.copy()

        # 1) 参数检查
        if not isinstance(keys, (list, tuple)):
            raise ValueError("keys 必须是列名列表")
        keys = [k for k in keys if k in df.columns]
        if len(keys) == 0:
            raise ValueError("聚合键 keys 为空或不在数据列中")

        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")

        # 2) 强制目标列转数值（无法转换的变 NaN）
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # 3) 分组
        grouped = df.groupby(keys, dropna=False, sort=False)

        # 4) 聚合目标
        agg = (agg or 'median').lower()
        if agg == 'median':
            target_agg = grouped[target_col].median()
        elif agg == 'mean':
            target_agg = grouped[target_col].mean()
        elif agg == 'min':
            target_agg = grouped[target_col].min()
        elif agg == 'max':
            target_agg = grouped[target_col].max()
        else:
            raise ValueError("agg 仅支持 median/mean/min/max")

        rep_n = grouped[target_col].count()
        rep_std = grouped[target_col].std()  # ddof=1

        # 5) 按列类型智能聚合（数值列 median，类别列众数）
        other_cols = [c for c in df.columns if c not in keys and c != target_col]
        numeric_other = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_other = [c for c in other_cols if c not in numeric_other]

        def _safe_mode(s):
            """取众数，全为空或众数不唯一时取 first"""
            m = s.mode()
            return m.iloc[0] if len(m) > 0 else s.iloc[0]

        agg_dict = {}
        for c in numeric_other:
            agg_dict[c] = 'median'
        for c in cat_other:
            agg_dict[c] = _safe_mode

        if agg_dict:
            base = grouped.agg(agg_dict).reset_index()
        else:
            base = grouped.first().reset_index()

        # 6) 写回聚合后的目标与统计量
        base[target_col] = target_agg.values

        # column name: tg_c -> tg_rep_n / tg_rep_std；其他目标用 <target>_rep_n / _rep_std
        target_lower = str(target_col).lower()
        prefix = 'tg' if target_lower in ['tg', 'tg_c'] or target_lower.startswith('tg') else str(target_col)
        rep_n_col = f"{prefix}_rep_n"
        rep_std_col = f"{prefix}_rep_std"

        # 避免覆盖已有列
        if rep_n_col in base.columns:
            rep_n_col = rep_n_col + "_agg"
        if rep_std_col in base.columns:
            rep_std_col = rep_std_col + "_agg"

        base[rep_n_col] = rep_n.values.astype(int)
        base[rep_std_col] = rep_std.values

        # 7) 可选：删除 target 为 NaN 的组
        if dropna_target:
            base = base[base[target_col].notna()].reset_index(drop=True)

        # 8) 更新状态
        before = len(self.cleaned_data)
        after = len(base)
        self.cleaned_data = base
        self.cleaning_log.append({
            'action': 'aggregate_by_keys',
            'keys': keys,
            'target_col': target_col,
            'agg': agg,
            'rows_before': int(before),
            'rows_after': int(after)
        })
        return self.cleaned_data

    def aggregate_by_similarity(self, smiles_cols, target_col, similarity_threshold=0.85,
                                 fp_type='Morgan', n_bits=2048, radius=2,
                                 agg='median', dropna_target=True):
        """按分子指纹相似度聚类后聚合（适用于结构相似配方的合并）

        核心逻辑：
        1. 提取 SMILES 列的分子指纹
        2. 计算 Tanimoto 距离矩阵
        3. 基于相似度阈值进行层次聚类
        4. 簇内按智能策略聚合（数值列 median，类别列众数）

        Args:
            smiles_cols: SMILES 列名列表（如 ['resin_smiles', 'curing_agent_smiles']）
            target_col: 需要聚合的目标列名
            similarity_threshold: Tanimoto 相似度阈值（默认 0.85，越高越严格）
            fp_type: 指纹类型 'MACCS' 或 'Morgan'
            n_bits: Morgan 指纹位数
            radius: Morgan 指纹半径
            agg: 聚合方法：median/mean/min/max
            dropna_target: 是否删除聚合后 target 仍为 NaN 的组

        Returns:
            聚合后的 DataFrame
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        df = self.cleaned_data.copy()

        # 1) 参数检查
        if not isinstance(smiles_cols, (list, tuple)) or len(smiles_cols) == 0:
            raise ValueError("smiles_cols 必须是非空列名列表")
        smiles_cols = [c for c in smiles_cols if c in df.columns]
        if len(smiles_cols) == 0:
            raise ValueError("smiles_cols 中的列不在数据中")
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")

        # 2) 提取分子指纹
        try:
            from .molecular_features import FingerprintExtractor
        except ImportError:
            from molecular_features import FingerprintExtractor

        extractor = FingerprintExtractor()

        # 提取第一个 SMILES 列的指纹
        smi_list_1 = df[smiles_cols[0]].astype(str).tolist()
        smi_list_2 = df[smiles_cols[1]].astype(str).tolist() if len(smiles_cols) >= 2 else None

        fp_df, valid_indices = extractor.smiles_to_fingerprints(
            smi_list_1, smiles_list_2=smi_list_2,
            fp_type=fp_type, n_bits=n_bits, radius=radius
        )

        if fp_df.empty:
            raise ValueError("指纹提取失败，请检查 SMILES 列是否有效")

        # 只保留指纹提取成功的行
        df_valid = df.iloc[valid_indices].reset_index(drop=True)

        # 3) 计算 Tanimoto 距离矩阵
        try:
            from .applicability_domain import _binarize_fingerprint_matrix
        except ImportError:
            from applicability_domain import _binarize_fingerprint_matrix

        fp_bin = _binarize_fingerprint_matrix(fp_df).astype(np.float32)
        n_samples = fp_bin.shape[0]

        # 分批计算距离矩阵（避免内存溢出）
        # 使用压缩距离向量（scipy linkage 需要）
        from scipy.spatial.distance import pdist

        # Tanimoto 距离 = 1 - Tanimoto 相似度
        # 对于二值向量：Tanimoto = intersection / union = dot(a,b) / (|a| + |b| - dot(a,b))
        def _tanimoto_dist_condensed(X):
            """计算压缩 Tanimoto 距离向量"""
            n = X.shape[0]
            # 预计算每行的 bit 数
            row_sums = X.sum(axis=1)
            # 使用 pdist 的自定义度量会很慢，改用矩阵运算分批计算
            batch_size = min(2000, n)
            dists = []
            for i in range(n):
                if i == n - 1:
                    break
                # 计算 i 与 i+1..n-1 的距离
                j_start = i + 1
                # 分批处理避免内存峰值
                for bs in range(j_start, n, batch_size):
                    be = min(bs + batch_size, n)
                    chunk = X[bs:be]
                    inter = np.dot(chunk, X[i])
                    a = row_sums[bs:be]
                    b = row_sums[i]
                    union = a + b - inter
                    sim = np.zeros(len(inter), dtype=np.float32)
                    mask = union > 0
                    sim[mask] = inter[mask] / union[mask]
                    dists.append(1.0 - sim)
            return np.concatenate(dists)

        dist_condensed = _tanimoto_dist_condensed(fp_bin)

        # 4) 层次聚类
        Z = linkage(dist_condensed, method='average')
        # distance_threshold = 1 - similarity_threshold
        cluster_labels = fcluster(Z, t=1.0 - similarity_threshold, criterion='distance')

        # 5) 按簇聚合（复用智能聚合逻辑）
        df_valid['_sim_cluster_'] = cluster_labels
        df_valid[target_col] = pd.to_numeric(df_valid[target_col], errors='coerce')

        grouped = df_valid.groupby('_sim_cluster_', dropna=False, sort=False)

        # 目标列聚合
        agg_lower = (agg or 'median').lower()
        agg_func = {'median': 'median', 'mean': 'mean', 'min': 'min', 'max': 'max'}
        if agg_lower not in agg_func:
            raise ValueError("agg 仅支持 median/mean/min/max")
        target_agg = grouped[target_col].agg(agg_func[agg_lower])
        rep_n = grouped[target_col].count()
        rep_std = grouped[target_col].std()

        # 其他列智能聚合
        other_cols = [c for c in df_valid.columns if c not in [target_col, '_sim_cluster_']]
        numeric_other = [c for c in other_cols if pd.api.types.is_numeric_dtype(df_valid[c])]
        cat_other = [c for c in other_cols if c not in numeric_other]

        def _safe_mode(s):
            m = s.mode()
            return m.iloc[0] if len(m) > 0 else s.iloc[0]

        agg_dict = {}
        for c in numeric_other:
            agg_dict[c] = 'median'
        for c in cat_other:
            agg_dict[c] = _safe_mode

        if agg_dict:
            base = grouped.agg(agg_dict).reset_index(drop=True)
        else:
            base = grouped.first().reset_index(drop=True)

        # 写回目标列聚合值
        base[target_col] = target_agg.values

        # 生成 rep_n / rep_std
        target_lower = str(target_col).lower()
        prefix = 'tg' if target_lower in ['tg', 'tg_c'] or target_lower.startswith('tg') else str(target_col)
        rep_n_col = f"{prefix}_rep_n"
        rep_std_col = f"{prefix}_rep_std"
        if rep_n_col in base.columns:
            rep_n_col = rep_n_col + "_agg"
        if rep_std_col in base.columns:
            rep_std_col = rep_std_col + "_agg"
        base[rep_n_col] = rep_n.values.astype(int)
        base[rep_std_col] = rep_std.values

        # 删除临时列
        if '_sim_cluster_' in base.columns:
            base.drop(columns=['_sim_cluster_'], inplace=True)

        # 可选：删除 target 为 NaN 的组
        if dropna_target:
            base = base[base[target_col].notna()].reset_index(drop=True)

        # 更新状态
        before = len(self.cleaned_data)
        after = len(base)
        self.cleaned_data = base
        self.cleaning_log.append({
            'action': 'aggregate_by_similarity',
            'smiles_cols': smiles_cols,
            'target_col': target_col,
            'similarity_threshold': float(similarity_threshold),
            'fp_type': fp_type,
            'agg': agg_lower,
            'n_clusters': int(cluster_labels.max()),
            'rows_before': int(before),
            'rows_after': int(after)
        })
        return self.cleaned_data
        """对类别列做 one-hot 编码，生成数值特征列"""
        df = self.cleaned_data.copy()
        if not isinstance(cols, (list, tuple)):
            raise ValueError("cols 必须是列名列表")
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            return df

        before_cols = df.shape[1]
        df_encoded = pd.get_dummies(
            df,
            columns=list(cols),
            drop_first=drop_first,
            dummy_na=dummy_na,
            prefix=list(cols),
            prefix_sep='_'
        )

        self.cleaned_data = df_encoded
        self.cleaning_log.append({
            'action': 'one_hot_encode',
            'cols': cols,
            'cols_before': int(before_cols),
            'cols_after': int(df_encoded.shape[1])
        })
        return self.cleaned_data

    def label_encode(self, cols):
        """对类别列做 Label 编码，将每个唯一值映射为 0, 1, 2, ... 的整数"""
        df = self.cleaned_data.copy()
        if not isinstance(cols, (list, tuple)):
            raise ValueError("cols 必须是列名列表")
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            return df, {}

        mapping = {}
        for c in cols:
            codes, uniques = pd.factorize(df[c], sort=True)
            df[c] = codes
            mapping[c] = {str(val): int(i) for i, val in enumerate(uniques)}

        self.cleaned_data = df
        self.cleaning_log.append({
            'action': 'label_encode',
            'cols': cols,
            'mapping': mapping
        })
        return self.cleaned_data, mapping

    def apply_kmeans_clustering(self, feature_cols, n_clusters=None, auto_tune=True, use_fast_pca=True, label_name='Cluster_Label'):
        """
        [修复版+优化版] 使用 K-Means 对数据进行聚类，生成聚类标签特征。
        修复了 ValueError: Number of labels is 1 的问题。
        优化了大规模数据的性能。

        Args:
            feature_cols: 用于聚类的特征列
            n_clusters: 聚类数量（None则自动搜索）
            auto_tune: 是否自动寻找最佳聚类数
            use_fast_pca: 是否使用快速PCA进行数据预处理（大规模数据推荐）
            label_name: 生成的聚类标签列名（支持多次聚类生成不同列）
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        df = self.cleaned_data.copy()
        # 确保只选取数值列并去除空值
        X = df[feature_cols].dropna()

        # [安全检查 1] 样本量太少无法聚类
        if len(X) < 2:
            print("⚠️ 样本数量不足 2 个，无法执行聚类。")
            return df, 0

        # 标准化数据用于聚类
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # [性能优化] 对于大规模数据，先用PCA降维
        if use_fast_pca and X_scaled.shape[1] > 50:
            if FAST_PCA_AVAILABLE:
                X_scaled, _ = fast_pca_transform(X_scaled, variance_threshold=0.95, min_components=10)
                print(f"✅ 使用快速PCA降维: {X.shape[1]} → {X_scaled.shape[1]} 特征")
            else:
                # 回退到标准PCA
                from sklearn.decomposition import PCA
                n_comp = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)
                pca = PCA(n_components=n_comp, random_state=42)
                X_scaled = pca.fit_transform(X_scaled)
                print(f"✅ 使用标准PCA降维: {X.shape[1]} → {X_scaled.shape[1]} 特征")

        best_n = n_clusters if n_clusters else 2
        best_score = -1.0
        best_model = None
        silhouette_scores = {}  # 记录每个 k 对应的轮廓系数

        # [性能优化] 对于大样本数，使用MiniBatchKMeans
        use_minibatch = len(X) > 10000
        KMeansClass = MiniBatchKMeans if use_minibatch else KMeans

        # 自动寻找最佳聚类数
        if auto_tune and n_clusters is None:
            # 搜索范围 2 到 20
            # [安全检查 2] 簇数量不能超过样本数
            max_k = min(21, len(X))
            search_range = range(2, max_k)

            for k in search_range:
                try:
                    if use_minibatch:
                        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
                    else:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)

                    # [关键修复] Silhouette Score 需要至少 2 个唯一的聚类标签
                    unique_labels = np.unique(labels)
                    if len(unique_labels) < 2:
                        score = -1.0  # 无效分数
                    else:
                        # [性能优化] 对于大样本，使用采样计算silhouette score
                        if len(X_scaled) > 5000:
                            sample_size = 5000
                            indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                            score = silhouette_score(X_scaled[indices], labels[indices])
                        else:
                            score = silhouette_score(X_scaled, labels)

                    silhouette_scores[k] = score
                    print(f"  K={k:2d} → Silhouette Score = {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_n = k
                        best_model = kmeans
                except Exception as e:
                    silhouette_scores[k] = -1.0
                    continue

            print(f"✅ 最佳簇数量: K={best_n}, Silhouette Score={best_score:.4f}")

            # 如果自动搜索失败（例如所有结果都只有1类），兜底使用 k=2
            if best_model is None:
                best_n = 2
                if use_minibatch:
                    best_model = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3, batch_size=1024)
                else:
                    best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
                best_model.fit(X_scaled)

        else:
            # 手动模式
            k_manual = min(n_clusters if n_clusters else 2, len(X) - 1)
            if k_manual < 2: k_manual = 2  # 最小为2

            best_n = k_manual
            if use_minibatch:
                best_model = MiniBatchKMeans(n_clusters=best_n, random_state=42, n_init=3, batch_size=1024)
            else:
                best_model = KMeans(n_clusters=best_n, random_state=42, n_init=10)
            best_model.fit(X_scaled)

        # 预测并赋值
        if best_model is not None:
            labels = best_model.predict(X_scaled)

            # 将聚类标签拼回原数据 (注意索引对齐)
            df.loc[X.index, label_name] = labels
            # 填充未参与聚类的行（如果有）为 -1
            df[label_name] = df[label_name].fillna(-1).astype(int)

            self.cleaning_log.append({
                'action': 'kmeans_clustering',
                'n_clusters': best_n,
                'silhouette_score': float(best_score) if auto_tune else 'N/A',
                'algorithm': 'MiniBatchKMeans' if use_minibatch else 'KMeans',
                'used_pca': use_fast_pca and X.shape[1] > 50
            })
            self.cleaned_data = df

        return self.cleaned_data, best_n, silhouette_scores

    def clean_smiles_columns(self, columns: list, strategy: str = 'standard', drop_invalid: bool = False,
                              preserve_original_on_fail: bool = False,
                              model_path: str = None,
                              use_transformer: bool = False,
                              beam_size: int = 5,
                              batch_size: int = 256,
                              show_progress: bool = True):
        """
        清洗指定的 SMILES 列 - 优化版（支持批量GPU推理）

        Args:
            columns: 需要清洗的列名列表
            strategy: 清洗策略
                - 'standard': 仅进行标准 RDKit canonicalize，无效变 NaN
                - 'repair': 尝试智能修复（去除立体信息、取最大片段等）
                - 'aggressive': 激进修复模式（尝试所有可能的修复策略）
                - 'ultra': 超级修复模式（最大化保留数据，返回详细修复状态）
                - 'transformer': [新增] 使用 Transformer 深度学习模型纠错
                - 'hybrid': [新增] 混合模式，先使用 Transformer，失败后使用规则方法
                - 'strict': 严格模式，任何解析失败直接 NaN
            drop_invalid: 是否删除清洗后 SMILES 为 NaN 的行
            preserve_original_on_fail: 当所有修复策略都失败时，是否保留原始字符串
            model_path: Transformer 模型路径（仅用于 transformer/hybrid 模式）
            use_transformer: 是否启用 Transformer 纠错（自动切换到 hybrid 模式）
            beam_size: Beam search 大小（仅用于 Transformer 模式）
            batch_size: ★ 批量处理大小（GPU推理优化）
            show_progress: ★ 是否显示详细进度

        Returns:
            cleaned_data DataFrame
        """
        # 导入增强修复函数
        try:
            from core.smiles_utils import (
                canonicalize_smiles, smart_repair_smiles, 
                aggressive_repair_smiles, ultra_repair_smiles,
                transformer_repair_smiles, hybrid_repair_smiles,
                get_correction_pipeline
            )
        except ImportError:
            try:
                from .smiles_utils import (
                    canonicalize_smiles, smart_repair_smiles,
                    aggressive_repair_smiles, ultra_repair_smiles,
                    transformer_repair_smiles, hybrid_repair_smiles,
                    get_correction_pipeline
                )
            except ImportError:
                # Fallback
                transformer_repair_smiles = None
                hybrid_repair_smiles = None
                get_correction_pipeline = None
        
        # ★ 尝试导入批量处理器
        batch_processor_available = False
        try:
            from core.smiles_batch_processor import (
                SMILESBatchProcessor,
                BatchProcessingConfig,
                batch_correct_smiles
            )
            batch_processor_available = True
        except ImportError:
            try:
                from .smiles_batch_processor import (
                    SMILESBatchProcessor,
                    BatchProcessingConfig,
                    batch_correct_smiles
                )
                batch_processor_available = True
            except ImportError:
                pass
        
        # 如果显式启用 Transformer，切换到 hybrid 模式
        if use_transformer and strategy not in ['transformer', 'hybrid']:
            strategy = 'hybrid'
        
        if not columns:
            return self.cleaned_data

        df = self.cleaned_data.copy()

        for col in columns:
            if col not in df.columns:
                continue

            original_valid_count = df[col].notna().sum()
            repair_stats = {'success': 0, 'repaired': 0, 'transformer': 0, 'rule': 0, 'preserved': 0, 'failed': 0,
                           'direct_valid': 0, 'dl_corrected': 0, 'rule_corrected': 0}

            # ★★★ 使用批量处理器进行Transformer/Hybrid模式 ★★★
            if strategy in ['transformer', 'hybrid'] and batch_processor_available:
                print(f"🚀 使用优化的批量处理器 (batch_size={batch_size})")
                
                # 配置批量处理器
                config = BatchProcessingConfig(
                    batch_size=batch_size,
                    use_fp16=True,
                    beam_size=beam_size,
                    use_transformer=True,
                    use_rules_fallback=(strategy == 'hybrid'),
                    preserve_original_on_fail=preserve_original_on_fail
                )
                
                processor = SMILESBatchProcessor(config=config, model_path=model_path)
                
                # 获取SMILES列表
                smiles_list = df[col].fillna('').astype(str).tolist()
                
                # ★ 批量处理 - 真正的GPU批量推理！
                result = processor.process_batch(
                    smiles_list,
                    show_progress=show_progress
                )
                
                # 更新DataFrame
                df[col] = result.corrected_smiles
                
                # 更新统计
                repair_stats.update({
                    'direct_valid': result.stats.get('direct_valid', 0),
                    'dl_corrected': result.stats.get('dl_corrected', 0),
                    'rule_corrected': result.stats.get('rule_corrected', 0),
                    'success': result.stats.get('direct_valid', 0),
                    'transformer': result.stats.get('dl_corrected', 0),
                    'rule': result.stats.get('rule_corrected', 0),
                    'failed': result.stats.get('failed', 0)
                })
                
            elif strategy == 'transformer':
                # 回退到逐条处理（如果批量处理器不可用）
                if transformer_repair_smiles is not None:
                    print("⚠️ 批量处理器不可用，使用逐条处理（性能较低）")
                    df[col] = df[col].apply(
                        lambda x: transformer_repair_smiles(
                            x, model_path=model_path, beam_size=beam_size, 
                            use_rules_fallback=False
                        )
                    )
                else:
                    print("⚠️ Transformer 纠错不可用，回退到 aggressive 模式")
                    df[col] = df[col].apply(
                        lambda x: aggressive_repair_smiles(x, keep_largest_frag=True)
                    )
                    
            elif strategy == 'hybrid':
                # 回退到逐条处理（如果批量处理器不可用）
                if hybrid_repair_smiles is not None:
                    print("⚠️ 批量处理器不可用，使用逐条处理（性能较低）")
                    results = df[col].apply(
                        lambda x: hybrid_repair_smiles(
                            x, 
                            use_transformer=(transformer_repair_smiles is not None),
                            use_aggressive=True,
                            use_ultra=True,
                            model_path=model_path,
                            keep_largest_frag=True
                        )
                    )
                    # 分离 SMILES 和方法
                    df[col] = results.apply(lambda x: x[0] if isinstance(x, tuple) else x)
                    # 统计修复方法
                    methods = results.apply(lambda x: x[1] if isinstance(x, tuple) else 'unknown')
                    for method in methods:
                        if method == 'direct':
                            repair_stats['success'] += 1
                        elif method == 'transformer':
                            repair_stats['transformer'] += 1
                        elif method in ['aggressive', 'smart']:
                            repair_stats['rule'] += 1
                        elif method.startswith('ultra'):
                            repair_stats['rule'] += 1
                        elif 'fail' in method:
                            repair_stats['failed'] += 1
                        else:
                            repair_stats['repaired'] += 1
                else:
                    # Fallback 到 aggressive
                    df[col] = df[col].apply(
                        lambda x: aggressive_repair_smiles(x, keep_largest_frag=True)
                    )

            elif strategy == 'aggressive':
                # 激进修复模式
                df[col] = df[col].apply(
                    lambda x: aggressive_repair_smiles(x, keep_largest_frag=True, 
                                                       preserve_original_on_fail=preserve_original_on_fail)
                )
            elif strategy == 'ultra':
                # 超级修复模式（返回详细状态）
                results = df[col].apply(
                    lambda x: ultra_repair_smiles(x, keep_largest_frag=True, 
                                                  try_all_fragments=True,
                                                  preserve_original=preserve_original_on_fail)
                )
                # 分离SMILES和状态
                df[col] = results.apply(lambda x: x[0] if isinstance(x, tuple) else x)
                # 统计修复状态
                statuses = results.apply(lambda x: x[1] if isinstance(x, tuple) else 'unknown')
                for status in statuses:
                    if status == 'success':
                        repair_stats['success'] += 1
                    elif status.startswith('repaired'):
                        repair_stats['repaired'] += 1
                    elif status == 'preserved':
                        repair_stats['preserved'] += 1
                    else:
                        repair_stats['failed'] += 1
                        
            elif strategy == 'repair':
                # 标准智能修复
                df[col] = df[col].apply(lambda x: smart_repair_smiles(x, keep_largest_frag=True))
            elif strategy == 'strict':
                # 严格模式
                df[col] = df[col].apply(canonicalize_smiles)
            else:
                # 默认标准模式
                df[col] = df[col].apply(canonicalize_smiles)

            new_valid_count = df[col].notna().sum()
            lost = original_valid_count - new_valid_count

            log_entry = {
                'action': 'clean_smiles',
                'column': col,
                'strategy': strategy,
                'valid_before': int(original_valid_count),
                'valid_after': int(new_valid_count),
                'lost_samples': int(lost),
                'preserve_original': preserve_original_on_fail,
                'use_transformer': strategy in ['transformer', 'hybrid']
            }
            
            # 添加详细修复统计
            if strategy in ['ultra', 'hybrid']:
                log_entry['repair_stats'] = repair_stats
            
            self.cleaning_log.append(log_entry)

        if drop_invalid:
            before_len = len(df)
            df = df.dropna(subset=columns, how='all').reset_index(drop=True)
            self.cleaning_log.append({
                'action': 'drop_invalid_smiles_rows',
                'columns': columns,
                'rows_dropped': int(before_len - len(df))
            })

        self.cleaned_data = df
        return self.cleaned_data
    
    def train_smiles_corrector_from_data(self, 
                                          smiles_column: str,
                                          model_save_path: str,
                                          valid_only: bool = True,
                                          **training_kwargs):
        """
        从当前数据中训练 SMILES 纠错模型
        
        Args:
            smiles_column: SMILES 列名
            model_save_path: 模型保存路径
            valid_only: 是否仅使用 RDKit 验证通过的 SMILES
            **training_kwargs: 额外的训练参数（传递给 train_smiles_corrector）
        
        Returns:
            训练历史字典
        """
        try:
            from core.smiles_utils import train_smiles_corrector, canonicalize_smiles
        except ImportError:
            from .smiles_utils import train_smiles_corrector, canonicalize_smiles
        
        if smiles_column not in self.cleaned_data.columns:
            raise ValueError(f"列 '{smiles_column}' 不存在")
        
        # 提取 SMILES 列表
        smiles_list = self.cleaned_data[smiles_column].dropna().astype(str).tolist()
        
        # 如果需要，仅保留有效 SMILES
        if valid_only:
            smiles_list = [s for s in smiles_list if canonicalize_smiles(s) is not None]
        
        if len(smiles_list) < 100:
            raise ValueError(f"训练数据不足，至少需要 100 个有效 SMILES，当前仅有 {len(smiles_list)} 个")
        
        print(f"📊 使用 {len(smiles_list)} 个 SMILES 训练纠错模型...")
        
        # 训练模型
        history = train_smiles_corrector(
            smiles_list,
            model_save_path,
            **training_kwargs
        )
        
        self.cleaning_log.append({
            'action': 'train_smiles_corrector',
            'smiles_column': smiles_column,
            'n_samples': len(smiles_list),
            'model_path': model_save_path
        })
        
        return history
