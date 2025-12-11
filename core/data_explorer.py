# -*- coding: utf-8 -*-
"""数据探索模块"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EnhancedDataExplorer:
    """增强版数据探索器"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    def generate_summary_stats(self):
        return {
            'basic_info': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'numeric_columns': len(self.numeric_cols),
                'categorical_columns': len(self.categorical_cols),
                'missing_values': int(self.data.isnull().sum().sum()),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 ** 2
            },
            'numeric_summary': self.data[self.numeric_cols].describe().to_dict() if self.numeric_cols else {},
            'missing_by_column': self.data.isnull().sum().to_dict()
        }

    def plot_correlation_matrix(self, width=1200, height=800):
        if len(self.numeric_cols) < 2:
            return None
        corr = self.data[self.numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0,
            text=corr.values.round(2), texttemplate='%{text}'
        ))
        fig.update_layout(title='相关性矩阵', width=width, height=height)
        return fig

    def plot_distributions(self, max_cols=15, width=1400, height=1000):
        if not self.numeric_cols:
            return None
        cols = self.numeric_cols[:max_cols]
        n_cols = min(3, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)
        for i, col in enumerate(cols):
            fig.add_trace(
                go.Histogram(x=self.data[col].dropna(), name=col, showlegend=False),
                row=i // n_cols + 1, col=i % n_cols + 1
            )
        fig.update_layout(title='特征分布', width=width, height=height, showlegend=False)
        return fig

    def plot_missing_values(self, width=1000, height=600):
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values()
        if len(missing) == 0:
            return None
        fig = go.Figure(go.Bar(
            x=missing.values, y=missing.index, orientation='h',
            text=[f"{v} ({v/len(self.data)*100:.1f}%)" for v in missing.values]
        ))
        fig.update_layout(title='缺失值分布', width=width, height=max(400, len(missing) * 25))
        return fig

    def plot_boxplots(self, max_cols=10, width=1200, height=600):
        if not self.numeric_cols:
            return None
        fig = go.Figure()
        for col in self.numeric_cols[:max_cols]:
            fig.add_trace(go.Box(y=self.data[col].dropna(), name=col))
        fig.update_layout(title='箱线图', width=width, height=height)
        return fig

    def get_high_correlation_pairs(self, threshold=0.8):
        if len(self.numeric_cols) < 2:
            return []
        corr = self.data[self.numeric_cols].corr()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pairs.append({
                        'feature1': corr.columns[i],
                        'feature2': corr.columns[j],
                        'correlation': corr.iloc[i, j]
                    })
        return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)
