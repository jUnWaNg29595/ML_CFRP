# -*- coding: utf-8 -*-
"""高级SHAP分析模块 - 用于深入理解特征影响模式"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class AdvancedSHAPAnalyzer:
    """高级SHAP分析器 - 提供更深入的特征解释"""

    def __init__(self, model, X_train, X_test, y_train, y_test,
                 feature_names, shap_values=None, explainer=None):
        """
        初始化高级SHAP分析器

        Args:
            model: 训练好的模型
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集目标
            y_test: 测试集目标
            feature_names: 特征名列表
            shap_values: 预计算的SHAP值（可选）
            explainer: SHAP解释器（可选）
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_names)
        self.X_test = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.shap_values = shap_values
        self.explainer = explainer

    def plot_dependence_analysis(self, feature_idx: int, interaction_idx: str = 'auto',
                                  max_samples: int = 500) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        绘制SHAP依赖图 - 显示单个特征的详细影响模式

        Args:
            feature_idx: 特征索引或名称
            interaction_idx: 交互特征索引（'auto'自动选择）
            max_samples: 最大采样数

        Returns:
            (fig, df): matplotlib图形和数据DataFrame
        """
        if self.shap_values is None:
            raise ValueError("需要先计算SHAP值")

        # 采样
        n_samples = min(max_samples, len(self.X_test))
        sample_idx = np.random.choice(len(self.X_test), n_samples, replace=False)
        X_sample = self.X_test.iloc[sample_idx]
        shap_sample = self.shap_values[sample_idx]

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制依赖图
        shap.dependence_plot(
            feature_idx,
            shap_sample,
            X_sample,
            interaction_index=interaction_idx,
            ax=ax,
            show=False
        )

        plt.title(f'SHAP Dependence Plot: {self.feature_names[feature_idx] if isinstance(feature_idx, int) else feature_idx}',
                  fontsize=14, pad=15)
        plt.tight_layout()

        # 导出数据
        feature_name = self.feature_names[feature_idx] if isinstance(feature_idx, int) else feature_idx
        df = pd.DataFrame({
            'feature_value': X_sample.iloc[:, feature_idx] if isinstance(feature_idx, int) else X_sample[feature_idx],
            'shap_value': shap_sample[:, feature_idx] if isinstance(feature_idx, int) else shap_sample[:, self.feature_names.index(feature_idx)]
        })

        return fig, df

    def analyze_feature_correlations(self, top_n: int = 20) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        分析特征与目标的相关性

        Args:
            top_n: 显示前N个特征

        Returns:
            (fig, df): matplotlib图形和相关性DataFrame
        """
        # 计算相关性
        correlations = []
        for col in self.X_train.columns:
            try:
                corr = np.corrcoef(self.X_train[col].fillna(0), self.y_train)[0, 1]
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'correlation_raw': corr
                })
            except:
                correlations.append({
                    'feature': col,
                    'correlation': 0,
                    'correlation_raw': 0
                })

        df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False).head(top_n)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

        colors = ['#d62728' if x > 0 else '#1f77b4' for x in df_corr['correlation_raw']]
        ax.barh(range(len(df_corr)), df_corr['correlation_raw'], color=colors)
        ax.set_yticks(range(len(df_corr)))
        ax.set_yticklabels(df_corr['feature'])
        ax.set_xlabel('Correlation with Target', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Correlations', fontsize=14, pad=15)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        return fig, df_corr

    def analyze_by_target_range(self, n_bins: int = 3) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        按目标变量范围分段分析SHAP值

        Args:
            n_bins: 分段数量

        Returns:
            (fig, df): matplotlib图形和分段统计DataFrame
        """
        if self.shap_values is None:
            raise ValueError("需要先计算SHAP值")

        # 将目标变量分段
        y_test_arr = np.array(self.y_test).ravel()
        bins = pd.qcut(y_test_arr, q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')

        # 计算每段的平均SHAP值
        results = []
        for bin_label in bins.unique():
            mask = (bins == bin_label)
            shap_mean = np.abs(self.shap_values[mask]).mean(axis=0)

            for i, feat in enumerate(self.feature_names):
                results.append({
                    'segment': bin_label,
                    'feature': feat,
                    'mean_abs_shap': shap_mean[i]
                })

        df_segments = pd.DataFrame(results)

        # 找出每段最重要的特征
        top_features_per_segment = []
        for segment in df_segments['segment'].unique():
            seg_data = df_segments[df_segments['segment'] == segment].nlargest(10, 'mean_abs_shap')
            top_features_per_segment.extend(seg_data['feature'].tolist())

        # 取并集作为要显示的特征
        features_to_show = list(set(top_features_per_segment))[:20]

        # 透视表
        pivot = df_segments[df_segments['feature'].isin(features_to_show)].pivot(
            index='feature', columns='segment', values='mean_abs_shap'
        )

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, max(8, len(features_to_show) * 0.4)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean |SHAP|'})
        ax.set_title(f'Feature Importance by Target Range ({n_bins} segments)', fontsize=14, pad=15)
        ax.set_xlabel('Target Range', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        plt.tight_layout()

        return fig, df_segments

    def detect_feature_interactions(self, top_n: int = 10) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        检测特征交互效应

        Args:
            top_n: 显示前N对交互

        Returns:
            (fig, df): matplotlib图形和交互强度DataFrame
        """
        if self.shap_values is None:
            raise ValueError("需要先计算SHAP值")

        # 计算特征交互强度（简化版：使用SHAP值的协方差）
        shap_df = pd.DataFrame(self.shap_values, columns=self.feature_names)

        interactions = []
        n_features = len(self.feature_names)

        for i in range(n_features):
            for j in range(i+1, n_features):
                # 计算两个特征SHAP值的相关性
                corr = np.corrcoef(shap_df.iloc[:, i], shap_df.iloc[:, j])[0, 1]
                interactions.append({
                    'feature_1': self.feature_names[i],
                    'feature_2': self.feature_names[j],
                    'interaction_strength': abs(corr)
                })

        df_interactions = pd.DataFrame(interactions).sort_values('interaction_strength', ascending=False).head(top_n)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))

        labels = [f"{row['feature_1']}\n×\n{row['feature_2']}" for _, row in df_interactions.iterrows()]
        ax.barh(range(len(df_interactions)), df_interactions['interaction_strength'], color='#ff7f0e')
        ax.set_yticks(range(len(df_interactions)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Interaction Strength (|correlation|)', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Interactions', fontsize=14, pad=15)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        return fig, df_interactions

    def comprehensive_report(self, output_dir: Optional[str] = None) -> dict:
        """
        生成综合分析报告

        Args:
            output_dir: 输出目录（可选）

        Returns:
            包含所有分析结果的字典
        """
        results = {}

        # 1. 特征相关性
        try:
            fig_corr, df_corr = self.analyze_feature_correlations()
            results['correlation'] = {'fig': fig_corr, 'data': df_corr}
        except Exception as e:
            print(f"相关性分析失败: {e}")

        # 2. 分段分析
        try:
            fig_seg, df_seg = self.analyze_by_target_range()
            results['segmentation'] = {'fig': fig_seg, 'data': df_seg}
        except Exception as e:
            print(f"分段分析失败: {e}")

        # 3. 交互检测
        try:
            fig_int, df_int = self.detect_feature_interactions()
            results['interactions'] = {'fig': fig_int, 'data': df_int}
        except Exception as e:
            print(f"交互检测失败: {e}")

        return results
