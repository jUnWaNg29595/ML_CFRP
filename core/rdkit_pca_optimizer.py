#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RDKit特征PCA降维优化器

针对RDKit描述符进行智能PCA降维，自动识别RDKit特征列并进行优化
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# 导入快速PCA优化器
try:
    from .fast_pca import FastPCAOptimizer
    FAST_PCA_AVAILABLE = True
except ImportError:
    try:
        from fast_pca import FastPCAOptimizer
        FAST_PCA_AVAILABLE = True
    except ImportError:
        FAST_PCA_AVAILABLE = False
        FastPCAOptimizer = None

# 统一图表风格
try:
    from .plot_style import apply_global_style
    apply_global_style()
except Exception:
    pass


class RDKitPCAOptimizer:
    """RDKit特征PCA降维优化器"""

    # RDKit常见描述符名称模式（用于自动识别）
    RDKIT_PATTERNS = [
        # 基本分子性质
        'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
        # 氢键相关
        'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
        # 环相关
        'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
        'RingCount', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
        'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        # 原子计数
        'HeavyAtomCount', 'NumHeteroatoms', 'NumValenceElectrons',
        'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
        'MaxAbsPartialCharge', 'MinAbsPartialCharge',
        # 碳相关
        'FractionCSP3', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles',
        # 拓扑指数
        'Chi0', 'Chi1', 'Chi2', 'Chi3', 'Chi4', 'Chi0n', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
        'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
        'Kappa1', 'Kappa2', 'Kappa3',
        'BalabanJ', 'BertzCT', 'Ipc',
        # 电子性质
        'HallKierAlpha',
        'PEOE_VSA', 'SMR_VSA', 'SlogP_VSA', 'EState_VSA', 'VSA_EState',
        # 功能团计数（fr_开头）
        'fr_',
        # 其他
        'qed', 'MolWt', 'ExactMolWt',
        # EState相关
        'EState', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
    ]

    # 指纹特征模式（需要排除）
    FINGERPRINT_PATTERNS = [
        'MACCS', 'maccs', 'Morgan', 'morgan', 'ECFP', 'ecfp', 'FCFP', 'fcfp',
        'fp_', 'fingerprint', 'Fingerprint', 'bit_', 'Bit_', 'FP_'
    ]

    def __init__(self, variance_threshold: float = 0.95, min_components: int = 5,
                 use_fast_pca: bool = True, pca_method: Optional[str] = None):
        """
        初始化优化器

        Args:
            variance_threshold: 累计解释方差阈值（默认95%）
            min_components: 最小保留主成分数（默认5）
            use_fast_pca: 是否使用快速PCA优化（默认True）
            pca_method: PCA方法 ('standard', 'randomized', 'incremental', 'truncated', None=自动)
        """
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.use_fast_pca = use_fast_pca and FAST_PCA_AVAILABLE
        self.pca_method = pca_method
        self.scaler = None
        self.pca = None
        self.rdkit_cols = []
        self.other_cols = []
        self.feature_importance = None
        self.pca_stats = None

    def detect_rdkit_features(self, df: pd.DataFrame) -> List[str]:
        """
        自动检测DataFrame中的RDKit特征列

        Args:
            df: 输入DataFrame

        Returns:
            RDKit特征列名列表
        """
        rdkit_cols = []

        for col in df.columns:
            col_str = str(col)

            # 首先排除指纹特征
            is_fingerprint = any(fp_pattern in col_str for fp_pattern in self.FINGERPRINT_PATTERNS)
            if is_fingerprint:
                continue

            # 策略1: 检查是否匹配RDKit描述符模式
            if any(pattern in col_str for pattern in self.RDKIT_PATTERNS):
                rdkit_cols.append(col)
                continue

            # 策略2: 检查是否有rdkit前缀（但不是指纹）
            if col_str.lower().startswith('rdkit_') and not is_fingerprint:
                rdkit_cols.append(col)
                continue

            # 策略3: 检查是否是纯数值列且名称像描述符
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # 如果列名包含大写字母和数字混合，可能是描述符
                if any(c.isupper() for c in col_str) and any(c.isdigit() for c in col_str):
                    # 但要再次确认不是指纹
                    if not is_fingerprint:
                        rdkit_cols.append(col)
                        continue

            # 策略4: 宽松模式 - 检查常见的RDKit描述符特征
            # 如果列名包含这些关键词，很可能是RDKit描述符
            rdkit_keywords = [
                'VSA', 'EState', 'Partial', 'Charge', 'Valence', 'Radical',
                'Aromatic', 'Aliphatic', 'Saturated', 'Hetero', 'Carbocycle',
                'Heterocycle', 'Donor', 'Acceptor', 'Rotatable', 'Exact',
                'Index', 'Abs', 'Max', 'Min'
            ]

            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if any(keyword in col_str for keyword in rdkit_keywords):
                    if not is_fingerprint:
                        rdkit_cols.append(col)
                        continue

        return rdkit_cols

    def fit_transform(self, df: pd.DataFrame,
                     rdkit_cols: Optional[List[str]] = None,
                     prefix: str = 'PC') -> Tuple[pd.DataFrame, Dict]:
        """
        对RDKit特征进行PCA降维

        Args:
            df: 输入DataFrame
            rdkit_cols: RDKit特征列名（None则自动检测）
            prefix: PCA特征前缀

        Returns:
            (降维后的DataFrame, 统计信息字典)
        """
        # 自动检测RDKit特征
        if rdkit_cols is None:
            self.rdkit_cols = self.detect_rdkit_features(df)
        else:
            self.rdkit_cols = rdkit_cols

        if not self.rdkit_cols:
            raise ValueError("未检测到RDKit特征列，请手动指定rdkit_cols参数")

        # 分离RDKit特征和其他特征
        self.other_cols = [c for c in df.columns if c not in self.rdkit_cols]

        # 提取RDKit特征
        X_rdkit = df[self.rdkit_cols].copy()

        # 处理缺失值
        X_rdkit = X_rdkit.fillna(X_rdkit.median())

        # 移除零方差特征
        var_mask = X_rdkit.var() > 1e-10
        X_rdkit = X_rdkit.loc[:, var_mask]
        removed_zero_var = len(self.rdkit_cols) - X_rdkit.shape[1]

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_rdkit)

        # 选择PCA方法
        if self.use_fast_pca and FAST_PCA_AVAILABLE:
            # 使用快速PCA优化器
            fast_optimizer = FastPCAOptimizer(
                variance_threshold=self.variance_threshold,
                min_components=self.min_components,
                auto_select_method=(self.pca_method is None)
            )

            X_pca, pca_stats = fast_optimizer.fit_transform(X_scaled, method=self.pca_method)
            self.pca = fast_optimizer.pca
            self.pca_stats = pca_stats

            # 兼容性：确保pca对象有必要的属性
            if not hasattr(self.pca, 'n_components_'):
                self.pca.n_components_ = X_pca.shape[1]
        else:
            # 使用标准PCA
            self.pca = PCA(n_components=self.variance_threshold, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)

            # 确保至少保留min_components个主成分
            if X_pca.shape[1] < self.min_components:
                self.pca = PCA(n_components=self.min_components, random_state=42)
                X_pca = self.pca.fit_transform(X_scaled)

            self.pca_stats = {
                'method': 'standard',
                'n_components': X_pca.shape[1],
                'explained_variance_ratio': self.pca.explained_variance_ratio_,
                'total_variance_explained': self.pca.explained_variance_ratio_.sum()
            }

        # 创建PCA特征DataFrame
        pca_cols = [f'{prefix}{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

        # 合并其他特征
        if self.other_cols:
            df_result = pd.concat([df[self.other_cols], df_pca], axis=1)
        else:
            df_result = df_pca

        # 计算特征重要性（基于载荷）
        loadings = np.abs(self.pca.components_)
        self.feature_importance = pd.DataFrame({
            'feature': X_rdkit.columns,
            'importance': loadings.sum(axis=0)
        }).sort_values('importance', ascending=False)

        # 统计信息
        stats = {
            'original_rdkit_features': len(self.rdkit_cols),
            'removed_zero_variance': removed_zero_var,
            'used_rdkit_features': X_rdkit.shape[1],
            'n_components': X_pca.shape[1],
            'explained_variance_ratio': self.pca_stats.get('explained_variance_ratio',
                                                           self.pca.explained_variance_ratio_ if hasattr(self.pca, 'explained_variance_ratio_') else []),
            'cumulative_variance': np.cumsum(self.pca_stats.get('explained_variance_ratio',
                                                                self.pca.explained_variance_ratio_ if hasattr(self.pca, 'explained_variance_ratio_') else [])),
            'total_variance_explained': self.pca_stats.get('total_variance_explained',
                                                           self.pca.explained_variance_ratio_.sum() if hasattr(self.pca, 'explained_variance_ratio_') else 0),
            'compression_ratio': X_rdkit.shape[1] / X_pca.shape[1],
            'other_features': len(self.other_cols),
            'total_features': df_result.shape[1],
            'pca_method': self.pca_stats.get('method', 'standard') if self.pca_stats else 'standard'
        }

        return df_result, stats

    def transform(self, df: pd.DataFrame, prefix: str = 'PC') -> pd.DataFrame:
        """
        对新数据应用已拟合的PCA转换

        Args:
            df: 输入DataFrame
            prefix: PCA特征前缀

        Returns:
            降维后的DataFrame
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("请先调用fit_transform()进行拟合")

        # 提取RDKit特征
        X_rdkit = df[self.rdkit_cols].copy()
        X_rdkit = X_rdkit.fillna(X_rdkit.median())

        # 标准化和PCA转换
        X_scaled = self.scaler.transform(X_rdkit)
        X_pca = self.pca.transform(X_scaled)

        # 创建PCA特征DataFrame
        pca_cols = [f'{prefix}{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

        # 合并其他特征
        if self.other_cols:
            df_result = pd.concat([df[self.other_cols], df_pca], axis=1)
        else:
            df_result = df_pca

        return df_result

    def plot_analysis(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        生成PCA分析可视化

        Args:
            figsize: 图表大小

        Returns:
            matplotlib Figure对象
        """
        if self.pca is None:
            raise ValueError("请先调用fit_transform()进行拟合")

        fig = plt.figure(figsize=figsize)

        # 1. 解释方差（碎石图）
        ax1 = plt.subplot(2, 3, 1)
        n_show = min(20, len(self.pca.explained_variance_ratio_))
        ax1.bar(range(1, n_show + 1), self.pca.explained_variance_ratio_[:n_show],
               alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('主成分', fontsize=11)
        ax1.set_ylabel('解释方差比例', fontsize=11)
        ax1.set_title('碎石图 (Scree Plot)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # 标注前5个
        for i in range(min(5, n_show)):
            ax1.text(i+1, self.pca.explained_variance_ratio_[i],
                    f'{self.pca.explained_variance_ratio_[i]:.1%}',
                    ha='center', va='bottom', fontsize=9)

        # 2. 累计解释方差
        ax2 = plt.subplot(2, 3, 2)
        cum_var = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cum_var) + 1), cum_var,
                marker='o', linewidth=2, markersize=6, color='darkred')
        ax2.fill_between(range(1, len(cum_var) + 1), cum_var, alpha=0.2, color='darkred')
        ax2.axhline(y=self.variance_threshold, color='green', linestyle='--',
                   linewidth=2, label=f'{self.variance_threshold:.0%} 阈值')
        ax2.set_xlabel('主成分数量', fontsize=11)
        ax2.set_ylabel('累计解释方差', fontsize=11)
        ax2.set_title('累计解释方差曲线', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. 特征重要性（Top 20）
        ax3 = plt.subplot(2, 3, 3)
        top_features = self.feature_importance.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax3.barh(range(len(top_features)), top_features['importance'].values,
                color=colors, edgecolor='black', alpha=0.7)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax3.set_xlabel('重要性得分', fontsize=11)
        ax3.set_title('特征重要性 (Top 20)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()

        # 4. PC1 vs PC2 载荷图
        ax4 = plt.subplot(2, 3, 4)
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        ax4.scatter(loadings[:, 0], loadings[:, 1], alpha=0.5, s=30,
                   c='steelblue', edgecolors='black', linewidths=0.3)
        ax4.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.3)
        ax4.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.3)
        ax4.set_xlabel(f'PC1 载荷 ({self.pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax4.set_ylabel(f'PC2 载荷 ({self.pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax4.set_title('PC1 vs PC2 载荷分布', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        # 标注重要特征
        importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        top_idx = np.argsort(importance)[-5:]
        for idx in top_idx:
            ax4.annotate(self.feature_importance.iloc[idx]['feature'],
                       (loadings[idx, 0], loadings[idx, 1]),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

        # 5. 降维效果对比
        ax5 = plt.subplot(2, 3, 5)
        categories = ['原始RDKit\n特征', '移除零方差后', 'PCA降维后', '其他特征', '总特征数']
        values = [
            len(self.rdkit_cols),
            len(self.rdkit_cols) - (len(self.rdkit_cols) - self.pca.n_features_in_),
            self.pca.n_components_,
            len(self.other_cols),
            self.pca.n_components_ + len(self.other_cols)
        ]
        colors_bar = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#cc99ff']
        bars = ax5.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.7)
        ax5.set_ylabel('特征数量', fontsize=11)
        ax5.set_title('特征降维效果对比', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 6. 统计信息文本
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        stats_text = f"""
        📊 PCA降维统计信息

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        原始RDKit特征数: {len(self.rdkit_cols)}
        移除零方差特征: {len(self.rdkit_cols) - self.pca.n_features_in_}
        实际使用特征数: {self.pca.n_features_in_}

        主成分数量: {self.pca.n_components_}
        累计解释方差: {self.pca.explained_variance_ratio_.sum():.2%}
        压缩比: {self.pca.n_features_in_ / self.pca.n_components_:.1f}x

        其他特征数: {len(self.other_cols)}
        最终总特征数: {self.pca.n_components_ + len(self.other_cols)}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ✅ 降维完成！
        特征数从 {len(self.rdkit_cols)} 降至 {self.pca.n_components_}
        保留了 {self.pca.explained_variance_ratio_.sum():.1%} 的信息
        """

        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        return fig

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """
        获取最重要的特征

        Args:
            n: 返回前n个特征

        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("请先调用fit_transform()进行拟合")

        return self.feature_importance.head(n)

    def export_report(self, filepath: str):
        """
        导出降维报告

        Args:
            filepath: 保存路径（支持.txt, .csv, .xlsx）
        """
        if self.pca is None:
            raise ValueError("请先调用fit_transform()进行拟合")

        report = {
            '原始RDKit特征数': len(self.rdkit_cols),
            '实际使用特征数': self.pca.n_features_in_,
            '主成分数量': self.pca.n_components_,
            '累计解释方差': f"{self.pca.explained_variance_ratio_.sum():.2%}",
            '压缩比': f"{self.pca.n_features_in_ / self.pca.n_components_:.1f}x",
            '其他特征数': len(self.other_cols),
            '最终总特征数': self.pca.n_components_ + len(self.other_cols)
        }

        if filepath.endswith('.txt'):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("RDKit特征PCA降维报告\n")
                f.write("=" * 50 + "\n\n")
                for key, val in report.items():
                    f.write(f"{key}: {val}\n")
                f.write("\n" + "=" * 50 + "\n")
                f.write("\n特征重要性 (Top 20):\n\n")
                f.write(self.feature_importance.head(20).to_string())

        elif filepath.endswith('.csv'):
            pd.DataFrame([report]).to_csv(filepath, index=False, encoding='utf-8-sig')

        elif filepath.endswith('.xlsx'):
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                pd.DataFrame([report]).to_excel(writer, sheet_name='Summary', index=False)
                self.feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)


def optimize_rdkit_features(df: pd.DataFrame,
                            rdkit_cols: Optional[List[str]] = None,
                            variance_threshold: float = 0.95,
                            min_components: int = 5,
                            prefix: str = 'PC',
                            show_plot: bool = True) -> Tuple[pd.DataFrame, Dict, Optional[plt.Figure]]:
    """
    一键优化RDKit特征（便捷函数）

    Args:
        df: 输入DataFrame
        rdkit_cols: RDKit特征列名（None则自动检测）
        variance_threshold: 累计解释方差阈值
        min_components: 最小保留主成分数
        prefix: PCA特征前缀
        show_plot: 是否生成可视化

    Returns:
        (降维后的DataFrame, 统计信息字典, 可视化Figure)
    """
    optimizer = RDKitPCAOptimizer(
        variance_threshold=variance_threshold,
        min_components=min_components
    )

    df_optimized, stats = optimizer.fit_transform(df, rdkit_cols=rdkit_cols, prefix=prefix)

    fig = None
    if show_plot:
        fig = optimizer.plot_analysis()

    return df_optimized, stats, fig


if __name__ == "__main__":
    # 测试示例
    print("RDKit特征PCA降维优化器")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100

    # 模拟RDKit特征
    rdkit_features = {
        'MolWt': np.random.randn(n_samples) * 50 + 300,
        'MolLogP': np.random.randn(n_samples) * 2 + 3,
        'TPSA': np.random.randn(n_samples) * 30 + 80,
        'NumRotatableBonds': np.random.randint(0, 10, n_samples),
        'NumHDonors': np.random.randint(0, 5, n_samples),
        'NumHAcceptors': np.random.randint(0, 8, n_samples),
    }

    # 添加其他特征
    other_features = {
        'temperature': np.random.randn(n_samples) * 20 + 100,
        'pressure': np.random.randn(n_samples) * 10 + 50,
    }

    df_test = pd.DataFrame({**rdkit_features, **other_features})

    print(f"\n原始数据形状: {df_test.shape}")
    print(f"特征列: {list(df_test.columns)}")

    # 执行优化
    df_optimized, stats, fig = optimize_rdkit_features(
        df_test,
        variance_threshold=0.95,
        show_plot=True
    )

    print(f"\n优化后数据形状: {df_optimized.shape}")
    print(f"\n统计信息:")
    for key, val in stats.items():
        if key not in ['explained_variance_ratio', 'cumulative_variance']:
            print(f"  {key}: {val}")

    plt.show()
