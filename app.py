# -*- coding: utf-8 -*-
"""
碳纤维复合材料智能预测平台 v1.2.8
更新内容：
1. 修复数据探索页面不显示最新（处理后）数据的问题
2. 修复导出功能只导出原始数据的问题
"""
try:
    import torchani

    TORCHANI_AVAILABLE = True
except ImportError:
    TORCHANI_AVAILABLE = False
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import traceback
import json
import io
from datetime import datetime
import multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="碳纤维复合材料智能预测平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入核心模块
from core.data_processor import AdvancedDataCleaner, SparseDataHandler, DataEnhancer
from core.data_explorer import EnhancedDataExplorer
from core.model_trainer import EnhancedModelTrainer
from core.model_interpreter import ModelInterpreter, EnhancedModelInterpreter
from core.molecular_features import AdvancedMolecularFeatureExtractor, RDKitFeatureExtractor
from core.feature_selector import SmartFeatureSelector, SmartSparseDataSelector, show_robust_feature_selection
from core.optimizer import HyperparameterOptimizer, InverseDesigner, generate_tuning_suggestions
from core.visualizer import Visualizer
from core.applicability_domain import ApplicabilityDomainAnalyzer
from core.ui_config import (
    MANUAL_TUNING_PARAMS,
    MODEL_PARAMETERS,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE
)

from config import APP_NAME, VERSION, DATA_DIR
from generate_sample_data import generate_hybrid_dataset, generate_pure_numeric_dataset

# 可选模块导入
try:
    from core.molecular_features import OptimizedRDKitFeatureExtractor, MemoryEfficientRDKitExtractor, \
        FingerprintExtractor

    OPTIMIZED_EXTRACTOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_EXTRACTOR_AVAILABLE = False

try:
    from core.graph_utils import GNNFeaturizer, smiles_to_pyg_graph

    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

try:
    from core.ann_model import ANNRegressor

    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@st.cache_data(ttl=3600, show_spinner="正在读取数据文件...")
def load_data_file(uploaded_file):
    """带缓存的数据加载函数，避免每次交互都重新读取文件"""
    # 必须重置文件指针到开头，因为Streamlit可能会多次读取同一个文件对象
    uploaded_file.seek(0)

    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


# --- 全局常量 ---
USER_DATA_DB = "datasets/user_data.csv"

# --- 自定义 CSS 样式 ---
CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #4F46E5;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --bg-card: #F8FAFC;
        --border-color: #E2E8F0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 8px 0;
    }

    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    .feature-badge {
        display: inline-block;
        background: #E0E7FF;
        color: #4338CA;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin: 2px;
    }

    .status-success {
        color: var(--success-color);
        font-weight: 600;
    }

    .status-warning {
        color: var(--warning-color);
        font-weight: 600;
    }

    .status-error {
        color: var(--error-color);
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
def init_session_state():
    """初始化所有session state变量"""
    defaults = {
        'data': None,
        'processed_data': None,
        'molecular_features': None,
        'target_col': None,
        'feature_cols': [],
        'model': None,
        'model_name': None,
        'train_result': None,
        'scaler': None,
        'pipeline': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'optimization_history': [],
        'best_params': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# 侧边栏渲染
# ============================================================
def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        st.title(f"🔬 {APP_NAME}")
        st.caption(f"版本 {VERSION}")
        st.markdown("---")

        page = st.radio(
            "📌 功能导航",
            [
                "🏠 首页",
                "📤 数据上传",
                "🔍 数据探索",
                "🧹 数据清洗",
                "✨ 数据增强",
                "🧬 分子特征",
                "🎯 特征选择",
                "🤖 模型训练",
                "📊 模型解释",
                "🔮 预测应用",
                "⚙️ 超参优化",
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📊 数据状态")

        # 优先获取 processed_data (清洗/处理后的数据)
        current_df = st.session_state.get('processed_data')
        original_df = st.session_state.get('data')

        # 确定要显示哪个数据的信息
        display_df = current_df if current_df is not None else original_df

        if display_df is not None:
            # 1. 显示行/列数
            status_label = "✅ 当前数据 (已清洗)" if current_df is not None else "✅ 原始数据"
            st.success(f"{status_label}\n\n**{display_df.shape[0]} 行 × {display_df.shape[1]} 列**")

            # 2. 显示分子特征状态
            if st.session_state.get('molecular_features') is not None:
                mf = st.session_state.molecular_features
                st.info(f"🧬 分子特征: {mf.shape[1]} 个")

            # 3. 显示特征选择状态
            feature_cols = st.session_state.get('feature_cols')
            target_col = st.session_state.get('target_col')

            if feature_cols:
                st.info(f"🎯 已选特征 (X): {len(feature_cols)} 个")

            if target_col:
                st.caption(f"🎯 目标变量 (Y): {target_col}")

        else:
            st.warning("⚠️ 未加载数据")

        if st.session_state.model is not None:
            st.success(f"🤖 已训练: {st.session_state.model_name}")
            # 如果有训练结果，也可以显示R2
            if st.session_state.get('train_result'):
                r2 = st.session_state.train_result.get('r2', 0)
                st.caption(f"当前 R²: {r2:.4f}")

        st.markdown("---")
        st.markdown("### 🔧 系统信息")
        st.caption(f"CPU核心: {mp.cpu_count()}")
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            st.caption(f"内存使用: {mem.percent}%")

        return page


# ============================================================
# 页面：首页
# ============================================================
def page_home():
    """首页"""
    st.title("🔬 碳纤维复合材料智能预测平台")
    st.markdown(f"**版本 {VERSION}** ")

    st.markdown("---")

    # 功能卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">数据处理</div>
            <div class="metric-value">📊</div>
            <p>智能清洗 · VAE增强 · 类别平衡</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card metric-card-success">
            <div class="metric-label">分子特征</div>
            <div class="metric-value">🧬</div>
            <p>RDKit · 指纹(MACCS) · 图特征</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card metric-card-warning">
            <div class="metric-label">模型训练</div>
            <div class="metric-value">🤖</div>
            <p>15+模型 · 手动调参 · Optuna优化</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 核心功能介绍
    st.markdown("## 🚀 核心功能")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 数据处理
        - **智能数据清洗**: 缺失值处理、异常值检测、数据类型修复
        - **VAE数据增强**: 基于变分自编码器的表格数据生成
        - **类别平衡**: **(新)** 解决化学单体样本不平衡问题

        ### 🧬 分子特征提取
        - **分子指纹**: **(新)** MACCS Keys, Morgan (ECFP) 指纹
        - **RDKit标准版**: 200+分子描述符
        - **图神经网络特征**: 分子拓扑结构特征
        - **ML力场特征**: ANI-2x 高精度能量/力
        """)

    with col2:
        st.markdown("""
        ### 🤖 模型训练
        - **传统模型**: 线性回归、SVR、决策树等
        - **集成模型**: 随机森林、XGBoost、LightGBM、CatBoost
        - **深度学习**: 自定义神经网络(ANN)
        - **AutoML**: TabPFN、AutoGluon
        - **手动调参**: 可视化参数配置界面

        ### 📊 模型解释
        - **SHAP分析**: 特征重要性可视化
        - **学习曲线**: 模型收敛分析
        - **适用域分析**: PCA凸包边界检测
        """)

    st.markdown("---")

    # 快速开始
    st.markdown("## ⚡ 快速开始")
    st.info("""
    1. **上传数据** → 支持CSV、Excel格式
    2. **数据清洗** → 使用“类别平衡”处理高频单体
    3. **分子特征** → 提取SMILES指纹或描述符
    4. **特征选择** → 选择目标变量和输入特征
    5. **模型训练** → 选择模型并调整参数
    6. **模型解释** → SHAP分析和性能评估
    """)


# ============================================================
# 页面：数据上传
# ============================================================
def page_data_upload():
    """数据上传页面"""
    st.title("📤 数据上传")

    tab1, tab2 = st.tabs(["📁 上传文件", "📝 生成示例数据"])

    with tab1:
        uploaded_file = st.file_uploader(
            "选择数据文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式"
        )

        if uploaded_file is not None:
            try:
                # 使用缓存函数加载数据
                df = load_data_file(uploaded_file)

                # 去重名列
                if df.columns.duplicated().any():
                    st.warning("⚠️ 检测到重名列，系统已自动重命名处理")
                    df = df.loc[:, ~df.columns.duplicated()]

                st.session_state.data = df
                st.session_state.processed_data = df.copy()

                st.success(f"✅ 成功加载数据: {df.shape[0]}行 × {df.shape[1]}列")

                # 数据预览
                st.markdown("### 📋 数据预览")
                st.dataframe(df.head(10), use_container_width=True)

                # 列信息
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 数值列")
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    for col in numeric_cols[:10]:
                        st.markdown(f"<span class='feature-badge'>{col}</span>", unsafe_allow_html=True)
                    if len(numeric_cols) > 10:
                        st.caption(f"... 等共 {len(numeric_cols)} 个数值列")

                with col2:
                    st.markdown("#### 文本列")
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    for col in text_cols[:10]:
                        st.markdown(f"<span class='feature-badge'>{col}</span>", unsafe_allow_html=True)
                    if len(text_cols) > 10:
                        st.caption(f"... 等共 {len(text_cols)} 个文本列")

            except Exception as e:
                st.error(f"❌ 加载失败: {str(e)}")

    with tab2:
        st.markdown("### 生成示例数据集")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🧪 混合数据集")
            st.caption("包含工艺参数和SMILES分子结构")
            n_samples_hybrid = st.number_input("样本数量", 100, 2000, 500, key="n_hybrid")
            if st.button("生成混合数据集", type="primary"):
                df = generate_hybrid_dataset(n_samples=n_samples_hybrid)
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                st.success(f"✅ 已生成混合数据集: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)

        with col2:
            st.markdown("#### 📊 纯数值数据集")
            st.caption("仅包含数值型工艺参数")
            n_samples_numeric = st.number_input("样本数量", 100, 2000, 500, key="n_numeric")
            if st.button("生成纯数值数据集", type="primary"):
                df = generate_pure_numeric_dataset(n_samples=n_samples_numeric)
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                st.success(f"✅ 已生成纯数值数据集: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)


# ============================================================
# 页面：数据探索 (修复版)
# ============================================================
def page_data_explore():
    """数据探索页面 - 修复版"""
    st.title("🔍 数据探索")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    # [关键修复] 优先使用处理后的数据(processed_data)
    # 这样提取特征、清洗后的数据才能显示出来
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data

    explorer = EnhancedDataExplorer(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 描述统计", "🔗 相关性分析", "📈 分布图", "❓ 缺失值", "💾 导出"
    ])

    with tab1:
        stats = explorer.generate_summary_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总行数", stats['basic_info']['total_rows'])
        col2.metric("总列数", stats['basic_info']['total_columns'])
        col3.metric("数值列", stats['basic_info']['numeric_columns'])
        col4.metric("缺失值", stats['basic_info']['missing_values'])

        st.markdown("### 数值特征统计")
        if explorer.numeric_cols:
            st.dataframe(df[explorer.numeric_cols].describe(), use_container_width=True)

    with tab2:
        st.markdown("### 特征相关性矩阵")
        fig = explorer.plot_correlation_matrix()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要至少2个数值列")

        # 高相关性特征对
        pairs = explorer.get_high_correlation_pairs(threshold=0.8)
        if pairs:
            st.markdown("### ⚠️ 高相关性特征对 (|r| > 0.8)")
            for p in pairs[:10]:
                st.write(f"- **{p['feature1']}** ↔ **{p['feature2']}**: {p['correlation']:.3f}")

    with tab3:
        st.markdown("### 数值特征分布")
        fig = explorer.plot_distributions()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 箱线图")
        fig_box = explorer.plot_boxplots()
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.markdown("### 缺失值分析")
        fig_missing = explorer.plot_missing_values()
        if fig_missing:
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ 数据无缺失值")

    with tab5:
        st.markdown("### 导出数据 (最新)")
        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 下载CSV",
                csv,
                "data_export.csv",
                "text/csv"
            )

        with col2:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            st.download_button(
                "📥 下载Excel",
                buffer.getvalue(),
                "data_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ============================================================
# 页面：数据清洗
# ============================================================
def page_data_cleaning():
    """数据清洗页面"""
    st.title("🧹 数据清洗")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    cleaner = AdvancedDataCleaner(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "❓ 缺失值处理", "📊 异常值检测", "🔄 重复数据", "🔧 数据类型", "⚖️ 类别平衡"
    ])

    with tab1:
        st.markdown("### 缺失值处理")

        # 缺失值统计
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.warning(f"检测到 {len(missing)} 列存在缺失值")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(pd.DataFrame({
                    '列名': missing.index,
                    '缺失数量': missing.values,
                    '缺失比例': (missing.values / len(df) * 100).round(2)
                }), use_container_width=True)

            with col2:
                strategy = st.selectbox(
                    "选择填充策略",
                    ["median", "mean", "mode", "knn", "drop_rows", "constant"]
                )

                fill_value = None
                if strategy == "constant":
                    fill_value = st.number_input("填充常数值", value=0.0)

                if st.button("🔧 执行缺失值处理", type="primary"):
                    cleaned_df = cleaner.handle_missing_values(strategy=strategy, fill_value=fill_value)
                    st.session_state.processed_data = cleaned_df
                    st.success("✅ 缺失值处理完成")
                    st.rerun()
        else:
            st.success("✅ 数据无缺失值")

    with tab2:
        st.markdown("### 异常值检测与处理")

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("检测方法", ["iqr", "zscore"])
            threshold = st.slider("阈值", 1.0, 5.0, 1.5 if method == "iqr" else 3.0)

        with col2:
            handle_method = st.selectbox("处理方法", ["clip", "replace_median", "remove"])

        if st.button("🔍 检测异常值"):
            outliers = cleaner.detect_outliers(method=method, threshold=threshold)
            if outliers:
                st.warning(f"检测到 {len(outliers)} 列存在异常值")
                st.json(outliers)
            else:
                st.success("✅ 未检测到显著异常值")

        if st.button("🔧 处理异常值", type="primary"):
            cleaned_df = cleaner.handle_outliers(method=handle_method, threshold=threshold)
            st.session_state.processed_data = cleaned_df
            st.success("✅ 异常值处理完成")

    with tab3:
        st.markdown("### 🔄 数据去重与分布优化")

        col_clean_1, col_clean_2 = st.columns(2)

        with col_clean_1:
            st.markdown("#### 1. 行去重")
            st.caption("删除完全重复的样本行")
            dup_count = df.duplicated().sum()
            st.metric("完全重复行数", dup_count)

            if dup_count > 0:
                if st.button("🗑️ 删除重复行", type="primary"):
                    cleaned_df = cleaner.remove_duplicates()
                    st.session_state.processed_data = cleaned_df
                    st.success(f"✅ 已删除 {dup_count} 行重复数据")
                    st.rerun()
            else:
                st.info("✅ 无重复行")

        st.markdown("---")

        with col_clean_2:
            st.markdown("#### 2. 特征分布优化 (针对数值)")
            st.caption("降低某一特征中众数（出现最多的值）的比例，平衡数据分布")

            # 检测阈值设置
            rep_threshold = st.slider("高重复率检测阈值", 0.5, 0.99, 0.8, 0.05,
                                      help="检测众数占比超过此比例的特征")

            high_rep_cols = cleaner.detect_high_repetition_columns(rep_threshold)

            if high_rep_cols:
                st.warning(f"⚠️ 检测到 {len(high_rep_cols)} 个特征存在高重复值")

                # 显示详情
                rep_data = []
                for col, info in high_rep_cols.items():
                    rep_data.append({
                        "特征": col,
                        "众数": str(info['most_frequent_value']),
                        "当前占比": f"{info['frequency'] * 100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(rep_data), use_container_width=True, hide_index=True)

                # 操作区
                st.markdown("##### 🔧 执行优化")
                target_col = st.selectbox("选择要优化的特征", list(high_rep_cols.keys()))

                # 智能计算滑块范围：不能比当前占比还高，也不能太低（如0%）
                current_freq = high_rep_cols[target_col]['frequency']
                target_rate = st.slider(
                    f"目标占比 (针对 {target_col})",
                    0.1, float(current_freq), 0.5, 0.05,
                    help="通过随机删除包含众数的样本，使其占比降低到此值"
                )

                if st.button(f"📉 降低 '{target_col}' 的重复率", type="primary"):
                    original_len = len(df)
                    cleaned_df = cleaner.reduce_feature_repetition(target_col, target_rate)
                    new_len = len(cleaned_df)
                    st.session_state.processed_data = cleaned_df

                    st.success(f"✅ 优化完成！删除了 {original_len - new_len} 个样本")
                    st.info(f"📊 当前行数: {new_len}，'{target_col}' 的众数占比已调整至 {target_rate * 100:.1f}%")
                    st.rerun()
            else:
                st.success("✅ 未检测到高重复率特征")

    with tab4:
        st.markdown("### 数据类型诊断")

        pseudo_numeric = cleaner.detect_pseudo_numeric_columns()

        if pseudo_numeric:
            st.warning(f"检测到 {len(pseudo_numeric)} 个伪数值列")
            st.json(pseudo_numeric)

            if st.button("🔧 修复伪数值列", type="primary"):
                cleaned_df = cleaner.fix_pseudo_numeric_columns()
                st.session_state.processed_data = cleaned_df
                st.success("✅ 数据类型修复完成")
        else:
            st.success("✅ 数据类型正常")

    with tab5:
        st.markdown("### ⚖️ 类别平衡 (针对化学结构)")
        st.info(
            "💡 解决特定单体/分子重复次数过多的问题。通过限制每个类别的最大样本数，强制数据分布更均匀，避免模型偏向常见分子。")

        # 1. 选择分类列
        # 默认尝试找 'smiles' 相关列
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            cat_col = st.selectbox("选择要平衡的类别列 (通常是SMILES)", text_cols)

            # 2. 分析当前分布
            counts = df[cat_col].value_counts()
            n_unique = len(counts)

            col1, col2, col3 = st.columns(3)
            col1.metric("唯一类别数", n_unique)
            col2.metric("最大样本数", counts.max())
            col3.metric("中位数样本数", int(counts.median()))

            st.markdown("#### Top 10 出现最频繁的分子")
            st.bar_chart(counts.head(10))

            # 3. 设置平衡参数
            st.markdown("#### 🔧 平衡设置")

            limit_val = st.slider(
                "每个类别的最大样本数 (Max Samples per Category)",
                min_value=1,
                max_value=int(counts.max()),
                value=int(counts.median()) if n_unique > 0 else 10,
                help="如果某分子的出现次数超过此值，多余的样本将被随机丢弃。"
            )

            if st.button(f"⚖️ 执行平衡 (限制为 {limit_val} 个)", type="primary"):
                old_len = len(df)
                cleaned_df = cleaner.balance_category_counts(cat_col, max_samples=limit_val)
                new_len = len(cleaned_df)

                st.session_state.processed_data = cleaned_df

                st.success(f"✅ 平衡完成！")
                st.info(f"📊 总样本数从 {old_len} 减少到 {new_len} (删除了 {old_len - new_len} 个过度重复样本)")
                st.rerun()
        else:
            st.warning("⚠️ 没有找到文本列，无法执行类别平衡")


# ============================================================
# 页面：数据增强
# ============================================================
def page_data_enhancement():
    """数据增强页面"""
    st.title("✨ 数据增强")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    enhancer = DataEnhancer(df)

    tab1, tab2 = st.tabs(["🔮 KNN智能填充", "🧬 VAE生成式增强"])

    with tab1:
        st.markdown("### KNN智能填充")
        st.info("使用K近邻算法预测并填充缺失值，比简单的均值/中位数填充更准确")

        n_neighbors = st.slider("K值（近邻数量）", 1, 20, 5)

        if st.button("🔧 执行KNN填充", type="primary"):
            with st.spinner("正在执行KNN填充..."):
                filled_df = enhancer.knn_impute(n_neighbors=n_neighbors)
                st.session_state.processed_data = filled_df
                st.success("✅ KNN填充完成")

                # 对比
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("原始缺失值", df.isnull().sum().sum())
                with col2:
                    st.metric("处理后缺失值", filled_df.isnull().sum().sum())

    with tab2:
        st.markdown("### VAE生成式数据增强")
        st.info("使用变分自编码器(VAE)学习数据分布，生成高保真虚拟数据点")

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("生成样本数量", 10, 1000, 100)
            latent_dim = st.slider("潜在空间维度", 4, 64, 16)
            h_dim = st.slider("隐藏层维度", 32, 256, 128)

        with col2:
            epochs = st.slider("训练轮数", 10, 500, 100)
            batch_size = st.selectbox("批大小", [16, 32, 64, 128], index=1)
            learning_rate = st.number_input("学习率", 0.0001, 0.1, 0.001, format="%.4f")

        if st.button("🚀 生成增强数据", type="primary"):
            try:
                with st.spinner("正在训练VAE模型..."):
                    progress_bar = st.progress(0)

                    generated_df, fig = enhancer.generate_with_vae(
                        n_samples=n_samples,
                        latent_dim=latent_dim,
                        h_dim=h_dim,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=learning_rate
                    )

                    progress_bar.progress(100)

                st.success(f"✅ 成功生成 {len(generated_df)} 个样本")

                # PCA可视化
                st.markdown("### 📊 PCA可视化对比")
                st.plotly_chart(fig, use_container_width=True)

                # 合并选项
                if st.checkbox("将生成数据合并到原始数据"):
                    merged_df = pd.concat([df, generated_df], ignore_index=True)
                    st.session_state.processed_data = merged_df
                    st.success(f"✅ 合并后数据: {merged_df.shape}")

            except Exception as e:
                st.error(f"❌ 生成失败: {str(e)}")


# ============================================================
# 页面：分子特征提取（完整5种方法）
# ============================================================
def page_molecular_features():
    """分子特征提取页面 - 完整还原5种方法 + 分子指纹 (适配双组分)"""
    st.title("🧬 分子特征提取")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    # 优先使用处理后的数据
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data

    # 检测SMILES列
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    smiles_candidates = [col for col in text_cols if 'smiles' in col.lower() or 'smi' in col.lower()]

    if not text_cols:
        st.warning("⚠️ 数据中未检测到文本列，无法提取分子特征")
        return

    st.markdown("### 🔬 SMILES列选择")

    col1, col2 = st.columns(2)
    with col1:
        default_idx = 0
        if smiles_candidates:
            default_idx = text_cols.index(smiles_candidates[0])

        # 这里明确这是第一组分（通常是树脂）
        smiles_col = st.selectbox(
            "选择包含SMILES的列 (树脂/主体)",
            text_cols,
            index=default_idx
        )

    with col2:
        st.markdown("**示例SMILES:**")
        samples = df[smiles_col].dropna().head(3).tolist()
        for s in samples:
            st.code(s[:50] + "..." if len(str(s)) > 50 else s)

    st.markdown("---")

    # 🔥 核心功能：5种提取方法选择
    st.markdown("### 🛠️ 提取方法选择")

    extraction_method = st.radio(
        "选择分子特征提取方法",
        [
            "👆 分子指纹 (MACCS/Morgan) [新]",
            "🔹 RDKit 标准版 (推荐新手)",
            "🚀 RDKit 并行版 (大数据集)",
            "💾 RDKit 内存优化版 (低内存)",
            "🔬 Mordred 描述符 (1600+特征)",
            "🕸️ 图神经网络特征 (拓扑结构)",
            "⚛️ ML力场特征 (ANI能量/力)",
            "⚗️ 环氧树脂反应特征 (基于领域知识)"
        ],
        help="不同方法适用于不同场景"
    )

    # UI 变量初始化
    fp_type = "MACCS"
    fp_bits = 2048
    fp_radius = 2
    hardener_col = None  # 初始化固化剂列变量
    phr_col = None

    # ============== [UI 修改] 指纹参数设置 ==============
    if "分子指纹" in extraction_method:
        st.info("💡 提示：对于环氧树脂体系，建议同时选择树脂和固化剂列，系统将自动拼接两者的指纹以描述完整网络结构。")

        col_fp1, col_fp2, col_fp3 = st.columns(3)
        with col_fp1:
            fp_type = st.selectbox("指纹类型", ["MACCS", "Morgan"])

        if fp_type == "Morgan":
            with col_fp2:
                fp_radius = st.selectbox("半径 (Radius)", [2, 3, 4], index=0)
            with col_fp3:
                fp_bits = st.selectbox("位长 (Bits)", [1024, 2048, 4096], index=1)

        # [新增] 双组分选择 UI
        st.markdown("#### 双组分设置 (推荐)")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            # 排除已选的树脂列，避免重复选择
            candidate_cols = ["无 (仅提取单列)"] + [c for c in text_cols if c != smiles_col]
            hardener_col_opt = st.selectbox("选择【固化剂】SMILES列", candidate_cols)

            if hardener_col_opt != "无 (仅提取单列)":
                hardener_col = hardener_col_opt

    # ============== [UI] 环氧树脂特征参数 ==============
    if "环氧树脂反应特征" in extraction_method:
        st.info("💡 该方法需要同时提供【树脂】和【固化剂】的SMILES结构。")
        col_h, col_p = st.columns(2)
        with col_h:
            candidate_cols = [c for c in text_cols if c != smiles_col]
            hardener_col = st.selectbox("选择【固化剂】SMILES列", candidate_cols)
        with col_p:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            phr_col = st.selectbox("选择【配比/PHR】列 (可选)", ["无 (假设理想配比)"] + num_cols)

    # 并行版参数
    if "并行版" in extraction_method and OPTIMIZED_EXTRACTOR_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            n_jobs = st.slider("并行进程数", 1, mp.cpu_count(), mp.cpu_count() // 2)
        with col2:
            batch_size = st.number_input("批处理大小", 100, 5000, 1000)

    st.markdown("---")

    # 执行提取
    if st.button("🚀 开始提取分子特征", type="primary"):
        smiles_list = df[smiles_col].tolist()

        # 准备固化剂列表
        hardener_list = None
        if hardener_col:
            hardener_list = df[hardener_col].tolist()

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # --- [逻辑修改] 分发提取任务 ---
            if "分子指纹" in extraction_method:
                from core.molecular_features import FingerprintExtractor

                # 提示用户当前模式
                mode_str = "双组分拼接" if hardener_list else "单组分"
                status_text.text(f"正在提取 {fp_type} 指纹 ({mode_str}模式)...")

                extractor = FingerprintExtractor()
                # 传入 smiles_list_2 (固化剂)
                features_df, valid_indices = extractor.smiles_to_fingerprints(
                    smiles_list,
                    smiles_list_2=hardener_list,
                    fp_type=fp_type, n_bits=fp_bits, radius=fp_radius
                )

            elif "标准版" in extraction_method:
                status_text.text("正在使用RDKit标准版提取...")
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "并行版" in extraction_method:
                if OPTIMIZED_EXTRACTOR_AVAILABLE:
                    status_text.text(f"正在使用RDKit并行版提取 ({n_jobs}进程)...")
                    extractor = OptimizedRDKitFeatureExtractor(n_jobs=n_jobs, batch_size=batch_size)
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)
                else:
                    st.warning("并行版不可用，回退到标准版")
                    extractor = AdvancedMolecularFeatureExtractor()
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "内存优化版" in extraction_method:
                status_text.text("正在使用RDKit内存优化版...")
                extractor = MemoryEfficientRDKitExtractor()
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "Mordred" in extraction_method:
                status_text.text("正在使用Mordred提取...")
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_mordred(smiles_list)

            elif "图神经网络" in extraction_method:
                status_text.text("正在提取图结构特征...")
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_graph_features(smiles_list)

            elif "ML力场" in extraction_method:
                from core.molecular_features import MLForceFieldExtractor
                status_text.text("正在计算ANI力场特征...")
                extractor = MLForceFieldExtractor()
                if not extractor.AVAILABLE:
                    st.error("TorchANI 未安装")
                    return
                features_df, valid_indices = extractor.smiles_to_ani_features(smiles_list)

            elif "环氧树脂" in extraction_method:
                from core.molecular_features import EpoxyDomainFeatureExtractor
                status_text.text("正在计算环氧树脂领域特征...")
                if hardener_col is None:
                    st.error("请选择固化剂列！")
                    return

                phr_list = None
                if phr_col and phr_col != "无 (假设理想配比)":
                    phr_list = df[phr_col].tolist()

                extractor = EpoxyDomainFeatureExtractor()
                features_df, valid_indices = extractor.extract_features(smiles_list, hardener_list, phr_list)

            progress_bar.progress(100)

            # --- 合并结果逻辑 (保持不变) ---
            if len(features_df) > 0:
                st.session_state.molecular_features = features_df
                prefix = f"{smiles_col}_"
                features_df = features_df.add_prefix(prefix)

                df_valid = df.iloc[valid_indices].reset_index(drop=True)
                features_df = features_df.reset_index(drop=True)

                # 防止列名冲突：如果新特征名已存在，先删除旧的
                cols_to_drop = [col for col in features_df.columns if col in df_valid.columns]
                if cols_to_drop:
                    df_valid = df_valid.drop(columns=cols_to_drop)

                merged_df = pd.concat([df_valid, features_df], axis=1)
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                st.session_state.processed_data = merged_df

                st.success(f"✅ 成功提取 {len(features_df)} 个样本的 {features_df.shape[1]} 个分子特征")

                # 结果统计
                col1, col2, col3 = st.columns(3)
                col1.metric("有效样本", len(valid_indices))
                col2.metric("特征数量", features_df.shape[1])
                col3.metric("双组分模式", "是" if hardener_list else "否")

                st.markdown("### 📋 特征预览")
                st.dataframe(features_df.head(), use_container_width=True)
            else:
                st.error("❌ 未能提取任何特征，请检查SMILES格式")

        except Exception as e:
            st.error(f"❌ 提取失败: {str(e)}")
            st.code(traceback.format_exc())


# ============================================================
# 页面：特征选择（完整版）
# ============================================================
def page_feature_selection():
    """特征选择页面 - 调用完整的show_robust_feature_selection"""
    st.title("🎯 特征选择")

    # 调用完整的特征选择UI
    show_robust_feature_selection()


# ============================================================
# 页面：模型训练（完整手动调参）
# ============================================================
def page_model_training():
    """模型训练页面 - 完整手动调参界面"""
    st.title("🤖 模型训练")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ 请先在特征选择页面选择特征和目标变量")
        return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    feature_cols = st.session_state.feature_cols
    target_col = st.session_state.target_col

    # 准备数据
    X = df[feature_cols]
    y = df[target_col]

    # 显示当前配置
    col1, col2, col3 = st.columns(3)
    col1.metric("特征数量", len(feature_cols))
    col2.metric("样本数量", len(df))
    col3.metric("目标变量", target_col)

    st.markdown("---")

    # 模型选择
    trainer = EnhancedModelTrainer()
    available_models = trainer.get_available_models()

    # 添加人工神经网络选项
    if ANN_AVAILABLE and "人工神经网络" not in available_models:
        available_models.append("人工神经网络")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📦 模型选择")
        selected_model = st.selectbox(
            "选择模型",
            available_models,
            help="选择要训练的机器学习模型"
        )

        st.markdown("### ⚙️ 训练设置")
        test_size = st.slider("测试集比例", 0.1, 0.4, DEFAULT_TEST_SIZE)
        random_state = st.number_input("随机种子", 0, 1000, DEFAULT_RANDOM_STATE)

    with col2:
        st.markdown("### 🎛️ 手动调参")

        # 🔥 核心功能：动态生成手动调参界面
        manual_params = {}

        if selected_model in MANUAL_TUNING_PARAMS:
            param_configs = MANUAL_TUNING_PARAMS[selected_model]

            if param_configs:
                st.info(f"为 **{selected_model}** 配置超参数")

                # 创建参数输入控件
                param_cols = st.columns(2)

                for i, config in enumerate(param_configs):
                    with param_cols[i % 2]:
                        param_name = config['name']
                        param_label = config['label']
                        widget_type = config['widget']
                        default_val = config['default']
                        args = config.get('args', {})

                        # 根据widget类型创建控件
                        if widget_type == 'slider':
                            manual_params[param_name] = st.slider(
                                param_label,
                                value=default_val,
                                key=f"param_{selected_model}_{param_name}",
                                **args
                            )
                        elif widget_type == 'number_input':
                            manual_params[param_name] = st.number_input(
                                param_label,
                                value=default_val,
                                key=f"param_{selected_model}_{param_name}",
                                **args
                            )
                        elif widget_type == 'selectbox':
                            options = args.get('options', [])
                            default_idx = options.index(default_val) if default_val in options else 0
                            manual_params[param_name] = st.selectbox(
                                param_label,
                                options=options,
                                index=default_idx,
                                key=f"param_{selected_model}_{param_name}"
                            )
                        elif widget_type == 'text_input':
                            manual_params[param_name] = st.text_input(
                                param_label,
                                value=default_val,
                                key=f"param_{selected_model}_{param_name}"
                            )
            else:
                st.info(f"**{selected_model}** 无需配置参数")

        # 显示当前参数
        if manual_params:
            st.markdown("**当前参数配置:**")
            st.json(manual_params)

    st.markdown("---")

    # 训练按钮
    if st.button("🚀 开始训练模型", type="primary"):
        try:
            with st.spinner(f"正在训练 {selected_model}..."):
                # 合并默认参数和手动参数
                final_params = MODEL_PARAMETERS.get(selected_model, {}).copy()
                final_params.update(manual_params)

                # 处理特殊参数
                if selected_model == "多层感知器" and 'hidden_layer_sizes' in final_params:
                    if isinstance(final_params['hidden_layer_sizes'], str):
                        try:
                            final_params['hidden_layer_sizes'] = tuple(
                                int(x.strip()) for x in final_params['hidden_layer_sizes'].split(',')
                            )
                        except:
                            final_params['hidden_layer_sizes'] = (100, 50)
                if 'random_state' in final_params:
                    final_params.pop('random_state')
                # 训练模型
                result = trainer.train_model(
                    X, y,
                    model_name=selected_model,
                    test_size=test_size,
                    random_state=random_state,
                    **final_params
                )

                # 保存结果
                st.session_state.model = result['model']
                st.session_state.model_name = selected_model
                st.session_state.train_result = result
                st.session_state.scaler = result.get('scaler')
                st.session_state.pipeline = result.get('pipeline')
                st.session_state.X_train = result['X_train']
                st.session_state.X_test = result['X_test']
                st.session_state.y_train = result['y_train']
                st.session_state.y_test = result['y_test']

                st.success(f"✅ 模型训练完成！")

                # 显示结果
                st.markdown("### 📊 训练结果")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² 分数", f"{result['r2']:.4f}")
                col2.metric("RMSE", f"{result['rmse']:.4f}")
                col3.metric("MAE", f"{result['mae']:.4f}")
                col4.metric("训练时间", f"{result['train_time']:.2f}秒")

                # 可视化
                visualizer = Visualizer()

                # 优先使用新风格绘图
                if 'y_pred_train' in result:
                    st.markdown("### 📈 实验值 vs 预测值")
                    fig = visualizer.plot_parity_train_test(
                        y_train=result['y_train'],
                        y_pred_train=result['y_pred_train'],
                        y_test=result['y_test'],
                        y_pred_test=result['y_pred'],
                        target_name=target_col
                    )
                    st.pyplot(fig)
                else:
                    # 回退旧版（防止未更新 trainer 导致报错）
                    fig, export_df = visualizer.plot_predictions_vs_true(
                        result['y_test'],
                        result['y_pred'],
                        selected_model
                    )
                    st.pyplot(fig)

                plt.close()

        except Exception as e:
            st.error(f"❌ 训练失败: {str(e)}")
            st.code(traceback.format_exc())


# ============================================================
# 页面：模型解释（完整版）
# ============================================================
def page_model_interpretation():
    """模型解释页面"""
    st.title("📊 模型解释")

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        return

    model = st.session_state.model
    model_name = st.session_state.model_name
    result = st.session_state.train_result

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 SHAP分析", "📈 预测性能", "📉 学习曲线", "🎯 特征重要性", "💾 数据导出"
    ])

    with tab1:
        st.markdown("### SHAP特征重要性分析")

        try:
            X_test = st.session_state.X_test

            # 采样用于SHAP计算
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42) if len(X_test) > sample_size else X_test

            interpreter = ModelInterpreter(model, X_sample, model_name)

            col1, col2 = st.columns(2)
            with col1:
                plot_type = st.selectbox("图表类型", ["bar", "beeswarm"])
            with col2:
                max_display = st.slider("显示特征数", 5, 30, 15)

            if st.button("🔍 计算SHAP值"):
                with st.spinner("正在计算SHAP值..."):
                    fig = interpreter.plot_summary(X_sample, plot_type=plot_type, max_display=max_display)
                    if fig:
                        st.pyplot(fig)
                        plt.close()
        except Exception as e:
            st.error(f"SHAP分析失败: {str(e)}")

    with tab2:
        st.markdown("### 预测性能可视化")

        visualizer = Visualizer()

        # 预测值 vs 真实值
        fig1, export_df = visualizer.plot_predictions_vs_true(
            result['y_test'],
            result['y_pred'],
            model_name
        )
        st.pyplot(fig1)
        plt.close()

        # 残差分析
        fig2 = visualizer.plot_residuals(
            result['y_test'],
            result['y_pred'],
            model_name
        )
        st.pyplot(fig2)
        plt.close()

    with tab3:
        st.markdown("### 学习曲线")

        try:
            from sklearn.model_selection import learning_curve

            X = st.session_state.X_train
            y = st.session_state.y_train

            if st.button("📉 生成学习曲线"):
                with st.spinner("正在计算学习曲线..."):
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X, y,
                        cv=5,
                        n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='r2'
                    )

                    fig, ax = plt.subplots(figsize=(10, 6))

                    train_mean = train_scores.mean(axis=1)
                    train_std = train_scores.std(axis=1)
                    test_mean = test_scores.mean(axis=1)
                    test_std = test_scores.std(axis=1)

                    ax.plot(train_sizes, train_mean, 'o-', label='训练集')
                    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

                    ax.plot(train_sizes, test_mean, 'o-', label='验证集')
                    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

                    ax.set_xlabel('训练样本数')
                    ax.set_ylabel('R² 分数')
                    ax.set_title('学习曲线')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close()
        except Exception as e:
            st.error(f"学习曲线生成失败: {str(e)}")

    with tab4:
        st.markdown("### 特征重要性")

        # 尝试获取特征重要性
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = st.session_state.feature_cols

                importance_df = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': importances
                }).sort_values('重要性', ascending=False)

                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))

                top_n = min(20, len(importance_df))
                top_features = importance_df.head(top_n)

                ax.barh(range(top_n), top_features['重要性'].values[::-1])
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(top_features['特征'].values[::-1])
                ax.set_xlabel('重要性')
                ax.set_title(f'{model_name} - 特征重要性 (Top {top_n})')

                st.pyplot(fig)
                plt.close()

                st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("该模型不支持直接获取特征重要性，请使用SHAP分析")
        except Exception as e:
            st.error(f"特征重要性获取失败: {str(e)}")

    with tab5:
        st.markdown("### 导出预测结果")

        export_df = pd.DataFrame({
            '真实值': result['y_test'],
            '预测值': result['y_pred'],
            '残差': result['y_test'] - result['y_pred']
        })

        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 下载预测结果CSV",
            csv,
            f"predictions_{model_name}.csv",
            "text/csv"
        )


# ============================================================
# 页面：预测应用
# ============================================================
def page_prediction():
    """预测应用页面"""
    st.title("🔮 预测应用")

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        return

    model = st.session_state.model
    model_name = st.session_state.model_name
    feature_cols = st.session_state.feature_cols
    scaler = st.session_state.scaler

    tab1, tab2, tab3 = st.tabs(["📝 手动输入", "📁 批量预测", "🎯 适用域分析"])

    with tab1:
        st.markdown("### 手动输入特征值")

        input_values = {}
        cols = st.columns(3)

        for i, feature in enumerate(feature_cols):
            with cols[i % 3]:
                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.4f",
                    key=f"input_{feature}"
                )

        if st.button("🔮 预测", type="primary"):
            try:
                input_df = pd.DataFrame([input_values])

                # 使用pipeline或直接预测
                if st.session_state.pipeline is not None:
                    prediction = st.session_state.pipeline.predict(input_df)
                else:
                    prediction = model.predict(input_df)

                st.success(f"### 预测结果: **{prediction[0]:.4f}**")

            except Exception as e:
                st.error(f"预测失败: {str(e)}")

    with tab2:
        st.markdown("### 批量预测")

        uploaded_file = st.file_uploader("上传待预测数据", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_excel(uploaded_file)

                st.dataframe(new_df.head(), use_container_width=True)

                # 检查特征列
                missing_cols = set(feature_cols) - set(new_df.columns)
                if missing_cols:
                    st.error(f"缺少特征列: {missing_cols}")
                else:
                    if st.button("🚀 执行批量预测"):
                        X_new = new_df[feature_cols]

                        if st.session_state.pipeline is not None:
                            predictions = st.session_state.pipeline.predict(X_new)
                        else:
                            predictions = model.predict(X_new)

                        new_df['预测值'] = predictions
                        st.dataframe(new_df, use_container_width=True)

                        # 下载
                        csv = new_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 下载预测结果",
                            csv,
                            "batch_predictions.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"加载失败: {str(e)}")

    with tab3:
        st.markdown("### 适用域分析")
        st.info("分析新样本是否在模型训练数据的适用范围内")

        if st.session_state.X_train is not None and scaler is not None:
            try:
                X_train = st.session_state.X_train

                # 创建分析器
                analyzer = ApplicabilityDomainAnalyzer(X_train)

                st.markdown("#### 输入待分析样本")

                input_values = {}
                cols = st.columns(3)
                for i, feature in enumerate(feature_cols):
                    with cols[i % 3]:
                        input_values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.4f",
                            key=f"ad_input_{feature}"
                        )

                if st.button("🎯 分析适用域"):
                    input_df = pd.DataFrame([input_values])
                    is_in_domain, fig = analyzer.analyze(input_df, scaler)

                    if is_in_domain:
                        st.success("✅ 样本在模型适用域内，预测结果可靠")
                    else:
                        st.warning("⚠️ 样本超出模型适用域，预测结果可能不可靠")

                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"适用域分析失败: {str(e)}")
        else:
            st.warning("需要训练数据和scaler才能进行适用域分析")


# ============================================================
# 页面：超参优化
# ============================================================
def page_hyperparameter_optimization():
    """超参数优化页面"""
    st.title("⚙️ 超参数优化")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ 请先在特征选择页面选择特征和目标变量")
        return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    feature_cols = st.session_state.feature_cols
    target_col = st.session_state.target_col

    X = df[feature_cols]
    y = df[target_col]

    st.markdown("### Optuna智能超参数优化")

    col1, col2 = st.columns(2)

    with col1:
        trainer = EnhancedModelTrainer()
        available_models = trainer.get_available_models()

        # 支持优化的模型
        optimizable_models = [
            "随机森林", "XGBoost", "LightGBM", "CatBoost",
            "SVR", "Ridge回归", "Lasso回归", "ElasticNet",
            "AdaBoost", "梯度提升树"
        ]
        optimizable_models = [m for m in optimizable_models if m in available_models]

        model_name = st.selectbox("选择模型", optimizable_models)

    with col2:
        n_trials = st.slider("优化轮数", 10, 200, DEFAULT_OPTUNA_TRIALS)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)

    if st.button("🚀 开始优化", type="primary"):
        try:
            optimizer = HyperparameterOptimizer()

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner(f"正在优化 {model_name}..."):
                best_params, best_score, study = optimizer.optimize(
                    model_name, X, y,
                    n_trials=n_trials,
                    cv=cv_folds
                )

            progress_bar.progress(100)

            st.success(f"✅ 优化完成！最佳R²分数: {best_score:.4f}")

            st.markdown("### 最佳参数")
            st.json(best_params)

            # 优化历史可视化
            if study is not None:
                st.markdown("### 优化历史")

                try:
                    import plotly.graph_objects as go

                    trials = study.trials
                    values = [t.value for t in trials if t.value is not None]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=values,
                        mode='lines+markers',
                        name='R² Score'
                    ))
                    fig.update_layout(
                        title='优化过程',
                        xaxis_title='Trial',
                        yaxis_title='R² Score'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

            # 使用最佳参数训练
            if st.button("🎯 使用最佳参数训练模型"):
                result = trainer.train_model(
                    X, y,
                    model_name=model_name,
                    **best_params
                )

                st.session_state.model = result['model']
                st.session_state.model_name = model_name
                st.session_state.train_result = result
                st.session_state.best_params = best_params

                st.success(f"✅ 模型训练完成！R²: {result['r2']:.4f}")

        except Exception as e:
            st.error(f"❌ 优化失败: {str(e)}")
            st.code(traceback.format_exc())


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    page = render_sidebar()

    if page == "🏠 首页":
        page_home()
    elif page == "📤 数据上传":
        page_data_upload()
    elif page == "🔍 数据探索":
        page_data_explore()
    elif page == "🧹 数据清洗":
        page_data_cleaning()
    elif page == "✨ 数据增强":
        page_data_enhancement()
    elif page == "🧬 分子特征":
        page_molecular_features()
    elif page == "🎯 特征选择":
        page_feature_selection()
    elif page == "🤖 模型训练":
        page_model_training()
    elif page == "📊 模型解释":
        page_model_interpretation()
    elif page == "🔮 预测应用":
        page_prediction()
    elif page == "⚙️ 超参优化":
        page_hyperparameter_optimization()


if __name__ == "__main__":
    main()