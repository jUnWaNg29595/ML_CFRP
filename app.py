# -*- coding: utf-8 -*-
"""
碳纤维复合材料智能预测平台 v1.2.0
(主应用程序 - 修复版)
"""

# [关键修复] 移除顶层可能导致死锁的导入，改为安全检查
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
# import torch  <-- [安全] 移除顶层 torch，防止 Windows 多进程死锁
import traceback
import io
import multiprocessing as mp
import warnings
import psutil

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
    from core.molecular_features import OptimizedRDKitFeatureExtractor, MemoryEfficientRDKitExtractor

    OPTIMIZED_EXTRACTOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_EXTRACTOR_AVAILABLE = False

try:
    from core.ann_model import ANNRegressor

    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False


@st.cache_data(ttl=3600, show_spinner="正在读取数据文件...")
def load_data_file(uploaded_file):
    """带缓存的数据加载函数"""
    uploaded_file.seek(0)
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


# --- 自定义 CSS 样式 ---
CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 8px 0; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; }
    .feature-badge {
        display: inline-block; background: #E0E7FF; color: #4338CA;
        padding: 4px 12px; border-radius: 16px; font-size: 0.85rem; margin: 2px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
def init_session_state():
    defaults = {
        'data': None, 'processed_data': None, 'molecular_features': None,
        'target_col': None, 'feature_cols': [],
        'model': None, 'model_name': None, 'train_result': None,
        'scaler': None, 'pipeline': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'optimization_history': [], 'best_params': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# 侧边栏渲染
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.title(f"🔬 {APP_NAME}")
        st.caption(f"版本 {VERSION}")
        st.markdown("---")

        page = st.radio(
            "📌 功能导航",
            ["🏠 首页", "📤 数据上传", "🔍 数据探索", "🧹 数据清洗", "✨ 数据增强",
             "🧬 分子特征", "🎯 特征选择", "🤖 模型训练", "📊 模型解释",
             "🔮 预测应用", "⚙️ 超参优化"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📊 数据状态")

        current_df = st.session_state.get('processed_data')
        original_df = st.session_state.get('data')
        display_df = current_df if current_df is not None else original_df

        if display_df is not None:
            status_label = "✅ 当前数据 (已清洗)" if current_df is not None else "✅ 原始数据"
            st.success(f"{status_label}\n\n**{display_df.shape[0]} 行 × {display_df.shape[1]} 列**")

            if st.session_state.get('molecular_features') is not None:
                st.info(f"🧬 分子特征: {st.session_state.molecular_features.shape[1]} 个")

            if st.session_state.get('feature_cols'):
                st.info(f"🎯 已选特征: {len(st.session_state.feature_cols)} 个")
        else:
            st.warning("⚠️ 未加载数据")

        if st.session_state.model is not None:
            st.success(f"🤖 已训练: {st.session_state.model_name}")

        return page


# ============================================================
# 页面功能函数
# ============================================================

def page_home():
    st.title("🔬 碳纤维复合材料智能预测平台")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">数据处理</div><div class="metric-value">📊</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="metric-card" style="background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"><div class="metric-label">分子特征</div><div class="metric-value">🧬</div></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="metric-card" style="background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"><div class="metric-label">模型训练</div><div class="metric-value">🤖</div></div>',
            unsafe_allow_html=True)
    st.info("👋 欢迎使用！请从左侧导航栏开始操作。")


def page_data_upload():
    st.title("📤 数据上传")
    tab1, tab2 = st.tabs(["📁 上传文件", "📝 生成示例数据"])

    with tab1:
        uploaded_file = st.file_uploader("选择数据文件 (CSV/Excel)", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df = load_data_file(uploaded_file)
                # 自动去重名列
                df = df.loc[:, ~df.columns.duplicated()]
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                st.success(f"✅ 加载成功: {df.shape[0]}行 × {df.shape[1]}列")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"加载失败: {e}")

    with tab2:
        if st.button("生成混合数据集"):
            df = generate_hybrid_dataset()
            st.session_state.data = df
            st.session_state.processed_data = df.copy()
            st.success("✅ 已生成示例数据")
            st.dataframe(df.head())


def page_data_explore():
    st.title("🔍 数据探索")
    if st.session_state.data is None: return st.warning("请先上传数据")

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    explorer = EnhancedDataExplorer(df)

    tab1, tab2, tab3 = st.tabs(["📊 统计", "🔗 相关性", "📈 分布"])
    with tab1:
        st.dataframe(df.describe(), use_container_width=True)
    with tab2:
        fig = explorer.plot_correlation_matrix()
        if fig: st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = explorer.plot_distributions()
        if fig: st.plotly_chart(fig, use_container_width=True)


def page_data_cleaning():
    st.title("🧹 数据清洗")
    if st.session_state.data is None: return st.warning("请先上传数据")

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    cleaner = AdvancedDataCleaner(df)

    tab1, tab2, tab3 = st.tabs(["❓ 缺失值", "📊 异常值", "🔄 去重与优化"])

    with tab1:
        st.markdown("### 缺失值处理")
        if df.isnull().sum().sum() > 0:
            strategy = st.selectbox("填充策略", ["median", "mean", "knn", "drop_rows"])
            if st.button("执行填充"):
                st.session_state.processed_data = cleaner.handle_missing_values(strategy)
                st.success("✅ 完成")
                st.rerun()
        else:
            st.success("无缺失值")

    with tab3:
        st.markdown("### 🔄 数据去重与分布优化")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. 行去重")
            dup = df.duplicated().sum()
            st.metric("重复行", dup)
            if dup > 0 and st.button("删除重复行"):
                st.session_state.processed_data = cleaner.remove_duplicates()
                st.success("✅ 已去重")
                st.rerun()

        with col2:
            st.markdown("#### 2. 特征分布优化")
            st.caption("降低某一特征中众数的比例")
            threshold = st.slider("高重复率检测阈值", 0.5, 0.99, 0.8)
            high_rep_cols = cleaner.detect_high_repetition_columns(threshold)

            if high_rep_cols:
                target_col = st.selectbox("选择要优化的特征", list(high_rep_cols.keys()))
                target_rate = st.slider("目标占比", 0.1, 0.8, 0.5)
                if st.button(f"📉 优化 '{target_col}'"):
                    st.session_state.processed_data = cleaner.reduce_feature_repetition(target_col, target_rate)
                    st.success("✅ 优化完成")
                    st.rerun()
            else:
                st.success("未检测到高重复率特征")


def page_data_enhancement():
    st.title("✨ 数据增强")
    if st.session_state.data is None: return st.warning("请先上传数据")

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    enhancer = DataEnhancer(df)

    if st.button("执行 KNN 智能填充", type="primary"):
        st.session_state.processed_data = enhancer.knn_impute()
        st.success("✅ 填充完成")


def page_molecular_features():
    st.title("🧬 分子特征提取")
    if st.session_state.data is None: return st.warning("请先上传数据")

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not text_cols: return st.error("未检测到 SMILES 列")

    smiles_col = st.selectbox("选择 SMILES 列", text_cols)

    # [修复] 逗号修复，确保选项独立
    method_options = [
        "🔹 RDKit 标准版 (推荐新手)",
        "🚀 RDKit 并行版 (大数据集)",
        "💾 RDKit 内存优化版 (低内存)",
        "🔬 Mordred 描述符 (1600+特征)",
        "🕸️ 图神经网络特征 (拓扑结构)",
        "⚛️ ML力场特征 (ANI能量/力)",
        "⚗️ 环氧树脂反应特征 (基于领域知识)"
    ]

    extraction_method = st.radio("选择提取方法", method_options)

    # [新增] 环氧树脂专用UI
    hardener_col = None
    phr_col = None
    if "环氧树脂" in extraction_method:
        st.info("需提供树脂和固化剂结构。")
        col_h, col_p = st.columns(2)
        with col_h:
            hardener_col = st.selectbox("选择【固化剂】列", [c for c in text_cols if c != smiles_col])
        with col_p:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            phr_col = st.selectbox("选择【PHR/配比】列 (可选)", ["无 (默认1:1)"] + num_cols)

    if st.button("🚀 开始提取", type="primary"):
        smiles_list = df[smiles_col].tolist()
        status_text = st.empty()
        status_text.text("正在提取...")

        try:
            features_df = pd.DataFrame()
            valid_indices = []

            if "标准版" in extraction_method:
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "并行版" in extraction_method:
                # 自动降级处理，防止 Windows 死锁
                if OPTIMIZED_EXTRACTOR_AVAILABLE:
                    extractor = OptimizedRDKitFeatureExtractor()
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)
                else:
                    st.warning("并行版不可用，使用标准版")
                    extractor = AdvancedMolecularFeatureExtractor()
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "内存优化" in extraction_method:
                extractor = MemoryEfficientRDKitExtractor()
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list)

            elif "Mordred" in extraction_method:
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_mordred(smiles_list)

            elif "图神经网络" in extraction_method:
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_graph_features(smiles_list)

            elif "ML力场" in extraction_method:
                from core.molecular_features import MLForceFieldExtractor
                status_text.text("正在计算ANI力场特征 (单线程稳定模式)...")
                extractor = MLForceFieldExtractor()
                if not extractor.AVAILABLE:
                    st.error("TorchANI 未安装或初始化失败")
                    return
                features_df, valid_indices = extractor.smiles_to_ani_features(smiles_list)

            elif "环氧树脂" in extraction_method:
                from core.molecular_features import EpoxyDomainFeatureExtractor
                status_text.text("计算 EEW、交联密度等物理特征...")

                if not hardener_col: return st.error("需选择固化剂列")

                r_list = df[smiles_col].tolist()
                h_list = df[hardener_col].tolist()
                p_list = df[phr_col].tolist() if phr_col != "无 (默认1:1)" else None

                extractor = EpoxyDomainFeatureExtractor()
                features_df, valid_indices = extractor.extract_features(r_list, h_list, p_list)

            if not features_df.empty:
                st.session_state.molecular_features = features_df
                # 合并数据
                features_df = features_df.add_prefix(f"{smiles_col}_")
                df_valid = df.iloc[valid_indices].reset_index(drop=True)
                features_df = features_df.reset_index(drop=True)

                # 智能去重合并
                cols_to_drop = [c for c in features_df.columns if c in df_valid.columns]
                if cols_to_drop: df_valid = df_valid.drop(columns=cols_to_drop)

                merged = pd.concat([df_valid, features_df], axis=1)
                st.session_state.processed_data = merged
                st.success(f"✅ 提取成功: {features_df.shape[1]} 个特征")
                st.dataframe(features_df.head())
            else:
                st.error("❌ 提取失败或无有效结果")

        except Exception as e:
            st.error(f"错误: {e}")
            st.code(traceback.format_exc())


def page_feature_selection():
    show_robust_feature_selection()


def page_model_training():
    st.title("🤖 模型训练")
    if st.session_state.data is None: return st.warning("请先上传数据")
    if not st.session_state.feature_cols: return st.warning("请先选择特征")

    df = st.session_state.processed_data
    X = df[st.session_state.feature_cols]
    y = df[st.session_state.target_col]

    col1, col2 = st.columns([1, 2])
    with col1:
        trainer = EnhancedModelTrainer()
        models = trainer.get_available_models()
        model_name = st.selectbox("选择模型", models)
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)

    if st.button("🚀 训练模型", type="primary"):
        with st.spinner("训练中..."):
            try:
                # 获取参数 (这里简化处理，实际可接手动调参面板)
                params = MODEL_PARAMETERS.get(model_name, {})
                result = trainer.train_model(X, y, model_name, test_size=test_size, **params)

                st.session_state.model = result['model']
                st.session_state.model_name = model_name
                st.session_state.train_result = result
                st.session_state.X_train = result['X_train']

                st.success(f"✅ 训练完成! R²: {result['r2']:.4f}")

                # [可视化] 优先使用 Parity Plot (Train vs Test)
                visualizer = Visualizer()
                if 'y_pred_train' in result:
                    st.markdown("### 📈 实验值 vs 预测值")
                    fig = visualizer.plot_parity_train_test(
                        result['y_train'], result['y_pred_train'],
                        result['y_test'], result['y_pred_test'],
                        target_name=st.session_state.target_col
                    )
                    st.pyplot(fig)
                else:
                    fig, _ = visualizer.plot_predictions_vs_true(result['y_test'], result['y_pred'], model_name)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"训练失败: {e}")
                st.code(traceback.format_exc())


def page_model_interpretation():
    st.title("📊 模型解释")
    if st.session_state.model is None: return st.warning("请先训练模型")

    model = st.session_state.model
    X_test = st.session_state.train_result['X_test']

    if st.button("计算 SHAP 值"):
        try:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP 计算失败: {e}")


def page_prediction():
    st.title("🔮 预测应用")
    if st.session_state.model is None: return st.warning("请先训练模型")
    st.info("在此上传新数据进行预测...")


def page_hyperparameter_optimization():
    st.title("⚙️ 超参优化")
    st.info("Optuna 自动调参模块...")


# ============================================================
# 主入口
# ============================================================
def main():
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