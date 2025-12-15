# -*- coding: utf-8 -*-
"""
碳纤维复合材料智能预测平台 v1.3.0
更新内容：
1. 修复SHAP图表显示和特征名缺失问题
2. 优化所有图表布局，防止缩放变形
3. 为所有图表增加数据导出(CSV)功能
4. 增加双组分分子指纹拼接功能
5. 增加训练脚本一键导出功能
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
from core.model_trainer import EnhancedModelTrainer, AutoGluonWrapper  # 确保引入 Wrapper
from core.model_interpreter import ModelInterpreter, EnhancedModelInterpreter
from core.molecular_features import AdvancedMolecularFeatureExtractor, RDKitFeatureExtractor
from core.feature_selector import SmartFeatureSelector, SmartSparseDataSelector, show_robust_feature_selection
from core.optimizer import HyperparameterOptimizer, InverseDesigner, generate_tuning_suggestions
from core.visualizer import Visualizer
from core.applicability_domain import ApplicabilityDomainAnalyzer, TanimotoADAnalyzer
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


# --- [新增] 生成独立训练脚本的函数 ---
def generate_training_script_code(model_name, params, feature_cols, target_col):
    """生成独立的 Python 训练脚本"""
    script_template = f'''# -*- coding: utf-8 -*-
"""
自动生成的机器学习训练脚本
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
模型类型: {model_name}
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
try: from xgboost import XGBRegressor
except ImportError: pass
try: from lightgbm import LGBMRegressor
except ImportError: pass
try: from catboost import CatBoostRegressor
except ImportError: pass

MODEL_NAME = "{model_name}"
FEATURE_COLS = {json.dumps(feature_cols, ensure_ascii=False)}
TARGET_COL = "{target_col}"
HYPERPARAMETERS = {json.dumps(params, indent=4, ensure_ascii=False)}
DATA_PATH = "data.csv" 

def load_and_train():
    print(f"正在加载数据: {{DATA_PATH}}...")
    try:
        if DATA_PATH.endswith('.csv'): df = pd.read_csv(DATA_PATH)
        else: df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        print("❌ 错误: 找不到数据文件")
        return

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    y = pd.to_numeric(y, errors='coerce')
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"正在初始化模型: {{MODEL_NAME}}...")
    model = None
    if MODEL_NAME == "随机森林": model = RandomForestRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "XGBoost": model = XGBRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "LightGBM": model = LGBMRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "CatBoost": model = CatBoostRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "SVR": model = SVR(**HYPERPARAMETERS)
    elif MODEL_NAME == "决策树": model = DecisionTreeRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "梯度提升树": model = GradientBoostingRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "AdaBoost": model = AdaBoostRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "多层感知器": model = MLPRegressor(**HYPERPARAMETERS)
    elif MODEL_NAME == "线性回归": model = LinearRegression(**HYPERPARAMETERS)
    elif MODEL_NAME == "Ridge回归": model = Ridge(**HYPERPARAMETERS)
    elif MODEL_NAME == "Lasso回归": model = Lasso(**HYPERPARAMETERS)
    elif MODEL_NAME == "ElasticNet": model = ElasticNet(**HYPERPARAMETERS)

    if model:
        print("开始训练...")
        model.fit(X_train, y_train)
        print("正在评估...")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"训练完成！R² Score: {{r2:.4f}}")

if __name__ == "__main__":
    load_and_train()
'''
    return script_template


# --- 全局常量 ---
USER_DATA_DB = "datasets/user_data.csv"

# --- 自定义 CSS 样式 ---
# --- 自定义 CSS 样式 (含图片防抖) ---
CUSTOM_CSS = """
<style>
    :root { --primary-color: #4F46E5; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
        margin: 8px 0;
    }
    /* 图片容器高度固定，防止页面抖动 */
    div[data-testid="stImage"] { min-height: 400px; display: flex; align-items: center; justify-content: center; }
    .stPlotlyChart { min-height: 400px; }
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
        'cv_result': None,
        'scaler': None,
        'imputer': None,
        'pipeline': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'optimization_history': [],
        'best_params': None,
        'molecular_feature_names': [],
        'optimized_model_name': None  # 新增：记录优化的模型名
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
        - **类别平衡**: 解决化学单体样本不平衡问题

        ### 🧬 分子特征提取
        - **分子指纹**: MACCS Keys, Morgan (ECFP) 指纹
        - **RDKit标准版**: 200+分子描述符
        - **图神经网络特征**: 分子拓扑结构特征
        - **ML力场特征**: ANI-2x 高精度能量/力
        """)

    with col2:
        st.markdown("""
        ### 🤖 模型训练
        - **集成模型**: 随机森林、XGBoost、LightGBM、CatBoost
        - **AutoML**: AutoGluon 自动建模
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
# 页面：数据探索
# ============================================================
def page_data_explore():
    """数据探索页面"""
    st.title("🔍 数据探索")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

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
        st.markdown("### 导出数据")
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "❓ 缺失值处理", "📊 异常值检测", "🔄 重复数据", "🔧 数据类型", "🧩 SMILES组分分列", "⚖️ 类别平衡"
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

        st.markdown("---")
        st.markdown("#### 3. 🧪 按配方/键聚合重复记录（推荐：Tg / 力学性质）")
        st.caption("同一配方(或同一测试口径)的重复测量往往会引入标签噪声。对 target 做稳健聚合（如 median）可显著提升泛化稳定性。")

        all_cols = df.columns.tolist()

        # 默认聚合键：resin_smiles + curing_agent_smiles (+ tg_method)
        default_keys = []
        for k in ["resin_smiles", "curing_agent_smiles", "tg_method"]:
            if k in all_cols:
                default_keys.append(k)

        keys = st.multiselect(
            "选择聚合键（Group By）",
            options=all_cols,
            default=default_keys,
            help="建议：resin_smiles + curing_agent_smiles；如果存在 tg_method 且目标为 Tg，建议也加入 tg_method 以统一口径。"
        )

        # 默认目标：优先用已选 target，其次 tg_c
        default_target = st.session_state.get("target_col") if st.session_state.get("target_col") in all_cols else ("tg_c" if "tg_c" in all_cols else all_cols[0])
        target_col_for_agg = st.selectbox("选择需要聚合的目标列", options=all_cols, index=all_cols.index(default_target))

        agg_method = st.selectbox("聚合方式", options=["median", "mean", "min", "max"], index=0)
        dropna_target = st.checkbox("删除聚合后目标仍为空(NaN)的组", value=True)

        if keys:
            try:
                n_unique = df[keys].drop_duplicates().shape[0]
                dup_like = len(df) - n_unique
                c1, c2, c3 = st.columns(3)
                c1.metric("当前样本数", len(df))
                c2.metric("按键唯一组数", n_unique)
                c3.metric("可合并的重复记录", dup_like)
            except Exception:
                pass

        if st.button("🔄 执行聚合（生成 *_rep_n / *_rep_std）", type="primary"):
            if not keys:
                st.error("请至少选择 1 个聚合键")
            else:
                try:
                    new_df = cleaner.aggregate_by_keys(keys=keys, target_col=target_col_for_agg, agg=agg_method, dropna_target=dropna_target)
                    st.session_state.processed_data = new_df
                    st.success(f"✅ 聚合完成：{len(df)} 行 → {len(new_df)} 行")
                    st.info("已生成重复统计列：tg_rep_n / tg_rep_std（或 <target>_rep_n / <target>_rep_std）")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 聚合失败: {e}")

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

        st.markdown("---")
        st.markdown("### 🔤 One-Hot 编码（把类别列转成数值特征）")
        st.caption("适合：tg_method、树脂体系类型等类别信息。如果你希望一个模型覆盖多口径，可将 tg_method one-hot 后加入特征。")

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            default_encode = ["tg_method"] if "tg_method" in cat_cols else []
            encode_cols = st.multiselect("选择要编码的列", options=cat_cols, default=default_encode)
            drop_first = st.checkbox("drop_first（可选：避免完全共线）", value=False)

            if st.button("🔤 执行 One-Hot 编码", type="primary"):
                if not encode_cols:
                    st.error("请至少选择 1 个要编码的列")
                else:
                    try:
                        new_df = cleaner.one_hot_encode(encode_cols, drop_first=drop_first, dummy_na=False)
                        st.session_state.processed_data = new_df
                        st.success(f"✅ One-Hot 编码完成：列数 {df.shape[1]} → {new_df.shape[1]}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ One-Hot 编码失败: {e}")
        else:
            st.info("未检测到可编码的类别列")

    with tab5:
        st.markdown("### 🧩 SMILES组分自动分列（树脂/固化剂/改性剂）")
        st.info(
            "💡 将单元格内的多组分 SMILES（如 'A;B' 或 'A + B' 或 'A.B'）自动拆分到多列："
            "例如 curing_agent_smiles_1 / curing_agent_smiles_2 …。"
            "同时可选做 RDKit canonical 化，生成 *_key（配方键），方便后续类别平衡与分组划分。"
        )

        from core.smiles_utils import split_smiles_column, build_formulation_key
        import re

        text_cols_local = df.select_dtypes(include=['object', 'category']).columns.tolist()
        smiles_cols = [c for c in text_cols_local if 'smiles' in c.lower()]
        candidate_cols = smiles_cols if smiles_cols else text_cols_local

        if not candidate_cols:
            st.warning("⚠️ 未检测到可分列的文本列（object/category）。")
        else:
            # 默认优先：resin_smiles / curing_agent_smiles
            default_cols = []
            for cand in ["resin_smiles", "curing_agent_smiles", "hardener_smiles", "curing_agent", "curing_agent_smiles"]:
                if cand in candidate_cols:
                    default_cols.append(cand)
            if not default_cols:
                default_cols = [candidate_cols[0]]

            cols_to_split = st.multiselect(
                "选择要分列的列",
                options=candidate_cols,
                default=default_cols,
                help="建议至少选择 resin_smiles 与 curing_agent_smiles 两列（如果存在）。"
            )

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                max_components = st.slider("最大分列组分数", 1, 12, 6, help="每列最多拆成多少个组分（*_1~*_k）")
            with col_s2:
                canonicalize = st.checkbox("RDKit canonical 化组分（推荐）", value=True)
            with col_s3:
                keep_original = st.checkbox("保留原始列", value=True)

            add_key = st.checkbox("生成 *_key 配方键（排序去重后 '.' 拼接）", value=True)
            add_n = st.checkbox("生成 *_n_components 组分数列", value=True)

            if st.button("🧩 执行分列", type="primary"):
                new_df = df.copy()
                created_cols = []

                for c in cols_to_split:
                    new_df, new_cols = split_smiles_column(
                        new_df,
                        column=c,
                        max_components=max_components,
                        canonicalize=canonicalize,
                        add_key=add_key,
                        add_n_components=add_n,
                        keep_original=keep_original,
                        prefix=None
                    )
                    created_cols.extend(new_cols)

                # 如果同时分列了树脂与固化剂，自动生成体系配方键 formulation_key
                if add_key:
                    resin_key = None
                    hard_key = None
                    for c in cols_to_split:
                        if resin_key is None and "resin" in c.lower():
                            if f"{c}_key" in new_df.columns:
                                resin_key = f"{c}_key"
                        if hard_key is None and ("curing" in c.lower() or "hardener" in c.lower()):
                            if f"{c}_key" in new_df.columns:
                                hard_key = f"{c}_key"
                    if resin_key and hard_key:
                        new_df = build_formulation_key(
                            new_df,
                            resin_key_col=resin_key,
                            hardener_key_col=hard_key,
                            new_col="formulation_key"
                        )
                        created_cols.append("formulation_key")

                st.session_state.processed_data = new_df
                st.success(f"✅ 分列完成：新增 {len(created_cols)} 列")
                if created_cols:
                    st.caption("新增列示例（前 20 个）： " + ", ".join(created_cols[:20]) + (" ..." if len(created_cols) > 20 else ""))
                st.rerun()

            st.markdown("---")
            st.markdown("#### 🔎 分列后的类别分布快速体检")
            st.caption("分列后通常会出现 *_1 / *_2 / *_key 等列；若发现某类占比过高，可在右侧“类别平衡”页对该列执行限制。")

            preview_cols = [c for c in df.columns if c.endswith("_key") or re.search(r"_\d+$", c)]
            if preview_cols:
                prev_col = st.selectbox("选择要查看分布的列", options=preview_cols)
                vc = df[prev_col].value_counts(dropna=False)
                if len(vc) > 0:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("唯一类别数", int(len(vc)))
                    col_m2.metric("最大样本数", int(vc.max()))
                    col_m3.metric("中位数样本数", int(vc.median()))
                    st.bar_chart(vc.head(10))

                    st.markdown("##### ⚖️ 一键类别优化（可选）")
                    default_cap = int(max(1, vc.median()))
                    cap = st.slider(
                        "每个类别最大样本数",
                        min_value=1,
                        max_value=int(vc.max()),
                        value=default_cap,
                        help="将超高频的单体/配方下采样到指定上限，减少数据中“单种分子单体过多”的偏置。"
                    )
                    if st.button("⚖️ 立即对该列执行平衡", key=f"quick_balance_{prev_col}"):
                        from core.data_processor import AdvancedDataCleaner
                        cleaner_tmp = AdvancedDataCleaner(df)
                        balanced_df = cleaner_tmp.balance_category_counts(prev_col, max_samples=int(cap))
                        st.session_state.processed_data = balanced_df
                        st.success(f"✅ 已对 {prev_col} 执行类别平衡（max_samples={int(cap)}）")
                        st.rerun()

            else:
                st.info("当前数据还没有 *_key 或 *_数字 的分列列。你可以先点击上方按钮执行分列。")

    with tab6:
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

    # -----------------------------
    # 多组分/混合物 SMILES 处理
    # 说明：
    # 1) 单列里可能用 ";" 等分隔符表示多个组分（RDKit 不能直接解析 ";"，但能解析 "."）
    # 2) 也可能每个组分单独占一列（如 resin_smiles_1, resin_smiles_2 ...）
    # 这里把多个组分统一转换为“多片段 SMILES”（用 "." 连接），再交给 RDKit/指纹提取器。
    # -----------------------------
    import re

    def _split_smiles_cell(x):
        """把单元格里的 SMILES 拆成组分列表。

        仅把常见“列表分隔符”当作组分边界：;、；、|、以及带空格的 +
        注意：不把“/”当分隔符（它是 SMILES 立体化学的一部分）。
        """
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        s = str(x).strip()
        if not s or s.lower() == 'nan':
            return []
        # 统一中文分号
        s = s.replace('；', ';')

        # 先按 ; 或 | 分割
        parts = re.split(r"\s*[;|]\s*", s)

        # 再按“带空格的 +”分割（避免误伤 [N+] 这类带电荷写法）
        final = []
        for p in parts:
            final.extend(re.split(r"\s+\+\s+", p))

        # 清理空串
        final = [p.strip() for p in final if p and p.strip()]
        return final

    def _combine_components(df_in: pd.DataFrame, cols: list[str]):
        """把多列/单列 SMILES 合并成多片段 SMILES，并返回(合并后的Series, 组分数量Series)"""
        if not cols:
            return pd.Series([np.nan] * len(df_in)), pd.Series([0] * len(df_in))

        combined = []
        counts = []
        for _, row in df_in[cols].iterrows():
            comps = []
            for c in cols:
                comps.extend(_split_smiles_cell(row[c]))
            counts.append(len(comps))
            combined.append('.'.join(comps) if comps else np.nan)
        return pd.Series(combined), pd.Series(counts)

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

    # --- 多组分设置（树脂侧） ---
    st.markdown("#### 🧩 多组分/混合物设置 (可选)")
    resin_mix_mode = st.checkbox(
        "树脂为多组分（或单元格内包含多个SMILES）",
        value=False,
        help="如果你的树脂列里出现 'A;B' 这种写法，或有 resin_smiles_1/resin_smiles_2 这种多列组分，请开启。"
    )

    resin_component_cols = [smiles_col]
    resin_mix_layout = "单列"  # 仅用于 UI 记录
    add_component_count_features = False

    if resin_mix_mode:
        resin_mix_layout = st.radio(
            "树脂组分在表格中的组织方式",
            ["单列（同一单元格用分隔符表示多个组分，如 A;B）", "多列（每列一个组分，如 resin_smiles_1/resin_smiles_2…）"],
            index=0
        )

        if resin_mix_layout.startswith("多列"):
            # 自动推荐：与所选列同前缀、且以 _数字 结尾的列
            pattern = re.compile(rf"^{re.escape(smiles_col)}_\d+$")
            auto_cols = [c for c in text_cols if pattern.match(c)]
            # 按末尾数字排序
            def _tail_num(colname: str):
                try:
                    return int(colname.split('_')[-1])
                except:
                    return 0
            auto_cols = sorted(auto_cols, key=_tail_num)
            resin_component_cols = st.multiselect(
                "选择树脂组分列",
                options=text_cols,
                default=auto_cols if auto_cols else [smiles_col],
                help="系统会把这些列的所有非空组分合并为一个多片段SMILES（用 '.' 连接）"
            )
        else:
            st.caption("将自动把 ';'、'；'、'|'、以及带空格的 ' + ' 转换为多组分分隔，并用 '.' 连接。")

        add_component_count_features = st.checkbox(
            "额外加入组分数量特征（resin_n_components / hardener_n_components）",
            value=True,
            help="对很多混配体系，组分数量本身也会影响性能；此选项会把组分数作为额外数值特征并入数据集。"
        )

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
            "🧊 3D构象描述符 (RDKit3D+Coulomb) [新]",
            "🧠 预训练SMILES Transformer Embedding (ChemBERTa等) [可选]",
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
    hardener_col = None
    hardener_fusion_mode = "仅用于指纹/反应特征（当前默认）"  # 初始化固化剂列变量
    phr_col = None

    # ============== [修改] 指纹参数设置 ==============
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


        # ---- 预训练 SMILES Transformer Embedding 参数（可选）----
        lm_model_name = "seyonec/ChemBERTa-zinc-base-v1"
        lm_pooling = "cls"
        lm_max_length = 128
        lm_batch_size = 16

        if "Transformer Embedding" in extraction_method:
            st.markdown("#### 🧠 预训练SMILES Transformer Embedding 参数")
            st.info("需要先安装 transformers；首次运行会下载模型权重（需要联网）。模型输出维度通常为 768，可配合后续特征选择/降维使用。")
            lm_model_name = st.text_input("HuggingFace 模型名", value=lm_model_name)
            col_lm1, col_lm2, col_lm3 = st.columns(3)
            with col_lm1:
                lm_pooling = st.selectbox("Pooling", ["cls", "mean"], index=0)
            with col_lm2:
                lm_max_length = st.selectbox("Max Length", [64, 128, 256], index=1)
            with col_lm3:
                lm_batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)

        # [新增] 双组分选择 UI
        st.markdown("#### 双组分设置 (推荐)")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            # 排除已选的树脂列，避免重复选择
            candidate_cols = ["无 (仅提取单列)"] + [c for c in text_cols if c != smiles_col]
            hardener_col_opt = st.selectbox("选择【固化剂】SMILES列", candidate_cols)

            if hardener_col_opt != "无 (仅提取单列)":
                hardener_col = hardener_col_opt

        with col_h2:
            hardener_fusion_mode = st.selectbox(
                "固化剂融入方式",
                [
                    "仅用于指纹/反应特征（当前默认）",
                    "拼接SMILES后用于所有分子特征（Resin.Hardener）"
                ],
                index=0,
                help="选择第二项后，RDKit/Mordred/3D/ANI/Transformer 等方法将对拼接后的 SMILES 提取特征。"
            )


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

    # [修改] 按钮区域：增加清除按钮
    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        run_extraction = st.button("🚀 开始提取分子特征", type="primary")

    with col_btn2:
        if st.button("🗑️ 清除已提取特征"):
            # 检查是否有记录的特征列名
            if st.session_state.get('molecular_feature_names'):
                current_df = st.session_state.processed_data
                # 找出当前数据中实际存在的特征列
                cols_to_remove = [c for c in st.session_state.molecular_feature_names if c in current_df.columns]

                if cols_to_remove:
                    # 从 processed_data 中移除这些列
                    st.session_state.processed_data = current_df.drop(columns=cols_to_remove)
                    # 重置状态
                    st.session_state.molecular_features = None
                    st.session_state.molecular_feature_names = []
                    st.success(f"✅ 已成功清除 {len(cols_to_remove)} 个分子特征列！")
                    st.rerun()
                else:
                    st.warning("⚠️ 数据表中未找到可清除的特征列（可能已被修改）。")

            elif st.session_state.get('molecular_features') is not None:
                # 兜底逻辑：如果有特征数据但没记录列名（旧状态），强制重置状态
                st.session_state.molecular_features = None
                st.warning("⚠️ 特征状态已重置，但无法自动从数据表中移除具体列（建议重新上传数据）。")
                st.rerun()
            else:
                st.info("ℹ️ 当前没有已提取的特征。")

    # 执行提取逻辑
    if run_extraction:
        # -----------------------------
        # 1) 生成“可被 RDKit 解析”的 SMILES 列表
        #    - 单列多组分：将 ';' 等分隔符转换为 '.'
        #    - 多列多组分：合并多列为 '.' 连接的多片段 SMILES
        # -----------------------------
        resin_smiles_series, resin_ncomp = _combine_components(df, resin_component_cols)
        smiles_list = resin_smiles_series.tolist()

        # 2) 固化剂（可选）——同样支持多组分
        hardener_list = None
        hardener_ncomp = None
        if hardener_col:
            # 如果用户选择了 curing_agent_smiles 这类列，同时存在 curing_agent_smiles_1/2/…，则给出多列模式
            hardener_component_cols = [hardener_col]
            if resin_mix_mode:
                # 仅在启用多组分模式时才展示/使用固化剂多列合并逻辑，避免 UI 过复杂
                st.caption("（提示）固化剂也支持多组分：如果有 hardener_col_1/2/… 可在下方自动合并。")

            # 自动合并：如果存在 hardener_col_\d 列，则优先使用它们（用户未显式多选时）
            pattern_h = re.compile(rf"^{re.escape(hardener_col)}_\d+$")
            auto_h_cols = [c for c in text_cols if pattern_h.match(c)]
            if auto_h_cols:
                def _tail_num_h(colname: str):
                    try:
                        return int(colname.split('_')[-1])
                    except:
                        return 0
                auto_h_cols = sorted(auto_h_cols, key=_tail_num_h)
                hardener_component_cols = auto_h_cols

            hardener_smiles_series, hardener_ncomp = _combine_components(df, hardener_component_cols)
            hardener_list = hardener_smiles_series.tolist()

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # --- [新增] 固化剂融合：可选将 Resin 与 Hardener SMILES 拼接后用于所有分子特征 ---
            smiles_list_input = smiles_list
            if hardener_list and isinstance(hardener_fusion_mode, str) and hardener_fusion_mode.startswith("拼接SMILES"):
                def _safe_smiles(x):
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return ""
                    s = str(x).strip()
                    return "" if s.lower() == "nan" else s

                smiles_list_input = []
                for r, h in zip(smiles_list, hardener_list):
                    rs = _safe_smiles(r)
                    hs = _safe_smiles(h)
                    if rs and hs:
                        smiles_list_input.append(f"{rs}.{hs}")
                    elif rs:
                        smiles_list_input.append(rs)
                    elif hs:
                        smiles_list_input.append(hs)
                    else:
                        smiles_list_input.append(np.nan)

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
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list_input)

            elif "并行版" in extraction_method:
                if OPTIMIZED_EXTRACTOR_AVAILABLE:
                    status_text.text(f"正在使用RDKit并行版提取 ({n_jobs}进程)...")
                    extractor = OptimizedRDKitFeatureExtractor(n_jobs=n_jobs, batch_size=batch_size)
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list_input)
                else:
                    st.warning("并行版不可用，回退到标准版")
                    extractor = AdvancedMolecularFeatureExtractor()
                    features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list_input)

            elif "内存优化版" in extraction_method:
                status_text.text("正在使用RDKit内存优化版...")
                extractor = MemoryEfficientRDKitExtractor()
                features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list_input)

            elif "Mordred" in extraction_method:
                status_text.text("正在使用Mordred提取...")
                extractor = AdvancedMolecularFeatureExtractor()
                features_df, valid_indices = extractor.smiles_to_mordred(smiles_list_input)

            elif "3D构象" in extraction_method:
                from core.molecular_features import RDKit3DDescriptorExtractor
                status_text.text("正在提取RDKit 3D构象描述符...")
                extractor = RDKit3DDescriptorExtractor()
                features_df, valid_indices = extractor.smiles_to_3d_descriptors(smiles_list_input)

            elif "Transformer Embedding" in extraction_method:
                from core.molecular_features import SmilesTransformerEmbeddingExtractor
                status_text.text("正在加载预训练Transformer并提取Embedding...")
                extractor = SmilesTransformerEmbeddingExtractor(
                    model_name=lm_model_name,
                    pooling=lm_pooling,
                    max_length=lm_max_length
                )
                if not getattr(extractor, "AVAILABLE", False):
                    st.error("❌ 未检测到 transformers，请先安装：pip install transformers")
                    st.stop()
                features_df, valid_indices = extractor.smiles_to_embeddings(smiles_list_input, batch_size=lm_batch_size)

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
                features_df, valid_indices = extractor.smiles_to_ani_features(smiles_list_input)

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

            # --- 合并结果逻辑 ---
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

                # 可选：追加组分数量特征
                if resin_mix_mode and add_component_count_features:
                    merged_df[f"{smiles_col}_resin_n_components"] = resin_ncomp.iloc[valid_indices].reset_index(drop=True)
                    if hardener_ncomp is not None:
                        merged_df[f"{smiles_col}_hardener_n_components"] = hardener_ncomp.iloc[valid_indices].reset_index(drop=True)

                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                st.session_state.processed_data = merged_df
                # [新增] 保存特征列名到 Session State，以便后续清除
                st.session_state.molecular_feature_names = features_df.columns.tolist()
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
# 页面：模型训练（更新版：含表格、一键输出、图片防抖）
# ============================================================
def page_model_training():
    """模型训练页面（稳健版：支持分层/分组划分 + Repeated KFold CV）"""
    st.title("🤖 模型训练")

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ 请先在特征选择页面选择特征")
        return

    # 数据源
    df_all = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    df = df_all.copy()

    # --- [P0-2] Tg 口径过滤：建议分方法建模 ---
    target_col = st.session_state.target_col
    if target_col and isinstance(target_col, str) and ("tg" in target_col.lower()) and ("tg_method" in df.columns):
        st.markdown("### 🧪 Tg 口径过滤（tg_method）")
        methods = df["tg_method"].dropna().astype(str).unique().tolist()
        methods = sorted(methods)
        method_options = ["全部"] + methods

        # 默认优先 DSC（如果存在），否则全部
        default_method = "全部"
        for prefer in ["DSC", "DMA-tanδ", "DMA-tanδ (tanδ)", "DMA", "DSC (onset)"]:
            if prefer in methods:
                default_method = prefer
                break

        selected_method = st.selectbox(
            "选择训练数据的 tg_method",
            options=method_options,
            index=method_options.index(default_method) if default_method in method_options else 0,
            help="不同测试口径（DSC / DMA-tanδ 等）会造成系统偏差；建议分口径建模获得更稳定的泛化。"
        )

        if selected_method != "全部":
            before_n = len(df)
            df = df[df["tg_method"].astype(str) == str(selected_method)].copy()
            st.info(f"已过滤 tg_method={selected_method}：{before_n} → {len(df)} 行")

    # 构造 X / y
    try:
        X = df[st.session_state.feature_cols]
        y = df[target_col]
    except Exception as e:
        st.error(f"❌ 构造训练数据失败：{e}")
        return

    trainer = EnhancedModelTrainer()

    # 分组划分可用性检测
    has_resin = "resin_smiles" in df.columns
    has_hardener = "curing_agent_smiles" in df.columns
    group_key_options = []
    if has_resin and has_hardener:
        group_key_options = ["resin_smiles + curing_agent_smiles", "resin_smiles", "curing_agent_smiles"]
    elif has_resin:
        group_key_options = ["resin_smiles"]
    elif has_hardener:
        group_key_options = ["curing_agent_smiles"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📦 模型选择")
        model_name = st.selectbox("选择模型", trainer.get_available_models())

        st.markdown("### ⚙️ 训练设置")
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
        random_state = st.number_input("随机种子", 0, 1000000, 42)

        # --- [P0-3 / P1-1] 划分策略 ---
        st.markdown("### 🧩 划分策略")
        split_ui = st.selectbox(
            "选择划分策略",
            options=["随机划分", "分层划分(回归分箱)", "按配方分组划分"],
            index=1 if len(df) >= 50 else 0,
            help="小样本/跨度大回归建议使用“分层划分”；真实配方泛化建议使用“按配方分组划分”。"
        )

        split_strategy = "random"
        n_bins = 10
        groups = None
        group_key = None

        if split_ui.startswith("分层"):
            split_strategy = "stratified"
            n_bins = st.slider("分层分箱数（建议 8~12）", 4, 20, 10)
        elif split_ui.startswith("按配方"):
            split_strategy = "group"
            if not group_key_options:
                st.warning("⚠️ 当前数据缺少 resin_smiles / curing_agent_smiles，无法使用分组划分，将回退为随机划分。")
                split_strategy = "random"
            else:
                group_key = st.selectbox("分组键", options=group_key_options, index=0)
                if group_key == "resin_smiles + curing_agent_smiles":
                    groups = df["resin_smiles"].astype(str) + "||" + df["curing_agent_smiles"].astype(str)
                elif group_key == "resin_smiles":
                    groups = df["resin_smiles"].astype(str)
                elif group_key == "curing_agent_smiles":
                    groups = df["curing_agent_smiles"].astype(str)

        # --- [P0-4 / P1-1] 交叉验证 ---
        st.markdown("### 🧪 交叉验证 (CV)")
        enable_cv = st.checkbox("同时计算交叉验证 (推荐)", value=True)
        cv_folds = 5
        cv_repeats = 5
        if enable_cv:
            cv_folds = st.slider("CV folds", 3, 10, 5)
            cv_repeats = st.slider("CV repeats（仅对 repeated kfold 有效）", 1, 10, 5)

    with col2:
        st.markdown("### 🎛️ 手动调参")

        # 应用优化参数
        if st.session_state.best_params and st.session_state.get('optimized_model_name') == model_name:
            st.info(f"💡 检测到优化参数 (R²: {st.session_state.get('best_score', 0):.4f})")
            if st.button("🔄 应用最佳参数"):
                for k, v in st.session_state.best_params.items():
                    st.session_state[f"param_{model_name}_{k}"] = v
                st.rerun()

        # 生成参数控件
        manual_params = {}
        if model_name in MANUAL_TUNING_PARAMS:
            configs = MANUAL_TUNING_PARAMS[model_name]
            p_cols = st.columns(2)
            for i, config in enumerate(configs):
                with p_cols[i % 2]:
                    key = f"param_{model_name}_{config['name']}"
                    if key not in st.session_state:
                        st.session_state[key] = config['default']

                    if config['widget'] == 'slider':
                        manual_params[config['name']] = st.slider(config['label'], key=key, **config.get('args', {}))
                    elif config['widget'] == 'number_input':
                        manual_params[config['name']] = st.number_input(config['label'], key=key, **config.get('args', {}))
                    elif config['widget'] == 'selectbox':
                        manual_params[config['name']] = st.selectbox(config['label'], options=config['args']['options'], key=key)
                    elif config['widget'] == 'text_input':
                        manual_params[config['name']] = st.text_input(config['label'], key=key)

    st.markdown("---")

    # 按钮区
    c_btn1, c_btn2 = st.columns(2)

    with c_btn1:
        if st.button("🚀 开始训练", type="primary"):
            with st.spinner("训练中..."):
                try:
                    # 准备参数
                    params = manual_params.copy()
                    if 'random_state' in params:
                        params.pop('random_state')

                    # 训练（支持 split_strategy / n_bins / groups）
                    res = trainer.train_model(
                        X, y,
                        model_name=model_name,
                        test_size=test_size,
                        random_state=int(random_state),
                        split_strategy=split_strategy,
                        n_bins=int(n_bins),
                        groups=groups,
                        **params
                    )

                    # 交叉验证（可选）
                    cv_res = None
                    if enable_cv:
                        if split_strategy == "group" and groups is not None:
                            cv_strategy = "group_kfold"
                        elif split_strategy == "stratified":
                            cv_strategy = "stratified_kfold"
                        else:
                            cv_strategy = "repeated_kfold"

                        cv_res = trainer.cross_validate_model(
                            X, y,
                            model_name=model_name,
                            cv_strategy=cv_strategy,
                            n_splits=int(cv_folds),
                            n_repeats=int(cv_repeats),
                            random_state=int(random_state),
                            groups=groups,
                            n_bins=int(n_bins),
                            **params
                        )

                    # 保存结果
                    st.session_state.model = res['model']
                    st.session_state.pipeline = res.get('pipeline')
                    st.session_state.scaler = res.get('scaler')
                    st.session_state.imputer = res.get('imputer')
                    st.session_state.train_result = res
                    st.session_state.cv_result = cv_res

                    st.session_state.X_train = res['X_train']
                    st.session_state.X_test = res['X_test']
                    st.session_state.y_train = res['y_train']
                    st.session_state.y_test = res['y_test']
                    st.session_state.model_name = model_name
                    st.session_state.manual_params = params  # 用于脚本导出

                    st.success("✅ 训练完成")

                    # --- 指标 ---
                    st.markdown("### 📌 单次划分（Test）指标")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R² (Test)", f"{res['r2']:.4f}")
                    m2.metric("RMSE (Test)", f"{res['rmse']:.4f}")
                    m3.metric("MAE (Test)", f"{res['mae']:.4f}")
                    m4.metric("Train Time (s)", f"{res.get('train_time', 0):.2f}")

                    if cv_res is not None:
                        st.markdown("### 🧪 交叉验证（CV）指标")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("CV R² (mean±std)", f"{cv_res['cv_r2_mean']:.4f} ± {cv_res['cv_r2_std']:.4f}")
                        c2.metric("OOF RMSE", f"{cv_res['oof_rmse']:.4f}")
                        c3.metric("OOF MAE", f"{cv_res['oof_mae']:.4f}")

                        # 折分数表
                        fold_df = pd.DataFrame({
                            "fold_r2": cv_res.get("fold_r2", []),
                            "fold_rmse": cv_res.get("fold_rmse", []),
                            "fold_mae": cv_res.get("fold_mae", []),
                        })
                        st.dataframe(fold_df, use_container_width=True, height=200)

                    # --- 结果表格与导出 ---
                    st.markdown("### 📈 测试集预测结果详情")
                    res_df = pd.DataFrame({
                        "真实值": res['y_test'],
                        "预测值": res['y_pred_test'] if 'y_pred_test' in res else res['y_pred']
                    })
                    res_df["残差"] = res_df["真实值"] - res_df["预测值"]

                    t1, t2 = st.columns([3, 1])
                    with t1:
                        st.dataframe(res_df, use_container_width=True, height=200)
                    with t2:
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 导出结果 CSV", csv, "predictions_test.csv", "text/csv")

                    # --- 可视化 ---
                    st.markdown("### 📉 性能可视化")
                    visualizer = Visualizer()

                    if cv_res is not None:
                        tab_a, tab_b = st.tabs(["Train/Test", "CV (OOF)"])
                        with tab_a:
                            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                            with col_img2:
                                fig_tt, _ = visualizer.plot_parity_train_test(
                                    res['y_train'], res['y_pred_train'],
                                    res['y_test'], res['y_pred_test'],
                                    target_name=target_col
                                )
                                st.pyplot(fig_tt, use_container_width=True)
                        with tab_b:
                            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                            with col_img2:
                                fig_oof, _ = visualizer.plot_predictions_vs_true(
                                    cv_res['oof_true'],
                                    cv_res['oof_pred'],
                                    model_name=f"{model_name} (OOF)"
                                )
                                st.pyplot(fig_oof, use_container_width=True)
                    else:
                        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                        with col_img2:
                            fig, _ = visualizer.plot_parity_train_test(
                                res['y_train'], res['y_pred_train'],
                                res['y_test'], res['y_pred_test'],
                                target_name=target_col
                            )
                            st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 训练失败: {e}")

    with c_btn2:
        # 脚本导出按钮
        if st.session_state.model and st.session_state.model_name == model_name:
            if 'generate_training_script_code' in globals():
                script = generate_training_script_code(
                    model_name,
                    manual_params,
                    st.session_state.feature_cols,
                    st.session_state.target_col
                )
                st.download_button("💾 导出 Python 训练脚本", script, "train_script.py")

def page_model_interpretation():
    """模型解释页面"""
    st.title("📊 模型解释")

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        return

    model = st.session_state.model
    model_name = st.session_state.model_name
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_names = st.session_state.feature_cols

    tab1, tab2, tab3 = st.tabs(["🔍 SHAP分析", "📈 预测性能", "🎯 特征重要性"])

    # --- 1. SHAP 分析 (恢复选项) ---
    with tab1:
        st.markdown("### SHAP特征重要性")

        # 恢复这两个选项控件
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            plot_type = st.selectbox("图表类型", ["bar", "beeswarm"], index=0)
        with c_opt2:
            max_display = st.slider("显示特征数量", 5, 50, 20)

        if st.button("🔍 计算SHAP值"):
            with st.spinner("正在计算 SHAP 值 (可能较慢)..."):
                try:
                    interp = EnhancedModelInterpreter(
                        model, X_train, y_train, X_test, y_test,
                        model_name, feature_names=feature_names
                    )
                    # 调用修改后的 plot_summary，获取图和数据
                    fig, df_shap = interp.plot_summary(plot_type=plot_type, max_display=max_display)

                    if fig:
                        # 限制图片宽度
                        c1, c2, c3 = st.columns([1, 6, 1])
                        with c2:
                            st.pyplot(fig, use_container_width=True)

                            # SHAP 数据导出
                            if df_shap is not None:
                                csv = df_shap.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 导出 SHAP 数据 (CSV)", csv, "shap_values.csv", "text/csv")
                    else:
                        st.error("无法生成 SHAP 图，请检查模型是否支持。")
                except Exception as e:
                    st.error(f"计算出错: {str(e)}")

    # --- 2. 预测性能 ---
    with tab2:
        st.markdown("### 预测性能")
        visualizer = Visualizer()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            fig, df_res = visualizer.plot_residuals(y_test, st.session_state.train_result['y_pred'], model_name)
            st.pyplot(fig, use_container_width=True)
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📥 导出残差数据", csv, "residuals.csv")

    # --- 3. 特征重要性 (含 MACCS 解释) ---
    with tab3:
        st.markdown("### 特征重要性")
        if hasattr(model, 'feature_importances_'):
            visualizer = Visualizer()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig, df_imp = visualizer.plot_feature_importance(model.feature_importances_, feature_names, model_name)
                st.pyplot(fig, use_container_width=True)
                csv = df_imp.to_csv(index=False).encode('utf-8')
                st.download_button("📥 导出重要性数据", csv, "importance.csv")

            # MACCS 解释表
            st.markdown("#### 🧬 特征含义解析")
            exps = []
            for f in df_imp.head(15)['Feature']:
                desc = "数值特征"
                if "MACCS" in f:
                    try:
                        # 动态导入防止报错
                        from core.molecular_features import get_maccs_description
                        idx = int(f.split('_')[-1])
                        desc = get_maccs_description(idx)
                    except:
                        desc = "MACCS 指纹片段"
                exps.append({"特征名": f, "含义": desc})
            st.table(pd.DataFrame(exps))
        else:
            st.info("该模型不支持原生特征重要性，请使用 SHAP 分析。")

def page_prediction():
    """预测应用页面（修复：预测阶段应用 imputer/scaler；支持指纹适用域）"""
    st.title("🔮 预测应用")

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        return

    model = st.session_state.model
    model_name = st.session_state.model_name
    feature_cols = st.session_state.feature_cols
    pipeline = st.session_state.get("pipeline", None)
    scaler = st.session_state.get("scaler", None)
    imputer = st.session_state.get("imputer", None)

    tab1, tab2, tab3 = st.tabs(["📝 单样本预测", "📁 批量预测", "🎯 适用域分析"])

    # ============================================================
    # Tab1: 单样本预测
    # ============================================================
    with tab1:
        st.markdown("### 单样本预测")

        input_df = None

        # 特征过多时，禁止渲染大量 number_input（会卡死）
        if len(feature_cols) <= 60:
            input_values = {}
            cols = st.columns(3)
            for i, feature in enumerate(feature_cols):
                with cols[i % 3]:
                    input_values[feature] = st.number_input(feature, value=0.0)
            input_df = pd.DataFrame([input_values])
        else:
            st.info(f"当前特征数量较多（{len(feature_cols)}），建议用“单行文件”上传进行单样本预测。")
            single_file = st.file_uploader("上传单样本文件（CSV/Excel，至少包含所选特征列）", type=['csv', 'xlsx', 'xls'], key="single_pred_file")
            if single_file is not None:
                try:
                    tmp_df = load_data_file(single_file)
                    if tmp_df.shape[0] == 0:
                        st.error("文件为空")
                    else:
                        row_idx = 0
                        if tmp_df.shape[0] > 1:
                            row_idx = st.number_input("选择预测的行号（从 0 开始）", 0, int(tmp_df.shape[0]-1), 0)
                        input_df = tmp_df.iloc[[int(row_idx)]].copy()
                except Exception as e:
                    st.error(f"读取文件失败: {e}")

        if st.button("🔮 开始预测", type="primary"):
            if input_df is None:
                st.error("请先提供输入样本")
            else:
                # 保证列齐全
                missing = [c for c in feature_cols if c not in input_df.columns]
                if missing:
                    st.error(f"输入样本缺少特征列: {missing[:10]}{'...' if len(missing)>10 else ''}")
                else:
                    X_in = input_df[feature_cols]

                    try:
                        if pipeline is not None:
                            pred = pipeline.predict(X_in)[0]
                        else:
                            # [P0-5] 修复：没有 pipeline 时也要应用 imputer + scaler
                            X_arr = X_in.values
                            if imputer is not None:
                                X_arr = imputer.transform(X_arr)
                            if scaler is not None:
                                X_arr = scaler.transform(X_arr)
                            pred = model.predict(X_arr)[0]

                        st.success(f"✅ 预测结果：{pred:.4f}")

                    except Exception as e:
                        st.error(f"❌ 预测失败: {e}")

    # ============================================================
    # Tab2: 批量预测
    # ============================================================
    with tab2:
        st.markdown("### 批量预测")
        uploaded_file = st.file_uploader("上传待预测数据", type=['csv', 'xlsx', 'xls'], key="batch_pred_file")

        if uploaded_file is not None:
            try:
                pred_df = load_data_file(uploaded_file)
                st.dataframe(pred_df.head(), use_container_width=True)

                if st.button("🚀 执行批量预测", type="primary"):
                    missing = [c for c in feature_cols if c not in pred_df.columns]
                    if missing:
                        st.error(f"缺少特征列: {missing[:10]}{'...' if len(missing)>10 else ''}")
                    else:
                        X_pred = pred_df[feature_cols]

                        if pipeline is not None:
                            preds = pipeline.predict(X_pred)
                        else:
                            X_arr = X_pred.values
                            if imputer is not None:
                                X_arr = imputer.transform(X_arr)
                            if scaler is not None:
                                X_arr = scaler.transform(X_arr)
                            preds = model.predict(X_arr)

                        pred_df['prediction'] = preds
                        st.success("✅ 批量预测完成")
                        st.dataframe(pred_df.head(20), use_container_width=True)

                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 下载预测结果 CSV", csv, "batch_predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ 读取文件失败: {e}")

    # ============================================================
    # Tab3: 适用域分析
    # ============================================================
    with tab3:
        st.markdown("### 🎯 适用域分析")
        st.caption("适用域用于判断新样本是否“超出训练数据覆盖范围”。PCA Hull 更通用；Tanimoto 更适用于指纹特征。")

        # 可用方法
        fp_cols = [c for c in feature_cols if ("morgan" in c.lower()) or ("maccs" in c.lower())]
        has_fp = len(fp_cols) > 0

        methods = ["PCA Hull（数值空间）"]
        if has_fp and st.session_state.get("train_result") is not None and "X_train_raw" in st.session_state.train_result:
            methods.append("Tanimoto 相似度（指纹）")

        ad_method = st.selectbox("选择适用域方法", options=methods, index=0)

        # -------- PCA Hull --------
        if ad_method.startswith("PCA"):
            st.info("PCA Hull：在降维空间构建凸包，判断新样本是否落在训练集覆盖范围内（对任意数值特征通用）。")

            input_df = None
            if len(feature_cols) <= 60:
                input_values = {}
                cols = st.columns(3)
                for i, feature in enumerate(feature_cols):
                    with cols[i % 3]:
                        input_values[feature] = st.number_input(feature, value=0.0, key=f"ad_pca_{feature}")
                input_df = pd.DataFrame([input_values])
            else:
                st.info(f"当前特征数量较多（{len(feature_cols)}），建议用“单行文件”上传进行适用域判断。")
                single_file = st.file_uploader("上传单样本文件（CSV/Excel，一行即可）", type=['csv', 'xlsx', 'xls'], key="ad_single_file_pca")
                if single_file is not None:
                    try:
                        tmp_df = load_data_file(single_file)
                        if tmp_df.shape[0] > 0:
                            row_idx = 0
                            if tmp_df.shape[0] > 1:
                                row_idx = st.number_input("选择分析的行号（从 0 开始）", 0, int(tmp_df.shape[0]-1), 0, key="ad_pca_row_idx")
                            input_df = tmp_df.iloc[[int(row_idx)]].copy()
                    except Exception as e:
                        st.error(f"读取文件失败: {e}")

            if st.button("🎯 适用域分析（PCA Hull）", type="primary"):
                if input_df is None:
                    st.error("请先提供输入样本")
                else:
                    missing = [c for c in feature_cols if c not in input_df.columns]
                    if missing:
                        st.error(f"输入样本缺少特征列: {missing[:10]}{'...' if len(missing)>10 else ''}")
                    else:
                        X_input = input_df[feature_cols].values
                        if imputer is not None:
                            X_input = imputer.transform(X_input)
                        if scaler is not None:
                            X_input = scaler.transform(X_input)

                        analyzer = ApplicabilityDomainAnalyzer(st.session_state.X_train.values)
                        analyzer.fit()

                        in_domain, distance = analyzer.is_within_domain(X_input)

                        if in_domain:
                            st.success(f"✅ 样本在适用域内 (distance={distance:.4f})")
                        else:
                            st.warning(f"⚠️ 样本可能超出适用域 (distance={distance:.4f})")

        # -------- Tanimoto Similarity --------
        else:
            if not has_fp:
                st.warning("当前特征中未检测到 MACCS/Morgan 指纹列，无法使用 Tanimoto 适用域。")
            else:
                st.info("Tanimoto：计算新样本与训练集中最近邻的指纹相似度 sim_max。sim_max 过低通常意味着 out-of-domain。")

                threshold = st.slider("相似度阈值（建议 0.20~0.30）", 0.0, 1.0, 0.25, 0.01)
                top_k = st.slider("Top-K 相似样本", 1, 20, 5)

                # 构造训练指纹矩阵
                try:
                    X_train_raw = st.session_state.train_result["X_train_raw"]
                    y_train = st.session_state.train_result.get("y_train")
                    X_train_fp = X_train_raw[fp_cols]
                    analyzer = TanimotoADAnalyzer(X_train_fp, threshold=threshold, max_train_samples=5000, random_state=42)
                except Exception as e:
                    st.error(f"初始化 Tanimoto 分析器失败: {e}")
                    analyzer = None

                st.markdown("#### 1) 单样本分析（推荐：上传单行文件）")
                single_file = st.file_uploader("上传单样本文件（CSV/Excel，一行即可，需包含指纹列）", type=['csv', 'xlsx', 'xls'], key="ad_single_file_tanimoto")
                if analyzer is not None and single_file is not None:
                    try:
                        qdf = load_data_file(single_file)
                        if qdf.shape[0] == 0:
                            st.error("文件为空")
                        else:
                            row_idx = 0
                            if qdf.shape[0] > 1:
                                row_idx = st.number_input("选择分析的行号（从 0 开始）", 0, int(qdf.shape[0]-1), 0, key="ad_tani_row_idx")
                            qrow = qdf.iloc[int(row_idx)]
                            missing = [c for c in fp_cols if c not in qdf.columns]
                            if missing:
                                st.error(f"缺少指纹列: {missing[:10]}{'...' if len(missing)>10 else ''}")
                            else:
                                is_in, sim_max, top_df, fig = analyzer.analyze_single(qrow[fp_cols].values, top_k=top_k, threshold=threshold)
                                if is_in:
                                    st.success(f"✅ 在适用域内：sim_max = {sim_max:.3f}")
                                else:
                                    st.warning(f"⚠️ 可能超出适用域：sim_max = {sim_max:.3f}")

                                # 补充：显示 top-k 的 y_train（若有）
                                if y_train is not None and len(y_train) >= top_df.shape[0]:
                                    top_df = top_df.copy()
                                    try:
                                        top_df["y_train"] = [float(y_train[int(i)]) for i in top_df["train_index"].values]
                                    except Exception:
                                        pass

                                st.dataframe(top_df, use_container_width=True, height=200)
                                st.pyplot(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"分析失败: {e}")

                st.markdown("---")
                st.markdown("#### 2) 批量分析（输出 sim_max / in_domain）")
                batch_file = st.file_uploader("上传批量样本文件（CSV/Excel）", type=['csv', 'xlsx', 'xls'], key="ad_batch_file_tanimoto")

                if analyzer is not None and batch_file is not None:
                    try:
                        qdf = load_data_file(batch_file)
                        missing = [c for c in fp_cols if c not in qdf.columns]
                        if missing:
                            st.error(f"缺少指纹列: {missing[:10]}{'...' if len(missing)>10 else ''}")
                        else:
                            sim_max_arr = analyzer.compute_batch_max_similarity(qdf[fp_cols])
                            out_df = qdf.copy()
                            out_df["sim_max"] = sim_max_arr
                            out_df["in_domain"] = out_df["sim_max"] >= threshold

                            st.success("✅ 批量适用域分析完成")
                            st.dataframe(out_df[["sim_max", "in_domain"]].head(20), use_container_width=True)

                            # 可选：如果包含目标列，可画 |error| vs sim_max
                            if st.session_state.get("target_col") in out_df.columns:
                                try:
                                    y_true = pd.to_numeric(out_df[st.session_state.target_col], errors="coerce")
                                    ok = y_true.notna()
                                    if ok.sum() >= 10:
                                        # 预测
                                        if pipeline is not None:
                                            y_pred = pipeline.predict(out_df.loc[ok, feature_cols])
                                        else:
                                            X_arr = out_df.loc[ok, feature_cols].values
                                            if imputer is not None:
                                                X_arr = imputer.transform(X_arr)
                                            if scaler is not None:
                                                X_arr = scaler.transform(X_arr)
                                            y_pred = model.predict(X_arr)
                                        abs_err = np.abs(y_true.loc[ok].values - y_pred)

                                        fig, ax = plt.subplots(figsize=(7, 4))
                                        ax.scatter(out_df.loc[ok, "sim_max"].values, abs_err, alpha=0.7, edgecolors="k", linewidth=0.3)
                                        ax.set_xlabel("sim_max (Tanimoto)")
                                        ax.set_ylabel("|error|")
                                        ax.set_title("|error| vs sim_max")
                                        ax.grid(True, linestyle="--", alpha=0.4)
                                        plt.tight_layout()
                                        st.pyplot(fig, use_container_width=True)
                                except Exception:
                                    pass

                            csv = out_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 下载适用域结果 CSV", csv, "tanimoto_ad_results.csv", "text/csv")

                    except Exception as e:
                        st.error(f"批量分析失败: {e}")

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

    # --- [新增] 进度条组件 ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    if st.button("🚀 开始优化", type="primary"):
        try:
            optimizer = HyperparameterOptimizer()

            # 定义进度更新回调
            def update_progress(p):
                progress_bar.progress(min(p, 1.0))
                status_text.text(f"正在进行优化... 进度: {int(p * 100)}%")

            with st.spinner(f"正在优化 {model_name}..."):
                # 传递 progress_callback
                best_params, best_score, study = optimizer.optimize(
                    model_name, X, y,
                    n_trials=n_trials,
                    cv=cv_folds,
                    progress_callback=update_progress
                )

            # 优化完成，进度条满
            progress_bar.progress(100)
            status_text.text("优化完成！")

            st.success(f"✅ 优化完成！最佳R²分数: {best_score:.4f}")

            # 保存到 session_state
            st.session_state.best_params = best_params
            st.session_state.best_score = best_score
            st.session_state.optimized_model_name = model_name  # 记录优化的是哪个模型

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
                st.session_state.pipeline = result.get('pipeline')
                st.session_state.scaler = result.get('scaler')
                st.session_state.imputer = result.get('imputer')
                st.session_state.X_train = result.get('X_train')
                st.session_state.X_test = result.get('X_test')
                st.session_state.y_train = result.get('y_train')
                st.session_state.y_test = result.get('y_test')
                st.session_state.model_name = model_name
                st.session_state.train_result = result
                st.session_state.cv_result = None
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