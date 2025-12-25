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
# [新增] TensorFlow Sequential (TFS) 模型支持（即使未安装 TF 也要显示入口）
try:
    from core.tf_model import (
        TFSequentialRegressor,
        TENSORFLOW_AVAILABLE,
        TFS_TUNING_PARAMS,
        TENSORFLOW_IMPORT_ERROR,
        get_tensorflow_version
    )
except Exception as e:
    # 任何异常都不应阻止应用启动
    TENSORFLOW_AVAILABLE = False
    TFSequentialRegressor = None
    TFS_TUNING_PARAMS = []
    TENSORFLOW_IMPORT_ERROR = repr(e)

    def get_tensorflow_version():
        return None

# [新增] 特征工程状态追踪器
from core.fe_tracker import (
    FeatureEngineeringTracker,
    render_status_sidebar,
    render_status_panel,
    render_data_export_panel,
    create_quick_export_button
)

try:
    import torchani

    TORCHANI_AVAILABLE = True
except ImportError:
    TORCHANI_AVAILABLE = False
import streamlit as st

# =========================
# Operation Log Utilities
# =========================
def _oplog_init():
    if "oplog" not in st.session_state:
        st.session_state["oplog"] = []

def oplog(msg: str):
    """Append a timestamped message to operation log and show it in UI."""
    _oplog_init()
    import datetime as _dtmod
    ts = _dtmod.datetime.now().strftime("%H:%M:%S")
    st.session_state["oplog"].append(f"[{ts}] {msg}")

def oplog_clear():
    st.session_state["oplog"] = []

def oplog_render():
    _oplog_init()
    with st.expander("🧾 Operation Log", expanded=False):
        if st.session_state["oplog"]:
            st.code("\n".join(st.session_state["oplog"]))
        else:
            st.caption("No operations yet.")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Clear Log"):
                oplog_clear()
                st.rerun()
import pandas as pd
<<<<<<< HEAD

from rdkit import Chem as _Chem

def _quick_rdkit_parse_stats(smiles_list, max_check: int = 200):
    """Fast parse-only check (no 3D). Returns (ok_count, checked_count, examples_bad)."""
    checked = 0
    ok = 0
    bad = []
    for s in smiles_list:
        if checked >= max_check:
            break
        if s is None:
            continue
        ss = str(s).strip()
        if (not ss) or (ss.lower() in {"nan", "none", "<na>", "na"}):
            continue
        checked += 1

        # split like 3D worker: ;,；,| and " + ", then "."
        parts = re.split(r"\s*[;；|]\s*", ss)
        frags = []
        for p in parts:
            if not p:
                continue
            for q in re.split(r"\s+\+\s+", p):
                frags.extend([x.strip() for x in str(q).split('.') if x and str(x).strip()])

        parsed_any = False
        for frag in frags:
            m = _Chem.MolFromSmiles(frag)
            if m is not None and m.GetNumAtoms() >= 2:
                parsed_any = True
                break

        if parsed_any:
            ok += 1
        else:
            if len(bad) < 5:
                bad.append(ss[:200])
    return ok, checked, bad
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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

# 统一 matplotlib 风格（全站图表一致）
try:
    from core.plot_style import apply_global_style
    apply_global_style()
except Exception:
    pass

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
from core.plot_utils import fig_to_png_bytes, fig_to_html
from core.training_curves import plot_history
from core.training_runs import TrainingRunManager
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



# === TensorFlow (for TFS model) ===
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None
    regularizers = None


def build_tfs_model(input_dim, params):
    # Parse hidden layers like "128,64,32"
    hidden_layers_str = str(params.get('hidden_layers', '128,64,32'))
    try:
        hidden = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
    except Exception:
        hidden = [128, 64, 32]

    activation = str(params.get('activation', 'relu'))
    dropout_rate = float(params.get('dropout_rate', 0.2))
    l2_reg = float(params.get('l2_reg', 0.001))
    learning_rate = float(params.get('learning_rate', 0.001))
    opt_name = str(params.get('optimizer', 'adam')).lower()

    if TENSORFLOW_AVAILABLE:
        try:
            tf.random.set_seed(42)
        except Exception:
            pass

    model = keras.Sequential(name='TFS_Regressor')
    model.add(layers.Input(shape=(int(input_dim),)))

    reg = None
    try:
        if l2_reg and l2_reg > 0:
            reg = regularizers.l2(l2_reg)
    except Exception:
        reg = None

    for units in hidden:
        units = int(units)
        if activation == 'leaky_relu':
            model.add(layers.Dense(units, kernel_regularizer=reg))
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=reg))
        if dropout_rate and dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    # Optimizer
    if opt_name == 'adamw' and hasattr(keras.optimizers, 'AdamW'):
        opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif opt_name == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif opt_name == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model
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
    elif MODEL_NAME == "TensorFlow Sequential":
        if not TENSORFLOW_AVAILABLE:
            print("❌ 错误: TensorFlow 未安装或导入失败，无法训练 TFS。请先安装: pip install tensorflow")
            return
        print("开始训练 (TFS)...")
        model = build_tfs_model(X_train.shape[1], HYPERPARAMETERS)
        cbs = []
        if bool(HYPERPARAMETERS.get('early_stopping', True)):
            try:
                patience = int(HYPERPARAMETERS.get('patience', 20))
            except Exception:
                patience = 20
            cbs.append(callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))

        model.fit(
            X_train, y_train,
            epochs=int(HYPERPARAMETERS.get('epochs', 200)),
            batch_size=int(HYPERPARAMETERS.get('batch_size', 32)),
            validation_split=float(HYPERPARAMETERS.get('validation_split', 0.1)),
            verbose=1,
            callbacks=cbs
        )
        y_pred = model.predict(X_test).ravel()
        r2 = r2_score(y_test, y_pred)
        print('训练完成！R² Score: %.4f' % r2)
        return

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
        'optimized_model_name': None,  # 新增：记录优化的模型名

        # --- [新增] Active Learning ---
        'al_pool_data': None,
        'al_recommendations': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

if 'fe_tracker' not in st.session_state:
    st.session_state.fe_tracker = FeatureEngineeringTracker()
tracker = st.session_state.fe_tracker


def log_fe_step(operation: str, description: str, params=None, input_df=None, output_df=None,
                features_added=None, features_removed=None, status: str = "success", message: str = ""):
    """记录特征工程/建模关键步骤到状态条（不会影响主流程）。"""
    tr = st.session_state.get("fe_tracker", None)
    if tr is None:
        return
    try:
        tr.log_step(
            operation=operation,
            description=description,
            params=params or {},
            input_df=input_df,
            output_df=output_df,
            features_added=features_added or [],
            features_removed=features_removed or [],
            status=status,
            message=message
        )
    except Exception:
        # 日志记录失败不应影响主流程
        pass


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
<<<<<<< HEAD
                "🖼️ 图像转SMILES",
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
                "🎯 特征选择",
                "🤖 模型训练",
                "📈 训练记录",
                "📊 模型解释",
                "🔮 预测应用",
                "⚙️ 超参优化",
                "🧠 主动学习",
                "📋 状态条记录",
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

        # --- [新增] 依赖检测（让 TFS 模型入口更“可见”） ---
        st.markdown("### 🧩 依赖检测")
        tf_ver = None
        try:
            tf_ver = get_tensorflow_version()
        except Exception:
            tf_ver = None

        if bool(TENSORFLOW_AVAILABLE) and tf_ver:
            st.success(f"TensorFlow: {tf_ver}")
        else:
            st.caption("TensorFlow: 未安装或不可用")
            try:
                err = globals().get('TENSORFLOW_IMPORT_ERROR', '')
                if err:
                    st.caption(f"TF 导入信息: {str(err)[:160]}")
            except Exception:
                pass

        st.caption(f"TorchANI: {'可用' if TORCHANI_AVAILABLE else '不可用'}")

        # [增强] 在侧边栏始终显示状态条入口（即使暂无记录，也避免“功能存在但界面不显示”）
        render_status_sidebar(st.session_state.get('fe_tracker', None))
        return page


def render_top_status_bar():
    """主区域顶部的轻量状态条（防止用户折叠侧边栏后找不到状态条/TF信息）。"""
    tr = st.session_state.get('fe_tracker', None)
    if tr is None:
        return

    try:
        stats = tr.get_stats()
        last = tr.get_last_step()
    except Exception:
        stats = {'success': 0, 'warning': 0, 'error': 0}
        last = None

    # TensorFlow 状态
    tf_ver = None
    try:
        tf_ver = get_tensorflow_version()
    except Exception:
        tf_ver = None
    tf_status = "✅" if (bool(TENSORFLOW_AVAILABLE) and tf_ver) else "⛔"
    tf_text = f"TensorFlow {tf_status} {tf_ver}" if (bool(TENSORFLOW_AVAILABLE) and tf_ver) else "TensorFlow ⛔ 未安装/不可用"

    with st.expander("📋 状态条（快捷）", expanded=False):
        c1, c2, c3 = st.columns([1.2, 1.2, 3.6])
        with c1:
            st.caption("记录统计")
            st.write(f"✅ {int(stats.get('success', 0))}  ·  ⚠️ {int(stats.get('warning', 0))}  ·  ❌ {int(stats.get('error', 0))}")
        with c2:
            st.caption("TFS 依赖")
            st.write(tf_text)
        with c3:
            st.caption("最近一次")
            if last:
                icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(last.get('status', 'success'), "ℹ️")
                st.write(f"{icon} [{last.get('timestamp','')}] {last.get('operation','')} - {last.get('description','')}")
                if last.get('message'):
                    st.caption(last.get('message'))
            else:
                st.write("暂无记录。完成一次数据上传/清洗/特征选择/训练后会自动出现。")
        st.caption("提示：更完整的时间线与导出在左侧导航「📋 状态条记录」。")


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

                # [新增] 记录到状态条
                log_fe_step(
                    operation="数据上传",
                    description=f"加载文件：{uploaded_file.name}",
                    params={"rows": int(df.shape[0]), "cols": int(df.shape[1]), "type": "csv" if uploaded_file.name.endswith('.csv') else "excel"},
                    output_df=df,
                    message=f"数据维度：{df.shape[0]}×{df.shape[1]}"
                )

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
                log_fe_step(
                    operation="数据生成",
                    description="生成混合示例数据集",
                    params={"n_samples": int(n_samples_hybrid), "type": "hybrid"},
                    output_df=df,
                    message=f"数据维度：{df.shape[0]}×{df.shape[1]}"
                )
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
                log_fe_step(
                    operation="数据生成",
                    description="生成纯数值示例数据集",
                    params={"n_samples": int(n_samples_numeric), "type": "numeric"},
                    output_df=df,
                    message=f"数据维度：{df.shape[0]}×{df.shape[1]}"
                )
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

        numeric_cols = explorer.numeric_cols
        if not numeric_cols or len(numeric_cols) < 2:
            st.info("需要至少2个数值列")
        else:
            # --- 自定义热图特征子集 ---
            col_a, col_b = st.columns([2, 1])
            with col_a:
                heatmap_mode = st.radio(
                    "热图特征来源",
                    ["全部数值特征", "自定义多选", "使用【特征选择】页当前子集", "按目标相关性Top-K"],
                    horizontal=True,
                    key="corr_heatmap_mode"
                )
            with col_b:
                max_show = st.number_input(
                    "最多显示特征数",
                    min_value=2,
                    max_value=min(80, len(numeric_cols)),
                    value=min(25, len(numeric_cols)),
                    key="corr_heatmap_max"
                )

            target = st.session_state.get("target_col")
            include_target = st.checkbox("包含目标变量（若为数值列）", value=True, key="corr_heatmap_include_target")

            selected_cols = None

            if heatmap_mode == "全部数值特征":
                selected_cols = numeric_cols.copy()

            elif heatmap_mode == "使用【特征选择】页当前子集":
                selected_cols = [c for c in st.session_state.get("feature_cols", []) if c in numeric_cols]
                if not selected_cols:
                    st.info("当前尚未在【特征选择】页选定特征子集，将回退为“自定义多选”。")
                    heatmap_mode = "自定义多选"

            if heatmap_mode == "自定义多选":
                default_cols = numeric_cols[:min(25, len(numeric_cols))]
                selected_cols = st.multiselect(
                    "选择用于热图的数值特征",
                    options=numeric_cols,
                    default=default_cols,
                    key="corr_heatmap_cols"
                )

            elif heatmap_mode == "按目标相关性Top-K":
                k = st.number_input(
                    "Top-K（按 |corr| 排序）",
                    min_value=2,
                    max_value=len(numeric_cols),
                    value=min(20, len(numeric_cols)),
                    key="corr_heatmap_topk"
                )
                if target in numeric_cols:
                    corrs = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                    selected_cols = corrs.head(int(k)).index.tolist()
                else:
                    st.info("目标变量不是数值列，无法按相关性排序；已使用前K个数值列。")
                    selected_cols = numeric_cols[:int(k)]

            # 可选：把目标变量也放到热图里
            if include_target and (target in numeric_cols) and (target not in selected_cols):
                selected_cols = selected_cols + [target]

            # 限制显示数量，避免热图过大
            if len(selected_cols) > int(max_show):
                st.warning(f"已选择 {len(selected_cols)} 个特征，为可读性仅显示前 {int(max_show)} 个。")
                selected_cols = selected_cols[:int(max_show)]

            fig = explorer.plot_correlation_matrix(cols=selected_cols)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 高相关性特征对（基于当前热图列）
            pairs = explorer.get_high_correlation_pairs(cols=selected_cols, threshold=0.8)
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "❓ 缺失值处理", "📊 异常值检测", "🔄 重复数据", "🔧 数据类型", "🧪 SMILES清洗", "🧩 SMILES组分分列", "⚖️ 类别平衡",
        "🧩 K-Means聚类"])

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
                    # 为了避免“伪数值列(object)无法填充”导致看起来没生效，
                    # 这里先自动尝试把可转换的列转为数值型再做缺失值处理。
                    df_in = df.copy()
                    try:
                        _tmp_cleaner = AdvancedDataCleaner(df_in)
                        df_in = _tmp_cleaner.fix_pseudo_numeric_columns()
                    except Exception:
                        pass

                    missing_before = int(df_in.isna().sum().sum())
                    rows_before = int(df_in.shape[0])

                    if strategy == "mode":
                        cleaned_df = df_in.copy()
                        # 众数填充：对所有列都生效（含文本列）
                        for _col in cleaned_df.columns:
                            if cleaned_df[_col].isna().any():
                                try:
                                    _mode = cleaned_df[_col].mode(dropna=True)
                                    if not _mode.empty:
                                        cleaned_df[_col] = cleaned_df[_col].fillna(_mode.iloc[0])
                                except Exception:
                                    # 某些列 mode 计算可能失败，跳过即可
                                    pass
                    elif strategy == "drop_rows":
                        cleaned_df = df_in.dropna().reset_index(drop=True)
                    else:
                        # 其余策略先走原清洗器（主要针对数值列）
                        _cleaner2 = AdvancedDataCleaner(df_in)
                        cleaned_df = _cleaner2.handle_missing_values(strategy=strategy, fill_value=fill_value)

                        # 如果还有非数值列缺失，给一个温和回退：用众数补齐
                        # 避免用户看到“按钮点了但没变化”
                        non_num_cols = cleaned_df.select_dtypes(exclude=np.number).columns.tolist()
                        for _col in non_num_cols:
                            if cleaned_df[_col].isna().any():
                                try:
                                    _mode = cleaned_df[_col].mode(dropna=True)
                                    if not _mode.empty:
                                        cleaned_df[_col] = cleaned_df[_col].fillna(_mode.iloc[0])
                                except Exception:
                                    if strategy == "constant":
                                        cleaned_df[_col] = cleaned_df[_col].fillna(fill_value if fill_value is not None else 0)

                    missing_after = int(cleaned_df.isna().sum().sum())
                    rows_after = int(cleaned_df.shape[0])

                    st.session_state.processed_data = cleaned_df

                    log_fe_step(
                        operation="缺失值处理",
                        description=f"策略: {strategy}",
                        params={"strategy": strategy, "fill_value": fill_value},
                        input_df=df_in,
                        output_df=cleaned_df,
                        message=f"缺失值: {missing_before} → {missing_after}; 行数: {rows_before} → {rows_after}"
                    )

                    if strategy == "drop_rows":
                        st.success(f"✅ 缺失值处理完成：删除 {rows_before - rows_after} 行（{rows_before} → {rows_after}）")
                    else:
                        st.success(f"✅ 缺失值处理完成：缺失值 {missing_before} → {missing_after}")

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
            # 同样先尝试把“伪数值列”转换为数值列，避免漏检
            _tmp_cleaner = AdvancedDataCleaner(df.copy())
            try:
                _tmp_cleaner.fix_pseudo_numeric_columns()
            except Exception:
                pass

            outliers = _tmp_cleaner.detect_outliers(method=method, threshold=threshold)
            if outliers:
                st.warning(f"检测到 {len(outliers)} 列存在异常值")
                st.json(outliers)
            else:
                st.success("✅ 未检测到显著异常值")

        if st.button("🔧 处理异常值", type="primary"):
            # 后端 cleaner.handle_outliers() 仅支持 IQR + clip/replace_median，
            # 为了让前端的 remove / zscore 选项真正生效，这里在 app.py 内做兼容实现。
            df_in = df.copy()
            _tmp_cleaner = AdvancedDataCleaner(df_in)
            try:
                df_in = _tmp_cleaner.fix_pseudo_numeric_columns()
            except Exception:
                pass

            numeric_cols = df_in.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("ℹ️ 未找到可用于异常值处理的数值列。")
                return

            any_outlier = pd.Series(False, index=df_in.index)
            total_affected = 0

            for col in numeric_cols:
                s = df_in[col]

                if method == "iqr":
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    if pd.isna(iqr) or iqr == 0:
                        continue
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    mask = (s < lower) | (s > upper)
                else:
                    mean = s.mean(skipna=True)
                    std = s.std(skipna=True)
                    if pd.isna(std) or std == 0:
                        continue
                    z = (s - mean) / std
                    mask = z.abs() > threshold
                    lower = mean - threshold * std
                    upper = mean + threshold * std

                mask = mask.fillna(False)

                if handle_method == "remove":
                    any_outlier = any_outlier | mask
                elif handle_method == "clip":
                    df_in[col] = s.clip(lower, upper)
                    total_affected += int(mask.sum())
                elif handle_method == "replace_median":
                    median_val = s.median(skipna=True)
                    df_in.loc[mask, col] = median_val
                    total_affected += int(mask.sum())

            if handle_method == "remove":
                removed_rows = int(any_outlier.sum())
                cleaned_df = df_in.loc[~any_outlier].reset_index(drop=True)
                st.session_state.processed_data = cleaned_df
                log_fe_step(
                    operation="异常值处理",
                    description=f"方法: {method}, 处理: {handle_method}",
                    params={"method": method, "threshold": float(threshold), "handle": handle_method},
                    input_df=df_in,
                    output_df=cleaned_df,
                    message=f"删除 {removed_rows} 行"
                )
                st.success(f"✅ 异常值处理完成：删除 {removed_rows} 行（{len(df_in)} → {len(cleaned_df)}）")
            else:
                st.session_state.processed_data = df_in
                log_fe_step(
                    operation="异常值处理",
                    description=f"方法: {method}, 处理: {handle_method}",
                    params={"method": method, "threshold": float(threshold), "handle": handle_method},
                    input_df=df,
                    output_df=df_in,
                    message=f"调整 {total_affected} 个异常值单元格"
                )
                st.success(f"✅ 异常值处理完成：已调整 {total_affected} 个异常值单元格")

            st.rerun()

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
                    log_fe_step(
                        operation="去重",
                        description="删除完全重复行",
                        params={"removed_rows": int(dup_count)},
                        input_df=df,
                        output_df=cleaned_df,
                        message=f"删除 {dup_count} 行"
                    )
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
                    log_fe_step(
                        operation="分布优化",
                        description=f"降低重复率: {target_col}",
                        params={"feature": target_col, "target_rate": float(target_rate)},
                        input_df=df,
                        output_df=cleaned_df,
                        message=f"行数: {original_len} → {new_len}"
                    )

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
                    log_fe_step(
                        operation="重复记录聚合",
                        description=f"keys={keys} / target={target_col_for_agg} / agg={agg_method}",
                        params={"keys": keys, "target": target_col_for_agg, "agg": agg_method, "dropna_target": bool(dropna_target)},
                        input_df=df,
                        output_df=new_df,
                        message=f"行数: {len(df)} → {len(new_df)}"
                    )
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
                log_fe_step(
                    operation="数据类型修复",
                    description="修复伪数值列",
                    input_df=df,
                    output_df=cleaned_df,
                    message="已将可转换的 object 列转换为数值类型"
                )
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
                        log_fe_step(
                            operation="One-Hot 编码",
                            description=f"编码列: {encode_cols}",
                            params={"cols": encode_cols, "drop_first": bool(drop_first)},
                            input_df=df,
                            output_df=new_df,
                            message=f"列数: {df.shape[1]} → {new_df.shape[1]}"
                        )
                        st.success(f"✅ One-Hot 编码完成：列数 {df.shape[1]} → {new_df.shape[1]}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ One-Hot 编码失败: {e}")
        else:
            st.info("未检测到可编码的类别列")

    with tab5:
        st.markdown("### 🧪 SMILES 字符串清洗与修复")
        st.info(
            "💡 针对原始数据中的不规范 SMILES（如包含引号、非标准字符、错误的立体化学标记等）进行清洗和智能修复。这能显著提高后续特征提取的成功率。")

        # 1. 筛选可能的 SMILES 列 (文本列)
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        # 简单启发式：默认选中列名包含 'smi' 的列
        default_candidates = [c for c in obj_cols if 'smi' in c.lower()]

        cols_to_clean = st.multiselect(
            "选择要清洗的 SMILES 列",
            options=obj_cols,
            default=default_candidates,
            help="选中列中的无效字符串将被尝试修复；无法修复的将被置为 NaN。"
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            strategy = st.selectbox(
                "清洗/修复策略",
                options=['standard', 'repair', 'strict'],
                index=1,
                format_func=lambda x: {
                    'standard': '标准模式 (基础清洗 + RDKit Canonical)',
                    'repair': '智能修复 (推荐：去除立体标记 / 提取最大片段 / 去除盐)',
                    'strict': '严格模式 (任何解析失败均置 NaN)'
                }[x],
                help="智能修复模式会尝试处理 'Salt.Component' 写法，或去除导致解析失败的手性标记。"
            )
        with col_c2:
            drop_invalid = st.checkbox(
                "删除清洗后仍无效(NaN)的样本行",
                value=False,
                help="如果勾选，那些经过修复仍无法解析为分子的行将被直接删除。"
            )

        # ==== 清洗结果预览（支持多列） ====
        if st.session_state.get("smiles_clean_preview") is not None:
            st.markdown("#### 👀 清洗前后预览（最多50行）")
            st.dataframe(st.session_state["smiles_clean_preview"], use_container_width=True)
            if st.button("🧹 清除预览", key="clear_smiles_preview"):
                st.session_state["smiles_clean_preview"] = None
                st.session_state["smiles_clean_cols"] = None
                st.rerun()

        st.markdown("---")

        if st.button("🧪 执行清洗与修复", type="primary"):
            if not cols_to_clean:
                st.warning("⚠️ 请至少选择一列进行清洗")
            else:
                try:
                    # 调用后端 AdvancedDataCleaner.clean_smiles_columns
                    # 注意：这依赖于您之前在 core/data_processor.py 中添加的方法
                    if not hasattr(cleaner, 'clean_smiles_columns'):
                        st.error("❌ 后端代码未更新：未在 AdvancedDataCleaner 中找到 `clean_smiles_columns` 方法。")
                    else:
                        df_before = df[cols_to_clean].copy()
                        new_df = cleaner.clean_smiles_columns(
                            columns=cols_to_clean,
                            strategy=strategy,
                            drop_invalid=drop_invalid
                        )
                        st.session_state.processed_data = new_df

                        # 保存清洗前后对比预览（多列支持），供 rerun 后展示
                        try:
                            _preview = pd.DataFrame(index=new_df.index)
                            for _c in cols_to_clean:
                                _preview[f"{_c} (before)"] = df_before.get(_c)
                                _preview[f"{_c} (after)"] = new_df.get(_c)
                            st.session_state["smiles_clean_preview"] = _preview.head(50)
                            st.session_state["smiles_clean_cols"] = cols_to_clean
                        except Exception:
                            # 预览失败不影响主流程
                            pass


                        st.success("✅ 清洗完成！")

                        # 显示日志摘要
                        logs = [x for x in cleaner.cleaning_log if x.get('action') == 'clean_smiles']
                        if logs:
                            st.markdown("#### 📊 清洗结果统计")
                            log_data = []
                            for l in logs:
                                log_data.append({
                                    "列名": l['column'],
                                    "原始有效数": l['valid_before'],
                                    "修复后有效数": l['valid_after'],
                                    "最终无效数": l['lost_samples']
                                })
                            st.dataframe(pd.DataFrame(log_data), use_container_width=True)

                        if drop_invalid:
                            dropped_logs = [x for x in cleaner.cleaning_log if
                                            x.get('action') == 'drop_invalid_smiles_rows']
                            if dropped_logs:
                                count = dropped_logs[-1]['rows_dropped']
                                st.warning(f"🗑️ 已删除 {count} 行无效样本")

                        st.rerun()

                except Exception as e:
                    st.error(f"❌ 执行失败: {str(e)}")
                    st.code(traceback.format_exc())

    # ================= [顺延] 原 Tab 5 -> Tab 6: SMILES组分分列 =================
    with tab6:
        # (这里是原来的 "with tab5:" 的内容，不做修改，直接粘贴过来)
        st.markdown("### 🧩 SMILES组分自动分列（树脂/固化剂/改性剂）")
        # ... (原 tab5 代码内容) ...
        # (请确保这里的代码逻辑与原文件一致，只是缩进在 with tab6 下)
        st.info("💡 将单元格内的多组分 SMILES（如 'A;B' 或 'A + B' 或 'A.B'）自动拆分到多列...")

        from core.smiles_utils import split_smiles_column, build_formulation_key
        import re

        text_cols_local = df.select_dtypes(include=['object', 'category']).columns.tolist()
        smiles_cols = [c for c in text_cols_local if 'smiles' in c.lower()]
        candidate_cols = smiles_cols if smiles_cols else text_cols_local

        if not candidate_cols:
            st.warning("⚠️ 未检测到可分列的文本列（object/category）。")
        else:
            # ... (保留原有的分列逻辑代码) ...
            # 为节省篇幅，此处省略中间未修改代码，请保留原 app.py 中该部分逻辑
            # ...
            # ...
            # 直到原 tab5 结束
            pass

            # (以下是原分列逻辑的 UI 组件，需确保它们现在位于 tab6 下)
            # 默认优先：resin_smiles / curing_agent_smiles
            default_cols = []
            for cand in ["resin_smiles", "curing_agent_smiles", "hardener_smiles", "curing_agent",
                         "curing_agent_smiles"]:
                if cand in candidate_cols:
                    default_cols.append(cand)
            if not default_cols:
                default_cols = [candidate_cols[0]]

            cols_to_split = st.multiselect(
                "选择要分列的列",
                options=candidate_cols,
                default=default_cols,
                help="建议至少选择 resin_smiles 与 curing_agent_smiles 两列（如果存在）。",
                key="split_cols_multiselect"  # 加个 key 防止冲突
            )

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                max_components = st.slider("最大分列组分数", 1, 12, 6, help="每列最多拆成多少个组分（*_1~*_k）")
            with col_s2:
                canonicalize = st.checkbox("RDKit canonical 化组分（推荐）", value=True, key="split_canon")
            with col_s3:
                keep_original = st.checkbox("保留原始列", value=True, key="split_keep")

            add_key = st.checkbox("生成 *_key 配方键（排序去重后 '.' 拼接）", value=True, key="split_add_key")
            add_n = st.checkbox("生成 *_n_components 组分数列", value=True, key="split_add_n")

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
                    st.caption("新增列示例（前 20 个）： " + ", ".join(created_cols[:20]) + (
                        " ..." if len(created_cols) > 20 else ""))
                st.rerun()

            st.markdown("---")
            st.markdown("#### 🔎 分列后的类别分布快速体检")

            preview_cols = [c for c in df.columns if c.endswith("_key") or re.search(r"_\d+$", c)]
            if preview_cols:
                prev_col = st.selectbox("选择要查看分布的列", options=preview_cols, key="split_view_col")
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
                        help="将超高频的单体/配方下采样到指定上限，减少数据中“单种分子单体过多”的偏置。",
                        key="split_cap_slider"
                    )
                    if st.button("⚖️ 立即对该列执行平衡", key=f"quick_balance_{prev_col}"):
                        cleaner_tmp = AdvancedDataCleaner(df)
                        balanced_df = cleaner_tmp.balance_category_counts(prev_col, max_samples=int(cap))
                        st.session_state.processed_data = balanced_df
                        st.success(f"✅ 已对 {prev_col} 执行类别平衡（max_samples={int(cap)}）")
                        st.rerun()
            else:
                st.info("当前数据还没有 *_key 或 *_数字 的分列列。你可以先点击上方按钮执行分列。")

    # ================= [顺延] 原 Tab 6 -> Tab 7: 类别平衡 =================
    with tab7:
        # (这里是原来的 "with tab6:" 的内容，不做修改，直接粘贴过来)
        st.markdown("### ⚖️ 类别平衡 (针对化学结构)")
        # ... (原 tab6 代码内容) ...
        # (确保缩进正确)
        st.info("💡 解决特定单体/分子重复次数过多的问题...")

        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            cat_col = st.selectbox("选择要平衡的类别列 (通常是SMILES)", text_cols, key="bal_col_select")

            counts = df[cat_col].value_counts()
            n_unique = len(counts)

            col1, col2, col3 = st.columns(3)
            col1.metric("唯一类别数", n_unique)
            col2.metric("最大样本数", counts.max())
            col3.metric("中位数样本数", int(counts.median()))

            st.markdown("#### Top 10 出现最频繁的分子")
            st.bar_chart(counts.head(10))

            st.markdown("#### 🔧 平衡设置")

            limit_val = st.slider(
                "每个类别的最大样本数 (Max Samples per Category)",
                min_value=1,
                max_value=int(counts.max()),
                value=int(counts.median()) if n_unique > 0 else 10,
                key="bal_slider"
            )

            if st.button(f"⚖️ 执行平衡 (限制为 {limit_val} 个)", type="primary", key="bal_btn"):
                old_len = len(df)
                cleaned_df = cleaner.balance_category_counts(cat_col, max_samples=limit_val)
                new_len = len(cleaned_df)

                st.session_state.processed_data = cleaned_df

                st.success(f"✅ 平衡完成！")
                st.info(f"📊 总样本数从 {old_len} 减少到 {new_len} (删除了 {old_len - new_len} 个过度重复样本)")
                st.rerun()
        else:
            st.warning("⚠️ 没有找到文本列，无法执行类别平衡")


    with tab8:
        st.markdown("### 🧩 K-Means 智能聚类 (文献核心策略)")
        st.info(
            "💡 文献 [Polymer 256 (2022) 125216] 指出，利用 K-Means 将环氧树脂体系分为 11 个簇，可将 R² 提升至 0.99。此功能将生成 'Cluster_Label' 列作为新特征。")

        # 选择用于聚类的特征（通常是分子描述符 + 温度）
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cluster_features = st.multiselect("选择用于聚类的特征", num_cols,
                                          default=num_cols[:5] if len(num_cols) > 5 else num_cols)

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            auto_k = st.checkbox("自动搜索最佳簇数量 (Silhouette Score)", value=True)
        with col_k2:
            n_clusters = st.slider("手动指定簇数量", 2, 20, 11, disabled=auto_k)

        if st.button("🚀 执行 K-Means 聚类", type="primary"):
            cleaned_df, final_k = cleaner.apply_kmeans_clustering(
                feature_cols=cluster_features,
                n_clusters=None if auto_k else n_clusters,
                auto_tune=auto_k
            )
            st.session_state.processed_data = cleaned_df
            st.success(f"✅ 聚类完成！最佳簇数量: {final_k}")
            st.rerun()

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

    # Render operation log panel
    oplog_render()

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
<<<<<<< HEAD
        st.session_state.selected_smiles_col = smiles_col
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

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
            "👆 分子指纹 (MACCS/Morgan)",
            "🔹 RDKit 标准版 (推荐新手)",
            "🚀 RDKit 并行版 (大数据集)",
            "💾 RDKit 内存优化版 (低内存)",
            "🔬 Mordred 描述符 (1600+特征)",
            "🧊 3D构象描述符 (RDKit3D+Coulomb) ",
            "🧩 TDA拓扑特征 (持续同调PH) ",
            "🧠 预训练SMILES Transformer Embedding (ChemBERTa等)",
            "🕸️ 图神经网络特征 (拓扑结构)",
            "⚛️ ML力场特征 (ANI能量/力)",
            "⚗️ 环氧树脂反应特征 (基于领域知识)",
            "📑 FGD 官能团区分",
        ],
        help="不同方法适用于不同场景"
    )


    # Log selected extraction method
    oplog(f"Selected molecular feature method: {extraction_method}")

    # UI 变量初始化
    fp_type = "MACCS"
    fp_bits = 2048
    fp_radius = 2
    hardener_col = None
    hardener_component_cols = None
    hardener_fusion_mode = "仅用于指纹/反应特征（当前默认）"  # 初始化固化剂列变量
    phr_col = None

    # --- [修复] 在这里补上 Transformer 变量的默认初始化 ---
    lm_model_name = "seyonec/ChemBERTa-zinc-base-v1"
    lm_pooling = "cls"
    lm_max_length = 128
    lm_batch_size = 16

    # [新增] TDA 参数默认值
    tda_maxdim = 2
    tda_use_pim = False
    tda_pim_pixels = 10
    tda_pim_spread = 1.0

    # [新增] ML力场(ANI) 参数默认值
    ani_batch_size = 64
    ani_cpu_workers = max(1, (os.cpu_count() or 1) - 1) if os.name != 'nt' else 1

<<<<<<< HEAD
    # [新增] 指纹默认参数
    fp_use_chirality = False
    fp_use_features = False

    # [新增] Mordred 默认参数
    mordred_batch_size = 1000
    mordred_ignore_3d = True

    # [新增] 3D 描述符默认参数
    keep_all_rows_3d = True
    rdkit3d_coulomb_top_k = 10
    rdkit3d_n_jobs = None


=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
    # ============== [修改] 指纹参数设置 ==============
    if "分子指纹" in extraction_method:
        st.info("💡 提示：对于环氧树脂体系，建议同时选择树脂和固化剂列，系统将自动拼接两者的指纹以描述完整网络结构。")

        col_fp1, col_fp2, col_fp3 = st.columns(3)
        with col_fp1:
            fp_type = st.selectbox("指纹类型", ["MACCS", "Morgan"])

<<<<<<< HEAD
        
            drop_all_zero_bits = st.checkbox("移除全为0的指纹位（不推荐：会造成列缺失，影响模型导入/复用）", value=False)
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
        if fp_type == "Morgan":
            with col_fp2:
                fp_radius = st.selectbox("半径 (Radius)", [2, 3, 4], index=0)
            with col_fp3:
                fp_bits = st.selectbox("位长 (Bits)", [1024, 2048, 4096], index=1)

<<<<<<< HEAD
            # [新增] Morgan 额外参数
            col_fpa, col_fpb = st.columns(2)
            with col_fpa:
                fp_use_chirality = st.checkbox("包含手性 (useChirality)", value=False,
                                              help="启用后会把手性信息编码到指纹中（可能提升对手性敏感体系的效果）")
            with col_fpb:
                fp_use_features = st.checkbox("使用 Feature Morgan (FCFP, useFeatures)", value=False,
                                             help="启用后使用 feature-based Morgan 指纹（更偏向官能团/药效团风格）")

=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

        # ---- 预训练 SMILES Transformer Embedding 参数（可选）----
        lm_model_name = "seyonec/ChemBERTa-zinc-base-v1"
        lm_pooling = "cls"
        lm_max_length = 128
        lm_batch_size = 16

        if "Transformer Embedding" in extraction_method:
            st.markdown("#### 🧠 预训练 Transformer 设置")
            st.info("💡 将调用 HuggingFace `transformers` 库。首次运行会自动下载模型权重（需联网）。")

            # 1. 模型名称输入框
            lm_model_name = st.text_input(
                "HuggingFace 模型名称 (Model ID)",
                value=lm_model_name,  # 使用默认值初始化
                help="例如: 'seyonec/ChemBERTa-zinc-base-v1' 或 'DeepChem/ChemBERTa-77M-MTR'"
            )

            col_lm1, col_lm2, col_lm3 = st.columns(3)

            # 2. 池化策略
            with col_lm1:
                lm_pooling = st.selectbox(
                    "池化策略 (Pooling)",
                    ["cls", "mean"],
                    index=["cls", "mean"].index(lm_pooling) if lm_pooling in ["cls", "mean"] else 0,
                    help="CLS: 取首个token向量; Mean: 取所有token均值"
                )

            # 3. 最大长度
            with col_lm2:
                lm_max_length = st.selectbox(
                    "最大序列长度 (Max Length)",
                    [64, 128, 256, 512],
                    index=[64, 128, 256, 512].index(lm_max_length) if lm_max_length in [64, 128, 256, 512] else 1
                )

            # 4. 批大小
            with col_lm3:
                lm_batch_size = st.selectbox(
                    "批处理大小 (Batch Size)",
                    [8, 16, 32, 64, 128],
                    index=[8, 16, 32, 64, 128].index(lm_batch_size) if lm_batch_size in [8, 16, 32, 64, 128] else 1,
                    help="显存越小，请调小此数值"
                )

        # [新增] 双组分选择 UI
        st.markdown("#### 双组分设置 (推荐)")

        # 初始化变量
        hardener_component_cols = []

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            # 1. 基础单列选择
            candidate_cols = ["无 (仅提取单列)"] + [c for c in text_cols if c != smiles_col]
            hardener_col_opt = st.selectbox("选择【固化剂】主列", candidate_cols)

            if hardener_col_opt != "无 (仅提取单列)":
                hardener_col = hardener_col_opt
                hardener_component_cols = [hardener_col]
            else:
                hardener_col = None

        # [新增] 固化剂多组分复选框
        if hardener_col:
            hardener_mix_mode = st.checkbox(
                "固化剂为多组分（多列复配）",
                value=False,
                help="如果你的配方包含多种固化剂（如 hardener_1, hardener_2），请勾选此项进行多列选择。"
            )

            if hardener_mix_mode:
                # 自动正则匹配推荐
                pattern_h = re.compile(rf"^{re.escape(hardener_col)}_\d+$")
                auto_h = [c for c in text_cols if pattern_h.match(c)]

                # 允许用户多选
                hardener_component_cols = st.multiselect(
                    "选择所有固化剂组分列",
                    options=text_cols,
                    default=auto_h if auto_h else [hardener_col],
                    help="系统会将这些列合并提取特征（例如：指纹叠加或结构拼接）"
                )

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

            # ✅ 支持多选：允许选择多个【固化剂】SMILES列（例如 hardener_smiles_1/2/3）
            # 使用 key 交给 Streamlit 管理状态，避免“需要点两次才能选上”的交互问题
            if "epoxy_hardener_cols" not in st.session_state:
                st.session_state["epoxy_hardener_cols"] = ([candidate_cols[0]] if candidate_cols else [])

            hardener_cols = st.multiselect(
                "选择【固化剂】SMILES列",
                options=candidate_cols,
                key="epoxy_hardener_cols",
                help="可多选：系统会把所选固化剂SMILES列合并用于环氧树脂反应特征提取"
            )

            # 为兼容后续逻辑：hardener_col 保留第一个选择；hardener_component_cols 为全部选择
            hardener_col = hardener_cols[0] if hardener_cols else None
            hardener_component_cols = hardener_cols if hardener_cols else None

        with col_p:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            phr_col = st.selectbox("选择【配比】列 (可选)", ["无 (假设理想配比)"] + num_cols)

            stoich_mode = "theoretical"
            if phr_col and phr_col != "无 (假设理想配比)":
                stoich_mode = st.selectbox(
                    "配比含义",
                    ["Resin/Hardener (总质量比, R/H)", "PHR (Hardener per 100 Resin)"],
                    index=0,
                    help="如果你的配比列是‘树脂总量/固化剂总量(R/H)’，选第一项；如果是传统 PHR（每100份树脂对应固化剂份数），选第二项。"
                )


    
    # ============== [新增 UI] ML力场(ANI) 参数 ==============
<<<<<<< HEAD
    
    # ============== [新增 UI] 3D 构象描述符 参数 ==============
    if "3D构象" in extraction_method:
        st.markdown("#### 🧊 3D 构象描述符参数")
        st.info("将生成 3D 构象并计算 Coulomb Matrix / 3D 描述符。耗时较高。")
        col_3d1, col_3d2 = st.columns(2)
        with col_3d1:
            rdkit3d_coulomb_top_k = st.selectbox("Coulomb Top-K", [5, 10, 15, 20, 30, 50], index=[5, 10, 15, 20, 30, 50].index(int(rdkit3d_coulomb_top_k)) if int(rdkit3d_coulomb_top_k) in [5, 10, 15, 20, 30, 50] else 1)
        with col_3d2:
            max_workers = max(1, (os.cpu_count() or 1) - 1) if os.name != 'nt' else 1
            if max_workers <= 1:
                rdkit3d_n_jobs = 1
                st.info("⚠️ 当前环境仅支持 1 个并行进程（Windows/单核环境）。")
            else:
                rdkit3d_n_jobs = st.slider("n_jobs", min_value=1, max_value=max_workers, value=min(max_workers, 2))

        keep_all_rows_3d = st.checkbox(
            "跳过空值/无效SMILES并保留原始行（失败行特征为NaN）",
            value=True,
            help="开启后：即使有大量 NaN/无效 SMILES，也不会丢行；仅对有效样本计算 3D 特征，其余样本特征留空（NaN）。关闭后：只保留成功提取特征的样本行。"
        )

        with st.expander("🧪 3D Diagnostics / 自检（看这里定位失败原因）", expanded=False):
            try:
                import rdkit as _rdkit
                from rdkit.Chem import AllChem as _AllChem
                from core import molecular_features as _mf
                st.write(f"RDKit version: **{getattr(_rdkit, '__version__', 'unknown')}**")
                st.write(f"RDKIT_AVAILABLE in system: **{getattr(_mf, 'RDKIT_AVAILABLE', False)}**")
                st.write(f"Has ETKDGv3: **{hasattr(_AllChem, 'ETKDGv3')}**")
            except Exception as _e:
                st.error(f"Import self-test failed: {_e}")

            col_a, col_b = st.columns(2)
            with col_a:
                run_self_test = st.button("Run 3D self-test", key="rdkit3d_selftest_btn")
            with col_b:
                quick_n = st.number_input("Test first N rows", min_value=1, max_value=200, value=30, step=1, key="rdkit3d_selftest_n")

            if run_self_test:
                try:
                    from core.molecular_features import _rdkit3d_feature_worker, rdkit3d_debug_one
                    st.caption("1) 用最简单分子 CCO 测试 3D pipeline（应当成功）。")
                    out0 = _rdkit3d_feature_worker("CCO", coulomb_top_k=int(rdkit3d_coulomb_top_k))
                    st.write("CCO worker:", "✅ OK" if out0 is not None else "❌ None (failed)")
                    if out0 is None:
                        st.json(rdkit3d_debug_one("CCO", coulomb_top_k=int(rdkit3d_coulomb_top_k)))

                    st.caption("2) 测试你当前选择列的前 N 行（仅统计成功/失败）。")
                    try:
                        # 兼容：页面主选择变量通常叫 smiles_col；若不存在再尝试其它名字
                        _col = (locals().get("smiles_col", None) or locals().get("resin_smiles_col", None) or st.session_state.get("selected_smiles_col", None))
                        if _col is None:
                            st.warning("无法自动确定要测试的 SMILES 列名（请先在页面上方选择 SMILES 列并重新展开自检）。")
                            sample_list = []
                        else:
                            exists = bool((_col in df.columns)) if "df" in locals() else False
                            st.write(f"Testing column: `{_col}` | exists in df: **{exists}**")
                            sample_list = df[_col].tolist()[: int(quick_n)] if exists else []
                            if len(sample_list) == 0:
                                st.warning("未获取到任何样本行用于测试：可能 df 未加载、列名不在 df 中，或该列全为空。")
                    except Exception as _e:
                        st.error(f"加载样本失败: {_e}")
                        sample_list = []
                    ok_cnt = 0
                    first_fail = None
                    for _s in sample_list:
                        out = _rdkit3d_feature_worker(_s, coulomb_top_k=int(rdkit3d_coulomb_top_k))
                        if out is None and first_fail is None:
                            first_fail = _s
                        if out is not None:
                            ok_cnt += 1
                    st.write(f"First {len(sample_list)} rows: ✅ {ok_cnt} success / ❌ {len(sample_list)-ok_cnt} fail")
                    if first_fail is not None and ok_cnt == 0:
                        st.warning("所有样本都失败了：下面给出第一条失败样本的详细诊断（解析/嵌入阶段原因）。")
                        st.code(str(first_fail)[:300])
                        st.json(rdkit3d_debug_one(first_fail, coulomb_top_k=int(rdkit3d_coulomb_top_k)))
                except Exception as e:
                    st.error(f"Self-test failed: {e}")

=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
    if "ML力场特征" in extraction_method:
        st.markdown("#### ⚛️ ML 力场 (ANI2x) 参数")
        st.info("该方法会先生成3D构象，再用 ANI2x 推理能量/力。较耗时，建议调大批量并使用多核CPU。")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            ani_batch_size = st.selectbox("ANI Batch Size", [16, 32, 64, 128], index=2)
        with col_a2:
            max_workers = max(1, (os.cpu_count() or 1) - 1) if os.name != 'nt' else 1
            if max_workers <= 1:
                ani_cpu_workers = 1
                st.info("⚠️ 当前环境仅支持 1 个 CPU worker（Windows / 单核环境）。")
            else:
                ani_cpu_workers = st.slider(
                    "CPU Workers (3D Generation)",
                    min_value=1,
                    max_value=max_workers,
                    value=min(max_workers, ani_cpu_workers)
                )
        st.caption("提示：3D 构象生成使用多进程；ANI 推理在主进程使用 Torch CPU 多线程。")
<<<<<<< HEAD

    # ============== [新增 UI] Mordred 参数 ==============
    if "Mordred" in extraction_method:
        st.markdown("#### 🔬 Mordred 参数")
        st.info("Mordred 输出 1600+ 描述符，耗时与内存都较高。建议先用较小 batch_size 测试。")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mordred_batch_size = st.number_input("batch_size", min_value=100, max_value=5000, value=int(mordred_batch_size), step=100)
        with col_m2:
            mordred_ignore_3d = st.checkbox("ignore_3D (推荐 True)", value=bool(mordred_ignore_3d),
                                            help="True: 仅计算 2D 描述符，更稳定、更快；False: 允许 3D 相关描述符（更慢、对构象敏感）")

=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
# ============== [新增 UI] TDA 参数 ==============
    if "TDA拓扑特征" in extraction_method:
        st.markdown("#### 🧩 TDA(持续同调) 参数")
        st.info("需要安装 ripser/persim：pip install ripser persim。TDA 将把 3D 构象点云转为 Betti0/1/2 的拓扑统计特征。")

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            tda_maxdim = st.selectbox("maxdim (0/1/2)", [0, 1, 2], index=2)
        with col_t2:
            tda_use_pim = st.checkbox("使用 Persistence Image（高维）", value=False)
        with col_t3:
            tda_pim_pixels = st.selectbox("PIM 像素边长", [8, 10, 16, 20], index=1, disabled=(not tda_use_pim))
        with col_t4:
            tda_pim_spread = st.number_input("PIM spread", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                             disabled=(not tda_use_pim))

        # 速度/稳定性选项（推荐默认即可）
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            tda_add_hs = st.checkbox("添加氢原子（更慢）", value=False)
        with col_s2:
            tda_do_optimize = st.checkbox("力场优化MMFF/UFF（更慢）", value=False)
        with col_s3:
            tda_max_points = st.selectbox("最大点数(下采样加速)", ["不限制", 64, 128, 256, 512, 1024], index=3)
        st.caption("提示：TDA 默认只用重原子坐标；通常不需要加氢/力场优化。点数越大越慢。")

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
            if hardener_component_cols is None:
                hardener_component_cols = [hardener_col]

            # 注意：这里直接使用 UI 中生成的 hardener_component_cols 列表
            # 移除了原有的自动正则覆盖逻辑，完全尊重用户在 UI 上的选择

            hardener_smiles_series, hardener_ncomp = _combine_components(df, hardener_component_cols)
            hardener_list = hardener_smiles_series.tolist()
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
<<<<<<< HEAD
                import inspect
                _kwargs = dict(
                    smiles_list_2=hardener_list,
                    fp_type=fp_type, n_bits=fp_bits, radius=fp_radius,
                    use_chirality=bool(fp_use_chirality),
                    use_features=bool(fp_use_features),
                )
                # 兼容旧版本：若 FingerprintExtractor 未实现该参数，则自动忽略，避免报错
                try:
                    if "drop_all_zero_bits" in inspect.signature(extractor.smiles_to_fingerprints).parameters:
                        _kwargs["drop_all_zero_bits"] = bool(drop_all_zero_bits)
                except Exception:
                    pass

                features_df, valid_indices = extractor.smiles_to_fingerprints(
                    smiles_list,
                    **_kwargs
=======
                features_df, valid_indices = extractor.smiles_to_fingerprints(
                    smiles_list,
                    smiles_list_2=hardener_list,
                    fp_type=fp_type, n_bits=fp_bits, radius=fp_radius
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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
<<<<<<< HEAD
                features_df, valid_indices = extractor.smiles_to_mordred(smiles_list_input, batch_size=int(mordred_batch_size), ignore_3D=bool(mordred_ignore_3d))
=======
                features_df, valid_indices = extractor.smiles_to_mordred(smiles_list_input)
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

            elif "3D构象" in extraction_method:
                from core.molecular_features import RDKit3DDescriptorExtractor
                status_text.text("正在提取RDKit 3D构象描述符...")
<<<<<<< HEAD
                extractor = RDKit3DDescriptorExtractor(coulomb_top_k=int(rdkit3d_coulomb_top_k))
                features_df, valid_indices = extractor.smiles_to_3d_descriptors(smiles_list_input, n_jobs=rdkit3d_n_jobs)
=======
                extractor = RDKit3DDescriptorExtractor()
                features_df, valid_indices = extractor.smiles_to_3d_descriptors(smiles_list_input)
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

            elif "TDA拓扑特征" in extraction_method:
                from core.tda_features import PersistentHomologyFeatureExtractor, TDAConfig
                status_text.text("正在提取 TDA 拓扑特征（持续同调）...")

                config = TDAConfig(
                    maxdim=int(tda_maxdim),
                    use_persistence_image=bool(tda_use_pim),
                    pim_size=(int(tda_pim_pixels), int(tda_pim_pixels)),
                    pim_spread=float(tda_pim_spread),
                    max_points=None if str(tda_max_points) == "不限制" else int(tda_max_points),
                    do_optimize=bool(tda_do_optimize),
                )
                extractor = PersistentHomologyFeatureExtractor(config)
                if not getattr(extractor, "AVAILABLE", False):
                    st.error("❌ 未检测到 ripser/persim，请先安装：pip install ripser persim")
                    return

                features_df, valid_indices = extractor.smiles_to_tda_features(smiles_list_input, add_hs=bool(tda_add_hs))

            elif "Transformer Embedding" in extraction_method:
                from core.molecular_features import SmilesTransformerEmbeddingExtractor
                oplog(f"Running Transformer embedding: model={lm_model_name}, pooling={lm_pooling}, max_length={lm_max_length}, batch={lm_batch_size}")
                oplog(f"SMILES sources (resin_component_cols): {resin_component_cols}")
                if hardener_component_cols is not None:
                    oplog(f"SMILES sources (hardener_component_cols): {hardener_component_cols}")
                oplog(f"Hardener fusion mode: {hardener_fusion_mode}")
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
                oplog("Running ML force field features (ANI2x): 3D generation + ANI inference")
                status_text.text("正在计算ANI力场特征...")
                extractor = MLForceFieldExtractor()
                if not extractor.AVAILABLE:
                    st.error("TorchANI 未安装")
                    return
                oplog(f"ANI params: batch_size={ani_batch_size}, cpu_workers(3D)={ani_cpu_workers}")
                features_df, valid_indices = extractor.smiles_to_ani_features(smiles_list_input, batch_size=ani_batch_size, n_jobs=ani_cpu_workers)
            elif "FGD" in extraction_method:
                from core.molecular_features import FGDFeatureExtractor
                status_text.text("正在执行 FGD 结构分类与编码...")

                extractor = FGDFeatureExtractor()
                # 使用 smiles_list_input (这是上面已经处理过多组分拼接的变量)
                features_df, valid_indices = extractor.categorize_smiles(smiles_list_input)

                if not features_df.empty:
                    # --- [关键步骤] 自动 One-Hot 编码 ---
                    # 文献中 FGD 必须配合 OHE (One-Hot Encoding) 使用
                    st.info("ℹ️ 已提取 FGD 类别特征，正在自动执行 One-Hot 编码以适配模型...")

                    features_df = pd.get_dummies(
                        features_df,
                        columns=["FGD_Substrate", "FGD_Group"],
                        prefix=["Substrate", "Group"],
                        dtype=int
                    )
                    # -----------------------------------


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
                features_df, valid_indices = extractor.extract_features(smiles_list, hardener_list, phr_list, stoich_mode)

            progress_bar.progress(100)

            # --- 合并结果逻辑 ---
            if len(features_df) > 0:
                st.session_state.molecular_features = features_df
                prefix = f"{smiles_col}_"  # default
                # ✅ 更清晰的命名：多组分/拼接模式不再使用“第一列名”作为前缀
                try:
                    if hardener_list and isinstance(hardener_fusion_mode, str) and hardener_fusion_mode.startswith("拼接SMILES"):
                        prefix = "resin_hardener_"
                    elif resin_mix_mode and isinstance(resin_component_cols, list) and len(resin_component_cols) > 1:
                        prefix = f"multi_smiles_{len(resin_component_cols)}_"
                except Exception:
                    pass
                features_df = features_df.add_prefix(prefix)

<<<<<<< HEAD
                # -----------------------------
                # 合并策略：
                # - keep_all_rows_3d=True：保留原始所有行；仅对 valid_indices 填充特征，其余为 NaN（推荐，适合大量空值场景）
                # - 否则：仅保留成功提取特征的样本行（原逻辑）
                # -----------------------------
                features_df = features_df.reset_index(drop=True)

                if 'keep_all_rows_3d' in locals() and keep_all_rows_3d:
                    base_df = df.reset_index(drop=True)

                    # 防止列名冲突：如果新特征名已存在，先删除旧的
                    cols_to_drop = [col for col in features_df.columns if col in base_df.columns]
                    if cols_to_drop:
                        base_df = base_df.drop(columns=cols_to_drop)

                    # 构建全量特征表并按 valid_indices 回填
                    full_feat = pd.DataFrame(index=range(len(base_df)), columns=features_df.columns, dtype=float)
                    if valid_indices:
                        full_feat.iloc[valid_indices, :] = features_df.values

                    merged_df = pd.concat([base_df, full_feat], axis=1)
                else:
                    df_valid = df.iloc[valid_indices].reset_index(drop=True)

                    # 防止列名冲突：如果新特征名已存在，先删除旧的
                    cols_to_drop = [col for col in features_df.columns if col in df_valid.columns]
                    if cols_to_drop:
                        df_valid = df_valid.drop(columns=cols_to_drop)

                    merged_df = pd.concat([df_valid, features_df], axis=1)
=======
                df_valid = df.iloc[valid_indices].reset_index(drop=True)
                features_df = features_df.reset_index(drop=True)

                # 防止列名冲突：如果新特征名已存在，先删除旧的
                cols_to_drop = [col for col in features_df.columns if col in df_valid.columns]
                if cols_to_drop:
                    df_valid = df_valid.drop(columns=cols_to_drop)

                merged_df = pd.concat([df_valid, features_df], axis=1)
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

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

                log_fe_step(
                    operation="分子特征提取",
                    description=f"方法: {extraction_method} / 列: {smiles_col}",
                    params={"method": extraction_method, "smiles_col": smiles_col, "n_features": int(features_df.shape[1]), "n_samples": int(len(valid_indices))},
                    input_df=df,
                    output_df=merged_df,
                    features_added=features_df.columns.tolist(),
                    message=f"新增分子特征 {features_df.shape[1]} 列，数据列数 {df.shape[1]} → {merged_df.shape[1]}"
                )

                # 结果统计
                col1, col2, col3 = st.columns(3)
                col1.metric("有效样本", len(valid_indices))
                col2.metric("特征数量", features_df.shape[1])
                col3.metric("双组分模式", "是" if hardener_list else "否")

                st.markdown("### 📋 特征预览")
                st.dataframe(features_df.head(), use_container_width=True)
            else:
<<<<<<< HEAD
                st.error("❌ 未能提取任何特征：当前选择的 SMILES 列可能全部为空/无效，或 3D 构象生成全部失败。")
                # 额外诊断：快速检查前 200 条是否能被 RDKit 解析（不做 3D）
                try:
                    ok, checked, bad = _quick_rdkit_parse_stats(smiles_list_input, max_check=200)
                    st.info(f"RDKit 解析自检：检查 {checked} 条中，有 {ok} 条至少包含一个可解析片段。")
                    if ok == 0 and bad:
                        st.warning("示例（可能不是合法 SMILES）：\n- " + "\n- ".join(bad))
                        st.caption("若这些字符串看起来像名称/配方键（含中文/单位/特殊分隔符等），请改选真正的 SMILES 列；或先清洗后再做 3D。")
                        st.caption("如果示例是正常 SMILES，但仍全部失败，常见原因是 RDKit 版本过旧（不支持 ETKDGv3）。本版本已自动回退 ETKDGv2/ETKDG；也建议升级 rdkit。")
                except Exception:
                    pass

                st.info(f"总行数={len(df)}，树脂/主体 SMILES 非空数≈{pd.Series(smiles_list_input).replace(['nan','NaN','<NA>'], np.nan).notna().sum()}（仅粗略统计）")
                st.caption("建议：1) 确认选择了正确的 SMILES 列；2) 先把 n_jobs 调到 1；3) 先用少量样本测试；4) 多组分/含盐/含金属体系更易失败。")
=======
                st.error("❌ 未能提取任何特征，请检查SMILES格式")
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4

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

        # 获取模型目录（包含可用性/缺失依赖原因）；兼容旧版本 trainer
        if hasattr(trainer, "get_model_catalog"):
            model_catalog = trainer.get_model_catalog()
            model_options = trainer.get_available_models(include_unavailable=True)
        else:
            # 旧版本：仅提供可用模型
            model_options = trainer.get_available_models()
            model_catalog = {m: {"available": True, "reason": ""} for m in model_options}

            # 兼容：即使当前环境未安装 TensorFlow，也显示 TFS 入口（训练会被禁用并提示安装）
            if "TensorFlow Sequential" not in model_options:
                model_options = model_options + ["TensorFlow Sequential"]
                model_catalog["TensorFlow Sequential"] = {
                    "available": bool(TENSORFLOW_AVAILABLE),
                    "reason": "" if TENSORFLOW_AVAILABLE else "未安装 TensorFlow（pip install tensorflow）"
                }

        def _fmt_model_name(n: str) -> str:
            label = "TFS（TensorFlow Sequential）" if n == "TensorFlow Sequential" else n
            meta = model_catalog.get(n, {})
            if meta.get("available", True):
                return f"{label} ✅"
            return f"{label} ⛔"

        model_name = st.selectbox("选择模型", model_options, format_func=_fmt_model_name)

        meta = model_catalog.get(model_name, {"available": True, "reason": ""})
        disable_train = (not meta.get("available", True))
        if disable_train:
            reason = meta.get("reason") or "当前环境缺少依赖"
            st.warning(f"该模型当前不可训练：{reason}")

        st.markdown("### ⚙️ 训练设置")
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
        random_state = st.number_input("随机种子", 0, 1000000, 42)

<<<<<<< HEAD
        # 并行训练核数（对支持 n_jobs/thread_count 的算法生效；其它算法自动忽略）
        cpu_total = os.cpu_count() or 1
        core_opts = ["Auto (all cores)"] + [str(i) for i in range(1, min(cpu_total, 64) + 1)]
        core_sel = st.selectbox("训练并行核数", core_opts, index=0,
                              help="会应用到 RandomForest/ExtraTrees/XGBoost/LightGBM/CatBoost/部分线性模型等。Auto=使用全部CPU核心。")
        train_n_jobs = -1 if core_sel.startswith("Auto") else int(core_sel)

=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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

        # 参数配置优先级：TFS 使用 core.tf_model 中的配置（支持 checkbox 等），其余模型使用 ui_config
        configs = []
        if model_name == "TensorFlow Sequential":
            configs = TFS_TUNING_PARAMS if TFS_TUNING_PARAMS else MANUAL_TUNING_PARAMS.get(model_name, [])
        elif model_name in MANUAL_TUNING_PARAMS:
            configs = MANUAL_TUNING_PARAMS[model_name]

        if configs:
            p_cols = st.columns(2)
            for i, config in enumerate(configs):
                with p_cols[i % 2]:
                    key = f"param_{model_name}_{config['name']}"
                    if key not in st.session_state:
                        st.session_state[key] = config.get('default')

                    help_txt = config.get('help', None)
                    widget = config.get('widget', 'text_input')
                    args = config.get('args', {}) or {}

                    # 对 TFS：若关闭 early_stopping，则 patience 不必填写
                    disabled_flag = False
                    if model_name == "TensorFlow Sequential" and config.get('name') == 'patience':
                        disabled_flag = (not bool(manual_params.get('early_stopping', True)))

                    if widget == 'slider':
                        manual_params[config['name']] = st.slider(config['label'], key=key, help=help_txt, disabled=disabled_flag, **args)
                    elif widget == 'number_input':
                        manual_params[config['name']] = st.number_input(config['label'], key=key, help=help_txt, disabled=disabled_flag, **args)
                    elif widget == 'selectbox':
                        manual_params[config['name']] = st.selectbox(config['label'], options=args.get('options', []), key=key, help=help_txt, disabled=disabled_flag)
                    elif widget == 'text_input':
                        manual_params[config['name']] = st.text_input(config['label'], key=key, help=help_txt, disabled=disabled_flag)
                    elif widget == 'checkbox':
                        manual_params[config['name']] = st.checkbox(config['label'], key=key, help=help_txt, disabled=disabled_flag)

    st.markdown("---")

    # 按钮区
    c_btn1, c_btn2 = st.columns(2)

    with c_btn1:
        if st.button("🚀 开始训练", type="primary", disabled=disable_train if "disable_train" in locals() else False):
            with st.spinner("训练中..."):
                try:
                    # 准备参数
                    params = manual_params.copy()
<<<<<<< HEAD
                    params['train_n_jobs'] = int(train_n_jobs)
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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

                    # [新增] 训练过程写入状态条
                    try:
                        log_fe_step(
                            operation="模型训练",
                            description=f"训练完成: {model_name}",
                            params={
                                "model": model_name,
                                "test_size": float(test_size),
                                "split_strategy": str(split_strategy),
                                "n_features": int(len(st.session_state.get('feature_cols') or [])),
                                **(params or {})
                            },
                            input_df=df,
                            status="success",
                            message=f"R²={res.get('r2', 0):.4f}, RMSE={res.get('rmse', 0):.4f}, MAE={res.get('mae', 0):.4f}"
                        )
                    except Exception:
                        pass


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

                    # --- [增强] 所有模型的训练曲线 + 训练记录落盘 ---
                    try:
                        history = res.get('training_history') or {}
                        # 图内标题尽量用英文，避免服务器环境缺少中文字体导致方块/乱码
                        fig_curve, hist_export_df = plot_history(history, title=f"{model_name} Training Curves")

                        st.markdown("### 📉 训练曲线（所有模型）")
                        st.pyplot(fig_curve, use_container_width=True)

                        if hist_export_df is not None and not hist_export_df.empty:
                            with st.expander("🧾 查看训练曲线数据", expanded=False):
                                st.dataframe(hist_export_df, use_container_width=True, height=240)
                                st.download_button(
                                    "📥 导出训练曲线 CSV",
                                    hist_export_df.to_csv(index=False).encode("utf-8-sig"),
                                    f"{model_name}_training_history.csv",
                                    "text/csv"
                                )

                        # 保存一次训练 Run（指标+参数+曲线）
                        manager = TrainingRunManager()
                        meta = {
                            "model_name": model_name,
                            "r2": float(res.get('r2', 0)),
                            "rmse": float(res.get('rmse', 0)),
                            "mae": float(res.get('mae', 0)),
                            "train_time": float(res.get('train_time', 0)),
                            "split_strategy": str(res.get('split_strategy', '')),
                            "test_size": float(test_size),
                            "random_state": int(random_state),
                            "params": params or {},
                            "n_samples": int(len(res.get('y_train', [])) + len(res.get('y_test', []))),
                            "n_features": int(len(st.session_state.get('feature_cols') or [])),
                        }
                        summary = manager.save_run(
                            model_name=model_name,
                            metadata=meta,
                            history_df=hist_export_df,
                            curve_fig=fig_curve,
                        )
                        st.session_state.last_training_run_id = summary.run_id
                        st.caption(f"🗂️ 已保存训练记录: {summary.run_id}（可在【📈 训练记录】查看）")
                    except Exception:
                        pass

                    # --- [新增] TFS 网络结构 Summary ---
                    if model_name == "TensorFlow Sequential":
                        try:
                            summary_str = ""
                            if hasattr(st.session_state.model, "get_model_summary_str"):
                                summary_str = st.session_state.model.get_model_summary_str() or ""
                            if summary_str.strip():
                                with st.expander("🧾 TFS 网络结构（Model Summary）", expanded=False):
                                    st.code(summary_str)
                        except Exception:
                            pass

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
                    # [新增] 失败也写入状态条
                    try:
                        log_fe_step(
                            operation="模型训练",
                            description=f"训练失败: {model_name}",
                            params={"model": model_name},
                            input_df=df,
                            status="error",
                            message=str(e)
                        )
                    except Exception:
                        pass

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

<<<<<<< HEAD
                st.markdown("---")
                with st.expander("📦 导出训练好的模型（.joblib）", expanded=False):
                    st.caption("导出的模型文件包含：pipeline/模型、特征列、目标列、评估指标等。可在“预测应用”页面直接导入使用。")
                    try:
                        from core.model_io import create_model_artifact_bytes
                        metrics = {}
                        tr = st.session_state.get("train_result") or {}
                        for k in ["r2", "rmse", "mae", "train_time", "split_strategy", "n_bins"]:
                            if k in tr:
                                metrics[k] = tr[k]
                        model_bytes = create_model_artifact_bytes(
                            model_name=str(st.session_state.get("model_name") or model_name),
                            target_col=str(st.session_state.get("target_col") or ""),
                            feature_cols=list(st.session_state.get("feature_cols") or []),
                            model=st.session_state.get("model"),
                            pipeline=st.session_state.get("pipeline"),
                            scaler=st.session_state.get("scaler"),
                            imputer=st.session_state.get("imputer"),
                            metrics=metrics,
                            extra={
                                "app_version": str(VERSION),
                            },
                        )
                        safe_name = (str(st.session_state.get("model_name") or model_name) or "model").replace(" ", "_")
                        st.download_button(
                            "⬇️ 下载模型文件",
                            data=model_bytes,
                            file_name=f"{safe_name}_artifact.joblib",
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"模型导出失败：{e}")
                        st.info("提示：若使用深度学习模型（TF/自定义网络），joblib 序列化可能失败。可改用“导出训练脚本”在目标环境复现训练。")


=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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
    feature_names = st.session_state.feature_cols or []
    n_features = len(feature_names)

    tab1, tab2, tab3 = st.tabs(["🔍 SHAP分析", "📈 预测性能", "🎯 特征重要性"])

    def _export_matplotlib_fig(fig, base_name: str, key_prefix: str):
        """Export matplotlib fig as PNG/HTML download buttons."""
        if fig is None:
            return
        try:
            png_bytes = fig_to_png_bytes(fig)
            st.download_button(
                "📥 导出图像 PNG",
                png_bytes,
                file_name=f"{base_name}.png",
                mime="image/png",
                key=f"{key_prefix}_png",
            )
            html_str = fig_to_html(fig, title=base_name)
            st.download_button(
                "📥 导出图像 HTML",
                html_str.encode("utf-8"),
                file_name=f"{base_name}.html",
                mime="text/html",
                key=f"{key_prefix}_html",
            )
        except Exception as e:
            st.warning(f"图像导出失败: {e}")

    # --- 1) SHAP / 快速解释 ---
    with tab1:
        st.markdown("### 特征解释")

        default_fast = n_features >= 300
        method = st.radio(
            "解释方法",
            ["SHAP（更准确，可能较慢）", "快速模式（Permutation Importance，推荐）"],
            index=1 if default_fast else 0,
            horizontal=True,
            key="interp_method",
        )

        if n_features >= 300:
            st.warning(
                f"当前特征数量为 {n_features}。若模型不是树/线性模型，SHAP（尤其 KernelExplainer）可能非常慢甚至卡住。建议使用“快速模式”。"
            )

        # ---------- SHAP ----------
        if method.startswith("SHAP"):
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            with c_opt1:
                plot_type = st.selectbox("图表类型", ["bar", "beeswarm"], index=0, key="shap_plot_type")
            with c_opt2:
                max_display = st.slider("显示特征数量", 5, 50, 20, key="shap_max_display")
            with c_opt3:
                # 限制采样条数，提速
                max_val = max(20, min(500, len(X_test)))
                max_samples = int(
                    st.number_input(
                        "采样条数（越小越快）",
                        min_value=20,
                        max_value=int(max_val),
                        value=int(min(200, len(X_test))),
                        step=10,
                        key="shap_max_samples",
                    )
                )

            c_k1, c_k2 = st.columns(2)
            with c_k1:
                kernel_bg_default = 20 if n_features >= 300 else 50
                kernel_bg = int(
                    st.number_input(
                        "Kernel 背景样本数",
                        min_value=5,
                        max_value=100,
                        value=int(kernel_bg_default),
                        step=5,
                        key="shap_kernel_bg",
                    )
                )
            with c_k2:
                kernel_ns = int(
                    st.number_input(
                        "Kernel nsamples",
                        min_value=50,
                        max_value=2000,
                        value=200,
                        step=50,
                        key="shap_kernel_nsamples",
                    )
                )

            if st.button("🔍 计算SHAP值", key="btn_compute_shap"):
                with st.spinner("正在计算 SHAP 值 (可能较慢)..."):
                    try:
                        interp = EnhancedModelInterpreter(
                            model,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            model_name,
                            feature_names=feature_names,
                            max_samples=max_samples,
                            kernel_background=kernel_bg,
                            kernel_nsamples=kernel_ns,
                        )
                        fig, df_shap = interp.plot_summary(plot_type=plot_type, max_display=max_display)

                        if fig:
                            c1, c2, c3 = st.columns([1, 6, 1])
                            with c2:
                                st.pyplot(fig, use_container_width=True)

                                if df_shap is not None:
                                    csv = df_shap.to_csv(index=False).encode("utf-8-sig")
                                    st.download_button(
                                        "📥 导出 SHAP 数据 (CSV)",
                                        csv,
                                        "shap_values.csv",
                                        "text/csv",
                                        key="shap_csv",
                                    )

                                _export_matplotlib_fig(fig, base_name="shap_summary", key_prefix="shap_fig")
                        else:
                            st.error("无法生成 SHAP 图，请检查模型是否支持。")
                    except Exception as e:
                        st.error(f"计算出错: {str(e)}")
            else:
                st.caption("提示：SHAP 结果会受到采样条数/背景样本的影响；特征数很大时推荐使用快速模式。")

        # ---------- Permutation (Fast) ----------
        else:
            st.markdown("#### ⚡ 快速重要性（Permutation Importance）")

            c_q1, c_q2, c_q3, c_q4 = st.columns(4)
            with c_q1:
                top_n = st.slider("显示特征数量", 5, 50, 20, key="perm_top_n")
            with c_q2:
                n_repeats = st.slider("重复次数", 1, 10, 3, key="perm_repeats")
            with c_q3:
                max_val = max(30, min(1000, len(X_test)))
                sample_n = int(
                    st.number_input(
                        "采样条数",
                        min_value=30,
                        max_value=int(max_val),
                        value=int(min(200, len(X_test))),
                        step=20,
                        key="perm_sample_n",
                    )
                )
            with c_q4:
                scoring = st.selectbox("评分指标", ["r2", "neg_root_mean_squared_error"], index=0, key="perm_scoring")

            if st.button("⚡ 计算快速重要性", key="btn_compute_perm"):
                with st.spinner("正在计算 permutation importance..."):
                    try:
                        from sklearn.inspection import permutation_importance

                        # 确保 X/y 的索引对齐
                        if isinstance(X_test, pd.DataFrame):
                            Xdf = X_test.copy()
                        else:
                            Xdf = pd.DataFrame(np.asarray(X_test), columns=feature_names)

                        if isinstance(y_test, pd.Series):
                            y_series = y_test.copy()
                        elif isinstance(y_test, pd.DataFrame):
                            y_series = y_test.iloc[:, 0].copy()
                        else:
                            y_series = pd.Series(np.asarray(y_test).ravel(), index=Xdf.index)

                        if len(Xdf) > sample_n:
                            X_sample = Xdf.sample(n=sample_n, random_state=42)
                            y_sample = y_series.loc[X_sample.index]
                        else:
                            X_sample = Xdf
                            y_sample = y_series

                        result = permutation_importance(
                            model,
                            X_sample,
                            np.asarray(y_sample).ravel(),
                            n_repeats=int(n_repeats),
                            random_state=42,
                            scoring=scoring,
                        )

                        df_perm = pd.DataFrame(
                            {
                                "Feature": feature_names,
                                "Importance": result.importances_mean,
                                "Std": result.importances_std,
                            }
                        ).sort_values("Importance", ascending=False)

                        viz = Visualizer()
                        fig, _ = viz.plot_feature_importance(
                            df_perm["Importance"].values,
                            df_perm["Feature"].values.tolist(),
                            f"{model_name} - Permutation",
                            top_n=int(top_n),
                        )

                        c1, c2, c3 = st.columns([1, 6, 1])
                        with c2:
                            st.pyplot(fig, use_container_width=True)

                            csv = df_perm.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "📥 导出 permutation 数据 (CSV)",
                                csv,
                                "permutation_importance.csv",
                                "text/csv",
                                key="perm_csv",
                            )

                            _export_matplotlib_fig(fig, base_name="permutation_importance", key_prefix="perm_fig")

                    except Exception as e:
                        st.error(f"计算失败: {e}")
            else:
                st.caption("Permutation importance 对模型无假设、速度更快；数值越大表示该特征越重要。")

    # --- 2) 预测性能 ---
    with tab2:
        st.markdown("### 预测性能")
        visualizer = Visualizer()

        try:
            y_pred = st.session_state.train_result["y_pred"] if st.session_state.get("train_result") else None
        except Exception:
            y_pred = None

        if y_pred is None:
            st.warning("缺少预测结果，请先在训练页完成一次训练。")
        else:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig, df_res = visualizer.plot_residuals(y_test, y_pred, model_name)
                st.pyplot(fig, use_container_width=True)

                if df_res is not None:
                    csv = df_res.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 导出残差数据 (CSV)",
                        csv,
                        "residuals.csv",
                        "text/csv",
                        key="res_csv",
                    )

                _export_matplotlib_fig(fig, base_name="residuals", key_prefix="res_fig")

    # --- 3) 特征重要性 ---
    with tab3:
        st.markdown("### 特征重要性")
        top_n = st.slider("显示特征数量", 5, 50, 20, key="fi_top_n")

        if hasattr(model, "feature_importances_"):
            visualizer = Visualizer()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig, df_imp = visualizer.plot_feature_importance(
                    model.feature_importances_, feature_names, model_name, top_n=int(top_n)
                )
                st.pyplot(fig, use_container_width=True)

                if df_imp is not None:
                    csv = df_imp.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 导出重要性数据 (CSV)",
                        csv,
                        "importance.csv",
                        "text/csv",
                        key="fi_csv",
                    )

                _export_matplotlib_fig(fig, base_name="feature_importance", key_prefix="fi_fig")

            # MACCS 解释表
            if df_imp is not None and not df_imp.empty:
                st.markdown("#### 🧬 特征含义解析（Top 15）")
                exps = []
                for f in df_imp.head(15)["Feature"]:
                    desc = "数值特征"
                    if "MACCS" in str(f):
                        try:
                            from core.molecular_features import get_maccs_description

                            idx = int(str(f).split("_")[-1])
                            desc = get_maccs_description(idx)
                        except Exception:
                            desc = "MACCS 指纹片段"
                    exps.append({"特征名": f, "含义": desc})
                st.table(pd.DataFrame(exps))
        else:
            st.info("该模型不支持原生 feature_importances_。可在【SHAP分析】中使用 SHAP 或快速模式。")
            st.markdown("#### （可选）用 permutation importance 作为替代")

            c_q1, c_q2, c_q3 = st.columns(3)
            with c_q1:
                n_repeats = st.slider("重复次数", 1, 10, 3, key="fi_perm_repeats")
            with c_q2:
                max_val = max(30, min(1000, len(X_test)))
                sample_n = int(
                    st.number_input(
                        "采样条数",
                        min_value=30,
                        max_value=int(max_val),
                        value=int(min(200, len(X_test))),
                        step=20,
                        key="fi_perm_sample",
                    )
                )
            with c_q3:
                scoring = st.selectbox(
                    "评分指标", ["r2", "neg_root_mean_squared_error"], index=0, key="fi_perm_scoring"
                )

            if st.button("⚡ 计算替代重要性", key="btn_fi_perm"):
                with st.spinner("正在计算 permutation importance..."):
                    try:
                        from sklearn.inspection import permutation_importance

                        if isinstance(X_test, pd.DataFrame):
                            Xdf = X_test.copy()
                        else:
                            Xdf = pd.DataFrame(np.asarray(X_test), columns=feature_names)

                        if isinstance(y_test, pd.Series):
                            y_series = y_test.copy()
                        elif isinstance(y_test, pd.DataFrame):
                            y_series = y_test.iloc[:, 0].copy()
                        else:
                            y_series = pd.Series(np.asarray(y_test).ravel(), index=Xdf.index)

                        if len(Xdf) > sample_n:
                            X_sample = Xdf.sample(n=sample_n, random_state=42)
                            y_sample = y_series.loc[X_sample.index]
                        else:
                            X_sample = Xdf
                            y_sample = y_series

                        result = permutation_importance(
                            model,
                            X_sample,
                            np.asarray(y_sample).ravel(),
                            n_repeats=int(n_repeats),
                            random_state=42,
                            scoring=scoring,
                        )

                        df_perm = pd.DataFrame(
                            {
                                "Feature": feature_names,
                                "Importance": result.importances_mean,
                                "Std": result.importances_std,
                            }
                        ).sort_values("Importance", ascending=False)

                        viz = Visualizer()
                        fig, _ = viz.plot_feature_importance(
                            df_perm["Importance"].values,
                            df_perm["Feature"].values.tolist(),
                            f"{model_name} - Permutation",
                            top_n=int(top_n),
                        )
                        st.pyplot(fig, use_container_width=True)

                        csv = df_perm.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 导出替代重要性 (CSV)",
                            csv,
                            "permutation_importance.csv",
                            "text/csv",
                            key="fi_perm_csv",
                        )

                        _export_matplotlib_fig(fig, base_name="permutation_importance", key_prefix="fi_perm_fig")

                    except Exception as e:
                        st.error(f"计算失败: {e}")

def page_prediction():
    """预测应用页面（修复：预测阶段应用 imputer/scaler；支持指纹适用域）"""
    st.title("🔮 预测应用")

<<<<<<< HEAD
    # =========================
    # 📦 导入模型（无需先训练）
    # =========================
    with st.expander("📦 导入训练好的模型（.joblib）", expanded=(st.session_state.model is None)):
        uploaded_model = st.file_uploader("上传模型文件（.joblib/.pkl）", type=["joblib", "pkl"], key="model_uploader")
        if uploaded_model is not None:
            try:
                import hashlib
                data_bytes = uploaded_model.getvalue()
                file_hash = hashlib.sha256(data_bytes).hexdigest()
                # 避免 Streamlit rerun 导致重复导入（看起来像“卡死”）
                if st.session_state.get("_last_import_hash") == file_hash and st.session_state.get("model") is not None:
                    st.info("模型已加载（检测到相同文件），已跳过重复导入。")
                else:
                    with st.spinner("正在加载模型…（首次可能较慢）"):
                        from core.model_io import load_model_artifact_bytes
                        artifact = load_model_artifact_bytes(data_bytes)
        
                    # 写入 session_state（用于后续页面复用）
                    st.session_state.model_name = artifact.get("model_name") or "ImportedModel"
                    st.session_state.target_col = artifact.get("target_col") or st.session_state.get("target_col", "")
                    st.session_state.feature_cols = artifact.get("feature_cols") or st.session_state.get("feature_cols", [])
                    st.session_state.pipeline = artifact.get("pipeline", None)
                    st.session_state.model = artifact.get("model", None) or artifact.get("pipeline", None)
                    st.session_state.scaler = artifact.get("scaler", None)
                    st.session_state.imputer = artifact.get("imputer", None)
                    st.session_state.imported_model_artifact = artifact
                    st.session_state._last_import_hash = file_hash
        
                    # AutoGluon / TabPFN 等重依赖模型提示
                    if (artifact.get("model_name") or "").strip() in ["AutoGluon", "TabPFN"]:
                        st.warning("该模型属于重依赖类型（如 AutoGluon/TabPFN）。若加载耗时较长，请确认依赖已安装且版本一致。")
        
                    st.success("✅ 模型导入成功！你现在可以直接进行预测。")
            except Exception as e:
                st.error(f"❌ 模型导入失败：{e}")
                st.info("请确认文件来自本系统导出（artifact.joblib），或是可被 joblib 正常加载的 sklearn Pipeline/模型。")



=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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
<<<<<<< HEAD
                        # 对齐特征列：避免因指纹“全零列被删”导致模型所需列缺失
                        missing_cols = [c for c in feature_cols if c not in pred_df.columns]
                        if missing_cols:
                            # 指纹缺失列填 0；其它缺失列填 NaN（后续 imputer 可处理）
                            fp_missing = [c for c in missing_cols if ("maccs" in c.lower()) or ("morgan" in c.lower())]
                            other_missing = [c for c in missing_cols if c not in fp_missing]
                            for c in fp_missing:
                                pred_df[c] = 0
                            for c in other_missing:
                                pred_df[c] = np.nan
                            st.warning(f"检测到模型所需特征列缺失 {len(missing_cols)} 个，已自动补齐（指纹列填0，其它列填NaN）。")
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
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


    disable_opt = False
    with col1:
        trainer = EnhancedModelTrainer()
        available_models = trainer.get_available_models()

        # 支持优化的模型
        optimizable_models = [
            "随机森林", "XGBoost", "LightGBM", "CatBoost",
            "SVR", "Ridge回归", "Lasso回归", "ElasticNet",
            "AdaBoost", "梯度提升树",
            "TensorFlow Sequential"
        ]
        optimizable_models = [m for m in optimizable_models if m in available_models]

        # 兼容：即使当前环境缺少依赖，也显示 TFS 入口（避免“功能存在但界面不显示”）
        if "TensorFlow Sequential" not in optimizable_models:
            optimizable_models.append("TensorFlow Sequential")


        def _fmt_model_name(n: str) -> str:
            if n == "TensorFlow Sequential":
                if TENSORFLOW_AVAILABLE:
                    return "TFS (TensorFlow Sequential) ✅"
                return "TFS (TensorFlow Sequential) ⛔ 需要安装 TensorFlow"
            return n

        model_name = st.selectbox("选择模型", optimizable_models, format_func=_fmt_model_name)

        # 未安装 TensorFlow 时：仍显示入口，但禁用优化按钮并提示安装
        disable_opt = (model_name == "TensorFlow Sequential" and (not TENSORFLOW_AVAILABLE))
        if disable_opt:
            st.warning("检测到当前环境未安装 TensorFlow，TFS 模型暂不可进行 Optuna 优化。请先安装依赖：`pip install tensorflow`（或按你的硬件选择 tensorflow-cpu / tensorflow-gpu）。")
    with col2:
        n_trials = st.slider("优化轮数", 10, 200, DEFAULT_OPTUNA_TRIALS)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)

    # --- [新增] 进度条组件 ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    if st.button("🚀 开始优化", type="primary", disabled=disable_opt):
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

            # [新增] 记录到状态条
            try:
                log_fe_step(
                    operation="超参优化",
                    description=f"优化完成: {model_name}",
                    params={
                        "model": model_name,
                        "n_trials": int(n_trials),
                        "cv_folds": int(cv_folds),
                        **(best_params or {})
                    },
                    input_df=df,
                    status="success",
                    message=f"best_r2={best_score:.4f}"
                )
            except Exception:
                pass

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
            # [新增] 失败也写入状态条
            try:
                log_fe_step(
                    operation="超参优化",
                    description=f"优化失败: {model_name}",
                    params={"model": model_name, "n_trials": int(n_trials), "cv_folds": int(cv_folds)},
                    input_df=df,
                    status="error",
                    message=str(e)
                )
            except Exception:
                pass


# ============================================================
# 页面：主动学习（Active Learning）
# ============================================================
def page_active_learning():
    """主动学习页面：基于不确定性推荐下一批实验/模拟样本"""
    st.title("🧠 主动学习 (Active Learning)")

    st.markdown(
        """
主动学习适用于 **高分子/环氧树脂/复材** 这类“小样本 + 单次实验/模拟成本高”的场景。

典型闭环：
1) 用少量已标注数据训练代理模型（surrogate）
2) 在候选池中用采集函数选择“最值得做”的下一批样本（不确定性/期望提升）
3) 做实验或 MD 模拟得到真实标签
4) 回填数据，重复 1-3
        """
    )

    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    if df is None or df.empty:
        st.warning("⚠️ 当前数据为空")
        return

    # ---- 选择目标与特征 ----
    st.markdown("### 1) 选择目标变量与特征")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("❌ 当前数据没有数值列；主动学习需要数值特征 X 和数值目标 y")
        return

    # 默认目标：优先使用 session_state.target_col
    default_target = st.session_state.get('target_col')
    if default_target not in num_cols:
        default_target = num_cols[-1]
    target_col = st.selectbox("目标变量 (y)", options=num_cols, index=num_cols.index(default_target))

    # 默认特征：优先使用 session_state.feature_cols
    default_features = [c for c in (st.session_state.get('feature_cols') or []) if c in df.columns and c != target_col]
    if not default_features:
        default_features = [c for c in num_cols if c != target_col]

    feature_cols = st.multiselect(
        "特征列 (X)",
        options=[c for c in df.columns if c != target_col],
        default=default_features,
        help="建议使用‘特征选择’页面筛过的特征；也可以在这里手动指定。"
    )

    if not feature_cols:
        st.warning("⚠️ 请至少选择 1 个特征列")
        return

    # ---- 构建 labeled/pool ----
    y_all = pd.to_numeric(df[target_col], errors='coerce')
    df_labeled = df.loc[y_all.notna()].copy()

    st.markdown("### 2) 选择候选池（unlabeled pool）")

    pool_mode = st.radio(
        "候选池来源",
        [
            "使用当前数据中目标缺失的行（推荐：先导入候选配方，再逐步补实验）",
            "上传候选池文件（CSV/Excel，需包含相同特征列）",
        ],
        index=0
    )

    df_pool = None
    if pool_mode.startswith("使用当前数据"):
        df_pool = df.loc[y_all.isna()].copy()
        st.caption(f"当前候选池大小: {0 if df_pool is None else len(df_pool)}")
    else:
        up = st.file_uploader("上传候选池文件", type=["csv", "xlsx", "xls"], key="al_pool_upload")
        if up is not None:
            try:
                df_pool = load_data_file(up)
                st.success(f"✅ 已加载候选池: {df_pool.shape}")
            except Exception as e:
                st.error(f"❌ 候选池文件读取失败: {e}")
                return

    if df_pool is None or df_pool.empty:
        st.warning("⚠️ 候选池为空：请在数据中准备一些目标缺失的候选样本，或上传候选池文件")
        return

    # 检查列
    missing_in_pool = [c for c in feature_cols if c not in df_pool.columns]
    if missing_in_pool:
        st.error(f"❌ 候选池缺少特征列: {missing_in_pool}\n请确保候选池与训练数据的特征列一致。")
        return

    # ---- 选择模型与采集策略 ----
    st.markdown("### 3) 选择不确定性模型与采集策略")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_ui = st.selectbox(
            "不确定性模型",
            [
                "Gaussian Process (GPR, 小样本推荐)",
                "随机森林 (RF, 适合强非线性)",
                "Extra Trees (ETR, 更强随机性)",
            ],
            index=0
        )
    with col_b:
        acq_ui = st.selectbox(
            "采集策略",
            [
                "最大不确定性 (Uncertainty)",
                "UCB 上置信界 (Exploration+Exploitation)",
                "EI 期望提升 (Expected Improvement)",
            ],
            index=0
        )
    with col_c:
        batch_size = st.slider("推荐数量", 1, 50, 10)

    minimize = st.checkbox(
        "目标是【最小化】（例如：黏度/成本/收缩率）",
        value=False,
        help="若目标是越大越好（例如 Tg/模量/强度），请保持不勾选。"
    )

    # EI/UCB 参数
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        xi = st.number_input("EI 参数 xi", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                             disabled=("EI" not in acq_ui))
    with col_p2:
        kappa = st.number_input("UCB 参数 kappa", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
                                disabled=("UCB" not in acq_ui))

    if st.button("🚀 生成下一批实验/模拟建议", type="primary"):
        try:
            from core.active_learning import recommend_from_dataframes

            model_kind = "gpr"
            if model_ui.startswith("随机森林"):
                model_kind = "rf"
            elif model_ui.startswith("Extra Trees"):
                model_kind = "etr"

            acq_kind = "uncertainty"
            if acq_ui.startswith("UCB"):
                acq_kind = "ucb"
            elif acq_ui.startswith("EI"):
                acq_kind = "ei"

            rec_df = recommend_from_dataframes(
                df_labeled=df_labeled,
                df_pool=df_pool,
                feature_cols=feature_cols,
                target_col=target_col,
                model_kind=model_kind,
                acq_kind=acq_kind,
                batch_size=int(batch_size),
                minimize=bool(minimize),
                xi=float(xi),
                kappa=float(kappa),
                random_state=DEFAULT_RANDOM_STATE,
            )

            st.session_state.al_recommendations = rec_df
            st.success(f"✅ 已生成推荐列表（Top-{len(rec_df)}）")
        except Exception as e:
            st.error(f"❌ 主动学习计算失败: {e}")
            st.code(traceback.format_exc())

    # ---- 展示结果 ----
    if st.session_state.get('al_recommendations') is not None:
        rec_df = st.session_state.al_recommendations
        st.markdown("### 4) 推荐结果")
        st.dataframe(rec_df, use_container_width=True)

        # 导出
        csv_bytes = rec_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "⬇️ 下载推荐列表（CSV）",
            data=csv_bytes,
            file_name="active_learning_recommendations.csv",
            mime="text/csv"
        )

        st.markdown(
            """
**下一步怎么做？**
- 对表中 Top-N 候选配方进行实验合成/固化/测试（或 MD 虚拟固化 + 性能计算）。
- 把测得的目标值回填到数据表对应行（填到你选择的目标列里）。
- 重新运行本页面，即可进入下一轮主动学习。
            """
        )


# ============================================================
# 页面：训练记录（历史 Run）
# ============================================================
def page_training_records():
    st.title("📈 训练记录")
    st.caption("自动保存每次训练的指标、参数、训练曲线（loss/迭代指标或学习曲线）。")

    manager = TrainingRunManager()
    runs = manager.list_runs(limit=200)
    if not runs:
        st.info("暂无训练记录。请先在【🤖 模型训练】页面完成一次训练。")
        return

    # 选择 Run
    def _label(r):
        s = f"{r.run_id}｜{r.model_name}"
        if r.r2 is not None:
            s += f"｜R²={r.r2:.4f}"
        return s

    options = { _label(r): r.run_id for r in runs }
    sel = st.selectbox("选择一条训练记录", options=list(options.keys()), index=0)
    run_id = options[sel]

    payload = manager.load_run(run_id)
    meta = payload.get("metadata") or {}
    hist_df = payload.get("history")

    # 指标概览
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型", str(meta.get("model_name", "")) or "-")
    c2.metric("R²", f"{float(meta.get('r2', 0)):.4f}" if meta.get("r2") is not None else "-")
    c3.metric("RMSE", f"{float(meta.get('rmse', 0)):.4f}" if meta.get("rmse") is not None else "-")
    c4.metric("MAE", f"{float(meta.get('mae', 0)):.4f}" if meta.get("mae") is not None else "-")

    st.markdown("---")
    st.markdown("### 📉 训练曲线")
    if payload.get("training_curve_png"):
        st.image(payload["training_curve_png"], use_container_width=True)
    else:
        st.info("该记录未包含训练曲线图片（可能为旧版本记录）。")

    if hist_df is not None and not hist_df.empty:
        st.markdown("### 🧾 训练历史数据")
        st.dataframe(hist_df, use_container_width=True, height=260)
        st.download_button(
            "📥 下载训练历史 CSV",
            data=hist_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{run_id}_history.csv",
            mime="text/csv",
        )

    with st.expander("🔎 查看元数据（参数/切分/时间等）", expanded=False):
        st.json(meta)
        st.download_button(
            "📥 下载 metadata.json",
            data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{run_id}_metadata.json",
            mime="application/json",
        )

    extra_pngs = payload.get("extra_pngs") or {}
    if extra_pngs:
        with st.expander("🖼️ 其它图表", expanded=False):
            for fn, b in extra_pngs.items():
                st.markdown(f"**{fn}**")
                st.image(b, use_container_width=True)


# ============================================================
# 页面：状态条记录 / 操作日志
# ============================================================
def page_status_log():
    """特征工程状态条记录与数据导出"""
    st.title("📋 状态条记录")

    tracker = st.session_state.get("fe_tracker", None)
    current_df = st.session_state.get('processed_data')
    if current_df is None:
        current_df = st.session_state.get('data')

    tab1, tab2, tab3 = st.tabs(["📜 操作记录", "💾 数据导出", "ℹ️ 当前状态"])

    with tab1:
        render_status_panel(tracker)

    with tab2:
        if current_df is not None and not getattr(current_df, "empty", True):
            render_data_export_panel(current_df, tracker)
        else:
            st.info("暂无可导出的数据，请先上传/生成数据。")

    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric("当前数据", "processed_data" if st.session_state.get('processed_data') is not None else ("data" if st.session_state.get('data') is not None else "无"))
        c2.metric("特征数 (X)", len(st.session_state.get("feature_cols") or []))
        c3.metric("目标 (Y)", st.session_state.get("target_col") or "未选择")

        last = None
        if tracker is not None:
            try:
                last = tracker.get_last_step()
            except Exception:
                last = None

        st.markdown("---")
        if last:
            icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(last.get("status", "success"), "ℹ️")
            st.markdown(f"### 最近一次操作\n{icon} **{last.get('operation', '')}** - {last.get('description', '')}")
            st.caption(f"#{last.get('step_id','?')} @ {last.get('timestamp','')}")
            if last.get("message"):
                st.info(last.get("message"))
        else:
            st.info("暂无操作记录。完成一次数据处理/特征选择/训练后，这里会自动显示。")


def render_export_section():
    """渲染数据导出区域"""
    st.markdown("---")
    st.markdown("## 📥 数据导出")

    current_data = st.session_state.get('processed_data')  # 注意：改为 processed_data
    tracker = st.session_state.get('fe_tracker')

    if current_data is not None and not current_data.empty:
        render_data_export_panel(current_data, tracker)
    else:
        st.info("请先加载数据后再使用导出功能")


<<<<<<< HEAD

# ============================================================
# 页面：图像/文件转SMILES（DECIMER）
# ============================================================
def page_image_to_smiles():
    """使用 DECIMER 从结构图片/文件中识别 SMILES"""
    st.title("🖼️ 图像/文件转 SMILES")

    st.markdown(
        """
**说明**
- 本功能使用 **DECIMER**（Image Transformer）将化学结构图像识别为 SMILES。
- **首次运行会自动下载预训练权重（需要联网）**，下载位置由 DECIMER/pystow 管理。
- 支持：png/jpg/jpeg/bmp/tif/tiff/webp/heif/heic，PDF（若安装 PyMuPDF 或 pdf2image）。
- ⚠️ **建议：一张图只放一个分子结构**（多分子拼图/带大量文字标注会显著降低识别准确率，请先裁剪后再上传）。
        """
    )

    from core.image_smiles_extractor import decimer_is_available, smiles_from_bytes

    ok, msg = decimer_is_available()
    if not ok:
        st.error("DECIMER 依赖未就绪，当前无法识别。")
        st.caption(msg)
        st.markdown("### ✅ 安装依赖（建议在已激活的环境中执行）")
        st.code(
            "\n".join(
                [
                    "pip install tensorflow>=2.12.0,<=2.20.0",
                    "pip install opencv-python pystow pillow-heif efficientnet selfies pyyaml",
                    "# 若需要 PDF 支持（二选一）：",
                    "pip install pymupdf",
                    "# 或：pip install pdf2image  （系统需额外安装 poppler）",
                ]
            ),
            language="bash",
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        hand_drawn = st.checkbox("手绘结构模式（Hand-Drawn）", value=False)
    with col2:
        with_conf = st.checkbox("返回置信度（Top-1）", value=False)

    uploaded_files = st.file_uploader(
        "上传结构图片或 PDF（可多选）",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "heif", "heic", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("请上传图片或 PDF 后开始识别。")
        return

    results = []
    for uf in uploaded_files:
        st.markdown("---")
        st.subheader(f"📄 {uf.name}")

        data = uf.getvalue()

        # Preview image (skip pdf)
        if uf.type and uf.type.startswith("image/"):
            try:
                st.image(data, caption=uf.name, use_container_width=True)
            except Exception:
                pass

        try:
            preds = smiles_from_bytes(
                data, uf.name, confidence=with_conf, hand_drawn=hand_drawn
            )
            for p in preds:
                page_tag = ""
                if p.page_index is not None:
                    page_tag = f" (Page {p.page_index + 1})"

                st.success(f"SMILES{page_tag}: {p.smiles}")
                if p.confidence is not None:
                    st.caption(f"Confidence (Top-1): {p.confidence:.4f}")

                results.append(
                    {
                        "filename": p.filename,
                        "page": None if p.page_index is None else int(p.page_index + 1),
                        "smiles": p.smiles,
                        "confidence": p.confidence,
                        "engine": p.engine,
                    }
                )
        except Exception as e:
            st.error(f"识别失败：{e}")
            results.append(
                {
                    "filename": uf.name,
                    "page": None,
                    "smiles": "",
                    "confidence": None,
                    "engine": "DECIMER",
                    "error": str(e),
                }
            )

    if results:
        st.markdown("---")
        st.markdown("### 📌 识别结果汇总")
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        csv_bytes = df_res.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 下载识别结果 CSV",
            data=csv_bytes,
            file_name="smiles_results.csv",
            mime="text/csv",
        )

        # Save for later pages
        st.session_state["smiles_results"] = df_res
        st.info("结果已保存到会话变量：st.session_state['smiles_results']。")

=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
# ============================================================
# 主程序入口（保持原有结构）
# ============================================================
page = render_sidebar()

# [增强] 顶部轻量状态条：避免侧边栏折叠后找不到 TFS 入口/操作记录
render_top_status_bar()

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
<<<<<<< HEAD

elif page == "🖼️ 图像转SMILES":
    page_image_to_smiles()
=======
>>>>>>> f168256419b9b557a70253c84666a6aee162abf4
elif page == "🎯 特征选择":
    page_feature_selection()
elif page == "🤖 模型训练":
    page_model_training()
elif page == "📈 训练记录":
    page_training_records()
elif page == "📊 模型解释":
    page_model_interpretation()
elif page == "🔮 预测应用":
    page_prediction()
elif page == "⚙️ 超参优化":
    page_hyperparameter_optimization()
elif page == "🧠 主动学习":
    page_active_learning()
elif page == "📋 状态条记录":
    page_status_log()