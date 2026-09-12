# -*- coding: utf-8 -*-
"""
材料机器学习平台 —— 应用入口（由原 28000 行 app.py 拆分自动生成）。

架构：
  app_lib.py    共享库：全部定义与幂等环境设置（每进程一次）
  app_pages/    19 个 st.Page 页面封装
  本文件        每 rerun 执行：UI 初始化 + 分组导航 + 任务锁 + 自动保存
"""

from app_lib import *

st.set_page_config(
    page_title="材料机器学习平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'render_optimization' not in st.session_state:
    st.session_state.render_optimization = {
        'cache_enabled': True,
        'lazy_load': True,
        'max_preview_rows': 50  # 限制预览行数
    }

try:
    from core.theme import inject_theme
    inject_theme()
except Exception:
    pass

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

init_session_state()

_get_or_create_session_id()

_maybe_auto_restore()

if 'fe_tracker' not in st.session_state:
    st.session_state.fe_tracker = FeatureEngineeringTracker()

tracker = st.session_state.fe_tracker

if 'task_manager_initialized' not in st.session_state:
    clear_cancel()  # 清除之前的取消标志
    st.session_state.task_manager_initialized = True

# ============================================================
# 页面导航（st.navigation 分组侧边栏）
# ============================================================

# ---------- 侧边栏（导航菜单固定在顶部；平台名品牌位移至底部） ----------
with st.sidebar:

        # 旧版四组 radio 会同时保存四个选中值，并在同一次 rerun 中互相覆盖。
        # 统一为一个页面选择源，避免侧边栏出现多个互相冲突的导航状态。
        for legacy_key in (
            "nav_data",
            "nav_feature",
            "nav_model",
            "nav_app",
            "nav_status_toggle",
            "_last_data_page",
            "_last_feature_page",
            "_last_model_page",
            "_last_app_page",
            "nav_page",
            "_last_active_page",
            "_nav_to",
        ):
            st.session_state.pop(legacy_key, None)


_NAV_GROUPS = {
    "数据准备": [
        st.Page("app_pages/home.py", title="🏠 首页", url_path="home"),
        st.Page("app_pages/data_upload.py", title="📤 数据上传", url_path="data_upload"),
        st.Page("app_pages/data_explore.py", title="🔍 数据探索", url_path="data_explore"),
        st.Page("app_pages/data_cleaning.py", title="🧹 数据清洗", url_path="data_cleaning"),
        st.Page("app_pages/data_enhancement.py", title="✨ 数据增强", url_path="data_enhancement"),
    ],
    "特征工程": [
        st.Page("app_pages/molecular_features.py", title="🧬 分子特征", url_path="molecular_features"),
        st.Page("app_pages/molecular_feature_reproduction.py", title="🧬 分子特征复现", url_path="molecular_feature_reproduction"),
        st.Page("app_pages/feature_registry.py", title="🧩 特征管理", url_path="feature_registry"),
        st.Page("app_pages/smiles_structure_tools.py", title="🧪 SMILES / BigSMILES 结构图像工具", url_path="smiles_structure_tools"),
        st.Page("app_pages/feature_selection.py", title="🎯 特征选择", url_path="feature_selection"),
    ],
    "建模分析": [
        st.Page("app_pages/model_training.py", title="🤖 模型训练", url_path="model_training"),
        st.Page("app_pages/training_records.py", title="📈 训练记录", url_path="training_records"),
        st.Page("app_pages/model_interpretation.py", title="📊 模型解释", url_path="model_interpretation"),
        st.Page("app_pages/hyperparameter_optimization.py", title="⚙️ 超参优化", url_path="hyperparameter_optimization"),
        st.Page("app_pages/active_learning.py", title="🧠 主动学习", url_path="active_learning"),
    ],
    "应用预测": [
        st.Page("app_pages/prediction.py", title="🔮 预测应用", url_path="prediction"),
        st.Page("app_pages/model_imputation.py", title="🔧 模型补齐数据", url_path="model_imputation"),
        st.Page("app_pages/virtual_screening.py", title="🧪 虚拟分子筛选", url_path="virtual_screening"),
    ],
    "系统记录": [
        st.Page("app_pages/status_log.py", title="📋 状态条记录", url_path="status_log"),
    ],
}
pg = st.navigation(_NAV_GROUPS)

with st.sidebar:
        active_task_lock = bool(get_task_manager().get_active_tasks())

        st.markdown("---")
        _render_sidebar_portal_panel()
        _render_network_proxy_panel(active_task_lock)
        _render_portal_ai_service_panel(active_task_lock)

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

        # 清除缓存按钮
        st.markdown("---")
        st.markdown("### 🔧 系统工具")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔄 刷新页面",
                help="任务运行期间已禁用，避免中断或重复提交",
                width="stretch",
                disabled=active_task_lock,
            ):
                st.rerun()

        with col2:
            if st.button(
                "🗑️ 清除缓存",
                help="任务运行期间已禁用，避免清掉任务依赖的模型/特征缓存",
                width="stretch",
                disabled=active_task_lock,
            ):
                try:
                    # 清除XGBoost缓存
                    xgb_cache = get_xgboost_cache()
                    xgb_cache.clear()

                    # 清除SHAP缓存
                    shap_cache = get_shap_cache()
                    shap_cache.clear()

                    # 清除Streamlit缓存
                    st.cache_data.clear()
                    st.cache_resource.clear()

                    st.success("✅ 缓存已清除！")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除缓存失败: {e}")

        # [关键修复] 使用get_current_model()检查模型
        if get_current_model() is not None:
            st.success(f"🤖 已训练: {st.session_state.model_name}")
            # 如果有训练结果，也可以显示R2
            if st.session_state.get('train_result'):
                r2 = st.session_state.train_result.get('r2', 0)
                st.caption(f"当前 R²: {r2:.4f}")

        _render_sidebar_session_panel(active_task_lock)

        _render_sidebar_compute_panel(active_task_lock)

        # [新增] 后台任务管理器 UI
        render_task_manager_ui()

        # [增强] 在侧边栏始终显示状态条入口（即使暂无记录，也避免"功能存在但界面不显示"）
        render_status_sidebar(st.session_state.get('fe_tracker', None))
        if active_task_lock:
            st.caption("🔒 主页面操作已锁定；请使用上方后台任务控制停止或等待完成。")

        # 底部品牌位（导航菜单固定在侧边栏顶部，平台名移到底部展示）
        st.markdown("---")
        st.caption(f"🔬 **{APP_NAME}**")
        st.caption(f"📌 v{VERSION}")


# ============================================================
# 主流程
# ============================================================
page = pg.title  # 与原 task-lock / 置顶逻辑兼容（title 含 emoji）
_prev_page = st.session_state.get("_prev_page", None)
if _prev_page is not None and _prev_page != page:
    st.html("""
    <style>
        section.main .block-container { animation: fadeIn 0.15s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
    <script>
        const main = window.parent.document.querySelector('section.main');
        if (main) main.scrollTo({top: 0, behavior: 'instant'});
    </script>
    """)
st.session_state["_prev_page"] = page
render_top_status_bar()
if _render_global_task_lock(page):
    _maybe_autosave_session()
    st.stop()
pg.run()

# 自动保存快照（断连保护）
_maybe_autosave_session()
