# -*- coding: utf-8 -*-
"""特征选择模块 - 完整修复版 (含 SmartFeatureSelector 类及回调修复)"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 回调函数定义区 (必须在组件渲染前定义)
# ==========================================

def _update_selection_state(new_selection):
    """通用回调：更新特征选择状态"""
    st.session_state.feature_cols = new_selection
    st.session_state.multiselect_features = new_selection


def _apply_variance_filter_callback(df, candidates, threshold_key):
    """方差筛选回调"""
    try:
        threshold = st.session_state[threshold_key]
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df.fillna(0))
        selected = [candidates[i] for i in selector.get_support(indices=True)]
        _update_selection_state(selected)
        st.session_state['feature_selector_msg'] = f"✅ 方差筛选完成：剩余 {len(selected)} 个特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_correlation_filter_callback(df, target_series, k_key):
    """相关性筛选回调"""
    try:
        k = st.session_state[k_key]
        corrs = df.corrwith(target_series).abs().sort_values(ascending=False)
        selected = corrs.head(int(k)).index.tolist()
        _update_selection_state(selected)
        st.session_state['feature_selector_msg'] = f"✅ 相关性筛选完成：已选 Top-{k} 特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _build_rfe_estimator(estimator_name: str, random_state: int = 42):
    """根据名称构建RFE/RFECV可用的估计器，并返回(estimator, needs_scaling)。"""
    name = (estimator_name or "").strip()
    # 说明：RFE/RFECV 需要 estimator 提供 coef_ 或 feature_importances_
    if name in ["Ridge线性回归", "Ridge"]:
        return Ridge(alpha=1.0), True
    if name in ["Lasso稀疏回归", "Lasso"]:
        return Lasso(alpha=0.001, max_iter=5000, random_state=random_state), True
    if name in ["线性SVR", "SVR(linear)"]:
        return SVR(kernel="linear", C=1.0), True

    if name in ["ExtraTrees", "极端随机森林"]:
        return ExtraTreesRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1
        ), False
    if name in ["梯度提升", "GBDT", "GradientBoosting"]:
        return GradientBoostingRegressor(random_state=random_state), False

    # 默认：随机森林
    return RandomForestRegressor(
        n_estimators=400, random_state=random_state, n_jobs=-1
    ), False


def _apply_rfe_filter_callback(df, target_series, candidates,
                              mode_key, est_key, k_key, step_key,
                              cv_key, scoring_key, min_k_key):
    """RFE / RFECV 递归特征消除回调"""
    try:
        mode = st.session_state.get(mode_key, "RFECV自动（推荐）")
        est_name = st.session_state.get(est_key, "随机森林")
        estimator, needs_scaling = _build_rfe_estimator(est_name, random_state=42)

        # --- 准备数据 ---
        X = df[candidates].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(numeric_only=True))
        # 只保留数值列（避免混入类别特征导致报错）
        X = X.select_dtypes(include=np.number)
        used_candidates = X.columns.tolist()

        if len(used_candidates) < 2:
            st.session_state['feature_selector_error'] = "可用于RFE的数值特征少于2个。"
            return

        y = pd.Series(target_series).copy()
        y = y.replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            y = y.fillna(method="ffill").fillna(method="bfill")

        # 可选标准化（对线性模型/线性SVR更稳健）
        if needs_scaling:
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
        else:
            X_values = X.values

        step = st.session_state.get(step_key, 1)
        try:
            step = int(step)
        except Exception:
            step = 1
        step = max(1, step)

        # --- 选择器 ---
        if str(mode).startswith("RFECV"):
            cv = int(st.session_state.get(cv_key, 5))
            cv = max(2, min(cv, 10))
            min_k = int(st.session_state.get(min_k_key, max(2, min(10, len(used_candidates)))))
            min_k = max(1, min(min_k, len(used_candidates) - 1))
            scoring = st.session_state.get(scoring_key, "r2")
            selector = RFECV(
                estimator=estimator,
                step=step,
                cv=KFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=scoring,
                min_features_to_select=min_k,
                n_jobs=-1
            )
        else:
            k = int(st.session_state.get(k_key, min(20, len(used_candidates))))
            k = max(1, min(k, len(used_candidates)))
            selector = RFE(estimator=estimator, n_features_to_select=k, step=step)

        selector.fit(X_values, y.values.ravel())

        selected = [used_candidates[i] for i, flag in enumerate(selector.support_) if flag]
        _update_selection_state(selected)

        # 保存排名表，便于UI展示/导出
        ranking_df = pd.DataFrame({
            "特征": used_candidates,
            "RFE排名": selector.ranking_
        }).sort_values("RFE排名", ascending=True)
        st.session_state["rfe_ranking_df"] = ranking_df

        if hasattr(selector, "n_features_"):
            n_selected = int(selector.n_features_)
        else:
            n_selected = len(selected)

        if str(mode).startswith("RFECV"):
            st.session_state['feature_selector_msg'] = f"✅ RFECV完成：CV最优特征数 = {n_selected}"
        else:
            st.session_state['feature_selector_msg'] = f"✅ RFE完成：已选 {n_selected} 个特征"

    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_smart_rec_callback(feature_analysis):
    """智能推荐回调"""
    try:
        recommended = feature_analysis[feature_analysis['推荐'] == '✓']['特征'].tolist()
        _update_selection_state(recommended)
        st.session_state['feature_selector_msg'] = f"✅ 智能推荐完成：已选 {len(recommended)} 个特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_importance_filter_callback(top_features, candidates):
    """模型重要性筛选回调 (底层逻辑)"""
    try:
        valid_selected = [f for f in top_features if f in candidates]
        ignored = len(top_features) - len(valid_selected)
        _update_selection_state(valid_selected)

        msg = f"✅ 模型筛选完成：已选 {len(valid_selected)} 个特征"
        if ignored > 0:
            msg += f" (忽略了 {ignored} 个缺失特征)"
        st.session_state['feature_selector_msg'] = msg
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_importance_filter_callback_v2(sorted_features, candidates, k_key):
    """模型重要性筛选回调 (全局版，读取 Session State)"""
    try:
        # 动态从 session_state 获取当前的 Top-K 值
        if k_key in st.session_state:
            k = st.session_state[k_key]
        else:
            k = 20  # 默认兜底

        # 截取前 K 个特征
        top_f = sorted_features[:int(k)]

        # 复用已有的筛选逻辑
        _apply_importance_filter_callback(top_f, candidates)
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_pca_callback(pca, scaler, numeric_df, current_df, feature_candidates):
    """PCA应用回调"""
    try:
        # 执行转换
        X = numeric_df.copy().fillna(numeric_df.mean())
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # 构建新 DataFrame
        pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
        df_pca = pd.DataFrame(X_pca, columns=pc_names, index=current_df.index)

        # 合并
        df_rest = current_df.drop(columns=feature_candidates)
        df_new = pd.concat([df_rest, df_pca], axis=1)

        # 更新全局数据状态
        st.session_state.processed_data = df_new
        # 更新特征选择状态
        _update_selection_state(pc_names)

        # 清理临时状态
        st.session_state.pop('_pca_model', None)
        st.session_state.pop('_pca_scaler', None)
        st.session_state.pop('_pca_ready', None)

        st.session_state['feature_selector_msg'] = "✅ PCA 转换已应用！数据集已更新。"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


# ==========================================
# 2. 类定义区 (SmartFeatureSelector & SmartSparseDataSelector)
# ==========================================

class SmartFeatureSelector:
    """智能特征选择器 (补回缺失的类)"""

    MISSING_VALUE_TOLERANT_MODELS = {'XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'ExtraTrees'}

    def __init__(self, data, feature_cols, target_col, model_name=None):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_name = model_name
        self.missing_info = {}

    def analyze_missing_values(self):
        selected = self.data[self.feature_cols + [self.target_col]]
        total = selected.size
        missing = selected.isnull().sum().sum()
        col_missing = selected.isnull().sum()

        self.missing_info = {
            'total_cells': total,
            'missing_cells': missing,
            'missing_rate': missing / total if total > 0 else 0,
            'column_missing': col_missing,
            'column_missing_rate': col_missing / len(selected),
            'rows_with_missing': (selected.isnull().sum(axis=1) > 0).sum(),
            'columns_with_high_missing': col_missing[col_missing / len(selected) > 0.3].index.tolist()
        }
        return self.missing_info

    def recommend_strategy(self):
        self.analyze_missing_values()
        recommendations = []

        if self.model_name in self.MISSING_VALUE_TOLERANT_MODELS:
            recommendations.append({
                'strategy': 'model_native',
                'priority': 1,
                'reason': f'{self.model_name}原生支持缺失值'
            })

        recommendations.append({
            'strategy': 'median',
            'priority': 2,
            'reason': '中位数填充，对异常值稳健'
        })

        return recommendations


class SmartSparseDataSelector:
    """稀疏数据选择器"""

    def __init__(self, data):
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.sparsity_info = self._analyze()

    def _analyze(self):
        return {col: {
            'non_null_count': self.data[col].notna().sum(),
            'non_null_ratio': self.data[col].notna().mean(),
            'null_count': self.data[col].isna().sum()
        } for col in self.numeric_cols}

    def get_target_analysis(self):
        analysis = []
        for col, info in self.sparsity_info.items():
            try:
                non_null_ratio = float(info['non_null_ratio'])
                non_null_count = int(info['non_null_count'])
                null_count = int(info['null_count'])
            except TypeError:
                non_null_ratio = float(np.mean(info['non_null_ratio']))
                non_null_count = int(np.mean(info['non_null_count']))
                null_count = int(np.mean(info['null_count']))

            analysis.append({
                '变量名': col,
                '有效样本数': non_null_count,
                '有效率': f"{non_null_ratio * 100:.1f}%",
                '缺失数': null_count
            })
        return pd.DataFrame(analysis).sort_values('有效样本数', ascending=False)

    def get_valid_samples_for_target(self, target_col):
        return self.data[self.data[target_col].notna()].copy()

    def analyze_features_for_target(self, target_col, min_valid_ratio=0.5):
        valid_data = self.get_valid_samples_for_target(target_col)
        n = len(valid_data)
        analysis = []
        for col in self.numeric_cols:
            if col == target_col:
                continue
            valid_count = valid_data[col].notna().sum()
            analysis.append({
                '特征': col,
                '有效数': valid_count,
                '有效率': f"{valid_count / n * 100:.1f}%" if n > 0 else "0%",
                '推荐': '✓' if valid_count / n >= min_valid_ratio else '✗'
            })
        return pd.DataFrame(analysis).sort_values('有效数', ascending=False)


# ==========================================
# 3. 界面渲染函数 (show_robust_feature_selection)
# ==========================================

def show_robust_feature_selection():
    """完整的特征选择界面（含 PCA 降维及模型重要性反馈，已修复状态同步问题）"""
    st.markdown("### 🛠️ 特征选择与数据集构建")

    # 显示回调消息（如果有）
    if 'feature_selector_msg' in st.session_state:
        st.success(st.session_state.pop('feature_selector_msg'))
    if 'feature_selector_error' in st.session_state:
        st.error(st.session_state.pop('feature_selector_error'))

    # 数据源回退机制
    if st.session_state.get('processed_data') is not None:
        current_df = st.session_state.processed_data
    elif st.session_state.get('data') is not None:
        current_df = st.session_state.data
        st.info("ℹ️ 使用原始数据")
    else:
        st.warning("⚠️ 请先上传数据")
        return

    # 重复列名检查
    if current_df.columns.duplicated().any():
        current_df = current_df.loc[:, ~current_df.columns.duplicated()]
        st.session_state.processed_data = current_df
    all_columns = current_df.columns.tolist()

    # 目标变量选择
    col1, col2 = st.columns([1, 2])
    with col1:
        default_idx = 0
        if st.session_state.get('target_col') and st.session_state.target_col in all_columns:
            default_idx = all_columns.index(st.session_state.target_col)

        target_col = st.selectbox("🎯 选择目标变量 (Y)", options=all_columns, index=default_idx)
        st.session_state.target_col = target_col

    # 特征完整性检查
    numeric_df = current_df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    feature_candidates = numeric_df.columns.tolist()

    # 自动清洗 session_state 中的 feature_cols
    if 'feature_cols' in st.session_state and st.session_state.feature_cols:
        valid_features = [f for f in st.session_state.feature_cols if f in feature_candidates]
        if len(valid_features) != len(st.session_state.feature_cols):
            st.session_state.feature_cols = valid_features

    with col2:
        st.metric("可用数值特征", f"{len(feature_candidates)} 个")

    # 检查无穷大值
    if len(feature_candidates) > 0:
        check_inf = np.isinf(numeric_df).sum().sum()
        if check_inf > 0:
            st.error(f"⚠️ 检测到 {check_inf} 个无穷大数值(Inf)")
            if st.button("🧹 一键修复异常值", type="primary"):
                clean_df = current_df.copy()
                cols = numeric_df.columns
                clean_df[cols] = clean_df[cols].replace([np.inf, -np.inf], np.nan)
                clean_df[cols] = clean_df[cols].fillna(clean_df[cols].mean())
                st.session_state.data = clean_df
                st.session_state.processed_data = clean_df
                st.success("✅ 修复完成！")
                st.rerun()
            return

    st.markdown("---")

    # 统一使用 Tabs 布局
    tabs = st.tabs(["👆 手动选择", "📉 方差筛选", "🔗 相关性筛选", "🌀 RFE递归消除", "🧩 PCA降维", "🤖 智能推荐", "⭐ 模型重要性"])

    # --- Tab 1: 手动选择 ---
    with tabs[0]:
        st.markdown("#### 手动选择特征")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            # 使用 on_click 回调
            st.button("全选", key="btn_all_features",
                      on_click=_update_selection_state, args=(feature_candidates,))
        with col_m2:
            # 使用 on_click 回调
            st.button("清空", key="btn_clear_features",
                      on_click=_update_selection_state, args=([],))

        # 多选框：依赖 session_state 自动同步
        # ✅ 修复：不要同时使用 default= 与 st.session_state["multiselect_features"]
        # 否则会触发 Streamlit 警告：
        # "...created with a default value but also had its value set via the Session State API."
        if 'multiselect_features' not in st.session_state:
            st.session_state['multiselect_features'] = st.session_state.get('feature_cols', [])

        # 清理无效特征（防止数据列变化导致旧状态残留）
        st.session_state['multiselect_features'] = [
            f for f in (st.session_state.get('multiselect_features') or [])
            if f in feature_candidates
        ]

        selected_features = st.multiselect(
            "选择特征",
            options=feature_candidates,
            key="multiselect_features"
        )
        st.session_state.feature_cols = selected_features

    # --- Tab 2: 方差筛选 ---
    with tabs[1]:
        st.markdown("#### 方差阈值筛选")
        st.caption("移除变化很小（包含信息量少）的特征。")
        st.slider("方差阈值", 0.0, 1.0, 0.0, 0.01, key="var_threshold")

        # 使用回调
        st.button("应用方差筛选", key="btn_var_filter",
                  on_click=_apply_variance_filter_callback,
                  args=(numeric_df, feature_candidates, "var_threshold"))

    # --- Tab 3: 相关性筛选 ---
    with tabs[2]:
        st.markdown("#### 相关性筛选")
        st.caption("保留与目标变量相关性最高的 Top-K 特征。")
        st.number_input("保留相关性最高的K个", 1, len(feature_candidates), min(20, len(feature_candidates)),
                        key="corr_k")

        # 使用回调
        st.button("应用相关性筛选", key="btn_corr_filter",
                  on_click=_apply_correlation_filter_callback,
                  args=(numeric_df, current_df[target_col], "corr_k"))

    # --- Tab 4: RFE / RFECV ---
    with tabs[3]:
        st.markdown("#### 🌀 递归特征消除 (RFE / RFECV)")
        st.caption("训练一个可解释的重要性模型，迭代移除最不重要的特征；RFECV 可通过交叉验证自动确定最优特征数。")

        if len(feature_candidates) < 2:
            st.warning("⚠️ 可用数值特征少于2个，无法进行RFE/RFECV。")
        else:
            col_r1, col_r2 = st.columns([1, 1])
            with col_r1:
                st.radio("模式", ["RFECV自动（推荐）", "RFE固定数量"], horizontal=True, key="rfe_mode")
            with col_r2:
                st.selectbox(
                    "基模型（需支持 coef_ 或 feature_importances_）",
                    ["随机森林", "ExtraTrees", "梯度提升", "Ridge线性回归", "Lasso稀疏回归", "线性SVR"],
                    index=0,
                    key="rfe_estimator"
                )

            col_r3, col_r4, col_r5 = st.columns(3)
            with col_r3:
                st.number_input("step（每轮剔除特征数）", 1, max(1, len(feature_candidates) // 2), 1, key="rfe_step")
            with col_r4:
                st.number_input("RFE: 选择特征数K", 1, len(feature_candidates),
                                min(20, len(feature_candidates)), key="rfe_k")
            with col_r5:
                st.number_input("RFECV: 最少保留特征数", 1, max(1, len(feature_candidates) - 1),
                                max(2, min(10, len(feature_candidates))), key="rfe_min_k")

            col_r6, col_r7 = st.columns(2)
            with col_r6:
                st.selectbox(
                    "RFECV评分指标",
                    ["r2", "neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error"],
                    index=0,
                    key="rfe_scoring"
                )
            with col_r7:
                st.number_input("RFECV折数CV", 2, 10, 5, key="rfe_cv")

            st.button(
                "运行 RFE / RFECV",
                type="primary",
                key="btn_rfe_filter",
                on_click=_apply_rfe_filter_callback,
                args=(numeric_df, current_df[target_col], feature_candidates,
                      "rfe_mode", "rfe_estimator", "rfe_k", "rfe_step",
                      "rfe_cv", "rfe_scoring", "rfe_min_k")
            )

            if "rfe_ranking_df" in st.session_state:
                st.markdown("##### RFE排名（1 表示越重要）")
                st.dataframe(st.session_state["rfe_ranking_df"], use_container_width=True)

    # --- Tab 5: PCA 降维 ---

    with tabs[4]:
        st.markdown("#### 🧩 主成分分析 (PCA) 降维")

        if len(feature_candidates) < 2:
            st.warning("⚠️ 可用数值特征少于2个，无法进行PCA分析。")
        else:
            col_pca1, col_pca2 = st.columns([1, 1])
            with col_pca1:
                pca_method = st.radio("降维目标", ["按保留方差比", "按指定维度"], horizontal=True, key="pca_method")
            with col_pca2:
                if pca_method == "按保留方差比":
                    var_thresh = st.slider("目标解释方差 (Variance Ratio)", 0.5, 0.999, 0.95, 0.01, key="pca_var")
                    pca_args = {'n_components': var_thresh}
                else:
                    max_comp = len(feature_candidates)
                    n_comp = st.slider("目标维度 (N Components)", 1, max_comp, min(5, max_comp), key="pca_n")
                    pca_args = {'n_components': n_comp}

            # 预览按钮 (不修改状态，可以用常规 button)
            if st.button("📊 预览 PCA 分析结果", key="btn_pca_preview"):
                try:
                    X = numeric_df.copy().fillna(numeric_df.mean())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    pca = PCA(**pca_args)
                    pca.fit(X_scaled)

                    n_pc = pca.n_components_
                    explained = pca.explained_variance_ratio_
                    cum_explained = np.cumsum(explained)

                    st.success(f"✅ 计算完成：生成了 {n_pc} 个主成分，累计解释方差 {cum_explained[-1]:.4f}")

                    st.markdown("##### 解释方差分布 (Scree Plot)")
                    chart_df = pd.DataFrame({
                        "Component": [f"PC{i + 1}" for i in range(n_pc)],
                        "Individual Variance": explained,
                        "Cumulative Variance": cum_explained
                    })
                    st.line_chart(chart_df.set_index("Component")[["Individual Variance", "Cumulative Variance"]])

                    st.session_state['_pca_model'] = pca
                    st.session_state['_pca_scaler'] = scaler
                    st.session_state['_pca_ready'] = True

                except Exception as e:
                    st.error(f"PCA 分析出错: {e}")

            # 应用按钮 (修改 DataFrame 和 Selectbox，必须用回调)
            if st.session_state.get('_pca_ready', False):
                st.markdown("---")
                st.warning("⚠️ **确认操作**：点击下方按钮将创建新的数据集。所有原始数值特征将被 PC1, PC2... 替换。")

                st.button("🚀 应用 PCA 转换并更新数据集", type="primary", key="btn_pca_apply",
                          on_click=_apply_pca_callback,
                          args=(st.session_state['_pca_model'],
                                st.session_state['_pca_scaler'],
                                numeric_df, current_df, feature_candidates))

    # --- Tab 5: 智能推荐 ---
    with tabs[5]:
        sparse_selector = SmartSparseDataSelector(current_df)

        st.markdown("#### 目标变量有效性分析")
        target_analysis = sparse_selector.get_target_analysis()
        st.dataframe(target_analysis.head(10), use_container_width=True)

        feature_analysis = sparse_selector.analyze_features_for_target(target_col)

        st.dataframe(feature_analysis, use_container_width=True)

        # 使用回调
        st.button("🎯 智能推荐特征", key="btn_smart_rec",
                  on_click=_apply_smart_rec_callback, args=(feature_analysis,))

    # --- Tab 6: 模型重要性 (新增) ---
    with tabs[6]:
        st.markdown("#### ⭐ 基于已训练模型的重要性筛选")
        st.caption("利用上一轮【模型训练】得到的特征重要性（或系数）来反向优化特征集。这对于剔除噪声特征非常有效。")

        model = st.session_state.get('model')
        if model is None:
            st.info("⚠️ 暂无模型记录。请先前往【🤖 模型训练】页面训练一个模型（如随机森林、XGBoost），然后再返回此处。")
        else:
            trained_model_name = st.session_state.get('model_name', 'Unknown')
            st.success(f"当前参考模型: **{trained_model_name}**")

            # 1. 尝试获取特征名
            feature_names = None
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            elif 'feature_cols' in st.session_state and len(st.session_state.feature_cols) > 0:
                feature_names = np.array(st.session_state.feature_cols)

            # 2. 尝试获取重要性
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)

            # 3. 展示与操作
            if importances is not None and feature_names is not None:
                if len(importances) != len(feature_names):
                    st.warning(
                        f"⚠️ 特征名与重要性维度不匹配 ({len(feature_names)} vs {len(importances)})，无法精确对应。")
                else:
                    imp_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

                    max_imp = imp_df['Importance'].max()
                    if max_imp > 0:
                        imp_df['Importance'] = imp_df['Importance'] / max_imp

                    col_imp1, col_imp2 = st.columns([2, 1])

                    with col_imp1:
                        st.markdown("##### 特征重要性排序 (Top 20)")
                        st.bar_chart(imp_df.set_index('Feature').head(20))

                    with col_imp2:
                        st.markdown("##### ✂️ 筛选设置")
                        max_k = len(feature_names)
                        # key="imp_top_k" 供回调读取
                        st.number_input("保留 Top-K", 1, max_k, min(20, max_k), key="imp_top_k")

                        sorted_features = imp_df['Feature'].tolist()

                        # 使用全局回调函数 _apply_importance_filter_callback_v2
                        st.button(
                            "✅ 应用筛选",
                            type="primary",
                            key="btn_imp_apply",
                            on_click=_apply_importance_filter_callback_v2,
                            args=(sorted_features, feature_candidates, "imp_top_k")
                        )
            else:
                st.warning("❌ 当前模型不支持直接提取特征重要性（或者未记录特征名），请尝试使用 SHAP 分析或手动筛选。")

    # 显示已选特征摘要
    if st.session_state.get('feature_cols'):
        st.markdown("---")
        st.markdown(f"### ✅ 已选择 {len(st.session_state.feature_cols)} 个特征")

        cols = st.columns(4)
        for i, feat in enumerate(st.session_state.feature_cols[:20]):
            with cols[i % 4]:
                st.markdown(
                    f"<span style='background:#E0E7FF;padding:4px 8px;border-radius:12px;font-size:0.85rem;'>{feat}</span>",
                    unsafe_allow_html=True)

        if len(st.session_state.feature_cols) > 20:
            st.caption(f"... 等共 {len(st.session_state.feature_cols)} 个特征")

        # 数据预览
        st.markdown("#### 📋 数据预览")
        available_preview = [c for c in st.session_state.feature_cols if c in current_df.columns]
        if available_preview:
            preview_cols = available_preview[:5] + [
                target_col] if target_col in current_df.columns else available_preview[:5]
            st.dataframe(current_df[preview_cols].head(), use_container_width=True)