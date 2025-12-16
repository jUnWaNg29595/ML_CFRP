# -*- coding: utf-8 -*-
"""特征选择模块 - 完整版 (含PCA降维优化)"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class SmartFeatureSelector:
    """智能特征选择器"""

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


def show_robust_feature_selection():
    """完整的特征选择界面（含 PCA 降维）"""
    st.markdown("### 🛠️ 特征选择与数据集构建")

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

    # 统一使用 Tabs 布局，无论特征多少都提供所有工具
    # 这样用户在少量特征时也能使用 PCA
    tabs = st.tabs(["👆 手动选择", "📉 方差筛选", "🔗 相关性筛选", "🧩 PCA降维", "🤖 智能推荐"])

    # --- Tab 1: 手动选择 ---
    with tabs[0]:
        st.markdown("#### 手动选择特征")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("全选", key="btn_all_features"):
                st.session_state.feature_cols = feature_candidates
        with col_m2:
            if st.button("清空", key="btn_clear_features"):
                st.session_state.feature_cols = []

        selected_features = st.multiselect(
            "选择特征",
            options=feature_candidates,
            default=st.session_state.get('feature_cols', []),
            key="multiselect_features"
        )
        st.session_state.feature_cols = selected_features

    # --- Tab 2: 方差筛选 ---
    with tabs[1]:
        st.markdown("#### 方差阈值筛选")
        st.caption("移除变化很小（包含信息量少）的特征。")
        threshold = st.slider("方差阈值", 0.0, 1.0, 0.0, 0.01, key="var_threshold")
        if st.button("应用方差筛选", key="btn_var_filter"):
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(numeric_df.fillna(0))
            selected = [feature_candidates[i] for i in selector.get_support(indices=True)]
            st.session_state.feature_cols = selected
            st.success(f"✅ 筛选后剩余 {len(selected)} 个特征")

    # --- Tab 3: 相关性筛选 ---
    with tabs[2]:
        st.markdown("#### 相关性筛选")
        st.caption("保留与目标变量相关性最高的 Top-K 特征。")
        k = st.number_input("保留相关性最高的K个", 1, len(feature_candidates), min(20, len(feature_candidates)),
                            key="corr_k")
        if st.button("应用相关性筛选", key="btn_corr_filter"):
            corrs = numeric_df.corrwith(current_df[target_col]).abs().sort_values(ascending=False)
            selected = corrs.head(int(k)).index.tolist()
            st.session_state.feature_cols = selected
            st.success(f"✅ 已选择 {len(selected)} 个特征")

    # --- Tab 4: PCA 降维 (新增) ---
    with tabs[3]:
        st.markdown("#### 🧩 主成分分析 (PCA) 降维")
        st.info(
            "通过线性变换将原始特征映射到低维空间，生成互不相关的主成分 (PC)。\n\n**注意：** 应用PCA转换后，原始数值特征将被替换为 PC1, PC2...，这会丢失物理含义的可解释性，但能有效消除共线性并压缩维度。")

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

            # 预览/分析按钮
            if st.button("📊 预览 PCA 分析结果", key="btn_pca_preview"):
                try:
                    # 准备数据：PCA不支持NaN，这里用均值填充（假设已经做过基本清洗）
                    X = numeric_df.copy()
                    X = X.fillna(X.mean())

                    # 标准化是 PCA 的前置必要步骤
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    pca = PCA(**pca_args)
                    pca.fit(X_scaled)

                    # 指标计算
                    n_pc = pca.n_components_
                    explained = pca.explained_variance_ratio_
                    cum_explained = np.cumsum(explained)

                    st.success(f"✅ 计算完成：生成了 {n_pc} 个主成分，累计解释方差 {cum_explained[-1]:.4f}")

                    # 可视化：Scree Plot
                    st.markdown("##### 解释方差分布 (Scree Plot)")
                    chart_df = pd.DataFrame({
                        "Component": [f"PC{i + 1}" for i in range(n_pc)],
                        "Individual Variance": explained,
                        "Cumulative Variance": cum_explained
                    })
                    st.line_chart(chart_df.set_index("Component")[["Individual Variance", "Cumulative Variance"]])

                    # 保存模型到 session 以便应用
                    st.session_state['_pca_model'] = pca
                    st.session_state['_pca_scaler'] = scaler
                    st.session_state['_pca_ready'] = True

                except Exception as e:
                    st.error(f"PCA 分析出错: {e}")

            # 应用按钮
            if st.session_state.get('_pca_ready', False):
                st.markdown("---")
                st.warning(
                    "⚠️ **确认操作**：点击下方按钮将创建新的数据集。所有原始数值特征将被 PC1, PC2... 替换。此操作不可逆（除非重新加载文件）。")

                if st.button("🚀 应用 PCA 转换并更新数据集", type="primary", key="btn_pca_apply"):
                    pca = st.session_state['_pca_model']
                    scaler = st.session_state['_pca_scaler']

                    # 执行转换
                    X = numeric_df.copy().fillna(numeric_df.mean())
                    X_scaled = scaler.transform(X)
                    X_pca = pca.transform(X_scaled)

                    # 构建新 DataFrame
                    pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
                    df_pca = pd.DataFrame(X_pca, columns=pc_names, index=current_df.index)

                    # 合并：删除旧特征，保留目标列和其他非数值列（如文本、元数据）
                    df_rest = current_df.drop(columns=feature_candidates)
                    df_new = pd.concat([df_rest, df_pca], axis=1)

                    # 更新全局状态
                    st.session_state.processed_data = df_new
                    st.session_state.feature_cols = pc_names

                    # 清理临时状态
                    st.session_state.pop('_pca_model', None)
                    st.session_state.pop('_pca_scaler', None)
                    st.session_state.pop('_pca_ready', None)

                    st.success(f"✅ 数据集已更新！当前特征集: {pc_names}")
                    st.rerun()

    # --- Tab 5: 智能推荐 ---
    with tabs[4]:
        # 智能稀疏数据分析
        sparse_selector = SmartSparseDataSelector(current_df)

        st.markdown("#### 目标变量有效性分析")
        target_analysis = sparse_selector.get_target_analysis()
        st.dataframe(target_analysis.head(10), use_container_width=True)

        if st.button("🎯 智能推荐特征", key="btn_smart_rec"):
            feature_analysis = sparse_selector.analyze_features_for_target(target_col)
            recommended = feature_analysis[feature_analysis['推荐'] == '✓']['特征'].tolist()
            st.session_state.feature_cols = recommended
            st.success(f"✅ 推荐 {len(recommended)} 个特征")
            st.dataframe(feature_analysis, use_container_width=True)

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
        # 确保 preview cols 存在于 current_df 中
        available_preview = [c for c in st.session_state.feature_cols if c in current_df.columns]
        preview_cols = available_preview[:5] + [target_col]
        # 过滤掉不存在的列（防止PCA转换后引用旧列名报错）
        preview_cols = [c for c in preview_cols if c in current_df.columns]

        st.dataframe(current_df[preview_cols].head(), use_container_width=True)