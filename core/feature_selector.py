# -*- coding: utf-8 -*-
"""特征选择模块 - 完整版"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import VarianceThreshold
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
        analysis = [{
            '变量名': col,
            '有效样本数': info['non_null_count'],
            '有效率': f"{info['non_null_ratio']*100:.1f}%",
            '缺失数': info['null_count']
        } for col, info in self.sparsity_info.items()]
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
                '有效率': f"{valid_count/n*100:.1f}%" if n > 0 else "0%",
                '推荐': '✓' if valid_count / n >= min_valid_ratio else '✗'
            })
        
        return pd.DataFrame(analysis).sort_values('有效数', ascending=False)


def show_robust_feature_selection():
    """完整的特征选择界面"""
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

    # 特征选择模式
    if len(feature_candidates) > 50:
        st.info(f"📊 特征数量较多 ({len(feature_candidates)}个)，使用批量筛选模式")
        
        tab_a, tab_b, tab_c = st.tabs(["方差筛选", "相关性筛选", "智能推荐"])

        with tab_a:
            threshold = st.slider("方差阈值", 0.0, 1.0, 0.0, 0.01)
            if st.button("应用方差筛选"):
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(numeric_df.fillna(0))
                selected = [feature_candidates[i] for i in selector.get_support(indices=True)]
                st.session_state.feature_cols = selected
                st.success(f"✅ 筛选后剩余 {len(selected)} 个特征")

        with tab_b:
            k = st.number_input("保留相关性最高的K个", 1, len(feature_candidates), min(20, len(feature_candidates)))
            if st.button("应用相关性筛选"):
                corrs = numeric_df.corrwith(current_df[target_col]).abs().sort_values(ascending=False)
                selected = corrs.head(int(k)).index.tolist()
                st.session_state.feature_cols = selected
                st.success(f"✅ 已选择 {len(selected)} 个特征")

        with tab_c:
            # 智能稀疏数据分析
            sparse_selector = SmartSparseDataSelector(current_df)
            
            st.markdown("#### 目标变量有效性分析")
            target_analysis = sparse_selector.get_target_analysis()
            st.dataframe(target_analysis.head(10), use_container_width=True)
            
            if st.button("🎯 智能推荐特征"):
                feature_analysis = sparse_selector.analyze_features_for_target(target_col)
                recommended = feature_analysis[feature_analysis['推荐'] == '✓']['特征'].tolist()
                st.session_state.feature_cols = recommended
                st.success(f"✅ 推荐 {len(recommended)} 个特征")
                st.dataframe(feature_analysis, use_container_width=True)
    else:
        # 少量特征时的多选模式
        st.markdown("#### 选择输入特征 (X)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("全选"):
                st.session_state.feature_cols = feature_candidates
        with col2:
            if st.button("清空"):
                st.session_state.feature_cols = []

        selected_features = st.multiselect(
            "选择特征",
            options=feature_candidates,
            default=st.session_state.get('feature_cols', [])
        )
        st.session_state.feature_cols = selected_features

    # 显示已选特征
    if st.session_state.get('feature_cols'):
        st.markdown("---")
        st.markdown(f"### ✅ 已选择 {len(st.session_state.feature_cols)} 个特征")
        
        cols = st.columns(4)
        for i, feat in enumerate(st.session_state.feature_cols[:20]):
            with cols[i % 4]:
                st.markdown(f"<span style='background:#E0E7FF;padding:4px 8px;border-radius:12px;font-size:0.85rem;'>{feat}</span>", unsafe_allow_html=True)
        
        if len(st.session_state.feature_cols) > 20:
            st.caption(f"... 等共 {len(st.session_state.feature_cols)} 个特征")

        # 数据预览
        st.markdown("#### 📋 数据预览")
        preview_cols = st.session_state.feature_cols[:5] + [target_col]
        st.dataframe(current_df[preview_cols].head(), use_container_width=True)
