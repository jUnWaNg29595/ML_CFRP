#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征前缀批量修改工具
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
import re


class FeaturePrefixManager:
    """特征前缀管理器"""

    @staticmethod
    def _apply_rename_to_session(rename_map: dict):
        """重命名后同步更新所有 session_state 中的特征名引用"""
        # 更新 feature_cols / multiselect_features
        if st.session_state.get('feature_cols'):
            st.session_state.feature_cols = [rename_map.get(f, f) for f in st.session_state.feature_cols]
            st.session_state.multiselect_features = st.session_state.feature_cols.copy()

        # 更新 molecular_feature_names（分子特征记录）
        if st.session_state.get('molecular_feature_names'):
            old_names = st.session_state.molecular_feature_names
            if isinstance(old_names, set):
                st.session_state.molecular_feature_names = {rename_map.get(f, f) for f in old_names}
            else:
                st.session_state.molecular_feature_names = [rename_map.get(f, f) for f in old_names]

        # 清除特征分类缓存（列名变了，必须重新分类）
        st.session_state.pop('feature_classification', None)
        # 清除 inf 检测缓存
        st.session_state.pop('_fs_inf_signature', None)
        st.session_state.pop('_fs_inf_count', None)

    def __init__(self, df: pd.DataFrame):
        """
        初始化

        Args:
            df: 输入DataFrame
        """
        self.df = df
        self.prefix_groups = self._analyze_prefixes()

    def _analyze_prefixes(self) -> Dict[str, List[str]]:
        """分析所有特征的前缀（优化版）"""
        prefix_groups = {}

        # [性能优化] 预先转换所有列名为字符串，避免重复转换
        columns_str = [str(col) for col in self.df.columns]

        # [性能优化] 预先计算所有前缀，避免重复split
        column_prefixes = []
        for col_str in columns_str:
            parts = col_str.split('_')

            if len(parts) >= 3:
                # 尝试2级前缀
                prefix_2 = '_'.join(parts[:2])
                prefix_1 = parts[0]
                column_prefixes.append((col_str, prefix_2, prefix_1))
            elif len(parts) >= 2:
                # 只有1级前缀
                prefix_1 = parts[0]
                column_prefixes.append((col_str, None, prefix_1))
            else:
                # 没有前缀
                column_prefixes.append((col_str, None, None))

        # [性能优化] 统计2级前缀的使用次数
        prefix_2_counts = {}
        for col_str, prefix_2, prefix_1 in column_prefixes:
            if prefix_2:
                prefix_2_counts[prefix_2] = prefix_2_counts.get(prefix_2, 0) + 1

        # 决定使用哪个前缀
        for col_str, prefix_2, prefix_1 in column_prefixes:
            if prefix_2 and prefix_2_counts.get(prefix_2, 0) > 1:
                # 使用2级前缀
                prefix = prefix_2
            elif prefix_1:
                # 使用1级前缀
                prefix = prefix_1
            else:
                # 没有前缀，跳过
                continue

            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            # 使用原始列名（不是字符串）
            original_col = self.df.columns[columns_str.index(col_str)]
            prefix_groups[prefix].append(original_col)

        # 按特征数量排序
        prefix_groups = dict(sorted(prefix_groups.items(), key=lambda x: len(x[1]), reverse=True))

        return prefix_groups

    def rename_prefix(self, old_prefix: str, new_prefix: str) -> pd.DataFrame:
        """
        重命名前缀（优化版）

        Args:
            old_prefix: 旧前缀
            new_prefix: 新前缀

        Returns:
            重命名后的DataFrame
        """
        # [性能优化] 使用字典推导式，一次性生成重命名映射
        prefix_with_underscore = old_prefix + '_'
        rename_map = {
            col: new_prefix + str(col)[len(old_prefix):]
            for col in self.df.columns
            if str(col).startswith(prefix_with_underscore)
        }

        if rename_map:
            df_renamed = self.df.rename(columns=rename_map)
        else:
            df_renamed = self.df.copy()

        return df_renamed, rename_map

    def remove_prefix(self, prefix_to_remove: str) -> pd.DataFrame:
        """
        移除前缀（优化版）

        Args:
            prefix_to_remove: 要移除的前缀

        Returns:
            移除前缀后的DataFrame
        """
        # [性能优化] 使用字典推导式
        prefix_with_underscore = prefix_to_remove + '_'
        prefix_len = len(prefix_with_underscore)

        rename_map = {
            col: str(col)[prefix_len:]
            for col in self.df.columns
            if str(col).startswith(prefix_with_underscore)
        }

        if rename_map:
            df_renamed = self.df.rename(columns=rename_map)
        else:
            df_renamed = self.df.copy()

        return df_renamed, rename_map

    def add_prefix(self, columns: List[str], prefix: str) -> pd.DataFrame:
        """
        添加前缀

        Args:
            columns: 要添加前缀的列
            prefix: 前缀

        Returns:
            添加前缀后的DataFrame
        """
        df_renamed = self.df.copy()

        rename_map = {}
        for col in columns:
            if col in df_renamed.columns:
                new_col = f"{prefix}_{col}"
                rename_map[col] = new_col

        if rename_map:
            df_renamed = df_renamed.rename(columns=rename_map)

        return df_renamed, rename_map

    def batch_rename(self, rename_rules: Dict[str, str]) -> pd.DataFrame:
        """
        批量重命名（优化版 - 避免嵌套循环）

        Args:
            rename_rules: 重命名规则字典 {old_prefix: new_prefix}

        Returns:
            重命名后的DataFrame
        """
        df_renamed = self.df.copy()
        all_rename_map = {}

        # [性能优化] 一次遍历所有列，避免嵌套循环
        for col in df_renamed.columns:
            col_str = str(col)
            # 检查是否匹配任何规则
            for old_prefix, new_prefix in rename_rules.items():
                if col_str.startswith(old_prefix + '_'):
                    new_col = new_prefix + col_str[len(old_prefix):]
                    all_rename_map[col] = new_col
                    break  # 找到匹配就跳出，避免重复处理

        if all_rename_map:
            df_renamed = df_renamed.rename(columns=all_rename_map)

        return df_renamed, all_rename_map

    def simplify_prefixes(self, max_prefix_parts: int = 1) -> pd.DataFrame:
        """
        简化前缀（移除多余的前缀部分）

        例如：hardener_MACCS_PC1 -> MACCS_PC1

        Args:
            max_prefix_parts: 保留的前缀部分数量

        Returns:
            简化后的DataFrame
        """
        df_renamed = self.df.copy()
        rename_map = {}

        for col in df_renamed.columns:
            parts = str(col).split('_')
            if len(parts) > max_prefix_parts + 1:  # +1 for the actual feature name
                # 保留最后的max_prefix_parts个前缀部分
                new_col = '_'.join(parts[-max_prefix_parts - 1:])
                rename_map[col] = new_col

        if rename_map:
            df_renamed = df_renamed.rename(columns=rename_map)

        return df_renamed, rename_map


def render_prefix_manager():
    """渲染特征前缀管理界面"""
    st.markdown("### 🏷️ 特征前缀批量修改")

    st.info("""
    **什么是前缀？**

    前缀是特征名称中下划线（_）前的第一部分。例如：
    - `hardener_MACCS_45` → 前缀是 `hardener_`
    - `resin_PC1` → 前缀是 `resin_`
    - `RDKit_PC2` → 前缀是 `RDKit_`

    💡 本工具可以批量修改、移除或简化这些前缀。
    """)

    # 检查数据
    if st.session_state.get('processed_data') is not None:
        current_df = st.session_state.processed_data
    elif st.session_state.get('data') is not None:
        current_df = st.session_state.data
    else:
        st.warning("⚠️ 请先上传数据")
        return

    # 创建管理器
    manager = FeaturePrefixManager(current_df)

    # 显示当前前缀统计
    st.markdown("#### 📊 当前前缀统计")

    if not manager.prefix_groups:
        st.info("未检测到带前缀的特征")
        return

    # 显示前缀表格
    prefix_stats = []
    for prefix, cols in manager.prefix_groups.items():
        prefix_stats.append({
            '前缀': f"{prefix}_",  # 添加下划线显示
            '特征数量': len(cols),
            '示例': ', '.join(cols[:3]) + ('...' if len(cols) > 3 else '')
        })

    st.dataframe(pd.DataFrame(prefix_stats), use_container_width=True, height=300)

    st.markdown("---")

    # 操作选项
    operation = st.radio(
        "选择操作",
        ["单个前缀重命名", "批量前缀重命名", "移除前缀", "简化前缀", "添加前缀"],
        horizontal=True
    )

    st.markdown("---")

    # 操作1: 单个前缀重命名
    if operation == "单个前缀重命名":
        st.markdown("#### 🔄 单个前缀重命名")

        col1, col2 = st.columns(2)

        with col1:
            old_prefix = st.selectbox(
                "选择要修改的前缀",
                options=list(manager.prefix_groups.keys()),
                format_func=lambda x: f"{x}_",  # 显示时添加下划线
                key="rename_old_prefix"
            )

        with col2:
            new_prefix = st.text_input(
                "新前缀",
                value=old_prefix,
                key="rename_new_prefix"
            )

        if old_prefix and new_prefix:
            st.info(f"预览：`{old_prefix}_PC1` → `{new_prefix}_PC1`")

            if st.button("🚀 执行重命名", type="primary", key="btn_rename_single"):
                df_renamed, rename_map = manager.rename_prefix(old_prefix, new_prefix)

                # 保存到 session_state
                st.session_state.pending_rename_df = df_renamed
                st.session_state.pending_rename_map = rename_map

                st.success(f"✅ 已重命名 {len(rename_map)} 个特征")

                with st.expander("查看重命名详情", expanded=False):
                    rename_df = pd.DataFrame([
                        {'原名称': old, '新名称': new}
                        for old, new in list(rename_map.items())[:50]
                    ])
                    st.dataframe(rename_df, use_container_width=True)
                    if len(rename_map) > 50:
                        st.caption(f"...还有 {len(rename_map) - 50} 个特征")

            # 应用到数据按钮（独立于执行按钮）
            if st.session_state.get('pending_rename_df') is not None:
                if st.button("✅ 应用到数据", type="primary", key="btn_apply_rename_single"):
                    df_renamed = st.session_state.pending_rename_df
                    rename_map = st.session_state.pending_rename_map

                    st.session_state.processed_data = df_renamed
                    FeaturePrefixManager._apply_rename_to_session(rename_map)

                    # 清除临时数据
                    del st.session_state.pending_rename_df
                    del st.session_state.pending_rename_map

                    st.success("✅ 已应用到数据！")
                    st.rerun()

    # 操作2: 批量前缀重命名
    elif operation == "批量前缀重命名":
        st.markdown("#### 🔄 批量前缀重命名")

        st.info("为多个前缀设置新名称")

        rename_rules = {}

        for prefix in list(manager.prefix_groups.keys())[:10]:  # 最多显示10个
            col1, col2 = st.columns([1, 1])

            with col1:
                st.text(f"原前缀: {prefix}_")  # 添加下划线显示

            with col2:
                new_name = st.text_input(
                    "新前缀",
                    value=prefix,
                    key=f"batch_rename_{prefix}",
                    label_visibility="collapsed"
                )
                if new_name != prefix:
                    rename_rules[prefix] = new_name

        if rename_rules:
            st.info(f"将重命名 {len(rename_rules)} 个前缀")

            if st.button("🚀 执行批量重命名", type="primary", key="btn_rename_batch"):
                df_renamed, rename_map = manager.batch_rename(rename_rules)

                # 保存到 session_state
                st.session_state.pending_rename_df = df_renamed
                st.session_state.pending_rename_map = rename_map

                st.success(f"✅ 已重命名 {len(rename_map)} 个特征")

                with st.expander("查看重命名详情", expanded=False):
                    rename_df = pd.DataFrame([
                        {'原名称': old, '新名称': new}
                        for old, new in list(rename_map.items())[:50]
                    ])
                    st.dataframe(rename_df, use_container_width=True)

            # 应用到数据按钮（独立于执行按钮）
            if st.session_state.get('pending_rename_df') is not None:
                if st.button("✅ 应用到数据", type="primary", key="btn_apply_rename_batch"):
                    df_renamed = st.session_state.pending_rename_df
                    rename_map = st.session_state.pending_rename_map

                    st.session_state.processed_data = df_renamed
                    FeaturePrefixManager._apply_rename_to_session(rename_map)

                    # 清除临时数据
                    del st.session_state.pending_rename_df
                    del st.session_state.pending_rename_map

                    st.success("✅ 已应用到数据！")
                    st.rerun()

    # 操作3: 移除前缀
    elif operation == "移除前缀":
        st.markdown("#### ✂️ 移除前缀")

        prefix_to_remove = st.selectbox(
            "选择要移除的前缀",
            options=list(manager.prefix_groups.keys()),
            format_func=lambda x: f"{x}_",  # 显示时添加下划线
            key="remove_prefix"
        )

        if prefix_to_remove:
            example_cols = manager.prefix_groups[prefix_to_remove][:3]
            st.info(f"预览：`{example_cols[0]}` → `{example_cols[0].split('_', 1)[1]}`")

            if st.button("🚀 执行移除", type="primary", key="btn_remove_prefix"):
                df_renamed, rename_map = manager.remove_prefix(prefix_to_remove)

                # 保存到 session_state
                st.session_state.pending_rename_df = df_renamed
                st.session_state.pending_rename_map = rename_map

                st.success(f"✅ 已移除 {len(rename_map)} 个特征的前缀")

                with st.expander("查看详情", expanded=False):
                    rename_df = pd.DataFrame([
                        {'原名称': old, '新名称': new}
                        for old, new in list(rename_map.items())[:50]
                    ])
                    st.dataframe(rename_df, use_container_width=True)

            # 应用到数据按钮（独立于执行按钮）
            if st.session_state.get('pending_rename_df') is not None:
                if st.button("✅ 应用到数据", type="primary", key="btn_apply_remove"):
                    df_renamed = st.session_state.pending_rename_df
                    rename_map = st.session_state.pending_rename_map

                    st.session_state.processed_data = df_renamed
                    FeaturePrefixManager._apply_rename_to_session(rename_map)

                    # 清除临时数据
                    del st.session_state.pending_rename_df
                    del st.session_state.pending_rename_map

                    st.success("✅ 已应用到数据！")
                    st.rerun()

    # 操作4: 简化前缀
    elif operation == "简化前缀":
        st.markdown("#### 🎯 简化前缀")

        st.info("移除多余的前缀部分，例如：`hardener_MACCS_PC1` → `MACCS_PC1`")

        max_parts = st.slider(
            "保留的前缀部分数量",
            1, 3, 1,
            key="simplify_max_parts",
            help="1 = 保留1个前缀部分，2 = 保留2个前缀部分"
        )

        # 预览
        preview_cols = []
        for cols in list(manager.prefix_groups.values())[:3]:
            if cols:
                col = cols[0]
                parts = col.split('_')
                if len(parts) > max_parts + 1:
                    new_col = '_'.join(parts[-max_parts - 1:])
                    preview_cols.append((col, new_col))

        if preview_cols:
            st.markdown("**预览：**")
            for old, new in preview_cols:
                st.text(f"{old} → {new}")

        if st.button("🚀 执行简化", type="primary", key="btn_simplify"):
            df_renamed, rename_map = manager.simplify_prefixes(max_parts)

            if rename_map:
                # 保存到 session_state
                st.session_state.pending_rename_df = df_renamed
                st.session_state.pending_rename_map = rename_map

                st.success(f"✅ 已简化 {len(rename_map)} 个特征")

                with st.expander("查看详情", expanded=False):
                    rename_df = pd.DataFrame([
                        {'原名称': old, '新名称': new}
                        for old, new in list(rename_map.items())[:50]
                    ])
                    st.dataframe(rename_df, use_container_width=True)
            else:
                st.info("没有需要简化的特征")

        # 应用到数据按钮（独立于执行按钮）
        if st.session_state.get('pending_rename_df') is not None:
            if st.button("✅ 应用到数据", type="primary", key="btn_apply_simplify"):
                df_renamed = st.session_state.pending_rename_df
                rename_map = st.session_state.pending_rename_map

                st.session_state.processed_data = df_renamed
                FeaturePrefixManager._apply_rename_to_session(rename_map)

                # 清除临时数据
                del st.session_state.pending_rename_df
                del st.session_state.pending_rename_map

                st.success("✅ 已应用到数据！")
                st.rerun()

    # 操作5: 添加前缀
    elif operation == "添加前缀":
        st.markdown("#### ➕ 添加前缀")

        new_prefix = st.text_input(
            "要添加的前缀",
            value="feature",
            key="add_prefix_name"
        )

        # 选择要添加前缀的列
        all_cols = current_df.columns.tolist()
        selected_cols = st.multiselect(
            "选择要添加前缀的特征",
            options=all_cols,
            key="add_prefix_cols"
        )

        if selected_cols and new_prefix:
            st.info(f"预览：`{selected_cols[0]}` → `{new_prefix}_{selected_cols[0]}`")

            if st.button("🚀 执行添加", type="primary", key="btn_add_prefix"):
                df_renamed, rename_map = manager.add_prefix(selected_cols, new_prefix)

                # 保存到 session_state
                st.session_state.pending_rename_df = df_renamed
                st.session_state.pending_rename_map = rename_map

                st.success(f"✅ 已为 {len(rename_map)} 个特征添加前缀")

                with st.expander("查看详情", expanded=False):
                    rename_df = pd.DataFrame([
                        {'原名称': old, '新名称': new}
                        for old, new in list(rename_map.items())[:50]
                    ])
                    st.dataframe(rename_df, use_container_width=True)

            # 应用到数据按钮（独立于执行按钮）
            if st.session_state.get('pending_rename_df') is not None:
                if st.button("✅ 应用到数据", type="primary", key="btn_apply_add"):
                    df_renamed = st.session_state.pending_rename_df
                    rename_map = st.session_state.pending_rename_map

                    st.session_state.processed_data = df_renamed
                    FeaturePrefixManager._apply_rename_to_session(rename_map)

                    # 清除临时数据
                    del st.session_state.pending_rename_df
                    del st.session_state.pending_rename_map

                    st.success("✅ 已应用到数据！")
                    st.rerun()


if __name__ == "__main__":
    # 测试代码
    import numpy as np

    # 创建测试数据
    test_data = {
        'hardener_MACCS_PC1': np.random.rand(10),
        'hardener_MACCS_PC2': np.random.rand(10),
        'resin_MACCS_PC1': np.random.rand(10),
        'resin_MACCS_PC2': np.random.rand(10),
        'RDKit_PC1': np.random.rand(10),
        'RDKit_PC2': np.random.rand(10),
        'temperature': np.random.rand(10),
        'pressure': np.random.rand(10),
    }

    df_test = pd.DataFrame(test_data)

    print("原始列名:")
    print(df_test.columns.tolist())

    manager = FeaturePrefixManager(df_test)

    print("\n前缀统计:")
    for prefix, cols in manager.prefix_groups.items():
        print(f"  {prefix}: {len(cols)} 个特征")

    # 测试重命名
    df_renamed, rename_map = manager.rename_prefix('hardener', 'H')
    print("\n重命名后:")
    print(df_renamed.columns.tolist())

    # 测试简化
    df_simplified, simplify_map = manager.simplify_prefixes(max_prefix_parts=1)
    print("\n简化后:")
    print(df_simplified.columns.tolist())
