# -*- coding: utf-8 -*-
"""
Formulation Fusion UI Module
跨表配方数据融合引擎前端界面
"""

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

from core.formulation_fusion import FormulationFusionEngine


@st.fragment
def render_formulation_fusion_ui(
    standalone: bool = True,
    default_dir: str = r"C:\Users\wangj\Desktop\ml_dataset"
):
    """
    渲染跨表配方数据融合引擎 UI (已片段化 @st.fragment，交互与按钮不触发整页刷新)

    Args:
        standalone: 是否作为独立页面渲染（带一级大标题），若作为 tab 则适配容器
        default_dir: 默认配方数据扫描目录
    """
    if standalone:
        st.title("🔗 跨表配方数据融合引擎 (Formulation Fusion Engine)")
        st.markdown(
            "将各个目标性能窄表（如 `ml_qspr_model_tg_c.csv`）与全组分配方母宽表（`ml_wide_samples.csv`）"
            "进行智能化学语义对齐，**一键补全各单体用量 (PHR)、当量 (EEW/AHEW) 及 18 项交联机理特征**。"
        )
    else:
        st.markdown("### 🔗 跨表配方数据融合 (One-Click Fusion)")
        st.caption("自动关联母配方宽表 `ml_wide_samples.csv`，补充各组分精确配比与物理机理参数。")

    engine = FormulationFusionEngine()

    # 1. 扫描可用文件
    scan_dirs = [default_dir, os.getcwd()]
    candidate_narrow_files = []
    candidate_wide_files = []

    for s_dir in scan_dirs:
        if os.path.exists(s_dir):
            try:
                for fname in os.listdir(s_dir):
                    if fname.endswith(".csv"):
                        fpath = os.path.join(s_dir, fname)
                        if "wide" in fname.lower():
                            if fpath not in candidate_wide_files:
                                candidate_wide_files.append(fpath)
                        elif "qspr_model" in fname.lower() or "performance" in fname.lower():
                            # 过滤掉衍生生成的文件如 fused_*，保证输入窄表干净规范
                            if fpath not in candidate_narrow_files and not fname.startswith("fused_"):
                                candidate_narrow_files.append(fpath)
            except Exception:
                pass

    # 智能排序：优先将核心 QSPR 模型文件置顶，首选 ml_qspr_model_tg_c.csv
    def sort_narrow_key(path_str):
        bn = os.path.basename(path_str).lower()
        if "tg_c" in bn:
            return 0
        if bn.startswith("ml_qspr_model_"):
            return 1
        if bn.startswith("ml_performance_"):
            return 2
        return 3

    candidate_narrow_files.sort(key=sort_narrow_key)

    # 2. 表格选择区
    st.markdown("#### 📂 1. 选择待融合的数据表格")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**🎯 目标性能窄表 (Narrow Table)**")
        narrow_source = st.radio(
            "数据来源",
            ["从 ml_dataset 目录选择", "当前工作区数据", "手动上传 CSV"],
            horizontal=True,
            key="fusion_narrow_source_radio"
        )

        df_narrow: Optional[pd.DataFrame] = None
        narrow_label = ""

        if narrow_source == "从 ml_dataset 目录选择":
            if candidate_narrow_files:
                # 默认优先选中 tg_c
                default_idx = 0
                for i_n, n_path in enumerate(candidate_narrow_files):
                    if "tg_c" in os.path.basename(n_path).lower():
                        default_idx = i_n
                        break
                selected_narrow_path = st.selectbox(
                    "选择目标性能文件",
                    candidate_narrow_files,
                    index=default_idx,
                    format_func=lambda p: os.path.basename(p),
                    key="fusion_narrow_file_select"
                )
                if selected_narrow_path and os.path.exists(selected_narrow_path):
                    try:
                        df_narrow = pd.read_csv(selected_narrow_path)
                        narrow_label = os.path.basename(selected_narrow_path)
                        st.caption(f"已加载: `{narrow_label}` ({df_narrow.shape[0]} 行 × {df_narrow.shape[1]} 列)")
                    except Exception as e:
                        st.error(f"读取失败: {e}")
            else:
                st.warning("⚠️ 未在目录下扫描到 performance/qspr 表格，请切换为手动上传。")

        elif narrow_source == "当前工作区数据":
            cur_df = st.session_state.get("processed_data") or st.session_state.get("data")
            if cur_df is not None:
                df_narrow = cur_df.copy()
                narrow_label = "工作区数据"
                st.caption(f"当前工作区: {df_narrow.shape[0]} 行 × {df_narrow.shape[1]} 列")
            else:
                st.info("💡 当前工作区暂未加载数据，请选择其他数据来源。")

        else:
            up_file = st.file_uploader("上传目标性能表格 (CSV)", type=["csv"], key="fusion_narrow_uploader")
            if up_file is not None:
                try:
                    df_narrow = pd.read_csv(up_file)
                    narrow_label = up_file.name
                    st.caption(f"上传成功: {df_narrow.shape[0]} 行 × {df_narrow.shape[1]} 列")
                except Exception as e:
                    st.error(f"解析失败: {e}")

    with col_right:
        st.markdown("**🌐 配方母宽表 (Wide Master Table)**")
        wide_source = st.radio(
            "母宽表来源",
            ["自动检测 / 推荐宽表", "手动上传 CSV"],
            horizontal=True,
            key="fusion_wide_source_radio"
        )

        df_wide: Optional[pd.DataFrame] = None
        wide_label = ""

        if wide_source == "自动检测 / 推荐宽表":
            default_wide = os.path.join(default_dir, "ml_wide_samples.csv")
            if os.path.exists(default_wide):
                if default_wide not in candidate_wide_files:
                    candidate_wide_files.insert(0, default_wide)

            if candidate_wide_files:
                selected_wide_path = st.selectbox(
                    "选择配方母宽表",
                    candidate_wide_files,
                    index=0,
                    format_func=lambda p: os.path.basename(p),
                    key="fusion_wide_file_select"
                )
                if selected_wide_path and os.path.exists(selected_wide_path):
                    try:
                        df_wide = pd.read_csv(selected_wide_path, low_memory=False)
                        wide_label = os.path.basename(selected_wide_path)
                        st.caption(f"已加载: `{wide_label}` ({df_wide.shape[0]} 行 × {df_wide.shape[1]} 列)")
                    except Exception as e:
                        st.error(f"读取失败: {e}")
            else:
                st.warning("⚠️ 未找到配方母宽表，请指定文件或上传。")
        else:
            up_wide = st.file_uploader("上传配方母宽表 (CSV)", type=["csv"], key="fusion_wide_uploader")
            if up_wide is not None:
                try:
                    df_wide = pd.read_csv(up_wide, low_memory=False)
                    wide_label = up_wide.name
                    st.caption(f"上传成功: {df_wide.shape[0]} 行 × {df_wide.shape[1]} 列")
                except Exception as e:
                    st.error(f"解析失败: {e}")

    # 3. 智能检测与选项
    st.markdown("---")
    st.markdown("#### ⚙️ 2. 融合配置与智能语义匹配")

    if df_narrow is not None and df_wide is not None:
        try:
            target_name = engine.extract_target_name_from_df(df_narrow, df_wide, filename=narrow_label)
        except Exception:
            target_name = df_narrow.columns[-1] if len(df_narrow.columns) > 0 else None
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(f"🎯 自动识别目标性能列: **`{target_name}`**" if target_name else "🎯 目标性能列: 未自动识别（将启用化学指纹比对）")
        with col_m2:
            st.success(f"🔗 准备对齐: 窄表 **{len(df_narrow):,}** 行  ⟷  母宽表 **{len(df_wide):,}** 行")

        # 选项
        st.markdown("##### 🔬 固化体系与特征纯化清洗配置")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            curing_filter_choice = st.selectbox(
                "固化体系筛选 (Curing Type Filter)",
                [
                    "仅保留外加固化剂 (external_hardener) [推荐，约 6,474 样本]",
                    "保留全部固化体系 (包含双固化、催化等全部样本)"
                ],
                index=0,
                help="筛选常规外加固化剂（胺类、酸酐等），避免自固化或催化体系干扰；筛选后自动剔除单值冗余的 curing_type_standard 列。"
            )
            curing_type_val = "external_hardener" if "external_hardener" in curing_filter_choice else None

        with col_c2:
            clean_mode_choice = st.selectbox(
                "特征纯化清洗模式 (Feature Cleaning Mode)",
                [
                    "🎯 类似 ml_qspr_model 标准紧凑特征集 (规范保留 resin_3 物理特征，剔除元数据，约 70-80 列)",
                    "🌐 全息非空配方特征集 (保留母宽表所有有效单体数值列，约 300-450 列)",
                    "📦 原始母宽表合并 (仅剔除基础元数据)"
                ],
                index=0,
                help="紧凑模式遵循 ml_qspr_model 规范，分子结构居前，保留精确 PHR、MW 及 18 项物理机理参数，剔除 format 及文本标签，信噪比最高。"
            )
            mode_val = "qspr_clean" if "紧凑" in clean_mode_choice else ("comprehensive" if "全息" in clean_mode_choice else "raw")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            drop_useless_metadata = st.checkbox(
                "🧹 彻底剔除无意义元数据列 (curing_type_standard, curing_mechanism, *_format, raw_unit 等)",
                value=True,
                help="自动剔除格式标记 (format)、固化类型、机理说明及原始单位说明等无法用于回归训练的纯文本/低信噪比元数据。"
            )
            include_resin_3 = st.checkbox(
                "🧪 规范化纳入三组分配方特征 (保留 resin_3_structure, phr, MW, 官能度)",
                value=True,
                help="保留三元树脂共混配方的结构、用量和物理量，同时剥离冗余的 format、unit 等元数据标签。"
            )

        with col_opt2:
            auto_load_workspace = st.checkbox(
                "📥 融合清洗完成后自动载入系统工作区 (直接供后续特征提取/训练使用)",
                value=True,
                help="若勾选，融合清洗后的 DataFrame 将直接同步至全局状态，可直接进入特征工程或模型训练。"
            )

        # 4. 执行按钮
        st.markdown("---")
        exec_clicked = st.button("🚀 执行一键跨表配方融合与特征纯化 (One-Click Fusion & Curation)", type="primary", use_container_width=True)

        if exec_clicked:
            with st.spinner("⏳ 正在执行跨表自适应对齐、固化体系筛选与特征纯化清洗..."):
                try:
                    fused_df, meta = engine.auto_align_and_fuse(
                        df_narrow,
                        df_wide,
                        target_col=target_name,
                        fill_missing_r=True
                    )

                    # 执行特征纯化清洗
                    fused_clean_df, clean_stats = engine.clean_features_for_ml(
                        fused_df,
                        target_col=target_name,
                        curing_type_filter=curing_type_val,
                        mode=mode_val,
                        drop_metadata=drop_useless_metadata,
                        base_df=df_narrow
                    )

                    meta.update(clean_stats)

                    st.session_state["fusion_result_df"] = fused_clean_df
                    st.session_state["fusion_result_meta"] = meta

                    if auto_load_workspace:
                        st.session_state["data"] = fused_clean_df
                        st.session_state["processed_data"] = fused_clean_df
                        st.session_state["target_col"] = target_name

                    curing_info = f"固化体系筛选: `{curing_type_val}` ({clean_stats.get('filtered_rows', len(fused_clean_df))} 样本)" if curing_type_val else "保留全部固化体系"
                    r3_info = "已规范保留 resin_3 物理特征" if clean_stats.get("resin_3_included") else ""

                    st.success(
                        f"🎉 **跨表融合与特征纯化完成！**\n\n"
                        f"* 📊 **最终训练维度**: **{fused_clean_df.shape[0]} 行 × {fused_clean_df.shape[1]} 列**\n"
                        f"* 🔬 **筛选与组织**: {curing_info} | {r3_info}\n"
                        f"* 🧹 **纯化保障**: 已彻底剥离 `curing_type_standard`、`curing_mechanism`、全部 `*_format` 格式列及原始单位元数据，特征集干净纯粹。"
                    )
                except Exception as e_fuse:
                    st.error(f"❌ 融合失败: {e_fuse}")

    else:
        st.warning("👈 请在上方选择或上传【目标性能窄表】与【配方母宽表】后再执行融合。")

    # 5. 结果查看与导出
    if "fusion_result_df" in st.session_state:
        res_df = st.session_state["fusion_result_df"]
        meta_res = st.session_state.get("fusion_result_meta", {})

        st.markdown("---")
        st.markdown("#### 📊 3. 融合后数据预览与操作")
        st.dataframe(res_df.head(20), width="stretch")

        # 净化文件名：去除重复的 fused_ 前缀与 .csv.csv
        clean_base_label = re.sub(r'^(fused_)+', '', narrow_label or 'dataset')
        clean_base_label = re.sub(r'(\.csv)+$', '', clean_base_label)
        clean_out_name = f"fused_{clean_base_label}.csv"

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("📥 重新载入当前工作区", use_container_width=True):
                st.session_state["data"] = res_df
                st.session_state["processed_data"] = res_df
                st.success("✅ 已同步至当前工作区！可直接进入【特征工程】或【模型训练】页面。")

        with col_b2:
            csv_data = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 下载融合数据集 (CSV)",
                data=csv_data,
                file_name=clean_out_name,
                mime="text/csv",
                use_container_width=True
            )

        with col_b3:
            # 快捷保存至 ml_dataset 目录
            save_path_default = os.path.join(default_dir, clean_out_name)
            if st.button("📁 直接保存至 ml_dataset 目录", use_container_width=True):
                try:
                    res_df.to_csv(save_path_default, index=False, encoding='utf-8-sig')
                    st.success(f"✅ 已保存至: `{save_path_default}`")
                except Exception as e_save:
                    st.error(f"保存失败: {e_save}")
