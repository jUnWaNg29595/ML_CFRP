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


def _file_signature(path: str) -> tuple:
    """文件指纹（路径 + mtime + 大小）。文件被改写后指纹变化，缓存自动失效。"""
    try:
        stat = os.stat(path)
        return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (os.path.abspath(path), None, None)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_read_csv_table(path: str, mtime_ns, size: int) -> pd.DataFrame:
    """按文件指纹缓存 CSV 读取结果。

    母宽表 ml_wide_samples.csv 约 20MB / 1200+ 列，直接读盘需 0.7~1.1s。
    本面板每次 rerun 都会重建，缓存后可把重复读盘降到 0。
    调用方只读该 DataFrame（引擎内部均先 .copy() 再改），不会污染缓存对象。
    """
    return pd.read_csv(path, low_memory=False)


def read_csv_cached(path: str) -> pd.DataFrame:
    """读取 CSV（带指纹缓存）。文件不存在时由 pandas 正常抛错。"""
    _abs, mtime_ns, size = _file_signature(path)
    return _cached_read_csv_table(path, mtime_ns, size)


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
            "进行智能化学语义对齐，**一键补全各单体用量 (PHR)、当量 (EEW/AHEW)、固化工艺温度/时间参数及 18 项交联机理特征**。"
        )
    else:
        st.markdown("### 🔗 跨表配方数据融合 (One-Click Fusion)")
        st.caption("自动关联母配方宽表 `ml_wide_samples.csv`，补充各组分精确配比、物理机理参数及固化工艺温度。")

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
                        df_narrow = read_csv_cached(selected_narrow_path)
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
                        df_wide = read_csv_cached(selected_wide_path)
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
            single_component_only = st.checkbox(
                "🧬 仅保留单组分配方 (树脂/固化剂各 1 组分，不含小分子添加剂)",
                value=False,
                help="**单组分配方模式**：\n\n"
                     "1. **样本筛选**：剔除树脂 2/3 组分、固化剂 2/3 组分任一非空的样本；\n"
                     "2. **小分子添加剂**：small_additive_1/2 均丢弃（含添加剂样本一并剔除，列不进入工作表）；\n"
                     "3. **组分数量特征**：筛选后 `*_component_count` 恒为 1/0，全部不放入工作表；\n"
                     "4. 输出仅保留 resin_1 + curing_agent_1 的纯净双组分体系。"
            )
            drop_useless_metadata = st.checkbox(
                "🧹 彻底剔除无意义元数据列 (curing_type_standard, curing_mechanism, *_format, raw_unit 等)",
                value=True,
                help="自动剔除格式标记 (format)、固化类型、机理说明及原始单位说明等无法用于回归训练的纯文本/低信噪比元数据。"
            )
            include_resin_3 = st.checkbox(
                "🧪 规范化纳入三组分配方特征 (保留 resin_3_structure, phr, MW, 官能度)",
                value=True,
                disabled=single_component_only,
                help="保留三元树脂共混配方的结构、用量和物理量，同时剥离冗余的 format、unit 等元数据标签。"
                + ("　⚠️ 已被「单组分配方模式」覆盖：多组分列不会进入工作表。" if single_component_only else "")
            )

        with col_opt2:
            augment_standards = st.checkbox(
                "📜 关联测试标准 (ASTM / ISO / GB / DIN / JIS)",
                value=True,
                help="从 `ml_performance_standards.csv` 自动关联样本使用的测试标准，"
                     "输出 `test_standard_organization`、`test_standard_canonical` (如 ASTM D3418-1982) 与引用数量。"
                     "窄表无 ID 时通过 record_id 桥接，未覆盖样本留空。"
            )
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

                    # 可选：关联测试标准 (ASTM / ISO / GB / DIN / JIS)
                    if augment_standards:
                        fused_df, std_meta = engine.augment_with_test_standards(fused_df)
                        meta.update(std_meta)

                    # 执行特征纯化清洗
                    fused_clean_df, clean_stats = engine.clean_features_for_ml(
                        fused_df,
                        target_col=target_name,
                        curing_type_filter=curing_type_val,
                        mode=mode_val,
                        drop_metadata=drop_useless_metadata,
                        base_df=df_narrow,
                        single_component_only=single_component_only
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

                    if single_component_only:
                        single_info = (
                            f"🧬 **单组分配方**: 仅树脂/固化剂各 1 组分且无小分子添加剂 "
                            f"({clean_stats.get('single_filtered_rows', len(fused_clean_df))} 样本保留，"
                            f"已剔除多组分列与 {clean_stats.get('single_dropped_cols_count', 0)} 个组分数量特征)"
                        )
                    else:
                        single_info = "含全部组分配方样本"

                    proc_backfilled = meta.get("process_backfilled") or []
                    temp_cols = [c for c in fused_clean_df.columns if "temperature" in str(c).lower()]
                    if proc_backfilled:
                        proc_info = f"🔥 工艺温度补齐: 从母宽表指纹回填 {len(proc_backfilled)} 列（含 {', '.join(temp_cols[:2])}）"
                    elif temp_cols:
                        proc_info = f"🔥 工艺温度: 已包含 {len(temp_cols)} 个工艺温度列（如 {temp_cols[0]}）"
                    else:
                        proc_info = "⚠️ 未见工艺温度列（母宽表可能缺失）"

                    if augment_standards and meta.get("standards_matched", 0) > 0:
                        std_info = (
                            f"📜 测试标准: {meta.get('standards_matched')} 样本已关联 "
                            f"({meta.get('standards_coverage', 0) * 100:.1f}% 覆盖，策略: {meta.get('standards_strategy')})"
                        )
                    elif augment_standards:
                        std_info = "📜 测试标准: 未找到可关联的标准记录（可检查 ml_performance_standards.csv）"
                    else:
                        std_info = None

                    success_lines = [
                        f"🎉 **跨表融合与特征纯化完成！**\n\n",
                        f"* 📊 **最终训练维度**: **{fused_clean_df.shape[0]} 行 × {fused_clean_df.shape[1]} 列**\n",
                        f"* 🔬 **筛选与组织**: {curing_info} | {r3_info}\n",
                        f"* {proc_info}\n",
                    ]
                    if std_info:
                        success_lines.append(f"* {std_info}\n")
                    success_lines.extend([
                        f"* {single_info}\n",
                        f"* 🧹 **纯化保障**: 已彻底剥离 `curing_type_standard`、`curing_mechanism`、全部 `*_format` 格式列及原始单位元数据，特征集干净纯粹。",
                    ])
                    st.success("".join(success_lines))
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
        # [性能] 融合结果是宽表（紧凑模式 ~70-80 列，全息模式 300-450 列，原始模式可达 1248 列）。
        # 直接渲染会在每次交互 rerun 时重发整份数据（实测 1248 列 × 20 行 = 439KB），
        # 因此复用 app_lib.render_capped_preview 默认只展示前 40 列。
        try:
            from core.preview_ui import render_capped_preview
            render_capped_preview(
                res_df, key="fusion_result_preview", caption_prefix="融合结果", height=320
            )
        except Exception:
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

    # ν 嵌入模型训练区（两阶段 PINN · 阶段1）
    try:
        render_nu_encoder_ui(default_dir=default_dir)
    except Exception as e_nu:
        st.warning(f"ν 编码器组件加载失败: {e_nu}")


@st.fragment
def render_nu_encoder_ui(default_dir: str = r"C:\Users\wangj\Desktop\ml_dataset"):
    """
    渲染交联密度 (ν) 嵌入模型训练 UI（两阶段 PINN · 阶段1）
    - 用实测 ν 数据一键训练专用编码器（含脏数据清理、留出评估）
    - 冻结产物自动被 EpoxyPINN / Transformer+PINN 发现并嵌入
    - 训练与预测均无需手工输入 ν
    """
    st.markdown("---")
    st.markdown("#### 🧬 交联密度 (ν) 嵌入模型（两阶段 PINN · 阶段1）")
    st.caption(
        "用实测交联密度训练专用编码器（自动清理脏数据），冻结后自动嵌入 PINN 作为物理基线——"
        "**训练与预测均无需手工输入 ν**。编码器激活后建议将 PINN 的 Physics Weight 调至 0.2~0.3。"
    )

    from core.crosslink_nu_model import CrosslinkNuEncoder, default_encoder_path

    enc_path = default_encoder_path()
    if os.path.exists(enc_path):
        try:
            enc = CrosslinkNuEncoder.load(enc_path)
            prov = getattr(enc, "provenance_", {}) or {}
            st.success(
                f"✅ ν 编码器已就绪并会被 PINN 自动嵌入 ｜ 训练样本: {prov.get('n_train', '?')} ｜ "
                f"特征数: {prov.get('n_features', '?')} ｜ 清理剔除: {prov.get('n_dropped_dirty', '?')} 条脏数据"
            )
        except Exception as e_enc:
            st.warning(f"⚠️ 编码器文件存在但加载失败（将回退理论基线）: {e_enc}")
    else:
        st.info("ℹ️ 尚未训练 ν 编码器 —— PINN 将回退为较弱的理论基线，建议先训练。")

    # 数据源扫描（与融合引擎同目录逻辑）
    nu_files: list = []
    for s_dir in dict.fromkeys([default_dir, os.getcwd()]):
        if os.path.exists(s_dir):
            try:
                for fname in os.listdir(s_dir):
                    if fname.lower().endswith(".csv") and "crosslink_density" in fname.lower():
                        nu_files.append(os.path.join(s_dir, fname))
            except Exception:
                pass
    if not nu_files:
        st.warning("未找到含 crosslink_density 的数据文件（可将实测 ν 表放入 ml_dataset 目录）")
        return

    src = st.selectbox("ν 数据源", nu_files,
                       format_func=lambda p: os.path.basename(p))

    c1, c2, c3 = st.columns(3)
    lo = c1.number_input("ν 合理下界 (mol/m³)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0,
                         help="低于下界的记录视为单位错误/异常，训练时剔除")
    hi = c2.number_input("ν 合理上界 (mol/m³)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=500.0,
                         help="实测数据显示 >1e4 的记录多为模量回算单位错误")
    holdout = c3.slider("留出评估比例", 0.1, 0.4, 0.2, 0.05,
                        help="先在留出集上评估编码器质量（Spearman），再用全量数据重训正式产物")

    if st.button("🚀 训练 / 更新 ν 编码器", type="primary", use_container_width=True):
        try:
            with st.spinner("清理数据 → 留出评估 → 全量重训 → 保存 ..."):
                df = pd.read_csv(src, encoding="utf-8", encoding_errors="replace")
                nu_col = "crosslink_density_mol_m3" if "crosslink_density_mol_m3" in df.columns else \
                    next((c for c in df.columns if "crosslink_density" in c.lower()
                          and pd.api.types.is_numeric_dtype(df[c])), None)
                if nu_col is None:
                    st.error("❌ 数据源中未找到数值型交联密度列")
                    return
                # 剔除 ν 自身的测量条件列（防止编码器学到测量方法偏差）
                drop_cols = [c for c in df.columns if c != nu_col and c.startswith(nu_col + "_")]
                X_df = df.drop(columns=[nu_col] + drop_cols)
                nu = pd.to_numeric(df[nu_col], errors="coerce").to_numpy(dtype=float)

                valid = np.isfinite(nu) & (nu >= lo) & (nu <= hi)
                st.caption(f"数据体检: 共 {len(nu)} 条 ｜ 有效 {int(valid.sum())} 条 ｜ "
                           f"剔除脏数据 {int((~valid).sum())} 条（区间 [{lo:.0f}, {hi:.0f}] 之外）")

                rng = np.random.RandomState(42)
                v_idx = np.where(valid)[0]
                perm = rng.permutation(v_idx)
                n_te = max(1, int(len(perm) * holdout))
                te_idx, tr_idx = perm[:n_te], perm[n_te:]

                enc_eval = CrosslinkNuEncoder(nu_bounds=(float(lo), float(hi)))
                enc_eval.fit(X_df.iloc[tr_idx].reset_index(drop=True), nu[tr_idx])
                lp = enc_eval.predict_log_nu(X_df.iloc[te_idx].reset_index(drop=True))
                from scipy.stats import spearmanr as _spr
                from sklearn.metrics import r2_score as _r2
                rho = float(_spr(np.exp(lp), nu[te_idx])[0])
                r2l = float(_r2(np.log(nu[te_idx]), lp))

                m1, m2 = st.columns(2)
                m1.metric("留出集 Spearman (ν)", f"{rho:.3f}", help="ν 编码器预测与实测的秩相关；理论基线约 0.24")
                m2.metric("留出集 R² (log ν)", f"{r2l:.3f}")

                final = CrosslinkNuEncoder(nu_bounds=(float(lo), float(hi)))
                final.fit(X_df, nu)
                final.save(enc_path)
                st.success(f"✅ 编码器已保存: `{enc_path}` —— PINN 下次训练自动嵌入，无需任何额外操作")
                st.caption("💡 提示：编码器激活后（ν 100% 覆盖），建议将 PINN 的 Physics Weight 调至 0.2~0.3")
        except ImportError as e_imp:
            st.error(f"❌ 缺少依赖: {e_imp}")
        except Exception as e_train:
            st.error(f"❌ 训练失败: {e_train}")
