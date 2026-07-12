# -*- coding: utf-8 -*-
"""
数据库规模展示页面 - 紧凑横向布局（适合PPT截图）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json


def render_database_showcase_compact(df):
    """渲染紧凑的横向数据库规模展示界面（适合截图）"""

    # 添加自定义CSS样式
    st.markdown("""
    <style>
    .compact-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .compact-number {
        font-size: 36px;
        font-weight: bold;
        margin: 5px 0;
    }
    .compact-label {
        font-size: 13px;
        opacity: 0.95;
    }
    .gradient-pink { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 5px 15px rgba(245, 87, 108, 0.3); }
    .gradient-blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3); }
    .gradient-green { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); box-shadow: 0 5px 15px rgba(67, 233, 123, 0.3); }
    .gradient-orange { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); box-shadow: 0 5px 15px rgba(250, 112, 154, 0.3); }
    .gradient-purple { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3); }
    .gradient-peach { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); box-shadow: 0 5px 15px rgba(252, 182, 159, 0.3); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 数据库规模统计")

    # 计算基础统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    total_features = len(df.columns)
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    complete_samples = len(df[df.notna().all(axis=1)])
    missing_samples = len(df) - complete_samples

    # 第一行：6个核心指标（横向排列）
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="compact-metric">
            <div class="compact-label">📊 总样本数</div>
            <div class="compact-number">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="compact-metric gradient-pink">
            <div class="compact-label">🔢 数值特征</div>
            <div class="compact-number">{len(numeric_cols)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="compact-metric gradient-blue">
            <div class="compact-label">📝 文本特征</div>
            <div class="compact-number">{len(text_cols)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="compact-metric gradient-green">
            <div class="compact-label">✅ 完整样本</div>
            <div class="compact-number">{complete_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="compact-metric gradient-orange">
            <div class="compact-label">⚠️ 含缺失</div>
            <div class="compact-number">{missing_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="compact-metric gradient-purple">
            <div class="compact-label">✅ 完整度</div>
            <div class="compact-number">{completeness:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第二行：特征详细表格 + 3个可视化图表（横向排列）
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown("### 📋 特征详细信息")

        # 创建特征统计表
        feature_stats = []
        display_limit = min(10, len(df.columns))  # 最多显示10个特征

        for col in df.columns[:display_limit]:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            unique = df[col].nunique()

            if df[col].dtype in [np.number]:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    range_str = f"{min_val:.1f}~{max_val:.1f}"
                    mean_str = f"{mean_val:.1f}"
                except:
                    range_str = "-"
                    mean_str = "-"
            else:
                range_str = "-"
                mean_str = "-"

            feature_stats.append({
                "特征": col[:15] + "..." if len(col) > 15 else col,
                "类型": dtype.replace('float64', 'num').replace('int64', 'int').replace('object', 'txt'),
                "唯一": unique,
                "缺失": f"{missing_pct:.0f}%",
                "范围": range_str,
            })

        feature_df = pd.DataFrame(feature_stats)
        st.dataframe(feature_df, use_container_width=True, height=280, hide_index=True)

        if len(df.columns) > display_limit:
            st.caption(f"显示前 {display_limit}/{len(df.columns)} 个特征")

    with col2:
        st.markdown("### 📊 类型分布")

        # 特征类型饼图
        type_counts = {
            '数值': len(numeric_cols),
            '文本': len(text_cols),
        }

        fig1 = go.Figure(data=[go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.5,
            marker=dict(colors=['#667eea', '#f5576c']),
            textinfo='label+value',
            textfont=dict(size=12),
        )])

        fig1.update_layout(
            height=280,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col3:
        st.markdown("### 📈 数据质量")

        # 数据质量柱状图
        quality_data = {
            '完整': complete_samples,
            '缺失': missing_samples,
            '重复': duplicate_count
        }

        fig2 = go.Figure(data=[go.Bar(
            x=list(quality_data.keys()),
            y=list(quality_data.values()),
            marker=dict(color=['#43e97b', '#f5576c', '#ffa502']),
            text=[f'{v:,}' for v in quality_data.values()],
            textposition='outside',
            textfont=dict(size=10)
        )])

        fig2.update_layout(
            height=280,
            showlegend=False,
            margin=dict(l=30, r=30, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='lightgray', range=[0, max(quality_data.values()) * 1.2]),
            xaxis=dict(tickfont=dict(size=10))
        )

        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.markdown("### 🎯 目标变量")

        if st.session_state.get('target_col'):
            target_col = st.session_state.target_col
            if target_col in df.columns and df[target_col].dtype in [np.number]:
                target_data = df[target_col].dropna()

                # 箱线图
                fig3 = go.Figure()
                fig3.add_trace(go.Box(
                    y=target_data,
                    marker=dict(color='#667eea'),
                    name=target_col[:10],
                    boxmean='sd'
                ))

                fig3.update_layout(
                    height=280,
                    showlegend=False,
                    margin=dict(l=30, r=30, t=10, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='lightgray', title=target_col[:10])
                )

                st.plotly_chart(fig3, use_container_width=True)

                # 统计信息
                st.markdown(f"""
                <div style="font-size: 11px; text-align: center; margin-top: -10px;">
                    <b>范围:</b> {target_data.min():.1f} ~ {target_data.max():.1f}<br>
                    <b>均值:</b> {target_data.mean():.1f} ± {target_data.std():.1f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("未选择数值型目标变量")
        else:
            st.info("未选择目标变量")

    st.markdown("<br>", unsafe_allow_html=True)

    # 导出按钮
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("📥 导出统计报告", use_container_width=True):
            stats_report = {
                "数据库概览": {
                    "总样本数": int(len(df)),
                    "总特征数": int(len(df.columns)),
                    "数值特征数": int(len(numeric_cols)),
                    "文本特征数": int(len(text_cols)),
                    "数据完整度(%)": float(f"{completeness:.2f}"),
                },
                "数据质量": {
                    "完整样本数": int(complete_samples),
                    "含缺失样本数": int(missing_samples),
                    "缺失值总数": int(missing_count),
                    "重复行数": int(duplicate_count),
                },
                "存储信息": {
                    "内存占用(MB)": float(f"{data_size_mb:.2f}"),
                },
            }

            if st.session_state.get('target_col') and st.session_state.target_col in df.columns:
                target_col = st.session_state.target_col
                if df[target_col].dtype in [np.number]:
                    target_data = df[target_col].dropna()
                    stats_report["目标变量统计"] = {
                        "变量名": target_col,
                        "最小值": float(target_data.min()),
                        "最大值": float(target_data.max()),
                        "均值": float(target_data.mean()),
                        "标准差": float(target_data.std()),
                    }

            stats_json = json.dumps(stats_report, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 下载 JSON",
                stats_json,
                "database_stats_compact.json",
                "application/json",
                use_container_width=True
            )
