# -*- coding: utf-8 -*-
"""
特征工程状态追踪器 - 完全重写版 v3
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import io


@dataclass
class FeatureEngineeringStep:
    """单个特征工程步骤"""
    step_id: int
    timestamp: str
    operation: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_shape: tuple = (0, 0)
    output_shape: tuple = (0, 0)
    features_added: List[str] = field(default_factory=list)
    features_removed: List[str] = field(default_factory=list)
    status: str = "success"
    message: str = ""


class FeatureEngineeringTracker:
    """特征工程状态追踪器"""
    
    # 数据存储键（带下划线前缀，避免冲突）
    _DATA_KEY = '_fe_tracker_internal_data'
    
    def __init__(self):
        self._init_data()
    
    def _init_data(self):
        """初始化数据存储"""
        if self._DATA_KEY not in st.session_state:
            st.session_state[self._DATA_KEY] = {
                'steps': [],
                'stats': {'success': 0, 'warning': 0, 'error': 0},
                'counter': 0
            }
    
    def _data(self) -> dict:
        """获取内部数据"""
        self._init_data()
        return st.session_state[self._DATA_KEY]
    
    def log_step(
        self,
        operation: str,
        description: str,
        params: Optional[Dict[str, Any]] = None,
        input_df: Optional[pd.DataFrame] = None,
        output_df: Optional[pd.DataFrame] = None,
        features_added: Optional[List[str]] = None,
        features_removed: Optional[List[str]] = None,
        status: str = "success",
        message: str = ""
    ) -> FeatureEngineeringStep:
        """记录一个步骤"""
        d = self._data()
        d['counter'] += 1
        
        in_shape = input_df.shape if input_df is not None else (0, 0)
        out_shape = output_df.shape if output_df is not None else (0, 0)
        
        step = FeatureEngineeringStep(
            step_id=d['counter'],
            timestamp=datetime.now().strftime("%H:%M:%S"),
            operation=operation,
            description=description,
            params=params or {},
            input_shape=in_shape,
            output_shape=out_shape,
            features_added=features_added or [],
            features_removed=features_removed or [],
            status=status,
            message=message
        )
        
        d['steps'].append(asdict(step))
        if status in d['stats']:
            d['stats'][status] += 1
        
        return step
    
    def get_steps(self) -> List[dict]:
        """获取所有步骤"""
        return self._data()['steps']
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计"""
        return self._data()['stats']
    
    def get_last_step(self) -> Optional[dict]:
        """获取最后一步"""
        steps = self.get_steps()
        return steps[-1] if steps else None
    
    def clear(self):
        """清除记录"""
        st.session_state[self._DATA_KEY] = {
            'steps': [],
            'stats': {'success': 0, 'warning': 0, 'error': 0},
            'counter': 0
        }
    
    def export_log_to_json(self) -> str:
        """导出JSON"""
        d = self._data()
        return json.dumps({
            'steps': d['steps'],
            'stats': d['stats'],
            'exported_at': datetime.now().isoformat()
        }, ensure_ascii=False, indent=2)


# ============================================================
# UI 函数
# ============================================================

def render_status_sidebar(tracker: FeatureEngineeringTracker):
    """侧边栏状态显示"""
    if tracker is None:
        return
    
    try:
        steps = tracker.get_steps()
        stats = tracker.get_stats()
    except Exception:
        return
    
    if not steps:
        return
    
    st.markdown("---")
    total = len(steps)
    s_c = stats.get('success', 0)
    w_c = stats.get('warning', 0)
    e_c = stats.get('error', 0)
    # 精简为单行摘要（原三列 metric 在侧边栏占用过大且信息密度低）
    st.caption(f"📋 特征工程: ✅{s_c} ⚠️{w_c} ❌{e_c} · 共 {total} 条")

    recent = steps[-5:][::-1]
    with st.expander("最近操作", expanded=False):
        for s in recent:
            icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(s.get('status', 'success'), "❓")
            st.caption(f"{icon} [{s.get('timestamp', '')}] {s.get('operation', '')}")


def render_status_panel(tracker: FeatureEngineeringTracker):
    """主界面状态面板"""
    if tracker is None:
        st.info("追踪器未初始化")
        return
    
    try:
        steps = tracker.get_steps()
        stats = tracker.get_stats()
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return
    
    st.markdown("## 📋 特征工程操作记录")
    
    if not steps:
        st.info("暂无操作记录")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总操作数", len(steps))
    col2.metric("✅ 成功", stats.get('success', 0))
    col3.metric("⚠️ 警告", stats.get('warning', 0))
    col4.metric("❌ 错误", stats.get('error', 0))
    
    st.markdown("---")
    st.markdown("### 📜 操作时间线")
    
    for step in reversed(steps):
        color = {'success': '🟢', 'warning': '🟡', 'error': '🔴'}.get(step.get('status', 'success'), '⚪')
        st.markdown(f"{color} **{step.get('operation', '未知')}** - {step.get('description', '')}")
        st.caption(f"#{step.get('step_id', '?')} @ {step.get('timestamp', '')}")
        st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 导出日志 (JSON)",
            data=tracker.export_log_to_json(),
            file_name=f"fe_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with col_b:
        if st.button("🗑️ 清除记录"):
            tracker.clear()
            st.rerun()


# 模块级单条目导出缓存：侧边栏导出面板在每次页面重跑时都会渲染，
# 宽表（如 1 万行 × 千余列）to_csv 需数秒，这里对同一 DataFrame+格式做记忆缓存，
# 重跑时直接复用字节，文件名也保持稳定（避免时间戳变化导致下载控件重建重传）。
_EXPORT_PAYLOAD_CACHE = {"key": None, "payload": None, "ts": None, "df_ref": None}


def _build_export_payload(data: pd.DataFrame, fmt: str, include_index: bool):
    """生成导出字节（带单条目记忆缓存）。

    df_ref 持有 DataFrame 强引用，防止对象被回收后 id 被复用导致误命中。
    缓存键除对象 id 外还包含形状/索引端点/末列名等廉价探针，
    以捕捉对同一对象就地增删行/列的修改（同形状同索引的单元格值就地
    编辑属于极端情况，可通过切换格式或重新生成数据来刷新）。
    """
    try:
        probe = (
            data.index[0],
            data.index[-1],
            str(data.columns[-1]),
        )
    except Exception:
        probe = ()
    key = (id(data), data.shape, fmt, include_index, probe)
    cache = _EXPORT_PAYLOAD_CACHE
    if cache["key"] == key and cache["df_ref"] is data:
        return cache["payload"], cache["ts"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = None
    if fmt == "CSV":
        payload = data.to_csv(index=include_index).encode('utf-8-sig')
    elif fmt == "Excel (.xlsx)":
        buf = io.BytesIO()
        try:
            data.to_excel(buf, index=include_index, engine='openpyxl')
            payload = buf.getvalue()
        except ImportError:
            payload = None
    else:
        payload = data.to_json(orient='records', force_ascii=False, indent=2)

    cache.update(key=key, payload=payload, ts=ts, df_ref=data)
    return payload, ts


def render_data_export_panel(data: pd.DataFrame = None, tracker: Optional[FeatureEngineeringTracker] = None, key_prefix: str = ""):
    """数据导出面板

    [性能修复] 本面板常驻侧边栏，任意控件交互触发的整页重跑都会执行它；
    此前即使折叠状态也会对整表执行 to_csv/to_excel（宽表耗时数秒），
    导致所有页面的选择栏操作都卡顿数秒。现在：
    1. 动态 expander（on_change="rerun"）：折叠时完全跳过预览与导出字节生成；
    2. 导出字节带记忆缓存：展开状态下重复重跑直接复用，不重新序列化。
    """
    if data is None or (hasattr(data, 'empty') and data.empty):
        st.warning("⚠️ 没有可导出的数据")
        return

    st.markdown("### 📥 数据导出")

    export_exp = st.expander("📋 导出选项与数据预览", expanded=False, on_change="rerun")
    if not export_exp.open:
        st.caption("💡 展开后选择格式并生成下载（数据较大时生成需要片刻）。")
        return

    with export_exp:
        st.dataframe(data.head(10), width="stretch")
        st.caption(f"共 {data.shape[0]} 行 × {data.shape[1]} 列")

        col1, col2 = st.columns(2)
        with col1:
            fmt = st.selectbox("导出格式", ["CSV", "Excel (.xlsx)", "JSON"], key=f"{key_prefix}export_fmt")
        with col2:
            idx = st.checkbox("包含索引", value=False, key=f"{key_prefix}export_idx")

        payload, ts = _build_export_payload(data, fmt, idx)
        if payload is None:
            if fmt == "Excel (.xlsx)":
                st.error("需要安装 openpyxl")
            return

        if fmt == "CSV":
            st.download_button("⬇️ 下载 CSV", payload,
                              f"data_{ts}.csv", "text/csv", type="primary")
        elif fmt == "Excel (.xlsx)":
            st.download_button("⬇️ 下载 Excel", payload,
                              f"data_{ts}.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              type="primary")
        else:
            st.download_button("⬇️ 下载 JSON", payload,
                              f"data_{ts}.json", "application/json", type="primary")


def create_quick_export_button(data: pd.DataFrame = None, filename_prefix: str = "data", label: str = "📥 快速导出 CSV"):
    """快速导出按钮"""
    if data is None or (hasattr(data, 'empty') and data.empty):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(label, data.to_csv(index=False).encode('utf-8-sig'),
                      f"{filename_prefix}_{ts}.csv", "text/csv")


# 操作类型常量
FE_OPERATION_TYPES = {
    'data_load': '数据加载',
    'missing_value': '缺失值处理',
    'outlier': '异常值处理',
    'duplicate': '重复值处理',
    'type_fix': '数据类型修复',
    'encoding': '特征编码',
    'scaling': '特征缩放',
    'feature_extract': '特征提取',
    'feature_select': '特征选择',
    'smiles_clean': 'SMILES清洗',
    'fingerprint': '分子指纹',
    'descriptor': '分子描述符',
    'balance': '类别平衡',
    'cluster': '聚类分析',
    'export': '数据导出',
}
