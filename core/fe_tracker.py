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


def render_data_export_panel(data: pd.DataFrame = None, tracker: Optional[FeatureEngineeringTracker] = None, key_prefix: str = ""):
    """数据导出面板"""
    if data is None or (hasattr(data, 'empty') and data.empty):
        st.warning("⚠️ 没有可导出的数据")
        return
    
    st.markdown("### 📥 数据导出")
    
    with st.expander("📋 数据预览", expanded=False):
        st.dataframe(data.head(10), use_container_width=True)
        st.caption(f"共 {data.shape[0]} 行 × {data.shape[1]} 列")
    
    col1, col2 = st.columns(2)
    with col1:
        fmt = st.selectbox("导出格式", ["CSV", "Excel (.xlsx)", "JSON"], key=f"{key_prefix}export_fmt")
    with col2:
        idx = st.checkbox("包含索引", value=False, key=f"{key_prefix}export_idx")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if fmt == "CSV":
        st.download_button("⬇️ 下载 CSV", data.to_csv(index=idx).encode('utf-8-sig'),
                          f"data_{ts}.csv", "text/csv", type="primary")
    elif fmt == "Excel (.xlsx)":
        buf = io.BytesIO()
        try:
            data.to_excel(buf, index=idx, engine='openpyxl')
            st.download_button("⬇️ 下载 Excel", buf.getvalue(),
                              f"data_{ts}.xlsx", 
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              type="primary")
        except ImportError:
            st.error("需要安装 openpyxl")
    else:
        st.download_button("⬇️ 下载 JSON", 
                          data.to_json(orient='records', force_ascii=False, indent=2),
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
