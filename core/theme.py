# -*- coding: utf-8 -*-
"""科研风格 UI 主题配置。

提供 CSS 变量定义和全局样式，通过 Streamlit 的 st.markdown 注入。
"""

# ─────────────────────────────────────────────────────────────
# CSS 变量定义
# ─────────────────────────────────────────────────────────────

CSS_VARIABLES = """
:root {
  /* 主色调 */
  --primary-blue: #2563EB;
  --primary-cyan: #0891B2;
  
  /* 辅助色 */
  --light-blue: #DBEAFE;
  --light-cyan: #CFFAFE;
  
  /* 背景色 */
  --page-bg: #FFFFFF;
  --card-bg: #FFFFFF;
  --sidebar-bg: #F8FAFC;
  
  /* 文字色 */
  --text-primary: #1E293B;
  --text-secondary: #64748B;
  --text-muted: #94A3B8;
  
  /* 状态色 */
  --success: #059669;
  --warning: #D97706;
  --error: #DC2626;
  --info: #0284C7;
  
  /* 边框色 */
  --border-default: #E2E8F0;
  --border-focus: #2563EB;
  
  /* 字体 */
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "Consolas", monospace;
  
  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
}
"""

# ─────────────────────────────────────────────────────────────
# 全局样式
# ─────────────────────────────────────────────────────────────

GLOBAL_STYLES = """
/* 导入 Inter 字体 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* 全局字体 */
html, body, [class*="css"] {
  font-family: var(--font-sans);
  color: var(--text-primary);
}

/* 页面背景 */
.stApp {
  background-color: var(--page-bg);
}

/* 主按钮样式 */
.stButton > button[kind="primary"] {
  background-color: var(--primary-blue) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 8px 16px !important;
  font-weight: 500 !important;
  transition: background-color 0.2s ease !important;
}

.stButton > button[kind="primary"]:hover {
  background-color: #1D4ED8 !important;
}

/* 次要按钮 */
.stButton > button[kind="secondary"] {
  background-color: var(--light-blue) !important;
  color: var(--primary-blue) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-md) !important;
}

.stButton > button[kind="secondary"]:hover {
  background-color: #BFDBFE !important;
}

/* 卡片容器 */
.card-container {
  background: var(--card-bg);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 16px;
  margin-bottom: 16px;
}

/* 数据卡片 */
.data-card {
  text-align: center;
  padding: 24px;
  background: var(--card-bg);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
}

.data-card .value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary-blue);
}

.data-card .label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* 表格样式 */
.stDataFrame {
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
}

/* 表头 */
[data-testid="stDataFrame"] thead th {
  background-color: #F1F5F9 !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  border-bottom: 2px solid var(--border-default) !important;
}

/* 交替行背景 */
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
  background-color: #F8FAFC !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
  background-color: var(--light-blue) !important;
}

/* 侧边栏样式 */
section[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg) !important;
}

section[data-testid="stSidebar"] .stRadio > label {
  color: var(--text-primary);
}

/* Expander 样式 */
.stExpander {
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}

.stExpander header {
  background-color: transparent !important;
  font-weight: 500 !important;
}

/* 标题样式 */
h1 {
  color: var(--text-primary) !important;
  font-weight: 700 !important;
}

h2 {
  color: var(--text-primary) !important;
  font-weight: 600 !important;
}

h3 {
  color: var(--text-primary) !important;
  font-weight: 600 !important;
}

/* 说明文字 */
.stCaption, .stMarkdown small {
  color: var(--text-secondary) !important;
}

/* 成功/警告/错误消息 */
.element-container .stSuccess {
  background-color: #D1FAE5 !important;
  border-left: 4px solid var(--success) !important;
  border-radius: var(--radius-md) !important;
}

.element-container .stWarning {
  background-color: #FEF3C7 !important;
  border-left: 4px solid var(--warning) !important;
  border-radius: var(--radius-md) !important;
}

.element-container .stError {
  background-color: #FEE2E2 !important;
  border-left: 4px solid var(--error) !important;
  border-radius: var(--radius-md) !important;
}

.element-container .stInfo {
  background-color: #DBEAFE !important;
  border-left: 4px solid var(--info) !important;
  border-radius: var(--radius-md) !important;
}

/* 分隔线 */
hr {
  border-color: var(--border-default) !important;
}

/* 输入框 */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-md) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--primary-blue) !important;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}

/* 进度条 */
.stProgress > div > div > div {
  background-color: var(--primary-blue) !important;
}

/* Radio 按钮 */
.stRadio > div {
  gap: 4px !important;
}

/* Metric 卡片 */
[data-testid="stMetric"] {
  background-color: var(--card-bg) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-lg) !important;
  padding: 16px !important;
}

[data-testid="stMetric"] label {
  color: var(--text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--primary-blue) !important;
}

/* 深色模式切换占位 */
.theme-toggle {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 999;
}
"""


# ─────────────────────────────────────────────────────────────
# 注入函数
# ─────────────────────────────────────────────────────────────

def inject_theme():
    """注入科研风格主题到 Streamlit 应用。"""
    import streamlit as st
    css = CSS_VARIABLES + GLOBAL_STYLES
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_status_color(status: str) -> str:
    """根据状态返回对应的颜色值。
    
    Args:
        status: 状态名称（success/warning/error/info）
    
    Returns:
        对应的颜色值
    """
    colors = {
        "success": "#059669",
        "warning": "#D97706",
        "error": "#DC2626",
        "info": "#0284C7",
    }
    return colors.get(status.lower(), "#64748B")


def format_number(value, decimals: int = 2) -> str:
    """格式化数字，添加千分位分隔符。
    
    Args:
        value: 数值
        decimals: 小数位数
    
    Returns:
        格式化后的字符串
    """
    if value is None:
        return "-"
    try:
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:,.{decimals}f}"
            else:
                return f"{value:,}"
        return str(value)
    except Exception:
        return str(value)


# ─────────────────────────────────────────────────────────────
# 智能表格增强样式
# ─────────────────────────────────────────────────────────────

SMART_TABLE_STYLES = """
/* 智能表格增强 */
.stDataFrame thead th {
  cursor: pointer !important;
  user-select: none !important;
  transition: background-color 0.15s ease !important;
}

.stDataFrame thead th:hover {
  background-color: #E2E8F0 !important;
}

/* 数值列右对齐 */
.stDataFrame td[data-testid*="numeric"],
.stDataFrame td[align="right"] {
  text-align: right !important;
  font-family: var(--font-mono) !important;
}

/* 关键列高亮 */
.stDataFrame td.highlight-value {
  font-weight: 600 !important;
  color: var(--primary-blue) !important;
}

/* 异常值标记 */
.stDataFrame td.anomaly-value {
  color: var(--error) !important;
}

/* 选中行 */
.stDataFrame tbody tr.selected {
  background-color: var(--light-blue) !important;
  box-shadow: inset 0 0 0 1px var(--primary-blue) !important;
}

/* 排序指示器 */
.sort-indicator {
  margin-left: 4px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* 分页样式 */
.pagination-container {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-top: 16px;
}

.pagination-btn {
  padding: 4px 12px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--card-bg);
  cursor: pointer;
  transition: all 0.15s ease;
}

.pagination-btn:hover {
  background: var(--light-blue);
  border-color: var(--primary-blue);
}

.pagination-btn.active {
  background: var(--primary-blue);
  color: white;
  border-color: var(--primary-blue);
}
"""


def inject_smart_table_styles():
    """注入智能表格增强样式。"""
    import streamlit as st
    st.markdown(f"<style>{SMART_TABLE_STYLES}</style>", unsafe_allow_html=True)
