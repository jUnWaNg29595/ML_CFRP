"""Scientific visual primitives for the standalone prediction portal."""

from __future__ import annotations

from html import escape

TOKENS = {
    'navy': '#071a2f', 'navy_surface': '#0d2742', 'cyan': '#22d3ee',
    'blue': '#38bdf8', 'green': '#34d399', 'amber': '#fbbf24',
    'red': '#fb7185', 'slate': '#94a3b8',
}

_ICONS = {
    'molecule': '<circle cx="12" cy="5" r="2.3"/><circle cx="5" cy="16" r="2.3"/><circle cx="19" cy="16" r="2.3"/><path d="m10.2 6.4-3.6 7.2m6.9-7.2 3.6 7.2M7.3 16h9.4"/>',
    'resin': '<path d="M8 4h8m-7 0v4l-3 4v6h8l3-6-3-4V4M6 18h12"/>',
    'hardener': '<path d="M7 4h10M9 4v5l-3 7a2 2 0 0 0 1.8 2.8h8.4A2 2 0 0 0 18 16l-3-7V4M8 14h8"/>',
    'carbon_fiber': '<path d="m5 5 14 14M19 5 5 19M4 9h16M4 15h16M9 4v16M15 4v16"/>',
    'input': '<path d="M12 3v11m0 0-4-4m4 4 4-4M5 19h14"/>',
    'validation': '<path d="m5 12 4 4L19 6"/>',
    'features': '<path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5zM14 14h5v5h-5zM10 7.5h4M7.5 10v4M16.5 10v4M10 16.5h4"/>',
    'calculation': '<path d="M6 3h12v18H6zM9 7h6M9 11h1m4 0h1M9 15h1m4 0h1M9 19h1m4 0h1"/>',
    'result': '<path d="M4 19V5m0 14h16M8 16v-5m4 5V7m4 9v-8"/>',
    'ai': '<path d="M12 3 13.8 8.2 19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Zm6 13 .7 2.3L21 19l-2.3.7L18 22l-.7-2.3L15 19l2.3-.7L18 16Z"/>',
    'download': '<path d="M12 3v11m0 0-4-4m4 4 4-4M5 20h14"/>',
    'pause': '<path d="M8 5v14M16 5v14"/>',
    'cancel': '<path d="m6 6 12 12M18 6 6 18"/>',
    'retry': '<path d="M20 11a8 8 0 0 0-14.7-4L4 9m0-5v5h5M4 13a8 8 0 0 0 14.7 4L20 15m0 5v-5h-5"/>',
}

_STAGES = [
    ('validated', '输入确认'), ('structure', '结构校验'), ('workflow', '特征准备'),
    ('predicting', '模型计算'), ('explaining', '结果解释'),
]


def svg_icon(name: str, size: int = 20, label: str | None = None) -> str:
    body = _ICONS.get(name, _ICONS['molecule'])
    aria = f' aria-label="{escape(label)}" role="img"' if label else ' aria-hidden="true"'
    return f'<svg class="portal-icon" width="{int(size)}" height="{int(size)}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"{aria}>{body}</svg>'


def render_status_badge(status: str) -> str:
    normalized = str(status or 'unknown').strip().lower()
    labels = {'queued': '排队中', 'validating': '校验中', 'featuring': '特征准备中',
              'predicting': '模型计算中', 'explaining': '结果解释中', 'completed': '已完成',
              'failed': '失败', 'cancelled': '已取消', 'published': '已发布',
              'enabled': '已启用', 'unknown': '未知'}
    tone = 'success' if normalized in {'completed', 'published', 'enabled'} else 'danger' if normalized in {'failed', 'cancelled'} else 'active'
    return f'<span class="portal-status-badge {tone}"><span class="portal-status-dot"></span>{escape(labels.get(normalized, normalized))}</span>'


def render_stage_timeline(stage: str, progress: int | float = 0) -> str:
    current = str(stage or '').lower()
    try:
        progress_value = max(0, min(100, int(float(progress))))
    except (TypeError, ValueError):
        progress_value = 0
    current_index = next((index for index, (key, _) in enumerate(_STAGES) if key == current), -1)
    parts = []
    for index, (key, label) in enumerate(_STAGES):
        state = 'done' if current_index >= 0 and index < current_index else 'active' if key == current else 'pending'
        parts.append(f'<div class="portal-stage {state}"><span class="portal-stage-marker">{index + 1}</span><span>{escape(label)}</span></div>')
    return f'<div class="portal-stage-wrap" data-progress="{progress_value}">{"".join(parts)}</div>'


def render_material_card(material_key: str, label: str, description: str = '', status: str = 'enabled', icon: str = 'molecule') -> str:
    return (f'<article class="portal-material-card" data-material="{escape(str(material_key))}">'
            f'<div class="portal-card-icon">{svg_icon(icon, 24, label)}</div>'
            f'<div class="portal-card-body"><div class="portal-card-heading"><h3>{escape(label)}</h3>{render_status_badge(status)}</div>'
            f'<p>{escape(description or "")}</p></div></article>')


def inject_scientific_theme() -> None:
    import streamlit as st
    st.markdown('''
<style>
:root { --portal-navy:#071a2f; --portal-surface:#0d2742; --portal-cyan:#22d3ee; --portal-blue:#38bdf8; --portal-green:#34d399; --portal-amber:#fbbf24; --portal-red:#fb7185; --portal-slate:#94a3b8; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 80% 0%, rgba(34,211,238,.08), transparent 35%), linear-gradient(135deg, #071a2f 0%, #0b1f36 46%, #0f2b46 100%); color:#e2e8f0; }
[data-testid="stHeader"] { background:rgba(7,26,47,.72); }
[data-testid="stSidebar"] { background:#061426; border-right:1px solid rgba(148,163,184,.16); }
.portal-shell, .portal-hero, .portal-material-card, .portal-model-card, .portal-status-panel { border:1px solid rgba(148,163,184,.18); border-radius:16px; background:rgba(13,39,66,.72); box-shadow:0 18px 45px rgba(0,0,0,.18); }
.portal-hero { padding:28px 32px; margin-bottom:18px; }
.portal-hero h1, .portal-hero h2, .portal-hero h3 { color:#f8fafc; letter-spacing:-.02em; }
.portal-kicker, .hero-kicker { color:var(--portal-cyan); text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
.portal-icon { display:inline-block; vertical-align:middle; }
.portal-material-card { min-height:130px; padding:20px; display:flex; gap:16px; transition:transform .15s ease, border-color .15s ease; }
.portal-material-card:hover { transform:translateY(-2px); border-color:rgba(34,211,238,.55); }
.portal-card-icon { color:var(--portal-cyan); flex:0 0 auto; }
.portal-card-heading { display:flex; gap:12px; align-items:center; justify-content:space-between; }
.portal-card-heading h3 { margin:0; font-size:1.05rem; }
.portal-card-body p { color:#a9bacb; margin:.45rem 0 0; font-size:.9rem; line-height:1.55; }
.portal-status-badge { display:inline-flex; align-items:center; gap:6px; color:#cbd5e1; font-size:.75rem; white-space:nowrap; }
.portal-status-dot { width:7px; height:7px; border-radius:50%; background:var(--portal-blue); box-shadow:0 0 0 3px rgba(56,189,248,.12); }
.portal-status-badge.success .portal-status-dot { background:var(--portal-green); box-shadow:0 0 0 3px rgba(52,211,153,.12); }
.portal-status-badge.danger .portal-status-dot { background:var(--portal-red); box-shadow:0 0 0 3px rgba(251,113,133,.12); }
.portal-stage-wrap { display:flex; gap:8px; align-items:flex-start; overflow-x:auto; padding:14px 4px; }
.portal-stage { flex:1 0 120px; display:flex; gap:8px; align-items:center; color:#64748b; font-size:.78rem; }
.portal-stage-marker { width:26px; height:26px; display:grid; place-items:center; border-radius:50%; border:1px solid #334155; color:#64748b; }
.portal-stage.done, .portal-stage.active { color:#dbeafe; }.portal-stage.done .portal-stage-marker { color:#06251d; background:var(--portal-green); border-color:var(--portal-green); }.portal-stage.active .portal-stage-marker { color:#082f49; background:var(--portal-cyan); border-color:var(--portal-cyan); }
.portal-help { color:#94a3b8; font-size:.82rem; }
</style>
''', unsafe_allow_html=True)


__all__ = ['inject_scientific_theme', 'svg_icon', 'render_status_badge', 'render_stage_timeline', 'render_material_card']
