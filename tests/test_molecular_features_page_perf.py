# -*- coding: utf-8 -*-
"""分子特征页首屏性能回归 + 按需面板功能回归。

背景（2026-09-18 实测）：
  app_lib.page_molecular_features 首屏中位数 1.01s，其中 98% 来自两个「折叠态」
  重面板 —— Streamlit 的 st.expander 无论是否展开都会执行内部代码，而
  core/formulation_fusion_ui 每次执行都会重读 20MB 母宽表 ml_wide_samples.csv
  （单次 ~0.7~1.1s）。改为 render_lazy_panel（点击后才构建）后首屏降至 0.016s。

本文件锁定两点：
  1. 首屏不得读母宽表，且首屏耗时保持在阈值内；
  2. 面板展开后必须照常构建（功能不退化），融合控件可见。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

WIDE_TABLE = r"C:\Users\wangj\Desktop\ml_dataset\ml_wide_samples.csv"

PAGE_SCRIPT = (
    "import sys\n"
    "import time\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import streamlit as st\n"
    "import core.formulation_fusion_ui as _ffu\n"
    "from app_lib import page_molecular_features, init_session_state\n"
    "init_session_state()\n"
    "n = 40\n"
    "rng = np.random.default_rng(7)\n"
    "st.session_state.data = pd.DataFrame({\n"
    "    'sample_id': [f'S{i:04d}' for i in range(n)],\n"
    "    'resin_smiles': ['CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'] * n,\n"
    "    'hardener_smiles': ['Nc1ccc(Cc2ccc(N)cc2)cc1'] * n,\n"
    "    'tg': rng.normal(150, 20, n).round(2),\n"
    "})\n"
    "st.session_state.processed_data = st.session_state.data\n"
    # 记录本页执行期间所有 CSV 读盘（含缓存穿透后的真实读盘）
    "_reads = []\n"
    "_orig = pd.read_csv\n"
    "def _traced(*a, **kw):\n"
    "    _reads.append(str(a[0] if a else kw.get('filepath_or_buffer'))[:200])\n"
    "    return _orig(*a, **kw)\n"
    "pd.read_csv = _traced\n"
    "_t0 = time.perf_counter()\n"
    "page_molecular_features()\n"
    "st.session_state['_page_seconds'] = time.perf_counter() - _t0\n"
    "st.session_state['_csv_reads'] = _reads\n"
)


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """st.cache_data 是进程级的，避免跨用例互相命中。"""
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _open_page(timeout: float = 900.0) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_first_render_does_not_read_wide_master_table():
    """折叠态面板不得触发母宽表读盘（这是首屏卡顿的直接原因）。"""
    at = _open_page()
    reads = list(at.session_state["_csv_reads"])
    wide_reads = [r for r in reads if "ml_wide_samples" in r]
    assert wide_reads == [], f"首屏不应读取母宽表，实际读了: {wide_reads}"


def test_first_render_stays_within_budget():
    """首屏应在 0.5s 内完成（修复前实测中位数 1.01s）。"""
    at = _open_page()
    elapsed = float(at.session_state["_page_seconds"])
    assert elapsed < 0.5, f"首屏 {elapsed:.3f}s 超出预算 0.5s"


def test_fusion_panel_builds_only_after_expanding():
    """展开后才构建融合面板：控件出现，且此时才允许读母宽表。"""
    at = _open_page()

    toggles = {t.key: t for t in at.toggle}
    assert "mf_fusion_panel_open" in toggles, toggles.keys()

    # 折叠态：融合面板的控件不应存在
    labels_before = "\n".join(str(getattr(el, "label", "")) for el in at.selectbox)
    assert "选择目标性能文件" not in labels_before

    at.toggle(key="mf_fusion_panel_open").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]

    labels_after = "\n".join(str(getattr(el, "label", "")) for el in at.selectbox)
    assert "选择目标性能文件" in labels_after, labels_after
    assert "选择配方母宽表" in labels_after, labels_after

    reads = list(at.session_state["_csv_reads"])
    assert any("ml_wide_samples" in r for r in reads), "展开后应加载母宽表"


def test_physics_panel_builds_only_after_expanding():
    """高分子物理指数面板同样按需构建。"""
    at = _open_page()

    before = "\n".join(str(getattr(el, "label", "")) for el in at.text_input)
    assert "SMILES / BigSMILES" not in before

    at.toggle(key="mf_physics_panel_open").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]

    after = "\n".join(str(getattr(el, "label", "")) for el in at.text_input)
    assert "SMILES / BigSMILES" in after, after


def test_collapsing_again_skips_panel_build():
    """再次折叠后应恢复零成本，不读母宽表。"""
    at = _open_page()
    at.toggle(key="mf_fusion_panel_open").set_value(True).run()
    at.toggle(key="mf_fusion_panel_open").set_value(False).run()
    assert not at.exception, [e.value for e in at.exception]

    reads = list(at.session_state["_csv_reads"])
    assert reads == [], f"折叠态不应有任何读盘，实际: {reads}"


def test_read_csv_cached_hits_cache_on_second_call():
    """指纹未变时第二次读取不再落盘（这才是真正的性能收益）。"""
    from core import formulation_fusion_ui as ui

    target = WIDE_TABLE
    if not Path(target).exists():
        pytest.skip(f"母宽表不存在: {target}")

    import streamlit as st

    st.cache_data.clear()

    calls = {"n": 0}
    original = pd.read_csv

    def _counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    pd.read_csv = _counting
    try:
        first = ui.read_csv_cached(target)
        second = ui.read_csv_cached(target)
    finally:
        pd.read_csv = original

    assert calls["n"] == 1, f"应只读盘一次，实际 {calls['n']} 次"
    assert first.shape == second.shape
    pd.testing.assert_frame_equal(first, second)


def test_read_csv_cached_invalidates_when_file_changes(tmp_path):
    """文件被改写后指纹变化，缓存必须失效（否则会拿到旧数据）。"""
    from core import formulation_fusion_ui as ui

    import streamlit as st

    st.cache_data.clear()

    path = tmp_path / "wide.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    first = ui.read_csv_cached(str(path))
    assert list(first["a"]) == [1, 2, 3]

    pd.DataFrame({"a": [9, 8, 7]}).to_csv(path, index=False)
    second = ui.read_csv_cached(str(path))
    assert list(second["a"]) == [9, 8, 7], "改写文件后必须重新读盘"


def test_render_lazy_panel_returns_build_flag():
    """render_lazy_panel 的返回值语义：构建了才返回 True。"""
    import app_lib

    calls = {"n": 0}

    def _builder():
        calls["n"] += 1

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import streamlit as st\n"
        "from app_lib import render_lazy_panel\n"
        "built = render_lazy_panel('panel', key='unit_test_panel', builder=lambda: None)\n"
        "st.session_state['_built'] = built\n"
    )
    at = AppTest.from_string(script, default_timeout=120.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_built"] is False

    at.toggle(key="unit_test_panel_open").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_built"] is True
    assert calls["n"] == 0  # 上面的 lambda 与本地 _builder 无关，仅确认无副作用


# =============================================================================
# 交互卡顿回归（2026-09-18 第二批）
#
# 现象：分子特征提取完成后（宽表 + 上千个指纹列），点击任意选择框/复选框都卡。
# 根因：_render_extracted_features_panel 里 `st.dataframe(features_df.head(20))`
#   把全部列转成 Arrow 并在**每次交互 rerun** 时重发给浏览器：
#   实测 2000 列 → 801KB payload / 91ms 序列化（6474×3000 时达 1.2MB）。
# 修复：render_capped_preview 默认只渲染前 40 列，其余列走按需展开的选择器。
# =============================================================================

INTERACTION_SCRIPT = (
    "import sys, time\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import streamlit as st\n"
    "import streamlit.runtime.scriptrunner_utils.script_run_context as SRC\n"
    "_msgs = []\n"
    "_orig = SRC.ScriptRunContext.enqueue\n"
    "def _patched(self, msg, *a, **kw):\n"
    "    try:\n"
    "        _msgs.append(msg.ByteSize())\n"
    "    except Exception:\n"
    "        pass\n"
    "    return _orig(self, msg, *a, **kw)\n"
    "SRC.ScriptRunContext.enqueue = _patched\n"
    "from app_lib import page_molecular_features, init_session_state\n"
    "if '_inited' not in st.session_state:\n"
    "    init_session_state()\n"
    "    st.session_state['_inited'] = True\n"
    "if '_setup' not in st.session_state:\n"
    "    n = int(st.session_state.get('_rows', 3000))\n"
    "    nfeat = int(st.session_state.get('_feats', 2000))\n"
    "    rng = np.random.default_rng(7)\n"
    "    df = pd.DataFrame({\n"
    "        'sample_id': [f'S{i:05d}' for i in range(n)],\n"
    "        'resin_smiles': ['CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'] * n,\n"
    "        'hardener_smiles': ['Nc1ccc(Cc2ccc(N)cc2)cc1'] * n,\n"
    "        'tg': rng.normal(150, 20, n).round(2),\n"
    "    })\n"
    "    if nfeat:\n"
    "        block = pd.DataFrame(\n"
    "            rng.normal(0, 1, (n, nfeat)).round(4),\n"
    "            columns=[f'resin_smiles_fp_{i}' for i in range(nfeat)],\n"
    "        )\n"
    "        df = pd.concat([df, block], axis=1)\n"
    "    feat = df[[c for c in df.columns if c.startswith('resin_smiles_fp_')]]\n"
    "    st.session_state.data = df\n"
    "    st.session_state.processed_data = df\n"
    "    st.session_state.molecular_features = feat\n"
    "    st.session_state.molecular_feature_names = list(feat.columns)\n"
    "    st.session_state.latest_extracted_features_df = feat\n"
    "    st.session_state.latest_extraction_summary = {\n"
    "        'n_samples': n, 'n_features': nfeat, 'dual': False, 'method': 'single',\n"
    "    }\n"
    "    st.session_state['_setup'] = True\n"
    "_msgs.clear()\n"
    "_t0 = time.perf_counter()\n"
    "page_molecular_features()\n"
    "st.session_state['_last'] = {\n"
    "    'script_s': time.perf_counter() - _t0,\n"
    "    'payload_mb': sum(_msgs) / 1024 / 1024,\n"
    "}\n"
)


def _interaction_app(rows: int = 3000, feats: int = 2000) -> AppTest:
    at = AppTest.from_string(INTERACTION_SCRIPT, default_timeout=1800.0)
    at.session_state["_rows"] = rows
    at.session_state["_feats"] = feats
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_wide_feature_table_does_not_resend_full_payload_on_interaction():
    """核心回归：交互 rerun 不得把上千列特征整份发给浏览器。

    修复前：2000 列 → 801KB；6474×3000 → 1.2MB。
    修复后：约 31KB（只发前 40 列预览）。
    """
    at = _interaction_app(rows=3000, feats=2000)

    payloads = []
    for i, role in enumerate(["resin", "hardener", "neutral"]):
        at.radio(key="molecular_feature_component_role").set_value(role).run()
        assert not at.exception, [e.value for e in at.exception]
        payloads.append(at.session_state["_last"]["payload_mb"])

    worst = max(payloads)
    assert worst < 0.15, f"交互 rerun 负载 {worst:.3f}MB 过大（修复前为 0.8MB）"


def test_wide_feature_table_interaction_script_stays_fast():
    """交互 rerun 的脚本耗时（修复前中位数 0.87s）。"""
    at = _interaction_app(rows=3000, feats=2000)

    times = []
    for role in ["resin", "hardener", "neutral"]:
        at.radio(key="molecular_feature_component_role").set_value(role).run()
        assert not at.exception, [e.value for e in at.exception]
        times.append(at.session_state["_last"]["script_s"])

    times.sort()
    assert times[len(times) // 2] < 0.3, f"交互脚本中位数 {times[len(times) // 2]:.3f}s 超出预算"


def test_preview_defaults_to_capped_columns():
    """预览默认只渲染前 40 列，其余列通过按需展开的选择器查看。"""
    at = _interaction_app(rows=3000, feats=2000)

    # 未展开时不应存在列选择器（否则上千个列名本身就会拖慢页面）
    assert "mf_extracted_preview_cols" not in [m.key for m in at.multiselect]

    at.toggle(key="mf_extracted_preview_custom_open").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]

    ms = [m for m in at.multiselect if m.key == "mf_extracted_preview_cols"]
    assert ms, "展开后应出现列选择器"
    assert len(list(ms[0].options)) == 2000, "应能访问全部列，而非被截断"
    assert len(ms[0].value) == 40, "默认选中前 40 列"


def test_preview_column_selection_works():
    """自定义列选择必须真的生效（功能不因性能优化而丢失）。"""
    at = _interaction_app(rows=3000, feats=2000)
    at.toggle(key="mf_extracted_preview_custom_open").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]

    ms = [m for m in at.multiselect if m.key == "mf_extracted_preview_cols"][0]
    ms.set_value(["resin_smiles_fp_0", "resin_smiles_fp_1999"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_last"]["payload_mb"] < 0.15
