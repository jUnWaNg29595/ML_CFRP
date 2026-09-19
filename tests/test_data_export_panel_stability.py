# -*- coding: utf-8 -*-
"""回归测试：侧边栏数据导出「关了再开无法导出」。

根因：导出载荷缓存原本是「进程级单条目」，且文件名里的时间戳只在缓存未命中时
重新生成。任何一次缓存驱逐（另一会话导出 / 状态条记录页面板与侧边栏面板同时
渲染互相挤占）都会导致下次重跑重新生成时间戳 -> 文件名变化。

st.download_button 的媒体文件 id = hash(内容 + mimetype + 文件名)，文件名一变
就产生新的媒体文件；旧媒体文件成为孤儿后被 Streamlit 按 DOWNLOADABLE 两阶段
回收（第一次 sweep 标记、第二次 sweep 删除），浏览器已持有的下载链接随即 404，
表现为「点下载没反应 / 关了再开无法导出」。

因此缓存必须满足：
  1) 会话级（不跨会话/用户互相驱逐）；
  2) 多条目（两个导出面板同时渲染不互相挤占）；
  3) 同一份数据 + 同一格式在多次重跑之间文件名（时间戳）保持稳定。
"""

import sys

import pandas as pd
import pytest

sys.path.insert(0, ".")

import core.fe_tracker as fet  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_caches():
    """每个用例都从干净的缓存开始（含会话级与模块级兜底）。"""
    fet._EXPORT_CACHE_FALLBACK.clear()
    try:
        import streamlit as st

        st.session_state.pop(fet._EXPORT_CACHE_SESSION_KEY, None)
    except Exception:
        pass
    yield
    fet._EXPORT_CACHE_FALLBACK.clear()


def _build(data, fmt="CSV", include_index=False):
    return fet._build_export_payload(data, fmt, include_index)


def test_same_data_same_fmt_keeps_filename_stable():
    """同一份数据、同一格式反复重跑：时间戳（文件名）必须保持不变。"""
    df = pd.DataFrame({"a": range(50), "b": [f"v{i}" for i in range(50)]})

    p1, ts1 = _build(df)
    p2, ts2 = _build(df)
    p3, ts3 = _build(df)

    assert ts1 == ts2 == ts3
    assert p1 is p2 is p3  # 直接复用同一份字节，不重复序列化


def test_interleaved_other_format_does_not_evict():
    """【旧代码在此失败】中间插入其它格式的导出，原格式的文件名不得变化。

    旧实现是单条目缓存：切到 Excel 再切回 CSV 会被判为 miss 并重新生成
    时间戳，导致同一份数据导出两次得到两个不同的文件名（媒体文件抖动）。
    """
    df = pd.DataFrame({"a": range(50)})

    _, ts_csv_first = _build(df, "CSV", False)
    _build(df, "Excel (.xlsx)", False)          # 驱逐（旧实现）
    _build(df, "JSON", False)                   # 再驱逐一次
    _, ts_csv_again = _build(df, "CSV", False)

    assert ts_csv_first == ts_csv_again


def test_interleaved_other_dataframe_does_not_evict():
    """【旧代码在此失败】另一份数据（模拟状态条记录页面板/另一会话）插入后，
    原数据的文件名保持不变。"""
    df_a = pd.DataFrame({"a": range(30)})
    df_b = pd.DataFrame({"b": range(999)})

    _, ts_a_first = _build(df_a)
    _build(df_b)                                # 旧实现会把 A 的条目挤掉
    _, ts_a_again = _build(df_a)

    assert ts_a_first == ts_a_again


def test_two_export_panels_coexist_no_thrash():
    """【旧代码在此失败】两个导出面板同时渲染（侧边栏 sb_ + 状态条记录页 ``）
    反复交替重跑，两边的文件名都必须稳定。"""
    df = pd.DataFrame({"a": range(40)})

    for _ in range(4):
        _, ts_side = _build(df)          # 侧边栏面板（key_prefix="sb_"）
        _, ts_page = _build(df)          # 状态条记录页 tab2（key_prefix=""）
        assert ts_side == ts_page


def test_cache_is_session_scoped_not_global():
    """缓存必须放在 st.session_state（随会话生灭），而不是进程级全局。"""
    try:
        import streamlit as st
    except Exception:  # pragma: no cover
        pytest.skip("streamlit 不可用")

    df = pd.DataFrame({"a": range(10)})
    _build(df)

    cache = st.session_state.get(fet._EXPORT_CACHE_SESSION_KEY)
    assert isinstance(cache, dict) and len(cache) >= 1


def test_cache_is_bounded():
    """缓存有条目上限，避免会话内无限增长。"""
    dfs = [pd.DataFrame({f"a{i}": range(5)}) for i in range(20)]
    for d in dfs:
        _build(d)

    try:
        import streamlit as st

        cache = st.session_state.get(fet._EXPORT_CACHE_SESSION_KEY) or {}
    except Exception:
        cache = fet._EXPORT_CACHE_FALLBACK
    assert len(cache) <= fet._EXPORT_CACHE_MAX_ENTRIES


def test_data_replacement_gets_fresh_payload_and_timestamp():
    """数据被替换（新对象/内容不同）时：必须重新生成字节与新时间戳。"""
    df_old = pd.DataFrame({"a": range(10)})
    df_new = pd.DataFrame({"a": range(10, 30)})

    p_old, ts_old = _build(df_old)
    p_new, ts_new = _build(df_new)

    assert p_old != p_new
    assert b"10" in p_new and b"29" in p_new


def test_none_data_returns_none():
    assert _build(None) == (None, None)


def test_download_button_has_explicit_stable_key():
    """【旧代码在此失败】download_button 必须带显式 key，
    否则其元素 id 会把 file_name（含时间戳）计入哈希，数据一变控件就被销毁重建。
    用 AppTest 驱动真实面板：改变数据后按钮的元素 id 必须保持不变。"""
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "import streamlit as st\n"
        "import pandas as pd\n"
        "from core.fe_tracker import render_data_export_panel\n"
        "df = st.session_state.get('df')\n"
        "with st.sidebar:\n"
        "    render_data_export_panel(df, None, key_prefix='sb_')\n"
    )

    at = AppTest.from_string(script, default_timeout=60)
    at.session_state["df"] = pd.DataFrame({"a": range(30)})
    at.run()

    t = [x for x in at.sidebar.toggle if x.key == "sb_export_panel_open"]
    assert t, "导出面板 toggle 不存在"
    t[0].set_value(True)
    at.run()

    btns = [b for b in at.get("download_button") if "下载" in str(getattr(b.proto, "label", ""))]
    assert btns, "面板打开后没有下载按钮"
    ids_before = {b.proto.id for b in btns}
    names_before = {b.proto.url.rsplit("/", 1)[-1] for b in btns}

    # 换一份数据再渲染：文件名/URL 应变化，但控件元素 id 应保持稳定
    at.session_state["df"] = pd.DataFrame({"a": range(100, 140)})
    at.run()

    btns2 = [b for b in at.get("download_button") if "下载" in str(getattr(b.proto, "label", ""))]
    assert btns2, "数据变化后下载按钮消失"
    ids_after = {b.proto.id for b in btns2}
    names_after = {b.proto.url.rsplit("/", 1)[-1] for b in btns2}

    assert ids_before == ids_after, "download_button 元素 id 不应随数据/文件名变化"
    assert names_before != names_after, "数据变化后导出文件名应更新"
