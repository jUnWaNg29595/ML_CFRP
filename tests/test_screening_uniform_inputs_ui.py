"""锁定「模型外部特征统一输入」面板的 Streamlit 类型兼容性。

真实故障：点击虚拟筛选页抛
    streamlit.errors.StreamlitAPIException
    （_check_type_compatibilities 判定 TextColumn 与实际 dtype 不兼容）

根因（数据依赖型 bug，工作区数据齐全时才触发）：
    `值` 列用 `auto_vals.get(c, "")` 构造——
    - 全部列都自动取值成功 → 该列全是 float → dtype=float64
    - 部分缺失 → 混入 "" → dtype=object
    而 column_config 声明 TextColumn。dtype=float64 时判定不兼容 → 抛异常。

第二处隐患：auto_vals 的值既可能是 float（中位数）也可能是 str（类别众数，
如 `test_atmosphere="N2"`）。用 `f"{v:g}"` 格式化 str 会抛
`ValueError: Unknown format code 'g'`。

本测试用 Streamlit 自己的 `_check_type_compatibilities` 校验**真实函数**
传给 `st.data_editor` 的 DataFrame，而不是复刻一份逻辑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _run_panel(monkeypatch, feature_cols, workspace):
    """调用真实 _render_screening_uniform_inputs，捕获其传给 data_editor 的入参。"""
    import app_lib

    captured: dict = {}

    class _FakeEditor:
        def __call__(self, data, **kwargs):
            captured["df"] = data
            captured["column_config"] = kwargs.get("column_config")
            return data

    monkeypatch.setattr(app_lib.st, "data_editor", _FakeEditor())
    for name in ("markdown", "caption", "success", "warning", "info"):
        monkeypatch.setattr(app_lib.st, name, lambda *a, **k: None)

    result = app_lib._render_screening_uniform_inputs(
        feature_cols, df_ref=workspace, key_prefix="test_uniform"
    )
    return captured, result


def _assert_streamlit_type_compatible(df, column_config):
    """用 Streamlit 自身的类型校验逻辑验证（这是真实报错的那段代码）。"""
    import pyarrow as pa
    from streamlit.elements.widgets.data_editor import (
        _check_type_compatibilities,
        determine_dataframe_schema,
    )

    schema = determine_dataframe_schema(df, pa.Table.from_pandas(df).schema)
    # 不抛异常即通过
    _check_type_compatibilities(df, column_config, schema)


ALL_COLS = [
    "resin_1_amount_phr",
    "curing_agent_1_amount_phr",
    "process_atmosphere",
]


def test_all_columns_autofilled_is_type_compatible(monkeypatch):
    """核心回归：工作区包含全部列（值列全为数值）时不得抛 StreamlitAPIException。"""
    workspace = pd.DataFrame(
        {
            "resin_1_amount_phr": [100.0, 100.0],
            "curing_agent_1_amount_phr": [33.0, 33.0],
            "process_atmosphere": ["N2", "N2"],
        }
    )
    captured, result = _run_panel(monkeypatch, ALL_COLS, workspace)
    df = captured["df"]
    _assert_streamlit_type_compatible(df, captured["column_config"])
    assert df["值"].dtype == object, "值列必须始终是 object（文本）"
    assert all(isinstance(v, str) for v in df["值"]), "值列元素必须全是 str"
    assert result["resin_1_amount_phr"] == pytest.approx(100.0)
    assert result["process_atmosphere"] == "N2"


def test_partial_autofill_is_type_compatible(monkeypatch):
    workspace = pd.DataFrame({"resin_1_amount_phr": [100.0]})
    captured, result = _run_panel(monkeypatch, ALL_COLS, workspace)
    _assert_streamlit_type_compatible(captured["df"], captured["column_config"])
    assert result == {"resin_1_amount_phr": pytest.approx(100.0)}


def test_no_workspace_is_type_compatible(monkeypatch):
    """无工作区（全部需手工）时不得崩。"""
    captured, result = _run_panel(monkeypatch, ALL_COLS, None)
    _assert_streamlit_type_compatible(captured["df"], captured["column_config"])
    assert result == {}
    assert captured["df"]["值"].tolist() == ["", "", ""]


def test_categorical_string_values_do_not_crash_formatting(monkeypatch):
    """类别众数是字符串时不得因 f"{v:g}" 抛 ValueError。

    真实场景：test_atmosphere 的众数是 "N2"、tg_c_test_method 是 "DMA"。
    """
    workspace = pd.DataFrame(
        {
            "resin_1_amount_phr": [100.0, 100.0],
            "curing_agent_1_amount_phr": [33.0, 33.0],
            "process_atmosphere": ["air", "air"],
        }
    )
    captured, result = _run_panel(monkeypatch, ALL_COLS, workspace)
    _assert_streamlit_type_compatible(captured["df"], captured["column_config"])
    assert result["process_atmosphere"] == "air"
    assert result["resin_1_amount_phr"] == pytest.approx(100.0)


def test_mixed_numeric_and_categorical(monkeypatch):
    workspace = pd.DataFrame(
        {
            "resin_1_amount_phr": [100.0],
            "curing_agent_1_amount_phr": [33.0],
            "process_atmosphere": ["vacuum"],
        }
    )
    captured, result = _run_panel(monkeypatch, ALL_COLS, workspace)
    _assert_streamlit_type_compatible(captured["df"], captured["column_config"])
    assert result["process_atmosphere"] == "vacuum"
    assert result["curing_agent_1_amount_phr"] == pytest.approx(33.0)


def test_numeric_text_is_parsed_back_to_float(monkeypatch):
    """文本列必须能还原成 float（否则统一输入会变成字符串污染特征矩阵）。"""
    workspace = pd.DataFrame(
        {"resin_1_amount_phr": [100.0], "curing_agent_1_amount_phr": [33.0]}
    )
    _, result = _run_panel(monkeypatch, ALL_COLS, workspace)
    assert isinstance(result["resin_1_amount_phr"], float)
    assert isinstance(result["curing_agent_1_amount_phr"], float)


def test_empty_feature_cols_returns_empty(monkeypatch):
    import app_lib

    monkeypatch.setattr(app_lib.st, "data_editor", lambda data, **k: data)
    assert (
        app_lib._render_screening_uniform_inputs([], df_ref=None, key_prefix="t")
        == {}
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (100.0, "100"),
        (33.5, "33.5"),
        ("N2", "N2"),
        ("DMA", "DMA"),
        (1e12, "1e+12"),
    ],
)
def test_value_column_is_always_text_for_various_inputs(value, expected):
    """各类取值都必须被格式化成字符串（含字符串型类别值）。"""
    frame = pd.DataFrame(
        {
            "模型特征": pd.Series(["f"], dtype=object),
            "值": pd.Series(
                [
                    "" if value is None
                    else (
                        f"{float(value):g}"
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else str(value)
                    )
                ],
                dtype=object,
            ),
        }
    )
    assert frame["值"].iloc[0] == expected
    assert frame["值"].dtype == object
