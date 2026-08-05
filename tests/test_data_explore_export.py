from io import BytesIO
from zipfile import ZipFile

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.data_explorer import EnhancedDataExplorer
from core.data_explore_export import (
    build_export_zip,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_bytes,
    preview_dataframe,
    resolve_data_source,
    sanitize_preview_columns,
)


def test_resolve_data_source_defaults_to_processed_data():
    raw = pd.DataFrame({"raw": [1]})
    processed = pd.DataFrame({"processed": [2]})

    result, label = resolve_data_source(raw, processed, "processed")

    assert result is processed
    assert label == "当前处理数据"


def test_resolve_data_source_falls_back_to_raw_when_processed_is_missing():
    raw = pd.DataFrame({"raw": [1]})

    result, label = resolve_data_source(raw, None, "processed")

    assert result is raw
    assert label == "原始数据"


def test_sanitize_preview_columns_drops_missing_and_duplicates_preserving_order():
    result = sanitize_preview_columns(
        ["b", "missing", "a", "b"],
        ["a", "b", "c"],
    )

    assert result == ["b", "a"]


def test_preview_dataframe_requires_explicit_columns_and_limits_rows():
    frame = pd.DataFrame({"a": range(8), "b": range(8, 16)})

    result = preview_dataframe(frame, ["b"], 3)

    assert list(result.columns) == ["b"]
    assert result["b"].tolist() == [8, 9, 10]
    with pytest.raises(ValueError, match="至少选择一列"):
        preview_dataframe(frame, [], 3)


def test_dataframe_to_csv_bytes_uses_utf8_bom():
    payload = dataframe_to_csv_bytes(pd.DataFrame({"特征": ["值"]}))

    assert payload.startswith(b"\xef\xbb\xbf")
    assert "特征".encode("utf-8") in payload


def test_dataframe_to_excel_bytes_is_readable():
    payload = dataframe_to_excel_bytes(pd.DataFrame({"a": [1, 2]}))

    restored = pd.read_excel(BytesIO(payload))

    assert restored["a"].tolist() == [1, 2]


def test_build_export_zip_contains_all_named_files():
    payload = build_export_zip({"data.csv": b"a,b\n1,2\n", "chart.html": b"<html/>"})

    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {"data.csv", "chart.html"}


def test_figure_to_bytes_uses_requested_format():
    class Figure:
        def savefig(self, buffer, format):
            assert format == "png"
            buffer.write(b"png")

    assert figure_to_bytes(Figure(), "png") == b"png"


def test_correlation_data_matches_selected_numeric_columns():
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0], "label": ["x", "y", "z"]}
    )
    explorer = EnhancedDataExplorer(frame)

    result = explorer.correlation_data(["a", "b"])

    assert list(result.index) == ["a", "b"]
    assert list(result.columns) == ["a", "b"]
    assert result.loc["a", "a"] == 1.0


def test_distribution_data_is_long_form_and_excludes_missing_values():
    frame = pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})
    result = EnhancedDataExplorer(frame).distribution_data(["a", "b"])

    assert list(result.columns) == ["feature", "value"]
    assert result.to_dict("records") == [
        {"feature": "a", "value": 1.0},
        {"feature": "b", "value": 3.0},
        {"feature": "b", "value": 4.0},
    ]


def test_missing_values_data_contains_count_and_percentage():
    frame = pd.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]})
    result = EnhancedDataExplorer(frame).missing_values_data()

    assert result.to_dict("records") == [
        {"feature": "a", "missing_count": 1, "missing_percent": 50.0}
    ]


def test_boxplot_data_uses_same_long_form_as_distribution_data():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = EnhancedDataExplorer(frame).boxplot_data(["b"])

    assert result.to_dict("records") == [
        {"feature": "b", "value": 3.0},
        {"feature": "b", "value": 4.0},
    ]


def test_figure_to_bytes_exports_plotly_html():
    figure = go.Figure(go.Bar(x=["a"], y=[1]))

    payload = figure_to_bytes(figure, "html")

    assert b"<html" in payload.lower()
    assert b"plotly" in payload.lower()


def test_figure_to_bytes_exports_matplotlib_html_and_png():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    try:
        html_payload = figure_to_bytes(figure, ".HTML")
        png_payload = figure_to_bytes(figure, ".PNG")
    finally:
        plt.close(figure)

    assert b"<html" in html_payload.lower()
    assert png_payload.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("fmt", ["png", "svg"])
def test_figure_to_bytes_reports_plotly_kaleido_dependency_for_image_formats(
    monkeypatch,
    fmt,
):
    figure = go.Figure()

    def raise_missing_dependency(*args, **kwargs):
        raise ValueError("Image export using the 'kaleido' engine requires the kaleido package")

    monkeypatch.setattr(figure, "to_image", raise_missing_dependency)

    with pytest.raises(
        RuntimeError,
        match=rf"(?=.*Plotly)(?=.*kaleido)(?=.*{fmt})",
    ):
        figure_to_bytes(figure, fmt)


@pytest.mark.parametrize("fmt", ["html", "png"])
def test_figure_to_bytes_reports_matplotlib_dependency_and_format(
    monkeypatch,
    fmt,
):
    figure, _ = plt.subplots()

    def raise_missing_dependency(*args, **kwargs):
        raise ImportError("No module named 'matplotlib'")

    monkeypatch.setattr(figure, "savefig", raise_missing_dependency)
    try:
        with pytest.raises(
            RuntimeError,
            match=rf"(?=.*Matplotlib)(?=.*matplotlib)(?=.*{fmt})",
        ):
            figure_to_bytes(figure, fmt)
    finally:
        plt.close(figure)


def test_figure_to_bytes_rejects_unknown_format():
    with pytest.raises(ValueError, match="不支持的图表格式"):
        figure_to_bytes(go.Figure(), "pdf")


def test_figure_to_bytes_exports_matplotlib_svg():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    try:
        payload = figure_to_bytes(figure, "svg")
    finally:
        plt.close(figure)

    assert payload.lstrip().startswith(b"<?xml")
