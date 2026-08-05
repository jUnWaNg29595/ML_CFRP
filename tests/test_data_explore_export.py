from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

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
