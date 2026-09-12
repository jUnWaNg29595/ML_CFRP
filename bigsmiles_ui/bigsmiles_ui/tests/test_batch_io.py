from pathlib import Path

import pandas as pd

from bigsmiles_ui.batch_io import process_table, write_table
from bigsmiles_ui.renderer import RenderOptions


def test_process_table_preserves_columns_and_appends_scalars(tmp_path: Path):
    source = pd.DataFrame({"样品": ["A", "B"], "结构": ["CCO", "C1(CC"]})
    result = process_table(source, "结构", tmp_path / "assets", RenderOptions())
    assert list(result.columns[:2]) == ["样品", "结构"]
    assert "原始字符串" in result.columns
    assert result.loc[0, "原始字符串"] == "CCO"
    assert result.loc[1, "原始字符串"] == "C1(CC"
    for column in result.columns[2:]:
        assert all(not isinstance(value, (list, dict, tuple, set)) for value in result[column])


def test_process_table_reports_progress_without_printing(tmp_path: Path):
    source = pd.DataFrame({"结构": ["CCO", ""]})
    calls: list[tuple[int, int, str]] = []
    result = process_table(
        source,
        "结构",
        tmp_path / "assets",
        RenderOptions(),
        progress_callback=lambda current, total, item: calls.append((current, total, item.parse_status)),
    )
    assert len(result) == 2
    assert calls == [(1, 2, "valid"), (2, 2, "empty")]


def test_write_table_supports_csv_and_xlsx(tmp_path: Path):
    frame = pd.DataFrame({"a": [1], "b": ["x"]})
    csv_path = write_table(frame, tmp_path / "out.csv")
    xlsx_path = write_table(frame, tmp_path / "out.xlsx")
    assert csv_path.exists()
    assert xlsx_path.exists()
    assert pd.read_csv(csv_path).to_dict(orient="records") == [{"a": 1, "b": "x"}]
    assert pd.read_excel(xlsx_path).to_dict(orient="records") == [{"a": 1, "b": "x"}]