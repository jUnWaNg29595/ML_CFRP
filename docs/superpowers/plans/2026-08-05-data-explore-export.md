# 数据探索页面导出与预览增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为数据探索页面增加统一的图表数据/图表文件导出能力，并将数据预览改为可搜索多选特征列和可调行数。

**Architecture:** 将数据源解析、预览列校验、图表数据整理、CSV/Excel/图片/HTML/ZIP 字节生成放入独立的 `core/data_explore_export.py`，保持其不依赖 Streamlit。`app.py` 只负责控件状态、图表生成和下载按钮。现有 `core/data_explorer.py` 的统计/绘图逻辑保持不变，只新增与图表输入一致的数据导出方法。

**Tech Stack:** Python 3.10、pandas、numpy、Plotly、Matplotlib、Streamlit、pytest、openpyxl（可选）。

## Global Constraints

- 图表数据与图表文件都必须支持导出。
- 导出中心默认使用当前处理数据，且允许切换原始数据。
- 预览必须支持可搜索、多选列和可调行数，空选择不得隐式显示全部列。
- 当前处理数据优先使用 `st.session_state.processed_data`，否则回退到 `st.session_state.data`。
- CSV 使用 UTF-8 with BOM；Excel 使用内存缓冲区，不落地临时文件。
- 图表导出失败只影响当前格式，不能阻断数据探索页面。
- 不修改数据清洗、特征工程、模型训练和虚拟筛选逻辑。
- 遵循测试驱动开发：每个新生产函数先有会失败的测试，再写最小实现。
- 不执行 `git commit`，保留现有工作区改动。

---

### Task 1: 建立无 Streamlit 依赖的导出与预览工具层

**Files:**
- Create: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explore_export.py`
- Test: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py`

**Interfaces:**
- `resolve_data_source(raw_df: pd.DataFrame | None, processed_df: pd.DataFrame | None, source: str) -> tuple[pd.DataFrame, str]`
- `sanitize_preview_columns(columns: Sequence[object], available_columns: Sequence[object]) -> list[object]`
- `preview_dataframe(df: pd.DataFrame, columns: Sequence[object], row_count: int) -> pd.DataFrame`
- `dataframe_to_csv_bytes(df: pd.DataFrame, include_index: bool = False) -> bytes`
- `dataframe_to_excel_bytes(df: pd.DataFrame, include_index: bool = False) -> bytes`
- `figure_to_bytes(figure: object, fmt: str) -> bytes`
- `build_export_zip(files: Mapping[str, bytes]) -> bytes`

- [ ] **Step 1: Write failing tests for source selection and preview bounds**

```python
import pandas as pd
import pytest

from core.data_explore_export import (
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
```

- [ ] **Step 2: Run the focused test and verify it fails for missing module**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: FAIL with `ModuleNotFoundError` or missing function errors from `core.data_explore_export`.

- [ ] **Step 3: Implement source selection and preview helpers**

```python
from collections.abc import Mapping, Sequence
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def resolve_data_source(raw_df, processed_df, source):
    if source == "processed" and isinstance(processed_df, pd.DataFrame):
        return processed_df, "当前处理数据"
    if isinstance(raw_df, pd.DataFrame):
        return raw_df, "原始数据"
    if isinstance(processed_df, pd.DataFrame):
        return processed_df, "当前处理数据"
    raise ValueError("没有可用的数据")


def sanitize_preview_columns(columns, available_columns):
    available = set(available_columns)
    result = []
    for column in columns:
        if column in available and column not in result:
            result.append(column)
    return result


def preview_dataframe(df, columns, row_count):
    selected = sanitize_preview_columns(columns, df.columns.tolist())
    if not selected:
        raise ValueError("至少选择一列")
    if row_count < 1:
        raise ValueError("预览行数必须大于零")
    return df.loc[:, selected].head(int(row_count))
```

实现时补齐类型标注、`dataframe_to_csv_bytes`、`dataframe_to_excel_bytes` 和 `build_export_zip`；Excel 依赖缺失时抛出带有 `openpyxl` 的明确错误。

- [ ] **Step 4: Run focused tests and verify they pass**

Run the same pytest command from Step 2.

Expected: 4 passed.

- [ ] **Step 5: Add failing tests for CSV/Excel/ZIP byte formats**

```python
from zipfile import ZipFile
from io import BytesIO

from core.data_explore_export import (
    build_export_zip,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
)


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
```

- [ ] **Step 6: Run the new format tests and verify they fail before implementation**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: FAIL only on the not-yet-implemented byte conversion helpers.

- [ ] **Step 7: Implement byte conversion and ZIP helpers**

```python
def dataframe_to_csv_bytes(df, include_index=False):
    return df.to_csv(index=include_index).encode("utf-8-sig")


def dataframe_to_excel_bytes(df, include_index=False):
    buffer = BytesIO()
    try:
        df.to_excel(buffer, index=include_index, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError("导出 Excel 需要安装 openpyxl") from exc
    return buffer.getvalue()


def build_export_zip(files):
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, payload in files.items():
            archive.writestr(str(filename), bytes(payload))
    return buffer.getvalue()
```

- [ ] **Step 8: Run Task 1 tests and verify all pass**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: all Task 1 tests pass.

---

### Task 2: 提供与现有图表一致的图表数据导出

**Files:**
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explorer.py`
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explore_export.py`
- Test: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py`

**Interfaces:**
- `EnhancedDataExplorer.correlation_data(cols=None) -> pd.DataFrame`
- `EnhancedDataExplorer.distribution_data(cols=None) -> pd.DataFrame`
- `EnhancedDataExplorer.missing_values_data() -> pd.DataFrame`
- `EnhancedDataExplorer.boxplot_data(cols=None) -> pd.DataFrame`
- `figure_to_bytes(figure, fmt) -> bytes`

- [ ] **Step 1: Write failing tests for correlation, distribution, missing and boxplot export data**

```python
from core.data_explorer import EnhancedDataExplorer


def test_correlation_data_matches_selected_numeric_columns():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0], "label": ["x", "y", "z"]})
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
```

- [ ] **Step 2: Run the focused tests and verify the new methods fail**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: FAIL with `AttributeError` for the four missing methods.

- [ ] **Step 3: Implement the four data extraction methods**

实现规则：

- 只保留现有 DataFrame 中存在且为数值型的列；
- 相关性返回矩阵，列顺序与输入顺序一致；
- 分布和箱线图返回 `feature/value` 长表，并移除缺失值；
- 缺失值只返回缺失数大于零的列，按缺失数升序，与现有缺失图一致；
- 不修改原始 DataFrame。

```python
def correlation_data(self, cols=None):
    cols = self._valid_numeric_columns(cols)
    if len(cols) < 2:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    return self.data.loc[:, cols].corr()


def distribution_data(self, cols=None):
    cols = self._valid_numeric_columns(cols)
    records = []
    for col in cols:
        records.extend(
            {"feature": col, "value": value}
            for value in self.data[col].dropna().tolist()
        )
    return pd.DataFrame(records, columns=["feature", "value"])


def missing_values_data(self):
    missing = self.data.isnull().sum()
    missing = missing[missing > 0].sort_values()
    return pd.DataFrame(
        {
            "feature": missing.index.tolist(),
            "missing_count": missing.astype(int).tolist(),
            "missing_percent": (missing / len(self.data) * 100).round(4).tolist()
            if len(self.data)
            else [0.0] * len(missing),
        }
    )


def boxplot_data(self, cols=None):
    return self.distribution_data(cols)
```

其中 `_valid_numeric_columns` 是 `EnhancedDataExplorer` 内部的小方法，默认返回全部数值列，输入列按请求顺序去重。

- [ ] **Step 4: Add failing tests for Plotly/Matplotlib export dispatch**

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from core.data_explore_export import figure_to_bytes


def test_figure_to_bytes_exports_plotly_html():
    figure = go.Figure(go.Bar(x=["a"], y=[1]))
    payload = figure_to_bytes(figure, "html")
    assert b"<html" in payload.lower()
    assert b"plotly" in payload.lower()


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
```

- [ ] **Step 5: Run dispatch tests and verify they fail**

Run the same focused pytest command.

Expected: FAIL because `figure_to_bytes` is not implemented.

- [ ] **Step 6: Implement figure dispatch with isolated format errors**

实现 `figure_to_bytes`：

- `fmt` 统一转小写并去掉前导点；
- Plotly Figure：
  - `html` 调用 `write_html` 到字符串并编码 UTF-8；
  - `png`/`svg` 调用 `to_image`；
- Matplotlib Figure：
  - `savefig` 写入内存缓冲区；
  - `html` 生成自包含 HTML，并以 base64 嵌入 PNG；
- 对未知格式抛出 `ValueError("不支持的图表格式")`；
- 图像依赖缺失时抛出包含格式和依赖名的 `RuntimeError`。

- [ ] **Step 7: Run Task 2 tests and verify all pass**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: all Task 1 and Task 2 tests pass. If Plotly image export dependency is unavailable, the PNG test must be skipped or reported as an expected optional-dependency failure; HTML, SVG and all data tests must remain green.

---

### Task 3: 接入数据探索页的预览选择和图表快速导出

**Files:**
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\app.py` near `page_data_explore`
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_app_scope_regressions.py`

**Interfaces:**
- 在 `app.py` 中增加 `_render_data_explore_preview(raw_df, processed_df) -> None`
- 在 `app.py` 中增加 `_render_figure_export_controls(figure, data_df, base_name, key_prefix) -> None`
- `page_data_explore()` 复用上述两个函数，不把导出格式转换逻辑直接写进页面。

- [ ] **Step 1: Write failing source-level regression tests**

```python
def test_data_explore_page_has_searchable_preview_and_explicit_selection():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "数据预览" in page_source
    assert "预览特征/列" in page_source
    assert "至少选择一列" in page_source
    assert "processed_data" in page_source
    assert "原始数据" in page_source


def test_data_explore_page_has_both_quick_export_formats():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "导出图表数据" in page_source
    assert "导出图表" in page_source
    assert "HTML" in page_source
    assert "Excel" in page_source
```

- [ ] **Step 2: Run the regression tests and verify they fail**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_app_scope_regressions.py' -q
```

Expected: FAIL because the page does not yet contain the new preview/export labels and helpers.

- [ ] **Step 3: Implement preview rendering**

在 `page_data_explore()` 得到 `raw_df` 和 `processed_df` 后，调用：

```python
def _render_data_explore_preview(raw_df, processed_df):
    source_options = ["当前处理数据"]
    if isinstance(raw_df, pd.DataFrame):
        source_options.append("原始数据")
    source_label = st.radio(
        "预览数据源",
        source_options,
        horizontal=True,
        key="explore_preview_source",
    )
    source_key = "raw" if source_label == "原始数据" else "processed"
    preview_df, _ = resolve_data_source(raw_df, processed_df, source_key)
    available_cols = preview_df.columns.tolist()
    previous = sanitize_preview_columns(
        st.session_state.get("explore_preview_cols", available_cols[:8]),
        available_cols,
    )
    row_count = st.slider(
        "预览行数",
        min_value=5,
        max_value=max(5, min(5000, len(preview_df))),
        value=min(max(50, 5), max(5, min(5000, len(preview_df)))),
        key="explore_preview_rows",
    )
    selected_cols = st.multiselect(
        "预览特征/列",
        options=available_cols,
        default=previous,
        key="explore_preview_cols",
        help="支持搜索和多选，显示顺序按选择顺序保留。",
    )
    st.caption(f"已选 {len(selected_cols)} / {len(available_cols)} 列")
    if not selected_cols:
        st.info("至少选择一列后显示预览表。")
        return
    st.dataframe(
        preview_dataframe(preview_df, selected_cols, row_count),
        width="stretch",
    )
```

实现时通过 `st.session_state` 在数据源变化后清理不存在的列；不使用空选择显示全部列。

- [ ] **Step 4: Implement per-chart quick export**

`_render_figure_export_controls` 需要：

- 生成当前图表实际使用的数据 DataFrame；
- 提供 `导出图表数据` 的 CSV/Excel 下载按钮；
- 提供 `导出图表` 的 PNG/SVG/HTML 下载按钮；
- 捕获单个格式的 `RuntimeError`，显示 `该格式暂不可用：...`；
- 使用稳定的 key 前缀，避免不同标签页按钮冲突；
- 默认数据使用当前处理数据，图表数据不重新使用原始数据。

```python
def _render_figure_export_controls(figure, data_df, base_name, key_prefix):
    if figure is None:
        return
    st.markdown("#### 导出当前图表")
    data_col, figure_col = st.columns(2)
    with data_col:
        st.download_button(
            "📥 导出图表数据 CSV",
            dataframe_to_csv_bytes(data_df),
            f"{base_name}.csv",
            "text/csv",
            key=f"{key_prefix}_csv",
        )
        try:
            excel_bytes = dataframe_to_excel_bytes(data_df)
            st.download_button(
                "📥 导出图表数据 Excel",
                excel_bytes,
                f"{base_name}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_xlsx",
            )
        except RuntimeError as exc:
            st.caption(str(exc))
    with figure_col:
        for fmt, mime in (("html", "text/html"), ("png", "image/png"), ("svg", "image/svg+xml")):
            try:
                payload = figure_to_bytes(figure, fmt)
            except (RuntimeError, ValueError) as exc:
                st.caption(f"{fmt.upper()} 暂不可用：{exc}")
                continue
            st.download_button(
                f"📈 导出图表 {fmt.upper()}",
                payload,
                f"{base_name}.{fmt}",
                mime,
                key=f"{key_prefix}_{fmt}",
            )
```

- [ ] **Step 5: Wire existing figures to quick export**

在 `page_data_explore()` 中：

- 相关性图使用 `EnhancedDataExplorer(df_corr).correlation_data(selected_cols)`；
- Pearson 图使用 `corr_matrix.reset_index()` 作为矩阵数据导出；
- 分布图使用 `distribution_data`；
- 箱线图使用 `boxplot_data`；
- 缺失值图使用 `missing_values_data`；
- 每个图表紧跟 `st.plotly_chart` 或 `st.pyplot` 调用放置快速导出控件。

图表没有生成时不显示导出按钮；数据为空时显示明确提示。

- [ ] **Step 6: Run page regression tests and focused core tests**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_app_scope_regressions.py' `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: all tests pass.

---

### Task 4: 增加统一导出中心与页面回归验证

**Files:**
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\app.py` in `page_data_explore` export tab
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explore_export.py`
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py`
- Modify: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_app_scope_regressions.py`

**Interfaces:**
- `build_export_zip(files: Mapping[str, bytes]) -> bytes` 承担多文件导出；
- 导出中心复用 `_render_figure_export_controls` 使用的同一数据和格式函数；
- 不新增后台线程、临时文件或常驻服务。

- [ ] **Step 1: Write failing tests for ZIP naming and explicit empty selection**

```python
def test_build_export_zip_preserves_nested_export_names():
    payload = build_export_zip({
        "correlation/data.csv": b"a,b\n1,2\n",
        "correlation/chart.html": b"<html></html>",
    })
    with ZipFile(BytesIO(payload)) as archive:
        assert archive.namelist() == [
            "correlation/data.csv",
            "correlation/chart.html",
        ]


def test_preview_dataframe_does_not_fallback_to_all_columns():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="至少选择一列"):
        preview_dataframe(frame, [], 50)
```

- [ ] **Step 2: Run tests and verify the new regression fails or is incomplete**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' -q
```

Expected: the explicit-selection test passes from Task 1; the ZIP-order test must fail if ZIP ordering or path preservation regresses. If it already passes, keep it as a regression guard and proceed.

- [ ] **Step 3: Implement export-center UI**

在“💾 导出”标签页增加：

- `导出数据源` 单选：当前处理数据/原始数据，默认当前处理数据；
- `导出内容` 多选：图表数据/图表文件；
- `图表` 多选：描述统计、相关性矩阵、分布图、箱线图、缺失值；
- `数据格式` 多选：CSV、Excel；
- `图表格式` 多选：HTML、PNG、SVG；
- 点击 `生成统一导出包` 后构建文件映射并调用 `build_export_zip`；
- ZIP 下载文件名使用 `data_explore_export_YYYYMMDD_HHMMSS.zip`；
- 没有可导出的图表或没有选中内容时显示提示并不生成空 ZIP。

统一导出包的文件命名规则：

```text
summary/data.csv
correlation/data.csv
correlation/chart.html
distribution/data.xlsx
missing_values/chart.svg
```

描述统计的图表文件没有可用图表时只导出数据；图表生成失败时在导出中心显示失败原因并继续处理其他图表。

- [ ] **Step 4: Add source-level assertions for export-center controls**

```python
def test_data_explore_export_center_supports_raw_and_processed_data():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "导出数据源" in page_source
    assert "生成统一导出包" in page_source
    assert "build_export_zip" in page_source
    assert "当前处理数据" in page_source
    assert "原始数据" in page_source
```

- [ ] **Step 5: Run the complete relevant test set**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_data_explore_export.py' `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_app_scope_regressions.py' `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\tests\test_navigation.py' -q
```

Expected: all relevant tests pass without modifying unrelated failures.

- [ ] **Step 6: Run a syntax/import smoke check**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explore_export.py' `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\core\data_explorer.py' `
  'C:\Users\wangj\Desktop\CFRP系统\CFRP系统\app.py'
```

Expected: command exits with code 0 and produces no traceback.

- [ ] **Step 7: Update the implementation notes**

在 `docs/superpowers/specs/2026-08-05-data-explore-export-design.md` 末尾补充实际实现差异（如果存在），包括可选依赖导致的格式限制；不修改其他历史文档。

