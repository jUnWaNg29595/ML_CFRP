from collections.abc import Mapping, Sequence
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def resolve_data_source(
    raw_df: pd.DataFrame | None,
    processed_df: pd.DataFrame | None,
    source: str,
) -> tuple[pd.DataFrame, str]:
    if source == "processed" and isinstance(processed_df, pd.DataFrame):
        return processed_df, "当前处理数据"
    if isinstance(raw_df, pd.DataFrame):
        return raw_df, "原始数据"
    if isinstance(processed_df, pd.DataFrame):
        return processed_df, "当前处理数据"
    raise ValueError("没有可用的数据")


def sanitize_preview_columns(
    columns: Sequence[object],
    available_columns: Sequence[object],
) -> list[object]:
    available = set(available_columns)
    result: list[object] = []
    for column in columns:
        if column in available and column not in result:
            result.append(column)
    return result


def preview_dataframe(
    df: pd.DataFrame,
    columns: Sequence[object],
    row_count: int,
) -> pd.DataFrame:
    selected = sanitize_preview_columns(columns, df.columns.tolist())
    if not selected:
        raise ValueError("至少选择一列")
    if row_count < 1:
        raise ValueError("预览行数必须大于零")
    return df.loc[:, selected].head(int(row_count))


def dataframe_to_csv_bytes(
    df: pd.DataFrame,
    include_index: bool = False,
) -> bytes:
    return df.to_csv(index=include_index).encode("utf-8-sig")


def dataframe_to_excel_bytes(
    df: pd.DataFrame,
    include_index: bool = False,
) -> bytes:
    buffer = BytesIO()
    try:
        df.to_excel(buffer, index=include_index, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError("导出 Excel 需要安装 openpyxl") from exc
    return buffer.getvalue()


def figure_to_bytes(figure: object, fmt: str) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format=fmt)
    return buffer.getvalue()


def build_export_zip(files: Mapping[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, payload in files.items():
            archive.writestr(str(filename), bytes(payload))
    return buffer.getvalue()
