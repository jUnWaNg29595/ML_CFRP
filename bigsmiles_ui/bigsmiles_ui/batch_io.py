"""Vendored batch I/O helpers for SMILES / BigSMILES structure checking.

Public API (kept compatible with the host app in ``app.py``):

- ``read_uploaded_table``   – parse an uploaded CSV/XLSX/XLSM file into a DataFrame
- ``list_structure_columns`` – candidate structure columns of a DataFrame
- ``process_table``         – render every row of one structure column
- ``write_table``           – persist the result frame back to CSV/XLSX
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from bigsmiles_ui.renderer import RenderOptions, render_structure

_STATUS_ORDER = {"valid": 0, "valid_but_not_renderable": 1, "invalid": 2, "empty": 3}


def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV / XLSX / XLSM file given its raw bytes."""
    suffix = Path(str(file_name)).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(
            __import__("io").BytesIO(file_bytes),
            engine="openpyxl",
        )
    return pd.read_csv(__import__("io").BytesIO(file_bytes))


def list_structure_columns(frame: pd.DataFrame) -> list[str]:
    """Heuristic: object columns look like structure candidates, ranked by name."""
    if frame is None or frame.empty:
        return []
    columns = [
        str(col)
        for col in frame.columns
        if frame[col].dtype == object
    ]
    priority = ("smiles", "bigsmiles", "结构", "结构式")

    def rank(name: str) -> tuple:
        lowered = name.lower()
        for index, key in enumerate(priority):
            if key in lowered:
                return (0, index, name)
        return (1, 0, name)

    return sorted(columns, key=rank)


def process_table(
    frame: pd.DataFrame,
    structure_column: str,
    output_dir,
    options: RenderOptions | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Render one structure column row by row, appending status/image columns."""
    options = options or RenderOptions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_frame = frame.copy()
    structure_column = str(structure_column)

    total = len(result_frame)
    statuses: list[str] = []
    detected: list[str] = []
    normalized: list[str] = []
    draw_statuses: list[str] = []
    error_messages: list[str] = []
    warning_messages: list[str] = []
    image_paths: list[str] = []

    for index, value in enumerate(result_frame[structure_column], start=1):
        raw = "" if pd.isna(value) else str(value).strip()
        result = render_structure(raw, output_dir, options)

        statuses.append(result.parse_status)
        detected.append(result.detected_type)
        normalized.append(result.normalized_structure)
        draw_statuses.append(result.draw_status)
        error_messages.append(result.error_message)
        warning_messages.append(result.warning_message)
        image_paths.append(
            f"images/{Path(result.main_image_path).name}"
            if result.main_image_path
            else ""
        )

        if progress_callback is not None:
            try:
                progress_callback(index, max(total, 1), result)
            except Exception:
                pass

    result_frame["解析状态"] = statuses
    result_frame["结构类型"] = detected
    result_frame["规范化结构"] = normalized
    result_frame["绘图状态"] = draw_statuses
    result_frame["错误信息"] = error_messages
    result_frame["警告信息"] = warning_messages
    result_frame["图片路径"] = image_paths
    return result_frame


def write_table(frame: pd.DataFrame, output_path) -> Path:
    """Write a DataFrame to ``.csv`` or ``.xlsx``; returns the written path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".xlsx", ".xlsm"):
        frame.to_excel(output_path, index=False, engine="openpyxl")
    else:
        frame.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
    return output_path
