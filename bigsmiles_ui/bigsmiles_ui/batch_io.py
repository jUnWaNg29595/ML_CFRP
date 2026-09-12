"""Vendored batch I/O helpers for SMILES / BigSMILES structure checking.

Public API (kept compatible with the host app in ``app.py``):

- ``read_uploaded_table``   – parse an uploaded CSV/XLSX/XLSM file into a DataFrame
- ``list_structure_columns`` – candidate structure columns of a DataFrame
- ``process_table``         – render every row of one structure column
- ``write_table``           – persist the result frame back to CSV/XLSX
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from .renderer import RenderOptions, render_structure


def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV / XLSX / XLSM file given its raw bytes.

    CSV 解码按 ``utf-8-sig`` → ``gb18030`` 顺序回退，兼容中文 Excel 导出文件。
    """
    suffix = Path(str(file_name)).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        frame = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        last_error: Exception | None = None
        frame = None
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                frame = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if frame is None:
            raise ValueError(f"CSV 编码无法识别：{last_error}")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def list_structure_columns(frame: pd.DataFrame) -> list[str]:
    """Heuristic: object columns look like structure candidates, ranked by name."""
    if frame is None or frame.empty:
        return []
    columns = [str(col) for col in frame.columns if frame[col].dtype == object]
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
    sample_paths: list[str] = []
    raw_values: list[str] = []

    for index, value in enumerate(result_frame[structure_column], start=1):
        raw = "" if pd.isna(value) else str(value).strip()
        result = render_structure(raw, output_dir, options)

        statuses.append(result.parse_status)
        detected.append(result.detected_type)
        normalized.append(result.normalized_structure)
        draw_statuses.append(result.draw_status)
        error_messages.append(result.error_message)
        warning_messages.append(result.warning_message)
        # 原版渲染器返回相对 output_dir 的文件名（如 main_xxxxx.png）。
        # 仅当路径仍然存在时写入，避免把失败行写成悬空引用。
        main_name = str(result.main_image_path or "").strip()
        image_paths.append(main_name if (output_dir / main_name).is_file() else "")
        sample_name = str(result.sample_image_path or "").strip()
        sample_paths.append(sample_name if (output_dir / sample_name).is_file() else "")
        raw_values.append(raw)

        if progress_callback is not None:
            try:
                progress_callback(index, max(total, 1), result)
            except Exception:
                pass

    result_frame["原始字符串"] = raw_values
    result_frame["解析状态"] = statuses
    result_frame["结构类型"] = detected
    result_frame["规范化结构"] = normalized
    result_frame["绘图状态"] = draw_statuses
    result_frame["错误信息"] = error_messages
    result_frame["警告信息"] = warning_messages
    result_frame["图片路径"] = image_paths
    result_frame["采样图路径"] = sample_paths
    return result_frame


def write_table(frame: pd.DataFrame, output_path) -> Path:
    """Write a DataFrame to ``.csv`` or ``.xlsx``; returns the written path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".xlsx", ".xlsm"):
        frame.to_excel(output_path, index=False, engine="openpyxl")
    else:
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
