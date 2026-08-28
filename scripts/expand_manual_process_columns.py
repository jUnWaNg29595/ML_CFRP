#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Expand multi-stage process columns from the manual CFRP workbook.

This script is designed for the current `手工数据.xls`-style workbook:
- keep every original non-empty column
- expand cure/post-cure list columns into fixed-width stage features
- add a few lightweight QA / training-oriented helper columns

Output defaults to a UTF-8 CSV next to the source workbook and does not
overwrite the original file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path

import pandas as pd
import xlrd

from core.process_features import compute_process_features


DEFAULT_NAME_KEYWORD = "\u624b\u5de5\u6570\u636e"
CURE_STAGE_LIMIT = 4
POST_CURE_STAGE_LIMIT = 2

SEQUENCE_COLUMN_SPECS: dict[str, int] = {
    "cure_temperature_c": CURE_STAGE_LIMIT,
    "cure_time_h": CURE_STAGE_LIMIT,
    "cure_temperature_list_c": CURE_STAGE_LIMIT,
    "cure_time_list_h": CURE_STAGE_LIMIT,
    "cure_ramp_rate_list_c_min": CURE_STAGE_LIMIT,
    "post_cure_temperature_c": POST_CURE_STAGE_LIMIT,
    "post_cure_time_h": POST_CURE_STAGE_LIMIT,
    "post_cure_temperature_list_c": POST_CURE_STAGE_LIMIT,
    "post_cure_time_list_h": POST_CURE_STAGE_LIMIT,
    "post_cure_ramp_rate_list_c_min": POST_CURE_STAGE_LIMIT,
}

GENERIC_NUMERIC_SEQUENCE_SPECS: dict[str, dict[str, object]] = {
    "resin_eew_list": {"limit": 3, "replace_mode": "mean"},
    "curing_agent_ahew_list": {"limit": 2, "replace_mode": "mean"},
    "activation_energy_ea_list": {"limit": 10, "replace_mode": "mean"},
    "dma_frequency_hz": {"limit": 4, "replace_mode": "mean"},
    "flexural_span_mm": {"limit": 2, "replace_mode": "mean"},
    "flexural_crosshead_speed_mm_min": {"limit": 2, "replace_mode": "mean"},
    "tga_sample_mass_mg": {"limit": 2, "replace_mode": "mean"},
    "tensile_test_temperature_c": {"limit": 2, "replace_mode": "mean"},
    "tga_heating_rate_c_min": {"limit": 8, "replace_mode": "mean"},
    "dma_heating_rate_c_min": {"limit": 8, "replace_mode": "mean"},
}

NUMERIC_RANGE_SPECS: dict[str, dict[str, object]] = {
    "tga_temperature_range_c": {"replace_mode": "span"},
}

MULTIVALUE_REDUCER_SPECS: dict[str, dict[str, object]] = {
    "activation_energy_ea_kj_mol": {"limit": 6, "replace_mode": "mean"},
    "tg_c": {"limit": 8, "replace_mode": "mean"},
    "td5_c": {"limit": 4, "replace_mode": "mean"},
    "td10_c": {"limit": 4, "replace_mode": "mean"},
    "tmax_c": {"limit": 12, "replace_mode": "mean"},
    "tensile_strength_mpa": {"limit": 4, "replace_mode": "mean"},
    "flexural_strength_mpa": {"limit": 4, "replace_mode": "mean"},
    "lap_shear_strength_mpa": {"limit": 6, "replace_mode": "mean"},
    "tensile_strain_at_break_pct": {"limit": 4, "replace_mode": "mean"},
}

TEXT_LIST_NORMALIZE_COLS = {
    "tensile_strength_standard",
    "compressive_modulus_standard",
    "compressive_strength_standard",
    "tensile_strain_at_break_standard",
    "tensile_modulus_standard",
    "activation_energy_ea_standard",
}

HEAVY_TEXT_DROP_COLS = {
    "cure_schedule",
    "post_cure_schedule",
    "mixing_process",
    "degassing_process",
    "activation_energy_ea_notes",
}

NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")
MDSC_RE = re.compile(r"([-+]?\d*\.?\d+)\s*°?\s*C\s*/\s*([-+]?\d*\.?\d+)\s*s")


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "<na>", "null"}


def _format_cell(value: object) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    return str(value).strip()


def _normalize_list_text(text: str) -> str:
    normalized = str(text).strip()
    normalized = normalized.replace("\uff1b", ";").replace("\uff0c", ",").replace("\u3001", ";")
    normalized = normalized.replace("|", ";")
    return normalized


def _parse_number_list(value: object) -> list[float]:
    if _is_blank(value):
        return []
    text = _normalize_list_text(_format_cell(value))
    if ";" in text:
        parts = [part.strip() for part in text.split(";") if part.strip()]
    elif "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parts = [text]

    values: list[float] = []
    for part in parts:
        match = NUMERIC_RE.search(part)
        if match:
            try:
                values.append(float(match.group(0)))
            except Exception:
                continue
    return values


def _safe_mean(values: list[float]) -> str:
    if not values:
        return ""
    return format(sum(values) / len(values), "g")


def _safe_sum(values: list[float]) -> str:
    if not values:
        return ""
    return format(sum(values), "g")


def _safe_min(values: list[float]) -> str:
    if not values:
        return ""
    return format(min(values), "g")


def _safe_max(values: list[float]) -> str:
    if not values:
        return ""
    return format(max(values), "g")


def _safe_std(values: list[float]) -> str:
    if len(values) <= 1:
        return ""
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return format(math.sqrt(var), "g")


def _get_first_sequence(row: dict[str, str], candidates: tuple[str, ...]) -> tuple[list[float], str]:
    for col in candidates:
        values = _parse_number_list(row.get(col, ""))
        if values:
            return values, col
    return [], candidates[0] if candidates else ""


def _schedule_flags(value: object, prefix: str) -> dict[str, str]:
    text = _format_cell(value)
    if not text:
        return {
            f"{prefix}_schedule_has_isothermal": "",
            f"{prefix}_schedule_has_mdsc": "",
            f"{prefix}_schedule_has_ramp": "",
            f"{prefix}_schedule_modulation_amplitude_c": "",
            f"{prefix}_schedule_modulation_period_s": "",
        }

    match = MDSC_RE.search(text)
    return {
        f"{prefix}_schedule_has_isothermal": "1" if "\u7b49\u6e29" in text else "0",
        f"{prefix}_schedule_has_mdsc": "1" if "MDSC" in text.upper() else "0",
        f"{prefix}_schedule_has_ramp": "1" if any(token in text for token in ["\u5347\u6e29", "ramp", "Ramp", "->", "\u81f3"]) else "0",
        f"{prefix}_schedule_modulation_amplitude_c": match.group(1) if match else "",
        f"{prefix}_schedule_modulation_period_s": match.group(2) if match else "",
    }


def _expand_stage_group(
    row: dict[str, str],
    prefix: str,
    temp_cols: tuple[str, ...],
    time_cols: tuple[str, ...],
    ramp_cols: tuple[str, ...],
    limit: int,
) -> dict[str, str]:
    temps, temp_source = _get_first_sequence(row, temp_cols)
    times, time_source = _get_first_sequence(row, time_cols)
    ramps, ramp_source = _get_first_sequence(row, ramp_cols)
    detected_stage_count = max(len(temps), len(times), len(ramps))

    expanded: dict[str, str] = {
        f"{prefix}_list_detected_stage_count": str(detected_stage_count) if detected_stage_count else "",
        f"{prefix}_extra_stage_count": str(max(0, detected_stage_count - limit)) if detected_stage_count else "0",
        f"{prefix}_temp_source_col": temp_source,
        f"{prefix}_time_source_col": time_source,
        f"{prefix}_ramp_source_col": ramp_source,
    }

    for stage_idx in range(limit):
        pos = stage_idx + 1
        expanded[f"{prefix}_stage{pos}_temp_c"] = format(temps[stage_idx], "g") if stage_idx < len(temps) else ""
        expanded[f"{prefix}_stage{pos}_time_h"] = format(times[stage_idx], "g") if stage_idx < len(times) else ""
        expanded[f"{prefix}_stage{pos}_ramp_c_min"] = format(ramps[stage_idx], "g") if stage_idx < len(ramps) else ""

    extra_temps = temps[limit:]
    extra_times = times[limit:]
    extra_ramps = ramps[limit:]
    expanded[f"{prefix}_extra_temp_mean_c"] = _safe_mean(extra_temps)
    expanded[f"{prefix}_extra_time_sum_h"] = _safe_sum(extra_times)
    expanded[f"{prefix}_extra_ramp_mean_c_min"] = _safe_mean(extra_ramps)
    expanded[f"{prefix}_temp_stage_count"] = str(len(temps)) if temps else ""
    expanded[f"{prefix}_time_stage_count"] = str(len(times)) if times else ""
    expanded[f"{prefix}_ramp_stage_count"] = str(len(ramps)) if ramps else ""
    expanded[f"{prefix}_stage_count_mismatch_flag"] = (
        "1" if len({n for n in [len(temps), len(times), len(ramps)] if n > 0}) > 1 else "0"
    )
    return expanded


def _expand_sequence_column(row: dict[str, str], source_col: str, limit: int) -> dict[str, str]:
    values = _parse_number_list(row.get(source_col, ""))
    expanded: dict[str, str] = {
        f"{source_col}_value_count": str(len(values)) if values else "",
        f"{source_col}_multi_value_flag": "1" if len(values) > 1 else "0",
        f"{source_col}_extra_value_count": str(max(0, len(values) - limit)) if values else "0",
        f"{source_col}_extra_value_mean": _safe_mean(values[limit:]),
        f"{source_col}_extra_value_sum": _safe_sum(values[limit:]),
    }
    for idx in range(limit):
        expanded[f"{source_col}_{idx + 1}"] = format(values[idx], "g") if idx < len(values) else ""
    return expanded


def _choose_scalar(values: list[float], replace_mode: str) -> str:
    if not values:
        return ""
    if replace_mode == "mean":
        return _safe_mean(values)
    if replace_mode == "sum":
        return _safe_sum(values)
    if replace_mode == "min":
        return _safe_min(values)
    if replace_mode == "max":
        return _safe_max(values)
    if replace_mode == "first":
        return format(values[0], "g")
    if replace_mode == "last":
        return format(values[-1], "g")
    raise ValueError(f"Unsupported replace_mode: {replace_mode}")


def _clean_numeric_sequence_value(
    row: dict[str, str],
    source_col: str,
    limit: int,
    replace_mode: str = "mean",
) -> dict[str, str]:
    raw = row.get(source_col, "")
    values = _parse_number_list(raw)
    cleaned: dict[str, str] = {
        f"{source_col}_raw": raw,
        f"{source_col}_value_count": str(len(values)) if values else "",
        f"{source_col}_multi_value_flag": "1" if len(values) > 1 else "0",
        f"{source_col}_mean": _safe_mean(values),
        f"{source_col}_min": _safe_min(values),
        f"{source_col}_max": _safe_max(values),
        f"{source_col}_sum": _safe_sum(values),
        f"{source_col}_std": _safe_std(values),
    }
    cleaned[source_col] = _choose_scalar(values, replace_mode) if values else ""
    for idx in range(limit):
        cleaned[f"{source_col}_{idx + 1}"] = format(values[idx], "g") if idx < len(values) else ""
    return cleaned


def _clean_numeric_range_value(
    row: dict[str, str],
    source_col: str,
    replace_mode: str = "span",
) -> dict[str, str]:
    raw = row.get(source_col, "")
    values = _parse_number_list(raw)
    start = values[0] if len(values) >= 1 else None
    end = values[1] if len(values) >= 2 else None
    span = (end - start) if (start is not None and end is not None) else None
    mid = ((start + end) / 2.0) if (start is not None and end is not None) else None

    cleaned: dict[str, str] = {
        f"{source_col}_raw": raw,
        f"{source_col}_value_count": str(len(values)) if values else "",
        f"{source_col}_start": format(start, "g") if start is not None else "",
        f"{source_col}_end": format(end, "g") if end is not None else "",
        f"{source_col}_span": format(span, "g") if span is not None else "",
        f"{source_col}_mid": format(mid, "g") if mid is not None else "",
    }
    if replace_mode == "span":
        cleaned[source_col] = cleaned[f"{source_col}_span"]
    elif replace_mode == "mid":
        cleaned[source_col] = cleaned[f"{source_col}_mid"]
    elif replace_mode == "end":
        cleaned[source_col] = cleaned[f"{source_col}_end"]
    else:
        raise ValueError(f"Unsupported range replace_mode: {replace_mode}")
    return cleaned


def _normalize_text_list_value(value: str) -> str:
    text = _format_cell(value)
    if not text:
        return ""
    text = text.replace("\uff1b", ";")
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if len(parts) <= 1:
        return text
    normalized_parts = []
    for part in parts:
        if part not in normalized_parts:
            normalized_parts.append(part)
    return " | ".join(normalized_parts)


def _augment_full_cleaning(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    helper_headers: list[str] = []
    out_rows: list[dict[str, str]] = []

    for source_col, spec in GENERIC_NUMERIC_SEQUENCE_SPECS.items():
        helper_headers.extend(
            list(
                _clean_numeric_sequence_value(
                    {},
                    source_col=source_col,
                    limit=int(spec["limit"]),
                    replace_mode=str(spec.get("replace_mode", "mean")),
                ).keys()
            )
        )
    for source_col, spec in NUMERIC_RANGE_SPECS.items():
        helper_headers.extend(
            list(
                _clean_numeric_range_value(
                    {},
                    source_col=source_col,
                    replace_mode=str(spec.get("replace_mode", "span")),
                ).keys()
            )
        )
    for source_col, spec in MULTIVALUE_REDUCER_SPECS.items():
        helper_headers.extend(
            list(
                _clean_numeric_sequence_value(
                    {},
                    source_col=source_col,
                    limit=int(spec["limit"]),
                    replace_mode=str(spec.get("replace_mode", "mean")),
                ).keys()
            )
        )

    for source_row in rows:
        row = dict(source_row)
        for source_col, spec in GENERIC_NUMERIC_SEQUENCE_SPECS.items():
            if source_col in headers:
                row.update(
                    _clean_numeric_sequence_value(
                        row,
                        source_col=source_col,
                        limit=int(spec["limit"]),
                        replace_mode=str(spec.get("replace_mode", "mean")),
                    )
                )
        for source_col, spec in NUMERIC_RANGE_SPECS.items():
            if source_col in headers:
                row.update(
                    _clean_numeric_range_value(
                        row,
                        source_col=source_col,
                        replace_mode=str(spec.get("replace_mode", "span")),
                    )
                )
        for source_col, spec in MULTIVALUE_REDUCER_SPECS.items():
            if source_col in headers:
                row.update(
                    _clean_numeric_sequence_value(
                        row,
                        source_col=source_col,
                        limit=int(spec["limit"]),
                        replace_mode=str(spec.get("replace_mode", "mean")),
                    )
                )
        for source_col in TEXT_LIST_NORMALIZE_COLS:
            if source_col in headers:
                row[source_col] = _normalize_text_list_value(row.get(source_col, ""))
        out_rows.append(row)

    return _dedupe_keep_order(headers + helper_headers), out_rows


def find_default_input() -> Path:
    candidates = sorted(glob.glob(r"C:\Users\wangj\Desktop\*.xls"))
    for item in candidates:
        if DEFAULT_NAME_KEYWORD in os.path.basename(item):
            return Path(item)
    raise FileNotFoundError("Could not find the default 手工数据.xls workbook on the Desktop.")


def read_workbook(input_path: Path, sheet_name: str) -> tuple[list[str], list[dict[str, str]]]:
    book = xlrd.open_workbook(str(input_path))
    sheet = book.sheet_by_name(sheet_name)
    raw_headers = [_format_cell(sheet.cell_value(0, c)) for c in range(sheet.ncols)]

    kept_indices: list[int] = []
    headers: list[str] = []
    for idx, header in enumerate(raw_headers):
        if header:
            kept_indices.append(idx)
            headers.append(header)

    rows: list[dict[str, str]] = []
    for r in range(1, sheet.nrows):
        item: dict[str, str] = {}
        for idx, header in zip(kept_indices, headers):
            item[header] = _format_cell(sheet.cell_value(r, idx))
        rows.append(item)
    return headers, rows


def build_output_rows(headers: list[str], rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    output_rows: list[dict[str, str]] = []

    helper_columns = list(_schedule_flags("", "cure").keys())
    helper_columns += list(_schedule_flags("", "post_cure").keys())
    helper_columns += list(
        _expand_stage_group(
            {},
            "cure",
            ("cure_temperature_list_c", "cure_temperature_c"),
            ("cure_time_list_h", "cure_time_h"),
            ("cure_ramp_rate_list_c_min",),
            CURE_STAGE_LIMIT,
        ).keys()
    )
    helper_columns += list(
        _expand_stage_group(
            {},
            "post_cure",
            ("post_cure_temperature_list_c", "post_cure_temperature_c"),
            ("post_cure_time_list_h", "post_cure_time_h"),
            ("post_cure_ramp_rate_list_c_min",),
            POST_CURE_STAGE_LIMIT,
        ).keys()
    )
    for source_col, limit in SEQUENCE_COLUMN_SPECS.items():
        helper_columns += list(_expand_sequence_column({}, source_col, limit).keys())

    output_headers = headers + helper_columns

    for source_row in rows:
        row = dict(source_row)
        row.update(_schedule_flags(source_row.get("cure_schedule", ""), "cure"))
        row.update(_schedule_flags(source_row.get("post_cure_schedule", ""), "post_cure"))
        row.update(
            _expand_stage_group(
                source_row,
                prefix="cure",
                temp_cols=("cure_temperature_list_c", "cure_temperature_c"),
                time_cols=("cure_time_list_h", "cure_time_h"),
                ramp_cols=("cure_ramp_rate_list_c_min",),
                limit=CURE_STAGE_LIMIT,
            )
        )
        row.update(
            _expand_stage_group(
                source_row,
                prefix="post_cure",
                temp_cols=("post_cure_temperature_list_c", "post_cure_temperature_c"),
                time_cols=("post_cure_time_list_h", "post_cure_time_h"),
                ramp_cols=("post_cure_ramp_rate_list_c_min",),
                limit=POST_CURE_STAGE_LIMIT,
            )
        )
        for source_col, limit in SEQUENCE_COLUMN_SPECS.items():
            row.update(_expand_sequence_column(source_row, source_col, limit))
        output_rows.append(row)

    # Canonical process features are computed by the shared registry-aware
    # module so offline expansion and portal/workflow execution stay aligned.
    process_names = [
        "cure_stage_count", "cure_total_time_h", "cure_max_temperature_c",
        "cure_final_temperature_c", "cure_temp_time_integral_c_h",
        "cure_time_weighted_avg_temperature_c", "post_cure_stage_count",
        "post_cure_total_time_h", "post_cure_max_temperature_c",
        "post_cure_final_temperature_c", "post_cure_temp_time_integral_c_h",
        "post_cure_time_weighted_avg_temperature_c", "post_cure_temperature_c",
        "has_post_cure", "total_cure_stage_count", "total_heat_treatment_time_h",
        "overall_max_cure_temperature_c", "overall_temp_time_integral_c_h",
    ]
    process_definitions = []
    for name in process_names:
        fields = ["post_cure_schedule" if name.startswith("post_cure") else "cure_schedule"]
        if name.startswith("total_") or name.startswith("overall_"):
            fields = ["cure_schedule", "post_cure_schedule"]
        process_definitions.append({
            "name": name,
            "source_type": "derived_workflow",
            "calculation_rule": {
                "implementation": "core.process_features:derive_declared_feature",
                "version": "1",
                "input_fields": fields,
                "null_policy": "reject",
                "invalid_policy": "reject",
            },
        })
    process_frame = pd.DataFrame(rows)
    if not process_frame.empty and any(column in process_frame.columns for column in {"cure_schedule", "post_cure_schedule"}):
        bindings = [{"raw_column": field, "source_field": field} for field in ("cure_schedule", "post_cure_schedule") if field in process_frame.columns]
        result = compute_process_features(process_frame, process_definitions, {"source_bindings": bindings})
        for row_number, target in enumerate(output_rows):
            for name in process_names:
                if name in result.features.columns:
                    value = result.features.iloc[row_number][name]
                    target[name] = "" if value != value else format(float(value), "g")
        helper_columns.extend(process_names)

    output_headers = headers + helper_columns
    return _dedupe_keep_order(output_headers), output_rows


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_final_headers(headers: list[str], generated_headers: list[str], drop_source_sequence_cols: bool) -> list[str]:
    if not drop_source_sequence_cols:
        return _dedupe_keep_order(generated_headers)

    drop_cols = set(SEQUENCE_COLUMN_SPECS.keys())
    kept_original_headers = [header for header in headers if header not in drop_cols]
    generated_only_headers = [header for header in generated_headers if header not in set(headers)]
    return _dedupe_keep_order(kept_original_headers + generated_only_headers)


def refine_headers_for_training(
    headers: list[str],
    drop_cleaning_raw_cols: bool,
    drop_heavy_text_cols: bool,
) -> list[str]:
    refined = list(headers)
    if drop_cleaning_raw_cols:
        refined = [header for header in refined if not header.endswith("_raw")]
    if drop_heavy_text_cols:
        refined = [header for header in refined if header not in HEAVY_TEXT_DROP_COLS]
    return _dedupe_keep_order(refined)


def write_csv(output_path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand multi-stage cure/post-cure columns for training.")
    parser.add_argument("--input", type=str, default="", help="Path to the source .xls workbook.")
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Worksheet name to read.")
    parser.add_argument("--output", type=str, default="", help="Path to the output CSV file.")
    parser.add_argument(
        "--drop-source-sequence-cols",
        action="store_true",
        help="Drop original multi-value source columns such as cure_temperature_c and keep only expanded columns.",
    )
    parser.add_argument(
        "--fully-clean-multivalue-cols",
        action="store_true",
        help="Further clean remaining numeric multi-value columns into scalar + expanded helper columns.",
    )
    parser.add_argument(
        "--drop-cleaning-raw-cols",
        action="store_true",
        help="Drop helper audit columns ending with _raw in the final output.",
    )
    parser.add_argument(
        "--drop-heavy-text-cols",
        action="store_true",
        help="Drop long free-text process/notes columns that are usually unsuitable for direct training.",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_stage_model_ready.csv" if args.drop_source_sequence_cols else "_stage_expanded.csv"
        output_path = input_path.with_name(f"{input_path.stem}{suffix}")

    headers, rows = read_workbook(input_path, args.sheet)
    generated_headers, output_rows = build_output_rows(headers, rows)
    if args.fully_clean_multivalue_cols:
        generated_headers, output_rows = _augment_full_cleaning(generated_headers, output_rows)
    final_headers = build_final_headers(headers, generated_headers, args.drop_source_sequence_cols)
    final_headers = refine_headers_for_training(
        final_headers,
        drop_cleaning_raw_cols=args.drop_cleaning_raw_cols,
        drop_heavy_text_cols=args.drop_heavy_text_cols,
    )
    write_csv(output_path, final_headers, output_rows)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows={len(output_rows)}")
    print(f"original_columns={len(headers)}")
    print(f"output_columns={len(final_headers)}")
    print(f"added_columns={len(final_headers) - len(headers)}")
    print(f"drop_source_sequence_cols={args.drop_source_sequence_cols}")
    print(f"fully_clean_multivalue_cols={args.fully_clean_multivalue_cols}")
    print(f"drop_cleaning_raw_cols={args.drop_cleaning_raw_cols}")
    print(f"drop_heavy_text_cols={args.drop_heavy_text_cols}")


if __name__ == "__main__":
    main()
