#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter rows containing uncertain molecular formulas like "*NCCN*".
Default behavior removes any row with an asterisk in string columns.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            errors.append((enc, exc))
    try:
        return pd.read_csv(path)
    except Exception as exc:
        detail = "; ".join([f"{enc}: {err}" for enc, err in errors])
        raise RuntimeError(f"Failed to read CSV: {path} ({detail})") from exc


def _parse_columns(items: list[str] | None) -> list[str]:
    cols: list[str] = []
    for item in items or []:
        if not item:
            continue
        cols.extend([c.strip() for c in item.split(",") if c.strip()])
    return cols


def _resolve_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Warning: columns not found and will be ignored: {missing}", file=sys.stderr)
        return [c for c in cols if c in df.columns]
    return [
        c for c in df.columns
        if df[c].dtype == object or getattr(df[c].dtype, "name", "") == "string"
    ]


def _build_mask(df: pd.DataFrame, cols: list[str], regex: re.Pattern) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for c in cols:
        s = df[c].astype(str)
        mask |= s.str.contains(regex, na=False)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove rows with uncertain molecular formulas (e.g., *NCCN*)."
    )
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("-o", "--output", help="Output CSV path")
    parser.add_argument(
        "-c",
        "--columns",
        action="append",
        help="Columns to scan (repeat or comma-separated). Default: all string columns.",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default=r"\*",
        help=r"Regex pattern to detect uncertain formulas (default: '\*').",
    )
    parser.add_argument("--ignore-case", action="store_true", help="Case-insensitive match")
    parser.add_argument("--report-only", action="store_true", help="Only report counts")

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 2

    df = _read_csv(in_path)
    cols = _resolve_columns(df, _parse_columns(args.columns))
    if not cols:
        print("No columns available to scan.", file=sys.stderr)
        return 2

    flags = re.IGNORECASE if args.ignore_case else 0
    regex = re.compile(args.pattern, flags=flags)
    mask = _build_mask(df, cols, regex)

    total = int(len(df))
    removed = int(mask.sum())
    kept = total - removed
    print(f"Rows total: {total}")
    print(f"Rows removed: {removed}")
    print(f"Rows kept: {kept}")

    if args.report_only:
        return 0

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_filtered.csv")
    df.loc[~mask].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
