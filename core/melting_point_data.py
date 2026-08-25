"""Parsing and cleaning utilities for melting-point training data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import math
import re

import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency fallback
    Chem = None


_VALUE = r"[-+]?\d+(?:[.,]\d+)?"
_UNIT = r"(?:°\s*|deg\s*|degrees?\s+)?(?:C|F|K|Celsius|Fahrenheit|Kelvin)\b"
_RANGE_RE = re.compile(
    rf"(?P<lower>{_VALUE})\s*(?P<lower_unit>{_UNIT})?\s*(?:-|–|—|to)\s*"
    rf"(?P<upper>{_VALUE})\s*(?P<upper_unit>{_UNIT})",
    re.IGNORECASE,
)
_VALUE_RE = re.compile(rf"(?P<value>{_VALUE})\s*(?P<unit>{_UNIT})", re.IGNORECASE)

_QUALITY_RANK = {
    "unparsed": 0,
    "range": 1,
    "estimated": 2,
    "mixture": 2,
    "decomp": 3,
    "high": 4,
}

MELTING_POINT_DATASET_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "melting_point_dataset"
)
MELTING_POINT_TARGET_COLUMN = "mp_c"
MELTING_POINT_TARGET_UNIT = "°C"
MELTING_POINT_TARGET_UNIT_CODE = "C"


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _to_celsius(value: float, unit: str | None) -> float:
    normalized = str(unit or "C").upper()
    normalized = (
        normalized.replace("°", "")
        .replace("DEGREES", "")
        .replace("DEG", "")
        .replace("CELSIUS", "C")
        .replace("FAHRENHEIT", "F")
        .replace("KELVIN", "K")
        .replace(" ", "")
    )
    if normalized == "F":
        return (value - 32) * 5 / 9
    if normalized == "K":
        return value - 273.15
    return value


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _empty_result(raw: object) -> dict:
    return {
        "mp_c": None,
        "mp_lower_c": None,
        "mp_upper_c": None,
        "mp_unit_raw": None,
        "mp_raw": raw,
        "mp_quality": "unparsed",
    }


def parse_melting_point_text(text: object) -> dict:
    """Parse a melting-point annotation into Celsius values and quality metadata."""
    result = _empty_result(text)
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return result
    raw = str(text).strip()
    if not raw:
        return result
    result["mp_raw"] = raw

    range_match = _RANGE_RE.search(raw)
    if range_match:
        lower_unit = range_match.group("lower_unit") or range_match.group("upper_unit")
        upper_unit = range_match.group("upper_unit") or lower_unit
        result["mp_lower_c"] = _to_celsius(_number(range_match.group("lower")), lower_unit)
        result["mp_upper_c"] = _to_celsius(_number(range_match.group("upper")), upper_unit)
        result["mp_unit_raw"] = lower_unit
        result["mp_quality"] = "range"
        return result

    value_match = _VALUE_RE.search(raw)
    if not value_match:
        numeric_match = re.fullmatch(rf"\s*(?P<value>{_VALUE})\s*", raw)
        if not numeric_match:
            return result
        result["mp_c"] = _number(numeric_match.group("value"))
        result["mp_quality"] = "high"
        return result

    unit = value_match.group("unit")
    result["mp_c"] = _to_celsius(_number(value_match.group("value")), unit)
    result["mp_unit_raw"] = unit
    lowered = raw.lower()
    if re.search(r"decomp|decompose", lowered):
        result["mp_quality"] = "decomp"
    elif re.search(r"mixture", lowered):
        result["mp_quality"] = "mixture"
    elif re.search(r"approx|about|soften|\bca\.?\b|~|[<>]", lowered):
        result["mp_quality"] = "estimated"
    else:
        result["mp_quality"] = "high"
    return result


def normalize_melting_point_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize all melting-point values to Celsius without losing provenance.

    ``mp_c`` is always Celsius after this function returns. A numeric value
    without an explicit unit is treated as Celsius, while ``mp_unit_raw`` is
    retained for auditability. Text annotations in ``mp_raw`` take precedence
    over numeric fallback columns when they can be parsed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    result = frame.copy().reset_index(drop=True)
    raw_values = result.get("mp_raw", pd.Series(None, index=result.index, dtype=object))
    explicit_values = pd.to_numeric(result.get("mp_c", pd.Series(None, index=result.index)), errors="coerce")
    explicit_lower = pd.to_numeric(result.get("mp_lower_c", pd.Series(None, index=result.index)), errors="coerce")
    explicit_upper = pd.to_numeric(result.get("mp_upper_c", pd.Series(None, index=result.index)), errors="coerce")
    explicit_units = result.get("mp_unit_raw", pd.Series(None, index=result.index, dtype=object))
    explicit_quality = result.get("mp_quality", pd.Series(None, index=result.index, dtype=object))

    normalized_values = []
    normalized_lower = []
    normalized_upper = []
    raw_units = []
    qualities = []

    for index in result.index:
        parsed = parse_melting_point_text(raw_values.loc[index]) if index in raw_values.index else _empty_result(None)
        parsed_value = _finite_float(parsed.get("mp_c"))
        parsed_lower_value = _finite_float(parsed.get("mp_lower_c"))
        parsed_upper_value = _finite_float(parsed.get("mp_upper_c"))
        parsed_unit = parsed.get("mp_unit_raw")
        explicit_unit = explicit_units.loc[index]
        if pd.isna(explicit_unit):
            explicit_unit = None
        unit = parsed_unit or explicit_unit or "C"
        unit = unit if str(unit).strip() else "C"

        if parsed_value is not None:
            value = parsed_value
        else:
            fallback_value = _finite_float(explicit_values.loc[index])
            value = _to_celsius(fallback_value, unit) if fallback_value is not None else None

        if parsed_lower_value is not None:
            lower = parsed_lower_value
        else:
            fallback_lower = _finite_float(explicit_lower.loc[index])
            lower = _to_celsius(fallback_lower, unit) if fallback_lower is not None else None

        if parsed_upper_value is not None:
            upper = parsed_upper_value
        else:
            fallback_upper = _finite_float(explicit_upper.loc[index])
            upper = _to_celsius(fallback_upper, unit) if fallback_upper is not None else None

        parsed_quality = str(parsed.get("mp_quality") or "unparsed").strip().lower()
        existing_quality_value = explicit_quality.loc[index]
        if pd.isna(existing_quality_value):
            existing_quality_value = None
        existing_quality = str(existing_quality_value or "").strip().lower()
        quality = parsed_quality if parsed_quality != "unparsed" else existing_quality
        if quality not in _QUALITY_RANK:
            quality = "high" if value is not None else "unparsed"

        normalized_values.append(value)
        normalized_lower.append(lower)
        normalized_upper.append(upper)
        raw_units.append(parsed_unit or explicit_unit)
        qualities.append(quality)

    result["mp_c"] = normalized_values
    result["mp_lower_c"] = normalized_lower
    result["mp_upper_c"] = normalized_upper
    result["mp_unit_raw"] = raw_units
    result["mp_unit_normalized"] = MELTING_POINT_TARGET_UNIT_CODE
    result["mp_unit"] = MELTING_POINT_TARGET_UNIT
    result["mp_quality"] = qualities
    return result


def _structure_flags(smiles: object) -> tuple[bool, bool]:
    """Return conservative ``(is_salt, is_mixture)`` structure flags."""
    if Chem is None or smiles is None:
        return False, False
    value = str(smiles).strip()
    if not value:
        return False, False
    try:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            return False, False
        fragments = Chem.GetMolFrags(molecule, asMols=False)
        is_mixture = len(fragments) > 1
        has_charge = any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms())
        return bool(is_mixture and has_charge), bool(is_mixture)
    except Exception:
        return False, False


def canonicalize_smiles(smiles: object) -> Optional[str]:
    """Return RDKit's canonical isomeric SMILES, or ``None`` when invalid."""
    if Chem is None or smiles is None:
        return None
    value = str(smiles).strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        molecule = Chem.MolFromSmiles(value)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else None
    except Exception:
        return None


def deduplicate_melting_point_records(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per structure/role/class, preferring better quality.

    The same molecule can legitimately be present in both the resin and
    hardener training subsets. Deduplicating on SMILES alone silently drops
    one of those labels, so role and hardener class are part of the key.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    if "hardener_class" in result.columns:
        result["hardener_class"] = result["hardener_class"].fillna("").astype(str)
    if "component_role" in result.columns:
        result["component_role"] = result["component_role"].fillna("").astype(str)
    if "canonical_smiles" not in result:
        result["canonical_smiles"] = result.get("smiles", pd.Series(index=result.index)).map(canonicalize_smiles)
    quality = result["mp_quality"] if "mp_quality" in result else pd.Series("unparsed", index=result.index)
    result["_quality_rank"] = quality.map(_QUALITY_RANK).fillna(0)
    source = result.get("source", pd.Series("", index=result.index)).fillna("").astype(str)
    source_name = result.get("source_name", pd.Series("", index=result.index)).fillna("").astype(str)
    result["_source_rank"] = (source.str.strip().ne("") | source_name.str.strip().ne("")).astype(int)
    result["_non_salt_rank"] = (~result.get("is_salt", pd.Series(False, index=result.index)).fillna(False).astype(bool)).astype(int)
    result["_non_mixture_rank"] = (~result.get("is_mixture", pd.Series(False, index=result.index)).fillna(False).astype(bool)).astype(int)
    component_role = result.get("component_role", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    hardener_class = result.get("hardener_class", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    result["_dedup_key"] = (
        result["canonical_smiles"].fillna("").astype(str)
        + "\x1f"
        + component_role
        + "\x1f"
        + hardener_class
    )
    result["_original_order"] = range(len(result))
    valid = result["canonical_smiles"].notna()
    valid_result = result[valid].sort_values(
        [
            "_dedup_key",
            "_quality_rank",
            "_source_rank",
            "_non_salt_rank",
            "_non_mixture_rank",
            "_original_order",
        ],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    )
    valid_result = valid_result.drop_duplicates("_dedup_key", keep="first")
    result = pd.concat([valid_result, result[~valid]], axis=0)
    return result.drop(
        columns=[
            "_quality_rank",
            "_source_rank",
            "_non_salt_rank",
            "_non_mixture_rank",
            "_dedup_key",
            "_original_order",
        ]
    ).reset_index(drop=True)

def _series_for_frame(frame: pd.DataFrame, column: str, default: object = None) -> pd.Series:
    """Return an index-aligned series without mutating the input frame."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def prepare_melting_point_dataset(
    raw_df: pd.DataFrame,
    include_low_quality: bool = False,
) -> pd.DataFrame:
    """Parse, validate, canonicalize, filter, and deduplicate raw records.

    ``raw_df`` is never modified in place. Range and unparsed records remain
    available in the raw table, while the returned table is intentionally
    limited to records with a finite single-value ``mp_c`` suitable for model
    training. By default only exact/high-quality annotations are retained;
    callers can opt into decomposition, estimated, and mixture annotations
    explicitly with ``include_low_quality``.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame")

    result = normalize_melting_point_units(raw_df)
    accepted_quality = {"high"}
    if include_low_quality:
        accepted_quality.update({"decomp", "estimated", "mixture"})
    result = result[result["mp_quality"].isin(accepted_quality)].copy()
    result = result[result["mp_c"].notna() & result["mp_c"].map(math.isfinite)].copy()

    result["canonical_smiles"] = _series_for_frame(result, "smiles").map(canonicalize_smiles)
    result = result[result["canonical_smiles"].notna()].copy()

    structure_flags = _series_for_frame(result, "smiles").map(_structure_flags)
    inferred_salt = structure_flags.map(lambda value: bool(value[0]))
    inferred_mixture = structure_flags.map(lambda value: bool(value[1]))
    if "is_salt" not in result.columns:
        result["is_salt"] = inferred_salt
    else:
        result["is_salt"] = result["is_salt"].fillna(inferred_salt).astype(bool)
    if "is_mixture" not in result.columns:
        result["is_mixture"] = inferred_mixture
    else:
        result["is_mixture"] = result["is_mixture"].fillna(inferred_mixture).astype(bool)

    return deduplicate_melting_point_records(result)


def _json_safe(value: object) -> object:
    """Convert pandas/numpy scalar containers into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def persist_melting_point_dataset(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    summary: Optional[dict] = None,
    output_dir: Optional[Path | str] = None,
) -> dict[str, str]:
    """Atomically persist raw records, cleaned training data, and summary."""
    if not isinstance(raw_df, pd.DataFrame) or not isinstance(cleaned_df, pd.DataFrame):
        raise TypeError("raw_df and cleaned_df must be pandas DataFrames")

    destination = Path(output_dir) if output_dir is not None else MELTING_POINT_DATASET_DIR
    destination.mkdir(parents=True, exist_ok=True)
    if summary is None:
        summary = summarize_melting_point_dataset(cleaned_df)

    paths = {
        "raw_path": destination / "melting_point_raw_records.csv",
        "cleaned_path": destination / "melting_point_training_dataset.csv",
        "summary_path": destination / "melting_point_dataset_summary.json",
    }

    normalized_raw_df = normalize_melting_point_units(raw_df)

    def atomic_write(path: Path, writer) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            writer(temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    atomic_write(paths["raw_path"], lambda path: normalized_raw_df.to_csv(path, index=False, encoding="utf-8-sig"))
    atomic_write(paths["cleaned_path"], lambda path: cleaned_df.to_csv(path, index=False, encoding="utf-8-sig"))
    atomic_write(
        paths["summary_path"],
        lambda path: path.write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        ),
    )
    return {key: str(path) for key, path in paths.items()}


def load_persisted_melting_point_dataset(
    output_dir: Optional[Path | str] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """Load the persisted raw table, cleaned table, and summary if available."""
    destination = Path(output_dir) if output_dir is not None else MELTING_POINT_DATASET_DIR
    raw_path = destination / "melting_point_raw_records.csv"
    cleaned_path = destination / "melting_point_training_dataset.csv"
    summary_path = destination / "melting_point_dataset_summary.json"

    raw_df = pd.read_csv(raw_path, encoding="utf-8-sig") if raw_path.exists() else None
    cleaned_df = pd.read_csv(cleaned_path, encoding="utf-8-sig") if cleaned_path.exists() else None
    summary: dict = {}
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            summary = {}
    return raw_df, cleaned_df, summary


__all__ = [
    "MELTING_POINT_DATASET_DIR",
    "MELTING_POINT_TARGET_COLUMN",
    "MELTING_POINT_TARGET_UNIT",
    "MELTING_POINT_TARGET_UNIT_CODE",
    "canonicalize_smiles",
    "deduplicate_melting_point_records",
    "load_persisted_melting_point_dataset",
    "normalize_melting_point_units",
    "parse_melting_point_text",
    "persist_melting_point_dataset",
    "prepare_melting_point_dataset",
    "summarize_melting_point_dataset",
]

def summarize_melting_point_dataset(df: pd.DataFrame) -> dict:
    """Return compact counts for a cleaned melting-point dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    qualities = df.get("mp_quality", pd.Series(dtype=object))
    roles = df.get("component_role", pd.Series(dtype=object))
    return {
        "row_count": int(len(df)),
        "target_column": MELTING_POINT_TARGET_COLUMN,
        "target_unit": MELTING_POINT_TARGET_UNIT_CODE,
        "target_unit_display": MELTING_POINT_TARGET_UNIT,
        "high_quality_count": int((qualities == "high").sum()),
        "quality_counts": qualities.value_counts(dropna=False).to_dict(),
        "role_counts": roles.value_counts(dropna=False).to_dict(),
    }
