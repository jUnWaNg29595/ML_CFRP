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

MELTING_POINT_COMPONENT_ROLES = ("resin", "hardener", "unknown")
MELTING_POINT_HARDENER_CLASSES = ("胺", "酸酐", "酚", "硫醇", "咪唑")
MELTING_POINT_TRAINING_CATEGORIES = ("环氧树脂", "胺", "酸酐", "酚", "硫醇", "咪唑")
MELTING_POINT_COMPONENT_CATEGORIES = MELTING_POINT_TRAINING_CATEGORIES + ("未分类",)

_ROLE_ALIASES = {
    "resin": "resin",
    "epoxy": "resin",
    "epoxy resin": "resin",
    "树脂": "resin",
    "环氧树脂": "resin",
    "基体": "resin",
    "hardener": "hardener",
    "curing agent": "hardener",
    "curing_agent": "hardener",
    "curative": "hardener",
    "固化剂": "hardener",
    "交联剂": "hardener",
    "unknown": "unknown",
    "general": "unknown",
    "generic": "unknown",
    "unclassified": "unknown",
    "other": "unknown",
    "未分类": "unknown",
    "未知": "unknown",
}
_HARDENER_CLASS_ALIASES = {
    "胺": "胺",
    "amine": "胺",
    "amines": "胺",
    "primary amine": "胺",
    "secondary amine": "胺",
    "酸酐": "酸酐",
    "anhydride": "酸酐",
    "anhydrides": "酸酐",
    "酚": "酚",
    "phenol": "酚",
    "phenols": "酚",
    "硫醇": "硫醇",
    "thiol": "硫醇",
    "thiols": "硫醇",
    "巯基": "硫醇",
    "咪唑": "咪唑",
    "imidazole": "咪唑",
    "imidazoles": "咪唑",
    "其他": "其他",
    "other": "其他",
}
_RESIN_ROLE_HINTS = ("resin", "epoxy", "树脂", "环氧", "基体")
_HARDENER_ROLE_HINTS = (
    "hardener",
    "curing agent",
    "curing_agent",
    "curative",
    "固化剂",
    "交联剂",
)
_HARDENER_CLASS_HINTS = {
    "胺": ("amine", "amines", "胺"),
    "酸酐": ("anhydride", "anhydrides", "酸酐"),
    "酚": ("phenol", "phenols", "酚"),
    "硫醇": ("thiol", "thiols", "巯基", "硫醇"),
    "咪唑": ("imidazole", "imidazoles", "咪唑"),
}
_CATEGORY_ALIASES = {
    "环氧树脂": "环氧树脂",
    "resin": "环氧树脂",
    "epoxy": "环氧树脂",
    "epoxy resin": "环氧树脂",
    "树脂": "环氧树脂",
    "环氧": "环氧树脂",
    "胺": "胺",
    "amine": "胺",
    "amines": "胺",
    "酸酐": "酸酐",
    "anhydride": "酸酐",
    "anhydrides": "酸酐",
    "酚": "酚",
    "phenol": "酚",
    "phenols": "酚",
    "硫醇": "硫醇",
    "thiol": "硫醇",
    "thiols": "硫醇",
    "巯基": "硫醇",
    "咪唑": "咪唑",
    "imidazole": "咪唑",
    "imidazoles": "咪唑",
    "其他": "未分类",
    "other": "未分类",
    "未分类": "未分类",
    "unknown": "未分类",
    "general": "未分类",
}
_STRUCTURE_CATEGORY_SMARTS = {
    "环氧树脂": "C1OC1",
    "胺": "[NX3;H1,H2;!$(N-C=O)]",
    "酸酐": "C(=O)OC(=O)",
    "酚": "[c][OX2H]",
    "硫醇": "[SX2H]",
    "咪唑": "[nH]1ccnc1",
}


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


def _normalized_label(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower().replace("-", " ").replace("/", " ")


def normalize_component_role(value: object) -> str:
    """Normalize role labels without inferring a role from a bare SMILES string."""
    normalized = _normalized_label(value)
    if normalized in _ROLE_ALIASES:
        return _ROLE_ALIASES[normalized]
    if normalized in {"resin", "hardener", "unknown"}:
        return normalized
    return "unknown"


def normalize_hardener_class(value: object) -> str:
    """Normalize hardener subclasses to the UI's canonical Chinese labels."""
    normalized = _normalized_label(value)
    if not normalized:
        return ""
    return _HARDENER_CLASS_ALIASES.get(normalized, str(value).strip())


def normalize_melting_point_category(value: object) -> str:
    """Normalize the six supervised MP categories plus explicit unknown labels."""
    normalized = _normalized_label(value)
    if not normalized:
        return ""
    return _CATEGORY_ALIASES.get(normalized, str(value).strip())


def _infer_structure_category(smiles: object) -> str:
    """Return a category only when exactly one supported SMARTS matches."""
    if Chem is None or smiles is None:
        return ""
    try:
        molecule = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        molecule = None
    if molecule is None:
        return ""
    matched = []
    for category, smarts in _STRUCTURE_CATEGORY_SMARTS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and molecule.HasSubstructMatch(pattern):
                matched.append(category)
        except Exception:
            continue
    return matched[0] if len(matched) == 1 else ""


def classify_melting_point_records(
    frame: pd.DataFrame,
    *,
    infer_from_structure: bool = False,
) -> pd.DataFrame:
    """Standardize role/class metadata and conservatively classify explicit records.

    PubChem records already carry a role and hardener class. For manually imported
    data, text metadata is preferred. When ``infer_from_structure`` is enabled,
    a molecule is assigned only if exactly one of the six supported SMARTS rules
    matches; ambiguous or unmatched structures remain ``未分类``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = frame.copy()
    if "component_role" not in result.columns:
        result["component_role"] = "unknown"
    if "hardener_class" not in result.columns:
        result["hardener_class"] = ""
    if "component_category" not in result.columns:
        result["component_category"] = ""

    roles = result["component_role"].map(normalize_component_role)
    classes = result["hardener_class"].map(normalize_hardener_class)
    categories = result["component_category"].map(normalize_melting_point_category)
    hint_columns = [
        column
        for column in (
            "source",
            "source_name",
            "source_category",
            "category",
            "component_name",
            "name",
            "title",
            "description",
            "query_smarts",
            "component_category",
        )
        if column in result.columns
    ]
    if hint_columns:
        hints = result[hint_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    else:
        hints = pd.Series("", index=result.index, dtype=object)

    inferred_classes = classes.copy()
    for category in MELTING_POINT_TRAINING_CATEGORIES:
        mask = categories.eq(category)
        if category != "环氧树脂":
            inferred_classes.loc[mask] = category
    for hardener_class, tokens in _HARDENER_CLASS_HINTS.items():
        mask = inferred_classes.eq("") & hints.map(
            lambda value: any(token in value for token in tokens)
        )
        inferred_classes.loc[mask] = hardener_class

    inferred_roles = roles.copy()
    inferred_roles.loc[categories.eq("环氧树脂")] = "resin"
    hardener_mask = inferred_classes.isin(MELTING_POINT_HARDENER_CLASSES) & ~inferred_roles.eq("resin")
    inferred_roles.loc[hardener_mask] = "hardener"
    unknown_mask = inferred_roles.eq("unknown")
    inferred_roles.loc[unknown_mask & hints.map(lambda value: any(token in value for token in _RESIN_ROLE_HINTS))] = "resin"
    unknown_mask = inferred_roles.eq("unknown")
    inferred_roles.loc[unknown_mask & hints.map(lambda value: any(token in value for token in _HARDENER_ROLE_HINTS))] = "hardener"

    if infer_from_structure:
        unresolved = categories.isin(["", "未分类"]) & (
            inferred_roles.eq("unknown")
            | (inferred_roles.eq("hardener") & ~inferred_classes.isin(MELTING_POINT_HARDENER_CLASSES))
        )
        structure_categories = result["smiles"].map(_infer_structure_category) if "smiles" in result.columns else pd.Series("", index=result.index)
        categories.loc[unresolved] = structure_categories.loc[unresolved]
        for category in MELTING_POINT_TRAINING_CATEGORIES:
            mask = unresolved & categories.eq(category)
            if category == "环氧树脂":
                inferred_roles.loc[mask] = "resin"
            else:
                inferred_roles.loc[mask] = "hardener"
                inferred_classes.loc[mask] = category

    categories.loc[inferred_roles.eq("resin")] = "环氧树脂"
    hardener_category_mask = inferred_roles.eq("hardener")
    supported_hardener_mask = hardener_category_mask & inferred_classes.isin(MELTING_POINT_HARDENER_CLASSES)
    categories.loc[supported_hardener_mask] = inferred_classes.loc[
        supported_hardener_mask
    ]
    categories.loc[hardener_category_mask & ~supported_hardener_mask] = "未分类"
    categories.loc[categories.eq("")] = "未分类"

    result["component_role"] = inferred_roles
    result["hardener_class"] = inferred_classes
    result["component_category"] = categories
    return result

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
    return classify_melting_point_records(result)


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
    infer_from_structure: bool = True,
    max_melting_point_c: Optional[float] = 500.0,
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

    result = classify_melting_point_records(
        normalize_melting_point_units(raw_df),
        infer_from_structure=bool(infer_from_structure),
    )
    accepted_quality = {"high"}
    if include_low_quality:
        accepted_quality.update({"decomp", "estimated", "mixture"})
    result = result[result["mp_quality"].isin(accepted_quality)].copy()
    result = result[result["mp_c"].notna() & result["mp_c"].map(math.isfinite)].copy()
    if max_melting_point_c is not None:
        result = result[result["mp_c"] <= float(max_melting_point_c)].copy()

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


def filter_melting_point_training_dataset(
    frame: pd.DataFrame,
    *,
    scope: str = "resin",
    hardener_classes: Optional[list[str] | tuple[str, ...] | set[str]] = None,
    category: Optional[str] = None,
    infer_from_structure: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Select a role-specific melting-point training subset.

    ``scope`` accepts ``resin``, ``hardener``, ``hardener_class``, ``category`` and ``all``.
    Unknown-role rows are retained only for ``all`` and are otherwise reported
    rather than silently mixed into a component model.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    normalized = classify_melting_point_records(
        normalize_melting_point_units(frame),
        infer_from_structure=infer_from_structure,
    )
    normalized["mp_c"] = pd.to_numeric(normalized.get("mp_c"), errors="coerce")
    normalized = normalized[normalized["mp_c"].map(lambda value: _finite_float(value) is not None)].copy()
    scope_value = _normalized_label(scope).replace(" ", "_")
    scope_aliases = {
        "树脂": "resin",
        "环氧树脂": "resin",
        "固化剂": "hardener",
        "固化剂类别": "hardener_class",
        "类别": "category",
        "全部": "all",
    }
    scope_value = scope_aliases.get(scope_value, scope_value)
    if scope_value not in {"resin", "hardener", "hardener_class", "category", "all"}:
        raise ValueError(f"unsupported melting-point training scope: {scope}")

    selected_classes = {
        normalize_hardener_class(value)
        for value in (hardener_classes or [])
        if normalize_hardener_class(value)
    }
    roles = normalized["component_role"].astype(str)
    classes = normalized["hardener_class"].astype(str)
    categories = normalized["component_category"].astype(str)
    if scope_value == "resin":
        mask = roles.eq("resin")
    elif scope_value == "hardener":
        mask = roles.eq("hardener")
    elif scope_value == "hardener_class":
        mask = roles.eq("hardener") & classes.isin(selected_classes)
    elif scope_value == "category":
        selected_category = normalize_melting_point_category(category)
        if selected_category not in MELTING_POINT_TRAINING_CATEGORIES:
            raise ValueError(f"unsupported melting-point category: {category}")
        mask = categories.eq(selected_category)
    else:
        mask = pd.Series(True, index=normalized.index)

    selected = normalized.loc[mask].reset_index(drop=True)
    role_counts = roles.value_counts(dropna=False).to_dict()
    class_counts = classes[roles.eq("hardener")].value_counts(dropna=False).to_dict()
    report = {
        "scope": scope_value,
        "selected_hardener_classes": sorted(selected_classes),
        "selected_category": normalize_melting_point_category(category) if category else "",
        "source_count": int(len(normalized)),
        "selected_count": int(len(selected)),
        "unknown_role_count": int(roles.eq("unknown").sum()),
        "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "hardener_class_counts": {str(key): int(value) for key, value in class_counts.items()},
    }
    return selected, report


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
    paths = {
        "raw_path": destination / "melting_point_raw_records.csv",
        "cleaned_path": destination / "melting_point_training_dataset.csv",
        "summary_path": destination / "melting_point_dataset_summary.json",
    }

    normalized_raw_df = classify_melting_point_records(
        normalize_melting_point_units(raw_df),
        infer_from_structure=True,
    )
    normalized_cleaned_df = classify_melting_point_records(
        normalize_melting_point_units(cleaned_df),
        infer_from_structure=True,
    )
    if summary is None:
        summary = summarize_melting_point_dataset(normalized_cleaned_df)

    def atomic_write(path: Path, writer) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            writer(temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    atomic_write(paths["raw_path"], lambda path: normalized_raw_df.to_csv(path, index=False, encoding="utf-8-sig"))
    atomic_write(paths["cleaned_path"], lambda path: normalized_cleaned_df.to_csv(path, index=False, encoding="utf-8-sig"))
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
    if isinstance(raw_df, pd.DataFrame):
        raw_df = classify_melting_point_records(
            normalize_melting_point_units(raw_df),
            infer_from_structure=True,
        )
    if isinstance(cleaned_df, pd.DataFrame):
        cleaned_df = classify_melting_point_records(
            normalize_melting_point_units(cleaned_df),
            infer_from_structure=True,
        )
    summary: dict = {}
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            summary = {}
    if isinstance(cleaned_df, pd.DataFrame) and not cleaned_df.empty:
        summary = summarize_melting_point_dataset(cleaned_df)
    return raw_df, cleaned_df, summary


__all__ = [
    "MELTING_POINT_DATASET_DIR",
    "MELTING_POINT_COMPONENT_ROLES",
    "MELTING_POINT_HARDENER_CLASSES",
    "MELTING_POINT_TARGET_COLUMN",
    "MELTING_POINT_TARGET_UNIT",
    "MELTING_POINT_TARGET_UNIT_CODE",
    "canonicalize_smiles",
    "classify_melting_point_records",
    "deduplicate_melting_point_records",
    "filter_melting_point_training_dataset",
    "load_persisted_melting_point_dataset",
    "normalize_component_role",
    "normalize_hardener_class",
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
    classified = classify_melting_point_records(df, infer_from_structure=True)
    qualities = classified.get("mp_quality", pd.Series(dtype=object))
    roles = classified.get("component_role", pd.Series(dtype=object))
    classes = classified.get("hardener_class", pd.Series("", index=classified.index, dtype=object))
    numeric_values = pd.to_numeric(classified.get("mp_c"), errors="coerce")

    def _stats(mask: pd.Series) -> dict:
        values = numeric_values.loc[mask].dropna()
        if values.empty:
            return {"count": 0, "mean_c": None, "median_c": None, "min_c": None, "max_c": None}
        return {
            "count": int(values.size),
            "mean_c": float(values.mean()),
            "median_c": float(values.median()),
            "min_c": float(values.min()),
            "max_c": float(values.max()),
        }

    role_stats = {
        str(role): _stats(roles.eq(role))
        for role in sorted(roles.dropna().astype(str).unique().tolist())
    }
    class_stats = {
        str(hardener_class): _stats(classes.eq(hardener_class) & roles.eq("hardener"))
        for hardener_class in sorted(
            value for value in classes.dropna().astype(str).unique().tolist() if value
        )
    }
    categories = classified.get(
        "component_category",
        pd.Series("未分类", index=classified.index, dtype=object),
    ).fillna("未分类").astype(str)
    category_counts = {
        category: int(categories.eq(category).sum())
        for category in MELTING_POINT_COMPONENT_CATEGORIES
    }
    return {
        "row_count": int(len(classified)),
        "target_column": MELTING_POINT_TARGET_COLUMN,
        "target_unit": MELTING_POINT_TARGET_UNIT_CODE,
        "target_unit_display": MELTING_POINT_TARGET_UNIT,
        "high_quality_count": int((qualities == "high").sum()),
        "quality_counts": qualities.value_counts(dropna=False).to_dict(),
        "role_counts": roles.value_counts(dropna=False).to_dict(),
        "hardener_class_counts": classes[roles.eq("hardener")].value_counts(dropna=False).to_dict(),
        "role_stats": role_stats,
        "hardener_class_stats": class_stats,
        "component_category_counts": category_counts,
        "unknown_role_count": int(roles.eq("unknown").sum()),
    }
