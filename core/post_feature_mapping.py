"""Canonical manual mapping for post-feature model inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


POST_FEATURE_MAPPING_SCHEMA_VERSION = 1
SOURCE_TYPES = ("pending", "computed", "candidate", "constant", "unused", "keep")
_UNSAFE_FEATURE_TEXT_MARKERS = (
    "DeltaGenerator",
    "LockedCursor",
    "RunningCursor",
    "ScriptRunContext",
    "page_virtual_screening()",
    "<function ",
)
_UNSAFE_FEATURE_COMPACT_MARKERS = (
    "deltagenerator",
    "lockedcursor",
    "runningcursor",
    "scriptruncontext",
    "page_virtual_screening",
)
MAPPING_SESSION_KEYS = frozenset(
    {
        "post_feature_mapping_default",
        "post_feature_mapping_draft",
        "post_feature_mapping_confirmation",
        "post_feature_mapping_model_fingerprint",
        "post_feature_mapping_catalog_fingerprint",
        "post_feature_mapping_snapshot",
    }
)


def sanitize_feature_columns(
    values: Any,
    *,
    return_rejected: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """Keep only stable text column names and never stringify arbitrary objects."""
    if values is None:
        raw_values = []
    elif isinstance(values, str):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = []

    columns: list[str] = []
    rejected_types: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            type_name = type(value).__name__
            if type_name not in rejected_types:
                rejected_types.append(type_name)
            continue
        column = value.strip()
        compact_column = "".join(column.split()).lower()
        if (
            any(marker in column for marker in _UNSAFE_FEATURE_TEXT_MARKERS)
            or any(marker in compact_column for marker in _UNSAFE_FEATURE_COMPACT_MARKERS)
        ):
            if "Streamlit内部对象文本" not in rejected_types:
                rejected_types.append("Streamlit内部对象文本")
            continue
        if column and column not in seen:
            seen.add(column)
            columns.append(column)

    if return_rejected:
        return columns, rejected_types
    return columns


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _feature_cols(values: Any) -> list[str]:
    return sanitize_feature_columns(values)


def _finite_numeric(value: Any) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted) or not np.isfinite(float(converted)):
        return None
    return float(converted)


def _normalize_rule(rule: Any) -> dict[str, Any]:
    rule = rule if isinstance(rule, dict) else {}
    source_type = _safe_text(rule.get("source_type"))
    return {
        "source_type": source_type,
        "source_column": _safe_text(rule.get("source_column")),
        "constant_value": rule.get("constant_value"),
        "unit": _safe_text(rule.get("unit")),
        "definition": _safe_text(rule.get("definition")),
        "confirmed": bool(rule.get("confirmed", False)),
    }


def normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic mapping representation without guessing sources."""
    payload = mapping if isinstance(mapping, dict) else {}
    model_cols = _feature_cols(payload.get("model_feature_cols"))
    source_rules = payload.get("rules")
    source_rules = source_rules if isinstance(source_rules, dict) else {}
    rules = {column: _normalize_rule(source_rules.get(column)) for column in model_cols}
    try:
        schema_version = int(payload.get("schema_version") or POST_FEATURE_MAPPING_SCHEMA_VERSION)
    except (TypeError, ValueError):
        schema_version = POST_FEATURE_MAPPING_SCHEMA_VERSION
    status = _safe_text(payload.get("status"))
    model_fingerprint = _safe_text(payload.get("model_fingerprint"))
    workflow_fingerprint = _safe_text(payload.get("workflow_fingerprint"))
    catalog_fingerprint_value = _safe_text(payload.get("catalog_fingerprint"))
    return {
        "schema_version": schema_version,
        "model_feature_cols": model_cols,
        "rules": rules,
        "confirmed": bool(payload.get("confirmed", False)),
        "status": status or ("confirmed" if payload.get("confirmed") else "draft"),
        "model_fingerprint": model_fingerprint,
        "workflow_fingerprint": workflow_fingerprint,
        "catalog_fingerprint": catalog_fingerprint_value,
    }


def build_post_feature_catalog(
    candidate_df: pd.DataFrame,
    *,
    computed_definitions: dict[str, dict[str, object]],
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Build a stable, explicit catalog for candidate-side columns."""
    if not isinstance(candidate_df, pd.DataFrame):
        raise TypeError("candidate_df must be a pandas DataFrame")
    definitions = computed_definitions if isinstance(computed_definitions, dict) else {}
    excluded = set(sanitize_feature_columns(excluded_columns or set()))
    rows: list[dict[str, Any]] = []
    row_count = len(candidate_df)
    for column in candidate_df.columns:
        column = _safe_text(column)
        if column is None:
            continue
        if column in excluded:
            continue
        definition = definitions.get(column) or {}
        source_type = "computed" if column in definitions else "candidate"
        numeric = pd.to_numeric(candidate_df[column], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan)
        valid = finite.notna()
        rows.append(
            {
                "column": column,
                "source_type": source_type,
                "category": definition.get("category") if definition else "未分类",
                "unit": definition.get("unit") if definition else "未声明",
                "definition": definition.get("definition") if definition else "原始候选列",
                "dtype": str(candidate_df[column].dtype),
                "valid_ratio": float(valid.mean()) if row_count else 0.0,
                "min": float(finite.min()) if valid.any() else None,
                "max": float(finite.max()) if valid.any() else None,
                "row_count": row_count,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "source_type",
            "category",
            "unit",
            "definition",
            "dtype",
            "valid_ratio",
            "min",
            "max",
            "row_count",
        ],
    )


def build_manual_mapping_choices(
    catalog: pd.DataFrame,
    *,
    include_constant: bool = True,
    include_unused: bool = True,
) -> list[dict[str, Any]]:
    """Build compact, explicit choices for the pre-screen mapping panel."""
    choices: list[dict[str, Any]] = [
        {
            "label": "请选择来源",
            "source_type": "pending",
            "source_column": None,
        }
    ]
    if isinstance(catalog, pd.DataFrame):
        for row in catalog.to_dict(orient="records"):
            column = _safe_text(row.get("column"))
            source_type = _safe_text(row.get("source_type"))
            if not column or source_type not in {"computed", "candidate"}:
                continue
            if source_type == "computed":
                category = _safe_text(row.get("category"))
                label = f"计算列：{column}"
                if category and category != "未分类":
                    label += f"（{category}）"
            else:
                label = f"原始列：{column}"
            choices.append(
                {
                    "label": label,
                    "source_type": source_type,
                    "source_column": column,
                }
            )
    if include_constant:
        choices.append(
            {
                "label": "常数值（手动输入）",
                "source_type": "constant",
                "source_column": None,
            }
        )
    if include_unused:
        choices.append(
            {
                "label": "不使用（仅在模型允许缺失时）",
                "source_type": "unused",
                "source_column": None,
            }
        )
    return choices


def create_mapping_draft(
    model_feature_cols: list[str],
    *,
    molecular_feature_cols: list[str],
    catalog: pd.DataFrame,
    model_fingerprint: str,
    workflow_fingerprint: str | None,
) -> dict[str, Any]:
    """Create an explicitly blank draft; no same-name matching is performed."""
    del molecular_feature_cols
    columns = _feature_cols(model_feature_cols)
    return {
        "schema_version": POST_FEATURE_MAPPING_SCHEMA_VERSION,
        "model_feature_cols": columns,
        "rules": {
            column: {
                "source_type": "pending",
                "source_column": None,
                "constant_value": None,
                "unit": None,
                "definition": None,
                "confirmed": False,
            }
            for column in columns
        },
        "confirmed": False,
        "status": "draft",
        "model_fingerprint": model_fingerprint,
        "workflow_fingerprint": workflow_fingerprint,
        "catalog_fingerprint": catalog_fingerprint(catalog),
    }


def validate_mapping(
    mapping: dict,
    *,
    model_feature_cols: list[str],
    candidate_df: pd.DataFrame,
    catalog: pd.DataFrame,
    missing_input_tolerant: bool,
) -> dict[str, Any]:
    """Validate every mapping decision before a model matrix can be produced."""
    expected_cols = _feature_cols(model_feature_cols)
    normalized = normalize_mapping(mapping)
    errors: list[str] = []
    warnings: list[str] = []
    if normalized["schema_version"] != POST_FEATURE_MAPPING_SCHEMA_VERSION:
        errors.append("mapping schema version is unsupported")
    if normalized["model_feature_cols"] != expected_cols:
        errors.append("model feature order does not match the current model")
    if not bool(mapping.get("confirmed", False)):
        errors.append("mapping has not been confirmed")
    catalog_rows = (
        catalog.set_index("column").to_dict("index")
        if isinstance(catalog, pd.DataFrame) and "column" in catalog.columns
        else {}
    )
    rules = normalized["rules"]
    for feature in expected_cols:
        rule = rules[feature]
        source_type = rule["source_type"]
        if source_type not in SOURCE_TYPES or source_type == "pending":
            errors.append(f"{feature}: source is not explicitly selected")
            continue
        if not rule["confirmed"]:
            errors.append(f"{feature}: mapping rule is not confirmed")
        if source_type in {"computed", "candidate"}:
            source_column = rule["source_column"]
            if not source_column or source_column not in candidate_df.columns:
                errors.append(f"{feature}: source column {source_column!r} is missing")
                continue
            catalog_row = catalog_rows.get(source_column)
            if catalog_row is None:
                errors.append(f"{feature}: source column {source_column!r} is not in the catalog")
            elif source_type == "computed" and catalog_row.get("source_type") != "computed":
                errors.append(f"{feature}: source column {source_column!r} is not a computed column")
            values = pd.to_numeric(candidate_df[source_column], errors="coerce")
            if values.isna().any():
                errors.append(f"{feature}: source column {source_column!r} contains nonnumeric or missing values")
            elif not np.isfinite(values.to_numpy(dtype=float)).all():
                errors.append(f"{feature}: source column {source_column!r} contains nonfinite values")
        elif source_type == "constant":
            if _finite_numeric(rule["constant_value"]) is None:
                errors.append(f"{feature}: constant value must be finite numeric")
        elif source_type == "unused":
            if not missing_input_tolerant:
                errors.append(f"{feature}: unused is not allowed for this model input")
            else:
                warnings.append(f"{feature}: explicitly unused; downstream preprocessing must allow missing input")
        elif source_type == "keep":
            warnings.append(f"{feature}: keep existing candidate feature value")
    mapping_hash = mapping_fingerprint(normalized)
    catalog_hash = catalog_fingerprint(catalog)
    stored_catalog_hash = mapping.get("catalog_fingerprint")
    if stored_catalog_hash and stored_catalog_hash != catalog_hash:
        errors.append("candidate catalog fingerprint changed; mapping must be reconfirmed")
    return {
        "ok": not errors,
        "status": "confirmed" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "mapped_feature_cols": expected_cols,
        "mapping_hash": mapping_hash,
        "catalog_fingerprint": catalog_hash,
    }


def validate_mapping_for_prediction(*args, **kwargs) -> dict[str, Any]:
    """Shared prediction boundary used by ordinary and formulation screening."""
    return validate_mapping(*args, **kwargs)


def apply_mapping(
    base_matrix: pd.DataFrame,
    candidate_df: pd.DataFrame,
    mapping: dict,
    *,
    model_feature_cols: list[str],
) -> pd.DataFrame:
    """Apply only explicitly selected sources and restore model feature order."""
    result = base_matrix.copy()
    if len(result) != len(candidate_df):
        raise ValueError("base_matrix and candidate_df row counts differ")
    normalized = normalize_mapping(mapping)
    for feature in _feature_cols(model_feature_cols):
        rule = normalized["rules"][feature]
        source_type = rule["source_type"]
        if source_type in {"computed", "candidate"}:
            source_column = rule["source_column"]
            if not source_column or source_column not in candidate_df.columns:
                raise ValueError(f"{feature}: source column {source_column!r} is missing")
            values = pd.to_numeric(candidate_df[source_column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{feature}: source column {source_column!r} is not finite numeric")
            result[feature] = values
        elif source_type == "constant":
            constant = _finite_numeric(rule["constant_value"])
            if constant is None:
                raise ValueError(f"{feature}: constant value must be finite numeric")
            result[feature] = constant
        elif source_type == "unused":
            result[feature] = np.nan
        elif source_type == "keep":
            pass
    return result.reindex(columns=_feature_cols(model_feature_cols))


def mapping_fingerprint(mapping: dict) -> str:
    normalized = normalize_mapping(mapping)
    normalized.pop("catalog_fingerprint", None)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def catalog_fingerprint(catalog: pd.DataFrame) -> str:
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas DataFrame")
    payload = catalog.copy()
    for column in payload.columns:
        payload[column] = payload[column].map(
            lambda value: None if pd.isna(value) else value
        )
    records = payload.to_dict(orient="records")
    return hashlib.sha256(
        _canonical_json({"columns": [str(column) for column in payload.columns], "rows": records}).encode("utf-8")
    ).hexdigest()


def mapping_snapshot(mapping: dict, *, confirmed_at: str | None = None) -> dict[str, Any]:
    snapshot = copy.deepcopy(normalize_mapping(mapping))
    snapshot["mapping_hash"] = mapping_fingerprint(snapshot)
    if confirmed_at is not None:
        snapshot["confirmed_at"] = confirmed_at
    snapshot["status"] = "confirmed"
    snapshot["confirmed"] = True
    return snapshot


def mapping_snapshot_restore_policy(
    meta: dict[str, Any] | None,
    *,
    current_session_id: str | None,
    minimum_version: int = 2,
) -> str:
    """Return the pure session restore action for mapping metadata."""
    payload = meta if isinstance(meta, dict) else {}
    snapshot_session_id = payload.get("sid")
    if (
        snapshot_session_id
        and current_session_id
        and snapshot_session_id != current_session_id
    ):
        return "session_mismatch"
    try:
        version = int(payload.get("version") or 1)
    except (TypeError, ValueError):
        version = 1
    if version < int(minimum_version) or not MAPPING_SESSION_KEYS.issubset(payload):
        return "clear"
    return "restore"
