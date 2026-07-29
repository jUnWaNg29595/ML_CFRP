"""Canonical manual mapping for post-feature model inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


POST_FEATURE_MAPPING_SCHEMA_VERSION = 1
SOURCE_TYPES = ("pending", "computed", "candidate", "constant", "unused")
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


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _feature_cols(values: Any) -> list[str]:
    return [str(value) for value in (values or []) if str(value).strip()]


def _finite_numeric(value: Any) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted) or not np.isfinite(float(converted)):
        return None
    return float(converted)


def _normalize_rule(rule: Any) -> dict[str, Any]:
    rule = rule if isinstance(rule, dict) else {}
    source_type = rule.get("source_type")
    return {
        "source_type": str(source_type) if source_type is not None else None,
        "source_column": (
            str(rule["source_column"])
            if rule.get("source_column") is not None
            else None
        ),
        "constant_value": rule.get("constant_value"),
        "unit": rule.get("unit"),
        "definition": rule.get("definition"),
        "confirmed": bool(rule.get("confirmed", False)),
    }


def normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic mapping representation without guessing sources."""
    payload = mapping if isinstance(mapping, dict) else {}
    model_cols = _feature_cols(payload.get("model_feature_cols"))
    source_rules = payload.get("rules")
    source_rules = source_rules if isinstance(source_rules, dict) else {}
    rules = {column: _normalize_rule(source_rules.get(column)) for column in model_cols}
    return {
        "schema_version": int(payload.get("schema_version") or POST_FEATURE_MAPPING_SCHEMA_VERSION),
        "model_feature_cols": model_cols,
        "rules": rules,
        "confirmed": bool(payload.get("confirmed", False)),
        "status": payload.get("status") or ("confirmed" if payload.get("confirmed") else "draft"),
        "model_fingerprint": payload.get("model_fingerprint"),
        "workflow_fingerprint": payload.get("workflow_fingerprint"),
        "catalog_fingerprint": payload.get("catalog_fingerprint"),
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
    excluded = {str(column) for column in (excluded_columns or set())}
    rows: list[dict[str, Any]] = []
    row_count = len(candidate_df)
    for column in candidate_df.columns:
        column = str(column)
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
