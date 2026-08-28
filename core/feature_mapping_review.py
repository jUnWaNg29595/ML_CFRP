"""Focused AI-assisted review of raw columns and semantic feature bindings."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .portal_ai_schema import parse_feature_mapping_response, sanitize_ai_context


_REVIEW_FEATURE_FIELDS = (
    "feature_id",
    "name",
    "label",
    "source_type",
    "data_type",
    "unit",
    "enum_values",
    "categorical_values",
    "valid_range",
    "required_for_prediction",
    "nullable",
    "default_policy",
    "aliases",
    "accepted_aliases",
    "status",
)
_COMMON_TARGET_COLUMNS = {
    "target",
    "target_col",
    "y",
    "label",
    "tg",
    "tg_c",
    "ud_property",
    "tensile_modulus",
    "tensile_strength",
    "compressive_modulus",
    "compressive_strength",
    "yield_strength",
    "shear_strength",
}


def _normalized_column(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def _target_column_names(profile: Mapping[str, Any], columns: list[Any]) -> set[str]:
    declared = {
        profile.get("target_col"),
        profile.get("target"),
        profile.get("target_column"),
    }
    target_names = {
        _normalized_column(value)
        for value in declared
        if str(value or "").strip()
    }
    target_names.update(_COMMON_TARGET_COLUMNS)
    return {
        str(column)
        for column in columns
        if _normalized_column(column) in target_names
    }


def _feature_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item[key])
        for key in _REVIEW_FEATURE_FIELDS
        if key in item
    }


def build_feature_review_context(frame: Any, registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    """Build a small, feature-only context; metrics and prediction state are excluded."""
    raw_columns = getattr(frame, "columns", [])
    columns = list(raw_columns) if raw_columns is not None else []
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    if not isinstance(profile, Mapping):
        profile = {}
    target_columns = _target_column_names(profile, columns)
    review_columns = [column for column in columns if str(column) not in target_columns]
    feature_ids = set(profile.get("feature_ids", []) if isinstance(profile, Mapping) else [])
    normalized_columns = {_normalized_column(column) for column in review_columns}

    def mappable(item: Mapping[str, Any]) -> bool:
        names = [item.get("name"), item.get("feature_id")]
        for key in ("aliases", "accepted_aliases"):
            values = item.get(key) or []
            names.extend([values] if isinstance(values, str) else values if isinstance(values, list) else [])
        for name in names:
            token = _normalized_column(name)
            if token and any(token in column or column in token for column in normalized_columns):
                return True
        return False
    definitions = [
        _feature_summary(item)
        for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping)
        and item.get("source_type") not in {"target", "metadata"}
        and ((feature_ids and item.get("feature_id") in feature_ids) or (not feature_ids and mappable(item)))
    ]
    dtypes = {
        str(column): str(getattr(frame[column], "dtype", ""))
        for column in review_columns
        if hasattr(frame, "__getitem__")
    }
    sample_rows = []
    try:
        sample_frame = frame.loc[:, review_columns] if hasattr(frame, "loc") else frame
        sample_rows = sanitize_ai_context(
            {"rows": sample_frame.head(3).to_dict(orient="records")}
        ).get("rows", [])
    except Exception:
        sample_rows = []
    return {
        "profile_id": str(profile_id),
        "raw_columns": [str(column) for column in review_columns],
        "column_dtypes": dtypes,
        "sample_rows": sample_rows,
        "candidate_features": definitions,
    }


def request_feature_mapping_review(client: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    if client is None or not callable(getattr(client, "review_feature_mapping", None)):
        raise ValueError("AI 特征审核客户端不可用")
    response = client.review_feature_mapping(dict(context))
    return parse_feature_mapping_response(response)


def apply_feature_review_decision(
    manifest: Mapping[str, Any], suggestion: Mapping[str, Any], action: str, reviewer: str,
    registry: Mapping[str, Any] | None = None, edited: Mapping[str, Any] | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(manifest))
    action = str(action).strip().lower()
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise ValueError("reviewer is required for feature review decisions")
    now = datetime.now(timezone.utc).isoformat()
    record = {"action": action, "reviewer": reviewer, "feature_id": suggestion.get("feature_id"), "recorded_at": now}
    if action == "reject":
        updated.setdefault("review_records", []).append(record)
        return updated
    if action not in {"accept", "edit_accept"}:
        raise ValueError("unsupported feature review action")
    suggestion_status = suggestion.get("status")
    if suggestion_status is None or str(suggestion_status).strip().lower() != "approved":
        raise ValueError("只能批准 status=approved 的特征审核建议")
    feature_id = suggestion.get("feature_id")
    registry_feature = None
    if isinstance(registry, Mapping):
        registry_feature = next(
            (item for item in registry.get("features", [])
             if isinstance(item, Mapping) and item.get("feature_id") == feature_id),
            None,
        )
        valid_ids = {
            str(item.get("feature_id")) for item in registry.get("features", [])
            if isinstance(item, Mapping) and item.get("feature_id")
        }
        if feature_id not in valid_ids:
            raise ValueError("feature_id 不属于 registry 合法候选")
        if profile_id:
            profile = registry.get("model_profiles", {}).get(profile_id) if isinstance(registry.get("model_profiles"), Mapping) else None
            profile_ids = set(profile.get("feature_ids", [])) if isinstance(profile, Mapping) else set()
            if feature_id not in profile_ids:
                raise ValueError("feature_id 不属于当前 profile 合法候选")
    raw_columns = suggestion.get("raw_columns")
    if not isinstance(feature_id, str) or not feature_id.strip() or not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("approved binding requires feature_id and raw_columns")
    if "source_role" not in suggestion or not str(suggestion.get("source_role") or "").strip():
        raise ValueError("source_role 必须显式提供")
    source_role = str(suggestion.get("source_role") or "").strip()
    if source_role not in {"manual_input", "molecular_workflow", "derived_workflow"}:
        raise ValueError("source_role 必须是允许的输入/工作流来源")
    if action == "edit_accept":
        if not isinstance(edited, Mapping):
            raise ValueError("edit_accept 必须提供编辑后的字段")
        suggestion = {**dict(suggestion), **dict(edited)}
        raw_columns = suggestion.get("raw_columns")
        if isinstance(raw_columns, str):
            raw_columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
        if not isinstance(raw_columns, list) or not raw_columns:
            raise ValueError("编辑后的 raw_columns 不能为空")
        source_role = str(suggestion.get("source_role") or "").strip()
        if source_role not in {"manual_input", "molecular_workflow", "derived_workflow"}:
            raise ValueError("编辑后的 source_role 无效")
        edited_status = suggestion.get("status")
        if edited_status is not None and str(edited_status).strip().lower() != "approved":
            raise ValueError("只能批准 status=approved 的特征审核建议")
    if isinstance(registry_feature, Mapping):
        source_type = str(registry_feature.get("source_type") or "").strip()
        if source_type != source_role:
            raise ValueError("source_role 必须与 registry feature.source_type 对齐")
    binding = {
        "feature_id": feature_id.strip(),
        "raw_columns": [str(column).strip() for column in raw_columns],
        "source_role": source_role,
        "unit": str(suggestion.get("unit") or "unknown").strip(),
        "review_status": "approved",
        "approved_by": reviewer,
        "approved_at": now,
    }
    bindings = [item for item in (updated.get("feature_bindings") or []) if isinstance(item, Mapping) and item.get("feature_id") != feature_id]
    bindings.append(binding)
    updated["feature_bindings"] = bindings
    updated.setdefault("approval", {})
    updated["approval"].update({"status": "approved", "approved_by": reviewer, "approved_at": binding["approved_at"]})
    record.update({"feature_id": feature_id.strip(), "raw_columns": list(binding["raw_columns"])})
    updated.setdefault("review_records", []).append(record)
    return updated


def save_feature_review_record(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    payload["record_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


__all__ = ["build_feature_review_context", "request_feature_mapping_review", "apply_feature_review_decision", "save_feature_review_record"]
