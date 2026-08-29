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
    "legacy_name",
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
    if suggestion_status is None or str(suggestion_status).strip().lower() != "pending_review":
        raise ValueError("只能批准 status=pending_review 的特征审核建议")
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
        if "status" in edited and str(edited.get("status") or "").strip().lower() != "pending_review":
            raise ValueError("编辑后的 status 必须为 pending_review")
        suggestion = {**dict(suggestion), **dict(edited)}
        raw_columns = suggestion.get("raw_columns")
        if isinstance(raw_columns, str):
            raw_columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
        if not isinstance(raw_columns, list) or not raw_columns:
            raise ValueError("编辑后的 raw_columns 不能为空")
        source_role = str(suggestion.get("source_role") or "").strip()
        if source_role not in {"manual_input", "molecular_workflow", "derived_workflow"}:
            raise ValueError("编辑后的 source_role 无效")
    if isinstance(registry_feature, Mapping):
        source_type = str(registry_feature.get("source_type") or "").strip()
        if source_type != source_role:
            raise ValueError("source_role 必须与 registry feature.source_type 对齐")
    if isinstance(registry, Mapping):
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("registry 审核必须提供 model profile_id")
        profiles = registry.get("model_profiles")
        profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            raise ValueError("model profile 不存在，无法批准特征绑定")
        profile_ids = set(profile.get("feature_ids", []))
        if feature_id not in profile_ids:
            raise ValueError("feature_id 不属于当前 profile 合法候选")
    if isinstance(registry_feature, Mapping):
        registry_status = str(registry_feature.get("status") or "unknown").strip().lower()
        if registry_status not in {"draft", "approved"}:
            raise ValueError("registry feature status 不允许批准：" + registry_status)
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


def save_atomic_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Safely write JSON using a temporary file and atomic replace."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f".tmp.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    try:
        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(target)
    except Exception:
        if temp_path.is_file():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def load_profile_manifest(profile_id: str, root_dir: str | Path | None = None) -> dict[str, Any]:
    """Load persisted manifest for a specific model profile or return default draft."""
    if not profile_id:
        return {"schema_version": 1, "status": "draft", "feature_bindings": []}
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1] / "prediction_portal"
    manifest_dir = root / "manifests"
    manifest_file = manifest_dir / f"manifest_{profile_id}.json"
    if manifest_file.is_file():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "schema_version": 1,
        "model_profile_id": str(profile_id),
        "dataset_id": f"dataset_{profile_id}",
        "status": "draft",
        "feature_bindings": [],
    }


def save_profile_manifest(profile_id: str, manifest: Mapping[str, Any], root_dir: str | Path | None = None) -> Path:
    """Atomically save profile feature mapping manifest with calculated hash."""
    from .dataset_manifest import compute_dataset_manifest_hash
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1] / "prediction_portal"
    manifest_dir = root / "manifests"
    manifest_file = manifest_dir / f"manifest_{profile_id}.json"
    payload = copy.deepcopy(dict(manifest))
    payload["schema_version"] = 1
    payload["model_profile_id"] = str(profile_id)
    payload["dataset_id"] = str(payload.get("dataset_id") or f"dataset_{profile_id}")
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["manifest_hash"] = compute_dataset_manifest_hash(payload)
    save_atomic_json(manifest_file, payload)
    return manifest_file


def load_profile_suggestions(profile_id: str, root_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load persisted feature proposals and suggestions for a profile."""
    if not profile_id:
        return []
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1] / "prediction_portal"
    suggestions_file = root / "manifests" / f"suggestions_{profile_id}.json"
    if suggestions_file.is_file():
        try:
            data = json.loads(suggestions_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("suggestions"), list):
                return data["suggestions"]
        except Exception:
            pass
    return []


def save_profile_suggestions(profile_id: str, suggestions: list[Mapping[str, Any]], root_dir: str | Path | None = None) -> Path:
    """Atomically persist feature proposals and suggestions for a profile."""
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1] / "prediction_portal"
    suggestions_file = root / "manifests" / f"suggestions_{profile_id}.json"
    payload = [copy.deepcopy(dict(item)) for item in suggestions if isinstance(item, Mapping)]
    save_atomic_json(suggestions_file, payload)
    return suggestions_file


# ============================================================
# Batch approval helpers (one-click safe-approval workflow)
# ============================================================

_APPROVABLE_SOURCE_ROLES = {"manual_input", "molecular_workflow", "derived_workflow"}
_MIN_SAFE_CONFIDENCE = 0.85


def classify_feature_suggestions(
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify suggestions into quickly-approvable vs human-required buckets.

    A suggestion is "safe" only when every condition holds:
    - status == pending_review
    - source_role normalized into {manual_input, molecular_workflow, derived_workflow}
    - feature_id exists in the current profile
    - registry feature status is draft or approved
    - raw_columns non-empty
    - AI confidence >= 0.85
    - no unit/source conflict or duplicate binding flags
    """
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    profile_feature_ids = set(profile.get("feature_ids", []) if isinstance(profile, Mapping) else [])
    definitions = {
        item.get("feature_id"): item
        for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping) and item.get("feature_id")
    }
    safe: list[dict[str, Any]] = []
    needs_attention: list[dict[str, Any]] = []
    seen_raw_columns: dict[str, str] = {}

    for suggestion in suggestions:
        if not isinstance(suggestion, Mapping):
            needs_attention.append(dict(suggestion) if isinstance(suggestion, Mapping) else {"status": "unknown"})
            continue
        copy_sugg = copy.deepcopy(dict(suggestion))
        feature_id = str(copy_sugg.get("feature_id") or "").strip()
        source_role = str(copy_sugg.get("source_role") or "unknown").strip()
        status = str(copy_sugg.get("status") or "unknown").strip().lower()
        raw_columns = [str(col).strip() for col in (copy_sugg.get("raw_columns") or []) if str(col).strip()]
        confidence = copy_sugg.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_val = 0.0
        is_new_proposal = bool(copy_sugg.get("is_new_proposal") or feature_id not in profile_feature_ids)

        reasons: list[str] = []
        if status != "pending_review":
            reasons.append(f"状态为 {status}，需要人工处理")
        if source_role not in _APPROVABLE_SOURCE_ROLES:
            reasons.append(f"来源类型 {source_role} 无法批准")
        if feature_id not in profile_feature_ids:
            reasons.append(f"feature_id {feature_id} 不属于当前 profile")
        elif feature_id in definitions:
            reg_status = str(definitions[feature_id].get("status") or "unknown").strip().lower()
            if reg_status not in {"draft", "approved"}:
                reasons.append(f"registry 状态 {reg_status} 不允许批准")
        if not raw_columns:
            reasons.append("缺少原始列")
        if confidence_val < _MIN_SAFE_CONFIDENCE:
            reasons.append(f"AI 置信度 {confidence_val:.2f} 低于安全阈值 0.85")
        if copy_sugg.get("source_role_downgraded"):
            reasons.append("AI 来源类型已降级，需要人工审核")
        if is_new_proposal:
            reasons.append("新特征提案需要人工登记")
        for raw in raw_columns:
            if raw in seen_raw_columns and seen_raw_columns[raw] != feature_id:
                reasons.append(f"原始列 {raw} 已被映射到 {seen_raw_columns[raw]}，存在冲突")
            else:
                seen_raw_columns[raw] = feature_id

        copy_sugg["_review_reasons"] = reasons
        if reasons:
            needs_attention.append(copy_sugg)
        else:
            safe.append(copy_sugg)
    return safe, needs_attention


def batch_approve_safe_feature_suggestions(
    manifest: Mapping[str, Any],
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
    reviewer: str,
) -> dict[str, Any]:
    """Atomically approve every safe suggestion into one manifest write.

    All candidates are validated first; if any single item fails the whole
    write is aborted and the original manifest is returned unchanged.
    """
    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ValueError("reviewer is required for batch approval")
    safe, _ = classify_feature_suggestions(suggestions, registry, profile_id)
    if not safe:
        raise ValueError("没有可批量批准的安全建议")

    updated = copy.deepcopy(dict(manifest))
    now = datetime.now(timezone.utc).isoformat()
    new_bindings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for item in safe:
        feature_id = str(item.get("feature_id") or "").strip()
        raw_columns = [str(col).strip() for col in (item.get("raw_columns") or []) if str(col).strip()]
        source_role = str(item.get("source_role") or "").strip()
        if not feature_id or not raw_columns or source_role not in _APPROVABLE_SOURCE_ROLES:
            raise ValueError(f"安全建议 {feature_id or '(空)'} 校验失败，整个批次已中止")
        new_bindings.append({
            "feature_id": feature_id,
            "raw_columns": raw_columns,
            "source_role": source_role,
            "unit": str(item.get("unit") or "unknown").strip()[:120],
            "confidence": item.get("confidence"),
            "rationale_zh": str(item.get("rationale_zh") or "")[:500],
            "review_status": "approved",
            "approved_by": reviewer_name,
            "approved_at": now,
        })
        records.append({
            "action": "accept",
            "reviewer": reviewer_name,
            "feature_id": feature_id,
            "raw_columns": raw_columns,
            "source_role": source_role,
            "recorded_at": now,
            "batch": True,
        })

    # All-or-nothing: write everything only after all items validated.
    existing_bindings = [item for item in (updated.get("feature_bindings") or []) if isinstance(item, Mapping)]
    new_fids = {b["feature_id"] for b in new_bindings}
    merged_bindings = [b for b in existing_bindings if b.get("feature_id") not in new_fids] + new_bindings
    updated["feature_bindings"] = merged_bindings
    updated["status"] = "approved"
    updated.setdefault("approval", {})
    updated["approval"].update({
        "status": "approved",
        "approved_by": reviewer_name,
        "approved_at": now,
        "batch_approved": True,
    })
    updated.setdefault("review_records", []).extend(records)
    updated["approved_at"] = now
    updated["approved_by"] = reviewer_name

    # Recompute manifest hash
    from .dataset_manifest import compute_dataset_manifest_hash
    updated["manifest_hash"] = compute_dataset_manifest_hash(updated)
    return updated


__all__ = [
    "build_feature_review_context",
    "request_feature_mapping_review",
    "apply_feature_review_decision",
    "save_feature_review_record",
    "save_atomic_json",
    "load_profile_manifest",
    "save_profile_manifest",
    "load_profile_suggestions",
    "save_profile_suggestions",
    "classify_feature_suggestions",
    "batch_approve_safe_feature_suggestions",
]