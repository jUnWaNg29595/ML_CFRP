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
# Batch acceptance helpers (batch accept safe mapping suggestions)
# ============================================================

_APPROVABLE_SOURCE_ROLES = {"manual_input", "molecular_workflow", "derived_workflow"}
_MIN_SAFE_CONFIDENCE = 0.85

# 工艺/测试/人工记录字段默认必须为 manual_input（业务硬规则）。
# AI 若把这类字段标为 workflow/derived，必须转入人工处理而不是进入安全建议。
_PROCESS_TEST_FIELD_KEYWORDS: tuple[str, ...] = (
    "固化温度", "固化时间", "升温速率", "保温", "后固化", "压力", "真空", "真空度",
    "湿度", "气氛", "混合", "脱泡", "成型", "加工", "工艺", "批次", "环境",
    "测试方法", "测试标准", "测试仪器", "测试温度", "测试湿度", "样品状态",
    "测试条件", "试验条件", "重复次数", "前处理", "测试", "试验",
    "备注", "等级", "目视", "观察", "异常", "人工", "外观", "评级",
    "curing temperature", "cure time", "curing time", "ramp rate", "heating rate",
    "postcure", "post-cure", "dwell", "hold time", "pressure", "vacuum",
    "humidity", "atmosphere", "degas", "batch", "specimen", "test method",
    "test standard", "test condition", "notes", "grade", "visual", "operator",
)


def _looks_like_process_test_field(suggestion: Mapping[str, Any], registry_feature: Mapping[str, Any] | None = None) -> bool:
    """判断建议是否命中工艺/测试字段关键词。

    检测范围：建议的 feature_id/name/label/原始列，以及 registry 特征定义的
    name/label/aliases（建议常常只有 feature_id 代号，中文语义在 registry 中）。
    """
    text_parts: list[str] = []
    feature_id = str(suggestion.get("feature_id") or "").strip().lower()
    if feature_id:
        text_parts.append(feature_id)
    for key in ("name", "label", "feature_name"):
        value = suggestion.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.lower())
    for raw in (suggestion.get("raw_columns") or []):
        if isinstance(raw, str) and raw.strip():
            text_parts.append(raw.lower())
    if isinstance(registry_feature, Mapping):
        for key in ("name", "label", "legacy_name"):
            value = registry_feature.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.lower())
        for key in ("aliases", "accepted_aliases"):
            values = registry_feature.get(key) or []
            if isinstance(values, str):
                text_parts.append(values.lower())
            elif isinstance(values, (list, tuple)):
                text_parts.extend(str(v).lower() for v in values if str(v).strip())
    text = " ".join(text_parts)
    return any(keyword in text for keyword in _PROCESS_TEST_FIELD_KEYWORDS)


def _normalized_suggestion(suggestion: Mapping[str, Any]) -> dict[str, Any]:
    """字段别名规范化（source_type→source_role、raw_column→raw_columns 等）。"""
    from .portal_ai_schema import normalize_feature_mapping_aliases, normalize_feature_source_role
    entry = normalize_feature_mapping_aliases(dict(suggestion))
    normalized = copy.deepcopy(dict(entry))
    role = normalize_feature_source_role(normalized.get("source_role"))
    if role is not None:
        normalized["source_role"] = role
    raw_columns = normalized.get("raw_columns")
    if isinstance(raw_columns, str):
        normalized["raw_columns"] = [c.strip() for c in raw_columns.split(",") if c.strip()]
    return normalized


def classify_feature_suggestions(
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify suggestions into batch-acceptable vs human-required buckets.

    兼容接口：仍返回 (safe, attention)。每个 attention 项带结构化
    `_review_reasons`（列表）与 `_diagnostics`（{feature_id, raw_columns,
    status, reasons, can_batch_accept, repair_action}）。
    """
    result = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    return result["safe"], result["attention"]


def classify_suggestions_with_diagnostics(
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    """结构化分类：safe / attention / diagnostics / counts。

    diagnostics 每条含：feature_id、raw_columns、status('safe'|'attention')、
    reasons、can_batch_accept、repair_action。
    """
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    profile_status = str(profile.get("status") or "").strip().lower() if isinstance(profile, Mapping) else "unknown"
    profile_feature_ids = set(profile.get("feature_ids", []) if isinstance(profile, Mapping) else [])
    definitions = {
        item.get("feature_id"): item
        for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping) and item.get("feature_id")
    }
    registry_approval = str((registry.get("approval") or {}).get("status") or "").strip().lower() if isinstance(registry, Mapping) else "unknown"

    safe: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen_raw_columns: dict[str, str] = {}

    for suggestion in suggestions:
        if not isinstance(suggestion, Mapping):
            diag = {
                "feature_id": "(非法建议)", "raw_columns": [], "status": "attention",
                "reasons": ["建议不是对象"], "can_batch_accept": False,
                "repair_action": "删除该建议或重新运行 AI 分析",
            }
            attention.append({"status": "unknown", "_diagnostics": diag})
            diagnostics.append(diag)
            continue

        copy_sugg = _normalized_suggestion(suggestion)
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
        repair_actions: list[str] = []
        if status != "pending_review":
            reasons.append(f"状态为 {status}，需要人工处理")
            repair_actions.append("进入逐项审核修改状态为 pending_review 后再接受")
        if source_role not in _APPROVABLE_SOURCE_ROLES:
            reasons.append(f"来源类型 {source_role} 无法批准")
            repair_actions.append("在逐项审核中把来源修改为 manual_input / molecular_workflow / derived_workflow")
        if feature_id not in profile_feature_ids:
            reasons.append(f"feature_id {feature_id} 不属于当前 profile")
            repair_actions.append("选择当前 profile 内的规范特征，或转为新特征提案")
        elif feature_id in definitions:
            reg_status = str(definitions[feature_id].get("status") or "unknown").strip().lower()
            if reg_status not in {"draft", "approved"}:
                reasons.append(f"registry 状态 {reg_status} 不允许批准")
                repair_actions.append("先人工审核 Registry 特征状态（legacy_observed/deprecated/blocked 不能自动批准）")
        if not raw_columns:
            reasons.append("缺少原始列")
            repair_actions.append("在逐项审核中补全原始列")
        if confidence_val < _MIN_SAFE_CONFIDENCE:
            reasons.append(f"AI 置信度 {confidence_val:.2f} 低于安全阈值 0.85")
            repair_actions.append("人工核对后编辑接受，或重新运行 AI 分析")
        if copy_sugg.get("source_role_downgraded"):
            reasons.append("AI 来源类型已降级，需要人工审核")
            repair_actions.append("人工确认来源类型后编辑接受")
        if is_new_proposal:
            reasons.append("新特征提案需要人工登记")
            repair_actions.append("使用【批准并登记新特征】单独登记，不进入批量接受")
        # 工艺/测试字段被 AI 标为 workflow/derived → 必须人工确认
        if source_role in {"molecular_workflow", "derived_workflow"} and _looks_like_process_test_field(copy_sugg, definitions.get(feature_id)):
            reasons.append("该字段属于工艺/测试输入，默认应为 manual_input，请人工确认")
            repair_actions.append("把来源类型修改为 manual_input（工艺/测试字段默认人工输入）")
        for raw in raw_columns:
            if raw in seen_raw_columns and seen_raw_columns[raw] != feature_id:
                reasons.append(f"原始列 {raw} 已被映射到 {seen_raw_columns[raw]}，存在冲突")
                repair_actions.append("解决原始列冲突（一个原始列只能映射一个特征）")
            else:
                seen_raw_columns[raw] = feature_id

        diag = {
            "feature_id": feature_id or "(空)",
            "raw_columns": raw_columns,
            "status": "attention" if reasons else "safe",
            "reasons": reasons,
            "can_batch_accept": not reasons,
            "repair_action": "；".join(dict.fromkeys(repair_actions)) if repair_actions else "",
            "profile_status": profile_status,
            "registry_approval": registry_approval,
        }
        copy_sugg["_review_reasons"] = reasons
        copy_sugg["_diagnostics"] = diag
        diagnostics.append(diag)
        if reasons:
            attention.append(copy_sugg)
        else:
            safe.append(copy_sugg)

    counts = {
        "pending_review": sum(1 for s in suggestions if isinstance(s, Mapping) and str(s.get("status") or "").strip().lower() == "pending_review"),
        "safe": len(safe),
        "attention": len(attention),
        "approved": sum(1 for s in suggestions if isinstance(s, Mapping) and str(s.get("status") or "").strip().lower() == "approved"),
        "conflict": sum(1 for s in suggestions if isinstance(s, Mapping) and str(s.get("status") or "").strip().lower() == "conflict"),
        "unprocessable": sum(1 for d in diagnostics if not d["can_batch_accept"]),
    }
    return {
        "safe": safe,
        "attention": attention,
        "diagnostics": diagnostics,
        "counts": counts,
        "profile_status": profile_status,
        "registry_approval": registry_approval,
        "has_suggestions": bool(suggestions),
        "has_frame": False,  # UI 层填充
        "has_profile": bool(profile_id),
        "has_ai": False,  # UI 层填充
    }


def batch_accept_feature_bindings(
    manifest: Mapping[str, Any],
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
    reviewer: str,
    *,
    selected_feature_ids: list[str] | None = None,
) -> dict[str, Any]:
    """原子化批量接受选中的安全映射建议，只写入当前 dataset manifest。

    边界（与 Registry 批准严格区分）：
    - 只更新 manifest.feature_bindings 的 review_status=approved；
    - 不修改 Registry 特征语义/status；
    - 不修改 model profile 状态；
    - 不自动批准全局 Registry；
    - 不自动创建新特征（新提案必须走登记流程）；
    - 不自动生成缺失值、不发布模型、不启用门户模型。
    manifest.status 只在 Registry、profile、manifest 全部满足正式条件时
    才升级为 approved；否则保持 draft/mapped（mapped=有已接受绑定但未正式批准）。

    原子性：先校验全部选中建议；任何一条失败则整个批次不写入，原 manifest 不变。
    """
    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ValueError("审核人（reviewer）必填：批量接受特征映射建议需要本地审核人")
    classification = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    safe_by_id = {str(item.get("feature_id")): item for item in classification["safe"]}
    selected = [str(fid) for fid in (selected_feature_ids or []) if str(fid).strip()]
    if not selected:
        raise ValueError("请至少选择一条建议")

    # 1) 全量预校验（原子化第一步）：任何一条失败 → 整个批次中止
    validations: list[str] = []
    chosen_items: list[dict[str, Any]] = []
    for feature_id in selected:
        item = safe_by_id.get(feature_id)
        if item is None:
            validations.append(f"{feature_id} 不在可批量接受列表中（请检查状态/来源/置信度/冲突）")
            continue
        fid = str(item.get("feature_id") or "").strip()
        raw_columns = [str(c).strip() for c in (item.get("raw_columns") or []) if str(c).strip()]
        role = str(item.get("source_role") or "").strip()
        if not fid:
            validations.append("存在 feature_id 为空的建议")
        if not raw_columns:
            validations.append(f"{fid} 缺少原始列")
        if role not in _APPROVABLE_SOURCE_ROLES:
            validations.append(f"{fid} 的 source_role={role} 非法")
        if item.get("_diagnostics", {}).get("can_batch_accept") is False:
            validations.append(f"{fid} 未通过安全检查：{'; '.join((item.get('_review_reasons') or [])[:3])}")
        chosen_items.append(item)
    if validations:
        raise ValueError("批量接受中止（未写入任何数据）：" + " | ".join(validations[:10]))

    # 2) 结构校验通过后一次性写入
    updated = copy.deepcopy(dict(manifest))
    now = datetime.now(timezone.utc).isoformat()
    new_bindings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for item in chosen_items:
        feature_id = str(item.get("feature_id") or "").strip()
        raw_columns = [str(c).strip() for c in (item.get("raw_columns") or []) if str(c).strip()]
        source_role = str(item.get("source_role") or "").strip()
        binding = {
            "feature_id": feature_id,
            "raw_columns": raw_columns,
            "source_role": source_role,
            "unit": str(item.get("unit") or "unknown").strip()[:120],
            "confidence": item.get("confidence"),
            "rationale_zh": str(item.get("rationale_zh") or "")[:500],
            "review_status": "approved",
            "approved_by": reviewer_name,
            "approved_at": now,
        }
        new_bindings.append(binding)
        records.append({
            "action": "accept",
            "reviewer": reviewer_name,
            "feature_id": feature_id,
            "raw_columns": raw_columns,
            "source_role": source_role,
            "recorded_at": now,
            "batch": True,
            "manifest_hash": "",  # 写入后统一计算
        })

    existing_bindings = [b for b in (updated.get("feature_bindings") or []) if isinstance(b, Mapping)]
    new_fids = {b["feature_id"] for b in new_bindings}
    merged_bindings = [b for b in existing_bindings if b.get("feature_id") not in new_fids] + new_bindings
    updated["feature_bindings"] = merged_bindings

    # 3) manifest 状态升级规则：只有 Registry/profile/manifest 全部满足正式条件才 approved
    profile = ((registry.get("model_profiles") or {}).get(profile_id) or {}) if isinstance(registry, Mapping) else {}
    registry_ok = isinstance(registry, Mapping) and str((registry.get("approval") or {}).get("status") or "").strip().lower() == "approved"
    profile_ok = str(profile.get("status") or "").strip().lower() == "approved" if isinstance(profile, Mapping) else False
    if registry_ok and profile_ok:
        updated["status"] = "approved"
        updated.setdefault("approval", {}).update({
            "status": "approved", "approved_by": reviewer_name, "approved_at": now, "batch_approved": True,
        })
        updated["approved_at"] = now
        updated["approved_by"] = reviewer_name
    else:
        # 仅接受映射：manifest 保持 mapped（已接受但未正式批准，不能训练）
        updated["status"] = "mapped"
        updated.setdefault("approval", {}).update({
            "status": "mapped", "mapped_by": reviewer_name, "mapped_at": now, "batch_approved": True,
        })
        updated["mapped_at"] = now
        updated["mapped_by"] = reviewer_name

    # 4) 重新计算 manifest_hash（含 review_records，保证审计可追溯）
    from .dataset_manifest import compute_dataset_manifest_hash
    updated.setdefault("review_records", []).extend(records)
    for record in updated["review_records"]:
        if isinstance(record, Mapping) and record.get("batch") and not record.get("manifest_hash"):
            record["manifest_hash"] = compute_dataset_manifest_hash(updated)
    updated["manifest_hash"] = compute_dataset_manifest_hash(updated)
    return updated


def batch_approve_safe_feature_suggestions(
    manifest: Mapping[str, Any],
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
    reviewer: str,
) -> dict[str, Any]:
    """兼容旧接口：等价于 batch_accept_feature_bindings 全选安全建议。

    保留为兼容模式（旧调用方/旧测试仍可用），但内部走新的原子化实现，
    且不再无条件把 manifest.status 升级为 approved（遵循状态规则）。
    """
    classification = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    safe_ids = [str(item.get("feature_id")) for item in classification["safe"]]
    return batch_accept_feature_bindings(
        manifest, suggestions, registry, profile_id, reviewer, selected_feature_ids=safe_ids,
    )


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
    "classify_suggestions_with_diagnostics",
    "batch_accept_feature_bindings",
    "batch_approve_safe_feature_suggestions",
]