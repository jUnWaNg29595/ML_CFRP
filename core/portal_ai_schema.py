"""Validated data contracts shared by the portal AI and prediction layers."""

from __future__ import annotations

import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


MATERIAL_TYPES = frozenset({"epoxy_resin", "ud_cfrp"})
TARGETS_BY_MATERIAL = {
    "epoxy_resin": frozenset(
        {
            "tg",
            "tensile_modulus",
            "tensile_strength",
            "compressive_modulus",
            "yield_strength",
        }
    ),
    "ud_cfrp": frozenset(
        {
            "ud_property",
            "tensile_modulus",
            "tensile_strength",
            "compressive_modulus",
            "compressive_strength",
            "shear_strength",
        }
    ),
}
TARGETS = frozenset().union(*TARGETS_BY_MATERIAL.values())
SOURCES = frozenset({"ai", "ai_confirmed", "manual", "batch", "user"})
FIELD_STATES = frozenset(
    {"recognized", "suggested", "uncertain", "missing", "confirmed", "rejected"}
)
EXPLANATION_STATUSES = frozenset({"available", "unavailable", "failed"})
MAX_TEXT_LENGTH = 4000
MAX_FIELD_NAME_LENGTH = 160
MAX_IDENTIFIER_LENGTH = 200

_UNCERTAIN_VALUES = frozenset(
    {"", "na", "n/a", "none", "null", "unknown", "uncertain", "未确定", "不确定", "未知"}
)
_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|authorization|bearer|client[_-]?secret|"
    r"credential|password|private[_-]?key|secret|token)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,})\b",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<name>\b(?:api[_-]?key|key|password|secret|token|access[_-]?token|"
    r"authorization|client[_-]?secret|private[_-]?key)\b)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;&]+)",
    re.IGNORECASE,
)
_UNSAFE_TEXT_PATTERNS = (
    re.compile(r"os\s*\.\s*system", re.IGNORECASE),
    re.compile(r"subprocess", re.IGNORECASE),
    re.compile(r"(?:^|[^\w])(?:eval|exec)\s*\(", re.IGNORECASE),
    re.compile(r"__import__\s*\(", re.IGNORECASE),
    re.compile(r"(?:powershell|cmd\.exe|bash\s+-c|sh\s+-c)", re.IGNORECASE),
    re.compile(r"(?:invoke-expression|start-process|rm\s+-rf|del\s+/[fq])", re.IGNORECASE),
)


@dataclass
class AIFieldSuggestion:
    field: str
    value: object | None = None
    state: str = "suggested"
    source: str = "ai"
    confidence: float | None = None
    rationale: str | None = None


@dataclass
class AIParseResponse:
    recognized_fields: dict[str, object | None] = field(default_factory=dict)
    suggestions: list[AIFieldSuggestion] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class ConfirmedPredictionRequest:
    material_type: str
    target: str
    inputs: dict[str, object]
    confirmed_by_user: bool
    source: str
    task_id: str | None = None
    model_id: str | None = None
    feature_workflow_id: str | None = None


@dataclass
class PredictionResultSummary:
    prediction: float | int | None
    unit: str | None = None
    warnings: list[str] = field(default_factory=list)
    model_version: str | None = None
    feature_workflow_id: str | None = None
    input_summary: dict[str, object] = field(default_factory=dict)
    status: str = "completed"
    completed_at: str | None = None


@dataclass
class AIExplanationResponse:
    status: str
    summary: str | None = None
    experiment_suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def _as_text(value: object, *, label: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value[:max_length]


def _normalize_uncertain(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _UNCERTAIN_VALUES:
        return None
    return value


def _validate_confidence(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be between 0 and 1")
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return confidence


def _string_list(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list of strings")
    result = []
    for item in value:
        result.append(_as_text(item, label=label, max_length=MAX_TEXT_LENGTH))
    return result


def _parse_suggestion(value: object, warnings: list[str]) -> AIFieldSuggestion:
    if not isinstance(value, Mapping):
        raise ValueError("suggestions must contain objects")
    unknown = set(value) - {"field", "value", "state", "source", "confidence", "rationale"}
    if unknown:
        warnings.append("建议包含未知字段：" + ", ".join(sorted(map(str, unknown))))
    field_name = _as_text(value.get("field"), label="suggestion.field", max_length=MAX_FIELD_NAME_LENGTH)
    state = value.get("state", "suggested")
    source = value.get("source", "ai")
    if not isinstance(state, str):
        raise ValueError("suggestion.state must be a string")
    if state not in FIELD_STATES:
        raise ValueError(f"unsupported field state: {state}")
    if not isinstance(source, str):
        raise ValueError("suggestion.source must be a string")
    if source not in SOURCES:
        raise ValueError(f"unsupported source: {source}")
    confidence = _validate_confidence(value.get("confidence"), label="suggestion.confidence")
    rationale = value.get("rationale")
    if rationale is not None:
        rationale = _as_text(rationale, label="suggestion.rationale")
    suggestion_value = _normalize_uncertain(value.get("value"))
    if state in {"uncertain", "missing"}:
        suggestion_value = None
    return AIFieldSuggestion(
        field=field_name,
        value=suggestion_value,
        state=state,
        source=source,
        confidence=confidence,
        rationale=rationale,
    )


def parse_ai_response(value: object) -> AIParseResponse:
    """Parse an AI payload while ignoring unknown top-level extensions safely."""

    if not isinstance(value, Mapping):
        raise ValueError("AI response must be an object")

    warnings = _string_list(value.get("warnings"), label="warnings")
    allowed_keys = {
        "recognized_fields",
        "suggestions",
        "missing_fields",
        "warnings",
        "assumptions",
        "confidence",
    }
    unknown_keys = set(value) - allowed_keys
    if unknown_keys:
        warnings.append("响应包含未知字段：" + ", ".join(sorted(map(str, unknown_keys))))

    recognized = value.get("recognized_fields", {})
    if not isinstance(recognized, Mapping):
        raise ValueError("recognized_fields must be an object")
    recognized_fields = {}
    for key, item in recognized.items():
        field_name = _as_text(key, label="recognized field", max_length=MAX_FIELD_NAME_LENGTH)
        normalized = _normalize_uncertain(item)
        recognized_fields[field_name] = normalized
        if normalized is None and item is not None:
            warnings.append(f"字段 {field_name} 不确定，已保留为 None")

    raw_suggestions = value.get("suggestions", [])
    if isinstance(raw_suggestions, (str, bytes)) or not isinstance(raw_suggestions, Sequence):
        raise ValueError("suggestions must be a list")
    suggestions = [_parse_suggestion(item, warnings) for item in raw_suggestions]
    confidence = _validate_confidence(value.get("confidence"), label="confidence")
    return AIParseResponse(
        recognized_fields=recognized_fields,
        suggestions=suggestions,
        missing_fields=_string_list(value.get("missing_fields"), label="missing_fields"),
        warnings=warnings,
        assumptions=_string_list(value.get("assumptions"), label="assumptions"),
        confidence=confidence,
    )


def _validate_optional_identifier(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{label} must not be empty")
    if len(identifier) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{label} must be at most {MAX_IDENTIFIER_LENGTH} characters")
    return identifier


def _validate_input_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (types.CodeType, types.ModuleType)):
        raise ValueError(f"inputs{path} contains code-like object")
    if callable(value):
        raise ValueError(f"inputs{path} contains callable object")
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"inputs{path} contains a non-string key")
            field_name = _as_text(key, label="input field", max_length=MAX_FIELD_NAME_LENGTH)
            normalized[field_name] = _validate_input_value(
                item, path=f"{path}.{field_name}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _validate_input_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"inputs{path} contains unsupported object type: {type(value).__name__}"
    )


def validate_confirmed_request(value: object) -> ConfirmedPredictionRequest:
    """Validate the user-confirmed request before it can reach prediction code."""

    if not isinstance(value, Mapping):
        raise ValueError("confirmed request must be an object")
    allowed_keys = {
        "task_id",
        "material_type",
        "target",
        "inputs",
        "model_id",
        "feature_workflow_id",
        "source",
        "confirmed_by_user",
    }
    unknown_keys = set(value) - allowed_keys
    if unknown_keys:
        raise ValueError("unknown request keys: " + ", ".join(sorted(map(str, unknown_keys))))
    required = {"material_type", "target", "inputs", "confirmed_by_user", "source"}
    missing = required - set(value)
    if missing:
        raise ValueError("confirmed request missing: " + ", ".join(sorted(missing)))

    material_type = _as_text(value["material_type"], label="material_type", max_length=80)
    if material_type not in MATERIAL_TYPES:
        raise ValueError(f"unsupported material: {material_type}")
    target = _as_text(value["target"], label="target", max_length=100)
    if target not in TARGETS_BY_MATERIAL[material_type]:
        raise ValueError(f"unsupported target for {material_type}: {target}")
    if not isinstance(value["inputs"], Mapping):
        raise ValueError("inputs must be an object")
    inputs = _validate_input_value(value["inputs"], path="")
    if value["confirmed_by_user"] is not True:
        raise ValueError("confirmed_by_user must be True")
    source = _as_text(value["source"], label="source", max_length=40)
    if source not in SOURCES:
        raise ValueError(f"unsupported source: {source}")

    return ConfirmedPredictionRequest(
        task_id=_validate_optional_identifier(value.get("task_id"), label="task_id"),
        material_type=material_type,
        target=target,
        inputs=inputs,
        model_id=_validate_optional_identifier(value.get("model_id"), label="model_id"),
        feature_workflow_id=_validate_optional_identifier(
            value.get("feature_workflow_id"), label="feature_workflow_id"
        ),
        source=source,
        confirmed_by_user=True,
    )


def _sanitize_text(value: str) -> str:
    value = value[:MAX_TEXT_LENGTH]
    value = _SECRET_VALUE_PATTERN.sub("[redacted secret]", value)
    value = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group('name')}=[redacted credential]", value
    )
    for pattern in _UNSAFE_TEXT_PATTERNS:
        value = pattern.sub("[removed unsafe instruction]", value)
    return value


def _sanitize_value(value: object, *, depth: int = 0) -> object | None:
    if depth > 6:
        return "[内容已截断]"
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in list(value.items())[:100]:
            if not isinstance(raw_key, str):
                continue
            if _SECRET_KEY_PATTERN.search(raw_key.replace(" ", "")):
                continue
            key = raw_key[:MAX_FIELD_NAME_LENGTH]
            result[key] = _sanitize_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_sanitize_value(item, depth=depth + 1) for item in list(value)[:100]]
    return None


def sanitize_ai_context(value: object) -> dict[str, object]:
    """Return a bounded, secret-free context suitable for an AI request."""

    if not isinstance(value, Mapping):
        return {}
    sanitized = _sanitize_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


_CANONICAL_SOURCE_ROLES = frozenset({"manual_input", "molecular_workflow", "derived_workflow", "unknown"})

_SOURCE_ROLE_ALIASES: dict[str, str] = {}
for _alias_group, _canonical in (
    (("manual", "manual_input", "manual input", "measured", "experimental", "人工输入", "手工输入"), "manual_input"),
    (("molecular", "molecular_workflow", "molecular workflow", "descriptor", "分子特征"), "molecular_workflow"),
    (("derived", "derived_workflow", "derived workflow", "computed", "calculated", "workflow_derived", "派生", "计算"), "derived_workflow"),
    (("unknown", "uncertain", "不确定", "无法判断"), "unknown"),
):
    for _alias in _alias_group:
        _SOURCE_ROLE_ALIASES[_alias.lower()] = _canonical


def normalize_feature_source_role(value: object) -> str | None:
    """Normalize an AI-returned source_role into one canonical value, or None.

    Only a finite alias table is honoured: unknown strings return None so the
    caller can downgrade to a conflict suggestion instead of guessing.
    target/metadata are never mapped to an approvable role.
    """
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if not token:
        return None
    # Limited standardization: collapse whitespace, hyphens and underscores
    token = " ".join(token.split())
    if token in _CANONICAL_SOURCE_ROLES:
        return token
    compact = token.replace("-", " ").replace("_", " ")
    compact = " ".join(compact.split())
    if compact in _SOURCE_ROLE_ALIASES:
        return _SOURCE_ROLE_ALIASES[compact]
    if token in _SOURCE_ROLE_ALIASES:
        return _SOURCE_ROLE_ALIASES[token]
    return None


def parse_feature_mapping_response(value: object) -> dict[str, object]:
    """Validate the narrow AI response used by feature mapping review.

    Unrecognized source_role values are NOT hard failures: the suggestion is
    downgraded to source_role=unknown / status=conflict with the raw value kept
    in source_role_raw so a human must reclassify it locally.
    """
    if not isinstance(value, Mapping):
        raise ValueError("feature review response must be an object")
    suggestions = value.get("suggestions", [])
    conflicts = value.get("conflicts", [])
    if not isinstance(suggestions, list) or not isinstance(conflicts, list):
        raise ValueError("feature review suggestions/conflicts must be lists")
    allowed_statuses = {"pending_review", "conflict", "unknown"}
    downgraded_conflicts: list[str] = []
    normalized: list[dict[str, object]] = []
    for item in suggestions:
        if not isinstance(item, Mapping):
            raise ValueError("feature review suggestions must contain objects")
        feature_id = item.get("feature_id")
        raw_columns = item.get("raw_columns", [])
        source_role_raw_value = item.get("source_role", "")
        if not isinstance(feature_id, str) or not feature_id.strip():
            raise ValueError("feature review feature_id is required")
        if not isinstance(raw_columns, list) or any(not isinstance(column, str) or not column.strip() for column in raw_columns):
            raise ValueError("feature review raw_columns must be a list of strings")
        if not isinstance(source_role_raw_value, str) or not source_role_raw_value.strip():
            raise ValueError("feature review source_role is invalid")
        normalized_role = normalize_feature_source_role(source_role_raw_value)
        source_role_downgraded = False
        if normalized_role is None:
            # Safe downgrade: never guess; flag for human review.
            source_role_downgraded = True
            normalized_role = "unknown"
        confidence = _validate_confidence(item.get("confidence"), label="feature review confidence")
        rationale = item.get("rationale_zh", "")
        if not isinstance(rationale, str):
            raise ValueError("feature review rationale_zh must be a string")
        if "status" not in item:
            raise ValueError("feature review suggestion status is required")
        status = item["status"]
        if not isinstance(status, str) or status.strip() not in allowed_statuses:
            raise ValueError("feature review status is invalid")
        unit = item.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise ValueError("feature review unit must be a non-empty string or null")
        if source_role_downgraded:
            status = "conflict"
            downgrade_note = f"AI 返回了未识别来源类型（原始值：{source_role_raw_value.strip()[:80]}），需要人工审核。"
            rationale = (rationale.strip() + "；" if rationale.strip() else "") + downgrade_note
            downgraded_conflicts.append(
                f"特征 {feature_id.strip()} 的来源类型 '{source_role_raw_value.strip()[:80]}' 无法归一化，已降级为 unknown/conflict，等待人工审核。"
            )
        elif normalized_role in {"target", "metadata"}:
            # Defensive: target/metadata must never become approvable roles.
            status = "conflict"
            normalized_role = "unknown"
        rationale_stripped = rationale.strip()
        if not rationale_stripped and status.strip() not in {"pending_review", "unknown", "conflict"}:
            raise ValueError("feature review suggestion requires evidence or explicit pending/unknown semantics")
        if not raw_columns and status.strip() not in {"pending_review", "unknown", "conflict"}:
            raise ValueError("feature review suggestion requires raw-column evidence")
        normalized_item: dict[str, object] = {
            "feature_id": feature_id.strip(),
            "raw_columns": [column.strip() for column in raw_columns],
            "source_role": normalized_role,
            "confidence": confidence,
            "rationale_zh": rationale_stripped[:MAX_TEXT_LENGTH],
            "unit": unit.strip()[:120] if isinstance(unit, str) else None,
            "status": status.strip(),
        }
        if source_role_downgraded:
            normalized_item["source_role_raw"] = source_role_raw_value.strip()[:160]
            normalized_item["source_role_downgraded"] = True
        normalized.append(normalized_item)
    if any(not isinstance(item, str) or not item.strip() for item in conflicts):
        raise ValueError("feature review conflicts must contain non-empty strings")
    normalized_conflicts = [str(item).strip() for item in conflicts] + downgraded_conflicts
    rationale = value.get("rationale_zh", "")
    if rationale is not None and not isinstance(rationale, str):
        raise ValueError("feature review rationale_zh must be a string")
    confidence = _validate_confidence(value.get("confidence"), label="feature review confidence")
    return {
        "suggestions": normalized,
        "conflicts": normalized_conflicts,
        "rationale_zh": (rationale or "")[:MAX_TEXT_LENGTH],
        "confidence": confidence,
    }
