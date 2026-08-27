"""Versioned semantic feature registry and validation helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SOURCE_TYPES = {"molecular_workflow", "derived_workflow", "manual_input", "target", "metadata", "unknown"}
DEFAULT_POLICIES = {"forbidden", "explicit_only", "workflow_only"}
FEATURE_STATUSES = {"draft", "approved", "legacy_observed", "deprecated", "blocked"}
_REVIEW_KEYS = {"approved_hash", "approved_at", "approved_by", "reviewer", "reviewed_at", "review_note", "review_notes", "change_summary", "change_summaries", "review_metadata"}

def _without_mutable_approval_metadata(payload: Any) -> Any:
    value = copy.deepcopy(payload)
    if isinstance(value, dict) and isinstance(value.get("approval"), dict):
        for key in _REVIEW_KEYS:
            value["approval"].pop(key, None)
    return value

def compute_registry_hash(payload: Mapping[str, Any]) -> str:
    clean = _without_mutable_approval_metadata(payload)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def validate_registry(payload: Mapping[str, Any], require_approved: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return {"ok": False, "errors": ["registry must be a JSON object"], "warnings": [], "registry_hash": compute_registry_hash({})}
    registry_hash = compute_registry_hash(payload)
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(payload.get("registry_version"), str) or not payload.get("registry_version"):
        errors.append("registry_version is required")
    features = payload.get("features")
    if not isinstance(features, list):
        errors.append("features must be a list")
        features = []
    profiles = payload.get("model_profiles")
    if not isinstance(profiles, Mapping):
        errors.append("model_profiles must be an object")
        profiles = {}
    approval = payload.get("approval")
    if not isinstance(approval, Mapping):
        errors.append("approval must be an object")
        approval = {}
    ids: set[str] = set()
    names: set[str] = set()
    definitions: dict[str, Mapping[str, Any]] = {}
    for index, feature in enumerate(features):
        prefix = f"features[{index}]"
        if not isinstance(feature, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        feature_id, name = feature.get("feature_id"), feature.get("name")
        if not isinstance(feature_id, str) or not feature_id.strip():
            errors.append(f"{prefix}.feature_id is required")
        elif feature_id in ids:
            errors.append(f"duplicate feature_id: {feature_id}")
        else:
            ids.add(feature_id)
            definitions[feature_id] = feature
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name is required")
        elif name in names:
            errors.append(f"duplicate feature name: {name}")
        else:
            names.add(name)
        source_type, policy, status = feature.get("source_type"), feature.get("default_policy"), feature.get("status")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{prefix}.source_type is invalid: {source_type}")
        if policy not in DEFAULT_POLICIES:
            errors.append(f"{prefix}.default_policy is invalid: {policy}")
        if status not in FEATURE_STATUSES:
            errors.append(f"{prefix}.status is invalid: {status}")
        if "required_for_prediction" in feature and not isinstance(feature.get("required_for_prediction"), bool):
            errors.append(f"{prefix}.required_for_prediction must be boolean")
        if "nullable" in feature and not isinstance(feature.get("nullable"), bool):
            errors.append(f"{prefix}.nullable must be boolean")
        if feature.get("required_for_prediction") is True and feature.get("nullable") is True:
            errors.append(f"{prefix} required_for_prediction cannot be nullable")
        if source_type == "manual_input" and isinstance(feature.get("default"), (int, float)) and not isinstance(feature.get("default"), bool):
            errors.append(f"{prefix}.default numeric values are forbidden for manual_input")
        if source_type in {"derived_workflow", "molecular_workflow"}:
            rule = feature.get("calculation_rule")
            if not isinstance(rule, Mapping):
                errors.append(f"{prefix}.calculation_rule is required for workflow feature")
            else:
                if not isinstance(rule.get("input_fields"), list) or not rule.get("input_fields"):
                    errors.append(f"{prefix}.calculation_rule.input_fields is required")
                if not isinstance(rule.get("implementation"), str) or not rule.get("implementation"):
                    errors.append(f"{prefix}.calculation_rule.implementation is required")
        if source_type == "unknown" and status != "blocked":
            errors.append(f"{prefix} unknown feature must be blocked")
        if status == "blocked" and not any(feature.get(key) for key in ("blocking_reason", "block_reason", "reason")):
            errors.append(f"{prefix} blocked feature requires a blocking reason")
        if status == "legacy_observed" and not any(feature.get(key) for key in ("legacy_source", "source_artifact", "source_dataset")):
            errors.append(f"{prefix} legacy_observed feature requires a legacy source")
        valid_range = feature.get("valid_range")
        if valid_range is not None:
            if not isinstance(valid_range, Mapping):
                errors.append(f"{prefix}.valid_range must be an object")
            else:
                lower, upper = valid_range.get("min"), valid_range.get("max")
                for label, bound in (("min", lower), ("max", upper)):
                    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound)):
                        errors.append(f"{prefix}.valid_range.{label} must be finite")
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
                    errors.append(f"{prefix}.valid_range min cannot exceed max")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            errors.append(f"model profile {profile_id} must be an object")
            continue
        refs = profile.get("feature_ids")
        if not isinstance(refs, list):
            errors.append(f"model profile {profile_id}.feature_ids must be a list")
            refs = []
        seen_refs: set[str] = set()
        for feature_id in refs:
            if feature_id not in definitions:
                errors.append(f"model profile {profile_id} references unknown feature_id: {feature_id}")
            if feature_id in seen_refs:
                errors.append(f"model profile {profile_id} references duplicate feature_id: {feature_id}")
            seen_refs.add(feature_id)
        blocked = profile.get("blocked_feature_ids", [])
        if blocked and not isinstance(blocked, list):
            errors.append(f"model profile {profile_id}.blocked_feature_ids must be a list")
        elif isinstance(blocked, list):
            for feature_id in blocked:
                if feature_id not in refs:
                    errors.append(f"model profile {profile_id}.blocked_feature_ids references feature outside profile: {feature_id}")
    approval_status = approval.get("status")
    if approval_status not in {"draft", "approved", "deprecated"}:
        errors.append(f"approval.status is invalid: {approval_status}")
    if require_approved:
        if approval_status != "approved":
            errors.append("registry approval status is not approved")
        if approval.get("approved_hash") != registry_hash:
            errors.append("approval.approved_hash does not match computed registry_hash")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "registry_hash": registry_hash}

def load_registry(path: str | Path, require_approved: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = validate_registry(payload, require_approved=require_approved)
    if not report["ok"]:
        raise ValueError("invalid feature registry: " + "; ".join(report["errors"]))
    return payload

def get_model_profile(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = registry.get("model_profiles", {})[profile_id]
    except (KeyError, TypeError):
        raise KeyError(f"unknown model profile: {profile_id}") from None
    if not isinstance(profile, Mapping):
        raise ValueError(f"model profile is not an object: {profile_id}")
    return copy.deepcopy(dict(profile))

def build_registry_snapshot(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    profile = get_model_profile(registry, profile_id)
    definitions = {item.get("feature_id"): item for item in registry.get("features", []) if isinstance(item, Mapping)}
    selected = []
    for feature_id in profile.get("feature_ids", []):
        if feature_id not in definitions:
            raise KeyError(f"model profile references unknown feature_id: {feature_id}")
        selected.append(copy.deepcopy(dict(definitions[feature_id])))
    return {"schema_version": registry.get("schema_version"), "registry_version": registry.get("registry_version"), "registry_hash": compute_registry_hash(registry), "profile_id": profile_id, "model_profile": profile, "features": selected}
