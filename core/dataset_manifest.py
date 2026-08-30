"""Versioned raw-column to semantic-feature dataset manifests."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

_REVIEW_KEYS = {
    "approved_hash", "approved_at", "approved_by", "reviewer", "reviewed_at", "review_note", "review_notes", "review_metadata",
    "updated_at", "created_at", "mapped_at", "manifest_hash_before", "manifest_hash_after", "manifest_hash", "record_hash",
}


def normalize_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value.pop("manifest_hash", None)
    for key in ("updated_at", "created_at", "mapped_at", "approved_at", "approved_by"):
        value.pop(key, None)
    approval = value.get("approval")
    if isinstance(approval, dict):
        for key in _REVIEW_KEYS:
            approval.pop(key, None)
    records = value.get("review_records")
    if isinstance(records, list):
        normalized_records = []
        for record in records:
            if isinstance(record, dict):
                normalized_records.append({key: item for key, item in record.items() if key not in _REVIEW_KEYS})
            else:
                normalized_records.append(record)
        value["review_records"] = normalized_records
    return value


def compute_dataset_manifest_hash(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(normalize_dataset_manifest(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_dataset_manifest(manifest: Mapping[str, Any], registry: Mapping[str, Any], frame_columns: list[str] | None = None, require_approved: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return {"ok": False, "errors": ["manifest must be an object"], "warnings": []}
    if manifest.get("schema_version") != 1:
        _error(errors, "schema_version must be 1")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest.get("dataset_id", "").strip():
        _error(errors, "dataset_id is required")
    source_bindings = manifest.get("source_bindings", [])
    if not isinstance(source_bindings, list):
        _error(errors, "source_bindings must be a list")
        source_bindings = []
    seen_raw: set[str] = set()
    for i, binding in enumerate(source_bindings):
        if not isinstance(binding, Mapping):
            _error(errors, f"source_bindings[{i}] must be an object")
            continue
        raw, field = binding.get("raw_column"), binding.get("source_field")
        if not isinstance(raw, str) or not raw.strip() or raw in seen_raw:
            _error(errors, f"source_bindings[{i}].raw_column must be unique and non-empty")
        elif frame_columns is not None and raw not in frame_columns:
            _error(errors, f"raw column not found: {raw}")
        seen_raw.add(raw) if isinstance(raw, str) else None
        if not isinstance(field, str) or not field.strip():
            _error(errors, f"source_bindings[{i}].source_field is required")
        if not isinstance(binding.get("parse_rule_version"), str) or not binding.get("parse_rule_version", "").strip():
            _error(errors, f"source_bindings[{i}].parse_rule_version is required")
    features = {item.get("feature_id"): item for item in registry.get("features", []) if isinstance(item, Mapping) and isinstance(item.get("feature_id"), str)}
    profiles = registry.get("model_profiles", {}) if isinstance(registry.get("model_profiles", {}), Mapping) else {}
    profile_id = manifest.get("model_profile_id")
    profile = profiles.get(profile_id) if isinstance(profile_id, str) else None
    if not isinstance(profile, Mapping):
        _error(errors, f"unknown model profile: {profile_id}")
        profile = {}
    bindings = manifest.get("feature_bindings", [])
    if not isinstance(bindings, list):
        _error(errors, "feature_bindings must be a list")
        bindings = []
    bound: dict[str, Mapping[str, Any]] = {}
    for i, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            _error(errors, f"feature_bindings[{i}] must be an object")
            continue
        fid = binding.get("feature_id")
        if not isinstance(fid, str) or not fid.strip():
            _error(errors, f"feature_bindings[{i}].feature_id is required")
            continue
        if fid in bound:
            _error(errors, f"duplicate feature_id binding: {fid}")
        bound[fid] = binding
        if fid not in features:
            _error(errors, f"unknown feature_id: {fid}")
        raw_columns = binding.get("raw_columns")
        if not isinstance(raw_columns, list) or not raw_columns or any(not isinstance(raw, str) or not raw.strip() for raw in raw_columns):
            _error(errors, f"feature_bindings[{i}].raw_columns must be non-empty strings")
        elif frame_columns is not None:
            for raw in raw_columns:
                if raw not in frame_columns:
                    _error(errors, f"raw column not found: {raw}")
        if not isinstance(binding.get("source_role"), str) or not binding.get("source_role", "").strip():
            _error(errors, f"feature_bindings[{i}].source_role is required")
        if not isinstance(binding.get("unit"), str) or not binding.get("unit", "").strip():
            _error(errors, f"feature_bindings[{i}].unit is required")
        if "value_mapping" in binding and not isinstance(binding["value_mapping"], Mapping):
            _error(errors, f"feature_bindings[{i}].value_mapping must be an object")
    aliases = manifest.get("aliases", [])
    if aliases:
        for alias in aliases if isinstance(aliases, list) else []:
            if isinstance(alias, Mapping) and alias.get("status") == "approved":
                _error(errors, "aliases are pending candidates and cannot be approved bindings")
    profile_ids = set(profile.get("feature_ids", []))
    if not profile.get("allow_feature_subset", False):
        missing = profile_ids - set(bound)
        if missing:
            _error(errors, "manifest missing required profile features: " + ", ".join(sorted(missing)))
    if require_approved:
        if registry.get("approval", {}).get("status") != "approved":
            _error(errors, "registry is not approved")
        if profile.get("status") != "approved":
            _error(errors, "model profile is not approved")
        if manifest.get("status") != "approved":
            _error(errors, "manifest is not approved")
        for fid in bound:
            if features.get(fid, {}).get("status") != "approved":
                _error(errors, f"feature is not approved: {fid}")
    return {"ok": not errors, "errors": errors, "warnings": [], "manifest_hash": compute_dataset_manifest_hash(manifest)}


def resolve_dataset_feature_bindings(manifest: Mapping[str, Any], registry: Mapping[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    if manifest.get("model_profile_id") != profile_id:
        raise ValueError("manifest model profile does not match requested profile")
    report = validate_dataset_manifest(manifest, registry)
    if not report["ok"]:
        raise ValueError("invalid dataset manifest: " + "; ".join(report["errors"]))
    return {binding["feature_id"]: copy.deepcopy(dict(binding)) for binding in manifest.get("feature_bindings", [])}
