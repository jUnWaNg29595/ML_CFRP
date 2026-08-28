"""Pure publication contracts and version helpers for the user portal."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .prediction_contract import resolve_prediction_feature_contract
from .prediction_molecular_baseline import collect_workflow_source_columns


CONTRACT_SCHEMA_VERSION = 2
_CONTRACT_FIELDS = {
    "schema_version",
    "feature_cols",
    "target_col",
    "workflow_hash",
    "workflow_schema_version",
    "source_columns",
    "workflow_source_columns",
    "workflow_present",
    "molecular_features_indicated",
    "pipeline_present",
    "imputer_present",
    "scaler_present",
    "numeric_ranges",
    "workflow_source_fields",
}
_MOLECULAR_FEATURE_TOKENS = (
    "xtb",
    "maccs",
    "fingerprint",
    "molecular",
    "smiles",
    "bigsmiles",
    "selfies",
    "resin_",
    "curing_agent_",
    "hardener_",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_workflow_mapping(workflow: Any) -> dict[str, Any]:
    if isinstance(workflow, Mapping):
        return copy.deepcopy(dict(workflow))
    to_dict = getattr(workflow, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    return {}


def _normalized_columns(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        normalized = "".join(text.split()).lower()
        if text and normalized not in seen:
            result.append(text)
            seen.add(normalized)
    return result


def _pipeline_steps(pipeline: Any) -> list[Any]:
    try:
        steps = pipeline.steps
        if not isinstance(steps, Sequence) or not steps:
            return []
        parsed: list[Any] = []
        for item in steps:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
                or not str(item[0]).strip()
                or item[1] is None
            ):
                return []
            parsed.append(item[1])
        return parsed
    except (AttributeError, TypeError, ValueError):
        return []


def _has_meaningful_learned_attribute(value: Any) -> bool:
    try:
        attributes = vars(value)
    except TypeError:
        return False
    ignored = {
        "n_features_in_",
        "feature_names_in_",
        "n_outputs_",
        "n_iter_",
    }
    for name, learned in attributes.items():
        if not name.endswith("_") or name.startswith("__") or name in ignored:
            continue
        if learned is None:
            continue
        if isinstance(learned, (str, bytes)):
            if learned:
                return True
            continue
        try:
            if np.asarray(learned).size > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_usable_preprocessor(value: Any) -> bool:
    if value is None or not callable(getattr(value, "transform", None)):
        return False
    return _has_meaningful_learned_attribute(value)


def _is_usable_pipeline(value: Any) -> bool:
    return callable(getattr(value, "predict", None)) and bool(_pipeline_steps(value))


def _has_usable_preprocessor(artifact: Mapping[str, Any], kind: str) -> bool:
    if _is_usable_preprocessor(artifact.get(kind)):
        return True
    pipeline = artifact.get("pipeline")
    for step in _pipeline_steps(pipeline):
        name = type(step).__name__.lower()
        if kind == "imputer" and "imput" in name and _is_usable_preprocessor(step):
            return True
        if (
            kind == "scaler"
            and ("scaler" in name or "standardize" in name)
            and _is_usable_preprocessor(step)
        ):
            return True
    return False


def _is_molecular_feature_set(feature_cols: Sequence[str]) -> bool:
    return any(
        any(token in str(column).strip().lower() for token in _MOLECULAR_FEATURE_TOKENS)
        for column in feature_cols
    )


def _numeric_ranges(frame: Any, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    if frame is None or not hasattr(frame, "columns"):
        return {}
    ranges: dict[str, dict[str, float]] = {}
    for column in feature_cols:
        if column not in frame.columns:
            continue
        try:
            values = np.asarray(frame[column], dtype=float)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            ranges[column] = {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
    return ranges


def compute_contract_hash(contract: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(contract))
    payload.pop("contract_hash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prediction_contract(
    *,
    artifact: Mapping[str, Any],
    feature_cols: Sequence[str],
    target_col: str,
    workflow: Any = None,
    training_frame: Any = None,
    source_frame: Any = None,
    registry_snapshot: Mapping[str, Any] | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
    model_profile_id: str | None = None,
    canonical_feature_cols: Sequence[str] | None = None,
    effective_feature_cols: Sequence[str] | None = None,
    removed_feature_cols: Sequence[str] | None = None,
    removed_feature_reasons: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact, serializable contract used for portal publication."""

    artifact = _as_mapping(artifact)
    model = artifact.get("model")
    pipeline = artifact.get("pipeline")
    requested_features = _normalized_columns(feature_cols)
    target = str(target_col or "").strip()
    if model is None and pipeline is None:
        raise ValueError("模型 artifact 缺少 model 或 pipeline。")
    if not target:
        raise ValueError("模型 artifact 缺少目标列。")
    if not requested_features:
        raise ValueError("模型 artifact 缺少精确特征列清单。")

    resolution = resolve_prediction_feature_contract(
        model=model,
        pipeline=pipeline,
        artifact={**dict(artifact), "feature_cols": requested_features},
        session_feature_cols=requested_features,
    )
    if not resolution.get("ok"):
        errors = "；".join(str(error) for error in resolution.get("errors", []))
        raise ValueError(f"无法解析精确模型特征契约：{errors}")
    resolved_features = list(resolution["feature_cols"])
    if resolved_features != requested_features:
        raise ValueError("模型特征列顺序与发布特征列清单不一致。")

    workflow_payload = _as_workflow_mapping(workflow)
    source_columns = collect_workflow_source_columns(workflow_payload)
    numeric_frame = training_frame if training_frame is not None else source_frame
    base = {
        "schema_version": 1 if registry_snapshot is None and dataset_manifest is None else 2,
        "feature_cols": resolved_features,
        "target_col": target,
        "workflow_hash": workflow_payload.get("workflow_hash"),
        "workflow_schema_version": workflow_payload.get("schema_version"),
        "source_columns": source_columns,
        "workflow_source_columns": copy.deepcopy(source_columns),
        "workflow_source_fields": copy.deepcopy(source_columns),
        "workflow_present": bool(workflow_payload),
        "molecular_features_indicated": bool(
            source_columns or _is_molecular_feature_set(resolved_features)
        ),
        "pipeline_present": _is_usable_pipeline(pipeline),
        "imputer_present": _has_usable_preprocessor(artifact, "imputer"),
        "scaler_present": _has_usable_preprocessor(artifact, "scaler"),
        "numeric_ranges": _numeric_ranges(numeric_frame, resolved_features),
    }
    if base["schema_version"] == 2:
        canonical = _normalized_columns(canonical_feature_cols or resolved_features)
        effective = _normalized_columns(effective_feature_cols or resolved_features)
        removed = _normalized_columns(removed_feature_cols or [])
        definitions = list((_as_mapping(registry_snapshot).get("features") or []))
        workflow_cols = [str(item.get("name")) for item in definitions if isinstance(item, Mapping) and item.get("source_type") in {"molecular_workflow", "derived_workflow"} and item.get("name") in canonical]
        molecular_cols = [str(item.get("name")) for item in definitions if isinstance(item, Mapping) and item.get("source_type") == "molecular_workflow" and item.get("name") in canonical]
        manual_cols = [str(item.get("name")) for item in definitions if isinstance(item, Mapping) and item.get("source_type") == "manual_input" and item.get("name") in canonical]
        derived_cols = [str(item.get("name")) for item in definitions if isinstance(item, Mapping) and item.get("source_type") == "derived_workflow" and item.get("name") in canonical]
        base.update({
            "canonical_feature_cols": canonical,
            "effective_feature_cols": effective,
            "removed_feature_cols": removed,
            "removed_feature_reasons": copy.deepcopy(dict(removed_feature_reasons or {})),
            "feature_registry_version": _as_mapping(registry_snapshot).get("registry_version"),
            "feature_registry_hash": _as_mapping(registry_snapshot).get("registry_hash"),
            "dataset_manifest_hash": _as_mapping(dataset_manifest).get("manifest_hash"),
            "model_profile_id": model_profile_id,
            "workflow_feature_cols": workflow_cols,
            "molecular_workflow_feature_cols": molecular_cols,
            "derived_feature_cols": derived_cols,
            "manual_input_feature_cols": manual_cols,
            "feature_definitions": copy.deepcopy(definitions),
            "workflow_source_fields": copy.deepcopy(source_columns),
            "contract_hash": "",
        })
        base["contract_hash"] = compute_contract_hash(base)
    return base


def _contract_from_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    extra = _as_mapping(artifact.get("extra"))
    contract = extra.get("prediction_contract")
    return contract if isinstance(contract, Mapping) else None


def _contracts_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def validate_publication_artifact(
    artifact: Mapping[str, Any], contract: Mapping[str, Any] | None = None,
    registry_snapshot: Mapping[str, Any] | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a publication diagnostic without mutating the artifact."""

    artifact = _as_mapping(artifact)
    errors: list[str] = []

    def diagnostics(items: list[str], *, code: str = "publication_invalid", source: str = "publication") -> list[dict[str, Any]]:
        return [{"code": code, "feature": None, "source": source, "rule": "publication_gate", "message": str(item)} for item in items]

    extra = _as_mapping(artifact.get("extra"))
    saved_contract = _contract_from_artifact(artifact)
    if contract is not None and saved_contract is not None:
        if not _contracts_match(_as_mapping(contract), saved_contract):
            errors.append("显式 prediction_contract 与 artifact 已保存的 prediction_contract 不一致。")
    resolved_contract = contract if contract is not None else saved_contract
    if resolved_contract is None:
        message = "artifact 缺少 prediction_contract，需要重新验证。"
        return {
            "ok": False,
            "status": "needs_validation",
            "errors": [message],
            "diagnostics": diagnostics([message], code="missing_contract", source="artifact"),
        }
    resolved_contract = _as_mapping(resolved_contract)

    schema_version = resolved_contract.get("schema_version")
    legacy_contract = schema_version in {None, 1}
    if schema_version not in {None, 1, 2}:
        message = "prediction_contract schema_version is unsupported."
        return {"ok": False, "status": "needs_validation", "errors": [message], "diagnostics": diagnostics([message], code="unsupported_schema", source="prediction_contract")}
    if legacy_contract:
        errors.append("schema-1/legacy prediction_contract 只能审计，必须重新验证后才能发布。")
    v2_errors: list[str] = []
    if resolved_contract.get("contract_hash") != compute_contract_hash(resolved_contract):
        v2_errors.append("prediction_contract contract_hash mismatch")
    artifact_registry = extra.get("registry_snapshot") if isinstance(extra.get("registry_snapshot"), Mapping) else None
    artifact_manifest = extra.get("dataset_manifest") if isinstance(extra.get("dataset_manifest"), Mapping) else None
    resolved_registry = registry_snapshot if isinstance(registry_snapshot, Mapping) else artifact_registry
    resolved_manifest = dataset_manifest if isinstance(dataset_manifest, Mapping) else artifact_manifest
    manifest_registry: Mapping[str, Any] | None = None
    if resolved_registry is None:
        v2_errors.append("v2 artifact 缺少 registry_snapshot")
    else:
        from .feature_registry import compute_registry_hash, get_model_profile, validate_registry
        # A full registry can be hashed directly. build_registry_snapshot
        # intentionally carries the parent registry hash, so for that compact
        # shape the hash is verified against the contract provenance while the
        # selected definitions are compared semantically below.
        if "model_profile" in resolved_registry and "model_profiles" not in resolved_registry:
            computed_registry_hash = resolved_registry.get("registry_hash")
            if not isinstance(computed_registry_hash, str) or not computed_registry_hash.strip():
                v2_errors.append("registry_snapshot registry_hash is missing")
            registry_payload = resolved_registry.get("registry_payload")
            if isinstance(registry_payload, Mapping):
                manifest_registry = registry_payload
                registry_report = validate_registry(registry_payload, require_approved=True)
                if not registry_report.get("ok"):
                    v2_errors.extend("registry_payload " + str(error) for error in registry_report.get("errors", []))
                computed_payload_hash = compute_registry_hash(registry_payload)
                if computed_registry_hash != computed_payload_hash:
                    v2_errors.append("compact registry_snapshot registry payload hash mismatch")
                if registry_payload.get("registry_version") != resolved_registry.get("registry_version"):
                    v2_errors.append("compact registry_snapshot registry version mismatch")
                profile_id = resolved_registry.get("profile_id")
                if resolved_contract.get("model_profile_id") != profile_id:
                    v2_errors.append("prediction_contract model_profile_id 与 compact registry_snapshot profile_id 不一致")
                try:
                    expected_profile = get_model_profile(registry_payload, profile_id)
                except (KeyError, TypeError, ValueError):
                    expected_profile = None
                if not isinstance(expected_profile, Mapping):
                    v2_errors.append("compact registry_snapshot profile cannot be proven from registry_payload")
                elif dict(resolved_registry.get("model_profile") or {}) != dict(expected_profile):
                    v2_errors.append("compact registry_snapshot model_profile mismatch with registry_payload")
                if isinstance(expected_profile, Mapping):
                    definitions = {
                        item.get("feature_id"): item
                        for item in registry_payload.get("features", [])
                        if isinstance(item, Mapping) and item.get("feature_id")
                    }
                    expected_features = [
                        copy.deepcopy(dict(definitions[feature_id]))
                        for feature_id in expected_profile.get("feature_ids", [])
                        if feature_id in definitions
                    ]
                    if list(resolved_registry.get("features") or []) != expected_features:
                        v2_errors.append("compact registry_snapshot features mismatch with registry_payload profile")
            else:
                v2_errors.append("compact registry_snapshot 缺少 registry_payload，无法证明 registry approved")
            selected = resolved_registry.get("features")
            selected_hash = resolved_registry.get("selected_features_hash")
            if not isinstance(selected, list) or not isinstance(selected_hash, str) or not selected_hash.strip():
                v2_errors.append("compact registry_snapshot 缺少 selected_features_hash")
            else:
                selected_payload = {
                    "profile_id": resolved_registry.get("profile_id"),
                    "model_profile": resolved_registry.get("model_profile"),
                    "features": selected,
                }
                expected_selected_hash = hashlib.sha256(
                    json.dumps(selected_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if selected_hash != expected_selected_hash:
                    v2_errors.append("compact registry_snapshot selected feature hash mismatch")
        else:
            manifest_registry = resolved_registry
            registry_for_validation = dict(resolved_registry)
            registry_for_validation.pop("registry_hash", None)
            registry_report = validate_registry(registry_for_validation, require_approved=True)
            if not registry_report.get("ok"):
                v2_errors.extend("registry " + str(error) for error in registry_report.get("errors", []))
            profile_id = resolved_contract.get("model_profile_id")
            profiles = resolved_registry.get("model_profiles")
            profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
            if not isinstance(profile, Mapping):
                v2_errors.append("registry model_profile cannot be proven")
            elif profile.get("status") != "approved":
                v2_errors.append("registry model_profile is not approved")
            computed_registry_hash = compute_registry_hash(registry_for_validation)
            if resolved_registry.get("registry_hash") != computed_registry_hash:
                v2_errors.append("registry_snapshot registry_hash mismatch")
        if resolved_contract.get("feature_registry_version") != resolved_registry.get("registry_version") or resolved_contract.get("feature_registry_hash") != computed_registry_hash:
            v2_errors.append("feature registry version/hash mismatch")
    if resolved_manifest is None:
        v2_errors.append("v2 artifact 缺少 dataset_manifest")
    else:
        try:
            from .dataset_manifest import compute_dataset_manifest_hash
            computed_manifest_hash = compute_dataset_manifest_hash(resolved_manifest)
        except Exception:
            computed_manifest_hash = None
        if resolved_manifest.get("manifest_hash") != computed_manifest_hash:
            v2_errors.append("dataset_manifest manifest_hash mismatch")
        if resolved_contract.get("dataset_manifest_hash") != computed_manifest_hash:
            v2_errors.append("dataset manifest hash mismatch")
        if manifest_registry is None:
            v2_errors.append("dataset_manifest 缺少可验证 registry")
        else:
            from .dataset_manifest import validate_dataset_manifest
            manifest_report = validate_dataset_manifest(
                resolved_manifest,
                manifest_registry,
                require_approved=True,
            )
            if not manifest_report.get("ok"):
                v2_errors.extend("dataset_manifest " + str(error) for error in manifest_report.get("errors", []))
            if resolved_manifest.get("model_profile_id") != resolved_contract.get("model_profile_id"):
                v2_errors.append("dataset_manifest model_profile_id 与 prediction_contract 不一致")
    canonical = _normalized_columns(resolved_contract.get("canonical_feature_cols"))
    effective = _normalized_columns(resolved_contract.get("effective_feature_cols"))
    removed = _normalized_columns(resolved_contract.get("removed_feature_cols"))
    feature_cols_v2 = _normalized_columns(resolved_contract.get("feature_cols"))
    workflow_cols = _normalized_columns(resolved_contract.get("workflow_feature_cols"))
    manual_cols = _normalized_columns(resolved_contract.get("manual_input_feature_cols"))
    molecular_cols = _normalized_columns(resolved_contract.get("molecular_workflow_feature_cols"))
    derived_cols = _normalized_columns(resolved_contract.get("derived_feature_cols"))
    if feature_cols_v2 != canonical:
        v2_errors.append("feature_cols must equal canonical_feature_cols")
    if feature_cols_v2 != workflow_cols + manual_cols:
        v2_errors.append("feature source partitions do not match feature_cols")
    if workflow_cols != molecular_cols + derived_cols:
        v2_errors.append("workflow source partitions do not match workflow_feature_cols")
    source_columns_decl = resolved_contract.get("source_columns")
    workflow_columns_decl = resolved_contract.get("workflow_source_columns")
    source_fields_decl = resolved_contract.get("workflow_source_fields")
    if isinstance(source_columns_decl, list) and isinstance(workflow_columns_decl, list) and source_columns_decl != workflow_columns_decl:
        v2_errors.append("workflow_source_columns must equal source_columns")
    if isinstance(source_fields_decl, list) and isinstance(source_columns_decl, list) and source_columns_decl != source_fields_decl:
        v2_errors.append("workflow_source_fields must equal source_columns")
    if isinstance(source_fields_decl, list) and isinstance(workflow_columns_decl, list) and workflow_columns_decl != source_fields_decl:
        v2_errors.append("workflow_source_columns must equal workflow_source_fields")
    if (set(effective) - set(canonical) or set(removed) - set(canonical) or set(effective) & set(removed) or set(effective) | set(removed) != set(canonical)):
        v2_errors.append("effective/removed feature columns are inconsistent with canonical columns")
    if resolved_registry is not None:
        registry_features = {
            str(item.get("name")): item
            for item in resolved_registry.get("features", [])
            if isinstance(item, Mapping) and item.get("name")
        }
        for feature in canonical:
            definition = registry_features.get(feature)
            if definition is None:
                v2_errors.append(f"feature is not registered: {feature}")
            elif definition.get("status") != "approved":
                v2_errors.append(f"feature is not approved: {feature}")
        contract_definitions = {
            str(item.get("name")): item
            for item in resolved_contract.get("feature_definitions") or []
            if isinstance(item, Mapping) and item.get("name")
        }
        for feature in canonical:
            expected = registry_features.get(feature)
            actual = contract_definitions.get(feature)
            if expected is None or actual is None or dict(expected) != dict(actual):
                v2_errors.append(f"feature definition mismatch: {feature}")
        registry_workflow = {
            str(item.get("name")) for item in registry_features.values()
            if item.get("source_type") in {"molecular_workflow", "derived_workflow"}
        }
        registry_manual = {
            str(item.get("name")) for item in registry_features.values()
            if item.get("source_type") == "manual_input"
        }
        if set(workflow_cols) != registry_workflow.intersection(canonical):
            v2_errors.append("workflow feature partition does not match registry source_type")
        if set(manual_cols) != registry_manual.intersection(canonical):
            v2_errors.append("manual feature partition does not match registry source_type")
    # Legacy contracts are still inspected for useful diagnostics, but their
    # v2-only hash/partition checks must not mask the underlying compatibility
    # errors or change their required needs_validation status.
    if legacy_contract:
        v2_errors.clear()
    if v2_errors:
        return {"ok": False, "status": "invalid", "errors": v2_errors, "diagnostics": diagnostics(v2_errors, code="contract_invalid", source="prediction_contract")}

    required_fields = _CONTRACT_FIELDS - {"source_columns", "workflow_source_columns"}
    if legacy_contract:
        required_fields = set(_CONTRACT_FIELDS)
    missing_contract_fields = sorted(required_fields - set(resolved_contract))
    if missing_contract_fields:
        errors.append(
            "prediction_contract 缺少必需字段：" + ", ".join(missing_contract_fields)
        )
    if schema_version is None:
        errors.append("prediction_contract schema_version is required")
    elif not legacy_contract and resolved_contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("prediction_contract 的 schema_version 不受支持。")

    if artifact.get("model") is None and artifact.get("pipeline") is None:
        errors.append("artifact 缺少 model 或 pipeline。")
    artifact_target = str(artifact.get("target_col") or "").strip()
    contract_target = str(resolved_contract.get("target_col") or "").strip()
    if not artifact_target or not contract_target:
        errors.append("artifact 或 prediction_contract 缺少 target_col。")
    elif artifact_target != contract_target:
        errors.append("artifact 与 prediction_contract 的 target_col 不一致。")

    contract_features = _normalized_columns(resolved_contract.get("feature_cols"))
    artifact_features = _normalized_columns(artifact.get("feature_cols"))
    if not contract_features or len(contract_features) != len(resolved_contract.get("feature_cols") or []):
        errors.append("prediction_contract 缺少无重复的精确 feature_cols。")
    if not artifact_features:
        errors.append("artifact 缺少精确 feature_cols。")
    elif contract_features and artifact_features != contract_features:
        errors.append("artifact 与 prediction_contract 的 feature_cols 顺序或内容不一致。")

    if not isinstance(resolved_contract.get("workflow_source_fields"), list):
        errors.append("prediction_contract 的 workflow_source_fields 必须是列表。")
    if not legacy_contract:
        for alias in ("source_columns", "workflow_source_columns"):
            value = resolved_contract.get(alias)
            if value is not None and not isinstance(value, list):
                errors.append(f"prediction_contract 的 {alias} 必须是列表（如提供）。")
        if isinstance(source_fields_decl, list) and isinstance(source_columns_decl, list) and source_columns_decl != source_fields_decl:
            errors.append("prediction_contract 的 workflow_source_fields 与 source_columns 不一致。")
        if isinstance(source_fields_decl, list) and isinstance(workflow_columns_decl, list) and workflow_columns_decl != source_fields_decl:
            errors.append("prediction_contract 的 workflow_source_fields 与 workflow_source_columns 不一致。")
    elif isinstance(resolved_contract.get("source_columns"), list) and isinstance(resolved_contract.get("workflow_source_columns"), list) and resolved_contract.get("source_columns") != resolved_contract.get("workflow_source_columns"):
        errors.append("prediction_contract 的 workflow source 列记录不一致。")
    if not isinstance(resolved_contract.get("numeric_ranges"), Mapping):
        errors.append("prediction_contract 的 numeric_ranges 必须是映射。")
    else:
        for column, value in resolved_contract["numeric_ranges"].items():
            if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
                errors.append(f"numeric_ranges[{column}] 缺少 min/max。")
                continue
            try:
                minimum = float(value["min"])
                maximum = float(value["max"])
            except (TypeError, ValueError):
                errors.append(f"numeric_ranges[{column}] 不是数值范围。")
                continue
            if not np.isfinite([minimum, maximum]).all() or minimum > maximum:
                errors.append(f"numeric_ranges[{column}] 不是有限的有效范围。")

    resolution = resolve_prediction_feature_contract(
        model=artifact.get("model"),
        pipeline=artifact.get("pipeline"),
        artifact=artifact,
    )
    if not resolution.get("ok"):
        errors.extend(str(error) for error in resolution.get("errors", []))
    elif contract_features and list(resolution.get("feature_cols", [])) != contract_features:
        errors.append("模型公开的特征列与 prediction_contract 不一致。")

    source_columns = (
        resolved_contract.get("workflow_source_fields")
        if isinstance(resolved_contract.get("workflow_source_fields"), list)
        else resolved_contract.get("source_columns") or []
    )
    workflow_present = bool(resolved_contract.get("workflow_present"))
    if source_columns and not workflow_present:
        errors.append("分子源列存在，但 prediction_contract 缺少可复现 workflow。")
    if workflow_present and not resolved_contract.get("workflow_hash"):
        errors.append("workflow 缺少 workflow_hash。")
    if workflow_present and resolved_contract.get("workflow_schema_version") is None:
        errors.append("workflow 缺少 schema_version。")
    molecular_indicated = bool(
        resolved_contract.get("molecular_features_indicated")
        or source_columns
        or _is_molecular_feature_set(contract_features)
    )
    if molecular_indicated and not workflow_present:
        errors.append("模型包含分子特征，但缺少可复现 molecular workflow。")
    actual_imputer = _has_usable_preprocessor(artifact, "imputer")
    actual_scaler = _has_usable_preprocessor(artifact, "scaler")
    if bool(resolved_contract.get("imputer_present")) != actual_imputer:
        errors.append("prediction_contract 的 imputer_present 与 artifact 不一致或不可用。")
    if bool(resolved_contract.get("scaler_present")) != actual_scaler:
        errors.append("prediction_contract 的 scaler_present 与 artifact 不一致或不可用。")
    actual_pipeline = _is_usable_pipeline(artifact.get("pipeline"))
    if bool(resolved_contract.get("pipeline_present")) != actual_pipeline:
        errors.append("prediction_contract 的 pipeline_present 与 artifact 不一致或不可用。")
    if molecular_indicated:
        if not (actual_pipeline or actual_imputer or actual_scaler):
            errors.append("分子特征 artifact 缺少可用 pipeline、imputer 或 scaler。")

    extra = _as_mapping(artifact.get("extra"))
    workflow_payload = extra.get("molecular_feature_workflow")
    if workflow_present and not isinstance(workflow_payload, Mapping):
        errors.append("prediction_contract 声明存在 workflow，但 artifact 未保存 workflow。")
    elif workflow_present and isinstance(workflow_payload, Mapping):
        if workflow_payload.get("workflow_hash") != resolved_contract.get("workflow_hash"):
            errors.append("artifact workflow_hash 与 prediction_contract 不一致。")
        if workflow_payload.get("schema_version") != resolved_contract.get(
            "workflow_schema_version"
        ):
            errors.append("artifact workflow schema_version 与 prediction_contract 不一致。")
        actual_sources = collect_workflow_source_columns(workflow_payload)
        expected_sources = resolved_contract.get("workflow_source_fields")
        if not isinstance(expected_sources, list):
            expected_sources = resolved_contract.get("workflow_source_columns")
        if actual_sources != expected_sources:
            errors.append("artifact workflow source 列与 prediction_contract 不一致。")
        workflow_features = _normalized_columns(workflow_payload.get("final_feature_names"))
        if not workflow_features:
            errors.append("artifact workflow 缺少 final_feature_names。")
        elif contract_features and workflow_features != contract_features:
            missing = [column for column in contract_features if column not in workflow_features]
            extra_features = [column for column in workflow_features if column not in contract_features]
            detail = []
            if missing:
                detail.append("缺少 " + ", ".join(missing[:8]))
            if extra_features:
                detail.append("多出 " + ", ".join(extra_features[:8]))
            errors.append(
                "artifact workflow 的 final_feature_names 必须与 prediction_contract.feature_cols "
                "完全一致" + ("（" + "；".join(detail) + "）。" if detail else ".")
            )

    status = "needs_validation" if legacy_contract else ("valid" if not errors else "invalid")
    return {"ok": not errors, "status": status, "errors": errors, "diagnostics": diagnostics(errors) if errors else []}


def make_publication_entry(
    *,
    material_key: str,
    target_key: str,
    artifact_path: str,
    artifact_hash: str,
    label: str,
    unit: str,
    description: str,
    contract: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    version: str,
    published_at: str,
    publication_status: str = "needs_validation",
    enabled: bool = False,
    gate_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate_ok = (
        isinstance(gate_report, Mapping)
        and gate_report.get("ok") is True
        and str(gate_report.get("status") or "").strip().lower() == "valid"
    )
    is_published = str(publication_status).strip().lower() == "published" and bool(enabled) and gate_ok
    return {
        "id": f"{material_key}:{target_key}:{version}",
        "material_key": str(material_key),
        "target_key": str(target_key),
        "artifact_path": str(artifact_path),
        "artifact_hash": str(artifact_hash),
        "label": str(label),
        "unit": str(unit),
        "description": str(description),
        "contract": copy.deepcopy(dict(contract)),
        "metrics": copy.deepcopy(dict(metrics or {})),
        "version": str(version),
        "published_at": str(published_at),
        "enabled": is_published,
        "publication_status": "published" if is_published else "needs_validation",
        "gate_report": copy.deepcopy(dict(gate_report or {})),
    }


def _publication_models(config: dict[str, Any], material_key: str, target_key: str) -> list[dict[str, Any]]:
    materials = config.setdefault("materials", {})
    material = materials.setdefault(material_key, {})
    targets = material.setdefault("targets", {})
    target = targets.setdefault(target_key, {})
    models = target.setdefault("models", [])
    if not isinstance(models, list):
        raise ValueError("门户配置中的 models 必须是列表。")
    return models


def _validate_entry_for_activation(entry: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Revalidate the artifact before changing active-release state."""
    gate_report = entry.get("gate_report")
    if not (isinstance(gate_report, Mapping) and gate_report.get("ok") is True
            and str(gate_report.get("status") or "").strip().lower() == "valid"):
        raise ValueError("只能激活已通过发布门禁的 published entry。")
    artifact = entry.get("_artifact")
    if not isinstance(artifact, Mapping):
        artifact_path = str(entry.get("artifact_path") or "").strip()
        if artifact_path:
            roots = [Path(str(config[key])) for key in ("project_root", "portal_root", "root") if config.get(key)]
            roots.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
            path = Path(artifact_path)
            if not path.is_absolute():
                path = next(((root / path).resolve() for root in roots if (root / path).is_file()), (roots[0] / path).resolve())
            try:
                from .model_io import load_model_artifact_bytes
                artifact = load_model_artifact_bytes(path.read_bytes()) if path.is_file() else None
            except Exception:
                artifact = None
    if not isinstance(artifact, Mapping):
        raise ValueError("发布 entry 缺少可验证 artifact。")
    contract = entry.get("contract")
    if not isinstance(contract, Mapping):
        contract = _as_mapping(_as_mapping(artifact.get("extra")).get("prediction_contract"))
    if not contract:
        raise ValueError("发布 entry 缺少可验证 prediction_contract。")
    artifact_path = str(entry.get("artifact_path") or "").strip()
    artifact_hash = str(entry.get("artifact_hash") or "").strip().lower()
    if not artifact_path or not artifact_hash:
        raise ValueError("发布 entry 缺少 artifact_path 或 artifact_hash。")
    if artifact_path or artifact_hash:
        roots = [Path(str(config[key])) for key in ("project_root", "portal_root", "root") if config.get(key)]
        roots.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
        path = Path(artifact_path)
        if not path.is_absolute():
            path = next(((root / path).resolve() for root in roots if (root / path).is_file()), (roots[0] / path).resolve())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().lower() != artifact_hash:
            raise ValueError("发布 artifact_hash 校验失败。")
    report = validate_publication_artifact(artifact, contract)
    if report.get("ok") is not True or str(report.get("status") or "").lower() != "valid":
        raise ValueError("发布 artifact 未通过重新验证：" + "; ".join(map(str, report.get("errors") or [])))


def activate_publication(
    config: dict[str, Any], *, material_key: str, target_key: str, entry: dict[str, Any]
) -> dict[str, Any]:
    """Activate one release and disable all other releases in place."""

    version = str(entry.get("version") or "").strip()
    if not version:
        raise ValueError("发布版本不能为空。")
    _validate_entry_for_activation(entry, config)
    if (
        str(entry.get("publication_status") or "").strip().lower() != "published"
        or entry.get("enabled") is not True
    ):
        raise ValueError("只能激活已通过发布门禁的 published entry。")
    models = _publication_models(config, material_key, target_key)
    for model in models:
        if isinstance(model, dict):
            model["enabled"] = False
    entry["enabled"] = True
    entry.setdefault("publication_status", "published")
    replaced = False
    for index, model in enumerate(models):
        if isinstance(model, dict) and str(model.get("version") or "") == version:
            models[index] = entry
            replaced = True
            break
    if not replaced:
        models.append(entry)
    return config


def rollback_publication(
    config: dict[str, Any], *, material_key: str, target_key: str, version: str
) -> dict[str, Any]:
    materials = config.get("materials")
    material = materials.get(material_key) if isinstance(materials, Mapping) else None
    targets = material.get("targets") if isinstance(material, Mapping) else None
    target = targets.get(target_key) if isinstance(targets, Mapping) else None
    models = target.get("models") if isinstance(target, Mapping) else None
    if not isinstance(models, list):
        raise ValueError("门户配置中不存在可回退的 models 列表。")
    requested = str(version).strip()
    matches = [
        model
        for model in models
        if isinstance(model, dict) and str(model.get("version") or "") == requested
    ]
    if not matches:
        raise ValueError(f"Unknown publication version（未知发布版本）: {requested}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate publication version（重复发布版本）: {requested}")
    target_entry = matches[0]
    _validate_entry_for_activation(target_entry, config)
    if str(target_entry.get("publication_status") or "").strip().lower() != "published":
        raise ValueError("只能回退到已通过发布门禁的 published entry。")
    for model in models:
        if not isinstance(model, dict):
            continue
        model["enabled"] = str(model.get("version") or "") == requested
    return config


def portal_health_label(running: bool) -> str:
    return "可访问" if bool(running) else "未启动"


def is_port_open(host: str = "127.0.0.1", port: int = 8555, *, timeout: float = 0.2) -> bool:
    """Return whether a TCP service accepts connections without starting it."""
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout)):
            return True
    except (OSError, TypeError, ValueError):
        return False



PORTAL_DEFAULT_PORT = 8555
PORTAL_SCRIPT_NAME = "UserPrediction.py"


def _portal_project_root(project_root: str | os.PathLike[str] | None = None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def portal_runtime_file(project_root: str | os.PathLike[str] | None = None) -> Path:
    return _portal_project_root(project_root) / "prediction_portal" / "portal_runtime.json"


def _read_portal_runtime_state(project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = portal_runtime_file(project_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write_portal_runtime_state(
    state: Mapping[str, Any], project_root: str | os.PathLike[str] | None = None
) -> None:
    path = portal_runtime_file(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(dict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _clear_portal_runtime_state(project_root: str | os.PathLike[str] | None = None) -> None:
    try:
        portal_runtime_file(project_root).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return


def _is_process_running(pid: Any) -> bool:
    try:
        process_id = int(pid)
        if process_id <= 0:
            return False
    except (TypeError, ValueError):
        return False

    if sys.platform.startswith("win"):
        try:
            import psutil

            process = psutil.Process(process_id)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except ImportError:
            try:
                completed = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {process_id}"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return str(process_id) in (completed.stdout or "")
            except (OSError, subprocess.SubprocessError):
                return False
        except (OSError, psutil.Error):
            return False

    try:
        os.kill(process_id, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False

def _is_managed_portal_process(pid: Any, port: int = PORTAL_DEFAULT_PORT) -> bool:
    """Confirm the tracked PID still belongs to UserPrediction.py."""
    try:
        import psutil

        command_line = " ".join(psutil.Process(int(pid)).cmdline()).lower()
        script_matches = PORTAL_SCRIPT_NAME.lower() in command_line
        port_matches = f"--server.port {int(port)}" in command_line or f"--server.port={int(port)}" in command_line
        return script_matches and port_matches
    except Exception:
        return False


def _portal_command(
    project_root: str | os.PathLike[str],
    python_executable: str | None = None,
    port: int = PORTAL_DEFAULT_PORT,
) -> list[str]:
    root = _portal_project_root(project_root)
    return [
        str(python_executable or sys.executable),
        "-m",
        "streamlit",
        "run",
        str(root / PORTAL_SCRIPT_NAME),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def portal_process_status(
    project_root: str | os.PathLike[str] | None = None,
    *,
    port: int = PORTAL_DEFAULT_PORT,
) -> dict[str, Any]:
    state = _read_portal_runtime_state(project_root)
    pid = state.get("pid")
    if pid is not None and _is_process_running(pid):
        return {
            "status": "running" if is_port_open("127.0.0.1", port) else "starting",
            "managed": True,
            "pid": int(pid),
            "port": int(port),
        }
    if pid is not None:
        _clear_portal_runtime_state(project_root)
    if is_port_open("127.0.0.1", port):
        return {"status": "running", "managed": False, "pid": None, "port": int(port)}
    return {"status": "stopped", "managed": False, "pid": None, "port": int(port)}


def start_prediction_portal(
    project_root: str | os.PathLike[str] | None = None,
    *,
    python_executable: str | None = None,
    port: int = PORTAL_DEFAULT_PORT,
) -> dict[str, Any]:
    root = _portal_project_root(project_root)
    current = portal_process_status(root, port=port)
    if current["status"] in {"running", "starting"}:
        return {**current, "started": False}

    script_path = root / PORTAL_SCRIPT_NAME
    if not script_path.is_file():
        return {
            "status": "error",
            "started": False,
            "error": f"未找到门户脚本：{script_path}",
            "port": int(port),
        }

    log_path = root / "prediction_portal" / "portal_runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    command = _portal_command(root, python_executable, port)
    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
    except (OSError, ValueError) as exc:
        return {"status": "error", "started": False, "error": str(exc), "port": int(port)}

    _write_portal_runtime_state(
        {
            "pid": int(process.pid),
            "port": int(port),
            "script": str(script_path),
            "command": command,
        },
        root,
    )
    return {"status": "starting", "started": True, "managed": True, "pid": int(process.pid), "port": int(port)}


def stop_prediction_portal(
    project_root: str | os.PathLike[str] | None = None,
    *,
    port: int = PORTAL_DEFAULT_PORT,
) -> dict[str, Any]:
    root = _portal_project_root(project_root)
    state = _read_portal_runtime_state(root)
    pid = state.get("pid")
    if pid is None:
        current = portal_process_status(root, port=port)
        if current.get("status") == "running" and not current.get("managed"):
            return {**current, "stopped": False, "error": "当前端口由外部进程占用，未执行强制停止。"}
        return {"status": "stopped", "stopped": False, "pid": None, "port": int(port)}

    if not _is_process_running(pid):
        _clear_portal_runtime_state(root)
        return {"status": "stopped", "stopped": False, "pid": int(pid), "port": int(port)}
    if not _is_managed_portal_process(pid, port):
        return {
            "status": "error",
            "stopped": False,
            "pid": int(pid),
            "port": int(port),
            "error": "记录的 PID 不再匹配 UserPrediction.py，未执行强制停止。",
        }

    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(int(pid), 15)
    except (OSError, ValueError) as exc:
        return {"status": "error", "stopped": False, "pid": int(pid), "port": int(port), "error": str(exc)}

    _clear_portal_runtime_state(root)
    return {"status": "stopped", "stopped": True, "pid": int(pid), "port": int(port)}

def should_show_publication(contract_report: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(contract_report, Mapping)
        and contract_report.get("ok") is True
        and str(contract_report.get("status") or "").strip().lower() == "valid"
        and not contract_report.get("errors")
    )


def select_active_publication(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    active = [
        model for model in models
        if isinstance(model, dict)
        and model.get("enabled") is True
        and str(model.get("publication_status") or "").strip().lower() == "published"
        and isinstance(model.get("gate_report"), Mapping)
        and model["gate_report"].get("ok") is True
        and str(model["gate_report"].get("status") or "").strip().lower() == "valid"
    ]
    if len(active) > 1:
        raise ValueError("Multiple enabled publication releases（多个启用版本）")
    return active[0] if active else None
