"""Pure publication contracts and version helpers for the user portal."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .prediction_contract import resolve_prediction_feature_contract
from .prediction_molecular_baseline import collect_workflow_source_columns


CONTRACT_SCHEMA_VERSION = 1


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
        return [step for _, step in pipeline.steps]
    except (AttributeError, TypeError, ValueError):
        return []


def _has_preprocessor(artifact: Mapping[str, Any], kind: str) -> bool:
    if artifact.get(kind) is not None:
        return True
    pipeline = artifact.get("pipeline")
    for step in _pipeline_steps(pipeline):
        name = type(step).__name__.lower()
        if kind == "imputer" and "imput" in name:
            return True
        if kind == "scaler" and ("scaler" in name or "standardize" in name):
            return True
    return False


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


def build_prediction_contract(
    *,
    artifact: Mapping[str, Any],
    feature_cols: Sequence[str],
    target_col: str,
    workflow: Any = None,
    training_frame: Any = None,
    source_frame: Any = None,
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
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_cols": resolved_features,
        "target_col": target,
        "workflow_hash": workflow_payload.get("workflow_hash"),
        "workflow_schema_version": workflow_payload.get("schema_version"),
        "source_columns": source_columns,
        "workflow_present": bool(workflow_payload),
        "pipeline_present": pipeline is not None,
        "imputer_present": _has_preprocessor(artifact, "imputer"),
        "scaler_present": _has_preprocessor(artifact, "scaler"),
        "numeric_ranges": _numeric_ranges(numeric_frame, resolved_features),
    }


def _contract_from_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    extra = _as_mapping(artifact.get("extra"))
    contract = extra.get("prediction_contract")
    return contract if isinstance(contract, Mapping) else None


def validate_publication_artifact(
    artifact: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a publication diagnostic without mutating the artifact."""

    artifact = _as_mapping(artifact)
    errors: list[str] = []
    resolved_contract = contract or _contract_from_artifact(artifact)
    if resolved_contract is None:
        return {
            "ok": False,
            "status": "needs_validation",
            "errors": ["artifact 缺少 prediction_contract，需要重新验证。"],
        }
    resolved_contract = _as_mapping(resolved_contract)

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

    resolution = resolve_prediction_feature_contract(
        model=artifact.get("model"),
        pipeline=artifact.get("pipeline"),
        artifact=artifact,
    )
    if not resolution.get("ok"):
        errors.extend(str(error) for error in resolution.get("errors", []))
    elif contract_features and list(resolution.get("feature_cols", [])) != contract_features:
        errors.append("模型公开的特征列与 prediction_contract 不一致。")

    source_columns = resolved_contract.get("source_columns") or []
    workflow_present = bool(resolved_contract.get("workflow_present"))
    if source_columns and not workflow_present:
        errors.append("分子源列存在，但 prediction_contract 缺少可复现 workflow。")
    if workflow_present and not resolved_contract.get("workflow_hash"):
        errors.append("workflow 缺少 workflow_hash。")
    if workflow_present and resolved_contract.get("workflow_schema_version") is None:
        errors.append("workflow 缺少 schema_version。")
    if source_columns and not (
        resolved_contract.get("pipeline_present")
        or resolved_contract.get("imputer_present")
        or resolved_contract.get("scaler_present")
    ):
        errors.append("分子特征 artifact 缺少 pipeline、imputer 或 scaler 预处理元数据。")

    status = "valid" if not errors else "invalid"
    return {"ok": not errors, "status": status, "errors": errors}


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
) -> dict[str, Any]:
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
        "enabled": True,
        "publication_status": "published",
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


def activate_publication(
    config: dict[str, Any], *, material_key: str, target_key: str, entry: dict[str, Any]
) -> dict[str, Any]:
    """Activate one release and disable all other releases in place."""

    models = _publication_models(config, material_key, target_key)
    version = str(entry.get("version") or "").strip()
    if not version:
        raise ValueError("发布版本不能为空。")
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
    models = _publication_models(config, material_key, target_key)
    requested = str(version).strip()
    found = False
    for model in models:
        if not isinstance(model, dict):
            continue
        is_requested = str(model.get("version") or "") == requested
        model["enabled"] = is_requested
        if is_requested:
            found = True
    if not found:
        raise ValueError(f"Unknown publication version（未知发布版本）: {requested}")
    return config


def portal_health_label(running: bool) -> str:
    return "可访问" if bool(running) else "未启动"


def should_show_publication(contract_report: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(contract_report, Mapping)
        and contract_report.get("ok")
        and contract_report.get("status") != "needs_validation"
        and not contract_report.get("errors")
    )


def select_active_publication(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    for model in models:
        if isinstance(model, dict) and model.get("enabled"):
            return model
    return None
