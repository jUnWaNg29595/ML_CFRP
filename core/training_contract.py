"""Training-time semantic contract locking and audit helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .dataset_manifest import compute_dataset_manifest_hash, validate_dataset_manifest
from .feature_registry import build_registry_snapshot, load_registry, validate_registry


def _frame_columns(frame: Any) -> list[str]:
    columns = getattr(frame, "columns", ())
    if columns is None:
        return []
    return [str(column) for column in columns]


def assert_training_context(context: Mapping[str, Any], frame_columns: Sequence[str]) -> None:
    expected = list(context.get("canonical_feature_cols") or context.get("feature_cols") or [])
    actual = [] if frame_columns is None else list(frame_columns)
    if len(expected) != len(set(expected)):
        raise ValueError("training contract contains duplicate feature columns")
    missing = [column for column in expected if column not in actual]
    if missing:
        raise ValueError("training frame missing contract columns: " + ", ".join(map(str, missing)))
    unknown = [column for column in actual if column not in expected]
    if unknown and context.get("reject_unknown_columns", True):
        raise ValueError("training frame contains unregistered columns: " + ", ".join(map(str, unknown)))


def select_training_context_for_model(
    context: Mapping[str, Any] | None,
    model_name: str,
    frame_columns: Sequence[str],
    *,
    graph_model_names: Sequence[str] = (),
    raw_frame_model_names: Sequence[str] = (),
) -> Mapping[str, Any] | None:
    """Apply the numeric feature contract only to models that consume that frame.

    Graph and raw-frame models have their own input contracts (for example SMILES
    strings or auxiliary raw columns), so the canonical numeric context must not
    be asserted against those frames.
    """
    if context is None:
        return None
    if str(model_name) in set(graph_model_names) or str(model_name) in set(raw_frame_model_names):
        return None
    assert_training_context(context, frame_columns)
    return context


def lock_training_contract(registry_path: str | Path, dataset_manifest: Mapping[str, Any], material_type: str, target: str, target_col: str, feature_cols: Sequence[str], frame: Any, workflow: Mapping[str, Any] | None) -> dict[str, Any]:
    registry = load_registry(registry_path)
    report = validate_registry(registry, require_approved=True)
    if not report["ok"]:
        raise ValueError("registry is not approved: " + "; ".join(report["errors"]))
    profile_id = dataset_manifest.get("model_profile_id")
    snapshot = build_registry_snapshot(registry, profile_id)
    frame_column_names = _frame_columns(frame)
    manifest_report = validate_dataset_manifest(dataset_manifest, registry, frame_columns=frame_column_names, require_approved=True)
    if not manifest_report["ok"]:
        raise ValueError("dataset manifest is not approved: " + "; ".join(manifest_report["errors"]))
    columns = list(feature_cols or [])
    assert_training_context({"canonical_feature_cols": columns}, frame_column_names)
    definitions = {item.get("name"): item for item in snapshot.get("features", []) if isinstance(item, Mapping)}
    missing = [column for column in columns if column not in definitions]
    if missing:
        raise ValueError("feature columns are not registered: " + ", ".join(missing))
    expected_order = [item.get("name") for item in snapshot.get("features", []) if item.get("name")]
    profile = snapshot.get("model_profile") or {}
    if not profile.get("allow_feature_subset", False) and columns != expected_order:
        raise ValueError(
            "feature columns do not match approved model profile order: "
            + ", ".join(expected_order)
        )
    if profile.get("allow_feature_subset", False):
        positions = {name: index for index, name in enumerate(expected_order)}
        if any(positions.get(column, -1) >= positions.get(next_column, -1)
               for column, next_column in zip(columns, columns[1:])):
            raise ValueError("feature columns do not preserve approved model profile order")
    computed_manifest_hash = manifest_report.get("manifest_hash") or compute_dataset_manifest_hash(dataset_manifest)
    supplied_manifest_hash = dataset_manifest.get("manifest_hash")
    if supplied_manifest_hash is not None and supplied_manifest_hash != computed_manifest_hash:
        raise ValueError("dataset manifest hash does not match computed manifest hash")
    source_partitions = {
        "workflow_feature_cols": [column for column in columns if definitions[column].get("source_type") in {"molecular_workflow", "derived_workflow"}],
        "manual_input_feature_cols": [column for column in columns if definitions[column].get("source_type") == "manual_input"],
    }
    return {
        "material_type": str(material_type), "target": str(target), "target_col": str(target_col),
        "canonical_feature_cols": columns, "effective_feature_cols": columns.copy(), "removed_feature_cols": [],
        "feature_registry_version": snapshot.get("registry_version"), "feature_registry_hash": snapshot.get("registry_hash"),
        "registry_snapshot": snapshot, "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_manifest_hash": computed_manifest_hash,
        "dataset_manifest": copy.deepcopy(dict(dataset_manifest)), "workflow": copy.deepcopy(dict(workflow or {})),
        **source_partitions, "reject_unknown_columns": True,
    }


def diagnose_training_blockers(
    registry_path: str | Path,
    manifest: Mapping[str, Any] | None,
) -> list[str]:
    """Return concrete, user-facing reasons why training cannot lock a contract."""
    blockers: list[str] = []
    if not isinstance(manifest, Mapping):
        blockers.append("manifest 不存在：请先在【特征管理】页面完成特征映射审核并批准")
        return blockers

    manifest_status = str(manifest.get("status") or "draft").strip().lower()
    if manifest_status != "approved":
        blockers.append(f"manifest 仍为 {manifest_status}，未批准；请先在【特征管理】页面批准全部特征映射")

    registry_report = None
    try:
        registry = load_registry(registry_path)
        registry_report = validate_registry(registry, require_approved=True)
        if not registry_report["ok"]:
            approval_status = str((registry.get("approval") or {}).get("status") or "draft")
            if approval_status != "approved":
                blockers.append(f"Registry 仍为 {approval_status}，需要先完成 Registry 审核（approval.approved_hash 必须匹配）")
            profile_statuses = []
            for pid, prof in (registry.get("model_profiles") or {}).items():
                if isinstance(prof, Mapping):
                    profile_statuses.append(f"{pid}={prof.get('status', 'unknown')}")
            if profile_statuses:
                blockers.append("model profile 状态: " + ", ".join(profile_statuses) + "；blocked/draft profile 不能锁定训练契约")
            other = [e for e in registry_report["errors"] if "approval" not in e and "profile" not in e]
            if other:
                blockers.append("Registry 校验错误: " + "; ".join(other[:5]))
    except Exception as exc:
        blockers.append(f"Registry 读取失败：{exc}")
        return blockers

    if registry_report is not None:
        try:
            registry = load_registry(registry_path)
            manifest_report = validate_dataset_manifest(manifest, registry, require_approved=False)
            if not manifest_report["ok"]:
                errors = manifest_report["errors"]
                # Split into useful categories
                missing_features = [e for e in errors if e.startswith("manifest missing required profile features")]
                conflicts = [e for e in errors if "unknown feature_id" in e]
                others = [e for e in errors if e not in missing_features and e not in conflicts]
                if missing_features:
                    blockers.append("缺少 feature binding：" + "; ".join(missing_features[:3]))
                if conflicts:
                    blockers.append("存在未登记/冲突的 feature binding：" + "; ".join(conflicts[:5]))
                if others:
                    blockers.append("manifest 校验错误：" + "; ".join(others[:5]))
        except Exception as exc:
            blockers.append(f"manifest 校验异常：{exc}")

    # conflict/unknown binding report runs regardless of registry validity
    for binding in manifest.get("feature_bindings", []) if isinstance(manifest.get("feature_bindings"), list) else []:
        if not isinstance(binding, Mapping):
            continue
        review_status = str(binding.get("review_status") or "unknown")
        if review_status in {"conflict", "unknown"}:
            blockers.append(f"binding {binding.get('feature_id')} 状态为 {review_status}，需要人工处理")
    return blockers


def audit_training_result(context: Mapping[str, Any], train_result: Mapping[str, Any]) -> dict[str, Any]:
    canonical = list(context.get("canonical_feature_cols") or [])
    feature_names = list(train_result.get("feature_names") or [])
    mask = train_result.get("feature_mask")
    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)) and len(mask) == len(canonical):
        effective = [column for column, keep in zip(canonical, mask) if bool(keep)]
    else:
        effective = feature_names or canonical.copy()
    removed = [column for column in canonical if column not in effective]
    return {"canonical_feature_cols": canonical, "effective_feature_cols": effective, "removed_feature_cols": removed, "removed_feature_reasons": {column: "feature_mask" for column in removed}, "publishable": not removed}
