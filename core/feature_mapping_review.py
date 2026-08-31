"""Focused AI-assisted review of raw columns and semantic feature bindings."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .portal_ai_schema import parse_feature_mapping_response, sanitize_ai_context


LOCAL_REVIEWER_ID = "local_user"


def get_local_reviewer_id(config_dir: str | Path | None = None) -> str:
    """Read configured local reviewer ID from portal config, or return 'local_user' default."""
    try:
        from .portal_ai_config import load_ai_config
        config = load_ai_config(config_dir)
        if isinstance(config, Mapping):
            reviewer = config.get("feature_review", {}).get("local_reviewer_id") if isinstance(config.get("feature_review"), Mapping) else None
            if not reviewer:
                reviewer = config.get("local_reviewer_id")
            if isinstance(reviewer, str) and reviewer.strip():
                return reviewer.strip()
    except Exception:
        pass
    return LOCAL_REVIEWER_ID


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


_MOLECULAR_PREFIXES = (
    "fp_", "morgan_", "maccs_", "rdkit_", "mordred_", "coulomb_", "ani_",
    "tda_", "gnn_", "chembert_", "transformer_", "embed_", "desc_", "3d_",
    "fgd_", "epoxy_", "reaction_", "polymer_", "ionic_", "xtb_", "ff_",
)


def _is_molecular_feature_column(column_name: Any) -> bool:
    name = str(column_name or "").strip().lower()
    return any(name.startswith(prefix) for prefix in _MOLECULAR_PREFIXES) or "maccs" in name or "morgan" in name or "rdkit" in name or "mordred" in name


def build_feature_review_context(frame: Any, registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    """Build a small, feature-only context; metrics, prediction state, and generated molecular features are excluded."""
    raw_columns = getattr(frame, "columns", [])
    columns = list(raw_columns) if raw_columns is not None else []
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    if not isinstance(profile, Mapping):
        profile = {}
    target_columns = _target_column_names(profile, columns)
    # 过滤掉目标列以及系统自动提取的分子特征列（仅处理原始特征/工艺列）
    review_columns = [
        column for column in columns
        if str(column) not in target_columns and not _is_molecular_feature_column(column)
    ]
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


def fast_local_feature_mapping(
    columns: list[str],
    registry: Mapping[str, Any],
    profile_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Perform fast, zero-network exact/alias mapping for high-confidence columns.

    Returns:
        (matched_suggestions, remaining_unmapped_columns)
    """
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    profile_feature_ids = set(profile.get("feature_ids", []) if isinstance(profile, Mapping) else [])
    definitions = [
        item for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping) and item.get("feature_id") and item.get("feature_id") in profile_feature_ids
    ]
    matched: list[dict[str, Any]] = []
    mapped_columns: set[str] = set()
    used_feature_ids: set[str] = set()

    for col in columns:
        col_clean = str(col).strip()
        col_norm = _normalized_column(col_clean)
        if not col_norm:
            continue
        for feat in definitions:
            fid = str(feat.get("feature_id") or "")
            if fid in used_feature_ids:
                continue
            names = [fid, str(feat.get("name") or ""), str(feat.get("label") or "")]
            aliases = feat.get("aliases") or []
            if isinstance(aliases, str):
                names.append(aliases)
            elif isinstance(aliases, (list, tuple)):
                names.extend(str(a) for a in aliases if str(a).strip())
            acc_aliases = feat.get("accepted_aliases") or []
            if isinstance(acc_aliases, str):
                names.append(acc_aliases)
            elif isinstance(acc_aliases, (list, tuple)):
                names.extend(str(a) for a in acc_aliases if str(a).strip())

            # 精确或标准化严格相等判定
            if any(col_norm == _normalized_column(n) for n in names if n):
                source_role = str(feat.get("source_type") or "manual_input").strip()
                if source_role not in _APPROVABLE_SOURCE_ROLES:
                    source_role = "manual_input"
                # 工艺/测试字段默认 manual_input
                if _looks_like_process_test_field({"feature_id": fid, "raw_columns": [col_clean]}, feat):
                    source_role = "manual_input"

                sugg = {
                    "feature_id": fid,
                    "raw_columns": [col_clean],
                    "source_role": source_role,
                    "unit": str(feat.get("unit") or "").strip() or None,
                    "confidence": 1.0,
                    "rationale_zh": "本地精确/别名完全匹配（高置信度预映射）",
                    "status": "pending_review",
                    "is_new_proposal": False,
                }
                matched.append(sugg)
                mapped_columns.add(col_clean)
                used_feature_ids.add(fid)
                break

    remaining = [c for c in columns if c not in mapped_columns]
    return matched, remaining


def request_feature_mapping_review(
    client: Any,
    context: Mapping[str, Any],
    *,
    batch_size: int = 30,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Request AI review with smart column chunking and progressive fallback.

    If raw_columns exceeds batch_size, it transparently splits into smaller chunks,
    combines all generated suggestions, preserves successful chunks even if one fails,
    and returns a unified result dictionary.
    """
    if client is None or not callable(getattr(client, "review_feature_mapping", None)):
        raise ValueError("AI 特征审核客户端不可用")

    raw_columns = list(context.get("raw_columns") or [])
    if len(raw_columns) <= batch_size:
        if progress_callback:
            progress_callback(1, 1, f"正在分析全部 {len(raw_columns)} 列...")
        response = client.review_feature_mapping(dict(context))
        return parse_feature_mapping_response(response)

    # 分批并发执行（ThreadPoolExecutor 提速）
    from concurrent.futures import ThreadPoolExecutor, as_completed
    chunks = [raw_columns[i:i + batch_size] for i in range(0, len(raw_columns), batch_size)]
    total_chunks = len(chunks)
    all_suggestions: list[dict[str, Any]] = []
    all_conflicts: list[str] = []
    rationales: list[str] = []
    confidences: list[float] = []
    errors: list[str] = []

    dtypes = context.get("column_dtypes") or {}
    sample_rows = context.get("sample_rows") or []
    all_candidates = list(context.get("candidate_features") or [])

    def _process_chunk(idx: int, chunk_cols: list[str]) -> tuple[int, dict[str, Any] | None, str | None]:
        chunk_dtypes = {c: dtypes[c] for c in chunk_cols if c in dtypes}
        chunk_sample_rows = []
        for row in sample_rows:
            if isinstance(row, dict):
                chunk_sample_rows.append({c: row[c] for c in chunk_cols if c in row})

        chunk_context = {
            "profile_id": context.get("profile_id", ""),
            "raw_columns": chunk_cols,
            "column_dtypes": chunk_dtypes,
            "sample_rows": chunk_sample_rows,
            "candidate_features": all_candidates,
        }
        try:
            resp = client.review_feature_mapping(chunk_context)
            parsed = parse_feature_mapping_response(resp)
            return idx, parsed, None
        except Exception as exc:
            return idx, None, f"批次 {idx}/{total_chunks} 分析异常：{exc}"

    # 使用最多 4 个并发线程同时发起分析，大幅缩短总等待时间
    max_workers = min(4, total_chunks)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_process_chunk, idx, chunk_cols): idx
            for idx, chunk_cols in enumerate(chunks, 1)
        }
        for future in as_completed(future_map):
            completed_count += 1
            idx, parsed, err = future.result()
            if progress_callback:
                progress_callback(completed_count, total_chunks, f"已完成 {completed_count}/{total_chunks} 批数据列分析...")
            if err:
                errors.append(err)
            elif parsed:
                all_suggestions.extend(parsed.get("suggestions") or [])
                all_conflicts.extend(parsed.get("conflicts") or [])
                if parsed.get("rationale_zh"):
                    rationales.append(str(parsed["rationale_zh"]))
                if parsed.get("confidence") is not None:
                    try:
                        confidences.append(float(parsed["confidence"]))
                    except Exception:
                        pass

    if not all_suggestions and errors:
        # 全部批次失败，抛出汇总错误
        raise RuntimeError("分批特征审核全部失败：" + "；".join(errors))

    # 去重合并 suggestions（后批次不覆盖已匹配好的 feature_id）
    seen_fids: set[str] = set()
    deduped_suggs: list[dict[str, Any]] = []
    for s in all_suggestions:
        fid = str(s.get("feature_id") or "")
        if fid and fid not in seen_fids:
            seen_fids.add(fid)
            deduped_suggs.append(s)
        elif not fid:
            deduped_suggs.append(s)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.9
    combined_rationale = "；".join(rationales) if rationales else "分批分析完成"
    if errors:
        combined_rationale += f"（部分批次遇到异常：{'; '.join(errors)}）"

    return {
        "suggestions": deduped_suggs,
        "conflicts": all_conflicts,
        "rationale_zh": combined_rationale[:4000],
        "confidence": avg_conf,
    }


def apply_feature_review_decision(
    manifest: Mapping[str, Any], suggestion: Mapping[str, Any], action: str, reviewer: str | None = None,
    registry: Mapping[str, Any] | None = None, edited: Mapping[str, Any] | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    from .dataset_manifest import compute_dataset_manifest_hash
    updated = copy.deepcopy(dict(manifest))
    action = str(action).strip().lower()
    reviewer = str(reviewer or "").strip() or get_local_reviewer_id()
    now = datetime.now(timezone.utc).isoformat()
    manifest_hash_before = compute_dataset_manifest_hash(manifest)
    record = {
        "action": action,
        "reviewer": reviewer,
        "approved_by": reviewer,
        "approved_at": now,
        "feature_id": suggestion.get("feature_id"),
        "raw_columns": suggestion.get("raw_columns"),
        "source_role": suggestion.get("source_role"),
        "recorded_at": now,
        "manifest_hash_before": manifest_hash_before,
    }
    if action == "reject":
        manifest_hash_after = compute_dataset_manifest_hash(updated)
        record["manifest_hash_after"] = manifest_hash_after
        updated.setdefault("review_records", []).append(record)
        return updated
    if action not in {"accept", "edit_accept"}:
        raise ValueError("unsupported feature review action")
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
        suggestion["raw_columns"] = raw_columns

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
    from .portal_ai_schema import normalize_feature_source_role
    source_role_raw = str(suggestion.get("source_role") or "").strip()
    source_role = normalize_feature_source_role(source_role_raw) or source_role_raw
    if source_role not in {"manual_input", "molecular_workflow", "derived_workflow"}:
        raise ValueError("source_role 必须是允许的输入/工作流来源")
    if isinstance(registry_feature, Mapping):
        source_type = str(registry_feature.get("source_type") or "").strip()
        if source_type != source_role:
            if action == "edit_accept":
                # 人工明确指定了合法来源类型，自动对齐并更新 Registry 特征定义
                registry_feature["source_type"] = source_role
                if str(registry_feature.get("status") or "").strip().lower() in {"unknown", "blocked", "deprecated"}:
                    registry_feature["status"] = "draft"
            else:
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
            if action == "edit_accept":
                # 人工审核接受后将被阻断/历史未知状态自动解除为 draft 允许状态
                registry_feature["status"] = "draft"
                registry_status = "draft"
            else:
                raise ValueError("registry feature status 不允许批准：" + registry_status)
    if isinstance(registry, Mapping) and profile_id and action == "edit_accept":
        # 检查并解除当前 profile 的 blocked 状态（若所有包含的特征均已处于 draft/approved）
        profiles = registry.get("model_profiles", {})
        curr_profile = profiles.get(profile_id, {})
        if isinstance(curr_profile, dict) and curr_profile.get("status") == "blocked":
            blocked_fids = curr_profile.get("blocked_feature_ids") or []
            if isinstance(blocked_fids, list) and feature_id in blocked_fids:
                curr_profile["blocked_feature_ids"] = [fid for fid in blocked_fids if fid != feature_id]
            # 如果没有其他 blocked 特征，自动解除 profile 的 blocked 状态为 draft
            if not curr_profile.get("blocked_feature_ids"):
                curr_profile["status"] = "draft"
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
    if isinstance(registry, Mapping):
        updated["status"] = compute_feature_manifest_status(updated, registry, str(profile_id or ""))
        approval_status = updated["status"]
        updated.setdefault("approval", {}).update({"status": approval_status})
        if approval_status == "approved":
            updated["approval"].update({"approved_by": reviewer, "approved_at": binding["approved_at"]})
        else:
            updated["approval"].update({"mapped_by": reviewer, "mapped_at": binding["approved_at"]})
    else:
        updated.setdefault("approval", {})
        updated["approval"].update({"status": "approved", "approved_by": reviewer, "approved_at": binding["approved_at"]})
    manifest_hash_after = compute_dataset_manifest_hash(updated)
    record.update({
        "feature_id": feature_id.strip(),
        "raw_columns": list(binding["raw_columns"]),
        "source_role": source_role,
        "manifest_hash_after": manifest_hash_after,
    })
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


def compute_feature_manifest_status(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    profile_id: str,
) -> str:
    """Derive the manifest lifecycle state from bindings and governance state.

    ``draft`` means no valid approved binding exists; ``mapped`` means a valid
    subset exists; ``pending_approval`` means all profile features are mapped
    but Registry/Profile approval is still missing; ``approved`` requires the
    complete binding set and approved Registry, profile, and feature entries.
    """
    if not isinstance(manifest, Mapping) or not isinstance(registry, Mapping):
        return "draft"
    profiles = registry.get("model_profiles")
    profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        return "draft"
    required_ids = {str(fid) for fid in (profile.get("feature_ids") or []) if str(fid).strip()}
    definitions = {
        str(item.get("feature_id")): item
        for item in (registry.get("features") or [])
        if isinstance(item, Mapping) and str(item.get("feature_id") or "").strip()
    }
    valid_ids: set[str] = set()
    seen: set[str] = set()
    for binding in (manifest.get("feature_bindings") or []):
        if not isinstance(binding, Mapping):
            continue
        fid = str(binding.get("feature_id") or "").strip()
        raw_columns = binding.get("raw_columns")
        source_role = str(binding.get("source_role") or "").strip()
        feature = definitions.get(fid)
        if (
            fid and fid not in seen and fid in required_ids
            and isinstance(raw_columns, list) and raw_columns
            and all(str(raw).strip() for raw in raw_columns)
            and source_role in _APPROVABLE_SOURCE_ROLES
            and str(binding.get("review_status") or "").strip().lower() == "approved"
            and isinstance(feature, Mapping)
            and str(feature.get("source_type") or "").strip() == source_role
            and str(feature.get("status") or "").strip().lower() in {"draft", "approved"}
        ):
            valid_ids.add(fid)
        seen.add(fid)
    if not valid_ids:
        return "draft"
    if valid_ids != required_ids:
        return "mapped"
    registry_approved = str((registry.get("approval") or {}).get("status") or "").strip().lower() == "approved"
    profile_approved = str(profile.get("status") or "").strip().lower() == "approved"
    features_approved = all(str(definitions[fid].get("status") or "").strip().lower() == "approved" for fid in required_ids if fid in definitions)
    return "approved" if registry_approved and profile_approved and features_approved and required_ids <= valid_ids else "pending_approval"


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
        source_role_original = str(suggestion.get("source_role") or suggestion.get("source_type") or source_role).strip()
        status = str(copy_sugg.get("status") or "unknown").strip().lower()
        raw_columns = [str(col).strip() for col in (copy_sugg.get("raw_columns") or []) if str(col).strip()]
        confidence = copy_sugg.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_val = 0.0
        is_new_proposal = bool(copy_sugg.get("is_new_proposal") or feature_id not in profile_feature_ids)
        registry_feature = definitions.get(feature_id)
        is_process_test_field = _looks_like_process_test_field(copy_sugg, registry_feature)
        original_source_role = source_role
        if is_process_test_field and source_role in {"molecular_workflow", "derived_workflow"}:
            # Process/test inputs default to manual entry. Keep the AI value for
            # audit and force human attention so this is never auto-accepted.
            copy_sugg["source_role"] = "manual_input"
            copy_sugg["source_role_raw"] = source_role_original
            copy_sugg["source_role_defaulted"] = True
            source_role = "manual_input"
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
        elif feature_id not in definitions:
            reasons.append(f"feature_id {feature_id} 不存在于 Registry")
            repair_actions.append("在 Registry 中登记该特征并加入当前 profile 后再接受")
        elif feature_id in definitions:
            reg_status = str(definitions[feature_id].get("status") or "unknown").strip().lower()
            if reg_status not in {"draft", "approved"}:
                reasons.append(f"registry 状态 {reg_status} 不允许批准")
                repair_actions.append("先人工审核 Registry 特征状态（legacy_observed/deprecated/blocked 不能自动批准）")
            registry_source_role = str(definitions[feature_id].get("source_type") or "").strip()
            if registry_source_role not in _APPROVABLE_SOURCE_ROLES:
                reasons.append(f"Registry source_type {registry_source_role or 'unknown'} 无法批准")
                repair_actions.append("先修正 Registry 特征的 source_type")
            elif registry_source_role != source_role:
                reasons.append(f"来源类型 {source_role} 与 Registry source_type {registry_source_role} 不一致")
                repair_actions.append(f"把来源类型修改为 {registry_source_role}，与 Registry 定义保持一致")
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
        # 工艺/测试字段被 AI 标为 workflow/derived → 默认改为 manual_input，且必须人工确认
        if is_process_test_field and original_source_role in {"molecular_workflow", "derived_workflow"}:
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
    reviewer: str | None = None,
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
    reviewer_name = str(reviewer or "").strip() or get_local_reviewer_id()
    classification = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    safe_by_id = {str(item.get("feature_id")): item for item in classification["safe"]}
    selected = list(dict.fromkeys(str(fid).strip() for fid in (selected_feature_ids or []) if str(fid).strip()))
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
    from .dataset_manifest import compute_dataset_manifest_hash
    updated = copy.deepcopy(dict(manifest))
    manifest_hash_before = compute_dataset_manifest_hash(manifest)
    now = datetime.now(timezone.utc).isoformat()
    existing_bindings = [b for b in (updated.get("feature_bindings") or []) if isinstance(b, Mapping)]
    existing_by_id = {str(b.get("feature_id")): b for b in existing_bindings if str(b.get("feature_id") or "").strip()}
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
        previous = existing_by_id.get(feature_id)
        same_binding = isinstance(previous, Mapping) and all(
            previous.get(key) == binding.get(key)
            for key in ("feature_id", "raw_columns", "source_role", "unit", "confidence", "rationale_zh", "review_status")
        )
        if same_binding:
            continue
        new_bindings.append(binding)
        records.append({
            "action": "batch_accept",
            "reviewer": reviewer_name,
            "approved_by": reviewer_name,
            "approved_at": now,
            "feature_id": feature_id,
            "raw_columns": raw_columns,
            "source_role": source_role,
            "recorded_at": now,
            "batch": True,
            "manifest_hash_before": manifest_hash_before,
            "manifest_hash_after": "",  # 写入后统一计算
        })

    merged_bindings = [b for b in existing_bindings if str(b.get("feature_id")) not in {item["feature_id"] for item in new_bindings}] + new_bindings
    updated["feature_bindings"] = merged_bindings

    # 3) manifest 状态升级规则：状态由实际绑定完整度和治理审批共同决定。
    computed_status = compute_feature_manifest_status(updated, registry, profile_id)
    previous_status = str(updated.get("status") or "").strip().lower()
    status_changed = previous_status != computed_status
    if status_changed:
        updated["status"] = computed_status
        approval = updated.setdefault("approval", {})
        if computed_status == "approved":
            approval.update({"status": "approved", "approved_by": reviewer_name, "approved_at": now, "batch_approved": True})
            updated["approved_at"] = now
            updated["approved_by"] = reviewer_name
        elif computed_status == "pending_approval":
            approval.update({"status": "pending_approval", "mapped_by": reviewer_name, "mapped_at": now, "batch_approved": True})
            updated["mapped_at"] = now
            updated["mapped_by"] = reviewer_name
        else:
            approval.update({"status": computed_status, "mapped_by": reviewer_name, "mapped_at": now, "batch_approved": True})
            updated["mapped_at"] = now
            updated["mapped_by"] = reviewer_name
    elif not isinstance(updated.get("approval"), Mapping):
        updated["approval"] = {"status": computed_status}

    # 4) 重新计算 manifest_hash（含审计记录，但排除审计 hash 元数据）。
    updated.setdefault("review_records", []).extend(records)
    current_hash = compute_dataset_manifest_hash(updated)
    for record in updated["review_records"]:
        if isinstance(record, Mapping) and record.get("batch") and not record.get("manifest_hash_after"):
            record["manifest_hash_after"] = current_hash
            record["manifest_hash"] = current_hash
    updated["manifest_hash"] = compute_dataset_manifest_hash(updated)
    return updated


def batch_approve_safe_feature_suggestions(
    manifest: Mapping[str, Any],
    suggestions: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
    profile_id: str,
    reviewer: str | None = None,
) -> dict[str, Any]:
    """兼容旧接口：等价于 batch_accept_feature_bindings 全选安全建议。

    保留为兼容模式（旧调用方/旧测试仍可用），但内部走新的原子化实现，
    且不再无条件把 manifest.status 升级为 approved（遵循状态规则）。
    """
    reviewer_name = str(reviewer or "").strip() or get_local_reviewer_id()
    classification = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    safe_ids = [str(item.get("feature_id")) for item in classification["safe"]]
    return batch_accept_feature_bindings(
        manifest, suggestions, registry, profile_id, reviewer_name, selected_feature_ids=safe_ids,
    )


__all__ = [
    "LOCAL_REVIEWER_ID",
    "get_local_reviewer_id",
    "build_feature_review_context",
    "fast_local_feature_mapping",
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
    "compute_feature_manifest_status",
    "batch_accept_feature_bindings",
    "batch_approve_safe_feature_suggestions",
]