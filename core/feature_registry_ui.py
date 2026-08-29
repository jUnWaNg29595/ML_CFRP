"""Minimal Streamlit page for feature mapping review and local approval."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


_REVIEW_SOURCE_ROLES = {"manual_input", "molecular_workflow", "derived_workflow"}


def frame_column_names(frame: Any) -> list[str]:
    """Return frame columns without evaluating pandas Index truthiness."""
    columns = getattr(frame, "columns", None)
    if columns is None:
        return []
    return [str(column) for column in list(columns)]


def build_manual_feature_suggestion(
    *,
    feature_id: str,
    raw_columns: list[str],
    source_role: str,
    unit: str | None = None,
    rationale: str = "",
) -> dict[str, Any]:
    """Normalize a user-created mapping without approving or writing registry data."""
    normalized_feature_id = str(feature_id or "").strip()
    normalized_columns = [str(column).strip() for column in (raw_columns or []) if str(column).strip()]
    normalized_role = str(source_role or "").strip()
    if not normalized_feature_id:
        raise ValueError("feature_id 不能为空")
    if not normalized_columns:
        raise ValueError("至少需要一个原始列")
    if normalized_role not in _REVIEW_SOURCE_ROLES:
        raise ValueError("来源类型无效")
    return {
        "feature_id": normalized_feature_id,
        "raw_columns": normalized_columns,
        "source_role": normalized_role,
        "unit": str(unit or "").strip()[:120] or None,
        "confidence": 1.0,
        "rationale_zh": str(rationale or "").strip()[:500],
        "status": "pending_review",
    }


def _reviewable_profile_features(registry: Mapping[str, Any], profile_id: str) -> list[dict[str, Any]]:
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    feature_ids = list(profile.get("feature_ids", [])) if isinstance(profile, Mapping) else []
    definitions = {
        item.get("feature_id"): item
        for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping) and item.get("feature_id")
    }
    return [
        dict(definitions[feature_id])
        for feature_id in feature_ids
        if feature_id in definitions
        and definitions[feature_id].get("source_type") in _REVIEW_SOURCE_ROLES
    ]


def build_feature_mapping_candidates(registry: Mapping[str, Any], profile_id: str) -> list[dict[str, Any]]:
    """Return profile candidates with explicit approval capability metadata.

    Legacy metadata and blocked/unknown definitions stay visible for audit, but
    are never presented as directly approvable mappings.
    """
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) else {}
    feature_ids = list(profile.get("feature_ids", [])) if isinstance(profile, Mapping) else []
    definitions = {
        item.get("feature_id"): item
        for item in (registry.get("features", []) if isinstance(registry, Mapping) else [])
        if isinstance(item, Mapping) and item.get("feature_id")
    }
    candidates: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        definition = definitions.get(feature_id)
        if not isinstance(definition, Mapping):
            continue
        candidate = dict(definition)
        source_type = str(candidate.get("source_type") or "unknown")
        status = str(candidate.get("status") or "unknown")
        allowed = source_type in _REVIEW_SOURCE_ROLES and status not in {"blocked", "deprecated"}
        candidate["approval_allowed"] = allowed
        if not allowed:
            reasons = []
            if source_type not in _REVIEW_SOURCE_ROLES:
                reasons.append(f"source_type={source_type}")
            if status in {"blocked", "deprecated"}:
                reasons.append(f"status={status}")
            candidate["approval_note"] = "不可直接批准：" + "，".join(reasons)
        else:
            candidate["approval_note"] = "可提交 pending_review，批准仍需本地审核"
        candidates.append(candidate)
    return candidates


def render_feature_registry_page(
    *,
    frame: Any = None,
    registry: Mapping[str, Any] | None = None,
    profile_id: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    import streamlit as st

    from .feature_mapping_review import (
        apply_feature_review_decision,
        build_feature_review_context,
        request_feature_mapping_review,
        save_feature_review_record,
    )

    review_root = Path(__file__).resolve().parents[1] / "prediction_portal" / "feature_reviews"

    def persist_review_event(record: Mapping[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = review_root / f"{timestamp}-{uuid4().hex}.json"
        try:
            save_feature_review_record(path, record)
        except Exception as exc:
            st.error(f"审核记录保存失败：{exc}")

    registry = registry if isinstance(registry, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {"status": "draft", "feature_bindings": []}
    st.title("🧩 特征管理")
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile_ids = list(profiles.keys()) if isinstance(profiles, Mapping) else []
    requested_profile_id = str(profile_id or st.session_state.get("model_profile_id") or "")
    if not profile_ids:
        st.error("特征登记库中没有可用的 model profile，暂时无法建立映射。")
        requested_profile_id = ""
    elif requested_profile_id not in profile_ids:
        requested_profile_id = profile_ids[0] if len(profile_ids) == 1 else profile_ids[0]
    if profile_ids:
        selected_profile_id = st.selectbox(
            "模型 profile",
            profile_ids,
            index=profile_ids.index(requested_profile_id),
            key="feature_review_profile_id",
        )
        profile_id = str(selected_profile_id)
        st.session_state["model_profile_id"] = profile_id
    else:
        profile_id = ""
    profile = profiles.get(profile_id, {}) if isinstance(profiles, Mapping) and profile_id else {}
    profile_status = str(profile.get("status") or "unknown") if isinstance(profile, Mapping) else "unknown"
    st.caption(f"当前 profile：{profile_id} · registry 状态：{profile_status}")
    if frame is not None and profile_id:
        context = build_feature_review_context(frame, registry, profile_id)
        st.session_state["feature_review_context"] = context
    else:
        st.session_state.pop("feature_review_context", None)

    mapping_candidates = build_feature_mapping_candidates(registry, profile_id)
    st.subheader("审核工作台")
    if frame is None:
        st.info("尚未加载数据；仍可先建立待审核映射，加载数据后再执行列存在性校验。")
    else:
        frame_columns = frame_column_names(frame)
        st.caption(f"当前数据列：{len(frame_columns)} 个；映射时请使用真实列名。")
    with st.container(border=True):
        st.markdown("#### 新建特征映射建议")
        feature_options = ["新建规范特征"] + [
            f"{item.get('feature_id')} · {item.get('label') or item.get('name')} [{item.get('status')}]"
            for item in mapping_candidates
        ]
        selected_feature_label = st.selectbox("规范特征候选", feature_options, key="feature_review_manual_feature")
        selected_feature = None
        if selected_feature_label != "新建规范特征":
            selected_feature = mapping_candidates[feature_options.index(selected_feature_label) - 1]
            st.caption(str(selected_feature.get("approval_note") or ""))
        custom_feature_id = st.text_input(
            "新建规范 feature_id",
            value="" if selected_feature else "cfrp.custom.feature",
            key="feature_review_custom_feature_id",
        )
        custom_feature_name = st.text_input(
            "新建规范 name（可选）",
            value="" if selected_feature else "",
            key="feature_review_custom_feature_name",
        )
        raw_text = st.text_input(
            "原始列（多个列用逗号分隔）",
            placeholder="例如：degree_of_cure 或 cure_temperature,cure_time",
            key="feature_review_manual_raw_columns",
        )
        source_role = str(selected_feature.get("source_type") or "") if selected_feature else "manual_input"
        if selected_feature is None:
            source_role = st.selectbox("来源类型", sorted(_REVIEW_SOURCE_ROLES), key="feature_review_custom_source_role")
        else:
            st.caption(f"来源类型由 registry 定义：{source_role}")
        unit = st.text_input(
            "单位",
            value=(str(selected_feature.get("unit") or "") if selected_feature and selected_feature.get("unit") != "unknown" else ""),
            key="feature_review_manual_unit",
        )
        rationale = st.text_input(
            "中文依据（可选）",
            placeholder="说明为何把该原始列映射到此规范特征",
            key="feature_review_manual_rationale",
        )
        can_save = selected_feature is None or bool(selected_feature.get("approval_allowed"))
        if not can_save:
            st.warning("该 registry 候选不可直接批准；请审核其历史语义，或改用新建候选。")
        if st.button("保存为待审核建议", key="feature_review_manual_save", disabled=not can_save):
            try:
                suggestion = build_manual_feature_suggestion(
                    feature_id=str((selected_feature or {}).get("feature_id") or custom_feature_id),
                    raw_columns=[column for column in raw_text.split(",")],
                    source_role=source_role,
                    unit=unit,
                    rationale=rationale,
                )
                if custom_feature_name.strip():
                    suggestion["feature_name"] = custom_feature_name.strip()[:200]
                suggestions_before = st.session_state.get("feature_review_suggestions", [])
                suggestions_before = [
                    item for item in suggestions_before
                    if not isinstance(item, Mapping) or item.get("feature_id") != suggestion["feature_id"]
                ]
                st.session_state["feature_review_suggestions"] = suggestions_before + [suggestion]
                persist_review_event({
                    "event": "manual_pending_suggestion",
                    "profile_id": profile_id,
                    "suggestion": suggestion,
                })
                st.success("新建候选仅保存为 pending_review，尚未批准，也未写入 registry。")
            except Exception as exc:
                st.error(str(exc))

    suggestions = st.session_state.get("feature_review_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []
    reviewer = st.text_input("本地审核人", key="feature_review_reviewer", placeholder="请输入审核身份")
    status_filter = st.selectbox("查看", ["pending_review", "conflict", "unknown", "approved"], key="feature_review_status")
    rows = []
    candidates = list(suggestions)
    for binding in manifest.get("feature_bindings", []) if isinstance(manifest.get("feature_bindings"), list) else []:
        if not isinstance(binding, Mapping):
            continue
        binding_status = str(binding.get("review_status") or "pending_review")
        if binding_status != status_filter:
            continue
        rows.append({
            "原始列": ", ".join(map(str, binding.get("raw_columns") or [])),
            "feature_id": binding.get("feature_id", ""),
            "来源": binding.get("source_role", ""),
            "状态": binding.get("review_status", "pending_review"),
        })
    for index, suggestion in enumerate(candidates):
        if not isinstance(suggestion, Mapping):
            continue
        status = str(suggestion.get("status") or "unknown")
        if status != status_filter:
            continue
        feature_id = str(suggestion.get("feature_id") or "")
        raw_columns = ", ".join(map(str, suggestion.get("raw_columns") or []))
        with st.expander(f"{raw_columns or '未绑定原始列'} -> {feature_id or '未确定'}", expanded=False):
            st.write(str(suggestion.get("rationale_zh") or "暂无中文依据"))
            st.caption(f"来源：{suggestion.get('source_role') or 'unknown'} | 状态：{status}")
            st.json(dict(suggestion))
            with st.expander("编辑字段", expanded=False):
                edited_raw_text = st.text_input(
                    "编辑原始列",
                    value=", ".join(map(str, suggestion.get("raw_columns") or [])),
                    key=f"feature_review_raw_{index}",
                )
                edited_source_role = st.selectbox(
                    "编辑来源",
                    ["manual_input", "molecular_workflow", "derived_workflow"],
                    index=(
                        ["manual_input", "molecular_workflow", "derived_workflow"].index(
                            str(suggestion.get("source_role") or "manual_input")
                        )
                        if str(suggestion.get("source_role") or "manual_input")
                        in _REVIEW_SOURCE_ROLES
                        else 0
                    ),
                    key=f"feature_review_role_{index}",
                )
                edited_unit = st.text_input(
                    "编辑单位",
                    value=str(suggestion.get("unit") or ""),
                    key=f"feature_review_unit_{index}",
                )
            action_cols = st.columns(3)
            with action_cols[0]:
                accept = st.button("接受", key=f"feature_review_accept_{index}", disabled=not bool(reviewer))
            with action_cols[1]:
                edit_accept = st.button("编辑后接受", key=f"feature_review_edit_accept_{index}", disabled=not bool(reviewer))
            with action_cols[2]:
                reject = st.button("拒绝", key=f"feature_review_reject_{index}", disabled=not bool(reviewer))
            action = "accept" if accept else "edit_accept" if edit_accept else "reject" if reject else None
            edited = {
                "raw_columns": edited_raw_text,
                "source_role": edited_source_role,
                "unit": edited_unit,
            } if edit_accept else None
            if action:
                event = {
                    "event": "local_decision",
                    "action": action,
                    "reviewer": reviewer,
                    "feature_id": suggestion.get("feature_id"),
                    "suggestion": dict(suggestion),
                }
                try:
                    updated = apply_feature_review_decision(manifest, suggestion, action, reviewer, registry=registry, edited=edited, profile_id=profile_id)
                    st.session_state["feature_mapping_manifest"] = updated
                    event["status"] = "applied"
                    st.success("已记录本地审核动作；AI 建议不会自动写入登记库。")
                except Exception as exc:
                    event["status"] = "failed"
                    event["error"] = str(exc)
                    st.error(str(exc))
                finally:
                    persist_review_event(event)
    review_context = st.session_state.get("feature_review_context")
    if review_context and st.button("请求 AI 特征审核", key="feature_review_request_ai"):
        client = st.session_state.get("portal_ai_client")
        try:
            response = request_feature_mapping_review(client, review_context)
            st.session_state["feature_review_suggestions"] = response.get("suggestions", [])
            persist_review_event({
                "event": "ai_response",
                "profile_id": profile_id,
                "response": response,
            })
            st.info("AI 建议已载入待审核列表，尚未批准。")
        except Exception as exc:
            st.error(f"AI 特征审核不可用：{exc}")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("当前数据集还没有已批准的特征绑定。")
    st.caption("AI 仅提供特征映射建议；写入 approved binding 需要本地单人显式批准。")


__all__ = ["build_feature_mapping_candidates", "build_manual_feature_suggestion", "frame_column_names", "render_feature_registry_page"]
