"""Minimal Streamlit page for feature mapping review and local approval."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


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
    profile_id = str(profile_id or st.session_state.get("model_profile_id") or "")
    if frame is not None and profile_id:
        context = build_feature_review_context(frame, registry, profile_id)
        st.session_state.setdefault("feature_review_context", context)
    suggestions = st.session_state.get("feature_review_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []
    reviewer = st.text_input("本地审核人", key="feature_review_reviewer", placeholder="请输入审核身份")
    status_filter = st.selectbox("查看", ["pending_review", "conflict", "approved"], key="feature_review_status")
    rows = []
    candidates = list(suggestions)
    for binding in manifest.get("feature_bindings", []) if isinstance(manifest.get("feature_bindings"), list) else []:
        if not isinstance(binding, Mapping):
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
        status = str(suggestion.get("status") or "pending_review")
        if status != status_filter:
            continue
        feature_id = str(suggestion.get("feature_id") or "")
        raw_columns = ", ".join(map(str, suggestion.get("raw_columns") or []))
        with st.expander(f"{raw_columns or '未绑定原始列'} -> {feature_id or '未确定'}", expanded=False):
            st.write(str(suggestion.get("rationale_zh") or "暂无中文依据"))
            st.caption(f"来源：{suggestion.get('source_role') or 'unknown'} | 状态：{status}")
            st.json(dict(suggestion))
            action_cols = st.columns(3)
            with action_cols[0]:
                accept = st.button("接受", key=f"feature_review_accept_{index}", disabled=not bool(reviewer))
            with action_cols[1]:
                edit_accept = st.button("编辑后接受", key=f"feature_review_edit_accept_{index}", disabled=not bool(reviewer))
            with action_cols[2]:
                reject = st.button("拒绝", key=f"feature_review_reject_{index}", disabled=not bool(reviewer))
            action = "accept" if accept else "edit_accept" if edit_accept else "reject" if reject else None
            if action:
                event = {
                    "event": "local_decision",
                    "action": action,
                    "reviewer": reviewer,
                    "feature_id": suggestion.get("feature_id"),
                    "suggestion": dict(suggestion),
                }
                try:
                    updated = apply_feature_review_decision(manifest, suggestion, action, reviewer)
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


__all__ = ["render_feature_registry_page"]
