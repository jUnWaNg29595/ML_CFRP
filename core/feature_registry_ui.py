"""Streamlit page for feature mapping review and local approval workflow."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .dataset_manifest import compute_dataset_manifest_hash
from .feature_mapping_review import (
    apply_feature_review_decision,
    batch_accept_feature_bindings,
    batch_approve_safe_feature_suggestions,
    build_feature_review_context,
    classify_feature_suggestions,
    classify_suggestions_with_diagnostics,
    load_profile_manifest,
    load_profile_suggestions,
    request_feature_mapping_review,
    save_feature_review_record,
    save_profile_manifest,
    save_profile_suggestions,
)
from .feature_registry import (
    compute_registry_hash,
    register_new_feature,
    save_registry_atomic,
    validate_registry,
)
from .portal_ai import PortalAIError
from .portal_ai_config import get_feature_review_ai_client

_REVIEW_SOURCE_ROLES = {"manual_input", "molecular_workflow", "derived_workflow"}

_STAGE_LABELS = {
    "authentication": "认证失败",
    "authorization": "权限不足",
    "transient_network": "网络或代理失败",
    "network_timeout": "网络超时",
    "network_connection": "网络连接失败",
    "proxy_connection": "代理连接失败",
    "dns_failure": "DNS 解析失败",
    "http_error": "HTTP 响应错误",
    "chat_completion_structure": "OpenAI 响应结构错误",
    "message_content_extraction": "message.content 提取失败",
    "http_response_json": "HTTP 响应体非 JSON",
    "http_payload_parsing": "AI 响应非 JSON 对象",
    "json_content_parsing": "模型返回非 JSON",
    "schema_validation": "特征审核 schema 校验失败",
    "unexpected": "未知程序异常",
}


def _redact_secrets(text: str) -> str:
    import re as _re
    cleaned = _re.sub(r"(?i)(?:api[_-]?key|bearer|token|secret|password)\s*[:=]\s*\S+", "[REDACTED]", str(text))
    cleaned = _re.sub(r"sk-[A-Za-z0-9_\-\.]{4,}", "[REDACTED_KEY]", cleaned)
    return cleaned


def format_feature_review_error(exc: BaseException) -> dict[str, str]:
    """Format an AI feature-review failure into sanitized, user-facing fields.

    Returns {"title", "detail", "suggestion"} without leaking API keys,
    authorization headers, full payloads or full model responses.
    """
    raw_text = str(exc) or exc.__class__.__name__
    # Defence-in-depth redaction (portal_ai already sanitizes; belt and braces)
    raw_text = _redact_secrets(raw_text)
    if len(raw_text) > 500:
        raw_text = raw_text[:500] + "..."

    if isinstance(exc, PortalAIError):
        stage_key = str(getattr(exc, "stage", "") or "")
        stage_label = _STAGE_LABELS.get(stage_key, stage_key or "AI 审核异常")
        status_code = getattr(exc, "status_code", None)
        service_id = getattr(exc, "service_id", None)
        suggestion = _redact_secrets(str(getattr(exc, "suggestion", "") or "").strip())
        raw_excerpt = _redact_secrets(str(getattr(exc, "raw_excerpt", "") or "").strip())
        if len(raw_excerpt) > 300:
            raw_excerpt = raw_excerpt[:300] + "..."

        title_parts = [f"[{stage_label}]"]
        if status_code:
            title_parts.append(f"HTTP {status_code}")
        if service_id:
            title_parts.append(f"服务: {service_id}")
        title = " ".join(title_parts)
        sanitized_excerpt = raw_excerpt
        detail = raw_text if not sanitized_excerpt or sanitized_excerpt in raw_text else f"{raw_text} 响应摘要: {sanitized_excerpt}"
        if not suggestion:
            if isinstance(exc, PortalAIError) and exc.__class__.__name__ == "PortalAIAuthError":
                suggestion = "请检查 API Key 是否有效，使用【替换 API Key】入口修复。"
            else:
                suggestion = "请前往左侧边栏【AI 服务管理】检查配置后重试。"
        return {"title": title, "detail": detail, "suggestion": suggestion}

    # Schema / value errors carry concrete field messages
    if isinstance(exc, ValueError):
        detail = raw_text[:500]
        if "source_role" in detail:
            return {
                "title": "[AI 已连接，但返回的特征来源类型不符合契约]",
                "detail": detail,
                "suggestion": "已转为冲突建议，未写入系统。请在人工审核工作台检查来源类型并修正后接受。",
            }
        return {
            "title": "[特征审核数据校验失败]",
            "detail": detail,
            "suggestion": "AI 返回的数据不符合特征审核契约（状态只能为 pending_review/conflict/unknown）。请重试或更换模型。",
        }

    return {
        "title": "[未知程序异常]",
        "detail": raw_text,
        "suggestion": "请检查网络连接与 AI 服务配置；如持续出现，请查看系统日志。",
    }


def frame_column_names(frame: Any) -> list[str]:
    """Return frame columns without evaluating pandas Index truthiness."""
    columns = getattr(frame, "columns", None)
    if columns is None:
        return []
    return [str(column) for column in list(columns)]


def sync_manifest_to_training_state(manifest: Mapping[str, Any]) -> None:
    """Mirror the feature mapping manifest into the training contract state keys.

    模型训练页按 training_dataset_manifest → dataset_manifest →
    feature_dataset_manifest → feature_mapping_manifest 的顺序读取；
    只有 manifest.status == approved（Registry/profile/manifest 全部满足正式
    审批条件）时才写入训练契约键。mapped / pending_approval / draft 只保留在
    feature_mapping_manifest，训练页不会误以为映射已获正式批准。
    """
    import streamlit as st

    if not isinstance(manifest, Mapping):
        return
    st.session_state["feature_mapping_manifest"] = dict(manifest)
    manifest_status = str(manifest.get("status") or "").strip().lower()
    # Only propagate approved manifests to the training contract keys so the
    # training page can never mistake a draft/mapped manifest for an approved one.
    if manifest_status == "approved":
        st.session_state["training_dataset_manifest"] = dict(manifest)
    else:
        # 状态降级（如重新编辑映射）时清除旧训练键，避免训练页读到过期批准
        st.session_state.pop("training_dataset_manifest", None)


def build_manual_feature_suggestion(
    *,
    feature_id: str,
    raw_columns: list[str],
    source_role: str,
    unit: str | None = None,
    rationale: str = "",
    feature_name: str | None = None,
    is_new_proposal: bool = False,
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
    suggestion: dict[str, Any] = {
        "feature_id": normalized_feature_id,
        "raw_columns": normalized_columns,
        "source_role": normalized_role,
        "unit": str(unit or "").strip()[:120] or None,
        "confidence": 1.0,
        "rationale_zh": str(rationale or "").strip()[:500],
        "status": "pending_review",
        "is_new_proposal": bool(is_new_proposal),
    }
    if feature_name:
        suggestion["feature_name"] = str(feature_name).strip()[:200]
    return suggestion


def build_feature_mapping_candidates(registry: Mapping[str, Any], profile_id: str) -> list[dict[str, Any]]:
    """Return profile candidates with explicit approval capability metadata."""
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
            candidate = {
                "feature_id": str(feature_id),
                "name": str(feature_id),
                "source_type": "unknown",
                "status": "unknown",
            }
        else:
            candidate = dict(definition)
        source_type = str(candidate.get("source_type") or "unknown").strip()
        status = str(candidate.get("status") or "unknown").strip().lower()
        allowed = source_type in _REVIEW_SOURCE_ROLES and status in {"draft", "approved"}
        candidate["approval_allowed"] = allowed
        if not allowed:
            reasons = []
            if source_type not in _REVIEW_SOURCE_ROLES:
                reasons.append(f"source_type={source_type}")
            if status not in {"draft", "approved"}:
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
    ai_client: Any | None = None,
    registry_path: str | Path | None = None,
    preferred_service_id: str | None = None,
) -> None:
    """Render the full feature mapping review, proposal and local approval workbench."""
    import streamlit as st

    review_root = Path(__file__).resolve().parents[1] / "prediction_portal" / "feature_reviews"
    portal_root = Path(__file__).resolve().parents[1] / "prediction_portal"
    reg_file = Path(registry_path) if registry_path else portal_root / "feature_registry.json"

    def persist_review_event(record: Mapping[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = review_root / f"{timestamp}-{uuid4().hex}.json"
        try:
            save_feature_review_record(path, record)
        except Exception as exc:
            st.error(f"审核记录保存失败：{exc}")

    # Load registry if missing or invalid
    if registry is None or not isinstance(registry, Mapping):
        try:
            if reg_file.is_file():
                import json
                registry = json.loads(reg_file.read_text(encoding="utf-8"))
            else:
                registry = {}
        except Exception:
            registry = {}
    registry = dict(registry)

    st.title("🧩 特征管理")

    # Profile resolution
    profiles = registry.get("model_profiles", {}) if isinstance(registry, Mapping) else {}
    profile_ids = list(profiles.keys()) if isinstance(profiles, Mapping) else []
    requested_profile_id = str(profile_id or st.session_state.get("model_profile_id") or "")
    if not profile_ids:
        st.error("特征登记库中没有可用的 model profile，暂时无法建立映射。")
        requested_profile_id = ""
    elif requested_profile_id not in profile_ids:
        requested_profile_id = profile_ids[0]

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

    # Restore persisted manifest and suggestions for the active profile
    if profile_id:
        if "feature_mapping_manifest" not in st.session_state or st.session_state.get("manifest_loaded_profile") != profile_id:
            persisted_manifest = load_profile_manifest(profile_id, portal_root)
            st.session_state["feature_mapping_manifest"] = persisted_manifest
            st.session_state["manifest_loaded_profile"] = profile_id
        if "feature_review_suggestions" not in st.session_state or st.session_state.get("suggestions_loaded_profile") != profile_id:
            persisted_suggestions = load_profile_suggestions(profile_id, portal_root)
            st.session_state["feature_review_suggestions"] = persisted_suggestions
            st.session_state["suggestions_loaded_profile"] = profile_id

    active_manifest = st.session_state.get("feature_mapping_manifest") or manifest or {"status": "draft", "feature_bindings": []}
    suggestions = st.session_state.get("feature_review_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []

    # AI Client resolution
    active_client = ai_client
    ai_status_msg = ""
    target_sid = preferred_service_id or st.session_state.get("preferred_feature_review_service_id")
    if active_client is None:
        discovered_client, ai_status_msg = get_feature_review_ai_client(portal_root.parent, preferred_service_id=target_sid)
        if discovered_client is not None:
            active_client = discovered_client
            st.session_state["portal_ai_client"] = discovered_client
    else:
        ai_status_msg = "已注入可用 AI 客户端"

    # Review Context
    review_context = None
    frame_columns = frame_column_names(frame)
    if frame is not None and profile_id:
        review_context = build_feature_review_context(frame, registry, profile_id)
        st.session_state["feature_review_context"] = review_context
    else:
        st.session_state.pop("feature_review_context", None)

    # 1. Top Compact AI Feature Review Workspace
    classification = classify_suggestions_with_diagnostics(suggestions, registry, profile_id)
    safe_suggestions = classification["safe"]
    attention_suggestions = classification["attention"]
    counts = classification["counts"]
    ai_ready = active_client is not None and callable(getattr(active_client, "review_feature_mapping", None))
    # 环境状态（供诊断与空态提示）
    classification["has_frame"] = frame is not None
    classification["has_ai"] = ai_ready
    classification["has_profile"] = bool(profile_id)
    classification["has_suggestions"] = bool(suggestions)

    # profile / Registry 阻断提示（十二：不让用户看到空白按钮区而不知道原因）
    profile_status_lower = profile_status.strip().lower()
    if profile_status_lower == "blocked":
        st.warning(
            "🚫 当前 profile 处于 blocked 状态。可以继续进行映射审核和 manifest 草稿整理，"
            "但不能提交正式训练或发布。请先完成人工审核并批准 profile。"
        )
    registry_approval_status = str((registry.get("approval") or {}).get("status") or "").strip().lower() if isinstance(registry, Mapping) else "unknown"
    if registry_approval_status == "draft":
        st.info(
            "ℹ️ 当前 Registry 仍为 draft。当前操作只会保存待审核映射，"
            "不会使模型获得正式发布资格。"
        )

    with st.container(border=True):
        st.markdown("### 🤖 AI 特征审核工作区")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("目标 Profile", profile_id or "未选择", help=f"Registry 状态: {profile_status}")
        with col_m2:
            st.metric("数据列状态", f"{len(frame_columns)} 列" if frame is not None else "未加载数据")
        with col_m3:
            st.metric("可快速批准", f"{len(safe_suggestions)} 项", help="高置信、无冲突、来源明确")
        with col_m4:
            st.metric("需人工处理", f"{len(attention_suggestions)} 项", help="低置信、来源未知、冲突或新提案")

        if ai_ready:
            st.caption(f"ℹ️ {ai_status_msg} · **安全约束**：AI 仅提出建议，批准必须由本地审核人确认。")
        else:
            st.warning(f"⚠️ {ai_status_msg or 'AI 服务未配置或未启用，请在左侧边栏【AI 服务管理】配置。'}")

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])
        can_run_ai = ai_ready and (frame is not None) and bool(profile_id)
        with btn_col1:
            analyze_clicked = st.button(
                "🔍 分析当前数据列",
                key="feature_review_analyze_btn",
                disabled=not can_run_ai,
                width="stretch",
            )
        with btn_col2:
            reanalyze_clicked = st.button(
                "🔄 重新分析",
                key="feature_review_reanalyze_btn",
                disabled=not can_run_ai,
                width="stretch",
            )
        with btn_col3:
            if not ai_ready:
                st.caption("提示：配置并启用 AI 服务后即可一键智能分析数据列。")
            elif frame is None:
                st.caption("提示：请先在数据页面加载或上传数据集。")
            elif not profile_id:
                st.caption("提示：请先选择目标模型 Profile。")

        if analyze_clicked or reanalyze_clicked:
            if active_client is None:
                st.error("未配置 AI 服务：请在左侧边栏【AI 服务管理】选择并启用特征审核服务。")
            elif not callable(getattr(active_client, "review_feature_mapping", None)):
                st.error("AI 客户端配置错误：当前客户端不支持特征审核接口（缺少 review_feature_mapping 方法）。请重新配置 AI 服务。")
            else:
                try:
                    with st.spinner("AI 正在分析数据列与语义特征映射..."):
                        response = request_feature_mapping_review(active_client, review_context)
                    new_suggs = response.get("suggestions", [])
                    if reanalyze_clicked:
                        updated_suggs = new_suggs
                    else:
                        existing_fids = {s.get("feature_id") for s in suggestions if isinstance(s, Mapping)}
                        updated_suggs = suggestions + [s for s in new_suggs if s.get("feature_id") not in existing_fids]
                    st.session_state["feature_review_suggestions"] = updated_suggs
                    save_profile_suggestions(profile_id, updated_suggs, portal_root)
                    persist_review_event({
                        "event": "ai_response",
                        "profile_id": profile_id,
                        "response": response,
                    })
                    st.success(f"AI 分析完成，已载入 {len(new_suggs)} 条待审核建议。")
                    st.rerun()
                except PortalAIError as exc:
                    formatted = format_feature_review_error(exc)
                    st.error(f"❌ {formatted['title']} {formatted['detail']}")
                    st.info(f"💡 修复建议：{formatted['suggestion']}")
                    st.caption("您可以前往左侧边栏【AI 服务管理】重新配置或测试该服务。")
                except ValueError as exc:
                    formatted = format_feature_review_error(exc)
                    st.error(f"❌ {formatted['title']} {formatted['detail']}")
                    st.info(f"💡 修复建议：{formatted['suggestion']}")
                except Exception as exc:
                    formatted = format_feature_review_error(exc)
                    st.error(f"❌ {formatted['title']} {formatted['detail']}")
                    st.info(f"💡 修复建议：{formatted['suggestion']}")

    mapping_candidates = build_feature_mapping_candidates(registry, profile_id)
    candidate_by_id = {
        str(item.get("feature_id")): item
        for item in mapping_candidates
        if isinstance(item, Mapping) and item.get("feature_id")
    }

    # 2. Main Flow: batch accept safe mapping suggestions (selectable + transparent)
    with st.container(border=True):
        st.markdown("#### ✅ 批量接受安全映射建议")
        st.caption(
            "AI 只提供建议；本地审核人确认后，建议写入当前 dataset manifest。"
            "此操作**不会**自动批准全局 Registry、不会自动发布模型、不会自动生成特征值，"
            "也不会绕过训练和发布门禁。"
        )

        # 审核人输入：稳定 key 由 Streamlit 自动跨 rerun 持久化；
        # 注意：widget key 在本次运行中已绑定，不允许再对同名 session key 赋值
        # （StreamlitAPIException），因此这里不做手动回写。
        reviewer = st.text_input(
            "本地审核人",
            key="feature_review_reviewer",
            placeholder="请输入审核身份（例如：reviewer-alice）",
        )
        if not str(reviewer or "").strip():
            st.warning("请先填写审核人（本地审核人），确认框才能勾选。")

        # 建议状态统计（三）
        stat_c1, stat_c2, stat_c3, stat_c4, stat_c5, stat_c6 = st.columns(6)
        stat_c1.metric("待审核", counts.get("pending_review", 0))
        stat_c2.metric("可批量接受", counts.get("safe", 0))
        stat_c3.metric("需人工处理", counts.get("attention", 0))
        stat_c4.metric("已接受", counts.get("approved", 0))
        stat_c5.metric("冲突", counts.get("conflict", 0))
        stat_c6.metric("无法处理", counts.get("unprocessable", 0))

        # 环境状态行（按钮禁用条件透明化，五）
        env_checks = []
        env_checks.append(("✅" if reviewer.strip() else "❌", "已填写审核人" if reviewer.strip() else "尚未填写审核人"))
        env_checks.append(("✅" if classification.get("has_ai") else "❌", "AI 服务可用" if classification.get("has_ai") else "AI 服务未配置"))
        env_checks.append(("✅" if classification.get("has_frame") else "❌", "数据集已加载" if classification.get("has_frame") else "数据集未加载"))
        env_checks.append(("✅" if classification.get("has_profile") else "❌", "已选择 profile" if classification.get("has_profile") else "未选择 profile"))
        st.caption("　|　".join(f"{icon} {label}" for icon, label in env_checks))

        # 工具按钮（三）：即使 safe 为空也始终显示
        tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(4)
        with tool_c1:
            refresh_classify = st.button(
                "🔄 刷新建议分类", key="feature_review_refresh_classify_btn", width="stretch",
                help="重新执行建议分类与诊断（不调用 AI）。",
            )
        with tool_c2:
            show_attention = st.button(
                "👀 查看需人工处理", key="feature_review_show_attention_btn", width="stretch",
            )
        with tool_c3:
            export_suggestions_btn = st.button(
                "📥 导出待审核建议", key="feature_review_export_suggestions_btn", width="stretch",
            )
        with tool_c4:
            clean_processed = st.button(
                "🧹 清理已处理建议", key="feature_review_clean_processed_btn", width="stretch",
                help="把 status 为 approved/rejected 的建议从列表移除（不影响 manifest）。",
            )
        if refresh_classify:
            st.success(f"已重新分类：可批量接受 {len(safe_suggestions)} 条，需人工处理 {len(attention_suggestions)} 条。")
        if export_suggestions_btn:
            export_rows = []
            for idx, item in enumerate(suggestions if isinstance(suggestions, list) else []):
                if isinstance(item, Mapping):
                    export_rows.append({
                        "feature_id": item.get("feature_id", ""),
                        "raw_columns": ", ".join(map(str, item.get("raw_columns") or [])),
                        "source_role": item.get("source_role", ""),
                        "unit": item.get("unit", ""),
                        "confidence": item.get("confidence", ""),
                        "status": item.get("status", ""),
                    })
            if export_rows:
                import json as _json
                st.download_button(
                    "⬇️ 下载建议清单 JSON",
                    data=_json.dumps(export_rows, ensure_ascii=False, indent=2),
                    file_name=f"suggestions_{profile_id}.json",
                    mime="application/json",
                    key="feature_review_export_download",
                )
            else:
                st.info("当前没有待导出的建议。")
        if clean_processed:
            remaining = [
                s for s in suggestions
                if isinstance(s, Mapping) and str(s.get("status") or "").strip().lower() not in {"approved", "rejected"}
            ]
            st.session_state["feature_review_suggestions"] = remaining
            if profile_id:
                save_profile_suggestions(profile_id, remaining, portal_root)
            st.success(f"已清理已处理建议，剩余 {len(remaining)} 条。")
            st.rerun()

        # 建议列表（可勾选，四）
        safe_feature_ids = [str(item.get("feature_id")) for item in safe_suggestions]
        selection_state_key = "feature_review_batch_selection"
        selection = set(st.session_state.get(selection_state_key, set(safe_feature_ids)))
        if refresh_classify:
            selection = set(safe_feature_ids)
        selection &= set(safe_feature_ids)

        if safe_suggestions:
            st.markdown(f"**可批量接受（{len(safe_suggestions)} 项）**")
            # 快捷选择工具
            quick_c1, quick_c2, quick_c3, quick_c4, quick_c5 = st.columns(5)
            with quick_c1:
                if st.button("全选安全建议", key="feature_review_select_all_btn", width="stretch"):
                    selection = set(safe_feature_ids)
            with quick_c2:
                if st.button("取消全选", key="feature_review_select_none_btn", width="stretch"):
                    selection = set()
            with quick_c3:
                if st.button("只选 manual_input", key="feature_review_select_manual_btn", width="stretch"):
                    selection = {fid for fid in safe_feature_ids if str(next((s.get("source_role") for s in safe_suggestions if str(s.get("feature_id")) == fid), "")).lower() == "manual_input"}
            with quick_c4:
                if st.button("只选 molecular/derived", key="feature_review_select_workflow_btn", width="stretch"):
                    selection = {fid for fid in safe_feature_ids if str(next((s.get("source_role") for s in safe_suggestions if str(s.get("feature_id")) == fid), "")).lower() in {"molecular_workflow", "derived_workflow"}}
            with quick_c5:
                if st.button("仅选择无冲突项", key="feature_review_select_noconflict_btn", width="stretch"):
                    selection = set(safe_feature_ids)
            st.caption(f"已选择 {len(selection)} 项")

            for idx, item in enumerate(safe_suggestions):
                feature_id = str(item.get("feature_id") or "")
                raw_cols = ", ".join(map(str, item.get("raw_columns") or []))
                role = str(item.get("source_role") or "unknown")
                conf = item.get("confidence")
                unit = str(item.get("unit") or "—")
                reg_status = ""
                registry_feature = candidate_by_id.get(feature_id) or {}
                reg_status = str(registry_feature.get("status") or "unknown")
                diag = item.get("_diagnostics") or {}
                is_checked = feature_id in selection
                with st.expander(f"{raw_cols} ➜ {feature_id}", expanded=False):
                    st.caption(
                        f"来源类型：`{role}` | AI 置信度：`{conf}` | 单位：`{unit}`"
                        f" | Registry 状态：`{reg_status}` | 冲突：`{'有' if diag.get('reasons') else '无'}`"
                    )
                    st.caption(f"当前状态：`{item.get('status', 'pending_review')}` | 可批量接受原因：通过全部安全检查")
                    st.write(str(item.get("rationale_zh") or "暂无中文依据"))
                checkbox_checked = st.checkbox(
                    f"选择 {feature_id}",
                    value=is_checked,
                    key=f"feature_review_sel_{feature_id}",
                )
                # checkbox 未被用户交互过（首帧渲染，widget 返回初值之外的假值）时
                # 保留 session 里的选择；只有显式 value=False 才取消选择。
                if checkbox_checked or is_checked:
                    selection.add(feature_id)
                else:
                    selection.discard(feature_id)
        else:
            # 空态诊断（三）：必须列出原因与可操作入口
            st.markdown("**当前没有可批量接受的安全建议。**")
            diag_lines: list[str] = []
            if not classification.get("has_ai"):
                diag_lines.append("没有可用的 AI 服务（尚未配置或未启用）。")
            if not classification.get("has_profile"):
                diag_lines.append("当前未选择 profile。")
            if not classification.get("has_frame"):
                diag_lines.append("当前数据集未加载。")
            if not classification.get("has_suggestions"):
                diag_lines.append("尚未运行 AI 分析（请点击【分析当前数据列】）。")
            attention_diags = [d for d in (classification.get("diagnostics") or []) if not d.get("can_batch_accept")]
            if attention_diags:
                reason_set: list[str] = []
                for d in attention_diags[:10]:
                    for r in (d.get("reasons") or [])[:2]:
                        reason_set.append(f"{d.get('feature_id')}: {r}")
                diag_lines.append("以下建议需要人工处理：" + "；".join(reason_set[:6]))
            for line in diag_lines:
                st.caption(f"• {line}")
            st.caption("请先点击上方【分析当前数据列】或【重新分析】获取 AI 建议，或进入【逐项审核】人工创建映射。")

        st.session_state[selection_state_key] = set(selection)

        # 按钮启用条件透明化（五）
        selected_count = len(selection)
        has_reviewer = bool(reviewer.strip())
        can_proceed = has_reviewer and selected_count > 0
        condition_rows = [
            ("已填写审核人", has_reviewer),
            (f"已选择 {selected_count} 条建议", selected_count > 0),
            ("建议均通过安全检查（当前选择来自可批量接受列表）", selected_count > 0),
            ("manifest 结构有效（可写入）", isinstance(active_manifest, Mapping)),
            ("当前 profile 允许写入", bool(profile_id) and profile_status_lower not in {"blocked"}),
        ]
        st.caption("启用条件：" + "　|　".join(
            f"{'✅' if ok else '❌'} {label}" for label, ok in condition_rows
        ))
        if selected_count == 0 and safe_suggestions:
            st.warning("请至少选择一条建议后再执行批量接受。")

        confirm_batch = st.checkbox(
            "我已核对以上选择与来源类型，确认批量接受",
            key="feature_review_batch_confirm",
            disabled=not can_proceed,
        )
        if not can_proceed and not reviewer.strip():
            st.caption("提示：请先填写本地审核人；确认框需审核人已填写且至少选择一条建议。")

        if st.button(
            f"✅ 接受已选择的 {selected_count} 条建议" if selected_count else "✅ 接受已选择的建议",
            key="feature_review_batch_approve_btn",
            type="primary",
            disabled=not (can_proceed and confirm_batch),
            width="stretch",
        ):
            try:
                selected_ids = [fid for fid in safe_feature_ids if fid in selection]
                updated_manifest = batch_accept_feature_bindings(
                    active_manifest,
                    suggestions,
                    registry,
                    profile_id,
                    reviewer,
                    selected_feature_ids=selected_ids,
                )
                # 成功后才同步 session state / 落盘（原子性：失败不写任何数据）
                sync_manifest_to_training_state(updated_manifest)
                save_profile_manifest(profile_id, updated_manifest, portal_root)
                accepted_fids = {
                    str(b.get("feature_id"))
                    for b in updated_manifest.get("feature_bindings", [])
                    if isinstance(b, Mapping) and str(b.get("review_status") or "") == "approved"
                }
                for s in suggestions:
                    if isinstance(s, Mapping) and str(s.get("feature_id") or "") in accepted_fids:
                        s["status"] = "approved"
                st.session_state["feature_review_suggestions"] = suggestions
                if profile_id:
                    save_profile_suggestions(profile_id, suggestions, portal_root)
                persist_review_event({
                    "event": "batch_accept",
                    "profile_id": profile_id,
                    "reviewer": reviewer,
                    "accepted_count": len(selected_ids),
                    "feature_ids": sorted(selected_ids),
                    "manifest_status": updated_manifest.get("status"),
                    "manifest_hash": updated_manifest.get("manifest_hash"),
                })
                if updated_manifest.get("status") == "approved":
                    st.success(
                        f"已由审核人 {reviewer} 接受 {len(selected_ids)} 条特征映射建议，已写入当前 manifest；"
                        "Registry、profile 与模型发布状态未自动改变。"
                    )
                else:
                    st.success(
                        f"已由审核人 {reviewer} 接受 {len(selected_ids)} 条特征映射建议，已写入当前 manifest；"
                        "Registry 或 profile 尚未正式批准，manifest 保持 mapped（未获得正式训练资格）。"
                    )
                st.session_state.pop(selection_state_key, None)
                st.rerun()
            except Exception as exc:
                st.error(f"批量接受失败（未写入任何数据）：{exc}")

        # 3. Conflict / attention list（十一：每条必须提供操作入口）
        st.markdown("---")
        st.markdown(f"#### ⚠️ 需人工处理（{len(attention_suggestions)} 项）")
        if attention_suggestions:
            for item in attention_suggestions:
                feature_id = str(item.get("feature_id") or "未知特征")
                raw_cols = ", ".join(map(str, item.get("raw_columns") or []))
                status = str(item.get("status") or "unknown")
                reasons = item.get("_review_reasons") or []
                diag = item.get("_diagnostics") or {}
                with st.expander(f"{raw_cols or '未绑定列'} ➜ {feature_id} [{status}]", expanded=False):
                    st.write(str(item.get("rationale_zh") or "暂无中文依据"))
                    st.caption(f"来源类型：`{item.get('source_role') or 'unknown'}` | AI 置信度：`{item.get('confidence', 'N/A')}`")
                    if item.get("source_role_raw"):
                        st.warning(f"AI 原始来源类型：`{item.get('source_role_raw')}`（已降级，需人工审核）")
                    for reason in reasons:
                        st.caption(f"• {reason}")
                    if diag.get("repair_action"):
                        st.info(f"💡 修复建议：{diag['repair_action']}")
                    # 逐项人工编辑入口：修改 source_role / raw_columns / unit
                    edit_col1, edit_col2, edit_col3 = st.columns(3)
                    with edit_col1:
                        edited_role = st.selectbox(
                            "修改来源",
                            ["manual_input", "molecular_workflow", "derived_workflow"],
                            index=(
                                ["manual_input", "molecular_workflow", "derived_workflow"].index(
                                    str(item.get("source_role") or "manual_input")
                                )
                                if str(item.get("source_role") or "") in _REVIEW_SOURCE_ROLES
                                else 0
                            ),
                            key=f"feature_review_att_role_{feature_id}",
                        )
                    with edit_col2:
                        edited_raw = st.text_input(
                            "修改原始列（逗号分隔）",
                            value=", ".join(map(str, item.get("raw_columns") or [])),
                            key=f"feature_review_att_raw_{feature_id}",
                        )
                    with edit_col3:
                        edited_unit = st.text_input(
                            "修改单位",
                            value=str(item.get("unit") or ""),
                            key=f"feature_review_att_unit_{feature_id}",
                        )
                    att_c1, att_c2, att_c3, att_c4 = st.columns(4)
                    with att_c1:
                        accept_edited = st.button(
                            "接受修改后结果", key=f"feature_review_att_accept_{feature_id}",
                            disabled=not bool(reviewer),
                        )
                    with att_c2:
                        mark_conflict = st.button(
                            "标记冲突", key=f"feature_review_att_conflict_{feature_id}",
                            disabled=not bool(reviewer),
                        )
                    with att_c3:
                        reject = st.button(
                            "拒绝", key=f"feature_review_att_reject_{feature_id}",
                            disabled=not bool(reviewer),
                        )
                    with att_c4:
                        retry_ai = st.button(
                            "重试 AI 分析", key=f"feature_review_att_retry_{feature_id}",
                            disabled=not (ai_ready and frame is not None and bool(profile_id)),
                        )
                    if accept_edited:
                        try:
                            edited_payload = {
                                "raw_columns": [c.strip() for c in edited_raw.split(",") if c.strip()],
                                "source_role": edited_role,
                                "unit": edited_unit,
                                "status": "pending_review",
                            }
                            updated_manifest = apply_feature_review_decision(
                                active_manifest, item, "edit_accept", reviewer,
                                registry=registry, edited=edited_payload, profile_id=profile_id,
                            )
                            sync_manifest_to_training_state(updated_manifest)
                            save_profile_manifest(profile_id, updated_manifest, portal_root)
                            for s in suggestions:
                                if isinstance(s, Mapping) and str(s.get("feature_id") or "") == feature_id:
                                    s["status"] = "approved"
                            st.session_state["feature_review_suggestions"] = suggestions
                            if profile_id:
                                save_profile_suggestions(profile_id, suggestions, portal_root)
                            persist_review_event({
                                "event": "local_decision", "action": "edit_accept",
                                "reviewer": reviewer, "feature_id": feature_id,
                                "suggestion": dict(item),
                            })
                            st.success(f"已接受修改后结果：{feature_id}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"接受修改后结果失败：{exc}")
                    if mark_conflict:
                        for s in suggestions:
                            if isinstance(s, Mapping) and str(s.get("feature_id") or "") == feature_id:
                                s["status"] = "conflict"
                        st.session_state["feature_review_suggestions"] = suggestions
                        if profile_id:
                            save_profile_suggestions(profile_id, suggestions, portal_root)
                        persist_review_event({
                            "event": "local_decision", "action": "conflict",
                            "reviewer": reviewer, "feature_id": feature_id,
                        })
                        st.success(f"已标记冲突：{feature_id}")
                        st.rerun()
                    if reject:
                        for s in suggestions:
                            if isinstance(s, Mapping) and str(s.get("feature_id") or "") == feature_id:
                                s["status"] = "rejected"
                        st.session_state["feature_review_suggestions"] = suggestions
                        if profile_id:
                            save_profile_suggestions(profile_id, suggestions, portal_root)
                        persist_review_event({
                            "event": "local_decision", "action": "reject",
                            "reviewer": reviewer, "feature_id": feature_id,
                        })
                        st.success(f"已拒绝：{feature_id}")
                        st.rerun()
                    if retry_ai:
                        st.info("请��用上方【重新分析】按钮重新运行整批 AI 分析。")
                    st.json(dict(item))
        else:
            st.info("当前没有需要人工处理的建议。")

    # 4. Approved bindings summary
    approved_bindings = [
        b for b in (active_manifest.get("feature_bindings") or [])
        if isinstance(b, Mapping) and str(b.get("review_status") or "") == "approved"
    ]
    if approved_bindings:
        st.markdown("#### ✅ 已生效的特征绑定")
        st.dataframe(
            [
                {
                    "原始列": ", ".join(map(str, b.get("raw_columns") or [])),
                    "feature_id": b.get("feature_id", ""),
                    "来源": b.get("source_role", ""),
                    "状态": b.get("review_status", "approved"),
                }
                for b in approved_bindings
            ],
            hide_index=True,
            width="stretch",
        )

    # 4.5 Manifest 导出 / 审核历史 / 提交审批（G 补全区块）
    if profile_id:
        with st.container(border=True):
            st.markdown("#### 📦 Manifest 工具")
            tool_c1, tool_c2, tool_c3 = st.columns(3)
            with tool_c1:
                download_payload = build_manifest_download(profile_id, active_manifest, portal_root)
                if download_payload:
                    st.download_button(
                        "⬇️ 导出 manifest JSON",
                        data=download_payload[1],
                        file_name=download_payload[0],
                        mime="application/json",
                        key="feature_review_manifest_download",
                    )
                else:
                    st.caption("当前 profile 无 manifest 可导出。")
            with tool_c2:
                submit_requested_by = st.text_input(
                    "提交人",
                    key="feature_review_submit_requester",
                    value=st.session_state.get("feature_review_reviewer", ""),
                    placeholder="填写提交审批的身份",
                )
                if st.button(
                    "📤 提交 manifest 审批",
                    key="feature_review_submit_approval_btn",
                    disabled=not bool(submit_requested_by),
                    help="记录提交请求；最终批准仍需本地单人显式执行。",
                ):
                    updated_manifest = submit_manifest_for_approval(profile_id, active_manifest, submit_requested_by, portal_root)
                    if updated_manifest is not None:
                        st.session_state["feature_mapping_manifest"] = updated_manifest
                        persist_review_event({
                            "event": "manifest_submitted_for_approval",
                            "profile_id": profile_id,
                            "requested_by": submit_requested_by,
                        })
                        st.success("manifest 已标记为 submitted_for_approval；批准仍需本地显式操作。")
                        st.rerun()
                    else:
                        st.warning("当前没有 manifest 可提交。")
            with tool_c3:
                approval_obj = active_manifest.get("approval") if isinstance(active_manifest.get("approval"), Mapping) else {}
                if approval_obj.get("submitted"):
                    st.caption(f"审批状态：已提交（{approval_obj.get('submitted_at', '')} by {approval_obj.get('requested_by', '')}）")
                else:
                    st.caption(f"审批状态：{'未提交' if active_manifest.get('feature_bindings') else '暂无绑定'}")

        with st.expander("🕘 审核历史（最近记录）", expanded=False):
            history_events = list_review_history(profile_id, review_root)
            if history_events:
                history_rows = []
                for record in history_events:
                    history_rows.append({
                        "时间": str(record.get("recorded_at") or record.get("timestamp") or ""),
                        "事件": str(record.get("event") or record.get("action") or "unknown"),
                        "审核人": str(record.get("reviewer") or record.get("requested_by") or "—"),
                        "特征": str(record.get("feature_id") or "—"),
                    })
                st.dataframe(history_rows, width="stretch", hide_index=True)
            else:
                st.info("暂无审核记录。")

    # 5. Advanced: manual mapping creation & per-item workbench (hidden by default)
    with st.expander("⚙️ 高级功能：手工新建映射 / 新特征提案 / 逐项审核", expanded=False):
        st.markdown("#### 📝 新建特征映射建议 / 新特征提案")
        mapping_mode = st.radio(
            "创建模式",
            ["映射已有规范特征", "新建规范特征提案 (Proposal)"],
            horizontal=True,
            key="feature_review_mode_radio",
        )
        selected_feature = None
        custom_feature_id = ""
        custom_feature_name = ""
        source_role = "manual_input"
        unit_default = ""

        if mapping_mode == "映射已有规范特征":
            feature_options = [
                f"{item.get('feature_id')} · {item.get('label') or item.get('name')} [{item.get('status')}]"
                for item in mapping_candidates
            ]
            if feature_options:
                selected_label = st.selectbox("选择目标规范特征", feature_options, key="feature_review_manual_feature")
                idx = feature_options.index(selected_label)
                selected_feature = mapping_candidates[idx]
                st.caption(str(selected_feature.get("approval_note") or ""))
                source_role = str(selected_feature.get("source_type") or "manual_input")
                unit_default = str(selected_feature.get("unit") or "") if selected_feature.get("unit") != "unknown" else ""
            else:
                st.warning("当前 profile 没有可用的候选特征。")
        else:
            custom_col1, custom_col2 = st.columns(2)
            with custom_col1:
                custom_feature_id = st.text_input(
                    "新建规范 feature_id",
                    value="cfrp.custom.feature",
                    key="feature_review_custom_feature_id",
                )
                custom_feature_name = st.text_input(
                    "新建规范 name（可选）",
                    value="",
                    key="feature_review_custom_feature_name",
                )
            with custom_col2:
                source_role = st.selectbox("来源类型", sorted(_REVIEW_SOURCE_ROLES), key="feature_review_custom_source_role")
                unit_default = ""

        if frame_columns:
            selected_cols = st.multiselect(
                "选择原始数据列",
                options=frame_columns,
                placeholder="从当前数据集中选择一列或多列",
                key="feature_review_manual_multiselect",
            )
            raw_text = ",".join(selected_cols)
        else:
            raw_text = st.text_input(
                "原始列（多个列用逗号分隔）",
                placeholder="例如：degree_of_cure 或 cure_temperature,cure_time",
                key="feature_review_manual_raw_columns",
            )

        unit = st.text_input("单位", value=unit_default, key="feature_review_manual_unit")
        rationale = st.text_input(
            "中文依据（可选）",
            placeholder="说明为何把该原始列映射到此规范特征",
            key="feature_review_manual_rationale",
        )

        can_save = (selected_feature is None) or bool(selected_feature.get("approval_allowed"))
        if not can_save:
            st.warning("该 registry 候选不可直接批准；请审核其历史语义，或改用新建候选。")

        if st.button("保存为待审核建议", key="feature_review_manual_save", disabled=not can_save or not raw_text.strip()):
            try:
                target_fid = str((selected_feature or {}).get("feature_id") or custom_feature_id)
                suggestion = build_manual_feature_suggestion(
                    feature_id=target_fid,
                    raw_columns=[c.strip() for c in raw_text.split(",") if c.strip()],
                    source_role=source_role,
                    unit=unit,
                    rationale=rationale,
                    feature_name=custom_feature_name,
                    is_new_proposal=(selected_feature is None),
                )
                suggestions_before = [
                    item for item in suggestions
                    if not isinstance(item, Mapping) or item.get("feature_id") != suggestion["feature_id"]
                ]
                updated_suggestions = suggestions_before + [suggestion]
                st.session_state["feature_review_suggestions"] = updated_suggestions
                save_profile_suggestions(profile_id, updated_suggestions, portal_root)
                persist_review_event({
                    "event": "manual_pending_suggestion",
                    "profile_id": profile_id,
                    "suggestion": suggestion,
                })
                st.success("新建候选仅保存为 pending_review，尚未批准，也未写入 registry。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.markdown("---")
        st.markdown("#### 🛡️ 逐项人工审核（高级）")
        status_filter = st.selectbox(
            "查看状态",
            ["pending_review", "conflict", "unknown", "approved", "rejected"],
            key="feature_review_status",
        )
        matching_suggestions = [
            (idx, s) for idx, s in enumerate(suggestions)
            if isinstance(s, Mapping) and str(s.get("status") or "unknown") == status_filter
        ]
        if not matching_suggestions:
            st.info(f"暂无状态为【{status_filter}】的建议。")
        for orig_idx, suggestion in matching_suggestions:
            feature_id = str(suggestion.get("feature_id") or "")
            status = str(suggestion.get("status") or "unknown")
            raw_columns_str = ", ".join(map(str, suggestion.get("raw_columns") or []))
            is_new_prop = bool(suggestion.get("is_new_proposal") or feature_id not in candidate_by_id)
            prop_badge = " [🆕 新特征提案]" if is_new_prop else ""
            with st.expander(f"{raw_columns_str or '未绑定原始列'} ➜ {feature_id}{prop_badge} [{status}]", expanded=False):
                st.write(f"**中文依据**：{suggestion.get('rationale_zh') or '暂无中文依据'}")
                st.caption(f"来源类型：`{suggestion.get('source_role') or 'unknown'}` | 置信度：`{suggestion.get('confidence', 'N/A')}` | 状态：`{status}`")
                st.json(dict(suggestion))
                with st.expander("✏️ 编辑字段", expanded=False):
                    edited_raw_text = st.text_input(
                        "编辑原始列",
                        value=", ".join(map(str, suggestion.get("raw_columns") or [])),
                        key=f"feature_review_raw_{orig_idx}",
                    )
                    edited_source_role = st.selectbox(
                        "编辑来源",
                        ["manual_input", "molecular_workflow", "derived_workflow"],
                        index=(
                            ["manual_input", "molecular_workflow", "derived_workflow"].index(
                                str(suggestion.get("source_role") or "manual_input")
                            )
                            if str(suggestion.get("source_role") or "manual_input") in _REVIEW_SOURCE_ROLES
                            else 0
                        ),
                        key=f"feature_review_role_{orig_idx}",
                    )
                    edited_unit = st.text_input(
                        "编辑单位",
                        value=str(suggestion.get("unit") or ""),
                        key=f"feature_review_unit_{orig_idx}",
                    )
                registry_candidate = candidate_by_id.get(feature_id)
                is_reg_approved_candidate = isinstance(registry_candidate, Mapping) and registry_candidate.get("approval_allowed")
                action_cols = st.columns(5)
                with action_cols[0]:
                    can_accept = bool(reviewer and profile_id and status == "pending_review" and is_reg_approved_candidate)
                    accept = st.button("接受", key=f"feature_review_accept_{orig_idx}", disabled=not can_accept)
                with action_cols[1]:
                    edit_accept = st.button("编辑后接受", key=f"feature_review_edit_accept_{orig_idx}", disabled=not can_accept)
                with action_cols[2]:
                    mark_conflict = st.button("标记冲突", key=f"feature_review_conflict_{orig_idx}", disabled=not bool(reviewer))
                with action_cols[3]:
                    reject = st.button("拒绝", key=f"feature_review_reject_{orig_idx}", disabled=not bool(reviewer))
                with action_cols[4]:
                    can_register = bool(reviewer and is_new_prop)
                    register_btn = st.button("批准并登记新特征", key=f"feature_review_register_{orig_idx}", disabled=not can_register)

                if register_btn:
                    try:
                        feat_def = {
                            "feature_id": feature_id,
                            "name": str(suggestion.get("feature_name") or feature_id),
                            "source_type": str(suggestion.get("source_role") or "manual_input"),
                            "unit": str(suggestion.get("unit") or "unknown"),
                            "status": "draft",
                        }
                        updated_registry = register_new_feature(
                            registry,
                            feat_def,
                            reviewer=reviewer,
                            target_profile_id=profile_id,
                            review_note=f"由审核人 {reviewer} 登记新特征",
                        )
                        save_registry_atomic(reg_file, updated_registry)
                        persist_review_event({
                            "event": "feature_registered",
                            "reviewer": reviewer,
                            "feature_id": feature_id,
                            "feature_definition": feat_def,
                        })
                        st.success(f"已成功将新特征 【{feature_id}】 登记至 Registry。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"登记新特征失败：{exc}")

                action = "accept" if accept else "edit_accept" if edit_accept else "reject" if reject else "conflict" if mark_conflict else None
                edited_payload = {
                    "raw_columns": [c.strip() for c in edited_raw_text.split(",") if c.strip()],
                    "source_role": edited_source_role,
                    "unit": edited_unit,
                    "status": "pending_review",
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
                        if action in {"accept", "edit_accept"}:
                            updated_manifest = apply_feature_review_decision(
                                active_manifest,
                                suggestion,
                                action,
                                reviewer,
                                registry=registry,
                                edited=edited_payload,
                                profile_id=profile_id,
                            )
                            sync_manifest_to_training_state(updated_manifest)
                            save_profile_manifest(profile_id, updated_manifest, portal_root)
                            suggestions[orig_idx]["status"] = "approved"
                            if edited_payload:
                                suggestions[orig_idx]["raw_columns"] = edited_payload["raw_columns"]
                                suggestions[orig_idx]["source_role"] = edited_payload["source_role"]
                                suggestions[orig_idx]["unit"] = edited_payload["unit"]
                        elif action == "reject":
                            suggestions[orig_idx]["status"] = "rejected"
                        elif action == "conflict":
                            suggestions[orig_idx]["status"] = "conflict"
                        st.session_state["feature_review_suggestions"] = suggestions
                        save_profile_suggestions(profile_id, suggestions, portal_root)
                        event["status"] = "applied"
                        st.success("已记录本地审核动作；AI 建议不会自动写入登记库。")
                        st.rerun()
                    except Exception as exc:
                        event["status"] = "failed"
                        event["error"] = str(exc)
                        st.error(str(exc))
                    finally:
                        persist_review_event(event)

    st.caption("AI 仅提供特征映射建议；写入 approved binding 需要本地单人显式批准。")


def build_manifest_download(profile_id: str, manifest: Mapping[str, Any] | None, portal_root: Path | None = None) -> tuple[str, bytes] | None:
    """构造 manifest 下载内容（纯函数）。返回 (文件名, JSON bytes)；无 manifest 返回 None。"""
    if not isinstance(manifest, Mapping) or not manifest:
        return None
    root = portal_root or Path(__file__).resolve().parents[1] / "prediction_portal"
    profile_slug = str(profile_id or "profile").replace("/", "_") or "profile"
    payload = json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    return f"manifest_{profile_slug}.json", payload


def list_review_history(profile_id: str, review_root: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """读取当前 profile 的审核历史事件（纯函数；目录不存在返回空列表）。"""
    root = review_root or Path(__file__).resolve().parents[1] / "prediction_portal" / "feature_reviews"
    if not root.is_dir():
        return []
    events: list[dict[str, Any]] = []
    try:
        files = sorted(root.glob("*.json"), reverse=True)
    except OSError:
        return []
    for path in files[: limit * 4]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, Mapping) and str(record.get("profile_id") or "") == str(profile_id):
            events.append(dict(record))
        if len(events) >= limit:
            break
    return events


def submit_manifest_for_approval(profile_id: str, manifest: Mapping[str, Any] | None, requested_by: str, portal_root: Path | None = None) -> dict[str, Any] | None:
    """把 manifest 标记为 submitted_for_approval（追加 approval.submitted 字段，不覆盖原状态）。

    审批动作本身仍由本地单人显式批准；这里只记录提交请求与审计。
    返回更新后的 manifest；无 manifest 时返回 None。
    """
    if not isinstance(manifest, Mapping) or not manifest:
        return None
    updated = copy.deepcopy(dict(manifest))
    approval = updated.get("approval")
    approval = dict(approval) if isinstance(approval, Mapping) else {}
    approval["submitted"] = True
    approval["submitted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    approval["requested_by"] = str(requested_by or "local")
    updated["approval"] = approval
    root = portal_root or Path(__file__).resolve().parents[1] / "prediction_portal"
    try:
        from .feature_mapping_review import save_profile_manifest
        if str(profile_id):
            save_profile_manifest(str(profile_id), updated, root)
    except Exception:
        pass
    return updated


__all__ = [
    "build_feature_mapping_candidates",
    "build_manual_feature_suggestion",
    "format_feature_review_error",
    "frame_column_names",
    "render_feature_registry_page",
    "sync_manifest_to_training_state",
    "build_manifest_download",
    "list_review_history",
    "submit_manifest_for_approval",
]
