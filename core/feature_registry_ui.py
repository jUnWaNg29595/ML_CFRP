"""Streamlit page for feature mapping review and local approval workflow."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .dataset_manifest import compute_dataset_manifest_hash
from .feature_mapping_review import (
    apply_feature_review_decision,
    batch_approve_safe_feature_suggestions,
    build_feature_review_context,
    classify_feature_suggestions,
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
    这里把批准后的 manifest 同步写入这些键，保证训练页可直接使用。
    """
    import streamlit as st

    if not isinstance(manifest, Mapping):
        return
    st.session_state["feature_mapping_manifest"] = dict(manifest)
    # Only propagate approved manifests to the training contract keys so the
    # training page can never mistake a draft manifest for an approved one.
    if str(manifest.get("status") or "").strip().lower() == "approved":
        st.session_state["training_dataset_manifest"] = dict(manifest)


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
    safe_suggestions, attention_suggestions = classify_feature_suggestions(suggestions, registry, profile_id)
    ai_ready = active_client is not None and callable(getattr(active_client, "review_feature_mapping", None))
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

    # 2. Main Flow: one-click batch approval of safe suggestions
    with st.container(border=True):
        st.markdown("#### ✅ 一键批准安全建议")
        st.caption(
            "AI 分析 → 自动分类 → 本地审核人填写姓名 → 一次确认 → 批量批准安全建议；"
            "低置信、来源未知或冲突项单独处理。"
        )
        reviewer = st.text_input(
            "本地审核人",
            key="feature_review_reviewer",
            placeholder="请输入审核身份（例如：reviewer-alice）",
        )

        if safe_suggestions:
            st.markdown(f"**可快速批准（{len(safe_suggestions)} 项）**")
            for idx, item in enumerate(safe_suggestions):
                feature_id = str(item.get("feature_id") or "")
                raw_cols = ", ".join(map(str, item.get("raw_columns") or []))
                role = str(item.get("source_role") or "unknown")
                conf = item.get("confidence")
                rationale = str(item.get("rationale_zh") or "暂无中文依据")
                with st.expander(f"{raw_cols} ➜ {feature_id}", expanded=False):
                    st.caption(f"来源类型：`{role}` | AI 置信度：`{conf}`")
                    st.write(rationale)

            confirm_batch = st.checkbox(
                "我已阅读以上安全建议，确认批量批准",
                key="feature_review_batch_confirm",
                disabled=not bool(reviewer),
            )
            if st.button(
                "✅ 批准全部安全建议",
                key="feature_review_batch_approve_btn",
                type="primary",
                disabled=not (reviewer and confirm_batch),
                width="stretch",
            ):
                try:
                    updated_manifest = batch_approve_safe_feature_suggestions(
                        active_manifest,
                        suggestions,
                        registry,
                        profile_id,
                        reviewer,
                    )
                    sync_manifest_to_training_state(updated_manifest)
                    save_profile_manifest(profile_id, updated_manifest, portal_root)
                    approved_fids = {
                        str(b.get("feature_id"))
                        for b in updated_manifest.get("feature_bindings", [])
                        if isinstance(b, Mapping)
                    }
                    for s in suggestions:
                        if isinstance(s, Mapping) and str(s.get("feature_id") or "") in approved_fids:
                            s["status"] = "approved"
                    st.session_state["feature_review_suggestions"] = suggestions
                    save_profile_suggestions(profile_id, suggestions, portal_root)
                    persist_review_event({
                        "event": "batch_approve",
                        "profile_id": profile_id,
                        "reviewer": reviewer,
                        "approved_count": len(safe_suggestions),
                        "feature_ids": sorted(approved_fids),
                    })
                    st.success(f"已批量批准 {len(safe_suggestions)} 条安全建议，manifest 已同步到训练页面。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"批量批准失败（未写入任何数据）：{exc}")
        else:
            st.info("当前没有可快速批准的安全建议。")

        # 3. Conflict / attention list
        st.markdown("---")
        st.markdown(f"#### ⚠️ 需人工处理（{len(attention_suggestions)} 项）")
        if attention_suggestions:
            for item in attention_suggestions:
                feature_id = str(item.get("feature_id") or "未知特征")
                raw_cols = ", ".join(map(str, item.get("raw_columns") or []))
                status = str(item.get("status") or "unknown")
                reasons = item.get("_review_reasons") or []
                with st.expander(f"{raw_cols or '未绑定列'} ➜ {feature_id} [{status}]", expanded=False):
                    st.write(str(item.get("rationale_zh") or "暂无中文依据"))
                    st.caption(f"来源类型：`{item.get('source_role') or 'unknown'}` | AI 置信度：`{item.get('confidence', 'N/A')}`")
                    if item.get("source_role_raw"):
                        st.warning(f"AI 原始来源类型：`{item.get('source_role_raw')}`（已降级，需人工审核）")
                    for reason in reasons:
                        st.caption(f"• {reason}")
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


__all__ = [
    "build_feature_mapping_candidates",
    "build_manual_feature_suggestion",
    "format_feature_review_error",
    "frame_column_names",
    "render_feature_registry_page",
    "sync_manifest_to_training_state",
]
