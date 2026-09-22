# -*- coding: utf-8 -*-
from __future__ import annotations

import os
# 必须在科学计算库加载前设置：conda 的 sklearn 经 MKL 间接加载 Intel Fortran 运行库
# libifcoremd.dll，其默认注册的 Ctrl+C 处理器会打印 forrtl: error (200) 并卡死退出，
# 导致 Streamlit 按 Ctrl+C 关不掉。设为 1 可禁用该处理器（Intel 官方开关）。
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import copy
import base64
import json
import hashlib
import re
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.model_io import load_model_artifact_bytes
from core.portal_prediction import load_published_portal_model, run_confirmed_prediction, validate_publication_artifact
from core.prediction_portal import activate_publication
from core.portal_ai import PortalAIClient, PortalAIError
from core.portal_ai_config import AIServiceConfig, load_ai_config, redacted_ai_config
from core.portal_ui import inject_scientific_theme, render_material_card, render_stage_timeline, render_status_badge, svg_icon
from core.portal_tasks import PortalTaskManager


APP_NAME = "邹华维课题组材料预测平台"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = PROJECT_ROOT / "prediction_portal"
MODEL_ROOT = PLATFORM_ROOT / "managed_models"
CONFIG_PATH = PLATFORM_ROOT / "prediction_config.json"
ASSET_ROOT = PLATFORM_ROOT / "assets"

DATA_UPLOAD_TYPES = ["csv", "xlsx", "xls"]
SMILES_UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "heif", "heic", "pdf"]
PARAMETER_KINDS = ["number", "integer", "text", "select", "smiles"]


def build_ai_confirmation_state(response: Any, confirmed_fields: set[str] | None = None) -> Dict[str, Any]:
    """Normalize AI suggestions into a state that requires explicit user decisions."""
    if isinstance(response, dict):
        payload = response
    else:
        payload = {
            "recognized_fields": getattr(response, "recognized_fields", {}),
            "suggestions": getattr(response, "suggestions", []),
            "warnings": getattr(response, "warnings", []),
            "assumptions": getattr(response, "assumptions", []),
        }
    fields: Dict[str, Dict[str, Any]] = {}
    recognized = payload.get("recognized_fields") if isinstance(payload.get("recognized_fields"), dict) else {}
    for field, value in recognized.items():
        if isinstance(field, str) and field.strip():
            fields[field.strip()] = {"value": value, "state": "recognized", "confidence": None}
    suggestions = payload.get("suggestions") or []
    for item in suggestions:
        if hasattr(item, "field"):
            field, value, state, confidence = item.field, item.value, item.state, item.confidence
        elif isinstance(item, dict):
            field, value, state, confidence = item.get("field"), item.get("value"), item.get("state", "suggested"), item.get("confidence")
        else:
            continue
        if isinstance(field, str) and field.strip():
            fields[field.strip()] = {"value": value, "state": state or "suggested", "confidence": confidence}
    confirmed = {str(field) for field in (confirmed_fields or set()) if str(field).strip()}
    return {"fields": fields, "confirmed_fields": confirmed, "rejected_fields": set(), "warnings": list(payload.get("warnings") or []), "assumptions": list(payload.get("assumptions") or [])}


def confirm_ai_field(state: Dict[str, Any], field: str, value: Any) -> Dict[str, Any]:
    """Return a copied state after the user confirms or edits one AI field."""
    updated = copy.deepcopy(state)
    name = str(field).strip()
    updated.setdefault("fields", {})[name] = {"value": value, "state": "confirmed", "confidence": 1.0}
    updated.setdefault("confirmed_fields", set()).add(name)
    updated.setdefault("rejected_fields", set()).discard(name)
    return updated


def reject_ai_field(state: Dict[str, Any], field: str) -> Dict[str, Any]:
    updated = copy.deepcopy(state)
    name = str(field).strip()
    updated.setdefault("fields", {}).setdefault(name, {"value": None})["value"] = None
    updated["fields"][name]["state"] = "rejected"
    updated.setdefault("confirmed_fields", set()).discard(name)
    updated.setdefault("rejected_fields", set()).add(name)
    return updated


def can_submit_ai_prediction(state: Dict[str, Any]) -> bool:
    fields = set((state or {}).get("fields") or {})
    resolved = set((state or {}).get("confirmed_fields") or set()) | set((state or {}).get("rejected_fields") or set())
    return bool(fields) and fields <= resolved


def fallback_input_mode(error: Any) -> str:
    return "manual" if isinstance(error, (str, Exception)) else "manual"


#: 多轮对话保留的最大历史轮数（避免上下文无限增长）
AI_CONVERSATION_MAX_TURNS = 12


def build_ai_field_descriptions(field_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构造传给 AI 的字段描述。

    **硬约束**：``allow_ai_generation`` 一律为 False —— AI 只能提取和整理
    用户提供的信息，不得生成 EEW/AHEW/PHR/分子特征/工艺参数等计算量。
    这些量必须由 Python 侧按公式推导（见 core/portal_formulation_inputs.py）。
    """
    return [
        {
            "name": item.get("name"),
            "label": item.get("label"),
            "kind": item.get("kind"),
            "required": bool(item.get("required", False)),
            "allow_ai_generation": False,
        }
        for item in (field_defs or [])
        if isinstance(item, dict) and item.get("name")
    ]


def build_ai_conversation_context(
    state: Dict[str, Any], field_labels: Dict[str, str] | None = None
) -> Dict[str, Any]:
    """构造本轮对话的上下文（供 AI 理解「修改」语义）。

    关键：必须把**已确认字段**带进去，否则 AI 不知道「温度」指哪个字段，
    也不知道其他字段已经定下来。被用户拒绝的字段不传（避免 AI 再提）。
    """
    state = state if isinstance(state, dict) else {}
    rejected = {str(f) for f in (state.get("rejected_fields") or set())}
    confirmed_fields: Dict[str, Any] = {}
    for name, detail in (state.get("fields") or {}).items():
        if name in rejected or not isinstance(detail, dict):
            continue
        value = detail.get("value")
        if value is None or value == "":
            continue
        confirmed_fields[str(name)] = value
    return {
        "confirmed_fields": confirmed_fields,
        "rejected_fields": sorted(rejected),
        "field_labels": dict(field_labels or {}),
    }


def build_ai_conversation_messages(
    turns: List[Dict[str, Any]], current_text: str
) -> List[Dict[str, str]]:
    """拼接对话消息：有界历史 + 当前输入。"""
    history = [
        {"role": str(t.get("role") or "user"), "text": str(t.get("text") or "")}
        for t in (turns or [])
        if isinstance(t, dict)
    ][-AI_CONVERSATION_MAX_TURNS:]
    return history + [{"role": "user", "text": str(current_text or "")}]


def append_conversation_turn(
    turns: List[Dict[str, Any]], *, role: str, text: str
) -> List[Dict[str, Any]]:
    """追加一轮对话（返回新列表，不原地修改）。"""
    updated = list(turns or [])
    updated.append({"role": str(role), "text": str(text)})
    return updated


def merge_ai_conversation_state(
    previous: Dict[str, Any], response: Any
) -> Dict[str, Any]:
    """把新一轮 AI 响应合并进已有状态（增量修正，不丢历史字段）。

    规则：
    1. 新一轮未提及的字段**保留原值**（不得清空）；
    2. 已拒绝字段在新一轮不得被静默恢复；
    3. 新提取的字段状态仍为 suggested/recognized，**必须用户确认**；
    4. 警告累积。
    """
    previous = previous if isinstance(previous, dict) else {}
    # 兼容两种 AI 响应形态：
    #   (a) 标准形态 {recognized_fields / suggestions / warnings}（build_ai_confirmation_state 消费）
    #   (b) 已是 fields 形态 {fields: {name: {value, state}}}（多轮对话/测试常用）
    if isinstance(response, dict) and isinstance(response.get("fields"), dict):
        fresh = {
            "fields": {
                str(name): (
                    {
                        "value": detail.get("value"),
                        # 新提取字段一律不得是 confirmed —— 必须用户确认（门禁不放松）
                        "state": str(detail.get("state") or "suggested"),
                        "confidence": detail.get("confidence"),
                    }
                    if isinstance(detail, dict)
                    else {"value": detail, "state": "suggested", "confidence": None}
                )
                for name, detail in response["fields"].items()
            },
            "warnings": list(response.get("warnings") or []),
        }
    else:
        fresh = build_ai_confirmation_state(response)

    merged_fields: Dict[str, Dict[str, Any]] = {}
    for name, detail in (previous.get("fields") or {}).items():
        merged_fields[str(name)] = copy.deepcopy(detail) if isinstance(detail, dict) else {"value": detail}

    rejected = {str(f) for f in (previous.get("rejected_fields") or set())}
    for name, detail in (fresh.get("fields") or {}).items():
        if name in rejected:
            # 已拒绝：保持拒绝，不恢复
            continue
        merged_fields[str(name)] = dict(detail)

    merged_confirmed = {str(f) for f in (previous.get("confirmed_fields") or set())}
    merged_warnings = list(previous.get("warnings") or [])
    for warning in fresh.get("warnings") or []:
        if warning not in merged_warnings:
            merged_warnings.append(warning)

    return {
        "fields": merged_fields,
        "confirmed_fields": merged_confirmed,
        "rejected_fields": rejected,
        "warnings": merged_warnings,
        "assumptions": list(previous.get("assumptions") or []),
    }


def render_task_snapshot(snapshot: Dict[str, Any]) -> str:
    task_id = html_escape(str(snapshot.get("task_id") or ""))
    stage = html_escape(str(snapshot.get("stage_label") or snapshot.get("stage") or ""))
    progress = max(0, min(100, int(snapshot.get("progress") or 0)))
    status = html_escape(str(snapshot.get("status") or "unknown"))
    return (f'<section class="portal-status-panel" data-task-id="{task_id}">'
            f'<div><strong>任务 {task_id}</strong> · {status}</div>'
            f'<div class="portal-help">阶段：{stage} · {progress}%</div>'
            f'<div class="portal-progress"><div style="width:{progress}%"></div></div></section>')


def render_result(result: Dict[str, Any]) -> str:
    prediction = html_escape(str(result.get("prediction", "")))
    unit = html_escape(str(result.get("unit") or ""))
    explanation = result.get("explanation") or {}
    if isinstance(explanation, dict) and explanation.get("status") == "unavailable":
        note = "AI 解释暂不可用"
    elif isinstance(explanation, dict) and explanation.get("summary"):
        note = html_escape(str(explanation.get("summary")))
    else:
        note = "Python 模型结果为权威值"
    return f'<div class="portal-result"><strong>{prediction} {unit}</strong><span>{note}</span></div>'


@st.cache_resource(show_spinner=False)
def get_portal_task_manager(root: str) -> PortalTaskManager:
    return PortalTaskManager(Path(root))


def submit_prediction_task(config: Dict[str, Any], material_key: str, target_key: str, input_df: pd.DataFrame, *, explain: bool = False) -> str:
    manager = get_portal_task_manager(str(PROJECT_ROOT))
    inputs: Any = input_df.iloc[0].to_dict() if len(input_df) == 1 else input_df.to_dict(orient="records")
    return manager.create_task({
        "request": {
            "material_type": material_key, "target": target_key, "inputs": inputs,
            "confirmed_by_user": True,
        },
        "config": config, "explain": bool(explain),
    })


def _ai_service_dataclass(service: Dict[str, Any]) -> AIServiceConfig:
    allowed = {field for field in AIServiceConfig.__dataclass_fields__}
    return AIServiceConfig(**{key: value for key, value in service.items() if key in allowed})


def _model_contract(model: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact = model.get("_artifact") if isinstance(model, dict) else None
    extra = artifact.get("extra") if isinstance(artifact, dict) else {}
    contract = model.get("contract") if isinstance(model, dict) else None
    if not isinstance(contract, dict) and isinstance(extra, dict):
        contract = extra.get("prediction_contract")
    snapshot = model.get("registry_snapshot") if isinstance(model, dict) else None
    if not isinstance(snapshot, dict) and isinstance(extra, dict):
        snapshot = extra.get("registry_snapshot")
    return (dict(contract) if isinstance(contract, dict) else {}, dict(snapshot) if isinstance(snapshot, dict) else {})


def _is_publishable_ui_model(model: Dict[str, Any]) -> bool:
    """判断模型是否可在预测页使用（第二道门禁）。

    硬条件（必须全满足）：已启用、已发布、门禁报告 ok 且 valid。

    注册表审核要求（v2 契约才强制）：契约 schema_version==2 时，仍要求
    registry_snapshot 存在、model_profile.status==approved、全部 feature approved。

    legacy(schema-1) 导入模型：从训练平台导出的 artifact 不带 registry_snapshot，
    契约 schema_version 为 None/1。这类模型已经过 publish_imported_entry 的
    validate_publication_artifact 门禁，不应因缺少 v2 注册表快照而被永久拒绝
    （否则所有训练平台下载的模型都无法启用）。
    """
    if not isinstance(model, dict) or model.get("enabled") is not True or str(model.get("publication_status") or "").strip().lower() != "published":
        return False
    gate = model.get("gate_report")
    if not isinstance(gate, dict) or gate.get("ok") is not True or str(gate.get("status") or "").strip().lower() != "valid":
        return False
    contract, snapshot = _model_contract(model)
    if contract.get("schema_version") == 2:
        # v2 契约：保持注册表审核的严格语义
        profile = snapshot.get("model_profile") if isinstance(snapshot, dict) else None
        if not snapshot or not isinstance(profile, dict) or profile.get("status") != "approved":
            return False
        if any(not isinstance(item, dict) or item.get("status") != "approved" for item in snapshot.get("features") or []):
            return False
    artifact = model.get("_artifact")
    if isinstance(artifact, dict):
        report = validate_publication_artifact(
            artifact,
            contract,
            registry_snapshot=snapshot,
            dataset_manifest=artifact.get("extra", {}).get("dataset_manifest") if isinstance(artifact.get("extra"), dict) else None,
        )
        if report.get("ok") is not True or str(report.get("status") or "").lower() != "valid":
            return False
    return True


def toggle_model_enabled(
    config: Dict[str, Any], material_key: str, target_key: str, model: Dict[str, Any]
) -> Dict[str, Any]:
    """Toggle a model release, gating every transition to enabled."""
    if not isinstance(model, dict):
        raise ValueError("模型记录无效。")
    if model.get("enabled") is True:
        model["enabled"] = False
        model["updated_at"] = now_iso()
        return config
    candidate = copy.deepcopy(model)
    candidate["enabled"] = True
    if not candidate.get("version"):
        # 导入型模型（如从训练平台下载后重新上传）本身没有发布版本号：
        # 走本地发布门禁，验证通过后补打版本号并正式上线。
        from core.prediction_portal import publish_imported_entry
        return publish_imported_entry(
            config, material_key=material_key, target_key=target_key, entry=candidate
        )
    return activate_publication(
        config, material_key=material_key, target_key=target_key, entry=candidate
    )


def _ai_parse_input(
    material_key: str, target_key: str, user_text: str,
    contract: Dict[str, Any] | None, registry_snapshot: Dict[str, Any] | None, service: Dict[str, Any],
    *, context: Dict[str, Any] | None = None, force_refresh: bool = False,
) -> Dict[str, Any]:
    """调用 AI 输入助手解析用户文本，返回统一的确认状态。

    供首页全自动流程、手动解析按钮与多轮对话复用。

    缓存（core/portal_ai_cache.py）：相同 service/model/输入直接复用，
    不重复消耗额度；``force_refresh=True`` 绕过读取但仍写入新结果。

    ``context`` 为多轮对话上下文（已确认字段），使 AI 理解「温度改成 200 度」
    这类增量修改的语义。
    """
    contract = contract if isinstance(contract, dict) else {}
    registry_snapshot = registry_snapshot if isinstance(registry_snapshot, dict) else {}
    field_defs = build_manual_input_fields(contract, registry_snapshot) + build_workflow_source_fields(contract, registry_snapshot)
    field_descriptions = build_ai_field_descriptions(field_defs)

    service_id = str(service.get("service_id") or "")
    model_name = str(service.get("model") or "")
    prompt_kind = "input_parse"
    cache = _ai_cache()
    cache_text = str(user_text)
    if context and context.get("confirmed_fields"):
        # 上下文参与缓存键：同一句话在不同已确认字段下语义不同
        cache_text = cache_text + "\x1e" + repr(sorted(context["confirmed_fields"].items()))
    if cache is not None and not force_refresh:
        hit = cache.get(
            service_id=service_id, model=model_name, prompt_kind=prompt_kind, text=cache_text
        )
        if hit is not None and isinstance(hit.get("value"), dict):
            return build_ai_confirmation_state(hit["value"])

    payload = {
        'material_type': material_key,
        'target': target_key,
        'field_descriptions': field_descriptions,
        'user_text': user_text,
    }
    if context:
        payload['context'] = context
    response = PortalAIClient(_ai_service_dataclass(service)).parse_input(payload)

    if cache is not None:
        try:
            raw = {
                "recognized_fields": getattr(response, "recognized_fields", {}) or {},
                "suggestions": [
                    {
                        "field": getattr(item, "field", None),
                        "value": getattr(item, "value", None),
                        "state": getattr(item, "state", "suggested"),
                        "confidence": getattr(item, "confidence", None),
                    }
                    for item in (getattr(response, "suggestions", None) or [])
                ],
                "warnings": list(getattr(response, "warnings", None) or []),
                "assumptions": list(getattr(response, "assumptions", None) or []),
            }
            cache.put(
                service_id=service_id,
                model=model_name,
                prompt_kind=prompt_kind,
                text=cache_text,
                value=raw,
            )
        except Exception:
            pass
    return build_ai_confirmation_state(response)


def _ai_cache():
    """获取 AI 缓存实例；任何异常降级为 None（缓存不可用不得阻断主流程）。"""
    try:
        from core.portal_ai_cache import PortalAICache

        return PortalAICache(root=PROJECT_ROOT)
    except Exception:
        return None


def _sync_ai_state_to_manual(material_key: str, target_key: str, state: Dict[str, Any]) -> None:
    """把 AI 已确认的字段同步写入手动输入各分区的 widget session_state。"""
    for field_name, detail in state.get('fields', {}).items():
        if field_name not in set(state.get('rejected_fields') or set()) and detail.get('value') not in (None, ''):
            val_str = str(detail.get('value'))
            for prefix in ("manual", f"manual_{material_key}_{target_key}_molecular", f"manual_{material_key}_{target_key}_required_manual", f"manual_{material_key}_{target_key}_optional_manual"):
                st.session_state[f"{prefix}_{field_name}"] = val_str
                st.session_state[f"{prefix}_{field_name}_number"] = default_number(val_str, 0.0)
                st.session_state[f"{prefix}_{field_name}_integer"] = default_integer(val_str, 0)
                st.session_state[f"{prefix}_{field_name}_text"] = val_str


def render_ai_assistant_tab(
    config: Dict[str, Any], material_key: str, target_key: str, target_cfg: Dict[str, Any],
    contract: Dict[str, Any] | None = None, registry_snapshot: Dict[str, Any] | None = None,
) -> None:
    st.markdown('### AI 辅助输入')
    text_key = f'ai_text_{material_key}_{target_key}'
    state_key = f'ai_state_{material_key}_{target_key}'
    # 首页「全自动」入口：带着用户粘贴的描述文本跳转过来，这里自动完成 解析→确认→回填→切换到手动输入。
    home_auto_text = st.session_state.pop('ai_home_pending_text', None)
    if home_auto_text:
        st.session_state[text_key] = home_auto_text
    st.caption('AI 只提取和整理你提供的信息；不能自动生成 EEW、AHEW、PHR、分子特征或工艺参数。首页「全自动解析」会把提取结果直接填入手动输入表单，仍需人工勾选确认后才会计算。')
    try:
        ai_config = load_ai_config(PROJECT_ROOT)
    except Exception as exc:
        st.error(f'AI 配置读取失败：{exc}')
        return
    services = [item for item in ai_config.get('services', []) if item.get('enabled') and item.get('purpose') in {'both', 'input_parsing'}]
    if not services:
        if home_auto_text:
            st.warning('尚未启用 AI 输入助手服务：你粘贴的描述已保留在下方，可在侧边栏配置 AI 后重新解析，或直接使用手动/批量输入。')
        else:
            st.info('暂无已启用的输入助手服务；请在主平台侧边栏配置，或继续使用手动/批量输入。')
        return
    service = st.selectbox('AI 服务', services, format_func=lambda item: item.get('label') or item.get('service_id'), key=f'ai_service_{material_key}_{target_key}')
    user_text = st.text_area('描述材料、配方和工艺信息', key=text_key, height=150, placeholder='例如：树脂 SMILES 为 CCO；固化温度 80 °C。')
    if home_auto_text:
        with st.spinner('AI 正在全自动解析你粘贴的描述…'):
            try:
                state = _ai_parse_input(material_key, target_key, home_auto_text, contract, registry_snapshot, service)
            except PortalAIError as exc:
                st.warning(f'AI 自动解析失败：{exc}；文本已保留，可修改后点击「解析输入」重试，或改用手动输入。')
                return
            # 全自动：把提取到的全部字段标记为已确认，再同步回填手动表单。
            for field_name in list(state.get('fields', {}).keys()):
                detail = state['fields'][field_name]
                if detail.get('value') not in (None, ''):
                    state = confirm_ai_field(state, field_name, detail.get('value'))
            st.session_state[state_key] = state
            _sync_ai_state_to_manual(material_key, target_key, state)
        st.session_state['ai_home_flash'] = '🤖 AI 已自动解析并填入手动输入表单；请核对参数、勾选确认框后即可预测。'
        st.session_state['predict_tab_intent'] = 'manual'
        st.rerun()
    if st.button('解析输入', key=f'ai_parse_{material_key}_{target_key}', type='secondary'):
        if not user_text.strip():
            st.warning('请先输入待解析的文本。')
        else:
            try:
                st.session_state[state_key] = _ai_parse_input(material_key, target_key, user_text, contract, registry_snapshot, service)
                st.success('解析完成，请逐项确认或拒绝。')
            except PortalAIError as exc:
                st.warning(f'AI 不可用：{exc}；已保留手动输入模式。')

    # ------------------------------------------------------------------
    # 多轮对话（st.chat_message / st.chat_input）
    #
    # 目的：用户不会一次说全配方。第二轮把**已确认字段**作为上下文传给 AI，
    # 使 AI 理解「温度改成 200 度」这类增量修改的语义。
    # 约束：AI 仍只能提取/整理，不得生成计算量；结果仍需用户确认。
    # ------------------------------------------------------------------
    st.markdown('##### 💬 多轮对话修正')
    st.caption('先解析一次，然后可直接说「温度改成 200 度」这类修改；系统会把已确认字段作为上下文。')
    turns_key = f'ai_turns_{material_key}_{target_key}'
    turns = st.session_state.setdefault(turns_key, [])
    for turn in turns[-AI_CONVERSATION_MAX_TURNS:]:
        with st.chat_message('user' if turn.get('role') == 'user' else 'assistant'):
            st.markdown(str(turn.get('text') or ''))

    chat_text = st.chat_input(
        '继续描述或修正（例如：温度改成 200 度）',
        key=f'ai_chat_{material_key}_{target_key}',
    )
    if chat_text:
        state = st.session_state.get(state_key) or {'fields': {}, 'confirmed_fields': set(), 'rejected_fields': set(), 'warnings': []}
        context = build_ai_conversation_context(state)
        st.session_state[turns_key] = append_conversation_turn(turns, role='user', text=chat_text)
        try:
            with st.spinner('AI 正在理解你的修正…'):
                fresh = _ai_parse_input(
                    material_key, target_key, chat_text, contract, registry_snapshot, service,
                    context=context,
                )
            merged = merge_ai_conversation_state(state, fresh)
            st.session_state[state_key] = merged
            changed = [
                name for name in (fresh.get('fields') or {})
                if name in (merged.get('fields') or {})
            ]
            reply = (
                f"已更新 {len(changed)} 个字段：{'、'.join(changed[:6])}"
                if changed else '本轮未识别到可更新的字段，请换种说法再试。'
            )
            st.session_state[turns_key] = append_conversation_turn(
                st.session_state[turns_key], role='assistant', text=reply
            )
        except PortalAIError as exc:
            st.session_state[turns_key] = append_conversation_turn(
                st.session_state[turns_key], role='assistant', text=f'AI 不可用：{exc}'
            )
        st.rerun()

    col_cache1, col_cache2 = st.columns([1, 1])
    with col_cache1:
        if st.button('🔄 强制重新解析（忽略缓存）', key=f'ai_force_{material_key}_{target_key}'):
            if not user_text.strip():
                st.warning('请先输入待解析的文本。')
            else:
                try:
                    st.session_state[state_key] = _ai_parse_input(
                        material_key, target_key, user_text, contract, registry_snapshot, service,
                        force_refresh=True,
                    )
                    st.success('已忽略缓存重新解析。')
                except PortalAIError as exc:
                    st.warning(f'AI 不可用：{exc}')
    with col_cache2:
        cache = _ai_cache()
        if cache is not None:
            st.caption(f'缓存条目：{cache.size()}')

    state = st.session_state.get(state_key)
    if not state:
        st.info('解析结果会显示在这里。')
        return
    for warning in state.get('warnings', []):
        st.warning(str(warning))
    for field, detail in state.get('fields', {}).items():
        current_value = '' if detail.get('value') is None else str(detail.get('value'))
        edited = st.text_input(f'{field}（{detail.get("state", "suggested")}）', value=current_value, key=f'ai_field_{material_key}_{target_key}_{field}')
        col_confirm, col_reject = st.columns(2)
        with col_confirm:
            if st.button('确认该字段', key=f'ai_confirm_{material_key}_{target_key}_{field}', width='stretch'):
                st.session_state[state_key] = confirm_ai_field(state, field, edited)
                st.rerun()
        with col_reject:
            if st.button('拒绝该字段', key=f'ai_reject_{material_key}_{target_key}_{field}', width='stretch'):
                st.session_state[state_key] = reject_ai_field(state, field)
                st.rerun()
    if can_submit_ai_prediction(state):
        c_sub1, c_sub2 = st.columns([1, 1.2])
        with c_sub1:
            if st.button('🚀 创建 AI 辅助预测任务', key=f'ai_submit_{material_key}_{target_key}', type='primary', width="stretch"):
                confirmed_inputs = {
                    field: detail.get('value')
                    for field, detail in state.get('fields', {}).items()
                    if field not in set(state.get('rejected_fields') or set()) and detail.get('value') not in (None, '')
                }
                if confirmed_inputs:
                    task_key = f'portal_task_ai_{material_key}_{target_key}'
                    st.session_state[task_key] = submit_prediction_task(config, material_key, target_key, pd.DataFrame([confirmed_inputs]), explain=True)
                    st.rerun()
                else:
                    st.warning('没有被确认的有效输入，无法创建任务。')
        with c_sub2:
            if st.button('📋 一键采纳并同步填入单组手动输入表单', key=f'ai_sync_to_manual_{material_key}_{target_key}', width="stretch", help="将 AI 确认后的参数一键同步回填到手动输入表单中，方便继续微调与结构图复核"):
                _sync_ai_state_to_manual(material_key, target_key, state)
                st.success("🎉 已成功将 AI 提取的参数同步填入手动输入表单！请在上方切换到【手动输入】标签页查看与预测。")
                st.rerun()
    render_portal_task_panel(st.session_state.get(f'portal_task_ai_{material_key}_{target_key}'), session_key=f'ai_{material_key}_{target_key}')


def render_portal_task_panel(task_id: str | None, *, session_key: str) -> None:
    if not task_id:
        return
    manager = get_portal_task_manager(str(PROJECT_ROOT))
    try:
        snapshot = manager.get_task_snapshot(task_id)
    except KeyError:
        st.error("找不到预测任务快照，请重新提交。")
        return
    st.markdown(render_task_snapshot(snapshot), unsafe_allow_html=True)
    st.markdown(render_stage_timeline(snapshot.get("stage", ""), snapshot.get("progress", 0)), unsafe_allow_html=True)
    status = snapshot.get("status")
    control_cols = st.columns(3)
    with control_cols[0]:
        if st.button("刷新任务", key=f"refresh_{session_key}_{task_id}", width="stretch"):
            st.rerun()
    with control_cols[1]:
        if status in {"queued", "validating", "featuring", "predicting", "explaining", "pending", "running"} and st.button("取消任务", key=f"cancel_{session_key}_{task_id}", width="stretch"):
            manager.cancel_task(task_id)
            st.rerun()
    with control_cols[2]:
        if status in {"failed", "cancelled"} and st.button("重新提交输入", key=f"retry_{session_key}_{task_id}", width="stretch"):
            st.info("请重新填写并提交预测输入；系统不会自动重放原始请求。")
    if status == "completed" and isinstance(snapshot.get("result"), dict):
        st.markdown(render_result(snapshot["result"]), unsafe_allow_html=True)
        result = snapshot["result"]
        st.success(f'预测完成：{result.get("prediction")} {result.get("unit") or ""}')
        if snapshot.get("explanation_error"):
            st.info("AI 解释暂不可用，Python 预测结果不受影响。")
    elif status == "failed":
        st.error(f'任务失败：{snapshot.get("error") or "请检查发布模型和输入契约。"}')
    elif status == "cancelled":
        st.warning("任务已取消，可以修改输入后重新提交。")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: str, fallback: str = "item") -> str:
    value = re.sub(r"[^\w\-]+", "_", (text or "").strip(), flags=re.UNICODE).strip("_")
    return value or fallback


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, pd.Series)):
        return [json_ready(v) for v in value.tolist()]
    if pd.isna(value) if not isinstance(value, str) else False:
        return None
    return value


DEFAULT_EPOXY_PARAMETERS = [
    {
        "name": "resin_smiles",
        "label": "树脂 SMILES",
        "kind": "smiles",
        "required": True,
        "default": "",
        "placeholder": "可手输，也可导入结构图/PDF识别",
        "help": "树脂主体的 SMILES 字符串",
        "options": [],
    },
    {
        "name": "curing_agent_smiles",
        "label": "固化剂 SMILES",
        "kind": "smiles",
        "required": False,
        "default": "",
        "placeholder": "可选",
        "help": "双组分体系可在这里录入固化剂",
        "options": [],
    },
    {
        "name": "phr",
        "label": "配比 / phr",
        "kind": "number",
        "required": False,
        "default": 0.0,
        "placeholder": "",
        "help": "可在管理端替换成你自己的字段",
        "options": [],
    },
    {
        "name": "curing_temperature_c",
        "label": "固化温度 (°C)",
        "kind": "number",
        "required": False,
        "default": 25.0,
        "placeholder": "",
        "help": "示例参数，可在管理端继续改",
        "options": [],
    },
    {
        "name": "curing_time_h",
        "label": "固化时间 (h)",
        "kind": "number",
        "required": False,
        "default": 1.0,
        "placeholder": "",
        "help": "示例参数，可在管理端继续改",
        "options": [],
    },
]


def make_target_config(label: str, description: str = "") -> Dict[str, Any]:
    return {
        "label": label,
        "enabled": True,
        "description": description,
        "parameters": copy.deepcopy(DEFAULT_EPOXY_PARAMETERS),
        "models": [],
    }


def make_numeric_target_config(label: str, description: str = "") -> Dict[str, Any]:
    return {
        "label": label,
        "enabled": True,
        "description": description,
        "parameters": [],
        "models": [],
    }


def default_config() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "updated_at": now_iso(),
        "materials": {
            "epoxy_resin": {
                "label": "环氧树脂",
                "enabled": True,
                "description": "面向树脂体系的性能预测入口，可由后台配置具体性能项、参数和模型。",
                "coming_soon_message": "",
                "targets": {
                    "tg": make_target_config("Tg", "玻璃化转变温度"),
                    "tensile_modulus": make_target_config("拉伸模量", "树脂体系拉伸模量预测"),
                    "tensile_strength": make_target_config("拉伸强度", "树脂体系拉伸强度预测"),
                    "compressive_modulus": make_target_config("压缩模量", "树脂体系压缩模量预测"),
                    "yield_strength": make_target_config("屈服强度", "树脂体系屈服强度预测"),
                },
            },
            "ud_cfrp": {
                "label": "单向碳纤维复合材料",
                "enabled": True,
                "description": "面向单向碳纤维复合材料的性能预测入口，可上传已训练模型后进行单样本或批量预测。",
                "coming_soon_message": "",
                "targets": {
                    "ud_property": make_numeric_target_config("综合性能", "用于挂载已训练的单向复材性能预测模型"),
                    "tensile_modulus": make_numeric_target_config("拉伸模量", "单向复材拉伸模量预测"),
                    "tensile_strength": make_numeric_target_config("拉伸强度", "单向复材拉伸强度预测"),
                    "compressive_modulus": make_numeric_target_config("压缩模量", "单向复材压缩模量预测"),
                    "compressive_strength": make_numeric_target_config("压缩强度", "单向复材压缩强度预测"),
                    "shear_strength": make_numeric_target_config("剪切强度", "单向复材剪切强度预测"),
                },
            },
        },
    }


def deep_merge_defaults(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in (current or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_storage() -> None:
    PLATFORM_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        payload = default_config()
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(json_ready(payload), fh, ensure_ascii=False, indent=2)


def load_config() -> Dict[str, Any]:
    ensure_storage()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            current = json.load(fh)
    except Exception:
        current = default_config()
    return deep_merge_defaults(default_config(), current)


def save_config(config: Dict[str, Any]) -> None:
    ensure_storage()
    payload = copy.deepcopy(config)
    payload["updated_at"] = now_iso()
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(json_ready(payload), fh, ensure_ascii=False, indent=2)


def material_items(config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return list((config.get("materials") or {}).items())


def target_items(material_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return list((material_cfg.get("targets") or {}).items())


def model_items(target_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(target_cfg.get("models") or [])


@st.cache_data(show_spinner=False)
def image_data_uri(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def portal_asset(name: str) -> str:
    return image_data_uri(str(ASSET_ROOT / name))


def material_image_name(material_key: str) -> str:
    return {
        "epoxy_resin": "card-epoxy.png",
        "ud_cfrp": "card-ud-cfrp.png",
    }.get(material_key, "portal-hero.png")


def resolve_model_path(model_entry: Dict[str, Any]) -> Path:
    path = Path(model_entry.get("artifact_path") or "")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_text_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def parse_options(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").replace("；", ";").replace("，", ",")
    tokens = re.split(r"[\n,;]+", text)
    return [token.strip() for token in tokens if token.strip()]


def parameter_from_feature(feature: str) -> Dict[str, Any]:
    return {
        "name": str(feature),
        "label": str(feature),
        "kind": "number",
        "required": True,
        "default": None,
        "placeholder": "",
        "help": "仅作为迁移提示；预测值必须来自已批准契约或用户显式输入",
        "options": [],
    }


def _contract_feature_definitions(contract: Dict[str, Any], registry_snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    definitions: Dict[str, Dict[str, Any]] = {}
    # The registry snapshot is authoritative. Contract definitions are only a
    # fallback for embedded v2 artifacts that carry their approved snapshot.
    if not isinstance(registry_snapshot, dict) or not registry_snapshot:
        return definitions
    if registry_snapshot.get("schema_version") != 1 or not isinstance(registry_snapshot.get("features"), list):
        return definitions
    sources = [registry_snapshot]
    for source in sources:
        items = source.get("features") if isinstance(source, dict) else None
        if not isinstance(items, list):
            items = source.get("feature_definitions") if isinstance(source, dict) else None
        for item in items or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    definitions[name] = dict(item)
    return definitions


def build_manual_input_fields(contract: Dict[str, Any], registry_snapshot: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Build editable fields only from the contract's approved manual partition."""
    contract = contract if isinstance(contract, dict) else {}
    registry_snapshot = registry_snapshot if isinstance(registry_snapshot, dict) else {}
    definitions = _contract_feature_definitions(contract, registry_snapshot)
    names = [str(name) for name in contract.get("manual_input_feature_cols") or [] if str(name).strip()]
    fields: List[Dict[str, Any]] = []
    for name in names:
        definition = definitions.get(name, {})
        if (
            definition.get("source_type") != "manual_input"
            or definition.get("status") != "approved"
            or definition.get("default_policy") != "explicit_only"
        ):
            continue
        data_type = str(definition.get("data_type") or "float").lower()
        kind = "integer" if data_type in {"integer", "int"} else "number" if data_type in {"float", "number"} else "select" if definition.get("enum_values") or definition.get("options") else "text"
        fields.append({
            "name": name,
            "label": definition.get("label") or definition.get("display_name") or name,
            "unit": definition.get("unit"),
            "data_type": definition.get("data_type") or "float",
            "kind": kind,
            "required": bool(definition.get("required_for_prediction", True)),
            "nullable": bool(definition.get("nullable", False)),
            "default": None,
            "valid_range": definition.get("valid_range"),
            "options": list(definition.get("enum_values") or definition.get("options") or []),
            "source_type": "manual_input",
        })
    return fields


def build_workflow_source_fields(contract: Dict[str, Any], registry_snapshot: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Build read-only workflow source slots declared by the prediction contract."""
    contract = contract if isinstance(contract, dict) else {}
    registry_snapshot = registry_snapshot if isinstance(registry_snapshot, dict) else {}
    profile = registry_snapshot.get("model_profile")
    if contract.get("schema_version") != 2 or not registry_snapshot or not isinstance(profile, dict) or profile.get("status") != "approved":
        return []
    if any(not isinstance(item, dict) or item.get("status") != "approved" for item in registry_snapshot.get("features") or []):
        return []
    # v2's canonical field name is intentionally the only accepted source for
    # the editable portal UI. Legacy aliases remain read-only compatibility
    # data in the backend and must not silently create controls here.
    declared = contract.get("workflow_source_fields") or []
    fields: List[Dict[str, Any]] = []
    for item in declared:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column") or item.get("name") or "").strip()
        if not name:
            continue
        roles = item.get("roles") or item.get("source_roles") or []
        if isinstance(roles, str):
            roles = [roles]
        fields.append({
            "name": name,
            "label": item.get("label") or name,
            "roles": [str(role) for role in roles],
            "required": bool(item.get("required", not bool(item.get("nullable", False)))),
            "nullable": bool(item.get("nullable", False)),
            "default": None,
            "kind": "smiles" if "smiles" in name.lower() else "text",
            "source_type": "workflow",
        })
    return fields


def sync_parameters_from_features(target_cfg: Dict[str, Any], feature_cols: List[str]) -> bool:
    if not feature_cols or target_cfg.get("parameters"):
        return False
    target_cfg["parameters"] = [parameter_from_feature(feature) for feature in feature_cols]
    return True


#: 输入分区的固定顺序（UI 依赖此顺序渲染）
AUTOFILL_SECTION_GROUPS = ("recipe", "cure_schedule", "test_conditions")

#: 配方类字段关键词（配方计量与组分描述）
_RECIPE_FIELD_TOKENS = (
    "resin_", "curing_agent_", "hardener_", "formulation_", "initiator_",
    "accelerator_", "catalyst_", "reactive_diluent", "reactive_toughener",
    "small_additive", "other_component", "_total_phr", "_phr_basis",
    "curing_type_standard", "curing_mechanism", "_component_count",
    "_equivalent_group_total", "_epoxy_group_total", "_active_hydrogen_total",
    "_total_eew", "_total_ahew", "_r_value", "_equivalent_ratio",
)

#: 固化制度（工艺）类字段关键词
_CURE_FIELD_TOKENS = (
    "process_", "cure_", "post_cure", "_cure_temperature", "_cure_time",
    "atmosphere", "pressure", "ramp", "dwell",
)

#: 测试条件类字段关键词
_TEST_FIELD_TOKENS = (
    "_test_method", "_test_standard", "_test_atmosphere", "_analysis_method",
    "_specimen_geometry", "_frequency_hz", "_heating_rate", "_loading_rate",
    "_strain_rate", "_test_temperature",
)


def classify_manual_field_group(feature: str) -> str:
    """把人工输入字段归入 3 个可编辑分区之一（纯函数）。

    优先级：测试条件 → 固化制度 → 配方 → 默认测试条件。
    默认落在测试条件（保守：不误入配方区，避免把测试标准当成配方输入）。
    """
    name = str(feature or "").strip().lower()
    if not name:
        return "test_conditions"
    if any(token in name for token in _TEST_FIELD_TOKENS):
        return "test_conditions"
    if any(token in name for token in _CURE_FIELD_TOKENS):
        return "cure_schedule"
    if any(token in name for token in _RECIPE_FIELD_TOKENS):
        return "recipe"
    return "test_conditions"


def build_autofilled_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造「已自动填充 N 项」明细（纯函数）。

    空值（None / ""）不计入，避免虚报项数；0 是合法取值，应计入。
    """
    rows: List[Dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if value is None or value == "":
            continue
        rows.append({
            "feature": str(entry.get("feature") or ""),
            "value": value,
            "origin": str(entry.get("origin") or "default"),
            "detail": str(entry.get("detail") or ""),
        })
    count = len(rows)
    title = f"已自动填充 {count} 项" if count else "未自动填充任何字段"
    return {"count": count, "rows": rows, "title": title}


def build_input_partition_plan(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按用户心智模型生成输入分区计划（纯函数，可独立测试）。

    返回列表，每项 {group, title, description, features, kind}，固定顺序：

    ① recipe          配方（必填）——可编辑
    ② cure_schedule   固化制度——可编辑（默认预填）
    ③ test_conditions 高级测试条件——可编辑（默认预填）
    ④ derived         自动推导——只读（kind=display）
    ⑤ computed        系统计算特征——只读（kind=display）

    ④⑤ 绝不出现在可编辑分区：它们必须由 workflow 真实计算，允许手填
    会让用户覆盖系统值，造成静默的预测错误。
    分区缺失时给出默认空组，保证 UI 不崩溃。
    """
    contract = contract if isinstance(contract, dict) else {}
    manual_cols = [str(c) for c in contract.get("manual_input_feature_cols") or [] if str(c).strip()]
    molecular_cols = [str(c) for c in contract.get("molecular_workflow_feature_cols") or [] if str(c).strip()]
    derived_cols = [str(c) for c in contract.get("derived_feature_cols") or [] if str(c).strip()]

    # legacy(schema-1) 契约没有分区字段，但模型仍需这些显式特征。
    # 此时按 core.portal_prediction._explicit_model_feature_names 的口径补齐：
    #   manual = contract.feature_cols − workflow.final_feature_names
    # 否则 UI 会一个输入框都不渲染，用户无法预测（真实 artifact 就是这种情况）。
    if not manual_cols and not molecular_cols and not derived_cols:
        workflow_features = {
            str(name)
            for name in contract.get("workflow_final_feature_names") or []
            if str(name).strip()
        }
        all_features = [
            str(c) for c in contract.get("feature_cols") or [] if str(c).strip()
        ]
        if workflow_features:
            manual_cols = [c for c in all_features if c not in workflow_features]
        elif all_features:
            # 无 workflow 信息时不能判断哪些是系统计算的，全部交给用户输入
            # （宁可多问，不可静默使用臆造值）
            manual_cols = list(all_features)

    buckets: Dict[str, List[str]] = {
        "recipe": [],
        "cure_schedule": [],
        "test_conditions": [],
    }
    for name in manual_cols:
        buckets[classify_manual_field_group(name)].append(name)

    plan: List[Dict[str, Any]] = [
        {
            "group": "recipe",
            "title": "① 配方（必填）",
            "description": "树脂/固化剂结构、配比（phr）与组分构成。只填配方即可预测，其余条件系统已按全表统计预填。",
            "features": buckets["recipe"],
            "kind": "input",
        },
        {
            "group": "cure_schedule",
            "title": "② 固化制度",
            "description": "固化温度、时间、阶段、后固化与气氛。已按典型值预填，请按实际实验修改。",
            "features": buckets["cure_schedule"],
            "kind": "input",
        },
        {
            "group": "test_conditions",
            "title": "③ 高级测试条件",
            "description": "测试方法/标准/频率/升温速率。已按全表主流口径预填，不修改也可预测。",
            "features": buckets["test_conditions"],
            "kind": "input",
        },
        {
            "group": "derived",
            "title": "④ 自动推导（只读）",
            "description": "由配方按物理/化学公式推导，无需填写；下方展示推导依据。",
            "features": derived_cols,
            "kind": "display",
        },
        {
            "group": "computed",
            "title": "⑤ 系统计算（只读）",
            "description": "由登记的分子特征 workflow 从结构自动计算，无需填写。",
            "features": molecular_cols,
            "kind": "display",
        },
    ]
    if contract.get("screening_fixed_input_cols"):
        plan.append({
            "group": "fixed_inputs",
            "title": "固定工艺条件（只读）",
            "description": "当前预测任务使用的固定工艺/测试条件（契约 screening_fixed_input_cols）。",
            "features": [str(c) for c in contract.get("screening_fixed_input_cols") or [] if str(c).strip()],
            "kind": "display",
        })
    return plan


def model_contract_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    """从模型条目 + contract 提取可核验信息（纯函数，可独立测试）。"""
    model = model if isinstance(model, dict) else {}
    contract, snapshot = _model_contract(model)
    extra = (model.get("_artifact") or {}).get("extra") if isinstance(model.get("_artifact"), dict) else {}
    entry_extra = model.get("extra") if isinstance(model.get("extra"), dict) else {}
    status_raw = str(model.get("publication_status") or "").strip().lower()
    status_labels = {
        "published": "已发布",
        "needs_validation": "待验证",
        "draft": "草稿",
        "disabled": "已停用",
        "legacy": "legacy",
    }
    return {
        "model_version": model.get("model_version") or model.get("updated_at") or "",
        "model_profile_id": contract.get("model_profile_id") or snapshot.get("profile_id") or "",
        "contract_schema_version": contract.get("schema_version"),
        "feature_registry_version": contract.get("feature_registry_version") or "",
        "feature_registry_hash": str(contract.get("feature_registry_hash") or "")[:8],
        "workflow_hash": str(contract.get("workflow_hash") or "")[:8],
        "artifact_hash": str(model.get("artifact_hash") or (extra or entry_extra or {}).get("artifact_hash") or "")[:8],
        "publication_status": status_labels.get(status_raw, status_raw or "未知"),
        "publication_status_raw": status_raw,
        "contract_features": len(contract.get("feature_cols") or []),
        "manual_features": len(contract.get("manual_input_feature_cols") or []),
        "workflow_source_fields": len(contract.get("workflow_source_fields") or []),
        "gate_status": str(((model.get("gate_report") or {}).get("status")) or "") if isinstance(model.get("gate_report"), dict) else "",
    }


MODEL_MANAGEMENT_AUDIT_FILE = PROJECT_ROOT / "prediction_portal" / "model_management_audit.jsonl"


def append_model_management_audit(action: str, model_id: str = "", model_version: str = "", detail: str = "", reviewer: str = "local") -> None:
    """把管理操作追加写入本地审计日志（JSONL，不覆盖旧记录）。"""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "action": str(action),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "detail": str(detail)[:500],
        "reviewer": str(reviewer),
    }
    try:
        MODEL_MANAGEMENT_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_MANAGEMENT_AUDIT_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def render_sync_diagnostics_panel(model: Dict[str, Any]) -> None:
    """平台同步诊断 UI：调用 diagnose_platform_sync 展示中文状态。"""
    try:
        from core.prediction_portal import diagnose_platform_sync
    except ImportError:
        st.info("诊断模块未加载，无法展示平台同步诊断。")
        return
    artifact = model.get("_artifact") if isinstance(model, dict) else None
    contract, snapshot = _model_contract(model)
    manifest = None
    if isinstance(artifact, dict) and isinstance(artifact.get("extra"), dict):
        manifest = artifact["extra"].get("dataset_manifest")
    try:
        report = diagnose_platform_sync(
            registry=snapshot if isinstance(snapshot, dict) and snapshot.get("registry_version") else None,
            profile=(snapshot or {}).get("model_profile") if isinstance(snapshot, dict) else None,
            manifest=manifest,
            artifact=artifact if isinstance(artifact, dict) else None,
            contract=contract if contract else None,
        )
    except Exception as exc:
        st.warning(f"同步诊断执行失败：{exc}")
        return
    status_labels = {
        "synced": "✅ 已同步", "partial": "🟡 部分同步", "needs_review": "⏳ 待人工审核",
        "needs_retrain": "🔁 待重新训练", "needs_republication": "📢 待重新发布",
        "prediction_blocked": "⛔ 不允许预测", "screening_blocked": "⛔ 不允许正式筛选",
    }
    overall = " | ".join(status_labels.get(s, s) for s in report.get("overall_status", []))
    st.markdown(f"**平台同步状态：** {overall}")
    rows = []
    for check in report.get("checks", []):
        icon = {"ok": "✅", "warn": "⚠️", "error": "❌", "missing": "➖"}.get(check.get("status"), "•")
        rows.append({"状态": icon, "检查项": check.get("check", ""), "说明": check.get("detail", "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    counts = report.get("feature_counts") or {}
    if counts:
        st.caption(
            f"特征数量：模型 {counts.get('model_features', 0)} | 契约 {counts.get('contract_features', 0)}"
            f" | 门户输入 {counts.get('portal_input_features', 0)} | 虚拟筛选 {counts.get('screening_features', 0)}"
        )
    flags = []
    flags.append(("可发布", report.get("can_publish")))
    flags.append(("可预测", report.get("can_predict")))
    flags.append(("可正式虚拟筛选", report.get("can_screen_formally")))
    st.caption("　|　".join(f"{'✅' if ok else '⛔'} {label}" for label, ok in flags))


def normalize_parameter_rows(editor_df: pd.DataFrame) -> List[Dict[str, Any]]:
    parameters: List[Dict[str, Any]] = []
    if editor_df is None or editor_df.empty:
        return parameters

    for _, row in editor_df.fillna("").iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        parameters.append(
            {
                "name": name,
                "label": str(row.get("label", "") or name).strip(),
                "kind": str(row.get("kind", "text") or "text").strip(),
                "required": bool(row.get("required", False)),
                "default": row.get("default", ""),
                "placeholder": str(row.get("placeholder", "")).strip(),
                "help": str(row.get("help", "")).strip(),
                "options": parse_options(row.get("options", "")),
            }
        )
    return parameters


def parameter_editor_df(parameters: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in parameters or []:
        rows.append(
            {
                "name": item.get("name", ""),
                "label": item.get("label", ""),
                "kind": item.get("kind", "text"),
                "required": bool(item.get("required", False)),
                "default": item.get("default", ""),
                "placeholder": item.get("placeholder", ""),
                "help": item.get("help", ""),
                "options": "\n".join(item.get("options") or []),
            }
        )
    if not rows:
        rows.append(
            {
                "name": "",
                "label": "",
                "kind": "text",
                "required": False,
                "default": "",
                "placeholder": "",
                "help": "",
                "options": "",
            }
        )
    return pd.DataFrame(rows)


def load_data_file(uploaded_file) -> pd.DataFrame:
    file_name = (uploaded_file.name or "").lower()
    uploaded_file.seek(0)
    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="gbk")
    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding="utf-8-sig")


def preview_artifact(file_bytes: bytes) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact = load_model_artifact_bytes(file_bytes)
    extra = artifact.get("extra") or {}
    feature_process = extra.get("molecular_feature_config") or extra.get("feature_process")
    preview = {
        "model_name": artifact.get("model_name") or "ImportedModel",
        "target_col": artifact.get("target_col") or "",
        "feature_count": len(artifact.get("feature_cols") or []),
        "metrics": json_ready(artifact.get("metrics") or {}),
        "has_feature_process": bool(feature_process),
        "feature_process_keys": sorted(list(feature_process.keys())) if isinstance(feature_process, dict) else [],
    }
    return artifact, preview


def upsert_model_entry(
    config: Dict[str, Any],
    material_key: str,
    target_key: str,
    file_name: str,
    file_bytes: bytes,
    label: str,
    notes: str,
    feature_override: List[str],
    replace_model_id: str = "",
) -> Dict[str, Any]:
    artifact, _ = preview_artifact(file_bytes)
    target_cfg = config["materials"][material_key]["targets"][target_key]
    target_cfg.setdefault("models", [])

    model_dir = MODEL_ROOT / material_key / target_key
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file_name).suffix or ".joblib"
    file_stub = slugify(label or Path(file_name).stem, fallback="model")
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_stub}{suffix}"
    saved_path = model_dir / saved_name
    saved_path.write_bytes(file_bytes)

    existing_entry = None
    if replace_model_id:
        for item in target_cfg["models"]:
            if item.get("id") == replace_model_id:
                existing_entry = item
                break

    feature_cols = feature_override or artifact.get("feature_cols") or []
    model_id = replace_model_id or f"{target_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    entry = {
        "id": model_id,
        "label": label or artifact.get("model_name") or Path(file_name).stem,
        # Imported artifacts enter validation first; publication requires an
        # explicit successful gate report and activation action.
        "enabled": False,
        "publication_status": "needs_validation",
        "gate_report": {},
        "notes": notes,
        "model_name": artifact.get("model_name") or "ImportedModel",
        "target_col": artifact.get("target_col") or target_key,
        "feature_cols": list(feature_cols),
        "feature_source": "override" if feature_override else ("artifact" if artifact.get("feature_cols") else "input_order"),
        "artifact_path": saved_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_hash": hashlib.sha256(file_bytes).hexdigest(),
        "source_filename": file_name,
        "created_at": existing_entry.get("created_at") if existing_entry else now_iso(),
        "updated_at": now_iso(),
        "metrics": json_ready(artifact.get("metrics") or {}),
        "has_feature_process": bool(
            ((artifact.get("extra") or {}).get("molecular_feature_config"))
            or ((artifact.get("extra") or {}).get("feature_process"))
        ),
    }

    if existing_entry is None:
        target_cfg["models"].append(entry)
    else:
        existing_entry.clear()
        existing_entry.update(entry)

    sync_parameters_from_features(target_cfg, list(feature_cols))

    return entry


def execute_predictions(
    selected_models: List[Dict[str, Any]],
    input_df: pd.DataFrame,
    *,
    config: Dict[str, Any],
    material_key: str,
    target_key: str,
    confirmed_by_user: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    result_df = input_df.copy()
    infos: List[str] = []
    errors: List[str] = []
    prediction_cols: List[str] = []

    if not confirmed_by_user:
        return result_df, infos, ['请先勾选用户确认，确认输入结构和工艺参数后再执行预测。']
    if len(selected_models) != 1:
        return result_df, infos, ['可信门户每个材料/目标只能使用一个已发布模型，请先在管理端只启用一个发布版本。']

    model_entry = selected_models[0]
    label = model_entry.get('label') or model_entry.get('model_name') or 'model'
    try:
        summary = run_confirmed_prediction(
            {
                'material_type': material_key,
                'target': target_key,
                'inputs': input_df,
                'confirmed_by_user': confirmed_by_user,
            },
            config=config,
        )
        predictions = summary.prediction
        if not isinstance(predictions, (list, tuple, np.ndarray, pd.Series)):
            predictions = [predictions]
        predictions = list(predictions)
        if len(predictions) != len(result_df):
            raise ValueError(f'模型返回 {len(predictions)} 条结果，但输入有 {len(result_df)} 行。')
        base_col = f'prediction__{label}'
        pred_col = base_col
        counter = 2
        while pred_col in result_df.columns:
            pred_col = f'{base_col}_{counter}'
            counter += 1
        result_df[pred_col] = predictions
        prediction_cols.append(pred_col)
        info_text = f'{label}: 已通过发布契约完成可信预测'
        if summary.model_version:
            info_text += f'，版本 {summary.model_version}'
        if summary.feature_workflow_id:
            info_text += f'，工作流 {summary.feature_workflow_id}'
        infos.append(info_text)
        infos.extend(f'{label}: {warning}' for warning in summary.warnings)
    except Exception as exc:
        errors.append(f'{label}: {exc}')

    if len(prediction_cols) > 1:
        result_df['prediction_mean'] = result_df[prediction_cols].mean(axis=1)
    return result_df, infos, errors

def metric_text(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "暂无指标"
    if metrics.get("r2") is not None:
        return f"R²={float(metrics['r2']):.4f}"
    if metrics.get("rmse") is not None:
        return f"RMSE={float(metrics['rmse']):.4f}"
    if metrics.get("mae") is not None:
        return f"MAE={float(metrics['mae']):.4f}"
    return "已登记"


def _material_icon(material_key: str) -> str:
    return {
        "epoxy_resin": "resin",
        "ud_cfrp": "carbon_fiber",
    }.get(material_key, "molecule")


def _material_statistics(material_cfg: Dict[str, Any]) -> Tuple[int, int]:
    targets = [target for _, target in target_items(material_cfg) if target.get("enabled", True)]
    published_models = sum(
        1
        for target in targets
        for model in model_items(target)
        if model.get("enabled") is True and str(model.get("publication_status") or "").strip().lower() == "published"
    )
    return len(targets), published_models


def _portal_service_status() -> Tuple[str, str, int]:
    try:
        ai_config = load_ai_config(PROJECT_ROOT)
        services = [
            item for item in ai_config.get("services", [])
            if item.get("enabled") and item.get("purpose") in {"both", "input_parsing", "result_explanation"}
        ]
    except Exception:
        return "不可用", "AI 配置读取失败，仍可使用手动与批量预测", 0
    if not services:
        return "未配置", "AI 为可选能力，不影响 Python 预测流程", 0
    return "已连接", f"{services[0].get('label') or services[0].get('service_id') or 'AI 服务'} 可用于输入解析", len(services)


def material_card_html(material_key: str, material_cfg: Dict[str, Any]) -> str:
    status = "enabled" if material_cfg.get("enabled") else "unknown"
    target_count, model_count = _material_statistics(material_cfg)
    label = html_escape(str(material_cfg.get("label") or material_key))
    description = html_escape(str(material_cfg.get("description") or ""))
    icon_markup = svg_icon(_material_icon(material_key), 28, label)
    state_markup = render_status_badge(status)
    return f"""
    <article class="portal-card portal-material-entry" data-material="{html_escape(material_key)}">
        <div class="portal-card-top">
            <div class="portal-card-icon">{icon_markup}</div>
            <div class="portal-title-group">
                <div class="portal-title">{label}</div>
                <div class="portal-card-code">{html_escape(material_key.upper())}</div>
            </div>
            <div class="portal-card-state">{state_markup}</div>
        </div>
        <div class="portal-desc">{description}</div>
        <div class="portal-card-metrics">
            <span><strong>{target_count}</strong> 个性能项</span>
            <span><strong>{model_count}</strong> 个已发布模型</span>
        </div>
    </article>
    """

# ==============================================================================
# 课题组经典常用树脂与固化剂预设库 (Portal Preset Libraries)
# ==============================================================================
PORTAL_PRESET_RESINS = {
    "双酚A二缩水甘油醚 (E-51 / DGEBA)": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
    "双酚F环氧树脂 (DGEBF)": "C1OC1COc1ccc(Cc2ccc(OCC3CO3)cc2)cc1",
    "双酚S环氧树脂 (DGEBS / 耐温极性)": "C1OC1COc1ccc(S(=O)(=O)c2ccc(OCC3CO3)cc2)cc1",
    "四缩水甘油基二氨基二苯甲烷 (AG-80 / TGDDM / 航空承力)": "C1OC1CN(CC2CO2)c3ccc(Cc4ccc(N(CC5CO5)CC6CO6)cc4)cc3",
    "三缩水甘油基对氨基苯酚 (AFG-90 / TGDAP)": "C1OC1CN(CC2CO2)c3ccc(OCC4CO4)cc3",
    "酚醛环氧树脂 (EPN)": "C1OC1COc1cccc(Cc2cccc(OCC3CO3)c2)c1",
    "芴基双酚环氧树脂 (BHPF-EP / 卡基耐热)": "C1OC1COc2ccc3c(c2)C4(c5ccccc5-c3c4)c6ccc(OCC7CO7)cc6",
    "间苯二甲酸二缩水甘油酯 (DGEP)": "O=C(OCC1CO1)c2cccc(C(=O)OCC3CO3)c2",
    "六氟双酚A二缩水甘油醚 (BPAF-EP / 低介电)": "FC(F)(F)C(c1ccc(OCC2CO2)cc1)(c1ccc(OCC2CO2)cc1)C(F)(F)F",
}

PORTAL_PRESET_HARDENERS = {
    "4,4'-二氨基二苯砜 (4,4'-DDS / 标杆耐高温)": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1",
    "3,3'-二氨基二苯砜 (3,3'-DDS / 宽工艺窗口)": "Nc1cccc(S(=O)(=O)c2cccc(N)c2)c1",
    "4,4'-二氨基二苯甲烷 (DDM)": "Nc1ccc(Cc2ccc(N)cc2)cc1",
    "4,4'-二氨基二苯醚 (ODA)": "Nc1ccc(Oc2ccc(N)cc2)cc1",
    "异佛尔酮二胺 (IPDA / 脂环高韧)": "CC1(C)CC(C)(CN)CC(N)C1",
    "甲基四氢苯酐 (MTHPA / 电绝缘)": "CC1=CCC2C(=O)OC(=O)C2C1",
    "甲基六氢苯酐 (MHHPA / 耐候)": "CC1CCC2C(=O)OC(=O)C2C1",
    "间苯二胺 (m-PDA)": "Nc1cccc(N)c1",
    "4,4'-二氨基联苯 (DAB / 液晶刚性)": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "9,9-双(4-氨基苯基)芴 (FDA / 芴二胺)": "Nc1ccc(C2(c3ccccc3-c3ccccc32)c2ccc(N)cc2)cc1",
}

# 课题组经典常用配方库：树脂 + 固化剂 + 配比 + 固化制度（一键整体载入）
# phr 为按当量化学计量的参考值；temp/time 为代表性等效单阶段固化制度，可按文献/实验调整。
def validate_recipe_schedule(
    schedule: str, *, declared_stages: int | None = None
) -> List[Tuple[float, float]]:
    """解析固化制度并校验阶段数（防止静默丢阶段）。

    背景：``core.process_features._schedule_pairs`` 的正则要求温度是数字，
    **非数字温度会被静默丢弃且不报错**：

        '室温/24 h + 80 °C/2 h'   → [(80.0, 2.0)]              ← 室温阶段丢失
        '25 °C/24 h + 80 °C/2 h'  → [(25.0,24.0),(80.0,2.0)]    ← 正确

    因此配方库一律使用**显式数字温度**，并在加载时用本函数断言阶段数一致。

    参数
    ----
    schedule : 形如 ``"80 °C/2 h + 150 °C/3 h"``
    declared_stages : 声明的阶段数；给定且与实际解析数不符时抛错

    抛出
    ----
    ValueError : 无法解析，或阶段数与声明不符
    """
    from core.process_features import _schedule_pairs

    text = str(schedule or "").strip()
    if not text:
        raise ValueError("固化制度为空。")
    try:
        pairs = _schedule_pairs(text)
    except ValueError as exc:
        raise ValueError(f"固化制度无法解析：{text!r}（{exc}）") from exc
    if not pairs:
        raise ValueError(f"固化制度无法解析：{text!r}")
    if declared_stages is not None and len(pairs) != int(declared_stages):
        raise ValueError(
            f"固化制度阶段数不符：声明 {int(declared_stages)} 阶段，"
            f"实际解析出 {len(pairs)} 阶段（{text!r}）。"
            "请检查是否使用了非数字温度（如「室温」会被静默丢弃）。"
        )
    return [(float(temperature), float(duration)) for temperature, duration in pairs]


PORTAL_PRESET_RECIPES = [
    {
        "name": "E-51 / DDM 通用结构",
        "tags": ["通用结构", "中温固化"],
        "resin_key": "双酚A二缩水甘油醚 (E-51 / DGEBA)",
        "resin_short": "E-51 / DGEBA",
        "hardener_key": "4,4'-二氨基二苯甲烷 (DDM)",
        "hardener_short": "DDM",
        "phr": 26.0,
        "temp": 120.0,
        "time": 3.0,
        "cure_schedule": "80 °C/2 h + 120 °C/2 h",
        "cure_stages": 2,
        "note": "标准双酚A环氧结构配方；典型制度 80 °C/2 h + 120 °C/2 h 阶段固化，此处取代表性等效单阶段。",
    },
    {
        "name": "E-51 / DDS 耐高温",
        "tags": ["耐高温", "高Tg"],
        "resin_key": "双酚A二缩水甘油醚 (E-51 / DGEBA)",
        "resin_short": "E-51 / DGEBA",
        "hardener_key": "4,4'-二氨基二苯砜 (4,4'-DDS / 标杆耐高温)",
        "hardener_short": "4,4'-DDS",
        "phr": 33.0,
        "temp": 160.0,
        "time": 4.0,
        "cure_schedule": "130 °C/2 h + 180 °C/2 h",
        "cure_stages": 2,
        "note": "DDS 砜基高刚性交联网络；典型制度 130 °C/2 h + 180 °C/2 h，取等效 160 °C/4 h。",
    },
    {
        "name": "AG-80 / DDS 航空承力",
        "tags": ["航空航天", "碳纤维基体"],
        "resin_key": "四缩水甘油基二氨基二苯甲烷 (AG-80 / TGDDM / 航空承力)",
        "resin_short": "AG-80 / TGDDM",
        "hardener_key": "4,4'-二氨基二苯砜 (4,4'-DDS / 标杆耐高温)",
        "hardener_short": "4,4'-DDS",
        "phr": 50.0,
        "temp": 170.0,
        "time": 4.0,
        "cure_schedule": "130 °C/1 h + 180 °C/2 h",
        "cure_stages": 2,
        "note": "经典航空预浸料基体体系（5208 类）；典型制度 130 °C/1 h + 180 °C/2 h，取等效 170 °C/4 h。",
    },
    {
        "name": "E-51 / MTHPA 电绝缘",
        "tags": ["电绝缘", "酸酐固化"],
        "resin_key": "双酚A二缩水甘油醚 (E-51 / DGEBA)",
        "resin_short": "E-51 / DGEBA",
        "hardener_key": "甲基四氢苯酐 (MTHPA / 电绝缘)",
        "hardener_short": "MTHPA",
        "phr": 85.0,
        "temp": 130.0,
        "time": 4.0,
        "cure_schedule": "80 °C/2 h + 140 °C/4 h",
        "cure_stages": 2,
        "note": "酸酐体系常配合叔胺促进剂（如 DMP-30）；典型制度 80 °C/2 h + 140 °C/4 h，取等效 130 °C/4 h。",
    },
    {
        "name": "DGEBF / IPDA 高韧常温",
        "tags": ["高韧性", "低温固化"],
        "resin_key": "双酚F环氧树脂 (DGEBF)",
        "resin_short": "DGEBF",
        "hardener_key": "异佛尔酮二胺 (IPDA / 脂环高韧)",
        "hardener_short": "IPDA",
        "phr": 25.0,
        "temp": 80.0,
        "time": 4.0,
        # 原写法为「室温/24 h + 80 °C/2 h」——「室温」是非数字温度，会被
        # core.process_features._schedule_pairs 静默丢弃（26 h 误算为 2 h），
        # 故改写为显式 25 °C。配方库一律使用数字温度。
        "cure_schedule": "25 °C/24 h + 80 °C/2 h",
        "cure_stages": 2,
        "note": "脂环胺体系低粘度高韧性；典型制度 25 °C/24 h + 80 °C/2 h 后固化，取等效 80 °C/4 h。",
    },
    {
        "name": "E-51 / m-PDA 中温经典",
        "tags": ["中温固化", "经典文献"],
        "resin_key": "双酚A二缩水甘油醚 (E-51 / DGEBA)",
        "resin_short": "E-51 / DGEBA",
        "hardener_key": "间苯二胺 (m-PDA)",
        "hardener_short": "m-PDA",
        "phr": 14.0,
        "temp": 100.0,
        "time": 3.0,
        "cure_schedule": "80 °C/2 h + 150 °C/2 h",
        "cure_stages": 2,
        "note": "间苯二胺经典中温体系；典型制度 80 °C/2 h + 150 °C/2 h，取等效 100 °C/3 h。",
    },
    {
        "name": "BPAF-EP / DDS 低介电",
        "tags": ["低介电", "电子封装"],
        "resin_key": "六氟双酚A二缩水甘油醚 (BPAF-EP / 低介电)",
        "resin_short": "BPAF-EP",
        "hardener_key": "4,4'-二氨基二苯砜 (4,4'-DDS / 标杆耐高温)",
        "hardener_short": "4,4'-DDS",
        "phr": 28.0,
        "temp": 160.0,
        "time": 4.0,
        "cure_schedule": "130 °C/2 h + 180 °C/2 h",
        "cure_stages": 2,
        "note": "含氟体系低吸湿低介电；典型制度 130 °C/2 h + 180 °C/2 h，取等效 160 °C/4 h。",
    },
    {
        "name": "EPN / DDM 高交联耐热",
        "tags": ["高交联", "耐热"],
        "resin_key": "酚醛环氧树脂 (EPN)",
        "resin_short": "EPN (酚醛环氧)",
        "hardener_key": "4,4'-二氨基二苯甲烷 (DDM)",
        "hardener_short": "DDM",
        "phr": 28.0,
        "temp": 150.0,
        "time": 4.0,
        "cure_schedule": "100 °C/2 h + 150 °C/3 h",
        "cure_stages": 2,
        "note": "酚醛环氧多官能高交联密度；典型制度 100 °C/2 h + 150 °C/3 h，取等效 150 °C/4 h。",
    },
]


def _self_check_preset_recipe_schedules() -> None:
    """模块加载时自检配方库：每个 cure_schedule 的阶段数必须与声明一致。

    防止未来新增配方时误写非数字温度（如「室温」），导致阶段被静默丢弃、
    工艺特征（process_total_time_h 等）算出错误值。

    自检失败即**在导入时立即报错**，而不是等到用户预测时才发现。
    """
    problems: List[str] = []
    for recipe in PORTAL_PRESET_RECIPES:
        name = recipe.get("name") or "<未命名>"
        try:
            validate_recipe_schedule(
                recipe.get("cure_schedule") or "",
                declared_stages=recipe.get("cure_stages"),
            )
        except (ValueError, TypeError) as exc:
            problems.append(f"{name}: {exc}")
    if problems:
        raise ValueError("配方库固化制度自检失败：\n  - " + "\n  - ".join(problems))


_self_check_preset_recipe_schedules()


@st.cache_data(show_spinner=False, max_entries=256)
def _render_2d_molecule_png_b64(smiles: str, width: int = 360, height: int = 180) -> Tuple[bool, str, str]:
    """Render 2D structure image and return (is_valid, base64_png_or_error, formula)."""
    if not smiles or not str(smiles).strip():
        return False, "empty", ""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdMolDescriptors
        import io
        import base64
        
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return False, "SMILES 语法无效（无法构建分子图）", ""
        Chem.SanitizeMol(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        img = Draw.MolToImage(mol, size=(int(width), int(height)), kekulize=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        return True, b64_str, formula
    except Exception as e:
        return False, f"化学解析异常: {e}", ""


def init_smiles_field_state(state_key: str, default: str = "") -> None:
    """确保 SMILES 输入字段在 session_state 中有初始值。"""
    if state_key not in st.session_state:
        st.session_state[state_key] = str(default or "")


def _recipe_value_for_field(field: Dict[str, Any], recipe: Dict[str, Any]) -> Any:
    """将配方值按字段名/标签模糊匹配到对应输入项；无匹配返回 None。"""
    name = str(field.get("name") or "").lower()
    label = str(field.get("label") or "")
    if "resin" in name or "树脂" in label:
        return recipe.get("resin_smiles")
    if "hardener" in name or "curing_agent" in name or "固化剂" in label:
        return recipe.get("hardener_smiles")
    if "phr" in name or "份数" in label:
        return recipe.get("phr")
    if "temp" in name or "温度" in label:
        return recipe.get("temp")
    if "time" in name or "时间" in label:
        return recipe.get("time")
    return None


def _recipe_display_value(value: Any) -> str:
    """配方数值转展示/预填字符串：整数浮点去掉多余小数位。"""
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return str(value)


def recipe_card_html(recipe: Dict[str, Any]) -> str:
    """渲染单张常用配方卡片（纯展示，应用按钮由调用方补充）。"""
    tags = "".join(
        f'<span class="portal-recipe-tag">{html_escape(str(tag))}</span>'
        for tag in (recipe.get("tags") or [])
    )
    note = html_escape(str(recipe.get("note") or ""))
    temp = _recipe_display_value(recipe.get("temp"))
    time_text = _recipe_display_value(recipe.get("time"))
    phr = _recipe_display_value(recipe.get("phr"))
    return f"""
    <article class="portal-recipe-card" title="{note}">
        <div class="portal-recipe-head">
            <span class="portal-recipe-name">{html_escape(str(recipe.get('name') or ''))}</span>
            <span class="portal-recipe-cure-chip">{temp} °C / {time_text} h</span>
        </div>
        <div class="portal-recipe-tags">{tags}</div>
        <div class="portal-recipe-line"><span class="portal-recipe-key">树脂</span><span>{html_escape(str(recipe.get('resin_short') or ''))}</span></div>
        <div class="portal-recipe-line"><span class="portal-recipe-key">固化剂</span><span>{html_escape(str(recipe.get('hardener_short') or ''))}</span></div>
        <div class="portal-recipe-line"><span class="portal-recipe-key">配比</span><span>100 : {phr} phr</span></div>
    </article>
    """


def render_recipe_library(material_key: str, target_key: str) -> Dict[str, Any] | None:
    """渲染『常用配方快速载入』面板。

    返回需要预填的配方 dict（仅在应用后的下一个渲染周期生效一次），否则 None。
    """
    pending_key = f"pending_recipe_{material_key}_{target_key}"
    flash_key = f"recipe_flash_{material_key}_{target_key}"

    active_recipe: Dict[str, Any] | None = None
    pending = st.session_state.pop(pending_key, None)
    if isinstance(pending, dict):
        active_recipe = {
            "resin_smiles": PORTAL_PRESET_RESINS.get(str(pending.get("resin_key") or ""), ""),
            "hardener_smiles": PORTAL_PRESET_HARDENERS.get(str(pending.get("hardener_key") or ""), ""),
            "phr": pending.get("phr"),
            "temp": pending.get("temp"),
            "time": pending.get("time"),
        }

    flash_name = st.session_state.pop(flash_key, "")
    if flash_name:
        st.success(f"已载入配方「{html_escape(str(flash_name))}」，结构与工艺参数已填入下方表单，可继续逐项微调后提交预测。")

    st.markdown("#### ⚗️ 常用配方快速载入")
    st.caption("课题组经典环氧配方库：一键整体填入树脂/固化剂结构、配比与固化制度；应用后仍可逐项微调。")
    recipe_cols = st.columns(2)
    for index, recipe in enumerate(PORTAL_PRESET_RECIPES):
        with recipe_cols[index % 2]:
            st.markdown(recipe_card_html(recipe), unsafe_allow_html=True)
            if st.button(
                "⚡ 载入此配方",
                key=f"apply_recipe_{material_key}_{target_key}_{index}",
                width="stretch",
            ):
                st.session_state[pending_key] = recipe
                st.session_state[flash_key] = str(recipe.get("name") or "配方")
                st.rerun()
    st.caption("注：配比为按当量化学计量的参考值，固化制度为代表性等效单阶段；请结合文献与实验方案确认后使用。")
    return active_recipe


def render_jsme_editor(field_key: str, current_smiles: str = "") -> None:
    """Render an embedded lightweight JSME 2D chemical structure editor."""
    jsme_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
        <script type="text/javascript">
            function jsmeOnLoad() {{
                var startingSmiles = "{current_smiles}";
                jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "280px", {{
                    "options": "query,hydrogens,autoez"
                }});
                if (startingSmiles && startingSmiles.trim() !== "") {{
                    jsmeApplet.readGenericMolecularInput(startingSmiles);
                }}
            }}
            function copyJsmeSmiles() {{
                var smiles = jsmeApplet.smiles();
                if (navigator.clipboard) {{
                    navigator.clipboard.writeText(smiles).then(function() {{
                        alert("已将画板 SMILES 复制到剪贴板:\\n" + smiles + "\\n\\n请直接粘贴到上方文本框即可！");
                    }});
                }} else {{
                    prompt("请复制生成的 SMILES:", smiles);
                }}
            }}
        </script>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; background: transparent; }}
            #jsme_container {{ width: 100%; height: 280px; }}
            .jsme-btn {{
                margin-top: 6px;
                padding: 6px 14px;
                background: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
            }}
            .jsme-btn:hover {{ background: #1d4ed8; }}
        </style>
    </head>
    <body>
        <div id="jsme_container"></div>
        <button class="jsme-btn" type="button" onclick="copyJsmeSmiles()">📋 复制画板 SMILES 到剪贴板</button>
    </body>
    </html>
    """
    st.components.v1.html(jsme_html, height=330, scrolling=False)


def render_smiles_field(field: Dict[str, Any], scope_key: str, recipe: Dict[str, Any] | None = None) -> str:
    """多模态分子结构输入复合组件：包含经典预设、手动粘贴、截图识别、2D在线画板与即时拓扑图看板。"""
    label = field.get("label") or field["name"]
    name_low = str(field.get("name") or "").lower()
    state_key = f"{scope_key}_{field['name']}"
    init_smiles_field_state(state_key, field.get("default", ""))
    if recipe is not None:
        recipe_value = _recipe_value_for_field(field, recipe)
        if recipe_value is not None and str(recipe_value).strip():
            st.session_state[state_key] = str(recipe_value).strip()

    is_hardener = any(token in name_low for token in ("hardener", "curing", "固化", "交联", "amine"))
    preset_dict = PORTAL_PRESET_HARDENERS if is_hardener else PORTAL_PRESET_RESINS
    role_label = "固化剂" if is_hardener else "树脂/主体"

    st.markdown(f"##### 🧪 {label}")
    
    # 左右双栏布局：左侧多模态输入，右侧即时 2D 分子图看板
    left_col, right_col = st.columns([1.35, 1.0], gap="medium")
    
    with left_col:
        mode_tab1, mode_tab2, mode_tab3, mode_tab4 = st.tabs(
            ["🏷️ 常用预设", "✏️ 手动粘贴", "📷 截图识别", "🎨 2D画板"]
        )
        
        # 1. 经典预设
        with mode_tab1:
            st.caption(f"一键载入课题组常用经典 {role_label} 单体：")
            preset_options = ["(自定义 / 保持当前输入)"] + list(preset_dict.keys())
            selected_preset = st.selectbox(
                f"选择经典 {role_label}",
                options=preset_options,
                key=f"{state_key}_preset_select",
                label_visibility="collapsed",
            )
            if selected_preset != "(自定义 / 保持当前输入)":
                chosen_smi = preset_dict[selected_preset]
                if st.session_state.get(state_key) != chosen_smi:
                    st.session_state[state_key] = chosen_smi
                    st.rerun()

        # 2. 手动粘贴
        with mode_tab2:
            st.caption("直接输入或粘贴分子标准 SMILES 字符串：")
            st.text_area(
                "SMILES 字符串",
                key=state_key,
                placeholder=field.get("placeholder") or "例如: CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
                height=95,
                label_visibility="collapsed",
            )

        # 3. 截图 OCR 识别
        with mode_tab3:
            st.caption("拖入分子结构图或从文献截图中识别 SMILES：")
            img_c1, img_c2 = st.columns([2, 1])
            with img_c1:
                uploaded = st.file_uploader(
                    "上传图片",
                    type=SMILES_UPLOAD_TYPES,
                    key=f"{state_key}_upload",
                    label_visibility="collapsed",
                )
            with img_c2:
                hand_drawn = st.checkbox("手绘结构", key=f"{state_key}_handdrawn")

            if uploaded is not None and st.button("🚀 识别并填入", key=f"{state_key}_recognize"):
                try:
                    from core.image_smiles_extractor import decimer_is_available, smiles_from_bytes
                    ok, msg = decimer_is_available()
                    if not ok:
                        st.error(msg)
                    else:
                        with st.spinner("正在通过 DECIMER 识别分子图像..."):
                            preds = smiles_from_bytes(
                                uploaded.getvalue(),
                                uploaded.name,
                                confidence=False,
                                hand_drawn=bool(hand_drawn),
                            )
                            if preds:
                                st.session_state[state_key] = preds[0].smiles
                                st.success(f"识别成功: `{preds[0].smiles}`")
                                st.rerun()
                except Exception as exc:
                    st.error(f"SMILES 识别失败: {exc}")

        # 4. 2D 在线分子画板
        with mode_tab4:
            st.caption("在下方画板中绘制分子，点击复制按钮后粘贴到文本框：")
            render_jsme_editor(state_key, current_smiles=st.session_state.get(state_key, ""))

    # 右侧：即时 2D 分子拓扑结构看板
    with right_col:
        current_smiles = str(st.session_state.get(state_key, "")).strip()
        st.markdown(f"**🔬 实时 2D 拓扑结构预览**")
        if current_smiles:
            is_valid, content, formula = _render_2d_molecule_png_b64(current_smiles)
            if is_valid:
                st.markdown(
                    f'<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:6px; text-align:center;">'
                    f'<img src="data:image/png;base64,{content}" style="max-width:100%; height:auto; display:block; margin:0 auto;" />'
                    f'<div style="font-size:12px; color:#475569; margin-top:4px;"><strong>分子式:</strong> {formula}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#fef2f2; border:1px dashed #ef4444; border-radius:8px; padding:24px 12px; text-align:center; color:#991b1b; font-size:13px;">'
                    f'⚠️ <strong>化学语法有误</strong><br/><span style="font-size:11px; color:#b91c1c;">{content}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:45px 12px; text-align:center; color:#64748b; font-size:12px;">'
                '👈 请在左侧选择预设、粘贴 SMILES 或画板绘制，结构图将在此即时呈现'
                '</div>',
                unsafe_allow_html=True,
            )

    return st.session_state.get(state_key, "")


def default_number(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def default_integer(value: Any, fallback: int = 0) -> int:
    try:
        if value is None or value == "":
            return fallback
        return int(float(value))
    except Exception:
        return fallback


def _portal_default_for_field(
    field: Dict[str, Any], *, target_col: str
) -> Dict[str, Any] | None:
    """查字段的全表统计默认值（仅 manual_input 分区）。

    spec 硬约束：只有 manual_input 字段可以拿到默认值。本函数的调用方
    （``render_parameter_inputs``）只处理 manual 字段，此处再传
    ``partition="manual_input"`` 双保险 —— 模块内已强制非 manual 分区返回 None。

    任何异常都降级为无默认值（门户不能因缺 JSON 而崩溃）。
    """
    try:
        from core.portal_input_defaults import default_for_feature

        return default_for_feature(
            str(field.get("name") or ""),
            partition="manual_input",
            target_col=str(target_col or ""),
        )
    except Exception:
        return None


def _portal_default_display_value(record: Dict[str, Any] | None) -> Any:
    """把默认值记录转成可直接预填的值（无记录时 None）。"""
    if not isinstance(record, dict):
        return None
    value = record.get("value")
    return None if value is None or value == "" else value


def _portal_default_audit_detail(record: Dict[str, Any] | None) -> str:
    """生成默认值的审计说明（供「已自动填充」明细条展示）。"""
    if not isinstance(record, dict):
        return ""
    share = record.get("share")
    support = record.get("support")
    table = record.get("source_table") or ""
    parts = []
    if isinstance(share, (int, float)):
        parts.append(f"占比 {share:.1%}")
    if isinstance(support, (int, float)):
        parts.append(f"n={int(support)}")
    if table:
        parts.append(str(table))
    return "全表统计：" + "，".join(parts) if parts else "全表统计默认值"


def render_parameter_inputs(
    parameters: List[Dict[str, Any]],
    scope_key: str,
    recipe: Dict[str, Any] | None = None,
    target_col: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    if not parameters:
        st.info("当前性能项还没有配置输入参数。请先在管理页面设置参数。")
        return pd.DataFrame([{}]), ["未配置参数"]

    values: Dict[str, Any] = {}
    errors: List[str] = []
    
    # 针对分子结构字段（smiles 类型），采用单列独占全宽渲染以提供最佳视觉预览体验
    smiles_fields = [f for f in parameters if str(f.get("kind") or "text").strip().lower() == "smiles"]
    other_fields = [f for f in parameters if str(f.get("kind") or "text").strip().lower() != "smiles"]

    # 1. 渲染分子结构复合字段
    for field in smiles_fields:
        label = field.get("label") or field["name"]
        required = bool(field.get("required", False))
        val = render_smiles_field(field, scope_key, recipe=recipe)
        if required and not str(val).strip():
            errors.append(f"【{label}】不能为空，请在上方输入合法分子结构。")
        values[field["name"]] = val

    # 2. 渲染数值/工艺/选择参数字段（双栏栅格）
    if other_fields:
        columns = st.columns(2)
        for index, field in enumerate(other_fields):
            label = field.get("label") or field["name"]
            kind = str(field.get("kind") or "text").strip()
            required = bool(field.get("required", False))
            key_base = f"{scope_key}_{field['name']}"
            name_low = str(field["name"]).lower()

            # 智能推断常用物理单位与占位提示
            unit_suffix = ""
            if "temp" in name_low or "温度" in label:
                unit_suffix = " (°C)"
            elif "time" in name_low or "时间" in label:
                unit_suffix = " (h)"
            elif "phr" in name_low or "份数" in label:
                unit_suffix = " (phr)"
            elif "ratio" in name_low or "比" in label:
                unit_suffix = " (比值)"
            elif "ahew" in name_low or "eew" in name_low or "当量" in label:
                unit_suffix = " (g/eq)"

            display_label = f"{label}{unit_suffix}" if not label.endswith(")") else label

            recipe_value = _recipe_value_for_field(field, recipe) if recipe is not None else None
            # 配方库优先；无配方时用全表统计默认值预填（仅 manual_input 分区）
            default_record = None
            if recipe_value is None:
                default_record = _portal_default_for_field(field, target_col=target_col)
            portal_default = _portal_default_display_value(default_record)
            effective_default = recipe_value if recipe_value is not None else portal_default
            if effective_default is not None:
                field.setdefault("_autofill", []).append({
                    "feature": field["name"],
                    "value": effective_default,
                    "origin": "recipe" if recipe_value is not None else "default",
                    "detail": (
                        "配方库预填" if recipe_value is not None
                        else _portal_default_audit_detail(default_record)
                    ),
                })

            with columns[index % 2]:
                if kind == "number":
                    if effective_default is not None:
                        st.session_state[f"{key_base}_number"] = default_number(effective_default, 0.0)
                        value = st.number_input(
                            display_label,
                            key=f"{key_base}_number",
                            help=field.get("help") or None,
                            format="%.4f",
                        )
                    else:
                        value = st.number_input(
                            display_label,
                            key=f"{key_base}_number",
                            value=None if field.get("default") is None else default_number(field.get("default"), 0.0),
                            help=field.get("help") or None,
                            format="%.4f",
                        )
                elif kind == "integer":
                    if effective_default is not None:
                        st.session_state[f"{key_base}_integer"] = default_integer(effective_default, 0)
                        value = st.number_input(
                            display_label,
                            key=f"{key_base}_integer",
                            help=field.get("help") or None,
                            step=1,
                        )
                    else:
                        value = st.number_input(
                            display_label,
                            key=f"{key_base}_integer",
                            value=None if field.get("default") is None else default_integer(field.get("default"), 0),
                            help=field.get("help") or None,
                            step=1,
                        )
                    value = None if value is None else int(value)
                elif kind == "select":
                    options = parse_options(field.get("options"))
                    default_value = str(
                        effective_default if effective_default is not None
                        else field.get("default", "") or ""
                    )
                    if not options:
                        options = [""]
                    elif not default_value:
                        options = [""] + options
                    if default_value and default_value not in options:
                        # 统计默认值可能不在契约枚举内（如 DMA 未列入 options）→ 补入并置于首位
                        options = [default_value] + options
                    default_index = options.index(default_value) if default_value in options else 0
                    value = st.selectbox(
                        display_label,
                        options=options,
                        index=default_index,
                        key=f"{key_base}_select",
                        help=field.get("help") or None,
                    )
                elif effective_default is not None:
                    st.session_state[f"{key_base}_text"] = _recipe_display_value(effective_default)
                    value = st.text_input(
                        display_label,
                        key=f"{key_base}_text",
                        placeholder=field.get("placeholder") or "",
                        help=field.get("help") or None,
                    )
                else:
                    value = st.text_input(
                        display_label,
                        key=f"{key_base}_text",
                        value="" if field.get("default") is None else str(field.get("default")),
                        placeholder=field.get("placeholder") or "",
                        help=field.get("help") or None,
                    )

            if required and (kind in {"text", "select"} and str(value).strip() == "" or kind in {"number", "integer"} and value is None):
                errors.append(f"【{display_label}】为必填项，不能为空")
            values[field["name"]] = value

    return pd.DataFrame([values]), errors


def reset_user_selection() -> None:
    st.session_state["predict_selected_material"] = ""
    st.session_state["predict_selected_target"] = ""
    st.session_state.pop("predict_tab_intent", None)


def render_user_home(config: Dict[str, Any]) -> None:
    materials = material_items(config)
    enabled_materials = [item for item in materials if item[1].get("enabled", True)]
    total_targets = sum(_material_statistics(material_cfg)[0] for _, material_cfg in materials)
    total_models = sum(_material_statistics(material_cfg)[1] for _, material_cfg in materials)
    ai_status, ai_detail, ai_service_count = _portal_service_status()
    ai_badge = render_status_badge("enabled" if ai_service_count else "unknown")

    st.markdown(
        f"""
        <section class="portal-hero portal-home-hero">
            <div class="portal-home-copy">
                <div class="portal-brand-mark"><span class="portal-brand-line"></span>邹华维课题组</div>
                <div class="hero-kicker">材料预测平台 · Scientific prediction workspace</div>
                <h1>从材料输入到可复现预测</h1>
                <p>选择材料方向，确认结构与工艺输入，沿着可追踪的特征流程调用已发布模型。AI 只负责辅助整理，Python 负责可信计算。</p>
                <div class="portal-hero-actions">
                    <span class="portal-assurance">{svg_icon("validation", 16)} 输入需人工确认</span>
                    <span class="portal-assurance">{svg_icon("features", 16)} 特征流程可追溯</span>
                    <span class="portal-assurance">{svg_icon("calculation", 16)} 模型版本可核验</span>
                </div>
            </div>
            <div class="portal-home-status">
                <div class="portal-status-heading">系统状态</div>
                <div class="portal-status-row"><span>{svg_icon("calculation", 18)} 预测门户</span>{render_status_badge("enabled")}</div>
                <div class="portal-status-row"><span>{svg_icon("result", 18)} 已发布模型</span><strong>{total_models}</strong></div>
                <div class="portal-status-row"><span>{svg_icon("ai", 18)} AI 输入助手</span>{ai_badge}</div>
                <div class="portal-status-detail">{html_escape(ai_detail)}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class=\"portal-section-heading\"><span>AI</span> 一句话智能输入</div>", unsafe_allow_html=True)
    home_ai_text = st.text_area(
        "描述材料、配方和工艺",
        key="home_ai_input",
        height=90,
        placeholder="例如：AG-80 环氧树脂，固化剂 DDS，50 phr，170 °C 固化 4 小时。也可以直接粘贴一段实验记录。",
        label_visibility="collapsed",
    )
    ai_col1, ai_col2 = st.columns([1.1, 2.4])
    with ai_col1:
        if st.button("🤖 AI 全自动解析并填入", key="home_open_ai", type="primary", width="stretch"):
            if not home_ai_text.strip():
                st.warning("请先在上方输入配方/工艺描述文本。")
            elif enabled_materials:
                st.session_state["ai_home_pending_text"] = home_ai_text
                st.session_state["predict_selected_material"] = enabled_materials[0][0]
                st.session_state["predict_selected_target"] = ""
                st.session_state["predict_tab_intent"] = "ai"
                st.rerun()
            else:
                st.warning("当前没有已开放的材料方向。")
        if st.button("仅打开 AI 输入助手", key="home_open_ai_manual", width="stretch"):
            if enabled_materials:
                st.session_state["predict_selected_material"] = enabled_materials[0][0]
                st.session_state["predict_selected_target"] = ""
                st.session_state["predict_tab_intent"] = "ai"
                st.rerun()
            else:
                st.warning("当前没有已开放的材料方向。")
    with ai_col2:
        st.caption(
            f"粘贴一段配方/工艺描述，AI 自动解析全部参数并直接填入手动输入表单（当前使用第一个可用方向："
            f"{enabled_materials[0][1].get('label') or enabled_materials[0][0]}），核对后勾选确认即可预测；"
            "AI 不会自动生成 EEW、AHEW、PHR 等推导值。未配置 AI 时仍可使用手动输入和批量上传。"
        )

    st.markdown("<div class=\"portal-section-heading\"><span>01</span> 选择材料方向</div>", unsafe_allow_html=True)
    if not materials:
        st.info("当前还没有配置材料方向，请先在管理页面完成配置。")
        return

    cols = st.columns(max(1, min(3, len(materials))))
    for index, (material_key, material_cfg) in enumerate(materials):
        with cols[index % len(cols)]:
            st.markdown(material_card_html(material_key, material_cfg), unsafe_allow_html=True)
            button_text = "进入预测工作台" if material_cfg.get("enabled") else "方向暂未开放"
            if st.button(button_text, key=f"open_{material_key}", disabled=not material_cfg.get("enabled"), width="stretch"):
                st.session_state["predict_selected_material"] = material_key
                st.session_state["predict_selected_target"] = ""
                st.session_state["predict_tab_intent"] = "manual"
                st.rerun()

    if len(materials) <= 2:
        st.markdown(
            f"<article class=\"portal-card portal-placeholder-card\">{svg_icon('molecule', 24)}<div><div class=\"portal-title\">更多材料方向</div><div class=\"portal-card-code\">EXTENSIBLE MATERIAL LIBRARY</div><div class=\"portal-desc\">后续可在管理页面新增材料体系、性能项和已发布模型，不影响现有预测入口。</div></div></article>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class=\"portal-section-heading\"><span>02</span> 可复现工作流</div>", unsafe_allow_html=True)
    st.markdown(render_stage_timeline("validated", 0), unsafe_allow_html=True)
    workflow_cols = st.columns(4)
    workflow_items = [
        ("input", "输入确认", "手动、批量或 AI 建议均需人工确认"),
        ("validation", "结构校验", "SMILES / BigSMILES 由 Python 复核"),
        ("features", "特征准备", "按发布模型绑定的流程和顺序执行"),
        ("result", "结果解释", "预测值、版本和警告一起保留"),
    ]
    for column, (icon_name, title, description) in zip(workflow_cols, workflow_items):
        with column:
            st.markdown(
                f"<div class=\"portal-workflow-card\">{svg_icon(icon_name, 22)}<strong>{title}</strong><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

def render_model_brief(models: List[Dict[str, Any]]) -> None:
    if not models:
        st.warning("当前性能项还没有可用模型，请先在管理页面上传训练好的模型。")
        return
    for model in models:
        state_text = "启用中" if model.get("enabled", True) else "已停用"
        summary = model_contract_summary(model)
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-card-title">{model.get('label', '')}</div>
                <div class="model-card-meta">
                    <span>{model.get('model_name', '')}</span>
                    <span>{metric_text(model.get('metrics') or {})}</span>
                    <span>{state_text}</span>
                    <span>{len(model.get('feature_cols') or [])} 个特征</span>
                    <span>发布状态：{summary['publication_status']}</span>
                </div>
                <div class="model-card-notes">{model.get('notes', '') or '无备注'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"版本 {summary['model_version']} | profile {summary['model_profile_id'] or '未声明'}"
            f" | contract v{summary['contract_schema_version'] or '?'}"
            f" | registry {summary['feature_registry_version']}@{summary['feature_registry_hash'] or '—'}"
            f" | workflow {summary['workflow_hash'] or '—'}"
            f" | artifact {summary['artifact_hash'] or '—'}"
        )


def render_prediction_results(result_df: pd.DataFrame, infos: List[str], errors: List[str], download_name: str) -> None:
    for info in infos:
        st.info(info)
    for error in errors:
        st.error(error)

    if result_df.empty:
        st.warning("没有可展示的预测结果。")
        return

    st.dataframe(result_df, width="stretch")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载预测结果 CSV",
        data=csv_bytes,
        file_name=download_name,
        mime="text/csv",
    )


def render_user_page(config: Dict[str, Any]) -> None:
    selected_material = st.session_state.get("predict_selected_material", "")
    if not selected_material:
        render_user_home(config)
        return

    material_cfg = (config.get("materials") or {}).get(selected_material)
    if not material_cfg:
        reset_user_selection()
        st.rerun()
        return

    target_count, model_count = _material_statistics(material_cfg)
    top1, top2 = st.columns([6, 1.4])
    with top1:
        st.markdown(
            f"""
            <section class="portal-hero compact portal-detail-hero">
                <div class="hero-kicker">Prediction workbench</div>
                <h1>{html_escape(str(material_cfg.get('label') or selected_material))}</h1>
                <p>{html_escape(str(material_cfg.get('description') or ''))}</p>
                <div class="portal-hero-actions">
                    <span class="portal-assurance">{svg_icon('result', 16)} {target_count} 个可用性能项</span>
                    <span class="portal-assurance">{svg_icon('calculation', 16)} {model_count} 个已发布模型</span>
                    <span class="portal-assurance">{svg_icon('validation', 16)} 输入确认后计算</span>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.write("")
        if st.button("返回方向选择", key=f"back_to_materials_{selected_material}", width="stretch"):
            reset_user_selection()
            st.rerun()
    if not material_cfg.get("enabled"):
        st.warning(material_cfg.get("coming_soon_message") or "该方向暂未开放。")
        return

    enabled_targets = [
        (target_key, target_cfg)
        for target_key, target_cfg in target_items(material_cfg)
        if target_cfg.get("enabled", True)
    ]
    if not enabled_targets:
        st.info("该方向下还没有可用的预测性能项，请先到管理页面配置。")
        return

    target_keys = [item[0] for item in enabled_targets]
    current_target = st.session_state.get("predict_selected_target")
    if current_target not in target_keys:
        current_target = target_keys[0]
        st.session_state["predict_selected_target"] = current_target

    selected_target = st.selectbox(
        "选择预测性能",
        options=target_keys,
        index=target_keys.index(current_target),
        format_func=lambda key: material_cfg["targets"][key].get("label") or key,
        key="predict_selected_target",
    )
    target_cfg = material_cfg["targets"][selected_target]

    st.caption(target_cfg.get("description") or "当前性能项描述未填写。")

    enabled_models = [model for model in model_items(target_cfg) if _is_publishable_ui_model(model)]
    render_model_brief(enabled_models)
    if not enabled_models:
        return

    selected_model_id = st.selectbox(
        "选择用于预测的模型",
        options=[model["id"] for model in enabled_models],
        index=0,
        format_func=lambda model_id: next(
            (
                f"{model.get('label', '')} | {metric_text(model.get('metrics') or {})}"
                for model in enabled_models
                if model["id"] == model_id
            ),
            model_id,
        ),
    )
    selected_models = [model for model in enabled_models if model["id"] == selected_model_id]
    if not selected_models:
        st.warning("请至少选择一个模型。")
        return
    selected_contract, registry_snapshot = _model_contract(selected_models[0])
    manual_fields = build_manual_input_fields(selected_contract, registry_snapshot)
    workflow_fields = build_workflow_source_fields(selected_contract, registry_snapshot)

    # 平台同步诊断（Agent C 后端 + 本页 UI）
    with st.expander("🩺 平台同步诊断", expanded=False):
        render_sync_diagnostics_panel(selected_models[0])

    confirmed_by_user = st.checkbox(
        "我确认已检查输入结构、配方/工艺参数，并同意调用已发布模型进行预测",
        key=f"prediction_confirmation_{selected_material}_{selected_target}",
    )
    st.caption("门户预测不会自动补齐缺失特征；模型必须处于已启用、已发布且通过契约验证状态。")
    st.markdown(render_stage_timeline("validated", 0), unsafe_allow_html=True)

    # 工作台功能区导航：用持久化的 segmented_control 取代原先根据 open_ai 标志动态换序的 st.tabs。
    # st.tabs 的激活状态不跨重跑保留（任何控件交互后都会回到第一个标签），
    # 导致：在 AI 标签里点「解析输入」后被弹回手动输入、解析结果看似丢失；
    # AI 标签置顶时，在手动输入里点确认又被弹回 AI。改用 segmented_control 后：
    # 1) 重跑后停留在当前功能区；2) 入口按钮可通过 predict_tab_intent 精确跳转。
    active_tab_key = f"predict_active_tab_{selected_material}_{selected_target}"
    tab_intent = st.session_state.pop("predict_tab_intent", None)
    if tab_intent == "ai":
        st.session_state[active_tab_key] = "AI 辅助输入"
    elif tab_intent == "manual":
        st.session_state[active_tab_key] = "手动输入"
    if st.session_state.get(active_tab_key) not in ("手动输入", "批量上传", "AI 辅助输入", "当前配置"):
        st.session_state[active_tab_key] = "手动输入"
    active_tab = st.segmented_control(
        "功能区",
        options=["手动输入", "批量上传", "AI 辅助输入", "当前配置"],
        key=active_tab_key,
        label_visibility="collapsed",
    )
    if not active_tab:
        active_tab = "手动输入"

    if active_tab == "手动输入":
        st.markdown("### 手动输入参数")
        home_flash = st.session_state.pop("ai_home_flash", None)
        if home_flash:
            st.success(home_flash)
        # 常用配方快速载入（仅对树脂/环氧类材料方向展示；返回本次需要预填的配方）
        active_recipe: Dict[str, Any] | None = None
        material_label_text = str(material_cfg.get("label") or "")
        if "epoxy" in selected_material.lower() or "环氧" in material_label_text or "树脂" in material_label_text:
            active_recipe = render_recipe_library(selected_material, selected_target)
        # 输入分区：必填人工 / 可选人工 / 分子结构 / 系统计算，全部由当前 contract 驱动。
        partition_plan = build_input_partition_plan(selected_contract)
        all_input_fields: List[Dict[str, Any]] = []

        # 分子/结构输入块（workflow source）：不属于 5 个可编辑分区，
        # 它是 workflow 的原始输入（SMILES），单独置于配方区之前。
        if workflow_fields:
            st.markdown("#### 分子结构输入")
            st.caption("树脂/固化剂 SMILES 等原始结构；由当前模型 workflow 消费。")
            source_df, source_errors = render_parameter_inputs(
                workflow_fields,
                f"manual_{selected_material}_{selected_target}_molecular",
                recipe=active_recipe,
            )

        for section in partition_plan:
            # ① 配方 / ② 固化制度 / ③ 高级测试条件：可编辑
            # ③ 默认折叠（已按全表统计预填，不修改也能预测）
            # ④⑤ 只读展示
            if section["kind"] == "display":
                st.markdown(f"#### {section['title']}")
                st.caption(section["description"])
                if section["features"]:
                    st.info(
                        "系统将自动计算："
                        + "、".join(section["features"][:12])
                        + ("…" if len(section["features"]) > 12 else "")
                    )
                else:
                    st.caption("该模型契约未声明此分区。")
                all_input_fields.append(pd.DataFrame([{}]))
                continue

            section_input_fields = [
                field for field in manual_fields if field["name"] in section["features"]
            ]
            collapsed = section["group"] == "test_conditions"
            container = (
                st.expander(f"{section['title']}（已预填，可展开修改）", expanded=False)
                if collapsed
                else st.container()
            )
            with container:
                st.markdown(f"#### {section['title']}") if not collapsed else None
                st.caption(section["description"])
                if section_input_fields:
                    section_df, section_errors = render_parameter_inputs(
                        section_input_fields,
                        f"manual_{selected_material}_{selected_target}_{section['group']}",
                        recipe=active_recipe,
                        target_col=selected_target,
                    )
                else:
                    section_df, section_errors = pd.DataFrame([{}]), []
                    st.caption("当前模型契约未声明此分区的可输入字段。")
            all_input_fields.append(section_df)
            if section_errors:
                for err in section_errors:
                    st.warning(err)

        # 「已自动填充 N 项」明细条：让用户看得见、可追溯系统代填了什么
        autofill_entries: List[Dict[str, Any]] = []
        for field in manual_fields:
            autofill_entries.extend(field.get("_autofill") or [])
        if workflow_fields:
            source_df, source_errors = render_parameter_inputs(
                workflow_fields,
                f"manual_{selected_material}_{selected_target}_molecular",
                recipe=active_recipe,
            )
            all_input_fields.append(source_df)
            for err in source_errors:
                st.warning(err)
        autofill_summary = build_autofilled_summary(autofill_entries)
        if autofill_summary["count"]:
            with st.expander(f"📋 {autofill_summary['title']}（点击查看依据）", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "字段": row["feature"],
                                "取值": row["value"],
                                "来源": "配方库" if row["origin"] == "recipe" else "全表统计",
                                "依据": row["detail"],
                            }
                            for row in autofill_summary["rows"]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
        # 合并各分区输入帧（dict 合并保持向后单行结构）
        merged_values: Dict[str, Any] = {}
        for frame in all_input_fields:
            if isinstance(frame, pd.DataFrame) and not frame.empty and frame.columns.tolist() != [None]:
                for column in frame.columns:
                    if column:
                        merged_values[column] = frame.iloc[0][column]
        manual_df = pd.DataFrame([merged_values]) if merged_values else pd.DataFrame([{}])
        validation_errors = []
        if not manual_fields and not workflow_fields:
            validation_errors = ["当前模型契约未声明任何输入字段，无法手动预测。"]
        
        # 预测前健康检查与就绪指示条 (Pre-flight Health Check)
        st.markdown("---")
        st.markdown("##### 🩺 预测输入就绪检查")
        check_cols = st.columns(3)
        with check_cols[0]:
            mol_ready = all(str(merged_values.get(f["name"], "")).strip() != "" for f in workflow_fields if f.get("required"))
            if mol_ready:
                st.markdown("✅ **分子结构**：就绪")
            else:
                st.markdown("⚠️ **分子结构**：待完善必填项")
        with check_cols[1]:
            proc_ready = all(merged_values.get(f["name"]) is not None and str(merged_values.get(f["name"])).strip() != "" for f in manual_fields if f.get("required"))
            if proc_ready:
                st.markdown("✅ **配方/工艺**：就绪")
            else:
                st.markdown("⚠️ **配方/工艺**：待完善必填项")
        with check_cols[2]:
            if confirmed_by_user:
                st.markdown("✅ **安全授权**：已确认")
            else:
                st.markdown("⚠️ **安全授权**：待勾选上方授权框")

        if validation_errors:
            for err in validation_errors:
                st.warning(err)

        if st.button("🚀 开始可复现模型预测", type="primary", key=f"predict_manual_{selected_material}_{selected_target}", use_container_width=True):
            if validation_errors:
                st.error("请先补全必填项后再预测。")
            elif not confirmed_by_user:
                st.error("请先勾选上方的『我确认已检查输入结构、配方/工艺参数...』授权复选框。")
            elif not mol_ready or not proc_ready:
                st.error("输入未通过就绪检查，请检查是否有遗漏的必填分子结构或工艺参数。")
            else:
                task_key = f"portal_task_manual_{selected_material}_{selected_target}"
                st.session_state[task_key] = submit_prediction_task(config, selected_material, selected_target, manual_df, explain=False)
                st.rerun()
        render_portal_task_panel(
            st.session_state.get(f"portal_task_manual_{selected_material}_{selected_target}"),
            session_key=f"manual_{selected_material}_{selected_target}",
        )

    elif active_tab == "批量上传":
        st.markdown("### 上传待预测数据")
        uploaded = st.file_uploader(
            "上传 CSV / Excel 数据文件",
            type=DATA_UPLOAD_TYPES,
            key=f"batch_input_{selected_material}_{selected_target}",
        )
        if uploaded is not None:
            try:
                batch_df = load_data_file(uploaded)
                st.dataframe(batch_df.head(20), width="stretch")
                if st.button("执行批量预测", type="primary", key=f"predict_batch_{selected_material}_{selected_target}"):
                    if not confirmed_by_user:
                        st.error("请先确认已检查批量输入结构和配方/工艺参数。")
                    else:
                        task_key = f"portal_task_batch_{selected_material}_{selected_target}"
                        st.session_state[task_key] = submit_prediction_task(config, selected_material, selected_target, batch_df, explain=False)
                        st.rerun()
                render_portal_task_panel(
                    st.session_state.get(f"portal_task_batch_{selected_material}_{selected_target}"),
                    session_key=f"batch_{selected_material}_{selected_target}",
                )
            except Exception as exc:
                st.error(f"读取文件失败: {exc}")
        else:
            st.info("适合已经准备好批量输入表格的场景。缺失特征不会被自动补齐，系统会按发布契约校验并提示具体问题。")

    elif active_tab == "AI 辅助输入":
        render_ai_assistant_tab(config, selected_material, selected_target, target_cfg, selected_contract, registry_snapshot)

    elif active_tab == "当前配置":
        st.markdown("### 当前性能项参数")
        param_df = parameter_editor_df(manual_fields + workflow_fields)
        st.dataframe(param_df, width="stretch", hide_index=True)

        st.markdown("### 当前已选模型")
        model_preview_rows = []
        for model in selected_models:
            summary = model_contract_summary(model)
            model_preview_rows.append(
                {
                    "模型标签": model.get("label", ""),
                    "模型类型": model.get("model_name", ""),
                    "指标": metric_text(model.get("metrics") or {}),
                    "特征数": len(model.get("feature_cols") or []),
                    "契约特征数": summary["contract_features"],
                    "特征来源": model.get("feature_source", ""),
                    "含分子特征流程": "是" if model.get("has_feature_process") else "否",
                    "发布状态": summary["publication_status"],
                    "版本": summary["model_version"],
                    "profile": summary["model_profile_id"],
                    "contract": f"v{summary['contract_schema_version'] or '?'}",
                    "registry": f"{summary['feature_registry_version']}@{summary['feature_registry_hash'] or '—'}",
                    "workflow": summary["workflow_hash"] or "—",
                    "artifact": summary["artifact_hash"] or "—",
                    "更新时间": model.get("updated_at", ""),
                }
            )
        st.dataframe(pd.DataFrame(model_preview_rows), width="stretch", hide_index=True)


def save_material_basic_settings(config: Dict[str, Any], material_key: str, label: str, enabled: bool, description: str, coming_soon_message: str) -> None:
    material_cfg = config["materials"][material_key]
    material_cfg["label"] = label.strip() or material_cfg["label"]
    material_cfg["enabled"] = bool(enabled)
    material_cfg["description"] = description.strip()
    material_cfg["coming_soon_message"] = coming_soon_message.strip()
    save_config(config)


def render_admin_page(config: Dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="portal-hero">
            <div class="hero-kicker">Admin Page</div>
            <h1>预测平台管理后台</h1>
            <p>在这里维护材料方向、性能项、输入参数和预测模型。用户端会直接读取这里的配置。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_material, tab_model, tab_export = st.tabs(["材料与性能", "模型管理", "配置导出"])

    with tab_material:
        material_keys = [item[0] for item in material_items(config)]
        selected_material = st.selectbox(
            "选择材料方向",
            options=material_keys,
            format_func=lambda key: config["materials"][key].get("label") or key,
            key="admin_material_selector",
        )
        material_cfg = config["materials"][selected_material]

        st.markdown("### 材料方向设置")
        col1, col2 = st.columns(2)
        with col1:
            material_label = st.text_input("名称", value=material_cfg.get("label", ""), key=f"material_label_{selected_material}")
            material_enabled = st.checkbox("开放入口", value=material_cfg.get("enabled", True), key=f"material_enabled_{selected_material}")
        with col2:
            material_desc = st.text_area("说明", value=material_cfg.get("description", ""), key=f"material_desc_{selected_material}", height=110)
            coming_msg = st.text_area("未开放提示", value=material_cfg.get("coming_soon_message", ""), key=f"coming_msg_{selected_material}", height=110)

        if st.button("保存材料方向设置", key=f"save_material_{selected_material}"):
            save_material_basic_settings(config, selected_material, material_label, material_enabled, material_desc, coming_msg)
            st.success("材料方向设置已保存。")

        st.markdown("---")
        st.markdown("### 新增性能项")
        add_col1, add_col2 = st.columns(2)
        with add_col1:
            new_target_key = st.text_input("性能项编码", value="", placeholder="例如 tg_new", key=f"new_target_key_{selected_material}")
        with add_col2:
            new_target_label = st.text_input("性能项名称", value="", placeholder="例如 Tg", key=f"new_target_label_{selected_material}")

        if st.button("新增性能项", key=f"add_target_{selected_material}"):
            target_key = slugify(new_target_key, fallback="")
            if not target_key:
                st.error("请先填写性能项编码。")
            elif target_key in material_cfg.get("targets", {}):
                st.error("该性能项编码已存在。")
            else:
                material_cfg.setdefault("targets", {})[target_key] = make_target_config(new_target_label or target_key)
                save_config(config)
                st.success("已新增性能项。")
                st.rerun()

        st.markdown("---")
        st.markdown("### 编辑现有性能项")
        target_key_list = [item[0] for item in target_items(material_cfg)]
        if not target_key_list:
            st.info("当前材料方向下还没有性能项。")
        else:
            selected_target = st.selectbox(
                "选择性能项",
                options=target_key_list,
                format_func=lambda key: material_cfg["targets"][key].get("label") or key,
                key=f"admin_target_selector_{selected_material}",
            )
            target_cfg = material_cfg["targets"][selected_target]

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                target_label = st.text_input("性能项名称", value=target_cfg.get("label", ""), key=f"target_label_{selected_target}")
                target_enabled = st.checkbox("性能项启用", value=target_cfg.get("enabled", True), key=f"target_enabled_{selected_target}")
            with tcol2:
                target_desc = st.text_area("性能项说明", value=target_cfg.get("description", ""), key=f"target_desc_{selected_target}", height=110)

            st.markdown("#### 输入参数配置")
            editor_df = st.data_editor(
                parameter_editor_df(target_cfg.get("parameters") or []),
                width="stretch",
                hide_index=True,
                num_rows="dynamic",
                key=f"parameter_editor_{selected_material}_{selected_target}",
                column_config={
                    "name": st.column_config.TextColumn("字段名", help="最终会作为输入 DataFrame 的列名"),
                    "label": st.column_config.TextColumn("显示名"),
                    "kind": st.column_config.SelectboxColumn("类型", options=PARAMETER_KINDS),
                    "required": st.column_config.CheckboxColumn("必填"),
                    "default": st.column_config.TextColumn("默认值"),
                    "placeholder": st.column_config.TextColumn("占位提示"),
                    "help": st.column_config.TextColumn("帮助文字"),
                    "options": st.column_config.TextColumn("选项列表", help="select 类型可换行填写多个选项"),
                },
            )

            save_col, delete_col = st.columns([1.2, 1])
            with save_col:
                if st.button("保存性能项与参数", type="primary", key=f"save_target_{selected_material}_{selected_target}"):
                    target_cfg["label"] = target_label.strip() or selected_target
                    target_cfg["enabled"] = bool(target_enabled)
                    target_cfg["description"] = target_desc.strip()
                    target_cfg["parameters"] = normalize_parameter_rows(editor_df)
                    save_config(config)
                    st.success("性能项配置已保存。")
            with delete_col:
                if st.button("移除当前性能项", key=f"delete_target_{selected_material}_{selected_target}"):
                    material_cfg["targets"].pop(selected_target, None)
                    save_config(config)
                    st.success("性能项已移除。")
                    st.rerun()

    with tab_model:
        material_keys = [item[0] for item in material_items(config)]
        selected_material = st.selectbox(
            "选择材料方向",
            options=material_keys,
            format_func=lambda key: config["materials"][key].get("label") or key,
            key="admin_model_material_selector",
        )
        material_cfg = config["materials"][selected_material]
        target_key_list = [item[0] for item in target_items(material_cfg)]
        if not target_key_list:
            st.info("请先在“材料与性能”中创建性能项。")
        else:
            selected_target = st.selectbox(
                "选择性能项",
                options=target_key_list,
                format_func=lambda key: material_cfg["targets"][key].get("label") or key,
                key="admin_model_target_selector",
            )
            target_cfg = material_cfg["targets"][selected_target]

            st.markdown("### 已登记模型")
            registered_models = model_items(target_cfg)
            if not registered_models:
                st.info("该性能项下还没有模型。")
            else:
                try:
                    from core.prediction_portal import rollback_publication
                    rollback_available = True
                except ImportError:
                    rollback_available = False
                status_labels_admin = {
                    "published": "✅ 已发布", "needs_validation": "⏳ 待验证",
                    "draft": "📝 草稿", "disabled": "⛔ 已停用", "legacy": "📦 legacy",
                }
                for model in registered_models:
                    c1, c2, c3, c4 = st.columns([6, 1, 1, 1])
                    with c1:
                        pub_raw = str(model.get("publication_status") or "").strip().lower()
                        pub_label = status_labels_admin.get(pub_raw, pub_raw or "未记录")
                        gate = model.get("gate_report") if isinstance(model.get("gate_report"), dict) else {}
                        gate_errors = gate.get("errors") or []
                        gate_hint = f"｜gate: {gate.get('status')}" if gate.get("status") else ""
                        failure_hint = ""
                        if gate_errors:
                            failure_hint = f"｜失败原因：{str(gate_errors[0])[:120]}"
                        st.markdown(
                            f"""
                            <div class="model-card">
                                <div class="model-card-title">{model.get('label', '')}</div>
                                <div class="model-card-meta">
                                    <span>{model.get('model_name', '')}</span>
                                    <span>{metric_text(model.get('metrics') or {})}</span>
                                    <span>{len(model.get('feature_cols') or [])} 个特征</span>
                                    <span>{'启用' if model.get('enabled', True) else '停用'}</span>
                                    <span>{pub_label}{gate_hint}</span>
                                </div>
                                <div class="model-card-notes">{model.get('notes', '') or '无备注'}{failure_hint}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"contract hash {str(model.get('contract_hash') or '—')[:8]}"
                            f" ｜ artifact hash {str(model.get('artifact_hash') or '—')[:8]}"
                            f" ｜ registry {str((model.get('registry_snapshot') or {}).get('registry_hash') or '—')[:8] if isinstance(model.get('registry_snapshot'), dict) else '—'}"
                        )
                    with c2:
                        toggle_label = "停用" if model.get("enabled", True) else "启用"
                        toggle_confirm_key = f"toggle_confirm_{model['id']}"
                        st.checkbox("确认", key=toggle_confirm_key, value=False, help="勾选后才能执行（二次确认）")
                        toggle_confirmed = bool(st.session_state.get(toggle_confirm_key))
                        if st.button(toggle_label, key=f"toggle_model_{model['id']}", disabled=not toggle_confirmed):
                            try:
                                toggle_model_enabled(config, selected_material, selected_target, model)
                            except ValueError as exc:
                                st.error(f"模型启用失败：{exc}")
                            else:
                                append_model_management_audit(
                                    action="enable" if not model.get("enabled", True) else "disable",
                                    model_id=model["id"], model_version=model.get("updated_at", ""),
                                    detail=f"{toggle_label} 模型 {model.get('label', '')}",
                                )
                                save_config(config)
                                st.rerun()
                    with c3:
                        rollback_confirm_key = f"rollback_confirm_{model['id']}"
                        st.checkbox("确认", key=rollback_confirm_key, value=False, help="回滚前请勾选（二次确认）")
                        if rollback_available and st.button("回滚发布", key=f"rollback_model_{model['id']}", disabled=not bool(st.session_state.get(rollback_confirm_key))):
                            try:
                                rollback_publication(config, material_key=selected_material, target_key=selected_target, version=str(model.get("version") or ""))
                            except (ValueError, TypeError) as exc:
                                st.error(f"回滚失败：{exc}")
                            else:
                                append_model_management_audit(
                                    action="rollback", model_id=model["id"], model_version=model.get("updated_at", ""),
                                    detail=f"回滚 {model.get('label', '')} 的发布",
                                )
                                save_config(config)
                                st.success(f"已回滚 {model.get('label', '')}")
                                st.rerun()
                    with c4:
                        remove_confirm_key = f"remove_confirm_{model['id']}"
                        st.checkbox("确认", key=remove_confirm_key, value=False, help="删除前请勾选（二次确认）")
                        if st.button("移除记录", key=f"remove_model_{model['id']}", disabled=not bool(st.session_state.get(remove_confirm_key))):
                            target_cfg["models"] = [item for item in target_cfg.get("models", []) if item.get("id") != model["id"]]
                            append_model_management_audit(
                                action="delete", model_id=model["id"], model_version=model.get("updated_at", ""),
                                detail=f"删除模型记录 {model.get('label', '')}",
                            )
                            save_config(config)
                            st.rerun()

            st.markdown("---")
            st.markdown("### 上传或替换模型")
            uploaded_model = st.file_uploader(
                "上传训练好的模型文件（推荐使用训练平台导出的 .joblib artifact）",
                type=["joblib", "pkl"],
                key=f"model_upload_{selected_material}_{selected_target}",
            )
            replace_options = [""] + [model["id"] for model in registered_models]
            replace_model_id = st.selectbox(
                "替换已有模型（可选）",
                options=replace_options,
                format_func=lambda model_id: (
                    "新增模型"
                    if model_id == ""
                    else next((model.get("label", model_id) for model in registered_models if model["id"] == model_id), model_id)
                ),
                key=f"replace_model_selector_{selected_material}_{selected_target}",
            )
            upload_label = st.text_input("模型显示名称", value="", placeholder="例如 Tg_XGBoost_v1", key=f"upload_label_{selected_material}_{selected_target}")
            upload_notes = st.text_area("模型备注", value="", placeholder="可记录训练日期、数据版本、适用范围等", key=f"upload_notes_{selected_material}_{selected_target}")
            feature_override_text = st.text_area(
                "特征列覆盖（可选，一行一个）",
                value="",
                placeholder="当模型文件里没有 feature_cols，或你想手动指定输入列顺序时使用",
                key=f"feature_override_{selected_material}_{selected_target}",
            )

            artifact_preview = None
            artifact_error = ""
            if uploaded_model is not None:
                try:
                    _, artifact_preview = preview_artifact(uploaded_model.getvalue())
                    st.json(artifact_preview)
                except Exception as exc:
                    artifact_error = str(exc)
                    st.error(f"模型预览失败: {exc}")

            if st.button("保存模型到平台", type="primary", key=f"save_model_btn_{selected_material}_{selected_target}"):
                if uploaded_model is None:
                    st.error("请先上传模型文件。")
                elif artifact_error:
                    st.error("当前上传文件无法解析，请先处理模型文件。")
                else:
                    feature_override = read_text_lines(feature_override_text)
                    entry = upsert_model_entry(
                        config=config,
                        material_key=selected_material,
                        target_key=selected_target,
                        file_name=uploaded_model.name,
                        file_bytes=uploaded_model.getvalue(),
                        label=upload_label.strip(),
                        notes=upload_notes.strip(),
                        feature_override=feature_override,
                        replace_model_id=replace_model_id.strip(),
                    )
                    save_config(config)
                    st.success(f"模型已保存: {entry.get('label', '')}")
                    st.rerun()

    with tab_export:
        st.markdown("### 当前配置预览")
        st.caption(f"配置文件位置: {CONFIG_PATH}")
        st.json(config)
        st.download_button(
            "下载 prediction_config.json",
            data=json.dumps(json_ready(config), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="prediction_config.json",
            mime="application/json",
        )


CUSTOM_CSS = """
<style>
:root {
    --portal-ink: #17324d;
    --portal-muted: #667085;
    --portal-line: #d8e1e8;
    --portal-accent: #0f7490;
    --portal-cyan: #0f7490;
    --portal-blue: #2563a8;
    --portal-green: #16805d;
    --portal-amber: #a46708;
    --portal-red: #b4233c;
    --portal-surface: #ffffff;
    --portal-surface-strong: #f7fafc;
}
.main .block-container {
    max-width: 1380px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}
[data-testid="stAppViewContainer"] { background: #f4f7fa; }
[data-testid="stHeader"] { background: rgba(244, 247, 250, 0.92); }
[data-testid="stSidebar"] { background: #edf3f7; border-right: 1px solid var(--portal-line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #17324d; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--portal-accent); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #475467; }
[data-testid="stTextInput"] label, [data-testid="stTextArea"] label, [data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label, [data-testid="stFileUploader"] label { color: #344054; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background: #ffffff; border-color: #cbd5e1; color: #172b3a; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color: #172b3a; }
[data-baseweb="select"] span { color: #344054; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    border: 1px solid #b8c7d4;
    border-radius: 7px;
    background: #ffffff;
    color: #17324d;
    transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { border-color: var(--portal-accent); background: #f0f8fa; transform: translateY(-1px); }
[data-testid="stButton"] button[kind="primary"] { background: #0f7490; border-color: #0f7490; color: #ffffff; }
[data-testid="stTabs"] [role="tab"] { color: #667085; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--portal-accent); }
[data-testid="stDataFrame"] { border: 1px solid var(--portal-line); border-radius: 8px; overflow: hidden; }
[data-testid="stSidebar"] .sidebar-brand { padding: .15rem 0 .85rem; border-bottom: 1px solid #d8e1e8; }
[data-testid="stSidebar"] .sidebar-brand-mark { color: #17324d; font-size: .9rem; font-weight: 800; letter-spacing: .04em; }
[data-testid="stSidebar"] .sidebar-brand-mark span { color: #17324d; }
[data-testid="stSidebar"] .sidebar-subtitle { margin-top: .28rem; color: #667085; font-size: .75rem; }
[data-testid="stSidebar"] .sidebar-section-title { margin: 1rem 0 .45rem; color: #667085; font-size: .7rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }
[data-testid="stSidebar"] .sidebar-context { padding: .75rem .8rem; border: 1px solid #cce4ea; border-radius: 8px; background: #f1fafb; }
[data-testid="stSidebar"] .sidebar-context-label { color: #667085; font-size: .7rem; }
[data-testid="stSidebar"] .sidebar-context-value { margin-top: .2rem; color: #17324d; font-size: .92rem; font-weight: 700; line-height: 1.35; }
[data-testid="stSidebar"] .sidebar-context-meta { margin-top: .25rem; color: #175b6b; font-size: .72rem; }
[data-testid="stSidebar"] .sidebar-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; }
[data-testid="stSidebar"] .sidebar-stat { padding: .55rem .45rem; border: 1px solid #d8e1e8; border-radius: 7px; background: #ffffff; text-align: center; }
[data-testid="stSidebar"] .sidebar-stat-value { color: #17324d; font-size: 1.08rem; font-weight: 750; line-height: 1.1; }
[data-testid="stSidebar"] .sidebar-stat-label { margin-top: .25rem; color: #667085; font-size: .65rem; line-height: 1.25; }
[data-testid="stSidebar"] .sidebar-flow { display: grid; gap: .45rem; }
[data-testid="stSidebar"] .sidebar-flow-item { display: flex; align-items: center; gap: .5rem; color: #667085; font-size: .76rem; }
[data-testid="stSidebar"] .sidebar-flow-item strong { display: grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; background: #e7edf2; color: #667085; font-size: .65rem; }
[data-testid="stSidebar"] .sidebar-flow-item.active { color: #175b6b; font-weight: 700; }
[data-testid="stSidebar"] .sidebar-flow-item.active strong { background: #0f7490; color: #ffffff; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { color: #667085; font-size: .72rem; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: .35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] { min-height: 2rem; padding: .35rem .55rem; border: 1px solid #d8e1e8; border-radius: 6px; background: #ffffff; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] { border-color: #8fc5cf; background: #f1fafb; color: #175b6b; }
[data-testid="stSidebar"] [data-testid="stButton"] button { min-height: 2.35rem; font-size: .78rem; }
.portal-hero, .portal-card, .model-card, .portal-status-panel, .portal-workflow-card {
    border: 1px solid var(--portal-line);
    border-radius: 10px;
    background: var(--portal-surface);
    box-shadow: 0 8px 24px rgba(23, 50, 77, 0.07);
}
.portal-hero { padding: 1.65rem 1.8rem; margin-bottom: 1.15rem; }
.portal-home-hero { min-height: 270px; display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(260px, 0.8fr); gap: 2rem; align-items: center; background: #ffffff; }
.portal-home-copy { min-width: 0; }
.portal-brand-mark { color: #17324d; font-size: .86rem; font-weight: 800; letter-spacing: .06em; }
.portal-brand-mark span { color: #17324d; }
.portal-brand-line { display: inline-block; width: 26px; height: 2px; margin-right: 8px; vertical-align: middle; background: var(--portal-cyan); }
.hero-kicker { display: inline-block; margin-top: .9rem; color: var(--portal-cyan); font-size: .73rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
.portal-hero h1 { margin: .35rem 0 .65rem; color: #17324d; font-size: clamp(2rem, 4vw, 3.15rem); letter-spacing: 0; line-height: 1.08; }
.portal-hero p { max-width: 48rem; margin: 0; color: #667085; line-height: 1.7; }
.portal-hero-actions { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.2rem; }
.portal-assurance { display: inline-flex; align-items: center; gap: .4rem; padding: .4rem .65rem; border: 1px solid #cce4ea; border-radius: 999px; color: #175b6b; background: #f1fafb; font-size: .77rem; }
.portal-assurance .portal-icon { color: var(--portal-cyan); }
.portal-home-status { padding: 1.05rem 1.15rem; border: 1px solid var(--portal-line); border-radius: 8px; background: #f7fafc; }
.portal-status-heading { margin-bottom: .8rem; color: #17324d; font-size: .82rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
.portal-status-row { display: flex; align-items: center; justify-content: space-between; gap: .7rem; padding: .65rem 0; border-top: 1px solid #e7edf2; color: #475467; font-size: .85rem; }
.portal-status-row > span { display: inline-flex; align-items: center; gap: .45rem; }
.portal-status-row .portal-icon { color: var(--portal-blue); }
.portal-status-row strong { color: #17324d; font-size: 1.15rem; }
.portal-status-detail { margin-top: .55rem; color: var(--portal-muted); font-size: .75rem; line-height: 1.5; }
.portal-section-heading { display: flex; align-items: center; gap: .7rem; margin: 1.65rem 0 .8rem; color: #17324d; font-size: 1.1rem; font-weight: 750; }
.portal-section-heading span { color: var(--portal-cyan); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .75rem; letter-spacing: .1em; }
.portal-card, .model-card { padding: 1.05rem 1.1rem; margin-bottom: .8rem; }
.portal-material-entry { min-height: 185px; transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease; }
.portal-material-entry:hover { border-color: #74b9c6; box-shadow: 0 12px 28px rgba(23, 50, 77, .12); transform: translateY(-2px); }
.portal-card-top { display: flex; align-items: flex-start; gap: .75rem; }
.portal-card-icon { display: grid; place-items: center; width: 44px; height: 44px; flex: 0 0 44px; border: 1px solid #cce4ea; border-radius: 9px; color: var(--portal-cyan); background: #f1fafb; }
.portal-title-group { min-width: 0; flex: 1; }
.portal-title, .model-card-title { color: #17324d; font-size: 1.08rem; font-weight: 750; }
.portal-card-code { margin-top: .2rem; color: #667085; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .68rem; letter-spacing: .08em; }
.portal-card-state { padding-top: .15rem; }
.portal-desc, .model-card-notes { color: var(--portal-muted); font-size: .88rem; line-height: 1.6; }
.portal-material-entry .portal-desc { min-height: 2.8rem; margin: 1rem 0 .85rem; }
.portal-card-metrics { display: flex; flex-wrap: wrap; gap: .55rem; padding-top: .75rem; border-top: 1px solid #e7edf2; color: #667085; font-size: .77rem; }
.portal-card-metrics strong { color: #17324d; }
.portal-workflow-card { min-height: 118px; padding: 1rem; background: #f7fafc; }
.portal-workflow-card .portal-icon { display: block; margin-bottom: .65rem; color: var(--portal-cyan); }
.portal-workflow-card strong { color: #17324d; font-size: .88rem; }
.portal-workflow-card p { margin: .4rem 0 0; color: var(--portal-muted); font-size: .76rem; line-height: 1.5; }
.portal-status-badge { display: inline-flex; align-items: center; gap: .38rem; color: #475467; font-size: .73rem; white-space: nowrap; }
.portal-status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--portal-blue); box-shadow: 0 0 0 3px rgba(37, 99, 168, .12); }
.portal-status-badge.success .portal-status-dot { background: var(--portal-green); box-shadow: 0 0 0 3px rgba(22, 128, 93, .12); }
.portal-status-badge.danger .portal-status-dot { background: var(--portal-red); box-shadow: 0 0 0 3px rgba(180, 35, 60, .12); }
.portal-stage-wrap { margin: .5rem 0 1.1rem; }
.portal-stage { color: #98a2b3; }
.portal-stage.done, .portal-stage.active { color: #344054; }
.portal-stage-marker { border-color: #cbd5e1; }
.portal-stage.done .portal-stage-marker { color: #ffffff; background: var(--portal-green); border-color: var(--portal-green); }
.portal-stage.active .portal-stage-marker { color: #ffffff; background: var(--portal-cyan); border-color: var(--portal-cyan); }
.status-pill { border-radius: 999px; padding: .24rem .62rem; font-size: .73rem; font-weight: 700; }
.status-open { color: #176b4f; background: #e8f5ef; }
.status-closed { color: #a61b35; background: #fdecef; }
.model-card-meta { display: flex; flex-wrap: wrap; gap: .5rem; margin: .55rem 0; }
.model-card-meta span { border: 1px solid #cce4ea; border-radius: 999px; padding: .18rem .55rem; color: #175b6b; background: #f1fafb; font-size: .74rem; }
/* 常用配方快速载入卡片 */
.portal-recipe-card { border: 1px solid var(--portal-line); border-radius: 10px; background: #ffffff; padding: .8rem .9rem .75rem; margin-bottom: .45rem; transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease; }
.portal-recipe-card:hover { border-color: #74b9c6; box-shadow: 0 10px 24px rgba(23, 50, 77, .10); transform: translateY(-1px); }
.portal-recipe-head { display: flex; align-items: center; justify-content: space-between; gap: .6rem; }
.portal-recipe-name { color: #17324d; font-weight: 750; font-size: .92rem; line-height: 1.3; }
.portal-recipe-cure-chip { flex: 0 0 auto; padding: .14rem .5rem; border-radius: 999px; border: 1px solid #cce4ea; background: #f1fafb; color: #175b6b; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .7rem; white-space: nowrap; }
.portal-recipe-tags { display: flex; flex-wrap: wrap; gap: .3rem; margin: .45rem 0 .55rem; }
.portal-recipe-tag { padding: .1rem .45rem; border-radius: 999px; background: #eef4f8; color: #344054; font-size: .68rem; }
.portal-recipe-line { display: flex; align-items: baseline; gap: .5rem; margin-top: .22rem; color: #475467; font-size: .8rem; }
.portal-recipe-line span:last-child { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .76rem; }
.portal-recipe-key { flex: 0 0 auto; color: #667085; font-size: .68rem; letter-spacing: .08em; }
.portal-recipe-card + [data-testid="stButton"] button { min-height: 2.1rem; font-size: .76rem; margin-bottom: .35rem; }
@media (max-width: 800px) {
    .main .block-container { padding: .75rem .65rem 2.25rem; max-width: 100%; }
    [data-testid="stHorizontalBlock"] { gap: .65rem; }
    [data-testid="stSidebar"] { min-width: 86vw; max-width: 86vw; }
    [data-testid="stSidebar"] .sidebar-stats { gap: .35rem; }
    [data-testid="stSidebar"] .sidebar-stat { padding: .5rem .3rem; }
    [data-testid="stSidebar"] .sidebar-stat-value { font-size: 1rem; }
    [data-testid="stSidebar"] .sidebar-context { padding: .7rem; }
    .portal-hero { padding: 1.1rem; margin-bottom: .85rem; }
    .portal-home-hero { grid-template-columns: 1fr; gap: 1.1rem; min-height: 0; }
    .portal-detail-hero { min-height: 0; }
    .portal-home-status { margin-top: .15rem; }
    .portal-hero-actions { gap: .4rem; margin-top: .95rem; }
    .portal-assurance { padding: .34rem .5rem; font-size: .7rem; }
    .portal-hero h1 { font-size: 1.85rem; line-height: 1.18; }
    .portal-hero p { font-size: .88rem; line-height: 1.6; }
    .portal-section-heading { margin-top: 1.25rem; font-size: 1rem; }
    .portal-card, .model-card { padding: .9rem; margin-bottom: .65rem; }
    .portal-material-entry { min-height: 0; }
    .portal-card-top { gap: .6rem; }
    .portal-card-state { margin-left: auto; }
    .portal-card-metrics { gap: .4rem; padding-top: .6rem; font-size: .72rem; }
    .portal-workflow-card { min-height: 0; padding: .85rem; }
    .portal-recipe-card { padding: .7rem .75rem; }
    .portal-recipe-name { font-size: .86rem; }
    .portal-recipe-cure-chip { font-size: .64rem; }
    .portal-stage-wrap { overflow-x: auto; padding: 10px 2px; margin-bottom: .8rem; }
    .portal-stage { min-width: 112px; font-size: .72rem; }
    [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; scrollbar-width: none; }
    [data-testid="stTabs"] [role="tab"] { flex: 0 0 auto; min-height: 2.5rem; padding: .45rem .65rem; font-size: .82rem; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height: 2.65rem; }
    [data-testid="stDataFrame"] { max-width: 100%; overflow-x: auto; }
}
@media (max-width: 480px) {
    .portal-brand-mark { font-size: .78rem; letter-spacing: .02em; }
    [data-testid="stSidebar"] .sidebar-brand-mark { font-size: .84rem; }
    .portal-hero h1 { font-size: 1.6rem; }
    .portal-home-status { padding: .85rem .9rem; }
    .portal-status-row { font-size: .78rem; }
    .portal-title, .model-card-title { font-size: 1rem; }
    .portal-card-code { font-size: .62rem; }
}
</style>
"""

def render_sidebar(config: Dict[str, Any]) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="sidebar-brand-mark"><span>邹华维课题组</span></div>'
            '<div class="sidebar-subtitle">材料预测平台</div></div>',
            unsafe_allow_html=True,
        )

        mode = st.radio("工作区", ["用户页面", "管理页面"], index=0, horizontal=True)
        material_count = len(config.get("materials") or {})
        target_count = sum(len(material.get("targets") or {}) for _, material in material_items(config))
        model_count = sum(
            len(target.get("models") or [])
            for _, material in material_items(config)
            for _, target in target_items(material)
        )

        selected_material = st.session_state.get("predict_selected_material", "")
        selected_target = st.session_state.get("predict_selected_target", "")
        material_cfg = (config.get("materials") or {}).get(selected_material, {})
        target_cfg = (material_cfg.get("targets") or {}).get(selected_target, {})
        selected_material_label = material_cfg.get("label") or "尚未选择材料方向"
        selected_target_label = target_cfg.get("label") or "请选择预测性能"

        if mode == "用户页面":
            st.markdown('<div class="sidebar-section-title">当前预测</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sidebar-context"><div class="sidebar-context-label">材料方向</div>'
                f'<div class="sidebar-context-value">{html_escape(str(selected_material_label))}</div>'
                f'<div class="sidebar-context-meta">目标：{html_escape(str(selected_target_label))}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sidebar-section-title">快速操作</div>', unsafe_allow_html=True)
            if st.button("回到材料方向", key="sidebar_home", width="stretch", disabled=not selected_material):
                reset_user_selection()
                st.rerun()
            if st.button(
                "跳转到 AI 输入助手",
                key="sidebar_ai",
                width="stretch",
                disabled=not selected_material,
                help="需要先进入某个材料方向的工作台；也可在首页点击「打开 AI 输入助手」自动进入。",
            ):
                st.session_state["predict_tab_intent"] = "ai"
                st.rerun()
            active_section = st.session_state.get(
                f"predict_active_tab_{selected_material}_{selected_target}", ""
            ) if selected_material else ""
            st.caption(f"当前功能区：{active_section or '尚未进入工作台'}")
        else:
            st.markdown('<div class="sidebar-section-title">管理导航</div>', unsafe_allow_html=True)
            st.caption("材料、性能项和模型配置集中在管理页面中。")

        st.markdown('<div class="sidebar-section-title">系统概览</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-stats">'
            f'<div class="sidebar-stat"><div class="sidebar-stat-value">{material_count}</div><div class="sidebar-stat-label">材料方向</div></div>'
            f'<div class="sidebar-stat"><div class="sidebar-stat-value">{target_count}</div><div class="sidebar-stat-label">性能项</div></div>'
            f'<div class="sidebar-stat"><div class="sidebar-stat-value">{model_count}</div><div class="sidebar-stat-label">模型数量</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"配置文件：{CONFIG_PATH.name}")

        st.markdown('<div class="sidebar-section-title">预测流程</div>', unsafe_allow_html=True)
        active_step = 1 if not selected_material else 2 if not selected_target else 3
        st.markdown(
            '<div class="sidebar-flow">'
            f'<div class="sidebar-flow-item {"active" if active_step == 1 else ""}"><strong>1</strong><span>选择材料</span></div>'
            f'<div class="sidebar-flow-item {"active" if active_step == 2 else ""}"><strong>2</strong><span>确认输入</span></div>'
            f'<div class="sidebar-flow-item {"active" if active_step == 3 else ""}"><strong>3</strong><span>调用模型</span></div>'
            '<div class="sidebar-flow-item"><strong>4</strong><span>查看结果</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    return mode


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=str(ASSET_ROOT / "portal-icon.svg"),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_scientific_theme()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = load_config()
    mode = render_sidebar(config)

    if mode == "管理页面":
        render_admin_page(config)
    else:
        render_user_page(config)


if __name__ == "__main__":
    main()
