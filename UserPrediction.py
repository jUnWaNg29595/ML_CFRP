# -*- coding: utf-8 -*-
from __future__ import annotations

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
    if not isinstance(model, dict) or model.get("enabled") is not True or str(model.get("publication_status") or "").strip().lower() != "published":
        return False
    gate = model.get("gate_report")
    if not isinstance(gate, dict) or gate.get("ok") is not True or str(gate.get("status") or "").strip().lower() != "valid":
        return False
    contract, snapshot = _model_contract(model)
    profile = snapshot.get("model_profile") if isinstance(snapshot, dict) else None
    if contract.get("schema_version") != 2 or not snapshot or not isinstance(profile, dict) or profile.get("status") != "approved":
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
    return activate_publication(
        config, material_key=material_key, target_key=target_key, entry=candidate
    )


def render_ai_assistant_tab(
    config: Dict[str, Any], material_key: str, target_key: str, target_cfg: Dict[str, Any],
    contract: Dict[str, Any] | None = None, registry_snapshot: Dict[str, Any] | None = None,
) -> None:
    st.markdown('### AI 辅助输入')
    st.caption('AI 只提取和整理你提供的信息；每个字段必须确认、修改后确认或明确拒绝，不能自动生成 EEW、AHEW、PHR、分子特征或工艺参数。')
    try:
        ai_config = load_ai_config(PROJECT_ROOT)
    except Exception as exc:
        st.error(f'AI 配置读取失败：{exc}')
        return
    services = [item for item in ai_config.get('services', []) if item.get('enabled') and item.get('purpose') in {'both', 'input_parsing'}]
    if not services:
        st.info('暂无已启用的输入助手服务；请在主平台侧边栏配置，或继续使用手动/批量输入。')
        return
    service = st.selectbox('AI 服务', services, format_func=lambda item: item.get('label') or item.get('service_id'), key=f'ai_service_{material_key}_{target_key}')
    text_key = f'ai_text_{material_key}_{target_key}'
    user_text = st.text_area('描述材料、配方和工艺信息', key=text_key, height=150, placeholder='例如：树脂 SMILES 为 CCO；固化温度 80 °C。')
    state_key = f'ai_state_{material_key}_{target_key}'
    if st.button('解析输入', key=f'ai_parse_{material_key}_{target_key}', type='secondary'):
        if not user_text.strip():
            st.warning('请先输入待解析的文本。')
        else:
            try:
                contract = contract if isinstance(contract, dict) else {}
                registry_snapshot = registry_snapshot if isinstance(registry_snapshot, dict) else {}
                field_defs = build_manual_input_fields(contract, registry_snapshot) + build_workflow_source_fields(contract, registry_snapshot)
                response = PortalAIClient(_ai_service_dataclass(service)).parse_input({
                    'material_type': material_key, 'target': target_key,
                    'field_descriptions': [
                        {'name': item.get('name'), 'label': item.get('label'), 'kind': item.get('kind'), 'required': item.get('required', False), 'allow_ai_generation': False}
                        for item in field_defs
                    ], 'user_text': user_text,
                })
                st.session_state[state_key] = build_ai_confirmation_state(response)
                st.success('解析完成，请逐项确认或拒绝。')
            except PortalAIError as exc:
                st.warning(f'AI 不可用：{exc}；已保留手动输入模式。')
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
        if st.button('创建 AI 辅助预测任务', key=f'ai_submit_{material_key}_{target_key}', type='primary'):
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

def init_smiles_field_state(state_key: str, default_value: Any) -> None:
    if state_key not in st.session_state:
        st.session_state[state_key] = "" if default_value is None else str(default_value)


def render_smiles_field(field: Dict[str, Any], scope_key: str) -> str:
    label = field.get("label") or field["name"]
    state_key = f"{scope_key}_{field['name']}"
    result_key = f"{state_key}_results"
    init_smiles_field_state(state_key, field.get("default", ""))

    st.text_area(
        label,
        key=state_key,
        placeholder=field.get("placeholder") or "直接输入 SMILES，或在下方上传结构图识别",
        help=field.get("help") or None,
        height=100,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            f"{label} 结构图 / PDF",
            type=SMILES_UPLOAD_TYPES,
            key=f"{state_key}_upload",
        )
    with col2:
        hand_drawn = st.checkbox("手绘结构", key=f"{state_key}_handdrawn")

    if uploaded is not None and st.button("识别为 SMILES", key=f"{state_key}_recognize"):
        try:
            from core.image_smiles_extractor import decimer_is_available, smiles_from_bytes

            ok, msg = decimer_is_available()
            if not ok:
                st.error(msg)
            else:
                preds = smiles_from_bytes(
                    uploaded.getvalue(),
                    uploaded.name,
                    confidence=False,
                    hand_drawn=bool(hand_drawn),
                )
                if preds:
                    st.session_state[state_key] = preds[0].smiles
                    st.session_state[result_key] = [
                        {
                            "filename": pred.filename,
                            "page": "" if pred.page_index is None else pred.page_index + 1,
                            "smiles": pred.smiles,
                        }
                        for pred in preds
                    ]
                    st.rerun()
        except Exception as exc:
            st.error(f"SMILES 识别失败: {exc}")

    if st.session_state.get(result_key):
        st.caption("最近一次结构图识别结果")
        st.dataframe(pd.DataFrame(st.session_state[result_key]), width="stretch", hide_index=True)

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


def render_parameter_inputs(parameters: List[Dict[str, Any]], scope_key: str) -> Tuple[pd.DataFrame, List[str]]:
    if not parameters:
        st.info("当前性能项还没有配置输入参数。请先在管理页面设置参数。")
        return pd.DataFrame([{}]), ["未配置参数"]

    values: Dict[str, Any] = {}
    errors: List[str] = []
    columns = st.columns(2)

    for index, field in enumerate(parameters):
        label = field.get("label") or field["name"]
        kind = str(field.get("kind") or "text").strip()
        required = bool(field.get("required", False))
        key_base = f"{scope_key}_{field['name']}"

        with columns[index % 2]:
            if kind == "number":
                value = st.number_input(
                    label,
                    key=f"{key_base}_number",
                    value=None if field.get("default") is None else default_number(field.get("default"), 0.0),
                    help=field.get("help") or None,
                    format="%.6f",
                )
            elif kind == "integer":
                value = st.number_input(
                    label,
                    key=f"{key_base}_integer",
                    value=None if field.get("default") is None else default_integer(field.get("default"), 0),
                    help=field.get("help") or None,
                    step=1,
                )
                value = None if value is None else int(value)
            elif kind == "select":
                options = parse_options(field.get("options"))
                default_value = str(field.get("default", "") or "")
                if not options:
                    options = [""]
                elif not default_value:
                    options = [""] + options
                default_index = options.index(default_value) if default_value in options else 0
                value = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f"{key_base}_select",
                    help=field.get("help") or None,
                )
            elif kind == "smiles":
                value = render_smiles_field(field, scope_key)
            else:
                value = st.text_input(
                    label,
                    key=f"{key_base}_text",
                    value="" if field.get("default") is None else str(field.get("default")),
                    placeholder=field.get("placeholder") or "",
                    help=field.get("help") or None,
                )

        if required and (kind in {"text", "select", "smiles"} and str(value).strip() == "" or kind in {"number", "integer"} and value is None):
            errors.append(f"{label} 不能为空")
        values[field["name"]] = value

    return pd.DataFrame([values]), errors


def reset_user_selection() -> None:
    st.session_state["predict_selected_material"] = ""
    st.session_state["predict_selected_target"] = ""


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

    action_col, status_col = st.columns([1.1, 2.4])
    with action_col:
        if st.button("打开 AI 输入助手", key="home_open_ai", type="primary", width="stretch"):
            if enabled_materials:
                st.session_state["predict_selected_material"] = enabled_materials[0][0]
                st.session_state["predict_selected_target"] = ""
                st.session_state["predict_open_ai_assistant"] = True
                st.rerun()
            else:
                st.warning("当前没有已开放的材料方向。")
    with status_col:
        st.caption("AI 未配置时仍可使用手动输入和批量上传；所有 AI 建议必须逐项确认后才会进入预测任务。")

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
                st.session_state["predict_open_ai_assistant"] = False
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
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-card-title">{model.get('label', '')}</div>
                <div class="model-card-meta">
                    <span>{model.get('model_name', '')}</span>
                    <span>{metric_text(model.get('metrics') or {})}</span>
                    <span>{state_text}</span>
                    <span>{len(model.get('feature_cols') or [])} 个特征</span>
                </div>
                <div class="model-card-notes">{model.get('notes', '') or '无备注'}</div>
            </div>
            """,
            unsafe_allow_html=True,
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
            st.session_state["predict_open_ai_assistant"] = False
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

    confirmed_by_user = st.checkbox(
        "我确认已检查输入结构、配方/工艺参数，并同意调用已发布模型进行预测",
        key=f"prediction_confirmation_{selected_material}_{selected_target}",
    )
    st.caption("门户预测不会自动补齐缺失特征；模型必须处于已启用、已发布且通过契约验证状态。")
    st.markdown(render_stage_timeline("validated", 0), unsafe_allow_html=True)

    open_ai = bool(st.session_state.get("predict_open_ai_assistant"))
    if open_ai:
        tab_ai, tab_manual, tab_batch, tab_config = st.tabs(["AI 辅助输入", "手动输入", "批量上传", "当前配置"])
    else:
        tab_manual, tab_batch, tab_ai, tab_config = st.tabs(["手动输入", "批量上传", "AI 辅助输入", "当前配置"])

    with tab_manual:
        st.markdown("### 手动输入参数")
        # Workflow source columns are collected here as explicit human inputs so
        # the submitted frame preserves the raw values required for replay.
        manual_df, validation_errors = render_parameter_inputs(manual_fields + workflow_fields, f"manual_{selected_material}_{selected_target}")
        if validation_errors:
            for err in validation_errors:
                st.warning(err)
        if st.button("开始手动预测", type="primary", key=f"predict_manual_{selected_material}_{selected_target}"):
            if validation_errors:
                st.error("请先补全必填项后再预测。")
            elif not confirmed_by_user:
                st.error("请先确认已检查输入结构和配方/工艺参数。")
            else:
                task_key = f"portal_task_manual_{selected_material}_{selected_target}"
                st.session_state[task_key] = submit_prediction_task(config, selected_material, selected_target, manual_df, explain=False)
                st.rerun()
        render_portal_task_panel(
            st.session_state.get(f"portal_task_manual_{selected_material}_{selected_target}"),
            session_key=f"manual_{selected_material}_{selected_target}",
        )

    with tab_batch:
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

    with tab_ai:
        render_ai_assistant_tab(config, selected_material, selected_target, target_cfg, selected_contract, registry_snapshot)

    with tab_config:
        st.markdown("### 当前性能项参数")
        param_df = parameter_editor_df(manual_fields + workflow_fields)
        st.dataframe(param_df, width="stretch", hide_index=True)

        st.markdown("### 当前已选模型")
        model_preview_rows = []
        for model in selected_models:
            model_preview_rows.append(
                {
                    "模型标签": model.get("label", ""),
                    "模型类型": model.get("model_name", ""),
                    "指标": metric_text(model.get("metrics") or {}),
                    "特征数": len(model.get("feature_cols") or []),
                    "特征来源": model.get("feature_source", ""),
                    "含分子特征流程": "是" if model.get("has_feature_process") else "否",
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
                for model in registered_models:
                    c1, c2, c3 = st.columns([6, 1, 1])
                    with c1:
                        st.markdown(
                            f"""
                            <div class="model-card">
                                <div class="model-card-title">{model.get('label', '')}</div>
                                <div class="model-card-meta">
                                    <span>{model.get('model_name', '')}</span>
                                    <span>{metric_text(model.get('metrics') or {})}</span>
                                    <span>{len(model.get('feature_cols') or [])} 个特征</span>
                                    <span>{'启用' if model.get('enabled', True) else '停用'}</span>
                                </div>
                                <div class="model-card-notes">{model.get('notes', '') or '无备注'}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c2:
                        toggle_label = "停用" if model.get("enabled", True) else "启用"
                        if st.button(toggle_label, key=f"toggle_model_{model['id']}"):
                            try:
                                toggle_model_enabled(config, selected_material, selected_target, model)
                            except ValueError as exc:
                                st.error(f"模型启用失败：{exc}")
                            else:
                                save_config(config)
                                st.rerun()
                    with c3:
                        if st.button("移除记录", key=f"remove_model_{model['id']}"):
                            target_cfg["models"] = [item for item in target_cfg.get("models", []) if item.get("id") != model["id"]]
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
        ai_open = bool(st.session_state.get("predict_open_ai_assistant"))

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
                st.session_state["predict_open_ai_assistant"] = False
                st.rerun()
            if st.button("打开 AI 输入助手", key="sidebar_ai", width="stretch", disabled=not selected_material):
                st.session_state["predict_open_ai_assistant"] = True
                st.rerun()
            st.caption("AI 状态：已打开" if ai_open else "AI 状态：可从工作台开启")
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
