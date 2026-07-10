# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.model_io import load_model_artifact_bytes


APP_NAME = "CFRP 预测应用平台"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = PROJECT_ROOT / "prediction_portal"
MODEL_ROOT = PLATFORM_ROOT / "managed_models"
CONFIG_PATH = PLATFORM_ROOT / "prediction_config.json"
ASSET_ROOT = PLATFORM_ROOT / "assets"

DATA_UPLOAD_TYPES = ["csv", "xlsx", "xls"]
SMILES_UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "heif", "heic", "pdf"]
PARAMETER_KINDS = ["number", "integer", "text", "select", "smiles"]


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
        "default": 0.0,
        "placeholder": "",
        "help": "由模型特征列自动生成，可在管理页面调整",
        "options": [],
    }


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
        "enabled": True if existing_entry is None else bool(existing_entry.get("enabled", True)),
        "notes": notes,
        "model_name": artifact.get("model_name") or "ImportedModel",
        "target_col": artifact.get("target_col") or target_key,
        "feature_cols": list(feature_cols),
        "feature_source": "override" if feature_override else ("artifact" if artifact.get("feature_cols") else "input_order"),
        "artifact_path": saved_path.relative_to(PROJECT_ROOT).as_posix(),
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


@st.cache_resource(show_spinner=False)
def load_artifact_from_disk(path_str: str, mtime: float) -> Dict[str, Any]:
    del mtime
    return load_model_artifact_bytes(Path(path_str).read_bytes())


def align_prediction_frame(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if not feature_cols:
        return df.copy(), []

    aligned = df.copy()
    missing = [col for col in feature_cols if col not in aligned.columns]
    for col in missing:
        col_lower = col.lower()
        if any(token in col_lower for token in ["morgan", "maccs", "fingerprint", "fp_", "_fp"]):
            aligned[col] = 0
        else:
            aligned[col] = np.nan
    return aligned[feature_cols], missing


def run_prediction(model_entry: Dict[str, Any], input_df: pd.DataFrame) -> Dict[str, Any]:
    artifact_path = resolve_model_path(model_entry)
    if not artifact_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {artifact_path}")

    artifact = load_artifact_from_disk(str(artifact_path), artifact_path.stat().st_mtime)
    pipeline = artifact.get("pipeline")
    model = artifact.get("model") or pipeline
    if model is None:
        raise RuntimeError("模型文件中未检测到可用模型对象")

    feature_cols = list(model_entry.get("feature_cols") or artifact.get("feature_cols") or input_df.columns.tolist())
    prepared_df, auto_filled = align_prediction_frame(input_df, feature_cols)

    if pipeline is not None:
        preds = pipeline.predict(prepared_df)
    else:
        values = prepared_df.values
        imputer = artifact.get("imputer")
        scaler = artifact.get("scaler")
        if imputer is not None:
            values = imputer.transform(values)
        if scaler is not None:
            values = scaler.transform(values)
        preds = model.predict(values)

    predictions = np.asarray(preds).reshape(-1)
    return {
        "predictions": predictions,
        "feature_cols": feature_cols,
        "auto_filled": auto_filled,
        "artifact": artifact,
    }


def execute_predictions(selected_models: List[Dict[str, Any]], input_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    result_df = input_df.copy()
    infos: List[str] = []
    errors: List[str] = []
    prediction_cols: List[str] = []

    for model_entry in selected_models:
        label = model_entry.get("label") or model_entry.get("model_name") or "model"
        try:
            result = run_prediction(model_entry, input_df)
            base_col = f"prediction__{label}"
            pred_col = base_col
            counter = 2
            while pred_col in result_df.columns:
                pred_col = f"{base_col}_{counter}"
                counter += 1

            result_df[pred_col] = result["predictions"]
            prediction_cols.append(pred_col)

            info_text = f"{label}: 使用 {len(result['feature_cols'])} 个特征完成预测"
            if result["auto_filled"]:
                info_text += f"，自动补齐 {len(result['auto_filled'])} 个缺失列"
            infos.append(info_text)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if len(prediction_cols) > 1:
        result_df["prediction_mean"] = result_df[prediction_cols].mean(axis=1)

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


def material_card_html(material_key: str, material_cfg: Dict[str, Any]) -> str:
    state_text = "已开放" if material_cfg.get("enabled") else "暂未开放"
    state_class = "status-open" if material_cfg.get("enabled") else "status-closed"
    description = material_cfg.get("description") or ""
    image_uri = portal_asset(material_image_name(material_key))
    image_html = f'<div class="portal-card-image" style="background-image: url({image_uri});"></div>' if image_uri else ""
    return f"""
    <div class="portal-card">
        {image_html}
        <div class="portal-card-top">
            <div class="portal-title">{material_cfg.get('label', '')}</div>
            <div class="status-pill {state_class}">{state_text}</div>
        </div>
        <div class="portal-desc">{description}</div>
    </div>
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
        st.dataframe(pd.DataFrame(st.session_state[result_key]), use_container_width=True, hide_index=True)

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
                    value=default_number(field.get("default"), 0.0),
                    help=field.get("help") or None,
                    format="%.6f",
                )
            elif kind == "integer":
                value = st.number_input(
                    label,
                    key=f"{key_base}_integer",
                    value=default_integer(field.get("default"), 0),
                    help=field.get("help") or None,
                    step=1,
                )
                value = int(value)
            elif kind == "select":
                options = parse_options(field.get("options"))
                if not options:
                    options = [""]
                default_value = str(field.get("default", "") or "")
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

        if required and kind in {"text", "select", "smiles"} and str(value).strip() == "":
            errors.append(f"{label} 不能为空")
        values[field["name"]] = value

    return pd.DataFrame([values]), errors


def reset_user_selection() -> None:
    st.session_state["predict_selected_material"] = ""
    st.session_state["predict_selected_target"] = ""


def render_user_home(config: Dict[str, Any]) -> None:
    hero_uri = portal_asset("portal-hero.png")
    st.markdown(
        f"""
        <div class="portal-hero visual" style="background-image: linear-gradient(90deg, rgba(248,250,252,0.96) 0%, rgba(248,250,252,0.84) 48%, rgba(248,250,252,0.22) 100%), url({hero_uri});">
            <div class="hero-kicker">Prediction Workspace</div>
            <h1>材料预测入口</h1>
            <p>先选择预测方向，再进入对应的性能预测流程。环氧树脂和单向碳纤维复合材料入口均已开放。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    materials = material_items(config)
    cols = st.columns(len(materials))
    for col, (material_key, material_cfg) in zip(cols, materials):
        with col:
            st.markdown(material_card_html(material_key, material_cfg), unsafe_allow_html=True)
            button_text = "进入该方向" if material_cfg.get("enabled") else "暂未开放"
            if st.button(button_text, key=f"open_{material_key}", disabled=not material_cfg.get("enabled"), use_container_width=True):
                st.session_state["predict_selected_material"] = material_key
                st.session_state["predict_selected_target"] = ""
                st.rerun()


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

    st.dataframe(result_df, use_container_width=True)
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

    image_uri = portal_asset(material_image_name(selected_material))
    top1, top2 = st.columns([6, 1.4])
    with top1:
        st.markdown(
            f"""
            <div class="portal-hero compact visual detail" style="background-image: linear-gradient(90deg, rgba(248,250,252,0.95) 0%, rgba(248,250,252,0.86) 54%, rgba(248,250,252,0.28) 100%), url({image_uri});">
                <div class="hero-kicker">User Page</div>
                <h1>{material_cfg.get('label', '')}</h1>
                <p>{material_cfg.get('description', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.write("")
        if st.button("返回方向选择", use_container_width=True):
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

    enabled_models = [model for model in model_items(target_cfg) if model.get("enabled", True)]
    render_model_brief(enabled_models)
    if not enabled_models:
        return

    default_model_ids = [enabled_models[0]["id"]]
    selected_model_ids = st.multiselect(
        "选择用于预测的模型",
        options=[model["id"] for model in enabled_models],
        default=default_model_ids,
        format_func=lambda model_id: next(
            (
                f"{model.get('label', '')} | {metric_text(model.get('metrics') or {})}"
                for model in enabled_models
                if model["id"] == model_id
            ),
            model_id,
        ),
    )
    selected_models = [model for model in enabled_models if model["id"] in selected_model_ids]
    if not selected_models:
        st.warning("请至少选择一个模型。")
        return

    if any(model.get("has_feature_process") for model in selected_models):
        st.warning("已检测到部分模型带有分子特征流程配置。当前版本先完成平台框架和 SMILES 录入/识别，尚未自动按训练流程生成分子特征。")

    tab_manual, tab_batch, tab_config = st.tabs(["手动输入", "批量上传", "当前配置"])

    with tab_manual:
        st.markdown("### 手动输入参数")
        manual_df, validation_errors = render_parameter_inputs(target_cfg.get("parameters") or [], f"manual_{selected_material}_{selected_target}")
        if validation_errors:
            for err in validation_errors:
                st.warning(err)
        if st.button("开始手动预测", type="primary", key=f"predict_manual_{selected_material}_{selected_target}"):
            if validation_errors:
                st.error("请先补全必填项后再预测。")
            else:
                result_df, infos, errors = execute_predictions(selected_models, manual_df)
                render_prediction_results(result_df, infos, errors, f"{selected_target}_manual_predictions.csv")

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
                st.dataframe(batch_df.head(20), use_container_width=True)
                if st.button("执行批量预测", type="primary", key=f"predict_batch_{selected_material}_{selected_target}"):
                    result_df, infos, errors = execute_predictions(selected_models, batch_df)
                    render_prediction_results(result_df, infos, errors, f"{selected_target}_batch_predictions.csv")
            except Exception as exc:
                st.error(f"读取文件失败: {exc}")
        else:
            st.info("适合已经准备好批量输入表格的场景。若模型缺少部分特征列，系统会按规则自动补齐。")

    with tab_config:
        st.markdown("### 当前性能项参数")
        param_df = parameter_editor_df(target_cfg.get("parameters") or [])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(model_preview_rows), use_container_width=True, hide_index=True)


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
                use_container_width=True,
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
                            model["enabled"] = not bool(model.get("enabled", True))
                            model["updated_at"] = now_iso()
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
    --portal-ink: #1f2937;
    --portal-muted: #6b7280;
    --portal-line: #e5e7eb;
    --portal-accent: #0f766e;
    --portal-warm: #fff8ef;
    --portal-closed: #b91c1c;
    --portal-open: #047857;
}
.main .block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.portal-hero {
    background:
        radial-gradient(circle at top right, rgba(20, 184, 166, 0.16), transparent 32%),
        linear-gradient(135deg, #fff7ed 0%, #f8fafc 100%);
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    overflow: hidden;
}
.portal-hero.compact {
    margin-bottom: 0;
}
.portal-hero.visual {
    min-height: 230px;
    background-size: cover;
    background-position: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-color: #cbd5e1;
}
.portal-hero.visual.detail {
    min-height: 160px;
    background-position: center right;
}
.portal-hero h1 {
    margin: 0.15rem 0 0.35rem 0;
    font-size: 2rem;
    color: var(--portal-ink);
}
.portal-hero p {
    margin: 0;
    color: var(--portal-muted);
    line-height: 1.6;
    max-width: 42rem;
}
.hero-kicker {
    display: inline-block;
    background: rgba(15, 118, 110, 0.12);
    color: var(--portal-accent);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.portal-card, .model-card {
    border: 1px solid var(--portal-line);
    background: white;
    border-radius: 8px;
    padding: 0.85rem 0.9rem 0.9rem 0.9rem;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
    margin-bottom: 0.8rem;
    overflow: hidden;
}
.portal-card {
    padding-top: 0;
    transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
}
.portal-card:hover {
    border-color: rgba(15, 118, 110, 0.36);
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
}
.portal-card-image {
    height: 168px;
    margin: 0 -0.9rem 0.85rem -0.9rem;
    background-size: cover;
    background-position: center;
    border-bottom: 1px solid var(--portal-line);
}
.portal-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.65rem;
}
.portal-title, .model-card-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--portal-ink);
}
.portal-desc, .model-card-notes {
    color: var(--portal-muted);
    line-height: 1.6;
    font-size: 0.94rem;
}
.status-pill {
    border-radius: 999px;
    padding: 0.24rem 0.62rem;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
}
.status-open {
    color: var(--portal-open);
    background: rgba(16, 185, 129, 0.12);
}
.status-closed {
    color: var(--portal-closed);
    background: rgba(239, 68, 68, 0.10);
}
.model-card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
}
.model-card-meta span {
    background: var(--portal-warm);
    border: 1px solid #fde68a;
    border-radius: 999px;
    padding: 0.16rem 0.52rem;
    font-size: 0.77rem;
    color: #92400e;
}
</style>
"""


def render_sidebar(config: Dict[str, Any]) -> str:
    with st.sidebar:
        st.title(APP_NAME)
        st.caption(f"版本 {VERSION}")
        st.markdown("---")

        mode = st.radio("工作区", ["用户页面", "管理页面"], index=0)
        material_count = len(config.get("materials") or {})
        target_count = sum(len(material.get("targets") or {}) for _, material in material_items(config))
        model_count = sum(
            len(target.get("models") or [])
            for _, material in material_items(config)
            for _, target in target_items(material)
        )

        st.markdown("### 当前配置")
        st.metric("材料方向", material_count)
        st.metric("性能项", target_count)
        st.metric("模型数量", model_count)
        st.caption(f"配置文件: {CONFIG_PATH.name}")

        st.markdown("---")
        st.caption("管理页面负责配置入口、参数和模型；用户页面负责选方向、传数据和执行预测。")

    return mode


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = load_config()
    mode = render_sidebar(config)

    if mode == "管理页面":
        render_admin_page(config)
    else:
        render_user_page(config)


if __name__ == "__main__":
    main()
