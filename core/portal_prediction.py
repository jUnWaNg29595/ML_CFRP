"""Trusted, publication-gated prediction entry point for the user portal."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from .model_io import load_model_artifact_bytes
from .molecular_feature_workflow import execute_molecular_feature_workflow
from .prediction_contract import resolve_prediction_feature_contract
from .prediction_molecular_baseline import (
    collect_workflow_source_columns,
    validate_single_row_source_values,
)
from .prediction_portal import validate_publication_artifact

_CONFIRMATION_KEYS = ('confirmed_by_user', 'user_confirmed', 'confirmed')
_REQUEST_KEYS = {
    'material_type', 'material_key', 'target', 'target_key', 'inputs', 'input',
    'parameters', 'data', 'source_values', *_CONFIRMATION_KEYS,
}


@dataclass(frozen=True)
class PredictionResultSummary:
    prediction: Any
    unit: str
    warnings: list[str] = field(default_factory=list)
    model_version: str = ''
    feature_workflow_id: str = ''
    summary: dict[str, Any] = field(default_factory=dict)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or '').strip()


def _material_target(request: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _text(request.get('material_type') or request.get('material_key')),
        _text(request.get('target') or request.get('target_key')),
    )


def _confirmed(request: Mapping[str, Any]) -> bool:
    return any(request.get(key) is True for key in _CONFIRMATION_KEYS)


def _models_for(config: Mapping[str, Any], material: str, target: str) -> list[dict[str, Any]]:
    materials = _mapping(config.get('materials'))
    material_config = _mapping(materials.get(material))
    targets = _mapping(material_config.get('targets'))
    target_config = _mapping(targets.get(target))
    models = target_config.get('models')
    return [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []


def _contract(entry: Mapping[str, Any], artifact: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    value = entry.get('contract')
    if isinstance(value, Mapping):
        return value
    value = _mapping((artifact or {}).get('extra')).get('prediction_contract')
    return value if isinstance(value, Mapping) else {}


def _source_columns(contract: Mapping[str, Any], workflow: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    declared = contract.get('source_columns') or contract.get('workflow_source_columns')
    if isinstance(declared, list) and all(isinstance(item, Mapping) for item in declared):
        return [dict(item) for item in declared if _text(item.get('column'))]
    return collect_workflow_source_columns(workflow)


def _artifact_path(entry: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    raw = _text(entry.get('artifact_path'))
    if not raw:
        raise ValueError('已发布模型缺少 artifact_path。')
    path = Path(raw)
    if path.is_absolute():
        return path
    roots = [Path(str(config[key])) for key in ('project_root', 'portal_root', 'root') if config.get(key)]
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate
    return (roots[0] / path).resolve()


def _load_artifact(entry: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = _mapping(config or entry.get('_portal_config'))
    embedded = entry.get('_artifact')
    if isinstance(embedded, Mapping):
        return dict(embedded)
    path = _artifact_path(entry, config)
    if not path.is_file():
        raise FileNotFoundError(f'已发布模型文件不存在：{path}')
    artifact = load_model_artifact_bytes(path.read_bytes())
    if not isinstance(artifact, Mapping):
        raise ValueError('模型 artifact 必须是映射对象。')
    return dict(artifact)


def _validate_contract_shape(contract: Mapping[str, Any], artifact: Mapping[str, Any], entry: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    features = contract.get('feature_cols')
    if not isinstance(features, list) or not features or any(not _text(item) for item in features):
        errors.append('发布契约缺少精确 feature_cols。')
    target = _text(contract.get('target_col') or entry.get('target_col'))
    if not target:
        errors.append('发布契约缺少 target_col。')
    artifact_target = _text(artifact.get('target_col'))
    if artifact_target and target and artifact_target != target:
        errors.append('发布契约 target_col 与模型 artifact 不一致。')
    ranges = contract.get('numeric_ranges', {})
    if ranges is not None and not isinstance(ranges, Mapping):
        errors.append('发布契约 numeric_ranges 必须是映射。')
    return errors


def load_published_portal_model(config: Mapping[str, Any], material_type: str, target: str) -> dict[str, Any]:
    """Load exactly one enabled, published and validated release."""
    material, target_key = _text(material_type), _text(target)
    if not material or not target_key:
        raise ValueError('材料类型和预测目标不能为空。')
    models = _models_for(config, material, target_key)
    if not models:
        raise ValueError('当前材料和目标没有登记模型。')
    enabled = [item for item in models if item.get('enabled') is True and item.get('disabled') is not True]
    if len(enabled) > 1:
        raise ValueError('当前目标存在多个启用发布版本，发布状态存在歧义。')
    if not enabled:
        raise ValueError('当前材料和目标没有启用的已发布模型。')
    entry = enabled[0]
    if _text(entry.get('publication_status')).lower() != 'published':
        raise ValueError('当前模型不是已发布版本，不能用于门户预测。')
    if entry.get('needs_validation') is True or _text(entry.get('status')).lower() in {'needs_validation', 'ambiguous', 'disabled'}:
        raise ValueError('当前模型仍需验证，不能用于门户预测。')
    load_entry = dict(entry)
    load_entry['_portal_config'] = config
    try:
        artifact = _load_artifact(load_entry)
    except TypeError as exc:
        if 'positional argument' not in str(exc) and 'required positional' not in str(exc):
            raise
        artifact = _load_artifact(load_entry, config)
    contract = _contract(entry, artifact)
    errors = _validate_contract_shape(contract, artifact, entry)
    if errors:
        raise ValueError('发布模型契约无效：' + '；'.join(errors))
    report = validate_publication_artifact(artifact, contract)
    if not report.get('ok') or report.get('status') in {'needs_validation', 'invalid'}:
        details = '；'.join(str(item) for item in report.get('errors') or [])
        raise ValueError('发布模型未通过验证' + (f'：{details}' if details else '。'))
    return {'entry': dict(entry), 'artifact': artifact, 'contract': dict(contract)}


def _contains_forbidden_value(value: Any, *, key: str = '') -> bool:
    lowered = key.lower()
    if callable(value):
        return True
    if any(token in lowered for token in ('callable', 'function', 'lambda', 'source_code', 'shell', 'command', 'exec', 'eval', 'ai_feature', 'feature_vector')):
        return True
    if isinstance(value, Mapping):
        return any(_contains_forbidden_value(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_forbidden_value(item, key=key) for item in value)
    return False


def _input_payload(request: Mapping[str, Any]) -> Any:
    for key in ('inputs', 'input', 'parameters', 'data', 'source_values'):
        if key in request:
            return request.get(key)
    return None


def _as_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    elif isinstance(payload, Mapping):
        frame = pd.DataFrame([dict(payload)])
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        rows = list(payload)
        if not rows or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError('inputs 必须是对象或对象列表。')
        frame = pd.DataFrame([dict(row) for row in rows])
    else:
        raise ValueError('inputs 不能为空，且必须是对象、对象列表或 DataFrame。')
    if frame.empty:
        raise ValueError('inputs 不能为空。')
    if any(not _text(column) for column in frame.columns):
        raise ValueError('inputs 不能包含空列名。')
    return frame.reset_index(drop=True)


def _structure_error(value: Any) -> str | None:
    if not isinstance(value, str):
        return '必须是字符串。'
    value = value.strip()
    if not value:
        return '不能为空。'
    if len(value) > 20000:
        return '长度超过安全上限。'
    if any(ord(char) < 32 and char not in '\\t\\n\\r' for char in value):
        return '包含不可用控制字符。'
    if value.count('(') != value.count(')') or value.count('[') != value.count(']'):
        return '括号不匹配。'
    try:
        from .smiles_utils import diagnose_chemical_string

        diagnosis = diagnose_chemical_string(value)
    except Exception:
        diagnosis = None
    if not isinstance(diagnosis, Mapping) or diagnosis.get('status') not in {'ok', 'proxy_ok'}:
        return '不是可解析的 SMILES/BigSMILES。'
    if diagnosis.get('needs_semantic_review'):
        return 'SMILES/BigSMILES 需要人工语义复核。'
    return None


def _validate_structure_columns(frame: pd.DataFrame, source_columns: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for item in source_columns:
        column = _text(item.get('column'))
        if not column or column not in frame.columns:
            continue
        for index, value in frame[column].items():
            error = _structure_error(value)
            if error:
                errors.append(f'{column}[{index}] {error}')
    return errors


def _unknown_input_columns(frame: pd.DataFrame, contract: Mapping[str, Any], workflow: Mapping[str, Any] | None) -> list[str]:
    allowed = set(_text(item) for item in contract.get('feature_cols') or [])
    allowed.update(item['column'] for item in _source_columns(contract, workflow) if _text(item.get('column')))
    return [str(column) for column in frame.columns if str(column) not in allowed]


def _validate_numeric_frame(frame: pd.DataFrame, columns: Sequence[str], ranges: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        if column not in frame.columns:
            errors.append(f'缺少模型特征列：{column}')
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            errors.append(f'模型特征列 {column} 必须是有限数值。')
            continue
        bounds = ranges.get(column) if isinstance(ranges, Mapping) else None
        if isinstance(bounds, Mapping):
            try:
                minimum, maximum = float(bounds['min']), float(bounds['max'])
            except (KeyError, TypeError, ValueError):
                errors.append(f'模型特征列 {column} 的数值范围契约无效。')
                continue
            if ((values < minimum) | (values > maximum)).any():
                errors.append(f'模型特征列 {column} 超出发布训练范围 [{minimum}, {maximum}]。')
    return errors


def validate_prediction_request(request: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Return blocking diagnostics without executing a model."""
    if not isinstance(request, Mapping):
        return ['预测请求必须是对象。']
    errors: list[str] = []
    if _contains_forbidden_value(request):
        errors.append('请求不能包含 callable、源代码、Shell 命令或任意 AI 特征向量。')
    if not _confirmed(request):
        errors.append('必须先确认输入数据和预测用途，才能执行预测。')
    unknown = sorted(set(request) - _REQUEST_KEYS)
    if unknown:
        errors.append('请求包含未知字段：' + ', '.join(map(str, unknown)))
    material, target = _material_target(request)
    if not material:
        errors.append('缺少 material_type。')
    if not target:
        errors.append('缺少 target。')
    if errors:
        return errors
    try:
        bundle = load_published_portal_model(config, material, target)
    except Exception as exc:
        return errors + [str(exc)]
    contract = bundle['contract']
    workflow = _mapping(_mapping(bundle['artifact'].get('extra')).get('molecular_feature_workflow'))
    try:
        frame = _as_frame(_input_payload(request))
    except Exception as exc:
        return errors + [str(exc)]
    unknown_columns = _unknown_input_columns(frame, contract, workflow)
    if unknown_columns:
        errors.append('inputs 包含未知列：' + ', '.join(unknown_columns))
    sources = _source_columns(contract, workflow)
    if sources:
        source_names = [str(item.get('column')) for item in sources if _text(item.get('column'))]
        missing_columns = [column for column in source_names if column not in frame.columns]
        if missing_columns:
            errors.append('缺少分子源列：' + ', '.join(missing_columns))
        source_report = validate_single_row_source_values(frame.iloc[[0]], sources)
        if len(frame) > 1:
            source_report['empty_columns'] = [
                column
                for column in source_names
                if column in frame.columns and frame[column].map(
                    lambda value: value is None
                    or (isinstance(value, float) and pd.isna(value))
                    or (isinstance(value, str) and not value.strip())
                ).any()
            ]
        if source_report['empty_columns']:
            errors.append('分子源列不能为空：' + ', '.join(source_report['empty_columns']))
        errors.extend(_validate_structure_columns(frame, sources))
    else:
        errors.extend(_validate_numeric_frame(frame, contract.get('feature_cols') or [], contract.get('numeric_ranges') or {}))
    return errors


def _progress(progress: Callable[..., Any] | None, stage: str, **details: Any) -> None:
    if progress is None:
        return
    payload = {'stage': stage, **details}
    try:
        progress(payload)
    except TypeError:
        progress(stage, details)


def _safe_prediction(value: Any) -> Any:
    array = np.asarray(value).reshape(-1)
    result = []
    for item in array.tolist():
        try:
            number = float(item)
        except (TypeError, ValueError):
            result.append(item)
        else:
            result.append(number if math.isfinite(number) else None)
    return result[0] if len(result) == 1 else result


def run_confirmed_prediction(
    request: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    progress: Callable[..., Any] | None = None,
) -> PredictionResultSummary:
    """Validate and execute one published model release exactly once."""
    errors = validate_prediction_request(request, config)
    if errors:
        raise ValueError('预测请求未通过校验：' + '；'.join(errors))
    material, target = _material_target(request)
    bundle = load_published_portal_model(config, material, target)
    entry, artifact, contract = bundle['entry'], bundle['artifact'], bundle['contract']
    frame = _as_frame(_input_payload(request))
    extra = _mapping(artifact.get('extra'))
    workflow = _mapping(extra.get('molecular_feature_workflow'))
    warnings: list[str] = []
    _progress(progress, 'validated', rows=len(frame), model_version=_text(entry.get('version')))

    if workflow:
        _progress(progress, 'workflow', rows=len(frame))
        workflow_result = execute_molecular_feature_workflow(
            frame,
            workflow,
            mode='portal',
            progress_callback=lambda trace: _progress(progress, 'workflow_step', **trace),
        )
        features = workflow_result.features.reindex(columns=list(contract['feature_cols']))
        warnings.extend(str(item) for item in workflow_result.warnings)
        workflow_id = _text(entry.get('feature_workflow_id') or workflow.get('workflow_hash'))
    else:
        features = frame.reindex(columns=list(contract['feature_cols']))
        workflow_id = _text(entry.get('feature_workflow_id') or contract.get('workflow_hash'))

    feature_errors = _validate_numeric_frame(
        features,
        contract['feature_cols'],
        contract.get('numeric_ranges') or {},
    )
    if feature_errors:
        raise ValueError('预测特征未通过发布契约校验：' + '；'.join(feature_errors))

    resolution = resolve_prediction_feature_contract(model=artifact.get('model'), pipeline=artifact.get('pipeline'), artifact=artifact)
    if not resolution.get('ok') or list(resolution.get('feature_cols') or []) != list(contract['feature_cols']):
        raise ValueError('发布模型的实际特征输入与 prediction_contract 不一致。')
    pipeline = artifact.get('pipeline')
    model = artifact.get('model') or pipeline
    if model is None or not callable(getattr(model, 'predict', None)):
        raise ValueError('发布模型缺少可执行 predict 接口。')
    _progress(progress, 'predicting', rows=len(features))
    if pipeline is not None:
        predicted = pipeline.predict(features)
    else:
        values = features.to_numpy()
        imputer, scaler = artifact.get('imputer'), artifact.get('scaler')
        if imputer is not None:
            values = imputer.transform(values)
        if scaler is not None:
            values = scaler.transform(values)
        predicted = model.predict(values)
    prediction = _safe_prediction(predicted)
    _progress(progress, 'completed', rows=len(features))
    summary = {
        'material_type': material,
        'target': target,
        'rows': int(len(features)),
        'feature_count': int(len(contract['feature_cols'])),
        'model_label': _text(entry.get('label')),
    }
    return PredictionResultSummary(
        prediction=prediction,
        unit=_text(entry.get('unit') or contract.get('unit')),
        warnings=warnings,
        model_version=_text(entry.get('version') or entry.get('id')),
        feature_workflow_id=workflow_id,
        summary=summary,
    )
