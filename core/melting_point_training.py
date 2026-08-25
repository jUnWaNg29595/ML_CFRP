"""Helpers for preparing PubChem melting-point records for model training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_GENERIC_SOURCE_TOKENS = (
    'smiles',
    'bigsmiles',
    'molecule',
    'structure',
    '分子',
    '结构',
)
_RESIN_TOKENS = ('resin', 'epoxy', '树脂', '基体')
_HARDENER_TOKENS = (
    'hardener',
    'curing_agent',
    'curingagent',
    'curative',
    'curing',
    '固化剂',
    '交联剂',
)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = []
    result = []
    for item in values:
        text = str(item or '').strip()
        if text and text not in result:
            result.append(text)
    return result


def _workflow_dict(workflow: Any) -> dict[str, Any]:
    if workflow is None:
        return {}
    if isinstance(workflow, Mapping):
        return dict(workflow)
    to_dict = getattr(workflow, 'to_dict', None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def collect_workflow_source_columns(workflow: Any) -> list[str]:
    payload = _workflow_dict(workflow)
    columns: list[str] = []
    for step in payload.get('steps') or []:
        if isinstance(step, Mapping):
            columns.extend(_as_list(step.get('source_columns')))
    input_contract = payload.get('input_contract')
    if isinstance(input_contract, Mapping):
        for key in (
            'selected_source_columns',
            'source_columns',
            'resin_component_cols',
            'hardener_component_cols',
            'smiles_col',
            'hardener_col',
        ):
            columns.extend(_as_list(input_contract.get(key)))
    for key in (
        'selected_source_columns',
        'source_columns',
        'resin_component_cols',
        'hardener_component_cols',
        'smiles_col',
        'hardener_col',
    ):
        columns.extend(_as_list(payload.get(key)))
    return list(dict.fromkeys(columns))


def classify_source_column_role(column: Any) -> str:
    text = str(column or '').strip().lower()
    resin_score = sum(text.count(token) for token in _RESIN_TOKENS)
    hardener_score = sum(text.count(token) for token in _HARDENER_TOKENS)
    if resin_score > hardener_score and resin_score > 0:
        return 'resin'
    if hardener_score > resin_score and hardener_score > 0:
        return 'hardener'
    return 'neutral'


def prepare_melting_point_source_frame(
    dataset: pd.DataFrame,
    workflow: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map canonical PubChem ``smiles`` into a compatible single-component workflow.

    Melting point records represent one molecule per row.  Numbered source columns
    such as ``resin_smiles_1`` are therefore materialized from that same molecule,
    but only when the workflow describes one role (or neutral source columns).
    A workflow that mixes resin and hardener source branches is rejected rather
    than silently training on a semantically incorrect input layout.
    """
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError('熔点数据集为空，无法准备分子特征。')
    if 'smiles' not in dataset.columns:
        raise ValueError('熔点数据集缺少 smiles 列。')
    out = dataset.reset_index(drop=True).copy()
    out['smiles'] = out['smiles'].fillna('').astype(str)
    out['bigsmiles'] = out['smiles']

    source_columns = collect_workflow_source_columns(workflow)
    if not source_columns:
        raise ValueError('熔点模型 workflow 未声明 SMILES/BigSMILES 源列。')
    roles = {
        classify_source_column_role(column)
        for column in source_columns
        if classify_source_column_role(column) != 'neutral'
    }
    if len(roles) > 1:
        raise ValueError(
            '当前 workflow 同时包含树脂和固化剂源列；熔点训练是单分子任务，'
            '请在分子特征复现页选择并锁定单一组分 workflow。'
        )

    mapped_columns = []
    for column in source_columns:
        if column in out.columns:
            continue
        lowered = column.lower()
        if any(token in lowered for token in _GENERIC_SOURCE_TOKENS) or len(roles) == 1:
            out[column] = out['smiles']
            mapped_columns.append(column)

    missing = [column for column in source_columns if column not in out.columns]
    if missing:
        raise ValueError(
            '熔点训练 workflow 的源列无法从单分子 smiles 映射：'
            + '、'.join(missing[:12])
        )
    return out, {
        'source_columns': source_columns,
        'mapped_columns': mapped_columns,
        'source_role': next(iter(roles), 'neutral'),
        'row_count': int(len(out)),
    }


def _regression_metrics(y_true: Any, y_pred: Any, min_r2_samples: int = 2) -> dict[str, Any]:
    """Return JSON-safe regression metrics for finite paired values."""
    true_values = pd.to_numeric(pd.Series(y_true), errors='coerce').to_numpy(dtype=float)
    pred_values = pd.to_numeric(pd.Series(y_pred), errors='coerce').to_numpy(dtype=float)
    size = min(true_values.size, pred_values.size)
    if size == 0:
        return {
            'n': 0,
            'mae': None,
            'rmse': None,
            'r2': None,
            'r2_status': 'insufficient_samples',
        }
    true_values = true_values[:size]
    pred_values = pred_values[:size]
    mask = np.isfinite(true_values) & np.isfinite(pred_values)
    true_values = true_values[mask]
    pred_values = pred_values[mask]
    n = int(true_values.size)
    if n == 0:
        return {
            'n': 0,
            'mae': None,
            'rmse': None,
            'r2': None,
            'r2_status': 'insufficient_samples',
        }
    residual = true_values - pred_values
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    r2 = None
    r2_status = 'ok'
    if n < int(min_r2_samples):
        r2_status = 'insufficient_samples'
    else:
        total_sum_squares = float(np.sum(np.square(true_values - np.mean(true_values))))
        if total_sum_squares <= np.finfo(float).eps:
            r2_status = 'constant_target'
        else:
            r2 = float(1.0 - np.sum(np.square(residual)) / total_sum_squares)
    return {
        'n': n,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'r2_status': r2_status,
    }


def build_melting_point_split_metrics(
    dataset: pd.DataFrame,
    *,
    train_indices: Any = None,
    test_indices: Any = None,
    y_train: Any = None,
    y_pred_train: Any = None,
    y_test: Any = None,
    y_pred_test: Any = None,
    min_r2_samples: int = 2,
) -> dict[str, Any]:
    """Build auditable overall and role/class metrics for an MP split.

    ``train_indices`` and ``test_indices`` are positional indices from the
    model trainer.  The function deliberately keeps missing or tiny subsets in
    the output with ``None`` metrics instead of silently dropping them.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError('dataset must be a pandas DataFrame')
    frame = dataset.reset_index(drop=True)

    def _positions(values: Any) -> np.ndarray:
        if values is None:
            return np.asarray([], dtype=int)
        try:
            arr = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(dtype=int)
        except Exception:
            return np.asarray([], dtype=int)
        return arr[(arr >= 0) & (arr < len(frame))]

    def _subset(split_name: str, indices: Any, true_values: Any, pred_values: Any) -> dict[str, Any]:
        positions = _positions(indices)
        true_array = np.asarray(true_values if true_values is not None else [], dtype=object).ravel()
        pred_array = np.asarray(pred_values if pred_values is not None else [], dtype=object).ravel()
        size = min(len(positions), len(true_array), len(pred_array))
        positions = positions[:size]
        result: dict[str, Any] = {
            'split': split_name,
            'metrics': _regression_metrics(true_array[:size], pred_array[:size], min_r2_samples),
        }
        if size == 0:
            result['positions'] = []
            result['roles'] = {}
            result['hardener_classes'] = {}
            return result
        metadata = frame.iloc[positions]
        roles = metadata.get('component_role', pd.Series('unknown', index=metadata.index))
        roles = roles.fillna('unknown').astype(str).str.strip().str.lower().replace('', 'unknown')
        classes = metadata.get('hardener_class', pd.Series('', index=metadata.index))
        classes = classes.fillna('').astype(str).str.strip()

        by_role: dict[str, Any] = {}
        for role in sorted(roles.unique().tolist()):
            mask = roles.to_numpy() == role
            by_role[str(role)] = _regression_metrics(
                true_array[:size][mask], pred_array[:size][mask], min_r2_samples
            )
        by_class: dict[str, Any] = {}
        for hardener_class in sorted(value for value in classes.unique().tolist() if value):
            mask = classes.to_numpy() == hardener_class
            by_class[str(hardener_class)] = _regression_metrics(
                true_array[:size][mask], pred_array[:size][mask], min_r2_samples
            )
        result['positions'] = [int(value) for value in positions.tolist()]
        result['roles'] = by_role
        result['hardener_classes'] = by_class
        return result

    train = _subset('train', train_indices, y_train, y_pred_train)
    test = _subset('test', test_indices, y_test, y_pred_test)
    return {
        'target': 'mp_c',
        'target_unit': 'C',
        'dataset_rows': int(len(frame)),
        'train': train,
        'test': test,
        'quality_policy': 'recorded_in_dataset',
        'role_column': 'component_role' if 'component_role' in frame.columns else None,
        'hardener_class_column': 'hardener_class' if 'hardener_class' in frame.columns else None,
    }


__all__ = [
    'build_melting_point_split_metrics',
    'collect_workflow_source_columns',
    'prepare_melting_point_source_frame',
]
