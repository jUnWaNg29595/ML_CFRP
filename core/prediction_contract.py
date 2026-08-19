"""Prediction-time feature contract resolution.

The prediction page must never infer a model input contract from column count
alone.  This module keeps the resolution logic independent from Streamlit so
it can be tested with small model/artifact doubles.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _as_columns(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Iterable):
        return []

    result: list[str] = []
    seen: set[str] = set()
    normalized_seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized = "".join(text.split()).lower()
        if text in seen or normalized in normalized_seen:
            continue
        seen.add(text)
        normalized_seen.add(normalized)
        result.append(text)
    return result


def _expected_count(model: Any = None, pipeline: Any = None) -> int | None:
    for obj in (pipeline, model):
        if obj is None:
            continue
        try:
            value = getattr(obj, "n_features_in_", None)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _model_feature_names(model: Any = None, pipeline: Any = None) -> list[str]:
    for obj in (pipeline, model):
        if obj is None:
            continue
        try:
            names = getattr(obj, "feature_names_in_", None)
        except Exception:
            names = None
        columns = _as_columns(names)
        if columns:
            return columns

    if model is not None and hasattr(model, "get_booster"):
        try:
            return _as_columns(model.get_booster().feature_names)
        except Exception:
            pass
    return []


def _feature_mask(artifact: Mapping[str, Any]) -> np.ndarray | None:
    extra = artifact.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    candidates = (
        artifact.get("feature_mask"),
        extra.get("feature_mask"),
    )
    pipeline = artifact.get("pipeline")
    if pipeline is not None:
        try:
            for _, step in getattr(pipeline, "steps", []):
                if getattr(step, "feature_mask", None) is not None:
                    candidates += (getattr(step, "feature_mask"),)
        except Exception:
            pass
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return np.asarray(candidate, dtype=bool).ravel()
        except Exception:
            continue
    return None


def _source_candidates(
    artifact: Mapping[str, Any],
    *,
    session_feature_cols: Any = None,
    train_result: Mapping[str, Any] | None = None,
) -> list[tuple[str, list[str]]]:
    extra = artifact.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    workflow = extra.get("molecular_feature_workflow")
    workflow = workflow if isinstance(workflow, Mapping) else {}

    candidates = [
        ("artifact.extra.effective_feature_cols", _as_columns(extra.get("effective_feature_cols"))),
        ("artifact.extra.workflow.final_feature_names", _as_columns(workflow.get("final_feature_names"))),
        ("artifact.feature_cols", _as_columns(artifact.get("feature_cols"))),
    ]
    if isinstance(train_result, Mapping):
        for key in ("X_train_raw", "X_train", "X_test_raw", "X_test"):
            frame = train_result.get(key)
            columns = _as_columns(getattr(frame, "columns", None))
            if columns:
                candidates.append((f"train_result.{key}", columns))
    candidates.append(("session.feature_cols", _as_columns(session_feature_cols)))
    return [(source, columns) for source, columns in candidates if columns]


def _apply_mask(columns: list[str], mask: np.ndarray | None) -> tuple[list[str], bool]:
    if mask is None or len(mask) != len(columns):
        return columns, False
    return [column for column, keep in zip(columns, mask) if bool(keep)], True


def _normalized_set(columns: list[str]) -> set[str]:
    return {"".join(column.split()).lower() for column in columns}


def resolve_prediction_feature_contract(
    *,
    model: Any = None,
    pipeline: Any = None,
    artifact: Mapping[str, Any] | None = None,
    session_feature_cols: Any = None,
    train_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and validate the exact model input contract.

    A model-provided feature-name sequence is authoritative.  If the model
    exposes only ``n_features_in_``, a saved feature mask may reduce a wider
    source list.  Otherwise a wider or ambiguous list is rejected instead of
    silently truncating it.
    """

    artifact = artifact if isinstance(artifact, Mapping) else {}
    expected = _expected_count(model=model, pipeline=pipeline)
    model_columns = _model_feature_names(model=model, pipeline=pipeline)
    mask = _feature_mask(artifact)
    candidates = _source_candidates(
        artifact,
        session_feature_cols=session_feature_cols,
        train_result=train_result,
    )

    selected: list[str] = []
    source: str | None = None
    masked_source = False
    errors: list[str] = []

    if model_columns:
        selected = model_columns
        source = "model.feature_names_in_"
        if expected is not None and len(selected) != expected:
            errors.append(
                f"模型公开了 {len(selected)} 个特征名，但 n_features_in_ 为 {expected}。"
            )
    else:
        for candidate_source, columns in candidates:
            masked, used_mask = _apply_mask(columns, mask)
            if expected is not None and len(masked) == expected:
                selected = masked
                source = candidate_source + ("+feature_mask" if used_mask else "")
                masked_source = used_mask
                break
        if not selected:
            if expected is None:
                for candidate_source, columns in candidates:
                    selected = columns
                    source = candidate_source
                    break

    if not selected:
        errors.append("未找到可用于预测的模型特征列。")
    if expected is not None and len(selected) != expected:
        errors.append(
            f"模型要求 {expected} 个特征，但当前只能解析到 {len(selected)} 个。"
        )

    extra_features: list[str] = []
    missing_features: list[str] = []
    order_mismatch = False
    artifact_columns = _as_columns(artifact.get("feature_cols"))
    if model_columns and artifact_columns:
        model_norm = _normalized_set(model_columns)
        artifact_norm = _normalized_set(artifact_columns)
        extra_features = [
            column for column in artifact_columns
            if "".join(column.split()).lower() not in model_norm
        ]
        missing_features = [
            column for column in model_columns
            if "".join(column.split()).lower() not in artifact_norm
        ]
        common_artifact = [
            column for column in artifact_columns
            if "".join(column.split()).lower() in model_norm
        ]
        common_model = [
            column for column in model_columns
            if "".join(column.split()).lower() in artifact_norm
        ]
        order_mismatch = common_artifact != common_model
        if missing_features:
            errors.append(
                "保存的模型特征清单缺少模型公开的特征列："
                + ", ".join(missing_features[:12])
            )
    elif selected and artifact_columns:
        if len(artifact_columns) > len(selected):
            extra_features = artifact_columns[len(selected):]
        order_mismatch = artifact_columns[: len(selected)] != selected

    if extra_features and not model_columns and not masked_source:
        errors.append(
            "候选特征列多于模型输入，且没有模型列名或有效 feature_mask，无法安全判断应删除哪些列。"
        )

    return {
        "ok": not errors,
        "feature_cols": selected,
        "expected_count": expected,
        "source": source,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "duplicate_features": [],
        "order_mismatch": order_mismatch,
        "errors": errors,
    }
