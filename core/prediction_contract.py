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
    audit = extra.get("feature_audit")
    audit = audit if isinstance(audit, Mapping) else {}

    candidates = [
        # canonical 是 feature_mask **之前**的完整列清单（如 2070 列），
        # 当 pipeline 第一层按 mask 前列数 fit 时必须优先用它。
        ("artifact.extra.feature_audit.canonical_feature_cols", _as_columns(audit.get("canonical_feature_cols"))),
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


def _widen_columns_to_expected(
    *,
    expected: int,
    model_columns: list[str],
    mask: np.ndarray | None,
    candidates: list[tuple[str, list[str]]],
    audit: Mapping[str, Any] | None = None,
) -> list[str] | None:
    """用 feature_mask 把模型公开的（mask 后）列名还原为 pipeline 期望的（mask 前）列清单。

    场景：``Pipeline(imputer(N) → feature_mask(N→M) → scaler(M) → model(M))``。
    模型只暴露 mask 后的 M 个列名，而 pipeline 第一层按 N 列 fit。

    还原条件（全部满足才返回，否则返回 None 由调用方报错）：
    1. ``mask`` 存在且长度为 ``expected``；
    2. ``mask`` 中 True 的数量等于 ``len(model_columns)``；
    3. 存在一个候选列清单，其长度 == ``expected``，且用 mask 过滤后
       **恰好等于** ``model_columns``（逐列比对，确保不是巧合）。

    额外处理：某些 artifact 的 ``canonical_feature_cols`` 会比实际多记录 1 列
    （重复列/常量列处理差异），而 mask 是按真实列数 fit 的。此时用
    ``feature_audit.removed_feature_cols`` 逐个试删一列，找到能让
    ``canonical − drop`` 重建出的 mask 与真实 mask 完全一致的那一列。
    （实测：TabPFN artifact 的 canonical 为 2071，而 pipeline 期望 2070，
    冗余列为 ``resin_3_molecular_weight_g_mol``。）

    返回：mask 之前的完整列清单（长度 == expected）；不满足则 None。
    """
    if mask is None or len(mask) != expected:
        return None
    if int(np.count_nonzero(mask)) != len(model_columns):
        return None

    model_norm = _normalized_set(model_columns)
    mask_list = [bool(keep) for keep in mask]
    for _source, columns in candidates:
        if len(columns) == expected:
            kept = [column for column, keep in zip(columns, mask_list) if keep]
            if _normalized_set(kept) == model_norm:
                return list(columns)
            continue
        # 候选列数偏多：尝试按 removed_feature_cols 删掉冗余列后重新校验
        if len(columns) <= expected:
            continue
        repaired = _repair_columns_to_length(columns, expected, audit, mask_list, model_norm)
        if repaired is not None:
            return repaired
    return None


def _repair_columns_to_length(
    columns: list[str],
    expected: int,
    audit: Mapping[str, Any] | None,
    mask_list: list[bool],
    model_norm: set[str],
) -> list[str] | None:
    """逐个试删 ``removed_feature_cols`` 中的一列，使重建的 mask 与真实 mask 一致。

    与 ``core/external_feature_augmenter._repair_columns_to_length`` 同思路：
    ``canonical − drop`` 后，按 ``drop 之后`` 的列顺序重建掩码（列在 effective 里
    则为 True），要求与真实 mask 完全相等，且保留列依次等于模型公开列名。
    """
    if not isinstance(audit, Mapping):
        return None
    removed = _as_columns(audit.get("removed_feature_cols"))
    if not removed:
        return None
    removed_set = set(removed)
    for drop in removed:
        built = [column for column in columns if column != drop]
        if len(built) != expected:
            continue
        rebuilt_mask = [column not in removed_set for column in built]
        if rebuilt_mask != mask_list:
            continue
        kept = [column for column, keep in zip(built, mask_list) if keep]
        if _normalized_set(kept) == model_norm:
            return built
    return None


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
            # 模型公开的列名数少于 pipeline 期望数 → 典型是
            # Pipeline(imputer(N) → feature_mask(N→M) → scaler(M) → model(M))：
            # 模型看到的是 mask **之后**的 M 列名，而 pipeline 的输入契约是
            # mask **之前**的 N 列。此时不能直接把 M 与 N 对比报错，
            # 而应用 feature_mask 从更宽的候选列还原出 N 列。
            #
            # 实测背景：TabPFN artifact 的 model.feature_names_in_ 为 1408 列，
            # pipeline.n_features_in_ 为 2070，feature_mask 为 2070 位
            # （True 数 = 1408）→ 旧逻辑必然报「1408 vs 2070」而无法启用模型。
            widened = None
            if len(selected) < expected:
                widened = _widen_columns_to_expected(
                    expected=expected,
                    model_columns=selected,
                    mask=mask,
                    candidates=candidates,
                    audit=(artifact.get("extra") or {}).get("feature_audit")
                    if isinstance(artifact.get("extra"), Mapping)
                    else None,
                )
            if widened is not None:
                selected = widened
                source = "feature_mask(canonical)"
                masked_source = True
            else:
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

    # 被 feature_mask 剔除的列：记录下来供审计（不得静默丢弃）
    removed_features: list[str] = []
    if masked_source and mask is not None and len(mask) == len(selected):
        removed_features = [
            column for column, keep in zip(selected, mask) if not bool(keep)
        ]

    return {
        "ok": not errors,
        "feature_cols": selected,
        "expected_count": expected,
        "source": source,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "duplicate_features": [],
        "order_mismatch": order_mismatch,
        "removed_features": removed_features,
        "errors": errors,
    }
