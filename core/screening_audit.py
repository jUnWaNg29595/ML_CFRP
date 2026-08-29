# -*- coding: utf-8 -*-
"""Candidate-level audit for high-throughput virtual screening.

This module is deliberately dependency-light (no rdkit, no streamlit, no model
code) so it can be unit tested in isolation.  It implements:

* a candidate pool with unique ``candidate_id`` plus a status machine that
  *keeps failed candidates* instead of deleting them;
* funnel (per-stage) summaries over that pool;
* merging of multi-objective (multi-model) screening results by
  ``candidate_id`` without mixing feature columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Candidate status enumeration.  Failed candidates are never deleted; the UI
# filters them for display only.
# ---------------------------------------------------------------------------
VALID = "valid"
STRUCTURE_INVALID = "structure_invalid"
WORKFLOW_FAILED = "workflow_failed"
MISSING_REQUIRED_INPUT = "missing_required_input"
FEATURE_INVALID = "feature_invalid"
OUT_OF_RANGE = "out_of_range"
MODEL_PREDICTION_FAILED = "model_prediction_failed"
FILTERED_BY_CHEMISTRY = "filtered_by_chemistry"
FILTERED_BY_MELTING_POINT = "filtered_by_melting_point"
FILTERED_BY_FEASIBILITY = "filtered_by_feasibility"
SELECTED = "selected"

CANDIDATE_STATUSES = (
    VALID,
    STRUCTURE_INVALID,
    WORKFLOW_FAILED,
    MISSING_REQUIRED_INPUT,
    FEATURE_INVALID,
    OUT_OF_RANGE,
    MODEL_PREDICTION_FAILED,
    FILTERED_BY_CHEMISTRY,
    FILTERED_BY_MELTING_POINT,
    FILTERED_BY_FEASIBILITY,
    SELECTED,
)

_FAILURE_STATUSES = frozenset(
    {
        STRUCTURE_INVALID,
        WORKFLOW_FAILED,
        MISSING_REQUIRED_INPUT,
        FEATURE_INVALID,
        OUT_OF_RANGE,
        MODEL_PREDICTION_FAILED,
        FILTERED_BY_CHEMISTRY,
        FILTERED_BY_MELTING_POINT,
        FILTERED_BY_FEASIBILITY,
    }
)

#: Ordered pipeline stages used by :func:`summarize_funnel` (Chinese labels so
#: exported audit reports read naturally for reviewers).
FUNNEL_STAGES = (
    ("候选池生成", "valid"),
    ("化学规则过滤", (FILTERED_BY_CHEMISTRY,)),
    ("熔点过滤", (FILTERED_BY_MELTING_POINT,)),
    ("可行性过滤", (FILTERED_BY_FEASIBILITY,)),
    ("结构校验", (STRUCTURE_INVALID,)),
    ("工作流特征提取", (WORKFLOW_FAILED,)),
    ("固定输入校验", (MISSING_REQUIRED_INPUT,)),
    ("特征有效性校验", (FEATURE_INVALID,)),
    ("适用域校验", (OUT_OF_RANGE,)),
    ("模型预测", (MODEL_PREDICTION_FAILED,)),
    ("入选", (SELECTED,)),
)

_CANDIDATE_TEXT_KEYS = (
    "resin_smiles",
    "hardener_smiles",
    "component_roles",
    "source",
    "generation_method",
)
_CANDIDATE_PASSTHROUGH_KEYS = (
    "structure_validation",
    "recipe_params",
)


def _as_plain_value(value: Any) -> Any:
    """Convert numpy scalars / arrays into JSON-friendly primitives."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _as_plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_plain_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _coerce_frame_row(candidate: Any) -> Dict[str, Any]:
    """Normalize one candidate (dict, Series or DataFrame row) into a dict."""
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if isinstance(candidate, pd.Series):
        return {str(k): v for k, v in candidate.items()}
    if isinstance(candidate, pd.DataFrame):
        if len(candidate) == 1:
            return {str(k): v for k, v in candidate.iloc[0].items()}
        raise ValueError("new_candidate_pool 不接受多行 DataFrame 作为单个候选。")
    try:
        return dict(candidate)
    except Exception:
        return {"value": candidate}


def new_candidate_pool(candidates: Any) -> List[Dict[str, Any]]:
    """Create an auditable candidate pool from raw candidate records.

    Each candidate gets a unique sequential ``candidate_id`` (``c0001``...)
    and starts with ``status='valid'``.  Failed candidates must be kept in the
    pool with ``status`` + ``failure_reason`` set; callers filter them only at
    display time.
    """
    if candidates is None:
        return []
    if isinstance(candidates, pd.DataFrame):
        records: List[Mapping[str, Any]] = [
            {str(k): v for k, v in row.items()} for _, row in candidates.iterrows()
        ]
    elif isinstance(candidates, Mapping):
        records = [candidates]
    elif isinstance(candidates, Iterable):
        records = [_coerce_frame_row(item) for item in candidates]
    else:
        raise ValueError("candidates 必须是 DataFrame、映射或映射的可迭代对象。")

    pool: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        entry: Dict[str, Any] = {
            "candidate_id": f"c{index:04d}",
            "resin_smiles": _as_plain_value(record.get("resin_smiles")),
            "hardener_smiles": _as_plain_value(record.get("hardener_smiles")),
            "component_roles": _as_plain_value(
                record.get("component_roles")
                if record.get("component_roles") is not None
                else record.get("roles")
            ),
            "source": _as_plain_value(record.get("source") or "unknown"),
            "generation_method": _as_plain_value(
                record.get("generation_method")
                if record.get("generation_method") is not None
                else record.get("method")
            ),
            "structure_validation": _as_plain_value(
                record.get("structure_validation")
            ),
            "status": VALID,
            "failure_reason": None,
            "recipe_params": _as_plain_value(record.get("recipe_params")),
            "filter_trace": [],
        }
        for key in _CANDIDATE_TEXT_KEYS + _CANDIDATE_PASSTHROUGH_KEYS:
            if key in record and key not in entry:
                entry[key] = _as_plain_value(record.get(key))
        pool.append(entry)
    return pool


def find_candidate(pool: List[Dict[str, Any]], candidate_id: str) -> Optional[Dict[str, Any]]:
    """Return the candidate record with ``candidate_id`` (or None)."""
    wanted = str(candidate_id or "")
    for entry in pool:
        if str(entry.get("candidate_id")) == wanted:
            return entry
    return None


def set_candidate_status(
    pool: List[Dict[str, Any]],
    candidate_id: str,
    status: str,
    reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update a candidate's status (and failure reason) without removing it.

    Unknown candidate ids are ignored (returns ``None``) so pipeline code can
    call this defensively while iterating frames that may have been pre-filtered.
    """
    entry = find_candidate(pool, candidate_id)
    if entry is None:
        return None
    status = str(status or "").strip()
    if status not in CANDIDATE_STATUSES:
        raise ValueError(f"未知候选状态: {status}")
    entry["status"] = status
    if reason is not None:
        entry["failure_reason"] = str(reason)
    elif status in _FAILURE_STATUSES and not entry.get("failure_reason"):
        entry["failure_reason"] = "未提供失败原因"
    return entry


def record_filter(
    pool: List[Dict[str, Any]],
    candidate_id: str,
    stage: str,
    rule: str,
    count_before: int,
    count_after: int,
) -> Optional[Dict[str, Any]]:
    """Append a filter event to a candidate's ``filter_trace``."""
    entry = find_candidate(pool, candidate_id)
    if entry is None:
        return None
    trace = entry.setdefault("filter_trace", [])
    trace.append(
        {
            "stage": str(stage or ""),
            "rule": str(rule or ""),
            "count_before": int(count_before),
            "count_after": int(count_after),
        }
    )
    return entry


def merge_model_results(
    *,
    pools: List[List[Dict[str, Any]]],
    by: str = "candidate_id",
) -> Dict[str, List[Any]]:
    """Merge multi-objective (multi-model) screening results by candidate id.

    Each pool's prediction columns are prefixed with the pool's model tag so
    that feature columns are never mixed between models.  Non-identifier
    columns are prefixed too; ``candidate_id`` (and the join column) stay
    unprefixed.  Returns ``{"merged": DataFrame, "candidate_ids": [...],
    "model_tags": [...]}``.
    """
    if by != "candidate_id":
        raise ValueError("当前仅支持按 candidate_id 合并多目标筛选结果。")
    if not pools:
        return {"merged": pd.DataFrame(), "candidate_ids": [], "model_tags": []}

    merged: Optional[pd.DataFrame] = None
    model_tags: List[str] = []
    for index, pool in enumerate(pools):
        tag = f"model_{index + 1}"
        model_tags.append(tag)
        frame = pd.DataFrame(pool if pool is not None else [])
        if frame.empty:
            frame = pd.DataFrame(columns=["candidate_id"])
        if "candidate_id" not in frame.columns:
            raise ValueError(f"第 {index + 1} 个 pool 缺少 candidate_id 列，无法合并。")
        rename = {
            column: f"{tag}__{column}"
            for column in frame.columns
            if column != "candidate_id"
        }
        frame = frame.rename(columns=rename)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="candidate_id", how="outer")
    if merged is None:
        merged = pd.DataFrame()
    candidate_ids = merged["candidate_id"].tolist() if "candidate_id" in merged.columns else []
    return {
        "merged": merged,
        "candidate_ids": candidate_ids,
        "model_tags": model_tags,
    }


def summarize_funnel(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize candidate counts per funnel stage (Chinese stage names).

    Returns ``{"stages": [{"stage", "status", "count", "filtered"}...],
    "total": int, "remaining": int, "status_counts": {...}}`` where ``count``
    is the number of candidates removed at that stage and ``filtered`` is the
    cumulative number removed before it.
    """
    pool = pool or []
    total = len(pool)
    status_counts: Dict[str, int] = {}
    for entry in pool:
        status = str(entry.get("status") or VALID)
        status_counts[status] = status_counts.get(status, 0) + 1

    stages: List[Dict[str, Any]] = []
    cumulative_filtered = 0
    for label, statuses in FUNNEL_STAGES:
        if isinstance(statuses, str):
            statuses = (statuses,)
        count = sum(status_counts.get(status, 0) for status in statuses)
        stages.append(
            {
                "stage": label,
                "status": list(statuses),
                "count": count,
                "filtered": cumulative_filtered,
            }
        )
        cumulative_filtered += count

    remaining = sum(
        1 for entry in pool if str(entry.get("status")) in {VALID, SELECTED}
    )
    return {
        "stages": stages,
        "total": total,
        "remaining": remaining,
        "status_counts": status_counts,
    }
