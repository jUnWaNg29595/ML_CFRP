"""Helpers for prediction-time molecular workflow inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    if type(missing).__name__ == "bool_":
        return bool(missing)
    return False


def _ordered_step_ids(workflow: Mapping[str, Any]) -> list[str]:
    steps = workflow.get("steps") or []
    by_id = {
        str(step.get("step_id")): step
        for step in steps
        if isinstance(step, Mapping) and step.get("step_id")
    }
    merge_order = [
        str(step_id)
        for step_id in (workflow.get("merge_order") or [])
        if str(step_id) in by_id
    ]
    remaining = [
        step_id
        for step_id, _ in sorted(
            by_id.items(),
            key=lambda item: (
                int(item[1].get("order", 10**9))
                if str(item[1].get("order", "")).isdigit()
                else 10**9,
                item[0],
            ),
        )
        if step_id not in merge_order
    ]
    return merge_order + remaining


def collect_workflow_source_columns(
    workflow: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Collect declared input columns in saved workflow order.

    A column can be referenced by multiple steps; its role list is merged
    without changing the first-seen order.
    """

    if not isinstance(workflow, Mapping):
        return []
    by_id = {
        str(step.get("step_id")): step
        for step in (workflow.get("steps") or [])
        if isinstance(step, Mapping) and step.get("step_id")
    }
    result: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for step_id in _ordered_step_ids(workflow):
        step = by_id[step_id]
        role = str(step.get("role") or "neutral").strip().lower() or "neutral"
        for value in step.get("source_columns") or []:
            column = str(value or "").strip()
            if not column:
                continue
            if column not in positions:
                positions[column] = len(result)
                result.append({"column": column, "roles": [role]})
                continue
            roles = result[positions[column]]["roles"]
            if role not in roles:
                roles.append(role)
    return result


def workflow_requires_manual_molecular_input(
    workflow: Mapping[str, Any] | None,
) -> bool:
    """Return whether prediction must collect molecular source values manually."""

    return isinstance(workflow, Mapping) and bool(
        collect_workflow_source_columns(workflow)
    )


def build_single_row_source_frame(
    source_columns: list[dict[str, Any]],
    values: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Build a one-row source frame without coercing SMILES/BigSMILES values."""

    values = values if isinstance(values, Mapping) else {}
    record = {
        str(item["column"]): (
            "" if values.get(str(item["column"])) is None else values.get(str(item["column"]))
        )
        for item in source_columns
        if isinstance(item, Mapping) and str(item.get("column") or "").strip()
    }
    return pd.DataFrame([record])


def validate_single_row_source_values(
    source_frame: pd.DataFrame,
    source_columns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate required source values before replaying a molecular workflow."""

    if not isinstance(source_frame, pd.DataFrame) or source_frame.shape[0] != 1:
        return {
            "ok": False,
            "missing_columns": [
                str(item.get("column"))
                for item in source_columns
                if isinstance(item, Mapping) and str(item.get("column") or "").strip()
            ],
            "empty_columns": [],
        }

    missing_columns = []
    empty_columns = []
    for item in source_columns:
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "").strip()
        if not column:
            continue
        if column not in source_frame.columns:
            missing_columns.append(column)
            continue
        value = source_frame.iloc[0][column]
        if _is_missing_scalar(value):
            empty_columns.append(column)
            continue

    return {
        "ok": not missing_columns and not empty_columns,
        "missing_columns": missing_columns,
        "empty_columns": empty_columns,
    }
