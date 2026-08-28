"""Versioned molecular feature workflow schema and validation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable

import numpy as np
import pandas as pd


_WORKFLOW_FIELDS = (
    "schema_version",
    "model_fingerprint",
    "mode",
    "input_contract",
    "steps",
    "merge_order",
    "final_feature_names",
    "feature_source_map",
    "derived_feature_steps",
    "random_seeds",
    "workflow_hash",
    "legacy",
    "missing_items",
)


def _deduplicate(values: list[Any]) -> list[Any]:
    result = []
    hashable_seen = set()
    serialized_seen = set()
    fallback_values = []
    for value in values:
        try:
            if value in hashable_seen:
                continue
            hashable_seen.add(value)
        except TypeError:
            try:
                marker = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                )
            except (TypeError, ValueError):
                if any(value == previous for previous in fallback_values):
                    continue
                fallback_values.append(value)
            else:
                if marker in serialized_seen:
                    continue
                serialized_seen.add(marker)
        result.append(value)
    return result


def align_extracted_features_to_rows(
    features_df: pd.DataFrame,
    valid_indices: Sequence[int] | None,
    total_rows: int,
) -> pd.DataFrame:
    """Place extracted rows back into the source-row coordinate system."""
    features = features_df.reset_index(drop=True)
    row_count = max(0, int(total_rows))
    indices = [
        int(index)
        for index in (valid_indices or [])
        if 0 <= int(index) < row_count
    ]
    n_rows = min(len(indices), len(features))
    if n_rows == 0:
        return features.iloc[:0].reindex(range(row_count))

    aligned = features.iloc[:n_rows].copy()
    aligned.index = indices[:n_rows]
    if aligned.index.has_duplicates:
        aligned = aligned[~aligned.index.duplicated(keep="last")]
    return aligned.reindex(range(row_count))


def merge_extracted_features(
    base_df: pd.DataFrame,
    features_df: pd.DataFrame,
    valid_indices: Sequence[int] | None,
    *,
    keep_all_rows: bool = True,
) -> pd.DataFrame:
    """Merge extracted rows while avoiding an unnecessary full-table backfill."""
    base = base_df.reset_index(drop=True)
    features = features_df.reset_index(drop=True)
    indices = [
        int(index)
        for index in (valid_indices or [])
        if 0 <= int(index) < len(base)
    ]

    if not keep_all_rows:
        return pd.concat(
            [base.iloc[indices].reset_index(drop=True), features.iloc[: len(indices)]],
            axis=1,
        )

    if (
        len(base) == len(features) == len(indices)
        and indices == list(range(len(base)))
    ):
        return pd.concat([base, features], axis=1)

    n_rows = min(len(indices), len(features))
    full_features = align_extracted_features_to_rows(
        features,
        indices[:n_rows],
        len(base),
    )
    return pd.concat([base, full_features], axis=1)


_FEATURE_COMPONENT_ROLE_TOKENS = {
    "resin": ("resin", "epoxy", "树脂", "基体"),
    "hardener": (
        "hardener",
        "curing_agent",
        "curingagent",
        "curative",
        "固化剂",
        "交联剂",
    ),
}


def resolve_feature_component_role(
    selected_role: Any = None,
    *,
    inferred_role: Any = None,
    default: str = "neutral",
) -> str:
    """Return the explicit workflow role, falling back to inference only when absent."""
    valid_roles = {"resin", "hardener", "neutral"}
    selected = str(selected_role or "").strip().lower()
    if selected in valid_roles:
        return selected
    inferred = str(inferred_role or "").strip().lower()
    if inferred in valid_roles:
        return inferred
    fallback = str(default or "").strip().lower()
    return fallback if fallback in valid_roles else "neutral"


def _feature_component_role_score(column: Any, role: str) -> int:
    target_role = str(role or "").strip().lower()
    opposite_role = "hardener" if target_role == "resin" else "resin"
    text = str(column or "").strip().lower()
    target_score = sum(text.count(token) for token in _FEATURE_COMPONENT_ROLE_TOKENS.get(target_role, ()))
    opposite_score = sum(text.count(token) for token in _FEATURE_COMPONENT_ROLE_TOKENS.get(opposite_role, ()))
    return target_score - opposite_score


def _feature_component_natural_key(column: str) -> tuple[str, int]:
    match = re.search(r"(\d+)(?:\D*)$", str(column))
    if not match:
        return str(column).lower(), -1
    return str(column)[: match.start()].lower(), int(match.group(1))


def _feature_component_columns(values: Sequence[Any] | str | None) -> list[str]:
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values or []:
        column = str(value or "").strip()
        if column and column not in seen:
            seen.add(column)
            result.append(column)
    return result


def get_feature_component_column_options(
    available_columns: Sequence[Any] | None,
    *,
    smiles_candidates: Sequence[Any] | None = None,
    role: str = "resin",
    primary_column: str | None = None,
    dedicated_columns: Sequence[Any] | None = None,
) -> list[str]:
    """Return role-compatible source columns for multi-component extraction."""
    target_role = str(role or "").strip().lower()
    if target_role not in {"resin", "hardener"}:
        target_role = "resin"

    available = _feature_component_columns(available_columns)
    available_set = set(available)
    detected = _feature_component_columns(
        smiles_candidates if smiles_candidates is not None else available
    )
    detected = [column for column in detected if column in available_set or not available_set]

    dedicated = _feature_component_columns(dedicated_columns)
    if dedicated:
        base = [
            column
            for column in _deduplicate(dedicated + detected)
            if column in available_set or not available_set
        ]
    else:
        base = detected
        if smiles_candidates is None:
            base = [
                column
                for column in base
                if re.search(r"(smiles|bigsmiles|smile|smi|molecule|structure)", column, flags=re.I)
            ]

    opposite_role = "hardener" if target_role == "resin" else "resin"
    filtered = [
        column
        for column in base
        if _feature_component_role_score(column, target_role)
        >= _feature_component_role_score(column, opposite_role)
    ]
    return sorted(_feature_component_columns(filtered), key=_feature_component_natural_key)


def resolve_feature_component_columns(
    available_columns: Sequence[Any] | None,
    *,
    role: str = "resin",
    primary_column: str | None = None,
    dedicated_columns: Sequence[Any] | None = None,
    selected_columns: Sequence[Any] | str | None = None,
    smiles_candidates: Sequence[Any] | None = None,
    mode: str = "auto",
) -> list[str]:
    """Resolve the actual extraction columns for a single feature workflow."""
    available = _feature_component_columns(available_columns)
    available_set = set(available)
    dedicated = _feature_component_columns(dedicated_columns)
    dedicated_options = (
        get_feature_component_column_options(
            available,
            smiles_candidates=smiles_candidates,
            role=role,
            primary_column=primary_column,
            dedicated_columns=dedicated,
        )
        if dedicated
        else []
    )
    if dedicated:
        dedicated_set = set(dedicated)
        dedicated_options = [
            column for column in dedicated_options if column in dedicated_set
        ]
    if str(mode or "").strip().lower() != "manual":
        if dedicated_options:
            return dedicated_options
        primary = str(primary_column or "").strip()
        if primary and (not available_set or primary in available_set):
            if _feature_component_role_score(primary, role) >= 0:
                return [primary]
        return []

    options = set(
        get_feature_component_column_options(
            available,
            smiles_candidates=smiles_candidates,
            role=role,
            primary_column=primary_column,
            dedicated_columns=dedicated if dedicated else None,
        )
    )
    selected = [
        column
        for column in _feature_component_columns(selected_columns)
        if column in options
    ]
    if selected:
        return selected
    if dedicated_options:
        return dedicated_options
    primary = str(primary_column or "").strip()
    if primary and (not available_set or primary in available_set):
        if _feature_component_role_score(primary, role) >= 0:
            return [primary]
    return []


def merge_feature_name_lists_in_order(
    existing: Sequence[str] | None,
    incoming: Sequence[str] | None,
) -> list[str]:
    """Merge feature names by first-seen order without duplicate columns."""
    return _deduplicate(
        [
            str(name)
            for name in list(existing or []) + list(incoming or [])
            if str(name)
        ]
    )


def find_duplicate_feature_names(names: Sequence[Any] | None) -> list[str]:
    """Return emitted feature names repeated in the supplied order."""
    duplicates = _find_duplicates([str(name) for name in list(names or [])])
    return [str(name) for name in duplicates]


def materialize_workflow_source_columns(
    candidate_df: pd.DataFrame,
    *,
    resin_columns: Sequence[str] | str | None = None,
    hardener_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Expand canonical formulation strings into declared workflow source columns.

    The screening layer stores canonical ``resin_smiles`` and
    ``hardener_smiles`` values.  A saved training workflow may instead require
    numbered source columns such as ``resin_smiles_1`` or
    ``curing_agent_smiles_2``.  Component splitting must use the same
    top-level-aware parser as feature extraction so dots inside BigSMILES
    blocks are not mistaken for formulation separators.
    """
    if not isinstance(candidate_df, pd.DataFrame):
        return pd.DataFrame()

    out = candidate_df.copy()

    def _as_columns(value: Sequence[str] | str | None) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [
            str(column).strip()
            for column in (value or [])
            if str(column).strip()
        ]

    def _expand(canonical_column: str, declared_columns: Sequence[str] | str | None) -> None:
        columns = _as_columns(declared_columns)
        if not columns or canonical_column not in out.columns:
            return

        from .smiles_utils import split_smiles_cell

        split_values = out[canonical_column].map(split_smiles_cell)
        for position, column in enumerate(columns):
            if column in out.columns:
                continue
            out[column] = split_values.map(
                lambda parts, index=position: (
                    parts[index] if index < len(parts) else None
                )
            )

    _expand("resin_smiles", resin_columns)
    _expand("hardener_smiles", hardener_columns)
    return out


def validate_feature_frame_contract(
    features: pd.DataFrame | None,
    valid_indices: Sequence[Any] | None,
    source_row_count: int,
    expected_feature_names: Sequence[Any] | None = None,
) -> list[str]:
    """Return contract violations for one extracted feature frame."""
    errors = []
    if not isinstance(features, pd.DataFrame):
        return ["feature extractor did not return a DataFrame"]
    if features.empty or len(features.columns) == 0:
        errors.append("feature extractor returned no features")
    duplicate_names = find_duplicate_feature_names(features.columns.tolist())
    if duplicate_names:
        errors.append(
            "duplicate emitted feature names: " + ", ".join(duplicate_names)
        )
    declared_indices = list(valid_indices or [])
    if len(features) != len(declared_indices):
        errors.append(
            "feature row count does not match valid-row indices "
            f"({len(features)} != {len(declared_indices)})"
        )
    invalid_indices = []
    seen_indices = set()
    for index in declared_indices:
        if not isinstance(index, Integral) or not 0 <= int(index) < int(source_row_count):
            invalid_indices.append(index)
        elif int(index) in seen_indices:
            invalid_indices.append(index)
        else:
            seen_indices.add(int(index))
    if invalid_indices:
        errors.append(
            "valid-row indices are invalid or duplicated: "
            + ", ".join(map(str, invalid_indices[:12]))
        )
    if expected_feature_names is not None:
        missing = [
            str(name)
            for name in expected_feature_names
            if str(name) and str(name) not in features.columns
        ]
        if missing:
            errors.append(
                "missing required emitted feature names: "
                + ", ".join(missing[:12])
            )
    return errors


def _find_duplicates(values: Sequence[Any]) -> list[Any]:
    seen = []
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        elif value not in seen:
            seen.append(value)
    return duplicates


def _collect_duplicate_workflow_items(
    payload: Mapping[str, Any],
) -> dict[str, list[Any]]:
    diagnostics: dict[str, list[Any]] = {}
    for field_name in ("merge_order", "final_feature_names"):
        duplicates = _find_duplicates(list(payload.get(field_name) or []))
        if duplicates:
            diagnostics[field_name] = duplicates

    steps = list(payload.get("steps") or [])
    step_ids = [
        step.get("step_id")
        for step in steps
        if isinstance(step, Mapping) and step.get("step_id")
    ]
    duplicate_step_ids = _find_duplicates(step_ids)
    if duplicate_step_ids:
        diagnostics["step_ids"] = duplicate_step_ids
    for step in steps:
        if not isinstance(step, Mapping) or not step.get("step_id"):
            continue
        duplicates = _find_duplicates(list(step.get("feature_names") or []))
        if duplicates:
            diagnostics[f"step:{step['step_id']}:feature_names"] = duplicates
    return diagnostics


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return _deduplicate([_normalize_value(item) for item in value])
    if isinstance(value, tuple):
        return _deduplicate([_normalize_value(item) for item in value])
    return value


def _hash_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_value(payload[key])
        for key in _WORKFLOW_FIELDS
        if key != "workflow_hash" and key in payload
    }


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _hash_input(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_workflow_hash(payload: Mapping[str, Any]) -> str:
    """Return the stable hash of a workflow payload, excluding its hash field."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_workflow_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a workflow payload into the current versioned shape."""
    if not isinstance(payload, Mapping):
        raise TypeError("workflow payload must be a mapping")

    normalized = _normalize_value(dict(payload))
    schema_version = normalized.get("schema_version", 1)
    legacy = schema_version < 2
    required_fields = [
        "mode",
        "input_contract",
        "steps",
        "merge_order",
        "final_feature_names",
        "feature_source_map",
        "derived_feature_steps",
        "random_seeds",
    ]
    if not legacy:
        required_fields = [
            "mode",
            "steps",
            "merge_order",
            "final_feature_names",
        ]
    missing_items = list(normalized.get("missing_items", []))
    missing_items.extend(
        field_name for field_name in required_fields if field_name not in normalized
    )
    required_defaults = {
        "schema_version": schema_version,
        "model_fingerprint": None,
        "mode": "single_batch",
        "input_contract": {},
        "steps": [],
        "merge_order": [],
        "final_feature_names": [],
        "feature_source_map": {},
        "derived_feature_steps": [],
        "random_seeds": {},
        "legacy": legacy or bool(missing_items),
        "missing_items": missing_items,
    }
    for key, default in required_defaults.items():
        normalized.setdefault(key, default)

    normalized["steps"] = _deduplicate(normalized["steps"])
    normalized["merge_order"] = _deduplicate(normalized["merge_order"])
    normalized["final_feature_names"] = _deduplicate(
        normalized["final_feature_names"]
    )
    normalized["derived_feature_steps"] = _deduplicate(
        normalized["derived_feature_steps"]
    )
    normalized["missing_items"] = _deduplicate(normalized["missing_items"])
    serialized = {
        key: normalized[key]
        for key in _WORKFLOW_FIELDS
        if key != "workflow_hash"
    }
    serialized["workflow_hash"] = compute_workflow_hash(serialized)
    return serialized


def append_workflow_step(
    payload: Mapping[str, Any],
    step: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one editable step while preserving existing workflow order."""
    normalized = normalize_workflow_config(payload)
    steps = [dict(item) for item in normalized.get("steps") or []]
    existing_ids = {
        str(item.get("step_id")).strip()
        for item in steps
        if isinstance(item, Mapping) and str(item.get("step_id") or "").strip()
    }
    next_index = len(steps) + 1
    step_id = f"step_{next_index}"
    while step_id in existing_ids:
        next_index += 1
        step_id = f"step_{next_index}"

    new_step = {
        "step_id": step_id,
        "order": len(steps),
        "role": "neutral",
        "source_columns": [],
        "method": "",
        "prefix": "",
        "params": {},
        "semantic_params": {},
        "feature_names": [],
        "valid_row_behavior": {
            "keep_all_rows": True,
            "invalid_row_policy": "nan_fill",
        },
    }
    if isinstance(step, Mapping):
        new_step.update(dict(step))
        requested_id = str(new_step.get("step_id") or step_id).strip()
        new_step["step_id"] = requested_id if requested_id not in existing_ids else step_id
        new_step["order"] = len(steps)

    steps.append(new_step)
    normalized["steps"] = steps
    merge_order = [
        str(item).strip()
        for item in (normalized.get("merge_order") or [])
        if str(item).strip()
    ]
    if new_step["step_id"] not in merge_order:
        merge_order.append(new_step["step_id"])
    normalized["merge_order"] = merge_order
    return normalize_workflow_config(normalized)


@dataclass
class MolecularFeatureWorkflow:
    schema_version: int
    model_fingerprint: str | None
    mode: str
    input_contract: dict
    steps: list[dict]
    merge_order: list[str]
    final_feature_names: list[str]
    feature_source_map: dict
    derived_feature_steps: list[dict]
    random_seeds: dict
    workflow_hash: str
    legacy: bool = False
    missing_items: list[str] = field(default_factory=list)
    _duplicate_items: dict[str, list[Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MolecularFeatureWorkflow":
        normalized = normalize_workflow_config(payload)
        workflow = cls(**{key: normalized[key] for key in _WORKFLOW_FIELDS})
        workflow._duplicate_items = _collect_duplicate_workflow_items(payload)
        return workflow

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "mode": self.mode,
            "input_contract": self.input_contract,
            "steps": self.steps,
            "merge_order": self.merge_order,
            "final_feature_names": self.final_feature_names,
            "feature_source_map": self.feature_source_map,
            "derived_feature_steps": self.derived_feature_steps,
            "random_seeds": self.random_seeds,
            "workflow_hash": self.workflow_hash,
            "legacy": self.legacy,
            "missing_items": self.missing_items,
        }

    def to_legacy_config(self) -> dict[str, Any]:
        """Project the first workflow step into the legacy extractor config."""
        first_step = dict(self.steps[0]) if self.steps else {}
        input_contract = dict(self.input_contract or {})
        legacy_fields = input_contract.get("legacy_fields")
        config = (
            dict(legacy_fields)
            if isinstance(legacy_fields, Mapping)
            else {}
        )
        config.setdefault("method", first_step.get("method"))
        source_columns = list(first_step.get("source_columns") or [])
        config.setdefault("smiles_col", source_columns[0] if source_columns else None)
        config.setdefault(
            "resin_component_cols",
            list(input_contract.get("resin_component_cols") or source_columns[:1]),
        )
        config.setdefault("hardener_col", input_contract.get("hardener_col"))
        config.setdefault(
            "hardener_component_cols",
            list(input_contract.get("hardener_component_cols") or []),
        )
        config.setdefault(
            "hardener_fusion_mode",
            input_contract.get("hardener_fusion_mode"),
        )
        config.setdefault("resin_mix_mode", input_contract.get("resin_mix_mode"))
        config.setdefault(
            "primary_component_role",
            input_contract.get("primary_component_role"),
        )
        config.setdefault(
            "component_roles",
            dict(input_contract.get("component_roles") or {}),
        )
        config.setdefault("prefix", first_step.get("prefix") or "")
        params = dict(first_step.get("params") or {})
        params.update(dict(first_step.get("semantic_params") or {}))
        config.setdefault("params", params)
        config.setdefault("feature_names", list(self.final_feature_names))
        config.setdefault("n_features", len(self.final_feature_names))
        config.setdefault("mode", self.mode)
        config.setdefault("workflow_mode", self.mode)
        valid_row_behavior = first_step.get("valid_row_behavior")
        if isinstance(valid_row_behavior, Mapping):
            config.setdefault(
                "n_valid_samples",
                valid_row_behavior.get("n_valid_samples"),
            )
            if "keep_all_rows" in valid_row_behavior:
                config.setdefault(
                    "keep_all_rows_3d",
                    bool(valid_row_behavior["keep_all_rows"]),
                )
        if self.mode == "multi_batch" or len(self.steps) > 1:
            config.setdefault("batch_mode", True)
            config.setdefault(
                "batch_smiles_cols",
                [
                    column
                    for step in self.steps
                    for column in step.get("source_columns") or []
                ],
            )
        return config


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence_copy(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value) if isinstance(value, Sequence) else [value]


def build_workflow_from_training_state(
    state: Mapping[str, Any],
    feature_names: Sequence[str],
) -> MolecularFeatureWorkflow:
    """Build a versioned workflow from the training extraction state."""
    if not isinstance(state, Mapping):
        raise TypeError("training state must be a mapping")

    default_source_columns = _sequence_copy(
        state.get("selected_source_columns")
    )
    if not default_source_columns:
        default_source_columns = _sequence_copy(state.get("source_columns"))
    if not default_source_columns:
        default_source_columns = _sequence_copy(
            state.get("resin_component_cols")
        )
    if not default_source_columns and state.get("smiles_col") is not None:
        default_source_columns = [state["smiles_col"]]
    for column in _sequence_copy(state.get("hardener_component_cols")):
        if column not in default_source_columns:
            default_source_columns.append(column)

    raw_steps = state.get("workflow_steps") or state.get("steps") or []
    if not raw_steps:
        raw_steps = [
            {
                "step_id": state.get("step_id", "molecular_features_1"),
                "order": 0,
                "role": state.get(
                    "role",
                    state.get("primary_component_role", "neutral"),
                ),
                "source_columns": default_source_columns,
                "method": state.get("method"),
                "prefix": state.get("prefix", ""),
                "params": _mapping_copy(state.get("params")),
                "semantic_params": _mapping_copy(state.get("semantic_params")),
                "feature_names": list(feature_names),
                "valid_row_behavior": _mapping_copy(
                    state.get("valid_row_behavior")
                ),
            }
        ]

    steps = []
    for order, raw_step in enumerate(raw_steps):
        step = dict(raw_step) if isinstance(raw_step, Mapping) else {}
        step.setdefault("step_id", f"molecular_features_{order + 1}")
        step.setdefault("order", order)
        step.setdefault(
            "source_columns",
            default_source_columns,
        )
        step.setdefault("method", state.get("method"))
        step.setdefault(
            "role",
            state.get(
                "role",
                state.get("primary_component_role", "neutral"),
            ),
        )
        step.setdefault("prefix", state.get("prefix", ""))
        step["source_columns"] = _sequence_copy(step.get("source_columns"))
        step["params"] = _mapping_copy(
            step.get("params") if "params" in step else state.get("params")
        )
        step["semantic_params"] = _mapping_copy(
            step.get("semantic_params")
            if "semantic_params" in step
            else state.get("semantic_params")
        )
        if "feature_names" not in step:
            step["feature_names"] = (
                list(feature_names) if len(raw_steps) == 1 else []
            )
        else:
            step["feature_names"] = _sequence_copy(step["feature_names"])
        if "valid_row_behavior" not in step:
            step["valid_row_behavior"] = _mapping_copy(
                state.get("valid_row_behavior")
            )
        steps.append(step)

    final_feature_names = merge_feature_name_lists_in_order([], feature_names)
    if not final_feature_names:
        final_feature_names = [
            str(name)
            for step in steps
            for name in step.get("feature_names") or []
            if str(name)
        ]
        final_feature_names = merge_feature_name_lists_in_order(
            [], final_feature_names
        )

    merge_order = _sequence_copy(state.get("merge_order"))
    if not merge_order:
        merge_order = [step["step_id"] for step in steps]
    merge_order = [str(step_id) for step_id in merge_order]

    feature_source_map = _mapping_copy(state.get("feature_source_map"))
    if not feature_source_map:
        for step in steps:
            for feature_name in step.get("feature_names") or []:
                feature_source_map[str(feature_name)] = step["step_id"]

    input_contract = _mapping_copy(state.get("input_contract"))
    for field_name in (
        "selected_source_columns",
        "resin_component_cols",
        "hardener_col",
        "hardener_component_cols",
        "hardener_fusion_mode",
        "resin_mix_mode",
        "primary_component_role",
        "component_roles",
        "valid_row_behavior",
        "legacy_fields",
    ):
        if field_name in state and field_name not in input_contract:
            value = state[field_name]
            input_contract[field_name] = (
                _sequence_copy(value)
                if field_name.endswith("columns")
                else value
            )

    payload = {
        "schema_version": 2,
        "model_fingerprint": state.get("model_fingerprint"),
        "mode": state.get(
            "mode",
            "multi_batch" if len(steps) > 1 else "single_batch",
        ),
        "input_contract": input_contract,
        "steps": steps,
        "merge_order": merge_order,
        "final_feature_names": final_feature_names,
        "feature_source_map": feature_source_map,
        "derived_feature_steps": _sequence_copy(
            state.get("derived_feature_steps")
        ),
        "random_seeds": _mapping_copy(state.get("random_seeds")),
        "legacy": False,
        "missing_items": _sequence_copy(state.get("missing_items")),
    }
    return MolecularFeatureWorkflow.from_dict(payload)


def _unique_training_step_id(existing_ids: set[str], requested_id: Any) -> str:
    requested = str(requested_id or "").strip() or "step"
    if requested not in existing_ids:
        return requested

    match = re.match(r"^(.*?)(?:_(\d+))?$", requested)
    prefix = (match.group(1) if match else requested) or "step"
    start = int(match.group(2) or 1) + 1 if match else 2
    candidate = f"{prefix}_{start}"
    while candidate in existing_ids:
        start += 1
        candidate = f"{prefix}_{start}"
    return candidate


def append_training_workflow(
    existing_payload: Mapping[str, Any] | None,
    state: Mapping[str, Any],
    feature_names: Sequence[str],
) -> MolecularFeatureWorkflow:
    """Append a completed extraction run without discarding prior steps."""
    incoming = build_workflow_from_training_state(state, feature_names)
    if not existing_payload:
        return incoming

    existing = MolecularFeatureWorkflow.from_dict(existing_payload)
    merged_steps = copy.deepcopy(existing.steps)
    existing_ids = {
        str(step.get("step_id")).strip()
        for step in merged_steps
        if isinstance(step, Mapping) and str(step.get("step_id") or "").strip()
    }
    incoming_id_map: dict[str, str] = {}

    for incoming_step in incoming.steps:
        step = copy.deepcopy(incoming_step)
        original_id = str(step.get("step_id") or "").strip()
        step_id = _unique_training_step_id(existing_ids, original_id)
        step["step_id"] = step_id
        step["order"] = len(merged_steps)
        merged_steps.append(step)
        existing_ids.add(step_id)
        if original_id:
            incoming_id_map[original_id] = step_id

    merged_merge_order = [
        str(step_id)
        for step_id in existing.merge_order
        if str(step_id).strip()
    ]
    for step in merged_steps[len(existing.steps):]:
        step_id = str(step.get("step_id") or "").strip()
        if step_id and step_id not in merged_merge_order:
            merged_merge_order.append(step_id)

    merged_feature_names = merge_feature_name_lists_in_order(
        existing.final_feature_names,
        incoming.final_feature_names,
    )
    merged_feature_names = merge_feature_name_lists_in_order(
        merged_feature_names,
        feature_names,
    )

    merged_feature_source_map = copy.deepcopy(existing.feature_source_map)
    for feature_name, original_step_id in incoming.feature_source_map.items():
        mapped_step_id = incoming_id_map.get(
            str(original_step_id),
            str(original_step_id),
        )
        if mapped_step_id in existing_ids:
            merged_feature_source_map[str(feature_name)] = mapped_step_id
    for step in merged_steps[len(existing.steps):]:
        step_id = str(step.get("step_id") or "").strip()
        for feature_name in step.get("feature_names") or []:
            if str(feature_name):
                merged_feature_source_map[str(feature_name)] = step_id

    merged_input_contract = copy.deepcopy(existing.input_contract or {})
    incoming_input_contract = incoming.input_contract or {}
    union_fields = {
        "selected_source_columns",
        "resin_component_cols",
        "hardener_component_cols",
    }
    for field_name, value in incoming_input_contract.items():
        if field_name in union_fields:
            merged_input_contract[field_name] = merge_feature_name_lists_in_order(
                merged_input_contract.get(field_name) or [],
                value or [],
            )
        elif field_name == "legacy_fields" and isinstance(value, Mapping):
            legacy_fields = dict(merged_input_contract.get(field_name) or {})
            legacy_fields.update(copy.deepcopy(dict(value)))
            merged_input_contract[field_name] = legacy_fields
        else:
            merged_input_contract[field_name] = copy.deepcopy(value)

    merged_derived_steps = list(existing.derived_feature_steps or [])
    merged_derived_steps.extend(copy.deepcopy(incoming.derived_feature_steps or []))
    merged_random_seeds = dict(existing.random_seeds or {})
    merged_random_seeds.update(copy.deepcopy(incoming.random_seeds or {}))

    merged_payload = existing.to_dict()
    merged_payload.update(
        {
            "model_fingerprint": (
                existing.model_fingerprint or incoming.model_fingerprint
            ),
            "mode": (
                "multi_batch"
                if len(merged_steps) > 1
                else incoming.mode or existing.mode
            ),
            "input_contract": merged_input_contract,
            "steps": merged_steps,
            "merge_order": merged_merge_order,
            "final_feature_names": merged_feature_names,
            "feature_source_map": merged_feature_source_map,
            "derived_feature_steps": merged_derived_steps,
            "random_seeds": merged_random_seeds,
            "legacy": False,
            "missing_items": [],
        }
    )
    return MolecularFeatureWorkflow.from_dict(merged_payload)


@dataclass
class WorkflowExecutionResult:
    features: pd.DataFrame
    step_trace: list[dict]
    warnings: list[str]
    workflow_hash: str
    valid_row_indices: dict[str, list[int]]


def _step_parameters(step: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(step.get("params") or {})
    params.update(dict(step.get("semantic_params") or {}))
    return params


def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _filter_step_components(
    components: list[str],
    params: Mapping[str, Any],
) -> list[str]:
    if not params.get("drop_catalyst_fragments"):
        return components

    try:
        from .virtual_screening import (
            RDKIT_AVAILABLE,
            Chem,
            _mol_from_smiles,
            rdMolDescriptors,
        )
    except Exception:
        return components
    if not RDKIT_AVAILABLE:
        return components

    only_multi = bool(params.get("drop_catalyst_only_multi", True))
    if only_multi and len(components) <= 1:
        return components

    allowed_elements = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    remove_invalid = bool(params.get("catalyst_remove_invalid", True))
    remove_cations = params.get("catalyst_remove_cations")
    remove_anions = params.get("catalyst_remove_anions")
    if remove_cations is None and remove_anions is None:
        remove_charged = bool(params.get("catalyst_remove_charged", True))
        remove_cations = remove_charged
        remove_anions = remove_charged
    else:
        remove_cations = bool(remove_cations)
        remove_anions = bool(remove_anions)
    remove_metals = bool(params.get("catalyst_remove_metals", True))
    min_heavy = int(params.get("catalyst_min_heavy_atoms", 0) or 0)
    min_mw = float(params.get("catalyst_min_mol_wt", 0.0) or 0.0)
    smarts_mols = []
    for pattern in params.get("catalyst_smarts") or []:
        if not isinstance(pattern, str):
            continue
        try:
            molecule = Chem.MolFromSmarts(pattern)
        except Exception:
            molecule = None
        if molecule is not None:
            smarts_mols.append(molecule)

    def is_catalyst_fragment(fragment: str) -> bool:
        molecule = _mol_from_smiles(fragment)
        if molecule is None:
            return remove_invalid
        try:
            has_positive = any(atom.GetFormalCharge() > 0 for atom in molecule.GetAtoms())
            has_negative = any(atom.GetFormalCharge() < 0 for atom in molecule.GetAtoms())
            if (remove_cations and has_positive) or (remove_anions and has_negative):
                return True
        except Exception:
            pass
        if remove_metals:
            try:
                if any(
                    atom.GetAtomicNum() not in allowed_elements
                    for atom in molecule.GetAtoms()
                ):
                    return True
            except Exception:
                pass
        if min_heavy and molecule.GetNumHeavyAtoms() <= min_heavy:
            return True
        if min_mw:
            try:
                if float(rdMolDescriptors.CalcExactMolWt(molecule)) <= min_mw:
                    return True
            except Exception:
                pass
        if smarts_mols:
            try:
                if any(molecule.HasSubstructMatch(pattern) for pattern in smarts_mols):
                    return True
            except Exception:
                pass
        return False

    return [fragment for fragment in components if not is_catalyst_fragment(fragment)]


def prepare_step_inputs(
    data: pd.DataFrame,
    step: Mapping[str, Any],
) -> tuple[list[str | None], list[int], dict]:
    """Prepare declared source columns without mutating the candidate dataframe."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    source_columns = list(step.get("source_columns") or [])
    optional_columns = set(step.get("optional_source_columns") or [])
    missing_columns = [column for column in source_columns if column not in data.columns and column not in optional_columns]
    if missing_columns:
        raise KeyError(
            "workflow step requires missing source columns: "
            + ", ".join(map(str, missing_columns))
        )

    try:
        from .smiles_utils import split_smiles_cell
    except Exception as exc:
        raise RuntimeError("split_smiles_cell is required for workflow execution") from exc

    params = _step_parameters(step)
    prepared: list[str | None] = []
    present_columns = [column for column in source_columns if column in data.columns]
    source_values = {column: [] for column in present_columns}
    component_values: list[list[str]] = []
    source_components: list[dict[Any, list[str]]] = []
    valid_indices: list[int] = []

    for row_index, row in enumerate(
        data.reindex(columns=present_columns).itertuples(index=False, name=None)
    ):
        components: list[str] = []
        row_source_components: dict[Any, list[str]] = {}
        for column, value in zip(present_columns, row):
            fragments = [] if _is_missing_cell(value) else split_smiles_cell(value)
            retained_fragments = _filter_step_components(fragments, params)
            row_source_components[column] = retained_fragments
            components.extend(retained_fragments)
        retained = components
        component_values.append(retained)
        remaining = list(retained)
        for column, fragments in row_source_components.items():
            kept = []
            for fragment in fragments:
                if fragment in remaining:
                    kept.append(fragment)
                    remaining.remove(fragment)
            source_values[column].append(".".join(kept) if kept else None)
        source_components.append(row_source_components)
        joined = ".".join(retained) if retained else None
        prepared.append(joined)
        if joined is not None:
            valid_indices.append(row_index)

    role = str(step.get("role") or "").lower()
    resin_columns = [
        column
        for column in present_columns
        if any(token in str(column).lower() for token in ("resin", "epoxy"))
    ]
    hardener_columns = [
        column
        for column in present_columns
        if any(
            token in str(column).lower()
            for token in ("hardener", "curing", "curer")
        )
    ]

    def join_source_values(columns: list[Any]) -> list[str | None]:
        joined_values = []
        for row_number in range(len(prepared)):
            values = [
                source_values[column][row_number]
                for column in columns
                if source_values[column][row_number]
            ]
            joined_values.append(".".join(values) if values else None)
        return joined_values

    resin_values = join_source_values(resin_columns)
    hardener_values = join_source_values(hardener_columns)
    if "resin" in role and not any(resin_values):
        resin_values = prepared.copy()
    if ("hardener" in role or "curing" in role) and not any(hardener_values):
        hardener_values = prepared.copy()
    if not any(resin_values) and not any(hardener_values):
        resin_values = prepared.copy()

    metadata = {
        "source_columns": source_columns,
        "source_values": source_values,
        "component_values": component_values,
        "source_components": source_components,
        "resin_smiles": resin_values,
        "hardener_smiles": hardener_values,
    }
    return prepared, valid_indices, metadata


def execute_feature_step(
    smiles: list[str | None],
    step: Mapping[str, Any],
    *,
    device: Any = None,
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Dispatch a saved step through the legacy extractor compatibility adapter."""
    from .virtual_screening import extract_features_from_config

    params = _step_parameters(step)
    adapter_config = dict(step)
    adapter_params = dict(params)
    if adapter_params.get("drop_catalyst_fragments"):
        adapter_params["drop_catalyst_fragments"] = False
    adapter_config["params"] = adapter_params
    adapter_config["prefix"] = ""
    adapter_config["feature_names"] = []
    prepared = adapter_config.pop("_prepared_inputs", {})
    role = str(step.get("role") or "").lower()
    resin_smiles = prepared.get("resin_smiles") or smiles
    hardener_smiles = prepared.get("hardener_smiles")
    if "hardener" in role or "curing" in role:
        resin_smiles, hardener_smiles = smiles, None
    if not any(value is not None for value in (hardener_smiles or [])):
        hardener_smiles = None

    features, error = extract_features_from_config(
        resin_smiles,
        hardener_smiles,
        adapter_config,
        device=device,
    )
    warnings = [str(error)] if error else []
    declared_valid = [
        index for index, value in enumerate(smiles) if not _is_missing_cell(value)
    ]
    if isinstance(features, pd.DataFrame) and len(features) == len(smiles):
        valid_indices = declared_valid
        if valid_indices:
            features = features.iloc[valid_indices].reset_index(drop=True)
        else:
            features = features.iloc[0:0].reset_index(drop=True)
    elif not isinstance(features, pd.DataFrame):
        features = pd.DataFrame()
        valid_indices = []
    else:
        candidate_indices = list(features.index)
        index_is_valid = (
            len(candidate_indices) == len(set(candidate_indices))
            and all(
                isinstance(index, Integral) and 0 <= int(index) < len(smiles)
                for index in candidate_indices
            )
        )
        if index_is_valid:
            valid_indices = [int(index) for index in candidate_indices]
        else:
            valid_indices = declared_valid[: len(features)]
            warnings.append(
                "legacy adapter returned compact rows without valid original-row "
                "indices; using declared non-null source positions"
            )
        features = features.reset_index(drop=True)
    return features, valid_indices, warnings


def execute_derived_feature_step(
    data: pd.DataFrame,
    step: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Execute a registry-declared process derivation step."""
    from .process_features import compute_process_features

    definitions = list(step.get("feature_definitions") or step.get("derived_feature_definitions") or [])
    if not definitions:
        raise ValueError("derived workflow step requires feature_definitions")
    manifest = step.get("manifest") or {
        "source_bindings": [
            {"raw_column": column, "source_field": column}
            for column in list(step.get("source_columns") or [])
        ]
    }
    result = compute_process_features(data, definitions, manifest)
    if result.errors:
        raise ValueError("; ".join(item.get("message", "派生特征计算失败") for item in result.errors))
    return result.features, list(range(len(data))), []


def _apply_workflow_prefix(
    features: pd.DataFrame,
    prefix: Any,
) -> pd.DataFrame:
    if not prefix:
        return features.copy()
    prefix_text = str(prefix).strip()
    if not prefix_text:
        return features.copy()
    separator_prefix = prefix_text if prefix_text.endswith("_") else f"{prefix_text}_"
    result = features.copy()
    result.columns = [
        str(column)
        if str(column).startswith(separator_prefix)
        else f"{separator_prefix}{column}"
        for column in result.columns
    ]
    return result


def _restore_step_rows(
    features: pd.DataFrame,
    valid_indices: Sequence[int],
    row_index: pd.Index,
) -> pd.DataFrame:
    valid = [int(index) for index in valid_indices]
    if len(valid) != len(features):
        raise ValueError(
            "feature extractor returned a row count inconsistent with valid indices"
        )
    if any(original_position < 0 or original_position >= len(row_index) for original_position in valid):
        raise ValueError("feature extractor returned an out-of-range row index")
    restored = align_extracted_features_to_rows(
        features,
        valid,
        len(row_index),
    )
    restored.index = row_index
    return restored


def _merge_workflow_feature_frames(
    frames: Sequence[pd.DataFrame],
    step_ids: Sequence[str],
    feature_source_map: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge step outputs while resolving cross-step names deterministically."""
    selected: dict[str, pd.Series] = {}
    selected_step: dict[str, str] = {}
    emitted_order: list[str] = []
    duplicate_steps: dict[str, list[str]] = {}
    source_map = dict(feature_source_map or {})

    for step_id, frame in zip(step_ids, frames):
        current_step = str(step_id)
        for position, column in enumerate(frame.columns):
            name = str(column)
            series = frame.iloc[:, position].rename(name)
            if name not in selected:
                selected[name] = series
                selected_step[name] = current_step
                emitted_order.append(name)
                duplicate_steps[name] = [current_step]
                continue

            if current_step not in duplicate_steps[name]:
                duplicate_steps[name].append(current_step)
            owner = str(source_map.get(name) or "").strip()
            if owner == current_step:
                selected[name] = series
                selected_step[name] = current_step

    if not selected:
        index = frames[0].index if frames else None
        return pd.DataFrame(index=index), []

    merged = pd.concat(
        [selected[name] for name in emitted_order],
        axis=1,
    )
    warnings = []
    for name in emitted_order:
        steps = duplicate_steps.get(name, [])
        if len(steps) < 2:
            continue
        retained_step = selected_step[name]
        owner = str(source_map.get(name) or "").strip()
        if owner == retained_step:
            reason = "according to feature_source_map"
        else:
            reason = "using first emitted step because feature_source_map has no matching owner"
        warnings.append(
            "workflow emitted duplicate feature "
            f"'{name}' in steps {', '.join(steps)}; retained '{retained_step}' "
            f"{reason}"
        )
    return merged, warnings


def execute_molecular_feature_workflow(
    data: pd.DataFrame,
    workflow: MolecularFeatureWorkflow | Mapping[str, Any],
    *,
    device: Any = None,
    mode: str = "screening",
    progress_callback: Callable[[dict], None] | None = None,
) -> WorkflowExecutionResult:
    """Execute normalized molecular feature steps and restore original row alignment."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if isinstance(workflow, MolecularFeatureWorkflow):
        duplicate_items = _collect_duplicate_workflow_items(workflow.to_dict())
        duplicate_items.update(workflow._duplicate_items)
    else:
        duplicate_items = _collect_duplicate_workflow_items(workflow)
    if duplicate_items:
        details = "; ".join(
            f"{field_name}={values}"
            for field_name, values in duplicate_items.items()
        )
        raise ValueError(f"workflow contains duplicate required items: {details}")
    normalized = (
        workflow.to_dict()
        if isinstance(workflow, MolecularFeatureWorkflow)
        else normalize_workflow_config(workflow)
    )
    steps = normalized["steps"]
    step_by_id = {}
    for step in steps:
        step_id = step.get("step_id")
        if not step_id:
            raise ValueError("every workflow step must declare step_id")
        if step_id in step_by_id:
            raise ValueError(f"duplicate workflow step id: {step_id}")
        step_by_id[step_id] = step

    ordered_ids = list(normalized["merge_order"])
    missing_steps = [step_id for step_id in ordered_ids if step_id not in step_by_id]
    if missing_steps:
        raise ValueError(
            "workflow merge order references missing steps: "
            + ", ".join(map(str, missing_steps))
        )

    frames = []
    step_trace = []
    warnings: list[str] = []
    valid_row_indices: dict[str, list[int]] = {}
    for step_id in ordered_ids:
        step = step_by_id[step_id]
        started = time.perf_counter()
        declared_method = str(step.get("method") or "").strip().lower()
        if declared_method == "derived_workflow":
            features, valid_indices, step_warnings = execute_derived_feature_step(data, step)
            input_count = len(data)
        else:
            smiles, prepared_indices, metadata = prepare_step_inputs(data, step)
            dispatch_step = dict(step)
            dispatch_step["_prepared_inputs"] = metadata
            features, valid_indices, step_warnings = execute_feature_step(
                smiles,
                dispatch_step,
                device=device,
            )
            input_count = len(smiles)
        valid_indices = [int(index) for index in valid_indices]
        prefixed = _apply_workflow_prefix(features, step.get("prefix"))
        duplicate_columns = prefixed.columns[prefixed.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(
                f"workflow step {step_id} produced duplicate feature columns: "
                + ", ".join(map(str, duplicate_columns))
            )
        restored = _restore_step_rows(prefixed, valid_indices, data.index)
        required = [str(name) for name in (step.get("feature_names") or []) if str(name)]
        missing = [name for name in required if name not in restored.columns]
        if missing:
            raise ValueError(
                f"workflow step {step_id} is missing required features: "
                + ", ".join(missing[:12])
            )
        frames.append(restored)
        valid_row_indices[step_id] = valid_indices
        warnings.extend(f"{step_id}: {warning}" for warning in step_warnings)
        trace = {
            "step_id": step_id,
            "mode": mode,
            "input_count": input_count,
            "valid_count": len(valid_indices),
            "output_columns": list(restored.columns),
            "elapsed_time": time.perf_counter() - started,
            "warnings": list(step_warnings),
        }
        step_trace.append(trace)
        if progress_callback is not None:
            progress_callback(dict(trace))

    merged, merge_warnings = _merge_workflow_feature_frames(
        frames,
        ordered_ids,
        normalized.get("feature_source_map"),
    )
    warnings.extend(merge_warnings)
    final_feature_names = [str(name) for name in normalized["final_feature_names"]]
    missing = [name for name in final_feature_names if name not in merged.columns]
    if missing:
        raise ValueError(
            "workflow is missing required final features: "
            + ", ".join(missing[:12])
        )
    return WorkflowExecutionResult(
        features=merged.reindex(columns=final_feature_names),
        step_trace=step_trace,
        warnings=warnings,
        workflow_hash=normalized["workflow_hash"],
        valid_row_indices=valid_row_indices,
    )


def validate_workflow_config(
    payload: Mapping[str, Any],
    model_feature_cols: Sequence[str] | None = None,
    available_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return strict compatibility diagnostics for a workflow configuration."""
    normalized = normalize_workflow_config(payload)
    steps = normalized["steps"]
    step_by_id = {
        step.get("step_id"): step
        for step in steps
        if isinstance(step, Mapping) and step.get("step_id")
    }
    missing_steps = [
        step_id for step_id in normalized["merge_order"] if step_id not in step_by_id
    ]

    missing_columns = []
    if available_columns is not None:
        available = set(available_columns)
        for step in steps:
            for column in step.get("source_columns", []):
                if column not in available and column not in missing_columns:
                    missing_columns.append(column)

    actual_features = normalized["final_feature_names"]
    expected_features = list(model_feature_cols or [])
    available_features = set(actual_features)
    missing_features = [
        feature for feature in expected_features if feature not in available_features
    ]
    order_mismatch = []
    if model_feature_cols is not None and expected_features != actual_features:
        order_mismatch.append(
            {"expected": expected_features, "actual": actual_features}
        )

    expected_fingerprint = payload.get("expected_model_fingerprint")
    model_fingerprint_match = None
    if expected_fingerprint is not None:
        model_fingerprint_match = (
            normalized["model_fingerprint"] == expected_fingerprint
        )

    return {
        "ok": not (
            missing_steps
            or missing_columns
            or missing_features
            or order_mismatch
            or model_fingerprint_match is False
        ),
        "missing_steps": missing_steps,
        "missing_columns": missing_columns,
        "missing_features": missing_features,
        "order_mismatch": order_mismatch,
        "model_fingerprint_match": model_fingerprint_match,
    }


def build_workflow_diff(
    training_workflow: Mapping[str, Any] | None,
    screening_workflow: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare two workflows without losing declared step order."""
    training = normalize_workflow_config(training_workflow or {})
    screening = normalize_workflow_config(screening_workflow or {})
    training_steps = {
        str(step.get("step_id")): step
        for step in training.get("steps", [])
        if isinstance(step, Mapping) and step.get("step_id")
    }
    screening_steps = {
        str(step.get("step_id")): step
        for step in screening.get("steps", [])
        if isinstance(step, Mapping) and step.get("step_id")
    }

    training_order = [
        str(step_id)
        for step_id in (
            training.get("merge_order")
            or [step.get("step_id") for step in training.get("steps", [])]
        )
        if step_id
    ]
    screening_order = [
        str(step_id)
        for step_id in (
            screening.get("merge_order")
            or [step.get("step_id") for step in screening.get("steps", [])]
        )
        if step_id
    ]

    changed_steps = []
    for step_id in training_order:
        if step_id not in screening_steps:
            continue
        left = training_steps[step_id]
        right = screening_steps[step_id]
        comparable_fields = (
            "role",
            "source_columns",
            "method",
            "prefix",
            "params",
            "semantic_params",
            "feature_names",
            "valid_row_behavior",
        )
        if any(left.get(field) != right.get(field) for field in comparable_fields):
            changed_steps.append(str(step_id))

    return {
        "changed_steps": changed_steps,
        "added_steps": [
            step_id for step_id in screening_order if step_id not in training_steps
        ],
        "removed_steps": [
            step_id for step_id in training_order if step_id not in screening_steps
        ],
        "feature_order_changed": (
            list(training.get("final_feature_names", []))
            != list(screening.get("final_feature_names", []))
        ),
        "training_hash": training.get("workflow_hash"),
        "screening_hash": screening.get("workflow_hash"),
    }


def can_lock_workflow(validation_report: Mapping[str, Any] | None) -> bool:
    """Return whether a workflow passed all blocking compatibility checks."""
    if not isinstance(validation_report, Mapping):
        return False
    return bool(
        validation_report.get("ok")
        and not validation_report.get("missing_steps")
        and not validation_report.get("missing_columns")
        and not validation_report.get("missing_features")
        and not validation_report.get("order_mismatch")
        and validation_report.get("model_fingerprint_match") is not False
    )
