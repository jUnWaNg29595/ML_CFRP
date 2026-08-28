"""Shared deterministic derivation of process and formulation features."""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

CURE_STAGE_LIMIT = 4
POST_CURE_STAGE_LIMIT = 2
_SCHEDULE_RE = re.compile(r"([-+]?\d*\.?\d+)\s*[^0-9;,:/]*C?\s*/\s*([-+]?\d*\.?\d+)")
_IMPLEMENTATIONS = {
    "core.process_features:derive_declared_feature",
    "core.process_features:derive_cure_stage_count",
    "core.process_features:derive_cure_total_time_h",
}
_IMPLEMENTATION_VERSIONS = {name: {"1"} for name in _IMPLEMENTATIONS}


@dataclass(frozen=True)
class DerivedFeatureResult:
    features: pd.DataFrame
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


def _schedule_pairs(value: Any) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    pairs = [(float(a), float(b)) for a, b in _SCHEDULE_RE.findall(text)]
    if not pairs:
        raise ValueError("固化工艺阶段格式无法解析")
    return pairs


def derive_cure_stage_count(series: pd.Series) -> pd.Series:
    return series.map(lambda value: len(_schedule_pairs(value)))


def derive_cure_total_time_h(series: pd.Series) -> pd.Series:
    return series.map(lambda value: sum(duration for _, duration in _schedule_pairs(value)))


def _derive_named(name: str, frame: pd.DataFrame, inputs: list[str]) -> pd.Series:
    source = frame[inputs[0]]
    if name == "cure_stage_count":
        return derive_cure_stage_count(source)
    if name == "cure_total_time_h":
        return derive_cure_total_time_h(source)
    prefix = "post_cure" if name.startswith("post_cure") else "cure"
    pairs = source.map(_schedule_pairs)
    if name.startswith("total_") or name.startswith("overall_"):
        other = frame[inputs[1]] if len(inputs) > 1 else pd.Series([None] * len(frame), index=frame.index)
        pairs = pd.Series([_schedule_pairs(a) + _schedule_pairs(b) for a, b in zip(source, other)], index=frame.index)
    if name.endswith("stage_count"):
        return pairs.map(len)
    if name.endswith("total_time_h"):
        return pairs.map(lambda values: sum(item[1] for item in values))
    if name.endswith("max_temperature_c"):
        return pairs.map(lambda values: max((item[0] for item in values), default=float("nan")))
    if name.endswith("final_temperature_c") or name == "post_cure_temperature_c":
        return pairs.map(lambda values: values[-1][0] if values else float("nan"))
    if name.endswith("temp_time_integral_c_h"):
        return pairs.map(lambda values: sum(temp * duration for temp, duration in values))
    if name.endswith("time_weighted_avg_temperature_c"):
        return pairs.map(lambda values: (sum(t * d for t, d in values) / sum(d for _, d in values)) if values and sum(d for _, d in values) else float("nan"))
    if name == "has_post_cure":
        return pairs.map(lambda values: int(bool(values)))
    raise ValueError(f"未注册的工艺派生特征: {name}")


def _dispatch_declared_rule(frame: pd.DataFrame, definition: dict, raw_columns: list[str]) -> pd.Series:
    rule = definition["calculation_rule"]
    implementation = str(rule.get("implementation") or "").strip()
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError(f"未允许的派生实现: {implementation}")
    version = str(rule.get("version") or "1").strip()
    if version not in _IMPLEMENTATION_VERSIONS[implementation]:
        raise ValueError(f"未允许的派生实现版本: {implementation}:{version or '<missing>'}")
    if implementation.endswith(":derive_cure_stage_count"):
        return derive_cure_stage_count(frame[raw_columns[0]])
    if implementation.endswith(":derive_cure_total_time_h"):
        return derive_cure_total_time_h(frame[raw_columns[0]])
    return _derive_named(str(definition["name"]), frame, raw_columns)


def compute_process_features(frame: pd.DataFrame, feature_definitions: list[dict], manifest: dict) -> DerivedFeatureResult:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    output: dict[str, pd.Series] = {}
    errors: list[dict] = []
    bindings = {item.get("source_field"): item.get("raw_column") for item in (manifest or {}).get("source_bindings", [])}
    for definition in feature_definitions or []:
        rule = definition.get("calculation_rule") or {}
        source_fields = list(rule.get("input_fields") or [])
        inputs = [bindings.get(field) or field for field in source_fields]
        missing = [field for field, column in zip(source_fields, inputs) if not column or column not in frame.columns]
        if missing:
            errors.append({"code": "missing_source_column", "feature": definition.get("name"), "source": "derived_workflow", "rule": rule.get("implementation"), "message": "缺少声明的原始输入", "columns": missing})
            continue
        if str(rule.get("null_policy") or "").strip().lower() == "reject":
            null_rows = [int(index) for index, row in frame.loc[:, inputs].iterrows() if any(pd.isna(value) or not str(value).strip() for value in row)]
            if null_rows:
                errors.append({"code": "null_source_value", "feature": definition.get("name"), "source": inputs, "rule": rule.get("implementation"), "message": "声明为 reject 的工艺输入存在空值", "rows": null_rows})
                continue
        try:
            output[str(definition["name"])] = _dispatch_declared_rule(frame, definition, inputs)
        except Exception as exc:
            errors.append({"code": "derivation_error", "feature": definition.get("name"), "source": inputs, "rule": rule.get("implementation"), "message": str(exc)})
    return DerivedFeatureResult(pd.DataFrame(output, index=frame.index), errors, [])


def _canonicalize_component(value: str) -> str | None:
    try:
        from . import smiles_utils
        if not smiles_utils.RDKIT_AVAILABLE:
            return None
        return smiles_utils.canonicalize_smiles(value)
    except Exception:
        return None


def split_component_structures(value: Any, allow_empty: bool = False) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        if allow_empty:
            return []
        raise ValueError("SMILES 不能为空")
    raw = [part.strip() for part in re.split(r"[.;。；]+", text) if part.strip()]
    canonical = [_canonicalize_component(part) for part in raw]
    if not canonical or any(item is None for item in canonical):
        raise ValueError("SMILES 结构非法")
    return sorted(item for item in canonical if item is not None)


def count_smiles_components(value: Any, role: str) -> int:
    return len(split_component_structures(value, allow_empty=str(role).strip().lower() in {"curing_agent", "hardener", "固化剂"}))


def materialize_component_count_features(frame: pd.DataFrame, resin_column: str, curing_agent_column: str) -> pd.DataFrame:
    out = frame.copy()
    out["resin_smiles_n_components"] = out[resin_column].map(lambda value: count_smiles_components(value, "resin"))
    out["curing_agent_smiles_n_components"] = out[curing_agent_column].map(lambda value: count_smiles_components(value, "curing_agent"))
    return out


__all__ = ["DerivedFeatureResult", "compute_process_features", "split_component_structures", "count_smiles_components", "materialize_component_count_features", "derive_declared_feature", "derive_cure_stage_count", "derive_cure_total_time_h"]


def derive_declared_feature(frame: pd.DataFrame, definition: dict, manifest: dict | None = None) -> pd.Series:
    result = compute_process_features(frame, [definition], manifest or {"source_bindings": []})
    if result.errors:
        raise ValueError(result.errors[0]["message"])
    return result.features.iloc[:, 0]
