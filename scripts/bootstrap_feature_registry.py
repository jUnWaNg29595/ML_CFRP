"""Bootstrap a semantic registry from the historical Tg artifact.

This is an audit/import utility.  It reads the artifact only and writes a new
JSON registry; it never mutates the joblib file or prediction configuration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib

# Allow direct invocation as `python scripts/bootstrap_feature_registry.py`
# from the repository root, where Python otherwise searches `scripts/` first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.feature_registry import compute_registry_hash

GAPS: dict[str, tuple[str, str | None]] = {
    "cure_stage_count": ("derived_workflow", "由固化温度/时间序列解析阶段数"),
    "cure_total_time_h": ("derived_workflow", "由固化阶段时间求和"),
    "cure_max_temperature_c": ("derived_workflow", "由固化阶段温度序列求最大值"),
    "cure_final_temperature_c": ("derived_workflow", "由固化阶段温度序列取末值"),
    "cure_temp_time_integral_c_h": ("derived_workflow", "由固化温度-时间序列积分"),
    "cure_time_weighted_avg_temperature_c": ("derived_workflow", "由固化温度-时间序列计算时间加权平均"),
    "post_cure_stage_count": ("derived_workflow", "由后固化温度/时间序列解析阶段数"),
    "post_cure_total_time_h": ("derived_workflow", "由后固化阶段时间求和"),
    "post_cure_max_temperature_c": ("derived_workflow", "由后固化阶段温度序列求最大值"),
    "post_cure_final_temperature_c": ("derived_workflow", "由后固化阶段温度序列取末值"),
    "post_cure_temp_time_integral_c_h": ("derived_workflow", "由后固化温度-时间序列积分"),
    "post_cure_time_weighted_avg_temperature_c": ("derived_workflow", "由后固化温度-时间序列计算时间加权平均"),
    "post_cure_temperature_c": ("derived_workflow", "后固化温度派生字段"),
    "has_post_cure": ("derived_workflow", "由是否存在后固化阶段派生"),
    "total_cure_stage_count": ("derived_workflow", "固化与后固化阶段数合计"),
    "total_heat_treatment_time_h": ("derived_workflow", "固化与后固化时间合计"),
    "overall_max_cure_temperature_c": ("derived_workflow", "固化与后固化温度最大值"),
    "overall_temp_time_integral_c_h": ("derived_workflow", "固化与后固化温度-时间积分"),
    "degree_of_cure_pct": ("manual_input", "实验记录的固化度百分比"),
    "gel_time_min": ("manual_input", "实验记录的凝胶时间"),
    "curing_pressure_mpa": ("manual_input", "实验记录的固化压力"),
    "eew_g_eq": ("manual_input", "实验记录的环氧当量"),
    "ahew_g_eq": ("manual_input", "实验记录的活泼氢当量"),
    "tg_method": ("manual_input", "Tg 测试方法编码"),
    "tg_standard": ("manual_input", "Tg 测试标准编码"),
    "stoichiometric_ratio_r": ("unknown", "待确认 PHR、EEW、AHEW 定义、质量基准、方向、单位和训练数据实际公式"),
    "resin_smiles_n_components": ("molecular_workflow", "由树脂结构输入拆分组件计数"),
    "curing_agent_smiles_n_components": ("molecular_workflow", "由固化剂结构输入拆分组件计数"),
}

DERIVED_IMPLEMENTATION = "core.process_features:derive_declared_feature"
MOLECULAR_IMPLEMENTATION = "core.molecular_features:derive_declared_feature"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "feature"


def _feature_id(name: str) -> str:
    return "cfrp.tg." + _slug(name)


def _legacy_definition(name: str, artifact_path: str) -> dict[str, Any]:
    return {
        "feature_id": _feature_id(name),
        "name": name,
        "label": name,
        "source_type": "molecular_workflow",
        "data_type": "float",
        "unit": None,
        "required_for_prediction": True,
        "nullable": False,
        "default_policy": "workflow_only",
        "calculation_rule": {
            "implementation": MOLECULAR_IMPLEMENTATION,
            "version": "legacy-observed-1",
            "input_fields": ["resin_smiles", "curing_agent_smiles"],
            "null_policy": "reject",
            "invalid_policy": "reject",
        },
        "description": "历史 artifact 中观察到的模型特征，尚未完成语义复核",
        "legacy_source": artifact_path,
        "status": "legacy_observed",
    }


def _gap_definition(name: str) -> dict[str, Any]:
    source_type, reason = GAPS[name]
    blocked = source_type == "unknown"
    definition: dict[str, Any] = {
        "feature_id": _feature_id(name),
        "name": name,
        "label": name,
        "source_type": source_type,
        "data_type": "integer" if name.endswith("count") or name in {"has_post_cure", "tg_method", "tg_standard"} else "float",
        "unit": "stage" if name.endswith("stage_count") else None,
        "required_for_prediction": True,
        "nullable": False,
        "default_policy": "explicit_only" if source_type == "manual_input" else ("forbidden" if blocked else "workflow_only"),
        "description": reason,
        "status": "blocked" if blocked else "draft",
    }
    if source_type in {"derived_workflow", "molecular_workflow"}:
        definition["calculation_rule"] = {
            "implementation": DERIVED_IMPLEMENTATION if source_type == "derived_workflow" else MOLECULAR_IMPLEMENTATION,
            "version": "1",
            "input_fields": ["cure_schedule"] if source_type == "derived_workflow" else ["resin_smiles", "curing_agent_smiles"],
            "null_policy": "reject",
            "invalid_policy": "reject",
        }
    if source_type == "unknown":
        definition["blocking_reason"] = reason
    if name in {"tg_method", "tg_standard"}:
        definition["enum_mapping"] = {"values": {}, "unknown_policy": "reject"}
    return definition


def build_registry(artifact_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("artifact must contain a dictionary payload")
    model_features = list(artifact.get("feature_cols") or [])
    extra = artifact.get("extra") if isinstance(artifact.get("extra"), dict) else {}
    workflow = extra.get("molecular_feature_workflow") if isinstance(extra.get("molecular_feature_workflow"), dict) else {}
    workflow_features = list(workflow.get("final_feature_names") or [])
    if len(model_features) != 532 or len(workflow_features) != 504:
        raise ValueError(f"unexpected artifact counts: {len(model_features)} model / {len(workflow_features)} workflow")
    if not set(workflow_features).issubset(set(model_features)):
        raise ValueError("workflow features are not a subset of model features")
    # Keep the model's exact feature order.  The workflow list is only used to
    # identify which of those ordered columns were produced by the old
    # workflow.
    workflow_set = set(workflow_features)
    definitions = [
        _legacy_definition(name, str(artifact_path)) if name in workflow_set else _gap_definition(name)
        for name in model_features
    ]
    # Ensure every historical gap is represented, even if a future artifact changes order.
    missing_gaps = [name for name in GAPS if name not in model_features]
    if missing_gaps:
        raise ValueError("artifact is missing historical gaps: " + ", ".join(missing_gaps))
    profile_ids = [item["feature_id"] for item in definitions]
    blocked = [item["feature_id"] for item in definitions if item.get("status") == "blocked"]
    registry: dict[str, Any] = {
        "schema_version": 1,
        "registry_version": "2026.08.27",
        "features": definitions,
        "model_profiles": {
            "epoxy_resin.tg": {
                "material_type": "epoxy_resin",
                "target": "tg",
                "target_col": artifact.get("target_col", "tg_c"),
                "feature_ids": profile_ids,
                "status": "blocked",
                "blocked_feature_ids": blocked,
                "source_artifact": str(artifact_path),
            }
        },
        "approval": {
            "status": "draft",
            "approved_by": None,
            "approved_at": None,
            "change_summary": "建立历史 Tg artifact 特征来源登记，待本地单人审核",
            "review_note": "legacy_observed 特征仅用于审计；计量比特征保持 blocked",
        },
    }
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    registry = build_registry(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("532 model features / 504 workflow features / 28 historical gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
