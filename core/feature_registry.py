"""Versioned semantic feature registry and validation helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SOURCE_TYPES = {"molecular_workflow", "derived_workflow", "manual_input", "target", "metadata", "unknown"}
DEFAULT_POLICIES = {"forbidden", "explicit_only", "workflow_only"}
FEATURE_STATUSES = {"draft", "approved", "legacy_observed", "deprecated", "blocked"}
PROFILE_STATUSES = {"draft", "approved", "deprecated", "blocked"}
_REVIEW_KEYS = {"approved_hash", "approved_at", "approved_by", "reviewer", "reviewed_at", "review_note", "review_notes", "change_summary", "change_summaries", "review_metadata"}

def _without_mutable_approval_metadata(payload: Any) -> Any:
    value = copy.deepcopy(payload)
    if isinstance(value, dict) and isinstance(value.get("approval"), dict):
        for key in _REVIEW_KEYS:
            value["approval"].pop(key, None)
    return value

def compute_registry_hash(payload: Mapping[str, Any]) -> str:
    clean = _without_mutable_approval_metadata(payload)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def validate_registry(payload: Mapping[str, Any], require_approved: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return {"ok": False, "errors": ["registry must be a JSON object"], "warnings": [], "registry_hash": compute_registry_hash({})}
    registry_hash = compute_registry_hash(payload)
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(payload.get("registry_version"), str) or not payload.get("registry_version"):
        errors.append("registry_version is required")
    features = payload.get("features")
    if not isinstance(features, list):
        errors.append("features must be a list")
        features = []
    profiles = payload.get("model_profiles")
    if not isinstance(profiles, Mapping):
        errors.append("model_profiles must be an object")
        profiles = {}
    approval = payload.get("approval")
    if not isinstance(approval, Mapping):
        errors.append("approval must be an object")
        approval = {}
    ids: set[str] = set()
    names: set[str] = set()
    definitions: dict[str, Mapping[str, Any]] = {}
    for index, feature in enumerate(features):
        prefix = f"features[{index}]"
        if not isinstance(feature, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        feature_id, name = feature.get("feature_id"), feature.get("name")
        if not isinstance(feature_id, str) or not feature_id.strip():
            errors.append(f"{prefix}.feature_id is required")
        elif feature_id in ids:
            errors.append(f"duplicate feature_id: {feature_id}")
        else:
            ids.add(feature_id)
            definitions[feature_id] = feature
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name is required")
        elif name in names:
            errors.append(f"duplicate feature name: {name}")
        else:
            names.add(name)
        source_type, policy, status = feature.get("source_type"), feature.get("default_policy"), feature.get("status")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{prefix}.source_type is invalid: {source_type}")
        if policy not in DEFAULT_POLICIES:
            errors.append(f"{prefix}.default_policy is invalid: {policy}")
        if status not in FEATURE_STATUSES:
            errors.append(f"{prefix}.status is invalid: {status}")
        unit = feature.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            errors.append(f"{prefix}.unit is required")
        if "required_for_prediction" in feature and not isinstance(feature.get("required_for_prediction"), bool):
            errors.append(f"{prefix}.required_for_prediction must be boolean")
        if "nullable" in feature and not isinstance(feature.get("nullable"), bool):
            errors.append(f"{prefix}.nullable must be boolean")
        if feature.get("required_for_prediction") is True and feature.get("nullable") is True:
            errors.append(f"{prefix} required_for_prediction cannot be nullable")
        if source_type == "manual_input" and isinstance(feature.get("default"), (int, float)) and not isinstance(feature.get("default"), bool):
            errors.append(f"{prefix}.default numeric values are forbidden for manual_input")
        if source_type in {"derived_workflow", "molecular_workflow"}:
            rule = feature.get("calculation_rule")
            if not isinstance(rule, Mapping):
                errors.append(f"{prefix}.calculation_rule is required for workflow feature")
            else:
                if not isinstance(rule.get("input_fields"), list) or not rule.get("input_fields"):
                    errors.append(f"{prefix}.calculation_rule.input_fields is required")
                elif any(not isinstance(field, str) or not field.strip() for field in rule["input_fields"]):
                    errors.append(f"{prefix}.calculation_rule.input_fields must contain non-empty strings")
                if not isinstance(rule.get("implementation"), str) or not rule.get("implementation").strip():
                    errors.append(f"{prefix}.calculation_rule.implementation is required")
                for policy_name in ("null_policy", "invalid_policy"):
                    if not isinstance(rule.get(policy_name), str) or not rule.get(policy_name).strip():
                        errors.append(f"{prefix}.calculation_rule.{policy_name} is required")
        if source_type == "unknown" and status != "blocked":
            errors.append(f"{prefix} unknown feature must be blocked")
        if status == "blocked" and not any(feature.get(key) for key in ("blocking_reason", "block_reason", "reason")):
            errors.append(f"{prefix} blocked feature requires a blocking reason")
        if status == "legacy_observed" and not any(feature.get(key) for key in ("legacy_source", "source_artifact", "source_dataset")):
            errors.append(f"{prefix} legacy_observed feature requires a legacy source")
        valid_range = feature.get("valid_range")
        if valid_range is not None:
            if not isinstance(valid_range, Mapping):
                errors.append(f"{prefix}.valid_range must be an object")
            else:
                lower, upper = valid_range.get("min"), valid_range.get("max")
                for label, bound in (("min", lower), ("max", upper)):
                    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound)):
                        errors.append(f"{prefix}.valid_range.{label} must be finite")
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
                    errors.append(f"{prefix}.valid_range min cannot exceed max")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            errors.append(f"model profile {profile_id} must be an object")
            continue
        refs = profile.get("feature_ids")
        if not isinstance(refs, list):
            errors.append(f"model profile {profile_id}.feature_ids must be a list")
            refs = []
        seen_refs: set[str] = set()
        for feature_id in refs:
            if not isinstance(feature_id, str) or not feature_id.strip():
                errors.append(f"model profile {profile_id}.feature_ids entries must be non-empty strings")
                continue
            if feature_id not in definitions:
                errors.append(f"model profile {profile_id} references unknown feature_id: {feature_id}")
            if feature_id in seen_refs:
                errors.append(f"model profile {profile_id} references duplicate feature_id: {feature_id}")
            seen_refs.add(feature_id)
        profile_status = profile.get("status")
        if profile_status not in PROFILE_STATUSES:
            errors.append(f"model profile {profile_id}.status is invalid: {profile_status}")
        target_col = profile.get("target_col")
        if not isinstance(target_col, str) or not target_col.strip():
            errors.append(f"model profile {profile_id}.target_col is required")
        if profile_status == "approved":
            for feature_id in refs:
                if not isinstance(feature_id, str):
                    continue
                definition = definitions.get(feature_id)
                if definition is not None and definition.get("status") != "approved":
                    errors.append(f"model profile {profile_id} approved profile requires approved feature: {feature_id}")
        blocked = profile.get("blocked_feature_ids", [])
        if not isinstance(blocked, list):
            errors.append(f"model profile {profile_id}.blocked_feature_ids must be a list")
            blocked = []
        if isinstance(blocked, list):
            if any(not isinstance(feature_id, str) or not feature_id.strip() for feature_id in blocked):
                errors.append(f"model profile {profile_id}.blocked_feature_ids entries must be non-empty strings")
            if len(blocked) != len(set(item for item in blocked if isinstance(item, str))):
                errors.append(f"model profile {profile_id}.blocked_feature_ids contains duplicate entries")
            blocked_set = {item for item in blocked if isinstance(item, str)}
            actual_blocked = {
                feature_id for feature_id in refs
                if isinstance(feature_id, str)
                if definitions.get(feature_id, {}).get("status") == "blocked"
            }
            for feature_id in blocked:
                if feature_id not in refs:
                    errors.append(f"model profile {profile_id}.blocked_feature_ids references feature outside profile: {feature_id}")
                elif definitions.get(feature_id, {}).get("status") != "blocked":
                    errors.append(f"model profile {profile_id}.blocked_feature_ids requires blocked feature: {feature_id}")
            if blocked_set != actual_blocked:
                errors.append(f"model profile {profile_id}.blocked_feature_ids must match blocked definitions")
    approval_status = approval.get("status")
    if approval_status not in {"draft", "approved", "deprecated"}:
        errors.append(f"approval.status is invalid: {approval_status}")
    if require_approved:
        if approval_status != "approved":
            errors.append("registry approval status is not approved")
        if approval.get("approved_hash") != registry_hash:
            errors.append("approval.approved_hash does not match computed registry_hash")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "registry_hash": registry_hash}

def load_registry(path: str | Path, require_approved: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = validate_registry(payload, require_approved=require_approved)
    if not report["ok"]:
        raise ValueError("invalid feature registry: " + "; ".join(report["errors"]))
    return payload

def get_model_profile(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = registry.get("model_profiles", {})[profile_id]
    except (KeyError, TypeError):
        raise KeyError(f"unknown model profile: {profile_id}") from None
    if not isinstance(profile, Mapping):
        raise ValueError(f"model profile is not an object: {profile_id}")
    return copy.deepcopy(dict(profile))

def build_registry_snapshot(registry: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    report = validate_registry(registry, require_approved=True)
    if not report["ok"]:
        raise ValueError("cannot build snapshot from unapproved registry: " + "; ".join(report["errors"]))
    profile = get_model_profile(registry, profile_id)
    if profile.get("status") != "approved":
        raise ValueError(f"model profile is not approved: {profile_id}")
    definitions = {item.get("feature_id"): item for item in registry.get("features", []) if isinstance(item, Mapping)}
    selected = []
    for feature_id in profile.get("feature_ids", []):
        if feature_id not in definitions:
            raise KeyError(f"model profile references unknown feature_id: {feature_id}")
        if definitions[feature_id].get("status") != "approved":
            raise ValueError(f"model profile references non-approved feature: {feature_id}")
        selected.append(copy.deepcopy(dict(definitions[feature_id])))
    snapshot = {"schema_version": registry.get("schema_version"), "registry_version": registry.get("registry_version"), "registry_hash": compute_registry_hash(registry), "registry_payload": copy.deepcopy(dict(registry)), "profile_id": profile_id, "model_profile": profile, "features": selected}
    selected_payload = {"profile_id": profile_id, "model_profile": profile, "features": selected}
    snapshot["selected_features_hash"] = hashlib.sha256(json.dumps(selected_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return snapshot


def register_new_feature(

    registry: Mapping[str, Any],
    feature_definition: Mapping[str, Any],
    *,
    reviewer: str,
    target_profile_id: str | None = None,
    review_note: str = "",
) -> dict[str, Any]:
    """Explicitly register a newly approved semantic feature into registry draft."""
    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ValueError("reviewer is required to register a new feature")
    feature_id = str(feature_definition.get("feature_id") or "").strip()
    name = str(feature_definition.get("name") or feature_id).strip()
    if not feature_id:
        raise ValueError("feature_id cannot be empty")
    if not name:
        raise ValueError("feature name cannot be empty")
    unit = str(feature_definition.get("unit") or "unknown").strip()
    if not unit:
        unit = "unknown"
    source_type = str(feature_definition.get("source_type") or "manual_input").strip()
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {source_type}")

    updated = copy.deepcopy(dict(registry))
    features = updated.setdefault("features", [])
    existing_ids = {item.get("feature_id") for item in features if isinstance(item, Mapping)}
    existing_names = {item.get("name") for item in features if isinstance(item, Mapping)}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    status = str(feature_definition.get("status") or "draft").strip().lower()
    if status not in {"draft", "approved"}:
        status = "draft"

    new_entry: dict[str, Any] = {
        "feature_id": feature_id,
        "name": name,
        "label": str(feature_definition.get("label") or name).strip(),
        "source_type": source_type,
        "data_type": str(feature_definition.get("data_type") or "float").strip(),
        "unit": unit,
        "required_for_prediction": bool(feature_definition.get("required_for_prediction", False)),
        "nullable": bool(feature_definition.get("nullable", True)),
        "default_policy": str(feature_definition.get("default_policy") or "explicit_only").strip(),
        "description": str(feature_definition.get("description") or review_note or "人工审核登记的新特征").strip(),
        "status": status,
    }
    if "aliases" in feature_definition:
        aliases = feature_definition["aliases"]
        new_entry["aliases"] = [str(a).strip() for a in (aliases if isinstance(aliases, list) else [aliases]) if str(a).strip()]
    if "accepted_aliases" in feature_definition:
        acc_aliases = feature_definition["accepted_aliases"]
        new_entry["accepted_aliases"] = [str(a).strip() for a in (acc_aliases if isinstance(acc_aliases, list) else [acc_aliases]) if str(a).strip()]
    if "legacy_name" in feature_definition and feature_definition["legacy_name"]:
        new_entry["legacy_name"] = str(feature_definition["legacy_name"]).strip()
    if source_type in {"derived_workflow", "molecular_workflow"}:
        rule = feature_definition.get("calculation_rule")
        if isinstance(rule, Mapping):
            new_entry["calculation_rule"] = dict(rule)
        else:
            new_entry["calculation_rule"] = {
                "implementation": "workflow.default",
                "version": "v1",
                "input_fields": feature_definition.get("input_fields") or ["resin_smiles"],
                "null_policy": "reject",
                "invalid_policy": "reject",
            }

    if feature_id in existing_ids:
        # Update existing definition in place
        for idx, item in enumerate(features):
            if isinstance(item, Mapping) and item.get("feature_id") == feature_id:
                features[idx] = new_entry
                break
    else:
        if name in existing_names:
            raise ValueError(f"duplicate feature name: {name}")
        features.append(new_entry)

    # Optionally add to profile feature_ids if explicit profile target given
    if target_profile_id:
        profiles = updated.setdefault("model_profiles", {})
        profile = profiles.get(target_profile_id)
        if isinstance(profile, dict):
            profile_feature_ids = profile.setdefault("feature_ids", [])
            if feature_id not in profile_feature_ids:
                profile_feature_ids.append(feature_id)

    # Bump version and validate
    ver = updated.get("registry_version", "2026.08.27")
    updated["registry_version"] = ver
    val_report = validate_registry(updated, require_approved=False)
    if not val_report["ok"]:
        raise ValueError("新特征加入后 registry 校验失败: " + "; ".join(val_report["errors"]))
    return updated


def save_registry_atomic(path: str | Path, registry: Mapping[str, Any]) -> None:
    """Save registry using temporary file and atomic replace."""
    from .feature_mapping_review import save_atomic_json
    target = Path(path).resolve()
    val = validate_registry(registry, require_approved=False)
    if not val["ok"]:
        raise ValueError("无法保存不合法的 registry: " + "; ".join(val["errors"]))
    save_atomic_json(target, dict(registry))


__all__ = [
    "SOURCE_TYPES",
    "DEFAULT_POLICIES",
    "FEATURE_STATUSES",
    "PROFILE_STATUSES",
    "compute_registry_hash",
    "validate_registry",
    "load_registry",
    "get_model_profile",
    "build_registry_snapshot",
    "register_new_feature",
    "save_registry_atomic",
    "MANUAL_INPUT_KEYWORDS",
    "MOLECULAR_WORKFLOW_KEYWORDS",
    "RECIPE_INPUT_KEYWORDS",
    "classify_feature_source_type",
    "normalize_manifest_entry",
    "validate_manifest_source_type_consistency",
]


# ---------------------------------------------------------------------------
# 特征来源分类规则（业务硬约束）：
# 工艺参数、测试字段、人工记录 → manual_input（即使是数值字段，不得因可计算而改为 derived）；
# 由 SMILES/结构自动提取 → molecular_workflow；
# 由配方原始输入按明确公式计算（calculation_rule 必填）→ derived_workflow；
# 配方原始输入（树脂/固化剂 SMILES、配比、PHR 等）本身不是派生特征；
# 无法确定来源 → unknown（阻断正式训练与发布）。
# ---------------------------------------------------------------------------
MANUAL_INPUT_KEYWORDS: tuple[str, ...] = (
    "固化温度", "固化时间", "升温速率", "保温", "后固化", "压力", "真空", "真空度",
    "湿度", "气氛", "混合", "脱泡", "成型", "加工", "工艺", "批次", "环境",
    "测试方法", "测试标准", "测试仪器", "测试温度", "测试湿度", "样品状态",
    "测试条件", "试验条件", "重复次数", "前处理", "测试", "试验",
    "备注", "等级", "目视", "观察", "异常", "人工", "外观", "评级",
    "curing temperature", "cure time", "curing time", "ramp rate", "heating rate",
    "postcure", "post-cure", "dwell", "hold time", "pressure", "vacuum",
    "humidity", "atmosphere", "degas", "batch", "specimen", "test method",
    "test standard", "test condition", "notes", "grade", "visual", "operator",
)

MOLECULAR_WORKFLOW_KEYWORDS: tuple[str, ...] = (
    "分子量", "分子描述", "原子数", "环数", "官能团", "指纹", "描述符",
    "morgan", "maccs", "rdkit", "拓扑",
    "molecular weight", "molwt", "atom count", "ring count", "hbd", "hba",
    "tpsa", "logp", "fingerprint", "descriptor", "xtb", "homo", "lumo",
    "fp_bit", "fr_", "num_", "n_atoms", "n_rings", "n_hbd", "n_hba",
)

RECIPE_INPUT_KEYWORDS: tuple[str, ...] = (
    "树脂名称", "固化剂名称", "树脂smiles", "固化剂smiles", "组分数量",
    "配比", "phr", "称量", "配方比例", "份数", "resin smiles",
    "hardener smiles", "curing agent smiles", "component count",
    "mix ratio", "weight ratio", "resin phr", "hardener phr",
)


def _feature_search_text(definition: Any) -> str:
    """拼出用于关键词匹配的小写文本（name + label + aliases + description）。"""
    if isinstance(definition, str):
        return definition.strip().lower()
    if not isinstance(definition, Mapping):
        return ""
    parts: list[str] = []
    for key in ("name", "label", "raw_column", "column"):
        value = definition.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    for key in ("aliases", "accepted_aliases", "legacy_name"):
        value = definition.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value if str(item).strip())
    description = definition.get("description") or definition.get("label_zh")
    if isinstance(description, str):
        parts.append(description)
    return " ".join(parts).lower()


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_feature_source_type(definition: Any, extra_hints: Mapping[str, Any] | None = None) -> str:
    """按业务语义把特征归类为 source_type（纯函数，不修改任何数据）。

    优先级：
    1. definition 已登记合法 source_type（且非 unknown 占位）→ 直接尊重；
    2. 有 calculation_rule/implementation 的系统计算 → derived_workflow；
    3. 名称/别名命中分子特征关键词 → molecular_workflow；
    4. 名称/别名命中工艺/测试/人工记录关键词 → manual_input（数值型不改变归类）；
    5. 名称命中配方原始输入 → manual_input（extra_hints["recipe_input"]=True）；
    6. 其他 → unknown（阻断正式训练与发布）。
    """
    hints: dict[str, Any] = dict(extra_hints) if isinstance(extra_hints, Mapping) else {}
    if isinstance(definition, Mapping):
        declared = str(definition.get("source_type") or "").strip()
        if declared in SOURCE_TYPES and declared != "unknown":
            return declared

    text = _feature_search_text(definition)
    rule = definition.get("calculation_rule") if isinstance(definition, Mapping) else None
    has_rule = isinstance(rule, Mapping) and bool(rule)
    implementation = str(definition.get("implementation") or "").strip() if isinstance(definition, Mapping) else ""

    # 配方原始输入优先于分子关键词：树脂/固化剂 SMILES 是用户提供的源字段，不是系统提取
    if _has_keyword(text, RECIPE_INPUT_KEYWORDS):
        if isinstance(extra_hints, dict):
            extra_hints["recipe_input"] = True
        hints["recipe_input"] = True
        return "manual_input"

    # 有明确 calculation_rule/implementation 的系统计算 → derived_workflow（配方统计、等当量比等）
    if (has_rule and not _has_keyword(text, MOLECULAR_WORKFLOW_KEYWORDS)) or (implementation and not _has_keyword(text, MOLECULAR_WORKFLOW_KEYWORDS)):
        return "derived_workflow"

    if _has_keyword(text, MOLECULAR_WORKFLOW_KEYWORDS):
        return "molecular_workflow"

    if _has_keyword(text, MANUAL_INPUT_KEYWORDS):
        # 数值型不改变归类：data_type/numeric 信息不参与判断，工艺/测试字段一律 manual_input
        return "manual_input"

    return "unknown"


def normalize_manifest_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """manifest 条目字段别名归一化（纯函数，返回规范化副本）。

    别名组（规范名在前）：source_role/source_type、raw_column/raw_columns、
    source_field/source_fields、feature_id/semantic_feature_id、unit/units、
    aliases/accepted_aliases。规范名优先取已有值，缺失时用别名值回填；
    回填后删除别名键。值取第一个非空。
    """
    if not isinstance(entry, Mapping):
        return {}
    value = copy.deepcopy(dict(entry))
    alias_groups = (
        ("source_role", "source_type"),
        ("raw_column", "raw_columns"),
        ("source_field", "source_fields"),
        ("feature_id", "semantic_feature_id"),
        ("unit", "units"),
        ("aliases", "accepted_aliases"),
    )
    for canonical, alias in alias_groups:
        canonical_value = value.get(canonical)
        alias_value = value.get(alias)
        if (canonical_value is None or canonical_value == "" or canonical_value == []) and alias_value not in (None, "", []):
            value[canonical] = alias_value
        if alias in value and alias != canonical:
            value.pop(alias, None)
    return value


def validate_manifest_source_type_consistency(manifest: Mapping[str, Any], registry: Mapping[str, Any]) -> list[str]:
    """校验 manifest 各条目声明的来源与 registry 特征 source_type 语义一致。

    返回中文错误列表（不抛异常）。配方原始输入行不得声明为 derived_workflow 输出；
    manual_input 特征不得声明为 workflow/derived 来源。
    """
    errors: list[str] = []
    if not isinstance(manifest, Mapping) or not isinstance(registry, Mapping):
        return ["manifest 或 registry 不是合法对象，无法校验来源一致性。"]
    definitions = {
        str(item.get("feature_id")): item
        for item in (registry.get("features") or [])
        if isinstance(item, Mapping) and item.get("feature_id")
    }
    name_to_definition = {
        str(item.get("name")): item
        for item in (registry.get("features") or [])
        if isinstance(item, Mapping) and item.get("name")
    }
    entries: list[Any] = []
    for key in ("feature_bindings", "mappings", "entries", "source_bindings"):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(value)
    for index, raw_entry in enumerate(entries):
        entry = normalize_manifest_entry(raw_entry) if isinstance(raw_entry, Mapping) else {}
        if not entry:
            errors.append(f"manifest 条目[{index}] 不是对象。")
            continue
        feature_id = str(entry.get("feature_id") or "").strip()
        declared_role = str(entry.get("source_role") or "").strip().lower()
        definition = definitions.get(feature_id) or name_to_definition.get(feature_id)
        if definition is None:
            continue  # 未登记特征由 manifest/registry 各自校验负责
        source_type = str(definition.get("source_type") or "").strip().lower()
        if source_type == "manual_input" and declared_role in {"molecular_workflow", "derived_workflow", "workflow", "derived", "computed"}:
            errors.append(
                f"特征 {definition.get('name') or feature_id} 在 registry 中为 manual_input（工艺/测试/人工记录字段），"
                f"但 manifest 声明为 {declared_role}，来源不一致。数值型工艺字段也不允许声明为派生特征。"
            )
        if source_type in {"molecular_workflow", "derived_workflow"} and declared_role == "manual_input":
            errors.append(
                f"特征 {definition.get('name') or feature_id} 在 registry 中为 {source_type}（系统计算），"
                f"但 manifest 声明为 manual_input，来源不一致。"
            )
    return errors