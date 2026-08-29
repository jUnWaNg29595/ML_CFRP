# -*- coding: utf-8 -*-
"""Virtual molecule generation + screening utilities."""

from __future__ import annotations
import copy
import hashlib
import json
import signal
import os
from pathlib import Path

import itertools
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False

from .smiles_utils import normalize_chemical_string, parse_chemical_string
from .molecular_feature_workflow import (
    MolecularFeatureWorkflow,
    materialize_workflow_source_columns,
)
from .post_feature_mapping import sanitize_feature_columns

# 候选级审计状态机（纯逻辑模块，无 rdkit/streamlit 依赖）。
from . import screening_audit as _screening_audit
from .screening_audit import (
    FILTERED_BY_CHEMISTRY,
    FILTERED_BY_FEASIBILITY,
    FILTERED_BY_MELTING_POINT,
    FEATURE_INVALID,
    MISSING_REQUIRED_INPUT,
    MODEL_PREDICTION_FAILED,
    OUT_OF_RANGE,
    SELECTED,
    STRUCTURE_INVALID,
    VALID,
    WORKFLOW_FAILED,
    CANDIDATE_STATUSES,
    find_candidate,
    merge_model_results,
    new_candidate_pool,
    record_filter,
    set_candidate_status,
    summarize_funnel,
)


DEFAULT_EPOXY_RULES = {
    "global": {
        "allowed_elements": {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53},
        "reject_charged": False,
    },
    "resin": {
        "min_epoxide": 1,
        "max_epoxide": None,
        "min_aromatic_rings": None,
        "min_mw": 40.0,
        "max_mw": 2500.0,
        "min_heavy_atoms": 3,
        "max_heavy_atoms": 200,
        "ban_strong_acids": False,
        "ban_amines": False,
    },
    "hardener": {
        "min_mw": 25.0,
        "max_mw": 1500.0,
        "min_heavy_atoms": 2,
        "max_heavy_atoms": 120,
        "ban_strong_acids": False,
        "ban_epoxide": False,
        "allowed_classes": None,
    },
    "pair": {
        "amine_ratio": (0.05, 20.00),
        "anhydride_ratio": (0.05, 20.00),
        "phenol_ratio": (0.05, 20.00),
        "thiol_ratio": (0.05, 20.00),
        "reject_mixed_class": False,
    },
}


# The former ``[OX2r3]1[CR2r3][CR2r3]1`` query uses RDKit's R2
# (membership in two rings), so it misses ordinary oxirane rings entirely.
EPOXIDE_SMARTS = "[O;r3]1[C;r3][C;r3]1"

POST_FEATURE_COMPUTED_DEFINITIONS = {
    "computed_resin_molecular_weight": {"category": "分子量", "unit": "g/mol", "definition": "RDKit MolWt(resin)"},
    "computed_hardener_molecular_weight": {"category": "分子量", "unit": "g/mol", "definition": "RDKit MolWt(hardener)"},
    "computed_resin_eew": {"category": "EEW", "unit": "g/eq", "definition": "resin molecular weight / epoxy functionality"},
    "computed_hardener_ahew": {"category": "AHEW", "unit": "g/eq", "definition": "hardener molecular weight / active equivalent functionality"},
    "computed_resin_functionality": {"category": "官能度", "unit": "eq/mol", "definition": "count of epoxy SMARTS matches"},
    "computed_hardener_functionality": {"category": "官能度", "unit": "eq/mol", "definition": "count of supported active-hydrogen sites"},
    "computed_theoretical_phr": {"category": "PHR", "unit": "phr", "definition": "100 * AHEW / EEW"},
    "computed_actual_phr": {"category": "PHR", "unit": "phr", "definition": "manual/original ratio when valid, otherwise theoretical PHR"},
    "computed_stoich_ratio": {"category": "配比", "unit": "ratio", "definition": "actual PHR / theoretical PHR"},
    "computed_stoich_delta": {"category": "配比", "unit": "phr", "definition": "actual PHR - theoretical PHR"},
    "computed_resin_eq_100": {"category": "配比", "unit": "eq", "definition": "100 / EEW"},
    "computed_hardener_eq": {"category": "配比", "unit": "eq", "definition": "actual PHR / AHEW"},
    "computed_equiv_ratio_h_to_r": {"category": "配比", "unit": "ratio", "definition": "hardener equivalents / resin equivalents"},
    "computed_equiv_ratio_r_to_h": {"category": "配比", "unit": "ratio", "definition": "resin equivalents / hardener equivalents"},
}

POST_FEATURE_DISPLAY_ALIASES = {
    "EEW", "AHEW", "Resin_Functionality", "Hardener_Functionality",
    "Theoretical_PHR", "Actual_PHR", "Stoich_Ratio", "Stoich_Delta",
    "Resin_Eq_100", "Hardener_Eq", "Equiv_Ratio_H_to_R",
    "Equiv_Ratio_R_to_H", "PHR", "eew", "ahew", "resin_functionality",
    "hardener_functionality", "theoretical_phr", "actual_phr", "stoich_ratio",
    "stoichiometric_ratio", "stoichiometric_ratio_r", "stoichiometry_r",
    "r_value", "resin_eq_100", "hardener_eq", "equiv_ratio_h_to_r",
    "equiv_ratio_r_to_h",
}


@dataclass
class CandidatePool:
    df: pd.DataFrame
    total_possible: int
    sampled: int


@dataclass
class FormulaDesignSpace:
    candidate_df: pd.DataFrame
    metadata: Dict[str, Any]


def _clean_smiles_list(items: Iterable) -> List[str]:
    out: List[str] = []
    for v in items or []:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "<na>", "na"}:
            continue
        out.append(s)
    return out


def count_smiles_components(value: Any) -> int:
    """Count disconnected molecular components represented by a SMILES value."""
    if value is None:
        return 0
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "na"}:
        return 0
    return sum(1 for part in text.split(".") if part.strip())


def filter_formulation_candidates_by_component_limits(
    candidate_df: Optional[pd.DataFrame],
    *,
    max_resin_components: int = 1,
    max_hardener_components: int = 1,
    resin_col: str = "resin_smiles",
    hardener_col: str = "hardener_smiles",
) -> pd.DataFrame:
    """Keep only formulation rows within the configured component limits."""
    if candidate_df is None:
        return pd.DataFrame()
    if candidate_df.empty:
        return candidate_df.copy()

    out = candidate_df.copy()
    keep_mask = pd.Series(True, index=out.index)
    limits = (
        (resin_col, max(1, int(max_resin_components))),
        (hardener_col, max(1, int(max_hardener_components))),
    )
    for column, limit in limits:
        if column not in out.columns:
            continue
        component_counts = out[column].map(count_smiles_components)
        keep_mask &= component_counts.le(limit)
    return out.loc[keep_mask].reset_index(drop=True)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _as_column_list(value: Any) -> List[str]:
    return sanitize_feature_columns(value)


def _smiles_lookup_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_feature_key(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _is_fingerprint_bit_column(name: Any) -> bool:
    text = str(name or "")
    return bool(re.search(r"(?:^|_)(?:Resin|Hardener)_(?:MACCS|Morgan)_\d+$", text, flags=re.I))


def resolve_molecular_feature_config(payload: Any) -> Optional[Dict[str, Any]]:
    """Extract the actual molecular-feature config from an artifact or JSON payload."""
    if not isinstance(payload, dict):
        return None

    if str(payload.get("method") or "").strip():
        return payload

    candidates: List[Any] = []
    extra = payload.get("extra")
    if isinstance(extra, dict):
        candidates.extend(
            [
                extra.get("molecular_feature_config"),
                extra.get("feature_process"),
            ]
        )
    candidates.extend(
        [
            payload.get("molecular_feature_config"),
            payload.get("feature_process"),
        ]
    )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("molecular_feature_config")
        if isinstance(nested, dict) and str(nested.get("method") or "").strip():
            return nested
        if str(candidate.get("method") or "").strip():
            return candidate
    return None


def legacy_config_to_workflow(
    payload: Any,
    model_feature_cols: Optional[Sequence[Any]] = None,
) -> MolecularFeatureWorkflow:
    """Adapt a legacy single-step config without claiming batch completeness."""
    config = dict(payload) if isinstance(payload, dict) else {}
    feature_names = sanitize_feature_columns(
        config.get("feature_names") or model_feature_cols or []
    )
    source_columns = []
    for key in ("smiles_col", "resin_component_cols", "hardener_component_cols"):
        value = config.get(key)
        for column in _as_column_list(value):
            if column not in source_columns:
                source_columns.append(column)

    step = {
        "step_id": "legacy_molecular_features",
        "order": 0,
        "role": config.get("primary_component_role") or "neutral",
        "source_columns": source_columns
        + [
            config["hardener_col"]
            for _ in [0]
            if isinstance(config.get("hardener_col"), str)
            and config["hardener_col"].strip()
            and config["hardener_col"] not in source_columns
        ],
        "method": config.get("method"),
        "prefix": config.get("prefix") or "",
        "params": dict(config.get("params") or {}),
        "semantic_params": dict(config.get("semantic_params") or {}),
        "feature_names": feature_names,
        "valid_row_behavior": dict(config.get("valid_row_behavior") or {}),
    }
    missing_items = [
        field_name
        for field_name in ("batch_mode", "batch_smiles_cols", "workflow_steps")
        if field_name not in config
    ]
    workflow_payload = {
        "schema_version": 2,
        "model_fingerprint": config.get("model_fingerprint"),
        "mode": "single_batch",
        "input_contract": {
            "legacy_fields": config,
            "resin_component_cols": (
                [config["resin_component_cols"]]
                if isinstance(config.get("resin_component_cols"), str)
                else list(config.get("resin_component_cols") or [])
            ),
            "hardener_component_cols": (
                [config["hardener_component_cols"]]
                if isinstance(config.get("hardener_component_cols"), str)
                else list(config.get("hardener_component_cols") or [])
            ),
            "hardener_col": config.get("hardener_col"),
        },
        "steps": [step],
        "merge_order": [step["step_id"]],
        "final_feature_names": feature_names,
        "feature_source_map": {
            name: step["step_id"] for name in feature_names
        },
        "derived_feature_steps": [],
        "random_seeds": {},
        "legacy": True,
        "missing_items": missing_items,
    }
    return MolecularFeatureWorkflow.from_dict(workflow_payload)


def resolve_molecular_feature_workflow(
    payload: Any,
    model_feature_cols: Optional[Sequence[Any]] = None,
) -> Optional[MolecularFeatureWorkflow]:
    """Resolve versioned workflow first, then adapt legacy config payloads."""
    if not isinstance(payload, dict):
        return None

    extra = payload.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    workflow_candidates = [
        extra.get("molecular_feature_workflow"),
        payload.get("molecular_feature_workflow"),
    ]
    for candidate in workflow_candidates:
        if isinstance(candidate, MolecularFeatureWorkflow):
            return candidate
        if isinstance(candidate, dict):
            return MolecularFeatureWorkflow.from_dict(candidate)

    legacy_candidates = [
        extra.get("molecular_feature_config"),
        payload.get("molecular_feature_config"),
        extra.get("feature_process"),
        payload.get("feature_process"),
    ]
    for candidate in legacy_candidates:
        if isinstance(candidate, dict):
            nested = candidate.get("molecular_feature_config")
            if isinstance(nested, dict):
                candidate = nested
            if str(candidate.get("method") or "").strip() or candidate.get("feature_names"):
                return legacy_config_to_workflow(
                    candidate,
                    model_feature_cols=model_feature_cols or payload.get("feature_cols"),
                )
    return None


def validate_molecular_feature_contract(
    model_feature_cols: Sequence[Any],
    mf_cfg: Optional[Dict[str, Any]],
    extracted_feature_cols: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Validate that the saved molecular-feature workflow feeds the model columns."""
    model_cols = sanitize_feature_columns(model_feature_cols)
    config_cols = sanitize_feature_columns((mf_cfg or {}).get("feature_names"))
    if not model_cols or not config_cols:
        return {
            "ok": True,
            "model_feature_cols": model_cols,
            "configured_feature_cols": config_cols,
            "overlap": [],
        }

    config_by_key = {_normalize_feature_key(col): col for col in config_cols}
    overlap = [col for col in model_cols if _normalize_feature_key(col) in config_by_key]
    if not overlap:
        method = str((mf_cfg or {}).get("method") or "未声明")
        model_preview = ", ".join(model_cols[:8])
        config_preview = ", ".join(config_cols[:8])
        raise ValueError(
            "特征契约不一致：模型需要的特征列与当前分子特征流程完全没有交集。"
            f"模型列示例=[{model_preview}]；当前流程={method}，"
            f"生成列示例=[{config_preview}]。"
            "不能用基线值、插补值或零值替代另一种分子特征；请重新导入与模型配套的 feature_process.json。"
        )

    if extracted_feature_cols is not None:
        extracted_cols = {
            _normalize_feature_key(col)
            for col in (extracted_feature_cols or [])
            if isinstance(col, str) and col.strip()
        }
        missing = [col for col in overlap if _normalize_feature_key(col) not in extracted_cols]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                "特征契约不一致：分子特征流程声明了模型需要的列，但实际提取结果缺少 "
                f"{len(missing)} 列 [{preview}]。"
                "请检查特征方法、参数、前缀以及双组分输入列。"
            )

    return {
        "ok": True,
        "model_feature_cols": model_cols,
        "configured_feature_cols": config_cols,
        "overlap": overlap,
    }


def infer_primary_component_role(mf_cfg: Optional[Dict]) -> str:
    """Infer the chemistry role represented by the saved primary SMILES inputs."""
    cfg = mf_cfg or {}
    explicit_role = str(cfg.get("primary_component_role") or "").strip().lower()
    if explicit_role in {"resin", "hardener", "neutral"}:
        return explicit_role
    names = _as_column_list(cfg.get("resin_component_cols"))
    if cfg.get("smiles_col"):
        names.append(cfg.get("smiles_col"))
    text = " ".join(str(name).lower() for name in names if name)
    hardener_tokens = (
        "hardener",
        "curing_agent",
        "curingagent",
        "curative",
        "固化剂",
        "交联剂",
    )
    resin_tokens = ("resin", "epoxy", "树脂", "基体")
    hardener_score = sum(text.count(token) for token in hardener_tokens)
    resin_score = sum(text.count(token) for token in resin_tokens)
    # 当列名冲突时（如 resin_component_cols 含 curing_agent 但 smiles_col 是 resin），
    # 以 smiles_col 为优先级更高的判断依据，因为它是用户明确指定的主列
    if resin_score > 0 and hardener_score > 0:
        selector = str(cfg.get("smiles_col") or "").lower()
        if any(t in selector for t in ("resin", "epoxy", "树脂", "基体")):
            return "resin"
        if any(t in selector for t in hardener_tokens):
            return "hardener"
    if hardener_score > resin_score:
        return "hardener"
    if resin_score > hardener_score:
        return "resin"
    return "neutral"


def resolve_component_smiles_cols(
    mf_cfg: Optional[Dict],
    role: str,
    available_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve source SMILES columns without confusing the model's primary role.

    Older feature-process files sometimes stored curing-agent columns in
    ``resin_component_cols`` because that field represented the model's
    primary input rather than the chemical role.  Prefer explicit role
    columns, then recover numbered columns from the source dataframe.
    """
    cfg = mf_cfg or {}
    target_role = str(role or "").strip().lower()
    if target_role not in {"resin", "hardener"}:
        return []

    hardener_tokens = (
        "hardener",
        "curing_agent",
        "curingagent",
        "curative",
        "固化剂",
        "交联剂",
    )
    resin_tokens = ("resin", "epoxy", "树脂", "基体")

    def _role_score(name: Any, candidate_role: str) -> int:
        text = str(name or "").lower()
        own_tokens = resin_tokens if candidate_role == "resin" else hardener_tokens
        other_tokens = hardener_tokens if candidate_role == "resin" else resin_tokens
        own = sum(text.count(token) for token in own_tokens)
        other = sum(text.count(token) for token in other_tokens)
        return own - other

    def _dedupe(values: Iterable[Any]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            name = str(value or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
        return result

    def _natural_key(name: str):
        match = re.search(r"(\d+)(?:\D*)$", str(name))
        return (str(name).lower() if not match else str(name)[: match.start()].lower(), int(match.group(1)) if match else -1)

    if target_role == "resin":
        configured = _as_column_list(cfg.get("resin_component_cols"))
        selector = cfg.get("smiles_col")
    else:
        configured = _as_column_list(cfg.get("hardener_component_cols"))
        selector = cfg.get("hardener_col")
        if not configured:
            configured = [
                name
                for name in _as_column_list(cfg.get("resin_component_cols"))
                if _role_score(name, "hardener") > 0
            ]
            if not selector and _role_score(cfg.get("smiles_col"), "hardener") > 0:
                selector = cfg.get("smiles_col")

    configured = _dedupe(configured)
    if selector:
        configured = _dedupe([*configured, selector])

    available = _dedupe(available_columns or [])
    available_set = set(available)
    role_columns = [name for name in configured if _role_score(name, target_role) > 0]
    opposite_role = "hardener" if target_role == "resin" else "resin"
    opposite_columns = [name for name in configured if _role_score(name, opposite_role) > 0]
    if role_columns:
        resolved = [name for name in role_columns if not available or name in available_set]
        if resolved:
            return sorted(resolved, key=_natural_key)
    if opposite_columns:
        if target_role == "hardener":
            resolved = [name for name in opposite_columns if not available or name in available_set]
            if resolved:
                return sorted(resolved, key=_natural_key)
    elif configured and not available:
        return sorted(configured, key=_natural_key)

    if target_role == "resin":
        matches = [
            name
            for name in available
            if _role_score(name, "resin") > 0
            and re.search(r"(smiles|bigsmiles|smile)", str(name), flags=re.I)
        ]
    else:
        matches = [
            name
            for name in available
            if _role_score(name, "hardener") > 0
            and re.search(r"(smiles|bigsmiles|smile)", str(name), flags=re.I)
        ]
    return sorted(matches, key=_natural_key)


def resolve_workflow_source_columns_by_role(
    workflow: Any,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Resolve declared workflow source columns into resin and hardener roles.

    Older artifacts used ``resin_component_cols`` as the primary model input
    field even when those columns were curing-agent columns.  Classification
    therefore uses both the saved step role and column-name evidence, instead
    of trusting the legacy field name blindly.
    """
    resin_columns: List[str] = []
    hardener_columns: List[str] = []
    resin_tokens = ("resin", "epoxy", "树脂", "基体")
    hardener_tokens = (
        "hardener",
        "curing",
        "curer",
        "curative",
        "固化剂",
        "交联剂",
    )

    def add_unique(target: List[str], values: Any) -> None:
        for value in _as_column_list(values):
            if value not in target:
                target.append(value)

    def role_score(column: Any) -> Tuple[int, int]:
        text = str(column or "").lower()
        return (
            sum(text.count(token) for token in resin_tokens),
            sum(text.count(token) for token in hardener_tokens),
        )

    def add_classified(values: Any, default_role: str) -> None:
        for column in _as_column_list(values):
            resin_score, hardener_score = role_score(column)
            if hardener_score > resin_score and hardener_score > 0:
                add_unique(hardener_columns, [column])
            elif resin_score > hardener_score and resin_score > 0:
                add_unique(resin_columns, [column])
            elif default_role == "hardener":
                add_unique(hardener_columns, [column])
            else:
                add_unique(resin_columns, [column])

    if isinstance(workflow, Mapping):
        workflow_steps = workflow.get("steps") or []
        workflow_contract = workflow.get("input_contract") or {}
    else:
        workflow_steps = getattr(workflow, "steps", []) or []
        workflow_contract = getattr(workflow, "input_contract", {}) or {}

    for step in workflow_steps:
        if not isinstance(step, Mapping):
            continue
        role = str(step.get("role") or "").lower()
        default_role = (
            "hardener"
            if ("hardener" in role or "curing" in role)
            else "resin"
            if ("resin" in role or "epoxy" in role)
            else "resin"
        )
        add_classified(step.get("source_columns"), default_role)

    contract = dict(workflow_contract) if isinstance(workflow_contract, Mapping) else {}
    add_classified(contract.get("resin_component_cols"), "resin")
    add_classified(contract.get("hardener_component_cols"), "hardener")
    add_classified(contract.get("hardener_col"), "hardener")

    cfg = config or {}
    add_classified(cfg.get("resin_component_cols"), "resin")
    add_classified(cfg.get("hardener_component_cols"), "hardener")
    add_classified(cfg.get("hardener_col"), "hardener")

    if not resin_columns:
        add_classified(
            cfg.get("smiles_col"),
            "hardener"
            if str(cfg.get("primary_component_role") or "").lower() == "hardener"
            else "resin",
        )
    return resin_columns, hardener_columns


def _pick_matching_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    if not columns:
        return None
    norm_map: Dict[str, str] = {}
    for col in columns:
        norm = _normalize_feature_key(col)
        if norm and norm not in norm_map:
            norm_map[norm] = str(col)
    for alias in aliases:
        match = norm_map.get(_normalize_feature_key(alias))
        if match is not None:
            return match
    return None


def _combine_smiles(resin: str, hardener: Optional[str]) -> str:
    r = (resin or "").strip()
    h = (hardener or "").strip() if hardener is not None else ""
    if r and h:
        return f"{r}.{h}"
    return r or h


def generate_candidate_pool(
    resin_smiles: Iterable,
    hardener_smiles: Optional[Iterable] = None,
    mode: str = "cartesian",
    max_candidates: int = 5000,
    random_state: int = 42,
    dedupe_inputs: bool = True,
) -> CandidatePool:
    """Generate candidate pool by pairing resin/hardener lists, never allocate full cartesian product."""
    resin_list = _clean_smiles_list(resin_smiles)
    if dedupe_inputs:
        resin_list = _dedupe_keep_order(resin_list)
    hardener_list = None
    if hardener_smiles is not None:
        hardener_list = _clean_smiles_list(hardener_smiles)
        if dedupe_inputs:
            hardener_list = _dedupe_keep_order(hardener_list)

    if not resin_list:
        return CandidatePool(df=pd.DataFrame(), total_possible=0, sampled=0)

    if hardener_list is None or len(hardener_list) == 0:
        df = pd.DataFrame({"resin_smiles": resin_list})
        df["combo_smiles"] = df["resin_smiles"]
        return CandidatePool(df=df, total_possible=len(resin_list), sampled=len(resin_list))

    mode = str(mode or "cartesian").lower()
    n_r = len(resin_list)
    n_h = len(hardener_list)
    total = n_r * n_h
    max_candidates = int(max_candidates)
    rng = np.random.default_rng(int(random_state))

    # ALWAYS sample via random indices, never generate full cartesian product
    # This prevents memory explosion for large libraries
    if mode in {"paired", "row", "zip"}:
        n = min(n_r, n_h)
        res_idx = np.arange(n)
        hard_idx = np.arange(n)
    else:
        # For cartesian/random: directly sample from flat index space
        # Never use np.repeat/np.tile on large arrays
        sample_size = min(max_candidates, total)
        if sample_size <= 0:
            return CandidatePool(df=pd.DataFrame(), total_possible=total, sampled=0)
        flat_indices = rng.choice(total, size=sample_size, replace=False)
        res_idx = flat_indices // n_h
        hard_idx = flat_indices % n_h

    df = pd.DataFrame(
        {
            "resin_smiles": [resin_list[i] for i in res_idx],
            "hardener_smiles": [hardener_list[j] for j in hard_idx],
        }
    )
    df["combo_smiles"] = [
        _combine_smiles(r, h) for r, h in zip(df["resin_smiles"], df["hardener_smiles"])
    ]
    return CandidatePool(df=df, total_possible=total, sampled=len(df))

def _clean_numeric_list(
    items: Optional[Iterable],
    *,
    default: Optional[Sequence[float]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> List[float]:
    vals: List[float] = []
    source = items if items is not None else default
    for v in source or []:
        try:
            num = float(v)
        except Exception:
            continue
        if not np.isfinite(num):
            continue
        if min_value is not None and num < float(min_value):
            continue
        if max_value is not None and num > float(max_value):
            continue
        vals.append(float(num))
    if not vals and default:
        return _clean_numeric_list(default, min_value=min_value, max_value=max_value)
    deduped = []
    seen = set()
    for v in vals:
        key = round(float(v), 8)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(v))
    return deduped


def _sample_or_limit_indices(total: int, max_candidates: int, random_state: int = 42) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=int)
    total = int(total)
    limit = max(1, int(max_candidates))
    if total <= limit:
        return np.arange(total, dtype=int)
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(total, size=limit, replace=False)
    idx = np.asarray(idx, dtype=int)
    idx.sort()
    return idx


def _sample_pair_grid_indices(
    pair_count: int,
    grid_size: int,
    limit: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a pair x process grid while covering pairs before repeats."""
    pair_count = max(0, int(pair_count))
    grid_size = max(1, int(grid_size))
    total = pair_count * grid_size
    limit = min(max(0, int(limit)), total)
    if pair_count == 0 or limit == 0:
        empty = np.asarray([], dtype=int)
        return empty, empty

    rng = np.random.default_rng(int(random_state))
    if total <= limit:
        return (
            np.repeat(np.arange(pair_count), grid_size),
            np.tile(np.arange(grid_size), pair_count),
        )

    if limit < pair_count:
        pair_idx = rng.choice(pair_count, size=limit, replace=False)
        grid_idx = rng.integers(0, grid_size, size=limit)
        return np.asarray(pair_idx, dtype=int), np.asarray(grid_idx, dtype=int)

    full_cycles, remainder = divmod(limit, pair_count)
    offsets = rng.integers(0, grid_size, size=pair_count)
    pair_parts = []
    grid_parts = []
    for cycle in range(full_cycles):
        order = rng.permutation(pair_count)
        pair_parts.append(order)
        grid_parts.append((offsets[order] + cycle) % grid_size)
    if remainder:
        order = rng.choice(pair_count, size=remainder, replace=False)
        pair_parts.append(order)
        grid_parts.append((offsets[order] + full_cycles) % grid_size)
    return (
        np.concatenate(pair_parts).astype(int, copy=False),
        np.concatenate(grid_parts).astype(int, copy=False),
    )


def _compute_grid_spec(feature_grid: Optional[Dict[str, Iterable]]) -> Tuple[List[str], List[List[Any]], int]:
    clean_grid: Dict[str, List[Any]] = {}
    for key, values in (feature_grid or {}).items():
        if key is None or not str(key).strip():
            continue
        vals = []
        for v in values or []:
            if isinstance(v, str) and not v.strip():
                continue
            vals.append(v)
        if vals:
            clean_grid[str(key)] = vals

    if not clean_grid:
        return [], [], 1

    grid_keys = list(clean_grid.keys())
    grid_values = [list(clean_grid[k]) for k in grid_keys]
    grid_size = 1
    for vals in grid_values:
        grid_size *= max(1, len(vals))
    return grid_keys, grid_values, int(grid_size)


def _decode_grid_index(grid_keys: Sequence[str], grid_values: Sequence[Sequence[Any]], flat_idx: int) -> Dict[str, Any]:
    if not grid_keys or not grid_values:
        return {}
    rem = int(flat_idx)
    out: Dict[str, Any] = {}
    for key, vals in zip(reversed(grid_keys), reversed(grid_values)):
        n = max(1, len(vals))
        pos = rem % n
        rem //= n
        out[key] = vals[pos]
    return {k: out[k] for k in grid_keys}


def _repeat_to_length(values: List[float], length: int, fallback: float) -> List[float]:
    if length <= 0:
        return []
    if not values:
        return [float(fallback)] * int(length)
    out = []
    n = len(values)
    for i in range(int(length)):
        out.append(float(values[i % n]))
    return out


def calc_mol_formula(smiles: str) -> Optional[str]:
    if not RDKIT_AVAILABLE:
        return None
    if smiles is None:
        return None
    try:
        mol = parse_chemical_string(smiles, repair=True, keep_largest_frag=False)
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def calc_mol_basic_stats(smiles: str) -> Optional[Dict[str, float]]:
    if not RDKIT_AVAILABLE:
        return None
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        mol_wt = float(rdMolDescriptors.CalcExactMolWt(mol))
        ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    except Exception:
        return None
    return {
        "heavy_atoms": heavy_atoms,
        "mol_wt": mol_wt,
        "ring_count": ring_count,
    }


def summarize_smiles_stats(
    smiles_list: Iterable[str],
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.9),
) -> Dict[str, Dict[str, float]]:
    if not RDKIT_AVAILABLE:
        return {}
    heavy_atoms = []
    mol_wt = []
    ring_count = []
    for s in smiles_list or []:
        stats = calc_mol_basic_stats(s)
        if not stats:
            continue
        heavy_atoms.append(stats["heavy_atoms"])
        mol_wt.append(stats["mol_wt"])
        ring_count.append(stats["ring_count"])

    def _summ(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {}
        vec = np.asarray(arr, dtype=float)
        vec = vec[np.isfinite(vec)]
        if vec.size == 0:
            return {}
        out = {
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "mean": float(np.mean(vec)),
        }
        for q in quantiles:
            key = f"q{int(round(q * 100))}"
            out[key] = float(np.quantile(vec, q))
        return out

    return {
        "heavy_atoms": _summ(heavy_atoms),
        "mol_wt": _summ(mol_wt),
        "ring_count": _summ(ring_count),
    }


def filter_candidates_by_size(
    df: pd.DataFrame,
    resin_col: str = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
    resin_constraints: Optional[Dict[str, float]] = None,
    hardener_constraints: Optional[Dict[str, float]] = None,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or not RDKIT_AVAILABLE:
        return df

    resin_constraints = resin_constraints or {}
    hardener_constraints = hardener_constraints or {}

    def _passes(smiles: str, constraints: Dict[str, float]) -> bool:
        if not constraints:
            return True
        stats = calc_mol_basic_stats(smiles)
        if stats is None:
            return bool(keep_invalid)
        if "min_heavy_atoms" in constraints and stats["heavy_atoms"] < float(constraints["min_heavy_atoms"]):
            return False
        if "min_mol_wt" in constraints and stats["mol_wt"] < float(constraints["min_mol_wt"]):
            return False
        if "min_rings" in constraints and stats["ring_count"] < float(constraints["min_rings"]):
            return False
        if "max_heavy_atoms" in constraints and stats["heavy_atoms"] > float(constraints["max_heavy_atoms"]):
            return False
        if "max_mol_wt" in constraints and stats["mol_wt"] > float(constraints["max_mol_wt"]):
            return False
        return True

    mask = []
    for _, row in df.iterrows():
        resin_ok = True
        if resin_col in df.columns:
            resin_ok = _passes(row.get(resin_col), resin_constraints)
        hard_ok = True
        if hardener_col and hardener_col in df.columns:
            hard_ok = _passes(row.get(hardener_col), hardener_constraints)
        mask.append(bool(resin_ok and hard_ok))

    return df.loc[mask].reset_index(drop=True)


def filter_candidates_by_epoxy_rules(
    df: pd.DataFrame,
    resin_col: Optional[str] = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
    rules: Optional[Dict] = None,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or not RDKIT_AVAILABLE:
        return df
    rules = rules or {}
    base = DEFAULT_EPOXY_RULES
    rules = {
        "global": {**base.get("global", {}), **(rules.get("global", {}) if isinstance(rules, dict) else {})},
        "resin": {**base.get("resin", {}), **(rules.get("resin", {}) if isinstance(rules, dict) else {})},
        "hardener": {**base.get("hardener", {}), **(rules.get("hardener", {}) if isinstance(rules, dict) else {})},
        "pair": {**base.get("pair", {}), **(rules.get("pair", {}) if isinstance(rules, dict) else {})},
    }

    allowed_elements = rules.get("global", {}).get("allowed_elements")
    if allowed_elements:
        try:
            allowed_elements = set(allowed_elements)
        except Exception:
            allowed_elements = None
    reject_charged = bool(rules.get("global", {}).get("reject_charged", True))

    resin_rules = rules.get("resin", {})
    hardener_rules = rules.get("hardener", {})
    pair_rules = rules.get("pair", {})

    cache: Dict[str, Dict[str, float]] = {}

    def _get_feat(smiles: str) -> Dict[str, float]:
        key = str(smiles or "")
        if key in cache:
            return cache[key]
        feats = _calc_rule_features(smiles, allowed_elements)
        cache[key] = feats
        return feats

    def _in_range(val: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if val is None or not np.isfinite(val):
            return False
        if lo is not None and val < float(lo):
            return False
        if hi is not None and val > float(hi):
            return False
        return True

    def _passes_resin(feat: Dict[str, float]) -> bool:
        if not feat.get("valid"):
            return bool(keep_invalid)
        if reject_charged and feat.get("has_charge"):
            return False
        if not _in_range(feat.get("mol_wt"), resin_rules.get("min_mw"), resin_rules.get("max_mw")):
            return False
        if not _in_range(feat.get("heavy_atoms"), resin_rules.get("min_heavy_atoms"), resin_rules.get("max_heavy_atoms")):
            return False
        if feat.get("epoxide", 0) < int(resin_rules.get("min_epoxide", 0)):
            return False
        max_epoxide = resin_rules.get("max_epoxide")
        if max_epoxide is not None and feat.get("epoxide", 0) > int(max_epoxide):
            return False
        min_ar = resin_rules.get("min_aromatic_rings")
        if min_ar is not None:
            ar_val = feat.get("aromatic_rings")
            if ar_val is None or not np.isfinite(ar_val) or ar_val < float(min_ar):
                return False
        if resin_rules.get("ban_strong_acids", True):
            if feat.get("carboxylic_acid", 0) > 0:
                return False
            if feat.get("sulfonic_acid", 0) > 0:
                return False
            if feat.get("phosphoric_acid", 0) > 0:
                return False
        if resin_rules.get("ban_amines", True):
            if feat.get("primary_amine", 0) > 0 or feat.get("secondary_amine", 0) > 0:
                return False
        return True

    def _passes_hardener(feat: Dict[str, float]) -> Tuple[bool, str]:
        if not feat.get("valid"):
            return bool(keep_invalid), ""
        if reject_charged and feat.get("has_charge"):
            return False, ""
        if not _in_range(
            feat.get("mol_wt"), hardener_rules.get("min_mw"), hardener_rules.get("max_mw")
        ):
            return False, ""
        if not _in_range(
            feat.get("heavy_atoms"),
            hardener_rules.get("min_heavy_atoms"),
            hardener_rules.get("max_heavy_atoms"),
        ):
            return False, ""
        if hardener_rules.get("ban_epoxide", True) and feat.get("epoxide", 0) > 0:
            return False, ""
        if hardener_rules.get("ban_strong_acids", True):
            if feat.get("carboxylic_acid", 0) > 0:
                return False, ""
            if feat.get("sulfonic_acid", 0) > 0:
                return False, ""
            if feat.get("phosphoric_acid", 0) > 0:
                return False, ""
        # 固化剂最小官能团限制（按固化剂类型分别判断）
        min_func = int(hardener_rules.get("min_active_hydrogen", 0) or 0)
        if min_func > 0:
            amine_h = 2 * int(feat.get("primary_amine", 0)) + int(feat.get("secondary_amine", 0))
            anhydride_n = int(feat.get("anhydride", 0))
            phenol_n = int(feat.get("phenol_oh", 0))
            thiol_n = int(feat.get("thiol", 0))
            imidazole_n = int(feat.get("imidazole", 0))
            tertiary_n = int(feat.get("tertiary_amine", 0))
            # 胺类：按活泼氢数（伯胺~2 + 仲胺~1）
            if amine_h > 0:
                if amine_h < min_func:
                    return False, ""
            # 酸酐：按酸酐基团数（每个酸酐开环消耗1个环氧基）
            elif anhydride_n > 0:
                if anhydride_n < min_func:
                    return False, ""
            # 酚类：按酚羟基数
            elif phenol_n > 0:
                if phenol_n < min_func:
                    return False, ""
            # 硫醇：按巯基数
            elif thiol_n > 0:
                if thiol_n < min_func:
                    return False, ""
            # 咪唑：按咪唑环数（催化型，每个咪唑催化多个环氧开环）
            elif imidazole_n > 0:
                if imidazole_n < min_func:
                    return False, ""
            # 叔胺：按叔胺基数（催化型）
            elif tertiary_n > 0:
                if tertiary_n < min_func:
                    return False, ""
            else:
                # 未能识别类型的固化剂，保守过滤
                if min_func > 1:
                    return False, ""

        has_amine = (feat.get("primary_amine", 0) + feat.get("secondary_amine", 0)) > 0
        has_anhydride = feat.get("anhydride", 0) > 0
        has_phenol = feat.get("phenol_oh", 0) > 0
        has_thiol = feat.get("thiol", 0) > 0
        has_imidazole = feat.get("imidazole", 0) > 0
        has_tertiary = feat.get("tertiary_amine", 0) > 0

        if pair_rules.get("reject_mixed_class", True) and has_amine and has_anhydride:
            return False, ""

        allowed = hardener_rules.get("allowed_classes")
        allowed_set = None
        if allowed:
            try:
                allowed_set = {str(x).strip().lower() for x in allowed if str(x).strip()}
            except Exception:
                allowed_set = None

        class_flags = {
            "amine": has_amine,
            "anhydride": has_anhydride,
            "phenol": has_phenol,
            "thiol": has_thiol,
            "imidazole": has_imidazole,
            "tertiary_amine": has_tertiary,
        }
        if allowed_set is not None and not any(class_flags.get(k, False) for k in allowed_set):
            return False, ""

        for name in ("amine", "anhydride", "phenol", "thiol", "imidazole", "tertiary_amine"):
            if class_flags.get(name) and (allowed_set is None or name in allowed_set):
                return True, name
        # The permissive default accepts other valid curing components. A
        # caller that needs class-specific chemistry can pass allowed_classes.
        return (True, "other") if allowed_set is None else (False, "")

    amine_ratio = pair_rules.get("amine_ratio", (0.0, float("inf")))
    anhydride_ratio = pair_rules.get("anhydride_ratio", (0.0, float("inf")))
    phenol_ratio = pair_rules.get("phenol_ratio", amine_ratio)
    thiol_ratio = pair_rules.get("thiol_ratio", amine_ratio)

    mask = []
    has_resin = bool(resin_col and resin_col in df.columns)
    has_hardener = bool(hardener_col and hardener_col in df.columns)
    for _, row in df.iterrows():
        resin_feat = {}
        if has_resin:
            resin_smiles = row.get(resin_col)
            resin_feat = _get_feat(resin_smiles)
            if not _passes_resin(resin_feat):
                mask.append(False)
                continue

        if not has_hardener:
            mask.append(True)
            continue

        hardener_smiles = row.get(hardener_col)
        hardener_feat = _get_feat(hardener_smiles)
        ok, cls = _passes_hardener(hardener_feat)
        if not ok:
            mask.append(False)
            continue

        if not has_resin:
            mask.append(True)
            continue

        if cls == "amine":
            H = 2 * int(hardener_feat.get("primary_amine", 0)) + int(
                hardener_feat.get("secondary_amine", 0)
            )
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, amine_ratio[0], amine_ratio[1]):
                mask.append(False)
                continue
        elif cls == "anhydride":
            A = int(hardener_feat.get("anhydride", 0))
            if A <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(2 * A)
            if not _in_range(ratio, anhydride_ratio[0], anhydride_ratio[1]):
                mask.append(False)
                continue
        elif cls == "phenol":
            H = int(hardener_feat.get("phenol_oh", 0))
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, phenol_ratio[0], phenol_ratio[1]):
                mask.append(False)
                continue
        elif cls == "thiol":
            H = int(hardener_feat.get("thiol", 0))
            if H <= 0:
                mask.append(False)
                continue
            ratio = float(resin_feat.get("epoxide", 0)) / float(H)
            if not _in_range(ratio, thiol_ratio[0], thiol_ratio[1]):
                mask.append(False)
                continue
        mask.append(True)

    return df.loc[mask].reset_index(drop=True)


def _mol_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE:
        return None
    if smiles is None:
        return None
    return parse_chemical_string(smiles, repair=True, keep_largest_frag=False)


_SMARTS_CACHE: Dict[str, Optional["Chem.Mol"]] = {}


def _get_smarts(pattern: str) -> Optional["Chem.Mol"]:
    if not RDKIT_AVAILABLE:
        return None
    if pattern in _SMARTS_CACHE:
        return _SMARTS_CACHE[pattern]
    try:
        mol = Chem.MolFromSmarts(pattern)
    except Exception:
        mol = None
    _SMARTS_CACHE[pattern] = mol
    return mol


def _count_smarts(mol, pattern: str) -> int:
    if mol is None:
        return 0
    patt = _get_smarts(pattern)
    if patt is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(patt))
    except Exception:
        return 0


def _count_active_hydrogens_on_nitrogen(mol) -> int:
    if mol is None:
        return 0
    count = 0
    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                count += int(atom.GetTotalNumHs())
    except Exception:
        return 0
    return int(count)


def _calc_rule_features(smiles: str, allowed_elements: Optional[set]) -> Dict[str, float]:
    out = {
        "valid": False,
        "heavy_atoms": np.nan,
        "mol_wt": np.nan,
        "aromatic_rings": np.nan,
        "epoxide": 0,
        "carboxylic_acid": 0,
        "sulfonic_acid": 0,
        "phosphoric_acid": 0,
        "primary_amine": 0,
        "secondary_amine": 0,
        "tertiary_amine": 0,
        "anhydride": 0,
        "phenol_oh": 0,
        "thiol": 0,
        "imidazole": 0,
        "has_charge": False,
    }
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return out

    if allowed_elements:
        try:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed_elements:
                    return out
        except Exception:
            return out

    try:
        out["has_charge"] = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    except Exception:
        out["has_charge"] = False

    try:
        out["heavy_atoms"] = float(mol.GetNumHeavyAtoms())
        out["mol_wt"] = float(rdMolDescriptors.CalcExactMolWt(mol))
        out["aromatic_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        pass

    # 对多组分SMILES（含.）拆分成单个分子分别检查官能度
    # 确保每个单分子都满足官能度要求，而非组合分子累加
    if "." in str(smiles):
        parts = [s.strip() for s in str(smiles).split(".") if s.strip()]
        epoxide_counts = []
        for part in parts:
            part_mol = _mol_from_smiles(part)
            if part_mol is not None:
                epoxide_counts.append(_count_smarts(part_mol, EPOXIDE_SMARTS))
        out["epoxide"] = min(epoxide_counts) if epoxide_counts else 0
    else:
        out["epoxide"] = _count_smarts(mol, EPOXIDE_SMARTS)
    out["carboxylic_acid"] = _count_smarts(mol, "[CX3](=O)[OX2H1]")
    out["sulfonic_acid"] = _count_smarts(mol, "[SX4](=O)(=O)[OX2H1]")
    out["phosphoric_acid"] = _count_smarts(mol, "[PX4](=O)([OX2H1])([OX2H1])[OX2H1]")
    out["primary_amine"] = _count_smarts(mol, "[NX3;H2;!$(NC=O)][#6]")
    out["secondary_amine"] = _count_smarts(mol, "[NX3;H1;!$(NC=O)]([#6])[#6]")
    out["tertiary_amine"] = _count_smarts(mol, "[NX3;H0;!$(NC=O)]([#6])[#6][#6]")
    out["anhydride"] = _count_smarts(mol, "[CX3](=O)O[CX3](=O)")
    out["phenol_oh"] = _count_smarts(mol, "[cX3][OX2H]")
    out["thiol"] = _count_smarts(mol, "[SX2H]")
    out["imidazole"] = _count_smarts(mol, "n1cc[nH]c1") + _count_smarts(mol, "n1cncc1")

    out["valid"] = True
    return out


def _score_to_level(score: Optional[float]) -> str:
    if score is None:
        return "invalid"
    try:
        val = float(score)
    except Exception:
        return "invalid"
    if not np.isfinite(val):
        return "invalid"
    if val >= 70.0:
        return "容易"
    if val >= 40.0:
        return "中等"
    return "困难"


def estimate_synthesizability(smiles: str) -> Tuple[Optional[float], str]:
    """Heuristic synthesizability score (0-100, higher is easier)."""
    if not RDKIT_AVAILABLE:
        return None, "unavailable"
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None, "invalid"

    heavy_atoms = mol.GetNumHeavyAtoms()
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    try:
        rings = mol.GetRingInfo().AtomRings()
        macrocycles = sum(1 for ring in rings if len(ring) >= 8)
    except Exception:
        macrocycles = 0

    penalty = 0.0
    penalty += max(0, heavy_atoms - 30) * 1.5
    penalty += ring_count * 2.0
    penalty += max(0, ring_count - 4) * 3.0
    penalty += stereo * 3.0
    penalty += max(0, rot_bonds - 6) * 1.5
    penalty += max(0, hetero - 10) * 1.0
    penalty += macrocycles * 10.0

    score = max(0.0, 100.0 - penalty)
    score = float(max(0.0, min(100.0, score)))
    return score, _score_to_level(score)


def add_synthesizability_scores(
    df: pd.DataFrame,
    resin_col: str = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
    mode: str = "min",
) -> pd.DataFrame:
    """Add heuristic synthesizability scores to a candidate dataframe."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if not RDKIT_AVAILABLE:
        out["synth_score"] = np.nan
        out["synth_level"] = "unavailable"
        return out

    resin_scores = []
    resin_levels = []
    if resin_col in out.columns:
        for s in out[resin_col].tolist():
            score, level = estimate_synthesizability(s)
            resin_scores.append(score if score is not None else np.nan)
            resin_levels.append(level)
        out["resin_synth_score"] = resin_scores
        out["resin_synth_level"] = resin_levels

    hard_scores = []
    hard_levels = []
    if hardener_col and hardener_col in out.columns:
        for s in out[hardener_col].tolist():
            score, level = estimate_synthesizability(s)
            hard_scores.append(score if score is not None else np.nan)
            hard_levels.append(level)
        out["hardener_synth_score"] = hard_scores
        out["hardener_synth_level"] = hard_levels

    if hard_scores and resin_scores:
        res_arr = np.asarray(resin_scores, dtype=float)
        hard_arr = np.asarray(hard_scores, dtype=float)
        stack = np.vstack([res_arr, hard_arr])
        mode = str(mode or "min").lower()
        if mode.startswith("mean") or mode.startswith("avg"):
            synth = np.nanmean(stack, axis=0)
        else:
            synth = np.nanmin(stack, axis=0)
    elif resin_scores:
        synth = np.asarray(resin_scores, dtype=float)
    else:
        synth = np.full(len(out), np.nan, dtype=float)

    out["synth_score"] = synth
    out["synth_level"] = [_score_to_level(v) for v in synth]
    return out


def _apply_prefix(df: pd.DataFrame, prefix: Optional[str]) -> pd.DataFrame:
    if prefix:
        df = df.copy()
        df.columns = [f"{prefix}{c}" for c in df.columns]
    return df


def _fill_missing_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    fill_fp_with_zero: bool = True,
) -> pd.DataFrame:
    if not required_cols:
        return df
    out = df.copy()
    lower_cols = {c.lower() for c in required_cols}
    for c in required_cols:
        if c in out.columns:
            continue
        is_fp = False
        if fill_fp_with_zero:
            cl = c.lower()
            is_fp = ("maccs" in cl) or ("morgan" in cl)
        out[c] = 0 if is_fp else np.nan
    # ensure order
    return out.reindex(columns=required_cols)


def _restore_full_rows(
    df: pd.DataFrame,
    valid_indices: List[int],
    total_rows: int,
) -> pd.DataFrame:
    if total_rows <= 0:
        return df
    if df is None or df.empty:
        return pd.DataFrame(index=range(total_rows))
    full = pd.DataFrame(index=range(total_rows), columns=df.columns, dtype=float)
    if valid_indices:
        full.iloc[valid_indices, :] = df.values
    return full


def extract_features_from_config(
    resin_smiles: List[str],
    hardener_smiles: Optional[List[str]],
    mf_cfg: Dict,
    device=None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Extract features for candidates using a saved molecular_feature_config."""
    method = str((mf_cfg or {}).get("method") or "")
    params = (mf_cfg or {}).get("params") or {}
    prefix = (mf_cfg or {}).get("prefix") or ""
    hardener_fusion_mode = (mf_cfg or {}).get("hardener_fusion_mode") or ""

    if params.get("drop_catalyst_fragments") and RDKIT_AVAILABLE:
        try:
            from .smiles_utils import split_smiles_cell
        except Exception:
            split_smiles_cell = None

        if split_smiles_cell is not None:
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
            only_multi = bool(params.get("drop_catalyst_only_multi", True))
            min_heavy = int(params.get("catalyst_min_heavy_atoms", 0) or 0)
            min_mw = float(params.get("catalyst_min_mol_wt", 0.0) or 0.0)
            smarts_list = params.get("catalyst_smarts") or []
            smarts_mols = []
            for patt in smarts_list:
                mol = Chem.MolFromSmarts(patt) if isinstance(patt, str) else None
                if mol is not None:
                    smarts_mols.append(mol)

            frag_cache: Dict[str, bool] = {}

            def _is_catalyst_fragment(frag: str) -> bool:
                if frag in frag_cache:
                    return frag_cache[frag]
                mol = _mol_from_smiles(frag)
                if mol is None:
                    frag_cache[frag] = bool(remove_invalid)
                    return frag_cache[frag]
                if remove_cations or remove_anions:
                    try:
                        has_pos = False
                        has_neg = False
                        for atom in mol.GetAtoms():
                            fc = atom.GetFormalCharge()
                            if fc > 0:
                                has_pos = True
                            elif fc < 0:
                                has_neg = True
                            if (remove_cations and has_pos) or (remove_anions and has_neg):
                                frag_cache[frag] = True
                                return True
                    except Exception:
                        pass
                if remove_metals:
                    try:
                        for atom in mol.GetAtoms():
                            if atom.GetAtomicNum() not in allowed_elements:
                                frag_cache[frag] = True
                                return True
                    except Exception:
                        pass
                if min_heavy and mol.GetNumHeavyAtoms() <= int(min_heavy):
                    frag_cache[frag] = True
                    return True
                if min_mw:
                    try:
                        mw = float(rdMolDescriptors.CalcExactMolWt(mol))
                        if mw <= float(min_mw):
                            frag_cache[frag] = True
                            return True
                    except Exception:
                        pass
                if smarts_mols:
                    try:
                        for patt in smarts_mols:
                            if mol.HasSubstructMatch(patt):
                                frag_cache[frag] = True
                                return True
                    except Exception:
                        pass
                frag_cache[frag] = False
                return False

            def _filter_smiles_cell(cell):
                if cell is None:
                    return None
                frags = split_smiles_cell(cell)
                if not frags:
                    return None
                if only_multi and len(frags) <= 1:
                    return ".".join(frags)
                kept = [f for f in frags if not _is_catalyst_fragment(f)]
                return ".".join(kept) if kept else None

            resin_smiles = [_filter_smiles_cell(s) for s in resin_smiles]
            if hardener_smiles:
                hardener_smiles = [_filter_smiles_cell(s) for s in hardener_smiles]

    # determine smiles input
    smiles_list_input = resin_smiles
    if hardener_smiles and isinstance(hardener_fusion_mode, str) and hardener_fusion_mode.startswith("拼接SMILES"):
        smiles_list_input = [
            _combine_smiles(r, h) for r, h in zip(resin_smiles, hardener_smiles)
        ]

    features_df = pd.DataFrame()
    valid_indices: List[int] = []

    try:
        if "分子指纹" in method:
            from .molecular_features import FingerprintExtractor

            extractor = FingerprintExtractor()
            features_df, valid_indices = extractor.smiles_to_fingerprints(
                resin_smiles,
                smiles_list_2=hardener_smiles,
                fp_type=str(params.get("fp_type", "MACCS")),
                n_bits=int(params.get("fp_bits", 2048)),
                radius=int(params.get("fp_radius", 2)),
                use_chirality=bool(params.get("fp_use_chirality", False)),
                use_features=bool(params.get("fp_use_features", False)),
                drop_all_zero_bits=bool(params.get("drop_all_zero_bits", False)),
            )

        elif ("标准版" in method) or ("并行版" in method) or ("内存优化版" in method) or ("RDKit" in method):
            from .molecular_features import AdvancedMolecularFeatureExtractor

            extractor = AdvancedMolecularFeatureExtractor()
            features_df, valid_indices = extractor.smiles_to_rdkit_features(smiles_list_input)

        elif "Mordred" in method:
            from .molecular_features import AdvancedMolecularFeatureExtractor

            extractor = AdvancedMolecularFeatureExtractor()
            features_df, valid_indices = extractor.smiles_to_mordred(
                smiles_list_input,
                batch_size=int(params.get("mordred_batch_size", 1000)),
                ignore_3D=bool(params.get("mordred_ignore_3d", True)),
                n_jobs=params.get("mordred_n_jobs", None),
            )

        elif "3D构象" in method:
            from .molecular_features import RDKit3DDescriptorExtractor

            extractor = RDKit3DDescriptorExtractor(
                coulomb_top_k=int(params.get("rdkit3d_coulomb_top_k", 10))
            )
            features_df, valid_indices = extractor.smiles_to_3d_descriptors(
                smiles_list_input, n_jobs=params.get("rdkit3d_n_jobs")
            )

        elif "Transformer Embedding" in method:
            from .molecular_features import SmilesTransformerEmbeddingExtractor

            extractor = SmilesTransformerEmbeddingExtractor(
                model_name=str(params.get("lm_model_name", "seyonec/ChemBERTa-zinc-base-v1")),
                pooling=str(params.get("lm_pooling", "cls")),
                max_length=int(params.get("lm_max_length", 128)),
                device=device,
                trust_remote_code=bool(params.get("lm_trust_remote_code", False)),
            )
            if not getattr(extractor, "AVAILABLE", False):
                return pd.DataFrame(), "transformers not available"
            features_df, valid_indices = extractor.smiles_to_embeddings(
                smiles_list_input, batch_size=int(params.get("lm_batch_size", 16))
            )

        elif "ML力场" in method:
            from .molecular_features import MLForceFieldExtractor

            extractor = MLForceFieldExtractor(device=device)
            if not getattr(extractor, "AVAILABLE", False):
                return pd.DataFrame(), "torchani not available"
            features_df, valid_indices = extractor.smiles_to_ani_features(
                smiles_list_input,
                batch_size=int(params.get("ani_batch_size", 256)),
                n_jobs=params.get("ani_cpu_workers", None),
            )

        elif "快速力场" in method:
            from .molecular_features import QuickForceFieldFeatureExtractor

            extractor = QuickForceFieldFeatureExtractor(
                ff_mode=str(params.get("ff_mode", "auto")),
                max_iters=int(params.get("ff_max_iters", 200)),
                minimize=bool(params.get("ff_minimize", True)),
                per_mol_timeout_s=params.get("ff_per_mol_timeout_s"),
                max_heavy_atoms=params.get("ff_max_heavy_atoms"),
                max_fragments=params.get("ff_max_fragments"),
                keep_largest_fragment=bool(params.get("ff_keep_largest_fragment", False)),
                skip_optimize_above_atoms=params.get("ff_skip_opt_above_atoms"),
            )
            features_df, valid_indices = extractor.smiles_to_ff_features(
                smiles_list_input, n_jobs=params.get("ff_n_jobs")
            )

        elif "xTB" in method:
            from .molecular_features import XTBFeatureExtractor

            total_timeout = params.get("xtb_total_timeout_s")
            if total_timeout is None:
                total_timeout = params.get("xtb_timeout_s", 300)

            extractor = XTBFeatureExtractor(
                xtb_path=str(params.get("xtb_path", "xtb")),
                method=str(params.get("xtb_method", "gfn2")),
                run_mode=str(params.get("xtb_run_mode", "sp")),
                charge=int(params.get("xtb_charge", 0)),
                uhf=int(params.get("xtb_uhf", 0)),
                timeout_s=int(params.get("xtb_timeout_s", 300)),
                max_iters=int(params.get("xtb_max_iters", 200)),
                per_mol_timeout_s=total_timeout,
                max_heavy_atoms=params.get("xtb_max_heavy_atoms"),
                max_fragments=params.get("xtb_max_fragments"),
                keep_largest_fragment=bool(params.get("xtb_keep_largest_fragment", False)),
                cache_size=int(params.get("xtb_cache_size", 10000)),
                random_state=int(params.get("xtb_random_state", 42)),
            )
            if not getattr(extractor, "AVAILABLE", False):
                return pd.DataFrame(), "xtb not available"
            features_df, valid_indices = extractor.featurize(
                smiles_list_input,
                n_jobs=max(1, int(params.get("xtb_n_jobs", 1))),
            )

        elif "FGD" in method:
            from .molecular_features import FGDFeatureExtractor

            extractor = FGDFeatureExtractor(cache_size=int(params.get("fgd_cache_size", 10000)))
            features_df, valid_indices = extractor.featurize(
                smiles_list_input,
                multi_label=bool(params.get("fgd_multi_label", True)),
                keep_largest_frag=bool(params.get("fgd_keep_largest_frag", True)),
                count_features=bool(params.get("fgd_count_features", False)),
            )

        elif "环氧树脂" in method:
            from .molecular_features import EpoxyDomainFeatureExtractor

            if hardener_smiles is None:
                return pd.DataFrame(), "hardener smiles required"
            extractor = EpoxyDomainFeatureExtractor(
                enable_reaction_simulation=bool(params.get("enable_reaction_simulation", True)),
                target_conversion=float(params.get("target_conversion", 0.5)),
            )
            features_df, valid_indices = extractor.extract_features(
                resin_smiles, hardener_smiles, params.get("phr_list"), params.get("stoich_mode", "theoretical")
            )

        elif "图神经网络" in method:
            from .graph_utils import GNNFeaturizer, TORCH_GEOMETRIC_AVAILABLE

            if not TORCH_GEOMETRIC_AVAILABLE:
                return pd.DataFrame(), "torch_geometric not available"
            gnn_cfg = {
                "device": device,
                "seed": int(params.get("gnn_seed", 42)),
                "deterministic": True,
                "add_hs": bool(params.get("gnn_add_hs", True)),
                "pooling": str(params.get("gnn_pooling", "sum")),
                "model_type": params.get("gnn_model_type", "gnn3d"),
                "hidden_dim": params.get("gnn_hidden_dim"),
                "num_layers": params.get("gnn_num_layers"),
                "dropout": params.get("gnn_dropout"),
                "gat_heads": params.get("gnn_gat_heads"),
                "num_timesteps": params.get("gnn_num_timesteps"),
                "output_dim": params.get("gnn_output_dim"),
                "cache_graphs": bool(params.get("gnn_cache_graphs", True)),
                "max_cache_size": int(params.get("gnn_cache_size", 5000)),
                "num_workers": int(params.get("gnn_num_workers", 0)),
                "chunk_size": int(params.get("gnn_chunk_size", 512)),
                "model_state_path": (params.get("gnn_weights_path") or None),
                "bigsmiles_mode": str(params.get("gnn_bigsmiles_mode", "auto")),
                "bigsmiles_num_samples": int(params.get("gnn_bigsmiles_num_samples", 4)),
                "bigsmiles_min_repeat_units": int(params.get("gnn_bigsmiles_min_repeat_units", 2)),
                "bigsmiles_max_repeat_units": int(params.get("gnn_bigsmiles_max_repeat_units", 6)),
            }
            extractor = GNNFeaturizer(**gnn_cfg)
            gnn_features, valid_indices = extractor.featurize(
                smiles_list_input,
                batch_size=int(params.get("gnn_batch_size", 32)),
                chunk_size=int(params.get("gnn_chunk_size", 512)),
                num_workers=int(params.get("gnn_num_workers", 0)),
                show_progress=False,
            )
            if gnn_features is not None and len(gnn_features) > 0:
                features_df = pd.DataFrame(gnn_features, columns=[f"gnn_{i}" for i in range(gnn_features.shape[1])])
            else:
                features_df = pd.DataFrame()

        else:
            return pd.DataFrame(), f"unsupported method: {method}"

    except Exception as e:  # pragma: no cover - best effort
        return pd.DataFrame(), str(e)

    try:
        from .molecular_features import append_configured_semantic_features

        features_df, valid_indices = append_configured_semantic_features(
            features_df,
            valid_indices,
            smiles_list_input,
            params,
            preserve_duplicate_columns=bool(
                params.get("preserve_duplicate_columns", False)
            ),
        )
    except Exception as e:  # pragma: no cover - best effort
        return pd.DataFrame(), f"semantic feature extraction failed: {e}"

    features_df = _apply_prefix(features_df, prefix)
    full_df = _restore_full_rows(features_df, valid_indices, len(resin_smiles))

    req_cols = [str(c) for c in ((mf_cfg or {}).get("feature_names") or []) if str(c)]
    if req_cols:
        duplicate_cols = full_df.columns[full_df.columns.duplicated()].tolist()
        if duplicate_cols:
            return pd.DataFrame(), (
                "feature contract produced duplicate columns: "
                + ", ".join(str(c) for c in duplicate_cols[:12])
            )
        missing_cols = [c for c in req_cols if c not in full_df.columns]
        missing_fp_cols = [col for col in missing_cols if _is_fingerprint_bit_column(col)]
        if missing_fp_cols and "分子指纹" in method:
            for col in missing_fp_cols:
                full_df[col] = 0
            missing_cols = [col for col in missing_cols if col not in set(missing_fp_cols)]
        if missing_cols:
            preview = ", ".join(missing_cols[:12])
            suffix = " ..." if len(missing_cols) > 12 else ""
            return pd.DataFrame(), (
                f"saved feature contract could not be reproduced: missing {len(missing_cols)} columns "
                f"[{preview}{suffix}]"
            )
        full_df = full_df.reindex(columns=req_cols)
    return full_df, None


def build_feature_matrix(
    feature_cols: List[str],
    mol_features: pd.DataFrame,
    base_row: Optional[pd.Series] = None,
    fill_missing_fp: bool = True,
    strict: bool = False,
    strict_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    required_cols = [str(column) for column in (feature_cols or [])]
    strict_required_cols = [
        str(column)
        for column in (strict_feature_cols if strict_feature_cols is not None else required_cols)
    ]
    n = len(mol_features) if mol_features is not None else 0
    X = pd.DataFrame(np.nan, index=range(n), columns=required_cols, dtype=float)
    if base_row is not None:
        for c in base_row.index:
            if c in X.columns:
                X[c] = base_row[c]
    if mol_features is not None and not mol_features.empty:
        for c in mol_features.columns:
            if c in X.columns:
                X[c] = mol_features[c].values
    if fill_missing_fp and not strict:
        fp_cols = [c for c in X.columns if ("maccs" in c.lower()) or ("morgan" in c.lower())]
        if fp_cols:
            X[fp_cols] = X[fp_cols].fillna(0)
    if strict:
        missing_columns = [
            column
            for column in strict_required_cols
            if mol_features is None or column not in mol_features.columns
        ]
        if strict_feature_cols is None and base_row is not None:
            missing_columns = [
                column
                for column in missing_columns
                if column not in base_row.index
            ]
        if missing_columns:
            preview = ", ".join(missing_columns[:12])
            suffix = " ..." if len(missing_columns) > 12 else ""
            raise ValueError(
                f"strict feature matrix is missing {len(missing_columns)} required columns "
                f"[{preview}{suffix}]"
            )
        invalid_value_columns = []
        for column in strict_required_cols:
            if mol_features is None or column not in mol_features.columns:
                continue
            values = pd.to_numeric(mol_features[column], errors="coerce")
            if values.isna().any():
                invalid_value_columns.append(column)
        if invalid_value_columns:
            preview = ", ".join(invalid_value_columns[:12])
            suffix = " ..." if len(invalid_value_columns) > 12 else ""
            raise ValueError(
                f"strict feature matrix contains missing or non-numeric values in "
                f"{len(invalid_value_columns)} extracted columns [{preview}{suffix}]"
            )
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).astype(float)


def _get_saved_process_pls_step(pipeline):
    if pipeline is None:
        return None
    named_steps = getattr(pipeline, "named_steps", None)
    if isinstance(named_steps, Mapping) and "process_pls" in named_steps:
        return named_steps.get("process_pls")
    for step_name, step_obj in getattr(pipeline, "steps", []) or []:
        if step_name == "process_pls":
            return step_obj
    return None


def apply_saved_process_pls(pipeline, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and order raw inputs for a fitted process-PLS pipeline.

    The fitted Pipeline owns the actual transform. This helper deliberately does
    not call ``fit`` or ``transform``; it only ensures high-throughput screening
    passes the exact raw input columns that the saved ``process_pls`` step saw
    during training.
    """
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    else:
        X_raw = X_raw.copy()

    process_pls = _get_saved_process_pls_step(pipeline)
    if process_pls is None:
        return X_raw

    required_cols = list(getattr(process_pls, "input_feature_cols_", []) or [])
    if not required_cols:
        configured_cols = list(getattr(process_pls, "process_feature_cols", []) or [])
        if configured_cols:
            raise ValueError(
                "模型包含工艺 PLS 配置，但缺少已拟合的 process_pls pipeline step；"
                "请重新训练并导出模型后再筛选。"
            )
        return X_raw

    missing_cols = [column for column in required_cols if column not in X_raw.columns]
    if missing_cols:
        preview = ", ".join(map(str, missing_cols[:12]))
        suffix = " ..." if len(missing_cols) > 12 else ""
        raise ValueError(
            f"高通量筛选缺少工艺 PLS 原始输入列: {preview}{suffix}"
        )

    ordered = X_raw.loc[:, required_cols].copy()
    for column in ordered.columns:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.replace([np.inf, -np.inf], np.nan)


def get_valid_feature_row_mask(
    mol_features: Optional[pd.DataFrame],
    required_cols: Optional[Sequence[str]],
) -> pd.Series:
    """Return rows whose required extracted features are numeric and finite."""
    if mol_features is None:
        return pd.Series(dtype=bool)

    mask = pd.Series(True, index=mol_features.index, dtype=bool)
    required = [str(column) for column in (required_cols or []) if str(column)]
    for column in required:
        if column not in mol_features.columns:
            return pd.Series(False, index=mol_features.index, dtype=bool)
        try:
            values = pd.to_numeric(mol_features[column], errors="coerce")
            finite = np.isfinite(np.asarray(values, dtype=float))
        except (TypeError, ValueError):
            finite = np.zeros(len(mol_features), dtype=bool)
        mask &= pd.Series(finite, index=mol_features.index, dtype=bool)
    return mask


def add_candidate_equivalent_metrics(
    df: pd.DataFrame,
    *,
    resin_col: str = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
) -> pd.DataFrame:
    """Backfill EEW/AHEW-style stoichiometry features from candidate chemistry.

    This keeps virtual screening inputs aligned with the resin/hardener pair instead
    of inheriting stale values from the template/base row.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if (not RDKIT_AVAILABLE) or Descriptors is None or resin_col not in out.columns:
        return out

    input_columns = list(df.columns)
    has_hardener = bool(hardener_col and hardener_col in out.columns)

    resin_keys = out[resin_col].fillna("").map(_smiles_lookup_key)
    if has_hardener:
        hardener_keys = out[hardener_col].fillna("").map(_smiles_lookup_key)
    else:
        hardener_keys = pd.Series([""] * len(out), index=out.index, dtype=object)

    def _build_stats(keys: pd.Series, *, include_epoxide: bool = False, include_active_h: bool = False) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for key in pd.unique(keys):
            lookup = _smiles_lookup_key(key)
            info = {
                "mw": 0.0,
                "epoxide": 0.0,
                "active_h": 0.0,
            }
            if lookup:
                mol = _mol_from_smiles(lookup)
                if mol is not None:
                    try:
                        info["mw"] = float(Descriptors.MolWt(mol))
                    except Exception:
                        info["mw"] = 0.0
                    if include_epoxide:
                        epoxy_count = _count_smarts(mol, EPOXIDE_SMARTS)
                        info["epoxide"] = float(epoxy_count)
                    if include_active_h:
                        info["active_h"] = float(_count_active_hydrogens_on_nitrogen(mol))
            stats[lookup] = info
        return stats

    resin_stats = _build_stats(resin_keys, include_epoxide=True)
    hardener_stats = _build_stats(hardener_keys, include_active_h=has_hardener)

    resin_mw_map = {key: val["mw"] for key, val in resin_stats.items()}
    resin_func_map = {key: val["epoxide"] for key, val in resin_stats.items()}
    hardener_mw_map = {key: val["mw"] for key, val in hardener_stats.items()}
    hardener_func_map = {key: val["active_h"] for key, val in hardener_stats.items()}

    resin_mw = pd.to_numeric(resin_keys.map(resin_mw_map), errors="coerce").fillna(0.0)
    resin_func = pd.to_numeric(resin_keys.map(resin_func_map), errors="coerce").fillna(0.0)
    hardener_mw = pd.to_numeric(hardener_keys.map(hardener_mw_map), errors="coerce").fillna(0.0)
    hardener_func = pd.to_numeric(hardener_keys.map(hardener_func_map), errors="coerce").fillna(0.0)

    resin_mw_arr = resin_mw.to_numpy(dtype=float)
    resin_func_arr = resin_func.to_numpy(dtype=float)
    hardener_mw_arr = hardener_mw.to_numpy(dtype=float)
    hardener_func_arr = hardener_func.to_numpy(dtype=float)

    eew_arr = np.where(resin_func_arr > 0, resin_mw_arr / resin_func_arr, resin_mw_arr)
    ahew_arr = np.where(hardener_func_arr > 0, hardener_mw_arr / hardener_func_arr, hardener_mw_arr)
    theo_phr_arr = np.where(eew_arr > 0, (ahew_arr / eew_arr) * 100.0, 0.0)
    actual_phr_arr = theo_phr_arr.copy()

    def _numeric_column(name: Optional[str]) -> Optional[pd.Series]:
        if not name or name not in out.columns:
            return None
        return pd.to_numeric(out[name], errors="coerce")

    phr_col = _pick_matching_column(
        input_columns,
        [
            "Actual_PHR",
            "actual_phr",
            "PHR",
            "phr",
            "hardener_phr",
            "curing_agent_phr",
            "curer_phr",
        ],
    )
    hardener_to_resin_col = _pick_matching_column(
        input_columns,
        [
            "hardener_resin_ratio",
            "hardener_to_resin_ratio",
            "mass_ratio_h_to_r",
            "ratio_h_r",
            "h_r_ratio",
        ],
    )
    resin_to_hardener_col = _pick_matching_column(
        input_columns,
        [
            "resin_hardener_ratio",
            "resin_to_hardener_ratio",
            "mass_ratio_r_to_h",
            "ratio_r_h",
            "r_h_ratio",
        ],
    )
    equiv_h_to_r_col = _pick_matching_column(
        input_columns,
        [
            "Stoich_Ratio",
            "stoich_ratio",
            "stoichiometric_ratio",
            "stoichiometric_ratio_r",
            "stoichiometry_r",
            "Equiv_Ratio_H_to_R",
            "equiv_ratio_h_to_r",
            "r_value",
        ],
    )
    equiv_r_to_h_col = _pick_matching_column(
        input_columns,
        [
            "Equiv_Ratio_R_to_H",
            "equiv_ratio_r_to_h",
        ],
    )

    phr_vals = _numeric_column(phr_col)
    if phr_vals is not None:
        phr_arr = phr_vals.to_numpy(dtype=float)
        actual_phr_arr = np.where(np.isfinite(phr_arr) & (phr_arr > 0), phr_arr, actual_phr_arr)
    else:
        hardener_to_resin_vals = _numeric_column(hardener_to_resin_col)
        resin_to_hardener_vals = _numeric_column(resin_to_hardener_col)
        equiv_h_to_r_vals = _numeric_column(equiv_h_to_r_col)
        equiv_r_to_h_vals = _numeric_column(equiv_r_to_h_col)

        if hardener_to_resin_vals is not None:
            ratio_arr = hardener_to_resin_vals.to_numpy(dtype=float)
            calc_arr = ratio_arr * 100.0
            actual_phr_arr = np.where(np.isfinite(ratio_arr) & (ratio_arr > 0), calc_arr, actual_phr_arr)
        elif resin_to_hardener_vals is not None:
            ratio_arr = resin_to_hardener_vals.to_numpy(dtype=float)
            calc_arr = np.divide(
                100.0,
                ratio_arr,
                out=np.zeros(len(out), dtype=float),
                where=np.isfinite(ratio_arr) & (ratio_arr > 0),
            )
            actual_phr_arr = np.where(np.isfinite(ratio_arr) & (ratio_arr > 0), calc_arr, actual_phr_arr)
        elif equiv_h_to_r_vals is not None:
            ratio_arr = equiv_h_to_r_vals.to_numpy(dtype=float)
            calc_arr = np.divide(
                ratio_arr * 100.0 * ahew_arr,
                eew_arr,
                out=np.zeros(len(out), dtype=float),
                where=np.isfinite(ratio_arr) & (ratio_arr > 0) & (eew_arr > 0) & (ahew_arr > 0),
            )
            actual_phr_arr = np.where(np.isfinite(ratio_arr) & (ratio_arr > 0), calc_arr, actual_phr_arr)
        elif equiv_r_to_h_vals is not None:
            ratio_arr = equiv_r_to_h_vals.to_numpy(dtype=float)
            calc_arr = np.divide(
                100.0 * ahew_arr,
                eew_arr * ratio_arr,
                out=np.zeros(len(out), dtype=float),
                where=np.isfinite(ratio_arr) & (ratio_arr > 0) & (eew_arr > 0) & (ahew_arr > 0),
            )
            actual_phr_arr = np.where(np.isfinite(ratio_arr) & (ratio_arr > 0), calc_arr, actual_phr_arr)

    stoich_ratio_arr = np.divide(
        actual_phr_arr,
        theo_phr_arr,
        out=np.zeros(len(out), dtype=float),
        where=theo_phr_arr > 0,
    )
    stoich_delta_arr = actual_phr_arr - theo_phr_arr
    resin_eq_arr = np.divide(100.0, eew_arr, out=np.zeros(len(out), dtype=float), where=eew_arr > 0)
    hardener_eq_arr = np.divide(actual_phr_arr, ahew_arr, out=np.zeros(len(out), dtype=float), where=ahew_arr > 0)
    equiv_h_to_r_arr = np.divide(
        hardener_eq_arr,
        resin_eq_arr,
        out=np.zeros(len(out), dtype=float),
        where=resin_eq_arr > 0,
    )
    equiv_r_to_h_arr = np.divide(
        resin_eq_arr,
        hardener_eq_arr,
        out=np.zeros(len(out), dtype=float),
        where=hardener_eq_arr > 0,
    )

    metric_columns = {
        "computed_resin_molecular_weight": resin_mw_arr,
        "computed_hardener_molecular_weight": hardener_mw_arr,
        "computed_resin_eew": eew_arr,
        "computed_hardener_ahew": ahew_arr,
        "computed_resin_functionality": resin_func_arr,
        "computed_hardener_functionality": hardener_func_arr,
        "computed_theoretical_phr": theo_phr_arr,
        "computed_actual_phr": actual_phr_arr,
        "computed_stoich_ratio": stoich_ratio_arr,
        "computed_stoich_delta": stoich_delta_arr,
        "computed_resin_eq_100": resin_eq_arr,
        "computed_hardener_eq": hardener_eq_arr,
        "computed_equiv_ratio_h_to_r": equiv_h_to_r_arr,
        "computed_equiv_ratio_r_to_h": equiv_r_to_h_arr,
        "EEW": eew_arr,
        "AHEW": ahew_arr,
        "Resin_Functionality": resin_func_arr,
        "Hardener_Functionality": hardener_func_arr,
        "Theoretical_PHR": theo_phr_arr,
        "Actual_PHR": actual_phr_arr,
        "Stoich_Ratio": stoich_ratio_arr,
        "Stoich_Delta": stoich_delta_arr,
        "Resin_Eq_100": resin_eq_arr,
        "Hardener_Eq": hardener_eq_arr,
        "Equiv_Ratio_H_to_R": equiv_h_to_r_arr,
        "Equiv_Ratio_R_to_H": equiv_r_to_h_arr,
        "PHR": actual_phr_arr,
        "eew": eew_arr,
        "ahew": ahew_arr,
        "resin_functionality": resin_func_arr,
        "hardener_functionality": hardener_func_arr,
        "theoretical_phr": theo_phr_arr,
        "actual_phr": actual_phr_arr,
        "stoich_ratio": stoich_ratio_arr,
        "stoichiometric_ratio": stoich_ratio_arr,
        "stoichiometric_ratio_r": stoich_ratio_arr,
        "stoichiometry_r": stoich_ratio_arr,
        "r_value": stoich_ratio_arr,
        "resin_eq_100": resin_eq_arr,
        "hardener_eq": hardener_eq_arr,
        "equiv_ratio_h_to_r": equiv_h_to_r_arr,
        "equiv_ratio_r_to_h": equiv_r_to_h_arr,
    }
    for col_name, values in metric_columns.items():
        out[col_name] = pd.to_numeric(values, errors="coerce")

    return out


def predict_with_model(
    model,
    X: pd.DataFrame,
    pipeline=None,
    imputer=None,
    scaler=None,
) -> np.ndarray:
    X = build_feature_matrix(
        list(X.columns) if isinstance(X, pd.DataFrame) else [],
        X if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
    )
    if pipeline is not None:
        X = apply_saved_process_pls(pipeline, X)
        return pipeline.predict(X)
    X_arr = X.values
    if imputer is not None:
        X_arr = imputer.transform(X_arr)
    if scaler is not None:
        X_arr = scaler.transform(X_arr)
    return model.predict(X_arr)


FRAGMENT_LIBRARY = [
    {"name": "benzene", "smiles": "c1ccccc1", "tags": ["Substrate_Benzene Ring"], "role": "resin"},
    {"name": "benzophenone", "smiles": "O=C(c1ccc(cc1))c2ccc(cc2)", "tags": ["Substrate_Benzophenone"], "role": "resin"},
    {"name": "bisphenol_a", "smiles": "CC(C)(C)c1ccc(cc1)C(c2ccc(cc2)O)O", "tags": ["Substrate_DGEBA"], "role": "resin"},
    {"name": "bisphenol_f", "smiles": "c1ccc(cc1)Cc2ccc(cc2)", "tags": ["Substrate_DGEBF"], "role": "resin"},
    {"name": "novolac", "smiles": "Oc1ccc(cc1)Cc2ccccc2", "tags": ["Substrate_Novolac"], "role": "resin"},
    {"name": "ester", "smiles": "CCOC(=O)C", "tags": ["Substrate_TDE-85 (Ester)"], "role": "resin"},
    {"name": "cyclohexane", "smiles": "C1CCCCC1", "tags": ["Substrate_Cycloaliphatic"], "role": "resin"},
    {"name": "isocyanurate", "smiles": "N1C(=O)NC(=O)NC1=O", "tags": ["Substrate_Isocyanurate"], "role": "resin"},
    {"name": "aliphatic", "smiles": "CCCC", "tags": ["Substrate_Aliphatic Chain"], "role": "resin"},
    {"name": "epoxide", "smiles": "C1OC1", "tags": ["Group_Epoxide"], "role": "resin"},
    {"name": "anhydride", "smiles": "O=C1OC(=O)C1", "tags": ["Group_Anhydride"], "role": "hardener"},
    {"name": "hydrazide", "smiles": "NNC(=O)N", "tags": ["Group_Hydrazide"], "role": "hardener"},
    {"name": "thiol", "smiles": "CS", "tags": ["Group_Thiol"], "role": "hardener"},
    {"name": "methacrylate", "smiles": "C=C(C)C(=O)O", "tags": ["Group_Methacrylate"], "role": "hardener"},
    {"name": "acrylate", "smiles": "C=CC(=O)O", "tags": ["Group_Acrylate"], "role": "hardener"},
    {"name": "amine_primary", "smiles": "CN", "tags": ["Group_Amine (Primary)"], "role": "hardener"},
    {"name": "amine_secondary", "smiles": "CN(C)C", "tags": ["Group_Amine (Secondary)"], "role": "hardener"},
    {"name": "hydroxyl", "smiles": "CO", "tags": ["Group_Hydroxyl"], "role": "both"},
    {"name": "vinyl", "smiles": "C=C", "tags": ["Group_Vinyl"], "role": "both"},
    {"name": "linker_ether", "smiles": "COC", "tags": ["Linker"], "role": "both"},
    {"name": "linker_alkyl", "smiles": "CC", "tags": ["Linker"], "role": "both"},
]


def _curated_virtual_component_smiles(role: str) -> List[str]:
    """Return complete, connected molecules for the unified HTVS workflow."""
    role_key = str(role or "resin").strip().lower()
    if role_key == "hardener":
        homologous_diamines = ["N" + ("C" * n) + "N" for n in range(2, 13)]
        return _dedupe_keep_order(
            homologous_diamines
            + [
                "NCCNCCN",
                "NCCNCCNCCN",
                "Nc1ccc(N)cc1",
                "Nc1ccc(cc1)c2ccc(N)cc2",
                "NCCc1ccccc1",
                "O=C1OC(=O)CC1",
                "O=C1OC(=O)C=C1",
                "O=C1OC(=O)c2ccccc12",
                "Oc1ccc(O)cc1",
                "Oc1ccc(cc1)C(c2ccc(O)cc2)(C)C",
                "SCCS",
                "SCCCS",
                "c1ncc[nH]1",
                "Cn1ccnc1",
                "CN(C)C",
                "CCN(CC)CC",
            ]
        )

    alkyl_glycidyl_ethers = [("C" * n) + "OCC1CO1" for n in range(1, 19)]
    return _dedupe_keep_order(
        alkyl_glycidyl_ethers
        + [
            "C1CO1",
            "C1CO1COCCOCC2CO2",
            "C1CO1COCCCOCC2CO2",
            "C1CO1COCCCCOCC2CO2",
            "c1ccc(OCC2CO2)cc1",
            "c1cc(OCC2CO2)cc(OCC3CO3)c1",
            "c1cc(OCC2CO2)ccc1Cc3ccc(OCC4CO4)cc3",
            "CC(C)(c1ccc(OCC2CO2)cc1)c3ccc(OCC4CO4)cc3",
        ]
    )


def _get_estimator(model):
    if hasattr(model, "named_steps"):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            return model
    return model


def _normalize_screening_weights(weights: Dict[str, float]) -> Dict[str, float]:
    keys = (
        "performance",
        "synth",
        "feasibility",
        "applicability",
        "uncertainty",
        "novelty",
        "feature_guidance",
    )
    cleaned = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    total = float(sum(cleaned.values()))
    if total <= 0:
        cleaned = {
            "performance": 0.40,
            "synth": 0.15,
            "feasibility": 0.15,
            "applicability": 0.12,
            "uncertainty": 0.10,
            "novelty": 0.05,
            "feature_guidance": 0.03,
        }
        total = float(sum(cleaned.values()))
    return {k: cleaned[k] / total for k in keys}


def rebalance_screening_weights(
    weights: Dict[str, float],
    *,
    changed_key: str,
    new_value: float,
) -> Dict[str, float]:
    """Change one score weight while proportionally preserving the 100% total."""
    keys = (
        "performance",
        "synth",
        "feasibility",
        "applicability",
        "uncertainty",
        "novelty",
        "feature_guidance",
    )
    current = _normalize_screening_weights(weights or {})
    if changed_key not in keys:
        return current

    try:
        requested = float(new_value)
    except (TypeError, ValueError):
        requested = current[changed_key]
    requested = float(np.clip(requested, 0.0, 1.0))

    other_keys = [key for key in keys if key != changed_key]
    other_total = float(sum(current[key] for key in other_keys))
    remaining = 1.0 - requested
    if other_total > 1e-12:
        for key in other_keys:
            current[key] = current[key] / other_total * remaining
    else:
        equal_share = remaining / max(1, len(other_keys))
        for key in other_keys:
            current[key] = equal_share
    current[changed_key] = requested
    return {key: float(current[key]) for key in keys}


def sample_pair_indices(
    total_pairs: int,
    sample_size: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Sample unique flat pair indices without allocating the full index space."""
    batches = iter_pair_indices(
        total_pairs,
        sample_size,
        batch_size=max(1, min(int(sample_size), 100_000)),
        random_state=int(random_state),
    )
    chunks = list(batches)
    if not chunks:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def iter_pair_indices(
    total_pairs: int,
    sample_size: int,
    *,
    batch_size: int = 50_000,
    random_state: int = 42,
) -> Iterator[np.ndarray]:
    """Yield unique flat pair indices in bounded-size batches.

    A modular permutation is used instead of materializing a full
    ``range(total_pairs)`` or a random-choice index array. This keeps memory
    proportional to one batch even when the Cartesian space is very large.
    """
    total_pairs = int(total_pairs)
    sample_size = min(max(0, int(sample_size)), max(0, total_pairs))
    batch_size = max(1, int(batch_size))
    if total_pairs <= 0 or sample_size <= 0:
        return
    if total_pairs > int(np.iinfo(np.int64).max):
        raise ValueError("total_pairs exceeds the supported int64 index range")

    if sample_size == total_pairs:
        for start in range(0, sample_size, batch_size):
            yield np.arange(
                start,
                min(start + batch_size, sample_size),
                dtype=np.int64,
            )
        return

    from math import gcd

    rng = np.random.default_rng(int(random_state))
    offset = int(rng.integers(0, total_pairs))
    stride = int(rng.integers(1, total_pairs))
    stride = max(1, stride)
    while gcd(stride, total_pairs) != 1:
        stride += 1
        if stride >= total_pairs:
            stride = 1

    for start in range(0, sample_size, batch_size):
        end = min(start + batch_size, sample_size)
        yield np.asarray(
            [
                (offset + int(position) * stride) % total_pairs
                for position in range(start, end)
            ],
            dtype=np.int64,
        )


def _predict_supports_return_std(estimator) -> bool:
    if estimator is None or not hasattr(estimator, "predict"):
        return False
    try:
        sig = inspect.signature(estimator.predict)
    except Exception:
        return False
    return "return_std" in sig.parameters


def summarize_model_screening_profile(
    model,
    *,
    pipeline=None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    estimator = _get_estimator(pipeline if pipeline is not None else model)
    estimator_name = type(estimator).__name__ if estimator is not None else ""
    pipeline_name = type(pipeline).__name__ if pipeline is not None else ""
    display_name = str(model_name or pipeline_name or estimator_name or type(model).__name__ or "UnknownModel")
    blob = " ".join(
        [
            str(display_name or ""),
            str(estimator_name or ""),
            str(type(model).__name__ if model is not None else ""),
            str(pipeline_name or ""),
        ]
    ).lower()

    if any(token in blob for token in ("autogluon", "tabpfn", "auto-sklearn", "tpot", "flaml", "chemsl", "superlearner")):
        family = "automl"
        family_label = "AutoML / 集成黑盒"
    elif any(token in blob for token in ("gaussian process", "gaussianprocess", "gpr")):
        family = "probabilistic_kernel"
        family_label = "概率核模型"
    elif any(token in blob for token in ("bayesian", "bnn")):
        family = "probabilistic_neural"
        family_label = "概率神经网络"
    elif any(token in blob for token in ("gnn", "graph", "chemberta", "molformer", "fusion")):
        family = "graph_sequence"
        family_label = "图 / 序列分子模型"
    elif any(token in blob for token in ("xgboost", "lightgbm", "catboost", "randomforest", "random forest", "extra trees", "gradientboost", "decisiontree", "forest", "boost")):
        family = "tree_ensemble"
        family_label = "树集成模型"
    elif hasattr(estimator, "coef_"):
        family = "linear"
        family_label = "线性 / 广义线性模型"
    elif any(token in blob for token in ("transformer", "tabnet", "ft-transformer", "neural", "tensorflow", "mlp", "pinn", "ann")):
        family = "deep_tabular"
        family_label = "深度表格模型"
    else:
        family = "generic"
        family_label = "通用黑盒模型"

    native_uncertainty = False
    uncertainty_mode = "proxy"
    uncertainty_label = "代理不确定度"
    for target in (pipeline, estimator):
        if target is None:
            continue
        if hasattr(target, "predict_with_uncertainty"):
            native_uncertainty = True
            uncertainty_mode = "native"
            uncertainty_label = "原生不确定度"
            break
        if _predict_supports_return_std(target):
            native_uncertainty = True
            uncertainty_mode = "return_std"
            uncertainty_label = "原生标准差"
            break
    if (not native_uncertainty) and estimator is not None and hasattr(estimator, "estimators_"):
        uncertainty_mode = "ensemble_spread"
        uncertainty_label = "集成方差"

    missing_tolerant = any(token in blob for token in ("xgboost", "lightgbm", "catboost"))

    if family == "linear":
        effect_method = "coef"
        effect_method_label = "自动（优先模型参数）"
    elif family == "tree_ensemble":
        effect_method = "importance"
        effect_method_label = "自动（优先模型参数）"
    elif family in {"automl", "graph_sequence"}:
        effect_method = "corr"
        effect_method_label = "仅相关性（根据当前数据/训练数据）"
    else:
        effect_method = "auto"
        effect_method_label = "自动（优先模型参数）"

    if family in {"probabilistic_kernel", "probabilistic_neural"}:
        strategy_bias = "explore"
        strategy_label = "偏探索"
        weights = {
            "performance": 0.32,
            "synth": 0.12,
            "feasibility": 0.14,
            "applicability": 0.12,
            "uncertainty": 0.16,
            "novelty": 0.10,
            "feature_guidance": 0.04,
        }
        mc_samples = 80 if native_uncertainty else 30
        similarity_threshold = 0.88
    elif family in {"tree_ensemble", "linear"}:
        strategy_bias = "exploit"
        strategy_label = "偏稳妥"
        weights = {
            "performance": 0.44,
            "synth": 0.16,
            "feasibility": 0.16,
            "applicability": 0.13,
            "uncertainty": 0.05,
            "novelty": 0.03,
            "feature_guidance": 0.03,
        }
        mc_samples = 20
        similarity_threshold = 0.93
    elif family == "automl":
        strategy_bias = "conservative"
        strategy_label = "偏保守"
        weights = {
            "performance": 0.38,
            "synth": 0.14,
            "feasibility": 0.18,
            "applicability": 0.18,
            "uncertainty": 0.05,
            "novelty": 0.03,
            "feature_guidance": 0.04,
        }
        mc_samples = 20
        similarity_threshold = 0.94
    elif family in {"graph_sequence", "deep_tabular"}:
        strategy_bias = "balanced"
        strategy_label = "均衡"
        weights = {
            "performance": 0.38,
            "synth": 0.14,
            "feasibility": 0.15,
            "applicability": 0.12,
            "uncertainty": 0.12,
            "novelty": 0.05,
            "feature_guidance": 0.04,
        }
        mc_samples = 50 if native_uncertainty else 25
        similarity_threshold = 0.91
    else:
        strategy_bias = "balanced"
        strategy_label = "均衡"
        weights = {
            "performance": 0.40,
            "synth": 0.15,
            "feasibility": 0.15,
            "applicability": 0.12,
            "uncertainty": 0.10,
            "novelty": 0.05,
            "feature_guidance": 0.03,
        }
        mc_samples = 25
        similarity_threshold = 0.92

    if uncertainty_mode == "proxy":
        weights["performance"] += 0.02
        weights["applicability"] += 0.02
        weights["uncertainty"] = max(0.02, weights["uncertainty"] - 0.04)
    weights = _normalize_screening_weights(weights)

    notes: List[str] = []
    if family == "tree_ensemble":
        notes.append("适合先按性能和可行性收敛，再用多样性补覆盖。")
    elif family in {"probabilistic_kernel", "probabilistic_neural"}:
        notes.append("可以更积极利用不确定度做主动探索。")
    elif family == "automl":
        notes.append("可解释性通常偏弱，建议更依赖适用域和可行性约束。")
    elif family in {"graph_sequence", "deep_tabular"}:
        notes.append("推荐保留一定探索比例，避免深度模型过度外推。")
    if missing_tolerant:
        notes.append("模型原生容忍缺失值，候选构造时可少做硬填补。")
    if uncertainty_mode == "proxy":
        notes.append("当前不确定度不是模型原生输出，筛选时建议不要给它过高权重。")

    return {
        "display_name": display_name,
        "family": family,
        "family_label": family_label,
        "uncertainty_mode": uncertainty_mode,
        "uncertainty_label": uncertainty_label,
        "supports_native_uncertainty": native_uncertainty,
        "missing_tolerant": missing_tolerant,
        "recommended_effect_method": effect_method,
        "recommended_effect_method_label": effect_method_label,
        "recommended_strategy_bias": strategy_bias,
        "recommended_strategy_label": strategy_label,
        "recommended_mc_samples": int(max(5, mc_samples)),
        "recommended_similarity_threshold": float(np.clip(similarity_threshold, 0.50, 0.99)),
        "recommended_batch_size": 12,
        "recommended_guided_top_pos": 10 if family != "automl" else 8,
        "recommended_guided_top_neg": 10 if family != "automl" else 8,
        "recommended_weights": weights,
        "notes": notes,
    }


def _feature_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    try:
        return float(np.corrcoef(x[mask], y[mask])[0, 1])
    except Exception:
        return 0.0


def compute_feature_effects(
    model,
    feature_cols: List[str],
    X_ref: Optional[pd.DataFrame] = None,
    y_ref: Optional[pd.Series] = None,
    pipeline=None,
    imputer=None,
    scaler=None,
    method: str = "auto",
    max_samples: int = 2000,
) -> pd.DataFrame:
    """Estimate feature effect direction for model-guided generation."""
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "weight", "sign", "effect", "source"])

    X_use = None
    if isinstance(X_ref, pd.DataFrame) and X_ref.shape[0] > 0:
        missing = [c for c in feature_cols if c not in X_ref.columns]
        if not missing:
            X_use = X_ref[feature_cols].copy()

    if X_use is not None and len(X_use) > int(max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_use), size=int(max_samples), replace=False)
        X_use = X_use.iloc[idx].copy()
        if isinstance(y_ref, (pd.Series, np.ndarray, list)):
            try:
                y_ref = pd.Series(y_ref).iloc[idx]
            except Exception:
                y_ref = None

    y_vec = None
    if y_ref is not None:
        y_vec = pd.to_numeric(pd.Series(y_ref), errors="coerce").values
    elif X_use is not None:
        try:
            y_vec = predict_with_model(model, X_use, pipeline=pipeline, imputer=imputer, scaler=scaler)
            y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        except Exception:
            y_vec = None

    estimator = _get_estimator(model)
    weights = None
    source = "corr"

    if method in {"auto", "coef"} and hasattr(estimator, "coef_"):
        try:
            coef = np.asarray(estimator.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            if coef.shape[0] == len(feature_cols):
                weights = coef
                source = "coef"
        except Exception:
            weights = None

    if weights is None and method in {"auto", "importance"} and hasattr(estimator, "feature_importances_"):
        try:
            imp = np.asarray(estimator.feature_importances_)
            if imp.shape[0] == len(feature_cols):
                weights = imp
                source = "importance"
        except Exception:
            weights = None

    effects = []
    for idx, feat in enumerate(feature_cols):
        weight = None
        sign = 1.0
        effect = 0.0

        if weights is not None:
            weight = float(weights[idx])
            if source == "coef":
                effect = weight
                sign = 1.0 if effect >= 0 else -1.0
                weight = abs(weight)
            else:
                if y_vec is not None and X_use is not None:
                    x_vec = pd.to_numeric(X_use[feat], errors="coerce").values
                    sign = 1.0 if _feature_corr(x_vec, y_vec) >= 0 else -1.0
                effect = float(weight) * float(sign)
        else:
            if y_vec is not None and X_use is not None:
                x_vec = pd.to_numeric(X_use[feat], errors="coerce").values
                effect = _feature_corr(x_vec, y_vec)
                weight = abs(effect)
                sign = 1.0 if effect >= 0 else -1.0
            else:
                weight = 0.0
                effect = 0.0

        effects.append(
            {
                "feature": feat,
                "weight": float(weight) if weight is not None else 0.0,
                "sign": float(sign),
                "effect": float(effect),
                "source": source,
            }
        )

    df = pd.DataFrame(effects)
    if not df.empty:
        df = df.assign(abs_effect=df["effect"].abs()).sort_values("abs_effect", ascending=False)
        df = df.drop(columns=["abs_effect"])
    return df


def _match_tag(feature: str, tag: str) -> bool:
    return feature == tag or feature.endswith(tag)


def extract_effect_tags(
    effect_df: pd.DataFrame,
    top_pos: int = 8,
    top_neg: int = 8,
) -> Tuple[List[str], List[str]]:
    tags = sorted({t for frag in FRAGMENT_LIBRARY for t in frag.get("tags", [])})
    pos_tags: List[str] = []
    neg_tags: List[str] = []
    if effect_df is None or effect_df.empty:
        return pos_tags, neg_tags

    pos_df = effect_df[effect_df["effect"] > 0]
    neg_df = effect_df[effect_df["effect"] < 0]

    for feat in pos_df["feature"].tolist():
        for tag in tags:
            if _match_tag(str(feat), tag) and tag not in pos_tags:
                pos_tags.append(tag)
                if len(pos_tags) >= int(top_pos):
                    break
        if len(pos_tags) >= int(top_pos):
            break

    for feat in neg_df["feature"].tolist():
        for tag in tags:
            if _match_tag(str(feat), tag) and tag not in neg_tags:
                neg_tags.append(tag)
                if len(neg_tags) >= int(top_neg):
                    break
        if len(neg_tags) >= int(top_neg):
            break

    return pos_tags, neg_tags


def _build_fragment_pool(role: str, pos_tags: List[str], neg_tags: List[str]) -> List[Dict]:
    role = str(role or "any")
    pool = []
    for frag in FRAGMENT_LIBRARY:
        frag_role = frag.get("role", "both")
        if role != "any" and frag_role not in {role, "both"}:
            continue
        frag_tags = frag.get("tags", [])
        if neg_tags and any(t in neg_tags for t in frag_tags):
            continue
        if pos_tags and not any(t in pos_tags for t in frag_tags):
            continue
        pool.append(frag)

    if pool:
        return pool

    # fallback: ignore pos_tags but keep role/neg_tags
    for frag in FRAGMENT_LIBRARY:
        frag_role = frag.get("role", "both")
        if role != "any" and frag_role not in {role, "both"}:
            continue
        frag_tags = frag.get("tags", [])
        if neg_tags and any(t in neg_tags for t in frag_tags):
            continue
        pool.append(frag)
    return pool


def _sample_component_smiles(
    pool: List[Dict],
    max_frags: int,
    rng: np.random.Generator,
    min_frags: int = 1,
) -> Tuple[str, List[str]]:
    if not pool:
        return "", []
    max_frags = max(1, int(max_frags))
    min_frags = max(1, min(int(min_frags), max_frags))
    n_frags = int(rng.integers(min_frags, max_frags + 1))
    idx = rng.integers(0, len(pool), size=n_frags)
    smiles_parts = []
    tags = []
    for i in idx:
        frag = pool[int(i)]
        smiles_parts.append(frag.get("smiles", ""))
        tags.extend(frag.get("tags", []))
    return ".".join([s for s in smiles_parts if s]), tags


def _count_fragments(smiles: str) -> int:
    if not smiles:
        return 0
    parts = [p for p in str(smiles).split(".") if p and str(p).strip()]
    return len(parts)


def _filter_pool_by_fragments(
    pool: List[str],
    min_fragments: int,
    max_fragments: int,
) -> List[str]:
    if not pool:
        return []
    min_fragments = max(1, int(min_fragments))
    max_fragments = max(1, int(max_fragments))
    if min_fragments > max_fragments:
        min_fragments = max_fragments
    filtered = [s for s in pool if min_fragments <= _count_fragments(s) <= max_fragments]
    return filtered if filtered else pool


def generate_feature_guided_candidates(
    n_samples: int = 200,
    pos_tags: Optional[List[str]] = None,
    neg_tags: Optional[List[str]] = None,
    component_count: int = 1,
    max_fragments: int = 2,
    min_fragments: int = 1,
    random_state: int = 42,
    resin_pool: Optional[List[str]] = None,
    hardener_pool: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate virtual molecules guided by positive/negative feature tags."""
    rng = np.random.default_rng(int(random_state))
    pos_tags = pos_tags or []
    neg_tags = neg_tags or []
    component_count = 2 if int(component_count) >= 2 else 1

    pool_resin = _dedupe_keep_order(_clean_smiles_list(resin_pool or []))
    pool_hardener = _dedupe_keep_order(_clean_smiles_list(hardener_pool or []))
    if pool_resin:
        pool_resin = _filter_pool_by_fragments(pool_resin, min_fragments, max_fragments)
    if pool_hardener:
        pool_hardener = _filter_pool_by_fragments(pool_hardener, min_fragments, max_fragments)

    if component_count == 1:
        pool_any = pool_resin or pool_hardener
        if pool_any:
            idx = rng.integers(0, len(pool_any), size=int(max(1, n_samples)))
            rows = []
            for i in idx:
                smi = pool_any[int(i)]
                rows.append(
                    {
                        "resin_smiles": smi,
                        "hardener_smiles": None,
                        "combo_smiles": smi,
                        "design_tags": "",
                    }
                )
            return pd.DataFrame(rows)

    if component_count == 2 and pool_resin and pool_hardener:
        res_idx = rng.integers(0, len(pool_resin), size=int(max(1, n_samples)))
        hard_idx = rng.integers(0, len(pool_hardener), size=int(max(1, n_samples)))
        rows = []
        for ri, hi in zip(res_idx, hard_idx):
            resin_smi = pool_resin[int(ri)]
            hard_smi = pool_hardener[int(hi)]
            combo = _combine_smiles(resin_smi, hard_smi)
            rows.append(
                {
                    "resin_smiles": resin_smi,
                    "hardener_smiles": hard_smi,
                    "combo_smiles": combo,
                    "design_tags": "",
                }
            )
        return pd.DataFrame(rows)

    resin_pool = _build_fragment_pool("resin", pos_tags, neg_tags)
    hardener_pool = _build_fragment_pool("hardener", pos_tags, neg_tags)
    any_pool = _build_fragment_pool("any", pos_tags, neg_tags)

    rows = []
    for _ in range(int(max(1, n_samples))):
        if component_count == 1:
            smi, tags = _sample_component_smiles(any_pool, max_fragments, rng, min_frags=min_fragments)
            rows.append(
                {
                    "resin_smiles": smi,
                    "hardener_smiles": None,
                    "combo_smiles": smi,
                    "design_tags": ";".join(sorted(set(tags))),
                }
            )
        else:
            resin_smi, resin_tags = _sample_component_smiles(resin_pool, max_fragments, rng, min_frags=min_fragments)
            hard_smi, hard_tags = _sample_component_smiles(hardener_pool, max_fragments, rng, min_frags=min_fragments)
            combo = _combine_smiles(resin_smi, hard_smi)
            rows.append(
                {
                    "resin_smiles": resin_smi,
                    "hardener_smiles": hard_smi,
                    "combo_smiles": combo,
                    "design_tags": ";".join(sorted(set(resin_tags + hard_tags))),
                }
            )

    return pd.DataFrame(rows)


SOURCE_AVAILABILITY_SCORES = {
    "train_data": 90.0,
    "training_data": 90.0,
    "dataset": 88.0,
    "uploaded": 80.0,
    "pubchem": 92.0,
    "guided_generated": 58.0,
    "generated": 55.0,
    "manual": 82.0,
    "custom": 78.0,
}


def _source_availability_score(source: str) -> float:
    if source is None:
        return 60.0
    parts = [p.strip().lower() for p in str(source).replace("+", "|").split("|") if p.strip()]
    if not parts:
        return 60.0
    scores = [float(SOURCE_AVAILABILITY_SCORES.get(p, 65.0)) for p in parts]
    return float(np.nanmax(scores)) if scores else 60.0


def build_component_library(
    smiles: Iterable,
    *,
    role: str,
    source: str,
    dedupe: bool = True,
    max_items: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    vals = _clean_smiles_list(smiles)
    if dedupe:
        vals = _dedupe_keep_order(vals)
    if max_items is not None and len(vals) > int(max_items):
        idx = _sample_or_limit_indices(len(vals), int(max_items), random_state=int(random_state))
        vals = [vals[int(i)] for i in idx]
    if not vals:
        return pd.DataFrame(columns=["smiles", "role", "source", "availability_score"])
    return pd.DataFrame(
        {
            "smiles": vals,
            "role": [str(role)] * len(vals),
            "source": [str(source)] * len(vals),
            "availability_score": [_source_availability_score(source)] * len(vals),
        }
    )


def merge_component_libraries(*libraries: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for lib in libraries:
        if lib is None or lib.empty:
            continue
        use = lib.copy()
        for col in ("smiles", "role", "source"):
            if col not in use.columns:
                use[col] = ""
        if "availability_score" not in use.columns:
            use["availability_score"] = use["source"].apply(_source_availability_score)
        frames.append(use)
    if not frames:
        return pd.DataFrame(columns=["smiles", "role", "source", "availability_score"])

    merged = pd.concat(frames, ignore_index=True)
    merged["smiles"] = merged["smiles"].astype(str).str.strip()
    merged = merged[merged["smiles"].ne("")]
    if merged.empty:
        return pd.DataFrame(columns=["smiles", "role", "source", "availability_score"])

    merged["source"] = merged["source"].fillna("").astype(str)
    merged["availability_score"] = pd.to_numeric(merged["availability_score"], errors="coerce").fillna(
        merged["source"].apply(_source_availability_score)
    )
    merged["_source_rank"] = merged["availability_score"]
    merged = merged.sort_values(["_source_rank", "smiles"], ascending=[False, True])

    rows = []
    for (_, role), grp in merged.groupby(["smiles", "role"], sort=False):
        sources = []
        seen = set()
        for src in grp["source"].tolist():
            for part in [p.strip() for p in str(src).split("|") if p.strip()]:
                if part not in seen:
                    seen.add(part)
                    sources.append(part)
        rows.append(
            {
                "smiles": grp["smiles"].iloc[0],
                "role": grp["role"].iloc[0],
                "source": "|".join(sources) if sources else grp["source"].iloc[0],
                "availability_score": float(grp["availability_score"].max()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["smiles", "role", "source", "availability_score"])
    return out.sort_values(["availability_score", "smiles"], ascending=[False, True]).reset_index(drop=True)


def limit_unique_candidates_for_expensive_features(
    df: pd.DataFrame,
    *,
    key_col: str = "_molecule_key",
    max_unique: int = 1000,
    random_state: int = 42,
    origin_col: str = "candidate_origin",
    source_cols: Sequence[str] = ("resin_source", "hardener_source"),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Limit expensive molecular calculations while preserving anchors and source diversity."""
    if df is None or df.empty or key_col not in df.columns:
        return df, {"before_unique": 0, "after_unique": 0, "limit": int(max_unique)}
    limit = max(1, int(max_unique))
    unique_df = df.drop_duplicates(subset=[key_col], keep="first").copy()
    before_unique = int(len(unique_df))
    if before_unique <= limit:
        return df.reset_index(drop=True), {
            "before_unique": before_unique,
            "after_unique": before_unique,
            "limit": limit,
        }

    origin = unique_df.get(origin_col, pd.Series("", index=unique_df.index)).fillna("").astype(str)
    anchor_df = unique_df[origin.eq("train_observed_pair")].copy()
    if len(anchor_df) > limit:
        if "availability_score" in anchor_df.columns:
            anchor_df = anchor_df.sort_values("availability_score", ascending=False).head(limit)
        else:
            anchor_df = anchor_df.head(limit)

    selected_keys = list(dict.fromkeys(anchor_df[key_col].tolist()))
    remaining_slots = max(0, limit - len(selected_keys))
    remaining_df = unique_df[~unique_df[key_col].isin(selected_keys)].copy()
    if remaining_slots > 0 and not remaining_df.empty:
        source_parts = []
        for col in source_cols:
            if col in remaining_df.columns:
                source_parts.append(remaining_df[col].fillna("").astype(str))
        if source_parts:
            source_group = source_parts[0]
            for part in source_parts[1:]:
                source_group = source_group + "|" + part
            remaining_df["_source_group"] = source_group.replace("", "unknown")
        else:
            remaining_df["_source_group"] = "unknown"

        groups = [grp for _, grp in remaining_df.groupby("_source_group", sort=True)]
        quota = max(1, remaining_slots // max(1, len(groups)))
        stratified_keys = []
        for group_idx, group in enumerate(groups):
            take_n = min(quota, len(group))
            sampled = group.sample(n=take_n, random_state=int(random_state) + group_idx)
            stratified_keys.extend(sampled[key_col].tolist())
        stratified_keys = list(dict.fromkeys(stratified_keys))[:remaining_slots]
        selected_keys.extend(stratified_keys)

        fill_slots = limit - len(selected_keys)
        if fill_slots > 0:
            leftovers = remaining_df[~remaining_df[key_col].isin(selected_keys)]
            if not leftovers.empty:
                sampled = leftovers.sample(
                    n=min(fill_slots, len(leftovers)),
                    random_state=int(random_state) + 1009,
                )
                selected_keys.extend(sampled[key_col].tolist())

    selected_key_set = set(selected_keys[:limit])
    out = df[df[key_col].isin(selected_key_set)].reset_index(drop=True).copy()
    after_unique = int(out[key_col].nunique()) if not out.empty else 0
    return out, {
        "before_unique": before_unique,
        "after_unique": after_unique,
        "limit": limit,
        "anchors_kept": int(sum(key in selected_key_set for key in anchor_df[key_col].tolist())),
    }


def generate_virtual_component_library(
    *,
    role: str,
    n_samples: int = 200,
    pos_tags: Optional[List[str]] = None,
    neg_tags: Optional[List[str]] = None,
    max_fragments: int = 3,
    min_fragments: int = 1,
    random_state: int = 42,
    seed_pool: Optional[Iterable[str]] = None,
    source: str = "guided_generated",
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    role_key = str(role or "resin").strip().lower()
    role_key = "any" if role_key not in {"resin", "hardener"} else role_key

    seed_vals = _clean_smiles_list(seed_pool or [])
    if seed_vals:
        seed_vals = _dedupe_keep_order(seed_vals)
        seed_vals = _filter_pool_by_fragments(seed_vals, min_fragments, max_fragments)

    curated_vals = _curated_virtual_component_smiles(role_key)
    if RDKIT_AVAILABLE:
        curated_vals = [s for s in curated_vals if _mol_from_smiles(s) is not None]
        if role_key == "resin":
            curated_vals = [
                s for s in curated_vals
                if _calc_rule_features(s, DEFAULT_EPOXY_RULES["global"]["allowed_elements"]).get("epoxide", 0) > 0
            ]

    limit = max(1, int(n_samples))
    out_smiles: List[str] = []
    if seed_vals:
        seed_quota = min(len(seed_vals), max(1, limit // 2))
        seed_idx = rng.choice(len(seed_vals), size=seed_quota, replace=False)
        out_smiles.extend([seed_vals[int(i)] for i in seed_idx])
    remaining = max(0, limit - len(out_smiles))
    if curated_vals and remaining:
        curated_order = rng.permutation(len(curated_vals))
        out_smiles.extend([curated_vals[int(i)] for i in curated_order[:remaining]])

    out_smiles = _dedupe_keep_order(out_smiles)

    return build_component_library(
        out_smiles,
        role=role_key if role_key in {"resin", "hardener"} else "resin",
        source=source,
        dedupe=True,
        random_state=int(random_state),
    )



def _generate_multicomponent_pool(
    smiles_list: List[str],
    max_components: int = 2,
    max_samples: int = 5000000,
    random_state: int = 42,
    dedupe: bool = True,
) -> List[str]:
    """从SMILES列表中随机采样最多max_components个组分的组合（复配），避免枚举全部组合。"""
    items = _dedupe_keep_order(_clean_smiles_list(smiles_list)) if dedupe else list(smiles_list)
    items = [
        item
        for item in items
        if count_smiles_components(item) <= max(1, int(max_components))
    ]
    if max_components <= 1:
        return items
    if not items:
        return []
    item_component_counts = {
        item: count_smiles_components(item)
        for item in items
    }

    def _valid_combination(combo: Sequence[str]) -> bool:
        return sum(
            item_component_counts.get(item, count_smiles_components(item))
            for item in combo
        ) <= int(max_components)

    rng = np.random.default_rng(int(random_state))
    n_items = len(items)
    results = list(items)
    remaining = max_samples - len(results)
    if remaining <= 0:
        return results[:max_samples] if len(results) > max_samples else results
    from math import comb
    safe_enum_limit = 100_000
    total_combos = 0
    can_enumerate = True
    for n in range(2, min(max_components, n_items) + 1):
        c = comb(n_items, n)
        total_combos += c
        if total_combos > safe_enum_limit or c > safe_enum_limit:
            can_enumerate = False
            break
    if can_enumerate and total_combos <= remaining:
        from itertools import combinations
        for n in range(2, min(max_components, n_items) + 1):
            for combo in combinations(items, n):
                if _valid_combination(combo):
                    results.append(".".join(combo))
    else:
        sampled = set()
        max_attempts = min(remaining * 5, 100_000)
        attempts = 0
        while len(sampled) < remaining and attempts < max_attempts:
            n = rng.integers(2, min(max_components, n_items) + 1)
            choices = rng.choice(items, size=n, replace=False)
            combo = tuple(sorted(choices))
            if _valid_combination(combo) and combo not in sampled:
                sampled.add(combo)
                results.append(".".join(combo))
            attempts += 1
    return results[:max_samples]

def enumerate_formulation_candidates(
    resin_library: pd.DataFrame,
    hardener_library: Optional[pd.DataFrame] = None,
    *,
    pair_mode: str = "cartesian",
    max_pairs: int = 5000000,
    random_state: int = 42,
    feature_grid: Optional[Dict[str, Iterable]] = None,
    max_formulations: int = 5000000,
    batch_size: int = 5000,
    hardener_required: bool = False,
    max_resin_components: int = 1,
    max_hardener_components: int = 1,
    comp_diversity: bool = True,
    batch_callback: Optional[callable] = None,
) -> FormulaDesignSpace:
    """Generate formulation design space via batched sampling to avoid OOM.
    Supports batch_callback(batch_idx, total_batches, batch_count, total_count) -> bool.
    Return False from callback to pause/abort."""
    feature_grid = feature_grid or {}
    resin_df = resin_library.copy() if resin_library is not None else pd.DataFrame()
    hard_df = hardener_library.copy() if hardener_library is not None else pd.DataFrame()

    resin_df = resin_df.dropna(subset=["smiles"]) if "smiles" in resin_df.columns else pd.DataFrame()
    hard_df = hard_df.dropna(subset=["smiles"]) if "smiles" in hard_df.columns else pd.DataFrame()
    total_pairs = 0

    if resin_df.empty:
        return FormulaDesignSpace(candidate_df=pd.DataFrame(), metadata={"total_pairs": total_pairs, "grid_size": 0, "total_possible": 0, "paused": True})
    if hardener_required and hard_df.empty:
        return FormulaDesignSpace(candidate_df=pd.DataFrame(), metadata={"total_pairs": total_pairs, "grid_size": 0, "total_possible": 0, "paused": True})

    use_hardener = not hard_df.empty

    # 多组分扩展
    resin_smiles_list = _generate_multicomponent_pool(
        resin_df["smiles"].tolist(),
        max_components=int(max_resin_components),
        max_samples=int(max_pairs),
        random_state=int(random_state),
        dedupe=bool(comp_diversity),
    )
    hardener_smiles_list = hard_df["smiles"].tolist() if use_hardener else None
    if use_hardener and hardener_smiles_list:
        hardener_smiles_list = _generate_multicomponent_pool(
            hardener_smiles_list,
            max_components=int(max_hardener_components),
            max_samples=int(max_pairs),
            random_state=int(random_state) + 7,
            dedupe=bool(comp_diversity),
        )

    # 分批生成配对，避免一次性创建巨大数组
    resin_meta = resin_df.drop_duplicates("smiles").set_index("smiles") if use_hardener else None
    hard_meta = hard_df.drop_duplicates("smiles").set_index("smiles") if use_hardener else None
    grid_keys, grid_values, grid_size = _compute_grid_spec(feature_grid)
    limit = max(1, int(max_formulations))
    accumulated_count = 0
    paused = False

    all_batches = []
    for batch_result in batch_generate_formulation_pairs(
        resin_smiles_list,
        hardener_smiles_list,
        batch_size=max(1, int(batch_size)),
        max_total=int(max_pairs),
        random_state=int(random_state),
    ):
        pair_df = batch_result.pair_df
        total_pairs = batch_result.metadata.get("total_pairs", 0)

        # 添加来源信息
        if resin_meta is not None and not resin_meta.empty:
            pair_df["resin_source"] = pair_df["resin_smiles"].map(resin_meta.get("source", pd.Series(dtype=str))).fillna("custom")
            pair_df["resin_availability_score"] = pair_df["resin_smiles"].map(resin_meta.get("availability_score", pd.Series(dtype=float))).fillna(70.0)
        else:
            pair_df["resin_source"] = "custom"
            pair_df["resin_availability_score"] = 70.0

        if hard_meta is not None and not hard_meta.empty and hardener_smiles_list:
            pair_df["hardener_source"] = pair_df["hardener_smiles"].map(hard_meta.get("source", pd.Series(dtype=str))).fillna("custom")
            pair_df["hardener_availability_score"] = pair_df["hardener_smiles"].map(hard_meta.get("availability_score", pd.Series(dtype=float))).fillna(70.0)
        else:
            pair_df["hardener_source"] = None
            pair_df["hardener_availability_score"] = np.nan

        remaining_limit = max(0, limit - accumulated_count)
        if remaining_limit <= 0:
            break
        if grid_size <= 1 and len(pair_df) > remaining_limit:
            pair_df = pair_df.iloc[:remaining_limit].copy()

        pair_df["formulation_id"] = np.arange(1, len(pair_df) + 1) + batch_result.batch_idx * batch_result.metadata.get("batch_size", 5000)
        pair_df["candidate_origin"] = (
            pair_df["resin_source"].fillna("custom").astype(str)
            + (" + " + pair_df["hardener_source"].fillna("none").astype(str) if "hardener_source" in pair_df.columns else "")
        )

        all_batches.append(pair_df)
        batch_count = min(
            len(pair_df) * max(1, grid_size),
            max(0, limit - accumulated_count),
        )
        accumulated_count += int(batch_count)
        if batch_callback is not None:
            should_continue = batch_callback(
                batch_idx=batch_result.batch_idx,
                total_batches=batch_result.total_batches,
                batch_count=int(batch_count),
                total_count=int(accumulated_count),
            )
            if not should_continue:
                paused = True
                break
        if accumulated_count >= limit:
            break

    if not all_batches:
        return FormulaDesignSpace(
            candidate_df=pd.DataFrame(),
            metadata={
                "total_pairs": total_pairs,
                "grid_size": 0,
                "total_possible": 0,
                "paused": bool(paused),
            },
        )

    base = pd.concat(all_batches, ignore_index=True)
    sampled_pairs = int(len(base))

    # 工艺网格采样
    if grid_keys and grid_values:
        pair_idx, grid_idx = _sample_pair_grid_indices(
            len(base),
            max(1, grid_size),
            min(limit, sampled_pairs * max(1, grid_size)),
            random_state=int(random_state),
        )
        if len(pair_idx) > 0:
            base = base.iloc[pair_idx].reset_index(drop=True).copy()
            overrides = pd.DataFrame(
                [_decode_grid_index(grid_keys, grid_values, int(i)) for i in grid_idx]
            )
            overrides = overrides.reset_index(drop=True)
            base = pd.concat([base.reset_index(drop=True), overrides], axis=1)

    base["formulation_id"] = np.arange(1, len(base) + 1)
    effective_total_possible = int(len(base))
    full_total_possible = int(total_pairs * max(1, grid_size))

    metadata = {
        "total_pairs": int(total_pairs),
        "sampled_pairs": sampled_pairs,
        "grid_size": int(grid_size),
        "effective_total_possible": effective_total_possible,
        "total_possible": full_total_possible,
        "sampled": int(len(base)),
        "paused": bool(paused),
    }
    return FormulaDesignSpace(candidate_df=base, metadata=metadata)

def apply_feature_overrides(
    X: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    skip_cols: Optional[Sequence[str]] = None,
    allowed_override_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if X is None or X.empty or candidate_df is None or candidate_df.empty:
        return X
    out = X.copy()
    skip = {
        "resin_smiles",
        "hardener_smiles",
        "combo_smiles",
        "resin_source",
        "hardener_source",
        "candidate_origin",
        "formulation_id",
        "resin_availability_score",
        "hardener_availability_score",
        "availability_score",
    }
    skip.update(POST_FEATURE_DISPLAY_ALIASES)
    skip.update(POST_FEATURE_COMPUTED_DEFINITIONS)
    if allowed_override_cols is not None:
        allowed = {str(column) for column in allowed_override_cols}
        skip.difference_update(allowed)
    if skip_cols:
        skip.update(str(c) for c in skip_cols if c is not None)

    for col in candidate_df.columns:
        if col in skip or col not in out.columns:
            continue
        vals = pd.to_numeric(candidate_df[col], errors="coerce")
        if vals.notna().any():
            valid = vals.notna()
            out.loc[valid, col] = vals.loc[valid].to_numpy()
    return out


def _tree_ensemble_predict_std(estimator, X_arr: np.ndarray) -> Optional[np.ndarray]:
    if estimator is None or not hasattr(estimator, "estimators_"):
        return None
    preds = []
    for est in getattr(estimator, "estimators_", []):
        try:
            preds.append(np.asarray(est.predict(X_arr), dtype=float).reshape(-1))
        except Exception:
            continue
    if not preds:
        return None
    stacked = np.vstack(preds)
    return np.nanstd(stacked, axis=0)


def _transform_for_final_estimator(X, pipeline):
    if pipeline is None or not hasattr(pipeline, "steps"):
        return X, _get_estimator(pipeline)
    steps = list(getattr(pipeline, "steps", []))
    if not steps:
        return X, _get_estimator(pipeline)
    estimator = steps[-1][1]
    X_proc = X
    for _, step in steps[:-1]:
        if not hasattr(step, "transform"):
            return X, estimator
        X_proc = step.transform(X_proc)
    return X_proc, estimator


def _predict_with_uncertainty_impl(
    model,
    X: pd.DataFrame,
    pipeline=None,
    imputer=None,
    scaler=None,
    mc_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if isinstance(X, pd.DataFrame):
        X = build_feature_matrix(list(X.columns), X)
    mean = np.asarray(predict_with_model(model, X, pipeline=pipeline, imputer=imputer, scaler=scaler), dtype=float).reshape(-1)

    if pipeline is not None and hasattr(pipeline, "predict_with_uncertainty"):
        try:
            mu, std = pipeline.predict_with_uncertainty(X, n_samples=mc_samples)
            return np.asarray(mu, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1), "pipeline_native"
        except TypeError:
            try:
                mu, std = pipeline.predict_with_uncertainty(X)
                return np.asarray(mu, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1), "pipeline_native"
            except Exception:
                pass
        except Exception:
            pass

    if pipeline is not None:
        try:
            X_proc, estimator = _transform_for_final_estimator(X, pipeline)
        except Exception:
            X_proc, estimator = X, _get_estimator(model)
    else:
        estimator = _get_estimator(model)
        X_proc = X.values
        if imputer is not None:
            X_proc = imputer.transform(X_proc)
        if scaler is not None:
            X_proc = scaler.transform(X_proc)

    if estimator is None:
        return mean, None, "point_only"

    if hasattr(estimator, "predict_with_uncertainty"):
        try:
            if mc_samples is not None:
                mu, std = estimator.predict_with_uncertainty(X_proc, n_samples=mc_samples)
            else:
                mu, std = estimator.predict_with_uncertainty(X_proc)
            mu = np.asarray(mu, dtype=float).reshape(-1)
            std = np.asarray(std, dtype=float).reshape(-1)
            return mu, std, "estimator_native"
        except TypeError:
            try:
                mu, std = estimator.predict_with_uncertainty(X_proc)
                return np.asarray(mu, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1), "estimator_native"
            except Exception:
                pass
        except Exception:
            pass

    if hasattr(estimator, "predict"):
        try:
            mu, std = estimator.predict(X_proc, return_std=True)
            return np.asarray(mu, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1), "return_std"
        except TypeError:
            pass
        except Exception:
            pass

    X_arr = X_proc.values if isinstance(X_proc, pd.DataFrame) else np.asarray(X_proc)
    std = _tree_ensemble_predict_std(estimator, X_arr)
    if std is not None:
        return mean, std, "tree_ensemble"
    return mean, None, "point_only"


def predict_with_uncertainty(
    model,
    X: pd.DataFrame,
    pipeline=None,
    imputer=None,
    scaler=None,
    mc_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mean, std, _ = _predict_with_uncertainty_impl(
        model,
        X,
        pipeline=pipeline,
        imputer=imputer,
        scaler=scaler,
        mc_samples=mc_samples,
    )
    return mean, std


def predict_with_uncertainty_info(
    model,
    X: pd.DataFrame,
    pipeline=None,
    imputer=None,
    scaler=None,
    mc_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    return _predict_with_uncertainty_impl(
        model,
        X,
        pipeline=pipeline,
        imputer=imputer,
        scaler=scaler,
        mc_samples=mc_samples,
    )


def _coerce_numeric_vector(values: Optional[Sequence]) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    ser = pd.to_numeric(pd.Series(arr), errors="coerce")
    if ser.empty:
        return None
    out = ser.to_numpy(dtype=float)
    finite = np.isfinite(out)
    if not finite.any():
        return None
    return out


def _fingerprint_cols(columns: Sequence[str]) -> List[str]:
    fp_cols = []
    for c in columns or []:
        cl = str(c).lower()
        if ("maccs" in cl) or ("morgan" in cl):
            fp_cols.append(str(c))
    return fp_cols


def _batch_max_tanimoto(query_bin: np.ndarray, ref_bin: np.ndarray, batch_size: int = 256) -> np.ndarray:
    if query_bin.size == 0 or ref_bin.size == 0:
        return np.zeros(query_bin.shape[0], dtype=float)
    ref_sum = ref_bin.sum(axis=1).astype(np.int32)
    out = np.zeros(query_bin.shape[0], dtype=float)
    for start in range(0, query_bin.shape[0], max(1, int(batch_size))):
        end = min(query_bin.shape[0], start + max(1, int(batch_size)))
        q = query_bin[start:end].astype(np.uint16)
        inter = np.dot(q, ref_bin.T.astype(np.uint16))
        q_sum = q.sum(axis=1).astype(np.int32)
        union = q_sum[:, None] + ref_sum[None, :] - inter
        sims = np.zeros_like(inter, dtype=float)
        mask = union > 0
        sims[mask] = inter[mask] / union[mask]
        out[start:end] = np.max(sims, axis=1) if sims.size else 0.0
    return out


def estimate_applicability_scores(
    X_query: pd.DataFrame,
    X_reference: Optional[pd.DataFrame],
    *,
    feature_cols: Optional[Sequence[str]] = None,
    tanimoto_threshold: float = 0.25,
    reference_max_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    out = pd.DataFrame(index=X_query.index)
    out["ad_score"] = np.nan
    out["ad_in_domain"] = False
    out["ad_similarity"] = np.nan
    out["ad_distance"] = np.nan
    out["novelty_score"] = np.nan

    if X_query is None or X_query.empty or X_reference is None or X_reference.empty:
        return out

    query_df = X_query.copy()
    ref_df = X_reference.copy()
    common = [c for c in (feature_cols or query_df.columns.tolist()) if c in query_df.columns and c in ref_df.columns]
    if not common:
        common = [c for c in query_df.columns if c in ref_df.columns]
    if not common:
        return out

    query_df = query_df[common].copy()
    ref_df = ref_df[common].copy()
    if len(ref_df) > int(reference_max_samples):
        idx = _sample_or_limit_indices(len(ref_df), int(reference_max_samples), random_state=int(random_state))
        ref_df = ref_df.iloc[idx].copy()

    fp_cols = _fingerprint_cols(common)
    if fp_cols:
        q_bin = (np.nan_to_num(query_df[fp_cols].values, nan=0.0) > 0).astype(np.uint8)
        r_bin = (np.nan_to_num(ref_df[fp_cols].values, nan=0.0) > 0).astype(np.uint8)
        max_sim = _batch_max_tanimoto(q_bin, r_bin)
        out["ad_similarity"] = max_sim
        out["ad_score"] = np.clip(max_sim * 100.0, 0.0, 100.0)
        out["novelty_score"] = np.clip((1.0 - max_sim) * 100.0, 0.0, 100.0)
        out["ad_in_domain"] = max_sim >= float(tanimoto_threshold)
        return out

    ref_num = ref_df.apply(pd.to_numeric, errors="coerce")
    query_num = query_df.apply(pd.to_numeric, errors="coerce")
    ref_med = ref_num.median(axis=0, numeric_only=True)
    ref_std = ref_num.std(axis=0, numeric_only=True).replace(0, 1).fillna(1)

    ref_scaled = ((ref_num.fillna(ref_med) - ref_med) / ref_std).to_numpy(dtype=float)
    query_scaled = ((query_num.fillna(ref_med) - ref_med) / ref_std).to_numpy(dtype=float)

    center = np.nanmean(ref_scaled, axis=0)
    ref_center_dist = np.linalg.norm(ref_scaled - center, axis=1)
    query_center_dist = np.linalg.norm(query_scaled - center, axis=1)
    thr = float(np.nanquantile(ref_center_dist, 0.95)) if ref_center_dist.size else 1.0
    scale = max(thr * 1.5, 1e-6)

    out["ad_distance"] = query_center_dist
    out["ad_score"] = np.clip((1.0 - (query_center_dist / scale)) * 100.0, 0.0, 100.0)
    novelty = np.clip(query_center_dist / max(float(np.nanquantile(ref_center_dist, 0.99)) if ref_center_dist.size else 1.0, 1e-6), 0.0, 1.0)
    out["novelty_score"] = novelty * 100.0
    out["ad_in_domain"] = query_center_dist <= thr
    return out


def estimate_proxy_uncertainty(
    X_query: Optional[pd.DataFrame],
    X_reference: Optional[pd.DataFrame] = None,
    *,
    y_reference: Optional[Sequence] = None,
    y_pred_reference: Optional[Sequence] = None,
    y_std_reference: Optional[Sequence] = None,
    feature_cols: Optional[Sequence[str]] = None,
    ad_frame: Optional[pd.DataFrame] = None,
    predictions: Optional[Sequence] = None,
    reference_max_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    if isinstance(X_query, pd.DataFrame):
        index = X_query.index
        n_query = len(X_query)
    elif isinstance(ad_frame, pd.DataFrame):
        index = ad_frame.index
        n_query = len(ad_frame)
    else:
        pred_arr = _coerce_numeric_vector(predictions)
        n_query = int(len(pred_arr)) if pred_arr is not None else 0
        index = pd.RangeIndex(n_query)

    out = pd.DataFrame(index=index)
    out["prediction_std_proxy"] = np.nan
    out["uncertainty_base_error"] = np.nan
    out["uncertainty_multiplier"] = np.nan
    out["uncertainty_source"] = "proxy_unavailable"
    if n_query <= 0:
        return out

    source_parts: List[str] = []
    residual_scale = np.nan

    y_true_arr = _coerce_numeric_vector(y_reference)
    y_pred_arr = _coerce_numeric_vector(y_pred_reference)
    if (
        y_true_arr is not None
        and y_pred_arr is not None
        and y_true_arr.size == y_pred_arr.size
        and y_true_arr.size > 0
    ):
        residuals = np.abs(y_true_arr - y_pred_arr)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size > 0:
            rmse = float(np.sqrt(np.mean(np.square(residuals))))
            mae = float(np.mean(residuals))
            q75 = float(np.quantile(residuals, 0.75))
            residual_scale = max(1e-9, 0.55 * rmse + 0.30 * mae + 0.15 * q75)
            source_parts.append("residual")

    if (not np.isfinite(residual_scale)) or residual_scale <= 0:
        y_std_arr = _coerce_numeric_vector(y_std_reference)
        if y_std_arr is not None:
            y_std_arr = y_std_arr[np.isfinite(y_std_arr) & (y_std_arr >= 0)]
            if y_std_arr.size > 0:
                residual_scale = max(1e-9, float(np.nanmedian(y_std_arr)))
                source_parts.append("native_train_std")

    if (not np.isfinite(residual_scale)) or residual_scale <= 0:
        if y_true_arr is not None:
            finite_y = y_true_arr[np.isfinite(y_true_arr)]
            if finite_y.size > 1:
                residual_scale = max(1e-9, float(np.nanstd(finite_y)) * 0.15)
                source_parts.append("target_scale")

    if (not np.isfinite(residual_scale)) or residual_scale <= 0:
        pred_arr = _coerce_numeric_vector(predictions)
        if pred_arr is not None:
            finite_pred = pred_arr[np.isfinite(pred_arr)]
            if finite_pred.size > 1:
                pred_std = float(np.nanstd(finite_pred))
                pred_iqr = float(np.subtract(*np.nanpercentile(finite_pred, [75, 25])))
                residual_scale = max(1e-9, pred_std * 0.10, pred_iqr * 0.06)
                source_parts.append("prediction_scale")

    if (not np.isfinite(residual_scale)) or residual_scale <= 0:
        residual_scale = 1.0
        source_parts.append("constant")

    ad_local = ad_frame.copy() if isinstance(ad_frame, pd.DataFrame) else None
    if (
        (ad_local is None or ad_local.empty)
        and isinstance(X_query, pd.DataFrame)
        and not X_query.empty
        and isinstance(X_reference, pd.DataFrame)
        and not X_reference.empty
    ):
        ad_local = estimate_applicability_scores(
            X_query,
            X_reference,
            feature_cols=feature_cols,
            reference_max_samples=reference_max_samples,
            random_state=random_state,
        )

    ad_score = pd.Series(55.0, index=index, dtype=float)
    novelty_score = pd.Series(45.0, index=index, dtype=float)
    in_domain = pd.Series(True, index=index, dtype=bool)
    if isinstance(ad_local, pd.DataFrame) and not ad_local.empty:
        ad_score = pd.to_numeric(ad_local.get("ad_score"), errors="coerce").reindex(index).fillna(55.0)
        novelty_score = pd.to_numeric(ad_local.get("novelty_score"), errors="coerce").reindex(index).fillna(45.0)
        in_domain_col = ad_local.get("ad_in_domain")
        if in_domain_col is not None:
            in_domain = pd.Series(in_domain_col, index=ad_local.index).reindex(index).fillna(False).astype(bool)
        source_parts.append("applicability")

    ad_penalty = ((100.0 - ad_score) / 100.0).clip(lower=0.0, upper=1.5)
    novelty_bonus = (novelty_score / 100.0).clip(lower=0.0, upper=1.5)
    domain_penalty = (~in_domain).astype(float) * 0.20
    multiplier = (0.60 + 0.85 * ad_penalty + 0.35 * novelty_bonus + domain_penalty).clip(lower=0.35, upper=4.5)

    proxy_std = float(residual_scale) * multiplier
    out["prediction_std_proxy"] = pd.to_numeric(proxy_std, errors="coerce")
    out["uncertainty_base_error"] = float(residual_scale)
    out["uncertainty_multiplier"] = pd.to_numeric(multiplier, errors="coerce")
    out["uncertainty_source"] = "+".join(source_parts) if source_parts else "proxy_unavailable"
    return out


def add_formulation_feasibility_scores(
    df: pd.DataFrame,
    *,
    resin_col: str = "resin_smiles",
    hardener_col: Optional[str] = "hardener_smiles",
    primary_role: str = "resin",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out = add_synthesizability_scores(out, resin_col=resin_col, hardener_col=hardener_col, mode="mean")

    resin_feats = {}
    hard_feats = {}
    reaction_scores = []
    process_scores = []
    availability_scores = []
    chemistry_labels = []
    primary_role = str(primary_role or "resin").strip().lower()

    def _get_feat(cache: Dict[str, Dict[str, float]], smiles: str) -> Dict[str, float]:
        key = str(smiles or "")
        if key not in cache:
            cache[key] = _calc_rule_features(smiles, DEFAULT_EPOXY_RULES.get("global", {}).get("allowed_elements"))
        return cache[key]

    for _, row in out.iterrows():
        primary_feat = _get_feat(resin_feats, row.get(resin_col))
        if primary_role == "hardener":
            resin_feat = {}
            hard_feat = primary_feat
        else:
            resin_feat = primary_feat
            hard_feat = _get_feat(hard_feats, row.get(hardener_col)) if hardener_col and hardener_col in out.columns else {}

        chemistry_score = 0.0
        labels = []
        if primary_feat.get("valid"):
            chemistry_score += 20.0
        if not hardener_col or hardener_col not in out.columns:
            chemistry_score += 15.0
            labels.append("single-component")
        elif hard_feat.get("valid"):
            chemistry_score += 20.0

        if resin_feat.get("epoxide", 0) > 0:
            chemistry_score += 20.0
            labels.append("epoxy")

        hard_class = None
        if hard_feat:
            if hard_feat.get("primary_amine", 0) + hard_feat.get("secondary_amine", 0) > 0:
                hard_class = "amine"
            elif hard_feat.get("anhydride", 0) > 0:
                hard_class = "anhydride"
            elif hard_feat.get("phenol_oh", 0) > 0:
                hard_class = "phenol"
            elif hard_feat.get("thiol", 0) > 0:
                hard_class = "thiol"
            elif hard_feat.get("imidazole", 0) > 0:
                hard_class = "imidazole"
            elif hard_feat.get("tertiary_amine", 0) > 0:
                hard_class = "tertiary_amine"
            if hard_class:
                chemistry_score += 25.0
                labels.append(hard_class)

        if resin_feat.get("has_charge") or (hard_feat and hard_feat.get("has_charge")):
            chemistry_score -= 12.0
        if resin_feat.get("carboxylic_acid", 0) > 0 or (hard_feat and hard_feat.get("carboxylic_acid", 0) > 0):
            chemistry_score -= 10.0
        if resin_feat.get("mol_wt") and np.isfinite(resin_feat.get("mol_wt")):
            if resin_feat.get("mol_wt") > 1600:
                chemistry_score -= 8.0
            elif resin_feat.get("mol_wt") >= 250:
                chemistry_score += 10.0
        if hard_feat and hard_feat.get("mol_wt") and np.isfinite(hard_feat.get("mol_wt")):
            if hard_feat.get("mol_wt") > 900:
                chemistry_score -= 8.0
            elif hard_feat.get("mol_wt") >= 70:
                chemistry_score += 8.0

        chemistry_score = float(np.clip(chemistry_score, 0.0, 100.0))
        reaction_scores.append(chemistry_score)
        chemistry_labels.append("|".join(labels) if labels else "unknown")

        process_score = 100.0
        heavy_atoms = float(resin_feat.get("heavy_atoms")) if resin_feat.get("valid") else np.nan
        if np.isfinite(heavy_atoms):
            process_score -= max(0.0, heavy_atoms - 45.0) * 0.8
        resin_rings = float(resin_feat.get("aromatic_rings")) if resin_feat.get("valid") else np.nan
        if np.isfinite(resin_rings):
            process_score -= max(0.0, resin_rings - 4.0) * 3.0
        hard_heavy = float(hard_feat.get("heavy_atoms")) if hard_feat and hard_feat.get("valid") else np.nan
        if np.isfinite(hard_heavy):
            process_score -= max(0.0, hard_heavy - 35.0) * 0.6
        process_scores.append(float(np.clip(process_score, 0.0, 100.0)))

        resin_src = row.get("resin_source")
        hard_src = row.get("hardener_source")
        if hardener_col and hardener_col in out.columns and pd.notna(hard_src):
            avail = np.nanmean([_source_availability_score(resin_src), _source_availability_score(hard_src)])
        else:
            avail = _source_availability_score(resin_src)
        availability_scores.append(float(np.clip(avail, 0.0, 100.0)))

    out["reaction_score"] = reaction_scores
    out["processability_score"] = process_scores
    out["availability_score"] = availability_scores
    out["chemistry_label"] = chemistry_labels

    synth_series = pd.to_numeric(out.get("synth_score"), errors="coerce")
    feas = (
        synth_series.fillna(50.0) * 0.35
        + pd.Series(reaction_scores, index=out.index).fillna(50.0) * 0.30
        + pd.Series(process_scores, index=out.index).fillna(50.0) * 0.20
        + pd.Series(availability_scores, index=out.index).fillna(50.0) * 0.15
    )
    out["feasibility_score"] = np.clip(feas, 0.0, 100.0)
    return out


def _normalize_score_series(series: pd.Series, *, inverse: bool = False) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < 1e-12:
        base = pd.Series(1.0, index=series.index, dtype=float)
    else:
        base = (vals - lo) / (hi - lo)
    base = base.clip(0.0, 1.0)
    if inverse:
        base = 1.0 - base
    return base


def rank_screening_candidates(
    df: pd.DataFrame,
    *,
    maximize: bool = True,
    target_value: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    weights = weights or {}
    w_perf = float(weights.get("performance", 0.40))
    w_synth = float(weights.get("synth", 0.15))
    w_feas = float(weights.get("feasibility", 0.15))
    w_ad = float(weights.get("applicability", 0.12))
    w_unc = float(weights.get("uncertainty", 0.10))
    w_novel = float(weights.get("novelty", 0.05))
    w_feat = float(weights.get("feature_guidance", 0.03))

    pred = pd.to_numeric(out.get("prediction"), errors="coerce")
    if target_value is not None and np.isfinite(float(target_value)):
        target_err = (pred - float(target_value)).abs()
        out["target_error"] = target_err
        perf_score = _normalize_score_series(target_err, inverse=True)
    else:
        perf_score = _normalize_score_series(pred, inverse=not bool(maximize))

    synth_score = _normalize_score_series(out.get("synth_score", pd.Series(index=out.index, dtype=float)))
    feas_score = _normalize_score_series(out.get("feasibility_score", pd.Series(index=out.index, dtype=float)))
    ad_score = _normalize_score_series(out.get("ad_score", pd.Series(index=out.index, dtype=float)))
    novel_score = _normalize_score_series(out.get("novelty_score", pd.Series(index=out.index, dtype=float)))
    feat_score = _normalize_score_series(out.get("feature_score", pd.Series(index=out.index, dtype=float)))

    unc_col = None
    for candidate in ("prediction_std", "uncertainty", "pred_std"):
        if candidate in out.columns:
            unc_col = candidate
            break
    if unc_col:
        unc_score = _normalize_score_series(out[unc_col], inverse=True)
    else:
        unc_score = pd.Series(np.nan, index=out.index, dtype=float)

    total_weight = 0.0
    total = pd.Series(0.0, index=out.index, dtype=float)
    for score, weight in (
        (perf_score, w_perf),
        (synth_score, w_synth),
        (feas_score, w_feas),
        (ad_score, w_ad),
        (unc_score, w_unc),
        (novel_score, w_novel),
        (feat_score, w_feat),
    ):
        valid = score.notna()
        if not valid.any() or weight <= 0:
            continue
        total = total.add(score.fillna(0.0) * weight, fill_value=0.0)
        total_weight += weight
    total_weight = max(total_weight, 1e-9)

    out["performance_score"] = perf_score * 100.0
    out["uncertainty_score"] = unc_score * 100.0
    out["total_score"] = np.clip((total / total_weight) * 100.0, 0.0, 100.0)
    out["exploit_score"] = np.nanmean(
        np.vstack(
            [
                out["performance_score"].fillna(0.0).values,
                pd.to_numeric(out.get("synth_score"), errors="coerce").fillna(0.0).values,
                pd.to_numeric(out.get("ad_score"), errors="coerce").fillna(0.0).values,
            ]
        ),
        axis=0,
    )
    out["explore_score"] = np.nanmean(
        np.vstack(
            [
                pd.to_numeric(out.get("novelty_score"), errors="coerce").fillna(0.0).values,
                pd.to_numeric(out.get("uncertainty_score"), errors="coerce").fillna(0.0).values,
            ]
        ),
        axis=0,
    )
    return out


def _max_similarity_numeric(query_vec: np.ndarray, selected: np.ndarray) -> float:
    if selected.size == 0:
        return 0.0
    q = np.asarray(query_vec, dtype=float).reshape(1, -1)
    s = np.asarray(selected, dtype=float)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    s_norm = np.linalg.norm(s, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    s_norm[s_norm == 0] = 1.0
    sims = (q / q_norm) @ (s / s_norm).T
    return float(np.nanmax(sims)) if sims.size else 0.0


def select_diverse_top_candidates(
    df: pd.DataFrame,
    *,
    feature_frame: Optional[pd.DataFrame] = None,
    top_k: int = 50,
    score_col: str = "total_score",
    similarity_threshold: float = 0.92,
    candidate_cap: int = 2500,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    top_k = max(1, int(top_k))
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked = ranked.head(max(top_k * 12, min(int(candidate_cap), len(ranked))))
    if feature_frame is None or feature_frame.empty:
        return ranked.head(top_k).copy()

    feat = feature_frame.reindex(ranked.index).copy()
    common_idx = ranked.index.intersection(feat.index)
    ranked = ranked.loc[common_idx].copy()
    feat = feat.loc[common_idx].copy()
    if ranked.empty or feat.empty:
        return ranked.head(top_k).copy()

    fp_cols = _fingerprint_cols(feat.columns.tolist())
    use_fp = bool(fp_cols)
    if use_fp:
        rep = (np.nan_to_num(feat[fp_cols].values, nan=0.0) > 0).astype(np.uint8)
    else:
        rep_df = feat.apply(pd.to_numeric, errors="coerce")
        med = rep_df.median(axis=0, numeric_only=True)
        std = rep_df.std(axis=0, numeric_only=True).replace(0, 1).fillna(1)
        rep = ((rep_df.fillna(med) - med) / std).to_numpy(dtype=float)

    selected_pos: List[int] = []
    for pos in range(len(ranked)):
        if not selected_pos:
            selected_pos.append(pos)
            if len(selected_pos) >= top_k:
                break
            continue

        if use_fp:
            q = rep[pos : pos + 1]
            s = rep[selected_pos]
            sim = float(_batch_max_tanimoto(q, s, batch_size=1)[0])
        else:
            sim = _max_similarity_numeric(rep[pos], rep[selected_pos])

        if sim <= float(similarity_threshold):
            selected_pos.append(pos)
        if len(selected_pos) >= top_k:
            break

    if len(selected_pos) < top_k:
        for pos in range(len(ranked)):
            if pos in selected_pos:
                continue
            selected_pos.append(pos)
            if len(selected_pos) >= top_k:
                break

    selected = ranked.iloc[selected_pos].copy()
    selected["diversity_rank"] = np.arange(1, len(selected) + 1)
    return selected


def build_experiment_recommendation_batches(
    df: pd.DataFrame,
    *,
    feature_frame: Optional[pd.DataFrame] = None,
    top_n_each: int = 12,
    score_col: str = "total_score",
    diversity_similarity_threshold: float = 0.88,
    candidate_cap: int = 2500,
    strategy_bias: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {
            "exploit": pd.DataFrame(),
            "explore": pd.DataFrame(),
            "diversify": pd.DataFrame(),
        }

    top_n_each = max(1, int(top_n_each))
    ranked = df.copy()
    sort_cols = [c for c in [score_col, "prediction"] if c in ranked.columns]
    if sort_cols:
        ranked = ranked.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    def _tag_batch(batch_df: pd.DataFrame, batch_key: str, batch_label: str, batch_reason: str) -> pd.DataFrame:
        if batch_df is None or batch_df.empty:
            return pd.DataFrame()
        out = batch_df.copy()
        out["recommended_batch"] = batch_key
        out["recommended_batch_label"] = batch_label
        out["recommended_reason"] = batch_reason
        return out

    strategy_bias = str(strategy_bias or "balanced").lower()
    exploit_min_feas = 25.0
    exploit_min_ad = 35.0
    if strategy_bias == "exploit":
        exploit_min_feas = 30.0
        exploit_min_ad = 40.0
    elif strategy_bias == "conservative":
        exploit_min_feas = 30.0
        exploit_min_ad = 45.0
    elif strategy_bias == "explore":
        exploit_min_feas = 20.0
        exploit_min_ad = 30.0

    exploit_pool = ranked.copy()
    if "feasibility_score" in exploit_pool.columns:
        feas = pd.to_numeric(exploit_pool["feasibility_score"], errors="coerce")
        good = exploit_pool.loc[feas.fillna(0.0) >= exploit_min_feas]
        if not good.empty:
            exploit_pool = good
    if "ad_score" in exploit_pool.columns:
        ad_vals = pd.to_numeric(exploit_pool["ad_score"], errors="coerce")
        good = exploit_pool.loc[ad_vals.fillna(0.0) >= exploit_min_ad]
        if not good.empty:
            exploit_pool = good
    exploit_sort = [c for c in ["exploit_score", score_col, "prediction"] if c in exploit_pool.columns]
    if exploit_sort:
        exploit_pool = exploit_pool.sort_values(exploit_sort, ascending=[False] * len(exploit_sort))
    exploit_df = _tag_batch(
        exploit_pool.head(top_n_each),
        "exploit",
        "稳妥冲高",
        "优先高性能、高可行性、较稳妥的候选",
    )

    used_index = set(exploit_df.index.tolist())
    explore_pool = ranked.drop(index=list(used_index), errors="ignore").copy()
    if explore_pool.empty:
        explore_pool = ranked.copy()
    if "feasibility_score" in explore_pool.columns:
        feas = pd.to_numeric(explore_pool["feasibility_score"], errors="coerce")
        good = explore_pool.loc[feas.fillna(0.0) >= 15.0]
        if not good.empty:
            explore_pool = good
    if strategy_bias == "explore":
        explore_priority = ["explore_score", "prediction_std", "novelty_score", score_col]
    else:
        explore_priority = ["explore_score", "novelty_score", "prediction_std", score_col]
    explore_sort = [c for c in explore_priority if c in explore_pool.columns]
    if explore_sort:
        explore_pool = explore_pool.sort_values(explore_sort, ascending=[False] * len(explore_sort))
    explore_df = _tag_batch(
        explore_pool.head(top_n_each),
        "explore",
        "主动探索",
        "优先高新颖性或高不确定度，适合主动学习补点",
    )

    used_index.update(explore_df.index.tolist())
    diversify_pool = ranked.drop(index=list(used_index), errors="ignore").copy()
    if diversify_pool.empty:
        diversify_pool = ranked.copy()
    diversify_feat = None
    if isinstance(feature_frame, pd.DataFrame) and not feature_frame.empty:
        diversify_feat = feature_frame.reindex(diversify_pool.index).copy()
    diversify_df = select_diverse_top_candidates(
        diversify_pool,
        feature_frame=diversify_feat,
        top_k=top_n_each,
        score_col=score_col,
        similarity_threshold=diversity_similarity_threshold,
        candidate_cap=candidate_cap,
    )
    diversify_df = _tag_batch(
        diversify_df,
        "diversify",
        "多样化拓展",
        "优先保留结构与配方空间的覆盖度",
    )

    return {
        "exploit": exploit_df,
        "explore": explore_df,
        "diversify": diversify_df,
    }


# ============================================================================
# 分批流式生成配方候选（避免一次性生成大量配方导致内存爆炸）
# ============================================================================

@dataclass
class BatchScreeningResult:
    """单批筛选结果"""
    batch_idx: int
    pair_df: pd.DataFrame
    total_batches: int
    flat_indices_used: np.ndarray
    metadata: Dict


def batch_generate_formulation_pairs(
    resin_smiles_list: list,
    hardener_smiles_list: list = None,
    *,
    batch_size: int = 5000,
    max_total: int = 5000000,
    random_state: int = 42,
):
    """按批次生成树脂-固化剂配对，避免物化完整笛卡尔积。"""
    batch_size = max(1, int(batch_size))
    max_total = max(1, int(max_total))

    resin_list = _dedupe_keep_order(_clean_smiles_list(resin_smiles_list))
    hardener_list = None
    if hardener_smiles_list is not None:
        hardener_list = _dedupe_keep_order(_clean_smiles_list(hardener_smiles_list))

    if not resin_list:
        return

    n_r = len(resin_list)
    n_h = len(hardener_list) if hardener_list else 0

    if hardener_list is None or n_h == 0:
        effective_max = min(n_r, max_total)
        total_batches = max(1, (effective_max + batch_size - 1) // batch_size)
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, effective_max)
            if start >= effective_max:
                break
            pair_df = pd.DataFrame({
                "resin_smiles": resin_list[start:end],
                "hardener_smiles": [None] * (end - start),
                "combo_smiles": resin_list[start:end],
            })
            yield BatchScreeningResult(
                batch_idx=batch_idx, pair_df=pair_df, total_batches=total_batches,
                flat_indices_used=np.arange(start, end),
                metadata={"n_resin": n_r, "n_hardener": 0, "total_pairs": n_r},
            )
        return

    total_pairs = n_r * n_h
    effective_max = min(total_pairs, max_total)
    total_batches = max(1, (effective_max + batch_size - 1) // batch_size)

    batch_size_parallel = max(1, int(batch_size))

    def _build_batch_df(batch_flat, batch_offset, n_h, resin_list, hardener_list):
        res_indices = batch_flat // n_h
        hard_indices = batch_flat % n_h
        pair_df = pd.DataFrame({
            "resin_smiles": [resin_list[i] for i in res_indices],
            "hardener_smiles": [hardener_list[j] for j in hard_indices],
        })
        pair_df["combo_smiles"] = [
            _combine_smiles(r, h) for r, h in zip(pair_df["resin_smiles"], pair_df["hardener_smiles"])
        ]
        return pair_df

    for batch_idx, batch_flat in enumerate(
        iter_pair_indices(
            total_pairs,
            effective_max,
            batch_size=batch_size_parallel,
            random_state=int(random_state),
        )
    ):
        pair_df = _build_batch_df(
            batch_flat,
            batch_idx * batch_size_parallel,
            n_h,
            resin_list,
            hardener_list,
        )
        yield BatchScreeningResult(
            batch_idx=batch_idx,
            pair_df=pair_df,
            total_batches=total_batches,
            flat_indices_used=batch_flat,
            metadata={
                "n_resin": n_r,
                "n_hardener": n_h,
                "total_pairs": total_pairs,
                "batch_size": batch_size_parallel,
                "effective_max": effective_max,
            },
        )


# =============================================================================
# 工业级分子过滤：排除药物、试剂、不适用的化合物
# =============================================================================

def filter_industrial_candidates(
    smiles_list: List[str],
    *,
    max_melting_point: float = 130.0,
    min_melting_point: float = -50.0,
    min_mol_wt: float = 80.0,
    max_mol_wt: float = 800.0,
    min_logp: float = -1.0,
    max_logp: float = 6.0,
    min_heavy_atoms: int = 4,
    max_heavy_atoms: int = 60,
    reject_lipinski_violators: bool = True,
    reject_pains: bool = True,
    max_rotatable_bonds: int = 15,
    label: str = "分子",
) -> Tuple[List[str], Dict[str, int]]:
    """对SMILES列表应用工业级过滤，排除药品/试剂/不适用的化合物。
    返回 (通过列表, 过滤统计)。"""
    if not RDKIT_AVAILABLE or not smiles_list:
        return list(smiles_list), {"total": len(smiles_list), "passed": len(smiles_list), "skipped_no_rdkit": len(smiles_list)}

    passed = []
    stats = {"total": len(smiles_list), "passed": 0, "failed_parse": 0, "failed_melting": 0, "failed_mol_wt": 0, "failed_logp": 0, "failed_heavy": 0, "failed_rotatable": 0, "failed_lipinski": 0, "failed_pains": 0}

    # 常见PAINS子结构（简化版）
    pains_smarts = [
        "[#6]=[#6]-[#6]=[#6]-[#6]=[#6]",  # 过长共轭
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",  # 酸酐
        "[#7][#7]=[#6]",  # 腙
        "[#16][#6](=[#8])[#6]",  # 硫酯
        "[#6](=[#8])[#6](=[#8])",  # 二酮
        "[#7]=[#6]-[#6]#[#7]",  # 烯腈
        "[#16]-[#16]",  # 二硫键
        "[#6]=[#6]-[#6]#[#6]",  # 烯炔
    ]
    pains_mols = []
    for sma in pains_smarts:
        try:
            pains_mols.append(Chem.MolFromSmarts(sma))
        except Exception:
            pains_mols.append(None)

    for smi in smiles_list:
        if not smi or not str(smi).strip():
            stats["failed_parse"] += 1
            continue
        mol = parse_chemical_string(smi, repair=True, keep_largest_frag=False)
        if mol is None:
            stats["failed_parse"] += 1
            continue

        # 分子量过滤
        mw = Descriptors.MolWt(mol)
        if mw < min_mol_wt or mw > max_mol_wt:
            stats["failed_mol_wt"] += 1
            continue

        # 重原子数
        heavy = mol.GetNumHeavyAtoms()
        if heavy < min_heavy_atoms or heavy > max_heavy_atoms:
            stats["failed_heavy"] += 1
            continue

        # 可旋转键
        rot = Descriptors.NumRotatableBonds(mol)
        if rot > max_rotatable_bonds:
            stats["failed_rotatable"] += 1
            continue

        # LogP
        logp = Descriptors.MolLogP(mol)
        if logp < min_logp or logp > max_logp:
            stats["failed_logp"] += 1
            continue

        # 熔点预测（Joback法简化：基于分子量+可旋转键的粗略估计）
        # 真实熔点预测需要更复杂的模型，这里用RDKit描述符估算
        # 使用经验公式：Tm ≈ 200 + 0.4*MW - 50*logP - 10*rot
        est_tm = 200.0 + 0.4 * mw - 50.0 * logp - 10.0 * rot
        if est_tm > max_melting_point or est_tm < min_melting_point:
            stats["failed_melting"] += 1
            continue

        # Lipinski规则检查（排除药物特征）
        if reject_lipinski_violators:
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if Descriptors.NumHDonors(mol) > 5: violations += 1
            if Descriptors.NumHAcceptors(mol) > 10: violations += 1
            if violations >= 2:  # 违反2条及以上
                stats["failed_lipinski"] += 1
                continue

        # PAINS过滤
        if reject_pains:
            is_pains = False
            for p_mol in pains_mols:
                if p_mol is not None and mol.HasSubstructMatch(p_mol):
                    is_pains = True
                    break
            if is_pains:
                stats["failed_pains"] += 1
                continue

        passed.append(str(smi).strip())
        stats["passed"] += 1

    # 去重
    seen = set()
    deduped = []
    for s in passed:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    stats["passed"] = len(deduped)
    stats["dedup_removed"] = len(passed) - len(deduped)

    return deduped, stats


def render_industrial_filter_ui(st_module, label: str = "固化剂"):
    """在Streamlit中渲染工业过滤控件，返回过滤配置。"""
    with st_module.expander(f"工业级候选过滤（{label}）", expanded=False):
        st_module.caption("排除药品/试剂/不适用的化合物，保留工业可用候选")
        col1, col2, col3 = st_module.columns(3)
        with col1:
            enable_filter = st_module.checkbox(f"启用{label}工业过滤", value=True, key=f"vs_ind_filter_enable_{label}")
            max_mp = st_module.number_input(f"最高估算熔点(°C)", 0, 500, 130, key=f"vs_ind_max_mp_{label}")
        with col2:
            min_mw = st_module.number_input(f"最小分子量", 20, 500, 80, key=f"vs_ind_min_mw_{label}")
            max_mw = st_module.number_input(f"最大分子量", 100, 2000, 800, key=f"vs_ind_max_mw_{label}")
        with col3:
            min_logp = st_module.number_input(f"最小LogP", -5, 5, -1, key=f"vs_ind_min_logp_{label}")
            max_logp = st_module.number_input(f"最大LogP", 0, 15, 6, key=f"vs_ind_max_logp_{label}")
        st_module.caption("已启用：PAINS排除 + Lipinski违规模块 + 重原子/可旋转键约束")
    return enable_filter, {"max_melting_point": float(max_mp), "min_mol_wt": float(min_mw), "max_mol_wt": float(max_mw), "min_logp": float(min_logp), "max_logp": float(max_logp)}

# ============================================================================
# 契约化高通量筛选（screening_plan / 正式-探索模式 / 候选级审计 / 持久化）
# 说明：正式模式（已发布且 gate 通过的模型 + prediction_contract）与探索模式
# （缺少 contract 或未发布模型）由 screening_mode_for_artifact 区分；工艺/测试
# 条件必须通过 fixed_inputs 显式固定，禁止用训练集均值/0/模板行填充。
# ============================================================================

_SCREENING_PLAN_REQUIRED_FIELDS = (
    "model_id",
    "model_version",
    "artifact_hash",
    "candidate_source",
    "resin_col",
    "hardener_col",
    "missing_value_rule",
    "chemistry_filter_rule",
    "melting_point_model_version",
    "random_state",
    "batch_params",
    "ranking_params",
    "fixed_inputs",
    "feature_cols",
    "screening_plan_hash",
)
_FIXED_INPUT_REQUIRED_KEYS = (
    "feature_id",
    "value",
    "unit",
    "source",
    "reviewer",
    "allow_missing",
    "allow_zero",
)
_PLAN_HASH_EXCLUDED_KEYS = (
    "screening_plan_hash",
    "contract",
    "created_at",
    "plan_metadata",
)
DEFAULT_MISSING_VALUE_RULE = "strict_required"
DEFAULT_CHEMISTRY_FILTER_RULE = "default_epoxy_rules"


def _as_plain_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise ValueError(f"期望 dict 类型，实际为 {type(value).__name__}")


def _as_list_of_mappings(value: Any, *, name: str) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是列表")
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name}[{index}] 必须是 dict")
        result.append(dict(item))
    return result


def _normalized_fixed_inputs(fixed_inputs: Any) -> List[Dict[str, Any]]:
    fixed = _as_list_of_mappings(fixed_inputs, name="fixed_inputs")
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(fixed):
        feature_id = str(item.get("feature_id") or "").strip()
        if not feature_id:
            raise ValueError(f"fixed_inputs[{index}] 缺少非空 feature_id")
        value = item.get("value")
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"fixed_inputs[{index}] 的 value 必须是数值或 None"
                )
        unit = str(item.get("unit") or "unknown").strip() or "unknown"
        source = str(item.get("source") or "unknown").strip() or "unknown"
        reviewer = str(item.get("reviewer") or "").strip()
        allow_missing = bool(item.get("allow_missing"))
        allow_zero = bool(item.get("allow_zero"))
        normalized.append(
            {
                "feature_id": feature_id,
                "value": value,
                "unit": unit,
                "source": source,
                "reviewer": reviewer,
                "allow_missing": allow_missing,
                "allow_zero": allow_zero,
            }
        )
    return normalized


def compute_screening_plan_hash(plan: Mapping[str, Any]) -> str:
    """sha256 over the plan minus hash/derived keys (stable + reproducible)."""
    payload = copy.deepcopy(dict(plan))
    for key in _PLAN_HASH_EXCLUDED_KEYS:
        payload.pop(key, None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_from_artifact_or_param(artifact: Any, contract: Any) -> Optional[Dict[str, Any]]:
    if isinstance(contract, Mapping):
        return dict(contract)
    if isinstance(artifact, Mapping):
        extra = artifact.get("extra")
        if isinstance(extra, Mapping):
            stored = extra.get("prediction_contract")
            if isinstance(stored, Mapping):
                return dict(stored)
    return None


def _screening_fixed_input_cols_from_contract(contract: Mapping[str, Any]) -> List[str]:
    """Collect fixed process columns declared by a prediction contract."""
    if not isinstance(contract, Mapping):
        return []
    result: List[str] = []
    seen: set = set()
    declared = contract.get("screening_fixed_input_cols")
    if isinstance(declared, (list, tuple)):
        for item in declared:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    manual = contract.get("manual_input_feature_cols")
    if isinstance(manual, (list, tuple)):
        for item in manual:
            if not isinstance(item, Mapping):
                text = str(item or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    result.append(text)
                continue
            marked = item.get("fixed") or item.get("is_fixed") or item.get("fixed_input")
            if bool(marked):
                text = str(item.get("name") or item.get("feature_id") or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    result.append(text)
    return result


def build_screening_plan(
    *,
    model_id: str,
    model_version: str,
    artifact_hash: str,
    contract: Optional[Mapping[str, Any]] = None,
    model_profile_id: Optional[str] = None,
    registry_hash: Optional[str] = None,
    workflow_hash: Optional[str] = None,
    candidate_source: Optional[str] = None,
    resin_col: Optional[str] = None,
    hardener_col: Optional[str] = None,
    fixed_inputs: Optional[Any] = None,
    missing_value_rule: Optional[str] = None,
    chemistry_filter_rule: Optional[Any] = None,
    melting_point_model_version: Optional[str] = None,
    random_state: int = 42,
    batch_params: Optional[Mapping[str, Any]] = None,
    ranking_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable, hash-locked screening plan.

    The plan records model identity, contract linkage, candidate source /
    component roles, explicitly fixed process-test inputs (``fixed_inputs``),
    missing-value / chemistry-filter rules, melting-point model version, the
    random seed and batch/ranking parameters.  With a ``contract`` the plan
    also mirrors ``contract.feature_cols`` and any contract-declared fixed
    input columns; without one the plan is flagged ``contract_missing: true``
    (exploratory screening).
    """
    normalized_contract = _contract_from_artifact_or_param(None, contract)
    contract_missing = normalized_contract is None

    if contract_missing:
        feature_cols: List[str] = []
        fixed_input_cols: List[str] = []
        contract_hash: Optional[str] = None
    else:
        raw_cols = normalized_contract.get("feature_cols")
        if isinstance(raw_cols, (list, tuple)):
            feature_cols = [str(col) for col in raw_cols]
        elif raw_cols is None:
            feature_cols = []
        else:
            feature_cols = [str(raw_cols)]
        fixed_input_cols = _screening_fixed_input_cols_from_contract(normalized_contract)
        contract_hash = str(normalized_contract.get("contract_hash") or "") or None

    normalized_fixed = _normalized_fixed_inputs(fixed_inputs)
    if contract_missing and not normalized_fixed and fixed_input_cols:
        # 正式契约声明了固定输入列但调用方未提供值：保留列名清单供上游补齐。
        pass

    plan: Dict[str, Any] = {
        "schema_version": 1,
        "model_id": str(model_id or "").strip(),
        "model_version": str(model_version or "").strip(),
        "artifact_hash": str(artifact_hash or "").strip(),
        "contract_hash": contract_hash,
        "contract_missing": contract_missing,
        "feature_cols": feature_cols,
        "fixed_input_cols": fixed_input_cols,
        "model_profile_id": str(model_profile_id or "").strip() or None,
        "registry_hash": str(registry_hash or "").strip() or None,
        "workflow_hash": str(workflow_hash or "").strip() or None,
        "candidate_source": str(candidate_source or "").strip() or "unknown",
        "resin_col": str(resin_col or "resin_smiles"),
        "hardener_col": str(hardener_col or "hardener_smiles"),
        "fixed_inputs": normalized_fixed,
        "missing_value_rule": str(missing_value_rule or DEFAULT_MISSING_VALUE_RULE),
        "chemistry_filter_rule": (
            copy.deepcopy(chemistry_filter_rule)
            if chemistry_filter_rule is not None
            else DEFAULT_CHEMISTRY_FILTER_RULE
        ),
        "melting_point_model_version": str(melting_point_model_version or "").strip() or None,
        "random_state": int(random_state) if random_state is not None else 42,
        "batch_params": copy.deepcopy(_as_plain_mapping(batch_params)),
        "ranking_params": copy.deepcopy(_as_plain_mapping(ranking_params)),
        "screening_plan_hash": "",
    }
    plan["screening_plan_hash"] = compute_screening_plan_hash(plan)
    return plan


def validate_screening_plan(plan: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(plan, Mapping):
        return ["screening_plan 必须是 dict"]
    for field in _SCREENING_PLAN_REQUIRED_FIELDS:
        if field not in plan:
            errors.append(f"缺少必需字段: {field}")
    if errors:
        return errors
    fixed = plan.get("fixed_inputs")
    if not isinstance(fixed, list):
        errors.append("fixed_inputs 必须是列表")
    else:
        for index, item in enumerate(fixed):
            if not isinstance(item, Mapping):
                errors.append(f"fixed_inputs[{index}] 必须是 dict")
                continue
            missing_keys = [k for k in _FIXED_INPUT_REQUIRED_KEYS if k not in item]
            if missing_keys:
                errors.append(
                    f"fixed_inputs[{index}] 缺少字段: {', '.join(missing_keys)}"
                )
                continue
            value = item.get("value")
            allow_missing = bool(item.get("allow_missing"))
            if value is None and not allow_missing:
                errors.append(
                    f"fixed_inputs[{index}]（{item.get('feature_id')}）value 为 None "
                    f"但 allow_missing=false"
                )
    if str(plan.get("screening_plan_hash") or "") != compute_screening_plan_hash(plan):
        errors.append("screening_plan_hash 与 plan 内容不一致")
    return errors


def apply_fixed_inputs(frame_or_dict: Any, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the plan's fixed process/test inputs to a candidate frame.

    Returns ``{"frame", "fixed_cols", "missing_records"}`` where
    ``missing_records`` lists ``{candidate_id or index, feature_id}`` for
    candidates that violate ``allow_missing=false`` fixed inputs (caller marks
    them ``MISSING_REQUIRED_INPUT``).  Fixed inputs are *explicit overrides*:
    they are never filled with training medians, zeros or template rows.
    """
    plan = plan if isinstance(plan, Mapping) else {}
    fixed_inputs = _normalized_fixed_inputs(plan.get("fixed_inputs"))
    if isinstance(frame_or_dict, pd.DataFrame):
        frame = frame_or_dict.copy()
        index_labels: List[Any] = list(frame.index)
        if "candidate_id" in frame.columns:
            index_labels = list(frame["candidate_id"])
    elif isinstance(frame_or_dict, Mapping):
        frame = pd.DataFrame([dict(frame_or_dict)])
        index_labels = [frame_or_dict.get("candidate_id", 0)]
    else:
        raise ValueError("apply_fixed_inputs 只接受 DataFrame 或 dict")

    missing_records: List[Dict[str, Any]] = []
    fixed_cols: List[str] = []
    for item in fixed_inputs:
        feature_id = str(item.get("feature_id") or "")
        value = item.get("value")
        allow_missing = bool(item.get("allow_missing"))
        allow_zero = bool(item.get("allow_zero"))
        if not feature_id:
            continue
        fixed_cols.append(feature_id)
        if value is None:
            # 值缺失：不允许静默填充均值/0，只记录违规候选。
            if not allow_missing:
                for pos, label in enumerate(index_labels):
                    missing_records.append(
                        {"candidate_id": label, "feature_id": feature_id}
                    )
            continue
        numeric_value = float(value)
        if numeric_value == 0.0 and not allow_zero:
            for pos, label in enumerate(index_labels):
                missing_records.append(
                    {"candidate_id": label, "feature_id": feature_id}
                )
            continue
        frame[feature_id] = numeric_value
    return {
        "frame": frame,
        "fixed_cols": fixed_cols,
        "missing_records": missing_records,
    }


def _entry_gate_ok(entry: Any) -> bool:
    gate = entry.get("gate_report") if isinstance(entry, Mapping) else None
    return (
        isinstance(gate, Mapping)
        and gate.get("ok") is True
        and str(gate.get("status") or "").strip().lower() == "valid"
    )


def screening_mode_for_artifact(
    artifact: Any = None,
    config: Any = None,
    material_type: Optional[str] = None,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Decide 'formal' vs 'exploratory' screening mode for one artifact.

    Formal mode requires ALL of: artifact.extra.prediction_contract, a
    publication entry with publication_status=published, a passing gate
    report, and a matching artifact hash.  Any missing item downgrades to
    exploratory mode with Chinese reasons recorded (and callers must persist
    ``result_is_formal=False``).

    When ``core.prediction_portal`` / ``core.portal_prediction`` can be
    imported (guarded — circular-import safe) they resolve the publication
    entry; otherwise this is a pure field check on ``artifact``/``entry``.
    """
    reasons: List[str] = []
    artifact_map = artifact if isinstance(artifact, Mapping) else {}
    extra = artifact_map.get("extra") if isinstance(artifact_map, Mapping) else None
    extra = extra if isinstance(extra, Mapping) else {}
    contract = extra.get("prediction_contract")
    has_contract = isinstance(contract, Mapping) and bool(contract)

    artifact_hash = ""
    if isinstance(artifact_map, Mapping):
        artifact_hash = str(artifact_map.get("artifact_hash") or "").strip().lower()

    entry: Optional[Mapping[str, Any]] = None
    if isinstance(config, Mapping) and material_type and target:
        try:
            from .portal_prediction import load_published_portal_model  # 防御性导入

            loaded = load_published_portal_model(config, material_type, target)
            entry = loaded.get("entry") if isinstance(loaded, Mapping) else None
        except Exception as exc:  # 未发布 / 未启用 / 校验失败等
            reasons.append(f"发布模型加载失败：{exc}")

    if entry is None and isinstance(config, Mapping):
        # 纯字段检查：在门户配置里寻找匹配 artifact_hash 的已发布 entry。
        try:
            materials = config.get("materials")
            if isinstance(materials, Mapping):
                material_block = materials.get(str(material_type)) if material_type else None
                targets = material_block.get("targets") if isinstance(material_block, Mapping) else None
                models = targets.get(str(target)) if isinstance(targets, Mapping) else None
                models = models.get("models") if isinstance(models, Mapping) else None
                if isinstance(models, list):
                    for item in models:
                        if not isinstance(item, Mapping):
                            continue
                        if str(item.get("artifact_hash") or "").strip().lower() == artifact_hash and artifact_hash:
                            entry = item
                            break
        except Exception:
            entry = None

    result_is_formal = True
    if not has_contract:
        result_is_formal = False
        reasons.append("artifact 缺少 prediction_contract，属于探索模式")

    if entry is None:
        result_is_formal = False
        if not reasons or "发布模型加载失败" not in str(reasons[0]):
            reasons.append("模型未在发布配置中登记为已发布版本")
    else:
        status = str(entry.get("publication_status") or "").strip().lower()
        if status != "published":
            result_is_formal = False
            reasons.append(f"模型发布状态为 {status or '未登记'}，不是 published")
        if entry.get("enabled") is not True:
            result_is_formal = False
            reasons.append("发布 entry 未启用（enabled != true）")
        if not _entry_gate_ok(entry):
            result_is_formal = False
            reasons.append("发布门禁报告缺失或未通过（gate_report 非 ok/valid）")
        entry_hash = str(entry.get("artifact_hash") or "").strip().lower()
        if artifact_hash and entry_hash and artifact_hash != entry_hash:
            result_is_formal = False
            reasons.append("artifact hash 与发布 entry 的 artifact_hash 不匹配")

    if result_is_formal:
        reasons = reasons or ["artifact 满足正式筛选要求：存在 prediction_contract 且已发布、门禁通过、hash 匹配"]
    if not artifact_map and not entry:
        reasons.append("未提供可判定的模型 artifact，按手动导入处理")

    mode = "formal" if result_is_formal else "exploratory"
    return {
        "mode": mode,
        "reasons": reasons,
        "contract": dict(contract) if has_contract else None,
        "result_is_formal": result_is_formal,
    }


def resolve_screening_feature_cols(artifact_or_contract: Any = None) -> Dict[str, Any]:
    """Resolve screening feature columns strictly for the current model.

    Formal mode: ``contract.feature_cols`` is the ONLY source.  Exploratory
    mode (no contract): fall back to artifact.feature_cols /
    extra.effective_feature_cols with warnings.  Session-state multi-source
    inference is intentionally not used here.
    """
    contract: Optional[Mapping[str, Any]] = None
    artifact: Optional[Mapping[str, Any]] = None
    if isinstance(artifact_or_contract, Mapping):
        if "feature_cols" in artifact_or_contract and (
            "schema_version" in artifact_or_contract or "contract_hash" in artifact_or_contract
        ):
            contract = artifact_or_contract
        else:
            artifact = artifact_or_contract
            extra = artifact.get("extra")
            if isinstance(extra, Mapping):
                stored = extra.get("prediction_contract")
                if isinstance(stored, Mapping):
                    contract = stored

    if isinstance(contract, Mapping) and contract.get("feature_cols"):
        cols = [str(col) for col in contract["feature_cols"]]
        return {
            "feature_cols": cols,
            "source": "contract",
            "warnings": [],
            "result_is_formal": True,
        }

    warnings: List[str] = []
    fallback: List[str] = []
    if isinstance(artifact, Mapping):
        extra = artifact.get("extra")
        extra = extra if isinstance(extra, Mapping) else {}
        for source_name, payload in (
            ("artifact.extra.effective_feature_cols", extra.get("effective_feature_cols")),
            ("artifact.feature_cols", artifact.get("feature_cols")),
        ):
            if isinstance(payload, (list, tuple)) and payload:
                fallback = [str(col) for col in payload]
                warnings.append(
                    f"无 prediction_contract，特征列回退自 {source_name}（探索模式）"
                )
                break
    if not fallback:
        warnings.append("未找到任何可用特征列（无 contract 且 artifact 缺少特征清单）")
    return {
        "feature_cols": fallback,
        "source": "artifact_fallback",
        "warnings": warnings,
        "result_is_formal": False,
    }


def screening_fixed_input_cols(contract: Any = None) -> List[str]:
    """Return contract-declared fixed process columns (empty when absent)."""
    return _screening_fixed_input_cols_from_contract(
        contract if isinstance(contract, Mapping) else {}
    )

def _audit_json_value(value: Any) -> Any:
    """Best-effort conversion of pandas/numpy values for JSONL output."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _audit_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_audit_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_screening_audit(
    plan: Mapping[str, Any],
    pool: Any,
    results: Any,
    out_dir: Any,
) -> str:
    """Persist plan + per-candidate audit + results + summary to disk.

    Writes ``screening_plan.json``, ``candidate_audit.jsonl``,
    ``screening_results.csv`` (only when ``results`` is a DataFrame) and
    ``audit_summary.json``; returns the audit directory path.
    """
    plan = plan if isinstance(plan, Mapping) else {}
    audit_dir = Path(str(out_dir))
    audit_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = copy.deepcopy(dict(plan))
    plan_path = audit_dir / "screening_plan.json"
    plan_path.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if pool is None:
        pool = []
    if isinstance(pool, pd.DataFrame):
        pool_records = [
            {str(k): _audit_json_value(v) for k, v in row.items()}
            for _, row in pool.iterrows()
        ]
    else:
        pool_records = [
            {str(k): _audit_json_value(v) for k, v in dict(item).items()}
            for item in pool
        ]
    audit_path = audit_dir / "candidate_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in pool_records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    results_payload = None
    if isinstance(results, pd.DataFrame):
        results_path = audit_dir / "screening_results.csv"
        results.to_csv(results_path, index=False, encoding="utf-8-sig")
        results_payload = str(results_path)

    funnel = summarize_funnel(pool_records)
    summary = {
        "model_id": plan.get("model_id"),
        "model_version": plan.get("model_version"),
        "contract_hash": plan.get("contract_hash"),
        "workflow_hash": plan.get("workflow_hash"),
        "registry_hash": plan.get("registry_hash"),
        "screening_plan_hash": plan.get("screening_plan_hash"),
        "result_is_formal": bool(
            plan.get("result_is_formal", not plan.get("contract_missing", True))
        ),
        "total_candidates": funnel.get("total"),
        "remaining_candidates": funnel.get("remaining"),
        "funnel": funnel.get("stages"),
        "status_counts": funnel.get("status_counts"),
        "results_file": results_payload,
    }
    summary_path = audit_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(audit_dir)
