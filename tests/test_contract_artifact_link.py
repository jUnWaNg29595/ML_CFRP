# -*- coding: utf-8 -*-
"""训练契约 → artifact 链路测试（Agent B 范围）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.feature_registry import compute_registry_hash
from core.training_contract import attach_feature_audit, audit_training_result, lock_training_contract
from core.model_io import (
    artifact_hash_from_bytes,
    compute_artifact_hash,
    create_model_artifact,
    create_model_artifact_bytes,
    loads_artifact,
    dumps_artifact,
)


def _approved_registry():
    registry = {
        "schema_version": 1,
        "registry_version": "v1",
        "features": [
            {
                "feature_id": "x", "name": "固化温度", "source_type": "manual_input",
                "unit": "C", "default_policy": "explicit_only", "status": "approved",
                "data_type": "float", "required_for_prediction": True, "nullable": False,
            },
            {
                "feature_id": "y", "name": "MolWt_resin", "source_type": "derived_workflow",
                "unit": "g/mol", "default_policy": "workflow_only", "status": "approved",
                "data_type": "float", "nullable": True,
                "calculation_rule": {
                    "input_fields": ["resin_smiles"], "implementation": "core.m",
                    "implementation_version": "1", "null_policy": "compute_zero",
                    "invalid_policy": "mark_invalid",
                },
            },
        ],
        "model_profiles": {
            "p": {
                "feature_ids": ["x", "y"], "status": "approved", "target_col": "tg_c",
                "material_type": "epoxy_resin", "target": "tg",
            }
        },
        "approval": {"status": "approved"},
    }
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    return registry


def _approved_manifest():
    return {
        "schema_version": 1, "dataset_id": "d", "model_profile_id": "p",
        "source_bindings": [],
        "feature_bindings": [
            {"feature_id": "x", "raw_columns": ["固化温度"], "source_role": "manual_input", "unit": "C", "parse_rule_version": "1"},
            {"feature_id": "y", "raw_columns": ["MolWt_resin"], "source_role": "derived_workflow", "unit": "g/mol", "parse_rule_version": "1"},
        ],
        "status": "approved",
    }


def _lock(tmp_path: Path):
    registry = _approved_registry()
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    manifest = _approved_manifest()
    frame = pd.DataFrame({"固化温度": [120.0, 150.0], "MolWt_resin": [340.0, 380.0]})
    context = lock_training_contract(
        path, manifest, "epoxy_resin", "tg", "tg_c",
        ["固化温度", "MolWt_resin"], frame, None,
    )
    return context


def test_lock_training_contract_includes_prediction_contract(tmp_path):
    context = _lock(tmp_path)
    contract = context.get("prediction_contract")
    assert isinstance(contract, dict)
    assert contract["schema_version"] == 2
    assert contract["manual_input_feature_cols"] == ["固化温度"]
    assert contract["workflow_feature_cols"] == ["MolWt_resin"]
    assert contract["feature_cols"] == ["固化温度", "MolWt_resin"]


def test_lock_training_contract_hash_is_stable(tmp_path):
    first = _lock(tmp_path)
    second = _lock(tmp_path)
    assert first.get("contract_hash")
    assert first["contract_hash"] == second["contract_hash"]
    assert first.get("contract_errors") == []


def test_attach_feature_audit_writes_context():
    context = {"canonical_feature_cols": ["a", "b"], "feature_cols": ["a", "b"]}
    result = {"feature_names": ["a", "b"], "feature_mask": [True, False]}
    updated = attach_feature_audit(context, result)
    assert updated["feature_audit"]["publishable"] is False
    assert updated["feature_audit"]["removed_feature_cols"] == ["b"]
    # audit_training_result 语义保持
    audit = audit_training_result({"canonical_feature_cols": ["a"]}, {"feature_names": ["a"]})
    assert audit["publishable"] is True


def test_create_model_artifact_extra_contains_contract_and_audit(tmp_path):
    context = _lock(tmp_path)
    attach_feature_audit(context, {"feature_names": ["固化温度", "MolWt_resin"]})
    artifact = create_model_artifact(
        model_name="demo", target_col="tg_c", feature_cols=["固化温度", "MolWt_resin"],
        model=None, contract_context=context,
    )
    assert artifact["extra"]["prediction_contract"]["schema_version"] == 2
    assert artifact["extra"]["feature_audit"]["publishable"] is True
    assert artifact["extra"]["registry_snapshot"]["registry_hash"]


def test_create_model_artifact_bytes_embeds_artifact_hash(tmp_path):
    context = _lock(tmp_path)
    data = create_model_artifact_bytes(
        model_name="demo", target_col="tg_c", feature_cols=["固化温度", "MolWt_resin"],
        model=None, contract_context=context,
    )
    artifact = loads_artifact(data)
    embedded = artifact["extra"].get("artifact_hash")
    assert isinstance(embedded, str) and len(embedded) == 64
    # round-trip：对去 hash 后的 payload 重算必须一致
    assert compute_artifact_hash(artifact) == embedded
    assert artifact_hash_from_bytes(data) == embedded


def test_compute_artifact_hash_changes_when_feature_cols_change():
    a1 = create_model_artifact(model_name="m", target_col="t", feature_cols=["x"], model=None)
    a2 = create_model_artifact(model_name="m", target_col="t", feature_cols=["x", "y"], model=None)
    assert compute_artifact_hash(a1) != compute_artifact_hash(a2)
