# -*- coding: utf-8 -*-
"""平台同步诊断（diagnose_platform_sync / publication_verdict）与发布门禁降级规则测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from core.prediction_portal import (
    build_prediction_contract,
    compute_contract_hash,
    diagnose_platform_sync,
    publication_verdict,
    validate_publication_artifact,
)
from core.feature_registry import compute_registry_hash
from core.dataset_manifest import compute_dataset_manifest_hash


def _manifest_hash(manifest):
    return compute_dataset_manifest_hash(manifest)


class _NamedModel:
    # 模型实际输入顺序：与 _v2_artifact 的 feature_cols（workflow 在前）一致
    feature_names_in_ = ["MolWt_resin", "固化温度"]
    n_features_in_ = 2


def _approved_registry():
    features = [
        {
            "feature_id": "f_temp",
            "name": "固化温度",
            "source_type": "manual_input",
            "data_type": "float",
            "unit": "C",
            "required_for_prediction": True,
            "nullable": False,
            "default_policy": "explicit_only",
            "status": "approved",
        },
        {
            "feature_id": "f_mw",
            "name": "MolWt_resin",
            "source_type": "derived_workflow",
            "data_type": "float",
            "unit": "g/mol",
            "nullable": True,
            "default_policy": "workflow_only",
            "status": "approved",
            "calculation_rule": {
                "input_fields": ["resin_smiles"],
                "implementation": "core.molecular_features.mol_weight",
                "implementation_version": "1",
                "null_policy": "compute_zero",
                "invalid_policy": "mark_invalid",
            },
        },
    ]
    profile = {
        "material_type": "epoxy_resin",
        "target": "tg",
        "target_col": "tg",
        "feature_ids": ["f_temp", "f_mw"],
        "status": "approved",
    }
    registry = {
        "schema_version": 1,
        "registry_version": "2026.08.30-001",
        "features": features,
        "model_profiles": {"epoxy_resin.tg": profile},
        "approval": {"status": "approved"},
    }
    registry["registry_hash"] = compute_registry_hash(registry)
    registry["approval"]["approved_hash"] = registry["registry_hash"]
    return registry, profile


def _approved_manifest(registry, profile_id="epoxy_resin.tg"):
    return {
        "schema_version": 1,
        "dataset_id": "ds_test",
        "model_profile_id": profile_id,
        "source_bindings": [],
        "feature_bindings": [
            {
                "feature_id": item["feature_id"],
                "raw_columns": [item["name"]],
                "source_role": item["source_type"],
                "unit": item["unit"],
                "parse_rule_version": "1",
            }
            for item in registry["features"]
        ],
        "status": "approved",
    }


def _v2_artifact(feature_cols, *, contract_overrides=None, extra_overrides=None):
    """构造完整、自洽的 v2 artifact（contract 分区/registry/manifest 全部内嵌）。

    门禁的分区校验是顺序敏感的：feature_cols == workflow_cols + manual_cols，
    且模型 feature_names_in_ 顺序必须与 feature_cols 一致。因此固定为
    workflow 列（MolWt_resin）在前、manual 列（固化温度）在后，并让模型
    feature_names_in_ 与该顺序一致。
    """
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    feature_cols = ["MolWt_resin", "固化温度"]
    contract = {
        "schema_version": 2,
        "feature_cols": feature_cols,
        "target_col": "tg",
        "canonical_feature_cols": feature_cols,
        "effective_feature_cols": feature_cols,
        "removed_feature_cols": [],
        "removed_feature_reasons": {},
        "workflow_feature_cols": ["MolWt_resin"],
        "molecular_workflow_feature_cols": [],
        "derived_feature_cols": ["MolWt_resin"],
        "manual_input_feature_cols": ["固化温度"],
        "feature_definitions": [dict(item) for item in registry["features"]],
        "feature_registry_version": registry["registry_version"],
        "feature_registry_hash": registry["registry_hash"],
        "dataset_manifest_hash": _manifest_hash(manifest),
        "model_profile_id": "epoxy_resin.tg",
        "workflow_source_fields": [],
        "source_columns": [],
        "workflow_source_columns": [],
        "workflow_present": False,
        "workflow_hash": None,
        "workflow_schema_version": None,
        "molecular_features_indicated": False,
        "pipeline_present": False,
        "imputer_present": False,
        "scaler_present": False,
        "numeric_ranges": {"固化温度": {"min": 120.0, "max": 150.0}, "MolWt_resin": {"min": 340.0, "max": 380.0}},
        "contract_hash": "",
    }
    contract["contract_hash"] = compute_contract_hash(contract)
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": feature_cols,
        "target_col": "tg",
        "extra": {
            "prediction_contract": contract,
            "registry_snapshot": dict(registry),
            "dataset_manifest": {**manifest, "manifest_hash": _manifest_hash(manifest)},
        },
    }
    if contract_overrides:
        contract.update(contract_overrides)
        contract["contract_hash"] = compute_contract_hash(contract)
    if extra_overrides:
        artifact["extra"].update(extra_overrides)
    return artifact, contract, registry, manifest


def test_sync_diagnoses_fully_aligned_platform():
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    artifact, contract, _, _ = _v2_artifact(["固化温度", "MolWt_resin"])
    report = diagnose_platform_sync(
        registry=registry,
        profile=profile,
        manifest=manifest,
        artifact=artifact,
        contract=contract,
    )
    assert report["can_predict"] is True
    assert report["can_screen_formally"] is True
    assert report["partition_matches"] is True
    assert report["feature_counts"]["contract_features"] == 2
    # 门户输入特征 = manual(1) + workflow source fields(1，resin_smiles 输入)
    # 本 fixture 的 workflow_source_fields 为空且 manual 1 列 → 1
    assert report["feature_counts"]["portal_input_features"] == 1
    assert report["feature_counts"]["screening_features"] == 2
    assert report["unknown"] == []


def test_sync_none_inputs_do_not_crash():
    report = diagnose_platform_sync()
    assert isinstance(report, dict)
    assert report["can_predict"] is False
    assert report["can_publish"] is False
    assert report["can_screen_formally"] is False
    assert any(check["status"] == "missing" for check in report["checks"])


def test_sync_missing_contract_blocks_prediction():
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    artifact = {
        "model": _NamedModel(),
        "feature_cols": ["固化温度", "MolWt_resin"],
        "target_col": "tg",
        "extra": {},
    }
    report = diagnose_platform_sync(
        registry=registry, profile=profile, manifest=manifest, artifact=artifact,
    )
    assert report["can_predict"] is False
    assert report["can_screen_formally"] is False
    assert "prediction_blocked" in report["overall_status"]
    assert "screening_blocked" in report["overall_status"]


def test_sync_feature_order_mismatch_blocks_publish():
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    artifact, contract, _, _ = _v2_artifact(["固化温度", "MolWt_resin"])
    # 篡改 artifact 实际输入顺序（与 contract.feature_cols 相反）
    artifact["feature_cols"] = ["固化温度", "MolWt_resin"]
    report = diagnose_platform_sync(
        registry=registry, profile=profile, manifest=manifest, artifact=artifact, contract=contract,
    )
    assert report["feature_order_matches"] is False
    assert report["can_publish"] is False
    assert any(check["status"] == "error" for check in report["checks"])


def test_sync_partition_mismatch_blocks():
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    artifact, contract, _, _ = _v2_artifact(["固化温度", "MolWt_resin"])
    contract["manual_input_feature_cols"] = []
    contract["derived_feature_cols"] = ["MolWt_resin", "固化温度"]
    report = diagnose_platform_sync(
        registry=registry, profile=profile, manifest=manifest, artifact=artifact, contract=contract,
    )
    assert report["partition_matches"] is False
    assert report["can_predict"] is False


def test_sync_unregistered_feature_blocks():
    registry, profile = _approved_registry()
    manifest = _approved_manifest(registry)
    artifact = {
        "model": _NamedModel(),
        "feature_cols": ["固化温度", "未登记特征"],
        "target_col": "tg",
        "extra": {
            "prediction_contract": {
                "schema_version": 2,
                "feature_cols": ["固化温度", "未登记特征"],
                "manual_input_feature_cols": ["固化温度", "未登记特征"],
                "molecular_workflow_feature_cols": [],
                "derived_feature_cols": [],
                "workflow_source_fields": [],
                "contract_hash": "x",
            }
        },
    }
    report = diagnose_platform_sync(
        registry=registry, profile=profile, manifest=manifest, artifact=artifact,
    )
    assert report["can_predict"] is False
    assert "未登记特征" in report["unknown"]


def test_publication_validation_blocks_removed_features():
    """训练删除了 contract 声明的特征（effective != canonical）→ needs_validation。"""
    artifact, contract, registry, manifest = _v2_artifact(["固化温度", "MolWt_resin"])
    contract.update({
        "effective_feature_cols": ["MolWt_resin"],
        "removed_feature_cols": ["固化温度"],
        "removed_feature_reasons": {"固化温度": "feature_mask"},
    })
    contract["contract_hash"] = compute_contract_hash(contract)
    artifact["extra"]["prediction_contract"] = contract
    report = validate_publication_artifact(artifact, contract)
    assert report["ok"] is False
    assert report["status"] == "needs_validation"


def test_publication_validation_blocks_failed_feature_audit():
    artifact, contract, registry, manifest = _v2_artifact(["固化温度", "MolWt_resin"])
    artifact["extra"]["feature_audit"] = {"publishable": False}
    report = validate_publication_artifact(artifact, contract)
    assert report["ok"] is False
    assert report["status"] == "needs_validation"


def test_publication_validation_legacy_artifact_needs_validation():
    artifact = {
        "model": _NamedModel(),
        "pipeline": None,
        "feature_cols": ["固化温度"],
        "target_col": "tg",
        "extra": {},
    }
    report = validate_publication_artifact(artifact)
    assert report["ok"] is False
    assert report["status"] == "needs_validation"


def test_publication_verdict_blocks_hash_mismatch(tmp_path: Path):
    artifact, contract, registry, manifest = _v2_artifact(["固化温度", "MolWt_resin"])
    payload = json.dumps(
        {"feature_cols": artifact["feature_cols"], "target_col": artifact["target_col"]},
        ensure_ascii=False,
    ).encode("utf-8")
    artifact_file = tmp_path / "model.joblib"
    artifact_file.write_bytes(payload)
    real_hash = hashlib.sha256(payload).hexdigest()

    entry = {
        "model_id": "m1",
        "model_version": "v1",
        "artifact_path": str(artifact_file),
        "artifact_hash": real_hash,
        "publication_status": "published",
        "enabled": True,
        "gate_report": {"ok": True, "status": "valid", "errors": []},
    }
    verdict = publication_verdict(entry, config=None, artifact=artifact)
    assert verdict["verdict"] == "publishable", verdict

    artifact_file.write_bytes(payload + b"tampered")
    verdict = publication_verdict(entry, config=None, artifact=artifact)
    assert verdict["verdict"] == "not_publishable"
    assert any("hash" in reason for reason in verdict["reasons"])


def test_publication_verdict_requires_published_status():
    entry = {
        "artifact_path": "",
        "artifact_hash": "",
        "publication_status": "needs_validation",
        "enabled": False,
        "gate_report": {"ok": False, "status": "needs_validation"},
    }
    verdict = publication_verdict(entry, config=None, artifact=None)
    assert verdict["verdict"] == "not_publishable"
    assert verdict["reasons"]
