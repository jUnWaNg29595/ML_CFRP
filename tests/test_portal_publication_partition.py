"""发布门禁：workflow 特征与 contract / removed 特征的关系校验。

背景（实测，见 docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.1.1）：
真实训练平台导出的 artifact 存在三层特征结构——

    workflow.final_feature_names           253  分子特征 workflow 产出
    contract.feature_cols == effective     284  模型实际消费的列
    feature_audit.removed_feature_cols       5  训练侧因 feature_mask 删除的列
    feature_audit.canonical_feature_cols   289  = 284 + 5（删除前全量）

恒等式（已实测）：
    contract.feature_cols ∪ removed == canonical_feature_cols
    workflow ∩ removed == 那 5 个"多出"特征

因此门禁规则必须是 workflow ⊆ (contract ∪ removed)，而不是
workflow ⊆ contract（后者会被那 5 个删除特征误判为失败）。
"""

import copy

import pytest

from core.prediction_portal import validate_publication_artifact


class _NamedModel:
    def __init__(self, feature_names):
        self.feature_names_in_ = list(feature_names)
        self.n_features_in_ = len(self.feature_names_in_)


class _SimpleImputer:
    """最小可用 imputer：让 imputer_present 与契约声明一致。"""

    def __init__(self):
        # 必须是实例属性：_has_meaningful_learned_attribute 用 vars() 检查
        self.statistics_ = [0.0, 0.0]
        self.n_features_in_ = 2

    def transform(self, values):
        return values


def _legacy_contract(**overrides):
    """schema-1（legacy）契约：训练平台导入模型的真实形态。"""
    contract = {
        "schema_version": 1,
        "feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"],
        "target_col": "Tg",
        "workflow_hash": "workflow-123",
        "workflow_schema_version": 3,
        "source_columns": [{"column": "resin_smiles_1", "roles": ["resin"]}],
        "workflow_source_columns": [{"column": "resin_smiles_1", "roles": ["resin"]}],
        "workflow_source_fields": [{"column": "resin_smiles_1", "roles": ["resin"]}],
        "workflow_present": True,
        "molecular_features_indicated": True,
        "pipeline_present": False,
        "imputer_present": True,
        "scaler_present": False,
        "numeric_ranges": {},
    }
    contract.update(overrides)
    return contract


def _artifact(*, workflow_features, feature_cols, removed=None, canonical=None):
    workflow = {
        "schema_version": 3,
        "workflow_hash": "workflow-123",
        "steps": [
            {
                "step_id": "resin",
                "role": "resin",
                "source_columns": ["resin_smiles_1"],
                "order": 1,
            }
        ],
        "merge_order": ["resin"],
        "final_feature_names": list(workflow_features),
    }
    extra = {"molecular_feature_workflow": workflow}
    if removed is not None or canonical is not None:
        extra["feature_audit"] = {
            "canonical_feature_cols": list(canonical or []),
            "effective_feature_cols": list(feature_cols),
            "removed_feature_cols": list(removed or []),
            "removed_feature_reasons": {c: "feature_mask" for c in (removed or [])},
            "publishable": not removed,
        }
    return {
        "model": _NamedModel(feature_cols),
        "pipeline": None,
        "imputer": _SimpleImputer(),
        "feature_cols": list(feature_cols),
        "target_col": "Tg",
        "extra": extra,
    }


def test_workflow_feature_removed_by_training_audit_is_publishable():
    """workflow 产出被训练侧删除的特征（在 removed 中）时，不得阻断发布。

    这是真实 artifact 的场景：workflow 的 5 个特征不在 contract.feature_cols 中，
    但都在 feature_audit.removed_feature_cols 中。
    """
    artifact = _artifact(
        workflow_features=["resin_xtb_gap", "curing_agent_xtb_gap", "resin_1_structure_xtb_dipole"],
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
        removed=["resin_1_structure_xtb_dipole"],
        canonical=["resin_xtb_gap", "curing_agent_xtb_gap", "resin_1_structure_xtb_dipole"],
    )
    contract = _legacy_contract()

    report = validate_publication_artifact(artifact, contract)

    assert not any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def test_workflow_feature_unknown_to_contract_and_audit_is_publishable():
    """workflow 产出契约未知的特征时**放行**（多余列由 reindex 安全丢弃）。

    规则演化（基于实测）：
    - 最初：workflow 必须与 contract 完全相等
    - 修正 1：workflow ⊆ contract ∪ removed（处理训练侧删除的特征）
    - 修正 2（本测试）：workflow **可以多于** contract —— 因为
      ``core/portal_prediction`` 会 ``reindex(columns=contract['feature_cols'])``
      丢弃多余列。实测 TabPFN artifact 的 workflow 有 2131 个特征名，
      contract 只有 2070 个，多出的 115 个 xtb/ff 描述符既不在 effective
      也不在 removed 中，但它们并不影响预测。

    安全性由**运行时**保证：契约要求的非 workflow 特征由
    ``_merge_explicit_model_features`` 强制补齐，缺一即抛错。
    """
    artifact = _artifact(
        workflow_features=["resin_xtb_gap", "totally_unknown_gap"],
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
        removed=[],
        canonical=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )
    contract = _legacy_contract()

    report = validate_publication_artifact(artifact, contract)

    assert not any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def test_missing_workflow_output_still_enforced_at_runtime():
    """workflow 多产出放行后，契约缺失的非 workflow 特征仍由运行时拦截。"""
    from core.portal_prediction import _explicit_model_feature_names

    explicit = _explicit_model_feature_names(
        contract={"feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"]},
        workflow={"final_feature_names": ["resin_xtb_gap"]},
    )

    assert "curing_agent_xtb_gap" in explicit


def test_workflow_subset_of_contract_is_publishable():
    """workflow ⊆ contract（无删除特征）→ 不得因 final_feature_names 报错。"""
    artifact = _artifact(
        workflow_features=["resin_xtb_gap"],
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
        removed=[],
        canonical=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )
    contract = _legacy_contract()

    report = validate_publication_artifact(artifact, contract)

    assert not any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def test_empty_workflow_feature_names_is_rejected():
    """final_feature_names 为空 → 必须拒绝（保留原校验）。"""
    artifact = _artifact(
        workflow_features=[],
        feature_cols=["resin_xtb_gap"],
        removed=[],
        canonical=["resin_xtb_gap"],
    )
    contract = _legacy_contract(feature_cols=["resin_xtb_gap"])

    report = validate_publication_artifact(artifact, contract)

    assert any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def _v2_registry(*, feature_cols, workflow_feature_cols):
    return {
        "registry_version": "v1",
        "registry_hash": "r1",
        "features": [
            {
                "feature_id": name,
                "name": name,
                "source_type": (
                    "molecular_workflow" if name in workflow_feature_cols else "manual_input"
                ),
                "status": "approved",
            }
            for name in feature_cols
        ],
    }


def _v2_contract(*, feature_cols, workflow_feature_cols, removed=None, canonical=None,
                 effective=None, artifact=None):
    """用 build_prediction_contract 构造合法的 v2 契约（含正确 contract_hash）。"""
    from core.prediction_portal import build_prediction_contract

    registry = _v2_registry(
        feature_cols=feature_cols, workflow_feature_cols=workflow_feature_cols
    )
    return build_prediction_contract(
        artifact=artifact or {"model": _NamedModel(feature_cols), "pipeline": None},
        feature_cols=feature_cols,
        target_col="Tg",
        registry_snapshot=registry,
        dataset_manifest={"manifest_hash": "m1"},
        model_profile_id="p",
        canonical_feature_cols=canonical or feature_cols,
        effective_feature_cols=effective or feature_cols,
        removed_feature_cols=removed or [],
        removed_feature_reasons={c: "feature_mask" for c in (removed or [])},
    )


def test_v2_contract_uses_removed_feature_cols_from_contract():
    """v2 契约从 contract.removed_feature_cols 读取删除特征（而非 feature_audit）。"""
    artifact = _artifact(
        workflow_features=["resin_xtb_gap", "dropped_gap"],
        feature_cols=["resin_xtb_gap"],
        removed=None,
        canonical=None,
    )
    contract = _v2_contract(
        feature_cols=["resin_xtb_gap"],
        workflow_feature_cols=["resin_xtb_gap"],
        canonical=["resin_xtb_gap", "dropped_gap"],
        effective=["resin_xtb_gap"],
        removed=["dropped_gap"],
        artifact=artifact,
    )
    registry = _v2_registry(
        feature_cols=["resin_xtb_gap"], workflow_feature_cols=["resin_xtb_gap"]
    )

    report = validate_publication_artifact(
        artifact, contract, registry_snapshot=registry,
        dataset_manifest={"manifest_hash": "m1"},
    )

    assert not any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def test_workflow_feature_outside_declared_workflow_partition_is_rejected():
    """契约声明 workflow 分区时，workflow 产出越界 → 拒绝。"""
    artifact = _artifact(
        workflow_features=["resin_xtb_gap", "not_a_workflow_feature"],
        feature_cols=["resin_xtb_gap", "not_a_workflow_feature"],
        removed=None,
        canonical=None,
    )
    contract = _legacy_contract(
        feature_cols=["resin_xtb_gap", "not_a_workflow_feature"],
        workflow_feature_cols=["resin_xtb_gap"],  # 未包含 not_a_workflow_feature
    )

    report = validate_publication_artifact(artifact, contract)

    assert any("not_a_workflow_feature" in str(e) for e in report["errors"]), report["errors"]


def test_workflow_feature_inside_declared_workflow_partition_is_publishable():
    """契约声明的 workflow 分区包含全部 workflow 产出 → 不得因分区报错。"""
    artifact = _artifact(
        workflow_features=["resin_xtb_gap"],
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
        removed=None,
        canonical=None,
    )
    contract = _legacy_contract(workflow_feature_cols=["resin_xtb_gap"])

    report = validate_publication_artifact(artifact, contract)

    assert not any("workflow_feature_cols" in str(e) for e in report["errors"]), report["errors"]


# ---------------------------------------------------------------------------
# 第二道门禁：_is_publishable_ui_model 对 legacy(schema-1) 导入模型的可见性
# ---------------------------------------------------------------------------

class _UiArtifact:
    """最小可用的 legacy artifact：让 validate_publication_artifact 返回 ok=True。"""

    def __init__(self, feature_cols):
        self._cols = list(feature_cols)

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def feature_cols(self):
        return self._cols


def _ui_model(*, schema_version, gate_ok=True, gate_status="valid", enabled=True,
              status="published", snapshot=None, artifact=None, removed=None):
    contract = _legacy_contract()
    if schema_version is not None:
        contract["schema_version"] = schema_version
    if removed is not None:
        contract["removed_feature_cols"] = list(removed)
    model = {
        "enabled": enabled,
        "publication_status": status,
        "gate_report": {"ok": gate_ok, "status": gate_status},
        "contract": contract,
    }
    if snapshot is not None:
        model["registry_snapshot"] = snapshot
    if artifact is not None:
        model["_artifact"] = artifact
    return model


def test_legacy_imported_model_is_publishable_in_ui():
    """方案 b：schema-1（训练平台导入）模型通过门禁后应在预测页可见。

    实测背景：真实导入模型 contract.schema_version=None、registry_snapshot 缺失，
    旧逻辑要求 schema_version==2 + approved profile，导致模型永远无法启用。
    """
    import UserPrediction

    model = _ui_model(schema_version=None)

    assert UserPrediction._is_publishable_ui_model(model) is True


def test_model_without_gate_report_is_not_publishable():
    """未经门禁（gate_report 缺失/失败）的模型一律不可见。"""
    import UserPrediction

    assert UserPrediction._is_publishable_ui_model(_ui_model(schema_version=None, gate_ok=False)) is False
    assert UserPrediction._is_publishable_ui_model(_ui_model(schema_version=None, gate_status="invalid")) is False
    assert UserPrediction._is_publishable_ui_model(
        {"enabled": True, "publication_status": "published"}
    ) is False


def test_disabled_or_unpublished_model_is_not_publishable():
    """未启用 / 未发布状态的模型不可见（硬条件不得放宽）。"""
    import UserPrediction

    assert UserPrediction._is_publishable_ui_model(_ui_model(schema_version=None, enabled=False)) is False
    assert UserPrediction._is_publishable_ui_model(_ui_model(schema_version=None, status="needs_validation")) is False


def test_v2_contract_still_requires_approved_registry_snapshot():
    """v2 契约仍保持严格语义：必须有 approved 的 snapshot 与 profile。"""
    import UserPrediction

    # v2 + 无 snapshot → 拒绝
    assert UserPrediction._is_publishable_ui_model(_ui_model(schema_version=2)) is False
    # v2 + snapshot 但 profile 未批准 → 拒绝
    assert UserPrediction._is_publishable_ui_model(
        _ui_model(schema_version=2, snapshot={"model_profile": {"status": "pending"}, "features": []})
    ) is False


# ---------------------------------------------------------------------------
# workflow 多余产出（mask/pipeline 场景）：会被 reindex 安全丢弃，不应阻断发布
# ---------------------------------------------------------------------------

def test_workflow_superset_of_contract_is_publishable():
    """workflow 产出多于 contract 时应放行（多余列由 reindex 丢弃）。

    实测背景：TabPFN artifact 的 workflow.final_feature_names 有 2131 个，
    而 contract（mask 前）只有 2070 个 —— workflow 多出 115 个分子特征
    （xtb/ff 描述符），它们既不在 effective 也不在 removed 中，但
    ``portal_prediction`` 会执行 ``features.reindex(columns=contract['feature_cols'])``
    把多余列安全丢弃，不影响预测。

    安全性由运行时保证：契约要求的 54 个非 workflow 特征由
    ``_merge_explicit_model_features`` 强制补齐，缺一即抛错。
    """
    artifact = _artifact(
        workflow_features=["resin_xtb_gap", "extra_a", "extra_b"],
        feature_cols=["resin_xtb_gap", "curing_agent_xtb_gap"],
        removed=[],
        canonical=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )
    contract = _legacy_contract()

    report = validate_publication_artifact(artifact, contract)

    assert not any("final_feature_names" in str(e) for e in report["errors"]), report["errors"]


def test_missing_non_workflow_features_still_enforced_at_runtime():
    """workflow 多余产出放行后，缺失的非 workflow 特征仍由运行时拦截。"""
    from core.portal_prediction import _explicit_model_feature_names

    workflow = {"final_feature_names": ["resin_xtb_gap"]}
    contract = {"feature_cols": ["resin_xtb_gap", "curing_agent_xtb_gap"]}

    explicit = _explicit_model_feature_names(contract, workflow)

    assert "curing_agent_xtb_gap" in explicit
