import pandas as pd
import pytest


def test_portal_renders_only_manual_input_as_editable_fields():
    from UserPrediction import build_manual_input_fields, build_workflow_source_fields

    registry = {"schema_version": 1, "model_profile": {"status": "approved"}, "features": [
        {"feature_id": "m", "name": "pressure", "label": "固化压力", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "valid_range": {"min": 0, "max": 20}, "status": "approved"},
        {"feature_id": "d", "name": "cure_stage_count", "label": "固化阶段数", "source_type": "derived_workflow", "status": "approved"},
    ]}
    contract = {"schema_version": 2, "manual_input_feature_cols": ["pressure"], "workflow_source_fields": [{"column": "cure_schedule", "roles": ["derived"]}]}
    assert [field["name"] for field in build_manual_input_fields(contract, registry)] == ["pressure"]
    assert [field["name"] for field in build_workflow_source_fields(contract, registry)] == ["cure_schedule"]


def test_missing_manual_input_is_rejected_without_default_fill(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    monkeypatch.setattr(
        "core.portal_prediction.load_published_portal_model",
        lambda *args: {
            "entry": {"id": "v1"},
            "artifact": {"model": object(), "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}},
            "contract": {
                "schema_version": 2,
                "feature_cols": ["pressure"],
                "manual_input_feature_cols": ["pressure"],
                "workflow_feature_cols": [],
                "molecular_workflow_feature_cols": [],
                "derived_feature_cols": [],
                "feature_definitions": [{"feature_id": "p", "name": "pressure", "source_type": "manual_input", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}],
                "target_col": "tg_c",
                "numeric_ranges": {"pressure": {"min": 0, "max": 20}},
                "missing_value_policy": "reject_user_missing",
            },
        },
    )
    errors = validate_prediction_request({"material_type": "epoxy_resin", "target": "tg", "inputs": {}, "confirmed_by_user": True}, {"materials": {}})
    assert any("pressure" in error for error in errors)


def test_manual_fields_require_approved_manual_registry_definition():
    from UserPrediction import build_manual_input_fields

    contract = {"manual_input_feature_cols": ["pressure", "draft_value"]}
    registry = {"schema_version": 1, "model_profile": {"status": "approved"}, "features": [
        {"name": "pressure", "source_type": "manual_input", "status": "approved", "default_policy": "explicit_only"},
        {"name": "draft_value", "source_type": "manual_input", "status": "draft", "default_policy": "explicit_only"},
    ]}
    fields = build_manual_input_fields(contract, registry)
    assert [field["name"] for field in fields] == ["pressure"]


def test_workflow_fields_ignore_legacy_aliases_and_use_canonical_contract_key():
    """v2 契约只认规范键 ``workflow_source_fields``，不认遗留别名。

    规则演化（2026-09-22 导入模型修复）：
    - v2 契约（有注册表审核）仍严格：快照无效不渲染，只读规范键。
    - legacy(schema-1) 契约改为**渲染** ``workflow_source_fields`` / ``source_columns``，
      因为运行时确实要求这些源列（见下面的 legacy 专项测试）。
    """
    from UserPrediction import build_workflow_source_fields

    registry = {"schema_version": 1, "model_profile": {"status": "approved"}, "features": []}
    # v2：忽略仅存在于 source_columns 的遗留列
    assert [field["name"] for field in build_workflow_source_fields({
        "schema_version": 2,
        "workflow_source_fields": [{"column": "cure_schedule", "roles": ["derived"]}],
        "source_columns": [{"column": "legacy"}],
    }, registry)] == ["cure_schedule"]
    # v2 且快照无效：不得渲染（保持严格语义）
    assert build_workflow_source_fields(
        {"schema_version": 2, "workflow_source_fields": [{"column": "legacy"}]}, {}
    ) == []


def test_legacy_contract_renders_workflow_source_fields_without_registry_snapshot():
    """legacy 契约必须渲染分子源列，否则用户无法提供运行时要求的源列。

    实测背景：训练平台导入的 Tg 模型契约 schema_version=1、registry_snapshot={}，
    但后端 ``core.portal_prediction`` 会要求 8 个 ``*_structure`` 源列。
    若此处不渲染，页面只剩「当前模型契约未声明任何输入字段」。
    """
    from UserPrediction import build_workflow_source_fields

    contract = {
        "schema_version": 1,
        "workflow_source_fields": [{"column": "resin_1_structure", "roles": ["resin"]}],
    }
    fields = build_workflow_source_fields(contract, {})
    assert [field["name"] for field in fields] == ["resin_1_structure"]
    # 训练侧列名是 *_structure 而非 *_smiles，同样必须走 SMILES 输入控件
    assert fields[0]["kind"] == "smiles"


def test_legacy_contract_derives_manual_fields_when_registry_definitions_absent():
    """legacy 契约无注册表定义时，人工字段按 feature_cols − workflow 合成。

    口径必须与 ``core.portal_prediction._explicit_model_feature_names`` 一致，
    否则会出现「UI 让用户填 A、后端却要 B」的错配。
    """
    from UserPrediction import build_manual_input_fields

    contract = {
        "schema_version": 1,
        "feature_cols": ["resin_1_amount_phr", "process_max_temperature_c", "resin_1_structure_xtb_gap"],
        "workflow_final_feature_names": ["resin_1_structure_xtb_gap"],
    }
    names = [field["name"] for field in build_manual_input_fields(contract, {})]
    assert names == ["resin_1_amount_phr", "process_max_temperature_c"]


def test_v2_contract_without_manual_partition_still_yields_no_fields():
    """v2 契约未声明人工分区 = 确实无人工输入，不得猜。"""
    from UserPrediction import build_manual_input_fields

    contract = {
        "schema_version": 2,
        "feature_cols": ["fp_bit_1", "fp_bit_2"],
        "workflow_final_feature_names": [],
    }
    assert build_manual_input_fields(contract, {}) == []


def test_nullable_optional_workflow_source_can_be_empty(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    contract = {
        "schema_version": 2,
        "feature_cols": ["resin_x"], "manual_input_feature_cols": [],
        "workflow_feature_cols": ["resin_x"], "molecular_workflow_feature_cols": ["resin_x"],
        "derived_feature_cols": [], "feature_definitions": [],
        "target_col": "tg_c", "numeric_ranges": {},
        "workflow_source_fields": [{"column": "resin_smiles", "roles": ["resin"], "required": True, "nullable": False},
                                    {"column": "curing_agent_smiles", "roles": ["hardener"], "required": False, "nullable": True}],
    }
    artifact = {"model": object(), "feature_cols": ["resin_x"], "target_col": "tg_c", "extra": {}}
    monkeypatch.setattr("core.portal_prediction.load_published_portal_model", lambda *a: {"entry": {}, "artifact": artifact, "contract": contract})
    monkeypatch.setattr("core.portal_prediction._structure_error", lambda value: None)
    monkeypatch.setattr("core.portal_prediction.validate_single_row_source_values", lambda frame, sources: {"ok": True, "missing_columns": [], "empty_columns": []})
    errors = validate_prediction_request({"material_type": "epoxy_resin", "target": "tg", "inputs": {"resin_smiles": "CCO", "curing_agent_smiles": ""}, "confirmed_by_user": True}, {})
    assert not any("curing_agent_smiles" in error and ("不能为空" in error or "empty" in error.lower()) for error in errors)


def test_optional_curating_agent_pandas_na_runs_real_request_validation(monkeypatch):
    from core.portal_prediction import validate_prediction_request

    contract = {
        "schema_version": 2,
        "feature_cols": ["resin_x"], "manual_input_feature_cols": [],
        "workflow_feature_cols": ["resin_x"], "molecular_workflow_feature_cols": ["resin_x"], "derived_feature_cols": [],
        "feature_definitions": [], "target_col": "tg_c", "numeric_ranges": {"resin_x": {"min": 0, "max": 10}},
        "workflow_source_fields": [
            {"column": "resin_smiles", "roles": ["resin"], "required": True, "nullable": False},
            {"column": "curing_agent_smiles", "roles": ["hardener"], "required": False, "nullable": True},
        ],
    }
    artifact = {"model": object(), "feature_cols": ["resin_x"], "target_col": "tg_c", "extra": {}}
    monkeypatch.setattr("core.portal_prediction.load_published_portal_model", lambda *args: {"entry": {}, "artifact": artifact, "contract": contract})
    errors = validate_prediction_request(
        {"material_type": "epoxy_resin", "target": "tg", "inputs": {"resin_smiles": "CCO", "curing_agent_smiles": pd.NA, "resin_x": 1.0}, "confirmed_by_user": True},
        {},
    )
    assert not any("curing_agent_smiles" in error for error in errors)


# ---------------------------------------------------------------------------
# 导入模型回归：发布成功但配置里 contract=null（2026-09-22）
# ---------------------------------------------------------------------------

def test_publish_imported_entry_writes_back_contract_and_registry_snapshot():
    """发布成功必须回写 entry.contract，否则 UI/后端都拿不到输入契约。

    根因（实测）：``publish_imported_entry`` 只在发布时**校验**推导出的契约，
    不写回 ``entry.contract``；而 artifact 自身也不带 ``extra.prediction_contract``
    → 配置里 ``contract: null`` → 预测页报
    「当前模型契约未声明任何输入字段，无法手动预测」。
    """
    from core.prediction_portal import publish_imported_entry

    class _NamedModel:
        feature_names_in_ = ["a_phr", "resin_1_structure_xtb_gap"]
        n_features_in_ = 2

    entry = {
        "id": "m1",
        "enabled": False,
        "publication_status": "needs_validation",
        "gate_report": {},
        "feature_cols": ["a_phr", "resin_1_structure_xtb_gap"],
        "target_col": "tg_c",
        "version": "",
        "_artifact": {
            "model": _NamedModel(),
            "pipeline": None,
            "feature_cols": ["a_phr", "resin_1_structure_xtb_gap"],
            "target_col": "tg_c",
            "extra": {"molecular_feature_workflow": {
                "schema_version": 3,
                "workflow_hash": "wf-1",
                "final_feature_names": ["resin_1_structure_xtb_gap"],
            }},
        },
    }
    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": []}}}}}

    publish_imported_entry(config, material_key="epoxy_resin", target_key="tg", entry=entry)

    assert isinstance(entry.get("contract"), dict), "发布后必须回写 contract"
    assert entry["contract"].get("feature_cols"), "回写的契约必须含精确 feature_cols"
    # 回写后 UI 才能推导出可编辑字段
    from UserPrediction import build_manual_input_fields, build_workflow_source_fields

    manual = build_manual_input_fields(entry["contract"], entry.get("registry_snapshot") or {})
    sources = build_workflow_source_fields(entry["contract"], entry.get("registry_snapshot") or {})
    assert [f["name"] for f in manual] == ["a_phr"]
    assert sources == []  # 该 artifact 未声明分子源列


# ---------------------------------------------------------------------------
# 分类型测试条件的整数编码（2026-09-22）
# ---------------------------------------------------------------------------

def test_categorical_test_conditions_are_encoded_to_integers():
    """分类型测试条件必须还原为训练时的整数编码，不能发文本。

    实测背景：内置默认值给的是原始表文本取值（``tg_c_test_method="DMA"``），
    但训练侧用 ``pd.Categorical(series).codes`` 编码（字典序），后端
    ``_validate_numeric_frame`` 要求契约内**全部**特征为有限数值。
    若直接把 "DMA" 交给 number_input 的预填逻辑，会静默回退成 0.0。
    """
    from UserPrediction import _categorical_code_for

    # 编码表与实测反推的 scaler 参数逐位吻合
    assert _categorical_code_for("tg_c_test_method", "DMA") == 0
    assert _categorical_code_for("tg_c_test_method", "DSC") == 1
    assert _categorical_code_for("tg_c_test_method", "other") == 6
    assert _categorical_code_for("tg_c_test_atmosphere", "N2") == 0
    assert _categorical_code_for("tg_c_test_atmosphere", "other") == 5
    # 大小写不敏感
    assert _categorical_code_for("tg_c_test_method", "dma") == 0
    # 不在受控词表内 → 不猜编码
    assert _categorical_code_for("tg_c_test_method", "rheology") is None
    assert _categorical_code_for("process_atmosphere", "N2") is None


def test_text_default_is_never_silently_coerced_to_zero():
    """文本默认值不得被静默当成 0.0 预填（那会造成静默的预测错误）。"""
    from UserPrediction import _coerce_default_for_kind

    value, warning = _coerce_default_for_kind("DMA", "number")
    assert value is None and warning, "文本默认值必须被拒绝并给出原因"

    value, warning = _coerce_default_for_kind(5.0, "number")
    assert value == 5.0 and warning is None

    value, warning = _coerce_default_for_kind("3", "integer")
    assert value == 3 and warning is None


def test_legacy_input_fields_are_numeric_even_for_method_columns():
    """legacy 合成字段不得把 *_test_method 当成文本控件。

    后端要求契约内全部特征为有限数值；文本控件产出的 "DMA" 会被判为
    「模型特征列 tg_c_test_method 必须是有限数值」。
    """
    from UserPrediction import build_manual_input_fields

    contract = {
        "schema_version": 1,
        "feature_cols": ["tg_c_test_method", "tg_c_test_atmosphere", "resin_component_count", "a_phr"],
        "workflow_final_feature_names": [],
    }
    fields = {f["name"]: f for f in build_manual_input_fields(contract, {})}
    assert fields["tg_c_test_method"]["kind"] == "number"
    assert fields["tg_c_test_atmosphere"]["kind"] == "number"
    assert fields["resin_component_count"]["kind"] == "integer"
    assert fields["a_phr"]["kind"] == "number"
    # 标记为已编码类别列，供 UI 提示取值含义
    assert fields["tg_c_test_method"]["categorical_encoded"] is True
    assert fields["a_phr"]["categorical_encoded"] is False


def test_ready_check_treats_nan_as_unfilled():
    """就绪检查必须把 NaN 视为未填（``str(nan) == 'nan'`` 不是空串）。"""
    import numpy as np
    from UserPrediction import _value_is_filled

    assert _value_is_filled(None) is False
    assert _value_is_filled(np.nan) is False
    assert _value_is_filled(float("nan")) is False
    assert _value_is_filled("") is False
    assert _value_is_filled("  ") is False
    assert _value_is_filled(0) is True
    assert _value_is_filled(0.0) is True
    assert _value_is_filled("CCO") is True
