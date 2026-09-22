"""输入端 5 分区重构与「已自动填充」明细条。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.4

背景
----
原设计按 required/optional 分组，与用户的心智模型不符：用户只关心
「我要配什么料、怎么固化、按什么标准测」，而不是「哪些字段必填」。

新分区（5 个）
--------------
① 配方（必填）        resin/curing_agent/formulation/initiator/... → 可编辑
② 固化制度            process_/cure_/post_cure/气氛/压力        → 可编辑（默认预填）
③ 高级测试条件        *_test_method/_test_standard/频率/速率     → 可编辑（折叠）
④ 自动推导（只读）    derived_feature_cols                      → 展示推导依据
⑤ 系统计算（只读）    molecular_workflow_feature_cols            → 展示 workflow 输出

关键约束：④⑤ 只读，绝不出现在可编辑分区（否则用户可能手填覆盖系统计算值）。
"""

from __future__ import annotations

from UserPrediction import (
    AUTOFILL_SECTION_GROUPS,
    build_autofilled_summary,
    build_input_partition_plan,
    classify_manual_field_group,
)


def _contract() -> dict:
    manual = [
        "resin_total_phr",
        "curing_agent_total_phr",
        "formulation_r_value",
        "process_max_temperature_c",
        "process_total_time_h",
        "cure_stage_count",
        "storage_modulus_25c_gpa_test_method",
        "storage_modulus_25c_gpa_test_standard",
        "storage_modulus_25c_gpa_frequency_hz",
    ]
    derived = ["cp_eew", "cp_ahew"]
    molecular = ["resin_1_structure_xtb_gap", "curing_agent_1_structure_maccs_1"]
    return {
        "schema_version": 2,
        "feature_cols": manual + derived + molecular,
        "canonical_feature_cols": manual + derived + molecular,
        "manual_input_feature_cols": manual,
        "derived_feature_cols": derived,
        "molecular_workflow_feature_cols": molecular,
        "workflow_source_fields": [{"column": "resin_smiles", "roles": ["resin"]}],
        "feature_definitions": [
            {"name": name, "source_type": "manual_input", "status": "approved"}
            for name in manual
        ],
    }


# ---------------------------------------------------------------------------
# 分区计划
# ---------------------------------------------------------------------------

def test_partition_plan_has_five_groups_in_order():
    """分区计划必须包含 5 组且顺序固定（配方 → 固化制度 → 测试条件 → 推导 → 计算）。"""
    plan = build_input_partition_plan(_contract())
    groups = [section["group"] for section in plan]

    assert groups == [
        "recipe",
        "cure_schedule",
        "test_conditions",
        "derived",
        "computed",
    ]


def test_recipe_group_is_editable_and_holds_formulation_fields():
    """① 配方区可编辑，含配方计量字段。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}

    assert groups["recipe"]["kind"] == "input"
    assert "resin_total_phr" in groups["recipe"]["features"]
    assert "curing_agent_total_phr" in groups["recipe"]["features"]


def test_cure_schedule_group_holds_process_fields():
    """② 固化制度区含 process_/cure_ 字段。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}

    assert groups["cure_schedule"]["kind"] == "input"
    assert "process_max_temperature_c" in groups["cure_schedule"]["features"]
    assert "process_total_time_h" in groups["cure_schedule"]["features"]
    assert "cure_stage_count" in groups["cure_schedule"]["features"]


def test_test_conditions_group_holds_test_fields():
    """③ 高级测试条件区含测试方法/标准/频率字段。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}

    assert groups["test_conditions"]["kind"] == "input"
    assert "storage_modulus_25c_gpa_test_method" in groups["test_conditions"]["features"]
    assert "storage_modulus_25c_gpa_test_standard" in groups["test_conditions"]["features"]
    assert "storage_modulus_25c_gpa_frequency_hz" in groups["test_conditions"]["features"]


def test_derived_group_is_display_only():
    """④ 自动推导区必须只读（kind=display），且不含任何可编辑字段。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}

    assert groups["derived"]["kind"] == "display"
    assert groups["derived"]["features"] == ["cp_eew", "cp_ahew"]


def test_computed_group_is_display_only():
    """⑤ 系统计算区必须只读，且列出 workflow 输出特征。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}

    assert groups["computed"]["kind"] == "display"
    assert "resin_1_structure_xtb_gap" in groups["computed"]["features"]


def test_display_groups_never_appear_in_editable_groups():
    """系统计算/推导特征绝不出现在可编辑分区（防止手填覆盖）。"""
    groups = {s["group"]: s for s in build_input_partition_plan(_contract())}
    editable = (
        groups["recipe"]["features"]
        + groups["cure_schedule"]["features"]
        + groups["test_conditions"]["features"]
    )

    assert "cp_eew" not in editable
    assert "resin_1_structure_xtb_gap" not in editable


def test_every_manual_feature_lands_in_exactly_one_editable_group():
    """每个 manual 字段必须恰好落在一个可编辑分区（不重不漏）。"""
    plan = build_input_partition_plan(_contract())
    groups = {s["group"]: s for s in plan}
    collected: list[str] = []
    for group in ("recipe", "cure_schedule", "test_conditions"):
        collected.extend(groups[group]["features"])

    assert sorted(collected) == sorted(_contract()["manual_input_feature_cols"])
    assert len(collected) == len(set(collected)), "字段不得重复出现在多个分区"


def test_empty_contract_returns_all_five_groups():
    """空契约也必须返回 5 个分组（UI 不崩溃）。"""
    plan = build_input_partition_plan({})

    assert [s["group"] for s in plan] == [
        "recipe",
        "cure_schedule",
        "test_conditions",
        "derived",
        "computed",
    ]
    for section in plan:
        assert section["features"] == []


def test_screening_fixed_input_cols_becomes_readonly_block():
    """screening_fixed_input_cols 作为只读固定条件块保留。"""
    contract = _contract()
    contract["screening_fixed_input_cols"] = ["process_atmosphere"]
    plan = build_input_partition_plan(contract)
    fixed = [s for s in plan if s["group"] == "fixed_inputs"]

    assert fixed and fixed[0]["kind"] == "display"
    assert fixed[0]["features"] == ["process_atmosphere"]


# ---------------------------------------------------------------------------
# 字段分类
# ---------------------------------------------------------------------------

def test_classify_manual_field_group():
    """分类函数：配方 / 固化制度 / 测试条件。"""
    assert classify_manual_field_group("resin_total_phr") == "recipe"
    assert classify_manual_field_group("curing_agent_total_phr") == "recipe"
    assert classify_manual_field_group("formulation_r_value") == "recipe"
    assert classify_manual_field_group("initiator_present") == "recipe"
    assert classify_manual_field_group("process_max_temperature_c") == "cure_schedule"
    assert classify_manual_field_group("process_total_time_h") == "cure_schedule"
    assert classify_manual_field_group("cure_stage_count") == "cure_schedule"
    assert classify_manual_field_group("post_cure_temperature_c") == "cure_schedule"
    assert classify_manual_field_group("storage_modulus_25c_gpa_test_method") == "test_conditions"
    assert classify_manual_field_group("tg_c_test_standard") == "test_conditions"
    assert classify_manual_field_group("tg_c_frequency_hz") == "test_conditions"


def test_classify_unknown_field_defaults_to_test_conditions():
    """无法识别的字段归入测试条件（保守：不误入配方区）。"""
    assert classify_manual_field_group("some_unknown_manual_field") == "test_conditions"


# ---------------------------------------------------------------------------
# 已自动填充明细条
# ---------------------------------------------------------------------------

def test_autofill_section_groups_constant():
    """明细条分组常量与分区计划一致。"""
    assert AUTOFILL_SECTION_GROUPS == ("recipe", "cure_schedule", "test_conditions")


def test_autofilled_summary_lists_source_for_each_field():
    """代填明细必须逐字段给出取值与依据（默认值来源 / 推导来源）。"""
    entries = [
        {
            "feature": "storage_modulus_25c_gpa_test_method",
            "value": "DMA",
            "origin": "default",
            "detail": "全表统计：share=0.856，n=2348（ml_performance_all.csv）",
        },
        {
            "feature": "formulation_r_value",
            "value": 0.905,
            "origin": "derived",
            "detail": "r = (固化剂phr/AHEW)/(树脂phr/EEW)",
        },
    ]
    summary = build_autofilled_summary(entries)

    assert summary["count"] == 2
    assert summary["title"] == "已自动填充 2 项"
    assert len(summary["rows"]) == 2
    for row in summary["rows"]:
        assert row["feature"] and row["detail"]
        assert row["origin"] in {"default", "derived"}


def test_autofilled_summary_empty():
    """无代填项时给出明确的空状态文案。"""
    summary = build_autofilled_summary([])

    assert summary["count"] == 0
    assert summary["rows"] == []
    assert "未" in summary["title"] or "0" in summary["title"]


def test_autofilled_summary_skips_empty_values():
    """空值不得计入代填（避免虚报项数）。"""
    entries = [
        {"feature": "a", "value": None, "origin": "default", "detail": "x"},
        {"feature": "b", "value": "", "origin": "default", "detail": "y"},
        {"feature": "c", "value": 0, "origin": "default", "detail": "z"},
    ]
    summary = build_autofilled_summary(entries)

    assert summary["count"] == 1
    assert summary["rows"][0]["feature"] == "c"


# ---------------------------------------------------------------------------
# legacy(schema-1) 契约：无分区字段时仍需渲染输入框
# ---------------------------------------------------------------------------

def test_legacy_contract_derives_manual_fields_from_workflow_split():
    """legacy 契约（无分区字段）必须从 feature_cols − workflow 产出 推导人工字段。

    实测背景：真实训练平台导出的 artifact 是 schema-1，contract 没有
    manual_input_feature_cols。若不推导，UI 一个输入框都不渲染，用户无法预测。
    """
    contract = {
        "schema_version": 1,
        "feature_cols": ["a_phr", "process_max_temperature_c", "resin_1_structure_xtb_gap"],
        "workflow_final_feature_names": ["resin_1_structure_xtb_gap"],
    }
    plan = build_input_partition_plan(contract)
    groups = {s["group"]: s for s in plan}
    editable = (
        groups["recipe"]["features"]
        + groups["cure_schedule"]["features"]
        + groups["test_conditions"]["features"]
    )

    assert sorted(editable) == sorted(["a_phr", "process_max_temperature_c"])
    # workflow 产出不得进入可编辑分区
    assert "resin_1_structure_xtb_gap" not in editable


def test_legacy_contract_without_workflow_info_asks_for_all_fields():
    """无 workflow 信息时不得猜哪些是系统计算 —— 全部交给用户输入（宁可多问）。"""
    contract = {
        "schema_version": 1,
        "feature_cols": ["a_phr", "b_test_method"],
    }
    plan = build_input_partition_plan(contract)
    groups = {s["group"]: s for s in plan}
    editable = (
        groups["recipe"]["features"]
        + groups["cure_schedule"]["features"]
        + groups["test_conditions"]["features"]
    )

    assert sorted(editable) == ["a_phr", "b_test_method"]


def test_legacy_real_artifact_yields_36_editable_fields():
    """回归：真实 storage_modulus_25c_gpa artifact 应产出 36 个可编辑字段。

    36 = 284（模型消费列） − 253（workflow 产出）+ 5（被删除的分子特征）… 实测为 36。
    """
    import joblib

    from core.prediction_portal import build_prediction_contract

    path = (
        "prediction_portal/managed_models/epoxy_resin/storage_modulus_25c_gpa/"
        "20260831_195237_r2_0_8_MAE_0_5.joblib"
    )
    try:
        artifact = joblib.load(path)
    except Exception:
        pytest.skip("真实 artifact 不在工作区")

    extra = artifact.get("extra") or {}
    workflow = extra.get("molecular_feature_workflow") or {}
    contract = build_prediction_contract(
        artifact=artifact,
        feature_cols=artifact.get("feature_cols") or [],
        target_col="storage_modulus_25c_gpa",
        workflow=workflow,
    )
    plan = build_input_partition_plan(contract)
    groups = {s["group"]: s for s in plan}

    assert len(groups["recipe"]["features"]) == 26
    assert len(groups["cure_schedule"]["features"]) == 7
    assert len(groups["test_conditions"]["features"]) == 3
    total = sum(len(groups[g]["features"]) for g in ("recipe", "cure_schedule", "test_conditions"))
    assert total == 36
