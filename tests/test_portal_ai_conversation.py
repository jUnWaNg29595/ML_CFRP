"""AI 输入助手多轮对话。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.5

动机
----
用户不会一次把配方说全。典型交互：

    用户：E-51/DDS，100:33，180度固化4小时
    AI  ：已提取 树脂/固化剂/配比/温度/时间 5 项
    用户：温度改成 200 度
    AI  ：已更新固化温度 180 → 200 ℃

关键点：第二轮必须把**已确认字段**作为上下文传给 AI，否则 AI 不知道
"温度"指哪个字段、也不知道其他字段已确定。

不可放松的约束
--------------
1. AI 只能提取/整理用户提供的信息，**不得生成** EEW/AHEW/PHR/分子特征/工艺参数；
2. AI 提取结果仍须用户确认（can_submit_ai_prediction 门禁保留）；
3. 缓存不得落 API key。
"""

from __future__ import annotations

from UserPrediction import (
    AI_CONVERSATION_MAX_TURNS,
    append_conversation_turn,
    build_ai_conversation_context,
    build_ai_conversation_messages,
    can_submit_ai_prediction,
    merge_ai_conversation_state,
)


def _state(fields: dict) -> dict:
    return {
        "fields": {
            name: {"value": value, "state": "confirmed"}
            for name, value in fields.items()
        },
        "rejected_fields": set(),
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# 上下文构建
# ---------------------------------------------------------------------------

def test_conversation_context_passes_confirmed_fields():
    """每轮必须把当前已确认字段作为上下文传给 AI。"""
    state = _state({"resin_smiles": "CC(C)(...)DGEBA", "curing_temperature_c": 180.0})

    context = build_ai_conversation_context(state)

    assert context["confirmed_fields"]["curing_temperature_c"] == 180.0
    assert "resin_smiles" in context["confirmed_fields"]


def test_conversation_context_excludes_rejected_fields():
    """被用户拒绝的字段不得作为上下文（避免 AI 又提它）。"""
    state = _state({"a": 1, "b": 2})
    state["rejected_fields"] = {"b"}

    context = build_ai_conversation_context(state)

    assert "b" not in context["confirmed_fields"]
    assert context["rejected_fields"] == ["b"]


def test_conversation_context_marks_field_semantics():
    """上下文需说明字段含义，AI 才能把「温度」映射到正确字段。"""
    state = _state({"curing_temperature_c": 180.0})

    context = build_ai_conversation_context(
        state, field_labels={"curing_temperature_c": "固化温度"}
    )

    assert context["field_labels"]["curing_temperature_c"] == "固化温度"


# ---------------------------------------------------------------------------
# 消息构建
# ---------------------------------------------------------------------------

def test_conversation_messages_include_history_and_current_input():
    """消息序列含历史轮次 + 当前输入。"""
    turns = [
        {"role": "user", "text": "E-51/DDS，100:33，180度固化4小时"},
        {"role": "assistant", "text": "已提取 5 项"},
    ]
    messages = build_ai_conversation_messages(turns, "温度改成 200 度")

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["text"] == "温度改成 200 度"


def test_conversation_history_is_bounded():
    """历史轮次必须截断，避免上下文无限增长。"""
    turns = [{"role": "user", "text": f"t{i}"} for i in range(50)]
    messages = build_ai_conversation_messages(turns, "new")

    assert len(messages) <= AI_CONVERSATION_MAX_TURNS + 1


def test_append_conversation_turn():
    """追加轮次并保持顺序。"""
    turns = append_conversation_turn([], role="user", text="a")
    turns = append_conversation_turn(turns, role="assistant", text="b")

    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "a"


# ---------------------------------------------------------------------------
# 状态合并（增量修正）
# ---------------------------------------------------------------------------

def test_followup_modification_updates_only_targeted_field():
    """'温度改成 200 度' 只更新固化温度，不动其他字段。"""
    previous = _state({"curing_temperature_c": 180.0, "resin_total_phr": 100.0})
    new_response = {"fields": {"curing_temperature_c": {"value": 200.0, "state": "suggested"}}}

    merged = merge_ai_conversation_state(previous, new_response)

    assert merged["fields"]["curing_temperature_c"]["value"] == 200.0
    # 其他字段保持原值
    assert merged["fields"]["resin_total_phr"]["value"] == 100.0


def test_merge_preserves_unmentioned_fields():
    """新一轮未提及的已确认字段必须保留（不得被清空）。"""
    previous = _state({"a": 1, "b": 2, "c": 3})
    merged = merge_ai_conversation_state(previous, {"fields": {"b": {"value": 20}}})

    assert set(merged["fields"]) == {"a", "b", "c"}
    assert merged["fields"]["a"]["value"] == 1
    assert merged["fields"]["c"]["value"] == 3


def test_merge_keeps_rejected_fields_rejected():
    """已拒绝字段在新一轮不得被静默恢复。"""
    previous = _state({"a": 1})
    previous["rejected_fields"] = {"a"}
    merged = merge_ai_conversation_state(previous, {"fields": {"a": {"value": 99}}})

    assert "a" in set(merged.get("rejected_fields") or set())


def test_merge_does_not_mark_new_fields_confirmed():
    """新提取的字段仍为 suggested，必须用户确认（门禁不得放松）。"""
    previous = _state({"a": 1})
    merged = merge_ai_conversation_state(previous, {"fields": {"b": {"value": 2}}})

    assert merged["fields"]["b"]["state"] != "confirmed"
    # 有未确认字段 → 不允许直接提交
    assert can_submit_ai_prediction(merged) is False


def test_merge_collects_warnings():
    """AI 返回的警告需累积，不丢失。"""
    previous = _state({"a": 1})
    previous["warnings"] = ["旧警告"]
    merged = merge_ai_conversation_state(previous, {"warnings": ["新警告"]})

    assert "旧警告" in merged["warnings"]
    assert "新警告" in merged["warnings"]


# ---------------------------------------------------------------------------
# 约束保持
# ---------------------------------------------------------------------------

def test_ai_field_definitions_forbid_generation():
    """字段描述必须声明 allow_ai_generation=False（AI 不得生成计算量）。"""
    from UserPrediction import build_ai_field_descriptions

    descriptions = build_ai_field_descriptions(
        [{"name": "cp_eew", "label": "EEW", "kind": "number", "required": True}]
    )

    assert descriptions
    for item in descriptions:
        assert item["allow_ai_generation"] is False


def test_ai_cannot_generate_derived_quantities():
    """EEW/AHEW/PHR 等计算量不得被标为可生成。"""
    from UserPrediction import build_ai_field_descriptions

    forbidden = [
        {"name": "formulation_resin_total_eew_g_eq", "label": "EEW", "kind": "number"},
        {"name": "formulation_hardener_total_ahew_g_eq", "label": "AHEW", "kind": "number"},
        {"name": "formulation_r_value", "label": "r", "kind": "number"},
    ]
    for item in build_ai_field_descriptions(forbidden):
        assert item["allow_ai_generation"] is False
