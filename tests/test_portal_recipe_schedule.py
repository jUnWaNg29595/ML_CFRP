"""配方库固化制度：解析校验与显式数字温度约束。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.4

已实测的解析器缺陷
------------------
``core/process_features.py::_schedule_pairs`` 的正则要求温度是数字，
**非数字温度会被静默丢弃且不报错**：

    '室温/24 h + 80 °C/2 h'   → [(80.0, 2.0)]              ← 室温阶段丢失，26h 误算为 2h
    '25 °C/24 h + 80 °C/2 h'  → [(25.0,24.0),(80.0,2.0)]    ← 正确

因此配方库必须：
1. 一律使用显式数字温度（禁止「室温」「RT」「常温」）；
2. 加载时断言解析出的阶段数与声明一致，防止未来回归。
"""

import pytest

from UserPrediction import (
    PORTAL_PRESET_RECIPES,
    validate_recipe_schedule,
)


def test_validate_recipe_schedule_returns_pairs():
    """正常固化制度解析出 (温度, 时长) 序列。"""
    pairs = validate_recipe_schedule("80 °C/2 h + 150 °C/3 h", declared_stages=2)
    assert pairs == [(80.0, 2.0), (150.0, 3.0)]


def test_validate_recipe_schedule_raises_on_stage_mismatch():
    """声明 3 阶段但只解析出 2 阶段 → 必须抛错，不静默。"""
    with pytest.raises(ValueError) as excinfo:
        validate_recipe_schedule("80 °C/2 h + 150 °C/3 h", declared_stages=3)
    message = str(excinfo.value)
    assert "3" in message and "2" in message


def test_validate_recipe_schedule_raises_on_unparseable():
    """完全无法解析 → 抛错。"""
    with pytest.raises(ValueError):
        validate_recipe_schedule("请参考厂家说明", declared_stages=1)


def test_room_temperature_text_is_silently_dropped_by_parser():
    """回归：记录已知缺陷行为 —— 非数字温度被静默丢弃。

    本测试**锁定缺陷现状**，确保将来若有人修好解析器，会立刻被提醒更新配方库约束。
    """
    pairs = validate_recipe_schedule("室温/24 h + 80 °C/2 h", declared_stages=1)
    assert pairs == [(80.0, 2.0)]


def test_numeric_room_temperature_parses_two_stages():
    """'25 °C/24 h + 80 °C/2 h' 必须解析出 2 阶段（正确写法）。"""
    pairs = validate_recipe_schedule("25 °C/24 h + 80 °C/2 h", declared_stages=2)
    assert pairs == [(25.0, 24.0), (80.0, 2.0)]


def test_all_preset_recipes_parse_declared_stage_count():
    """遍历全部 PORTAL_PRESET_RECIPES，断言 cure_schedule 解析阶段数 == 声明阶段数。"""
    assert PORTAL_PRESET_RECIPES, "配方库不得为空"
    for recipe in PORTAL_PRESET_RECIPES:
        name = recipe.get("name")
        schedule = recipe.get("cure_schedule")
        declared = recipe.get("cure_stages")
        assert schedule, f"{name} 缺少 cure_schedule"
        assert isinstance(declared, int) and declared >= 1, f"{name} 缺少合法的 cure_stages"
        pairs = validate_recipe_schedule(schedule, declared_stages=declared)
        assert len(pairs) == declared, f"{name}: {schedule!r} 解析出 {len(pairs)} 阶段"


def test_preset_recipes_use_numeric_temperatures_only():
    """配方库 cure_schedule 禁止出现 '室温'/'RT'/'常温' 等非数字表述。"""
    forbidden = ("室温", "常温", "rt", "room temp", "环境温度")
    for recipe in PORTAL_PRESET_RECIPES:
        schedule = str(recipe.get("cure_schedule") or "")
        lowered = schedule.lower()
        for token in forbidden:
            assert token not in lowered, (
                f"{recipe.get('name')} 的 cure_schedule 含非数字温度 {token!r}：{schedule!r}"
            )


def test_preset_recipes_have_positive_temperature_and_time():
    """每个阶段的温度与时长必须为正数（防止 0/负数混入）。"""
    for recipe in PORTAL_PRESET_RECIPES:
        pairs = validate_recipe_schedule(
            recipe["cure_schedule"], declared_stages=recipe["cure_stages"]
        )
        for temperature, duration in pairs:
            assert temperature > 0, f"{recipe['name']}: 温度非正"
            assert duration > 0, f"{recipe['name']}: 时长非正"


def test_preset_recipes_keep_phr_and_note_fields():
    """扩展后仍保留 phr/note 等既有字段（不破坏现有 UI 契约）。"""
    for recipe in PORTAL_PRESET_RECIPES:
        assert recipe.get("name")
        assert recipe.get("resin_key")
        assert recipe.get("hardener_key")
        assert isinstance(recipe.get("phr"), (int, float))
        assert recipe.get("note")


def test_dgebf_ipda_recipe_uses_numeric_room_temperature():
    """DGEBF/IPDA 条目原写「室温/24 h」，必须已改写为 25 °C。"""
    match = [r for r in PORTAL_PRESET_RECIPES if "IPDA" in str(r.get("name", ""))]
    assert match, "配方库应保留 DGEBF/IPDA 条目"
    schedule = match[0]["cure_schedule"]
    assert "25 °C/24 h" in schedule
    assert match[0]["cure_stages"] == 2
