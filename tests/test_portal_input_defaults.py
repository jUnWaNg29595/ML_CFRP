"""默认值读取模块：只对 manual_input 分区字段返回默认值。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.2

核心硬约束
----------
1. **只对 ``manual_input`` 分区字段返回默认值**；
   derived_workflow / molecular_workflow 字段必须返回 None
   —— 它们必须由 workflow 真实计算，给默认值等同于伪造数据。
2. JSON 缺失/损坏时降级返回空默认值，**不得抛异常**（门户不能因缺文件而崩溃）。
3. 默认值必须可追溯到 share / support / source_table（审计要求）。
"""

import json

import pytest

from core.portal_input_defaults import (
    default_for_feature,
    defaults_for_target,
    load_portal_input_defaults,
    recipe_defaults,
)

SAMPLE = {
    "schema_version": 1,
    "generated_at": "2026-09-22",
    "source_dataset": "ml_dataset",
    "min_share": 0.3,
    "defaults": {
        "storage_modulus_25c_gpa": {
            "target_col": "storage_modulus_25c_gpa",
            "sample_size": 2744,
            "fields": {
                "storage_modulus_25c_gpa_test_method": {
                    "value": "DMA",
                    "share": 0.8557,
                    "support": 2348,
                    "field": "test_method",
                    "source_table": "ml_performance_all.csv",
                },
                "storage_modulus_25c_gpa_frequency_hz": {
                    "value": 1.0,
                    "share": 0.8262,
                    "support": 2267,
                    "field": "frequency_hz",
                    "source_table": "ml_performance_all.csv",
                },
            },
        }
    },
    "recipe_defaults": {
        "curing_type_standard": {
            "value": "external_hardener",
            "share": 0.8663,
            "support": 8816,
            "source_table": "ml_qspr_selected.csv",
        }
    },
}


@pytest.fixture()
def portal_root(tmp_path):
    path = tmp_path / "prediction_portal"
    path.mkdir(parents=True)
    (path / "portal_input_defaults.json").write_text(
        json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_only_manual_input_fields_get_defaults(portal_root):
    """manual_input 分区字段返回默认值。"""
    found = default_for_feature(
        "storage_modulus_25c_gpa_test_method",
        partition="manual_input",
        target_col="storage_modulus_25c_gpa",
        root=str(portal_root),
    )

    assert found is not None
    assert found["value"] == "DMA"
    assert found["share"] == pytest.approx(0.8557)
    assert found["support"] == 2348
    assert found["source_table"] == "ml_performance_all.csv"


def test_derived_and_workflow_partitions_never_get_defaults(portal_root):
    """derived_workflow / molecular_workflow 分区一律返回 None（不得伪造）。"""
    for partition in ("derived_workflow", "molecular_workflow"):
        assert default_for_feature(
            "storage_modulus_25c_gpa_test_method",
            partition=partition,
            target_col="storage_modulus_25c_gpa",
            root=str(portal_root),
        ) is None

    # 工艺派生字段本身也不在 JSON 中，即使误标 manual_input 也拿不到
    assert default_for_feature(
        "process_max_temperature_c",
        partition="manual_input",
        target_col="storage_modulus_25c_gpa",
        root=str(portal_root),
    ) is None


def test_unknown_partition_is_treated_as_not_manual(portal_root):
    """未声明/未知分区不得返回默认值（保守失败）。"""
    for partition in ("", "unknown", None, "target", "metadata"):
        assert default_for_feature(
            "storage_modulus_25c_gpa_test_method",
            partition=partition,
            target_col="storage_modulus_25c_gpa",
            root=str(portal_root),
        ) is None


def test_missing_json_degrades_gracefully(tmp_path):
    """JSON 缺失时返回空默认值，不抛异常。"""
    empty_root = tmp_path / "no_such_root"
    empty_root.mkdir()

    assert load_portal_input_defaults(root=str(empty_root)) == {}
    assert defaults_for_target("storage_modulus_25c_gpa", root=str(empty_root)) == {}
    assert recipe_defaults(root=str(empty_root)) == {}
    assert default_for_feature(
        "storage_modulus_25c_gpa_test_method",
        partition="manual_input",
        target_col="storage_modulus_25c_gpa",
        root=str(empty_root),
    ) is None


def test_corrupt_json_degrades_gracefully(tmp_path):
    """JSON 损坏时降级，不抛异常。"""
    path = tmp_path / "prediction_portal"
    path.mkdir(parents=True)
    (path / "portal_input_defaults.json").write_text("{ not valid json", encoding="utf-8")

    assert load_portal_input_defaults(root=str(tmp_path)) == {}
    assert recipe_defaults(root=str(tmp_path)) == {}


def test_defaults_lookup_by_target_col(portal_root):
    """按目标列查得该列的测试条件默认值集合。"""
    fields = defaults_for_target("storage_modulus_25c_gpa", root=str(portal_root))

    assert set(fields) == {
        "storage_modulus_25c_gpa_test_method",
        "storage_modulus_25c_gpa_frequency_hz",
    }
    assert fields["storage_modulus_25c_gpa_frequency_hz"]["value"] == 1.0
    assert defaults_for_target("no_such_target", root=str(portal_root)) == {}


def test_recipe_defaults_lookup(portal_root):
    """配方默认值可查得，且带审计字段。"""
    found = recipe_defaults(root=str(portal_root))

    assert found["curing_type_standard"]["value"] == "external_hardener"
    assert found["curing_type_standard"]["support"] == 8816


def test_lookup_without_target_col_still_finds_recipe_fields(portal_root):
    """未给 target_col 时仍能查到 recipe_defaults 中的字段。"""
    found = default_for_feature(
        "curing_type_standard",
        partition="manual_input",
        target_col="",
        root=str(portal_root),
    )

    assert found is not None
    assert found["value"] == "external_hardener"


def test_real_repository_defaults_are_loadable():
    """真实仓库产物必须可加载，且包含已实测的关键默认值。"""
    payload = load_portal_input_defaults()
    assert payload.get("schema_version") == 1
    assert payload.get("defaults")

    fields = defaults_for_target("storage_modulus_25c_gpa")
    assert fields["storage_modulus_25c_gpa_test_method"]["value"] == "DMA"
