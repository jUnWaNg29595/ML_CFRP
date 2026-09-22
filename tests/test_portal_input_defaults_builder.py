"""默认值生成脚本：离线统计产出可审计的 portal_input_defaults.json。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.2

核心约束：
1. 每条默认值必须带 share（占比）+ support（样本数）+ source_table，可审计；
2. 占比 < 0.30 不生成（证据不足）；
3. 绝不生成 derived_workflow / molecular_workflow 字段的默认值；
4. 哨兵值 'other' / 'unknown' 不得作为默认值（无信息量）；
5. curing_mechanism 因 unknown 占 84% 必须被排除。
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_portal_input_defaults.py"


def _write_dataset(root: Path, *, performance: pd.DataFrame, standards: pd.DataFrame | None = None,
                   qspr: pd.DataFrame | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    performance.to_csv(root / "ml_performance_all.csv", index=False)
    (standards if standards is not None else pd.DataFrame(
        columns=["performance_row_id", "standard_canonical"]
    )).to_csv(root / "ml_performance_standards.csv", index=False)
    (qspr if qspr is not None else pd.DataFrame()).to_csv(
        root / "ml_qspr_selected.csv", index=False
    )
    return root


def _run_script(dataset: Path, output: Path, generated_at: str = "2026-09-22") -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset", str(dataset),
            "--output", str(output),
            "--generated-at", generated_at,
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"脚本执行失败：{result.stderr}"
    return json.loads(output.read_text(encoding="utf-8"))


def test_builder_records_share_support_and_source_table(tmp_path):
    """每条默认值必须带 share / support / source_table，以便审计。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 90 + ["other"] * 10,
        "test_atmosphere": ["N2"] * 90 + ["air"] * 10,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    field = payload["defaults"]["storage_modulus_25c_gpa"]["fields"][
        "storage_modulus_25c_gpa_test_method"
    ]
    assert field["value"] == "DMA"
    assert field["share"] == pytest.approx(0.9)
    assert field["support"] == 90
    assert field["source_table"] == "ml_performance_all.csv"
    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-09-22"


def test_builder_omits_fields_below_share_threshold(tmp_path):
    """占比 < 0.30 的字段不得生成默认值（证据不足）。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        # 最高占比仅 0.25 → 低于 0.30 阈值
        "test_method": ["DMA"] * 25 + ["flexural"] * 25 + ["tensile"] * 25 + ["other"] * 25,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    # 所有取值占比均低于 0.30 → 整个目标列不应产生任何测试条件默认值
    fields = (payload["defaults"].get("storage_modulus_25c_gpa") or {}).get("fields", {})
    assert "storage_modulus_25c_gpa_test_method" not in fields
    assert fields == {}


def test_builder_never_uses_sentinel_values(tmp_path):
    """哨兵值 'other' / 'unknown' 无信息量，不得作为默认值。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 40 + ["other"] * 60,
        "test_atmosphere": ["other"] * 100,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    fields = payload["defaults"]["storage_modulus_25c_gpa"]["fields"]
    assert fields["storage_modulus_25c_gpa_test_method"]["value"] == "DMA"
    # test_atmosphere 全为 'other' → 不得生成
    assert "storage_modulus_25c_gpa_test_atmosphere" not in fields


def test_builder_never_emits_derived_or_workflow_defaults(tmp_path):
    """绝不生成 derived_workflow / molecular_workflow 字段的默认值。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 100,
        "storage_modulus_25c_gpa": [1.0] * 100,
        # 工艺派生字段：必须由 workflow 从 cure_schedule 计算，不得给默认值
        "process_max_temperature_c": [120.0] * 100,
        "process_total_time_h": [4.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    blob = json.dumps(payload, ensure_ascii=False)
    assert "process_max_temperature_c" not in blob
    assert "process_total_time_h" not in blob


def test_builder_skips_curing_mechanism(tmp_path):
    """curing_mechanism 因 unknown 占 84% 必须被排除（无效口径）。"""
    qspr = pd.DataFrame({
        "curing_mechanism": ["unknown"] * 84 + ["amine_epoxy"] * 16,
        "curing_type_standard": ["external_hardener"] * 90 + ["self_cure"] * 10,
    })
    performance = pd.DataFrame({
        "performance_row_id": [0],
        "test_method": ["DMA"],
        "storage_modulus_25c_gpa": [1.0],
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance, qspr=qspr)
    payload = _run_script(dataset, tmp_path / "out.json")

    assert "curing_mechanism" not in payload["recipe_defaults"]
    assert payload["recipe_defaults"]["curing_type_standard"]["value"] == "external_hardener"


def test_builder_records_median_for_continuous_conditions(tmp_path):
    """连续量（频率/升温速率/加载速率）取中位数。"""
    performance = pd.DataFrame({
        "performance_row_id": range(10),
        "frequency_hz": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        "storage_modulus_25c_gpa": [1.0] * 10,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    field = payload["defaults"]["storage_modulus_25c_gpa"]["fields"][
        "storage_modulus_25c_gpa_frequency_hz"
    ]
    assert field["value"] == 1.0
    assert field["support"] == 10


def test_builder_uses_standards_table_when_coverage_sufficient(tmp_path):
    """标准号经 ml_performance_standards 按 performance_row_id 关联后取众数。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 100,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    standards = pd.DataFrame({
        "performance_row_id": range(100),
        "standard_canonical": ["ASTM D5026"] * 80 + ["ASTM D7028"] * 20,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance, standards=standards)
    payload = _run_script(dataset, tmp_path / "out.json")

    field = payload["defaults"]["storage_modulus_25c_gpa"]["fields"][
        "storage_modulus_25c_gpa_test_standard"
    ]
    assert field["value"] == "ASTM D5026"
    assert field["support"] == 80
    assert field["source_table"] == "ml_performance_standards.csv"


def test_builder_is_deterministic(tmp_path):
    """相同输入两次运行必须产出完全一致的 JSON（可复现）。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 90 + ["other"] * 10,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    first = _run_script(dataset, tmp_path / "a.json")
    second = _run_script(dataset, tmp_path / "b.json")
    assert first == second


def test_builder_survives_missing_optional_tables(tmp_path):
    """缺少可选表（standards / qspr）时降级不报错。"""
    root = tmp_path / "ds"
    root.mkdir(parents=True)
    pd.DataFrame({
        "performance_row_id": range(10),
        "test_method": ["DMA"] * 10,
        "storage_modulus_25c_gpa": [1.0] * 10,
    }).to_csv(root / "ml_performance_all.csv", index=False)
    payload = _run_script(root, tmp_path / "out.json")
    assert payload["defaults"]["storage_modulus_25c_gpa"]["fields"]


def test_share_denominator_is_target_sample_size_not_condition_nonnull(tmp_path):
    """share 的分母必须是该目标列的样本数，而非该条件自身的非空数。

    否则只覆盖部分样本的条件会得到 share=1.0，误导读者以为"全部样本都有此条件"。
    实测背景：storage_modulus_25c_gpa 有 2744 样本，但 frequency_hz 仅 2267 非空，
    share 应为 0.826 而非 1.0。
    """
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 100,
        # 仅前 50 个样本有频率记录
        "frequency_hz": [1.0] * 50 + [None] * 50,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    field = payload["defaults"]["storage_modulus_25c_gpa"]["fields"][
        "storage_modulus_25c_gpa_frequency_hz"
    ]
    assert field["support"] == 50
    assert field["share"] == pytest.approx(0.5)


def test_condition_columns_are_never_treated_as_targets(tmp_path):
    """测试条件列（如 strain_rate_s_1 / test_temperature_c）本身不是性能指标，
    不得作为目标列出现在 defaults 中。"""
    performance = pd.DataFrame({
        "performance_row_id": range(100),
        "test_method": ["DMA"] * 100,
        "strain_rate_s_1": [0.01] * 100,
        "test_temperature_c": [25.0] * 100,
        "storage_modulus_25c_gpa": [1.0] * 100,
    })
    dataset = _write_dataset(tmp_path / "ds", performance=performance)
    payload = _run_script(dataset, tmp_path / "out.json")

    assert "strain_rate_s_1" not in payload["defaults"]
    assert "test_temperature_c" not in payload["defaults"]
    assert "test_method" not in payload["defaults"]
    assert "storage_modulus_25c_gpa" in payload["defaults"]
