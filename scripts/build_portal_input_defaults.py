"""离线统计生成门户输入默认值（portal_input_defaults.json）。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.2

用途
----
门户手工输入需要用户填写测试标准/反应条件/测试条件等"对应关系很难找"的字段。
本脚本从真实全表离线统计出这些字段的主流取值，作为**可审计的**默认值预填，
让用户只需输入配方即可预测。

硬约束（与 spec 一致）
--------------------
1. 每条默认值必须带 ``share``（占比）+ ``support``（样本数）+ ``source_table``；
2. 占比 < ``--min-share``（默认 0.30）不生成 —— 证据不足；
3. **绝不**为 derived_workflow / molecular_workflow 字段生成默认值
   （它们必须由 workflow 真实计算，伪造会造成静默错误）；
4. 哨兵值 ``other`` / ``unknown`` / ``nan`` 无信息量，不得作为默认值；
5. ``curing_mechanism`` 因 ``unknown`` 占 84% 必须排除（无效口径）。

用法
----
    python scripts/build_portal_input_defaults.py \
        --dataset /path/to/ml_dataset \
        --output prediction_portal/portal_input_defaults.json \
        --generated-at 2026-09-22
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

#: 无信息量的哨兵取值，不得作为默认值
SENTINEL_VALUES = {"other", "unknown", "nan", "none", "", "null", "unspecified", "na"}

#: 分类型测试条件字段（取众数）
CATEGORICAL_CONDITIONS = ("test_method", "test_atmosphere", "analysis_method", "specimen_geometry")

#: 连续型测试条件字段（取中位数）
CONTINUOUS_CONDITIONS = (
    "frequency_hz",
    "heating_rate_c_min",
    "loading_rate_mm_min",
    "strain_rate_s_1",
    "test_temperature_c",
)

#: 非目标列（元数据与测试条件）—— 这些列本身不是性能指标，不得作为目标列统计
NON_TARGET_COLUMNS = frozenset({
    "source_id",
    "performance_row_id",
    "record_id",
    "performance_category",
    "test_condition_id",
    "measurement_index",
    "property_family",
    "value_semantics",
    "ml_primary_candidate",
    "ml_extrapolated",
    "semantic_note",
}) | frozenset(CATEGORICAL_CONDITIONS) | frozenset(CONTINUOUS_CONDITIONS)

#: 配方聚合类字段（来自 ml_qspr_selected.csv）
RECIPE_CATEGORICAL = ("curing_type_standard", "formulation_resin_phr_basis_type", "initiator_present")

#: 明确排除的字段（口径无效或语义特殊）
EXCLUDED_RECIPE_FIELDS = frozenset({"curing_mechanism"})

#: 必须排除的派生/工艺字段名模式 —— 它们由 workflow 计算，不得给默认值
DERIVED_NAME_TOKENS = (
    "process_",
    "_test_standard",  # 由本脚本从 standards 表生成，但同名 workflow 列需排除
)


def _is_derived_field(name: str) -> bool:
    """判断字段是否为 workflow 派生字段（不得给默认值）。

    工艺类（process_*）与分子类（structure_* / *_xtb_* / *_maccs_*）均由
    workflow 从 cure_schedule / SMILES 真实计算，提供默认值等同于伪造数据。
    """
    text = str(name or "").strip().lower()
    if not text:
        return True
    if text.startswith("process_"):
        return True
    if "_structure_" in text or text.startswith("structure_"):
        return True
    if "_xtb_" in text or text.endswith("_xtb") or "_maccs_" in text:
        return True
    if "_mordred_" in text or "_polymer_" in text or "_bigsmiles_" in text:
        return True
    return False


def _clean_series(series: pd.Series) -> pd.Series:
    """去掉空值与哨兵值，返回可统计的取值序列。"""
    cleaned = series.dropna()
    if cleaned.empty:
        return cleaned
    as_text = cleaned.astype(str).str.strip()
    keep = ~as_text.str.lower().isin(SENTINEL_VALUES)
    return cleaned[keep]


def _mode_default(
    series: pd.Series, *, min_share: float, denominator: int | None = None
) -> dict[str, Any] | None:
    """取众数；占比低于 min_share 或全为哨兵值则返回 None。

    分母用**该目标列的样本数**（``denominator``）；未给定时退化为非空取值数。
    注意：哨兵值 ``other`` / ``unknown`` 也是真实记录值（意为“不在受控词表内”），
    因此计入分母 —— 否则会抬高其余取值的占比、高估置信度。
    但哨兵值自身**不得被选为默认值**（无信息量）。
    """
    non_null = series.dropna()
    total = int(denominator) if denominator else int(len(non_null))
    if total <= 0:
        return None
    cleaned = _clean_series(series)
    if cleaned.empty:
        return None
    counts = cleaned.value_counts()
    support = int(counts.iloc[0])
    share = support / total
    if share < min_share:
        return None
    value = counts.index[0]
    if isinstance(value, bool):
        value = bool(value)
    elif hasattr(value, "item"):
        value = value.item()
    return {"value": value, "share": round(float(share), 4), "support": support}


def _median_default(
    series: pd.Series, *, min_share: float, denominator: int | None = None
) -> dict[str, Any] | None:
    """取中位数；有效样本占比低于 min_share 则返回 None。

    ``denominator`` 必须是**该目标列的样本数**（而非该条件自身的非空数），
    否则 share 会恒为 1.0，误导读者以为“全部样本都有此条件”。
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    total = int(denominator) if denominator else int(series.notna().sum())
    if total <= 0:
        return None
    share = len(numeric) / total
    if share < min_share:
        return None
    value = float(numeric.median())
    return {"value": value, "share": round(float(share), 4), "support": int(len(numeric))}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - 防御性
        print(f"[warn] 无法读取 {path.name}：{exc}", file=sys.stderr)
        return None


def build_test_condition_defaults(
    performance: pd.DataFrame,
    standards: pd.DataFrame | None,
    *,
    min_share: float,
) -> dict[str, Any]:
    """为每个目标列统计测试条件默认值。"""
    target_columns = [
        column
        for column in performance.columns
        if column not in NON_TARGET_COLUMNS and not _is_derived_field(column)
    ]

    standard_by_row: dict[Any, str] = {}
    if standards is not None and not standards.empty:
        if "performance_row_id" in standards.columns and "standard_canonical" in standards.columns:
            usable = standards.dropna(subset=["performance_row_id", "standard_canonical"])
            standard_by_row = dict(
                zip(usable["performance_row_id"], usable["standard_canonical"].astype(str))
            )

    defaults: dict[str, Any] = {}
    for target in target_columns:
        series = performance[target]
        measured = performance[series.notna()]
        if measured.empty:
            continue
        fields: dict[str, Any] = {}
        target_sample_size = int(len(measured))
        for condition in CATEGORICAL_CONDITIONS:
            if condition not in performance.columns:
                continue
            found = _mode_default(
                measured[condition], min_share=min_share, denominator=target_sample_size
            )
            if found:
                fields[f"{target}_{condition}"] = {
                    **found,
                    "field": condition,
                    "source_table": "ml_performance_all.csv",
                }
        for condition in CONTINUOUS_CONDITIONS:
            if condition not in performance.columns:
                continue
            found = _median_default(
                measured[condition], min_share=min_share, denominator=target_sample_size
            )
            if found:
                fields[f"{target}_{condition}"] = {
                    **found,
                    "field": condition,
                    "source_table": "ml_performance_all.csv",
                }
        if standard_by_row:
            mapped = measured["performance_row_id"].map(standard_by_row).dropna()
            found = _mode_default(
                mapped, min_share=min_share, denominator=target_sample_size
            )
            if found:
                fields[f"{target}_test_standard"] = {
                    **found,
                    "field": "standard_canonical",
                    "source_table": "ml_performance_standards.csv",
                }
        if fields:
            defaults[target] = {
                "target_col": target,
                "sample_size": int(len(measured)),
                "fields": fields,
            }
    return defaults


def build_recipe_defaults(qspr: pd.DataFrame | None, *, min_share: float) -> dict[str, Any]:
    """统计配方聚合类字段默认值（来自 ml_qspr_selected.csv）。"""
    if qspr is None or qspr.empty:
        return {}
    defaults: dict[str, Any] = {}
    for field in RECIPE_CATEGORICAL:
        if field in EXCLUDED_RECIPE_FIELDS:
            continue
        if field not in qspr.columns:
            continue
        found = _mode_default(qspr[field], min_share=min_share)
        if found:
            defaults[field] = {**found, "source_table": "ml_qspr_selected.csv"}
    return defaults


def build_payload(
    *,
    dataset: Path,
    generated_at: str,
    min_share: float,
) -> dict[str, Any]:
    performance = _read_csv(dataset / "ml_performance_all.csv")
    if performance is None:
        raise SystemExit(f"缺少必需表：{dataset / 'ml_performance_all.csv'}")
    standards = _read_csv(dataset / "ml_performance_standards.csv")
    qspr = _read_csv(dataset / "ml_qspr_selected.csv")

    defaults = build_test_condition_defaults(performance, standards, min_share=min_share)
    recipe_defaults = build_recipe_defaults(qspr, min_share=min_share)

    # 最终防线：任何派生字段都不得出现在结果中
    for target, block in defaults.items():
        block["fields"] = {
            name: item
            for name, item in block["fields"].items()
            if not _is_derived_field(name.replace(f"{target}_", "", 1))
        }
    defaults = {k: v for k, v in defaults.items() if v["fields"]}
    recipe_defaults = {k: v for k, v in recipe_defaults.items() if k not in EXCLUDED_RECIPE_FIELDS}

    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_dataset": dataset.name,
        "min_share": min_share,
        "defaults": defaults,
        "recipe_defaults": recipe_defaults,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成门户输入默认值 JSON")
    parser.add_argument("--dataset", required=True, help="ml_dataset 目录")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--generated-at", default="", help="生成日期（保证可复现）")
    parser.add_argument("--min-share", type=float, default=0.30, help="最低占比阈值")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"数据集目录不存在：{dataset}")

    payload = build_payload(
        dataset=dataset,
        generated_at=args.generated_at,
        min_share=float(args.min_share),
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    n_targets = len(payload["defaults"])
    n_fields = sum(len(block["fields"]) for block in payload["defaults"].values())
    print(f"已写入 {output}")
    print(f"  目标列 {n_targets} 个，测试条件默认值 {n_fields} 条，配方默认值 {len(payload['recipe_defaults'])} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
