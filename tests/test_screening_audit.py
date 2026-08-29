# -*- coding: utf-8 -*-
"""Tests for the screening plan / candidate audit layer.

Covers the high-throughput screening audit additions:

* ``build_screening_plan`` / ``validate_screening_plan`` (hash-locked plan)
* ``apply_fixed_inputs`` (explicit process/test inputs, no mean/zero fill)
* candidate status machine in ``core.screening_audit`` (failed candidates kept)
* ``merge_model_results`` (multi-objective merge by ``candidate_id``)
* ``summarize_funnel`` (per-stage counts)
* ``screening_mode_for_artifact`` (formal vs exploratory)
* ``resolve_screening_feature_cols`` (contract-first feature resolution)
* ``save_screening_audit`` (plan / candidate audit / summary on disk)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from core.screening_audit import (
    FILTERED_BY_CHEMISTRY,
    MISSING_REQUIRED_INPUT,
    SELECTED,
    STRUCTURE_INVALID,
    WORKFLOW_FAILED,
    merge_model_results,
    new_candidate_pool,
    set_candidate_status,
    summarize_funnel,
)
from core.virtual_screening import (
    apply_fixed_inputs,
    build_screening_plan,
    compute_screening_plan_hash,
    resolve_screening_feature_cols,
    save_screening_audit,
    screening_fixed_input_cols,
    screening_mode_for_artifact,
    validate_screening_plan,
)

ARTIFACT_HASH = "a" * 64


def _make_contract():
    return {
        "schema_version": 2,
        "contract_hash": "c" * 64,
        "feature_cols": ["resin_smiles", "cure_temp_C", "cure_time_min"],
        "manual_input_feature_cols": [
            {"name": "cure_temp_C", "fixed": True},
            {"name": "cure_time_min", "fixed": False},
        ],
    }


def _fixed_temp(value=180.0, allow_missing=False, allow_zero=False):
    return {
        "feature_id": "cure_temp_C",
        "value": value,
        "unit": "C",
        "source": "process_spec",
        "reviewer": "tester",
        "allow_missing": allow_missing,
        "allow_zero": allow_zero,
    }


def _plan_kwargs():
    return dict(
        model_id="epoxy-tg",
        model_version="v2",
        artifact_hash=ARTIFACT_HASH,
        contract=_make_contract(),
        registry_hash="r" * 64,
        workflow_hash="w" * 64,
        candidate_source="virtual_library",
        fixed_inputs=[_fixed_temp()],
        random_state=1234,
    )


def _candidate_frame():
    # cure_temp_C pre-filled with a fake "training median" of 999.0.
    return pd.DataFrame(
        {
            "candidate_id": ["c0001", "c0002"],
            "resin_smiles": ["CCO", "CCC"],
            "cure_temp_C": [999.0, 999.0],
        }
    )


def _published_entry(artifact_hash=ARTIFACT_HASH, **overrides):
    entry = {
        "artifact_hash": artifact_hash,
        "publication_status": "published",
        "enabled": True,
        "gate_report": {"ok": True, "status": "valid"},
    }
    entry.update(overrides)
    return entry


def _portal_config(entry, material_type="epoxy_resin", target="glass_transition"):
    return {
        "materials": {
            material_type: {
                "targets": {
                    target: {"models": [entry]},
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# plan build + hash stability
# ---------------------------------------------------------------------------

def test_build_screening_plan_hash_is_stable_and_reproducible():
    plan_a = build_screening_plan(**_plan_kwargs())
    plan_b = build_screening_plan(**_plan_kwargs())
    assert plan_a["screening_plan_hash"] == plan_b["screening_plan_hash"]
    assert plan_a["screening_plan_hash"] == compute_screening_plan_hash(plan_a)
    assert len(plan_a["screening_plan_hash"]) == 64
    assert validate_screening_plan(plan_a) == []
    assert plan_a["random_state"] == 1234


def test_build_screening_plan_mirrors_contract_fields():
    plan = build_screening_plan(**_plan_kwargs())
    assert plan["contract_missing"] is False
    assert plan["contract_hash"] == "c" * 64
    assert plan["feature_cols"] == ["resin_smiles", "cure_temp_C", "cure_time_min"]
    assert plan["fixed_input_cols"] == ["cure_temp_C"]
    assert plan["registry_hash"] == "r" * 64
    assert plan["workflow_hash"] == "w" * 64
    assert plan["fixed_inputs"] == [_fixed_temp()]
    assert screening_fixed_input_cols(_make_contract()) == ["cure_temp_C"]


def test_build_screening_plan_without_contract_is_flagged_exploratory():
    plan = build_screening_plan(
        model_id="legacy",
        model_version="v1",
        artifact_hash=ARTIFACT_HASH,
    )
    assert plan["contract_missing"] is True
    assert plan["contract_hash"] is None
    assert plan["feature_cols"] == []
    assert plan["fixed_input_cols"] == []
    assert validate_screening_plan(plan) == []


# ---------------------------------------------------------------------------
# plan validation
# ---------------------------------------------------------------------------

def test_validate_screening_plan_detects_tampered_hash():
    plan = build_screening_plan(**_plan_kwargs())
    tampered = dict(plan)
    tampered["screening_plan_hash"] = "0" * 64
    errors = validate_screening_plan(tampered)
    assert errors and any("screening_plan_hash" in message for message in errors)

    errors_missing = validate_screening_plan({"model_id": "x"})
    assert errors_missing
    assert any("缺少必需字段" in message for message in errors_missing)


def test_validate_screening_plan_rejects_unfilled_required_fixed_input():
    plan = build_screening_plan(
        model_id="epoxy-tg",
        model_version="v2",
        artifact_hash=ARTIFACT_HASH,
        fixed_inputs=[_fixed_temp(value=None, allow_missing=False)],
    )
    errors = validate_screening_plan(plan)
    assert errors and any("allow_missing" in message for message in errors)


# ---------------------------------------------------------------------------
# fixed inputs
# ---------------------------------------------------------------------------

def test_apply_fixed_inputs_overrides_process_columns_from_plan():
    plan = build_screening_plan(**_plan_kwargs())
    outcome = apply_fixed_inputs(_candidate_frame(), plan)
    frame = outcome["frame"]
    assert outcome["fixed_cols"] == ["cure_temp_C"]
    assert outcome["missing_records"] == []
    # Explicit plan value wins over the 999.0 "training median" pre-fill.
    assert list(frame["cure_temp_C"]) == [180.0, 180.0]
    assert list(frame["candidate_id"]) == ["c0001", "c0002"]


def test_apply_fixed_inputs_accepts_single_candidate_dict():
    plan = build_screening_plan(**_plan_kwargs())
    outcome = apply_fixed_inputs({"candidate_id": "c0001", "cure_temp_C": 999.0}, plan)
    assert outcome["missing_records"] == []
    assert float(outcome["frame"]["cure_temp_C"].iloc[0]) == 180.0


def test_apply_fixed_inputs_zero_value_requires_allow_zero():
    plan = build_screening_plan(
        model_id="m",
        model_version="v1",
        artifact_hash=ARTIFACT_HASH,
        fixed_inputs=[_fixed_temp(value=0.0, allow_zero=False)],
    )
    outcome = apply_fixed_inputs(_candidate_frame(), plan)
    assert outcome["fixed_cols"] == ["cure_temp_C"]
    assert len(outcome["missing_records"]) == 2
    # Zero is never silently written over the pre-fill.
    assert list(outcome["frame"]["cure_temp_C"]) == [999.0, 999.0]

    plan_zero_ok = build_screening_plan(
        model_id="m",
        model_version="v1",
        artifact_hash=ARTIFACT_HASH,
        fixed_inputs=[_fixed_temp(value=0.0, allow_zero=True)],
    )
    outcome_ok = apply_fixed_inputs(_candidate_frame(), plan_zero_ok)
    assert outcome_ok["missing_records"] == []
    assert list(outcome_ok["frame"]["cure_temp_C"]) == [0.0, 0.0]


def test_apply_fixed_inputs_missing_value_flags_candidates_not_filled():
    raw_plan = {"fixed_inputs": [_fixed_temp(value=None, allow_missing=False)]}
    outcome = apply_fixed_inputs(_candidate_frame(), raw_plan)
    assert [r["feature_id"] for r in outcome["missing_records"]] == [
        "cure_temp_C",
        "cure_temp_C",
    ]
    assert sorted(r["candidate_id"] for r in outcome["missing_records"]) == [
        "c0001",
        "c0002",
    ]

    pool = new_candidate_pool([{"resin_smiles": "CCO"}, {"resin_smiles": "CCC"}])
    for record in outcome["missing_records"]:
        set_candidate_status(
            pool,
            record["candidate_id"],
            MISSING_REQUIRED_INPUT,
            "缺少固定输入 " + record["feature_id"],
        )
    # Failed candidates are kept in the pool, not deleted.
    assert len(pool) == 2
    assert all(entry["status"] == MISSING_REQUIRED_INPUT for entry in pool)
    assert all("cure_temp_C" in entry["failure_reason"] for entry in pool)


def test_apply_fixed_inputs_allow_missing_true_records_nothing():
    raw_plan = {"fixed_inputs": [_fixed_temp(value=None, allow_missing=True)]}
    outcome = apply_fixed_inputs(_candidate_frame(), raw_plan)
    assert outcome["missing_records"] == []
    assert list(outcome["frame"]["cure_temp_C"]) == [999.0, 999.0]


# ---------------------------------------------------------------------------
# candidate status machine
# ---------------------------------------------------------------------------

def test_candidate_pool_keeps_failed_candidates_with_status_and_reason():
    pool = new_candidate_pool(
        [
            {"resin_smiles": "CCO", "source": "library"},
            {"resin_smiles": "not-a-smile", "source": "generated"},
        ]
    )
    assert [entry["candidate_id"] for entry in pool] == ["c0001", "c0002"]
    assert all(entry["status"] == "valid" for entry in pool)
    assert all(entry["failure_reason"] is None for entry in pool)

    updated = set_candidate_status(pool, "c0002", STRUCTURE_INVALID, "RDKit 解析失败")
    assert updated["status"] == STRUCTURE_INVALID
    assert updated["failure_reason"] == "RDKit 解析失败"
    assert len(pool) == 2  # retained, not removed
    assert pool[1]["status"] == STRUCTURE_INVALID

    # Failure status without an explicit reason gets a placeholder reason.
    pool_single = new_candidate_pool([{"resin_smiles": "CCC"}])
    set_candidate_status(pool_single, "c0001", WORKFLOW_FAILED)
    assert pool_single[0]["status"] == WORKFLOW_FAILED
    assert pool_single[0]["failure_reason"] == "未提供失败原因"

    # Unknown candidate id is ignored defensively.
    assert set_candidate_status(pool, "c9999", "valid") is None


def test_set_candidate_status_rejects_unknown_status():
    pool = new_candidate_pool([{"resin_smiles": "CCO"}])
    with pytest.raises(ValueError):
        set_candidate_status(pool, "c0001", "totally_bogus_status")


# ---------------------------------------------------------------------------
# multi-objective merge
# ---------------------------------------------------------------------------

def test_merge_model_results_merges_by_candidate_id_with_prefixed_columns():
    pool_a = new_candidate_pool([{"resin_smiles": "CCO"}, {"resin_smiles": "CCC"}])
    pool_b = new_candidate_pool([{"resin_smiles": "CCO"}])
    pool_a[0]["prediction"] = 1.5
    pool_a[1]["prediction"] = 2.0
    pool_b[0]["prediction"] = 9.9

    outcome = merge_model_results(pools=[pool_a, pool_b])
    merged = outcome["merged"]
    assert outcome["model_tags"] == ["model_1", "model_2"]
    assert "candidate_id" in merged.columns
    assert "model_1__prediction" in merged.columns
    assert "model_2__prediction" in merged.columns
    assert "prediction" not in merged.columns  # model columns never mixed
    assert len(merged) == 2  # outer join keeps the union of candidate ids
    row = merged.loc[merged["candidate_id"] == "c0001"].iloc[0]
    assert float(row["model_1__prediction"]) == 1.5
    assert float(row["model_2__prediction"]) == 9.9


def test_merge_model_results_requires_candidate_id_column():
    with pytest.raises(ValueError):
        merge_model_results(pools=[[{"resin_smiles": "CCO"}]])
    empty = merge_model_results(pools=[])
    assert empty["merged"].empty
    assert empty["candidate_ids"] == []
    assert empty["model_tags"] == []


# ---------------------------------------------------------------------------
# funnel summary
# ---------------------------------------------------------------------------

def test_summarize_funnel_reports_per_stage_counts():
    pool = new_candidate_pool([{}, {}, {}, {}])
    set_candidate_status(pool, "c0003", FILTERED_BY_CHEMISTRY, "无环氧基团")
    set_candidate_status(pool, "c0004", SELECTED)
    summary = summarize_funnel(pool)
    assert summary["total"] == 4
    assert summary["remaining"] == 3  # 2 valid + 1 selected
    assert summary["status_counts"] == {
        "valid": 2,
        "filtered_by_chemistry": 1,
        "selected": 1,
    }
    stages = {stage["stage"]: stage for stage in summary["stages"]}
    assert stages["候选池生成"]["count"] == 2
    assert stages["化学规则过滤"]["count"] == 1
    assert stages["入选"]["count"] == 1


# ---------------------------------------------------------------------------
# formal vs exploratory mode
# ---------------------------------------------------------------------------

def test_screening_mode_formal_requires_contract_and_publication():
    artifact = {
        "artifact_hash": ARTIFACT_HASH,
        "extra": {
            "prediction_contract": {
                "schema_version": 2,
                "contract_hash": "c" * 64,
                "feature_cols": ["f1", "f2"],
            }
        },
    }
    config = _portal_config(_published_entry())
    outcome = screening_mode_for_artifact(
        artifact=artifact,
        config=config,
        material_type="epoxy_resin",
        target="glass_transition",
    )
    assert outcome["mode"] == "formal"
    assert outcome["result_is_formal"] is True
    assert outcome["contract"]["feature_cols"] == ["f1", "f2"]


def test_screening_mode_exploratory_without_contract():
    artifact = {"artifact_hash": "b" * 64, "feature_cols": ["x"], "extra": {}}
    outcome = screening_mode_for_artifact(artifact=artifact)
    assert outcome["mode"] == "exploratory"
    assert outcome["result_is_formal"] is False
    assert any("prediction_contract" in reason for reason in outcome["reasons"])
    assert outcome["contract"] is None


def test_screening_mode_exploratory_when_gate_report_fails():
    artifact = {
        "artifact_hash": ARTIFACT_HASH,
        "extra": {
            "prediction_contract": {
                "schema_version": 2,
                "contract_hash": "c" * 64,
                "feature_cols": ["f1"],
            }
        },
    }
    config = _portal_config(
        _published_entry(gate_report={"ok": True, "status": "invalid"})
    )
    outcome = screening_mode_for_artifact(
        artifact=artifact,
        config=config,
        material_type="epoxy_resin",
        target="glass_transition",
    )
    assert outcome["mode"] == "exploratory"
    assert outcome["result_is_formal"] is False
    assert any(
        ("门禁" in reason) or ("gate" in reason.lower())
        for reason in outcome["reasons"]
    )


# ---------------------------------------------------------------------------
# feature column resolution
# ---------------------------------------------------------------------------

def test_resolve_screening_feature_cols_prefers_contract():
    contract = {
        "schema_version": 2,
        "contract_hash": "c" * 64,
        "feature_cols": ["f1", "f2"],
    }
    outcome = resolve_screening_feature_cols(contract)
    assert outcome == {
        "feature_cols": ["f1", "f2"],
        "source": "contract",
        "warnings": [],
        "result_is_formal": True,
    }

    artifact_with_contract = {
        "feature_cols": ["stale"],
        "extra": {
            "prediction_contract": {
                "schema_version": 2,
                "feature_cols": ["p", "q"],
                "contract_hash": "z",
            }
        },
    }
    outcome = resolve_screening_feature_cols(artifact_with_contract)
    assert outcome["source"] == "contract"
    assert outcome["feature_cols"] == ["p", "q"]


def test_resolve_screening_feature_cols_falls_back_to_artifact_with_warnings():
    artifact = {"feature_cols": ["a", "b"]}
    outcome = resolve_screening_feature_cols(artifact)
    assert outcome["source"] == "artifact_fallback"
    assert outcome["feature_cols"] == ["a", "b"]
    assert outcome["warnings"]
    assert any("artifact.feature_cols" in warning for warning in outcome["warnings"])
    assert outcome["result_is_formal"] is False

    empty = resolve_screening_feature_cols({})
    assert empty["feature_cols"] == []
    assert empty["warnings"] and any("未找到" in warning for warning in empty["warnings"])


# ---------------------------------------------------------------------------
# audit persistence
# ---------------------------------------------------------------------------

def test_save_screening_audit_writes_parseable_files(tmp_path):
    plan = build_screening_plan(**_plan_kwargs())
    pool = new_candidate_pool(
        [
            {"resin_smiles": "CCO", "component_roles": ["resin"]},
            {"resin_smiles": "CCC"},
        ]
    )
    set_candidate_status(pool, "c0002", FILTERED_BY_CHEMISTRY, "无环氧基团")
    results = pd.DataFrame({"candidate_id": ["c0001"], "prediction": [1.23]})

    out_dir = Path(save_screening_audit(plan, pool, results, tmp_path / "audit_run"))
    assert (out_dir / "screening_plan.json").is_file()
    assert (out_dir / "candidate_audit.jsonl").is_file()
    assert (out_dir / "audit_summary.json").is_file()
    assert (out_dir / "screening_results.csv").is_file()

    plan_loaded = json.loads(
        (out_dir / "screening_plan.json").read_text(encoding="utf-8")
    )
    assert plan_loaded["screening_plan_hash"] == plan["screening_plan_hash"]
    assert plan_loaded["model_id"] == "epoxy-tg"

    audit_lines = [
        json.loads(line)
        for line in (out_dir / "candidate_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [record["candidate_id"] for record in audit_lines] == ["c0001", "c0002"]
    assert audit_lines[0]["status"] == "valid"
    assert audit_lines[1]["status"] == FILTERED_BY_CHEMISTRY
    assert audit_lines[1]["failure_reason"] == "无环氧基团"

    summary = json.loads(
        (out_dir / "audit_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total_candidates"] == 2
    assert summary["screening_plan_hash"] == plan["screening_plan_hash"]
    assert summary["status_counts"]["filtered_by_chemistry"] == 1
    assert summary["funnel"]

    results_df = pd.read_csv(out_dir / "screening_results.csv")
    assert list(results_df["candidate_id"]) == ["c0001"]
