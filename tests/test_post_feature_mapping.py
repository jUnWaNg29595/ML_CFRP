import copy

import numpy as np
import pandas as pd
import pytest

from core.virtual_screening import add_candidate_equivalent_metrics, apply_feature_overrides
from core.post_feature_mapping import (
    MAPPING_SESSION_KEYS,
    POST_FEATURE_MAPPING_SCHEMA_VERSION,
    apply_mapping,
    build_post_feature_catalog,
    catalog_fingerprint,
    create_mapping_draft,
    mapping_fingerprint,
    mapping_snapshot,
    mapping_snapshot_restore_policy,
    validate_mapping,
)


def test_candidate_metrics_expose_stable_computed_columns_and_legacy_aliases():
    result = add_candidate_equivalent_metrics(
        pd.DataFrame({"resin_smiles": ["C1CO1"], "hardener_smiles": ["NCCN"]})
    )
    assert "computed_resin_eew" in result.columns
    assert "computed_hardener_ahew" in result.columns
    assert "computed_resin_molecular_weight" in result.columns
    assert "computed_hardener_molecular_weight" in result.columns
    assert "EEW" in result.columns
    assert "AHEW" in result.columns


def test_apply_feature_overrides_does_not_inject_reserved_post_features():
    result = apply_feature_overrides(
        pd.DataFrame({"EEW": [100.0], "temperature": [20.0]}),
        pd.DataFrame({"EEW": [180.0], "temperature": [80.0]}),
    )
    assert result.loc[0, "EEW"] == 100.0
    assert result.loc[0, "temperature"] == 80.0


def test_empty_draft_does_not_match_same_named_column():
    catalog = build_post_feature_catalog(
        pd.DataFrame({"EEW": [180.0], "computed_resin_eew": [181.0]}),
        computed_definitions={
            "computed_resin_eew": {
                "category": "EEW",
                "unit": "g/eq",
                "definition": "resin molecular weight / epoxy functionality",
            }
        },
    )
    draft = create_mapping_draft(
        ["EEW"],
        molecular_feature_cols=[],
        catalog=catalog,
        model_fingerprint="model-a",
        workflow_fingerprint=None,
    )
    assert draft["status"] == "draft"
    assert draft["confirmed"] is False
    assert draft["rules"]["EEW"]["source_type"] == "pending"
    assert draft["rules"]["EEW"]["source_column"] is None


def test_default_mapping_is_only_a_draft():
    draft = create_mapping_draft(
        ["EEW"],
        molecular_feature_cols=[],
        catalog=pd.DataFrame(
            [{"column": "computed_resin_eew", "source_type": "computed"}]
        ),
        model_fingerprint="m",
        workflow_fingerprint="w",
    )
    assert draft["status"] == "draft"
    assert draft["confirmed"] is False


def test_mapping_draft_is_blank_even_when_catalog_contains_all_model_names():
    candidate_df = pd.DataFrame({"temperature": [20.0], "pressure": [1.0]})
    catalog = build_post_feature_catalog(candidate_df, computed_definitions={})
    draft = create_mapping_draft(
        ["temperature", "pressure"],
        molecular_feature_cols=[],
        catalog=catalog,
        model_fingerprint="model-a",
        workflow_fingerprint="workflow-a",
    )
    assert all(rule["source_type"] == "pending" for rule in draft["rules"].values())
    assert all(rule["source_column"] is None for rule in draft["rules"].values())
    assert draft["workflow_fingerprint"] == "workflow-a"


def test_apply_mapping_supports_all_explicit_source_types_in_model_order():
    candidate_df = pd.DataFrame(
        {
            "computed_resin_eew": [181.0, 182.0],
            "raw_ratio": ["0.8", "0.9"],
        }
    )
    base_matrix = pd.DataFrame(
        {
            "computed_target": [np.nan, np.nan],
            "raw_target": [np.nan, np.nan],
            "constant_target": [np.nan, np.nan],
            "unused_target": [7.0, 7.0],
        }
    )
    mapping = {
        "schema_version": 1,
        "model_feature_cols": [
            "computed_target",
            "raw_target",
            "constant_target",
            "unused_target",
        ],
        "rules": {
            "computed_target": {
                "source_type": "computed",
                "source_column": "computed_resin_eew",
                "confirmed": True,
            },
            "raw_target": {
                "source_type": "candidate",
                "source_column": "raw_ratio",
                "confirmed": True,
            },
            "constant_target": {
                "source_type": "constant",
                "source_column": None,
                "constant_value": 3.5,
                "confirmed": True,
            },
            "unused_target": {
                "source_type": "unused",
                "source_column": None,
                "confirmed": True,
            },
        },
        "confirmed": True,
    }
    result = apply_mapping(
        base_matrix,
        candidate_df,
        mapping,
        model_feature_cols=list(base_matrix.columns),
    )
    assert result.columns.tolist() == list(base_matrix.columns)
    assert result["computed_target"].tolist() == [181.0, 182.0]
    assert result["raw_target"].tolist() == [0.8, 0.9]
    assert result["constant_target"].tolist() == [3.5, 3.5]
    assert result["unused_target"].isna().all()


def test_validate_mapping_blocks_unconfirmed_missing_invalid_and_nonfinite_inputs():
    candidate_df = pd.DataFrame(
        {
            "bad_raw": ["bad", "2.0"],
            "nonfinite": [1.0, float("inf")],
        }
    )
    catalog = build_post_feature_catalog(candidate_df, computed_definitions={})
    mapping = {
        "schema_version": 1,
        "model_feature_cols": ["missing", "bad", "nonfinite", "constant"],
        "rules": {
            "missing": {"source_type": "pending", "confirmed": False},
            "bad": {"source_type": "candidate", "source_column": "bad_raw", "confirmed": True},
            "nonfinite": {"source_type": "candidate", "source_column": "nonfinite", "confirmed": True},
            "constant": {"source_type": "constant", "constant_value": "nan", "confirmed": True},
        },
    }
    report = validate_mapping(
        mapping,
        model_feature_cols=["missing", "bad", "nonfinite", "constant"],
        candidate_df=candidate_df,
        catalog=catalog,
        missing_input_tolerant=False,
    )
    assert report["ok"] is False
    assert any("missing" in error for error in report["errors"])
    assert any("bad_raw" in error for error in report["errors"])
    assert any("nonfinite" in error for error in report["errors"])


def test_missing_input_tolerance_only_allows_explicit_unused_columns():
    candidate_df = pd.DataFrame(index=[0, 1])
    catalog = build_post_feature_catalog(candidate_df, computed_definitions={})
    mapping = {
        "schema_version": 1,
        "model_feature_cols": ["optional"],
        "rules": {
            "optional": {
                "source_type": "unused",
                "confirmed": True,
            }
        },
        "confirmed": True,
    }
    strict = validate_mapping(
        mapping,
        model_feature_cols=["optional"],
        candidate_df=candidate_df,
        catalog=catalog,
        missing_input_tolerant=False,
    )
    tolerant = validate_mapping(
        mapping,
        model_feature_cols=["optional"],
        candidate_df=candidate_df,
        catalog=catalog,
        missing_input_tolerant=True,
    )
    assert strict["ok"] is False
    assert tolerant["ok"] is True


def test_mapping_and_catalog_fingerprints_are_stable_but_catalog_changes_invalidate():
    mapping = {
        "schema_version": POST_FEATURE_MAPPING_SCHEMA_VERSION,
        "model_feature_cols": ["b", "a"],
        "rules": {
            "b": {"source_type": "constant", "constant_value": 1.0, "confirmed": True},
            "a": {"source_type": "unused", "confirmed": True},
        },
    }
    reordered = copy.deepcopy(mapping)
    reordered["rules"] = {"a": reordered["rules"]["a"], "b": reordered["rules"]["b"]}
    assert mapping_fingerprint(mapping) == mapping_fingerprint(reordered)
    first_catalog = build_post_feature_catalog(
        pd.DataFrame({"temperature": [20.0]}),
        computed_definitions={},
    )
    second_catalog = build_post_feature_catalog(
        pd.DataFrame({"temperature": [80.0]}),
        computed_definitions={},
    )
    assert catalog_fingerprint(first_catalog) != catalog_fingerprint(second_catalog)


def test_validation_rejects_model_order_and_catalog_fingerprint_changes():
    first_candidate = pd.DataFrame({"temperature": [20.0]})
    first_catalog = build_post_feature_catalog(
        first_candidate,
        computed_definitions={},
    )
    mapping = {
        "schema_version": 1,
        "model_feature_cols": ["temperature", "pressure"],
        "rules": {
            "temperature": {
                "source_type": "candidate",
                "source_column": "temperature",
                "confirmed": True,
            },
            "pressure": {
                "source_type": "constant",
                "constant_value": 1.0,
                "confirmed": True,
            },
        },
        "confirmed": True,
        "catalog_fingerprint": catalog_fingerprint(first_catalog),
    }
    changed_catalog = build_post_feature_catalog(
        pd.DataFrame({"temperature": [80.0]}),
        computed_definitions={},
    )
    report = validate_mapping(
        mapping,
        model_feature_cols=["pressure", "temperature"],
        candidate_df=first_candidate,
        catalog=changed_catalog,
        missing_input_tolerant=False,
    )
    assert report["ok"] is False
    assert any("model feature order" in error for error in report["errors"])
    assert any("catalog fingerprint changed" in error for error in report["errors"])


def test_mapping_snapshot_is_immutable_and_contains_hash_and_feature_order():
    mapping = {
        "schema_version": 1,
        "model_feature_cols": ["b", "a"],
        "rules": {
            "b": {"source_type": "unused", "confirmed": True},
            "a": {"source_type": "constant", "constant_value": 1.0, "confirmed": True},
        },
    }
    snapshot = mapping_snapshot(mapping, confirmed_at="2026-07-29T12:00:00+08:00")
    mapping["rules"]["a"]["constant_value"] = 99.0
    assert snapshot["model_feature_cols"] == ["b", "a"]
    assert snapshot["mapping_hash"] == mapping_fingerprint({
        **mapping,
        "rules": {
            "b": {"source_type": "unused", "confirmed": True},
            "a": {"source_type": "constant", "constant_value": 1.0, "confirmed": True},
        },
    })
    assert snapshot["confirmed_at"] == "2026-07-29T12:00:00+08:00"


def test_ordinary_and_formula_prediction_gate_share_same_validation():
    from core.post_feature_mapping import validate_mapping_for_prediction

    candidate_df = pd.DataFrame({"computed_resin_eew": [181.0]})
    catalog = build_post_feature_catalog(
        candidate_df,
        computed_definitions={
            "computed_resin_eew": {
                "category": "EEW",
                "unit": "g/eq",
                "definition": "test",
            }
        },
    )
    mapping = {
        "schema_version": 1,
        "model_feature_cols": ["EEW"],
        "rules": {
            "EEW": {
                "source_type": "computed",
                "source_column": "computed_resin_eew",
                "confirmed": True,
            }
        },
        "confirmed": True,
    }
    ordinary = validate_mapping_for_prediction(
        mapping,
        model_feature_cols=["EEW"],
        candidate_df=candidate_df,
        catalog=catalog,
        missing_input_tolerant=False,
    )
    formula = validate_mapping_for_prediction(
        mapping,
        model_feature_cols=["EEW"],
        candidate_df=candidate_df,
        catalog=catalog,
        missing_input_tolerant=False,
    )
    assert ordinary == formula
    assert ordinary["ok"] is True


def test_mapping_restore_policy_is_pure_and_clears_old_versions_or_missing_keys():
    assert (
        mapping_snapshot_restore_policy(
            {"version": 1, "sid": "session-a"},
            current_session_id="session-a",
        )
        == "clear"
    )
    assert (
        mapping_snapshot_restore_policy(
            {"version": 2, "sid": "session-a"},
            current_session_id="session-a",
        )
        == "clear"
    )
    complete = {"version": 2, "sid": "session-a"}
    complete.update({key: None for key in MAPPING_SESSION_KEYS})
    assert (
        mapping_snapshot_restore_policy(
            complete,
            current_session_id="session-a",
        )
        == "restore"
    )
    assert (
        mapping_snapshot_restore_policy(
            complete,
            current_session_id="session-b",
        )
        == "session_mismatch"
    )
