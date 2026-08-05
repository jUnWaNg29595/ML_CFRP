import copy
import time

import pandas as pd
import pytest
import core.molecular_features as molecular_features

from core.molecular_feature_workflow import (
    MolecularFeatureWorkflow,
    append_training_workflow,
    append_workflow_step,
    build_workflow_from_training_state,
    build_workflow_diff,
    can_lock_workflow,
    compute_workflow_hash,
    execute_feature_step,
    execute_molecular_feature_workflow,
    find_duplicate_feature_names,
    merge_feature_name_lists_in_order,
    normalize_workflow_config,
    prepare_step_inputs,
    get_feature_component_column_options,
    merge_extracted_features,
    resolve_feature_component_columns,
    resolve_feature_component_role,
    validate_feature_frame_contract,
    validate_workflow_config,
)
from core.molecular_features import append_configured_semantic_features, extract_configured_semantic_features


def test_ionic_semantic_features_reuses_duplicate_fragment_parse(monkeypatch):
    class FakeAtom:
        def __init__(self, symbol, charge):
            self._symbol = symbol
            self._charge = charge

        def GetFormalCharge(self):
            return self._charge

        def GetSymbol(self):
            return self._symbol

    class FakeMol:
        def __init__(self, atoms):
            self._atoms = atoms

        def GetAtoms(self):
            return self._atoms

    calls = []

    def fake_parse(fragment, **_kwargs):
        calls.append(fragment)
        if fragment == "[Na+]":
            return FakeMol([FakeAtom("Na", 1)])
        if fragment == "[Cl-]":
            return FakeMol([FakeAtom("Cl", -1)])
        return FakeMol([])

    monkeypatch.setattr(molecular_features, "RDKIT_AVAILABLE", True)
    monkeypatch.setattr(
        molecular_features,
        "diagnose_chemical_string",
        lambda _text: {
            "proxy_smiles_ok": True,
            "rdkit_direct_ok": True,
            "normalized_ok": True,
        },
    )
    monkeypatch.setattr(
        molecular_features,
        "split_smiles_cell",
        lambda text: [part for part in str(text).split(".") if part],
    )
    monkeypatch.setattr(molecular_features, "parse_chemical_string", fake_parse)

    cache_clear = getattr(molecular_features._safe_fragment_mol_from_text, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    result = molecular_features.extract_ionic_semantic_features(
        ["[Na+].[Cl-]", "[Na+].[Cl-]"]
    )

    assert calls == ["[Na+]", "[Cl-]"]
    assert result["ionic_has_cation"].tolist() == [1.0, 1.0]
    assert result["ionic_has_anion"].tolist() == [1.0, 1.0]


def test_ionic_semantic_features_fast_paths_nonionic_bigsmiles(monkeypatch):
    monkeypatch.setattr(molecular_features, "RDKIT_AVAILABLE", True)
    monkeypatch.setattr(
        molecular_features,
        "detect_chem_string_format",
        lambda _text: "bigsmiles",
    )
    monkeypatch.setattr(molecular_features, "parse_smiles_quiet", lambda _text: None)
    monkeypatch.setattr(
        molecular_features,
        "diagnose_chemical_string",
        lambda _text: pytest.fail("non-ionic BigSMILES should not run full diagnosis"),
    )
    monkeypatch.setattr(
        molecular_features,
        "split_smiles_cell",
        lambda _text: pytest.fail("non-ionic BigSMILES should not be converted for ionic features"),
    )

    result = molecular_features.extract_ionic_semantic_features(
        ["{[>][<]CC(C)O}"]
    )

    assert result["ionic_missing"].tolist() == [0.0]
    assert result["ionic_has_any_charge"].tolist() == [0.0]
    assert result["ionic_proxy_parse_ok"].tolist() == [1.0]
    assert result["ionic_normalized_ok"].tolist() == [1.0]
    assert result["ionic_fragment_count"].tolist() == [1.0]


def test_merge_extracted_features_preserves_contiguous_rows_without_reindexing():
    base = pd.DataFrame({"source": range(4)})
    features = pd.DataFrame({"feature": [10, 20, 30, 40]})

    merged = merge_extracted_features(
        base,
        features,
        [0, 1, 2, 3],
        keep_all_rows=True,
    )

    assert merged.to_dict("list") == {
        "source": [0, 1, 2, 3],
        "feature": [10, 20, 30, 40],
    }


def test_merge_extracted_features_restores_sparse_rows():
    base = pd.DataFrame({"source": range(4)})
    features = pd.DataFrame({"feature": [10, 30]})

    merged = merge_extracted_features(
        base,
        features,
        [0, 2],
        keep_all_rows=True,
    )

    assert merged["source"].tolist() == [0, 1, 2, 3]
    assert merged["feature"].iloc[0] == 10
    assert pd.isna(merged["feature"].iloc[1])
    assert merged["feature"].iloc[2] == 30
    assert pd.isna(merged["feature"].iloc[3])


def test_append_configured_semantic_features_only_processes_valid_rows(monkeypatch):
    calls = []

    def fake_extract(smiles_like_list, params=None, **kwargs):
        calls.append(list(smiles_like_list))
        return pd.DataFrame({"semantic": [len(value) for value in smiles_like_list]})

    monkeypatch.setattr(
        molecular_features,
        "extract_configured_semantic_features",
        fake_extract,
    )

    base = pd.DataFrame({"xtb": [1, 2]})
    result, valid_indices = append_configured_semantic_features(
        base,
        [1, 3],
        ["invalid-row-0", "a", "invalid-row-2", "abcd"],
        {"append_ionic_semantic_features": True},
    )

    assert calls == [["a", "abcd"]]
    assert valid_indices == [1, 3]
    assert result["semantic"].tolist() == [1, 4]


def _workflow_payload():
    return {
        "schema_version": 2,
        "model_fingerprint": "model-1",
        "mode": "multi_batch",
        "steps": [
            {
                "step_id": "resin_1",
                "order": 0,
                "role": "resin",
                "source_columns": ["resin_smiles_1"],
                "method": "xTB",
                "prefix": "resin_1",
                "params": {"xtb_method": "gfn2"},
                "semantic_params": {},
                "feature_names": ["resin_1_xtb_gap"],
            },
            {
                "step_id": "hardener_1",
                "order": 1,
                "role": "hardener",
                "source_columns": ["hardener_smiles_1"],
                "method": "xTB",
                "prefix": "hardener_1",
                "params": {"xtb_method": "gfn2"},
                "semantic_params": {},
                "feature_names": ["hardener_1_xtb_gap"],
            },
        ],
        "merge_order": ["resin_1", "hardener_1"],
        "final_feature_names": ["resin_1_xtb_gap", "hardener_1_xtb_gap"],
    }


def make_test_step_output(smiles, step, **_kwargs):
    valid_indices = [index for index, value in enumerate(smiles) if value is not None]
    columns = step.get("test_columns", ["value"])
    values = [
        [smiles[index] for index in valid_indices]
        for _column in columns
    ]
    return (
        pd.DataFrame(
            list(zip(*values)) if values else [],
            columns=columns,
            index=valid_indices,
        ),
        valid_indices,
        [],
    )


def test_merge_feature_name_lists_preserves_first_seen_order():
    assert merge_feature_name_lists_in_order(
        ["base_1", "resin_1_x"],
        ["resin_1_x", "hardener_1_x", "resin_2_x"],
    ) == ["base_1", "resin_1_x", "hardener_1_x", "resin_2_x"]


def test_workflow_diff_reports_changed_prefix_and_missing_step():
    report = build_workflow_diff(
        {
            "steps": [
                {"step_id": "resin_1", "prefix": "resin_1", "method": "xTB"}
            ]
        },
        {
            "steps": [
                {"step_id": "resin_1", "prefix": "epoxy_1", "method": "xTB"},
                {"step_id": "hardener_1", "prefix": "hardener_1", "method": "xTB"},
            ]
        },
    )

    assert report["changed_steps"] == ["resin_1"]
    assert report["added_steps"] == ["hardener_1"]


def test_workflow_cannot_lock_with_missing_required_items():
    assert can_lock_workflow({"ok": False, "missing_steps": ["hardener_1"]}) is False
    assert can_lock_workflow({"ok": True, "missing_steps": []}) is True


def test_build_workflow_from_training_state_preserves_steps_parameters_and_features():
    workflow = build_workflow_from_training_state(
        {
            "mode": "multi_batch",
            "model_fingerprint": "model-1",
            "input_contract": {"row_policy": "keep_all_rows"},
            "workflow_steps": [
                {
                    "step_id": "resin_1",
                    "order": 0,
                    "role": "resin",
                    "source_columns": ["resin_smiles_1"],
                    "prefix": "resin_1",
                    "method": "xTB",
                    "params": {"xtb_method": "gfn2"},
                    "semantic_params": {"append_ionic_semantic_features": True},
                    "feature_names": ["resin_1_xtb_gap"],
                }
            ],
        },
        ["resin_1_xtb_gap"],
    )

    assert workflow.schema_version == 2
    assert workflow.mode == "multi_batch"
    assert workflow.merge_order == ["resin_1"]
    assert workflow.final_feature_names == ["resin_1_xtb_gap"]
    assert workflow.steps[0]["source_columns"] == ["resin_smiles_1"]
    assert workflow.steps[0]["params"]["xtb_method"] == "gfn2"
    assert workflow.steps[0]["semantic_params"]["append_ionic_semantic_features"] is True


def test_build_workflow_uses_selected_source_column_order_for_default_step():
    workflow = build_workflow_from_training_state(
        {
            "mode": "single_batch",
            "selected_source_columns": ["hardener_smiles", "resin_smiles"],
            "method": "xTB",
            "role": "hardener",
            "prefix": "ordered",
            "params": {"xtb_method": "gfn2"},
            "semantic_params": {"append_ionic_semantic_features": True},
            "valid_row_behavior": {
                "valid_indices": [1],
                "keep_all_rows": True,
                "invalid_row_policy": "nan_fill",
            },
        },
        ["ordered_xtb_gap"],
    )

    step = workflow.steps[0]
    assert step["source_columns"] == ["hardener_smiles", "resin_smiles"]
    assert step["role"] == "hardener"
    assert step["prefix"] == "ordered"
    assert step["method"] == "xTB"
    assert step["params"] == {"xtb_method": "gfn2"}
    assert step["semantic_params"] == {"append_ionic_semantic_features": True}
    assert step["valid_row_behavior"]["valid_indices"] == [1]
    assert step["feature_names"] == ["ordered_xtb_gap"]


def test_build_workflow_preserves_ionic_filter_contract_and_source_mapping():
    workflow = build_workflow_from_training_state(
        {
            "mode": "multi_batch",
            "selected_source_columns": ["resin_smiles"],
            "input_contract": {
                "skip_ionic_compounds": True,
                "ionic_filter_behavior": "clear_source_cells",
                "source_row_indices": [0, 1, 2],
                "source_row_mapping": [
                    {"source_row_index": 0, "input_position": 0},
                    {"source_row_index": 1, "input_position": 1},
                    {"source_row_index": 2, "input_position": 2},
                ],
            },
            "workflow_steps": [
                {
                    "step_id": "batch_1",
                    "source_columns": ["resin_smiles"],
                    "method": "xTB",
                    "params": {
                        "skip_ionic_compounds": True,
                        "ionic_filter_behavior": "clear_source_cells",
                    },
                    "valid_row_behavior": {
                        "source_valid_indices": [0, 2],
                        "ionic_excluded_indices": [1],
                        "valid_indices": [0, 2],
                    },
                    "feature_names": ["resin_gap"],
                }
            ],
        },
        ["resin_gap"],
    )

    assert workflow.input_contract["skip_ionic_compounds"] is True
    assert workflow.input_contract["source_row_indices"] == [0, 1, 2]
    assert workflow.steps[0]["params"]["ionic_filter_behavior"] == "clear_source_cells"
    assert workflow.steps[0]["valid_row_behavior"]["ionic_excluded_indices"] == [1]


def test_workflow_to_legacy_config_preserves_single_path_fields():
    workflow = build_workflow_from_training_state(
        {
            "mode": "single_batch",
            "method": "xTB",
            "smiles_col": "resin_smiles",
            "resin_component_cols": ["resin_smiles"],
            "hardener_col": "hardener_smiles",
            "hardener_component_cols": ["hardener_smiles"],
            "hardener_fusion_mode": "拼接SMILES后用于所有分子特征",
            "params": {"xtb_method": "gfn2"},
            "prefix": "resin",
            "valid_row_behavior": {"keep_all_rows": True, "valid_indices": [0, 2]},
        },
        ["resin_xtb_gap"],
    )

    legacy = workflow.to_legacy_config()

    assert legacy["method"] == "xTB"
    assert legacy["smiles_col"] == "resin_smiles"
    assert legacy["resin_component_cols"] == ["resin_smiles"]
    assert legacy["hardener_component_cols"] == ["hardener_smiles"]
    assert legacy["hardener_fusion_mode"] == "拼接SMILES后用于所有分子特征"
    assert legacy["feature_names"] == ["resin_xtb_gap"]


def test_workflow_to_legacy_config_preserves_explicit_single_step_batch_metadata():
    workflow = build_workflow_from_training_state(
        {
            "mode": "multi_batch",
            "selected_source_columns": ["resin_smiles_1"],
            "method": "xTB",
            "prefix": "resin_1",
            "params": {"xtb_method": "gfn2"},
        },
        ["resin_1_xtb_gap"],
    )

    legacy = workflow.to_legacy_config()

    assert legacy["mode"] == "multi_batch"
    assert legacy["workflow_mode"] == "multi_batch"
    assert legacy["batch_mode"] is True
    assert legacy["batch_smiles_cols"] == ["resin_smiles_1"]


def test_resin_component_options_keep_primary_dedicated_column():
    available_columns = [
        "resin_smiles_1",
        "resin_smiles_2",
        "resin_smiles_9",
        "curing_agent_smiles_1",
    ]

    options = get_feature_component_column_options(
        available_columns,
        smiles_candidates=available_columns,
        role="resin",
        primary_column="resin_smiles_1",
        dedicated_columns=[
            "resin_smiles_1",
            "resin_smiles_2",
            "resin_smiles_9",
        ],
    )

    assert options == ["resin_smiles_1", "resin_smiles_2", "resin_smiles_9"]


def test_resolve_resin_component_columns_keeps_all_detected_numbered_columns():
    available_columns = [
        "resin_smiles_1",
        "resin_smiles_2",
        "resin_smiles_9",
    ]

    resolved = resolve_feature_component_columns(
        available_columns,
        role="resin",
        primary_column="resin_smiles_1",
        dedicated_columns=available_columns,
        smiles_candidates=available_columns,
        mode="auto",
    )

    assert resolved == available_columns


def test_find_duplicate_feature_names_preserves_first_duplicate_order():
    assert find_duplicate_feature_names(
        ["resin_x", "hardener_x", "resin_x", "hardener_x", "resin_x"]
    ) == ["resin_x", "hardener_x"]


def test_validate_feature_frame_contract_rejects_partial_or_duplicate_batch_output():
    errors = validate_feature_frame_contract(
        pd.DataFrame([[1, 2]], columns=["feature", "feature"]),
        [0, 0],
        source_row_count=2,
    )

    assert any("duplicate emitted feature names" in error for error in errors)
    assert any("invalid or duplicated" in error for error in errors)


def test_semantic_duplicate_opt_in_preserves_collisions_for_strict_workflow_rejection(
    monkeypatch,
):
    duplicate_frame = lambda values: pd.DataFrame({"semantic_collision": [1] * len(values)})
    monkeypatch.setattr(
        molecular_features,
        "extract_polymer_string_features",
        duplicate_frame,
    )
    monkeypatch.setattr(
        molecular_features,
        "extract_ionic_semantic_features",
        duplicate_frame,
    )
    params = {
        "append_polymer_string_features": True,
        "append_ionic_semantic_features": True,
    }

    default_result = extract_configured_semantic_features(["CC"], params)
    preserved_result = extract_configured_semantic_features(
        ["CC"],
        {**params, "preserve_duplicate_columns": True},
    )
    default_base, default_indices = append_configured_semantic_features(
        pd.DataFrame({"semantic_collision": [0]}),
        [0],
        ["CC"],
        params,
    )
    preserved_base, preserved_indices = append_configured_semantic_features(
        pd.DataFrame({"semantic_collision": [0]}),
        [0],
        ["CC"],
        {**params, "preserve_duplicate_columns": True},
    )

    assert default_result.columns.tolist() == ["semantic_collision"]
    assert preserved_result.columns.tolist() == [
        "semantic_collision",
        "semantic_collision",
    ]
    assert default_indices == [0]
    assert default_base.columns.tolist() == ["semantic_collision"]
    assert preserved_indices == [0]
    assert preserved_base.columns.tolist() == [
        "semantic_collision",
        "semantic_collision",
        "semantic_collision",
    ]
    errors = validate_feature_frame_contract(
        preserved_base,
        preserved_indices,
        source_row_count=1,
    )
    assert any("duplicate emitted feature names" in error for error in errors)


def test_executor_runs_batches_in_merge_order_and_restores_rows(monkeypatch):
    data = pd.DataFrame(
        {
            "resin_smiles_1": ["A", None, "C"],
            "hardener_smiles_1": [None, "B", "D"],
        }
    )
    workflow = MolecularFeatureWorkflow.from_dict(
        {
            "schema_version": 2,
            "mode": "multi_batch",
            "steps": [
                {
                    "step_id": "resin_1",
                    "order": 0,
                    "role": "resin",
                    "source_columns": ["resin_smiles_1"],
                    "method": "test",
                    "prefix": "resin_1",
                    "feature_names": ["resin_1_value"],
                },
                {
                    "step_id": "hardener_1",
                    "order": 1,
                    "role": "hardener",
                    "source_columns": ["hardener_smiles_1"],
                    "method": "test",
                    "prefix": "hardener_1",
                    "feature_names": ["hardener_1_value"],
                },
            ],
            "merge_order": ["resin_1", "hardener_1"],
            "final_feature_names": ["resin_1_value", "hardener_1_value"],
        }
    )
    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step",
        lambda *args, **kwargs: make_test_step_output(*args, **kwargs),
    )

    result = execute_molecular_feature_workflow(data, workflow)

    assert list(result.features.columns) == [
        "resin_1_value",
        "hardener_1_value",
    ]
    assert result.features.index.tolist() == [0, 1, 2]
    assert result.features["resin_1_value"].iloc[[0, 2]].tolist() == ["A", "C"]
    assert result.features["hardener_1_value"].iloc[[1, 2]].tolist() == ["B", "D"]
    assert result.features["resin_1_value"].isna().iloc[1]
    assert result.features["hardener_1_value"].isna().iloc[0]
    assert result.valid_row_indices == {"resin_1": [0, 2], "hardener_1": [1, 2]}
    assert [trace["step_id"] for trace in result.step_trace] == [
        "resin_1",
        "hardener_1",
    ]
    assert [trace["input_count"] for trace in result.step_trace] == [3, 3]
    assert [trace["valid_count"] for trace in result.step_trace] == [2, 2]


def test_executor_rejects_duplicate_feature_columns(monkeypatch):
    data = pd.DataFrame({"smiles": ["A"]})
    workflow = MolecularFeatureWorkflow.from_dict(
        {
            "schema_version": 2,
            "steps": [
                {
                    "step_id": "single",
                    "source_columns": ["smiles"],
                    "method": "test",
                    "prefix": "single",
                    "test_columns": ["value", "value"],
                    "feature_names": ["single_value"],
                }
            ],
            "merge_order": ["single"],
            "final_feature_names": ["single_value"],
        }
    )
    def duplicate_feature_output(smiles, _step, **_kwargs):
        return (
            pd.DataFrame([["A", "A"]], columns=["value", "value"]),
            [0],
            [],
        )

    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step",
        duplicate_feature_output,
    )

    with pytest.raises(ValueError, match="duplicate"):
        execute_molecular_feature_workflow(data, workflow)


def test_executor_resolves_cross_step_duplicates_using_feature_source_map(monkeypatch):
    data = pd.DataFrame({"smiles": ["A", "B"]})
    workflow = MolecularFeatureWorkflow.from_dict(
        {
            "schema_version": 2,
            "steps": [
                {
                    "step_id": "resin_xtb",
                    "source_columns": ["smiles"],
                    "method": "test",
                    "prefix": "resin",
                    "feature_names": ["resin_shared", "resin_xtb_only"],
                },
                {
                    "step_id": "resin_fingerprint",
                    "source_columns": ["smiles"],
                    "method": "test",
                    "prefix": "resin",
                    "feature_names": ["resin_shared", "resin_fp_only"],
                },
            ],
            "merge_order": ["resin_xtb", "resin_fingerprint"],
            "final_feature_names": [
                "resin_shared",
                "resin_xtb_only",
                "resin_fp_only",
            ],
            "feature_source_map": {"resin_shared": "resin_fingerprint"},
        }
    )

    def duplicate_across_steps(smiles, step, **_kwargs):
        values = [1.0, 2.0] if step["step_id"] == "resin_xtb" else [10.0, 20.0]
        return (
            pd.DataFrame(
                {
                    "shared": values,
                    "xtb_only" if step["step_id"] == "resin_xtb" else "fp_only": values,
                }
            ),
            [0, 1],
            [],
        )

    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step",
        duplicate_across_steps,
    )

    result = execute_molecular_feature_workflow(data, workflow)

    assert result.features.columns.tolist() == [
        "resin_shared",
        "resin_xtb_only",
        "resin_fp_only",
    ]
    assert result.features["resin_shared"].tolist() == [10.0, 20.0]
    assert any("resin_shared" in warning for warning in result.warnings)


def test_executor_rejects_missing_required_feature(monkeypatch):
    data = pd.DataFrame({"smiles": ["A"]})
    workflow = MolecularFeatureWorkflow.from_dict(
        {
            "schema_version": 2,
            "steps": [
                {
                    "step_id": "single",
                    "source_columns": ["smiles"],
                    "method": "test",
                    "prefix": "single",
                    "feature_names": ["single_value"],
                }
            ],
            "merge_order": ["single"],
            "final_feature_names": ["single_value"],
        }
    )

    def missing_feature_output(smiles, step, **_kwargs):
        return pd.DataFrame({"other": ["A"]}), [0], []

    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step",
        missing_feature_output,
    )

    with pytest.raises(ValueError, match="missing"):
        execute_molecular_feature_workflow(data, workflow)


def test_executor_rejects_raw_duplicate_workflow_items():
    workflow = MolecularFeatureWorkflow.from_dict(
        {
            "schema_version": 2,
            "steps": [
                {
                    "step_id": "single",
                    "source_columns": ["smiles"],
                    "method": "test",
                    "feature_names": ["single_value", "single_value"],
                }
            ],
            "merge_order": ["single", "single"],
            "final_feature_names": ["single_value", "single_value"],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        execute_molecular_feature_workflow(
            pd.DataFrame({"smiles": ["A"]}),
            workflow,
        )


def test_legacy_full_row_adapter_uses_declared_positions_even_for_all_nan_rows(
    monkeypatch,
):
    def full_row_adapter(_resin, _hardener, _config, device=None):
        return pd.DataFrame({"value": [10.0, float("nan"), float("nan")]}), None

    monkeypatch.setattr(
        "core.virtual_screening.extract_features_from_config",
        full_row_adapter,
    )

    features, valid_indices, warnings = execute_feature_step(
        ["A", None, "C"],
        {"method": "test", "params": {}},
    )

    assert valid_indices == [0, 2]
    assert features["value"].iloc[0] == 10.0
    assert pd.isna(features["value"].iloc[1])
    assert warnings == []


def test_legacy_compact_adapter_with_invalid_index_warns_and_uses_declared_positions(
    monkeypatch,
):
    def compact_adapter(_resin, _hardener, _config, device=None):
        return pd.DataFrame({"value": [10.0]}, index=["not-a-row"]), None

    monkeypatch.setattr(
        "core.virtual_screening.extract_features_from_config",
        compact_adapter,
    )

    features, valid_indices, warnings = execute_feature_step(
        ["A", None, "C"],
        {"method": "test", "params": {}},
    )

    assert valid_indices == [0]
    assert features["value"].tolist() == [10.0]
    assert any("declared non-null source positions" in warning for warning in warnings)


def test_prepare_step_inputs_filters_each_source_cell_before_joining():
    data = pd.DataFrame({"second": ["A"], "first": ["B"]})
    step = {
        "source_columns": ["second", "first"],
        "params": {
            "drop_catalyst_fragments": True,
            "drop_catalyst_only_multi": True,
            "catalyst_remove_invalid": True,
        },
    }

    smiles, valid_indices, metadata = prepare_step_inputs(data, step)

    assert smiles == ["A.B"]
    assert valid_indices == [0]
    assert metadata["source_values"]["second"] == ["A"]
    assert metadata["source_values"]["first"] == ["B"]


def test_execute_feature_step_forwards_parameter_maps(monkeypatch):
    captured = {}

    def forwarding_adapter(_resin, _hardener, config, device=None):
        captured.update(config)
        return pd.DataFrame({"value": [1.0]}), None

    monkeypatch.setattr(
        "core.virtual_screening.extract_features_from_config",
        forwarding_adapter,
    )

    execute_feature_step(
        ["A"],
        {
            "method": "xTB",
            "params": {"xtb_method": "gfn2"},
            "semantic_params": {
                "append_ionic_semantic_features": True,
                "preserve_duplicate_columns": True,
            },
        },
    )

    assert captured["params"] == {
        "xtb_method": "gfn2",
        "append_ionic_semantic_features": True,
        "preserve_duplicate_columns": True,
    }


def test_workflow_round_trip_preserves_step_and_feature_order():
    workflow = MolecularFeatureWorkflow.from_dict(_workflow_payload())

    restored = workflow.to_dict()

    assert restored["merge_order"] == ["resin_1", "hardener_1"]
    assert restored["final_feature_names"] == [
        "resin_1_xtb_gap",
        "hardener_1_xtb_gap",
    ]
    assert [step["step_id"] for step in restored["steps"]] == [
        "resin_1",
        "hardener_1",
    ]


def test_append_workflow_step_preserves_existing_steps_and_updates_order_and_hash():
    payload = _workflow_payload()
    original_steps = copy.deepcopy(payload["steps"])

    appended = append_workflow_step(payload)

    assert payload["steps"] == original_steps
    assert [step["step_id"] for step in appended["steps"]] == [
        "resin_1",
        "hardener_1",
        "step_3",
    ]
    assert [step["order"] for step in appended["steps"]] == [0, 1, 2]
    assert appended["merge_order"] == ["resin_1", "hardener_1", "step_3"]
    assert appended["steps"][-1]["role"] == "neutral"
    assert appended["steps"][-1]["source_columns"] == []
    assert appended["workflow_hash"] == compute_workflow_hash(appended)


def test_append_workflow_step_skips_existing_generated_id_and_keeps_custom_fields():
    payload = _workflow_payload()
    payload["steps"].append({"step_id": "step_3", "order": 2})

    appended = append_workflow_step(
        payload,
        {
            "step_id": "step_3",
            "role": "resin",
            "source_columns": ["resin_smiles_2"],
            "method": "xTB",
            "feature_names": ["resin_2_gap"],
        },
    )

    assert appended["steps"][-1]["step_id"] == "step_4"
    assert appended["steps"][-1]["role"] == "resin"
    assert appended["steps"][-1]["source_columns"] == ["resin_smiles_2"]
    assert appended["steps"][-1]["method"] == "xTB"
    assert appended["steps"][-1]["feature_names"] == ["resin_2_gap"]
    assert appended["merge_order"][-1] == "step_4"


def test_append_training_workflow_preserves_previous_extraction_steps():
    existing = _workflow_payload()

    appended = append_training_workflow(
        existing,
        {
            "mode": "single_batch",
            "selected_source_columns": ["resin_smiles_2"],
            "input_contract": {
                "selected_source_columns": ["resin_smiles_2"],
                "source_row_count": 3,
            },
            "workflow_steps": [
                {
                    "step_id": "single_1",
                    "order": 0,
                    "role": "resin",
                    "source_columns": ["resin_smiles_2"],
                    "method": "MACCS",
                    "prefix": "resin_2",
                    "feature_names": ["resin_2_maccs_1"],
                }
            ],
        },
        ["resin_2_maccs_1"],
    )

    assert [step["step_id"] for step in appended.steps] == [
        "resin_1",
        "hardener_1",
        "single_1",
    ]
    assert appended.merge_order == [
        "resin_1",
        "hardener_1",
        "single_1",
    ]
    assert appended.final_feature_names == [
        "resin_1_xtb_gap",
        "hardener_1_xtb_gap",
        "resin_2_maccs_1",
    ]
    assert appended.steps[-1]["source_columns"] == ["resin_smiles_2"]


def test_workflow_normalization_scales_with_large_row_metadata():
    rows = 4000
    row_mapping = [
        {"source_row_index": index, "input_position": index}
        for index in range(rows)
    ]
    payload = {
        "schema_version": 2,
        "mode": "single_batch",
        "input_contract": {
            "source_row_count": rows,
            "source_row_indices": list(range(rows)),
            "source_row_mapping": row_mapping,
        },
        "steps": [
            {
                "step_id": "single_1",
                "order": 0,
                "source_columns": ["resin_smiles"],
                "method": "MACCS",
                "feature_names": ["resin_0"],
                "valid_row_behavior": {
                    "source_row_count": rows,
                    "source_row_indices": list(range(rows)),
                    "source_row_mapping": row_mapping,
                    "valid_indices": list(range(rows)),
                },
            }
        ],
        "merge_order": ["single_1"],
        "final_feature_names": ["resin_0"],
    }

    started = time.perf_counter()
    normalized = normalize_workflow_config(payload)

    assert normalized["input_contract"]["source_row_indices"] == list(range(rows))
    assert time.perf_counter() - started < 1.0


def test_append_training_workflow_generates_unique_ids_for_repeated_single_steps():
    existing = {
        "schema_version": 2,
        "mode": "single_batch",
        "steps": [
            {
                "step_id": "single_1",
                "order": 0,
                "source_columns": ["resin_smiles_1"],
                "method": "xTB",
                "feature_names": ["resin_1_gap"],
            }
        ],
        "merge_order": ["single_1"],
        "final_feature_names": ["resin_1_gap"],
    }

    appended = append_training_workflow(
        existing,
        {
            "mode": "single_batch",
            "workflow_steps": [
                {
                    "step_id": "single_1",
                    "order": 0,
                    "source_columns": ["hardener_smiles_1"],
                    "method": "MACCS",
                    "feature_names": ["hardener_1_maccs"],
                }
            ],
        },
        ["hardener_1_maccs"],
    )

    assert [step["step_id"] for step in appended.steps] == [
        "single_1",
        "single_2",
    ]
    assert appended.steps[-1]["source_columns"] == ["hardener_smiles_1"]
    assert appended.mode == "multi_batch"


def test_normalization_deduplicates_lists_without_changing_order():
    payload = _workflow_payload()
    payload["merge_order"] = ["hardener_1", "resin_1", "hardener_1"]
    payload["final_feature_names"] = [
        "hardener_1_xtb_gap",
        "resin_1_xtb_gap",
        "hardener_1_xtb_gap",
    ]
    payload["steps"][0]["source_columns"] = ["resin_smiles_1", "resin_smiles_1"]

    normalized = normalize_workflow_config(payload)

    assert normalized["merge_order"] == ["hardener_1", "resin_1"]
    assert normalized["final_feature_names"] == [
        "hardener_1_xtb_gap",
        "resin_1_xtb_gap",
    ]
    assert normalized["steps"][0]["source_columns"] == ["resin_smiles_1"]


def test_workflow_hash_is_stable_for_mapping_order_and_excludes_existing_hash():
    first = _workflow_payload()
    second = copy.deepcopy(first)
    second["steps"][0]["params"] = {"another": 1, "xtb_method": "gfn2"}
    first["steps"][0]["params"] = {"xtb_method": "gfn2", "another": 1}
    second["workflow_hash"] = "stale"

    assert compute_workflow_hash(first) == compute_workflow_hash(second)


def test_hash_matches_serialized_round_trip_and_ignores_unknown_keys():
    payload = _workflow_payload()
    payload["unknown_review_metadata"] = {"ignored": True}

    workflow = MolecularFeatureWorkflow.from_dict(payload)
    restored = workflow.to_dict()

    assert workflow.workflow_hash == compute_workflow_hash(restored)


def test_incomplete_workflow_is_marked_legacy_and_lists_missing_items():
    normalized = normalize_workflow_config(
        {
            "schema_version": 1,
            "steps": [],
        }
    )

    assert normalized["legacy"] is True
    assert normalized["missing_items"] == [
        "mode",
        "input_contract",
        "merge_order",
        "final_feature_names",
        "feature_source_map",
        "derived_feature_steps",
        "random_seeds",
    ]


def test_validation_reports_missing_steps_columns_features_and_order_mismatch():
    payload = _workflow_payload()
    payload["merge_order"] = ["hardener_1", "missing_step", "resin_1"]

    result = validate_workflow_config(
        payload,
        model_feature_cols=["resin_1_xtb_gap", "missing_feature"],
        available_columns=["resin_smiles_1"],
    )

    assert result == {
        "ok": False,
        "missing_steps": ["missing_step"],
        "missing_columns": ["hardener_smiles_1"],
        "missing_features": ["missing_feature"],
        "order_mismatch": [
            {
                "expected": ["resin_1_xtb_gap", "missing_feature"],
                "actual": ["resin_1_xtb_gap", "hardener_1_xtb_gap"],
            }
        ],
        "model_fingerprint_match": None,
    }


def test_validation_reports_unknown_model_fingerprint_match():
    payload = _workflow_payload()

    result = validate_workflow_config(
        payload,
        model_feature_cols=["resin_1_xtb_gap", "hardener_1_xtb_gap"],
    )

    assert result["ok"] is True
    assert result["model_fingerprint_match"] is None


def test_validation_reports_matching_model_fingerprint():
    payload = _workflow_payload()
    payload["expected_model_fingerprint"] = "model-1"

    result = validate_workflow_config(
        payload,
        model_feature_cols=payload["final_feature_names"],
    )

    assert result["ok"] is True
    assert result["model_fingerprint_match"] is True


def test_validation_reports_mismatching_model_fingerprint():
    payload = _workflow_payload()
    payload["expected_model_fingerprint"] = "model-2"

    result = validate_workflow_config(
        payload,
        model_feature_cols=payload["final_feature_names"],
    )

    assert result["ok"] is False
    assert result["model_fingerprint_match"] is False


def test_multicomponent_auto_mode_prefers_dedicated_resin_columns_without_primary_column():
    columns = [
        "resin_smiles",
        "resin_smiles_1",
        "resin_smiles_2",
        "curing_agent_smiles_1",
        "temperature",
    ]

    resolved = resolve_feature_component_columns(
        columns,
        role="resin",
        primary_column="resin_smiles",
        dedicated_columns=["resin_smiles_1", "resin_smiles_2"],
        mode="auto",
    )

    assert resolved == ["resin_smiles_1", "resin_smiles_2"]


def test_multicomponent_manual_options_exclude_primary_hardener_and_metadata_columns():
    columns = [
        "resin_smiles",
        "resin_smiles_1",
        "resin_smiles_2",
        "curing_agent_smiles_1",
        "temperature",
    ]

    options = get_feature_component_column_options(
        columns,
        smiles_candidates=[
            "resin_smiles",
            "resin_smiles_1",
            "resin_smiles_2",
            "curing_agent_smiles_1",
        ],
        role="resin",
        primary_column="resin_smiles",
        dedicated_columns=["resin_smiles_1", "resin_smiles_2"],
    )

    assert options == ["resin_smiles", "resin_smiles_1", "resin_smiles_2"]


def test_multicomponent_manual_selection_filters_invalid_columns_and_duplicates():
    columns = [
        "resin_smiles",
        "resin_smiles_1",
        "resin_smiles_2",
        "curing_agent_smiles_1",
        "temperature",
    ]

    resolved = resolve_feature_component_columns(
        columns,
        smiles_candidates=[
            "resin_smiles",
            "resin_smiles_1",
            "resin_smiles_2",
            "curing_agent_smiles_1",
        ],
        role="resin",
        primary_column="resin_smiles",
        dedicated_columns=["resin_smiles_1", "resin_smiles_2"],
        selected_columns=[
            "resin_smiles_2",
            "curing_agent_smiles_1",
            "temperature",
            "resin_smiles_2",
        ],
        mode="manual",
    )

    assert resolved == ["resin_smiles_2"]


def test_multicomponent_manual_options_include_other_smiles_columns():
    columns = [
        "resin_smiles_1",
        "resin_smiles_2",
        "additive_smiles_1",
        "curing_agent_smiles_1",
        "temperature",
    ]

    options = get_feature_component_column_options(
        columns,
        smiles_candidates=[
            "resin_smiles_1",
            "resin_smiles_2",
            "additive_smiles_1",
            "curing_agent_smiles_1",
        ],
        role="resin",
        primary_column="resin_smiles_1",
        dedicated_columns=["resin_smiles_1", "resin_smiles_2"],
    )

    assert options == [
        "additive_smiles_1",
        "resin_smiles_1",
        "resin_smiles_2",
    ]

    resolved = resolve_feature_component_columns(
        columns,
        role="resin",
        primary_column="resin_smiles_1",
        dedicated_columns=["resin_smiles_1", "resin_smiles_2"],
        selected_columns=["additive_smiles_1"],
        smiles_candidates=[
            "resin_smiles_1",
            "resin_smiles_2",
            "additive_smiles_1",
            "curing_agent_smiles_1",
        ],
        mode="manual",
    )

    assert resolved == ["additive_smiles_1"]


def test_multicomponent_without_dedicated_columns_falls_back_to_primary_column():
    resolved = resolve_feature_component_columns(
        [
            "resin_smiles",
            "resin_smiles_1",
            "resin_smiles_2",
            "curing_agent_smiles",
            "temperature",
        ],
        role="resin",
        primary_column="resin_smiles",
        dedicated_columns=[],
        mode="auto",
    )

    assert resolved == ["resin_smiles"]


def test_hardener_role_keeps_selected_primary_hardener_column_available():
    columns = [
        "curing_agent_smiles",
        "resin_smiles",
        "temperature",
    ]

    options = get_feature_component_column_options(
        columns,
        smiles_candidates=["curing_agent_smiles", "resin_smiles"],
        role="hardener",
        primary_column="curing_agent_smiles",
    )

    assert options == ["curing_agent_smiles"]


def test_explicit_component_role_overrides_column_name_inference():
    assert resolve_feature_component_role(
        "hardener",
        inferred_role="resin",
    ) == "hardener"
    assert resolve_feature_component_role(
        "neutral",
        inferred_role="hardener",
    ) == "neutral"
