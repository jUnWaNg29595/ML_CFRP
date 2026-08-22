import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import core.virtual_screening as virtual_screening
from core.training_runs import TrainingRunManager

from core.molecular_features import (
    OptimizedRDKitFeatureExtractor,
    _safe_fragment_mol_from_text,
    extract_bigsmiles_ensemble_features,
    extract_configured_semantic_features,
    extract_ionic_semantic_features,
    extract_polymer_string_features,
)
from core.virtual_screening import (
    DEFAULT_EPOXY_RULES,
    _calc_rule_features,
    apply_saved_process_pls,
    apply_feature_overrides,
    build_component_library,
    build_feature_matrix,
    enumerate_formulation_candidates,
    extract_features_from_config,
    filter_formulation_candidates_by_component_limits,
    filter_candidates_by_epoxy_rules,
    generate_candidate_pool,
    generate_virtual_component_library,
    get_valid_feature_row_mask,
    infer_primary_component_role,
    limit_unique_candidates_for_expensive_features,
    predict_with_model,
    resolve_component_smiles_cols,
    resolve_molecular_feature_config,
    resolve_molecular_feature_workflow,
    resolve_workflow_source_columns_by_role,
    legacy_config_to_workflow,
    materialize_workflow_source_columns,
    rebalance_screening_weights,
    iter_pair_indices,
    sample_pair_indices,
)


def test_virtual_screening_page_exposes_design_engine_only():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    source = source[source.rfind("def page_virtual_screening") :]
    assert "分子设计引擎" in source
    assert "叠加 PubChem 候选" not in source
    assert "虚拟完整分子上限" not in source


def test_exact_replay_rejects_missing_required_feature_without_fingerprint_fill():
    with pytest.raises(ValueError, match="missing"):
        build_feature_matrix(
            ["resin_1_x", "hardener_1_x"],
            pd.DataFrame({"resin_1_x": [1.0]}),
            strict=True,
        )


def test_screening_reuses_saved_process_pls_without_refit():
    from core.process_pls import ProcessPLSTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame(
        {
            "temperature": [1.0, 2.0, 3.0, 4.0],
            "time": [10.0, 11.0, 12.0, 13.0],
            "other": [5.0, 6.0, 7.0, 8.0],
        }
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pipeline = Pipeline(
        [
            (
                "process_pls",
                ProcessPLSTransformer(
                    process_feature_cols=["temperature", "time"],
                    max_components=1,
                    random_state=42,
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(train, y)
    statistics_before = pipeline.named_steps["process_pls"].imputer_.statistics_.copy()

    candidate = pd.DataFrame(
        {"temperature": [1000.0], "time": [999.0], "other": [8.5]}
    )
    validated = apply_saved_process_pls(pipeline, candidate)
    prediction = pipeline.predict(validated)

    assert prediction.shape == (1,)
    assert validated.columns.tolist() == ["temperature", "time", "other"]
    np.testing.assert_array_equal(
        pipeline.named_steps["process_pls"].imputer_.statistics_,
        statistics_before,
    )


def test_screening_reports_missing_raw_process_columns():
    from core.process_pls import ProcessPLSTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame({"temperature": [1.0, 2.0, 3.0], "time": [4.0, 5.0, 6.0]})
    pipeline = Pipeline(
        [
            (
                "process_pls",
                ProcessPLSTransformer(
                    process_feature_cols=["temperature", "time"],
                    max_components=1,
                    random_state=42,
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(train, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="高通量筛选缺少工艺 PLS 原始输入列"):
        apply_saved_process_pls(pipeline, pd.DataFrame({"time": [1.0]}))


def test_saved_process_pls_pipeline_exposes_raw_input_feature_names():
    from core.process_pls import ProcessPLSTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame(
        {
            "temperature": [1.0, 2.0, 3.0, 4.0],
            "time": [10.0, 11.0, 12.0, 13.0],
            "other": [5.0, 6.0, 7.0, 8.0],
        }
    )
    pipeline = Pipeline(
        [
            (
                "process_pls",
                ProcessPLSTransformer(
                    process_feature_cols=["temperature", "time"],
                    max_components=1,
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(train, np.array([1.0, 2.0, 3.0, 4.0]))

    assert pipeline.feature_names_in_.tolist() == ["temperature", "time", "other"]


def test_feature_row_mask_excludes_nan_inf_and_missing_required_features():
    features = pd.DataFrame(
        {
            "resin_a": [1.0, np.nan, 3.0],
            "resin_b": [2.0, 4.0, np.inf],
        }
    )

    mask = get_valid_feature_row_mask(
        features,
        ["resin_a", "resin_b"],
    )

    assert mask.tolist() == [True, False, False]
    missing_mask = get_valid_feature_row_mask(
        features,
        ["resin_a", "missing_feature"],
    )
    assert missing_mask.tolist() == [False, False, False]


from core.smiles_utils import parse_chemical_string


def test_artifact_round_trip_preserves_workflow_hash_and_feature_order():
    artifact = {
        "feature_cols": ["resin_1_x", "hardener_1_x"],
        "extra": {
            "molecular_feature_workflow": {
                "schema_version": 2,
                "merge_order": ["resin_1", "hardener_1"],
                "final_feature_names": ["resin_1_x", "hardener_1_x"],
            }
        },
    }
    workflow = resolve_molecular_feature_workflow(artifact)
    assert workflow is not None
    assert workflow.final_feature_names == ["resin_1_x", "hardener_1_x"]


def test_workflow_only_payload_can_be_projected_to_legacy_extractor_config():
    workflow = resolve_molecular_feature_workflow(
        {
            "molecular_feature_workflow": {
                "schema_version": 2,
                "mode": "single_batch",
                "steps": [
                    {
                        "step_id": "single",
                        "source_columns": ["resin_smiles"],
                        "method": "xTB",
                        "feature_names": ["resin_gap"],
                    }
                ],
                "merge_order": ["single"],
                "final_feature_names": ["resin_gap"],
            }
        }
    )

    assert workflow is not None
    legacy = workflow.to_legacy_config()
    assert legacy["method"] == "xTB"
    assert legacy["smiles_col"] == "resin_smiles"
    assert legacy["feature_names"] == ["resin_gap"]


def test_legacy_single_config_is_marked_partial_when_batch_fields_are_absent():
    workflow = legacy_config_to_workflow(
        {"method": "xTB", "feature_names": ["resin_xtb_gap"]},
        model_feature_cols=["resin_xtb_gap"],
    )
    assert workflow.legacy is True
    assert workflow.missing_items


def test_legacy_string_component_columns_remain_single_columns_and_keep_hardener_source():
    workflow = legacy_config_to_workflow(
        {
            "method": "xTB",
            "smiles_col": "resin_smiles",
            "resin_component_cols": "resin_smiles",
            "hardener_col": "hardener_smiles",
            "hardener_component_cols": "hardener_smiles",
            "feature_names": ["resin_gap"],
        }
    )

    assert workflow.input_contract["resin_component_cols"] == ["resin_smiles"]
    assert workflow.input_contract["hardener_component_cols"] == ["hardener_smiles"]
    assert workflow.input_contract["hardener_col"] == "hardener_smiles"
    assert workflow.steps[0]["source_columns"] == [
        "resin_smiles",
        "hardener_smiles",
    ]


def test_training_run_model_artifact_receives_molecular_feature_metadata(tmp_path, monkeypatch):
    captured = {}

    def fake_create_model_artifact_bytes(**kwargs):
        captured.update(kwargs)
        return b"artifact"

    monkeypatch.setattr(
        "core.model_io.create_model_artifact_bytes",
        fake_create_model_artifact_bytes,
    )

    extra = {
        "molecular_feature_workflow": {"schema_version": 2},
        "molecular_feature_config": {"method": "xTB"},
        "molecular_feature_trace": [{"step_id": "single"}],
    }
    manager = TrainingRunManager(base_dir=str(tmp_path))
    summary = manager.save_run(
        model_name="demo",
        metadata={"r2": 0.5},
        model=object(),
        feature_cols=["resin_gap"],
        target_col="target",
        extra=extra,
    )

    assert captured["extra"] == extra
    assert (tmp_path / summary.run_id / "model.pkl").read_bytes() == b"artifact"


def test_app_metadata_restore_prefers_extra_and_clears_missing_fields():
    import app

    workflow_from_extra = {
        "schema_version": 2,
        "mode": "single_batch",
        "steps": [
            {
                "step_id": "extra_step",
                "source_columns": ["resin_smiles"],
                "method": "xTB",
                "feature_names": ["extra_gap"],
            }
        ],
        "merge_order": ["extra_step"],
        "final_feature_names": ["extra_gap"],
    }
    workflow_from_top_level = {
        **workflow_from_extra,
        "steps": [{**workflow_from_extra["steps"][0], "step_id": "top_step"}],
        "merge_order": ["top_step"],
        "final_feature_names": ["top_gap"],
    }

    app._restore_versioned_molecular_feature_metadata(
        {
            "molecular_feature_workflow": workflow_from_top_level,
            "molecular_feature_trace": [{"step_id": "top"}],
            "extra": {
                "molecular_feature_workflow": workflow_from_extra,
                "molecular_feature_trace": [{"step_id": "extra"}],
            },
        }
    )
    assert app.st.session_state["molecular_feature_workflow"] == workflow_from_extra
    assert app.st.session_state["molecular_feature_trace"] == [{"step_id": "extra"}]

    app._restore_versioned_molecular_feature_metadata(
        {"extra": {"molecular_feature_config": {"method": "xTB"}}}
    )
    assert app.st.session_state["molecular_feature_workflow"] is None
    assert app.st.session_state["molecular_feature_trace"] == []


def test_app_workflow_only_restore_projects_legacy_config_into_session():
    import app

    workflow_payload = {
        "schema_version": 2,
        "mode": "single_batch",
        "steps": [
            {
                "step_id": "single",
                "source_columns": ["resin_smiles"],
                "method": "xTB",
                "prefix": "resin",
                "feature_names": ["resin_gap"],
            }
        ],
        "merge_order": ["single"],
        "final_feature_names": ["resin_gap"],
    }

    workflow, config = app._restore_molecular_feature_metadata(
        {"molecular_feature_workflow": workflow_payload},
        model_feature_cols=["resin_gap"],
    )

    assert workflow is not None
    assert config["method"] == "xTB"
    assert config["smiles_col"] == "resin_smiles"
    assert config["feature_names"] == ["resin_gap"]
    assert app.st.session_state["molecular_feature_workflow"]["workflow_hash"]
    assert app.st.session_state["molecular_feature_config"] == config


def test_app_imported_molecular_workflow_replays_and_replaces_columns(monkeypatch):
    import app

    data = pd.DataFrame(
        {
            "resin_smiles": ["A", None, "C"],
            "resin_value": ["stale", "stale", "stale"],
            "target": [1.0, 2.0, 3.0],
        }
    )
    workflow_payload = {
        "molecular_feature_workflow": {
            "schema_version": 2,
            "mode": "single_batch",
            "steps": [
                {
                    "step_id": "single",
                    "source_columns": ["resin_smiles"],
                    "method": "test",
                    "prefix": "resin",
                    "feature_names": ["resin_value"],
                }
            ],
            "merge_order": ["single"],
            "final_feature_names": ["resin_value"],
        }
    }

    def fake_step(smiles, step, **_kwargs):
        valid_indices = [
            index for index, value in enumerate(smiles) if value is not None
        ]
        return (
            pd.DataFrame(
                {"value": [str(smiles[index]).lower() for index in valid_indices]},
                index=valid_indices,
            ),
            valid_indices,
            [],
        )

    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step",
        fake_step,
    )

    imported = app._run_imported_molecular_feature_workflow(
        data,
        workflow_payload,
    )

    assert imported["feature_names"] == ["resin_value"]
    assert imported["features"].index.tolist() == [0, 1, 2]
    assert imported["data"].columns.tolist().count("resin_value") == 1
    assert imported["data"]["resin_value"].iloc[0] == "a"
    assert pd.isna(imported["data"]["resin_value"].iloc[1])
    assert imported["data"]["resin_value"].iloc[2] == "c"
    assert imported["workflow"]["workflow_hash"]
    assert imported["trace"][0]["mode"] == "training_import"


def test_artifact_extra_contains_post_feature_mapping_default(monkeypatch):
    import app

    app.st.session_state["post_feature_mapping_default"] = {
        "schema_version": 1,
        "model_feature_cols": ["temperature"],
        "rules": {
            "temperature": {
                "source_type": "candidate",
                "source_column": "temperature",
                "confirmed": True,
            }
        },
        "mapping_hash": "mapping-hash",
    }
    extra = app._current_molecular_feature_artifact_extra()
    assert extra["post_feature_mapping_default"]["mapping_hash"] == "mapping-hash"


def test_restore_model_metadata_loads_mapping_as_unconfirmed_draft():
    import app

    app._restore_molecular_feature_metadata(
        {
            "extra": {
                "post_feature_mapping_default": {
                    "schema_version": 1,
                    "model_feature_cols": ["EEW"],
                    "rules": {
                        "EEW": {
                            "source_type": "computed",
                            "source_column": "computed_resin_eew",
                            "confirmed": True,
                        }
                    },
                }
            }
        },
        model_feature_cols=["EEW"],
    )
    assert app.st.session_state["post_feature_mapping_draft"]["rules"]["EEW"]["source_type"] == "computed"
    assert app.st.session_state["post_feature_mapping_confirmation"] is False


def test_post_feature_mapping_invalidation_keeps_molecular_metadata():
    import app

    app.st.session_state["molecular_feature_workflow"] = {"workflow": "keep"}
    app.st.session_state["molecular_feature_trace"] = [{"step": "keep"}]
    app.st.session_state["post_feature_mapping_draft"] = {
        "model_feature_cols": ["temperature"],
        "catalog_fingerprint": "catalog-a",
    }
    app.st.session_state["post_feature_mapping_model_fingerprint"] = "model-a"
    app.st.session_state["post_feature_mapping_catalog_fingerprint"] = "catalog-a"
    app.st.session_state["post_feature_mapping_confirmation"] = True

    app._invalidate_post_feature_mapping_if_changed(
        ["pressure"],
        "catalog-b",
    )

    assert app.st.session_state["molecular_feature_workflow"] == {"workflow": "keep"}
    assert app.st.session_state["molecular_feature_trace"] == [{"step": "keep"}]
    assert app.st.session_state["post_feature_mapping_draft"]["invalid"] is True
    assert app.st.session_state["post_feature_mapping_confirmation"] is False


def test_app_v1_snapshot_restore_clears_new_workflow_metadata(monkeypatch):
    import app

    app.st.session_state["molecular_feature_workflow"] = {"stale": True}
    app.st.session_state["molecular_feature_trace"] = [{"stale": True}]
    monkeypatch.setattr(
        app,
        "_load_snapshot_meta",
        lambda tag="latest": {
            "version": 1,
            "saved_at": "2026-07-29T00:00:00",
            "df_keys": [],
        },
    )
    monkeypatch.setattr(
        app,
        "_snapshot_paths",
        lambda tag="latest": ("unused.json", {}),
    )

    restored, reason = app._restore_session_snapshot(override=True)

    assert restored is True
    assert reason == "ok"
    assert app.st.session_state["molecular_feature_workflow"] is None
    assert app.st.session_state["molecular_feature_trace"] == []


def test_snapshot_restore_does_not_restore_stringified_optimization_result(monkeypatch):
    import app

    app.st.session_state["optimization_result"] = "OptimizationResult(...)"
    monkeypatch.setattr(
        app,
        "_load_snapshot_meta",
        lambda tag="latest": {
            "version": 2,
            "saved_at": "2026-07-29T00:00:00",
            "df_keys": [],
            "optimization_result": "OptimizationResult(...)",
            "molecular_feature_workflow": None,
            "molecular_feature_trace": [],
        },
    )
    monkeypatch.setattr(
        app,
        "_snapshot_paths",
        lambda tag="latest": ("unused.json", {}),
    )

    restored, reason = app._restore_session_snapshot(override=True)

    assert restored is True
    assert reason == "ok"
    assert app.st.session_state["optimization_result"] is None


class VirtualScreeningRuleTests(unittest.TestCase):
    def test_ordinary_oxirane_is_counted(self):
        features = _calc_rule_features(
            "C1CO1",
            DEFAULT_EPOXY_RULES["global"]["allowed_elements"],
        )
        self.assertTrue(features["valid"])
        self.assertEqual(features["epoxide"], 1)

    def test_basic_rules_keep_small_aliphatic_epoxy(self):
        candidates = pd.DataFrame(
            {
                "resin_smiles": ["C1CO1", "COCC1CO1", "c1ccccc1"],
                "hardener_smiles": ["NCCN", "NCCN", "NCCN"],
            }
        )
        filtered = filter_candidates_by_epoxy_rules(candidates)
        self.assertEqual(filtered["resin_smiles"].tolist(), ["C1CO1", "COCC1CO1"])

    def test_hardener_only_rules_do_not_require_epoxide(self):
        candidates = pd.DataFrame({"resin_smiles": ["NCCN", "O=C1OC(=O)CC1", "C1CO1"]})
        filtered = filter_candidates_by_epoxy_rules(
            candidates,
            resin_col=None,
            hardener_col="resin_smiles",
            rules={
                "hardener": {
                    "allowed_classes": ["amine", "anhydride"],
                    "ban_epoxide": True,
                }
            },
        )
        self.assertEqual(filtered["resin_smiles"].tolist(), ["NCCN", "O=C1OC(=O)CC1"])


class VirtualScreeningFeatureContractTests(unittest.TestCase):
    def test_incompatible_molecular_feature_config_is_rejected(self):
        validator = getattr(virtual_screening, "validate_molecular_feature_contract", None)
        self.assertIsNotNone(validator)
        with self.assertRaisesRegex(ValueError, "特征契约不一致"):
            validator(
                ["xtb_gap", "xtb_homo"],
                {
                    "method": "分子指纹",
                    "feature_names": ["Resin_MACCS_1", "Resin_MACCS_2"],
                },
                extracted_feature_cols=["Resin_MACCS_1", "Resin_MACCS_2"],
            )

    def test_matching_molecular_feature_config_is_accepted(self):
        result = virtual_screening.validate_molecular_feature_contract(
            ["temperature", "xtb_gap", "xtb_homo"],
            {
                "method": "xTB",
                "feature_names": ["xtb_gap", "xtb_homo", "xtb_lumo"],
            },
            extracted_feature_cols=["xtb_gap", "xtb_homo", "xtb_lumo"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["overlap"], ["xtb_gap", "xtb_homo"])

    def test_imported_artifact_config_overrides_stale_session_method(self):
        config = resolve_molecular_feature_config(
            {
                "extra": {
                    "molecular_feature_config": {
                        "method": "xTB",
                        "params": {"xtb_method": "gfn2"},
                    }
                }
            }
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["method"], "xTB")

    def test_feature_process_wrapper_is_unwrapped(self):
        config = resolve_molecular_feature_config(
            {
                "molecular_feature_config": {
                    "method": "xTB",
                    "feature_names": ["xtb_gap"],
                }
            }
        )

        self.assertEqual(config["method"], "xTB")
        self.assertEqual(config["feature_names"], ["xtb_gap"])

    def test_bigsmiles_is_converted_before_rdkit_screening(self):
        molecule = parse_chemical_string("{[<]CCO[>]}")
        self.assertIsNotNone(molecule)
        self.assertGreaterEqual(molecule.GetNumAtoms(), 1)

    def test_feature_matrix_coerces_string_values_before_prediction(self):
        feature_matrix = build_feature_matrix(
            ["feature_a", "feature_b"],
            pd.DataFrame({"feature_a": ["1.25"], "feature_b": ["bad"]}),
        )

        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in feature_matrix.dtypes))
        self.assertEqual(feature_matrix.loc[0, "feature_a"], 1.25)
        self.assertTrue(np.isnan(feature_matrix.loc[0, "feature_b"]))

        class InfCheckingModel:
            def predict(self, values):
                np.isinf(values)
                return np.asarray([0.0])

        prediction = predict_with_model(InfCheckingModel(), feature_matrix)
        self.assertEqual(prediction.tolist(), [0.0])

    def test_pipeline_receives_numeric_dataframe_with_original_columns(self):
        class RecordingPipeline:
            def __init__(self):
                self.seen = None

            def predict(self, values):
                self.seen = values
                return np.asarray([float(values["feature_a"].iloc[0])])

        pipeline = RecordingPipeline()
        feature_matrix = pd.DataFrame(
            {"feature_a": ["2.5"], "feature_b": ["3.0"]},
        )

        prediction = predict_with_model(
            model=None,
            X=feature_matrix,
            pipeline=pipeline,
        )

        self.assertEqual(prediction.tolist(), [2.5])
        self.assertIsInstance(pipeline.seen, pd.DataFrame)
        self.assertEqual(pipeline.seen.columns.tolist(), ["feature_a", "feature_b"])
        self.assertTrue(np.issubdtype(pipeline.seen.dtypes["feature_a"], np.floating))

    def test_imputer_and_scaler_receive_numeric_arrays_in_order(self):
        class RecordingTransform:
            def __init__(self, offset):
                self.offset = offset
                self.seen = None

            def transform(self, values):
                self.seen = np.asarray(values)
                return self.seen + self.offset

        class RecordingModel:
            def __init__(self):
                self.seen = None

            def predict(self, values):
                self.seen = np.asarray(values)
                return np.asarray([self.seen[0, 0]])

        imputer = RecordingTransform(1.0)
        scaler = RecordingTransform(10.0)
        model = RecordingModel()
        feature_matrix = pd.DataFrame(
            {"feature_a": ["4.0"], "feature_b": ["5.0"]},
        )

        prediction = predict_with_model(
            model=model,
            X=feature_matrix,
            imputer=imputer,
            scaler=scaler,
        )

        self.assertEqual(prediction.tolist(), [15.0])
        np.testing.assert_array_equal(imputer.seen, np.asarray([[4.0, 5.0]]))
        np.testing.assert_array_equal(scaler.seen, np.asarray([[5.0, 6.0]]))
        np.testing.assert_array_equal(model.seen, np.asarray([[15.0, 16.0]]))

    def test_fast_rdkit_extractor_converts_bigsmiles_before_direct_parse(self):
        extractor = OptimizedRDKitFeatureExtractor(n_jobs=1, backend="single")

        features, valid_indices = extractor.smiles_to_rdkit_features(["{[<]CCO[>]}", "CCN"])

        self.assertEqual(valid_indices, [0, 1])
        self.assertFalse(features.empty)

    def test_saved_rdkit_feature_config_accepts_bigsmiles_candidates(self):
        features, error = extract_features_from_config(
            ["{[<]CCO[>]}", "CCN"],
            None,
            {
                "method": "RDKit标准版",
                "feature_names": [],
                "params": {},
            },
        )

        self.assertIsNone(error)
        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 0)

    def test_ionic_fragment_parser_accepts_bigsmiles_and_ion_aliases(self):
        molecule = _safe_fragment_mol_from_text("{[<]CCO[>]}[Cl-]")

        self.assertIsNotNone(molecule)
        self.assertGreaterEqual(molecule.GetNumAtoms(), 1)

    def test_configured_semantic_assembler_reproduces_all_feature_families(self):
        values = ["{[<]CCO[>]}[Cl-]", "NCCN"]
        params = {
            "append_polymer_string_features": True,
            "append_polymer_semantic_features": True,
            "append_ionic_semantic_features": True,
            "bigsmiles_semantic_num_samples": 3,
            "bigsmiles_semantic_min_repeat_units": 1,
            "bigsmiles_semantic_max_repeat_units": 2,
            "bigsmiles_semantic_random_state": 17,
        }
        result = extract_configured_semantic_features(values, params)
        expected = pd.concat(
            [
                extract_polymer_string_features(values),
                extract_bigsmiles_ensemble_features(
                    values,
                    n_samples=3,
                    min_repeat_units=1,
                    max_repeat_units=2,
                    random_state=17,
                ),
                extract_ionic_semantic_features(values),
            ],
            axis=1,
        ).loc[:, lambda df: ~df.columns.duplicated()]
        self.assertEqual(result.columns.tolist(), expected.columns.tolist())
        self.assertEqual(result.shape, expected.shape)

    def test_missing_non_fingerprint_feature_is_a_contract_error(self):
        _, error = extract_features_from_config(
            ["CCO"],
            None,
            {
                "method": "RDKit标准版",
                "feature_names": ["feature_that_does_not_exist"],
                "params": {},
            },
        )
        self.assertIsNotNone(error)
        self.assertIn("could not be reproduced", error)

    def test_missing_fingerprint_bit_is_safely_zero_filled(self):
        features, error = extract_features_from_config(
            ["CCO"],
            None,
            {
                "method": "分子指纹",
                "feature_names": ["Resin_MACCS_999"],
                "params": {"fp_type": "MACCS"},
            },
        )
        self.assertIsNone(error)
        self.assertEqual(features.columns.tolist(), ["Resin_MACCS_999"])
        self.assertEqual(features.iloc[0, 0], 0)

    def test_fingerprint_flow_does_not_hide_missing_semantic_feature(self):
        _, error = extract_features_from_config(
            ["CCO"],
            None,
            {
                "method": "分子指纹",
                "feature_names": ["Resin_MACCS_999", "ionic_missing"],
                "params": {"fp_type": "MACCS"},
            },
        )
        self.assertIsNotNone(error)
        self.assertIn("ionic_missing", error)


class VirtualScreeningRoleTests(unittest.TestCase):
    def test_primary_role_inference(self):
        self.assertEqual(
            infer_primary_component_role(
                {"resin_component_cols": [f"curing_agent_smiles_{i}" for i in range(4)] + ["resin_note"]}
            ),
            "hardener",
        )
        self.assertEqual(
            infer_primary_component_role({"resin_component_cols": ["epoxy_resin_smiles"]}),
            "resin",
        )
        self.assertEqual(
            infer_primary_component_role({"resin_component_cols": ["component_smiles"]}),
            "neutral",
        )

    def test_numbered_role_columns_are_recovered_independently(self):
        columns = [
            "resin_smiles_1",
            "resin_smiles_2",
            "curing_agent_smiles_1",
            "curing_agent_smiles_3",
        ]
        cfg = {
            "smiles_col": "resin_smiles",
            "resin_component_cols": [
                "curing_agent_smiles_1",
                "curing_agent_smiles_3",
            ],
        }
        self.assertEqual(
            resolve_component_smiles_cols(cfg, "resin", columns),
            ["resin_smiles_1", "resin_smiles_2"],
        )
        self.assertEqual(
            resolve_component_smiles_cols(cfg, "hardener", columns),
            ["curing_agent_smiles_1", "curing_agent_smiles_3"],
        )

    def test_explicit_hardener_columns_are_not_hidden_by_resin_primary_role(self):
        cfg = {
            "resin_component_cols": ["resin_smiles_1", "resin_smiles_2"],
            "hardener_component_cols": ["curing_agent_smiles_1"],
            "primary_component_role": "hardener",
        }
        self.assertEqual(
            resolve_component_smiles_cols(cfg, "resin"),
            ["resin_smiles_1", "resin_smiles_2"],
        )
        self.assertEqual(
            resolve_component_smiles_cols(cfg, "hardener"),
            ["curing_agent_smiles_1"],
        )

    def test_legacy_resin_field_with_curing_columns_is_split_by_chemical_role(self):
        resin_columns, hardener_columns = resolve_workflow_source_columns_by_role(
            None,
            {
                "smiles_col": "resin_smiles",
                "resin_component_cols": [
                    "curing_agent_smiles_1",
                    "curing_agent_smiles_2",
                ],
            },
        )

        self.assertEqual(resin_columns, ["resin_smiles"])
        self.assertEqual(
            hardener_columns,
            ["curing_agent_smiles_1", "curing_agent_smiles_2"],
        )

    def test_workflow_source_roles_merge_explicit_steps_without_cross_contamination(self):
        resin_columns, hardener_columns = resolve_workflow_source_columns_by_role(
            {
                "steps": [
                    {
                        "role": "resin",
                        "source_columns": ["resin_smiles_1"],
                    },
                    {
                        "role": "hardener",
                        "source_columns": ["curing_agent_smiles_1"],
                    },
                ],
                "input_contract": {
                    "resin_component_cols": [
                        "resin_smiles_1",
                        "curing_agent_smiles_1",
                    ],
                },
            },
            {},
        )

        self.assertEqual(resin_columns, ["resin_smiles_1"])
        self.assertEqual(hardener_columns, ["curing_agent_smiles_1"])

    def test_materialize_uses_top_level_component_parser_for_bigsmiles(self):
        candidates = pd.DataFrame(
            {
                "resin_smiles": ["CC.O", "{[<]CCO[>]}.CC"],
                "hardener_smiles": ["NCCN.CN", "{[<]CCO[>]}"],
            }
        )

        materialized = materialize_workflow_source_columns(
            candidates,
            resin_columns=["resin_smiles_1", "resin_smiles_2"],
            hardener_columns=[
                "curing_agent_smiles_1",
                "curing_agent_smiles_2",
            ],
        )

        self.assertEqual(
            materialized["resin_smiles_1"].tolist(),
            ["CC", "{[<]CCO[>]}"],
        )
        self.assertEqual(materialized["resin_smiles_2"].tolist(), ["O", "CC"])
        self.assertEqual(
            materialized["curing_agent_smiles_1"].tolist(),
            ["NCCN", "{[<]CCO[>]}"],
        )
        self.assertEqual(
            materialized["curing_agent_smiles_2"].tolist(),
            ["CN", None],
        )

        existing = candidates.assign(resin_smiles_1=["existing", None])
        preserved = materialize_workflow_source_columns(
            existing,
            resin_columns=["resin_smiles_1", "resin_smiles_2"],
        )
        self.assertEqual(preserved["resin_smiles_1"].tolist(), ["existing", None])


class VirtualScreeningSamplingTests(unittest.TestCase):
    def test_single_resin_mode_rejects_precomposed_resin_smiles(self):
        resin = build_component_library(
            ["C1CO1.CCO", "COCC1CO1"],
            role="resin",
            source="test",
        )
        hardener = build_component_library(
            ["NCCN"],
            role="hardener",
            source="test",
        )

        design = enumerate_formulation_candidates(
            resin,
            hardener,
            max_pairs=10,
            max_formulations=10,
            max_resin_components=1,
        )

        assert design.candidate_df["resin_smiles"].tolist() == ["COCC1CO1"]

    def test_component_limit_filter_removes_multicomponent_observed_rows(self):
        candidates = pd.DataFrame(
            {
                "resin_smiles": ["C1CO1.CCO", "COCC1CO1"],
                "hardener_smiles": ["NCCN", "NCCCN"],
            }
        )

        filtered = filter_formulation_candidates_by_component_limits(
            candidates,
            max_resin_components=1,
            max_hardener_components=1,
        )

        assert filtered["resin_smiles"].tolist() == ["COCC1CO1"]

    def test_observed_feature_override_does_not_erase_generated_rows(self):
        feature_matrix = pd.DataFrame({"feature_a": [1.0, 2.0]})
        candidates = pd.DataFrame({"feature_a": [9.0, float("nan")]})
        result = apply_feature_overrides(feature_matrix, candidates)
        self.assertEqual(result["feature_a"].tolist(), [9.0, 2.0])

    def test_virtual_resins_are_connected_epoxy_molecules(self):
        library = generate_virtual_component_library(
            role="resin",
            n_samples=100,
            random_state=7,
        )
        self.assertGreater(len(library), 10)
        for smiles in library["smiles"]:
            self.assertNotIn(".", smiles)
            features = _calc_rule_features(
                smiles,
                DEFAULT_EPOXY_RULES["global"]["allowed_elements"],
            )
            self.assertTrue(features["valid"])
            self.assertGreater(features["epoxide"], 0)

    def test_virtual_hardeners_are_connected_valid_molecules(self):
        library = generate_virtual_component_library(
            role="hardener",
            n_samples=100,
            random_state=7,
        )
        self.assertGreater(len(library), 10)
        for smiles in library["smiles"]:
            self.assertNotIn(".", smiles)
            features = _calc_rule_features(
                smiles,
                DEFAULT_EPOXY_RULES["global"]["allowed_elements"],
            )
            self.assertTrue(features["valid"])

    def test_random_pair_sampling_has_no_duplicates(self):
        pool = generate_candidate_pool(
            ["C1CO1", "COCC1CO1", "CCCCOCC1CO1"],
            ["NCCN", "NCCCN"],
            mode="random",
            max_candidates=6,
            random_state=7,
        )
        self.assertEqual(len(pool.df), 6)
        self.assertEqual(pool.df["combo_smiles"].nunique(), 6)

    def test_process_sampling_covers_every_pair_before_repeating(self):
        resin = build_component_library(
            ["C1CO1", "COCC1CO1", "CCCCOCC1CO1"],
            role="resin",
            source="test",
        )
        hardener = build_component_library(
            ["NCCN", "NCCCN"],
            role="hardener",
            source="test",
        )
        design = enumerate_formulation_candidates(
            resin,
            hardener,
            max_pairs=6,
            feature_grid={"temperature": [80, 120, 160, 200]},
            max_formulations=12,
            random_state=7,
        )
        result = design.candidate_df
        self.assertEqual(len(result), 12)
        self.assertEqual(result["combo_smiles"].nunique(), 6)
        self.assertEqual(result.duplicated(["combo_smiles", "temperature"]).sum(), 0)

    def test_expensive_feature_limit_preserves_anchors_and_sources(self):
        candidates = pd.DataFrame(
            {
                "_molecule_key": [f"k{i}" for i in range(20)],
                "resin_source": ["train_data"] * 5 + ["pubchem"] * 10 + ["guided_generated"] * 5,
                "candidate_origin": ["train_observed_pair", ""] + [""] * 18,
            }
        )
        limited, metadata = limit_unique_candidates_for_expensive_features(
            candidates,
            max_unique=6,
            random_state=7,
        )
        self.assertEqual(limited["_molecule_key"].nunique(), 6)
        self.assertIn("k0", limited["_molecule_key"].tolist())
        self.assertGreaterEqual(limited["resin_source"].nunique(), 2)
        self.assertEqual(metadata["before_unique"], 20)
        self.assertEqual(metadata["after_unique"], 6)


def test_enumeration_callback_can_pause_after_one_batch():
    resin = build_component_library(
        ["C1CO1", "COCC1CO1", "CCCCOCC1CO1"],
        role="resin",
        source="test",
    )
    hardener = build_component_library(
        ["NCCN", "NCCCN"],
        role="hardener",
        source="test",
    )
    calls = []

    def pause_after_first_batch(batch_idx, total_batches, batch_count, total_count):
        calls.append((batch_idx, total_batches, batch_count, total_count))
        return False

    design = enumerate_formulation_candidates(
        resin,
        hardener,
        max_pairs=6,
        feature_grid={"temperature": [80, 120]},
        max_formulations=12,
        random_state=7,
        batch_callback=pause_after_first_batch,
    )

    assert len(calls) == 1
    assert calls[0][0] == 0
    assert calls[0][2] > 0
    assert design.metadata["paused"] is True
    assert len(design.candidate_df) == calls[0][3]


def test_enumeration_stops_pair_generation_at_formulation_limit():
    resin = build_component_library(
        [f"C1CO1CC{'C' * index}" for index in range(1, 8)],
        role="resin",
        source="test",
    )
    hardener = build_component_library(
        [f"NCC{'C' * index}N" for index in range(1, 8)],
        role="hardener",
        source="test",
    )
    calls = []

    design = enumerate_formulation_candidates(
        resin,
        hardener,
        max_pairs=49,
        max_formulations=4,
        batch_size=10,
        random_state=7,
        batch_callback=lambda **kwargs: calls.append(kwargs) or True,
    )

    assert len(design.candidate_df) == 4
    assert len(calls) == 1
    assert calls[0]["total_count"] == 4


def test_iter_pair_indices_does_not_require_full_cartesian_index_array():
    batches = list(
        iter_pair_indices(
            total_pairs=10**12,
            sample_size=257,
            batch_size=64,
            random_state=7,
        )
    )
    sampled = np.concatenate(batches)

    assert len(sampled) == 257
    assert len(set(sampled.tolist())) == 257
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < 10**12


def test_manual_weight_change_rebalances_other_weights_proportionally():
    base = {
        "performance": 0.40,
        "synth": 0.15,
        "feasibility": 0.15,
        "applicability": 0.12,
        "uncertainty": 0.10,
        "novelty": 0.05,
        "feature_guidance": 0.03,
    }

    updated = rebalance_screening_weights(
        base,
        changed_key="performance",
        new_value=0.60,
    )

    assert updated["performance"] == pytest.approx(0.60)
    assert sum(updated.values()) == pytest.approx(1.0)
    assert updated["synth"] == pytest.approx(0.15 * 0.40 / 0.60)
    assert updated["feature_guidance"] == pytest.approx(0.03 * 0.40 / 0.60)


def test_pair_sampling_handles_huge_cartesian_space_without_materializing_it():
    sampled = sample_pair_indices(
        total_pairs=10**12,
        sample_size=128,
        random_state=7,
    )

    assert len(sampled) == 128
    assert len(set(sampled.tolist())) == 128
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < 10**12


if __name__ == "__main__":
    unittest.main()
