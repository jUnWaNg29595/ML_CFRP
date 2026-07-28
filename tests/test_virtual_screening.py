import unittest

import numpy as np
import pandas as pd

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
    apply_feature_overrides,
    build_component_library,
    build_feature_matrix,
    enumerate_formulation_candidates,
    extract_features_from_config,
    filter_candidates_by_epoxy_rules,
    generate_candidate_pool,
    generate_virtual_component_library,
    infer_primary_component_role,
    limit_unique_candidates_for_expensive_features,
    predict_with_model,
    resolve_component_smiles_cols,
)
from core.smiles_utils import parse_chemical_string


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


class VirtualScreeningSamplingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
