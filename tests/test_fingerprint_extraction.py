import numpy as np
import pandas as pd

import core.molecular_features as molecular_features
import core.molecular_feature_workflow as molecular_feature_workflow
from core.molecular_features import append_configured_semantic_features


def test_fingerprint_extraction_reuses_duplicate_molecule_work(monkeypatch):
    parse_calls = []

    def fake_parse(raw, **_kwargs):
        parse_calls.append(raw)
        return raw

    monkeypatch.setattr(molecular_features, "parse_chemical_string", fake_parse)

    extractor = molecular_features.FingerprintExtractor()
    monkeypatch.setattr(
        extractor,
        "_gen_fp_array",
        lambda *_args, **_kwargs: np.array([1, 0, 1], dtype=np.uint8),
    )

    features, valid_indices = extractor.smiles_to_fingerprints(
        ["duplicate", "duplicate", "unique"],
        fp_type="Morgan",
        n_bits=3,
    )

    assert valid_indices == [0, 1, 2]
    assert features.shape == (3, 3)
    assert parse_calls == ["duplicate", "unique"]
    assert features.dtypes.tolist() == [np.dtype("uint8")] * 3


def test_merge_extracted_features_restores_sparse_rows_without_changing_order():
    base = pd.DataFrame({"smiles": ["a", "b", "c", "d"]})
    features = pd.DataFrame(
        {"resin_MACCS_0": [1, 2], "resin_MACCS_1": [0, 1]},
        dtype=np.uint8,
    )

    merge_extracted_features = getattr(
        molecular_feature_workflow,
        "merge_extracted_features",
        None,
    )
    assert callable(merge_extracted_features)

    merged = merge_extracted_features(
        base,
        features,
        valid_indices=[1, 3],
        keep_all_rows=True,
    )

    assert merged["smiles"].tolist() == ["a", "b", "c", "d"]
    assert pd.isna(merged["resin_MACCS_0"].iloc[0])
    assert merged["resin_MACCS_0"].iloc[1] == 1.0
    assert pd.isna(merged["resin_MACCS_0"].iloc[2])
    assert merged["resin_MACCS_0"].iloc[3] == 2.0


def test_align_extracted_features_to_rows_vectorizes_sparse_row_backfill():
    features = pd.DataFrame(
        {"resin_MACCS_0": [1, 2], "resin_MACCS_1": [0, 1]},
        dtype=np.uint8,
    )

    align_features = getattr(
        molecular_feature_workflow,
        "align_extracted_features_to_rows",
        None,
    )
    assert callable(align_features)

    aligned = align_features(
        features,
        valid_indices=[1, 3],
        total_rows=4,
    )

    assert aligned.index.tolist() == [0, 1, 2, 3]
    assert aligned["resin_MACCS_0"].isna().tolist() == [True, False, True, False]
    assert aligned["resin_MACCS_0"].iloc[1] == 1
    assert aligned["resin_MACCS_0"].iloc[3] == 2
    assert aligned["resin_MACCS_1"].iloc[1] == 0
    assert aligned["resin_MACCS_1"].iloc[3] == 1


def test_disabled_semantic_features_do_not_copy_extracted_frame():
    features = pd.DataFrame(
        {"resin_MACCS_0": [1, 0, 1]},
        dtype=np.uint8,
    )

    returned, valid_indices = append_configured_semantic_features(
        features,
        [0, 1, 2],
        ["CCO", "CCN", "CCC"],
        {},
    )

    assert returned is features
    assert valid_indices == [0, 1, 2]


def test_xtb_extraction_calculates_duplicate_inputs_once(monkeypatch):
    extractor = molecular_features.XTBFeatureExtractor(xtb_path='xtb')
    monkeypatch.setattr(extractor, 'AVAILABLE', True)
    calls = []

    def fake_calc(raw):
        calls.append(raw)
        return {'xtb_gap': float(len(str(raw)))}

    monkeypatch.setattr(extractor, '_calc_features', fake_calc)

    features, valid_indices = extractor.featurize(
        ['CCO', 'CCO', 'CCN', 'CCO'],
        n_jobs=1,
    )

    assert calls == ['CCO', 'CCN']
    assert valid_indices == [0, 1, 2, 3]
    assert features['xtb_gap'].tolist() == [3.0, 3.0, 3.0, 3.0]


def test_xtb_extraction_reuses_chemically_equivalent_inputs(monkeypatch):
    extractor = molecular_features.XTBFeatureExtractor(xtb_path='xtb')
    monkeypatch.setattr(extractor, 'AVAILABLE', True)
    calls = []

    def fake_calc(raw):
        calls.append(raw)
        return {'xtb_gap': 1.0}

    monkeypatch.setattr(extractor, '_calc_features', fake_calc)

    features, valid_indices = extractor.featurize(
        ['C(C)O', 'CCO'],
        n_jobs=1,
    )

    assert len(calls) == 1
    assert valid_indices == [0, 1]
    assert features['xtb_gap'].tolist() == [1.0, 1.0]
