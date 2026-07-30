import numpy as np
import pandas as pd
import pytest

from core.process_pls import (
    ProcessPLSTransformer,
    compute_vip_scores,
    fingerprint_process_pls_workflow,
)


def test_process_pls_outputs_components_vip_features_and_masks():
    X = pd.DataFrame({
        "cure_temp": [80.0, 90.0, np.nan, 110.0, 120.0, 130.0],
        "cure_time": [30.0, 40.0, 50.0, np.nan, 70.0, 80.0],
        "resin_MolWt": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    })
    y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    transformer = ProcessPLSTransformer(
        process_feature_cols=["cure_temp", "cure_time"],
        max_components=2,
        vip_top_k=1,
        random_state=42,
    ).fit(X, y)
    result = transformer.transform(X)

    assert list(result.columns) == transformer.get_feature_names_out().tolist()
    assert "process_pls_1" in result.columns
    assert any(column.endswith("__missing") for column in result.columns)
    assert "resin_MolWt" in result.columns
    assert np.isfinite(result.to_numpy(dtype=float)).all()


def test_process_pls_does_not_refit_on_transform():
    train = pd.DataFrame({
        "temperature": [1.0, 2.0, 3.0, np.nan],
        "time": [10.0, 11.0, 12.0, 13.0],
    })
    test = pd.DataFrame({"temperature": [1000.0], "time": [999.0]})
    y = np.array([1.0, 2.0, 3.0, 4.0])

    transformer = ProcessPLSTransformer(
        process_feature_cols=["temperature", "time"],
        max_components=1,
        random_state=42,
    ).fit(train, y)
    imputer_statistics = transformer.imputer_.statistics_.copy()
    transformer.transform(test)

    np.testing.assert_array_equal(transformer.imputer_.statistics_, imputer_statistics)


def test_process_pls_rejects_missing_required_columns():
    transformer = ProcessPLSTransformer(process_feature_cols=["temperature"])
    with pytest.raises(ValueError, match="missing required process columns"):
        transformer.fit(pd.DataFrame({"time": [1.0, 2.0]}), np.array([1.0, 2.0]))


def test_vip_scores_are_finite_and_match_feature_count():
    class FakePLS:
        x_weights_ = np.array([[1.0], [2.0]])
        x_scores_ = np.array([[1.0], [2.0], [3.0]])
        y_loadings_ = np.array([[1.0]])

    scores = compute_vip_scores(FakePLS())
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_process_pls_workflow_fingerprint_is_order_sensitive():
    first = fingerprint_process_pls_workflow({
        "schema_version": 1,
        "process_feature_cols": ["temperature", "time"],
        "output_feature_names": ["process_pls_1"],
    })
    second = fingerprint_process_pls_workflow({
        "schema_version": 1,
        "process_feature_cols": ["time", "temperature"],
        "output_feature_names": ["process_pls_1"],
    })
    assert first != second


def test_process_candidate_inference_excludes_molecular_and_text_columns():
    from core.feature_selector import infer_process_feature_candidates

    frame = pd.DataFrame({
        "cure_temperature": [80.0, 90.0],
        "cure_time": [30.0, 40.0],
        "resin_smiles1": ["CCO", "CCC"],
        "resin_Morgan_0": [0, 1],
        "sample_id": ["A", "B"],
        "target": [1.0, 2.0],
    })
    result = infer_process_feature_candidates(
        frame,
        original_features=["cure_temperature", "cure_time", "resin_smiles1", "resin_Morgan_0"],
        molecular_features=["resin_smiles1", "resin_Morgan_0"],
        target_col="target",
    )
    assert result == ["cure_temperature", "cure_time"]


def test_process_pls_training_pipeline_fits_only_training_rows():
    from core.model_trainer import EnhancedModelTrainer

    X = pd.DataFrame({
        "temperature": [1.0, 2.0, 3.0, 4.0, 1000.0, 1001.0],
        "time": [10.0, 11.0, 12.0, 13.0, 999.0, 1000.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["temperature", "time"],
        "max_components": 1,
        "vip_top_k": 1,
        "missing_threshold": 0.85,
        "cv_splits": 2,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
    }

    trainer = EnhancedModelTrainer()
    result = trainer.train_model(
        X,
        y,
        model_name="线性回归",
        test_size=1 / 3,
        random_state=42,
        process_pls_config=config,
        use_process_pls=True,
    )
    fitted = result["pipeline"].named_steps["process_pls"]
    expected_median = float(X.iloc[result["train_indices"]]["temperature"].median())
    assert fitted.imputer_.statistics_[0] == expected_median
    assert expected_median != float(X["temperature"].median())
    assert result["feature_names"]


def test_process_pls_is_not_applied_when_disabled():
    from core.model_trainer import EnhancedModelTrainer

    X = pd.DataFrame({
        "temperature": [1.0, 2.0, 3.0, 4.0],
        "time": [10.0, 11.0, 12.0, 13.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = EnhancedModelTrainer().train_model(
        X,
        y,
        model_name="线性回归",
        use_process_pls=False,
    )
    assert "process_pls" not in result["pipeline"].named_steps


def test_process_pls_cross_validation_fits_each_outer_training_fold(monkeypatch):
    import core.model_trainer as model_trainer
    from core.model_trainer import EnhancedModelTrainer
    from core.process_pls import ProcessPLSTransformer

    fit_row_counts = []

    class RecordingProcessPLS(ProcessPLSTransformer):
        def fit(self, X, y):
            fit_row_counts.append(len(X))
            return super().fit(X, y)

    monkeypatch.setattr(model_trainer, "ProcessPLSTransformer", RecordingProcessPLS)

    X = pd.DataFrame({
        "temperature": [1.0, 2.0, 3.0, 4.0, 1000.0, 1001.0, 1002.0, 1003.0],
        "time": [10.0, 11.0, 12.0, 13.0, 999.0, 1000.0, 1001.0, 1002.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["temperature", "time"],
        "max_components": 1,
        "vip_top_k": 1,
        "missing_threshold": 0.85,
        "cv_splits": 2,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
    }

    result = EnhancedModelTrainer().cross_validate_model(
        X,
        y,
        model_name="线性回归",
        cv_strategy="repeated_kfold",
        n_splits=4,
        n_repeats=1,
        random_state=42,
        process_pls_config=config,
        use_process_pls=True,
    )

    assert result["fold_r2"]
    assert fit_row_counts
    assert all(row_count < len(X) for row_count in fit_row_counts)


def test_process_pls_artifact_round_trip_preserves_workflow_metadata():
    from core.model_io import (
        create_model_artifact,
        dumps_artifact,
        loads_artifact,
        process_pls_to_artifact_extra,
    )

    config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["temperature", "time"],
        "max_components": 8,
        "vip_top_k": 8,
        "missing_threshold": 0.85,
        "cv_splits": 5,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
        "workflow_hash": "abc123",
    }
    artifact = create_model_artifact(
        model_name="test",
        target_col="target",
        feature_cols=["process_pls_1"],
        model=object(),
        extra=process_pls_to_artifact_extra(config),
    )
    restored = loads_artifact(dumps_artifact(artifact))

    assert restored["extra"]["process_pls_workflow"]["process_feature_cols"] == [
        "temperature",
        "time",
    ]
    assert restored["extra"]["process_pls_schema_version"] == 1
    assert restored["extra"]["process_pls_workflow_hash"] == "abc123"


def test_legacy_artifact_without_process_pls_is_unchanged():
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact

    artifact = create_model_artifact(
        model_name="legacy",
        target_col="target",
        feature_cols=["temperature"],
        model=object(),
        extra={"molecular_feature_workflow": {"schema_version": 1}},
    )
    restored = loads_artifact(dumps_artifact(artifact))

    assert "process_pls_workflow" not in restored["extra"]
    assert restored["extra"]["molecular_feature_workflow"] == {"schema_version": 1}


def test_process_pls_output_order_is_stable_after_joblib_round_trip(tmp_path):
    import joblib

    X = pd.DataFrame({
        "temperature": [1.0, 2.0, 3.0, 4.0],
        "time": [10.0, 11.0, 12.0, 13.0],
        "other": [5.0, 6.0, 7.0, 8.0],
    })
    y = np.array([1.0, 2.0, 3.0, 4.0])
    transformer = ProcessPLSTransformer(
        process_feature_cols=["temperature", "time"],
        max_components=1,
    ).fit(X, y)
    path = tmp_path / "process_pls.joblib"

    joblib.dump(transformer, path)
    restored = joblib.load(path)

    assert restored.get_feature_names_out().tolist() == transformer.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(restored.transform(X), transformer.transform(X))
