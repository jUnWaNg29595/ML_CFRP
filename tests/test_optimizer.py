from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold as RealKFold
from sklearn.model_selection import StratifiedKFold as RealStratifiedKFold

from core.optimizer import (
    HyperparameterOptimizer,
    OptimizationEvaluationConfig,
    build_adaptive_regression_strata,
    prepare_regression_optimization,
    select_stable_trial,
    select_stratified_training_budget,
)


def _reliable_frame(rows=80):
    index = pd.Index(np.arange(1000, 1000 + rows), name="source_row")
    X = pd.DataFrame(
        {
            "resin_feature": np.arange(rows, dtype=float),
            "process_temperature": np.linspace(80.0, 180.0, rows),
        },
        index=index,
    )
    y = pd.Series(np.repeat(np.arange(10, dtype=float), rows // 10), index=index, name="tg")
    return X, y


def test_preflight_drops_only_invalid_targets_and_preserves_dataframe_columns():
    X, y = _reliable_frame()
    y.iloc[0] = np.nan
    y.iloc[1] = np.inf
    y.iloc[2] = -np.inf
    config = OptimizationEvaluationConfig(cv_folds=4, test_size=0.20, random_state=7)

    preflight = prepare_regression_optimization(X, y, config)

    assert list(preflight.X.columns) == ["resin_feature", "process_temperature"]
    assert preflight.removed_target_rows == 3
    assert preflight.X.index.equals(preflight.y.index)
    assert np.isfinite(preflight.y.to_numpy(dtype=float)).all()
    assert set(preflight.outer_train_indices).isdisjoint(preflight.outer_test_indices)
    assert len(preflight.outer_train_indices) + len(preflight.outer_test_indices) == len(preflight.y)


def test_adaptive_strata_reduces_bins_until_every_bin_supports_test_and_cv():
    y = np.asarray([0.0] * 12 + [1.0] * 12 + [2.0] * 12 + [3.0] * 12)

    labels, actual_bins = build_adaptive_regression_strata(
        y,
        cv_folds=4,
        test_size=0.20,
        requested_bins=10,
    )

    counts = pd.Series(labels).value_counts()
    assert 2 <= actual_bins <= 10
    assert counts.min() >= 5


def test_preflight_rejects_targets_that_cannot_support_stratified_cv():
    X = pd.DataFrame({"feature": np.arange(8, dtype=float)})
    y = pd.Series(np.arange(8, dtype=float), name="tg")

    with pytest.raises(ValueError, match="无法构建满足独立测试集和 5 折交叉验证"):
        prepare_regression_optimization(
            X,
            y,
            OptimizationEvaluationConfig(cv_folds=5, test_size=0.20),
        )


def test_preflight_reduces_strata_until_outer_training_supports_cv():
    X = pd.DataFrame({"feature": np.arange(20, dtype=float)})
    y = pd.Series(np.repeat(np.arange(4, dtype=float), 5), name="tg")
    config = OptimizationEvaluationConfig(
        cv_folds=5,
        test_size=0.20,
        quantile_bins=4,
        random_state=7,
    )

    preflight = prepare_regression_optimization(X, y, config)
    training_positions = preflight.X.index.get_indexer(preflight.outer_train_indices)
    training_counts = pd.Series(preflight.strata[training_positions]).value_counts()

    assert preflight.quantile_bins < 4
    assert training_counts.min() >= config.cv_folds


def test_training_budget_rejects_strata_that_cannot_support_inner_cv():
    X, y = _reliable_frame()
    config = OptimizationEvaluationConfig(
        cv_folds=4,
        test_size=0.20,
        quantile_bins=4,
        max_samples=15,
        random_state=7,
    )
    preflight = prepare_regression_optimization(X, y, config)

    with pytest.raises(ValueError, match="训练样本预算无法保证每个连续目标分层至少保留 4 行"):
        select_stratified_training_budget(preflight, config)


def test_explicit_kfold_uses_legacy_kfold_strategy(monkeypatch):
    X, y = _reliable_frame()
    optimizer = HyperparameterOptimizer()
    optimizer.get_model_params = lambda trial, model_name, fast_mode=False: {}
    optimizer.trainer._get_model = lambda model_name, **params: LinearRegression()
    calls = {"kfold": 0, "stratified": 0}

    def make_kfold(*args, **kwargs):
        calls["kfold"] += 1
        return RealKFold(*args, **kwargs)

    def make_stratified(*args, **kwargs):
        calls["stratified"] += 1
        return RealStratifiedKFold(*args, **kwargs)

    monkeypatch.setattr("core.optimizer.KFold", make_kfold)
    monkeypatch.setattr("core.optimizer.StratifiedKFold", make_stratified)

    optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=1,
        cv=4,
        cv_strategy="kfold",
        n_jobs=1,
        use_pruner=False,
    )

    assert calls == {"kfold": 1, "stratified": 0}


def test_explicit_kfold_bypasses_reliable_preflight_when_strata_impossible(monkeypatch):
    X = pd.DataFrame({"feature": np.arange(10, dtype=float)})
    y = pd.Series(np.arange(10, dtype=float), name="tg")
    optimizer = HyperparameterOptimizer()
    optimizer.get_model_params = lambda trial, model_name, fast_mode=False: {}
    optimizer.trainer._get_model = lambda model_name, **params: LinearRegression()

    best_params, best_score, study = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=1,
        cv=5,
        cv_strategy="kfold",
        n_jobs=1,
        use_pruner=False,
    )

    assert best_params == {}
    assert np.isfinite(best_score)
    assert len(study.trials) == 1


def test_explicit_holdout_bypasses_reliable_preflight_when_strata_impossible(monkeypatch):
    X = pd.DataFrame({"feature": np.arange(10, dtype=float)})
    y = pd.Series(np.arange(10, dtype=float), name="tg")
    optimizer = HyperparameterOptimizer()
    optimizer.get_model_params = lambda trial, model_name, fast_mode=False: {}
    optimizer.trainer._get_model = lambda model_name, **params: LinearRegression()

    best_params, best_score, study = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=1,
        cv=5,
        cv_strategy="holdout",
        val_size=0.2,
        n_jobs=1,
        use_pruner=False,
    )

    assert best_params == {}
    assert np.isfinite(best_score)
    assert len(study.trials) == 1


def test_duplicate_source_labels_support_preflight_and_training_budget():
    X = pd.DataFrame(
        {"feature": np.arange(80, dtype=float)},
        index=pd.Index(["duplicate"] * 80, name="source_row"),
    )
    y = pd.Series(np.repeat(np.arange(10, dtype=float), 8), name="tg")
    config = OptimizationEvaluationConfig(
        cv_folds=4,
        test_size=0.20,
        quantile_bins=4,
        max_samples=32,
        random_state=7,
    )

    preflight = prepare_regression_optimization(X, y, config)
    budgeted = select_stratified_training_budget(preflight, config)

    assert len(preflight.source_indices) == 80
    assert set(preflight.outer_train_indices) == {"duplicate"}
    assert len(budgeted.outer_train_indices) == 32
    assert budgeted.outer_test_indices == preflight.outer_test_indices


def test_optimizer_pipeline_keeps_pls_and_preprocessing_fold_local(monkeypatch):
    import core.model_trainer as trainer_module
    from core.model_trainer import EnhancedModelTrainer
    from core.process_pls import ProcessPLSTransformer

    fit_row_counts = []

    class RecordingProcessPLS(ProcessPLSTransformer):
        def fit(self, X, y):
            fit_row_counts.append(len(X))
            return super().fit(X, y)

    monkeypatch.setattr(trainer_module, "ProcessPLSTransformer", RecordingProcessPLS)
    X, y = _reliable_frame()
    config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["process_temperature"],
        "max_components": 1,
        "vip_top_k": 1,
        "missing_threshold": 0.85,
        "cv_splits": 2,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
    }

    pipeline = EnhancedModelTrainer(use_gpu=False).build_regression_cv_pipeline(
        "线性回归",
        X.columns.tolist(),
        random_state=42,
        process_pls_config=config,
        use_process_pls=True,
    )
    pipeline.fit(X.iloc[:60], y.iloc[:60])
    pipeline.predict(X.iloc[60:])

    assert list(pipeline.named_steps) == [
        "process_pls",
        "inf_cleaner",
        "imputer",
        "nan_col_dropper",
        "scaler",
        "model",
    ]
    assert fit_row_counts == [60]


def test_optimizer_pipeline_clones_pls_inside_real_cv(monkeypatch):
    import core.model_trainer as trainer_module
    from core.model_trainer import EnhancedModelTrainer
    from core.process_pls import ProcessPLSTransformer

    fit_row_counts = []

    class RecordingProcessPLS(ProcessPLSTransformer):
        def fit(self, X, y):
            fit_row_counts.append(len(X))
            return super().fit(X, y)

    monkeypatch.setattr(trainer_module, "ProcessPLSTransformer", RecordingProcessPLS)
    X, y = _reliable_frame(rows=90)
    config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["process_temperature"],
        "max_components": 1,
        "vip_top_k": 1,
        "missing_threshold": 0.85,
        "cv_splits": 2,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
    }

    pipeline = EnhancedModelTrainer(use_gpu=False).build_regression_cv_pipeline(
        "线性回归",
        X.columns.tolist(),
        random_state=42,
        process_pls_config=config,
        use_process_pls=True,
    )
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=RealKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="r2",
    )

    assert len(scores) == 3
    assert fit_row_counts == [60, 60, 60]


def test_optimizer_pipeline_rejects_raw_frame_and_classification_models():
    from core.model_trainer import EnhancedModelTrainer

    trainer = EnhancedModelTrainer(use_gpu=False)

    with pytest.raises(ValueError, match="不支持通用回归优化 pipeline"):
        trainer.build_regression_cv_pipeline(
            "Transformer + BNN",
            ["resin_feature"],
        )

    with pytest.raises(ValueError, match="仅支持回归模型"):
        trainer.build_regression_cv_pipeline(
            "随机森林分类",
            ["resin_feature"],
        )


def test_stable_trial_selection_prefers_lower_standard_deviation_within_tolerance():
    trials = [
        SimpleNamespace(
            number=4,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={"mean_cv_r2": 0.801, "std_cv_r2": 0.081, "min_cv_r2": 0.70},
        ),
        SimpleNamespace(
            number=2,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={"mean_cv_r2": 0.798, "std_cv_r2": 0.021, "min_cv_r2": 0.75},
        ),
    ]

    selected = select_stable_trial(trials, stability_tolerance=0.005)

    assert selected.number == 2


def test_reliable_optimization_never_uses_outer_test_rows_in_cv(monkeypatch):
    X, y = _reliable_frame()
    optimizer = HyperparameterOptimizer()
    result = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=2,
        evaluation_config=OptimizationEvaluationConfig(
            cv_folds=4,
            test_size=0.20,
            random_state=42,
        ),
        use_pruner=False,
    )

    test_rows = set(result.test_indices)
    assert result.status == "completed"
    assert result.independent_test["evaluated"] is True
    assert result.inner_cv["completed_folds"] == 4
    assert all(
        test_rows.isdisjoint(fold["train_indices"])
        and test_rows.isdisjoint(fold["valid_indices"])
        for fold in result.fold_source_indices
    )


def test_failed_trials_are_recorded_and_all_failures_return_a_readable_result(monkeypatch):
    class AlwaysFailRegressor:
        def fit(self, X, y):
            raise RuntimeError("intentional fold failure")

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    X, y = _reliable_frame()
    optimizer = HyperparameterOptimizer()
    monkeypatch.setattr(
        optimizer.trainer,
        "_get_model",
        lambda model_name, random_state=42, **params: AlwaysFailRegressor(),
    )

    result = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=2,
        evaluation_config=OptimizationEvaluationConfig(cv_folds=4, random_state=42),
        use_pruner=False,
    )

    assert result.status == "failed"
    assert result.best_params == {}
    assert result.trial_summary["failed"] == 2
    assert "intentional fold failure" in result.failure_reasons


def test_exploratory_optimization_with_pls_uses_fold_local_pipeline(monkeypatch):
    X, y = _reliable_frame(rows=40)
    optimizer = HyperparameterOptimizer()
    optimizer.get_model_params = lambda trial, model_name, fast_mode=False: {}
    calls = []
    original_builder = optimizer.trainer.build_regression_cv_pipeline

    def recording_builder(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        optimizer.trainer,
        "build_regression_cv_pipeline",
        recording_builder,
    )
    process_pls_config = {
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": ["process_temperature"],
        "max_components": 1,
        "vip_top_k": 1,
        "missing_threshold": 0.85,
        "cv_splits": 2,
        "random_state": 42,
        "selection_mode": "auto_combined_score",
    }

    result = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=1,
        cv=2,
        cv_strategy="kfold",
        n_jobs=1,
        use_pruner=False,
        evaluation_config=OptimizationEvaluationConfig(
            cv_folds=2,
            mode="exploratory",
            use_process_pls=True,
            process_pls_config=process_pls_config,
        ),
    )

    assert result.status == "completed"
    assert calls
    assert all(call["use_process_pls"] is True for call in calls)
    assert all(call["process_pls_config"] == process_pls_config for call in calls)


def test_optimizer_reuses_training_contract_context_for_trials_and_final_pipeline(monkeypatch):
    X, y = _reliable_frame(rows=30)
    optimizer = HyperparameterOptimizer()
    context = {
        "canonical_feature_cols": list(X.columns),
        "feature_registry_hash": "r1",
        "dataset_manifest_hash": "m1",
    }
    calls = []
    original_builder = optimizer.trainer.build_regression_cv_pipeline

    def recording_builder(*args, **kwargs):
        calls.append(kwargs.get("feature_contract_context"))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(optimizer.trainer, "build_regression_cv_pipeline", recording_builder)
    result = optimizer.optimize(
        "线性回归",
        X,
        y,
        n_trials=1,
        cv=2,
        use_pruner=False,
        feature_contract_context=context,
    )

    assert result.status == "completed"
    assert calls and all(item is context for item in calls)
