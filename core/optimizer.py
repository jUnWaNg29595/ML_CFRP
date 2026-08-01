# -*- coding: utf-8 -*-
"""
超参数优化模块
更新内容：
1. optimize 方法增加 progress_callback 参数，支持实时进度条。
2. 保持了之前的自动数据清洗逻辑。
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from math import ceil
import time
from typing import Any

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    KFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# 抑制 Optuna 的日志输出，只显示进度条
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

from core.model_trainer import EnhancedModelTrainer
from core.process_pls import fingerprint_process_pls_workflow


@dataclass(frozen=True)
class OptimizationEvaluationConfig:
    test_size: float = 0.20
    cv_folds: int = 5
    quantile_bins: int | None = None
    random_state: int = 42
    max_samples: int | None = None
    stability_tolerance: float = 0.005
    use_process_pls: bool = False
    process_pls_config: dict[str, Any] | None = None
    mode: str = "reliable"

    def validate(self) -> None:
        if not 0.05 <= float(self.test_size) <= 0.40:
            raise ValueError("独立测试集比例必须在 0.05 到 0.40 之间")
        if int(self.cv_folds) < 2:
            raise ValueError("内层交叉验证折数至少为 2")
        if float(self.stability_tolerance) < 0:
            raise ValueError("稳定性容差不能为负数")
        if self.mode not in {"reliable", "exploratory"}:
            raise ValueError("优化模式必须为 reliable 或 exploratory")


class TrialEvaluationError(RuntimeError):
    """标记一个 trial 的模型评估失败，但允许 Optuna 继续搜索。"""


@dataclass(frozen=True)
class OptimizationProgress:
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    total_trials: int
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    current_best_mean_r2: float | None
    current_best_std_r2: float | None
    stage: str


@dataclass
class OptimizationResult:
    model_name: str
    best_params: dict[str, Any]
    selected_trial_number: int | None
    inner_cv: dict[str, Any]
    independent_test: dict[str, Any]
    train_indices: list[Any]
    test_indices: list[Any]
    fold_source_indices: list[dict[str, list[Any]]]
    feature_columns: list[str]
    process_pls_workflow_hash: str | None
    evaluation_config: dict[str, Any]
    trial_summary: dict[str, int]
    failure_reasons: dict[str, int]
    study: Any = None
    status: str = "completed"
    message: str = ""

    def as_legacy_tuple(self):
        return (
            self.best_params,
            self.inner_cv.get("mean_r2"),
            self.study,
        )

    def __iter__(self):
        """让旧页面仍可使用 best_params, best_score, study = result。"""
        return iter(self.as_legacy_tuple())

    def __len__(self):
        return 3

    def __getitem__(self, item):
        return self.as_legacy_tuple()[item]


def select_stable_trial(trials, stability_tolerance):
    """在最优均值附近优先选择方差更小、最差折更好的 trial。"""
    valid_trials = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and np.isfinite(trial.user_attrs.get("mean_cv_r2", np.nan))
        and np.isfinite(trial.user_attrs.get("std_cv_r2", np.nan))
        and np.isfinite(trial.user_attrs.get("min_cv_r2", np.nan))
    ]
    if not valid_trials:
        return None

    best_mean = max(float(trial.user_attrs["mean_cv_r2"]) for trial in valid_trials)
    candidates = [
        trial
        for trial in valid_trials
        if best_mean - float(trial.user_attrs["mean_cv_r2"])
        <= float(stability_tolerance)
    ]
    return min(
        candidates,
        key=lambda trial: (
            float(trial.user_attrs["std_cv_r2"]),
            -float(trial.user_attrs["min_cv_r2"]),
            int(trial.number),
        ),
    )


@dataclass
class OptimizationPreflight:
    X: pd.DataFrame
    y: pd.Series
    source_indices: list[Any]
    strata: np.ndarray
    quantile_bins: int
    removed_target_rows: int
    outer_train_indices: list[Any]
    outer_test_indices: list[Any]
    validation_messages: list[str] = field(default_factory=list)
    outer_train_positions: list[int] = field(default_factory=list, repr=False)
    outer_test_positions: list[int] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "original_rows": int(len(self.source_indices) + self.removed_target_rows),
            "valid_target_rows": int(len(self.y)),
            "removed_target_rows": int(self.removed_target_rows),
            "quantile_bins": int(self.quantile_bins),
            "outer_train_rows": int(len(self.outer_train_indices)),
            "outer_test_rows": int(len(self.outer_test_indices)),
        }


def build_adaptive_regression_strata(
    y,
    cv_folds,
    test_size,
    requested_bins=None,
) -> tuple[np.ndarray, int]:
    """为连续目标构建同时支持独立测试集与交叉验证的自适应分层标签。"""
    y_series = pd.Series(pd.to_numeric(pd.Series(y), errors="coerce"))
    y_series = y_series.replace([np.inf, -np.inf], np.nan)
    n_samples = len(y_series)
    required_per_bin = max(int(cv_folds), ceil(1 / float(test_size)))
    candidate_bins = (
        int(requested_bins)
        if requested_bins is not None
        else min(10, max(2, n_samples // (2 * int(cv_folds))))
    )

    for bins in range(candidate_bins, 1, -1):
        try:
            categories = pd.qcut(y_series, q=bins, duplicates="drop")
        except ValueError:
            continue

        labels, categories = pd.factorize(categories, sort=True)
        actual_bins = len(categories)
        if actual_bins < 2:
            continue

        counts = pd.Series(labels).value_counts()
        if len(counts) == actual_bins and counts.min() >= required_per_bin:
            return labels.astype(int), int(actual_bins)

    raise ValueError(
        "无法构建满足独立测试集和 "
        f"{int(cv_folds)} 折交叉验证的连续目标分层；有效样本={n_samples}，"
        "请减少折数、降低分箱或补充数据"
    )


def prepare_regression_optimization(
    X,
    y,
    config: OptimizationEvaluationConfig,
) -> OptimizationPreflight:
    """清理连续目标并生成可靠优化所需的外层分层切分。"""
    config.validate()

    if isinstance(X, pd.DataFrame):
        X_frame = X.copy()
    else:
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("特征 X 必须是二维数组或 DataFrame")
        X_frame = pd.DataFrame(
            X_array,
            columns=[f"Feature_{column_index}" for column_index in range(X_array.shape[1])],
        )

    y_name = getattr(y, "name", None)
    target_values = pd.to_numeric(pd.Series(y), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if len(X_frame) != len(target_values):
        raise ValueError("特征 X 与目标 y 的样本数必须一致")
    y_series = pd.Series(target_values.to_numpy(), index=X_frame.index, name=y_name)

    valid_target_mask = np.isfinite(y_series.to_numpy(dtype=float))
    removed_target_rows = int((~valid_target_mask).sum())
    X_valid = X_frame.iloc[np.flatnonzero(valid_target_mask)].copy()
    y_valid = y_series.iloc[np.flatnonzero(valid_target_mask)].copy()
    source_indices = X_valid.index.tolist()

    candidate_bins = (
        config.quantile_bins
        if config.quantile_bins is not None
        else min(10, max(2, len(y_valid) // (2 * int(config.cv_folds))))
    )
    for requested_bins in range(int(candidate_bins), 1, -1):
        strata, actual_bins = build_adaptive_regression_strata(
            y_valid,
            cv_folds=config.cv_folds,
            test_size=config.test_size,
            requested_bins=requested_bins,
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        outer_train_positions, outer_test_positions = next(splitter.split(X_valid, strata))
        outer_training_counts = pd.Series(strata[outer_train_positions]).value_counts()
        if outer_training_counts.min() < int(config.cv_folds):
            continue

        return OptimizationPreflight(
            X=X_valid,
            y=y_valid,
            source_indices=source_indices,
            strata=strata,
            quantile_bins=actual_bins,
            removed_target_rows=removed_target_rows,
            outer_train_indices=X_valid.index.take(outer_train_positions).tolist(),
            outer_test_indices=X_valid.index.take(outer_test_positions).tolist(),
            validation_messages=[
                f"已移除 {removed_target_rows} 行无效目标值。",
                f"连续目标已使用 {actual_bins} 个自适应分层。",
            ],
            outer_train_positions=outer_train_positions.tolist(),
            outer_test_positions=outer_test_positions.tolist(),
        )

    raise ValueError(
        "无法构建满足独立测试集和 "
        f"{int(config.cv_folds)} 折交叉验证的连续目标分层；有效样本={len(y_valid)}，"
        "请减少折数、降低分箱或补充数据"
    )


def select_stratified_training_budget(
    preflight: OptimizationPreflight,
    config: OptimizationEvaluationConfig,
) -> OptimizationPreflight:
    """在不触碰外层测试集的前提下，按分层比例限制优化训练预算。"""
    max_samples = int(config.max_samples) if config.max_samples is not None else 0
    if max_samples <= 0 or max_samples >= len(preflight.outer_train_indices):
        return preflight

    train_positions = np.asarray(preflight.outer_train_positions, dtype=int)
    train_strata = preflight.strata[train_positions]
    train_counts = pd.Series(train_strata).value_counts()
    minimum_budget = int(config.cv_folds) * len(train_counts)
    if max_samples < minimum_budget:
        raise ValueError(
            "训练样本预算无法保证每个连续目标分层至少保留 "
            f"{int(config.cv_folds)} 行用于交叉验证；当前预算={max_samples}，"
            f"至少需要={minimum_budget}，请增加 max_samples、减少折数或降低分箱"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_samples,
        random_state=config.random_state,
    )
    selected_positions, _ = next(
        splitter.split(np.zeros(len(train_positions)), train_strata)
    )
    selected_source_positions = train_positions[selected_positions]
    selected_indices = preflight.X.index.take(selected_source_positions).tolist()
    selected_counts = pd.Series(train_strata[selected_positions]).value_counts()
    if selected_counts.min() < int(config.cv_folds):
        raise ValueError(
            "训练样本预算无法保证每个连续目标分层至少保留 "
            f"{int(config.cv_folds)} 行用于交叉验证；当前预算={max_samples}，"
            f"请增加 max_samples、减少折数或降低分箱"
        )

    return OptimizationPreflight(
        X=preflight.X,
        y=preflight.y,
        source_indices=preflight.source_indices,
        strata=preflight.strata,
        quantile_bins=preflight.quantile_bins,
        removed_target_rows=preflight.removed_target_rows,
        outer_train_indices=selected_indices,
        outer_test_indices=preflight.outer_test_indices,
        validation_messages=[
            *preflight.validation_messages,
            f"优化训练集已按分层预算缩减为 {len(selected_indices)} 行。",
        ],
        outer_train_positions=selected_source_positions.tolist(),
        outer_test_positions=list(preflight.outer_test_positions),
    )


def _coerce_evaluation_config(
    evaluation_config,
    *,
    cv,
    random_state,
    val_size,
    max_samples,
    cv_strategy,
):
    if evaluation_config is None:
        strategy = str(cv_strategy or "").lower()
        mode = (
            "exploratory"
            if strategy.startswith("hold") or strategy.startswith("kfold")
            else "reliable"
        )
        return OptimizationEvaluationConfig(
            test_size=float(val_size),
            cv_folds=int(cv),
            random_state=int(random_state),
            max_samples=max_samples,
            mode=mode,
        )
    if isinstance(evaluation_config, OptimizationEvaluationConfig):
        return evaluation_config
    if isinstance(evaluation_config, dict):
        allowed = {
            field_name
            for field_name in OptimizationEvaluationConfig.__dataclass_fields__
        }
        payload = {
            key: value
            for key, value in evaluation_config.items()
            if key in allowed
        }
        return OptimizationEvaluationConfig(**payload)
    raise TypeError("evaluation_config 必须是 OptimizationEvaluationConfig、dict 或 None")


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _trial_state_summary(study) -> dict[str, int]:
    states = [trial.state for trial in study.trials]
    return {
        "completed": int(sum(state == optuna.trial.TrialState.COMPLETE for state in states)),
        "pruned": int(sum(state == optuna.trial.TrialState.PRUNED for state in states)),
        "failed": int(sum(state == optuna.trial.TrialState.FAIL for state in states)),
        "total": int(len(states)),
    }


def _failure_reason_summary(study) -> dict[str, int]:
    reasons = [
        str(trial.user_attrs.get("failure_reason") or "").strip()
        for trial in study.trials
    ]
    return dict(Counter(reason for reason in reasons if reason))


def _emit_optimization_progress(progress_callback, progress: OptimizationProgress) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback(progress)
        return
    except Exception:
        pass
    try:
        fraction = (
            float(progress.completed_trials + progress.pruned_trials + progress.failed_trials)
            / max(1, int(progress.total_trials))
        )
        progress_callback(min(max(fraction, 0.0), 1.0))
    except Exception:
        pass


def _valid_process_pls_workflow_hash(config: OptimizationEvaluationConfig) -> str | None:
    if not config.use_process_pls:
        return None
    payload = config.process_pls_config
    if not isinstance(payload, dict):
        return None
    if not payload.get("process_feature_cols"):
        return None
    if payload.get("schema_version") != 1:
        return None
    return fingerprint_process_pls_workflow(payload)


class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self):
        self.trainer = EnhancedModelTrainer()

    def get_model_params(self, trial, model_name, fast_mode: bool = False):
        """定义各模型的参数搜索空间"""
        params = {}
        fast_mode = bool(fast_mode)

        if model_name == "随机森林":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "Extra Trees":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "决策树":
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "线性回归":
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'positive': trial.suggest_categorical('positive', [True, False])
            }

        elif model_name == "XGBoost":
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }

        elif model_name == "LightGBM":
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1
            }

        elif model_name == "CatBoost":
            max_iters = 300 if fast_mode else 500
            params = {
                'iterations': trial.suggest_int('iterations', 50, max_iters),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'verbose': 0
            }

        elif model_name == "SVR":
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            }

        elif model_name in ["Ridge回归", "Lasso回归", "ElasticNet"]:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True)
            }
            if model_name == "ElasticNet":
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)

        elif model_name == "AdaBoost":
            max_estimators = 300 if fast_mode else 500
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
            }

        elif model_name == "梯度提升树":
            max_estimators = 200 if fast_mode else 300
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

        elif model_name == "多层感知器":
            params = {
                'hidden_layer_sizes': trial.suggest_categorical(
                    'hidden_layer_sizes',
                    [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)]
                ),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'max_iter': trial.suggest_int('max_iter', 200 if fast_mode else 300, 600)
            }

        elif model_name == "人工神经网络":
            hidden_opts = ["128,64", "256,128,64", "256,256,128", "512,256,128"]
            if fast_mode:
                hidden_opts = ["128,64", "256,128,64"]
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', hidden_opts),
                'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'epochs': trial.suggest_int('epochs', 60 if fast_mode else 100, 200)
            }

        elif model_name == "TensorFlow Sequential":
            hidden_opts = ["128,64", "128,64,32", "256,128,64", "256,256,128"]
            if fast_mode:
                hidden_opts = ["128,64", "128,64,32"]
            params = {
                'hidden_layers': trial.suggest_categorical('hidden_layers', hidden_opts),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'swish']),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.4),
                'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'epochs': trial.suggest_int('epochs', 60 if fast_mode else 100, 200),
                'early_stopping': True,
                'patience': trial.suggest_int('patience', 10, 30),
                'validation_split': trial.suggest_float('validation_split', 0.1, 0.2)
            }

        elif model_name == "Epoxy PINN (Physics-Informed)":
            max_epochs = 150 if fast_mode else 300
            params = {
                'mode': trial.suggest_categorical('mode', ['auto', 'tg', 'mechanics', 'generic']),
                'hidden_dim': trial.suggest_int('hidden_dim', 128, 768, step=64),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
                'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 0.0, 1e-3),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'epochs': trial.suggest_int('epochs', 60, max_epochs, step=30),
                'patience': trial.suggest_int('patience', 10, 40, step=5),
                'physics_weight': trial.suggest_float('physics_weight', 0.0, 5.0, step=0.5),
            }

        return params


    def _create_sampler(self, optimization_method, random_state):
        method = str(optimization_method or "tpe").lower()
        if method == "random":
            return optuna.samplers.RandomSampler(seed=int(random_state))
        if method == "gp":
            try:
                return optuna.samplers.GPSampler(seed=int(random_state))
            except AttributeError:
                return optuna.samplers.TPESampler(seed=int(random_state))
        if method == "cmaes":
            try:
                return optuna.samplers.CmaEsSampler(seed=int(random_state))
            except Exception:
                return optuna.samplers.TPESampler(seed=int(random_state))
        if method == "grid":
            return optuna.samplers.RandomSampler(seed=int(random_state))
        return optuna.samplers.TPESampler(seed=int(random_state))

    def _optimize_reliable(
        self,
        model_name,
        X,
        y,
        *,
        n_trials,
        config,
        progress_callback,
        fast_mode,
        use_pruner,
        timeout,
        optimization_method,
    ):
        y_name = getattr(y, "name", None)
        y_target = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        preflight = prepare_regression_optimization(X, y_target, config)
        preflight = select_stratified_training_budget(preflight, config)

        train_positions = np.asarray(preflight.outer_train_positions, dtype=int)
        test_positions = np.asarray(preflight.outer_test_positions, dtype=int)
        X_train = preflight.X.iloc[train_positions].copy()
        y_train = preflight.y.iloc[train_positions].copy()
        X_test = preflight.X.iloc[test_positions].copy()
        y_test = preflight.y.iloc[test_positions].copy()
        train_strata = preflight.strata[train_positions]

        splitter = StratifiedKFold(
            n_splits=int(config.cv_folds),
            shuffle=True,
            random_state=int(config.random_state),
        )
        fixed_splits = list(splitter.split(X_train, train_strata))
        fold_source_indices = [
            {
                "train_indices": X_train.index.take(train_idx).tolist(),
                "valid_indices": X_train.index.take(valid_idx).tolist(),
            }
            for train_idx, valid_idx in fixed_splits
        ]
        feature_columns = preflight.X.columns.tolist()
        workflow_hash = _valid_process_pls_workflow_hash(config)
        total_trials = max(1, int(n_trials))
        started_at = time.monotonic()

        def set_score_attrs(trial, scores):
            trial.set_user_attr("fold_scores", [float(score) for score in scores])
            trial.set_user_attr("completed_folds", int(len(scores)))
            if scores:
                trial.set_user_attr("mean_cv_r2", float(np.mean(scores)))
                trial.set_user_attr(
                    "std_cv_r2",
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                )
                trial.set_user_attr("min_cv_r2", float(np.min(scores)))

        def objective(trial):
            scores = []
            try:
                params = self.get_model_params(
                    trial,
                    model_name,
                    fast_mode=bool(fast_mode),
                )
                if model_name == "Epoxy PINN (Physics-Informed)" and y_name is not None:
                    params.setdefault("target_name", y_name)

                for fold_index, (train_idx, valid_idx) in enumerate(fixed_splits):
                    pipeline = self.trainer.build_regression_cv_pipeline(
                        model_name,
                        feature_columns,
                        random_state=int(config.random_state),
                        process_pls_config=config.process_pls_config,
                        use_process_pls=bool(config.use_process_pls),
                        **params,
                    )
                    pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    y_pred = pipeline.predict(X_train.iloc[valid_idx])
                    score = float(r2_score(y_train.iloc[valid_idx], y_pred))
                    if not np.isfinite(score):
                        raise ValueError(f"fold {fold_index + 1} produced non-finite R²")
                    scores.append(score)
                    trial.report(float(np.mean(scores)), step=int(fold_index))
                    if use_pruner and trial.should_prune():
                        set_score_attrs(trial, scores)
                        trial.set_user_attr("failure_reason", None)
                        raise optuna.TrialPruned()

                set_score_attrs(trial, scores)
                trial.set_user_attr("failure_reason", None)
                return float(np.mean(scores))
            except optuna.TrialPruned:
                raise
            except TrialEvaluationError:
                raise
            except Exception as exc:
                reason = str(exc).strip()[:300] or exc.__class__.__name__
                trial.set_user_attr("failure_reason", reason)
                if scores:
                    set_score_attrs(trial, scores)
                raise TrialEvaluationError(reason) from exc

        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=3)
            if use_pruner
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=self._create_sampler(optimization_method, config.random_state),
        )

        def trial_finished_callback(current_study, _trial):
            summary = _trial_state_summary(current_study)
            elapsed = float(time.monotonic() - started_at)
            finished = summary["completed"] + summary["pruned"] + summary["failed"]
            estimated = (
                float(elapsed / finished * max(0, total_trials - finished))
                if finished > 0 and finished < total_trials
                else (0.0 if finished >= total_trials else None)
            )
            complete_trials = [
                item
                for item in current_study.trials
                if item.state == optuna.trial.TrialState.COMPLETE
                and np.isfinite(item.user_attrs.get("mean_cv_r2", np.nan))
            ]
            best_mean = None
            best_std = None
            if complete_trials:
                best_item = max(
                    complete_trials,
                    key=lambda item: float(item.user_attrs["mean_cv_r2"]),
                )
                best_mean = float(best_item.user_attrs["mean_cv_r2"])
                best_std = float(best_item.user_attrs.get("std_cv_r2", np.nan))
            stage = "completed" if finished >= total_trials else "running"
            _emit_optimization_progress(
                progress_callback,
                OptimizationProgress(
                    completed_trials=summary["completed"],
                    pruned_trials=summary["pruned"],
                    failed_trials=summary["failed"],
                    total_trials=total_trials,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=estimated,
                    current_best_mean_r2=best_mean,
                    current_best_std_r2=best_std,
                    stage=stage,
                ),
            )

        study.optimize(
            objective,
            n_trials=total_trials,
            timeout=timeout,
            callbacks=[trial_finished_callback],
            catch=(TrialEvaluationError,),
        )
        summary = _trial_state_summary(study)
        failure_reasons = _failure_reason_summary(study)
        selected_trial = select_stable_trial(study.trials, config.stability_tolerance)

        if selected_trial is None:
            _emit_optimization_progress(
                progress_callback,
                OptimizationProgress(
                    completed_trials=summary["completed"],
                    pruned_trials=summary["pruned"],
                    failed_trials=summary["failed"],
                    total_trials=total_trials,
                    elapsed_seconds=float(time.monotonic() - started_at),
                    estimated_remaining_seconds=0.0,
                    current_best_mean_r2=None,
                    current_best_std_r2=None,
                    stage="failed",
                ),
            )
            return OptimizationResult(
                model_name=str(model_name),
                best_params={},
                selected_trial_number=None,
                inner_cv={
                    "mean_r2": None,
                    "std_r2": None,
                    "min_r2": None,
                    "fold_scores": [],
                    "completed_folds": 0,
                },
                independent_test={"evaluated": False},
                train_indices=X_train.index.tolist(),
                test_indices=X_test.index.tolist(),
                fold_source_indices=fold_source_indices,
                feature_columns=feature_columns,
                process_pls_workflow_hash=workflow_hash,
                evaluation_config=asdict(config),
                trial_summary=summary,
                failure_reasons=failure_reasons,
                study=study,
                status="failed",
                message="全部 trial 无效，请查看失败原因并检查模型、特征或数据",
            )

        selected_scores = [
            float(score) for score in selected_trial.user_attrs.get("fold_scores", [])
        ]
        mean_cv_r2 = float(selected_trial.user_attrs["mean_cv_r2"])
        std_cv_r2 = float(selected_trial.user_attrs.get("std_cv_r2", 0.0))
        min_cv_r2 = float(selected_trial.user_attrs.get("min_cv_r2", np.nan))
        best_params = dict(selected_trial.params)

        try:
            final_pipeline = self.trainer.build_regression_cv_pipeline(
                model_name,
                feature_columns,
                random_state=int(config.random_state),
                process_pls_config=config.process_pls_config,
                use_process_pls=bool(config.use_process_pls),
                **best_params,
            )
            final_pipeline.fit(X_train, y_train)
            train_pred = final_pipeline.predict(X_train)
            test_pred = final_pipeline.predict(X_test)
            train_r2 = float(r2_score(y_train, train_pred))
            test_r2 = float(r2_score(y_test, test_pred))
            test_rmse = _rmse(y_test, test_pred)
            test_mae = float(mean_absolute_error(y_test, test_pred))
            independent_test = {
                "evaluated": True,
                "r2": test_r2,
                "rmse": test_rmse,
                "mae": test_mae,
                "train_r2": train_r2,
                "cv_test_gap": float(mean_cv_r2 - test_r2),
            }
            status = "completed"
            message = "可靠模式优化完成，独立测试集仅评估一次"
        except Exception as exc:
            reason = str(exc).strip()[:300] or exc.__class__.__name__
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            independent_test = {"evaluated": False}
            status = "failed"
            message = f"最佳 trial 在独立测试集评估时失败：{reason}"

        return OptimizationResult(
            model_name=str(model_name),
            best_params=best_params,
            selected_trial_number=int(selected_trial.number),
            inner_cv={
                "mean_r2": mean_cv_r2,
                "std_r2": std_cv_r2,
                "min_r2": min_cv_r2,
                "fold_scores": selected_scores,
                "completed_folds": int(
                    selected_trial.user_attrs.get("completed_folds", len(selected_scores))
                ),
            },
            independent_test=independent_test,
            train_indices=X_train.index.tolist(),
            test_indices=X_test.index.tolist(),
            fold_source_indices=fold_source_indices,
            feature_columns=feature_columns,
            process_pls_workflow_hash=workflow_hash,
            evaluation_config=asdict(config),
            trial_summary=summary,
            failure_reasons=failure_reasons,
            study=study,
            status=status,
            message=message,
        )

    def _optimize_exploratory(
        self,
        model_name,
        X,
        y,
        *,
        n_trials,
        config,
        cv_strategy,
        progress_callback,
        fast_mode,
        use_pruner,
        timeout,
        optimization_method,
    ):
        y_name = getattr(y, "name", None)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(X, pd.DataFrame):
            X_work = X.copy()
        else:
            X_array = np.asarray(X)
            if X_array.ndim != 2:
                raise ValueError("特征 X 必须是二维数组或 DataFrame")
            X_work = pd.DataFrame(
                X_array,
                columns=[f"Feature_{index}" for index in range(X_array.shape[1])],
            )

        if len(X_work) != len(y):
            raise ValueError("特征 X 与目标 y 的样本数必须一致")
        if config.max_samples is not None and int(config.max_samples) > 0:
            max_samples = int(config.max_samples)
            if len(X_work) > max_samples:
                rng = np.random.default_rng(int(config.random_state))
                positions = rng.choice(len(X_work), size=max_samples, replace=False)
                X_work = X_work.iloc[positions]
                y = pd.Series(y).iloc[positions]

        target_values = pd.to_numeric(pd.Series(y), errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        valid_mask = np.isfinite(target_values.to_numpy(dtype=float))
        X_work = X_work.iloc[np.flatnonzero(valid_mask)].copy()
        y_work = target_values.iloc[np.flatnonzero(valid_mask)].copy()
        feature_columns = X_work.columns.tolist()
        all_indices = X_work.index.tolist()
        strategy = str(cv_strategy or "kfold").lower()
        if strategy.startswith("hold"):
            train_idx, valid_idx = train_test_split(
                np.arange(len(X_work)),
                test_size=float(config.test_size),
                random_state=int(config.random_state),
            )
            fixed_splits = [(np.asarray(train_idx), np.asarray(valid_idx))]
        else:
            cv_obj = KFold(
                n_splits=int(config.cv_folds),
                shuffle=True,
                random_state=int(config.random_state),
            )
            fixed_splits = list(cv_obj.split(X_work, y_work))

        fold_source_indices = [
            {
                "train_indices": X_work.index.take(train_idx).tolist(),
                "valid_indices": X_work.index.take(valid_idx).tolist(),
            }
            for train_idx, valid_idx in fixed_splits
        ]
        workflow_hash = _valid_process_pls_workflow_hash(config)
        total_trials = max(1, int(n_trials))
        started_at = time.monotonic()

        def build_legacy_pipeline(params):
            base_model = self.trainer._get_model(
                model_name,
                random_state=int(config.random_state),
                **params,
            )
            if model_name == "Epoxy PINN (Physics-Informed)":
                return base_model
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                base_model,
            )

        def set_score_attrs(trial, scores):
            trial.set_user_attr("fold_scores", [float(score) for score in scores])
            trial.set_user_attr("completed_folds", int(len(scores)))
            if scores:
                trial.set_user_attr("mean_cv_r2", float(np.mean(scores)))
                trial.set_user_attr(
                    "std_cv_r2",
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                )
                trial.set_user_attr("min_cv_r2", float(np.min(scores)))

        def objective(trial):
            scores = []
            try:
                params = self.get_model_params(
                    trial,
                    model_name,
                    fast_mode=bool(fast_mode),
                )
                if model_name == "Epoxy PINN (Physics-Informed)" and y_name is not None:
                    params.setdefault("target_name", y_name)
                for fold_index, (train_idx, valid_idx) in enumerate(fixed_splits):
                    pipeline = build_legacy_pipeline(params)
                    pipeline.fit(X_work.iloc[train_idx], y_work.iloc[train_idx])
                    y_pred = pipeline.predict(X_work.iloc[valid_idx])
                    score = float(r2_score(y_work.iloc[valid_idx], y_pred))
                    if not np.isfinite(score):
                        raise ValueError(f"fold {fold_index + 1} produced non-finite R²")
                    scores.append(score)
                    if use_pruner:
                        trial.report(float(np.mean(scores)), step=int(fold_index))
                        if trial.should_prune():
                            set_score_attrs(trial, scores)
                            trial.set_user_attr("failure_reason", None)
                            raise optuna.TrialPruned()
                set_score_attrs(trial, scores)
                trial.set_user_attr("failure_reason", None)
                return float(np.mean(scores))
            except optuna.TrialPruned:
                raise
            except TrialEvaluationError:
                raise
            except Exception as exc:
                reason = str(exc).strip()[:300] or exc.__class__.__name__
                trial.set_user_attr("failure_reason", reason)
                if scores:
                    set_score_attrs(trial, scores)
                raise TrialEvaluationError(reason) from exc

        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=3)
            if use_pruner
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=self._create_sampler(optimization_method, config.random_state),
        )

        def trial_finished_callback(current_study, _trial):
            summary = _trial_state_summary(current_study)
            elapsed = float(time.monotonic() - started_at)
            finished = summary["completed"] + summary["pruned"] + summary["failed"]
            estimated = (
                float(elapsed / finished * max(0, total_trials - finished))
                if finished > 0 and finished < total_trials
                else (0.0 if finished >= total_trials else None)
            )
            complete_trials = [
                item
                for item in current_study.trials
                if item.state == optuna.trial.TrialState.COMPLETE
                and np.isfinite(item.user_attrs.get("mean_cv_r2", np.nan))
            ]
            best_mean = None
            best_std = None
            if complete_trials:
                best_item = max(
                    complete_trials,
                    key=lambda item: float(item.user_attrs["mean_cv_r2"]),
                )
                best_mean = float(best_item.user_attrs["mean_cv_r2"])
                best_std = float(best_item.user_attrs.get("std_cv_r2", np.nan))
            _emit_optimization_progress(
                progress_callback,
                OptimizationProgress(
                    completed_trials=summary["completed"],
                    pruned_trials=summary["pruned"],
                    failed_trials=summary["failed"],
                    total_trials=total_trials,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=estimated,
                    current_best_mean_r2=best_mean,
                    current_best_std_r2=best_std,
                    stage="completed" if finished >= total_trials else "running",
                ),
            )

        study.optimize(
            objective,
            n_trials=total_trials,
            timeout=timeout,
            callbacks=[trial_finished_callback],
            catch=(TrialEvaluationError,),
        )
        summary = _trial_state_summary(study)
        failure_reasons = _failure_reason_summary(study)
        selected_trial = select_stable_trial(study.trials, config.stability_tolerance)
        independent_test = {
            "evaluated": False,
            "label": "探索模式，不可作为最终泛化报告",
        }
        if selected_trial is None:
            return OptimizationResult(
                model_name=str(model_name),
                best_params={},
                selected_trial_number=None,
                inner_cv={
                    "mean_r2": None,
                    "std_r2": None,
                    "min_r2": None,
                    "fold_scores": [],
                    "completed_folds": 0,
                },
                independent_test=independent_test,
                train_indices=all_indices,
                test_indices=[],
                fold_source_indices=fold_source_indices,
                feature_columns=feature_columns,
                process_pls_workflow_hash=workflow_hash,
                evaluation_config=asdict(config),
                trial_summary=summary,
                failure_reasons=failure_reasons,
                study=study,
                status="failed",
                message="全部 trial 无效，请查看失败原因并检查模型、特征或数据",
            )

        selected_scores = [
            float(score) for score in selected_trial.user_attrs.get("fold_scores", [])
        ]
        return OptimizationResult(
            model_name=str(model_name),
            best_params=dict(selected_trial.params),
            selected_trial_number=int(selected_trial.number),
            inner_cv={
                "mean_r2": float(selected_trial.user_attrs["mean_cv_r2"]),
                "std_r2": float(selected_trial.user_attrs.get("std_cv_r2", 0.0)),
                "min_r2": float(selected_trial.user_attrs.get("min_cv_r2", np.nan)),
                "fold_scores": selected_scores,
                "completed_folds": int(
                    selected_trial.user_attrs.get("completed_folds", len(selected_scores))
                ),
            },
            independent_test=independent_test,
            train_indices=all_indices,
            test_indices=[],
            fold_source_indices=fold_source_indices,
            feature_columns=feature_columns,
            process_pls_workflow_hash=workflow_hash,
            evaluation_config=asdict(config),
            trial_summary=summary,
            failure_reasons=failure_reasons,
            study=study,
            status="completed",
            message="探索模式优化完成；当前分数不代表独立测试集泛化性能",
        )

    def optimize(
        self,
        model_name,
        X,
        y,
        n_trials=50,
        cv=5,
        random_state=42,
        progress_callback=None,
        cv_strategy: str | None = None,
        val_size: float = 0.2,
        max_samples: int | None = None,
        n_jobs: int = -1,
        fast_mode: bool = False,
        use_pruner: bool = True,
        timeout: int | None = None,
        optimization_method: str = "tpe",
        evaluation_config: OptimizationEvaluationConfig | dict[str, Any] | None = None,
    ):
        config = _coerce_evaluation_config(
            evaluation_config,
            cv=cv,
            random_state=random_state,
            val_size=val_size,
            max_samples=max_samples,
            cv_strategy=cv_strategy,
        )
        config.validate()
        if config.mode == "exploratory":
            return self._optimize_exploratory(
                model_name,
                X,
                y,
                n_trials=n_trials,
                config=config,
                cv_strategy=cv_strategy,
                progress_callback=progress_callback,
                fast_mode=fast_mode,
                use_pruner=use_pruner,
                timeout=timeout,
                optimization_method=optimization_method,
            )
        return self._optimize_reliable(
            model_name,
            X,
            y,
            n_trials=n_trials,
            config=config,
            progress_callback=progress_callback,
            fast_mode=fast_mode,
            use_pruner=use_pruner,
            timeout=timeout,
            optimization_method=optimization_method,
        )


class InverseDesigner:
    """反向设计器 (占位，预留未来功能)"""

    def __init__(self):
        pass


def generate_tuning_suggestions(model_name, current_score):
    """生成调参建议"""
    return f"建议增加 {model_name} 的搜索空间或增加迭代次数。"
