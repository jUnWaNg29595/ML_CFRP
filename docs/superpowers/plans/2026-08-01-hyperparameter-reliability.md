# 超参数优化可靠性基线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将回归模型的 Optuna 超参数优化升级为保留独立测试集、分位数分层交叉验证、折内预处理且可解释的科研评估基线。

**Architecture:** `core/optimizer.py` 负责目标清理、自适应连续目标分层、外层训练/测试切分、Optuna trial 统计、稳定性选参和最终独立测试评估；它始终保留 `DataFrame` 列名。`core/model_trainer.py` 提供唯一的可复用回归 pipeline builder，使插补、无穷值清理、标准化和可选工艺 PLS 都只能在每个折的训练行拟合。`app.py` 只负责可靠性设置、优化前检查、真实 trial 进度、结果渲染和把结构化结果留在 `st.session_state`。

**Tech Stack:** Python 3.10、pandas、NumPy、scikit-learn、Optuna、Streamlit、pytest。

## Global Constraints

- 默认模式必须是“可信优化基线”；Holdout/KFold 快速评估只能作为明确标注为“探索模式”的非最终报告。
- 仅实现回归模型；不得在本期加入嵌套 CV、分类调参、Pareto、多进程/分布式 trial、配方分组隔离或时间分组隔离。
- 独立测试集比例默认 `0.20`，内层折数默认 `5`，随机种子默认 `42`，稳定性容差默认 `0.005` R²。
- `max_samples` 只能在外层切分之后对训练部分做分层采样；独立测试集不得参与抽样、trial 拟合、trial 验证或 Optuna 剪枝。
- 每个内层折必须独立 `fit` 缺失值处理、无穷值清理、标准化和已启用的工艺 PLS；验证折和独立测试集只能 `transform` 或 `predict`。
- 保持 `HyperparameterOptimizer` 名称，并保留旧调用参数的兼容映射；不得修改既有模型 artifact 或历史优化记录。
- 所有新 UI 文案使用中文；任何“探索模式”分数都必须显示“不可作为最终泛化报告”。
- PowerShell 命令中的字符串值使用单引号；不要纳入 `.merge-backups/`、`backups/`、`cache/` 或现有备份脚本。

---

## File Structure

- `core/optimizer.py`：新增可靠性配置、预检与分层 helper、结构化优化结果、稳定性选择，以及可信/探索模式的执行分支。
- `core/model_trainer.py`：新增面向 Optuna 的可复用回归 pipeline builder，复用现有 `_make_process_pls_step`、`InfCleaner`、`AllNaNColumnDropper` 与 `_get_model`。
- `app.py`：将超参优化页替换为“预检 → 可靠性设置 → 真实进度 → 内层 CV / 独立测试分区”的界面，并持久化结构化结果。
- `tests/test_optimizer.py`：新增可靠性优化器的单元与集成测试。
- `tests/test_app_scope_regressions.py`：增加页面静态回归测试，确保不再在结果区二次随机切分。

### Task 1: 建立可靠性配置、目标清理与自适应分层

**Files:**
- Create: `tests/test_optimizer.py`
- Modify: `core/optimizer.py:9-25`
- Modify: `core/optimizer.py:198-392`

**Interfaces:**
- Consumes: `X: pandas.DataFrame`、`y: pandas.Series | numpy.ndarray` 和页面的基础设置。
- Produces:
  - `OptimizationEvaluationConfig(test_size: float = 0.20, cv_folds: int = 5, quantile_bins: int | None = None, random_state: int = 42, max_samples: int | None = None, stability_tolerance: float = 0.005, use_process_pls: bool = False, process_pls_config: dict | None = None, mode: str = 'reliable')`
  - `OptimizationPreflight`，包含 `X`、`y`、原始行索引、分层标签、实际分箱数、移除目标行数、外层训练/测试行索引和可读摘要。
  - `prepare_regression_optimization(X, y, config) -> OptimizationPreflight`
  - `build_adaptive_regression_strata(y, cv_folds, test_size, requested_bins=None) -> tuple[numpy.ndarray, int]`
  - `select_stratified_training_budget(preflight, config) -> OptimizationPreflight`

- [x] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from core.optimizer import (
    OptimizationEvaluationConfig,
    build_adaptive_regression_strata,
    prepare_regression_optimization,
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
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py' -q
```

Expected: FAIL during collection because `OptimizationEvaluationConfig`, `build_adaptive_regression_strata` and `prepare_regression_optimization` do not exist.

- [x] **Step 3: Write minimal implementation**

Replace the optimizer’s early NumPy conversion and global random sampling with the following concrete data boundary API:

```python
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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

    def summary(self) -> dict[str, Any]:
        return {
            "original_rows": int(len(self.source_indices) + self.removed_target_rows),
            "valid_target_rows": int(len(self.y)),
            "removed_target_rows": int(self.removed_target_rows),
            "quantile_bins": int(self.quantile_bins),
            "outer_train_rows": int(len(self.outer_train_indices)),
            "outer_test_rows": int(len(self.outer_test_indices)),
        }
```

Implement `prepare_regression_optimization` in this order:

1. Convert `X` to a copied `DataFrame`; for ndarray input create ordered `Feature_0` through `Feature_(n-1)` column names.
2. Convert `y` with `pd.to_numeric(errors='coerce')`, replace `±inf` with `NaN`, and remove only invalid target rows from `X` and `y`.
3. Preserve the retained original index as `source_indices`; do not drop rows because input features contain `NaN` or `±inf`.
4. Compute the initial candidate bin count as `min(10, max(2, len(y) // (2 * cv_folds)))` unless `quantile_bins` is explicitly supplied.
5. In `build_adaptive_regression_strata`, call `pd.qcut(y_series, q=candidate_bins, duplicates='drop')`; reduce candidate bin count one at a time until every populated bin has at least `max(cv_folds, ceil(1 / test_size))` rows. Raise `ValueError("无法构建满足独立测试集和 {cv_folds} 折交叉验证的连续目标分层；有效样本={n_samples}，请减少折数、降低分箱或补充数据")` instead of falling back to KFold.
6. Use `StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)` on the retained rows and strata. Store the *original* row indexes in `outer_train_indices` and `outer_test_indices`.
7. Implement `select_stratified_training_budget` with the same labels only on `outer_train_indices`; when `max_samples` is positive and smaller than the outer training set, retain every stratum proportionally with `StratifiedShuffleSplit(train_size=max_samples, random_state=config.random_state)`. Never modify `outer_test_indices`.

- [x] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py' -q
```

Expected: PASS with three tests; valid targets retain both named feature columns, every stratum supports the requested folds, and insufficient strata produce the actionable Chinese error.

- [x] **Step 5: Commit**

```powershell
git add 'core/optimizer.py' 'tests/test_optimizer.py'
git commit -m 'feat: add reliable optimization preflight'
```

### Task 2: 复用训练页的折内回归 Pipeline

**Files:**
- Modify: `core/model_trainer.py:761-796`
- Modify: `core/model_trainer.py:1936-2179`
- Modify: `tests/test_optimizer.py`

**Interfaces:**
- Consumes: `model_name: str`、已保留列名的 `feature_columns: list[str]`、`random_state: int`、模型超参数和可选 `process_pls_config`。
- Produces: `EnhancedModelTrainer.build_regression_cv_pipeline(model_name, feature_columns, *, random_state=42, process_pls_config=None, use_process_pls=False, **params) -> sklearn.pipeline.Pipeline`。
- Guarantees: 所有返回的 pipeline 均含 `inf_cleaner`、`imputer`、`nan_col_dropper`、`scaler`、`model`；启用工艺 PLS 时 `process_pls` 为第一个步骤，并使用当前训练页的 `_make_process_pls_step` 校验配置。

- [x] **Step 1: Write the failing test**

Append the following test to `tests/test_optimizer.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py::test_optimizer_pipeline_keeps_pls_and_preprocessing_fold_local' -q
```

Expected: FAIL with `AttributeError` because `build_regression_cv_pipeline` does not exist.

- [x] **Step 3: Write minimal implementation**

Add this method to `EnhancedModelTrainer` immediately before `train_model`:

```python
def build_regression_cv_pipeline(
    self,
    model_name,
    feature_columns,
    *,
    random_state=42,
    process_pls_config=None,
    use_process_pls=False,
    **params,
):
    if str(model_name) in RAW_FRAME_MODEL_NAMES:
        raise ValueError("当前模型不支持通用回归优化 pipeline，请在训练页使用专用训练流程")
    if _is_classification_model(model_name):
        raise ValueError("可信超参数优化当前仅支持回归模型")

    columns = list(feature_columns or [])
    process_pls_step = _make_process_pls_step(
        process_pls_config,
        bool(use_process_pls),
        columns,
    )
    model = self._get_model(
        model_name,
        random_state=int(random_state),
        **dict(params),
    )
    steps = []
    if process_pls_step is not None:
        steps.append(process_pls_step)
    steps.extend([
        ("inf_cleaner", InfCleaner()),
        ("imputer", SimpleImputer(strategy="median")),
        ("nan_col_dropper", AllNaNColumnDropper()),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return Pipeline(steps=steps)
```

Do not add a second preprocessing implementation to `core/optimizer.py`. For models that cannot participate in this sklearn-compatible path, raise the explicit error above during preflight; keep their existing training-page flow unchanged.

- [x] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py::test_optimizer_pipeline_keeps_pls_and_preprocessing_fold_local' 'tests/test_process_pls.py' -q
```

Expected: PASS; the PLS transformer fits exactly once on 60 training rows and the existing PLS workflow tests remain green.

- [x] **Step 5: Commit**

```powershell
git add 'core/model_trainer.py' 'tests/test_optimizer.py'
git commit -m 'refactor: share fold-safe optimization pipeline'
```

### Task 3: 实现固定内层折、稳定性选参和独立测试报告

**Files:**
- Modify: `core/optimizer.py:198-392`
- Modify: `tests/test_optimizer.py`

**Interfaces:**
- Consumes: `HyperparameterOptimizer.optimize(model_name, X, y, n_trials=50, evaluation_config=None, progress_callback=None, cv=5, random_state=42, cv_strategy='kfold', val_size=0.2, max_samples=None, fast_mode=False, use_pruner=True, timeout=None, optimization_method='tpe')`。
- Produces:
  - `OptimizationProgress(completed_trials, pruned_trials, failed_trials, total_trials, elapsed_seconds, estimated_remaining_seconds, current_best_mean_r2, current_best_std_r2, stage)`
  - `OptimizationResult(model_name, best_params, selected_trial_number, inner_cv, independent_test, train_indices, test_indices, fold_source_indices, feature_columns, process_pls_workflow_hash, evaluation_config, trial_summary, failure_reasons, study, status, message)`
  - `select_stable_trial(trials, stability_tolerance) -> optuna.trial.FrozenTrial | None`
- Compatibility: `OptimizationResult.as_legacy_tuple()` returns `(best_params, inner_cv['mean_r2'], study)`; callers that still pass `cv`, `random_state`, `max_samples`, `cv_strategy` or `val_size` without `evaluation_config` are mapped to a config. Default mapping is `mode='reliable'`; only an explicit `cv_strategy='holdout'` maps to `mode='exploratory'`.

- [x] **Step 1: Write the failing test**

Append these tests to `tests/test_optimizer.py`:

```python
from types import SimpleNamespace

import optuna

from core.optimizer import HyperparameterOptimizer, select_stable_trial


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
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py::test_stable_trial_selection_prefers_lower_standard_deviation_within_tolerance' 'tests/test_optimizer.py::test_reliable_optimization_never_uses_outer_test_rows_in_cv' 'tests/test_optimizer.py::test_failed_trials_are_recorded_and_all_failures_return_a_readable_result' -q
```

Expected: FAIL because `select_stable_trial`, `OptimizationResult`, fixed outer-test boundaries and structured failure reporting do not exist.

- [x] **Step 3: Write minimal implementation**

Implement the result and stable-selection contracts:

```python
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
        return self.best_params, self.inner_cv.get("mean_r2"), self.study


def select_stable_trial(trials, stability_tolerance):
    valid = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and np.isfinite(trial.user_attrs.get("mean_cv_r2", np.nan))
        and np.isfinite(trial.user_attrs.get("std_cv_r2", np.nan))
        and np.isfinite(trial.user_attrs.get("min_cv_r2", np.nan))
    ]
    if not valid:
        return None
    best_mean = max(float(t.user_attrs["mean_cv_r2"]) for t in valid)
    candidates = [
        trial
        for trial in valid
        if best_mean - float(trial.user_attrs["mean_cv_r2"]) <= float(stability_tolerance)
    ]
    return min(
        candidates,
        key=lambda trial: (
            float(trial.user_attrs["std_cv_r2"]),
            -float(trial.user_attrs["min_cv_r2"]),
            int(trial.number),
        ),
    )
```

Replace the existing `objective` implementation with this exact control flow:

1. Build `preflight = prepare_regression_optimization(X, y, config)` and derive `X_train`, `y_train` only from `preflight.outer_train_indices`; apply the optional training-only budget after that split.
2. Build one `StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)` split list from the outer-training strata before starting Optuna. Convert each split to original source indexes and save it in `fold_source_indices`.
3. For each trial, call `get_model_params`, then for each fixed fold call `self.trainer.build_regression_cv_pipeline(model_name, preflight.X.columns.tolist(), random_state=config.random_state, process_pls_config=config.process_pls_config, use_process_pls=config.use_process_pls, **params)`, `fit(X_fold_train, y_fold_train)` and `predict(X_fold_valid)`. Compute one R² per fold.
4. After every completed fold set `trial.report(float(np.mean(fold_scores)), step=fold_index)`. When pruning is enabled and `trial.should_prune()` is true, persist the completed scores in `user_attrs` then raise `optuna.TrialPruned()`.
5. On successful completion set these JSON-safe `user_attrs`: `mean_cv_r2`, `std_cv_r2`, `min_cv_r2`, `fold_scores`, `completed_folds`, `failure_reason=None`. Return `mean_cv_r2`.
6. On a non-pruning exception, set `failure_reason` to the first 300 characters of the exception, then raise a local `TrialEvaluationError`. Invoke `study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[trial_finished_callback], catch=(TrialEvaluationError,))`; do not return `-inf`.
7. Use an Optuna callback invoked after a trial becomes complete, pruned or failed to calculate exact state counts, elapsed time and estimated remaining time. Emit `OptimizationProgress` only from this callback, not when an objective starts.
8. Select the final trial using `select_stable_trial`; never use `study.best_params` as the final decision. If no trial is valid, return `OptimizationResult(model_name=model_name, best_params={}, selected_trial_number=None, inner_cv={'mean_r2': None, 'std_r2': None, 'min_r2': None, 'fold_scores': [], 'completed_folds': 0}, independent_test={'evaluated': False}, train_indices=list(preflight.outer_train_indices), test_indices=list(preflight.outer_test_indices), fold_source_indices=fold_source_indices, feature_columns=preflight.X.columns.tolist(), process_pls_workflow_hash=workflow_hash, evaluation_config=asdict(config), trial_summary=trial_summary, failure_reasons=failure_reasons, study=study, status='failed', message='全部 trial 无效，请查看失败原因并检查模型、特征或数据')` and aggregate identical failure strings into `failure_reasons`.
9. Only after a selected trial exists, build one fresh pipeline on the complete outer training set, fit once, and evaluate the untouched outer test set once with `r2_score(y_test, y_pred_test)`, `mean_squared_error(y_test, y_pred_test, squared=False)` and `mean_absolute_error(y_test, y_pred_test)`. Store `r2`, `rmse`, `mae`, `evaluated=True`, plus `train_r2` and `cv_test_gap=mean_cv_r2-test_r2`.
10. In explicit exploratory mode preserve the old Holdout/KFold calculation in a separate private `_optimize_exploratory` branch, return `independent_test={'evaluated': False, 'label': '探索模式，不可作为最终泛化报告'}`, and never label its score as an independent-test metric.

Import `fingerprint_process_pls_workflow` from `core.process_pls` and set `process_pls_workflow_hash` only when `config.use_process_pls` is true and its configuration is valid. Keep the selected feature-column order exactly as `preflight.X.columns.tolist()`.

- [x] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_optimizer.py' 'tests/test_process_pls.py' 'tests/test_missing_target_training.py' -q
```

Expected: PASS; stable trial ranking follows the `0.005` tolerance rule, no outer-test source row reaches an inner fold, PLS remains fold-local, and complete failure produces a result object rather than a `study.best_params` exception.

- [x] **Step 5: Commit**

```powershell
git add 'core/optimizer.py' 'tests/test_optimizer.py'
git commit -m 'feat: add stable nested holdout optimization'
```

### Task 4: 将超参优化页改为科研结果分区并持久化结果

**Files:**
- Modify: `app.py:2890-2905`
- Modify: `app.py:20946-21425`
- Modify: `tests/test_app_scope_regressions.py`

**Interfaces:**
- Consumes: `OptimizationEvaluationConfig`、`OptimizationPreflight`、`OptimizationProgress` 和 `OptimizationResult`。
- Produces:
  - `st.session_state.optimization_result: OptimizationResult | None`
  - `st.session_state.best_params`、`st.session_state.best_score`、`st.session_state.optimized_model_name`，作为既有训练页的兼容镜像。
  - 一个不重新运行 `study.optimize` 的结果区；其内容由 `optimization_result` 渲染。

- [x] **Step 1: Write the failing test**

Append this static regression test to `tests/test_app_scope_regressions.py`:

```python
from pathlib import Path


def test_hyperparameter_page_uses_persisted_result_without_second_random_split():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    start = source.index("def page_hyperparameter_optimization():")
    end = source.index("\n\n# ============================================================\n# 页面：主动学习", start)
    page_source = source[start:end]

    assert "optimization_result" in page_source
    assert "可信优化基线" in page_source
    assert "独立测试集结果" in page_source
    assert "未参与调参" in page_source
    assert "train_test_split(" not in page_source
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_app_scope_regressions.py::test_hyperparameter_page_uses_persisted_result_without_second_random_split' -q
```

Expected: FAIL because the page has no `optimization_result` state, still defaults to Holdout, and its “使用最优参数评估” branch calls `train_test_split`.

- [x] **Step 3: Write minimal implementation**

1. Update the `app.py` optimizer import at line 1074 to import `OptimizationEvaluationConfig`, `OptimizationProgress` and `prepare_regression_optimization` alongside `HyperparameterOptimizer`. Add `"optimization_result": None` to the existing session defaults near `best_params`; do not delete the existing compatibility keys.
2. At the start of `page_hyperparameter_optimization`, obtain the current process-PLS configuration exactly as training does:

```python
process_pls_workflow = st.session_state.get("process_pls_workflow")
use_process_pls = bool(
    isinstance(process_pls_workflow, dict)
    and process_pls_workflow.get("enabled")
    and st.session_state.get(
        "process_pls_use_in_training",
        st.session_state.get("process_pls_enabled_default", False),
    )
)
```

3. Replace the “优化加速设置” expander with a `st.form(key='optimization_reliability_form')` containing:
   - mode selector with `["可信优化基线", "探索模式（快速评估）"]`, defaulting to the first item;
   - independent-test ratio, CV folds, random seed, training-only sample budget, trial count, timeout and pruning controls;
   - exploratory-only Holdout/KFold selector and the immutable warning `⚠️ 探索模式结果不可作为最终泛化报告`;
   - a disabled start button when TensorFlow is missing, preflight fails, no numeric target is available, or the model is unsupported by the pipeline builder.
4. Before the start button, call `prepare_regression_optimization(X, y, config)` inside `try/except ValueError` and show one compact “优化前检查” block with raw rows, valid target rows, removed rows, actual quantile bins, outer-train rows, independent-test rows, fold count, random seed and budget. On error, show the exact exception and do not run optimization.
5. Pass a callback with the concrete signature `def update_progress(progress: OptimizationProgress) -> None:`. Update the progress bar from `(completed + pruned + failed) / total`, and render `完成 / 剪枝 / 失败`、`当前稳定性最佳 mean±std R²`、elapsed/ETA and stage. Do not emit counts before a trial changes state.
6. On completion store the full `OptimizationResult` in `st.session_state.optimization_result`, then mirror `best_params`, `inner_cv['mean_r2']` and `model_name` to the existing compatibility keys. Store `None` for the result before a new valid optimization starts so stale results cannot be mistaken for current data.
7. Replace the current “最佳参数 + 可视化 + 使用最优参数评估” layout with fixed sections:
   - **内层交叉验证结果（参数选择依据）**: mean R², std R², minimum R², each fold score, selected trial number, best params and a compact trial-state table.
   - **独立测试集结果（未参与调参）**: R², RMSE, MAE, training R² and CV/test gap. In exploratory mode show only its warning label.
   - **试验状态与失败原因**: completed/pruned/failed counts plus the aggregated failure table from `result.failure_reasons`.
   - **导出**: one CSV built from completed trial `user_attrs`, and one JSON download generated from a dictionary containing only serializable result metadata, metrics, settings and selected parameters.
8. Remove the “使用最优参数评估” button and all `train_test_split` use from this page. The existing “使用最佳参数训练模型” button stays, trains the selected model on all valid current data, and must precede with `if not result.best_params: st.error("没有可用的优化参数，请先完成至少一个有效 trial")`.
9. Update `log_fe_step` to use `result.inner_cv['mean_r2']`, `result.evaluation_config` metadata and `result.trial_summary`; do not log an exploratory score as `best_r2` without an `exploratory=True` marker.

- [x] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_app_scope_regressions.py::test_hyperparameter_page_uses_persisted_result_without_second_random_split' 'tests/test_optimizer.py' -q
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile 'app.py' 'core\optimizer.py' 'core\model_trainer.py'
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest -q --ignore='backups' --ignore='.merge-backups'
```

Expected: PASS; the source no longer contains a second random split in this page, the baseline wording is present, all modified Python modules compile, and no non-ignored repository regression fails.

- [x] **Step 5: Update verification record（不提交）**

After the commands in Step 4 pass, change the header in `docs/superpowers/specs/2026-08-01-hyperparameter-reliability-design.md` to `**状态：** 已实现并完成回归验证`. Append a `## 验证记录` section that records the actual passing commands and states: the default evaluation is outer stratified holdout plus inner stratified KFold; the untouched independent test set is evaluated once after stable-trial selection; and exploratory Holdout/KFold scores are non-final.

```powershell
git add 'app.py' 'tests/test_app_scope_regressions.py' 'docs/superpowers/specs/2026-08-01-hyperparameter-reliability-design.md' 'docs/superpowers/plans/2026-08-01-hyperparameter-reliability.md'
git commit -m 'feat: present reliable optimization results'
```

## Plan Self-Review

### Spec coverage

- 连续目标自适应分位数分层、目标清理、独立测试集和训练侧预算：Task 1。
- 折内缺失值处理、标准化和工艺 PLS 拟合：Task 2。
- 固定内层折、trial 属性、剪枝/失败统计、稳定性排序、独立测试一次性评估、探索模式兼容：Task 3。
- 优化前检查、可靠性设置、真实进度、分区结果、失败原因、下载、刷新不重跑、全数据正式训练入口：Task 4。
- 全量回归、编译和实施状态记录：Task 4。

### Placeholder scan

已检查本计划；没有任何待定标记、延后实现标记、笼统错误处理要求、空泛测试要求或跨任务省略指令。

### Type consistency

- `OptimizationEvaluationConfig` 由 Task 1 定义，并由 Task 3 的 `optimize` 与 Task 4 的页面构造。
- `OptimizationPreflight` 由 Task 1 定义，并作为 Task 3 的固定外层边界输入。
- `build_regression_cv_pipeline` 由 Task 2 定义，并由 Task 3 的每一个内层折调用。
- `OptimizationProgress` 与 `OptimizationResult` 由 Task 3 定义，并由 Task 4 的进度与结果区消费。
- `OptimizationResult.as_legacy_tuple()` 保留旧三元组消费者的迁移出口；页面迁移完成后不依赖该出口。

## 实施记录（2026-08-01）

- Task 1-3 已完成并提交；优化器已具备可靠预检、折内安全 pipeline、稳定 trial 选择和一次性独立测试评估。
- Task 4 已将超参优化页改为“可信优化基线”默认模式，结果持久化在 `st.session_state.optimization_result`，并移除了页面内二次随机切分评估。
- 已验证命令：`pytest tests\test_app_scope_regressions.py::test_hyperparameter_page_uses_persisted_result_without_second_random_split tests\test_optimizer.py -q` 与 `py_compile app.py core\optimizer.py core\model_trainer.py`。
