# 工艺特征 PLS 工作流 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在“特征选择”页面配置并锁定只作用于工艺特征的 PLS 工作流，并让训练、交叉验证、模型导出、预测和高通量筛选使用同一套无数据泄漏的转换流程。

**Architecture:** 新增可序列化的 `ProcessPLSTransformer`，把工艺特征的缺失处理、RobustScaler、PLS 成分选择、VIP 选择和缺失掩码封装成一个 sklearn 兼容转换器。特征选择页面只保存配置和诊断结果，正式训练时在训练折内重新拟合转换器；最终拟合的转换器放进模型 Pipeline，模型导出额外保存版本化配置和指纹，预测及高通量筛选只调用 `transform`，不允许重新 `fit`。分子特征仍由现有 molecular workflow 负责，工艺 PLS 不改变分子特征提取顺序。

**Tech Stack:** Python 3.10、pandas、NumPy、scikit-learn `BaseEstimator`/`TransformerMixin`/`PLSRegression`/`Pipeline`、Streamlit、joblib、pytest。

## Global Constraints

- PLS 只处理特征选择页识别出的数值工艺/原始特征，不处理 SMILES、BigSMILES、分子指纹、RDKit 分子描述符或目标列。
- 正式训练、交叉验证、预测和高通量筛选不得使用全数据拟合的预览 PLS 对象。
- 每个训练折都必须独立拟合缺失值处理器、RobustScaler、PLS 和 VIP；验证集、测试集和筛选候选只能调用 `transform`。
- 关闭 PLS 时保持当前原始特征路径、分子特征 workflow 和模型输入顺序不变。
- PLS 失败、列缺失、版本不匹配或 workflow 指纹不匹配时必须显式报错，不能静默用全局均值、零值或 MACCS 伪造特征。
- 分类模型和图模型不启用普通回归 PLS；训练页只对支持的回归模型显示轻量启用选项。
- 新增状态键、artifact 字段和输出列名必须使用 `process_pls_` 前缀，避免与现有 `molecular_feature_*`、PCA 和后处理映射字段冲突。
- 不修改 `backups/`、`cache/`、`catboost_info/`、`.bak` 文件、临时脚本或用户未提交的无关文件。
- 所有新增测试使用现有 pytest 配置；PowerShell 命令中的字符串值使用单引号。

---

### Task 1: 建立可序列化的工艺 PLS 转换器

**Files:**
- Create: `core/process_pls.py`
- Test: `tests/test_process_pls.py`

**Interfaces:**
- Produces `ProcessPLSTransformer(process_feature_cols, max_components=8, vip_top_k=8, missing_threshold=0.85, random_state=42)`.
- Produces `ProcessPLSTransformer.fit(X: pandas.DataFrame, y) -> ProcessPLSTransformer`.
- Produces `ProcessPLSTransformer.transform(X: pandas.DataFrame) -> pandas.DataFrame`.
- Produces `ProcessPLSTransformer.get_feature_names_out(input_features=None) -> numpy.ndarray`.
- Produces `select_pls_components_cv(X, y, max_components, cv_splits, random_state) -> dict`.
- Produces `compute_vip_scores(pls_model) -> numpy.ndarray`.
- Produces `process_pls_config_to_dict(config) -> dict` and `fingerprint_process_pls_workflow(payload) -> str`.

- [ ] **Step 1: Write the failing unit tests**

```python
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
        'cure_temp': [80.0, 90.0, np.nan, 110.0, 120.0, 130.0],
        'cure_time': [30.0, 40.0, 50.0, np.nan, 70.0, 80.0],
        'resin_MolWt': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    })
    y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    transformer = ProcessPLSTransformer(
        process_feature_cols=['cure_temp', 'cure_time'],
        max_components=2,
        vip_top_k=1,
        random_state=42,
    ).fit(X, y)
    result = transformer.transform(X)

    assert list(result.columns) == transformer.get_feature_names_out().tolist()
    assert 'process_pls_1' in result.columns
    assert any(column.endswith('__missing') for column in result.columns)
    assert 'resin_MolWt' in result.columns
    assert np.isfinite(result.to_numpy(dtype=float)).all()


def test_process_pls_does_not_refit_on_transform():
    train = pd.DataFrame({
        'temperature': [1.0, 2.0, 3.0, np.nan],
        'time': [10.0, 11.0, 12.0, 13.0],
    })
    test = pd.DataFrame({'temperature': [1000.0], 'time': [999.0]})
    y = np.array([1.0, 2.0, 3.0, 4.0])

    transformer = ProcessPLSTransformer(
        process_feature_cols=['temperature', 'time'],
        max_components=1,
        random_state=42,
    ).fit(train, y)
    imputer_statistics = transformer.imputer_.statistics_.copy()
    transformer.transform(test)

    np.testing.assert_array_equal(transformer.imputer_.statistics_, imputer_statistics)


def test_process_pls_rejects_missing_required_columns():
    transformer = ProcessPLSTransformer(process_feature_cols=['temperature'])
    with pytest.raises(ValueError, match='missing required process columns'):
        transformer.fit(pd.DataFrame({'time': [1.0, 2.0]}), np.array([1.0, 2.0]))


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
        'schema_version': 1,
        'process_feature_cols': ['temperature', 'time'],
        'output_feature_names': ['process_pls_1'],
    })
    second = fingerprint_process_pls_workflow({
        'schema_version': 1,
        'process_feature_cols': ['time', 'temperature'],
        'output_feature_names': ['process_pls_1'],
    })
    assert first != second
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
pytest -q 'tests/test_process_pls.py'
```

Expected: collection or import failure because `core/process_pls.py` does not exist.

- [ ] **Step 3: Implement the minimal transformer**

Implement the following behavior in `core/process_pls.py`:

```python
PROCESS_PLS_SCHEMA_VERSION = 1


class ProcessPLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        process_feature_cols,
        max_components=8,
        vip_top_k=8,
        missing_threshold=0.85,
        random_state=42,
        cv_splits=5,
    ):
        self.process_feature_cols = list(process_feature_cols or [])
        self.max_components = int(max_components)
        self.vip_top_k = int(vip_top_k)
        self.missing_threshold = float(missing_threshold)
        self.random_state = int(random_state)
        self.cv_splits = int(cv_splits)

    def fit(self, X, y):
        frame = _coerce_numeric_frame(X)
        self.input_feature_cols_ = frame.columns.tolist()
        missing = [column for column in self.process_feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f'missing required process columns: {", ".join(missing[:12])}'
            )
        self.kept_process_feature_cols_ = [
            column for column in self.process_feature_cols
            if frame[column].notna().mean() >= 1.0 - self.missing_threshold
        ]
        if not self.kept_process_feature_cols_:
            raise ValueError('no process feature remains after missingness filtering')
        y_array = _coerce_finite_target(y)
        process_frame = frame[self.kept_process_feature_cols_].copy()
        self.missing_mask_cols_ = [
            f'{column}__missing' for column in self.kept_process_feature_cols_
        ]
        self.imputer_ = SimpleImputer(strategy='median')
        imputed = self.imputer_.fit_transform(process_frame)
        self.scaler_ = RobustScaler()
        scaled = self.scaler_.fit_transform(imputed)
        self.n_components_, self.cv_report_ = select_pls_components_cv(
            scaled,
            y_array,
            max_components=self.max_components,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
        )
        self.pls_ = PLSRegression(n_components=self.n_components_)
        self.pls_.fit(scaled, y_array)
        self.vip_scores_ = compute_vip_scores(self.pls_)
        order = np.argsort(-self.vip_scores_)
        self.selected_original_features_ = [
            self.kept_process_feature_cols_[index]
            for index in order[:min(self.vip_top_k, len(order))]
        ]
        self.output_feature_names_ = (
            [f'process_pls_{index + 1}' for index in range(self.n_components_)]
            + self.selected_original_features_
            + self.missing_mask_cols_
            + [
                column for column in self.input_feature_cols_
                if column not in self.kept_process_feature_cols_
            ]
        )
        self.workflow_hash_ = fingerprint_process_pls_workflow(self.to_workflow_dict())
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ['input_feature_cols_', 'imputer_', 'scaler_', 'pls_', 'output_feature_names_'],
        )
        frame = _coerce_numeric_frame(X)
        missing = [column for column in self.process_feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f'missing required process columns: {", ".join(missing[:12])}'
            )
        frame = frame.reindex(columns=self.input_feature_cols_)
        process_frame = frame[self.kept_process_feature_cols_]
        masks = process_frame.isna().astype(float)
        scaled = self.scaler_.transform(self.imputer_.transform(process_frame))
        components = self.pls_.transform(scaled)
        output = pd.DataFrame(
            components,
            index=frame.index,
            columns=[f'process_pls_{index + 1}' for index in range(self.n_components_)],
        )
        output = pd.concat(
            [
                output,
                frame[self.selected_original_features_].reset_index(drop=True),
                masks.set_axis(self.missing_mask_cols_, axis=1).reset_index(drop=True),
                frame[
                    [
                        column for column in self.input_feature_cols_
                        if column not in self.kept_process_feature_cols_
                    ]
                ].reset_index(drop=True),
            ],
            axis=1,
        )
        return output.reindex(columns=self.output_feature_names_).replace(
            [np.inf, -np.inf], np.nan
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['output_feature_names_'])
        return np.asarray(self.output_feature_names_, dtype=object)

    def to_workflow_dict(self):
        return {
            'schema_version': PROCESS_PLS_SCHEMA_VERSION,
            'process_feature_cols': list(self.process_feature_cols),
            'kept_process_feature_cols': list(self.kept_process_feature_cols_),
            'selected_original_features': list(self.selected_original_features_),
            'missing_mask_cols': list(self.missing_mask_cols_),
            'output_feature_names': list(self.output_feature_names_),
            'n_components': int(self.n_components_),
            'max_components': int(self.max_components),
            'vip_top_k': int(self.vip_top_k),
            'missing_threshold': float(self.missing_threshold),
            'random_state': int(self.random_state),
            'workflow_hash': self.workflow_hash_,
        }
```

`select_pls_components_cv` 必须只接收已由调用者从训练折得到的数据，候选成分数不得超过 `min(n_samples - 1, n_features)`；每个候选值单独执行 K 折拟合，并返回 `cv_r2_mean`、`cv_r2_std`、`cv_rmse_mean`、`cv_rmse_std`、`rmse_improvement` 和 `selection_score`。候选评分使用：

```python
selection_score = (
    0.45 * normalized_cv_r2
    + 0.30 * normalized_rmse_improvement
    + 0.15 * normalized_fold_stability
    + 0.10 * (1.0 - component_count / max_component_count)
)
```

`compute_vip_scores` 使用 Wold VIP 公式，遇到零平方和时返回有限的零分数，不得产生 `NaN` 或 `Inf`。`fingerprint_process_pls_workflow` 使用排序后的 JSON、UTF-8 和 SHA-256，保留列表顺序以检测列顺序变化。

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```powershell
pytest -q 'tests/test_process_pls.py'
```

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated core implementation**

```powershell
git add 'core/process_pls.py' 'tests/test_process_pls.py'
git commit -m 'feat: add leakage-safe process PLS transformer'
```

### Task 2: 在特征选择页面配置、预览并锁定工艺 PLS

**Files:**
- Modify: `core/feature_selector.py:641-980, 2142-2781`
- Modify: `app.py:2060-2090, 2760-2825`
- Test: `tests/test_process_pls.py`

**Interfaces:**
- Consumes `SmartFeatureClassifier` 的 `original_features` 与 `molecular_features` 分类结果。
- Produces `infer_process_feature_candidates(frame, original_features, molecular_features, target_col) -> list[str]`.
- Produces `build_process_pls_config(process_feature_cols, random_state=42) -> dict`.
- Stores `st.session_state.process_pls_workflow` as a versioned configuration dict, not a full-data fitted transformer.
- Stores `st.session_state.process_pls_preview_report` as a DataFrame or serializable dict for UI display.
- Stores `st.session_state.process_pls_enabled_default` as `False` until the user explicitly locks the workflow.

- [ ] **Step 1: Add candidate-exclusion tests**

```python
def test_process_candidate_inference_excludes_molecular_and_text_columns():
    from core.feature_selector import infer_process_feature_candidates

    frame = pd.DataFrame({
        'cure_temperature': [80.0, 90.0],
        'cure_time': [30.0, 40.0],
        'resin_smiles1': ['CCO', 'CCC'],
        'resin_Morgan_0': [0, 1],
        'sample_id': ['A', 'B'],
        'target': [1.0, 2.0],
    })
    result = infer_process_feature_candidates(
        frame,
        original_features=['cure_temperature', 'cure_time', 'resin_smiles1', 'resin_Morgan_0'],
        molecular_features=['resin_smiles1', 'resin_Morgan_0'],
        target_col='target',
    )
    assert result == ['cure_temperature', 'cure_time']
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
pytest -q 'tests/test_process_pls.py::test_process_candidate_inference_excludes_molecular_and_text_columns'
```

Expected: FAIL because `infer_process_feature_candidates` is not defined.

- [ ] **Step 3: Implement candidate inference and feature-selection state**

Add `infer_process_feature_candidates` near the existing classifier helpers. It must:

```python
def infer_process_feature_candidates(
    frame,
    original_features,
    molecular_features,
    target_col,
):
    molecular = set(molecular_features or [])
    excluded = {str(target_col)} if target_col else set()
    result = []
    for column in list(original_features or []):
        if column in molecular or column in excluded:
            continue
        if column.lower().endswith(('_smiles', '_bigsmiles')):
            continue
        if frame[column].dtype == 'object':
            continue
        numeric = pd.to_numeric(frame[column], errors='coerce')
        if numeric.notna().sum() == 0:
            continue
        result.append(column)
    return result
```

在 `render_feature_selector()` 的 PCA tab 后增加“⚗️ 工艺特征 PLS”tab，不删除现有 PCA。界面只显示：

1. 工艺候选列的多选框，默认不自动全选，按数据列原顺序展示；
2. 候选列的缺失率和有效样本数预览；
3. “生成 PLS 诊断”按钮；
4. 诊断表：候选成分数、CV R²、CV RMSE、折间标准差、RMSE 改善、综合得分；
5. VIP Top 结果；
6. “锁定工艺 PLS 工作流”按钮；
7. 当前锁定工作流的一行摘要和“清除锁定工作流”按钮。

PLS 参数采用自动策略，不在此页暴露大块参数区。固定配置写入：

```python
{
    'schema_version': 1,
    'enabled': True,
    'process_feature_cols': selected_process_cols,
    'max_components': 8,
    'vip_top_k': 8,
    'missing_threshold': 0.85,
    'cv_splits': 5,
    'random_state': 42,
    'selection_mode': 'auto_combined_score',
}
```

“生成 PLS 诊断”只在内存中调用 `ProcessPLSTransformer.fit` 进行探索性预览，并明确显示“此预览不用于正式训练”。锁定时只保存配置，不保存全数据拟合的 `imputer_`、`scaler_` 或 `pls_`。锁定前验证：

```python
if len(selected_process_cols) < 2:
    st.error('至少选择 2 个工艺数值特征')
elif not y_available:
    st.error('当前数据没有可用于 PLS 诊断的数值目标列')
elif st.session_state.get('process_pls_preview_report') is None:
    st.error('请先生成 PLS 诊断')
else:
    st.session_state.process_pls_workflow = build_process_pls_config(
        selected_process_cols,
        random_state=42,
    )
    st.session_state.process_pls_enabled_default = False
```

初始化 session state 时加入：

```python
'process_pls_workflow': None,
'process_pls_preview_report': None,
'process_pls_enabled_default': False,
```

- [ ] **Step 4: Run focused tests and a syntax check**

Run:

```powershell
pytest -q 'tests/test_process_pls.py'
python -m py_compile 'core/feature_selector.py' 'app.py'
```

Expected: tests pass and `py_compile` exits with code 0.

- [ ] **Step 5: Commit the feature-selection UI**

```powershell
git add 'core/feature_selector.py' 'app.py' 'tests/test_process_pls.py'
git commit -m 'feat: configure and lock process PLS from feature selection'
```

### Task 3: 将 PLS 插入回归训练 Pipeline，并保持数据泄漏隔离

**Files:**
- Modify: `core/model_trainer.py:1697-1800, 2754-3178, 3377-4175`
- Modify: `app.py` model-training call sites near the existing `EnhancedModelTrainer.train_model` and `cross_validate` calls
- Test: `tests/test_process_pls.py`

**Interfaces:**
- `EnhancedModelTrainer.train_model(..., process_pls_config=None, use_process_pls=False, **params)`.
- `EnhancedModelTrainer.cross_validate(..., process_pls_config=None, use_process_pls=False, **params)`.
- Internal helper `_make_process_pls_step(process_pls_config, enabled) -> tuple[str, ProcessPLSTransformer] | None`.
- Internal helper `_validate_process_pls_config(config, feature_columns) -> None`.

- [ ] **Step 1: Add leakage and train/test separation tests**

```python
def test_process_pls_training_pipeline_fits_only_training_rows():
    from core.model_trainer import EnhancedModelTrainer

    X = pd.DataFrame({
        'temperature': [1.0, 2.0, 3.0, 4.0, 1000.0, 1001.0],
        'time': [10.0, 11.0, 12.0, 13.0, 999.0, 1000.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    config = {
        'schema_version': 1,
        'enabled': True,
        'process_feature_cols': ['temperature', 'time'],
        'max_components': 1,
        'vip_top_k': 1,
        'missing_threshold': 0.85,
        'cv_splits': 2,
        'random_state': 42,
        'selection_mode': 'auto_combined_score',
    }

    trainer = EnhancedModelTrainer()
    result = trainer.train_model(
        X,
        y,
        model_name='线性回归',
        test_size=1 / 3,
        random_state=42,
        process_pls_config=config,
        use_process_pls=True,
    )
    fitted = result['pipeline'].named_steps['process_pls']
    assert fitted.imputer_.statistics_[0] < 100.0
    assert result['feature_names']


def test_process_pls_is_not_applied_when_disabled():
    from core.model_trainer import EnhancedModelTrainer

    X = pd.DataFrame({'temperature': [1.0, 2.0, 3.0, 4.0], 'time': [10.0, 11.0, 12.0, 13.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = EnhancedModelTrainer().train_model(
        X, y, model_name='线性回归', use_process_pls=False
    )
    assert 'process_pls' not in result['pipeline'].named_steps
```

- [ ] **Step 2: Run the tests and verify the new behavior fails**

Run:

```powershell
pytest -q 'tests/test_process_pls.py::test_process_pls_training_pipeline_fits_only_training_rows' 'tests/test_process_pls.py::test_process_pls_is_not_applied_when_disabled'
```

Expected: FAIL because the trainer does not accept or insert `process_pls_config`.

- [ ] **Step 3: Add the pipeline transformer without changing the disabled path**

Extend `train_model` and `cross_validate` with explicit keyword arguments before `**params`:

```python
process_pls_config=None,
use_process_pls=False,
```

For supported regression models, keep the input as a DataFrame until the split is resolved. Build the PLS step only after the train/test indices are known:

```python
process_pls_step = None
if use_process_pls:
    _validate_process_pls_config(process_pls_config, feature_names)
    process_pls_step = (
        'process_pls',
        ProcessPLSTransformer(**_config_to_transformer_kwargs(process_pls_config)),
    )
```

The final normal-regression pipeline must use this order:

```python
steps = []
if process_pls_step is not None:
    steps.append(process_pls_step)
steps.extend([
    ('imputer', imputer or SimpleImputer(strategy='median')),
    ('inf_cleaner', InfCleaner()),
    ('nan_col_dropper', AllNaNColumnDropper()),
])
if scaler is not None:
    steps.append(('scaler', scaler))
steps.append(('model', base_model))
pipeline = Pipeline(steps=steps)
```

Do not pre-fit `ProcessPLSTransformer` on `X` before `_resolve_split`. Fit the pipeline on `X_train_df` and call `pipeline.predict(X_test_df)`. To preserve existing metrics and artifacts, obtain transformed feature names with:

```python
feature_names = list(
    pipeline.named_steps['process_pls'].get_feature_names_out()
    if 'process_pls' in pipeline.named_steps
    else X_train_df.columns
)
```

For `cross_validate`, construct a new `Pipeline` containing a new `ProcessPLSTransformer` inside every fold. The validation fold is passed only to `predict`; it must never be passed to `fit` or the component-selection CV. The existing XGBoost early-stopping path must either use the same PLS-transformed fold matrices or be routed through a pipeline-compatible fit helper; do not leave XGBoost using raw columns when `use_process_pls=True`.

For `Transformer + BNN` and other raw-frame regression models, add the same first-stage transform inside `_train_raw_frame_model` and `_cross_validate_raw_frame_model`. The model must receive the transformed DataFrame with `process_pls_*`, selected original process columns and `__missing` columns. When disabled, retain the existing raw-frame path exactly.

Reject these cases before fitting:

```python
if not isinstance(process_pls_config, dict):
    raise ValueError('已启用工艺 PLS，但没有有效的工艺 PLS 配置')
if process_pls_config.get('schema_version') != PROCESS_PLS_SCHEMA_VERSION:
    raise ValueError('工艺 PLS 工作流版本不匹配，请回到特征选择页面重新锁定')
if not set(process_pls_config.get('process_feature_cols', [])).issubset(set(feature_columns)):
    raise ValueError('工艺 PLS 所需原始列不完整，请检查特征选择和数据列映射')
```

- [ ] **Step 4: Run focused training tests and existing regression tests**

Run:

```powershell
pytest -q 'tests/test_process_pls.py' 'tests/test_missing_target_training.py' 'tests/test_regression_target_balance.py' 'tests/test_transformer_bnn_training.py'
```

Expected: all selected tests pass; the disabled PLS path has identical feature count and no `process_pls` pipeline step.

- [ ] **Step 5: Commit training integration**

```powershell
git add 'core/model_trainer.py' 'app.py' 'tests/test_process_pls.py'
git commit -m 'feat: fit process PLS inside regression training folds'
```

### Task 4: 连接训练页面的轻量启用选项

**Files:**
- Modify: `app.py` near the model-training controls and all trainer invocation blocks
- Modify: `core/model_trainer.py` only if a call-site adapter is needed
- Test: `tests/test_navigation.py`

**Interfaces:**
- Consumes `st.session_state.process_pls_workflow`.
- Produces a single lightweight checkbox/selectbox labelled `使用已锁定工艺 PLS`.
- Passes `process_pls_config` and `use_process_pls` consistently to both single-model training and CV.

- [ ] **Step 1: Add a UI-state regression test**

```python
def test_process_pls_training_option_defaults_to_disabled_without_locked_workflow():
    workflow = None
    enabled = bool(workflow and workflow.get('enabled'))
    assert enabled is False
```

- [ ] **Step 2: Run the test before changing the call sites**

Run:

```powershell
pytest -q 'tests/test_navigation.py::test_process_pls_training_option_defaults_to_disabled_without_locked_workflow'
```

Expected: FAIL because the named test does not yet exist.

- [ ] **Step 3: Add the compact training-page control**

Near the existing feature summary, render only:

```python
process_pls_workflow = st.session_state.get('process_pls_workflow')
if isinstance(process_pls_workflow, dict) and process_pls_workflow.get('enabled'):
    use_process_pls = st.checkbox(
        '使用已锁定工艺 PLS',
        value=bool(st.session_state.get('process_pls_use_in_training', False)),
        key='process_pls_use_in_training',
        help='只转换锁定的工艺特征；分子特征 workflow 不变。正式拟合只使用训练折。',
    )
    st.caption(
        f"工艺 PLS 已锁定：{len(process_pls_workflow.get('process_feature_cols', []))} 个原始列，"
        f"workflow={process_pls_workflow.get('workflow_hash', '')[:12]}…"
    )
else:
    use_process_pls = False
    st.caption('未锁定工艺 PLS；当前模型使用原始工艺特征流程。')
```

所有训练和交叉验证调用统一追加：

```python
process_pls_config=process_pls_workflow if use_process_pls else None,
use_process_pls=use_process_pls,
```

不得在模型训练页面显示成分数、VIP 阈值、缺失率阈值或诊断表。

- [ ] **Step 4: Run navigation and training tests**

Run:

```powershell
pytest -q 'tests/test_navigation.py' 'tests/test_process_pls.py' 'tests/test_transformer_bnn_training.py'
```

Expected: all tests pass.

- [ ] **Step 5: Commit the compact training-page integration**

```powershell
git add 'app.py' 'core/model_trainer.py' 'tests/test_navigation.py'
git commit -m 'feat: expose locked process PLS as lightweight training option'
```

### Task 5: 保存、恢复和校验工艺 PLS artifact 元数据

**Files:**
- Modify: `core/model_io.py:28-140`
- Modify: `app.py:2645-2728, 16390-16580, 21640-21710`
- Test: `tests/test_process_pls.py`

**Interfaces:**
- Produces `process_pls_to_artifact_extra(config) -> dict`.
- Produces `restore_process_pls_metadata(payload) -> dict | None`.
- Artifact `extra` contains `process_pls_workflow`, `process_pls_schema_version`, and `process_pls_workflow_hash`.
- The fitted `ProcessPLSTransformer` remains inside `artifact['pipeline'].named_steps['process_pls']`; metadata is a compact audit/config record.

- [ ] **Step 1: Add artifact round-trip tests**

```python
def test_process_pls_artifact_round_trip_preserves_workflow_metadata():
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact

    config = {
        'schema_version': 1,
        'enabled': True,
        'process_feature_cols': ['temperature', 'time'],
        'max_components': 8,
        'vip_top_k': 8,
        'missing_threshold': 0.85,
        'cv_splits': 5,
        'random_state': 42,
        'selection_mode': 'auto_combined_score',
        'workflow_hash': 'abc123',
    }
    artifact = create_model_artifact(
        model_name='test',
        target_col='target',
        feature_cols=['process_pls_1'],
        model=object(),
        extra={'process_pls_workflow': config},
    )
    restored = loads_artifact(dumps_artifact(artifact))
    assert restored['extra']['process_pls_workflow']['process_feature_cols'] == [
        'temperature',
        'time',
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
pytest -q 'tests/test_process_pls.py::test_process_pls_artifact_round_trip_preserves_workflow_metadata'
```

Expected: FAIL because no restore/normalization helper and no app artifact integration exists.

- [ ] **Step 3: Add versioned metadata helpers and app restoration**

In `core/model_io.py`, add:

```python
def process_pls_to_artifact_extra(config):
    if not isinstance(config, dict):
        return {}
    return {
        'process_pls_workflow': dict(config),
        'process_pls_schema_version': config.get('schema_version'),
        'process_pls_workflow_hash': config.get('workflow_hash'),
    }
```

Merge this payload into the existing `extra` at every model export path without replacing molecular workflow or post-feature mapping metadata.

In `app.py`, add restore logic parallel to the existing molecular metadata restore:

```python
def _restore_process_pls_metadata(payload):
    extra = payload.get('extra') if isinstance(payload, dict) else {}
    extra = extra if isinstance(extra, dict) else {}
    workflow = extra.get('process_pls_workflow')
    if not isinstance(workflow, dict):
        st.session_state['process_pls_workflow'] = None
        return None
    if int(workflow.get('schema_version', -1)) != 1:
        raise ValueError('导入模型的工艺 PLS workflow 版本不受支持')
    st.session_state['process_pls_workflow'] = workflow
    st.session_state['process_pls_enabled_default'] = bool(workflow.get('enabled'))
    return workflow
```

模型导入、结果包恢复和 session snapshot 恢复都调用该函数；缺少该字段的旧模型必须保持兼容并显示“未包含工艺 PLS workflow”。

- [ ] **Step 4: Run artifact and import-related tests**

Run:

```powershell
pytest -q 'tests/test_process_pls.py' 'tests/test_molecular_feature_workflow.py' 'tests/test_post_feature_mapping.py'
python -m py_compile 'core/model_io.py' 'app.py'
```

Expected: all tests pass and old artifacts without `process_pls_workflow` still load.

- [ ] **Step 5: Commit artifact support**

```powershell
git add 'core/model_io.py' 'app.py' 'tests/test_process_pls.py'
git commit -m 'feat: persist and restore process PLS workflow metadata'
```

### Task 6: 让预测和高通量筛选复用已保存的 PLS

**Files:**
- Modify: `core/virtual_screening.py:1837-1900`
- Modify: `app.py` prediction/screening paths around `build_feature_matrix` and `page_virtual_screening`
- Test: `tests/test_virtual_screening.py`
- Test: `tests/test_process_pls.py`

**Interfaces:**
- Produces `apply_saved_process_pls(pipeline, X_raw) -> pandas.DataFrame`.
- `build_feature_matrix(...)` continues to build raw model input columns and does not independently calculate PLS columns.
- Screening calls `pipeline.predict(X_raw)` or an equivalent single pipeline transform; it must not call `fit`, `fit_transform`, or rebuild PLS from candidate rows.

- [ ] **Step 1: Add screening reproducibility tests**

```python
def test_screening_reuses_saved_process_pls_without_refit():
    from core.process_pls import ProcessPLSTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame({
        'temperature': [1.0, 2.0, 3.0, 4.0],
        'time': [10.0, 11.0, 12.0, 13.0],
    })
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pipeline = Pipeline([
        ('process_pls', ProcessPLSTransformer(
            process_feature_cols=['temperature', 'time'],
            max_components=1,
            random_state=42,
        )),
        ('model', LinearRegression()),
    ])
    pipeline.fit(train, y)
    statistics_before = pipeline.named_steps['process_pls'].imputer_.statistics_.copy()

    candidate = pd.DataFrame({'temperature': [1000.0], 'time': [999.0]})
    prediction = pipeline.predict(candidate)

    assert prediction.shape == (1,)
    np.testing.assert_array_equal(
        pipeline.named_steps['process_pls'].imputer_.statistics_,
        statistics_before,
    )


def test_screening_reports_missing_raw_process_columns():
    from core.process_pls import ProcessPLSTransformer

    transformer = ProcessPLSTransformer(process_feature_cols=['temperature'])
    with pytest.raises(ValueError, match='missing required process columns'):
        transformer.transform(pd.DataFrame({'time': [1.0]}))
```

- [ ] **Step 2: Run the screening tests and verify the new test fails**

Run:

```powershell
pytest -q 'tests/test_virtual_screening.py' 'tests/test_process_pls.py::test_screening_reuses_saved_process_pls_without_refit'
```

Expected: the existing screening tests pass, while the new helper/integration test fails until the raw-frame path is connected.

- [ ] **Step 3: Route screening through raw columns plus the saved pipeline**

Before `build_feature_matrix`, resolve the raw process columns from the artifact workflow and retain them in `feature_cols`. Do not add `process_pls_*` to `build_feature_matrix` input. When a saved pipeline contains `process_pls`, call:

```python
raw_required = list(
    pipeline.named_steps['process_pls'].input_feature_cols_
)
missing_raw = [column for column in raw_required if column not in X_raw.columns]
if missing_raw:
    raise ValueError(
        f'高通量筛选缺少工艺 PLS 原始输入列: {", ".join(missing_raw[:12])}'
    )
predictions = pipeline.predict(X_raw[raw_required + untouched_feature_cols])
```

The actual implementation must preserve the complete saved pipeline input order, including molecular feature columns and non-process features. Use the saved artifact feature contract to build `X_raw`; then let the pipeline transform process columns. If an imported legacy model has PLS metadata but no fitted `process_pls` pipeline step, stop with a clear re-training message rather than fitting a new transformer.

Add the screening summary:

```python
if 'process_pls' in getattr(pipeline, 'named_steps', {}):
    st.caption(
        f"已复用模型内工艺 PLS：{len(pipeline.named_steps['process_pls'].process_feature_cols)} 个原始工艺列"
    )
```

For non-PLS artifacts, leave the current strict molecular feature contract and `build_feature_matrix` behavior unchanged.

- [ ] **Step 4: Run all screening and PLS tests**

Run:

```powershell
pytest -q 'tests/test_virtual_screening.py' 'tests/test_process_pls.py'
python -m py_compile 'core/virtual_screening.py' 'app.py'
```

Expected: all tests pass; a candidate without required raw process columns receives a readable error instead of a silent numeric fill.

- [ ] **Step 5: Commit screening reuse**

```powershell
git add 'core/virtual_screening.py' 'app.py' 'tests/test_virtual_screening.py' 'tests/test_process_pls.py'
git commit -m 'feat: reuse saved process PLS during prediction and screening'
```

### Task 7: 完成回归、兼容性和文档验收

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-process-pls-design.md`
- Create: `docs/process-pls-workflow.md`
- Modify: `README.md` or the current version documentation file identified by `rg -n 'version|版本'`
- Test: `tests/test_process_pls.py`

**Interfaces:**
- Produces a user-facing Chinese workflow document explaining configuration, leakage protection, artifact compatibility and screening behavior.
- Produces no new training-page parameter block beyond the single lightweight toggle.

- [ ] **Step 1: Add compatibility tests**

```python
def test_legacy_artifact_without_process_pls_is_unchanged():
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact

    artifact = create_model_artifact(
        model_name='legacy',
        target_col='target',
        feature_cols=['temperature'],
        model=object(),
        extra={'molecular_feature_workflow': {'schema_version': 1}},
    )
    restored = loads_artifact(dumps_artifact(artifact))
    assert 'process_pls_workflow' not in restored['extra']


def test_process_pls_output_order_is_stable_after_joblib_round_trip(tmp_path):
    import joblib
    from core.process_pls import ProcessPLSTransformer

    X = pd.DataFrame({
        'temperature': [1.0, 2.0, 3.0, 4.0],
        'time': [10.0, 11.0, 12.0, 13.0],
        'other': [5.0, 6.0, 7.0, 8.0],
    })
    y = np.array([1.0, 2.0, 3.0, 4.0])
    transformer = ProcessPLSTransformer(
        process_feature_cols=['temperature', 'time'],
        max_components=1,
    ).fit(X, y)
    path = tmp_path / 'process_pls.joblib'
    joblib.dump(transformer, path)
    restored = joblib.load(path)
    assert restored.get_feature_names_out().tolist() == transformer.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(restored.transform(X), transformer.transform(X))
```

- [ ] **Step 2: Run the complete focused regression suite**

Run:

```powershell
pytest -q 'tests/test_process_pls.py' 'tests/test_virtual_screening.py' 'tests/test_missing_target_training.py' 'tests/test_regression_target_balance.py' 'tests/test_transformer_bnn_training.py' 'tests/test_molecular_feature_workflow.py'
```

Expected: all selected tests pass with no new warnings except warnings already present in the repository.

- [ ] **Step 3: Update user-facing documentation**

Create `docs/process-pls-workflow.md` with these exact sections:

```markdown
# 工艺特征 PLS 工作流

## 使用位置
在“特征选择”页面选择工艺数值特征、生成诊断并锁定 workflow；模型训练页只选择是否使用已锁定 workflow。

## 处理顺序
原始工艺列 → 缺失率过滤 → 训练折中位数插补 → RobustScaler → 训练折内 CV 选择成分数 → VIP 选择少量原始列 → 缺失掩码 → 模型。

## 数据泄漏规则
预览可以使用全数据，但正式训练、交叉验证、预测和筛选不会使用预览拟合对象；每个训练折独立拟合。

## 输出列
`process_pls_1...n`、VIP 保留的原始工艺列、`<原始列>__missing`，以及未参与 PLS 的其他模型输入列。

## 与分子特征的关系
工艺 PLS 不读取 SMILES、BigSMILES、MACCS、Morgan、RDKit 描述符，也不改变 molecular workflow。

## 旧模型
没有 `process_pls_workflow` 的旧模型继续按原流程运行；包含 PLS 配置但缺少 fitted pipeline step 的模型必须重新训练。
```

更新设计稿的验收状态，补充实际 artifact 字段和测试命令；更新当前版本文档中的特性列表，但不在本任务中擅自修改版本号。

- [ ] **Step 4: Run repository-wide validation**

Run:

```powershell
pytest -q
git diff --check
$terms = @(
    ('T' + 'ODO'),
    ('T' + 'BD'),
    ('待' + '定'),
    ('la' + 'ter'),
    ('appro' + 'priate')
)
Select-String -LiteralPath 'docs/superpowers/plans/2026-07-30-process-pls.md' -Pattern $terms
```

Expected:

- `pytest -q` passes, or only reports failures already present before this feature and they are recorded in the handoff;
- `git diff --check` produces no output;
- the placeholder scan produces no matches.

- [ ] **Step 5: Commit documentation and validation**

```powershell
git add 'docs/process-pls-workflow.md' 'docs/superpowers/specs/2026-07-30-process-pls-design.md' 'README.md'
git commit -m 'docs: document process PLS workflow and validation'
```

## Self-Review Checklist

- [ ] Spec coverage: feature-selection configuration, preview-only fit, locked config, leakage-safe train/CV, raw-frame model compatibility, artifact round-trip, prediction, high-throughput screening and legacy fallback each have a task.
- [ ] Interface consistency: `process_pls_config`, `use_process_pls`, `ProcessPLSTransformer`, `process_pls_workflow` and artifact keys use the same names in every task.
- [ ] Molecular isolation: candidate inference excludes molecular workflow columns and the screening path does not synthesize PLS columns from molecular features.
- [ ] Disabled-path safety: no PLS step is added when the checkbox is off.
- [ ] Failure visibility: missing raw columns, unsupported schema and absent fitted pipeline are explicit errors.
- [ ] No placeholder scan matches.

Plan complete and saved to `docs/superpowers/plans/2026-07-30-process-pls.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task and review between tasks.
2. **Inline Execution** — execute tasks in this session using `executing-plans` with checkpoints.
