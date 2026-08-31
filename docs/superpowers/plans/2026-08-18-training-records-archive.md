# 训练记录归档与清理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每次训练的模型、完整切分数据和流程配置可靠落盘、可校验恢复，并提供安全的训练记录扫描与可恢复清理功能。

**Architecture:** 将 `TrainingRunManager` 改为基于项目根目录的原子记录包写入器，使用 `manifest.json` 描述文件状态、大小和 SHA-256。训练记录页面只通过 manager 读取并恢复 artifact 与数据；清理逻辑放在独立模块中，默认把候选记录移动到回收目录并记录日志，不直接删除。

**Tech Stack:** Python 3.10、pandas、joblib、hashlib、JSON、Streamlit、pytest。

## Global Constraints

- 记录根目录必须基于项目根目录解析，并支持 `CFRP_TRAINING_RUNS_DIR` 覆盖。
- 每次新训练必须保存完整 `X_train/X_test/y_train/y_test`、特征列、目标列、原始索引、切分参数、模型 artifact 和特征流程。
- 旧记录只读兼容扫描，不重写旧文件，不删除现有备份、缓存和数据源。
- 清理必须先预览、再二次确认，默认移动到 `.training_run_trash`，保留恢复能力。
- 模型或必需数据保存失败时，记录状态为 `incomplete`，不得伪装成可加载记录。
- 不改变训练算法、模型选择逻辑或已有模型文件格式。
- 不提交 Git；每个任务通过测试和 `git diff --check` 验证。

---

### Task 1: 建立可校验的训练记录存储层

**Files:**
- Modify: `core/training_runs.py`
- Create: `tests/test_training_runs_archive.py`

**Interfaces:**
- Extend `TrainingRunManager.__init__(base_dir: str | None = None)` to resolve the default directory from the project root and honor `CFRP_TRAINING_RUNS_DIR`.
- Extend `TrainingRunManager.save_run(...)` with optional `X_train`, `X_test`, `y_train`, `y_test`, `feature_process`, and `source_metadata` arguments.
- Add `TrainingRunManager.inspect_run(run_id: str) -> dict` returning `status`, `missing_required`, `invalid_files`, `total_bytes`, and `manifest`.
- Add `TrainingRunManager.restore_split_data(run_id: str) -> dict` returning `X_train`, `X_test`, `y_train`, `y_test`, and `metadata`.
- Add private helpers `_sha256_file(path)`, `_write_csv_preserving_index(df, path)`, `_build_manifest(run_dir, required_files, errors)`, and `_safe_run_dir(run_id)`.

- [ ] **Step 1: Write failing tests for stable paths and full split persistence**

```python
def test_default_manager_uses_project_root_not_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = TrainingRunManager()
    assert manager.base_dir.endswith('results/training_runs')
    assert Path(manager.base_dir).is_absolute()


def test_save_run_restores_split_data_and_manifest(tmp_path):
    class SerializableModel:
        def predict(self, values):
            return np.zeros(len(values))

    manager = TrainingRunManager(base_dir=str(tmp_path / 'runs'))
    x_train = pd.DataFrame({'a': [1.0, 2.0]}, index=[10, 11])
    x_test = pd.DataFrame({'a': [3.0]}, index=[12])
    summary = manager.save_run(
        model_name='test',
        metadata={'target_col': 'y', 'n_train': 2, 'n_test': 1},
        model=SerializableModel(),
        feature_cols=['a'],
        target_col='y',
        X_train=x_train,
        X_test=x_test,
        y_train=pd.Series([4.0, 5.0], index=[10, 11], name='y'),
        y_test=pd.Series([6.0], index=[12], name='y'),
        feature_process={'steps': [{'method': 'RDKit'}]},
    )
    loaded = manager.restore_split_data(summary.run_id)
    pd.testing.assert_frame_equal(loaded['X_train'], x_train)
    pd.testing.assert_frame_equal(loaded['X_test'], x_test)
    assert loaded['y_train'].tolist() == [4.0, 5.0]
    assert manager.inspect_run(summary.run_id)['status'] == 'complete'
```

- [ ] **Step 2: Run the focused tests and verify they fail for the current manager**

Run: `python -m pytest tests/test_training_runs_archive.py -q`

Expected: FAIL because the current manager has no split-data arguments, no manifest, and resolves a relative default path.

- [ ] **Step 3: Implement project-root resolution and safe run paths**

Use `Path(__file__).resolve().parents[1]` as the project root, then resolve `base_dir` or `CFRP_TRAINING_RUNS_DIR`. Reject absolute paths outside an explicitly supplied custom directory only when resolving a run id; reject path traversal such as `../x` and path separators in `run_id`.

- [ ] **Step 4: Implement atomic record writing and manifest generation**

Write all files to `.<run_id>.tmp-<pid>`, including `metadata.json`, split CSVs, `feature_process.json`, existing history/tables/images, and model artifact. Write a manifest containing:

```python
{
    'schema_version': 2,
    'run_id': run_id,
    'status': 'complete',
    'required_files': [...],
    'files': {'relative/path': {'size': 123, 'sha256': '...', 'kind': 'data'}},
    'errors': [],
}
```

Use `os.replace` only after the manifest is successfully written. On required-file failure, keep an `incomplete` manifest with `errors` and do not publish it as a normal completed directory.

- [ ] **Step 5: Implement split-data restoration and integrity inspection**

Read CSVs with `index_col=0`, restore Series names from metadata, validate manifest size/hash when present, and return a structured diagnostic instead of raising for missing optional files. `load_run(load_model=True)` must refuse only when the model artifact is missing or invalid, while preserving metadata and readable tables.

- [ ] **Step 6: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_training_runs_archive.py -q`

Expected: PASS with stable absolute path, round-trip split data, manifest status, and traversal rejection covered.

---

### Task 2: Persist all training outputs from the training page

**Files:**
- Modify: `app.py:13770-14050`
- Modify: `app.py:12160-12235`
- Modify: `core/training_runs.py` if a serialization adapter is required
- Modify: `tests/test_training_runs_archive.py`

**Interfaces:**
- Both regression and classification save paths call the same extended `save_run` contract.
- Add `_build_training_run_save_payload(result: dict, feature_cols: list[str], target_col: str, model_name: str, workflow_payload: dict | None, source_metadata: dict | None) -> dict` in `app.py`; it returns the complete keyword payload for `TrainingRunManager.save_run`.
- The call passes the actual training outputs from `res`: `X_train`, `X_test`, `y_train`, `y_test`; it does not reconstruct them from the current session.
- `feature_process` is taken from the complete current molecular/process workflow payload, and `source_metadata` includes source data shape, column names, target and split settings.

- [ ] **Step 1: Add a regression test for the save payload contract**

```python
def test_training_save_payload_uses_actual_result_splits(monkeypatch):
    payload = app._build_training_run_save_payload(
        result={'X_train': pd.DataFrame({'f': [1]}), 'X_test': pd.DataFrame({'f': [2]}),
                'y_train': pd.Series([3]), 'y_test': pd.Series([4]), 'model': object()},
        feature_cols=['f'], target_col='y', model_name='XGBoost',
        workflow_payload={'steps': []}, source_metadata={'split_strategy': 'random'},
    )
    assert payload['X_train'].iloc[0, 0] == 1
    assert payload['X_test'].iloc[0, 0] == 2
    assert payload['feature_process'] == {'steps': []}
```

- [ ] **Step 2: Run the test and verify the existing save path lacks the required payload**

Run: `python -m pytest tests/test_training_runs_archive.py::test_training_save_payload_uses_actual_result_splits -q`

Expected: FAIL until both classification and regression save calls pass the split matrices and workflow payload.

- [ ] **Step 3: Add the complete payload to regression save calls**

Pass `res.get('X_train')`, `res.get('X_test')`, `res.get('y_train')`, and `res.get('y_test')` to `save_run`. Pass the assembled `_extra` workflow payload or its JSON-safe feature-process subset. Preserve predictions and existing tables as separate files.

- [ ] **Step 4: Add the same payload to classification save calls**

Use the classification result object as the source of truth. Do not make the classification path depend on regression-only keys such as `r2`, and keep class labels and positive label in metadata.

- [ ] **Step 5: Ensure model export and run artifact share one run id**

Remove any second independent directory creation. The exported `model_<name>.joblib`, `feature_process.json`, and `model.pkl` must all be written into `summary.path`, and metadata must contain `training_run_id` before the manifest is finalized.

- [ ] **Step 6: Run focused save and existing training tests**

Run: `python -m pytest tests/test_training_runs_archive.py tests/test_virtual_screening.py -q`

Expected: PASS; existing training and virtual-screening tests remain green.

---

### Task 3: Make training-record loading complete and diagnosable

**Files:**
- Modify: `core/training_runs.py`
- Modify: `app.py:22513-22890`
- Modify: `tests/test_training_runs_archive.py`

**Interfaces:**
- `TrainingRunManager.list_runs(limit: int | None = None, status: str | None = None, model_name: str | None = None)` returns summaries with `status`, `total_bytes`, `missing_required`, and `can_load_model`.
- `load_run(run_id, load_model=False, verify=True)` returns `integrity`, `split_data`, and `model_artifact` in addition to existing keys.

- [ ] **Step 1: Add tests for incomplete records and model restoration**

```python
def test_load_run_reports_missing_model_without_hiding_other_data(tmp_path):
    manager = TrainingRunManager(base_dir=str(tmp_path / 'runs'))
    run_dir = Path(manager.base_dir) / 'legacy'
    run_dir.mkdir(parents=True)
    (run_dir / 'metadata.json').write_text(json.dumps({'model_name': 'legacy'}), encoding='utf-8')
    pd.DataFrame({'f': [1]}).to_csv(run_dir / 'split_X_train.csv', index=False)
    payload = manager.load_run('legacy', load_model=True)
    assert payload['model_artifact'] is None
    assert payload['integrity']['status'] in {'legacy', 'incomplete'}
    assert payload['split_data']['X_train'] is not None
```

- [ ] **Step 2: Run the test and verify current loading drops split data and diagnostics**

Run: `python -m pytest tests/test_training_runs_archive.py::test_load_run_reports_missing_model_without_hiding_other_data -q`

Expected: FAIL until `load_run` returns integrity and split-data fields.

- [ ] **Step 3: Add integrity-aware summaries and pagination**

Replace the hard-coded `limit=200` page call with a user-selectable page size and manager-side filtering. Keep the default page bounded for rendering performance, but expose total count and page navigation so older records are not silently hidden.

- [ ] **Step 4: Restore split data when loading a model**

After a valid artifact loads, assign the restored `X_train`, `X_test`, `y_train`, and `y_test` to session state only when present. Preserve the artifact feature contract and workflow fields; never replace missing features with MACCS, zeros, or current-page columns.

- [ ] **Step 5: Render status and missing-file explanations**

Show status, total size, and missing/invalid file names beside the selected run. Disable or reject model loading for invalid artifacts with the exact diagnostic from `inspect_run`; keep downloads for readable tables and metadata.

- [ ] **Step 6: Run focused loading tests**

Run: `python -m pytest tests/test_training_runs_archive.py tests/test_prediction_feature_contract.py tests/test_prediction_molecular_baseline.py -q`

Expected: PASS with old and new artifact contracts handled.

---

### Task 4: Add safe scan, quarantine, restore, and cleanup operations

**Files:**
- Create: `core/training_run_cleanup.py`
- Create: `tests/test_training_run_cleanup.py`
- Modify: `app.py:22513-22890`

**Interfaces:**
- `scan_training_runs(base_dir: str, exclude_run_ids: set[str] | None = None) -> pd.DataFrame` returns one row per record with `run_id`, `status`, `model_name`, `created_at`, `total_bytes`, `missing_required`, and `duplicate_group`.
- `build_cleanup_candidates(scan_df, status=None, before=None, keep_latest=None, max_total_bytes=None) -> pd.DataFrame` returns candidates only; it never mutates files.
- `quarantine_runs(base_dir: str, run_ids: list[str], trash_dir: str | None = None) -> dict` moves selected directories, writes an operation log, and returns moved/error rows.
- `restore_quarantined_run(trash_dir: str, operation_id: str, run_id: str) -> dict` restores only into the original safe parent when no conflicting active run exists.

- [ ] **Step 1: Write failing tests for scan, duplicate detection, quarantine and traversal safety**

```python
def test_cleanup_scan_marks_missing_required_files(tmp_path):
    run = tmp_path / 'runs' / 'r1'
    run.mkdir(parents=True)
    (run / 'metadata.json').write_text('{}', encoding='utf-8')
    scan = scan_training_runs(str(tmp_path / 'runs'))
    assert scan.loc[0, 'status'] in {'legacy', 'incomplete'}
    assert 'model.pkl' in scan.loc[0, 'missing_required']


def test_quarantine_moves_only_selected_run_and_can_restore(tmp_path):
    runs = tmp_path / 'runs'
    run = runs / 'r1'
    run.mkdir(parents=True)
    (run / 'metadata.json').write_text('{}', encoding='utf-8')
    result = quarantine_runs(str(runs), ['r1'])
    assert result['moved'] == ['r1']
    assert not run.exists()
    restored = restore_quarantined_run(result['trash_dir'], result['operation_id'], 'r1')
    assert restored['restored'] is True
    assert run.exists()


def test_cleanup_rejects_run_id_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        quarantine_runs(str(tmp_path / 'runs'), ['..\\outside'])
```

- [ ] **Step 2: Run the cleanup tests and verify they fail before implementation**

Run: `python -m pytest tests/test_training_run_cleanup.py -q`

Expected: FAIL because the cleanup module and quarantine interfaces do not exist.

- [ ] **Step 3: Implement read-only scan and duplicate grouping**

Use `TrainingRunManager.inspect_run` for status and size. Group duplicates by manifest content fingerprint when available; otherwise use metadata model/target/config plus file hashes. Never mark the newest record as an automatic deletion target when a duplicate group exists.

- [ ] **Step 4: Implement candidate generation with conservative filters**

Support status filters, cutoff timestamps, keep-latest counts, and total-byte budgets. Return a preview DataFrame including why each row is a candidate. Exclude current session run id, active model run id, `.training_run_trash`, temporary directories, and any path outside the base directory.

- [ ] **Step 5: Implement quarantine and restore with operation logs**

Create `.training_run_trash/<operation_id>/`, move directories with `shutil.move`, write `operation.json` containing original paths, timestamps, run ids and errors, and use a restore operation that checks conflicts before moving back. Do not use recursive deletion in the normal cleanup path.

- [ ] **Step 6: Add the Streamlit cleanup panel**

Add a collapsible panel on the training-record page with scan refresh, status/model/date filters, candidate preview, total size, selected checkboxes, a two-step confirmation flag, quarantine action, and restore list. Keep cleanup controls separate from model loading so a rerun cannot accidentally delete anything.

- [ ] **Step 7: Run cleanup tests**

Run: `python -m pytest tests/test_training_run_cleanup.py -q`

Expected: PASS for scan, selection, path safety, quarantine, restore and operation logging.

---

### Task 5: Compatibility migration report and end-to-end verification

**Files:**
- Modify: `core/training_runs.py`
- Modify: `app.py`
- Create: `tests/test_training_runs_compatibility.py`
- Modify: `README.md` or `DEVELOPMENT.md`

**Interfaces:**
- Add `TrainingRunManager.scan_legacy_runs() -> list[dict]` that reports old records without rewriting them.
- Add a UI download for the scan report as CSV/JSON.

- [ ] **Step 1: Add compatibility tests for existing directory layouts**

```python
def test_legacy_record_is_read_without_rewriting_files(tmp_path):
    run_dir = tmp_path / 'runs' / 'legacy'
    run_dir.mkdir(parents=True)
    metadata = run_dir / 'metadata.json'
    metadata.write_text(json.dumps({'model_name': 'XGBoost'}), encoding='utf-8')
    before = metadata.stat().st_mtime_ns
    report = TrainingRunManager(base_dir=str(tmp_path / 'runs')).scan_legacy_runs()
    assert report[0]['run_id'] == 'legacy'
    assert metadata.stat().st_mtime_ns == before
```

- [ ] **Step 2: Run compatibility tests and verify legacy behavior**

Run: `python -m pytest tests/test_training_runs_compatibility.py -q`

Expected: FAIL until the read-only legacy scan is implemented.

- [ ] **Step 3: Implement legacy scan and report download**

Classify a directory without `manifest.json` as `legacy`, list missing required files, and expose the report from the training-record page. Do not generate files in legacy directories.

- [ ] **Step 4: Document storage and cleanup behavior**

Document the absolute-path override, record contents, integrity statuses, recovery directory, and safe cleanup workflow in `DEVELOPMENT.md` or the existing training-record section of `README.md`.

- [ ] **Step 5: Run the complete targeted test set and static checks**

Run:

```powershell
python -m pytest tests/test_training_runs_archive.py tests/test_training_run_cleanup.py tests/test_training_runs_compatibility.py tests/test_prediction_feature_contract.py tests/test_prediction_molecular_baseline.py -q
python -m py_compile core/training_runs.py core/training_run_cleanup.py app.py
git diff --check
```

Expected: all targeted tests pass, compilation succeeds, and `git diff --check` produces no output.

- [ ] **Step 6: Produce a read-only report for the existing records**

Run the manager scan against the project `results/training_runs` and report the count of complete, legacy, incomplete and invalid records plus total size. Do not quarantine or delete anything until the user explicitly requests the cleanup operation.

## Self-Review Checklist

- [x] Covers stable absolute path and environment override.
- [x] Covers complete training/test matrices, targets, indices, feature/process workflow, models, predictions, histories and figures.
- [x] Covers atomic writes and manifest hash validation.
- [x] Covers model loading diagnostics and session restoration.
- [x] Covers paginated records rather than silently hiding records after 200.
- [x] Covers read-only legacy compatibility without rewriting old data.
- [x] Covers preview, quarantine, restore and operation logging without destructive default deletion.
- [x] Covers tests for failure paths, path traversal and incomplete records.
- [x] Contains no TODO/TBD/placeholders.
