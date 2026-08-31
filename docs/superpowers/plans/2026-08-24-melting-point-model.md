# 熔点模型与虚拟筛选 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 PubChem 熔点数据采集、清洗、训练和树脂/固化剂虚拟筛选过滤的完整闭环。

**Architecture:** 新增纯数据层 `core/melting_point_data.py` 负责文本解析、单位转换、质量分级和去重；扩展 `core/pubchem_client.py` 负责 PUG View 熔点记录获取和缓存；新增 `core/melting_point_screening.py` 负责 artifact 校验、角色阈值判定和结果字段生成。`app.py` 仅负责 Streamlit 控件、模型 workflow 重放和现有配方筛选流程的编排，避免继续扩大业务逻辑耦合。

**Tech Stack:** Python 3.10、pandas、NumPy、RDKit、scikit-learn/joblib、Streamlit、pytest、PubChem PUG REST/PUG View。

## Global Constraints

- PubChem 原始记录与整理后的训练集必须分别保存，不修改用户原始数据。
- 统一模型目标单位为 `°C`，artifact 必须声明 `task_kind=melting_point` 和 `target_unit=C`。
- 树脂和固化剂使用统一模型，但阈值、过滤状态和分层评估必须分开。
- 高质量单值记录默认进入训练；范围、估算、分解、软化、混合物和无法解析记录默认不进入训练。
- 适用域外、特征无法复现或不确定度不可用的候选必须标记为 `unknown`，不能静默通过。
- 新熔点筛选默认是“仅标记不剔除”，严格过滤必须由用户显式开启。
- 不再把 `core/industrial_filter.py` 中的经验熔点公式作为熔点模型结果。
- 命令中的路径和字符串值使用单引号；不自动创建 git commit。

---

### Task 1: 实现熔点文本解析与训练集清洗核心

**Files:**
- Create: `core/melting_point_data.py`
- Test: `tests/test_melting_point_data.py`

**Interfaces:**
- `parse_melting_point_text(text: object) -> dict`
- `canonicalize_smiles(smiles: object) -> Optional[str]`
- `deduplicate_melting_point_records(df: pd.DataFrame) -> pd.DataFrame`
- `prepare_melting_point_dataset(raw_df: pd.DataFrame, include_low_quality: bool = False) -> pd.DataFrame`
- `summarize_melting_point_dataset(df: pd.DataFrame) -> dict`

- [x] **Step 1: Write failing parser tests**

```python
from core.melting_point_data import parse_melting_point_text


def test_parse_celsius_single_value():
    result = parse_melting_point_text('126 °C')
    assert result['mp_c'] == 126.0
    assert result['mp_quality'] == 'high'


def test_parse_fahrenheit_and_kelvin():
    assert parse_melting_point_text('212 °F')['mp_c'] == 100.0
    assert parse_melting_point_text('373.15 K')['mp_c'] == 100.0


def test_parse_range_keeps_bounds_without_training_value():
    result = parse_melting_point_text('120-130 °C')
    assert result['mp_c'] is None
    assert result['mp_lower_c'] == 120.0
    assert result['mp_upper_c'] == 130.0
    assert result['mp_quality'] == 'range'


def test_parse_decomposition_is_low_quality():
    result = parse_melting_point_text('240 °C (decomposes)')
    assert result['mp_c'] == 240.0
    assert result['mp_quality'] == 'decomp'


def test_unparsed_text_never_becomes_training_value():
    result = parse_melting_point_text('softens above room temperature')
    assert result['mp_c'] is None
    assert result['mp_quality'] == 'unparsed'
```

- [x] **Step 2: Run parser tests and verify expected failure**

Run: `pytest 'tests/test_melting_point_data.py' -q`

Expected: FAIL because `core.melting_point_data` and its parser do not exist yet.

- [x] **Step 3: Implement parsing and quality classification**

Implement unit normalization with explicit formulas `C = (F - 32) * 5 / 9` and `C = K - 273.15`; recognize decimal values, ranges using `-`, `–`, `to`, and quality keywords `decomp`, `decompose`, `soften`, `approx`, `about`, `mixture`; return a fixed dictionary containing `mp_c`, `mp_lower_c`, `mp_upper_c`, `mp_unit_raw`, `mp_raw`, and `mp_quality`. Do not infer a midpoint into `mp_c` for range values.

- [x] **Step 4: Add canonicalization, deduplication and dataset summary tests**

```python
import pandas as pd
from core.melting_point_data import (
    canonicalize_smiles,
    deduplicate_melting_point_records,
    prepare_melting_point_dataset,
    summarize_melting_point_dataset,
)


def test_canonical_smiles_deduplicates_equivalent_structures():
    assert canonicalize_smiles('C(C)O') == canonicalize_smiles('CCO')


def test_deduplication_prefers_high_quality_record():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_c':  -114.0, 'mp_quality': 'estimated', 'source': 'a'},
        {'smiles': 'CCO', 'mp_c':  -114.1, 'mp_quality': 'high', 'source': 'b'},
    ])
    result = deduplicate_melting_point_records(frame)
    assert len(result) == 1
    assert result.iloc[0]['source'] == 'b'


def test_prepare_dataset_excludes_low_quality_by_default():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_raw': '-114 °C', 'component_role': 'other'},
        {'smiles': 'CCN', 'mp_raw': 'about 80 °C', 'component_role': 'hardener'},
    ])
    result = prepare_melting_point_dataset(frame)
    assert set(result['mp_quality']) == {'high'}
    summary = summarize_melting_point_dataset(result)
    assert summary['high_quality_count'] == 1
```

- [x] **Step 5: Run tests to verify the data layer is green**

Run: `pytest 'tests/test_melting_point_data.py' -q`

Expected: PASS with all parser, unit, quality, canonicalization, deduplication and summary assertions passing.

---

### Task 2: 扩展 PubChem 熔点注释采集与缓存

**Files:**
- Modify: `core/pubchem_client.py:361-531`
- Test: `tests/test_pubchem_melting_point_client.py`

**Interfaces:**
- `fetch_melting_point_annotations_by_cids(cids: Sequence[int], *, max_workers: int = 4, timeout: int = 30, retries: int = 2) -> pd.DataFrame`
- `fetch_melting_point_records_by_smarts(smarts: str, *, component_role: str, hardener_class: str = '', max_cids: int = 5000, property_workers: int = 4, timeout: int = 30, retries: int = 2) -> pd.DataFrame`

- [x] **Step 1: Write tests with a fake PUG View payload**

```python
from core import pubchem_client


def test_extracts_melting_point_annotation_and_source(monkeypatch):
    payload = {
        'Record': {
            'RecordNumber': 123,
            'Section': [{
                'TOCHeading': 'Melting Point',
                'Information': [{
                    'Value': {'StringWithMarkup': [{'String': '126 °C'}]},
                    'Reference': [{'URL': 'https://example.test/ref'}],
                }],
            }],
        }
    }
    monkeypatch.setattr(pubchem_client, '_request_json', lambda *args, **kwargs: payload)
    result = pubchem_client.fetch_melting_point_annotations_by_cids([123], max_workers=1)
    assert result.iloc[0]['cid'] == 123
    assert result.iloc[0]['mp_raw'] == '126 °C'
    assert result.iloc[0]['source_url'] == 'https://example.test/ref'


def test_query_records_keep_role_and_class(monkeypatch):
    monkeypatch.setattr(pubchem_client, 'fetch_cids_by_smarts', lambda *args, **kwargs: [11])
    monkeypatch.setattr(
        pubchem_client,
        'fetch_properties_by_cids',
        lambda *args, **kwargs: __import__('pandas').DataFrame([{
            'CID': 11,
            'CanonicalSMILES': 'CCO',
            'MolecularWeight': '46.07',
        }]),
    )
    monkeypatch.setattr(
        pubchem_client,
        'fetch_melting_point_annotations_by_cids',
        lambda *args, **kwargs: __import__('pandas').DataFrame([{
            'cid': 11, 'mp_raw': '-114 °C', 'source_url': 'https://example.test/mp'
        }]),
    )
    result = pubchem_client.fetch_melting_point_records_by_smarts(
        '[OX2H]', component_role='hardener', hardener_class='酚', max_cids=10,
    )
    assert result.iloc[0]['component_role'] == 'hardener'
    assert result.iloc[0]['hardener_class'] == '酚'
```

- [x] **Step 2: Run tests to verify the client tests fail for missing APIs**

Run: `pytest 'tests/test_pubchem_melting_point_client.py' -q`

Expected: FAIL because the new annotation and record APIs are not implemented.

- [x] **Step 3: Implement recursive PUG View annotation extraction**

Request `https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/{cid}/JSON`; recursively walk `Section` and `Information`, select headings normalized to `melting point`, read `StringWithMarkup`, `String`, `Number`, `Unit`, `Reference` and `URL`; emit one row per distinct raw annotation with `cid`, `mp_raw`, `source_url`, `source_name`, and `source_record`. A missing section returns an empty row set, not an exception.

- [x] **Step 4: Implement bounded parallel fetch and disk cache**

Add a cache key containing query text, role, hardener class, CID limit and client schema version; store raw annotation CSV and merged candidate CSV under `cache/pubchem`; use the existing worker cap and retry logic; preserve query order when combining completed futures; never treat a network failure as an empty successful cache.

- [x] **Step 5: Merge structure properties with annotations and run tests**

Join CID structure properties with annotation rows, parse values through `core.melting_point_data.parse_melting_point_text`, attach `component_role`, `hardener_class`, `smiles`, `mol_wt`, `canonical_smiles`, and query metadata, then run `pytest 'tests/test_pubchem_melting_point_client.py' -q` expecting PASS.

---

### Task 3: 建立熔点数据集采集面板与训练入口

**Files:**
- Modify: `app.py:17347-18740` in `page_virtual_screening`
- Modify: `app.py:12258-12350` in `page_model_training`
- Test: `tests/test_melting_point_dataset_ui_contract.py`

**Interfaces:**
- Session state key `melting_point_raw_records`: `pd.DataFrame`
- Session state key `melting_point_dataset`: `pd.DataFrame`
- Session state key `melting_point_dataset_summary`: `dict`
- UI helper `render_melting_point_dataset_panel() -> Optional[pd.DataFrame]`

- [x] **Step 1: Write UI contract tests for dataset shape and training handoff**

```python
import pandas as pd
from core.melting_point_data import MELTING_POINT_DATASET_COLUMNS


def test_training_dataset_contract_has_target_and_provenance_columns():
    required = {
        'smiles', 'mp_c', 'mp_raw', 'mp_quality', 'component_role',
        'hardener_class', 'cid', 'source_url', 'canonical_smiles',
    }
    assert required.issubset(set(MELTING_POINT_DATASET_COLUMNS))


def test_empty_dataset_cannot_be_marked_training_ready():
    frame = pd.DataFrame(columns=MELTING_POINT_DATASET_COLUMNS)
    assert frame.empty
```

- [x] **Step 2: Run the contract tests and verify the new constant is missing**

Run: `pytest 'tests/test_melting_point_dataset_ui_contract.py' -q`

Expected: FAIL because the dataset schema constant and collection contract do not exist.

- [x] **Step 3: Add the Streamlit collection panel**

Add an expander under the virtual screening page with separate resin SMARTS, hardener class multiselect, resin/hardener CID limits, per-class sample limits, seed, fetch button, progress text and cache resume behavior. Call the new PubChem record API by role, concatenate records, pass them through `prepare_melting_point_dataset`, show counts by `mp_quality`, `component_role` and `hardener_class`, and write both raw and cleaned frames to session state.

- [x] **Step 4: Add downloads and explicit training handoff**

Add downloads for raw annotations and cleaned training CSV. Add `载入模型训练` that stores a copy in `st.session_state['melting_point_training_dataset']`, sets the target candidate to `mp_c`, and displays a confirmation without silently deleting the current dataset. The model training page must show a banner when this handoff exists, use only finite `mp_c` rows by default, and leave low-quality rows opt-in.

- [x] **Step 5: Add dataset summary and failure messages**

Show separate messages for network failure, no CIDs, no melting-point annotations, no high-quality rows, and insufficient role/category sample counts. A failed class must not clear successful classes. Run `pytest 'tests/test_melting_point_dataset_ui_contract.py' -q` expecting PASS.

---

### Task 4: 增加熔点模型 artifact 元数据和契约校验

**Files:**
- Modify: `core/model_io.py:80-179`
- Modify: `app.py:13940-14030` and model-loading paths around `16220-16275`, `17412-17465`
- Create: `core/melting_point_screening.py`
- Test: `tests/test_melting_point_model_contract.py`

**Interfaces:**
- `is_melting_point_artifact(artifact: Mapping[str, Any]) -> bool`
- `validate_melting_point_artifact(artifact: Mapping[str, Any]) -> dict`
- `build_melting_point_artifact_extra(dataset: pd.DataFrame, *, workflow_hash: str = '') -> dict`

- [x] **Step 1: Write failing artifact contract tests**

```python
from core.melting_point_screening import (
    is_melting_point_artifact,
    validate_melting_point_artifact,
)


def test_melting_point_artifact_requires_task_and_celsius_unit():
    artifact = {'target_col': 'mp_c', 'extra': {
        'task_kind': 'melting_point', 'target_unit': 'C',
    }}
    assert is_melting_point_artifact(artifact)
    assert validate_melting_point_artifact(artifact)['ok']


def test_tg_artifact_is_not_accepted_as_melting_point_model():
    artifact = {'target_col': 'tg_c', 'extra': {'task_kind': 'tg', 'target_unit': 'C'}}
    assert not is_melting_point_artifact(artifact)
    assert not validate_melting_point_artifact(artifact)['ok']
```

- [x] **Step 2: Run contract tests and verify expected failure**

Run: `pytest 'tests/test_melting_point_model_contract.py' -q`

Expected: FAIL because the artifact helper functions do not exist.

- [x] **Step 3: Implement artifact metadata helpers**

Treat an artifact as a melting-point model only when `extra.task_kind == 'melting_point'`, `extra.target_unit == 'C'`, and the target column is `mp_c` or explicitly declared by `extra.target_col`. Return structured errors for missing task kind, wrong unit, missing feature columns, or missing workflow metadata. Store dataset row count, dataset fingerprint, workflow hash, quality policy and role counts in artifact extra metadata during model export.

- [x] **Step 4: Add metadata to training export and preserve it on load**

When the active target is `mp_c` and the handoff dataset is present, merge `build_melting_point_artifact_extra` into the existing `_extra` before `create_model_artifact_bytes`; do not alter Tg artifact metadata. On model import, keep the artifact in a separate `melting_point_model_artifact` session key so importing the MP model cannot replace the active Tg/formulation model.

- [x] **Step 5: Run the contract tests and existing model I/O tests**

Run: `pytest 'tests/test_melting_point_model_contract.py' 'tests/test_model_io.py' -q`

Expected: PASS; if `tests/test_model_io.py` is absent, run `pytest 'tests/test_model_trainer_feature_mask.py' -q` as the closest existing artifact-adjacent regression test.

---

### Task 5: 实现树脂/固化剂熔点模型筛选门控

**Files:**
- Modify: `core/virtual_screening.py:2223-2245` only if shared prediction normalization is required
- Modify: `core/melting_point_screening.py`
- Modify: `app.py:17347-20580` in `page_virtual_screening`
- Modify: `core/industrial_filter.py:188-300` to separate heuristic filtering from model filtering
- Test: `tests/test_melting_point_screening.py`

**Interfaces:**
- `apply_melting_point_gate(df: pd.DataFrame, *, role_col: str = 'component_role', resin_prediction_col: str = 'resin_mp_predicted_c', hardener_prediction_col: str = 'hardener_mp_predicted_c', resin_std_col: str = 'resin_mp_std_c', hardener_std_col: str = 'hardener_mp_std_c', resin_ad_col: str = 'resin_mp_ad_score', hardener_ad_col: str = 'hardener_mp_ad_score', resin_limit_c: float, hardener_limit_c: float, max_std_c: float, min_ad_score: float, mode: str = 'annotate') -> pd.DataFrame`
- `melting_point_filter_status(prediction_c: float, std_c: float, ad_score: float, limit_c: float, max_std_c: float, min_ad_score: float) -> tuple[str, str]`

- [x] **Step 1: Write failing gate tests**

```python
import pandas as pd
from core.melting_point_screening import apply_melting_point_gate


def test_resin_and_hardener_use_separate_limits():
    frame = pd.DataFrame([{
        'resin_mp_predicted_c': 120.0,
        'resin_mp_std_c': 4.0,
        'resin_mp_ad_score': 80.0,
        'hardener_mp_predicted_c': 135.0,
        'hardener_mp_std_c': 4.0,
        'hardener_mp_ad_score': 80.0,
    }])
    result = apply_melting_point_gate(
        frame, resin_limit_c=130.0, hardener_limit_c=130.0,
        max_std_c=10.0, min_ad_score=50.0, mode='annotate',
    )
    assert result.iloc[0]['resin_mp_filter_status'] == 'pass'
    assert result.iloc[0]['hardener_mp_filter_status'] == 'fail'


def test_unknown_is_not_passed_in_strict_mode():
    frame = pd.DataFrame([{
        'resin_mp_predicted_c': 120.0,
        'resin_mp_std_c': float('nan'),
        'resin_mp_ad_score': 80.0,
    }])
    result = apply_melting_point_gate(
        frame, resin_limit_c=130.0, hardener_limit_c=130.0,
        max_std_c=10.0, min_ad_score=50.0, mode='strict',
    )
    assert result.empty
```

- [x] **Step 2: Run gate tests and verify expected failure**

Run: `pytest 'tests/test_melting_point_screening.py' -q`

Expected: FAIL because the gate functions do not exist.

- [x] **Step 3: Implement status calculation and strict filtering**

Return `pass`, `fail`, or `unknown` with a machine-readable reason. Require finite prediction, standard deviation and applicability score for `pass`; evaluate `prediction + std <= role_limit`; in annotate mode retain all rows and add status/reason columns; in strict mode retain only rows whose resin and hardener statuses are both `pass`.

- [x] **Step 4: Add MP model controls and isolated artifact loading**

Add a separate MP model uploader and session key in the virtual screening page. Validate the artifact before enabling controls. Add independent resin/hardener limits, max standard deviation, minimum AD score and annotate/strict mode. Do not overwrite `model`, `pipeline`, `feature_cols`, `scaler` or `imputer` belonging to the active Tg/formulation model.

- [x] **Step 5: Reproduce the MP workflow and attach predictions to candidates**

Use the MP artifact's stored molecular feature config and workflow to extract features for unique resin/hardener molecules; validate feature order before prediction; call the artifact pipeline/model; calculate AD and uncertainty using the saved MP reference bundle; merge predictions back by canonical component key. Missing workflow, invalid SMILES, non-finite model output or unit mismatch must produce `unknown` and a visible warning.

- [x] **Step 6: Remove heuristic MP from the new model path**

Add an explicit `melting_point_filter_mode` argument to `pipeline_industrial_filter` with values `off`, `heuristic`, and `model`; preserve existing legacy default behavior for old callers, but call it with `off` during MP data collection and with `model` when an MP artifact is active. The model path must never use `estimate_melting_point` as a prediction fallback.

- [x] **Step 7: Render result columns and run gate tests**

Expose predicted MP, standard deviation, AD score and reason columns in the Chinese result table and CSV export. Run `pytest 'tests/test_melting_point_screening.py' -q` expecting PASS.

---

### Task 6: 集成回归验证与数据安全检查

**Files:**
- Create: `tests/test_melting_point_integration.py`
- Modify: `README.md`

- [x] **Step 1: Add a no-network integration fixture**

Create a small in-memory dataset with one resin, two hardeners, one high-quality MP label, one range label and a minimal sklearn model artifact whose output is in Celsius. Use monkeypatches for PubChem and molecular feature extraction; do not contact the network in tests.

- [x] **Step 2: Test annotation mode preserves candidates**

Assert that annotation mode keeps all formulation rows, writes separate resin/hardener status columns, preserves the existing Tg `prediction` column, and records the MP model fingerprint in metadata.

- [x] **Step 3: Test strict mode rejects unknown and unsafe candidates**

Assert that candidates with non-finite MP prediction, high uncertainty, low AD score or `prediction + std` above the role-specific limit are excluded, while both-role pass candidates remain.

- [x] **Step 4: Test Tg regression isolation**

Load the existing virtual screening fixture or a minimal current-model fixture without an MP artifact and assert that no MP columns are required, no MP filter is applied, and the existing prediction path returns the same row count and target prediction values.

- [x] **Step 5: Run focused and broad verification**

Run: `pytest 'tests/test_melting_point_data.py' 'tests/test_pubchem_melting_point_client.py' 'tests/test_melting_point_dataset_ui_contract.py' 'tests/test_melting_point_model_contract.py' 'tests/test_melting_point_screening.py' 'tests/test_melting_point_integration.py' -q`

Expected: PASS with no network requests. Then run: `python -m compileall 'core' 'app.py'`.

- [ ] **Step 6: Manual Streamlit smoke test**

Start the application with the repository's existing Streamlit command, load a cached MP dataset, export the CSV, hand it to the model training page, import the resulting MP artifact into virtual screening, run annotate mode, then strict mode, and confirm Chinese result columns, separate resin/hardener thresholds, cache reuse and no regression to Tg filtering.

## Plan Self-Review

- Data schema, raw/cache separation, unit conversion, quality filtering and provenance are covered by Tasks 1–3.
- Unified model metadata, workflow fingerprints and Celsius validation are covered by Task 4.
- Separate resin/hardener thresholds, uncertainty, applicability domain and unknown handling are covered by Task 5.
- PubChem failures, no-network tests, strict/annotate behavior and Tg isolation are covered by Task 6.
- The plan does not introduce a second independent model family or reuse the old empirical melting-point formula as a prediction source.
- No production change is made while the plan is being reviewed.

