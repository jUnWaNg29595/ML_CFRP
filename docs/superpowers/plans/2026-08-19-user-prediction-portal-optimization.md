# CFRP User Prediction Portal Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the external prediction portal consume only administrator-published, versioned models from the CFRP platform and reproduce the exact training-time input and molecular feature contract for form and batch predictions.

**Architecture:** Keep UserPrediction.py as a separate Streamlit service on port 8555, but move model publishing and validation into the main app.py. Add pure, Streamlit-independent portal contract/runtime helpers under core/; the main app writes controlled published artifacts and the portal reads only enabled releases. Both manual and CSV/Excel inputs pass through one strict prediction path, with no silent feature padding.

**Tech Stack:** Python 3.10, Streamlit, pandas, NumPy, joblib, scikit-learn-compatible pipelines, existing molecular workflow and prediction contract helpers, pytest.

## Global Constraints

- External users only see Chinese input fields, validation results, predictions, applicability warnings, and downloads.
- The portal remains an independent Streamlit process at http://localhost:8555.
- The main platform never starts or stops the 8555 process from a Streamlit callback.
- Missing model features, missing molecular workflows, and ambiguous feature order are blocking errors; never silently fill them with zero or NaN.
- Published model versions are manually released, versioned, and rollback-capable; users see only enabled releases.
- Form and CSV/Excel prediction must call the same pure prediction preparation/runtime path.
- Preserve unrelated dirty-worktree files; do not reset, clean, or overwrite user changes.
- Use the project interpreter C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe for tests.

---

### Task 1: Add the portal publication contract and version store

**Files:**
- Create: core/prediction_portal.py
- Create: tests/test_prediction_portal.py
- Modify: core/prediction_contract.py:resolve_prediction_feature_contract only if the new publication validator needs a shared diagnostic field

**Interfaces:**
- Consumes: model artifact dictionaries, optional training DataFrame, molecular workflow payload, and existing feature-contract reports.
- Produces pure helpers used by app.py and UserPrediction.py:
  - build_prediction_contract(*, artifact, feature_cols, target_col, workflow=None, training_frame=None, source_frame=None) -> dict[str, Any]
  - validate_publication_artifact(artifact, contract=None) -> dict[str, Any]
  - make_publication_entry(*, material_key, target_key, artifact_path, artifact_hash, label, unit, description, contract, metrics, version, published_at) -> dict[str, Any]
  - activate_publication(config, *, material_key, target_key, entry) -> dict[str, Any]
  - rollback_publication(config, *, material_key, target_key, version) -> dict[str, Any]
  - portal_health_label(running: bool) -> str
  - should_show_publication(contract_report: dict[str, Any]) -> bool
  - select_active_publication(models: list[dict[str, Any]]) -> dict[str, Any] | None

- [ ] Step 1: Write failing tests for strict contract and version behavior

    import pandas as pd

    from core.prediction_portal import (
        activate_publication,
        build_prediction_contract,
        rollback_publication,
        validate_publication_artifact,
    )


    def test_publication_contract_records_numeric_ranges_and_workflow_sources():
        artifact = {
            'model': object(),
            'pipeline': None,
            'feature_cols': ['resin_xtb_gap', 'curing_agent_xtb_gap'],
            'target_col': 'Tg',
            'metrics': {'r2': 0.91},
            'extra': {},
        }
        training_frame = pd.DataFrame({
            'resin_xtb_gap': [1.0, 2.0],
            'curing_agent_xtb_gap': [3.0, 5.0],
        })

        contract = build_prediction_contract(
            artifact=artifact,
            feature_cols=artifact['feature_cols'],
            target_col='Tg',
            training_frame=training_frame,
        )

        assert contract['feature_cols'] == artifact['feature_cols']
        assert contract['numeric_ranges']['resin_xtb_gap'] == {'min': 1.0, 'max': 2.0}
        assert contract['source_columns'] == []


    def test_publication_rejects_missing_molecular_workflow_for_molecular_sources():
        artifact = {
            'model': object(),
            'pipeline': None,
            'feature_cols': ['resin_xtb_gap'],
            'target_col': 'Tg',
            'extra': {},
        }
        contract = {
            'feature_cols': ['resin_xtb_gap'],
            'source_columns': [{'column': 'resin_smiles_1', 'roles': ['resin']}],
            'workflow_present': False,
        }

        report = validate_publication_artifact(artifact, contract)

        assert report['ok'] is False
        assert any('workflow' in error.lower() for error in report['errors'])


    def test_activate_and_rollback_keep_one_active_version():
        config = {'materials': {'epoxy_resin': {'targets': {'tg': {'models': []}}}}}
        first = {'id': 'tg-v1', 'version': 'v1', 'enabled': True}
        second = {'id': 'tg-v2', 'version': 'v2', 'enabled': True}

        activate_publication(config, material_key='epoxy_resin', target_key='tg', entry=first)
        activate_publication(config, material_key='epoxy_resin', target_key='tg', entry=second)
        rollback_publication(config, material_key='epoxy_resin', target_key='tg', version='v1')

        models = config['materials']['epoxy_resin']['targets']['tg']['models']
        assert [item['enabled'] for item in models] == [True, False]

- [ ] Step 2: Run the focused tests and verify the expected missing-interface failures

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' -q

Expected: FAIL because core.prediction_portal does not yet expose the four functions.

- [ ] Step 3: Implement the pure contract and publication helpers

Implement core/prediction_portal.py with these rules:

1. Resolve the exact feature list through resolve_prediction_feature_contract() and reject a report whose ok value is false.
2. Extract molecular source columns through collect_workflow_source_columns() from core/prediction_molecular_baseline.py when a workflow is present.
3. Store workflow_hash, workflow_schema_version, feature_cols, target_col, source_columns, workflow_present, pipeline_present, imputer_present, scaler_present, and numeric_ranges.
4. Compute each numeric range from finite values only; omit a range when no finite values exist.
5. Reject artifacts without a model/pipeline, target column, exact feature contract, or required workflow/preprocessor metadata.
6. On activation, append the entry and set every other version for that material/target to enabled=False.
7. On rollback, enable only the requested existing version and raise a clear ValueError for unknown versions.

- [ ] Step 4: Run focused tests and the existing contract tests

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' 'tests/test_prediction_feature_contract.py' 'tests/test_prediction_molecular_baseline.py' -q

Expected: PASS.

- [ ] Step 5: Commit the contract layer

    git add -- 'core/prediction_portal.py' 'tests/test_prediction_portal.py'
    git commit -m 'feat: add published prediction portal contract'

### Task 2: Persist complete prediction metadata in exported artifacts

**Files:**
- Modify: core/model_io.py:create_model_artifact
- Modify: app.py:13850-14050 and the equivalent export block around 14260-14330
- Modify: tests/test_prediction_portal.py

**Interfaces:**
- Consumes: build_prediction_contract() from Task 1, the current train_result, effective feature columns, workflow metadata, and preprocessing objects.
- Produces: every newly exported model artifact contains extra.prediction_contract and the portal can validate it without depending on Streamlit session state.

- [ ] Step 1: Add a failing artifact round-trip test

    from core.model_io import create_model_artifact


    def test_model_artifact_preserves_prediction_contract():
        contract = {
            'schema_version': 1,
            'feature_cols': ['resin_xtb_gap'],
            'target_col': 'Tg',
            'source_columns': [],
        }
        artifact = create_model_artifact(
            model_name='XGBoost',
            target_col='Tg',
            feature_cols=['resin_xtb_gap'],
            model=object(),
            extra={'prediction_contract': contract},
        )

        assert artifact['extra']['prediction_contract']['schema_version'] == 1
        assert artifact['extra']['prediction_contract']['feature_cols'] == ['resin_xtb_gap']

- [ ] Step 2: Run the test before implementation

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py::test_model_artifact_preserves_prediction_contract' -q

Expected: FAIL if the exported artifact does not preserve the contract in the tested path.

- [ ] Step 3: Add contract construction to both main-platform export paths

Before calling create_model_artifact_bytes() in each export block, build one contract from the same effective_feature_cols, target_col, workflow, preprocessing objects, and training reference used for the model. Add it to _extra as prediction_contract. Do not create a second feature list or infer order from the processed table.

Update create_model_artifact() to preserve an explicitly supplied extra.prediction_contract unchanged and add an artifact schema version inside the contract. Keep old artifact compatibility when the field is absent.

- [ ] Step 4: Add migration diagnostics for legacy artifacts

Make validate_publication_artifact() report legacy artifacts as status='needs_validation' when they lack prediction_contract; do not silently mark them publishable. Keep ordinary main-platform prediction behavior unchanged for legacy imports until the publication flow is used.

- [ ] Step 5: Run artifact, contract, and model I/O tests

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' 'tests/test_prediction_feature_contract.py' 'tests/test_prediction_molecular_baseline.py' -q

Expected: PASS.

- [ ] Step 6: Commit artifact metadata changes

    git add -- 'core/model_io.py' 'app.py' 'tests/test_prediction_portal.py'
    git commit -m 'feat: export prediction contract with trained models'

### Task 3: Add main-platform portal switch and model publishing UI

**Files:**
- Modify: app.py:3464-3630 for the sidebar entry and port health display
- Modify: app.py:16123-16520 for the current-model publication panel
- Modify: tests/test_prediction_portal.py

**Interfaces:**
- Consumes: prediction_portal contract/publication helpers, current imported_model_artifact or newly exported artifact bytes, current material/target selection, and prediction_portal/prediction_config.json.
- Produces: a sidebar switch named 用户预测门户, a new-tab link to http://localhost:8555, port state feedback, and a guarded 发布到用户门户 workflow.

- [ ] Step 1: Add pure tests for portal URL and release gating

    from core.prediction_portal import portal_health_label, should_show_publication


    def test_portal_health_label_distinguishes_running_and_stopped():
        assert portal_health_label(True) == '可访问'
        assert portal_health_label(False) == '未启动'


    def test_publication_button_is_blocked_when_contract_is_invalid():
        assert should_show_publication({'ok': False}) is False
        assert should_show_publication({'ok': True}) is True

- [ ] Step 2: Run the new tests and observe the missing helper failures

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py::test_portal_health_label_distinguishes_running_and_stopped' 'tests/test_prediction_portal.py::test_publication_button_is_blocked_when_contract_is_invalid' -q

Expected: FAIL because the pure UI decision helpers are not implemented.

- [ ] Step 3: Implement the sidebar entry without process management

Add a pure portal_health_label() helper using a short socket connection to 127.0.0.1:8555. In render_sidebar(), render a checkbox stored in st.session_state['user_prediction_portal_enabled']; only when true render the port state and st.link_button('打开用户预测门户', 'http://localhost:8555'). Disable the checkbox only while the existing global task lock is active. Do not call subprocess, os.startfile, or streamlit run from the callback.

- [ ] Step 4: Implement the guarded publication panel

Add _render_user_prediction_publish_panel() near page_prediction() and call it only after the current model feature contract has been resolved. The panel must:

1. Load current artifact bytes from st.session_state['_trained_model_artifact'], last_export_model_path, or imported_model_artifact.
2. Build/validate prediction_contract using the current training frame and workflow.
3. Display contract errors and disable publication if validation fails.
4. Accept only user-facing label, unit, description, and applicability notes; do not offer feature override.
5. Compute SHA-256, assign the next vN version for the material/target, copy the artifact to prediction_portal/managed_models/<material>/<target>/, and atomically update the JSON config.
6. Deactivate the previous version after successful write and show the published version.
7. Provide a rollback selector for existing versions and call rollback_publication().

Use st.form so editing fields does not repeatedly write config. Keep release state separate from the existing session model state.

- [ ] Step 5: Run pure tests and compile-check the main app

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' -q
    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile 'app.py' 'core/prediction_portal.py'

Expected: PASS with no syntax errors.

- [ ] Step 6: Commit main-platform publishing and switch

    git add -- 'app.py' 'core/prediction_portal.py' 'tests/test_prediction_portal.py'
    git commit -m 'feat: add main platform portal publishing controls'

### Task 4: Make the external portal release-only and strict

**Files:**
- Modify: UserPrediction.py:217-510 for model loading and prediction preparation
- Modify: UserPrediction.py:770-912 for the external user page
- Modify: UserPrediction.py:1302-1345 for sidebar/main mode removal
- Create: tests/test_user_prediction_runtime.py

**Interfaces:**
- Consumes: enabled publication entries and prediction_contract from Task 1.
- Produces: external-only portal behavior with one model per performance target, no admin/model upload UI, and strict missing-column handling.

- [ ] Step 1: Write failing tests for release filtering and strict missing input behavior

    import pandas as pd

    from core.prediction_portal import select_active_publication
    from UserPrediction import validate_external_input


    def test_only_enabled_publication_is_selected():
        models = [
            {'id': 'v1', 'version': 'v1', 'enabled': False},
            {'id': 'v2', 'version': 'v2', 'enabled': True},
        ]

        assert select_active_publication(models)['id'] == 'v2'


    def test_external_input_rejects_missing_features_instead_of_padding():
        report = validate_external_input(
            pd.DataFrame([{'resin_xtb_gap': 1.0}]),
            {'feature_cols': ['resin_xtb_gap', 'curing_agent_xtb_gap'], 'source_columns': []},
        )

        assert report['ok'] is False
        assert report['missing_columns'] == ['curing_agent_xtb_gap']

- [ ] Step 2: Run the tests and verify the current unsafe behavior is caught

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_user_prediction_runtime.py' -q

Expected: FAIL because the release selector and strict input validator do not yet exist.

- [ ] Step 3: Remove external model administration and model selection

Change render_sidebar() and main() so the user process renders only render_user_page(config). Remove the radio that exposes 管理页面, the admin tabs, model upload/replace controls, feature override text area, and the user-facing model multiselect. Keep the underlying config loader and registered entries needed for read-only operation.

- [ ] Step 4: Add release-only selection and strict input validation

Implement select_active_publication() to return the one enabled release or None when no release exists. Implement validate_external_input() to check required source/feature columns, duplicate normalized columns, numeric conversion, finite values, and configured ranges. Return structured missing_columns, duplicate_columns, invalid_values, out_of_range_columns, and errors lists. Keep this task independent of molecular replay; Task 5 will consume this validator when it routes predictions.

Replace align_prediction_frame() in the external path with validation followed by exact column selection. A missing required value stops the row; no zero/NaN padding is permitted.

- [ ] Step 5: Run the focused runtime tests and compile-check the portal

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_user_prediction_runtime.py' 'tests/test_prediction_portal.py' -q
    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile 'UserPrediction.py'

Expected: PASS.

- [ ] Step 6: Commit the external-only portal changes

    git add -- 'UserPrediction.py' 'tests/test_user_prediction_runtime.py' 'core/prediction_portal.py'
    git commit -m 'feat: restrict user portal to published models'

### Task 5: Share exact molecular workflow replay for form and batch inputs

**Files:**
- Create: core/prediction_runtime.py
- Modify: UserPrediction.py:430-510 and UserPrediction.py:858-900
- Modify: tests/test_user_prediction_runtime.py

**Interfaces:**
- Consumes: published artifact, prediction_contract, molecular workflow, source-column role metadata, and raw form/batch DataFrame.
- Produces: prepare_external_prediction_frame(input_df, artifact, contract) -> tuple[pd.DataFrame, dict[str, Any]] and run_published_prediction(entry, input_df) -> dict[str, Any].

- [ ] Step 1: Write failing tests proving both input modes share workflow order

    import pandas as pd

    from core.prediction_runtime import prepare_external_prediction_frame


    def test_prediction_runtime_replays_declared_source_order():
        workflow = {
            'steps': [
                {'step_id': 'hardener', 'order': 1, 'role': 'hardener', 'source_columns': ['curing_agent_smiles_1']},
                {'step_id': 'resin', 'order': 0, 'role': 'resin', 'source_columns': ['resin_smiles_1']},
            ],
            'merge_order': ['resin', 'hardener'],
        }
        artifact = {'extra': {'molecular_feature_workflow': workflow}}
        contract = {
            'feature_cols': ['resin_smiles_1', 'curing_agent_smiles_1'],
            'source_columns': [
                {'column': 'resin_smiles_1', 'roles': ['resin']},
                {'column': 'curing_agent_smiles_1', 'roles': ['hardener']},
            ],
        }
        frame = pd.DataFrame([{'resin_smiles_1': 'C1CO1', 'curing_agent_smiles_1': 'NCCN'}])

        prepared, report = prepare_external_prediction_frame(frame, artifact, contract)

        assert report['ok'] is True
        assert list(prepared.columns) == ['resin_smiles_1', 'curing_agent_smiles_1']

- [ ] Step 2: Run the molecular runtime test before implementation

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_user_prediction_runtime.py::test_prediction_runtime_replays_declared_source_order' -q

Expected: FAIL because core.prediction_runtime does not yet exist.

- [ ] Step 3: Implement the shared preparation runtime

Implement the runtime in this order:

1. Validate the raw input against contract.source_columns using collect_workflow_source_columns() and preserve declared order.
2. For workflow-bearing artifacts, call the existing execute_molecular_feature_workflow() path with the raw source frame and stored workflow payload; do not substitute MACCS, xTB defaults, or input-column order.
3. Apply the saved post-feature mapping and feature mask when present.
4. Resolve the exact model feature contract with resolve_prediction_feature_contract().
5. Return a frame with exactly the resolved feature columns and a structured report containing workflow hash, feature count, missing columns, and warnings.
6. For non-molecular artifacts, validate and select the exact numeric input columns without replaying a molecular workflow.

run_published_prediction() must load the artifact through the existing cached joblib loader, call prepare_external_prediction_frame(), then call either the pipeline or model plus saved imputer/scaler. It must return predictions, validation, feature_cols, publication_version, and operation_id.

- [ ] Step 4: Route both portal tabs through the same runtime

Update manual form construction and batch upload handling so both create a DataFrame and call run_published_prediction(). Remove the warning that says molecular workflows are not yet executed. Display workflow/replay failure as a blocking error, preserve row-level errors for batch input, and never call the old execute_predictions() path for a published model.

- [ ] Step 5: Add tests for row-level batch failures and non-molecular models

Cover one valid row and one missing-source row, assert only the valid row reaches model prediction, and assert a numeric-only artifact still uses the exact feature order. Use a tiny estimator double with predict(); do not load a real trained model in unit tests.

- [ ] Step 6: Run focused runtime tests and commit

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_user_prediction_runtime.py' 'tests/test_prediction_molecular_baseline.py' 'tests/test_prediction_feature_contract.py' -q
    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile 'UserPrediction.py' 'core/prediction_runtime.py'

Expected: PASS.

    git add -- 'core/prediction_runtime.py' 'UserPrediction.py' 'tests/test_user_prediction_runtime.py'
    git commit -m 'feat: replay published molecular workflows in portal'

### Task 6: Add configuration migration, user-facing validation output, and end-to-end verification

**Files:**
- Modify: UserPrediction.py:191-215 for config migration and atomic reads/writes
- Modify: UserPrediction.py:500-770 for Chinese validation/result tables and downloads
- Modify: 启动预测平台.bat only if the existing launcher needs a clearer portal status message
- Modify: README.md and 局域网访问配置.bat for port 8555 usage and optional firewall setup
- Create: tests/test_prediction_portal_integration.py

**Interfaces:**
- Consumes: release entries and runtime results from Tasks 1-5.
- Produces: backward-compatible config migration, Chinese external UI, stable result exports, and verified startup/documentation.

- [ ] Step 1: Write migration and result-export tests

    import pandas as pd

    from UserPrediction import deep_merge_defaults, render_result_export_frame


    def test_legacy_config_is_read_as_unpublished_until_released():
        migrated = deep_merge_defaults(
            {'materials': {'epoxy_resin': {'targets': {'tg': {'models': [{'id': 'old'}]}}}}},
            {'materials': {'epoxy_resin': {'targets': {'tg': {'models': [{'id': 'old'}]}}}}},
        )

        assert migrated['materials']['epoxy_resin']['targets']['tg']['models'][0].get('enabled') is False


    def test_result_export_keeps_validation_status_columns():
        frame = render_result_export_frame(
            pd.DataFrame([{'resin_smiles_1': 'C1CO1'}]),
            predictions=[120.5],
            statuses=['可预测'],
            errors=[''],
        )

        assert list(frame.columns)[-2:] == ['预测状态', '校验信息']
        assert frame.iloc[0]['预测值'] == 120.5

- [ ] Step 2: Run migration/export tests and verify they fail for the current UI

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal_integration.py' -q

Expected: FAIL because legacy releases are currently enabled by default and the result helper does not exist.

- [ ] Step 3: Migrate config safely and make writes atomic

On load, add release defaults without changing existing unrelated material/target settings. Mark legacy model entries as enabled=False and publication_status='needs_validation' when they lack a valid publication contract. Write JSON to a sibling temporary file and replace the target only after a successful UTF-8 JSON write. Keep a timestamped backup before changing a non-empty config.

- [ ] Step 4: Replace external-facing tables and downloads

Add render_result_export_frame() that combines original input columns, Chinese prediction columns, 预测状态, and 校验信息. Update manual and batch result rendering to show only user-facing labels, units, range warnings, and row-level errors. Remove the current 当前配置 tab from external users because it exposes internal model/feature details.

- [ ] Step 5: Update launch and documentation text

Document both services:

    conda activate 'CFRP_env'
    Set-Location 'C:\Users\wangj\Desktop\CFRP系统\CFRP系统'
    python -m streamlit run 'app.py' --server.port 8501
    python -m streamlit run 'UserPrediction.py' --server.port 8555

Keep the existing 启动预测平台.bat launcher for UserPrediction.py. Add a separate optional firewall command for 8555; do not silently replace the main platform firewall rule for 8501.

- [ ] Step 6: Run the focused suite and full relevant regression suite

Run:

    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' 'tests/test_prediction_portal_integration.py' 'tests/test_user_prediction_runtime.py' 'tests/test_prediction_feature_contract.py' 'tests/test_prediction_molecular_baseline.py' 'tests/test_navigation.py' -q
    & 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m py_compile 'app.py' 'UserPrediction.py' 'core/prediction_portal.py' 'core/prediction_runtime.py'

Expected: all focused tests pass and all files compile.

- [ ] Step 7: Perform a manual two-process smoke test

1. Start the main platform on 8501 and verify the sidebar switch is off by default.
2. Turn on the switch and verify the 8555 link reports “未启动”.
3. Start 启动预测平台.bat, refresh the main platform, and verify the link reports “可访问”.
4. Publish a tiny valid numeric artifact from the main platform and confirm exactly one enabled release appears in prediction_config.json.
5. Open http://localhost:8555, submit one form prediction, and verify the result includes prediction value, status, and applicability information.
6. Upload a CSV containing one valid and one invalid row and verify the valid row predicts while the invalid row retains a reason.
7. Publish a second version, confirm the first is disabled, then roll back and confirm the first becomes active.

- [ ] Step 8: Commit final integration and documentation

    git add -- 'UserPrediction.py' 'README.md' '局域网访问配置.bat' 'tests/test_prediction_portal_integration.py'
    git commit -m 'docs: document published user prediction portal'

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover artifact and contract requirements; Task 3 covers the main-platform switch, publication, versioning, and rollback; Tasks 4-5 cover external-only behavior, strict validation, shared form/batch execution, and molecular workflow replay; Task 6 covers migration, user-facing output, startup, and regression verification.
- Placeholder scan: No TBD, TODO, or unspecified implementation steps are used.
- Type consistency: build_prediction_contract(), validate_publication_artifact(), prepare_external_prediction_frame(), and run_published_prediction() are introduced before later tasks consume them, with concrete parameters and return types.
- Scope: No automatic process spawning, unrelated UI redesign, or model-training changes are included.
- Worktree safety: Existing dirty and untracked files remain untouched unless explicitly listed in a task.
