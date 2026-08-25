# 用户预测门户科研 UI 与多模型 AI 集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏现有 `8501` 主平台、`8555` 独立门户、模型发布和手动预测能力的前提下，构建深蓝实验室风格的用户预测门户，并接入可切换的 OpenAI 兼容 AI 服务，实现人工确认后的结构化输入、受控 Python 预测任务和结果解释。

**Architecture:** `UserPrediction.py` 只负责页面、交互和任务轮询；`core/portal_ai*.py` 组成独立 AI 服务层；`core/portal_prediction.py` 和 `core/portal_tasks.py` 组成可信 Python 计算层。AI 只能生成建议和解释，所有结构校验、特征流程、模型预处理、预测和任务状态由 Python 完成。

**Tech Stack:** Python 3.10、Streamlit、pandas、numpy、现有 `core.model_io` 与预测契约、pytest、OpenAI 兼容 `chat/completions` 接口、本地 JSON 配置和任务快照文件。

## Global Constraints

- 保留主平台默认端口 `8501`、独立用户门户默认端口 `8555` 和现有 `core.prediction_portal` 启停行为。
- 不覆盖或删除现有模型、特征流程、训练数据、缓存、备份和未提交工作区文件；配置迁移前先创建带时间戳的备份。
- Python 继续负责 SMILES/BigSMILES、多组分、分子特征、xTB/描述符、特征顺序、预处理、模型加载和 CFRP/环氧预测。
- AI 不得执行任意 Python/PowerShell/系统命令，不得修改模型、特征流程、训练数据、配置或预测数值。
- AI 解析字段默认“待确认”；未完成用户确认、结构校验和范围校验时禁止创建任务。
- API Key 只在服务端本地配置读取，不得进入浏览器、公开配置、导出文件、日志或异常文本。
- AI 未配置、超时、认证失败、限流或解析失败时，手动输入、CSV/Excel 批量预测和已确认的 Python 预测必须继续可用。
- 任务状态固定为 `queued`、`validating`、`featuring`、`predicting`、`explaining`、`completed`、`failed`、`cancelled`。
- PowerShell 验证命令中的字符串值使用单引号；不执行 `git reset --hard`、`git clean` 或批量删除未跟踪文件。
- 每个任务完成独立测试后再进入下一任务；提交只包含本计划对应文件。

## 文件与边界

**新增文件**

- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_ai_schema.py`：AI 解析、确认字段、任务请求、结果解释和错误契约。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_ai_config.py`：多服务配置、备份、脱敏、导出过滤和校验。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_ai.py`：OpenAI 兼容客户端、重试、限流、响应解析和错误分类。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_prediction.py`：受控预测入口，复用发布模型和特征契约。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_tasks.py`：任务持久化、后台执行、取消、重试和状态快照。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/core/portal_ui.py`：深蓝主题、CSS token、线性 SVG 图标和状态组件。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/tests/test_portal_ai_schema.py`、`test_portal_ai_config.py`、`test_portal_ai.py`、`test_portal_prediction.py`、`test_portal_tasks.py`：对应模块测试。

**修改文件**

- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/app.py`：在现有门户管理区增加 AI 服务配置和连通性测试；不改变门户启停。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/UserPrediction.py`：接入主题、首页材料选择、AI 助手、人工确认、任务轮询和结果解释；保留手动/批量路径。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/README.md`：补充 `8501/8555` 启动、AI 配置、密钥保护、降级和测试说明。
- `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/prediction_portal/prediction_config.json`：仅在迁移确有需要时增加无密钥版本字段，绝不保存 API Key。

**现有接口复用点**

- `core/prediction_portal.py` 的 `portal_process_status`、`start_prediction_portal`、`stop_prediction_portal` 和发布契约继续有效。
- `UserPrediction.py` 的 `load_config`、`save_config`、`resolve_model_path` 与现有单样本/批量预测函数继续作为兼容路径。
- `core/prediction_contract.py` 和既有分子特征工作流解析逻辑是 Python 预测入口的唯一特征契约来源。

**测试 fixture 约定**

- `tests/conftest.py` 中新增 `_config()`、`_request(confirmed_by_user=True, inputs=None)`、`_fake_artifact()` 和 `_result()`；它们只返回最小内存对象，不读取真实模型、不访问网络、不写入用户目录。
- `tests/test_user_prediction_ai_flow.py` 中定义 `build_ai_confirmation_state(fields, confirmed_fields)`、`confirm_ai_field(state, name, value)`、`can_submit_ai_prediction(state)` 和 `fallback_input_mode(error_code)`，这些 helper 只测试确认门禁，不调用 Streamlit。
- `tests/test_portal_task_view.py` 中定义纯字符串渲染 helper `render_task_snapshot(snapshot)` 和 `render_result(result)`，后续再由 `core.portal_ui` 提供同名生产实现。
- `tests/test_portal_integration.py` 中定义 `load_config_from_fixture(name)`、`manual_prediction_is_available(config)`、`ai_mode_is_optional(root)` 和 `migrate_ai_config(root)` 的测试适配器，所有路径指向 `tmp_path` 或固定 fixture，不调用真实端口。

---

### Task 1: 建立 AI 数据契约

**Files:** Create `core/portal_ai_schema.py`; create `tests/test_portal_ai_schema.py`.

**Interfaces:** `AIFieldSuggestion`、`AIParseResponse`、`ConfirmedPredictionRequest`、`PredictionResultSummary`、`AIExplanationResponse`；`parse_ai_response(value: object) -> AIParseResponse`；`validate_confirmed_request(value: object) -> ConfirmedPredictionRequest`；`sanitize_ai_context(value: object) -> dict[str, object]`。

- [ ] **Step 1: Write the failing test**

```python
def test_confirmation_is_required_and_secrets_are_removed():
    with pytest.raises(ValueError, match='confirmed'):
        validate_confirmed_request({'material_type': 'epoxy_resin', 'inputs': {}})
    response = parse_ai_response({'recognized_fields': {'resin_smiles': 'CCO'}, 'unexpected': 'x'})
    assert response.recognized_fields == {'resin_smiles': 'CCO'}
    assert response.warnings
    safe = sanitize_ai_context({'api_key': 'secret', 'user_text': 'run os.system()'})
    assert 'api_key' not in safe and 'os.system' not in safe['user_text']
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_ai_schema.py'`; **Expected:** FAIL because the module and contracts do not exist.
- [ ] **Step 3: Implement:** use strict allow-lists for material, target, source and field state; preserve uncertain values as `None`; require `material_type`、`target`、`inputs`、`confirmed_by_user=True` and `source`; reject unknown request keys; cap text length and remove secret-like keys and code-execution instructions from AI context.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_portal_ai_schema.py'`; **Expected:** PASS without network access.
- [ ] **Step 5: Commit:** `git add 'core/portal_ai_schema.py' 'tests/test_portal_ai_schema.py'; git commit -m 'feat: add portal AI data contracts'`。

### Task 2: 增加安全的多服务 AI 配置

**Files:** Create `core/portal_ai_config.py`; create `tests/test_portal_ai_config.py`; only migrate `prediction_portal/prediction_config.json` through code if necessary.

**Interfaces:** `AIServiceConfig` fields `service_id`、`label`、`provider`、`base_url`、`model`、`purpose`、`timeout_seconds`、`max_tokens`、`temperature`、`enabled`；`load_ai_config(root: Path)`、`save_ai_config(root, config)`、`redacted_ai_config(config)`、`exportable_ai_config(config)`、`validate_ai_config(config)`。

- [ ] **Step 1: Write the failing test**

```python
def test_key_is_stored_locally_but_never_exported(tmp_path):
    save_ai_config(tmp_path, {'services': [{'service_id': 'deepseek', 'api_key': 'sk-secret', 'base_url': 'https://api.deepseek.com/v1', 'model': 'deepseek-chat', 'enabled': True}]})
    payload = load_ai_config(tmp_path)
    assert payload['services'][0]['api_key'] == 'sk-secret'
    assert 'sk-secret' not in json.dumps(exportable_ai_config(payload))
    assert list((tmp_path / 'prediction_portal').glob('ai_config.*.bak'))
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_ai_config.py'`; **Expected:** FAIL because secure configuration helpers do not exist.
- [ ] **Step 3: Implement:** store full config in `prediction_portal/ai_config.json`, back up before each write, write atomically, validate URL/model/purpose/ranges, require HTTPS for non-local endpoints, allow `localhost`/`127.0.0.1` for Ollama/internal services, and mask keys in UI/export. Do not put the key into `prediction_config.json` or Streamlit widget defaults.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_portal_ai_config.py'`; **Expected:** PASS and a backup exists after the second save.
- [ ] **Step 5: Commit:** `git add 'core/portal_ai_config.py' 'tests/test_portal_ai_config.py'; git commit -m 'feat: add secure portal AI provider config'`。

### Task 3: 实现 OpenAI 兼容客户端

**Files:** Create `core/portal_ai.py`; create `tests/test_portal_ai.py`.

**Interfaces:** `PortalAIClient(config: AIServiceConfig, transport=None)`；`parse_chat_completion(payload) -> str`；`parse_json_or_markdown_json(text) -> object`；`parse_input(context) -> AIParseResponse`；`explain_result(summary) -> AIExplanationResponse`。

- [ ] **Step 1: Write the failing test**

```python
def test_fenced_json_and_transient_retry_are_supported():
    assert parse_json_or_markdown_json('```json\n{"ok": true}\n```') == {'ok': True}
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PortalAITransientError('timeout')
        return {'choices': [{'message': {'content': '{"recognized_fields": {}}'}}]}
    result = PortalAIClient(_config('sk-secret'), transport=transport, sleep=lambda _: None).parse_input({'material_type': 'epoxy_resin', 'user_text': '树脂'})
    assert result.recognized_fields == {} and len(calls) == 2
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_ai.py'`; **Expected:** FAIL because client/parsers do not exist.
- [ ] **Step 3: Implement:** POST to `base_url.rstrip('/') + '/chat/completions'` with Bearer auth, bounded timeout/temperature/tokens and a fixed safety prompt; inject a transport seam for tests; retry only timeout/connection/408/409/425/429/5xx, at most two retries; classify 401/403/429/timeout/connection/malformed JSON; parse JSON and fenced JSON but reject ambiguous prose; never include key, headers or full request in errors.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_portal_ai.py'`; **Expected:** PASS without network.
- [ ] **Step 5: Commit:** `git add 'core/portal_ai.py' 'tests/test_portal_ai.py'; git commit -m 'feat: add bounded OpenAI-compatible portal client'`。

### Task 4: 建立可信 Python 预测入口

**Files:** Create `core/portal_prediction.py`; create `tests/test_portal_prediction.py`.

**Interfaces:** `validate_prediction_request(request, config) -> list[str]`；`load_published_portal_model(config, material_type, target)`；`run_confirmed_prediction(request, *, config, progress=None) -> PredictionResultSummary`。

- [ ] **Step 1: Write the failing test**

```python
def test_prediction_requires_confirmation_and_published_workflow(monkeypatch):
    with pytest.raises(ValueError, match='确认'):
        run_confirmed_prediction(_request(confirmed_by_user=False), config=_config())
    monkeypatch.setattr('core.portal_prediction._load_artifact', lambda entry: _fake_artifact())
    result = run_confirmed_prediction(_request(), config=_config(), progress=lambda *args: None)
    assert result.model_version == 'v1'
    assert result.feature_workflow_id == 'workflow-1'
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_prediction.py'`; **Expected:** FAIL because the trusted entry point does not exist.
- [ ] **Step 3: Implement:** resolve the active publication through existing contracts; reject disabled, ambiguous or `needs_validation` releases; validate target fields, current SMILES/BigSMILES rules, workflow source columns, numeric ranges and unknown keys; run published preprocessing/model exactly once; return prediction, unit, warnings, model version, workflow ID and non-sensitive summary; reject callables, source code, shell commands and arbitrary AI feature vectors.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_portal_prediction.py'`; **Expected:** PASS with mocked artifacts and no external xTB.
- [ ] **Step 5: Commit:** `git add 'core/portal_prediction.py' 'tests/test_portal_prediction.py'; git commit -m 'feat: add trusted portal prediction entry point'`。

---
### Task 5: 增加可恢复的后台任务执行

**Files:** Create `core/portal_tasks.py`; create `tests/test_portal_tasks.py`; modify `core/prediction_portal.py` only if a shared root helper is needed and preserve public signatures.

**Interfaces:** `PortalTaskManager(root: Path, executor=None)`；`create_task(request) -> str`；`get_task_snapshot(task_id) -> dict`；`cancel_task(task_id) -> dict`；`retry_task(task_id) -> str`；`wait_for_task(task_id, timeout) -> dict`。

- [ ] **Step 1: Write the failing test**

```python
def test_task_lifecycle_persists_completed_result(tmp_path, monkeypatch):
    monkeypatch.setattr('core.portal_tasks.run_confirmed_prediction', lambda request, config, progress: _result())
    manager = PortalTaskManager(tmp_path)
    task_id = manager.create_task(_request())
    snapshot = manager.wait_for_task(task_id, timeout=2)
    assert snapshot['status'] == 'completed'
    assert snapshot['progress'] == 100
    assert (tmp_path / 'prediction_portal' / 'tasks' / f'{task_id}.json').exists()
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_tasks.py'`; **Expected:** FAIL because task persistence and worker lifecycle do not exist.
- [ ] **Step 3: Implement:** persist one atomic JSON snapshot per task under `prediction_portal/tasks`; start work in a bounded executor without calling Streamlit APIs from workers; publish only serialized updates; honor cancellation between validation, feature, prediction and explanation; make retry create a new task ID; mark stale active tasks failed after restart; keep explanation optional so AI failure leaves Python prediction `completed`.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_portal_tasks.py'`; **Expected:** PASS without `ScriptRunContext` warnings.
- [ ] **Step 5: Commit:** `git add 'core/portal_tasks.py' 'tests/test_portal_tasks.py'; git commit -m 'feat: add reload-safe portal background tasks'`。

### Task 6: 在主平台增加 AI 服务管理页

**Files:** Modify `C:/Users/wangj/Desktop/CFRP系统/CFRP系统/app.py` near the existing portal management block; use `core/portal_ai_config.py`; extend its tests.

**Interfaces:** UI only calls `load_ai_config`、`validate_ai_config`、`save_ai_config`、`redacted_ai_config` and `PortalAIClient`; existing `开启门户`、`关闭门户`、`portal_process_status(port=8555)` behavior remains unchanged.

- [ ] **Step 1: Write the regression test**

```python
def test_ai_export_has_no_key_and_runtime_port_is_separate():
    config = {'services': [{'service_id': 'deepseek', 'api_key': 'sk-live-secret', 'model': 'deepseek-chat'}]}
    assert 'sk-live-secret' not in json.dumps(exportable_ai_config(config))
    assert 'port' not in default_ai_config()
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_ai_config.py'`; **Expected:** PASS before UI editing; fix the schema if runtime portal fields leak into AI config.
- [ ] **Step 3: Implement:** add an “AI 服务配置” expander next to portal controls with service selection, provider, OpenAI-compatible `base_url`, model, purpose, timeout, token cap, temperature, enable switch, masked key replacement, save, key-free export and connection test. Use a short non-sensitive client request. Show validation errors; backup before writes; never access Streamlit APIs from worker threads.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_prediction_portal.py' 'tests/test_portal_ai_config.py'`; **Expected:** PASS; manually verify portal status and no key display.
- [ ] **Step 5: Commit:** `git add 'app.py' 'core/portal_ai_config.py' 'tests/test_portal_ai_config.py'; git commit -m 'feat: add main platform AI service management'`。

### Task 7: 建立深蓝科研风门户 UI

**Files:** Create `core/portal_ui.py`; modify `UserPrediction.py`; create `tests/test_portal_ui.py`.

**Interfaces:** `inject_scientific_theme()`；`svg_icon(name, size=20, label=None) -> str`；`render_status_badge(status)`；`render_stage_timeline(stage, progress)`；`render_material_card(...)`。

- [ ] **Step 1: Write the failing test**

```python
def test_icons_are_linear_svg_and_timeline_has_trusted_stages():
    markup = svg_icon('molecule', size=20, label='分子')
    assert markup.startswith('<svg') and 'stroke=' in markup and '🧪' not in markup
    html = render_stage_timeline('predicting', 60)
    assert all(x in html for x in ('输入确认', '结构校验', '特征准备', '模型计算', '结果解释'))
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_ui.py'`; **Expected:** FAIL because UI module/components do not exist.
- [ ] **Step 3: Implement:** define navy workspace, blue-cyan active, green success, amber confirmation, red blocking and slate auxiliary tokens; register linear SVG icons for material, resin, hardener, carbon fiber, input, validation, features, calculation, result, AI, download, pause, cancel and retry; avoid emoji in headings/buttons/status labels; make columns responsive and keep task ID/model/service status visible.
- [ ] **Step 4: Implement portal shell:** retain configuration-driven targets and parameters; add explicit “手动输入 / 批量上传 / AI 辅助输入” modes; AI mode is additive and cannot remove legacy modes.
- [ ] **Step 5: Run:** `pytest -q 'tests/test_portal_ui.py' 'tests/test_prediction_portal.py'`; **Expected:** PASS; manually check `8555` at desktop and narrow viewport for no horizontal overflow.
- [ ] **Step 6: Commit:** `git add 'core/portal_ui.py' 'UserPrediction.py' 'tests/test_portal_ui.py'; git commit -m 'feat: add scientific portal UI shell'`。

### Task 8: 接入人工确认的 AI 输入助手

**Files:** Modify `UserPrediction.py` and, if needed, `core/portal_ai_schema.py`; create `tests/test_user_prediction_ai_flow.py`.

**Interfaces:** The page calls `PortalAIClient.parse_input(context)` and stores only validated `AIParseResponse`; a confirmation form emits `ConfirmedPredictionRequest` only after every returned suggestion is accepted, edited and confirmed, or explicitly rejected.

- [ ] **Step 1: Write the failing test**

```python
def test_ai_suggestions_cannot_submit_without_confirmation():
    state = build_ai_confirmation_state({'resin_smiles': 'CCO'}, confirmed_fields=set())
    assert can_submit_ai_prediction(state) is False
    state = confirm_ai_field(state, 'resin_smiles', 'CCN')
    assert state['confirmed_fields'] == {'resin_smiles'}
    assert fallback_input_mode('authentication_error') == 'manual'
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_user_prediction_ai_flow.py'`; **Expected:** FAIL because confirmation helpers and AI mode do not exist.
- [ ] **Step 3: Implement:** place the AI entry in the home hero; send only material type, target, configured field descriptions and safe user text; display recognized values, missing values, warnings, assumptions and confidence with edit controls; require “确认 / 修改后确认 / 拒绝”; keep uncertain/null fields missing; never invent EEW/AHEW/PHR, molecular features or process values; run Python validation before enabling “创建预测任务”.
- [ ] **Step 4: Run:** `pytest -q 'tests/test_user_prediction_ai_flow.py'`; **Expected:** PASS without API key or network.
- [ ] **Step 5: Commit:** `git add 'UserPrediction.py' 'core/portal_ai_schema.py' 'tests/test_user_prediction_ai_flow.py'; git commit -m 'feat: add confirmed AI input assistant'`。

---
### Task 9: 连接后台预测、轮询和结果解释

**Files:** Modify `UserPrediction.py` and `core/portal_tasks.py`; create `tests/test_portal_task_view.py`.

**Interfaces:** 页面使用 `PortalTaskManager.create_task`、`get_task_snapshot`、`cancel_task`、`retry_task`；结果页只把 Python 返回的 `PredictionResultSummary.prediction` 作为权威值，AI 解释是独立可选字段。

- [ ] **Step 1: Write the failing test**

```python
def test_task_view_shows_id_progress_and_python_result_when_ai_is_unavailable():
    html = render_task_snapshot({'task_id': 'task-1', 'status': 'featuring', 'progress': 42, 'stage_label': '特征准备'})
    assert 'task-1' in html and '42%' in html and '特征准备' in html
    result_html = render_result({'prediction': 123.4, 'unit': '°C', 'explanation': {'status': 'unavailable'}})
    assert '123.4' in result_html and 'AI 解释暂不可用' in result_html
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_task_view.py'`; **Expected:** FAIL because task/result views do not exist.
- [ ] **Step 3: Implement:** submit immediately creates a task and renders its ID/`queued`; poll with bounded reruns, never run long feature extraction in the page thread; provide pause/cancel/retry controls; refresh reloads the persisted snapshot; show stage, progress, model version, workflow ID and sanitized error; never render raw tracebacks or secrets.
- [ ] **Step 4: Implement explanation:** after Python prediction is `completed`, send only result summary, units, warnings, model version and workflow ID to AI; if explanation fails, keep Python result visible and mark unavailable; prompt forbids changing values, causal claims, warning suppression and unsafe instructions.
- [ ] **Step 5: Run:** `pytest -q 'tests/test_portal_tasks.py' 'tests/test_portal_task_view.py' 'tests/test_prediction_portal.py'`; **Expected:** PASS and no `missing ScriptRunContext` warnings from workers.
- [ ] **Step 6: Commit:** `git add 'UserPrediction.py' 'core/portal_tasks.py' 'tests/test_portal_task_view.py'; git commit -m 'feat: connect portal tasks and result explanation'`。

### Task 10: 迁移文档并完成验证

**Files:** Modify `README.md`; create `tests/test_portal_integration.py`; modify `prediction_portal/prediction_config.json` only for a non-secret schema version migration.

**Interfaces:** Documented commands are `streamlit run 'app.py' --server.port 8501` and `streamlit run 'UserPrediction.py' --server.port 8555`; integration tests use temporary roots and mocked models/AI transport, never public APIs or data deletion.

- [ ] **Step 1: Write the failing integration test**

```python
def test_unconfigured_ai_keeps_manual_prediction_path(tmp_path):
    config = load_config_from_fixture('prediction_config.json')
    assert config['materials']['epoxy_resin']['enabled'] is True
    assert manual_prediction_is_available(config) is True
    assert ai_mode_is_optional(tmp_path) is True

def test_ai_migration_does_not_modify_prediction_config(tmp_path):
    before = (tmp_path / 'prediction_config.json').read_bytes()
    migrate_ai_config(tmp_path)
    assert (tmp_path / 'prediction_config.json').read_bytes() == before
```

- [ ] **Step 2: Run:** `pytest -q 'tests/test_portal_integration.py'`; **Expected:** FAIL because the migration helper has not yet created `ai_config.json`; the fixture adapters are defined in the test module and must not touch real model files.
- [ ] **Step 3: Implement:** document both startup commands, provider fields, local endpoint exception, key storage, redaction, manual/batch fallback, task stages, cancellation/retry and task snapshot inspection; migration creates `ai_config.json` and a backup without rewriting model config.
- [ ] **Step 4: Run all tests:** `pytest -q 'tests/test_portal_ai_schema.py' 'tests/test_portal_ai_config.py' 'tests/test_portal_ai.py' 'tests/test_portal_prediction.py' 'tests/test_portal_tasks.py' 'tests/test_portal_ui.py' 'tests/test_user_prediction_ai_flow.py' 'tests/test_portal_task_view.py' 'tests/test_portal_integration.py' 'tests/test_prediction_portal.py'`; **Expected:** PASS. Historical temporary-directory permission warnings are not source changes.
- [ ] **Step 5: Run syntax check:** `python -m compileall -q 'core' 'UserPrediction.py' 'app.py'`; **Expected:** exit code `0` and no import error from optional AI dependencies.
- [ ] **Step 6: Manually verify:** only when ports are stopped, run `streamlit run 'app.py' --server.port 8501` and `streamlit run 'UserPrediction.py' --server.port 8555`; check AI configuration, existing portal toggle, standalone portal, manual input, batch upload, confirmation gate, progress, cancel/retry, refresh recovery and AI-off fallback without terminating existing processes.
- [ ] **Step 7: Commit:** `git add 'README.md' 'tests/test_portal_integration.py'; git commit -m 'docs: document scientific portal AI workflow'`。

## Self-review checklist

- [ ] Spec coverage: UI、DeepSeek/OpenAI-compatible services、密钥安全、人工确认、Python-only 计算、后台状态、结果解释、`8555` 兼容和无 AI 降级均已分配到 Tasks 1–10。
- [ ] Placeholder scan: 计划中的每一步都给出了具体文件、接口、代码行为、验证命令和预期结果，没有未完成占位语句。
- [ ] Interface consistency: `AIParseResponse`、`ConfirmedPredictionRequest`、`PredictionResultSummary`、`PortalAIClient`、`PortalTaskManager` 在定义后才被后续任务引用。
- [ ] Safety: 不删除数据、不重置分支、不终止用户现有平台进程、不把 API Key 写入公开配置。
- [ ] Validation: 每个代码任务都有失败测试、验证命令、通过预期和提交边界。

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-25-user-prediction-portal-ai-ui.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task and review between tasks.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, with checkpoints.

Choose `1` or `2` when ready.


