# CFRP 特征登记库与灵活预测契约实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立以稳定语义 `feature_id` 为核心、支持不同数据集列名和特征子集的版本化特征登记库，并让数据集映射、训练、artifact、发布门禁、用户预测门户和特征专用 AI 审核使用同一份可审计契约。

**Architecture:** `feature_registry.json` 保存 approved/draft/deprecated 特征语义和模型 profile；`core/feature_registry.py` 负责规范化、校验、哈希和快照。每次训练另有 dataset manifest，显式记录原始列到 `feature_id` 的绑定，派生 workflow 允许一个源字段生成多个特征。训练前锁定 registry snapshot + manifest hash，生成 `prediction_contract` v2，保存进 artifact；发布和门户只消费已批准快照，AI 只生成特征映射建议，最终由本地单人批准。

**Tech Stack:** Python 3.10+、pandas、NumPy、scikit-learn、joblib、Streamlit、现有 `core.portal_ai` OpenAI-compatible 客户端、pytest；不新增第三方依赖。

## Global Constraints

- 所有运行和验证命令使用 `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe`。
- 只修改本计划列出的文件；保留工作区已有改动、`.workbuddy`、pytest 临时目录、缓存和备份文件。
- 不删除或覆盖现有 Tg artifact；当前 Tg profile 保持 `blocked/needs_validation`，不执行真实 Tg 预测。
- 特征语义使用稳定 `feature_id`；原始列名、模型列名和中文 label 不得互相自动推断。
- 只有 `approved` registry snapshot 和 approved dataset manifest 可以训练或发布。
- 门户手工输入不得使用默认 0、均值、中位数、训练均值或 imputer 补齐；训练集内部 imputer 与用户输入缺失策略必须分离。
- AI 只处理特征候选映射、中文依据、单位/编码冲突和变更摘要；AI 不得写 registry、批准映射、生成特征值、发布模型或执行代码。
- 每个任务完成后只提交该任务涉及的文件，提交前运行任务专属测试。

## File Map

### Create

- `core/feature_registry.py`：registry schema、规范化、校验、hash、profile 查询和不可变 snapshot。
- `prediction_portal/feature_registry.json`：首版语义登记库和历史 Tg 审计 profile。
- `core/dataset_manifest.py`：数据集原始列到 `feature_id` 的显式映射、单位/枚举转换和 hash。
- `core/process_features.py`：训练、门户和离线脚本共享的固化/后固化派生特征及组件计数实现。
- `core/training_contract.py`：训练前锁定、contract v2 构建和训练结果特征审计。
- `core/feature_mapping_review.py`：特征专用 AI 审核建议、人工动作和 review record。
- `core/feature_registry_ui.py`：主平台精简特征管理页面，只展示当前数据集/profile 相关项目。
- `scripts/bootstrap_feature_registry.py`：从当前 Tg artifact 生成历史 profile 的可审计初始 JSON，不修改 artifact 和门户配置。
- `tests/test_feature_registry.py`
- `tests/test_dataset_manifest.py`
- `tests/test_process_features.py`
- `tests/test_training_contract.py`
- `tests/test_training_runs.py`
- `tests/test_contract_v2.py`
- `tests/test_feature_mapping_review.py`
- `tests/test_feature_registry_ui.py`
- `tests/test_portal_feature_sources.py`
- `tests/test_legacy_tg_gate.py`
- `tests/test_feature_registry_end_to_end.py`

### Modify

- `core/prediction_portal.py`：发出/校验 contract v2，验证 registry、manifest、来源分区和发布状态。
- `core/portal_prediction.py`：按 contract 来源分区执行 workflow、手工字段校验和可选固化剂槽位语义。
- `core/portal_tasks.py`：任务快照只保存脱敏请求摘要和输入 hash，不持久化 API key、完整敏感输入或未确认 AI 数值。
- `core/model_io.py`：artifact 保存 contract、registry snapshot、manifest 和 feature audit。
- `core/training_runs.py`：训练记录 metadata 保存 registry/manifest/contract 摘要。
- `core/model_trainer.py`：普通训练、CV、优化 pipeline 接受同一个锁定的 feature contract context。
- `core/optimizer.py`：超参优化所有 trial 和最终 pipeline 透传 contract context。
- `scripts/expand_manual_process_columns.py`：改为调用 `core.process_features`，保留命令行输入输出兼容性。
- `core/navigation.py`：增加“🧩 特征管理”页面入口。
- `app.py`：训练前锁定、训练导出传递 context、特征管理页面分发和最小审核 UI 接入。
- `UserPrediction.py`：移除模型列自动生成参数和默认值路径，改为 contract/registry 驱动字段。
- `core/portal_ai.py`：增加仅面向特征映射的受控 AI 请求入口。
- `core/portal_ai_schema.py`：增加特征映射建议的结构化响应校验，不复用预测输入/结果解释 schema。
- `tests/test_prediction_portal.py`、`tests/test_portal_prediction.py`、`tests/test_prediction_feature_contract.py`、`tests/test_model_trainer_feature_mask.py`：补充 v2、来源分区、无默认补齐和训练锁定断言。
- `tests/test_portal_tasks.py`：补充任务快照脱敏和输入摘要 hash 断言。

Canonical naming rule for this plan: `workflow_source_fields` is the v2 contract name. `source_columns` and `workflow_source_columns` remain read-only schema-1 compatibility fields and must be emitted consistently when a legacy consumer needs them; no new code may infer a source field from a model feature name. `request_feature_mapping_review(client, context)` is the only orchestration entry point; `PortalAIClient.review_feature_mapping(context)` is its injected transport method, not a second public workflow.

---

### Task 1: 实现语义特征登记库与历史 profile

**Files:**
- Create: `core/feature_registry.py`
- Create: `prediction_portal/feature_registry.json`
- Create: `scripts/bootstrap_feature_registry.py`
- Create: `tests/test_feature_registry.py`

**Interfaces:**
- `load_registry(path, require_approved=False) -> dict`
- `validate_registry(payload, require_approved=False) -> dict`，返回 `ok`、`errors`、`warnings`、`registry_hash`。
- `compute_registry_hash(payload) -> str`
- `get_model_profile(registry, profile_id) -> dict`
- `build_registry_snapshot(registry, profile_id) -> dict`

- [ ] **Step 1: 写失败测试**

```python
def test_registry_rejects_duplicate_feature_names():
    from core.feature_registry import validate_registry
    payload = {"schema_version": 1, "registry_version": "2026.08.27", "features": [
        {"feature_id": "a", "name": "same", "source_type": "manual_input", "data_type": "float", "default_policy": "explicit_only", "required_for_prediction": False, "nullable": True},
        {"feature_id": "b", "name": "same", "source_type": "manual_input", "data_type": "float", "default_policy": "explicit_only", "required_for_prediction": False, "nullable": True},
    ], "model_profiles": {}, "approval": {"status": "draft"}}
    report = validate_registry(payload)
    assert report["ok"] is False
    assert any("name" in error for error in report["errors"])


def test_registry_rejects_manual_numeric_default():
    from core.feature_registry import validate_registry
    payload = {"schema_version": 1, "registry_version": "2026.08.27", "features": [
        {"feature_id": "a", "name": "pressure", "source_type": "manual_input", "data_type": "float", "default": 0, "default_policy": "explicit_only", "required_for_prediction": True, "nullable": False},
    ], "model_profiles": {}, "approval": {"status": "draft"}}
    report = validate_registry(payload)
    assert report["ok"] is False
    assert any("default" in error for error in report["errors"])


def test_registry_hash_ignores_review_metadata_but_changes_semantics():
    from core.feature_registry import compute_registry_hash
    base = {"schema_version": 1, "registry_version": "2026.08.27", "features": [{"feature_id": "a", "name": "temperature", "source_type": "manual_input", "data_type": "float", "default_policy": "explicit_only", "required_for_prediction": False, "nullable": True}], "model_profiles": {}, "approval": {"status": "approved", "approved_at": "2026-08-27T00:00:00+08:00"}}
    changed_time = {**base, "approval": {"status": "approved", "approved_at": "2026-08-28T00:00:00+08:00", "approved_by": "another-user", "change_summary": "same semantics", "review_note": "same semantics"}}
    changed_unit = {**base, "features": [{**base["features"][0], "unit": "°C"}]}
    assert compute_registry_hash(base) == compute_registry_hash(changed_time)
    assert compute_registry_hash(base) != compute_registry_hash(changed_unit)


def test_approved_hash_is_recomputed_and_mismatch_is_rejected():
    from core.feature_registry import compute_registry_hash, validate_registry
    payload = {"schema_version": 1, "registry_version": "2026.08.27", "features": [], "model_profiles": {}, "approval": {"status": "approved"}}
    payload["approval"]["approved_hash"] = "not-the-computed-hash"
    report = validate_registry(payload, require_approved=True)
    assert report["ok"] is False
    assert report["registry_hash"] == compute_registry_hash(payload)
    assert any("approved_hash" in error for error in report["errors"])
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_registry.py -q`  
Expected: FAIL，因为 `core.feature_registry` 尚不存在。

- [ ] **Step 3: 实现校验、hash 和 snapshot**

实现 `SOURCE_TYPES`、`DEFAULT_POLICIES`、`FEATURE_STATUSES` 常量；规范化 JSON 后稳定计算 SHA-256，排除 `approval.approved_hash` 以及可变审核时间/人员/备注；校验 feature_id/name 唯一性、来源、默认策略、nullable、范围、公式、profile 引用和审批状态。`manual_input` 拒绝数值 default；workflow 要求 calculation_rule.input_fields 和 implementation；`legacy_observed` 只能用于审计；`blocked` 必须带阻断原因。`build_registry_snapshot` 只复制 profile 引用的 definitions，并保留 registry version/hash。

- [ ] **Step 4: 生成历史 JSON**

`bootstrap_feature_registry.py` 读取当前 Tg artifact 的 532 个 feature_cols 和 504 个 workflow.final_feature_names，生成 532 个有序 feature identity；504 个旧 workflow 列标记 `legacy_observed`，28 个历史缺口按设计文档登记，其中 `stoichiometric_ratio_r` 为 `unknown` + `blocked`。脚本不得写 artifact 或 prediction_config。

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe scripts/bootstrap_feature_registry.py --artifact prediction_portal/managed_models/epoxy_resin/tg/20260825_200526_model_XGBoost_6.joblib --output prediction_portal/feature_registry.json`  
Expected: 输出 `532 model features / 504 workflow features / 28 historical gaps`。

- [ ] **Step 5: 运行测试并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_registry.py -q`  
Expected: PASS。

```powershell
git add core/feature_registry.py prediction_portal/feature_registry.json scripts/bootstrap_feature_registry.py tests/test_feature_registry.py
git commit -m "feat: add versioned semantic feature registry"
```

### Task 2: 实现 dataset manifest 与显式列映射

**Files:**
- Create: `core/dataset_manifest.py`
- Create: `tests/test_dataset_manifest.py`
- Modify: `core/feature_registry.py`

**Interfaces:**
- `normalize_dataset_manifest(payload) -> dict`
- `validate_dataset_manifest(manifest, registry, frame_columns=None, require_approved=False) -> dict`
- `compute_dataset_manifest_hash(manifest) -> str`
- `resolve_dataset_feature_bindings(manifest, registry, profile_id) -> dict[str, dict]`

- [ ] **Step 1: 写失败测试**

```python
def test_one_source_column_can_feed_multiple_derived_features():
    from core.dataset_manifest import validate_dataset_manifest
    registry = {"features": [
        {"feature_id": "stage_count", "name": "cure_stage_count", "source_type": "derived_workflow", "status": "approved"},
        {"feature_id": "total_time", "name": "cure_total_time_h", "source_type": "derived_workflow", "status": "approved"},
    ], "model_profiles": {"p": {"feature_ids": ["stage_count", "total_time"], "status": "approved"}}, "approval": {"status": "approved"}}
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [{"raw_column": "schedule", "source_field": "cure_schedule", "parse_rule_version": "core.process_features:1"}], "feature_bindings": [{"feature_id": "stage_count", "raw_columns": ["schedule"], "source_role": "derived_workflow"}, {"feature_id": "total_time", "raw_columns": ["schedule"], "source_role": "derived_workflow"}], "status": "approved"}
    assert validate_dataset_manifest(manifest, registry, frame_columns=["schedule"], require_approved=True)["ok"] is True


def test_manifest_rejects_multiple_bindings_for_one_feature():
    from core.dataset_manifest import validate_dataset_manifest
    registry = {"features": [{"feature_id": "x", "name": "temperature", "source_type": "manual_input", "status": "approved"}], "model_profiles": {"p": {"feature_ids": ["x"], "status": "approved"}}, "approval": {"status": "approved"}}
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "x", "raw_columns": ["a"]}, {"feature_id": "x", "raw_columns": ["b"]}], "status": "approved"}
    report = validate_dataset_manifest(manifest, registry, frame_columns=["a", "b"], require_approved=True)
    assert report["ok"] is False
    assert any("feature_id" in error for error in report["errors"])


def test_manifest_allows_profile_subset_and_rejects_missing_required_feature():
    from core.dataset_manifest import validate_dataset_manifest
    registry = {"features": [{"feature_id": "x", "name": "temperature", "source_type": "manual_input", "status": "approved"}, {"feature_id": "y", "name": "pressure", "source_type": "manual_input", "status": "approved"}], "model_profiles": {"p": {"feature_ids": ["x"], "status": "approved"}}, "approval": {"status": "approved"}}
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "x", "raw_columns": ["temperature_raw"]}], "status": "approved"}
    assert validate_dataset_manifest(manifest, registry, frame_columns=["temperature_raw"], require_approved=True)["ok"] is True
    assert validate_dataset_manifest({**manifest, "feature_bindings": []}, registry, frame_columns=["temperature_raw"], require_approved=True)["ok"] is False
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_dataset_manifest.py -q`  
Expected: FAIL，因为 manifest 模块尚不存在。

- [ ] **Step 3: 实现 manifest 规范化、hash 和 binding 查询**

`source_bindings.raw_column` 必须唯一且只声明一个规范化 `source_field`；同一 source field 可以被多个 `feature_bindings` 使用。`feature_bindings.feature_id` 在一个 manifest 内必须唯一，每个 profile feature_id 恰好绑定一次；`raw_columns` 必须存在于输入 frame；单位换算、value_mapping、parse_rule_version 和人工审批状态必须显式记录。aliases 只生成 pending 候选，不参与 approved manifest。数据集可使用 profile 的子集，但模型 profile 需要而 manifest 缺失的 feature_id 必须阻断。manifest 的 `manifest_hash` 由规范化 payload 重算，任何 raw/canonical 映射、单位、编码或解析规则变化都必须改变 hash。

- [ ] **Step 4: 运行测试并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_dataset_manifest.py -q`  
Expected: PASS。

```powershell
git add core/dataset_manifest.py core/feature_registry.py tests/test_dataset_manifest.py
git commit -m "feat: add explicit dataset feature mappings"
```

### Task 3: 抽取共享工艺派生特征与组件计数

**Files:**
- Create: `core/process_features.py`
- Create: `tests/test_process_features.py`
- Modify: `scripts/expand_manual_process_columns.py`
- Modify: `core/molecular_feature_workflow.py`

**Interfaces:**
- `compute_process_features(frame, feature_definitions, manifest) -> DerivedFeatureResult`
- `split_component_structures(value, allow_empty=False) -> list[str]`
- `count_smiles_components(value, role) -> int`
- `materialize_component_count_features(frame, resin_column, curing_agent_column) -> pandas.DataFrame`

- [ ] **Step 1: 写失败测试**

```python
def test_process_derivation_is_deterministic_and_does_not_fill_missing():
    import pandas as pd
    from core.process_features import compute_process_features

    frame = pd.DataFrame({"cure_schedule": ["80C/2h;120C/1h"]})
    definitions = [
        {"name": "cure_stage_count", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_stage_count", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}},
        {"name": "cure_total_time_h", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_total_time_h", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}},
    ]
    result = compute_process_features(frame, definitions, {"source_bindings": [{"raw_column": "cure_schedule", "source_field": "cure_schedule"}]})
    assert result.errors == []
    assert result.features.loc[0, "cure_stage_count"] == 2
    assert result.features.loc[0, "cure_total_time_h"] == 3.0


def test_process_error_has_machine_readable_rule_and_source():
    import pandas as pd
    from core.process_features import compute_process_features

    result = compute_process_features(
        pd.DataFrame({"other": [1]}),
        [{"name": "cure_stage_count", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_stage_count", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}}],
        {"source_bindings": []},
    )
    assert result.errors and {"code", "feature", "source", "rule", "message"} <= set(result.errors[0])


def test_empty_curing_agent_is_structural_zero_but_invalid_nonempty_is_blocked():
    from core.process_features import count_smiles_components

    assert count_smiles_components(None, role="curing_agent") == 0
    assert count_smiles_components("", role="curing_agent") == 0
    assert count_smiles_components("CCO.CCN", role="curing_agent") == 2
    try:
        count_smiles_components("(", role="curing_agent")
    except ValueError as exc:
        assert "SMILES" in str(exc)
    else:
        raise AssertionError("invalid non-empty curing agent must fail")


def test_offline_script_delegates_business_derivation(monkeypatch):
    import scripts.expand_manual_process_columns as script

    assert hasattr(script, "compute_process_features")
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_process_features.py -q`  
Expected: FAIL，因为共享模块和 `DerivedFeatureResult` 尚不存在。

- [ ] **Step 3: 实现纯函数解析器**

在 `core/process_features.py` 中集中迁移脚本已有的数值列表、阶段解析和统计规则。固定 `CURE_STAGE_LIMIT = 4`、`POST_CURE_STAGE_LIMIT = 2`，并返回结构化错误而不是填充值。实现时保持下面的可测试接口和错误形状：

```python
import re
import pandas as pd
from dataclasses import dataclass, field

@dataclass(frozen=True)
class DerivedFeatureResult:
    features: pd.DataFrame
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

def compute_process_features(frame, feature_definitions, manifest):
    output = {}
    errors = []
    for definition in feature_definitions:
        rule = definition["calculation_rule"]
        by_source = {binding["source_field"]: binding["raw_column"] for binding in manifest["source_bindings"]}
        inputs = [by_source.get(source_field) for source_field in rule["input_fields"]]
        missing = [source_field for source_field, raw_column in zip(rule["input_fields"], inputs)
                   if not raw_column or raw_column not in frame.columns]
        if missing:
            errors.append({"code": "missing_source_column", "feature": definition["name"],
                           "source": "derived_workflow", "rule": rule.get("implementation"),
                           "message": "缺少声明的原始输入", "columns": missing})
            continue
        values = _dispatch_declared_rule(frame, definition, inputs)
        output[definition["name"]] = values
    return DerivedFeatureResult(pd.DataFrame(output, index=frame.index), errors, [])

def split_component_structures(value, allow_empty=False):
    text = "" if value is None else str(value).strip()
    if not text:
        if allow_empty:
            return []
        raise ValueError("SMILES 不能为空")
    components = [item.strip() for item in re.split(r"[.;。；]+", text) if item.strip()]
    canonical = [_canonicalize_component(item) for item in components]
    if not canonical or any(item is None for item in canonical):
        raise ValueError("SMILES 结构非法")
    return sorted(item for item in canonical if item is not None)

def count_smiles_components(value, role):
    allow_empty = str(role) == "curing_agent"
    return len(split_component_structures(value, allow_empty=allow_empty))
```

`_dispatch_declared_rule(frame, definition, raw_columns) -> pandas.Series` 只允许 registry 白名单中的 implementation/version，并对每一行返回同索引 Series；任何解析/阶段配对错误都写入 `code/feature/source/rule/message` 诊断。`_canonicalize_component` 调用现有 `core.smiles_utils` 的安静解析器，不能在 RDKit 不可用时把非空文本当作有效结构；`_is_valid_component` 只作为该函数的内部谓词，不接受模糊字符串匹配。

适配 `scripts/expand_manual_process_columns.py` 时保留既有 CSV 输出列名，但 18 个目标派生特征的业务计算必须来自共享模块。`core.molecular_feature_workflow.execute_molecular_feature_workflow` 增加显式 derived step 分支，只接受 registry 声明的 implementation/version，不通过字符串前缀猜测。workflow input contract 同时声明可选固化剂槽位，避免把未使用 numbered slot 一律当作错误。

- [ ] **Step 4: 运行共享和 workflow 测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_process_features.py tests/test_molecular_feature_workflow.py -q`  
Expected: PASS，旧 workflow round-trip 测试不回归。

- [ ] **Step 5: 提交**

```powershell
git add core/process_features.py core/molecular_feature_workflow.py scripts/expand_manual_process_columns.py tests/test_process_features.py
git commit -m "feat: share process feature derivation across training and portal"
```

### Task 4: 升级 prediction_contract 到 v2 并接入 artifact

**Files:**
- Modify: `core/prediction_portal.py`
- Modify: `core/model_io.py`
- Create: `tests/test_contract_v2.py`
- Modify: `tests/test_prediction_portal.py`
- Modify: `tests/test_prediction_feature_contract.py`

**Interfaces:**
- `build_prediction_contract(..., registry_snapshot, dataset_manifest, model_profile_id, canonical_feature_cols, effective_feature_cols, removed_feature_cols, removed_feature_reasons) -> dict` emits schema 2。
- `compute_contract_hash(contract) -> str` computes a stable hash over the semantic contract excluding its own hash field。
- `validate_publication_artifact(artifact, contract=None, registry_snapshot=None, dataset_manifest=None) -> dict` validates v2；schema 1 只作为 legacy needs-validation 报告。
- `create_model_artifact(..., contract_context=None) -> dict` stores immutable contract context under `extra`。

- [ ] **Step 1: 写失败测试**

```python
def test_contract_v2_keeps_workflow_and_manual_partitions_separate():
    from core.prediction_portal import build_prediction_contract

    class Model:
        feature_names_in_ = ["molecular_x", "derived_temperature", "manual_pressure"]
        n_features_in_ = 3

    artifact = {"model": Model(), "pipeline": None, "feature_cols": ["molecular_x", "derived_temperature", "manual_pressure"], "target_col": "tg_c", "extra": {}}
    snapshot = {"registry_version": "2026.08.27", "registry_hash": "r1", "features": [
        {"feature_id": "m", "name": "molecular_x", "source_type": "molecular_workflow", "status": "approved"},
        {"feature_id": "d", "name": "derived_temperature", "source_type": "derived_workflow", "status": "approved"},
        {"feature_id": "p", "name": "manual_pressure", "source_type": "manual_input", "default_policy": "explicit_only", "status": "approved"},
    ]}
    contract = build_prediction_contract(artifact=artifact, feature_cols=artifact["feature_cols"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest={"dataset_id": "d1", "manifest_hash": "m1", "status": "approved"}, model_profile_id="epoxy_resin.tg.v2", canonical_feature_cols=artifact["feature_cols"], effective_feature_cols=artifact["feature_cols"], removed_feature_cols=[], removed_feature_reasons={})
    assert contract["schema_version"] == 2
    assert contract["workflow_feature_cols"] == ["molecular_x", "derived_temperature"]
    assert contract["manual_input_feature_cols"] == ["manual_pressure"]
    assert contract["workflow_source_fields"] == []


def test_publication_rejects_registry_hash_or_effective_feature_mismatch():
    from core.prediction_portal import validate_publication_artifact

    class LegacyModel:
        feature_names_in_ = ["x"]
        n_features_in_ = 1
    artifact = {"model": LegacyModel(), "pipeline": None, "feature_cols": ["x"], "target_col": "tg_c", "extra": {}}
    contract = {"schema_version": 2, "feature_cols": ["x"], "canonical_feature_cols": ["x"], "effective_feature_cols": [], "removed_feature_cols": ["x"], "removed_feature_reasons": {"x": "all_nan"}, "feature_registry_version": "2026.08.27", "feature_registry_hash": "wrong", "dataset_manifest_hash": "m1", "workflow_feature_cols": ["x"], "molecular_workflow_feature_cols": ["x"], "derived_feature_cols": [], "manual_input_feature_cols": [], "feature_definitions": [{"feature_id": "x", "name": "x", "source_type": "molecular_workflow", "status": "approved"}], "target_col": "tg_c", "model_profile_id": "p", "workflow_present": True, "workflow_hash": "w1", "workflow_schema_version": 2, "workflow_source_fields": [], "source_columns": [], "workflow_source_columns": [], "pipeline_present": False, "imputer_present": False, "scaler_present": False, "numeric_ranges": {}, "missing_value_policy": "reject_user_missing", "training_missing_policy": "pipeline_imputer_only", "model_fingerprint": "f1"}
    report = validate_publication_artifact(artifact, contract, registry_snapshot={"registry_version": "2026.08.27", "registry_hash": "r1", "features": contract["feature_definitions"]}, dataset_manifest={"manifest_hash": "m1"})
    assert report["ok"] is False
    assert any("effective" in error or "registry" in error for error in report["errors"])
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_contract_v2.py -q`  
Expected: FAIL，因为 v2 字段和 context 参数尚不存在。

- [ ] **Step 3: 实现 v2 构建和校验**

将 `core.prediction_portal.CONTRACT_SCHEMA_VERSION` 升为 2，保留 schema 1 的 legacy diagnostics。v2 的 canonical source 字段统一命名为 `workflow_source_fields`；`source_columns` 与 `workflow_source_columns` 仅在读取/写出 schema-1 兼容信息时使用，不能作为 v2 必需字段。v2 必须验证：

```python
contract["feature_cols"] == contract["canonical_feature_cols"]
contract["feature_cols"] == contract["workflow_feature_cols"] + contract["manual_input_feature_cols"]
contract["workflow_feature_cols"] == contract["molecular_workflow_feature_cols"] + contract["derived_feature_cols"]
```

同时验证 registry version/hash、dataset manifest hash、完整 feature definitions、来源状态、unknown/blocked 特征和 effective 列集合；失败诊断使用统一 `diagnostics` 项并同步生成可读 `errors`。schema 1 artifact 返回 `ok=False, status="needs_validation"`，不自动推断来源。

`core/model_io.py` 的 `create_model_artifact` 和 `create_model_artifact_bytes` 接受 `contract_context`；至少写入 `prediction_contract`、`registry_snapshot`、`dataset_manifest` 和 `feature_audit`。调用方 `extra` 与 context 同键但内容不同时抛 `ValueError`，不静默覆盖。`prediction_contract.contract_hash` 必须由 `compute_contract_hash` 重算并在发布校验时复核。

- [ ] **Step 4: 运行契约测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_contract_v2.py tests/test_prediction_portal.py tests/test_prediction_feature_contract.py -q`  
Expected: PASS；旧 schema 测试只验证 needs-validation。

- [ ] **Step 5: 提交**

```powershell
git add core/prediction_portal.py core/model_io.py tests/test_contract_v2.py tests/test_prediction_portal.py tests/test_prediction_feature_contract.py
git commit -m "feat: enforce registry-backed prediction contract v2"
```

### Task 5: 建立训练前锁定并贯穿普通训练、CV 和优化

**Files:**
- Create: `core/training_contract.py`
- Create: `tests/test_training_contract.py`
- Modify: `core/model_trainer.py`
- Modify: `core/optimizer.py`
- Modify: `app.py`

**Interfaces:**
- `lock_training_contract(registry_path, dataset_manifest, material_type, target, target_col, feature_cols, frame, workflow) -> dict`
- `assert_training_context(context, frame_columns) -> None`
- `audit_training_result(context, train_result) -> dict`
- Add optional `feature_contract_context=None` to `EnhancedModelTrainer.train_model`, `cross_validate_model`, `build_regression_cv_pipeline` and `HyperparameterOptimizer.optimize`。

- [ ] **Step 1: 写失败测试**

```python
def test_training_lock_rejects_unregistered_column_before_model_creation(tmp_path):
    import json
    import pandas as pd
    from core.training_contract import lock_training_contract

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "registry_version": "2026.08.27", "features": [{"feature_id": "x", "name": "temperature", "source_type": "manual_input", "default_policy": "explicit_only", "status": "approved"}], "model_profiles": {"p": {"feature_ids": ["x"], "status": "approved"}}, "approval": {"status": "approved"}}, ensure_ascii=False), encoding="utf-8")
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "x", "raw_columns": ["temperature"]}], "status": "approved"}
    try:
        lock_training_contract(registry_path, manifest, "epoxy_resin", "tg", "tg_c", ["temperature", "not_registered"], pd.DataFrame({"temperature": [100], "not_registered": [1]}), None)
    except ValueError as exc:
        assert "not_registered" in str(exc)
    else:
        raise AssertionError("unregistered feature must block training")


def test_training_result_with_removed_canonical_feature_is_not_publishable():
    from core.training_contract import audit_training_result

    context = {"canonical_feature_cols": ["x", "y"], "feature_registry_hash": "r1", "dataset_manifest_hash": "m1"}
    audit = audit_training_result(context, {"feature_names": ["x"], "feature_mask": [True, False]})
    assert audit["publishable"] is False
    assert audit["removed_feature_cols"] == ["y"]
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_training_contract.py -q`  
Expected: FAIL，因为训练 contract context 尚不存在。

- [ ] **Step 3: 实现训练锁定模块**

`lock_training_contract` 读取 approved registry、校验 profile 和 approved manifest、检查 frame columns 与 canonical feature order，生成 registry snapshot、dataset manifest hash、来源分区和 contract draft。`assert_training_context` 拒绝未知、重复或缺失列。`audit_training_result` 比较 canonical 与 result.feature_names/feature_mask，返回 effective、removed、reason 和 publishable。

在 `app.py:page_model_training` 构造 `X/y` 后、任何 split/CV/模型实例化前调用 lock，并放入 `st.session_state["training_feature_contract_context"]`。普通训练、CV、优化和每个 trial 都传入同一个 context。`core/model_trainer.py` 三个公开入口最前面取出该参数调用 assert，再进入已有模型分支；不得把参数传给模型构造器。`core/optimizer.py` 的 reliable/exploratory 路径及最终 pipeline 复用同一 hash。

- [ ] **Step 4: 运行训练相关测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_training_contract.py tests/test_model_trainer_feature_mask.py tests/test_optimizer.py -q`  
Expected: PASS；现有 feature mask 行为保留，但删除 canonical 特征会产生不可发布 audit。

- [ ] **Step 5: 提交**

```powershell
git add core/training_contract.py core/model_trainer.py core/optimizer.py app.py tests/test_training_contract.py
git commit -m "feat: lock feature semantics before training"
```

### Task 6: 保存训练记录、artifact 和导出 context

**Files:**
- Modify: `core/training_runs.py`
- Modify: `core/model_io.py`
- Modify: `app.py`
- Modify: `tests/test_training_contract.py`
- Create: `tests/test_training_runs.py`

**Interfaces:**
- Add `contract_context=None` to `TrainingRunManager.save_run`。
- Write metadata keys `feature_registry_version`, `feature_registry_hash`, `dataset_id`, `dataset_manifest_hash`, `canonical_feature_count`, `effective_feature_count`, `removed_feature_cols`。
- `create_model_artifact_bytes(..., contract_context=context)` serializes the same context。

- [ ] **Step 1: 写 round-trip 失败测试**

```python
def test_artifact_round_trip_keeps_registry_and_manifest_hash():
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact

    context = {"prediction_contract": {"schema_version": 2, "feature_registry_hash": "r1", "dataset_manifest_hash": "m1"}, "registry_snapshot": {"registry_hash": "r1"}, "dataset_manifest": {"manifest_hash": "m1"}, "feature_audit": {"canonical_feature_cols": ["x"], "effective_feature_cols": ["x"], "removed_feature_cols": []}}
    artifact = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["x"], model=object(), contract_context=context)
    restored = loads_artifact(dumps_artifact(artifact))
    assert restored["extra"]["prediction_contract"]["feature_registry_hash"] == "r1"
    assert restored["extra"]["dataset_manifest"]["manifest_hash"] == "m1"
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_training_contract.py::test_artifact_round_trip_keeps_registry_and_manifest_hash -q`  
Expected: FAIL，因为 contract_context 尚未接入 artifact。

- [ ] **Step 3: 保存 context 并阻止冲突覆盖**

在 `core/model_io.py` 中把 context 的四个键写入 `extra`；调用方 `extra` 已有相同键且内容不同时抛 ValueError。`core/training_runs.py` 写 metadata 摘要和 `contract.json`，并把相同 context 传给 artifact。`app.py` 训练记录和下载导出都从 session context 传入；训练结束后调用 audit，`publishable=False` 时导出状态写 `needs_validation` 但保留审计信息。`tests/test_training_runs.py` 覆盖 metadata 中的 registry/manifest/contract hash、canonical/effective/removed 计数以及敏感值不落盘。

- [ ] **Step 4: 运行测试并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_training_contract.py tests/test_training_runs.py -q`  
Expected: PASS。

```powershell
git add core/model_io.py core/training_runs.py app.py tests/test_training_contract.py tests/test_training_runs.py
git commit -m "feat: persist registry context in artifacts and runs"
```

### Task 7: 改造可信预测层和用户门户字段

**Files:**
- Modify: `core/portal_prediction.py`
- Modify: `UserPrediction.py`
- Create: `tests/test_portal_feature_sources.py`
- Modify: `tests/test_portal_prediction.py`
- Modify: `tests/test_user_prediction_ai_flow.py`

**Interfaces:**
- `build_workflow_source_fields(contract, registry_snapshot) -> list[dict]`
- `build_manual_input_fields(contract, registry_snapshot) -> list[dict]`
- `validate_prediction_request` 增加 strict manual missing/range/enum validation。
- `run_confirmed_prediction` 通过 `core.process_features` 执行 derived steps，只合并 approved manual columns。

- [ ] **Step 1: 写失败测试**

```python
def test_portal_renders_only_manual_input_as_editable_fields():
    from UserPrediction import build_manual_input_fields, build_workflow_source_fields
    registry = {"features": [
        {"feature_id": "m", "name": "pressure", "label": "固化压力", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "valid_range": {"min": 0, "max": 20}, "status": "approved"},
        {"feature_id": "d", "name": "cure_stage_count", "label": "固化阶段数", "source_type": "derived_workflow", "status": "approved"},
    ]}
    contract = {"manual_input_feature_cols": ["pressure"], "workflow_source_fields": [{"column": "cure_schedule", "roles": ["derived"]}]}
    assert [field["name"] for field in build_manual_input_fields(contract, registry)] == ["pressure"]
    assert [field["name"] for field in build_workflow_source_fields(contract, registry)] == ["cure_schedule"]


def test_missing_manual_input_is_rejected_without_default_fill(monkeypatch):
    from core.portal_prediction import validate_prediction_request
    # monkeypatch load_published_portal_model to return a synthetic validated bundle.
    monkeypatch.setattr("core.portal_prediction.load_published_portal_model", lambda *args: {"entry": {"id": "v1"}, "artifact": {"model": object(), "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}}, "contract": {"schema_version": 2, "feature_cols": ["pressure"], "manual_input_feature_cols": ["pressure"], "workflow_feature_cols": [], "molecular_workflow_feature_cols": [], "derived_feature_cols": [], "feature_definitions": [{"feature_id": "p", "name": "pressure", "source_type": "manual_input", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}], "target_col": "tg_c", "numeric_ranges": {"pressure": {"min": 0, "max": 20}}, "missing_value_policy": "reject_user_missing"}})
    errors = validate_prediction_request({"material_type": "epoxy_resin", "target": "tg", "inputs": {}, "confirmed_by_user": True}, {"materials": {}})
    assert any("pressure" in error for error in errors)
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_feature_sources.py -q`  
Expected: FAIL，因为门户字段仍来自 target.parameters，来源构建函数不存在。

- [ ] **Step 3: 改造后端和页面**

后端以 v2 contract 的 `manual_input_feature_cols` 为唯一手工白名单；required + nullable=false 的空值、非法值、越界值和未映射枚举一律阻断。workflow source 使用显式 alias 和 input contract 支持 `curing_agent_smiles` 到 declared numbered slots；可选固化剂空值返回结构性组件数 0，未使用槽位不报错。derived 通过 `compute_process_features` 执行，最终按 contract.feature_cols 有序重排。

`UserPrediction.py` 新增两个纯函数，根据 contract + registry snapshot 生成 workflow source 和 manual 字段；所有 default 为 None，删除 `parameter_from_feature`/`sync_parameters_from_features` 的可信路径。旧 target.parameters 仅作为迁移提示，不再作为契约来源。

- [ ] **Step 4: 运行门户测试并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_feature_sources.py tests/test_portal_prediction.py tests/test_user_prediction_ai_flow.py -q`  
Expected: PASS；已有禁止默认/均值补齐测试不回归。

```powershell
git add core/portal_prediction.py UserPrediction.py tests/test_portal_feature_sources.py tests/test_portal_prediction.py tests/test_user_prediction_ai_flow.py
git commit -m "feat: render and validate portal fields from feature contract"
```

### Task 8: 实现精简 AI 特征审核与人工批准

**Files:**
- Create: `core/feature_mapping_review.py`
- Create: `core/feature_registry_ui.py`
- Create: `tests/test_feature_mapping_review.py`
- Create: `tests/test_feature_registry_ui.py`
- Modify: `core/portal_ai.py`
- Modify: `core/portal_ai_schema.py`
- Modify: `core/navigation.py`
- Modify: `app.py`

**Interfaces:**
- `build_feature_review_context(frame, registry, profile_id) -> dict`
- `request_feature_mapping_review(client, context) -> dict`
- `apply_feature_review_decision(manifest, suggestion, action, reviewer) -> dict`
- `save_feature_review_record(path, record) -> None`
- `render_feature_registry_page() -> None`

- [ ] **Step 1: 写失败测试**

```python
def test_rejected_ai_candidate_does_not_write_binding():
    from core.feature_mapping_review import apply_feature_review_decision
    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = {"feature_id": "cfrp.tg.pressure", "raw_columns": ["压强"], "source_role": "manual_input", "confidence": 0.96, "rationale_zh": "列名接近但单位未确认"}
    updated = apply_feature_review_decision(manifest, suggestion, "reject", "local-user")
    assert updated["status"] == "draft"
    assert updated["feature_bindings"] == []


def test_accept_action_writes_approved_binding():
    from core.feature_mapping_review import apply_feature_review_decision
    updated = apply_feature_review_decision({"status": "draft", "feature_bindings": []}, {"feature_id": "pressure", "raw_columns": ["pressure_raw"], "source_role": "manual_input", "confidence": 0.8, "rationale_zh": "人工确认"}, "accept", "local-user")
    assert updated["feature_bindings"][0]["feature_id"] == "pressure"
    assert updated["approval"]["approved_by"] == "local-user"
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py -q`  
Expected: FAIL，因为审核模块和页面尚不存在。

- [ ] **Step 3: 实现特征专用 AI 管理**

`build_feature_review_context` 只包含当前 frame columns、目标 profile definitions、dtype、少量样本摘要和单位/编码候选；不发送模型指标、预测结果、全部历史 registry 或无关页面状态。`request_feature_mapping_review(client, context)` 调用注入的 `client.review_feature_mapping(context)`，由 `core/portal_ai_schema.py` 校验 `suggestions/conflicts/rationale_zh/confidence`；unknown 保持 pending_review，不生成数值。`apply_feature_review_decision` 只有 accept/edit_accept 才写 approved binding，reject 只写 review record。

在 `core/portal_ai.py` 增加 `PortalAIClient.review_feature_mapping(context)`，复用现有 bounded request、JSON 解析和错误脱敏；`core/feature_mapping_review.py` 提供唯一编排函数 `request_feature_mapping_review(client, context)`。review record 写入 `prediction_portal/feature_reviews/`，不进入 registry hash。`core/feature_registry_ui.py` 默认只显示原始列名、候选 feature_id、来源、中文依据、冲突和三个动作；样本统计、完整规则、历史版本和 AI 原文放在 expander。导航增加“🧩 特征管理”，但不把预测结果或模型管理混入审核页。

- [ ] **Step 4: 运行审核/UI 测试并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py tests/test_portal_ai_schema.py tests/test_portal_ai.py -q`  
Expected: PASS；AI 不可用时页面显示错误，manifest 仍可手工编辑和批准。

```powershell
git add core/feature_mapping_review.py core/feature_registry_ui.py core/portal_ai.py core/portal_ai_schema.py core/navigation.py app.py tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py
git commit -m "feat: add focused AI feature review workflow"
```

### Task 9: 收紧发布门禁、旧 artifact 兼容和配置状态

**Files:**
- Modify: `core/prediction_portal.py`
- Modify: `core/portal_prediction.py`
- Modify: `core/portal_tasks.py`
- Modify: `UserPrediction.py`
- Modify: `app.py`
- Create: `tests/test_legacy_tg_gate.py`
- Modify: `tests/test_portal_tasks.py`

**Interfaces:**
- `validate_publication_artifact(artifact, contract=None, registry_snapshot=None, dataset_manifest=None) -> dict`
- `make_publication_entry(..., publication_status="needs_validation", enabled=False, gate_report=None) -> dict`
- `should_show_publication(contract_report) -> bool`
- `select_active_publication(models) -> dict | None`

- [ ] **Step 1: 写失败测试**

```python
def test_legacy_tg_artifact_is_never_published_by_missing_status():
    from core.prediction_portal import should_show_publication, select_active_publication, validate_publication_artifact

    class LegacyModel:
        feature_names_in_ = ["x"]
        n_features_in_ = 1
    artifact = {"model": LegacyModel(), "pipeline": None, "feature_cols": ["x"], "target_col": "tg_c", "extra": {}}
    report = validate_publication_artifact(artifact)
    assert report["status"] == "needs_validation"
    assert should_show_publication(report) is False
    assert select_active_publication([{"id": "legacy", "enabled": True}]) is None


def test_missing_publication_status_is_not_counted_in_portal_statistics():
    from UserPrediction import _material_statistics

    targets, models = _material_statistics({"targets": {"tg": {"models": [{"id": "legacy", "enabled": True}]}}})
    assert targets == 1
    assert models == 0


def test_publication_entry_defaults_to_disabled_until_gate_passes():
    from core.prediction_portal import make_publication_entry

    entry = make_publication_entry(
        material_key="epoxy_resin", target_key="tg", artifact_path="x.joblib", artifact_hash="h",
        label="Tg", unit="°C", description="legacy", contract={}, metrics={}, version="v1",
        published_at="2026-08-27T00:00:00Z",
    )
    assert entry["publication_status"] == "needs_validation"
    assert entry["enabled"] is False


def test_publication_entry_cannot_claim_published_without_a_valid_gate_report():
    from core.prediction_portal import make_publication_entry

    entry = make_publication_entry(
        material_key="epoxy_resin", target_key="tg", artifact_path="x.joblib", artifact_hash="h",
        label="Tg", unit="°C", description="demo", contract={}, metrics={}, version="v2",
        published_at="2026-08-27T00:00:00Z", publication_status="published", enabled=True,
        gate_report={"ok": False, "status": "invalid", "errors": [{"code": "blocked_feature"}]},
    )
    assert entry["publication_status"] == "needs_validation"
    assert entry["enabled"] is False


def test_failed_activation_keeps_previous_release_enabled():
    from core.prediction_portal import activate_publication

    config = {"materials": {"epoxy_resin": {"targets": {"tg": {"models": [
        {"id": "old", "version": "v1", "publication_status": "published", "enabled": True}
    ]}}}}}
    try:
        activate_publication(config, material_key="epoxy_resin", target_key="tg", entry={"id": "bad", "version": "v2", "publication_status": "needs_validation", "enabled": False})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid publication must be rejected")
    assert config["materials"]["epoxy_resin"]["targets"]["tg"]["models"][0]["enabled"] is True


def test_publication_diagnostics_have_fixed_fields():
    from core.prediction_portal import validate_publication_artifact

    report = validate_publication_artifact({"model": None, "pipeline": None, "feature_cols": [], "target_col": "tg_c", "extra": {}})
    assert report["diagnostics"]
    assert {"code", "feature", "source", "rule", "message"} <= set(report["diagnostics"][0])


def test_task_snapshot_does_not_persist_api_key_or_full_inputs(tmp_path, monkeypatch):
    from core.portal_tasks import PortalTaskManager

    monkeypatch.setattr(PortalTaskManager, "_run_task", lambda self, task_id: None)
    manager = PortalTaskManager(tmp_path)
    task_id = manager.create_task({"request": {"inputs": {"secret_measurement": "private"}, "confirmed_by_user": True}, "ai_config": {"api_key": "sk-secret"}})
    saved = (tmp_path / "prediction_portal" / "tasks" / f"{task_id}.json").read_text(encoding="utf-8")
    assert "sk-secret" not in saved
    assert "private" not in saved
    assert "request_summary_hash" in saved
    manager.shutdown()
```

- [ ] **Step 2: 运行失败测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_legacy_tg_gate.py -q`
Expected: FAIL，因为旧 artifact 和缺省 publication status 目前仍可被当成发布对象。

- [ ] **Step 3: 实现发布门禁和状态传播**

`validate_publication_artifact` 对 schema 1 或缺少 contract 的 artifact 只返回 `ok=False,status="needs_validation"`，不从模型列名猜来源。registry/manifest 参数优先使用显式参数；若未传入则从 artifact `extra` 读取，二者缺失、版本/hash 不一致均阻断。v2 逐项验证 registry snapshot/hash、manifest hash、profile target/status、完整 feature definitions、unknown/blocked/deprecated 特征、来源分区、workflow hash/schema/输出列、canonical/effective 等式、artifact 文件 hash 和可重算的 `contract_hash`；失败时返回固定结构 `{"code", "feature", "source", "rule", "message"}` 的 `diagnostics`，同时保留兼容性的可读字符串 `errors`（现有调用方继续读取该字段）。`make_publication_entry` 默认 `publication_status="needs_validation"`、`enabled=False`；即使调用方请求 published/true，也必须提供 `gate_report["ok"] is True`，否则强制降级。`activate_publication` 先校验 entry 再原子替换，拒绝非 published entry，失败时保留旧 active 版本。门户加载、统计和 active release 选择均只接受 `publication_status == "published" and enabled is True`，不能把缺失字段当作已发布。`core/portal_tasks.py` 在内存中保留执行所需的原始 request，磁盘快照只写 `request_summary_hash`、允许字段名、contract/registry/manifest 摘要和脱敏诊断；重启后不能从脱敏快照重放任务，`retry_task` 必须要求用户重新提交原始输入，不写 API key、完整敏感输入或未确认 AI 数值。旧 Tg 两个 artifact 保持原文件不变，加载时显示中文阻断原因且不执行预测。

- [ ] **Step 4: 运行回归并提交**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_legacy_tg_gate.py tests/test_prediction_portal.py tests/test_portal_prediction.py tests/test_portal_tasks.py -q`
Expected: PASS；既有合法 contract 测试更新为显式 `publication_status="published"`，旧 artifact 仍是 `needs_validation`。

```powershell
git add core/prediction_portal.py core/portal_prediction.py core/portal_tasks.py UserPrediction.py app.py tests/test_legacy_tg_gate.py tests/test_portal_tasks.py
git commit -m "fix: block legacy artifacts from publication"
```

### Task 10: 端到端验收、文档和全量回归

**Files:**
- Create: `tests/test_feature_registry_end_to_end.py`
- Modify: `docs/superpowers/specs/2026-08-27-feature-registry-contract-design.md`
- Modify: `docs/superpowers/plans/2026-08-27-feature-registry-contract-implementation.md`

**Interfaces:**
- Compose the public interfaces from Tasks 1-9; no new runtime API is introduced.

- [x] **Step 1: 写失败验收测试**

```python
def test_approved_registry_manifest_contract_artifact_round_trip(tmp_path):
    import hashlib
    import pandas as pd
    from core.dataset_manifest import compute_dataset_manifest_hash, validate_dataset_manifest
    from core.feature_registry import build_registry_snapshot, compute_registry_hash, validate_registry
    from core.model_io import create_model_artifact, dumps_artifact, loads_artifact
    from core.prediction_portal import build_prediction_contract, validate_publication_artifact

    registry = {"schema_version": 1, "registry_version": "2026.08.27", "features": [
        {"feature_id": "pressure", "name": "pressure", "source_type": "manual_input", "data_type": "float", "unit": "MPa", "required_for_prediction": True, "nullable": False, "default_policy": "explicit_only", "status": "approved"}
    ], "model_profiles": {"p": {"material_type": "epoxy_resin", "target": "tg", "target_col": "tg_c", "feature_ids": ["pressure"], "status": "approved"}}, "approval": {"status": "approved", "approved_by": "local-user"}}
    registry["approval"]["approved_hash"] = compute_registry_hash(registry)
    assert validate_registry(registry, require_approved=True)["ok"] is True
    snapshot = build_registry_snapshot(registry, "p")
    manifest = {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [], "feature_bindings": [{"feature_id": "pressure", "raw_columns": ["p_raw"], "canonical_name": "pressure", "source_role": "manual_input"}], "status": "approved"}
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    assert validate_dataset_manifest(manifest, registry, frame_columns=["p_raw"], require_approved=True)["ok"] is True
    class Model:
        feature_names_in_ = ["pressure"]
        n_features_in_ = 1
        def predict(self, frame):
            return [0.0] * len(frame)
    artifact = {"model": Model(), "pipeline": None, "feature_cols": ["pressure"], "target_col": "tg_c", "extra": {}}
    contract = build_prediction_contract(artifact=artifact, feature_cols=["pressure"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest=manifest, model_profile_id="p", canonical_feature_cols=["pressure"], effective_feature_cols=["pressure"], removed_feature_cols=[], removed_feature_reasons={})
    bundle = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["pressure"], model=Model(), contract_context={"prediction_contract": contract, "registry_snapshot": snapshot, "dataset_manifest": manifest, "feature_audit": {"canonical_feature_cols": ["pressure"], "effective_feature_cols": ["pressure"], "removed_feature_cols": []}})
    serialized = dumps_artifact(bundle)
    restored = loads_artifact(serialized)
    artifact_path = tmp_path / "embedded.joblib"
    artifact_path.write_bytes(serialized)
    report = validate_publication_artifact(restored, registry_snapshot=snapshot, dataset_manifest=manifest)
    assert report["ok"] is True

    from core.portal_prediction import load_published_portal_model, run_confirmed_prediction
    from core.prediction_portal import make_publication_entry
    entry = make_publication_entry(
        material_key="epoxy_resin", target_key="tg", artifact_path="embedded.joblib", artifact_hash=hashlib.sha256(serialized).hexdigest(),
        label="Tg", unit="°C", description="synthetic", contract=contract, metrics={}, version="v1",
        published_at="2026-08-27T00:00:00Z", publication_status="published", enabled=True,
        gate_report={"ok": True, "status": "valid", "errors": []},
    )
    config = {"project_root": str(tmp_path), "materials": {"epoxy_resin": {"targets": {"tg": {"models": [entry]}}}}}
    loaded = load_published_portal_model(config, "epoxy_resin", "tg")
    result = run_confirmed_prediction(
        {"material_type": "epoxy_resin", "target": "tg", "inputs": {"pressure": 5.0}, "confirmed_by_user": True},
        config=config,
    )
    assert loaded["contract"]["feature_cols"] == ["pressure"]
    assert result.summary["feature_count"] == 1
```

- [x] **Step 2: 运行失败验收测试**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_registry_end_to_end.py -q`
Expected: FAIL，直到 registry、manifest、contract 和 artifact 互相引用并可 round-trip 校验。

- [x] **Step 3: 编写端到端验收用例并更新文档状态**

编写并接入验收用例，覆盖：不同 raw column 映射到同一 semantic feature、一个 source field 生成多个 derived feature、manual 缺失/越界/未映射枚举阻断、结构性 curing-agent 0、AI reject 不入 manifest、registry/manifest hash 变更导致发布失败、effective 特征删除导致 `needs_validation`，以及通过批准 synthetic release 走完 `load_published_portal_model` → `run_confirmed_prediction` 的可信入口。对 legacy Tg 测试在调用 `predict` 前即失败，并对两个真实 artifact 做读取前后字节/hash 不变断言；同时验证普通训练、CV、优化接收到相同 registry/manifest/contract hash。实现后在设计文档增加“实施验收记录”小节，只记录实际测试命令和结果，不修改历史审计事实；计划文件勾选已完成步骤并保留每个提交的 hash。

已存在的提交记录：Task 1 `b1a0b67`；Task 2 `7bb608c`；Task 3 `3b28824`；Task 4 `68f0128`；Task 5 `7867dba`；Task 6 `e35cfb7`；task-lock fix `07acb24`。Task 7-9 的工作区改动未分配虚构的提交 hash。

- [ ] **Step 4: 运行全量验证**

Run: `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests -q`
Expected: PASS；若环境中存在可选依赖缺失，必须单独记录失败测试和原因，不得把未运行说成通过。另运行 `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m compileall core scripts UserPrediction.py app.py`，Expected: 无语法错误。

当前核实结果：聚焦验收测试 62/65 passed，端到端验收 1 passed；全量 `pytest tests -q` 为 555 passed、5 failed。因全量测试仍有失败，Step 4 保持未完成并阻断最终验收，不能勾选为已完成。

- [x] **Step 5: 最终提交和交付审查**

```powershell
git add docs/superpowers/specs/2026-08-27-feature-registry-contract-design.md docs/superpowers/plans/2026-08-27-feature-registry-contract-implementation.md tests/test_feature_registry_end_to_end.py
git commit -m "test: verify feature contract end to end"
```

提交前确认：`git diff --check` 无空白错误；`git status --short` 中只出现本次任务明确列出的已提交/未提交文件；不删除或覆盖既有 artifact、配置、缓存和用户修改。最后输出中文交付摘要，明确当前 Tg 仍不可发布、不可真实预测，下一阶段只能从 approved registry + manifest 开始训练。

## Plan Self-Review

### Spec coverage

- 登记库、稳定 `feature_id`、版本/hash、状态和历史 Tg profile：Task 1。
- 不同数据集原始列、显式 binding、子集和多派生映射：Task 2。
- 工艺解析、结构性 0、非法输入阻断和共享 workflow：Task 3。
- contract v2 来源分区、artifact 嵌入、发布不变量：Task 4、Task 9。
- 训练前锁定、普通训练/CV/优化一致性和 effective 特征审计：Task 5。
- training run、artifact、导出 context 持久化：Task 6。
- 门户来源字段、无默认补齐、可选固化剂槽位：Task 7。
- 精简 AI 特征建议、中文解释、人工批准和审计记录：Task 8。
- 旧 artifact 兼容、发布状态、Tg 阻断和全量回归：Task 9、Task 10。
- 结构化错误与安全审计（`diagnostics` 固定字段、任务快照脱敏、contract/registry/manifest/workflow/model fingerprint 摘要）：Task 4、Task 6、Task 9、Task 10。

验收矩阵：

- §7 发布配置：`test_publication_entry_defaults_to_disabled_until_gate_passes`、`test_publication_entry_cannot_claim_published_without_a_valid_gate_report`、active 选择只接受 published+enabled。
- §8 门户：`test_portal_renders_only_manual_input_as_editable_fields`、manual 缺失/范围/枚举测试、synthetic `load_published_portal_model` → `run_confirmed_prediction` 测试。
- §9 诊断与安全：`diagnostics` 五字段断言、`test_task_snapshot_does_not_persist_api_key_or_full_inputs`、artifact/run metadata hash round-trip 测试。
- §10 遗留 Tg：两个真实 artifact 的 532/504/28 审计事实、blocked `stoichiometric_ratio_r`、缺 contract/status、字节 hash 不变和 predict 前阻断测试。
- §11 workflow/AI：离线线上派生一致、阶段/SMILES 错误阻断、结构性 0、AI pending/reject 不入 manifest、人工批准后才可训练测试。

### Placeholder scan

已搜索计划中的 `TBD`、`TODO`、`Similar to`、`write tests for the above`、`适当错误处理` 和空泛的“实现后补充”表述；运行时代码占位符均已改为明确的接口、错误形状或测试断言。Task 3 的 `_dispatch_declared_rule`、`_is_valid_component` 仅作为同一任务内部私有函数名，必须在该任务中定义；Task 9 的 `gate_report`、`diagnostics`、`publication_status` 和 artifact extra 来源优先级已在接口与测试中固定；Task 10 明确 synthetic portal 入口、旧 artifact 字节不变和全量验证命令。

### Type and interface consistency

Task 1 的 `build_registry_snapshot` 输出 `registry_version/registry_hash/features`，并以 `approval.approved_hash` 复核；Task 2 接收 registry 并产生 `manifest_hash`；Task 4/6 使用相同四键 `prediction_contract/registry_snapshot/dataset_manifest/feature_audit`，contract 同时含可重算 `contract_hash`，且 v2 canonical 字段为 `workflow_source_fields`，schema-1 别名仅兼容读取；Task 5 的 `contract_context` 直接传给 Task 6 和 Task 7，raw columns 通过 manifest 绑定而非直接当 canonical 名；Task 8 的 approved binding 通过 Task 2 校验后才进入 Task 5。Task 9 的可选 registry/manifest 参数与 Task 4 接口一致，显式参数优先、缺失则读 artifact extra，`make_publication_entry` 的 `gate_report` 是 published 状态的必要条件，`diagnostics` 与兼容 `errors` 并存；Task 10 只组合既有接口，不再引入第二套契约。

### Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-27-feature-registry-contract-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
