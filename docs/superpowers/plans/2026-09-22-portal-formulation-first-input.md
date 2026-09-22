# 实施计划：配方优先输入、内置条件默认值与模型上传门禁修复

> **设计规范**：`docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md`
> **目标**：让「从训练平台下载的模型能真正启用」+「用户只输配方即可预测」+「AI 助手可缓存、可多轮对话」

---

## 前置约束（每个任务都必须遵守）

1. **Python 解释器**：所有命令使用 `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe`，不得用裸 `python`。
2. **不新增第三方依赖**；不改训练侧代码；不改 artifact 格式；不重训模型。
3. **不修改 `core/component_physics.py` 的 `cp_r_value` 计算逻辑**，只加 docstring 语义警示。
4. **禁止用 `cp_r_value` 填充 `formulation_r_value`**。
5. **不得为 `derived_workflow` / `molecular_workflow` 字段提供默认值**，只允许 `manual_input`。
6. 门户手工输入**不得**用默认 0 / 均值 / 中位数 / imputer 补齐。
7. 配方库固化制度必须用**显式数字温度**（`25 °C` 而非「室温」）。
8. 保留工作区已有未提交改动、`.pi`、pytest 临时目录、缓存、备份文件。
9. 每个任务**只提交该任务涉及的文件**，提交前跑该任务专属测试。
10. TDD：先写失败测试，再实现，再跑全绿。

---

## 任务总览

| # | 任务 | 主要文件 | 依赖 |
|---|---|---|---|
| T1 | 修复模型上传门禁（子集校验） | `core/prediction_portal.py` | — |
| T2 | 默认值生成脚本（离线统计） | `scripts/build_portal_input_defaults.py`、`prediction_portal/portal_input_defaults.json` | — |
| T3 | 默认值读取模块 | `core/portal_input_defaults.py` | T2 |
| T4 | 配方自动推导模块 | `core/portal_formulation_inputs.py` | — |
| T5 | 配方库扩展 + 固化制度校验 | `UserPrediction.py` | T4 |
| T6 | 输入端 5 分区重构 + 代填明细条 | `UserPrediction.py` | T3,T4,T5 |
| T7 | AI 缓存层 | `core/portal_ai_cache.py` | — |
| T8 | AI 多轮对话 | `UserPrediction.py`、`core/portal_ai.py` | T7 |

> T1 / T2+T3 / T4 / T7 之间无依赖，可并行开发。

---

## T1：修复模型上传门禁（子集校验）

### 问题
`core/prediction_portal.py:637-654` 要求 `workflow.final_feature_names == contract.feature_cols`。
现行 `storage_modulus_25c_gpa` artifact：workflow 产出 **253** 个分子特征，contract 有 **284** 个（253 分子 + 31 配方/工艺/测试）→ **永远不等 → 永远无法启用**。

### 关键前提（**已实测，勿再假设**）

**⚠️ 实测校正 —— 最初的「workflow ⊆ contract」假设是错的**：

| 集合 | 数量 | 说明 |
|---|---|---|
| `workflow.final_feature_names` | 253 | 分子特征 workflow 产出 |
| `artifact.feature_cols` = `feature_audit.effective_feature_cols` = `contract.feature_cols` | 284 | 模型实际消费的列 |
| `feature_audit.removed_feature_cols` | 5 | 训练侧因 `feature_mask` 删除的列 |
| `feature_audit.canonical_feature_cols` | 289 | 删除前全量 = 284 + 5 |

已实测恒等式：
```
contract.feature_cols ∪ removed_feature_cols == canonical_feature_cols   # 289 == 289 ✓
contract.feature_cols == effective_feature_cols                          # 顺序一致 ✓
workflow ∩ removed_feature_cols == 那 5 个“多出”特征                   # 已定位 ✓
```

→ **正确规则**：`workflow.final_feature_names ⊆ contract.feature_cols ∪ removed_feature_cols`
（单纯 `⊆ contract.feature_cols` 仍会因那 5 个删除特征而失败）

**其他实测前提**：
- 现行 artifact 的 `extra.prediction_contract` 为 **None**，`extra` 中**无** `registry_snapshot` / `dataset_manifest` → `build_prediction_contract` 构建出的是 **`schema_version=1`（legacy）**。
- 该 `final_feature_names` 校验位于 `if molecular_indicated:` 之后的**公共分支**（legacy 与 v2 都执行）→ **修复必须改公共分支**。
- 安全前提：`core/portal_prediction.py` 的 `_explicit_model_feature_names()` 返回 36 列，`_merge_explicit_model_features()` 缺一即抛 `模型需要显式工艺/实验特征，当前输入缺少：...` → 不存在静默丢特征。

**⚠️ 第二道门禁（`UserPrediction.py:165 _is_publishable_ui_model`）**：即使 `publish_imported_entry` 放行，UI 还要求 `contract.schema_version == 2` + `registry_snapshot` 非空 + `model_profile.status == approved`。实测真实模型：`schema_version=None`、`registry_snapshot={}` → **`_is_publishable_ui_model=False`，UI 仍不展示该模型**。

实测：`publish_imported_entry` **不写回** `entry.contract` / `entry.registry_snapshot`（均为空）→ 必须同时解决此点。

**T1.0 前置决策（✅ 已确认：方案 b）**
- **方案 b（✅ 采用）**：直接放宽 `_is_publishable_ui_model`，允许 `gate_report.ok=True` + `publication_status=published` + `enabled=True` 的 legacy 模型通过；保留 v2 + `registry_snapshot` + `approved` 的严格校验作为「当 v2 契约存在时」的额外要求。
- 方案 a（未采用）：新增独立的 `_is_predictable_ui_model` 放宽版。

> 方案 b 的约束：不得因此让**未经门禁**的模型进入预测页——`gate_report.ok is True` 与 `publication_status == published` 仍是硬条件。

### 步骤

**T1.1 先写失败测试** — 新增 `tests/test_portal_publication_partition.py`

```python
def test_workflow_features_subset_of_contract_union_removed_is_publishable():
    """workflow ⊆ contract ∪ removed 时应通过（真实 artifact 场景）。"""
    # contract.feature_cols = ["resin_xtb_gap", "curing_agent_xtb_gap"]
    # workflow.final_feature_names = ["resin_xtb_gap", "removed_gap"]
    # artifact.extra.feature_audit.removed_feature_cols = ["removed_gap"]
    # 断言 report["ok"] is True

def test_workflow_feature_unknown_to_contract_and_audit_is_rejected():
    """workflow 产出既不在 contract 也不在 removed 中 → 必须拒绝。"""
    # workflow = ["resin_xtb_gap", "totally_unknown"]
    # 断言 ok is False，错误含 "totally_unknown"

def test_empty_workflow_feature_names_is_rejected():
    """final_feature_names 为空 → 必须拒绝（保留原校验）。"""

def test_v2_contract_workflow_features_must_be_in_workflow_partition():
    """v2 契约声明 workflow_feature_cols 时，workflow 产出必须落在该分区内。"""

def test_legacy_schema_one_artifact_still_publishable():
    """回归：schema_version=1 legacy 契约不得被 v2 逻辑误判。"""
```

**T1.2 改写现有冲突测试** — `tests/test_prediction_portal.py:151`

`test_publication_rejects_workflow_feature_gap_before_activation` 目前**正是断言本行为必须被拒绝**，其 fixture 与真实 artifact 结构一致。本改动是**有意识的策略变更**（理由见 spec §4.1.1）。

改写为两个测试，**保留验证意图**：
```python
def test_publication_rejects_workflow_feature_outside_contract():
    """子集方向：workflow 多出契约未声明特征 → 拒绝。"""

def test_publication_accepts_workflow_subset_but_runtime_still_requires_missing_features():
    """workflow 是子集 → 发布门禁通过；但缺失特征仍由运行时拦截（不是静默丢失）。"""
    # 断言 validate_publication_artifact ok is True
    # 并断言 core.portal_prediction._merge_explicit_model_features 对缺失特征抛错
```

**T1.3 修改实现** — `core/prediction_portal.py` 公共分支（约 637-654 行）

把 `workflow_features != contract_features` 改为**基于 `contract ∪ removed` 的子集校验**：

```python
workflow_features = _normalized_columns(workflow_payload.get("final_feature_names"))
if not workflow_features:
    errors.append("artifact workflow 缺少 final_feature_names。")
elif contract_features:
    # workflow 可以产出被训练侧 feature_mask 删除的特征（不影响预测），
    # 但不得产出契约完全未知的特征。
    removed = _normalized_columns(
        (extra.get("feature_audit") or {}).get("removed_feature_cols")
        if isinstance(extra.get("feature_audit"), Mapping) else None
    ) or _normalized_columns(resolved_contract.get("removed_feature_cols"))
    known = set(contract_features) | set(removed)
    undeclared = [c for c in workflow_features if c not in known]
    if undeclared:
        errors.append(
            "artifact workflow 的 final_feature_names 含 prediction_contract.feature_cols "
            "未声明的特征（多出 " + ", ".join(undeclared[:8]) + "）。"
        )
    # v2 契约若声明了 workflow 分区，额外校验归属
    workflow_partition = _normalized_columns(resolved_contract.get("workflow_feature_cols"))
    if workflow_partition:
        outside = [c for c in workflow_features if c not in workflow_partition]
        if outside:
            errors.append(
                "artifact workflow 的 final_feature_names 必须落在 prediction_contract."
                "workflow_feature_cols 内（越界 " + ", ".join(outside[:8]) + "）。"
            )
```

**不要**新增「差集必须全为 manual_input」的静态校验——运行时已强制。

**T1.3b 解决第二道门禁（方案 b）** — `UserPrediction.py:165 _is_publishable_ui_model`

放宽规则：
- 硬条件（必须全满足）：`enabled is True`、`publication_status == "published"`、`gate_report.ok is True`、`gate_report.status == "valid"`；
- 当 `contract.schema_version == 2` 时：**仍要求** `registry_snapshot` 非空、`model_profile.status == approved`、所有 feature `status == approved`（保持原严格语义）；
- 当 `contract.schema_version != 2`（legacy/schema-1，如训练平台导入的模型）：**跳过** snapshot/profile 校验，但**仍执行** `validate_publication_artifact` 二次校验；
- 若 artifact 存在且 `validate_publication_artifact` 返回 `ok is not True` 或 `status != valid` → 返回 False。

```python
def _is_publishable_ui_model(model: Dict[str, Any]) -> bool:
    if not isinstance(model, dict) or model.get("enabled") is not True:
        return False
    if str(model.get("publication_status") or "").strip().lower() != "published":
        return False
    gate = model.get("gate_report")
    if not isinstance(gate, dict) or gate.get("ok") is not True \
       or str(gate.get("status") or "").strip().lower() != "valid":
        return False
    contract, snapshot = _model_contract(model)
    # v2 契约：保持原有的注册表审核严格语义
    if contract.get("schema_version") == 2:
        profile = snapshot.get("model_profile") if isinstance(snapshot, dict) else None
        if not snapshot or not isinstance(profile, dict) or profile.get("status") != "approved":
            return False
        if any(not isinstance(item, dict) or item.get("status") != "approved"
               for item in snapshot.get("features") or []):
            return False
    artifact = model.get("_artifact")
    if isinstance(artifact, dict):
        report = validate_publication_artifact(
            artifact,
            contract,
            registry_snapshot=snapshot,
            dataset_manifest=artifact.get("extra", {}).get("dataset_manifest")
            if isinstance(artifact.get("extra"), dict) else None,
        )
        if report.get("ok") is not True or str(report.get("status") or "").lower() != "valid":
            return False
    return True
```

**注意**：legacy 契约下 `validate_publication_artifact` 返回的 `status` 是 `"needs_validation"`（因 `legacy_contract` 分支），而 `ok` 为 `True`。因此**不能**用 `status != "valid"` 一刀切，需在 legacy 时接受 `status in {"valid", "needs_validation"}` 且 `ok is True`。这一点必须在测试中覆盖。

→ 实施时以**实测**为准：先跑 `publish_imported_entry` 后检查该模型的 `gate_report`，再据此定稿判定条件。

**T1.4 验证**
```bash
cd "C:/Users/wangj/Desktop/CFRP系统/CFRP系统"
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest \
  tests/test_portal_publication_partition.py tests/test_prediction_portal.py \
  tests/test_portal_prediction.py tests/test_legacy_tg_gate.py tests/test_contract_v2.py -q
```

**T1.5 端到端确认**（真 artifact 过门禁，复现 `publish_imported_entry` 的真实路径）

注意：`build_prediction_contract` 是 **keyword-only**；且 `publish_imported_entry`（`core/prediction_portal.py:857`）会过滤 `ignored_errors = ("schema-1/legacy", "缺少可复现 molecular workflow", "缺少可用 pipeline")`，端到端复现必须同样过滤，否则会误判。

```bash
cd "C:/Users/wangj/Desktop/CFRP系统/CFRP系统"
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -c "
import joblib, json
from core.prediction_portal import validate_publication_artifact, build_prediction_contract

p='prediction_portal/managed_models/epoxy_resin/storage_modulus_25c_gpa/20260831_195237_r2_0_8_MAE_0_5.joblib'
a=joblib.load(p)
extra=a.get('extra') or {}
wf=(extra.get('molecular_feature_workflow') or extra.get('molecular_feature_config')
    or extra.get('feature_process'))
c=build_prediction_contract(artifact=a, feature_cols=a.get('feature_cols') or [],
                            target_col=a.get('target_col') or 'storage_modulus_25c_gpa',
                            workflow=wf)
r=validate_publication_artifact(a,c)
ignored=('schema-1/legacy','缺少可复现 molecular workflow','缺少可用 pipeline')
blocking=[e for e in (r.get('errors') or []) if not any(i in str(e) for i in ignored)]
print(json.dumps({'contract_schema_version':c.get('schema_version'),
                  'raw_ok':r['ok'],'blocking_errors':blocking[:5]},ensure_ascii=False,indent=2))
print('→ 可发布' if not blocking else '→ 仍被阻断')
"
```
预期：`contract_schema_version == 1`（legacy），`blocking_errors == []`，输出「→ 可发布」。

**修复前对照（已实测确认）**：修复前同一命令输出 `n_blocking: 1`，唯一 blocking error 为：
```
artifact workflow 的 final_feature_names 必须与 prediction_contract.feature_cols 完全一致
（缺少 formulation_resin_hardener_equivalent_ratio, process_max_temperature_c, ...；
  多出 resin_1_structure_xtb_dipole, resin_2_structure_xtb_dipole, ...）
```

**第二道门禁验证**（`publish_imported_entry` 之后）
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -c "
import copy, sys; sys.path.insert(0,'.')
import UserPrediction as UP
from core.prediction_portal import publish_imported_entry
cfg=UP.load_config(); mk,tk='epoxy_resin','storage_modulus_25c_gpa'
c2=copy.deepcopy(cfg)
m2=UP.model_items(c2['materials'][mk]['targets'][tk])[0]
out=publish_imported_entry(c2, material_key=mk, target_key=tk, entry=m2)
e=UP.model_items(out['materials'][mk]['targets'][tk])[0]
print('status=',e.get('publication_status'),'enabled=',e.get('enabled'))
print('gate_report=',e.get('gate_report'))
print('_is_publishable_ui_model=',UP._is_publishable_ui_model(e))
"
```
预期：`status=published`、`enabled=True`、`_is_publishable_ui_model=True`。

### 验收
- [ ] 5 个新测试通过
- [ ] 改写的 2 个测试通过
- [ ] 回归测试全绿
- [ ] 真 artifact 通过 `publish_imported_entry`（不再抛 `模型未通过发布门禁验证`）
- [ ] 发布后 `_is_publishable_ui_model(model) is True`，模型出现在预测页

---

## T2：默认值生成脚本（离线统计）

### 目标
从真实全表统计生成**可审计**的默认值 JSON：每条默认值带 `share`（占比）+ `support`（样本数）+ `source_table`。

### 数据源表归属（**务必读对表**，已实测）

| 表 | 行数 | 用途 |
|---|---|---|
| `ml_performance_all.csv` | 47012 | 测试条件类（`test_method`/`test_atmosphere`/`analysis_method`） |
| `ml_performance_standards.csv` | 9851 | 测试标准（按 `performance_row_id` join） |
| `ml_qspr_selected.csv` | 10177 | 配方聚合类（`*_component_count`、`*_total_phr`、`curing_type_standard`） |
| `ml_wide_samples.csv` | 10749 | 工艺类（`process_*`） |
| `ml_process_stages.csv` | 54593 | 工艺阶段（`temperature_c`/`time_h`） |

数据集根目录：`C:/Users/wangj/Desktop/ml_dataset`

### 已实测的统计口径（直接采用，勿重新猜）

| 目标列 | 主口径 | 覆盖 | 样本数 |
|---|---|---|---|
| `storage_modulus_25c_gpa` | DMA / ASTM D5026 / 1 Hz | 86% | 2744 |
| `tg_c` | DMA或DSC / ASTM D4065 | — | 13681 |
| `td5_c`/`td10_c`/`tmax_c`/`char_yield_pct` | TGA / N2 / 10℃/min / ASTM E1868或E1131 | 98-99% | 5123 |
| `tensile_*` | ASTM D638 | 59-60% | 7205 |
| `flexural_*` | ASTM D790 | 69-71% | 2758 |
| `lap_shear`/`gic,kic`/`impact`/`cte` | D1002 / D5045 / D256 / E831 | — | — |

配方聚合类（`ml_qspr_selected.csv`）：
- `curing_type_standard` = `external_hardener`，share 0.866，n=10177
- `formulation_resin_phr_basis_type` = `resin_100_basis`，share 0.684，n=10177
- `initiator_present` = `False`，share 0.94，n=10177
- **`curing_mechanism` = `unknown` 占 84% → 不生成默认值**（无效口径）

### 步骤

**T2.1 先写测试** — `tests/test_portal_input_defaults_builder.py`
```python
def test_builder_omits_low_share_fields(tmp_path):
    """占比 < 0.30 的字段不得生成默认值。"""

def test_builder_records_share_and_support(tmp_path):
    """每条默认值必须带 share / support / source_table 以便审计。"""

def test_builder_never_emits_workflow_or_derived_fields(tmp_path):
    """derived_workflow / molecular_workflow 字段绝不出现在 defaults 中。"""

def test_builder_skips_curing_mechanism(tmp_path):
    """curing_mechanism 因 unknown 占 84% 必须被排除。"""
```

**T2.2 实现** — 新增 `scripts/build_portal_input_defaults.py`

要点：
- 纯标准库 + pandas（项目已有依赖），不新增依赖。
- 对每个目标列：
  - 分类字段（`test_method`/`test_atmosphere`/`analysis_method`/`specimen_geometry`/标准号）取**众数**；
  - 连续量（`frequency_hz`/`heating_rate_c_min`/`loading_rate_mm_min`）取**中位数**；
  - 标准列经 `ml_performance_standards.csv` 按 `performance_row_id` 关联后取众数。
- 占比 `< 0.30` → **不写入**（证据不足），并在脚本 stdout 打印被跳过的字段清单。
- 输出 `prediction_portal/portal_input_defaults.json`，结构见 spec §4.2。
- `generated_at` 使用**脚本参数传入**或环境变量，不依赖运行时刻（保证可复现；且脚本在 pytest 外的 CLI 场景允许用系统时间）。

**T2.3 生成并人工抽检**
```bash
cd "C:/Users/wangj/Desktop/CFRP系统/CFRP系统"
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe scripts/build_portal_input_defaults.py \
  --dataset "C:/Users/wangj/Desktop/ml_dataset" \
  --output prediction_portal/portal_input_defaults.json \
  --generated-at 2026-09-22
```
抽检 `storage_modulus_25c_gpa` 的 `test_method` 应为 `DMA`、`share` 约 0.856。

### 验收
- [ ] 4 个测试通过
- [ ] JSON 已生成并提交，`storage_modulus_25c_gpa.test_method.value == "DMA"`
- [ ] 无 `derived_workflow`/`molecular_workflow` 字段
- [ ] 无 `curing_mechanism` 默认值

---

## T3：默认值读取模块

### 步骤

**T3.1 先写测试** — `tests/test_portal_input_defaults.py`
```python
def test_only_manual_input_fields_get_defaults():
    """只对 manual_input 分区字段返回默认值。"""

def test_derived_and_workflow_fields_never_get_defaults():
    """derived_workflow / molecular_workflow 字段一律返回 None。"""

def test_missing_json_degrades_gracefully():
    """JSON 缺失/损坏时返回空默认值，不抛异常。"""

def test_defaults_lookup_by_target_col():
    """按目标列查得对应测试条件默认值。"""
```

**T3.2 实现** — 新增 `core/portal_input_defaults.py`

```python
def load_portal_input_defaults(root: str | None = None) -> Dict[str, Any]: ...
def defaults_for_target(target_col: str, root: str | None = None) -> Dict[str, Dict[str, Any]]: ...
def recipe_defaults(root: str | None = None) -> Dict[str, Dict[str, Any]]: ...
def default_for_feature(feature: str, *, partition: str, target_col: str = "",
                        root: str | None = None) -> Dict[str, Any] | None:
    """仅当 partition == 'manual_input' 时返回默认值，否则返回 None。"""
```

**硬校验**：`default_for_feature` 必须显式检查 `partition == "manual_input"`，其他分区直接返回 `None`。这是 spec 明令约束，需有对应测试。

**T3.3 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_input_defaults.py -q
```

### 验收
- [ ] 4 个测试通过
- [ ] `default_for_feature(..., partition="derived_workflow")` 返回 `None`

---

## T4：配方自动推导模块

### 步骤

**T4.1 先写测试** — `tests/test_portal_formulation_inputs.py`
```python
def test_dgeba_dds_derives_expected_values():
    """DGEBA/DDS 100:33 → EEW≈170.2、AHEW≈62.08、r≈0.905。"""

def test_missing_phr_yields_none_r_with_hint():
    """缺 phr → r 为 None 且给出提示（不得猜 0 或均值）。"""

def test_multicomponent_smiles_counted_correctly():
    """多组分 SMILES（'.' 分隔）计数正确。"""

def test_invalid_smiles_raises_not_silent():
    """非法 SMILES → 报错不静默。"""

def test_cp_r_value_never_used_as_r_source():
    """断言 cp_r_value 未被用作 formulation_r_value 来源。"""
    # 构造 DGEBA/DDS 100:33：正确 r≈0.905，cp_r_value≈0.365
    # 断言推导结果接近 0.905 而非 0.365
```

**T4.2 实现** — 新增 `core/portal_formulation_inputs.py`

**正确公式（已用全表验证，corr 0.944，中位误差 0.0002，n=3451）**：
```
r = (固化剂 phr / AHEW) / (树脂 phr / EEW)
```

**禁止**：`cp_r_value`（`component_physics.py:731`，当量重比 AHEW/EEW）**不得**用于填充 `formulation_r_value`——全表 `formulation_r_value` 是化学计量比 r，两者相关系数仅 **-0.088**。

**复用原则**：优先复用
- `core/epoxy_mechanism_features.py` 的 `EpoxyMechanismEngine.get_epoxide_count` / `get_active_hydrogen_count`（**注意类名是 `EpoxyMechanismEngine`，不是 `EpoxyMechanismFeatureExtractor`**）
- `core/component_physics.py` 的 `compute_component_physics` / `compute_formulation_summary`
- `core/auto_feature_resolver.py` 的 `_component_feature`

**不新写 RDKit 化学逻辑。**

推导清单见 spec §4.3 表格（13 类字段）。

**T4.3 加 docstring 语义警示** — `core/component_physics.py`

**只加注释，不改计算逻辑**。在 `cp_r_value` 处补充：
```python
# ⚠️ 语义警示：cp_r_value 是当量重比（AHEW/EEW），不是环氧/固化剂化学计量比 r。
# 全表 formulation_r_value 是 r = (固化剂phr/AHEW)/(树脂phr/EEW)，两者相关系数仅 -0.088，
# 不可互换。formulation_r_value 请用 core/portal_formulation_inputs.py 推导。
```

**T4.4 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_formulation_inputs.py -q
```

### 验收
- [ ] 5 个测试通过
- [ ] DGEBA/DDS 100:33 得 r≈0.905（**不是** 0.365）
- [ ] `cp_r_value` 计算逻辑未被修改（`git diff` 仅注释行）

---

## T5：配方库扩展 + 固化制度校验

### 已实测的解析器缺陷（本任务必须规避）

`core/process_features.py:29 _schedule_pairs` 的正则 `([-+]?\d*\.?\d+)\s*[^0-9;,:/]*C?\s*/\s*([-+]?\d*\.?\d+)` **要求温度是数字，非数字温度被静默丢弃且不报错**：

```
'室温/24 h + 80 °C/2 h'   → [(80.0, 2.0)]              ← 室温阶段被丢弃，总时长 26h 误算为 2h
'25 °C/24 h + 80 °C/2 h'  → [(25.0,24.0),(80.0,2.0)]    ← 正确
```

现行 `PORTAL_PRESET_RECIPES` 第 5 条（`DGEBF / IPDA`）的 `note` 中即写有「室温/24 h + 80 °C/2 h」。

### 步骤

**T5.1 先写测试** — `tests/test_portal_recipe_schedule.py`
```python
def test_validate_recipe_schedule_raises_on_stage_mismatch():
    """声明 3 阶段但只解析出 2 阶段 → 必须抛错，不静默。"""

def test_all_preset_recipes_parse_declared_stage_count():
    """遍历全部 PORTAL_PRESET_RECIPES，断言 cure_schedule 解析阶段数 == 声明阶段数。"""

def test_preset_recipes_use_numeric_temperatures_only():
    """配方库 cure_schedule 禁止出现 '室温'/'RT'/'常温' 等非数字表述。"""

def test_room_temperature_text_is_silently_dropped_by_parser():
    """回归：记录已知缺陷行为 —— '室温/24 h + 80 °C/2 h' 只解析出 1 阶段。"""

def test_numeric_room_temperature_parses_two_stages():
    """'25 °C/24 h + 80 °C/2 h' 必须解析出 2 阶段。"""
```

**T5.2 实现** — `UserPrediction.py`

1. 新增 `validate_recipe_schedule(schedule: str, *, declared_stages: int | None = None) -> List[Tuple[float, float]]`
   - 调用 `core.process_features._schedule_pairs`（或等价正则）；
   - 若 `declared_stages` 给定且解析数不等 → 抛 `ValueError`，错误信息含两个数字；
   - 若解析数为 0 → 抛错。

2. 扩展 `PORTAL_PRESET_RECIPES`：每条新增
   - `cure_schedule`: `str`，**显式数字温度**，如 `"80 °C/2 h + 150 °C/3 h"`；
   - `cure_stages`: `int`，声明阶段数（供校验）。

3. 修正第 5 条（DGEBF / IPDA）：
   - `note` 中「室温/24 h + 80 °C/2 h」→「25 °C/24 h + 80 °C/2 h」；
   - `cure_schedule` = `"25 °C/24 h + 80 °C/2 h"`，`cure_stages` = 2。

4. 模块加载时（或首次渲染配方库时）对所有条目跑 `validate_recipe_schedule` 断言，防止未来回归。

**T5.3 配方库与 ①② 区联动** — `render_recipe_library` / `_recipe_value_for_field`

一键载入配方时，除 `phr`/`temp`/`time` 外，还要把 `cure_schedule` 写入固化制度源字段。

**T5.4 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_recipe_schedule.py -q
```

### 验收
- [ ] 5 个测试通过
- [ ] 全部配方库条目阶段数校验通过
- [ ] 配方库中无「室温」等非数字温度
- [ ] 第 5 条已修正为 `25 °C/24 h + 80 °C/2 h`

---

## T6：输入端 5 分区重构 + 代填明细条

### 步骤

**T6.1 先写测试** — `tests/test_portal_input_partition.py`
```python
def test_partition_plan_has_five_groups():
    """分区计划必须包含配方/固化制度/高级测试条件/自动推导/系统计算 5 组。"""

def test_derived_group_is_display_only():
    """自动推导区 kind 必须为 display，不含输入框。"""

def test_recipe_group_is_required():
    """配方区必须标记为必填。"""

def test_autofilled_summary_lists_source_for_each_field():
    """代填明细必须逐字段给出取值与依据（默认值来源 / 推导来源）。"""
```

**T6.2 重构 `build_input_partition_plan`** — `UserPrediction.py:711`

从 4 区改为 5 区（spec §4.4）：

| 分区 | title | 内容 | 可编辑 |
|---|---|---|---|
| ① | 配方（必填） | 树脂/固化剂 SMILES、phr | ✅ |
| ② | 固化制度 | 温度、时间、阶段、后固化、气氛 | ✅（默认预填） |
| ③ | 高级测试条件 | 测试方法/标准/频率/升温速率 | ✅（折叠，默认预填） |
| ④ | 自动推导（只读） | EEW/AHEW/r/组分计数/官能团汇总 | ❌ |
| ⑤ | 系统计算（只读） | 分子特征 workflow 输出 | ❌ |

保留现有 `screening_fixed_input_cols` 的「固定工艺条件」插入逻辑（作为 ③ 的子块）。

**T6.3 渲染层** — `render_parameter_inputs` / `render_user_page`

- ②③ 区用 `st.expander` 承载，默认展开 ②、折叠 ③；
- 默认值经 `core.portal_input_defaults.default_for_feature(..., partition="manual_input")` 预填；
- ④ 区以「自动推导」标签展示 `core.portal_formulation_inputs` 的输出 + 推导说明，视觉上区别于手填；
- 用户可覆盖推导值，覆盖后标记 `user_confirmed`。

**T6.4 新增「已自动填充 N 项」明细条**

列出系统代填的字段、取值、依据（默认值来源 `share`/`support` 或推导公式），让用户看得见、可追溯，避免黑箱填充。

**T6.5 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_input_partition.py -q
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m streamlit run UserPrediction.py --server.headless true --server.port 8599
```
手动确认：只填 SMILES + phr 即可完成一次预测；②③ 区已预填且可改。

### 验收
- [ ] 4 个测试通过
- [ ] 手工输入字段数从 36 降到「SMILES + phr + 固化制度」级别
- [ ] 代填明细条逐字段可追溯
- [ ] 门户手工输入**未**用 0/均值/中位数补齐

---

## T7：AI 缓存层

### 步骤

**T7.1 先写测试** — `tests/test_portal_ai_cache.py`
```python
def test_same_input_hits_cache():
    """相同 service/model/prompt_kind/输入 → 命中缓存，不重复调用。"""

def test_different_service_or_model_does_not_collide():
    """不同 service/model 不串味。"""

def test_lru_evicts_beyond_limit():
    """超过 LRU 上限（500）时淘汰最旧条目。"""

def test_cache_never_stores_api_key():
    """缓存文件不得包含 API key。"""

def test_input_normalization_collapses_whitespace_but_keeps_case():
    """空白规范化，但 SMILES 大小写必须保留（大小写有意义）。"""
```

**T7.2 实现** — 新增 `core/portal_ai_cache.py`

- 存储：`prediction_portal/ai_cache/<sha256>.json`
- 键：`sha256(service_id | model | prompt_kind | 规范化输入文本)`
- 值：原始响应 + 时间戳 + 命中次数
- 输入规范化：去首尾空白、压缩连续空白；**大小写敏感保留**（SMILES 大小写有意义）
- LRU 上限 500，超出淘汰最久未命中
- 提供「强制重新解析」旁路（跳过读取但写入结果）
- **脱敏**：沿用 `core/portal_tasks.py` 的口径，不落 API key、不落完整敏感输入

**T7.3 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_ai_cache.py -q
```

### 验收
- [ ] 5 个测试通过
- [ ] 缓存目录中无 API key 明文
- [ ] LRU 上限生效

---

## T8：AI 多轮对话

### 步骤

**T8.1 先写测试** — `tests/test_portal_ai_conversation.py`
```python
def test_conversation_passes_confirmed_fields_as_context():
    """每轮必须把当前已确认字段作为上下文传给 AI。"""

def test_followup_modification_updates_only_targeted_field():
    """'温度改成 200 度' 只更新固化温度，不动其他字段。"""

def test_ai_cannot_generate_eeq_ahew_phr():
    """AI 仍不得生成 EEW/AHEW/PHR/分子特征/工艺参数。"""

def test_ai_result_still_requires_user_confirmation():
    """AI 提取结果仍须用户确认（can_submit_ai_prediction 门禁保留）。"""
```

**T8.2 重构 `render_ai_assistant_tab`** — `UserPrediction.py:244`

- 用 `st.chat_message` + `st.chat_input` 实现对话式修正：
  ```
  用户：E-51/DDS，100:33，180度固化4小时
  AI  ：已提取 树脂/固化剂/配比/温度/时间 5 项 → [填表] [修改]
  用户：温度改成 200 度
  AI  ：已更新固化温度 180 → 200 ℃
  ```
- 对话历史存 `st.session_state`，可持久化至 `prediction_portal/ai_sessions/`
- 每轮把**当前已确认字段**作为上下文传给 AI，使 AI 理解「修改」语义
- 修正后立即回填表单（复用现有 `_sync_ai_state_to_manual`）
- 接入 T7 缓存：相同输入命中缓存；提供「强制重新解析」按钮

**T8.3 约束保持（不得放松）**
- AI 仍然**只能提取和整理**用户提供的信息，不得生成 EEW/AHEW/PHR/分子特征/工艺参数（现有 `_INPUT_PROMPT` 约束保留）
- AI 提取结果仍须用户确认（`can_submit_ai_prediction` 门禁保留）

**T8.4 验证**
```bash
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_ai_conversation.py tests/test_portal_ai_cache.py -q
```

### 验收
- [ ] 4 个测试通过
- [ ] 多轮对话可修正单个字段而不破坏其他字段
- [ ] AI 约束未被放松

---

## 收尾

### 全量回归
```bash
cd "C:/Users/wangj/Desktop/CFRP系统/CFRP系统"
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/ -q
```

### 文档
- 更新 `CHANGELOG.md`，说明：
  1. 模型上传门禁由「完全一致」改为「子集校验」（**策略变更**，附理由与安全论证）；
  2. 新增配方优先输入 + 内置默认值；
  3. 修正 `cp_r_value` 语义误用风险（仅注释）；
  4. 配方库固化制度显式数字温度；
  5. AI 缓存 + 多轮对话。

### 提交粒度（每个任务独立提交）
| 提交 | 文件 |
|---|---|
| T1 | `core/prediction_portal.py`、`UserPrediction.py`、`tests/test_portal_publication_partition.py`、`tests/test_prediction_portal.py` |
| T2 | `scripts/build_portal_input_defaults.py`、`prediction_portal/portal_input_defaults.json`、`tests/test_portal_input_defaults_builder.py` |
| T3 | `core/portal_input_defaults.py`、`tests/test_portal_input_defaults.py` |
| T4 | `core/portal_formulation_inputs.py`、`core/component_physics.py`（仅注释）、`tests/test_portal_formulation_inputs.py` |
| T5 | `UserPrediction.py`、`tests/test_portal_recipe_schedule.py` |
| T6 | `UserPrediction.py`、`tests/test_portal_input_partition.py` |
| T7 | `core/portal_ai_cache.py`、`tests/test_portal_ai_cache.py` |
| T8 | `UserPrediction.py`、`core/portal_ai.py`、`tests/test_portal_ai_conversation.py` |

### 明确不做（YAGNI，见 spec §8）
- 不重训模型、不改 artifact 格式、不改训练侧代码
- 不为 `derived_workflow` / `molecular_workflow` 字段提供默认值
- 不用 `cp_r_value` 填充 `formulation_r_value`
- 不用 0/均值/中位数/imputer 补齐门户手工输入
