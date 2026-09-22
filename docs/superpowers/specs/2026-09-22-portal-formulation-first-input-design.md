# 材料预测平台：配方优先输入、内置条件默认值与模型上传门禁修复

**设计日期**：2026-09-22
**所属模块**：邹华维课题组材料预测平台（`UserPrediction.py`、`core/prediction_portal.py`）
**状态**：待批准（Pending Approval）

---

## 1. 背景与问题

### 1.1 问题 A：模型上传后永远无法启用（阻断级）

`core/prediction_portal.py:637` 的发布门禁要求：

```python
workflow_features = workflow_payload["final_feature_names"]
if workflow_features != contract_features:   # 要求【完全相等】
    errors.append("...必须与 prediction_contract.feature_cols 完全一致")
```

但训练导出时 `workflow.final_feature_names` 只登记**分子特征**。以现行 `epoxy_resin/storage_modulus_25c_gpa` 模型实测：

| 项 | 数量 |
|---|---|
| `workflow.final_feature_names` | 253（纯分子特征） |
| `contract.feature_cols` | 284 = 253 分子 + 31 配方/工艺/测试 |
| 门禁判定 | ❌ 失败：缺少 8 个配方工艺列、多出 5 个 `*_xtb_dipole` |

实测报错：

```
模型未通过发布门禁验证：artifact workflow 的 final_feature_names 必须与
prediction_contract.feature_cols 完全一致（缺少 formulation_resin_hardener_equivalent_ratio,
process_max_temperature_c, curing_agent_component_count, resin_total_phr,
curing_agent_equivalent_group_total, storage_modulus_25c_gpa_test_atmosphere,
curing_type_standard, resin_component_count；多出 resin_1_structure_xtb_dipole, ...）
```

**这不是配置错误，而是门禁逻辑与训练导出格式不匹配。** 任何「分子特征 + 配方/工艺特征」混合模型都无法发布，属于系统性缺陷。

### 1.2 问题 B：输入端要求手工填写无法获知的字段

现行「手动输入」完全由 contract 驱动，模型要哪些列就渲染哪些输入框。以储能模量模型为例，31 个非分子特征全部要求手工填写，其中：

| 类别 | 字段示例 | 用户可获知性 |
|---|---|---|
| 配方计量 | `formulation_resin_total_eew_g_eq`、`formulation_hardener_total_ahew_g_eq`、`formulation_r_value`、`resin_total_phr`、`curing_agent_total_phr`、`formulation_epoxy_binder_total_phr`、`formulation_resin_phr_basis_type` | **可由 SMILES + phr 自动推导**，却要求手填 |
| 组分计数 | `resin_component_count`、`curing_agent_component_count`、`catalyst_component_count` 等 | 同上，可自动 |
| 官能团汇总 | `resin_epoxy_group_total`、`curing_agent_active_hydrogen_total`、`resin_equivalent_group_total`、`curing_agent_equivalent_group_total` | 同上，可自动 |
| 分类 | `curing_type_standard`、`curing_mechanism`、`initiator_present` | 手填，且用户不知道合法取值 |
| 测试条件 | `storage_modulus_25c_gpa_test_method`、`_test_atmosphere`、`_frequency_hz` | **"标准对应关系很难找"** —— 用户无法凭记忆填写 |
| 工艺条件 | `process_max_temperature_c`、`process_total_time_h`、`process_temperature_time_integral_c_h`、`process_final_cure_temperature_c/time_h`、`process_atmosphere`、`process_has_post_cure` | 是真实实验变量，但可从内置配方库/统计默认大幅降低填写成本 |
| 活性稀释/增韧 | `reactive_diluent_component_count`、`reactive_toughener_component_count`、`reactive_toughener_total_phr`、`accelerator_component_count`、`accelerator_total_phr` | 可由配方条目自动计数 |

### 1.3 问题 C：AI 输入助手未发挥应有作用

现有 `render_ai_assistant_tab` 只做单轮「文本 → 解析 → 逐字段确认」，存在：

1. 每次点击都重新调用 AI，无缓存，重复解析同一段文本浪费额度与时间；
2. 解析结果有误时只能逐字段「确认/拒绝」，无法用自然语言修正（如「固化温度其实是 180 ℃」）；
3. 与「配方优先」的新输入结构未打通，AI 提取结果无法直接落到配方区。

### 1.4 设计依据：全表统计（离线核验）

以下统计取自 `C:\Users\wangj\Desktop\ml_dataset`（2026-09-11 快照），用于生成内置默认值。

**表归属（生成脚本须读对表）**：

| 数据表 | 行数 | 提供字段 |
|---|---|---|
| `ml_performance_all.csv` + `ml_performance_standards.csv` | 47012 | 测试条件类（`test_method`/`test_atmosphere`/`frequency_hz`/`heating_rate_c_min`/`specimen_geometry`/`standard_canonical`），按目标列取有值行 |
| `ml_qspr_selected.csv` | 10177 | 配方聚合类（`*_component_count`、`*_total_phr`、`*_equivalent_group_total`、`formulation_*`、`curing_type_standard`、`curing_mechanism`、`initiator_present`、`formulation_resin_phr_basis_type`） |
| `ml_wide_samples.csv` | 10749 | 工艺条件类（`process_*`）、逐组分明细 |

> 注意：配方聚合类特征（如 `resin_component_count`、`curing_agent_total_phr`、`curing_agent_equivalent_group_total`）**只存在于 `ml_qspr_selected.csv`**，`ml_wide_samples.csv` 中不存在。`process_*` 类则相反。

| 目标列 | 主测试方法 | 覆盖 | 主标准 | 气氛 | 其他 | 样本数 |
|---|---|---|---|---|---|---|
| `storage_modulus_25c_gpa` | DMA | 86% | ASTM D5026 | — | 频率 1 Hz（2036） | 2744 |
| `tg_c` | DMA/DSC | — | ASTM D4065 | — | 频率 1 Hz | 13681 |
| `td5_c` / `td10_c` / `tmax_c` / `char_yield_pct` | TGA | 98–99% | ASTM E1868 / E1131 | N2 | 升温 10 ℃/min | 5123 |
| `tensile_modulus_gpa` / `tensile_strength_mpa` | tensile | 71–72% | ASTM D638 | — | — | 7205 |
| `flexural_modulus_gpa` / `flexural_strength_mpa` | flexural | 53% | ASTM D790 | — | — | 2758 |
| `compressive_strength_mpa` | compression | — | ASTM D695 | — | — | 654 |
| `lap_shear_strength_mpa` / `shear_strength_mpa` | shear | 58–62% | ASTM D1002 | — | — | 896 |
| `impact_strength_kj_m2` | impact | 52% | ASTM D256 | — | — | 1584 |
| `fracture_toughness_kic_mpa_m05` / `gic_j_m2` | — | — | ASTM D5045 | — | — | 144 / 53 |
| `cte_glassy_per_k` | — | 83% | ASTM E831 | — | — | 503 |
| `dsc_cure_peak_c` | — | 86% | GBT 19466-5-2022 | N2 | 升温 10 ℃/min | 2694 |
| `curing_type_standard` | external_hardener | 87% | — | — | — | 10749 |
| `formulation_resin_phr_basis_type` | resin_100_basis | 68.4% | — | — | — | 10177 |
| `initiator_present` | False | 94% | — | — | — | 10177 |
| `curing_type_standard` | external_hardener | 86.6% | — | — | — | 10177 |

**配方聚合字段实测分布**（`ml_qspr_selected.csv`，用于校验推导合理性）：

| 字段 | 样本数 | 中位数 | 范围 | 与推导对照 |
|---|---|---|---|---|
| `resin_component_count` | 10177 | 1 | 0–4 | 单树脂配方为主 ✓ |
| `curing_agent_component_count` | 10177 | 1 | 0–5 | 单固化剂为主 ✓ |
| `formulation_r_value` | 6054 | 1.000 | 0.02–20 | 化学计量配比 ✓ |
| `formulation_resin_total_eew_g_eq` | 5044 | 187.55 | 81–2285.7 | DGEBA 170.2 落在区间 ✓ |
| `formulation_hardener_total_ahew_g_eq` | 3428 | 49.57 | 12.01–1004.5 | DDS 62.08 落在区间 ✓ |
| `resin_epoxy_group_total` | 3827 | 2 | 1–508 | DGEBA 官能度 2 ✓ |
| `curing_agent_active_hydrogen_total` | 4107 | 4 | 1–17 | DDS 活泼氢 4 ✓ |
| `curing_agent_equivalent_group_total` | 3821 | 4 | 0–16 | 同上 ✓ |
| `resin_equivalent_group_total` | 3382 | 2 | 0–12 | 同上 ✓ |
| `resin_total_phr` | 8188 | 100.0 | 0–500 | `resin_100_basis` 一致 ✓ |
| `curing_agent_total_phr` | 7012 | 32.35 | 0–3333.33 | DDS 33 phr 高度吻合 ✓ |
| `formulation_epoxy_binder_total_phr` | 8205 | 100.0 | 0–848.21 | — |
| `curing_mechanism` | 10177 | unknown(84%) | — | 覆盖不足，**不生成默认值** |

**发现的正确性缺陷（本次一并修复）**：`core/component_physics.py:731` 的 `cp_r_value` 实际是**当量重比 AHEW/EEW**，而全表 `formulation_r_value` 是**化学计量比 r**。在 3518 条重叠样本上两者相关系数仅 **-0.088**，语义不同。实测 DGEBA/DDS（100:33 phr）：`cp_r_value` = 0.365，而正确 r = **0.905**，偏差 2.5 倍。

正确公式经全表验证：

```
r = (固化剂 phr / AHEW) / (树脂 phr / EEW)
```

与实测 `formulation_r_value` 相关系数 **0.944**，中位绝对误差 **0.0002**，85% 样本误差 < 10%（n=3451）。

---

## 2. 设计目标

1. **修复模型上传门禁**，使现行 artifact 无需重训即可发布；
2. **输入端只需输入配方**（SMILES + phr），配方计量与官能团特征自动推导；
3. **测试条件与分类字段内置默认值**，依据来自全表统计且可审计、可展开查看；
4. **固化制度保持可编辑**（真实实验变量，不由默认值掩盖）；
5. **AI 输入助手增加缓存与多轮对话**，让用户能用自然语言修正输入。

---

## 3. 架构

```
[ 用户端 · 配方优先输入工作台 ]
  │
  ├─ ① 配方区（用户唯一必填）
  │    树脂 SMILES / 固化剂 SMILES / phr 配比
  │    └─ 常用配方库一键载入（PORTAL_PRESET_RECIPES）
  │
  ├─ ② 固化制度区（统计默认预填，可编辑）
  │    固化温度 / 时间 / 阶段数 / 后固化 / 气氛
  │
  ├─ ③ 高级条件区（默认预填，折叠）
  │    测试方法 / 标准 / 频率 / 升温速率
  │
  └─ ④ AI 对话助手（缓存 + 多轮修正）
       自然语言 → 结构化 → 回填 ① ② ③
  │
  ▼
[ 输入装配层 · core/portal_formulation_inputs.py ]  ← 新增
  SMILES + phr
    → RDKit 官能团计数（环氧基 / 活泼氢 / 分子量）
    → EEW / AHEW / 正确 r
    → 组分计数 / 总量汇总
    → 标记 derived / user_confirmed
  │
  ▼
[ 默认值层 · core/portal_input_defaults.py ]  ← 新增
  读 prediction_portal/portal_input_defaults.json（由脚本离线生成）
  │
  ▼
[ 发布门禁 · core/prediction_portal.py ]  ← 修复分区子集校验
  │
  ▼
[ 预测执行 · core/portal_prediction.py ]（不改动契约消费逻辑）
```

---

## 4. 详细设计

### 4.1 修复模型上传门禁（改动 1）

**文件**：`core/prediction_portal.py`（`validate_publication_artifact` 内 `final_feature_names` 校验段，约 637–654 行）

**现状**：`workflow_features != contract_features` → 报错。

**重要实测前提**：现行 artifact **不含** `registry_snapshot` / `dataset_manifest`，构建出的契约是 **`schema_version = 1`（legacy）**，**不是 v2**。而该段校验位于 `if molecular_indicated:` 之后的**公共分支**，legacy 与 v2 **都会执行**。因此修复必须作用于公共分支，不能只放松 v2 分支。

**改为**以下校验（legacy 与 v2 共同适用）：

1. `workflow.final_feature_names` 必须非空（保持原有检查）；
2. `workflow.final_feature_names` **⊆ `contract.feature_cols` ∪ `removed_feature_cols`**，否则报错「workflow 产出了契约未声明的特征」；
3. 对 v2 contract，额外要求 workflow 特征 **⊆** `contract.workflow_feature_cols`（分子/派生分区），否则报错「workflow 产出了未登记的分子特征」；
4. **不新增**「差集必须全为 `manual_input`」的静态校验。

> 第 2 条中 `removed_feature_cols` 的来源：优先 `artifact.extra.feature_audit.removed_feature_cols`，其次 `contract.removed_feature_cols`。原因见 §4.1.1：workflow 会产出已被训练侧 `feature_mask` 删除的特征（真实 artifact 中有 5 个），这些特征不影响预测，不应阻断发布。

**为何第 4 条安全（已实测验证）**：非 workflow 特征的强制性由**运行时**保证，而非门禁：

- `core/portal_prediction.py` 的 `_explicit_model_feature_names()` 返回全部非 workflow 特征名（本 artifact 实测 **36 个**）；
- `_merge_explicit_model_features()` 对这 36 列执行 `missing = [n for n in explicit_names if n not in inputs.columns]`，**缺一即抛 ValueError**：
  > `模型需要显式工艺/实验特征，当前输入缺少：... 请填写这些字段，或重新训练并发布包含完整特征工作流的模型。`

因此不存在「静默丢特征」路径：任何未被 workflow 产出的特征，用户不提供就无法预测。门禁只负责保证「workflow 不会产出契约外的额外列」，职责边界清晰。

**对 legacy 契约的影响**：`legacy_contract` 分支原有的「schema-1 只能审计，必须重新验证后才能发布」提示**保持不删**（`errors.append`），故 legacy 契约仍为 `needs_validation`；但 `publish_imported_entry` 已显式过滤该类非阻断性提示（`ignored_errors` 含 `"schema-1/legacy"`），因此可正常发布。**本任务不修改 `publish_imported_entry` 的过滤列表。**

**预期效果**：现行 `storage_modulus_25c_gpa` artifact 通过第一道门禁（`publish_imported_entry`），无需重训。但**仍需解决 §4.1.2 的第二道门禁**才能在 UI 中真正预测。

### 4.1.1 ⚠️ 实测校正（原假设错误，以此为准）

**原假设**：`workflow ⊆ contract.feature_cols` 即可。

**实测结果：该假设错误。** 真实 artifact 的 `workflow.final_feature_names` 有 **5 个特征不在 `contract.feature_cols` 中**（`resin_1_structure_xtb_dipole` 等），单纯子集校验**仍会失败**。

实测出的真实三层结构（`20260831_195237` artifact）：

| 集合 | 数量 | 说明 |
|---|---|---|
| `workflow.final_feature_names` | 253 | 分子特征 workflow 产出 |
| `artifact.feature_cols` = `feature_audit.effective_feature_cols` = `contract.feature_cols` | 284 | 模型实际消费的列 |
| `feature_audit.removed_feature_cols` | 5 | 训练侧因 `feature_mask` 删除的列 |
| `feature_audit.canonical_feature_cols` | 289 | 删除前的全量 = 284 + 5 |

已实测验证的恒等式：
```
contract.feature_cols ∪ removed_feature_cols == canonical_feature_cols   # 289 == 289 ✓
contract.feature_cols == effective_feature_cols                          # 顺序一致 ✓
workflow ∩ removed_feature_cols == 那 5 个“多出”特征                   # 已定位 ✓
```

**正确规则**：
```
workflow.final_feature_names ⊆ contract.feature_cols ∪ removed_feature_cols
```
即 workflow 可以产出已被训练侧删除的特征（它们会被 `feature_mask` 丢弃，不影响预测），但不得产出契约完全未知的特征。

### 4.1.2 ⚠️ 第二道门禁：`_is_publishable_ui_model`

即使 `publish_imported_entry` 放行，UI 还有**第二道更严格的门禁**（`UserPrediction.py:165`，被 `render_user_page:1967` 用于筛选可预测模型）：

```python
if contract.get("schema_version") != 2 or not snapshot \
   or not isinstance(profile, dict) or profile.get("status") != "approved":
    return False
```

它要求：`contract.schema_version == 2` **且** `registry_snapshot` 非空 **且** `model_profile.status == approved` **且** 所有 feature `status == approved`。

**实测真实模型状态**：`contract.schema_version=None`、`registry_snapshot={}`、`_is_publishable_ui_model=False`。

**已实测的发布后行为**：`publish_imported_entry` 写回 `entry.contract` 与 `entry.registry_snapshot` 均为空（它不生成 v2 契约，只基于 artifact 现场构建 schema-1 契约用于门禁校验，**不写回 entry**）→ 即使修复第一道门禁，**UI 仍无法启用该模型**。

**决策**：`_is_publishable_ui_model` 的 v2 要求**保持不动**（它保护的是注册表审核流程，不应为 legacy artifact 放松）。改为在 `publish_imported_entry` 成功发布后，**为 legacy 导入模型补齐 UI 可见性**：

- 方案：让 `render_user_page` 的模型筛选条件接受「`gate_report.ok=True` 且 `publication_status=published` 且 `enabled=True`」的模型，**但仅在 `contract.schema_version != 2` 时额外要求 `registry_snapshot` 缺失不得导致崩溃**；
- 更保守的替代：新增 `_is_predictable_ui_model`（放宽版，用于预测页筛选），保留 `_is_publishable_ui_model` 原语义用于「发布/审计」场景。

> **此点需用户在实施 T1 前最终确认**，因为它决定改动范围（是只改 `prediction_portal.py`，还是同时改 `UserPrediction.py` 的筛选函数）。

### 4.1.3 ⚠️ 与现有测试的冲突（必须显式决策）

现有测试 `tests/test_prediction_portal.py:151` `test_publication_rejects_workflow_feature_gap_before_activation` **正是断言本行为必须被拒绝**，且其 fixture 结构与真实 artifact 完全一致（分子特征 + 非 workflow 特征混合）：

```python
artifact.feature_cols  = ["resin_xtb_gap", "curing_agent_xtb_gap"]
artifact.workflow.final_feature_names = ["resin_xtb_gap"]   # 缺 curing_agent_xtb_gap
contract.feature_cols  = ["resin_xtb_gap", "curing_agent_xtb_gap"]
# 断言：report["ok"] is False，且错误信息包含 "final_feature_names" 与 "curing_agent_xtb_gap"
```

这说明当前门禁是**刻意设计**，而非疏漏。因此本改动不是单纯的 bug fix，而是**有意识的策略变更**，理由如下：

| 论点 | 说明 |
|---|---|
| 该门禁的表达力过强 | 它无法区分「`curing_agent_xtb_gap` 是分子特征但 workflow 漏产」与「它是配方/工艺特征，本就不应由 workflow 产出」——两种情况被同样拒绝 |
| 真实数据中后者是常态 | 现行 artifact 有 31 个此类特征（配方计量/工艺/测试条件），均属后者 |
| 前者已有运行时防护 | `_merge_explicit_model_features` 对任何非 workflow 特征缺一即抛错（§4.1 第 4 条） |
| 代价可控 | 真丢分子特征时，运行时会在用户提交后报「缺少显式特征」而非在发布时拦截，报错时机推后但**不会静默出错** |

**决策（已获用户确认）**：接受该策略变更。原测试需**改写而非删除**——保留其验证意图，改为断言「workflow 产出契约未声明的特征时被拒绝」（即子集方向），并新增一个测试断言「workflow 是子集时通过发布门禁、但缺失特征仍被运行时拦截」。

**已实测验证**：该 artifact 的 36 个残留字段全部可覆盖，无遗漏类别——

| 覆盖来源 | 字段数 | 说明 |
|---|---|---|
| 配方推导（§4.3） | 23 | EEW/AHEW/r/组分计数/官能团汇总/活性稀释增韧 |
| 工艺条件（用户填） | 7 | `process_*` 固化制度 |
| 测试默认值（§4.2） | 4 | `*_test_method` / `_test_atmosphere` / `_frequency_hz` / `curing_type_standard` |
| 分类默认值（§4.2） | 2 | `curing_mechanism`、`initiator_present` |
| **合计** | **36** | **其他（无来源）：0** |

### 4.2 内置默认值（改动 2）

**新增文件**：
- `scripts/build_portal_input_defaults.py` — 离线统计脚本
- `prediction_portal/portal_input_defaults.json` — 生成物（版本化提交）
- `core/portal_input_defaults.py` — 读取与查询

**JSON 结构**（每条默认值必须带可审计依据）：

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-22",
  "source_dataset": "ml_dataset@2026-09-11",
  "defaults": {
    "storage_modulus_25c_gpa": {
      "target_col": "storage_modulus_25c_gpa",
      "sample_size": 2744,
      "fields": {
        "storage_modulus_25c_gpa_test_method": {
          "value": "DMA", "share": 0.856, "support": 2348,
          "field": "test_method", "source_table": "ml_performance_all.csv"
        },
        "storage_modulus_25c_gpa_frequency_hz": {
          "value": 1.0, "share": 0.742, "support": 2036,
          "field": "frequency_hz", "source_table": "ml_performance_all.csv"
        }
      }
    }
  },
  "recipe_defaults": {
    "process_atmosphere": {"value": "air", "share": 0.61, "support": 6553},
    "curing_type_standard": {"value": "external_hardener", "share": 0.871, "support": 9356},
    "formulation_resin_phr_basis_type": {"value": "resin_100_basis", "share": 0.683, "support": 7337},
    "initiator_present": {"value": false, "share": 0.944, "support": 10152}
  }
}
```

**生成规则**：
- 对每个目标列，取该列非空行的 `test_method` / `test_atmosphere` / `analysis_method` / `specimen_geometry` 众数；
- 连续量（`frequency_hz`、`heating_rate_c_min`、`loading_rate_mm_min`）取中位数；
- 标准列经 `ml_performance_standards.csv` 按 `performance_row_id` 关联后取众数；
- 记录 `share`（占比）与 `support`（支持样本数），占比低于 0.30 的字段不生成默认值（证据不足）。

**契约约束**：默认值**只作用于 `manual_input` 分区字段**，不得为 `derived_workflow` / `molecular_workflow` 字段提供默认值（后者必须由 workflow 真实计算，避免伪造）。此约束在 `core/portal_input_defaults.py` 中硬校验。

### 4.3 配方自动推导（改动 3）

**新增文件**：`core/portal_formulation_inputs.py`

**输入**：树脂 SMILES（可多组分，`.` 分隔）、固化剂 SMILES、各组分 phr
**输出**：`{feature_name: {"value": v, "origin": "derived"|"user_confirmed", "detail": str}}`

**推导清单**：

| 特征 | 推导方式 | 依据 |
|---|---|---|
| `formulation_resin_total_eew_g_eq` | RDKit 官能度 + 分子量（质量加权调和平均） | 与全表 `cp_eew` 同口径，实测 DGEBA = 170.21 ✓ |
| `formulation_hardener_total_ahew_g_eq` | 同上 | 实测 DDS = 62.08 ✓ |
| `formulation_r_value` | `(固化剂phr/AHEW)/(树脂phr/EEW)` | 全表验证 corr 0.944，中位误差 0.0002 |
| `formulation_resin_hardener_equivalent_ratio` | 同 r | 全表实测两列 100% 相同（n=3518） |
| `resin_total_phr` / `curing_agent_total_phr` | phr 求和 | — |
| `formulation_epoxy_binder_total_phr` | 树脂 + 活性稀释剂 + 活性增韧剂 phr | — |
| `resin_component_count` / `curing_agent_component_count` 等 | 组分条目计数 | 全表众数 1，与配方一致 |
| `resin_epoxy_group_total` | 各树脂组分环氧基数求和 | SMARTS `C1OC1` |
| `curing_agent_active_hydrogen_total` | 各固化剂组分活泼氢数求和 | 复用 `epoxy_mechanism_features.get_active_hydrogen_count` |
| `resin_equivalent_group_total` / `curing_agent_equivalent_group_total` | 官能团总数 | — |
| `initiator_present` | 配方中是否存在引发剂组分 | — |

**复用原则**：优先复用 `core/epoxy_mechanism_features.py` 的 `get_epoxide_count` / `get_active_hydrogen_count` 与 `core/auto_feature_resolver.py` 的 `_component_feature`，**不新写 RDKit 化学逻辑**。

**明令禁止**：`cp_r_value` **不得**用于填充 `formulation_r_value`（语义不符，见 §1.4）。在 `core/component_physics.py` 的 `cp_r_value` docstring 中补充语义警示。

**UI 呈现**：推导值以「自动推导」标签显示并附推导说明，与用户手填值视觉区分；用户可覆盖，覆盖后标记 `user_confirmed`。

### 4.4 输入端重构（改动 4）

**文件**：`UserPrediction.py`（`render_user_page` 手动输入分区、`build_input_partition_plan`）

**分区从 4 个改为 5 个**：

| 分区 | 标题 | 内容 | 可编辑 |
|---|---|---|---|
| ① | 配方（必填） | 树脂/固化剂 SMILES、phr | ✅ |
| ② | 固化制度 | 温度、时间、阶段、后固化、气氛 | ✅（默认预填） |
| ③ | 高级测试条件 | 测试方法/标准/频率/升温速率 | ✅（折叠，默认预填） |
| ④ | 自动推导（只读） | EEW/AHEW/r/组分计数/官能团汇总 | ❌（展示推导依据） |
| ⑤ | 系统计算（只读） | 分子特征 workflow 输出 | ❌ |

**新增「已自动填充 N 项」明细条**：列出系统代填的字段、取值、依据（默认值来源 / 推导来源），让用户看得见、可追溯，避免「黑箱填充」。

**配方库增强**：`PORTAL_PRESET_RECIPES` 现只含 `phr`/`temp`/`time`，扩展为携带**完整多阶段固化制度**（`cure_schedule`），一键载入同时填满 ①② 区。

**已实测的解析器缺陷（必须在配方库中规避）**：`core/process_features.py` 的 `_schedule_pairs` 正则要求温度是数字，**非数字温度会被静默丢弃且不报错**：

```
'室温/24 h + 80 °C/2 h'   → [(80.0, 2.0)]              ← 室温阶段被丢弃，总时长 26h 误算为 2h
'25 °C/24 h + 80 °C/2 h'  → [(25.0,24.0),(80.0,2.0)]    ← 正确
```

现行 `PORTAL_PRESET_RECIPES` 第 5 条（`DGEBF / IPDA`）的 note 中即写有「室温/24 h + 80 °C/2 h」，扩展时必须改写为 `25 °C/24 h + 80 °C/2 h`。

**约束**：
1. 配方库中所有 `cure_schedule` 一律使用**显式数字温度**（摄氏度），禁止「室温」「RT」「常温」等非数字表述；
2. 提供 `validate_recipe_schedule(schedule) -> list[tuple[float,float]]`，加载时断言解析出的阶段数与声明阶段数一致，不一致即抛错；
3. 新增测试 `test_preset_recipes_schedule_parses_all_stages` 遍历全部配方库条目，断言阶段数匹配（防止静默丢阶段）。

**固化制度字段映射**：`cure_schedule` 形如 `"80 °C/2 h + 150 °C/3 h"` 写入 workflow 的 `cure_schedule` 源字段，由 `core/process_features.py` 派生 `process_max_temperature_c` / `process_total_time_h` / `process_temperature_time_integral_c_h` / `process_final_cure_temperature_c` / `process_final_cure_time_h` / `process_has_post_cure` / `cure_stage_count` 等。

**全表工艺统计依据**（`ml_process_stages.csv`，54593 行，38797 行含温度与时长）：

| 量 | 中位数 | p10 | p90 |
|---|---|---|---|
| `temperature_c` | 120.0 ℃ | 25.0 | 180.0 |
| `time_h` | 2.0 h | 0.333 | 24.0 |
| `pressure_mpa` | 0.14（1444 行有值） | — | — |

`stage_type` 分布：cure 22273 / mixing 13623 / post_cure 9254 / degassing 6512。

### 4.5 AI 输入助手优化（改动 5）

**文件**：`core/portal_ai_cache.py`（新增）、`UserPrediction.py`（`render_ai_assistant_tab` 重构）

**5.1 缓存层**（新增 `core/portal_ai_cache.py`）

- 存储：`prediction_portal/ai_cache/<sha256>.json`
- 键：`sha256(service_id | model | prompt_kind | 规范化输入文本)`
- 值：原始响应 + 时间戳 + 命中次数
- 策略：命中即复用（不重复扣额度）；提供「强制重新解析」按钮；LRU 上限 500 条
- 输入规范化：去首尾空白、压缩连续空白、大小写敏感保留（SMILES 大小写有意义）

**5.2 多轮对话**

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

**5.3 约束保持**

- AI 仍然**只能提取和整理**用户提供的信息，不得生成 EEW/AHEW/PHR/分子特征/工艺参数（现有 `_INPUT_PROMPT` 约束保留）
- AI 提取结果仍须用户确认（`can_submit_ai_prediction` 门禁保留）
- 缓存不得存储 API key、完整敏感输入（沿用 `core/portal_tasks.py` 的脱敏口径）

---

## 5. 数据流

```
用户输入配方 (SMILES + phr)
   │
   ├─→ [配方推导] ─→ EEW/AHEW/r/计数/官能团  (origin=derived)
   │
   ├─→ [默认值层] ─→ 测试条件/分类字段       (origin=default, 带依据)
   │
   └─→ [用户填写] ─→ 固化制度               (origin=user_confirmed)
   │
   ▼
合并为输入 DataFrame（每个字段带 origin 元数据）
   │
   ▼
validate_prediction_request → 契约校验（分区校验保持不变）
   │
   ▼
run_confirmed_prediction → 预测
```

---

## 6. 错误处理

| 场景 | 处理 |
|---|---|
| SMILES 无法解析 | 阻断该字段推导，提示「结构无法解析」，不静默填 0 |
| 缺少 phr | EEW/AHEW 可算，但 **r 不可算**；明确提示「需提供配比才能计算化学计量比」 |
| 默认值缺失（占比 < 0.30） | 该字段留空并要求用户填写，不猜测 |
| 默认值 JSON 缺失/损坏 | 降级为无默认值模式，页面提示，不阻断预测 |
| AI 服务不可用 | 保留手动输入路径（现有行为不变） |
| AI 缓存损坏 | 忽略缓存条目，重新请求 |
| 门禁仍失败 | 保留原有详细错误信息（列出具体缺哪些列） |

---

## 7. 测试策略

**新增测试文件**：
- `tests/test_portal_publication_partition.py` — 门禁子集校验
  - workflow 特征 ⊂ contract.feature_cols → 通过
  - workflow 产出契约未声明的特征 → 拒绝
  - workflow.final_feature_names 为空 → 拒绝
  - v2 contract 且 workflow 特征落在 manual 分区 → 拒绝
  - **回归**：用现行 `storage_modulus_25c_gpa` artifact（legacy schema-1）断言可发布，且 36 个非 workflow 特征仍由运行时强制
  - **回归**：`publish_imported_entry` 对 legacy 契约仍可发布（`ignored_errors` 过滤生效）
- `tests/test_portal_input_defaults.py` — 默认值读取
  - 只为 `manual_input` 字段提供默认值
  - 占比 < 0.30 不生成
  - JSON 缺失时降级不报错
- `tests/test_portal_formulation_inputs.py` — 配方推导
  - DGEBA/DDS 100:33 → EEW≈170.2、AHEW≈62.08、r≈0.905
  - 缺 phr → r 为 None 且给出提示
  - 多组分 SMILES（`.` 分隔）计数正确
  - 非法 SMILES → 报错不静默
  - **断言 `cp_r_value` 未被用作 r 来源**
- `tests/test_portal_recipe_schedule.py` — 配方库固化制度
  - `validate_recipe_schedule` 阶段数不匹配时抛错
  - 遍历全部 `PORTAL_PRESET_RECIPES`，断言每个 `cure_schedule` 解析出的阶段数 == 声明阶段数
  - **回归**：`'室温/24 h + 80 °C/2 h'` 必须解析为 1 阶段（记录已知缺陷行为），而 `'25 °C/24 h + 80 °C/2 h'` 为 2 阶段
- `tests/test_portal_ai_cache.py` — AI 缓存
  - 相同输入命中缓存
  - 不同 service/model 不串味
  - LRU 上限生效
  - 缓存不落 API key

**回归测试**：`tests/test_prediction_portal.py`、`tests/test_portal_prediction.py`、`tests/test_legacy_tg_gate.py`、`tests/test_contract_v2.py` 必须全绿。

> 注意：`tests/test_legacy_tg_gate.py` 与 `tests/test_contract_v2.py` 含针对 `final_feature_names` 完全相等的现有断言，本改动需同步更新这些断言（从「完全相等」改为「子集」），并保留其验证意图。

**验证命令**：
```
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_portal_publication_partition.py tests/test_portal_input_defaults.py tests/test_portal_formulation_inputs.py tests/test_portal_ai_cache.py -v
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_prediction_portal.py tests/test_portal_prediction.py tests/test_legacy_tg_gate.py tests/test_contract_v2.py -v
```

---

## 8. 明确不做（YAGNI）

- ❌ 不改训练侧代码、不改 artifact 格式
- ❌ 不重训任何模型
- ❌ 不修改 `component_physics` 的 `cp_r_value` 计算逻辑（仅加 docstring 语义警示，避免影响其他下游）
- ❌ 不为 `derived_workflow` / `molecular_workflow` 字段提供默认值
- ❌ 不放松 schema-1 legacy contract 的严格校验
- ❌ 不引入新的第三方依赖

---

## 9. 风险与取舍

| 风险 | 取舍 |
|---|---|
| 门禁放松可能放过「真丢特征」的模型 | 非 workflow 特征的强制性由运行时 `_merge_explicit_model_features` 保证（缺一即抛错），已实测验证 |
| 统计默认值可能不适用于特定实验 | 默认值可编辑 + 显示依据与占比 + 低占比不生成 |
| 自动推导的 r 与实际配方有偏差 | 显示推导依据；用户可覆盖；缺 phr 时拒绝推导 |
| AI 缓存导致陈旧结果 | 提供强制重解析按钮；键含 service/model |
| 固化制度仍要求用户填写 | 这是刻意取舍——固化制度是真实实验变量，用默认值掩盖会造成系统性预测偏差 |
| `_schedule_pairs` 对非数字温度静默丢阶段 | 配方库统一使用显式数字温度（`25 °C` 而非 `室温`），并在加载时断言阶段数（见 §4.4） |
