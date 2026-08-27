# CFRP 特征登记库与统一预测契约设计

> 日期：2026-08-27  
> 状态：已获确认，待编写实施计划  
> 范围：第一阶段只建立版本化特征登记库，并将来源定义贯穿训练、artifact、发布门禁和用户预测门户；不自动修复或发布当前 Tg 模型。

## 1. 目标

建立一套唯一、可审计、可复现的特征来源契约，使以下链路使用同一份定义：

```text
特征登记库
  → 训练前锁定
  → 数据清洗与特征工程
  → 训练输入矩阵
  → artifact / prediction_contract
  → 发布门禁
  → UserPrediction.py 字段
  → 预测矩阵
  → 模型 predict
```

系统必须禁止：

- 使用默认值伪造缺失特征；
- 使用 0、均值、中位数、训练均值或 imputer 静默补齐门户输入；
- 根据特征数量、列位置或页面配置猜测列含义；
- 只修改门户页面而不修改训练契约；
- workflow、registry、artifact 和模型特征不一致时强行发布；
- 让 AI 生成、猜测或覆盖模型输入特征。

## 2. 已确认的范围与决策

### 2.1 存储与审核

采用版本化 JSON + Python 校验层，不首期引入 SQLite。登记库文件为：

```text
prediction_portal/feature_registry.json
```

校验与快照逻辑位于：

```text
core/feature_registry.py
```

采用本地单人显式批准：

```text
draft → approved → deprecated
```

只有 approved snapshot 可以进入可发布训练。变更摘要、审核人、审核时间、批准 hash 和意见必须保留。未批准的 draft 不得被训练或发布流程悄悄采用。

### 2.2 特征身份

每个特征有三种身份：

- `feature_id`：稳定的内部身份，例如 `cfrp.tg.curing_agent_smiles_n_components`；改显示名称或模型列名时不变。
- `name`：模型实际使用的列名；属于版本化契约，不能原地改写。
- `label`：门户显示名称；可独立修改。

只改 label、help 或描述属于非破坏性变更；改 name、单位、来源类型、公式、输入字段、空值规则、有效范围或枚举编码都必须生成新 registry 版本并重新批准。旧定义保留为 deprecated，不删除。`aliases` 只允许在人工批准的迁移中使用，门户和训练流程不得自动猜别名。

### 2.3 28 个当前 Tg 缺口的初始来源

| 来源 | 特征 |
|---|---|
| `derived_workflow`（18 个） | `cure_stage_count`、`cure_total_time_h`、`cure_max_temperature_c`、`cure_final_temperature_c`、`cure_temp_time_integral_c_h`、`cure_time_weighted_avg_temperature_c`、`post_cure_stage_count`、`post_cure_total_time_h`、`post_cure_max_temperature_c`、`post_cure_final_temperature_c`、`post_cure_temp_time_integral_c_h`、`post_cure_time_weighted_avg_temperature_c`、`post_cure_temperature_c`、`has_post_cure`、`total_cure_stage_count`、`total_heat_treatment_time_h`、`overall_max_cure_temperature_c`、`overall_temp_time_integral_c_h` |
| `manual_input`（7 个） | `degree_of_cure_pct`、`gel_time_min`、`curing_pressure_mpa`、`eew_g_eq`、`ahew_g_eq`、`tg_method`、`tg_standard` |
| `unknown` + `blocked`（1 个） | `stoichiometric_ratio_r`；待确认 PHR、EEW、AHEW 定义、质量基准、方向、单位和训练数据实际公式 |
| `molecular_workflow`（2 个） | `resin_smiles_n_components`、`curing_agent_smiles_n_components` |

组件计数规则已经确定：树脂结构为空或非法时阻断；固化剂为空时允许并由系统明确计算 `curing_agent_smiles_n_components = 0`；固化剂非空但非法时阻断。空值导致的结构性 0 必须由 registry 的规则明确声明，不能与缺失补齐混淆。

`tg_method` 和 `tg_standard` 当前训练参考数据为整数编码。registry 必须保存枚举的显示值、编码值和未知值策略；编码未确认前，依赖这两个字段的模型不能发布。

## 3. 登记库数据模型

登记库顶层包含：

```json
{
  "schema_version": 1,
  "registry_version": "2026.08.27",
  "features": [],
  "model_profiles": {},
  "approval": {
    "status": "approved",
    "approved_by": "local-user",
    "approved_at": "2026-08-27T00:00:00+08:00",
    "approved_hash": "sha256-of-normalized-registry-payload",
    "change_summary": "建立首版 Tg 特征来源登记",
    "review_note": "单人审核通过；计量比特征保持 blocked"
  }
}
```

`registry_hash` 在运行时由规范化 JSON 计算，计算时排除 hash 自身和可变审核时间；文件中已有的 hash 只作为声明，校验器必须重新计算并比较 `approved_hash`，不能盲信文件内容。

每个 feature 至少包含：

```json
{
  "feature_id": "cfrp.tg.cure_stage_count",
  "name": "cure_stage_count",
  "label": "固化阶段数",
  "source_type": "derived_workflow",
  "data_type": "integer",
  "unit": "stage",
  "required_for_prediction": true,
  "nullable": false,
  "default_policy": "workflow_only",
  "calculation_rule": {
    "implementation": "core.process_features:derive_cure_stage_count",
    "version": "1",
    "input_fields": ["cure_temperature_c", "cure_time_h"],
    "null_policy": "reject",
    "invalid_policy": "reject"
  },
  "description": "由固化温度/时间序列解析得到",
  "valid_range": {"min": 0, "max": 4},
  "enum_mapping": null,
  "aliases": [],
  "targets": ["epoxy_resin.tg"],
  "status": "approved"
}
```

字段约束：

- `source_type` 只能是 `molecular_workflow`、`derived_workflow`、`manual_input`、`target`、`metadata`、`unknown`；
- `default_policy` 只能是 `forbidden`、`explicit_only`、`workflow_only`；
- `manual_input` 不得有数值默认值，建议固定为 `forbidden` 或 `explicit_only`；
- `required_for_prediction=true` 时不能 `nullable=true`；
- workflow 来源必须有完整的 calculation rule、input fields、单位、空值和异常策略；
- `unknown` 或 `blocked` 特征必须带阻断原因；
- `valid_range` 的上下界必须有限且下界不大于上界；
- `feature_id`、`name` 在同一 registry snapshot 内都必须唯一；
- 一个特征不能同时属于 workflow 和 manual_input；
- `target` 与 model profile 的 target_col 必须一致。

`model_profiles` 按材料和目标保存有序 feature identity：

```json
{
  "epoxy_resin.tg": {
    "material_type": "epoxy_resin",
    "target": "tg",
    "target_col": "tg_c",
    "feature_ids": [
      "cfrp.tg.cure_stage_count",
      "cfrp.tg.degree_of_cure_pct",
      "cfrp.tg.stoichiometric_ratio_r"
    ],
    "status": "blocked",
    "blocked_feature_ids": ["cfrp.tg.stoichiometric_ratio_r"]
  }
}
```

初始 Tg profile 必须保留现有模型的 532 个有序列身份，包括 28 个缺口；不能为了让旧模型看起来可运行而删除 `stoichiometric_ratio_r` 或其他缺口。该 profile 的状态为 blocked，直到所有来源和计算语义获得批准。

## 4. `prediction_contract` v2

contract 是某一次训练 artifact 对 registry snapshot 的不可变引用，至少包含：

```json
{
  "schema_version": 2,
  "material_type": "epoxy_resin",
  "target": "tg",
  "target_col": "tg_c",
  "model_profile_id": "epoxy_resin.tg",
  "feature_cols": [],
  "feature_definitions": [],
  "workflow_feature_cols": [],
  "molecular_workflow_feature_cols": [],
  "derived_feature_cols": [],
  "manual_input_feature_cols": [],
  "workflow_source_fields": [],
  "feature_registry_version": "2026.08.27",
  "feature_registry_hash": "sha256-of-approved-registry",
  "workflow_hash": "sha256-of-workflow-payload",
  "workflow_schema_version": 2,
  "missing_value_policy": "reject_user_missing",
  "training_missing_policy": "pipeline_imputer_only",
  "numeric_ranges": {},
  "model_fingerprint": "sha256-of-model-fingerprint",
  "canonical_feature_cols": [],
  "effective_feature_cols": [],
  "removed_feature_cols": [],
  "removed_feature_reasons": {}
}
```

字段语义：

- `workflow_feature_cols` 是所有由可执行 workflow 生成的模型列；`molecular_workflow_feature_cols` 和 `derived_feature_cols` 是其两个来源子集。
- `manual_input_feature_cols` 是模型直接读取的实验/工艺输入列。
- `workflow_source_fields` 是生成 workflow 特征所需的原始输入，不等同于模型列。例如 SMILES 和固化温度/时间序列属于 source fields。
- `feature_definitions` 是训练时 registry snapshot 中对应定义的完整深拷贝，不引用运行时最新文件。
- `training_missing_policy` 只描述训练数据内部的显式 Pipeline 行为；`missing_value_policy` 固定约束门户预测。

发布不变量为：

```text
feature_cols == canonical_feature_cols
feature_cols == workflow_feature_cols + manual_input_feature_cols
workflow_feature_cols == molecular_workflow_feature_cols + derived_feature_cols
```

上述等式均为有序列表等式；每个特征必须且只能出现在一个来源分区。若未来需要交错来源列，必须显式改变 contract schema，不能在旧 schema 中静默放宽顺序规则。

## 5. 训练数据流

训练入口在构造 `X/y` 后、split/CV/超参优化和模型实例化前执行：

```text
load approved registry
  → resolve model profile
  → validate source fields and feature profile
  → freeze registry snapshot/version/hash
  → build contract draft
  → pass the same snapshot to normal training, CV and optimization
```

所有训练路径必须复用同一 snapshot，包括普通训练、交叉验证和超参搜索。训练代码不再通过 `n_features_in_`、feature mask 或列数量倒推业务输入含义；这些信息只能用于一致性校验。

现有训练 Pipeline 的训练集缺失值处理可以保留，但必须记录：

```text
canonical_feature_cols
effective_feature_cols
removed_feature_cols
removed_feature_reasons
```

如果训练内部删除全空列、零方差列或其他登记特征，`effective_feature_cols` 不再等于 canonical 列，artifact 只能标记为 needs_validation，不能按原 profile 发布；需要重新登记 profile 或重新训练。

artifact 和训练记录必须保存：

- 完整 `prediction_contract`；
- 完整 registry snapshot；
- registry version/hash；
- workflow snapshot/hash/schema；
- canonical/effective/removed 特征及删除原因；
- target、model fingerprint 和预处理器状态。

## 6. 共享 workflow 与派生特征

新增共享纯 Python 实现：

```text
core/process_features.py
```

训练、门户和离线脚本都调用同一实现；`scripts/expand_manual_process_columns.py` 不再拥有独立的业务解析规则。

共享实现必须：

- 统一解析逗号、分号、中文分隔符和允许的数值单位；
- 固化阶段最多 4 个，后固化阶段最多 2 个；
- 计算阶段数、总时间、最大/最终温度、温度-时间积分、时间加权平均温度及总热处理统计量；
- 对阶段数量不匹配、非法数值、超出上限、缺少必要输入返回结构化错误；
- 不使用 0、均值、训练统计量或 imputer 生成派生值；
- 只有 registry 明确声明为结构性零值的字段才允许输出 0；
- 记录 implementation version，并使 workflow hash 随规则变化而变化。

后固化不存在时，`has_post_cure=0`、阶段数和总时间是否为结构性 0，必须由各字段的 registry rule 明确声明；最大温度、最终温度、积分和加权平均温度若没有物理定义，应返回不可计算并阻断，而不是擅自填 0。

SMILES/BigSMILES 组件拆分必须由一个共享函数实现，固定规范化顺序、空值语义、非法结构行为和最大组件槽位。`curing_agent_smiles`、`hardener_smiles` 等名称只能作为 registry 中显式声明的 alias；不能根据字符串前缀猜测。可选固化剂槽位由 workflow input contract 声明，不能把所有未使用编号槽位一律当作错误。

## 7. 发布门禁与配置行为

`validate_publication_artifact()` 在现有模型、pipeline、预处理器和 workflow 校验之上增加：

1. registry version/hash 与嵌入 snapshot 一致；
2. contract 的 feature definitions 与 snapshot 一致；
3. artifact、model、pipeline、contract 的特征名、数量和顺序一致；
4. 三个来源分区无重复、无遗漏；
5. workflow/derived 实现能生成声明列，hash 和 schema 一致；
6. 不存在 unknown/blocked 特征；
7. effective feature set 等于 canonical feature set；
8. 所有 manual input 的 default policy、nullable、范围和枚举定义合法；
9. 预处理器状态与 contract 声明一致。

只有全部通过后才能写入：

```text
publication_status = "published"
enabled = true
```

否则必须写入：

```text
publication_status = "needs_validation"
enabled = false
```

`make_publication_entry()` 不得再默认把任意上传 artifact 标为 published。旧 artifact 不自动补 contract、不覆盖文件、不删除文件；现有两个字节级重复的 Tg artifact 保持原样，均按 needs_validation 处理。模型统计和门户入口只认 `publication_status == "published"`，不能因 `published` 字段缺失而显示为已发布。

## 8. 门户行为

`UserPrediction.py` 删除按模型列生成输入字段的逻辑，不再使用 `parameter_from_feature()` 或 `sync_parameters_from_features()` 作为契约来源。

门户从发布 artifact 的 contract 和 registry snapshot 构造字段：

- workflow source fields：展示树脂 SMILES、固化剂 SMILES、固化/后固化温度时间序列等原始输入；
- molecular workflow：执行后可显示只读计算预览，不显示为普通模型数值输入；
- derived workflow：展示原始工艺输入和只读派生结果；
- manual input：显示 label、单位、有效范围、实验含义、枚举选项和必填状态。

所有 manual input 控件初始为空，不存在数值默认值。空值、非法值、越界值、枚举未映射值均阻断。系统不得使用 0、均值、中位数、训练均值或 imputer 伪造门户字段。AI 只能解析用户已经提供的文本；AI 建议必须逐项人工确认后才能进入同一校验流程。

预测请求仍采用单一可信入口：

```text
{material_type, target, inputs, confirmed_by_user}
```

后端按 registry source type 和 contract 白名单检查未知列、必填项、有限值、范围、workflow 可计算性和确认状态，然后按 `contract.feature_cols` 严格排序调用模型。

## 9. 错误与审计

校验器返回结构化诊断，至少包含错误代码、特征名、来源、规则和用户可读消息。以下情况不得降级为警告：

- registry 未批准或 hash 不一致；
- 特征未登记、重复、来源冲突；
- unknown/blocked 特征进入训练或发布；
- workflow 输出少于 contract；
- manual input 缺失或不合法；
- 模型、artifact、contract、workflow 顺序或数量不一致。

训练 run、artifact、发布条目和门户任务快照至少记录 registry version/hash、contract hash、workflow hash、model fingerprint、canonical/effective feature count 和阻断原因。日志不得记录 API key、完整敏感输入或任意 AI 生成的未确认数值。

## 10. 迁移与兼容边界

第一阶段不执行当前 Tg 的真实预测，也不把旧模型改造成可发布模型。当前 Tg 模型的 532 列、504 个 workflow 输出、28 个缺口、缺少 prediction_contract 和 blocked 的计量比特征都必须在审计报告中保留。

旧配置和旧 artifact 可以继续存在，读取时统一进入 needs_validation；不存在 registry/contract 的旧 artifact 不自动推断来源。新模型必须从 approved registry snapshot 开始训练。若最终确认计量比公式或某些派生字段无法可靠实现，则选择重新定义 profile 并重新训练，而不是给旧模型塞默认值。

## 11. 验收测试

### 登记库

- 特征名和 feature_id 唯一；
- manual_input 禁止数值默认值；
- workflow 特征不能重复声明为 manual_input；
- required/nullable、范围、公式、input_fields、枚举编码校验；
- registry hash 稳定且审核 hash 可复核；
- draft、deprecated、blocked 状态行为明确；
- 本地单人批准记录完整。

### 训练与 artifact

- 训练开始前拒绝未登记列、重复列和 blocked profile；
- 普通训练、CV、超参优化使用同一 registry snapshot；
- contract 保存 registry version/hash 和完整 definitions；
- effective 特征被删除时 artifact 不能发布；
- artifact 与 registry profile 不一致时不能发布。

### workflow 与门户

- 派生特征线上输出与离线清洗结果一致；
- 阶段数量不匹配、非法数值和缺少必要输入被阻断；
- 单组分体系固化剂组件计数明确为 0；
- canonical curing-agent alias 和可选槽位行为一致；
- 页面只显示 manual input 和 workflow source fields；
- 缺少 manual input 不能预测；
- 不允许 0、均值或 imputer 伪造门户字段；
- AI 未确认字段不能进入预测。

### 当前 Tg

- 缺少 contract、workflow 少于模型特征、存在 blocked 特征时发布门禁失败；
- 配置中缺少 `publication_status` 时不显示为已发布；
- 不执行真实 Tg 预测。

## 12. 第一阶段明确不做的事情

- 不引入 SQLite 或远程多用户审批；
- 不自动修复两个现有 Tg artifact；
- 不猜测 `stoichiometric_ratio_r` 公式；
- 不用页面配置覆盖训练契约；
- 不把所有 532 个特征直接变成用户输入框；
- 不让 AI 生成缺失特征或执行代码；
- 不在契约闭合前执行真实 Tg 预测。
