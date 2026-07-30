# 工艺特征 PLS 降维设计

## 目标

针对工艺特征缺失率高、共线性强、直接输入神经网络不稳定的问题，在“特征选择”页面增加工艺特征专用 PLS 处理流程。PLS 只处理工艺特征，不改变分子特征工程顺序，也不让模型训练页面承担复杂的降维配置。

## 实现状态

- 已新增 `core/process_pls.py`，提供可序列化的 `ProcessPLSTransformer`、VIP 评分、成分数 CV 选择和 workflow 指纹。
- 特征选择页负责 PLS 预览与锁定；训练页只暴露“使用已锁定工艺 PLS”的轻量开关。
- 训练和交叉验证把 `process_pls` 放入 sklearn Pipeline，正式拟合发生在训练折内。
- 模型 artifact 的 `extra` 中保存 `process_pls_workflow`、`process_pls_schema_version` 和 `process_pls_workflow_hash`。
- 预测和高通量筛选复用已保存 Pipeline 中的 fitted `process_pls` step；缺少原始工艺列或缺少 fitted step 时显式报错。

## 用户界面边界

所有 PLS 配置、拟合、诊断和锁定操作放在“特征选择”页面：

- 选择工艺特征组，排除目标列、ID 列、分子特征和非数值列。
- 显示缺失率、有效样本数、候选成分数、交叉验证结果和 VIP 排名。
- 默认自动选择成分数；不在模型训练页面增加 PLS 参数区。
- 用户确认后生成并锁定一个可复用的工艺 PLS 工作流。
- 模型训练页面只显示当前已锁定的工作流摘要，并提供“使用/不使用已锁定工艺 PLS”的轻量选项。

## 训练数据流

PLS 工作流按以下顺序执行：

```text
原始工艺特征
→ 缺失率过滤
→ 缺失指示变量
→ 仅训练数据拟合缺失值处理器
→ 仅训练数据拟合 RobustScaler
→ 仅训练数据交叉验证选择 PLS 成分数
→ 仅训练数据计算 VIP
→ PLS 潜变量 + VIP 原始工艺特征 + 缺失指示变量
```

特征选择页面可以提供探索性预览，但不能把使用全数据拟合得到的 PLS 结果直接作为正式训练输入。

## 数据泄漏防护

- 普通训练：先划分训练、验证、测试集，再只用训练集拟合缺失处理器、缩放器和 PLS。
- 交叉验证：每个 fold 独立拟合完整预处理与 PLS，验证 fold 只能调用 `transform`。
- 自动选择成分数时使用训练集内部交叉验证，综合 CV R²、RMSE、折间稳定性和成分数复杂度。
- VIP 排名只从训练数据产生。
- 测试、预测和高通量筛选禁止重新 `fit`，必须加载锁定的 PLS 工作流。

## 输出特征

输出由三部分组成：

1. `process_pls_1 ... process_pls_n`；
2. VIP 达标或排名靠前的少量原始工艺特征；
3. 与保留工艺特征对应的缺失指示变量。

模型 artifact 必须同时保存两类信息：`pipeline.named_steps['process_pls']` 中的 fitted 转换器，以及 `extra.process_pls_workflow` 中的紧凑审计元数据。元数据保存原始列顺序、配置参数、输出列顺序、workflow hash 和 schema version；缺失处理器、缩放器、PLS 权重与 VIP 结果由 fitted Pipeline step 承载。

## 模型兼容

- 回归模型可按模型选择是否使用已锁定的工艺 PLS。
- 未启用时保持现有原始工艺特征流程。
- 分类模型不直接使用普通 PLS；后续若需要，可单独设计 PLS-DA。
- 分子特征不参与工艺 PLS，原有分子特征工作流保持不变。

## 异常与回退

- 工艺特征不足、有效样本不足或候选成分不可行时，阻止锁定并给出原因。
- 工艺特征全部缺失时，不生成 PLS。
- PLS 失败时不得静默改用全局均值或零值伪造结果。
- 工作流指纹、列顺序或版本不匹配时，训练和筛选必须要求重新确认。

## 验收标准

- 已验收：特征选择页能够预览、自动选择并锁定工艺 PLS 工作流。
- 已验收：训练页不出现大块 PLS 参数配置，只显示工作流摘要和轻量启用选项。
- 已验收：训练、验证、测试、预测和高通量筛选复用同一套已保存转换流程。
- 已验收：回归交叉验证中不存在全数据拟合造成的泄漏。
- 已验收：关闭 PLS 时，现有模型输入和结果路径不发生变化。
- 已验收：测试覆盖训练折拟合边界、保存/加载一致性、列顺序校验、缺失掩码、旧模型兼容和筛选复现。

## 验证命令

```powershell
& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest -q 'tests/test_process_pls.py' 'tests/test_virtual_screening.py' 'tests/test_missing_target_training.py' 'tests/test_regression_target_balance.py' 'tests/test_transformer_bnn_training.py' 'tests/test_molecular_feature_workflow.py'
```
