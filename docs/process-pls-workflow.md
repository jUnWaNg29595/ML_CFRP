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
