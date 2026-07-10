# BigSMILES 随机图采样

当前系统新增了 BigSMILES 的 stochastic graph 路线：

1. 识别 BigSMILES
2. 解析 repeat block、end group、连接位点
3. 随机采样多个具体 SMILES
4. 对采样结果做 GNN 特征聚合

可在“图神经网络特征”页面启用：

- `BigSMILES 图模式`
- `采样次数`
- `最少重复单元`
- `最多重复单元`

说明：

- `auto`：检测到 BigSMILES 时自动采样
- `sample`：单次采样
- `ensemble`：多次采样后平均
- `off`：关闭随机图采样

这不是严格的 BigSMILES 完整语义解析，但比单一代表性 SMILES 更适合聚合物多构象/多重复单元场景。
