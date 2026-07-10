# Hybrid Model Parameter Guide

适用模型：
- `FT-Transformer`
- `Transformer + BNN`
- `Epoxy PINN (Physics-Informed)`
- `Transformer + PINN`
- `GNN + Transformer Fusion`

对应实现位置：
- `core/fttransformer_model.py`
- `core/transformer_bnn_model.py`
- `core/pinn_model.py`
- `core/transformer_pinn_model.py`
- `core/gnn_transformer_fusion_model.py`
- `core/missing_value_handler.py`
- `core/ui_config.py`

## 1. 先看结论

如果你的树脂特征维度很多、样本量中等或偏小、还有不少空特征，调参优先级建议是：

1. `missing_value_strategy`
2. `ffn_dropout` / `weight_decay`
3. `learning_rate`
4. `d_token`
5. `n_blocks`
6. `sub_attention_dim`
7. 各模型专属参数

对高维树脂数据，通常先把模型调“稳”，再去加大容量。

## 2. 当前缺失特征机制

现在这套系统不再只依赖简单中位数填补，而是支持三层机制同时配合：

1. 缺失值插补
2. 缺失位置掩码
3. Transformer 对缺失模式的感知

### 2.1 `missing_value_strategy`

可选值：
- `median`
- `bayesian`
- `multiple_bayesian`

含义：

| 参数 | 原理 | 优点 | 风险 | 适用场景 |
|---|---|---|---|---|
| `median` | 每列中位数填补 | 快，稳，便宜 | 容易压缩分布，削弱相关性 | 缺失少、先跑基线 |
| `bayesian` | 基于 `IterativeImputer + BayesianRidge` 的迭代插补 | 能利用特征间相关性 | 比中位数慢，样本太少时会不稳 | 中等缺失，特征相关性较强 |
| `multiple_bayesian` | 多次贝叶斯插补后取平均 | 比单次插补更稳，能减小单次插补偶然性 | 更慢，参数更多 | 缺失较多且你愿意换精度 |

建议：
- 缺失少于 `5%`：先用 `median`
- 缺失约 `5%~20%`：优先 `bayesian`
- 缺失高于 `20%` 且样本量还能支撑：试 `multiple_bayesian`

### 2.2 `missing_imputer_max_iter`

作用：控制贝叶斯插补的迭代次数。

经验范围：
- `10~15`：一般够用
- `15~25`：高维、相关性强时可增加
- 超过 `30`：收益通常开始变小

现象判断：
- 插补后效果仍明显偏硬、偏平均：可略增
- 训练明显变慢但指标没改善：应回调

### 2.3 `missing_n_imputations`

仅对 `multiple_bayesian` 生效。

作用：重复做几次贝叶斯插补，再取均值。

建议：
- 起步 `5`
- 常用 `5~10`
- 如果数据很小，不建议拉太高

### 2.4 `use_missingness_embedding`

作用：把“这个特征原本缺失”作为额外信号注入 token 表示。

理解方式：
- 普通插补只告诉模型“填了什么值”
- 缺失嵌入额外告诉模型“这个值原本是空的”

建议：
- 对 `FT-Transformer`、`Transformer + BNN`、`Transformer + PINN`、`GNN + Transformer Fusion`，默认保持 `True`
- 如果缺失非常少，也可以关掉做对照

### 2.5 `use_missingness_attention`

作用：让子注意力网络在做特征门控时，也参考缺失模式。

适合：
- 空特征并不是随机发生
- 某些配方、工艺、测试条件经常一起缺

风险：
- 样本很少时，模型可能把“缺失模式”学得过重

建议：
- 默认 `True`
- 如果发现模型过度依赖“有没有值”而不是“值本身”，可试着关掉对照

## 3. FT-Transformer / 混合 Transformer 主干参数

这部分参数适用于：
- `FT-Transformer`
- `Transformer + BNN`
- `Transformer + PINN`
- `GNN + Transformer Fusion` 的表格分支

| 参数 | 作用 | 调大后 | 调小后 | 建议 |
|---|---|---|---|---|
| `d_token` | 每个特征 token 的维度 | 表达更强，显存更高，更易过拟合 | 更稳，但容量下降 | 高维数据先用 `64~128` |
| `n_blocks` | Transformer 层数 | 交互更深 | 更稳，更浅 | 常用 `2~4` |
| `attention_n_heads` | 多头注意力头数 | 表达更细 | 更简单 | 常用 `4` 或 `8` |
| `ffn_d_hidden` | FFN 隐层宽度 | 非线性更强 | 更保守 | 常用 `2x~4x d_token` |
| `pooling` | token 汇聚方式 | `attention` 更灵活 | `cls/mean` 更稳 | 先用 `cls` |
| `head_hidden_dim` | 输出头宽度 | 头部更灵活 | 更简单 | 常用 `64~160` |
| `layer_norm_eps` | LayerNorm 稳定项 | 数值更保守 | 更敏感 | 一般保持默认 |

高维树脂数据起步建议：
- `d_token=96` 或 `128`
- `n_blocks=3`
- `attention_n_heads=4` 或 `8`
- `ffn_d_hidden=256` 或 `384`

## 4. 子注意力网络参数

子注意力网络负责先做一轮特征级门控，再交给 Transformer。

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `sub_attention_dim` | 门控网络宽度 | 高维起步 `32~64` |
| `sub_attention_dropout` | 门控层 dropout | 常用 `0.1~0.2` |
| `sub_attention_temperature` | 门控分布锐度 | 常用 `0.8~1.5` |
| `feature_gate_type` | `softmax` 或 `sigmoid` | 高维场景优先 `softmax` |
| `feature_gate_scale` | 门控放大强度 | 默认 `1.0`，不稳就降到 `0.6~0.8` |
| `use_feature_residual` | 是否保留原始 token 残差 | 建议保持 `True` |

经验解释：
- `softmax` 更适合做特征竞争，能压住冗余特征
- `sigmoid` 更宽松，可能让太多特征同时放大
- 高维表格数据通常更需要“收缩”而不是“放开”

## 5. 正则化与训练稳定性

| 参数 | 作用 | 建议范围 |
|---|---|---|
| `attention_dropout` | 注意力层 dropout | `0.05~0.15` |
| `ffn_dropout` | FFN dropout | `0.1~0.25` |
| `residual_dropout` | 残差 dropout | `0~0.1` |
| `token_dropout` | 输入 token dropout | `0.0~0.1` |
| `weight_decay` | 权重衰减 | `1e-4 ~ 1e-3` |
| `gradient_clip_norm` / `grad_clip` | 梯度裁剪 | `0.5~1.5` |

判断规则：
- 训练好、验证差：先加 `ffn_dropout` 和 `weight_decay`
- loss 抖动大：先降 `learning_rate`，再加裁剪
- 很容易塌成常数：先减小正则，再看缺失插补策略

## 6. 学习率和训练控制

| 参数 | 建议 |
|---|---|
| `learning_rate` / `lr` | Transformer 起步 `5e-4 ~ 1e-3` |
| `batch_size` | 高维数据常用 `16~64` |
| `epochs` | 让早停决定，不要只看上限 |
| `patience` | 一般设置为 `20~80`，看模型大小 |
| `validation_split` | 常用 `0.1~0.2` |
| `scheduler_factor` | 常用 `0.5~0.7` |
| `scheduler_patience` | 常用 `5~10` |
| `min_learning_rate` | 一般 `1e-6` 即可 |

经验：
- 样本偏少时，小 batch 往往比大 batch 泛化更稳
- 如果你的数据噪声较大，不要把学习率设得过低，否则容易学到插补误差

## 7. Transformer + BNN 专属参数

`Transformer + BNN` 的结构是：
- 前面用 FT-Transformer 抓特征交互
- 后面用概率回归头输出均值和不确定性

关键参数：

| 参数 | 作用 | 建议 |
|---|---|---|
| `mc_samples` | 推理时 MC Dropout 采样次数 | `30~80` |
| `loss_name` | `gaussian_nll` 或 `mse` | 要不确定性就用 `gaussian_nll` |
| `min_logvar` / `max_logvar` | 限制方差输出范围 | 先保持默认 |

建议：
- 如果你重视不确定性排序，先把 `mc_samples` 提到 `50`
- 如果预测均值比不确定性更重要，先把主干调稳，再调 `mc_samples`

## 8. Epoxy PINN 专属参数

适用于：
- `Epoxy PINN (Physics-Informed)`

| 参数 | 作用 | 建议 |
|---|---|---|
| `mode` | `tg` / `mechanics` / `generic` / `auto` | 任务明确时不要总用 `auto` |
| `physics_weight` | 物理约束权重 | 常从 `0.001~0.003` 起 |
| `physics_formula` | `standard` / `advanced` | 先 `standard` |
| `target_name` | 帮 `auto` 识别任务 | 只对 PINN 系列有意义 |
| `hidden_dim` | MLP 宽度 | 常用 `256~768` |
| `n_layers` | MLP 深度 | 常用 `3~6` |

说明：
- `physics_weight` 太大时，预测会变“硬”
- 数据噪声较大时，优先减小 `physics_weight`
- 如果你发现模型总被公式拉走，先把 `physics_formula` 退回 `standard`

## 9. Transformer + PINN 专属参数

这是“Transformer 编码 + 物理损失”的组合。

建议理解为：
- Transformer 负责高维特征交互
- PINN 负责物理边界和机理约束

优先调参顺序：
1. 先把 Transformer 主干调稳
2. 再调 `physics_weight`
3. 最后再决定是否上 `advanced`

推荐起步：
- `d_token=96`
- `n_blocks=3`
- `ffn_d_hidden=256`
- `sub_attention_dim=64`
- `ffn_dropout=0.15`
- `lr=0.0008`
- `weight_decay=0.0003`
- `physics_weight=0.001~0.002`
- `physics_formula=standard`
- `missing_value_strategy=bayesian`

## 10. GNN + Transformer Fusion 专属参数

这是“SMILES 图分支 + 表格分支”的融合模型。

| 参数 | 作用 | 建议 |
|---|---|---|
| `graph_model_type` | 图编码器类型 | 先 `gcn`，再尝试 `gat` |
| `graph_hidden_dim` | 图分支宽度 | `96~192` |
| `graph_num_layers` | 图层数 | `2~4` |
| `graph_dropout` | 图分支 dropout | `0.1~0.2` |
| `graph_pooling` | 图池化方式 | 先 `mean` |
| `gat_heads` | GAT 头数 | 只对 `gat` 生效，常用 `4` |
| `fusion_hidden_dim` | 融合头宽度 | `128~384` |
| `num_workers` | DataLoader 并行数 | Windows 下先 `0` |

建议：
- 不要一开始同时把图分支和表格分支都做大
- 先固定图分支为中等容量，再调表格分支
- 如果数值特征缺失很多，优先先调 `missing_value_strategy` 和缺失感知参数

## 11. 高维树脂特征的推荐起步配置

### 11.1 FT-Transformer

```text
d_token=96
n_blocks=3
attention_n_heads=4
ffn_d_hidden=256
sub_attention_dim=64
attention_dropout=0.1
ffn_dropout=0.15
token_dropout=0.05
feature_gate_type=softmax
feature_gate_scale=1.0
learning_rate=0.0008
weight_decay=0.0005
batch_size=32
missing_value_strategy=bayesian
use_missingness_embedding=True
use_missingness_attention=True
```

### 11.2 Transformer + BNN

```text
d_token=96
n_blocks=3
attention_n_heads=4
ffn_d_hidden=256
sub_attention_dim=64
ffn_dropout=0.2
learning_rate=0.0006
weight_decay=0.0005
batch_size=32
mc_samples=50
missing_value_strategy=bayesian
use_missingness_embedding=True
use_missingness_attention=True
```

### 11.3 Transformer + PINN

```text
d_token=96
n_blocks=3
ffn_d_hidden=256
sub_attention_dim=64
ffn_dropout=0.15
lr=0.0008
weight_decay=0.0003
batch_size=32
physics_weight=0.0015
physics_formula=standard
missing_value_strategy=bayesian
use_missingness_embedding=True
use_missingness_attention=True
```

### 11.4 GNN + Transformer Fusion

```text
graph_model_type=gcn
graph_hidden_dim=128
graph_num_layers=3
graph_dropout=0.1
d_token=96
n_blocks=3
ffn_d_hidden=256
fusion_hidden_dim=256
learning_rate=0.0008
batch_size=16~32
missing_value_strategy=bayesian
use_missingness_embedding=True
use_missingness_attention=True
```

## 12. 缺失值场景的实用建议

### 情况 A：只有少量空特征

```text
missing_value_strategy=median
use_missingness_embedding=True
use_missingness_attention=False 或 True
```

适合先跑基线。

### 情况 B：空特征中等，且特征间相关性明显

```text
missing_value_strategy=bayesian
missing_imputer_max_iter=15
use_missingness_embedding=True
use_missingness_attention=True
```

这是目前最推荐的通用方案。

### 情况 C：空特征较多，且你担心单次插补失真

```text
missing_value_strategy=multiple_bayesian
missing_imputer_max_iter=15~25
missing_n_imputations=5~10
use_missingness_embedding=True
use_missingness_attention=True
```

这是当前最完整的空特征方案，但训练更慢。

### 情况 D：很多特征并不是“缺失”，而是“结构性不存在”

例如某类样品天生没有某个测试字段。

这时建议：
- 保留缺失掩码
- 不要只看插补值
- 优先保留 `use_missingness_embedding=True`
- 尝试 `use_missingness_attention=True`

因为这类问题里，“缺不缺”本身就是信号。

## 13. 症状到调参动作

| 现象 | 优先处理 |
|---|---|
| 训练集很好，验证集差 | 降 `d_token`，降 `n_blocks`，升 `ffn_dropout`，升 `weight_decay` |
| 训练和验证都学不动 | 先试升 `learning_rate`，再加 `d_token` 或 `ffn_d_hidden` |
| loss 抖动大 | 降 `learning_rate`，加 `gradient_clip_norm`，降 `feature_gate_scale` |
| 结果像被“平均化” | 把 `median` 换成 `bayesian`，或打开缺失感知 |
| 贝叶斯插补很慢 | 降 `missing_imputer_max_iter`，或改回 `bayesian` 而不是 `multiple_bayesian` |
| 不确定性过大 | 检查 `mc_samples`，再看主干是否过弱 |
| PINN 预测太僵硬 | 降 `physics_weight`，先用 `physics_formula=standard` |
| GNN 融合模型太慢 | 先用 `gcn`，减 `batch_size`，Windows 下 `num_workers=0` |

## 14. 最后一个原则

对你的这类树脂高维数据，最容易出问题的不是模型不够大，而是：
- 缺失机制没处理对
- 正则不够
- 学习率和容量一起拉太高

所以实际调参时，建议总是按这个顺序：

1. 先定缺失策略
2. 再定正则和学习率
3. 最后再加大模型容量

这样最稳。
