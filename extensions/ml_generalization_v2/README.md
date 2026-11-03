# CFRP 机器学习分支：训练准入控制与文献基线泛化瓶颈诊断 (ml_generalization_v2)

---

## 0. 最高优先级设计：绝对保留原系统，新增独立评估分支

本项目独立建立于 `/home/wangj/project/CFRP/extensions/ml_generalization_v2/`，与 MD 模拟任务明确分工：
- **明确分工**：本分支**专门负责机器学习接入、训练准入防御与文献基线泛化诊断**，不负责重跑 MD 模拟，不修改 HTPolyNet / LAMMPS 共享模板和力场缓存，避免与 MD Agent 产生协同冲突；
- **旧系统绝对保留**：未修改 `app.py`、`app_lib.py`、`core/`、数据库 CSV、已有训练模型及旧特征注册逻辑；原系统 17 个受保护核心文件保持 100% 字节级一致（SHA256 完全对齐）；
- **原服务端口无抢占**：原系统在 8501 端口的 Streamlit 服务（PID 167411）持续稳定运行；
- **训练数据零污染**：当前允许进入正式 ML 训练的 MD 记录数严格为 0 条，历史伪造小试数据已全部隔离在 `artifacts/quarantine_synthetic_v1/`，绝不混入模型训练。

```text
/home/wangj/project/CFRP/extensions/ml_generalization_v2/
├── README.md                      # 架构设计、缺陷修复与泛化诊断深度报告
├── REPRODUCE_TESTS.sh             # 全量验收测试与旧系统保护终验脚本
├── REPRODUCE_BASELINE_AUDIT.sh    # 文献基线独立复现与泛化诊断一键脚本
├── SHA256_MANIFEST.txt            # 全量工件 SHA256 清单
├── configs/                       # 契约配置与参数
├── src/                           # 核心重构与准入安全模块
│   ├── __init__.py
│   ├── admission.py               # 严格训练准入控制器 (严格布尔解析与三种路径隔离)
│   ├── feature_contract.py        # 特征契约与防静默退化控制器 (强校验增量列)
│   ├── split_manager.py           # 严格无泄漏分组划分 (随机留出 vs 配方严格留出)
│   ├── baseline_reproducer.py     # 真实文献基线复现与泛化瓶颈诊断引擎
│   └── safe_comparison.py         # 修复漏洞后的 FeatureComparisonEngine 替代实现
├── tests/                         # 验收测试套件 (包含两项实测攻击反例与8大测试场景)
│   ├── test_fault_injection_admission.py
│   ├── test_ml_pipeline_full.py
│   └── test_unbroken_old_system.py
├── reports/                       # 核心审计账本、冻结划分与科学报告
│   ├── TRAINING_ADMISSION_AUDIT.json      # 训练准入审计报告
│   ├── FEATURE_CONTRACT.json              # 特征契约与阻断规则
│   ├── DATA_QUALITY_AUDIT.csv             # 数据粒度、重复性与测试条件账本
│   ├── SPLIT_MANIFEST.json                # 冻结的样本折叠划分清单与哈希
│   ├── LEAKAGE_AUDIT.json                 # 特征/标签泄漏风险排查报告
│   ├── BASELINE_METRICS.json              # 真实文献基线多模型指标汇总
│   ├── BASELINE_AND_GENERALIZATION_REPORT.md # 泛化瓶颈深度归因报告
│   └── MD_ABLATION_READINESS.json         # MD 接入契约与就绪度 (B/C/D 阻断声明)
└── artifacts/
    ├── snapshots/                         # comparison.py 旧版快照与 diff
    │   └── comparison_v1_legacy.py
    └── oof_predictions/                   # 导出的实际样本 Pooled OOF 预测与残差
        ├── oof_predictions_tg_random.csv
        ├── oof_predictions_tg_group.csv
        ├── oof_predictions_modulus_random.csv
        └── oof_predictions_modulus_group.csv
```

---

## 1. 已实测的两项 ML 入口缺陷彻底修复证据

经复核桌面实测审核脚本（`audit_comparison.py` 与 `audit_results.json`），旧评估代码存在两大致命漏洞：
1. **漏洞 1（不合格样本非法放行）**：
   旧 `FeatureComparisonEngine` 仅按 `target` 与 `group` 是否缺失过滤，完全忽略 `eligible_for_training`、`scientific_gate_status` 与 `data_origin`。当所有记录显式标记为 `eligible_for_training=False`、`scientific_gate_status='FAIL'`、`data_origin='SYNTHETIC_FIXTURE'` 时，旧代码依然训练并返回 `status: PASS`。
2. **漏洞 2（特征缺失静默退化）**：
   旧代码采用 `active_cols = [c for c in feat_cols if c in sub_df.columns]`，当删除全部 MD 列时，B、C、D 组声明的增量特征全部丢失，代码静默把四组特征缩减为只剩旧特征 `['old']`，导致四组模型实际上都在跑 A 组，仍然返回假 PASS！

### 修复方案：
- **`src/admission.py` (训练准入控制器)**：
  - 引入严格布尔解析 `strict_parse_bool()`，彻底杜绝字符串 `'False'`、`'false'`、`'0'` 被 `bool()` 误判为 True 的漏洞；
  - 实行三种路径隔离：`baseline_literature_only`（允许真实文献基线，不强制要求 MD）、`formal_md_ablation`（正式 MD 消融，实行最高准入防御）、`diagnostic_fixture_test`（诊断测试 fixture 专用，标注 DIAGNOSTIC_ONLY）；
  - 当样本在 `formal_md_ablation` 路径下标记为不合格时，**坚决返回 `ADMISSION_REJECTED`，样本数置 0，拒绝放行任何模型训练，绝不伪造指标**！
- **`src/feature_contract.py` (防静默退化控制器)**：
  - 显式声明每组的核心增量特征列；
  - 请求 B、C、D 组时，若声明的增量列存在缺失，**坚决将该组标记为 `BLOCKED` 或 `NOT_RUN`，并列出 `missing_columns`，绝不允许悄悄退化成 A 组旧特征**！

---

## 2. 真实文献数据粒度核验与泄漏审计

从真实文献数据库中查明：
- **Tg 任务**：
  - 数据文件：`data_fixed/latest_data_clean_594.csv`
  - 样本量：**644 行**（非空真实文献配方 644 条，均值 136.46 ℃，标准差 72.41 ℃）
  - 特征数：168 列数值机理特征
  - 粒度：644 条均为独立规范配方
- **25℃ 储能模量 E' 任务**：
  - 数据文件：`cache/session_snapshots/latest_data.csv`
  - 样本量：**566 行**（非空真实测量 566 条，均值 2.17 GPa，标准差 0.96 GPa）
  - 特征数：484 列数值特征
  - 粒度与关键事实：**566 条测量仅对应 256 个独立配方**！大量样本为同一配方在不同测试频率（1 Hz vs 10 Hz）或不同批次的重复测量！
- **泄漏与质量审计清单**：已导出 `reports/DATA_QUALITY_AUDIT.csv` 与 `reports/LEAKAGE_AUDIT.json`。

---

## 3. 严格无泄漏双重划分体系 (冻结 SPLIT_MANIFEST.json)

- **Random 5-Fold**：5 折随机划分（用于严格复现与对照历史随机自报基线）；
- **Formulation Group 5-Fold**：基于 `formulation_id` 的 5 折严格配方留出（确保同一配方的所有测试记录被严格隔离在单折内，绝不发生跨折配方数据泄漏）。
- 划分清单已完全冻结导出至 `reports/SPLIT_MANIFEST.json`。
- **训练折内拟合铁律**：插补器中位数与 StandardScaler 严格仅在训练折内 `fit`，测试折仅 `transform`，严禁测试集分布反向渗透。

---

## 4. 真实文献基线复现与泛化瓶颈诊断结论

在固定计算预算下，对 4 种代表性模型（Dummy_Median, Ridge_Linear, RandomForest, GradientBoosting）分别在两套划分下进行全折交叉验证，实际 OOF 预测值已保存至 `artifacts/oof_predictions/`：

### 4.1 指标对照总表 (真实测量，绝无粉饰)

| 任务 | 目标 (单位) | 划分方式 | 模型 | Pooled OOF MAE | Pooled OOF RMSE | Pooled OOF R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tg** | tg_c (℃) | 随机 5-Fold (历史对照) | Dummy_Median | 56.46 ℃ | 72.41 ℃ | -0.0016 |
| | | | Ridge_Linear | 34.72 ℃ | 48.18 ℃ | 0.5566 |
| | | | RandomForest | 30.63 ℃ | 42.47 ℃ | **0.6555** |
| | | | GradientBoosting | 29.77 ℃ | 41.79 ℃ | **0.6664** |
| **Tg** | tg_c (℃) | 配方留出 5-Fold (无泄漏) | Dummy_Median | 56.49 ℃ | 72.41 ℃ | -0.0016 |
| | | | Ridge_Linear | 34.84 ℃ | 48.77 ℃ | 0.5456 |
| | | | RandomForest | 31.72 ℃ | 44.07 ℃ | **0.6290** |
| | | | GradientBoosting | 30.28 ℃ | 42.62 ℃ | **0.6529** |
| **E'** | storage_modulus (GPa) | 随机 5-Fold (历史对照) | Dummy_Median | 0.7451 GPa | 0.9671 GPa | -0.0203 |
| | | | Ridge_Linear | 0.5382 GPa | 0.7395 GPa | 0.4034 |
| | | | RandomForest | 0.4655 GPa | 0.6391 GPa | **0.5544** |
| | | | GradientBoosting | 0.4592 GPa | 0.6299 GPa | **0.5672** |
| **E'** | storage_modulus (GPa) | 配方留出 5-Fold (无泄漏) | Dummy_Median | 0.7647 GPa | 0.9871 GPa | -0.0629 |
| | | | Ridge_Linear | 0.7520 GPa | 0.9609 GPa | **-0.0071** |
| | | | RandomForest | 0.5805 GPa | 0.7296 GPa | **0.4193** |
| | | | GradientBoosting | 0.5883 GPa | 0.7448 GPa | **0.3948** |

### 4.2 泛化瓶颈核心归因（区分事实与推论）：
1. **同配方跨折泄漏对历史高 R² 的贡献（事实）**：
   在储能模量任务中，由于 566 条样本只对应 256 个配方，随机划分时同配方的多次重复测试被分散到训练与测试两侧。配方留出后，树模型的 R² 出现**断崖式下跌（从 0.567 跌落至 0.395，MAE 恶化 28%）**，Ridge 线性模型的 R² 直接跌为负数 (-0.007)！证明历史自报的 R²≈0.5~0.9 包含显著的数据泄漏成分。
2. **测试协议与测量模式混杂带来的标签噪声（事实）**：
   文献中储能模量跨越了 1 Hz 与 10 Hz 两种频率，夹具涵盖拉伸与双悬臂弯曲；Tg 混杂了 DSC 中点与 DMA tan δ 峰值（系统偏差达 15~25 ℃）。
3. **二维指纹对立体交联网络信息的表达缺失（物理依据）**：
   二维拓扑无法表达空间受限下的交联加成率与微观自由体积空穴分布，为后续引入合格的三维 MD 物理特征提供了明确的立足点。

---

## 5. MD 特征接入契约与就绪度声明

在 `reports/MD_ABLATION_READINESS.json` 中明确声明：
- **当前允许进入正式 ML 训练的 MD 记录数**：**严格为 0 条**；
- **B / C / D 特征组就绪度**：全量标记为 **`BLOCKED`**；
- **阻断原因**：当前 MD 成果属于单配方微小体系（262 原子）探索性诊断闭环，未达到宏观文献样本规模，且 CED 孤立参考态尚未推导完毕；
- **后续接入最小契约**：必须提供 `literature_record_id`、`formulation_id`、`network_id` 链条、通过 S0~S5 门禁的拓扑字段（`md_alpha_actual`, `md_largest_cluster_fraction`）与通过 S3/S6 门禁的自由体积字段（`md_density_g_cm3`, `md_ffv_probe_0A`, `md_ffv_probe_1_2A`）。

---

## 6. 一键复现指令

### 6.1 全量验收测试（包含 7 类故障注入、两项实测反例与旧系统保护审计）
```bash
cd /home/wangj/project/CFRP/extensions/ml_generalization_v2
./REPRODUCE_TESTS.sh
```

### 6.2 真实文献基线复现与泛化诊断
```bash
cd /home/wangj/project/CFRP/extensions/ml_generalization_v2
./REPRODUCE_BASELINE_AUDIT.sh
```
