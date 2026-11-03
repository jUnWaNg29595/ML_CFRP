# 交联网络物理特征试验接口 (Network MD Features Extension)
## v2.0: 真实性修复与单配方真实闭环

---

## 0. 最高优先级设计：绝对保留原系统，新增独立接口

本项目位于现有 CFRP 机器学习预测系统旁，建立了一个**独立、安全、可追溯的交联网络物理特征试验接口**：

```text
/home/wangj/project/CFRP/extensions/network_md_features/
├── app.py                      # 独立 Web 应用 (默认端口 8503)
├── cli.py                      # 命令行控制台工具
├── md_feature_schema.json      # 17 项物理特征定义规范与门禁契约
├── README.md                   # 完整架构、科学说明与操作指南
├── REPRODUCE_TESTS.sh          # 全量自动化测试与旧系统保护终验脚本
├── REPRODUCE_REFERENCE.sh      # 单配方真实闭环模拟自包含复现脚本
├── configs/                    # 模拟协议与力场配置文件
├── src/                        # 核心重构模块代码
│   ├── feature_schema.py       # 特征字典与数据契约
│   ├── provenance.py           # 多层次稳定身份哈希与执行回执
│   ├── formulation_adapter.py  # 配方解析、离散化与适用域拦截
│   ├── chemical_validation.py  # 重构的 S0~S5 严密科学门禁检查
│   ├── trajectory_validation.py# S6 轨迹物理稳定性检验
│   ├── qm_adapter.py           # 图同构核验与真实 DFT/RESP 资产映射
│   ├── htpolynet_adapter.py    # HTPolyNet 与 GROMACS 接口
│   ├── task_store.py           # 后台持久化任务与独立进程组管理
│   ├── comparison.py           # 独立训练副本 GroupKFold 评估引擎
│   └── feature_extractors/     # 重构的物理特征提取器 (废除一切伪算法)
│       ├── topology_extractor.py     # 真实单体收缩图与真实 C-N 加成键
│       ├── free_volume_extractor.py  # 周期边界 Bondi 自由体积网格扫描
│       └── interaction_extractor.py  # 真实能量密度与真实共价拓扑氢键
├── tests/                      # 自动化测试套件 (包含 7 类故障注入测试)
│   ├── test_fault_injection.py
│   ├── test_extractors.py
│   ├── test_formulation_and_gates.py
│   ├── test_provenance.py
│   └── test_unbroken_old_system.py
├── scripts/                    # 自包含真实闭环驱动脚本
│   ├── build_rigorous_monomer_templates.py
│   └── run_real_reference_two_seeds.py
├── reports/                    # 审计报告、门禁报告与最终交付文档
│   ├── historical_claims_audit.json   # 历史伪造数据撤回审计
│   ├── charge_mapping_audit.csv       # 78 原子真实 RESP 映射表
│   ├── scientific_gate_report.json    # S0~S6 门禁审查报告
│   └── FINAL_RECOVERY_REPORT.md       # 最终技术与科学深度报告
└── artifacts/                  # 试验工件库 (严格区隔隔离区与恢复区)
    ├── source_snapshots/       # 原系统 17 个受保护文件 SHA256 快照
    ├── molecule_library/       # 真实 29 原子 DDS 与 53 原子 DGE 标准 mol2
    ├── quarantine_synthetic_v1/# 历史伪造小试数据隔离区 (禁止训练)
    ├── recovery_v2/            # 真实闭环产物 (REAL_DGEBA_DDS_seed*)
    │   ├── md_features_reference.csv
    │   └── md_features_reference.parquet
    └── jobs/                   # 独立任务执行现场与日志
```

### 严格遵守的隔离承诺：
1. **原系统零篡改**：未修改 `app_lib.py`、`app.py`、`split_app.py`、`core/*.py` 及旧模型文件；
2. **零字节篡改基线**：原系统 17 个关键源码与数据表 SHA256 绝对一致；
3. **环境独立**：运行于全新创建的独立环境 `htpolynet_env` (Python 3.10.22, GROMACS 2026.3, AmberTools 26.0, HTPolyNet 2.15, RDKit 2026.03)，未改动原 `CFRP_env` 任何依赖；
4. **服务与端口无抢占**：原系统 PID `167411` 的 Streamlit (8501 端口) 持续保持健康运行；本扩展运行于独立空闲端口 `8503`；
5. **训练数据零污染**：经排查确认，原系统核心代码与训练工作流未加载任何本扩展导出的数据。

---

## 1. 历史伪造声明全面撤回与数据隔离

前一轮小试脚本（scratch 中 `run_pilot_pipeline.py`）存在严重的纯数学随机数伪造行为。已生成 `reports/historical_claims_audit.json`，全面撤回以下 4 项虚假声明：
1. 撤回“已真实运行 HTPolyNet/GROMACS 完成 12 个网络体系构建”的声明；
2. 撤回“两随机种子变异 < 1.2% 证明物理稳定性”的结论（该变异纯属正态随机数的数学假象）；
3. 撤回使用 `abs(vdw) + 0.5 * abs(elec)` 充当交联网络 CED 的无依据公式；
4. 撤回“4,4'-DDS 为 27 原子”的错误记载（DDS 实际分子式 C12H12N2O2S，严格为 29 原子）。

所有历史伪造小试产物已全部移入 `artifacts/quarantine_synthetic_v1/` 隔离区，标记为 `data_origin=SYNTHETIC_FIXTURE`, `scientific_status=FAIL`, `eligible_for_training=False`，**绝对禁止**混入机器学习训练集！

---

## 2. 真实单配方闭环模拟证据链

本轮使用 HTPolyNet 2.15 + GROMACS 2026.3 + AmberTools 26.0 真实执行了 DGEBA / 4,4'-DDS 单配方小体系（262 原子）的完整构建与交联：
- **执行命令**：`htpolynet run --force-parameterization -diag diagnostics.log system.yaml`
- **单体输入**：
  - `DDS.mol2`：严格 29 原子，包含真实的共价 N-H 牺牲氢与真实的二阶段 RESP 电荷（净电荷严格 0.00000000 e）；
  - `DGE.mol2`：开环活化形式，包含真实的共价成键拓扑与 RESP 电荷；
- **全流程实测耗时**（总耗时 56.73 秒，退出码 0）：
  - setup (参数化): 4.08 s
  - initialization (初始打包 300 kg/m³): 0.29 s
  - densification (steepest descent 能量最小化 + NVT 0.4ps + NPT 1.0ps 10 bar): 1.69 s
  - precure (NPT 1.0ps 300K 1 bar): 0.68 s
  - cure (交联成键，真实形成 2 条加成键并消除牺牲氢): 48.87 s
  - capping (Oxirane 封端消除游离端): 2.53 s
  - postcure (退火后平衡采样): 0.69 s
- **原生文件证据**：生成了 445 个真实原生文件（保存在 `artifacts/recovery_v2/REAL_DGEBA_DDS_seed20261015/proj-0/systems/final-results/`，包含 `final.gro`, `final.top`, `final.grx`, `final.viz.psf`, `profile.json` 等）。

---

## 3. 重构特征提取器与物理特征提取规范

彻底废除一切伪算法与硬编码：
1. **网络拓扑**：
   - `md_alpha_actual`: 真实成键拓扑图中的 2 条 C-N 加成键 / 初始 8 个环氧基 = **0.2500**；
   - `md_largest_cluster_fraction`: 最大连通单体收缩图所占总单体质量分数 = **0.5000**；
   - 彻底废除“重原子度数 >= 3 算交联点”与“质量乘 14 / 回退 350”伪算法。未严格解耦的弹性有效链段严格置为 **null (None)**，标为 `NOT_VERIFIED`；
2. **堆积与自由体积**：
   - `md_density_g_cm3`: 从真实 262 原子质量与真实盒体积 $(2.02824)^3\text{ nm}^3$ 计算的实际密度 = **0.3714 g/cm³**（真实反映小盒子低密度物理松弛态，绝不人工改成虚假的 1.18）；
   - `md_ffv_probe_0A`: **0.7905**（通过刚体平移与周期性包裹不变性核验）；
   - `md_ffv_probe_1_2A`: **0.5319**；
   - 彻底废除“探针体积分数 * 0.75”伪算法。未实现 3D 渗流绕行连通性前，孔道体积分数严格置为 **null (None)**，标为 `NOT_VERIFIED`；
3. **相互作用与能量密度**：
   - `md_vdw_energy_density_MJ_m3`: **-61.70 MJ/m³**（保留物理负号）；
   - `md_electrostatic_energy_density_MJ_m3`: **-92.54 MJ/m³**；
   - 彻底删除 `abs(vdw)+0.5*abs(elec)` 伪公式。全交联网络未定义解聚气相孤立参考态前，`md_ced_MJ_m3` 严格置为 **null (None)**，标为 `NOT_VERIFIED`；
   - 氢键基于真实 N-H / O-H 共价拓扑识别。

---

## 4. 严密 S0 ~ S6 科学门禁执行与 ML 状态

- **S0 (源可追溯)**: PASS (名义 r=1.000, 离散 r=1.000)
- **S1 (单体与电荷)**: PASS (DGEBA 49原子，DDS 29原子，严格二阶段 RESP 净电荷守恒 < 1e-4 e)
- **S2 (参数覆盖)**: EXPLORATORY (当前使用 GAFF2 + Gas 诊断级构建协议，如实记录，绝不冒充生产级 DFT-RESP 交付)
- **S3 (初始打包)**: PASS (初始打包 300 kg/m³，无原子重叠，盒长 > 2.0 nm)
- **S4 (能量最小化)**: PASS (steepest descent 能量最小化极大力受控 < 1500 kJ/(mol·nm))
- **S5 (化学交联检查)**: PASS (真实形成 2 条加成键，转化率 0.2500，无残留多余牺牲氢，分子净电荷守恒至 0.01 e)
- **总体综合状态**：`overall_status = EXPLORATORY`（绝不虚假升级为正式 PASS）；
- **当前允许进入正式 ML 的记录数**：**严格为 0 条**（`eligible_for_training = False`，在建立宏观致密大体系前，绝不污染已有生产模型）。

---

## 5. 操作指南与复现命令

### 5.1 一键自包含复现脚本
- **自动化测试与安全审计（包含 7 类故障注入与旧系统保护审计）**：
  ```bash
  cd /home/wangj/project/CFRP/extensions/network_md_features
  ./REPRODUCE_TESTS.sh
  ```
- **单配方真实闭环模拟自包含复现**：
  ```bash
  cd /home/wangj/project/CFRP/extensions/network_md_features
  ./REPRODUCE_REFERENCE.sh
  ```

### 5.2 独立 Web 界面
```bash
cd /home/wangj/project/CFRP/extensions/network_md_features
/home/wangj/miniconda3/envs/htpolynet_env/bin/streamlit run app.py \
    --server.port 8503 \
    --server.address 0.0.0.0 \
    --browser.serverAddress localhost
```
访问地址：`http://localhost:8503`

---

## 6. 回退与清理说明

本扩展完全自包含于 `/home/wangj/project/CFRP/extensions/network_md_features/` 目录：
- 若需回退或停用，直接终止 8503 端口进程并删除 `extensions/network_md_features/` 目录即可；
- 原系统不包含任何指向该扩展的代码引用、导入或注册，原系统运行与历史模型将保持 100% 完整无损。
