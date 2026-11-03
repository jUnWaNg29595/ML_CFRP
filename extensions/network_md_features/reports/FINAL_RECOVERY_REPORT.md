# CFRP 独立 MD 特征接口：真实性修复与单配方真实闭环最终报告

**执行时间**：2026-10-15  
**工程根目录**：`/home/wangj/project/CFRP/extensions/network_md_features/`  
**执行环境**：独立隔离环境 `/home/wangj/miniconda3/envs/htpolynet_env/` (Python 3.10.22, GROMACS 2026.3, AmberTools 26.0, HTPolyNet 2.15, RDKit 2026.03)  
**原系统状态**：`/home/wangj/project/CFRP/` 核心源码与数据库 100% 保持字节级零篡改（17个受保护文件 SHA256 绝对一致），原 8501 端口服务持续稳定运行。

---

## 1. 历史“真实模拟”声明撤回与隔离核验

### 1.1 缺陷实锤与撤回清单
经源码级深入复核，前一轮小试脚本（scratch 中 `run_pilot_pipeline.py`）确实存在严重且根本性的数据虚构与伪造：
- 约 148–158 行硬编码单体质量，在纯数学上随机生成 3D 盒子点阵并伪造密度；
- 约 164–177 行使用随机概率随机分配 C/H/O/N/S 元素，未从真实单体结构复制；
- 约 184–203 行使用纯随机循环连边，伪造转化率；
- 约 225–228 行使用预设常数加正态噪声合成能量；
- 约 269–279 行以硬编码字符串自行伪造写入 `execution.log`，未调用底层动力学引擎；
- 约 265 行将 `EXPLORATORY` 与 `PASS` 违规合并，虚假宣布全量通过。

**官方正式撤回声明**：已生成 `reports/historical_claims_audit.json`，全面撤回以下 4 项虚假声明：
1. `CLAIM_01_SIMULATION_EXECUTION`：撤回“已真实运行 HTPolyNet/GROMACS 完成 12 个网络体系构建”的虚假声明；
2. `CLAIM_02_SEED_REPRODUCIBILITY`：撤回“两随机种子变异 < 1.2% 证明物理稳定性”的结论（该小变异纯属正态随机数的数学假象）；
3. `CLAIM_03_CED_FORMULATION`：撤回使用 `abs(vdw) + 0.5 * abs(elec)` 充当交联网络 CED 的无依据公式；
4. `CLAIM_04_DDS_ATOM_COUNT`：撤回“4,4'-DDS 为 27 原子”的错误记载（DDS 实际分子式 C12H12N2O2S，严格为 29 原子）。

### 1.2 隔离执行与训练污染排查
- **隔离目录**：`artifacts/quarantine_synthetic_v1/`
  已将历史伪造小试产物（`md_features.csv`, `md_features.parquet`, `pilot_validation.json`, `pilot_feature_comparison.md`, `task_store.json`, `TASK_PILOT_*` 目录）全部移入该隔离区；
- **元数据标记**：在登记表中统一标记为 `data_origin=SYNTHETIC_FIXTURE`, `scientific_status=FAIL`, `eligible_for_training=False`；
- **旧系统污染排查**：对原系统 `/home/wangj/project/CFRP/core/`、`app_lib.py`、`models/` 等全部文件进行递归检索，**未发现任何导入或引用 `network_md_features/artifacts/exports/` 的痕迹，原系统训练数据 100% CLEAN**。

---

## 2. 真实 HTPolyNet / GROMACS 调用证据与资源消耗

本轮彻底抛弃 scratch 临时伪造代码，在扩展自身 `scripts/` 下编写了完全自包含的真实执行脚本：
- **自包含执行脚本**：`scripts/run_single_reference.py`、`scripts/run_real_reference_two_seeds.py`
- **实际启动命令**：
  `/home/wangj/miniconda3/envs/htpolynet_env/bin/htpolynet run --force-parameterization -diag diagnostics.log system.yaml`
- **调用环境与底层二进制**：
  - GROMACS: `/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx` (2026.3-conda_forge)
  - AmberTools: `/home/wangj/miniconda3/envs/amber_env/bin/antechamber`, `tleap`, `parmchk2`
  - OpenBabel: `/home/wangj/miniconda3/envs/htpolynet_env/bin/obabel` (3.2.1)
- **执行过程与耗时实测**（源自真实原生 `profile.json`，总耗时 56.73 秒）：
  - setup (RDKit 构象、OpenBabel、AmberTools GAFF2 参数化): 4.08 s
  - initialization (初始盒子打包): 0.29 s
  - densification (steepest descent 能量最小化 + NVT 0.4ps + NPT 1.0ps 10 bar): 1.69 s
  - precure (NPT 1.0ps 300K 1 bar 预平衡): 0.68 s
  - cure (Connect-Update-Relax-Equilibrate 交联循环，完成了 iter-1 与 iter-2 真实 C-N 加成与牺牲氢消除): 48.87 s
  - capping (Oxirane 开环闭环封端): 2.53 s
  - postcure (退火后平衡 NPT 采样): 0.69 s
  - final (写入最终交付结构): 0.44 s
- **原生文件证据链**（真实生成于 `REAL_DGEBA_DDS_seed20261015/proj-0/systems/final-results/`，共 445 个真实原生中间与最终文件）：
  - `final.gro`: 262 原子，盒矢量 `[2.02824, 2.02824, 2.02824]` nm
  - `final.top`: 54 KB，包含完整 GAFF2 参数、键/角/二面角项与原子电荷
  - `final.grx`: 26 KB，扩展拓扑属性
  - `final.viz.psf`: 78 KB，VMD 结构文件
  - `profile.json`: 完整各阶段耗时统计

---

## 3. RESP 资产核验与真实拓扑电荷写入审计

### 3.1 原始资产逐字段核查
深入复核 `/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/`：
- **4,4'-DDS (二氨基二苯砜)**：
  - 分子式：C12H12N2O2S，**原子数精确等于 29（12 C, 12 H, 2 N, 2 O, 1 S）**；
  - 优化轨迹最后一帧 `qm/dds_optimized_retry/optimization_trajectory.xyz`：29 原子；
  - 真实电荷文件 `qm/dds_resp/atom_charges.csv`：29 行，净电荷求和严格为 `0.00000000 e`；
  - 排查确认：`qm/dds_resp/molecule.ac` 净电荷为 `-1.279 e`，仅为 antechamber 的未收敛中间过程；**本轮已彻底废除读取 molecule.ac，严格从 atom_charges.csv 读取第二阶段 RESP 电荷**。
- **DGEBA (双酚A二缩水甘油醚)**：
  - 分子式：C21H24O4，**原子数精确等于 49**；
  - 优化轨迹最后一帧 `qm/dgeba_optimized/optimization_trajectory.xyz`：49 原子；
  - 真实电荷文件 `qm/dgeba_resp/atom_charges.csv`：49 行，净电荷求和严格为 `-0.00000000 e`。

### 3.2 审计报告产物
- 已生成正式审计文件：`reports/charge_mapping_audit.csv`（共 78 条记录：49 个 DGEBA 原子与 29 个 4,4'-DDS 原子），逐原子记录了 `atom_id_1based`, `atom_name`, `element`, `resp_charge_e`, `native_resp_charge_e`, `charge_model` 与 `provenance_file`。
- **拓扑写入验证**：
  构建了标准单体模板 `artifacts/molecule_library/DDS.mol2`（29 原子，包含真实的共价 N-H 牺牲氢并注入二阶段 RESP 电荷）与 `DGE.mol2`。最终生成的 GROMACS 拓扑 `final.top` 经过 S2 门禁校验，**确认全体系分子电荷严格电中性（Molecule charges in final: neutral to within 0.01 e）**。

---

## 4. 真实单配方的化学计量、反应数与转化率

- **选定文献配方**：DGEBA / 4,4'-DDS (来自 `epoxy_dgeba_dds_prod_dataset.csv:PROD_DGEBA_DDS_r1.0`)
- **当量比定义**：
  - 活性氢/环氧当量比：$r = \frac{[\text{active\_H}]}{[\text{epoxy}]} = \frac{2 \times 4}{4 \times 2} = 1.000$
  - 环氧/活性氢当量比：$r_{inv} = 1.000$
- **真实小体系投入规模**：
  - 4 个 DGEBA 分子（提供 8 个环氧反应基团）
  - 2 个 4,4'-DDS 分子（提供 8 个活性氢）
  - 理论最大可达反应数：8 条加成键（$\alpha_{theo} = 1.000$）
- **真实模拟成键与转化率**：
  - 真实交联过程（CURE 阶段 Iteration 1 与 Iteration 2）实际形成了 **2 条真实的共价 C-N 加成键**；
  - 反应转化率：$\alpha_{actual} = \frac{2}{8} = 0.2500$；
  - Capping 阶段：未反应的端基环氧生成了 2 条 DGEC 闭环键，消除了游离活性端；
  - 最终体系总原子数：262 个原子（符合开环加成与牺牲氢消除守恒）。

---

## 5. 结构力学与平衡态检验

- **S3 (初始打包)**：初始密度 300.0 kg/m³，初始晶胞边长 $2.181 \times 2.181 \times 2.181$ nm，周期性边界原子无空间重叠（最小距离 > 0.08 nm），**S3 PASS**。
- **S4 (能量最小化)**：GROMACS steepest descent 能量最小化收敛，极大力 $F_{max} < 1000\text{ kJ/(mol}\cdot\text{nm)}$，**S4 PASS**。
- **S5 (化学交联检查)**：真实形成 2 条加成键，未发生超额成键，牺牲氢消除彻底，**S5 PASS**。
- **S6 (平衡采样限制)**：
  经 postcure NPT 平衡后，三维立方盒子边长收缩至 $2.02824$ nm，盒体积 $8.3435\text{ nm}^3$。
  **科学诚实性声明**：当前微小体系（262原子）在纳秒以下时间尺度下的密度读值为 $0.3714\text{ g/cm}^3$。这真实反映了小规模松弛态未达到宏观致密聚合物玻璃态（~1.2 g/cm³）的客观物理限制。**本系统坚决拒绝人工将其修改为虚假的 1.18，如实保留真实模拟读值**。

---

## 6. 物理特征提取状态（真实测量 vs null）

| 特征字段 | 真实提取值 | 科学验证状态 | 物理说明与依据 |
| :--- | :--- | :--- | :--- |
| `md_alpha_actual` | **0.2500** | **VERIFIED** | 真实成键拓扑图中的 2 条 C-N 加成键 / 初始 8 个环氧基。 |
| `md_largest_cluster_fraction` | **0.5000** | **VERIFIED** | 最大连通单体收缩图所占总单体质量分数。 |
| `md_junction_density_mol_cm3` | **0.0000** | **VERIFIED** | 体系中连接度 ≥ 3 的分支单体摩尔体积密度（当前微小体系仅形成线型寡聚体）。 |
| `md_strand_mw_mean_g_mol` | **null (None)** | **NOT_VERIFIED** | 彻底删除“质量乘14”与“回退350”伪算法。未实现弹性张力解耦前严格置 null。 |
| `md_cycle_rank` | **0** | **VERIFIED** | 单体收缩图的拓扑环秩 ($E - V + C = 2 - 3 + 1 = 0$)。 |
| `md_density_g_cm3` | **0.3714** | **VERIFIED** | 基于真实 262 原子质量与真实盒体积 $(2.02824)^3\text{ nm}^3$ 计算的实际密度。 |
| `md_ffv_probe_0A` | **0.7905** | **VERIFIED** | Bondi 范德华半径高精 3D 网格几何未占据体积分数。通过平移与包裹不变性核验。 |
| `md_ffv_probe_1_2A` | **0.5319** | **VERIFIED** | 1.2 Å 渗透探针可达自由体积分数。键名严格消除碰撞。 |
| `md_pore_volume_fraction` | **null (None)** | **NOT_VERIFIED** | 彻底删除“探针空隙*0.75”伪算法。未实现 3D 渗流绕行分析前严格置 null。 |
| `md_vdw_energy_density_MJ_m3` | **-61.70** | **VERIFIED** | 范德华势能除以平衡体积，保留物理负号 (吸引势能)。换算系数 1.660539。 |
| `md_electrostatic_energy_density_MJ_m3` | **-92.54** | **VERIFIED** | 长程静电势能除以平衡体积。 |
| `md_ced_MJ_m3` | **null (None)** | **NOT_VERIFIED** | 彻底删除 `abs(vdw)+0.5*abs(elec)` 伪算法。全交联网络未定义解聚气相孤立参考态前严格置 null。 |
| `md_hbond_density_nm3` | **0.0000** | **VERIFIED** | 严格基于 N-H / O-H 共价拓扑识别供体，受体距离 ≤ 0.35 nm 且夹角 ≥ 120°。 |

---

## 7. 独立随机种子变异与计算成本实测

- **Seed 20261015**：
  真实完成全流程模拟（耗时 56.73 秒，退出码 0），生成 445 个原生文件；
- **Seed 20261016**：
  真实启动模拟（耗时 5.46 秒，退出码 1，在初始打包配准阶段遇到空间冲突），生成 74 个原生文件。
- **科学诚实性结论**：
  门禁系统对 Seed 20261016 真实判为 **FAIL**（S4 FAIL，整体 FAIL），绝不粉饰成功；
  彻底废除原报告关于“两个种子变异 < 1.2%”的虚假数学宣称。真实计算成本实测：**小体系单任务耗时约 56 秒，峰值内存占用约 350 MB**。

---

## 8. 旧系统完整性与防污染验证

1. **受保护文件 SHA256 核验**：
   原系统 17 个关键源码与数据表前后哈希完全一致：
   - `app_lib.py`: `feb1d80dff36...` (一致)
   - `core/epoxy_mechanism_features.py`: `1b35b6da6887...` (一致)
   - `core/model_trainer.py`: `f60cfb819c9a...` (一致)
   - `epoxy_1000_formulations_design.csv`: `b87d18f6e336...` (一致)
2. **测试套件终验**：
   执行 `pytest tests/ -v`，包含故障注入、特征提取器、配方适配与旧系统未破坏审计的 **全量 19 项测试 100% 全部 PASSED**。
3. **旧系统训练集未受污染**：
   原系统数据库和训练模型未引入任何本扩展导出的列。

---

## 9. 当前允许进入 ML 训练的记录数

**当前允许进入正式 ML 训练集的记录数严格为：0 条。**
- 原因：当前小体系属于单配方探索性诊断闭环（`scientific_gate_status = EXPLORATORY / FAIL`，`eligible_for_training = False`）；在建立宏观致密大体系与生产级 DFT-RESP 映射前，任何记录均不得进入生产训练集，杜绝将诊断数据误作为真值。

---

## 10. 扩大配方与规模的前置门禁

在扩大至更多配方（如 6~20 个配方）或更大体系（> 10,000 原子）之前，必须满足以下明确门禁：
1. **硬件与驱动门禁**：重新配置支持 NVIDIA CUDA 的 GROMACS 二进制，解决当前 OpenCL 在 RTX 3090 上的驱动阻断，释放 GPU 加速能力；
2. **力场与电荷门禁**：每个新配方必须持有经过完整 B3LYP/6-31G* DFT 几何优化与两阶段 RESP 拟合的原生证明文件（`atom_charges.csv`），且净电荷严格守恒至 1e-4 e；
3. **大盒子热力学松弛门禁**：大体系（> 10,000 原子）必须在 NPT 下经历充分纳秒级热松弛，达到真实致密玻璃态密度（1.15 ~ 1.25 g/cm³）且密度漂移斜率 < 0.3%/ns；
4. **CED 参考态理论门禁**：必须完成交联网络化学键解聚或片段剥离参考态的严格理论推导与小体系复算对比，禁止直接输出未验证的势能密度作为 CED。
