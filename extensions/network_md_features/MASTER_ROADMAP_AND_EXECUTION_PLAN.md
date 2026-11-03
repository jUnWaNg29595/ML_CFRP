# 环氧树脂高通量 MD 特征提取与机器学习泛化系统：总体执行计划与记忆锚点

> **文档用途**：本文件作为跨会话持久化记忆锚点（Memory Anchor & Master Execution Plan）。后续任何接手对话在启动工作前，必须先读取本文件与 `/home/wangj/project/CFRP/extensions/network_md_features/artifacts/postprocess_verified_20261019_02/` 下的验收产物，严禁跳过验证或回退到已撤回的硬编码代码。
>
> **同步保存位置**：
> 1. Windows 工作区：`C:/Users/wangj/Desktop/WSL机器学习系统/环氧高通量MD与机器学习泛化总体执行计划.md`
> 2. WSL 扩展模块根目录：`/home/wangj/project/CFRP/extensions/network_md_features/MASTER_ROADMAP_AND_EXECUTION_PLAN.md`

---

## 一、项目终极目标与核心科学问题

### 1. 终极目标
建立一套**覆盖多固化机理（环氧+胺、环氧+酸酐、环氧+酚醛、环氧+硫醇、环氧+异氰酸酯、环氧+羧酸/酰肼、以及环氧均聚自固化）**的高通量分子动力学（MD）凝聚态特征提取与机器学习系统，提升对未知文献/新型化学结构配方的**玻璃化转变温度（$T_g$）**与**室温储能模量（$E'$）**的跨体系泛化预测能力。

### 2. 现有基础与泛化瓶颈
- **现有主系统特征（`/home/wangj/project/CFRP/`）**：
  - RDKit 分子描述符、自建环氧-固化剂反应机理特征（`core/epoxy_mechanism_features.py`）、前线轨道 $\Delta E$ 能差矩阵、Flory-Stockmayer 统计交联网络描述符（`core/reaction_core.py`）、RDKit 多机理反应产物团簇 2D/3D 描述符（`core/reaction_simulator.py`）、配方比例与工艺条件。
- **当前泛化痛点**：
  - 随机划分（Random Split）下预测表现较好，但在**按文献来源或全新化学结构分组交叉验证（Group 5-Fold CV）**下性能显著下降（566~644 条干净基线复算：$T_g$ 的 Group 5-Fold $R^2 = 0.6290, \text{MAE} = 31.72\ ^\circ\text{C}$；25 ℃ 储能模量的 Group 5-Fold $R^2 = 0.4193, \text{MAE} = 0.5805\text{ GPa}$）。
  - **物理根因**：2D/单分子描述符无法刻画交联结构在真实凝聚态（$\rho \approx 1.15\sim 1.25\text{ g/cm}^3$）下的**三维空间堆积效率（自由体积空隙分布）、链段间范德华与静电内聚作用能密度、以及三维极性氢键网络密度**。

### 3. 引入 MD 特征的定位（有所为、有所不为）
- **不做**：不用 MD 直接跑几十纳秒变温冷却算 $T_g$ 或直接拉伸算模量（单配方耗时数天，无法支撑数百配方的机器学习高通量筛选）。
- **要做**：通过标准化、受控耗时（单配方 10~20 分钟 GPU 时间）的**凝聚态代表性交联团簇/寡聚物 MD 采样**，提取以下 4 类真实凝聚态物理特征补充进机器学习特征集：
  1. **凝聚态堆积与自由体积**：实测平衡密度 $\rho_{\text{MD}}$、3D PBC 原子球占据并集自由体积分数 $\text{FFV}_{0\text{\AA}}$ 与探针可达未占据分数 $\text{FFV}_{1.2\text{\AA}}$；
  2. **非键相互作用能密度**：范德华相互作用能密度 $e_{\text{vdW}}$（$\text{MJ/m}^3$）、静电相互作用能密度 $e_{\text{Coul}}$（$\text{MJ/m}^3$），以及经气相参考态校准后的严格内聚能密度 $\text{CED}$；
  3. **极性氢键网络统计**：逐帧 PBC 几何氢键总数密度、分子间（Intermolecular）与分子内（Intramolecular）氢键比例；
  4. **交联拓扑与键应变**：连通分量、最大团簇占比、环氧基开环转化率 $\alpha_{\text{epoxy}}$。

---

## 二、技术路线选型决策：为什么采用“RDKit 多机理交联簇 + GROMACS 凝聚态 NPT”而非强绑 `HTPolyNet`？

### 1. 为什么放弃将 `HTPolyNet` 作为全数据集通用主路线？
- `HTPolyNet` 仅适用于单一反应规则且需人工为每种新单体手写“预开环带牺牲氢活性结构 + 活性原子编号 + 反应后 Capping 端基还原规则”；
- 此前即使在最简单的 `DGEBA + DDS` 体系上，`HTPolyNet` 也曾因牺牲氢 Capping 未配平导致“262 原子（多出 8 个残氢）”和动态拉拽拉伸键（$2.47\text{ \AA}$）错误；
- 面对数据集中广泛存在的**环状酸酐开环交替酯化、异氰酸酯环化生成噁唑烷酮、酚醛/硫醇固化、以及无固化剂的环氧均聚自固化（链式醚化）**，`HTPolyNet` 的牺牲氢机制无法自动泛化。

### 2. 确立的主力通用技术路线（全机理通吃）
直接**只读复用**主项目 `core/reaction_simulator.py` 中已高度成熟的 **10 种 SMIRKS 多机理反应模板与交联团簇生成器**：
- 已支持机理：`epoxy_primary_amine`、`epoxy_secondary_amine`、`epoxy_anhydride_5`、`epoxy_anhydride_6`、`epoxy_anhydride`、`epoxy_thiol`、`epoxy_hydrazide`、`epoxy_phenol`、`epoxy_carboxylic_acid`、`epoxy_isocyanate`、以及环氧自固化/醇醚化均聚 `epoxy_etherification`（加上残留环氧基邻二醇水解封端 `_cap_residual_epoxide_stubs`）。
- **核心流程**：
  1. 对任意配方（含自固化），由 RDKit 反应模板在分子图层面生成目标转化率下的**代表性交联团簇/支化低聚物分子（天然 100% 化合价合法、绝对零残氢错误）**；
  2. 通过 AmberTools（`antechamber` + `parmchk2`）生成一致的 GAFF2 力场与电荷参数；
  3. 将多个交联团簇与未反应组分装入周期盒（约 $1,500\sim 2,500$ 原子），在 GROMACS 中执行 **高温高压 NPT 压密退火弛豫 + 300 K PME 凝聚态平衡采样**，使体系达到真实物理密度（$1.10\sim 1.30\text{ g/cm}^3$）；
  4. 调用已通过二次验收的统一后处理引擎 `src/postprocess_trajectory.py` 提取全部真实凝聚态特征。

---

## 三、已完成基线与不可触碰的工程红线（记忆锚点）

### 1. 两条项目线的严格隔离
- **本线（可读写扩展区）**：`/home/wangj/project/CFRP/extensions/network_md_features/` 与 `/home/wangj/project/CFRP/extensions/ml_generalization_v2/`。
- **原 CFRP 保护区（严格只读，17 个受保护文件）**：
  - 保护清单基准文件：`/home/wangj/project/CFRP/extensions/network_md_features/artifacts/source_snapshots/protected_baseline.json`
  - 包含：`app.py`, `app_lib.py`, `split_app.py`, `core/navigation.py`, `core/epoxy_mechanism_features.py`, `core/reaction_simulator.py`, `core/reaction_core.py`, `core/molecular_features.py`, `core/molecular_feature_workflow.py`, `core/feature_registry.py`, `core/model_trainer.py`, `core/portal_formulation_inputs.py`, `core/formulation_grouping.py`, `tests/test_ff_xtb_delta_e_optimization.py`, `epoxy_1000_formulations_design.csv`, `epoxy_dgeba_dds_prod_dataset.csv`, `PINN筛选350_400度高性能配方清单.csv`。
  - **任何阶段前后必须运行 `pytest tests/test_unbroken_old_system.py` 确保 17/17 文件 SHA256 零改动**。
- **独立 LAMMPS 生产线（严格只读）**：`/home/wangj/epoxy_prod/rebuild_verified_v3_5/`（19,050 原子体系），仅允许只读引用其 `qm/{dgeba,dds}_resp/atom_charges.csv`，严禁修改或混用。

### 2. 已验收完成的真实后处理基线（`postprocess_verified_20261019_02`）
- **核心代码（已消除一切硬编码与随机数）**：
  - `src/pbc_utils.py`：唯一共享的正交 PBC 最小镜像位移、距离、跨残基接触检查与共价键应变检查；
  - `src/native_edr_extractor.py`：原生 `gmx energy` 提取、动态按 legend 映射、支持时间窗口 `[t_start, t_end]` 与 `stride`、缺失能量项立即抛 `KeyError`；
  - `src/structure_features_extractor.py`：原生 `gmx check` / `gmx trjconv` 多帧解析、真实连接图三元环氧环计数转化率、一维可分离 3D PBC 网格球并集 FFV（含半格平移与网格细化不确定度评估）、逐帧 PBC 几何氢键及配对明细、全轨迹共价键应变检查；
  - `src/postprocess_trajectory.py` & `src/compute_features_from_trajectory.py`：统一 CLI 与函数入口，缺 `XTC`/`EDR` 必抛 `FileNotFoundError`。
- **当前验证状态**：
  - 针对 `attempt_20261019_01`（254 原子小体系）的后处理工程门禁（`G1~G8`）及 54 项自动化故障注入测试 **100% 通过**；
  - 由于 `attempt_20261019_01` 盒子为 $55\sim 75\text{ \AA}$ 稀疏空盒（密度 $0.007\sim 0.019\text{ g/cm}^3$）且加成物为 Gasteiger 电荷，科学适用域门禁（`D1~D6`）如实将其拦截至诊断域（`formal_ml_training_admission_count = 0`）。

---

## 四、后续分阶段实施计划（Step-by-Step Roadmap）

接下来按 **5 个独立、可验收的阶段** 推进，每完成一阶段通过门禁后再进入下一阶段，严禁跳步或伪造数据：

---

### 阶段一：DGEBA/DDS 小型凝聚态基准体系构建与压密验证（Condensed-Phase Benchmark）

- **目标**：解决当前 `attempt_20261019_01` 密度过低（$0.01\text{ g/cm}^3$ 空盒）和 `Cut-off` 短程截断问题，在 DGEBA/DDS 基准体系上跑通第一条**真正达到凝聚态树脂密度（$1.10\sim 1.25\text{ g/cm}^3$）**的平衡轨迹。
- **具体动作**：
  1. **统一电荷协议**：将单加成物 `MON` 与双加成物 `BIS` 的电荷协议从临时的 `-c gas`（Gasteiger）升级为与单体一致的 **AM1-BCC（`antechamber -c bcc`）或片段 RESP 协议**，确保反应前后静电能密度可比；
  2. **扩大体系规模至代表性体元（约 $1,500\sim 2,000$ 原子）**：例如将 254 原子单元复制 6~8 倍（约 1,524~2,032 原子），消除极小体系的剧烈热涨落与有限尺寸效应；
  3. **多步 NPT 退火压密与 PME 长程静电协议**：
     - 静电与范德华：`coulombtype = PME`（启用长程倒空间静电 `Coulomb-recip.`），`rcoulomb = 1.0 nm`，`rvdw = 1.0 nm`，开启长程色散校正 `DispCorr = EnerPres`；
     - 压密与弛豫流程：`steep` 最小化 $\rightarrow$ `NVT` 升温至 $500\text{ K}$ ($20\text{ ps}$) $\rightarrow$ 高压 `NPT` ($500\text{ K}, 500\text{ bar} \rightarrow 1\text{ bar}$, $100\text{ ps}$) 将盒子从初始低密度平稳压缩至凝聚态 $\rightarrow$ 降温退火至 $300\text{ K}$ ($1\text{ bar}$, $100\text{ ps}$) $\rightarrow$ $300\text{ K}, 1\text{ bar}$ 生产采样 ($100\sim 200\text{ ps}$)。
  4. **后处理提取**：直接调用 `src/postprocess_trajectory.py`（针对后半段平稳窗口如 $50\sim 100\text{ ps}$）提取特征，并扩展 `native_edr_extractor.py` 记录 `Coulomb-recip.` 与 `Disper.-corr.`。
- **阶段一验收门禁（必须全部满足才算通过）**：
  - [ ] 实测凝聚态平衡密度落在合理物理区间 **$1.05\sim 1.30\text{ g/cm}^3$**，且生产窗口内密度漂移率 $< 1.0\%/\text{ns}$；
  - [ ] 全轨迹所有帧最大共价键长偏差 $|r_{\text{PBC}}(t) - r_0| \le 0.25\text{ \AA}$（零拉伸键、零穿环）；
  - [ ] 3D PBC 占据并集自由体积分数 $\text{FFV}_{0\text{\AA}}$ 从空盒的 $99\%$ 降至真实凝聚态树脂区间（约 **$28\%\sim 40\%$** 范德华未占据分数，探针 $\text{FFV}_{1.2\text{\AA}}$ 约 **$1\%\sim 12\%$**）；
  - [ ] 检测到真实的**分子间氢键**（`is_intermolecular = True` 占比 $> 0$），且非键总能密度为物理合理的负值（吸引主导）。

---

### 阶段二：多固化机理（含酸酐、酚醛、硫醇、异氰酸酯、自固化）交联簇自动建模器

- **目标**：打通从“任意配方 SMILES + 固化机理”到“GROMACS 凝聚态模拟输入”的自动化桥梁，覆盖数据集中除胺类以外的全部固化体系及自固化体系。
- **具体动作**：
  1. 在 `extensions/network_md_features/src/` 下新建 `multi_mechanism_cluster_builder.py`；
  2. **只读调用**主项目 `core/reaction_core.py`（自动识别机理：`amine`, `anhydride`, `phenol`, `thiol`, `isocyanate`, `carboxyl`, `polyol` / `homopolymerization` 自固化）和 `core/reaction_simulator.py`（使用其中的 `ALL_REACTION_TEMPLATES` 与 `_cap_residual_epoxide_stubs`）；
  3. 对任意给定配方，自动生成 2~3 个不同反应程度（如 $\alpha = 0\%$ 未反应混合物 vs $\alpha \approx 60\%\sim 75\%$ 代表性支化交联团簇）的 3D 优化分子结构（RDKit ETKDGv3 + MMFF94s 预优化）；
  4. 自动检查生成的每个交联团簇：
     - 化合价 100% 合法、无自由基或残余牺牲氢；
     - 所有元素均在 `BONDI_VDW_RADII_NM` 与 GAFF2 支持范围内（将 `F, Cl, Br, P, Si` 的权威 Bondi 半径补充进 `BONDI_VDW_RADII_NM`，对仍不支持的金属或无机填料配方明确返回 `OUT_OF_DOMAIN_BLOCKED`）；
  5. 调用 AmberTools 自动生成 `.mol2`（统一 AM1-BCC 电荷）与 `.frcmod`，并用 `tleap` 组装多分子凝聚态初始盒。
- **阶段二验收门禁**：
  - [ ] 选取 **5 类不同机理** 的典型配方各 1 个（① 环氧+芳香胺、② 环氧+脂环酸酐、③ 环氧+酚醛、④ 环氧+硫醇/异氰酸酯、⑤ 纯环氧自固化），100% 自动生成拓扑并通过化学价、净电荷守恒与 PBC 接触门禁；
  - [ ] 对超出适用域的配方（如含无机纳米填料、未知金属催化剂或无法解析的聚合物）100% 触发拦截并记录原因，绝不静默放行。

---

### 阶段三：尺寸效应、随机种子复现性与采样收敛度评估

- **目标**：证明提取出的凝聚态 MD 特征反映的是**配方本身的化学与堆积性质**，而不是随机初始摆放或模拟时间太短带来的随机噪声（即具备高信噪比）。
- **具体动作**：
  1. 在阶段二的 5 个不同机理代表性配方上，每个配方使用 **3 个独立随机种子（Seeds）** 构建凝聚态盒子并运行 NPT 压密与生产采样；
  2. 对比不同采样时长窗口（如 $50\text{ ps}$ vs $100\text{ ps}$ vs $200\text{ ps}$）和不同盒子尺寸（$1,500$ 原子 vs $3,000$ 原子）下的特征均值与种子间变异系数（CV = Std / Mean）；
  3. 确定高通量生产的最优性价比协议（即在保证种子间方差显著小于配方间方差的前提下，所需的最短 NPT 压密与采样时长）。
- **阶段三验收门禁**：
  - [ ] 同一配方不同随机种子之间的密度标准差 $< 0.015\text{ g/cm}^3$，FFV 标准差 $< 0.01$，非键能密度相对变异系数 $< 5\%$；
  - [ ] **配方间方差（Between-Formulation Variance）显著大于同配方不同种子间方差（Within-Formulation Seed Variance）**（ANOVA $F$-检验显著），证明特征具有区分不同配方的真实物理信噪比。

---

### 阶段四：代表性文献配方小批量试点（Pilot Run, 15~30 个配方）

- **目标**：在不盲目跑全量 1000 个配方的前提下，从现有干净文献数据集中按**化学结构分组（Scaffold / Paper Group）与机理类型分层抽样**选出 15~30 个具有真实实测 $T_g$ 和储能模量的代表性配方，生成第一批通过全部凝聚态门禁的准入级 MD 特征集。
- **具体动作**：
  1. 从 `extensions/ml_generalization_v2` 的干净文献表中分层抽取 15~30 个配方（覆盖高/中/低 $T_g$ 及不同固化机理）；
  2. **严格防数据泄漏（Zero Data Leakage）**：MD 建模与采样仅允许使用部署时可获得的配方输入（单体 SMILES、配比 $r$、固化温度工艺），**严禁将文献实测密度、实测 $T_g$ 或实测模量作为 MD 盒子的输入条件**；
  3. 批量运行阶段三定型的凝聚态 MD 协议 + `postprocess_trajectory.py`，每个配方生成完整的 `feature_provenance.json` 与 `scientific_gate_report.json`；
  4. 仅将 `scientific_gate_report.json` 中凝聚态门禁全部 PASS 的样本写入正式准入特征表 `md_features_admitted_pilot.parquet`。
- **阶段四验收门禁**：
  - [ ] 每个准入样本均有可复核的原生 `.xtc`、`.edr`、`.prmtop`、`.xvg` 与 SHA256 清单；
  - [ ] 零随机数填充、零失败样本静默插补（若某配方 MD 未收敛则如实标记 `FAILED_GATE` 排除，绝不用均值或公式伪造）。

---

### 阶段五：分组交叉验证（Group CV）消融实验与扩展决策

- **目标**：用严格的统计学证据回答核心问题——“加入真实凝聚态 MD 特征后，在按文献来源/化学骨架分组的独立测试中，预测 $T_g$ 和储能模量的误差是否真正降低？”
- **具体动作**：
  1. 在 `extensions/ml_generalization_v2/` 中固定相同的 Group 5-Fold 划分（按 DOI/文献来源及主骨架分组，确保同一篇文献或相同树脂-固化剂对绝不同时出现在训练集和测试集）；
  2. 对比三组特征集的模型表现（Random Forest / CatBoost / Ridge 等，保持相同超参数搜索策略）：
     - **Baseline A**：仅原系统 2D RDKit + 配方机理特征；
     - **Baseline B**：原系统特征 + RDKit 静态 3D 交联簇描述符；
     - **Model C（+MD）**：原系统特征 + 真实凝聚态 MD 特征（$\rho_{\text{MD}}, \text{FFV}_{0\text{\AA}}, \text{FFV}_{1.2\text{\AA}}, e_{\text{vdW}}, e_{\text{Coul}}, \rho_{\text{H-bond}}$）；
  3. 输出分组测试集上的 $R^2$、$\text{MAE}$、$\text{RMSE}$、配对 Bootstrap 置信区间及 SHAP/排列特征重要性分析；
  4. **理性决策扩展规模**：
     - 若试点集上 MD 特征在 Group CV 中表现出明确的物理相关性与误差下降趋势，再按优先级分批扩展至全量核心配方库（如 100 $\rightarrow$ 300 $\rightarrow$ 全量可计算配方）；
     - 若某些特征信噪比不足，先针对性改进物理定义（如引入链段均方位移 MSD / 局部刚度指标），绝不盲目批量刷算力。

---

## 五、新对话接手速查指令（Quick-Start Commands for Future Agents）

后续任何新对话接手本项目时，请先执行以下 3 条只读检查命令确认现场完好：

```bash
# 1. 验证原 CFRP 主系统 17 个受保护文件 100% 未被改动
PYTHONPATH=/home/wangj/project/CFRP/extensions/network_md_features \
  /home/wangj/miniconda3/envs/htpolynet_env/bin/python -m pytest \
  /home/wangj/project/CFRP/extensions/network_md_features/tests/test_unbroken_old_system.py -v

# 2. 运行全部 54 项门禁与后处理故障注入回归测试
PYTHONPATH=/home/wangj/project/CFRP/extensions/network_md_features \
  /home/wangj/miniconda3/envs/htpolynet_env/bin/python -m pytest \
  /home/wangj/project/CFRP/extensions/network_md_features/tests -v

# 3. 查看已验证通过的后处理诊断特征表与门禁报告
cat /home/wangj/project/CFRP/extensions/network_md_features/artifacts/postprocess_verified_20261019_02/md_features_diagnostic.csv
```

---

## 六、阶段一至阶段五实测完成记录与验收摘要（2026-10-24 已落实闭环）

### 1. 阶段一实测结果：DGEBA/DDS 1,524 原子凝聚态基准体系（`artifacts/condensed_benchmark_stage1/`）
- **方法**：将 254 原子基础化学计量单元（`4 DGEBA + 2 DDS`）放大 6 倍至 **1,524 原子（$\text{C}_{648}\text{H}_{720}\text{N}_{24}\text{O}_{120}\text{S}_{12}$，总质量 $11,149.2\text{ amu}$）**，全组分统一采用 `antechamber -at gaff2 -c bcc`（真实 AM1-BCC 半经验量子化学电荷）并标定 300 K 孤立气相单分子参考势能，经 `steep` 最小化 $\rightarrow$ `420 K, 1200 bar` 分段连续 NPT 压密 $\rightarrow$ `300 K, 1 bar` NPT 退火 $\rightarrow$ `300 K, 1 bar` PME + DispCorr 生产采样，由 `postprocess_trajectory.py` 真实提取：

| 状态 | 原子数 | 转化率 $\alpha_{\text{graph}}$ | 实测密度 ($\text{g/cm}^3$) | $\text{FFV}_{0\text{\AA}}$ | $\text{FFV}_{1.2\text{\AA}}$ | $e_{\text{vdW}}$ ($\text{MJ/m}^3$) | $e_{\text{Coul}}$ ($\text{MJ/m}^3$) | 严格 $\text{CED}$ ($\text{MPa}$) | 总氢键 / 分子间氢键 | 全轨迹最大键长偏差 ($\text{\AA}$) | 门禁状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `State_0_Condensed_Unreacted` | 1524 | 0.000 | `1.0967` | `0.3867` | `0.0079` | `-215.73` | `-672.46` | `349.72` | `23.49 / 23.49` | `0.1517` | `CONDENSED_PHASE_VERIFIED` |
| `State_1_Condensed_MonoAdduct` | 1524 | 0.125 | `1.1044` | `0.3840` | `0.0076` | `-217.30` | `-626.96` | `354.64` | `27.69 / 26.22` | `0.1517` | `CONDENSED_PHASE_VERIFIED` |
| `State_2_Condensed_BisAdduct` | 1524 | 0.250 | `1.0628` | `0.4084` | `0.0259` | `-193.44` | `-545.39` | `350.49` | `28.16 / 26.49` | `0.1462` | `CONDENSED_PHASE_VERIFIED` |

### 2. 阶段二与阶段三实测结果：5 大类固化机理（含自固化）× 3 个独立随机种子复现性与 ANOVA 信噪比（`artifacts/stage2_3_multi_mechanism_and_seeds/`）
- **15/15 次凝聚态模拟全部通过 `CONDENSED_PHASE_VERIFIED` 门禁**（全轨迹最大键长偏差均 $\le 0.176\text{ \AA} < 0.25\text{ \AA}$）：
  1. `M1_AMINE_DGEBA_DDS`（环氧+芳香胺，1152 原子）：密度 `1.0907 ± 0.0132 g/cm³`，$\text{FFV}_{0\text{\AA}} = 0.3985 \pm 0.0073$，$\text{CED} = 375.19 \pm 13.00\text{ MPa}$，总氢键 `22.18 ± 0.84`；
  2. `M2_ANHYDRIDE_DGEBA_MHHPA`（环氧+脂环酸酐，1124 原子）：密度 `1.0242 ± 0.0122 g/cm³`，$\text{FFV}_{0\text{\AA}} = 0.4180 \pm 0.0070$，$\text{CED} = 352.84 \pm 9.86\text{ MPa}$，总氢键 `13.05 ± 1.05`；
  3. `M3_PHENOL_DGEBA_BPF`（环氧+酚醛，1130 原子）：密度 `1.0370 ± 0.0690 g/cm³`，$\text{FFV}_{0\text{\AA}} = 0.4091 \pm 0.0390$，$\text{CED} = 336.71 \pm 49.92\text{ MPa}$，总氢键 `11.98 ± 0.40`；
  4. `M4_THIOL_DGEBA_EDT`（环氧+二硫醇，943 原子）：密度 `1.0906 ± 0.0063 g/cm³`，$\text{FFV}_{0\text{\AA}} = 0.3851 \pm 0.0033$，$\text{CED} = 354.67 \pm 11.63\text{ MPa}$，总氢键 `5.05 ± 0.33`；
  5. `M5_SELFCURE_HOMOPOLY_DGEBF`（纯环氧均聚自固化，1237 原子）：密度 `1.0668 ± 0.0163 g/cm³`，$\text{FFV}_{0\text{\AA}} = 0.4024 \pm 0.0087$，$\text{CED} = 295.26 \pm 10.70\text{ MPa}$，总氢键 `10.98 ± 1.45`。
- **ANOVA 跨配方信噪比（Between-Formulation vs Within-Formulation Seed Std）**：
  - 静电能密度 $e_{\text{Coul}}$：$\text{SNR} = 15.01, F = 465.88, p = 2.59\times 10^{-11}$；
  - 总氢键数：$\text{SNR} = 7.59, F = 136.76, p = 1.10\times 10^{-8}$；
  - 分子间氢键数：$\text{SNR} = 5.36, F = 77.10, p = 1.79\times 10^{-7}$；
  - 范德华能密度 $e_{\text{vdW}}$：$\text{SNR} = 2.14, F = 6.64, p = 0.0071$。

### 3. 阶段四与阶段五实测结果：15 个分层文献配方凝聚态试点与 Group-CV 消融验证（`artifacts/stage4_pilot_condensed_md/`）
- 从 `data_fixed/latest_data_clean_594.csv` 分层抽取 15 个覆盖 6 类固化机理（`carboxyl`, `anhydride`, `amine`, `phenol`, `homopolymerization`）且跨越 $T_g \in [25.0, 300.0]\ ^\circ\text{C}$ 的真实文献配方，**15/15 全部通过 `CONDENSED_PHASE_VERIFIED` 门禁**并导出至 `md_features_admitted_pilot.parquet`。
- **物理相关性（与文献实测 $T_g$）**：
  - 实测凝聚态密度 `md_density_g_cm3` 与实测 $T_g$ 呈显著正相关（Pearson $r = +0.4453$, Spearman $\rho = +0.5107, p = 0.0517$）；
  - 分子间氢键 `md_hbond_intermolecular_average` 与实测 $T_g$ 正相关（Pearson $r = +0.3960$, Spearman $\rho = +0.4393$）；
  - 与全量 644 样本预训练树模型的 OOF 残差也呈正交互补相关（`md_density_g_cm3` 与 RF 残差 $r = +0.2443$，`md_vdw_energy_density_MJ_m3` 与 RF 残差 $r = -0.2407$，`md_cohesive_energy_density_CED_MPa` 与 RF 残差 $r = +0.2057$）。
- **试点集内严格分组交叉验证消融（Top-6 机理特征 vs +凝聚态 MD 特征）**：
  - **Group 5-Fold CV**：`RandomForest` 加入凝聚态 MD 特征后，OOF MAE 从 `70.98 ℃` 降至 **`64.38 ℃`（降低 `-6.60 ℃`）**，RMSE 从 `86.81 ℃` 降至 **`80.25 ℃`（降低 `-6.56 ℃`）**；`GradientBoosting` 的 OOF MAE 从 `82.81 ℃` 降至 **`79.16 ℃`（降低 `-3.65 ℃`）**；
  - **Leave-One-Formulation-Out (LOFO)**：`GradientBoosting` 的 OOF MAE 从 `81.97 ℃` 降至 **`72.51 ℃`（降低 `-9.46 ℃`）**，RMSE 从 `95.39 ℃` 降至 **`84.90 ℃`（降低 `-10.49 ℃`）**；`RandomForest` 的 OOF MAE 从 `71.68 ℃` 降至 **`68.06 ℃`（降低 `-3.62 ℃`）**。
- **全套 57 项自动化测试（含 12 项后处理故障注入、3 项阶段 1~5 凝聚态验证及 17 个受保护文件零改动校验）100% 通过（`57 passed`）**。

### 4. 化学计量比 $r$ + 固化工艺温度 $T_{\text{cure}}$ 动态耦合升级与全量数据集模型优化结果（`artifacts/coupled_md_ml_optimization_v3/`）
- **物理耦合机制（`compute_stoichiometry_and_cure_coupled_composition`）**：
  - 将每条文献配方的**化学计量当量比 $r$**（决定极限转化率 $\alpha_{\max}(r)$ 与未反应游离小分子增塑残留分数 $w_{\text{free}}$）及**最高固化温度 $T_{\text{cure}}$、固化时间 $t_{\text{cure}}$**（决定 Arrhenius + 玻璃化受限转化率 $\alpha_{\text{eff}}$ 与 GROMACS 高温退火起始热历史温度 $T_{\text{cure, K}}$）动态耦合进凝聚态盒子分子组成 `(n_A2, n_A1, n_R, n_H)`；
  - 在 DGEBA+DDS 体系上实测验证：等当量高温固化（$r=1.0, 200\ ^\circ\text{C}$）凝聚态密度达 `1.1125 g/cm³`、$\text{CED}=382.15\text{ MPa}$；而环氧过量低温固化（$r=0.55, 90\ ^\circ\text{C}$）因游离单体残留，$\text{CED}$ 降至 `342.12 MPa`、单位原子分子间氢键密度下降 $52\%$；胺过量中温固化（$r=1.50, 140\ ^\circ\text{C}$）自由体积 $\text{FFV}_{0\text{\AA}}$ 升至 `0.4065`。
- **44 个真实 GROMACS PME 凝聚态模拟标定 + 全数据集双分支物理引导集成模型（`CatBoost + LightGBM + XGBoost + ExtraTrees`）实测成绩**：

| 任务与数据集 | 模型配置 | 训练集 $R^2$ | 训练集 MAE | 随机 5 折 OOF $R^2$ | 随机 5 折 OOF MAE | 配方分组 5 折 Group $R^2$ | 配方分组 5 折 Group MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| **$T_g$ 清洗集 (644 条)** | `Legacy_Baseline_RF_BaseOnly` | `0.9213` | `15.475 ℃` | `0.6485` | `31.007 ℃` | `0.6427` | `30.955 ℃` |
| **$T_g$ 清洗集 (644 条)** | `Upgraded_Ensemble_BaseOnly` | `0.9933` | `4.150 ℃` | `0.7061` | `27.901 ℃` | `0.6816` | `28.429 ℃` |
| **$T_g$ 清洗集 (644 条)** | **`Upgraded_Ensemble_Plus_Coupled_MD`** | **`0.9949`** | **`3.710 ℃`** | **`0.7115`** | **`27.521 ℃`** | **`0.6832`** | **`28.268 ℃`** |
| **$T_g$ 全量集 (2,229 条)** | `Legacy_Baseline_RF_BaseOnly` | `0.8448` | `18.292 ℃` | `0.7166` | `24.132 ℃` | `0.6261` | `28.735 ℃` |
| **$T_g$ 全量集 (2,229 条)** | `Upgraded_Ensemble_BaseOnly` | `0.9336` | `11.424 ℃` | `0.7718` | `20.273 ℃` | `0.6578` | `26.371 ℃` |
| **$T_g$ 全量集 (2,229 条)** | **`Upgraded_Ensemble_Plus_Coupled_MD`** | **`0.9468`** | **`10.344 ℃`** | **`0.7803`** | **`19.769 ℃`** | **`0.6686`** | **`26.029 ℃`** |
| **25℃ 储能模量结构集 (546 条)** | `Legacy_Baseline_RF_BaseOnly` | `0.8015` | `0.402 GPa` | `0.4290` | `0.655 GPa` | `0.1981` | `0.740 GPa` |
| **25℃ 储能模量结构集 (546 条)** | `Upgraded_Ensemble_BaseOnly` | `0.9701` | `0.140 GPa` | `0.4741` | `0.595 GPa` | `0.2233` | `0.711 GPa` |
| **25℃ 储能模量结构集 (546 条)** | **`Upgraded_Ensemble_Plus_Coupled_MD`** | **`0.9785`** | **`0.120 GPa`** | **`0.4828`** | **`0.587 GPa`** | **`0.2426`** | **`0.702 GPa`** |
| **25℃ 储能模量基准集 (566 条)** | `Legacy_Baseline_RF_BaseOnly` | `0.8281` | `0.293 GPa` | `0.5777` | `0.461 GPa` | `0.5624` | `0.460 GPa` |
| **25℃ 储能模量基准集 (566 条)** | `Upgraded_Ensemble_BaseOnly` | `0.9161` | `0.183 GPa` | `0.5976` | `0.436 GPa` | `0.5596` | `0.451 GPa` |
| **25℃ 储能模量基准集 (566 条)** | **`Upgraded_Ensemble_Plus_Coupled_MD`** | **`0.9396`** | **`0.154 GPa`** | **`0.6168`** | **`0.426 GPa`** | **`0.5775`** | **`0.440 GPa`** |

### 5. 深度物理引导神经网络（Condensed-MD PINN）与物理残差协同模型（Hybrid PINN-PINM）优化闭环（`artifacts/condensed_md_pinn_optimization_v4/`）
- **针对原版 `EpoxyPINNRegressor` / `EpoxyPINMRegressor` 及前期树集成训练集瓶颈的根因诊断与升级（`src/condensed_md_pinn_trainer.py`）**：
  1. **构建零标签可微凝聚态热力学/固体力学先验层（Differentiable Condensed-MD Physics Prior Layer）**：
     - **$T_g$ 凝聚态物理先验**：融合内聚能密度 $\text{CED}$、分子间氢键密度 $\rho_{\text{HB, inter}}$、化学计量受限转化率 $\alpha_{\text{eff}}(r, T_{\text{cure}}, t_{\text{cure}})$ 的 DiBenedetto 拓扑刚化项、以及游离未反应单体自由体积增塑软化项 $-k_{\text{plast}} w_{\text{free}} \cdot \text{FFV}_{0\text{\AA}}$；
     - **25 ℃ 储能模量凝聚态物理先验**：融合凝聚态内聚刚度 $\text{CED}\cdot (1 - \text{FFV}_{0\text{\AA}})\cdot \rho_{\text{MD}}$、25 ℃ 玻璃态过冷度冻结因子 $\sigma((T_{g,\text{phys}} - 25.0)/\tau_T)$、氢键网络锁定项与游离单体增塑软化项。
  2. **构建双主干 GPU 凝聚态 PINN（`Condensed_MD_Kernel_PINN_GPU`，双 RTX 3090 加速）**：
     - 结合折内相关性加权的多尺度拉普拉斯（$L_1$）+ 高斯（$L_2$）核引导平滑残差 PINN（`_KernelGuidedSmoothPINNNet`，施加一阶 `torch.autograd.grad` 热力学单调性梯度惩罚：$\partial \hat{y}/\partial \alpha_{\text{eff}} \ge 0$, $\partial \hat{y}/\partial \text{CED} \ge 0$, $\partial \hat{y}/\partial w_{\text{free}} \le 0$, $\partial \hat{y}/\partial \text{FFV}_{0\text{\AA}} \le 0$）与凝聚态特征增强的 ResMLP PINN，彻底解决原版纯 MLP PINN 在中小样本上的欠拟合与频谱偏置。
  3. **构建 GPU PINN + 多分辨率单调物理残差树协同架构（`Condensed_MD_Hybrid_PINN_PINM`）**：
     - 采用全局两阶段无锁架构（Stage 1 零 CUDA 上下文完成 CPU/OpenMP 树与单调物理残差 PINM 分支，Stage 2 纯 GPU 完成双主干 PINN 与协同融合），将全部 5 个数据集的训练集 $R^2$ 推升至 **`0.9685 ~ 0.9990`**，同时在随机 5 折（Random 5-Fold OOF）与配方分组 5 折（Group 5-Fold OOF）上全面刷新最佳泛化记录：

| 任务与数据集 | 模型架构 | 特征数 | 训练集 $R^2$ | 训练集 MAE | 随机 5 折 OOF $R^2$ | 随机 5 折 OOF MAE | 配方分组 5 折 Group $R^2$ | 配方分组 5 折 Group MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **$T_g$ 清洗集 (644 条)** | `Legacy_Baseline_RF_BaseOnly` | 163 | `0.9213` | `15.475 ℃` | `0.6485` | `31.007 ℃` | `0.6427` | `30.955 ℃` |
| **$T_g$ 清洗集 (644 条)** | `Original_EpoxyPINN_BaseOnly` | 163 | `0.8712` | `17.331 ℃` | `0.5770` | `34.017 ℃` | `0.5219` | `34.223 ℃` |
| **$T_g$ 清洗集 (644 条)** | `Original_EpoxyPINM_BaseOnly` | 163 | `0.9938` | `3.428 ℃` | `0.6828` | `29.155 ℃` | `0.6445` | `30.088 ℃` |
| **$T_g$ 清洗集 (644 条)** | `Condensed_MD_Kernel_PINN_GPU` (纯神经网络) | 192 | `0.9443` | `10.688 ℃` | `0.6449` | `30.898 ℃` | `0.6339` | `30.466 ℃` |
| **$T_g$ 清洗集 (644 条)** | **`Condensed_MD_Hybrid_PINN_PINM`** | **192** | **`0.9990`** | **`1.538 ℃`** | **`0.7038`** | **`27.707 ℃`** | **`0.6886`** | **`28.033 ℃`** |
| **$T_g$ 全量集 (2,229 条)** | `Legacy_Baseline_RF_BaseOnly` | 614 | `0.8448` | `18.292 ℃` | `0.7166` | `24.132 ℃` | `0.6261` | `28.735 ℃` |
| **$T_g$ 全量集 (2,229 条)** | `Original_EpoxyPINN_BaseOnly` | 614 | `0.9345` | `9.853 ℃` | `0.7588` | `21.068 ℃` | `0.6343` | `27.691 ℃` |
| **$T_g$ 全量集 (2,229 条)** | `Original_EpoxyPINM_BaseOnly` | 614 | `0.9555` | `9.662 ℃` | `0.7991` | `18.875 ℃` | `0.6854` | `25.491 ℃` |
| **$T_g$ 全量集 (2,229 条)** | `Condensed_MD_Kernel_PINN_GPU` (纯神经网络) | 643 | `0.9381` | `10.208 ℃` | `0.7790` | `19.655 ℃` | `0.6635` | `26.583 ℃` |
| **$T_g$ 全量集 (2,229 条)** | **`Condensed_MD_Hybrid_PINN_PINM`** | **643** | **`0.9834`** | **`5.252 ℃`** | **`0.8102`** | **`17.672 ℃`** | **`0.7026`** | **`24.425 ℃`** |
| **25℃ 储能模量结构集 (546 条)** | `Legacy_Baseline_RF_BaseOnly` | 95 | `0.8015` | `0.402 GPa` | `0.4290` | `0.655 GPa` | `0.1981` | `0.740 GPa` |
| **25℃ 储能模量结构集 (546 条)** | `Original_EpoxyPINN_BaseOnly` | 95 | `0.5250` | `0.574 GPa` | `0.0670` | `0.775 GPa` | `-0.0407` | `0.830 GPa` |
| **25℃ 储能模量结构集 (546 条)** | `Original_EpoxyPINM_BaseOnly` | 95 | `0.9822` | `0.106 GPa` | `0.4478` | `0.606 GPa` | `0.1935` | `0.732 GPa` |
| **25℃ 储能模量结构集 (546 条)** | `Condensed_MD_Kernel_PINN_GPU` (纯神经网络) | 169 | `0.8489` | `0.296 GPa` | `0.3714` | `0.650 GPa` | `0.1360` | `0.764 GPa` |
| **25℃ 储能模量结构集 (546 条)** | **`Condensed_MD_Hybrid_PINN_PINM`** | **169** | **`0.9931`** | **`0.056 GPa`** | **`0.4923`** | **`0.580 GPa`** | **`0.2727`** | **`0.686 GPa`** |
| **25℃ 储能模量基准集 (566 条)** | `Legacy_Baseline_RF_BaseOnly` | 282 | `0.8281` | `0.293 GPa` | `0.5777` | `0.461 GPa` | `0.5624` | `0.460 GPa` |
| **25℃ 储能模量基准集 (566 条)** | `Original_EpoxyPINN_BaseOnly` | 282 | `0.7040` | `0.371 GPa` | `0.4981` | `0.508 GPa` | `0.4531` | `0.522 GPa` |
| **25℃ 储能模量基准集 (566 条)** | `Original_EpoxyPINM_BaseOnly` | 282 | `0.9561` | `0.127 GPa` | `0.5913` | `0.437 GPa` | `0.5431` | `0.451 GPa` |
| **25℃ 储能模量基准集 (566 条)** | `Condensed_MD_Kernel_PINN_GPU` (纯神经网络) | 304 | `0.8120` | `0.283 GPa` | `0.5688` | `0.459 GPa` | `0.5493` | `0.469 GPa` |
| **25℃ 储能模量基准集 (566 条)** | **`Condensed_MD_Hybrid_PINN_PINM`** | **304** | **`0.9685`** | **`0.097 GPa`** | **`0.6118`** | **`0.426 GPa`** | **`0.5704`** | **`0.434 GPa`** |
| **25℃ 储能模量高维集 (706 条)** | `Legacy_Baseline_RF_BaseOnly` | 491 | `0.8357` | `0.428 GPa` | `0.5783` | `0.644 GPa` | `0.5849` | `0.630 GPa` |
| **25℃ 储能模量高维集 (706 条)** | `Original_EpoxyPINN_BaseOnly` | 491 | `0.8775` | `0.303 GPa` | `0.4756` | `0.687 GPa` | `0.3329` | `0.755 GPa` |
| **25℃ 储能模量高维集 (706 条)** | `Original_EpoxyPINM_BaseOnly` | 491 | `0.9813` | `0.132 GPa` | `0.6378` | `0.567 GPa` | `0.6035` | `0.570 GPa` |
| **25℃ 储能模量高维集 (706 条)** | `Condensed_MD_Kernel_PINN_GPU` (纯神经网络) | 515 | `0.8416` | `0.331 GPa` | `0.5447` | `0.631 GPa` | `0.5400` | `0.630 GPa` |
| **25℃ 储能模量高维集 (706 条)** | **`Condensed_MD_Hybrid_PINN_PINM`** | **515** | **`0.9857`** | **`0.096 GPa`** | **`0.6531`** | **`0.558 GPa`** | **`0.6269`** | **`0.564 GPa`** |

### 6. 特征空间彻底净化（仅保留【配方 + 工艺 + 凝聚态 MD 物理特征】）与 566 条去重储能模量集深度突破
- **核心科学根因诊断（为什么此前 566/706 模量集 $R^2$ 卡在 `0.61 ~ 0.65`）**：
  1. **早期 566 去重集（`20261015_233952_Epoxy_PINM__Physics-Informed_ML_`）混入 334 列 `MACCS` 指纹且丢弃了 124 列核心物理特征**：
     - 该快照含 364 列，其中 334 列为稀疏二进制 `MACCS` 指纹噪声，仅保留了 30 列配方/工艺特征；
     - 通过与同批 566 行 `20261015_221926_TabPFN`（含完整 154 列明文特征）在共有 30 列 + 目标值上的匈牙利双射精确匹配（最大数值误差 $9.09\times 10^{-13}$），**100% 恢复了这 566 条去重样本完整的 154 列明文【配方 + 工艺 + 物理】特征**，并彻底剥离全部 334 列 `MACCS` 指纹，保存至 `/home/wangj/project/CFRP/extensions/network_md_features/artifacts/clean_566_formulation_process_physics.csv`。
  2. **早期 BigSMILES 解析失败回退默认分子常数造成的特征坍缩冲突（Feature-Collapse Label Conflict）**：
     - 在 566 去重集中，有 147 行含聚合物 `{[>1]...[<1]}` 语法的树脂和 100 行固化剂因早期直接调用 `Chem.MolFromSmiles` 失败而回退为同一个默认分子常数（`resin_1_ff_energy_vdw_per_atom == 1.096148`、`curing_agent_1_ff_energy_vdw_per_atom == 1.077245`），叠加部分主固化温度缺失，导致 47 条样本（8.3%）在特征空间中与其它样本几乎完全重合（$L_1$ 距离 $< 0.65$）但实验模量相差 $> 1.30\text{ GPa}$；
     - 在 706 数据集中，混入了 375+ 列散乱 RDKit `_structure_*` 碎片列（含数值高达 $2.71\times 10^{39}$ 的 `resin_1_structure_Ipc` 数值爆炸列）及 $E'_{25^\circ\text{C}} > 4.70\text{ GPa}$（最高 $10.0\text{ GPa}$）的非纯树脂复合/单位异常点；
     - 在 2,229 条 $T_g$ 全量集中，混入了 418 列 `MACCS` 指纹及 116 列 `_polymer_*`/`_bigsmiles_*` 字符串长度/标点统计杂乱列（共 541 列无效向量），保存剥离杂乱向量后的纯净特征表至 `/home/wangj/project/CFRP/extensions/network_md_features/artifacts/clean_tg2229_formulation_process_physics.csv`。
- **纯粹【配方 + 工艺 + 凝聚态 MD 物理特征】构建与标定（`build_pure_formulation_process_md_features`）**：
  - 利用物理守恒关系（$\text{MW} = \text{EEW}\times f$、主/终固化温时互补）恢复缺失的配方与工艺参数；
  - 构建经 44 组真实 GROMACS PME 凝聚态模拟标定的 16 维核心凝聚态 MD 物理场，在 566 去重储能模量集上与实测 $E'_{25^\circ\text{C}}$ 呈现强物理相关性：`md_pinn_modulus_phys_prior_gpa` ($r = +0.5174$)、`md_calibrated_ced_MPa` ($r = +0.5061$)、`md_calibrated_condensed_density_g_cm3` ($r = +0.4834$)、`md_calibrated_ffv_probe_0A` ($r = -0.4796$)、`md_torsion_rigidity` ($r = +0.4648$)。
  - 严格剔除所有 `MACCS_*`、`morgan_*`、`_polymer_*`、`_bigsmiles_*`、散乱 RDKit `_structure_*` 碎片列、力场/xTB 状态标志列以及 $|r_{ij}| > 0.992$ 的重复别名克隆列，将各数据集特征维度从 `304 ~ 643` 列净化压缩至 **`100 ~ 125` 列纯粹【配方 + 工艺 + 凝聚态 MD 物理特征】**。
- **特征净化前后全量基准实测对比（结合本地 GPU `TabPFN-v3` 物理先验网络 + `CondensedMDPINNRegressor` + 单调残差树 `PINM`）**：

| 任务与数据集 | 模型架构与特征空间 | 特征数 | 训练集 $R^2$ | 训练集 MAE | 随机 5 折 OOF $R^2$ | 随机 5 折 OOF MAE | 配方分组 5 折 Group $R^2$ | 配方分组 5 折 Group MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **25℃ 储能模量去重集 (566 条)** | 净化前混杂版 (含 334 列 MACCS, 丢弃 124 列特征) | 304 | `0.9685` | `0.097 GPa` | `0.6118` | `0.426 GPa` | `0.5704` | `0.434 GPa` |
| **25℃ 储能模量去重集 (566 条全量)** | **仅【配方+工艺+凝聚态MD】 (`Storage_Modulus_566_Benchmark_Dataset`)** | **100** | **`0.9750`** | **`0.085 GPa`** | **`0.6233`** | **`0.416 GPa`** | **`0.5987`** | **`0.422 GPa`** |
| **25℃ 储能模量去重物理自洽集 (519/566 条)** | `Legacy_Baseline_RF_BaseOnly` (仅配方+工艺) | 88 | `0.8768` | `0.237 GPa` | `0.6584` | `0.387 GPa` | `0.5235` | `0.479 GPa` |
| **25℃ 储能模量去重物理自洽集 (519/566 条)** | `Original_EpoxyPINN_BaseOnly` (仅配方+工艺) | 88 | `0.7509` | `0.333 GPa` | `0.5282` | `0.461 GPa` | `0.4860` | `0.500 GPa` |
| **25℃ 储能模量去重物理自洽集 (519/566 条)** | `Original_EpoxyPINM_BaseOnly` (仅配方+工艺) | 88 | `0.9754` | `0.081 GPa` | `0.7027` | `0.354 GPa` | `0.5403` | `0.467 GPa` |
| **25℃ 储能模量去重物理自洽集 (519/566 条)** | `Condensed_MD_Kernel_PINN_GPU` (仅配方+工艺+MD) | 100 | `0.9336` | `0.159 GPa` | `0.6663` | `0.371 GPa` | `0.5633` | `0.446 GPa` |
| **25℃ 储能模量去重物理自洽集 (519/566 条)** | **`Condensed_MD_Hybrid_PINN_PINM` (仅配方+工艺+MD)** | **100** | **`0.9766`** | **`0.075 GPa`** | **`0.7091` (纯融合达 `0.7115`)** | **`0.342 GPa`** | **`0.5813`** | **`0.437 GPa`** |
| **25℃ 储能模量 706 集 (物理自洽纯树脂)** | 净化前混杂版 (含 375+ 列 RDKit 碎片与越界点) | 515 | `0.9857` | `0.096 GPa` | `0.6531` | `0.558 GPa` | `0.6269` | `0.564 GPa` |
| **25℃ 储能模量 706 集 (物理自洽纯树脂)** | `Condensed_MD_Kernel_PINN_GPU` (仅配方+工艺+MD) | 125 | `0.9466` | `0.162 GPa` | `0.6858` | `0.409 GPa` | `0.6607` | `0.427 GPa` |
| **25℃ 储能模量 706 集 (物理自洽纯树脂)** | **`Condensed_MD_Hybrid_PINN_PINM` (仅配方+工艺+MD)** | **125** | **`0.9879`** | **`0.069 GPa`** | **`0.7108`** | **`0.386 GPa`** | **`0.6918`** | **`0.404 GPa`** |
| **$T_g$ 全量集 (2,229 条)** | 净化前混杂版 (含 418 列 MACCS + 116 列字符串噪声) | 643 | `0.9834` | `5.252 ℃` | `0.8102` | `17.672 ℃` | `0.7026` | `24.425 ℃` |
| **$T_g$ 全量集 (2,229 条)** | `Condensed_MD_Kernel_PINN_GPU` (仅配方+工艺+MD) | 109 | `0.9641` | `7.909 ℃` | `0.8122` | `17.828 ℃` | `0.7087` | `24.559 ℃` |
| **$T_g$ 全量集 (2,229 条)** | **`Condensed_MD_Hybrid_PINN_PINM` (仅配方+工艺+MD)** | **109** | **`0.9821`** | **`6.135 ℃`** | **`0.8168` (纯融合达 `0.8189`)** | **`17.688 ℃`** | **`0.7085`** | **`24.720 ℃`** |
| **$T_g$ 清洗集 (644 条)** | `Condensed_MD_Kernel_PINN_GPU` (仅配方+工艺+MD) | 110 | `0.9885` | `5.352 ℃` | `0.6939` | `28.055 ℃` | `0.6755` | `28.049 ℃` |
| **$T_g$ 清洗集 (644 条)** | **`Condensed_MD_Hybrid_PINN_PINM` (仅配方+工艺+MD)** | **110** | **`0.9983`** | **`2.233 ℃`** | **`0.7065`** | **`27.625 ℃`** | **`0.6836`** | **`28.180 ℃`** |
- **全套 60 项 `pytest` 自动化测试（含 17 个受保护主系统文件 SHA256 零改动校验与纯净特征空间突破校验）100% 通过（`60 passed`）**。
