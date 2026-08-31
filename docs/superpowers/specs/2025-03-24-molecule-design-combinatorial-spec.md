# 虚拟分子设计引擎重构规范：基于真实有机反应的母核正交组合展开

**设计日期**：2025-03-24  
**所属模块**：CFRP 树脂基复合材料体系 / 虚拟分子筛选 · 分子设计引擎  
**状态**：已批准（Approved）

---

## 1. 概述与设计目标

### 1.1 背景与现状问题
现有的分子设计模块存在以下严重缺陷：
1. **生成逻辑失真**：大量依赖不可控的苯环自由基加甲基（`aryl_methyl_substitution`），产物在真实实验中无合成路线或无选择性；
2. **生成与评价过度耦合**：在单体生成步骤中强制使用回归模型进行 Beam Search 贪心截断，导致模型偏好畸形变体（只保留了 64 个分数最高的无意义局部变体），阻碍了真正多样化的大批量分子库生成；
3. **官能团与交联网络受损**：无法稳定保证树脂/固化剂形成热固性网络所必需的 $\ge 2$ 官能度。

### 1.2 重构目标
1. **解耦生成与评估**：分子设计引擎纯化为“前置可合成单体库生成器”，输出大规模（数百至上千个）、化学结构严谨、合成可及的候选单体；性能评估完全交由下游【配方级高通量筛选】在真实的配比与工艺网格下完成。
2. **基于真实合成砌块与反应模板（Reaction SMARTS）**：采用工业及前沿科研中标准的高性能母核骨架（双酚类、多芳类、芴基类、芳香多胺等）与成熟有机反应（酚缩水甘油醚化、胺缩水甘油化、炔丙基化、氰酸酯化等）进行正交组合展开。
3. **内置多层化学合理性与合成可及性门禁（Quality Gate）**：严格进行价态校验、交联官能度断言、SAScore（合成难易度）评估与分子量/LogP 物理边界控制。

---

## 2. 系统架构与数据流

```
[ 用户交互界面 (UI) ]
  ├── 1. 勾选前驱体母核库 (双酚类/稠环类/芴基类/芳香胺类/酸酐类...)
  ├── 2. 勾选启用的有机反应模板 (缩水甘油醚化/胺化/酯化/炔丙基化/氰酸酯化...)
  └── 3. 设定过滤阈值 (最大产物数、SAScore 阈值、官能度下限)
       │
       ▼
[ 组合正交展开引擎 (Combinatorial Assembly Engine) ]
  ├── 遍历 Precursor Core × Reaction Template
  ├── 调用 RDKit ChemicalReaction 执行严谨反应变换
  ├── 自动标准化为 Canonical SMILES 并去重
       │
       ▼
[ 化学合理性与合成可及性过滤器 (Quality Gate) ]
  ├── RDKit SanitizeMol (价态/自由基/未闭环检查)
  ├── 官能度过滤器 (Functionality >= 2 确保热固交联能力)
  ├── SAScore 评估 (合成难度打分，过滤 > 6.0 的畸形结构)
  └── 分子量与物理边界过滤 (MW 150~1500 g/mol)
       │
       ▼
[ 产物交付与下游无缝联动 (Delivery & Handoff) ]
  ├── 结构化表格呈现 (SMILES, 结构名, 起始母核, 反应类型, 官能度, MW, SAScore)
  ├── 导出 CSV 供科研与实验记录
  └── 自动注入 st.session_state["vs_design_result_df"]，下游配方筛选一键调用
```

---

## 3. 核心前驱体母核库（Precursor Cores）

### 3.1 树脂母核（Resin Precursors, 多元酚/多元酸）
1. **双酚A (BPA)**: `CC(C)(c1ccc(O)cc1)c1ccc(O)cc1`
2. **双酚F (BPF)**: `Oc1ccc(Cc2ccc(O)cc2)cc1`
3. **双酚S (BPS)**: `Oc1ccc(S(=O)(=O)c2ccc(O)cc2)cc1`
4. **六氟双酚A (BPAF)**: `Oc1ccc(C(C(F)(F)F)(C(F)(F)F)c2ccc(O)cc2)cc1`
5. **四甲基双酚A (TMBPA)**: `Cc1cc(C(C)(C)c2cc(C)c(O)c(C)c2)cc(C)c1O`
6. **9,9-双(4-羟基苯基)芴 (BHPF, 芴基双酚)**: `Oc1ccc(C2(c3ccccc3-c3ccccc32)c2ccc(O)cc2)cc1`
7. **1,5-萘二酚 (1,5-DHN)**: `Oc1cccc2c(O)cccc12`
8. **2,7-萘二酚 (2,7-DHN)**: `Oc1ccc2cc(O)ccc2c1`
9. **间苯二酚 (Resorcinol)**: `Oc1cccc(O)c1`
10. **4,4'-联苯二酚 (BP)**: `Oc1ccc(-c2ccc(O)cc2)cc1`
11. **三(4-羟苯基)甲烷 (THPM)**: `Oc1ccc(C(c2ccc(O)cc2)c2ccc(O)cc2)cc1`
12. **四(4-羟苯基)乙烷 (THPE)**: `Oc1ccc(C(c2ccc(O)cc2)C(c2ccc(O)cc2)c2ccc(O)cc2)cc1`
13. **间苯二甲酸 (Isophthalic acid)**: `O=C(O)c1cccc(C(=O)O)c1`
14. **对苯二甲酸 (Terephthalic acid)**: `O=C(O)c1ccc(C(=O)O)cc1`
15. **六氢邻苯二甲酸 (HHPA-acid)**: `O=C(O)C1CCCCC1C(=O)O`

### 3.2 固化剂母核（Hardener Precursors, 芳香胺/脂环胺/酸酐）
1. **4,4'-二氨基二苯甲烷 (DDM)**: `Nc1ccc(Cc2ccc(N)cc2)cc1`
2. **4,4'-二氨基二苯砜 (4,4'-DDS)**: `Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1`
3. **3,3'-二氨基二苯砜 (3,3'-DDS)**: `Nc1cccc(S(=O)(=O)c2cccc(N)c2)c1`
4. **4,4'-二氨基二苯醚 (ODA)**: `Nc1ccc(Oc2ccc(N)cc2)cc1`
5. **间苯二胺 (m-PDA)**: `Nc1cccc(N)c1`
6. **对苯二胺 (p-PDA)**: `Nc1ccc(N)cc1`
7. **4,4'-二氨基联苯 (DAB)**: `Nc1ccc(-c2ccc(N)cc2)cc1`
8. **9,9-双(4-氨基苯基)芴 (FDA)**: `Nc1ccc(C2(c3ccccc3-c3ccccc32)c2ccc(N)cc2)cc1`
9. **异佛尔酮二胺 (IPDA)**: `CC1(C)CC(C)(CN)CC(N)C1`
10. **甲基四氢苯酐 (MTHPA)**: `CC1=CCC2C(=O)OC(=O)C2C1`
11. **甲基六氢苯酐 (MHHPA)**: `CC1CCC2C(=O)OC(=O)C2C1`
12. **均苯四甲酸二酐 (PMDA)**: `O=C1OC(=O)c2cc3C(=O)OC(=O)c3cc21`

---

## 4. 真实有机反应模板库（Reaction SMARTS）

1. **多元酚缩水甘油醚化 (`glycidyl_etherification`)**
   - 反应：`[c:1][OH:2] >> [c:1]OCC1CO1`
   - 适用：树脂多酚 $\rightarrow$ 典型缩水甘油醚环氧树脂（如 DGEBA、DGEBF、四官能芴基环氧等）。
2. **芳香多胺缩水甘油胺化 (`glycidyl_amination`)**
   - 反应：`[c:1][NH2:2] >> [c:1]N(CC2CO2)CC3CO3`
   - 适用：芳香多胺 $\rightarrow$ 缩水甘油胺耐高温环氧（如 TGDDM/AG-80、TGDAP）。
3. **多元羧酸缩水甘油酯化 (`glycidyl_esterification`)**
   - 反应：`[C:1](=O)[OH:2] >> [C:1](=O)OCC2CO2`
   - 适用：二元酸 $\rightarrow$ 缩水甘油酯低粘度树脂。
4. **多元酚炔丙基醚化 (`propargyl_etherification`)**
   - 反应：`[c:1][OH:2] >> [c:1]OCC#C`
   - 适用：耐高温耐烧蚀热固性树脂单体。
5. **多元酚烯丙基醚化 (`allyl_etherification`)**
   - 反应：`[c:1][OH:2] >> [c:1]OCC=C`
   - 适用：双烯丙基双酚增韧/共聚树脂单体。
6. **多元酚氰酸酯化 (`cyanate_esterification`)**
   - 反应：`[c:1][OH:2] >> [c:1]OC#N`
   - 适用：超低介电常数与超高耐热氰酸酯单体。
7. **芳香胺部分烷基化 (`amine_mono_alkylation`)**
   - 反应：`[c:1][NH2:2] >> [c:1]NCC`
   - 适用：调控固化剂反应活性与降低交联脆性。

---

## 5. 质量门禁与过滤规则（Quality Gate）

1. **化学合法性检查**：
   - 经 RDKit `SanitizeMol`，过滤五价碳、奇数价氮、非法自由基等非物理结构。
2. **交联官能度保护**：
   - 树脂产物中有效反应性基团（环氧基、炔基、烯基、氰酸酯基）数量必须 $\ge 2$；
   - 固化剂产物中活性氢/反应位点数必须 $\ge 2$。
3. **合成可及性打分（SAScore）**：
   - 基于 RDKit `contrib.SA_Score` 计算，范围 1.0 ~ 10.0；
   - 默认过滤 $\text{SAScore} > 6.0$ 的高难度/非物理空间位阻结构。
4. **物理化学合理性约束**：
   - 分子量 $MW \in [150, 1500]\text{ g/mol}$；
   - 包含的环数 $\ge 1$ 且 $\le 12$。

---

## 6. UI 呈现与下游配方高通量联动规范

1. **界面精简与清晰反馈**：
   - 提供前驱体母核多选面板（分组为：双酚系列、稠环芳香类、特种耐热类、芳香多胺类、酸酐类等）；
   - 提供反应模板多选面板（缩水甘油醚化、胺化、炔丙基化、氰酸酯化等）；
   - 点击 **【🚀 组合正交展开生成单体库】**，动态呈现生成进度条与生成数量。
2. **产物列表数据结构**：
   - 输出 DataFrame 包含：`product_smiles`, `role`, `precursor_name`, `reaction_name`, `functionality`, `molecular_weight`, `sa_score`。
   - 提供一键 CSV 下载。
3. **下游无缝联动**：
   - 生成的单体列表自动固化入 `st.session_state["vs_design_result_df"]`；
   - 在【配方级高通量筛选】中，用户可直接勾选“叠加分子设计引擎产出的候选”，这批单体将无缝参与配方配比、固化温度、工艺热积分的网格化全自动模型预测与多目标优化。

---

## 7. 验收标准

1. 点击生成后，系统基于勾选的母核与反应模板进行稳定正交组合，产出数百个结构合规单体；
2. 不再产生任何杂乱无章的在苯环上乱加甲基的畸形结构；
3. 不再在单体生成阶段强制依赖模型打分与截断；
4. 产出的单体库可以直接被【配方级高通量筛选】识别并完整参与配方网格计算。
