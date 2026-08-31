# 分子设计引擎重构实施计划（基于真实有机反应的正交组合展开）

> **设计规范文件**：`docs/superpowers/specs/2025-03-24-molecule-design-combinatorial-spec.md`  
> **目标**：彻底重构分子设计引擎，废弃旧的不可控苯环自由基加甲基算子与贪心模型截断，实现基于工业/科研前驱体母核与真实有机反应模板（Reaction SMARTS）的正交组合展开，输出大规模、可合成、合规的热固性树脂/固化剂单体库，并与下游配方高通量筛选无缝联动。

---

## 阶段一：重构核心分子设计库与反应模板 (`core/molecule_design.py`)

### 1.1 前驱体母核库构建 (`PRECURSOR_CATALOG`)
- [ ] 定义包含双酚类（BPA, BPF, BPS, BPAF, TMBPA）、稠环类（1,5-DHN, 2,7-DHN, 间苯二酚, 联苯二酚）、特种耐热多酚（芴基双酚 BHPF, 三酚甲烷 THPM, 四酚乙烷 THPE）、芳香多胺（DDM, 4,4'-DDS, 3,3'-DDS, ODA, m-PDA, p-PDA, DAB, 芴基二胺 FDA）、脂环胺（IPDA）及酸酐（MTHPA, MHHPA, PMDA）的标准 SMILES 与元数据字典。

### 1.2 真实有机反应模板体系 (`SYNTHETIC_REACTION_TEMPLATES`)
- [ ] 构建严格的 RDKit `ChemicalReaction` 模板：
  - `glycidyl_etherification`: 酚羟基 $\rightarrow$ 缩水甘油醚（生成 DGEBA, DGEBF, 芴基环氧等）
  - `glycidyl_amination`: 芳香伯胺 $\rightarrow$ 缩水甘油胺（生成 TGDDM/AG-80, TGDAP 等）
  - `glycidyl_esterification`: 羧酸 $\rightarrow$ 缩水甘油酯（生成 TDE-85 等）
  - `propargyl_etherification`: 酚羟基 $\rightarrow$ 炔丙基醚（耐烧蚀热固树脂）
  - `allyl_etherification`: 酚羟基 $\rightarrow$ 烯丙基醚（增韧与共聚单体）
  - `cyanate_esterification`: 酚羟基 $\rightarrow$ 氰酸酯（超低介电/超高耐热单体）
  - `amine_mono_alkylation`: 芳香胺部分烷基化（活性调控）

### 1.3 正交组合展开引擎 (`generate_combinatorial_monomers`)
- [ ] 实现 `Precursor Core × Reaction Template` 的笛卡尔积正交展开算法。
- [ ] 对反应产物执行多步彻底展开（如多元酚上的全部 `-OH` 均完成醚化）。
- [ ] 产物标准化与规范化（Canonical SMILES 去重）。

### 1.4 化学合理性与合成可及性过滤器 (`validate_designed_product`)
- [ ] RDKit `SanitizeMol` 严格价态校验（五价碳、奇数价氮、非法自由基快速剔除）。
- [ ] 交联官能度保护校验（树脂反应基团数 $\ge 2$；固化剂活性氢 $\ge 2$）。
- [ ] 物理属性边界过滤（$MW \in [150, 1500]\text{ g/mol}$，环数 $1 \sim 12$）。
- [ ] SAScore（合成可及性难度）评估计算，过滤 $\text{SAScore} > 6.0$ 的非物理畸形结构。

---

## 阶段二：重构 Streamlit UI 交互界面 (`app.py`)

### 2.1 界面组件重构 (`_render_molecule_design_engine`)
- [ ] 移除旧的 Beam Search / 模型贪心截断配置项（搜索深度、探索比例等）。
- [ ] 增加前驱体母核分类多选面板（双酚系列、稠环芳香类、特种耐热多酚、芳香多胺、脂环胺与酸酐）。
- [ ] 增加合成反应类型多选面板（缩水甘油醚化、缩水甘油胺化、酯化、炔丙基化、烯丙基化、氰酸酯化等）。
- [ ] 增加质量门禁与数量控制（目标最大产物数、最小官能度下限、SAScore 最大容忍度）。

### 2.2 极速生成与实时状态反馈
- [ ] 点击 **【🚀 组合正交展开生成单体库】**，使用 `st.status` 动态展示正交组合进度、过滤通过率与最终入库数量。
- [ ] 产物列表表格化呈现：包含结构预览、起始母核、合成反应路径、官能度、分子量 $MW$、合成难度 $SAScore$。
- [ ] 一键导出 CSV 文件功能。

---

## 阶段三：全流程联动与下游配方高通量筛选打通

### 3.1 成果会话持久化与下游自动继承
- [ ] 将正交生成的数百/上千个优质单体结构化注入 `st.session_state["vs_design_result_df"]`。
- [ ] 在【配方级高通量筛选】（`_page_virtual_screening_formula`）中，确保“🧩 叠加分子设计引擎产出的候选”能无缝读取这批单体。
- [ ] 下游系统将自动将这批单体与目标树脂/固化剂、phr 配比、固化温度曲线进行全网格组合，并由已训练好的机器学习模型进行打分与多目标 Pareto 筛选。

---

## 阶段四：单元测试与全链路验证

### 4.1 核心算法测试
- [ ] 编写测试用例验证 15+ 种前驱体与 7 种反应模板的正交组合产物正确性。
- [ ] 验证产物的交联官能度是否稳定 $\ge 2$。
- [ ] 验证产物是否 100% 能够通过 RDKit 化学解析与特征工程 Pipeline。
