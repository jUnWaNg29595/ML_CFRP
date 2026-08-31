# 虚拟分子设计与配方级高通量衔接引擎实施计划 (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建支持环结构、主链桥联、侧链R基修饰与核心官能团多维演化的组合分子设计引擎，生成 2,000~20,000+ 可合成单体产物库，并自动计算 EEW/AHEW 化学计量特征，实现与配方级高通量筛选（HTVS）的无缝一键注入。

**Architecture:** 
采用两阶段流水线架构：阶段 1 由 24 类环骨架、20 类主链桥联与 12 类侧链 R 基通过多拓扑组装裂变生成数万级多元中间体前驱体；阶段 2 通过 12 大 Reaction SMARTS 反应模板进行穷举转化；阶段 3 施加 RDKit 有效性、SAScore 难度及官能度门禁并自动计算 EEW/AHEW；阶段 4 提供 Streamlit 前端交互与一键向配方级高通量筛选注入的桥接协议。

**Tech Stack:** Python 3.10+, RDKit, Pandas, Streamlit, Pytest

## Global Constraints
- 严格遵循 RDKit 化学有效性校验（SanitizeMol）。
- 元素白名单限定为 `{H, C, N, O, F, Si, P, S, Cl, Br}`。
- 默认过滤条件：`functionality >= 2`, `SAScore <= 6.0`, `140.0 <= MW <= 1500.0`。
- 所有 UI 文本与提示信息保持中文。

---

### Task 1: 建立全维度拓扑积木库 (Ring Scaffolds, Linkers, R-Groups)

**Files:**
- Modify: `core/molecule_design.py`
- Test: `tests/test_combinatorial_molecule_design.py`

**Interfaces:**
- Produces: 
  - `RING_SCAFFOLDS: list[RingScaffold]` (24 类骨架环)
  - `LINKER_BRIDGES: list[LinkerBridge]` (20 类主链桥联)
  - `R_GROUPS: list[RGroupSubstituent]` (12 类侧链基团)
  - `generate_scaffold_intermediates(...) -> list[PrecursorCore]` (前驱体骨架生成函数)

- [ ] **Step 1: 编写积木库与骨架裂变单元测试**
在 `tests/test_combinatorial_molecule_design.py` 中添加对 24 类环、20 类桥联、12 类 R 基以及骨架生成函数 `generate_scaffold_intermediates` 的测试用例。

- [ ] **Step 2: 运行测试并验证失败**
执行 `pytest tests/test_combinatorial_molecule_design.py` 确认新测试因缺少定义而失败。

- [ ] **Step 3: 实现全量积木字典与骨架组装生成器**
在 `core/molecule_design.py` 中实现 `RingScaffold`, `LinkerBridge`, `RGroupSubstituent` 数据结构与骨架组装逻辑，支持双核、三核星型、四核空间及稠环多取代拓扑。

- [ ] **Step 4: 运行测试并验证通过**
执行 `pytest tests/test_combinatorial_molecule_design.py` 确保积木组装生成的 SMILES 均合法且数量达标。

---

### Task 2: 扩充 12 大多通道合成反应模板库与完全性穷举转化

**Files:**
- Modify: `core/molecule_design.py`
- Test: `tests/test_combinatorial_molecule_design.py`

**Interfaces:**
- Produces: 
  - `SYNTHETIC_REACTION_TEMPLATES: list[SyntheticReaction]` (包含 R01~R12 的完整模板列表)
  - `apply_multi_channel_reactions(...) -> list[CombinatorialProduct]`

- [ ] **Step 1: 编写多通道反应模板转换与角色分类测试**
针对酚缩水甘油醚、芳胺缩水甘油胺、酸缩水甘油酯、脂环环氧、酚醛胺、酸酐、BMI、氰酸酯、苯并噁嗪等编写独立化学转化测试，验证产物结构的合法性与 role 分类。

- [ ] **Step 2: 运行测试并确认失败**
执行 `pytest tests/test_combinatorial_molecule_design.py`。

- [ ] **Step 3: 实现 12 大 Reaction SMARTS 反应模板与深度穷举反应器**
在 `core/molecule_design.py` 中配置完整的 SMARTS 字符串与反应产物后处理逻辑。

- [ ] **Step 4: 运行测试验证转化通过**
执行 `pytest tests/test_combinatorial_molecule_design.py`。

---

### Task 3: 完善质量门禁与化学计量当量 (EEW/AHEW) 自动计算

**Files:**
- Modify: `core/molecule_design.py`
- Test: `tests/test_combinatorial_molecule_design.py`

**Interfaces:**
- Produces:
  - `calculate_stoichiometry(mol, role, warhead) -> dict[str, Any]` (计算 EEW、AHEW、官能度、分子量、SAScore)
  - `run_combinatorial_monomer_design(...) -> tuple[pd.DataFrame, list[str]]` (全流水线执行入口，支持万级生成)

- [ ] **Step 1: 编写 EEW、AHEW 与门禁过滤测试**
测试 DGEBA 的 EEW 约为 170 g/eq、TGDDM 的 EEW 约为 105 g/eq、4,4'-DDS 的 AHEW 约为 62 g/eq 等已知标杆分子。

- [ ] **Step 2: 运行测试并确认失败**
执行 `pytest tests/test_combinatorial_molecule_design.py`。

- [ ] **Step 3: 编写化学计量计算函数与全流水线展开函数**
在 `core/molecule_design.py` 中实现精确的子结构匹配统计与当量换算公式。

- [ ] **Step 4: 运行测试并确认全部通过**
执行 `pytest tests/test_combinatorial_molecule_design.py`。

---

### Task 4: UI 交互重构与配方级高通量筛选（HTVS）一键注入联动

**Files:**
- Modify: `app.py`
- Test: `tests/test_virtual_screening.py`

**Interfaces:**
- Consumes: `core.molecule_design.run_combinatorial_monomer_design`
- Produces:
  - 分子设计引擎新版 Streamlit 交互面板（支持环选择、桥联选择、R基选择、反应路径选择与规模配置）
  - 「一键注入配方级高通量筛选」按钮及 session_state 数据桥接

- [ ] **Step 1: 编写配方级筛选对接集成测试**
在 `tests/test_virtual_screening.py` 中验证 `molecule_design_engine` 产生的单体库能够被 `build_component_library` 和配方生成器正确消费。

- [ ] **Step 2: 运行测试并确认失败**
执行 `pytest tests/test_virtual_screening.py`。

- [ ] **Step 3: 更新 app.py 中 `_render_molecule_design_engine` 与配方筛选联动逻辑**
实现拓扑维度配置组件、产物库统计卡片（树脂数、固化剂数、EEW/AHEW 分布）及一键注入按钮。

- [ ] **Step 4: 运行测试并验证全流程通过**
执行 `pytest tests/test_virtual_screening.py` 与 `pytest tests/test_combinatorial_molecule_design.py`。
