# 虚拟分子筛选 · 全维度拓扑演化设计引擎 —— 合成转化路径细分与骨架多样性修复设计规范

- **日期**：2026-09-22
- **模块**：`core/molecule_design.py`（主）、`app_lib.py`（UI）、`core/screening_audit.py`（审计）
- **上游规范**：`docs/superpowers/specs/2026-08-28-combinatorial-molecule-design-engine-spec.md`
- **目标**：① 把「细分有机合成转化路径」从扁平反应模板列表升级为**路线感知（route-aware）**的合成路径设计器，补齐"羟基环氧化"的四条真实工业路线；② 修复产物库中**桥联类主导、骨架类稀少**的分布失衡问题。

---

## 1. 问题溯源（实测证据）

### 1.1 实测环境

- RDKit `2026.03.4`
- 复现命令基线：`run_combinatorial_monomer_design(enable_scaffold_fission=True, min_functionality=2, max_sa_score=6.0, mw_range=(140.0, 1500.0), max_total_products=5000)`
- 实测产物总数 **3625**，涉及独特母核 **1271**（不含 44 个经典母核）

### 1.2 现状：转化路径 = 扁平单步模板，无"路径"语义

| 项 | 现状 |
|---|---|
| 模板注册表 | `SYNTHETIC_REACTION_TEMPLATES`，**17 条**（文件注释写"12 大"已过时） |
| 穷举器 | `_apply_reaction_exhaustively` 只做**单模板自身**多位点穷举，**不做跨模板链式串联** |
| 去重语义 | `run_combinatorial_monomer_design` 的 `seen_smiles` 全局去重，同产物只保留**第一条**命中的 `reaction_id` |
| 产物字段 | `CombinatorialProduct.reaction_id` 为**标量**，只能记录一条路线 |

**结论**：当前"转化路径"实质是 17 个互不串联的单步反应，无法表达多步路线，也无法回答"这个产物能用哪几条路线合成"。

### 1.3 化学硬伤（RDKit 实测确认）

**硬伤 1｜R01 羟基缩水甘油醚化 SMARTS 过宽**

```python
reaction_smarts = "[#6:1]-[OX2H:2]>>[#6:1]-OCC1CO1"   # 无差别攻击所有羟基
```

| 底物 | 现有 R01 产物 | 化学判断 |
|---|---|---|
| 对羟基苯甲酸 `O=C(O)c1ccc(O)cc1` | `O=C(OCC1CO1)c1ccc(O)cc1` | ❌ 羧酸被当醇醚化，抢 R03 的活 |
| 己二酸 `O=C(O)CCCCC(=O)O` | `O=C(O)CCCCC(=O)OCC1CO1` | ❌ 多元酸前驱体被污染 |
| 叔丁醇 `CC(C)(C)O` | `CC(C)(C)OCC1CO1` | ❌ 叔醇位阻下 ECH 醚化需 NaH/相转移，工业不成立 |

**硬伤 2｜R04 双键环氧化无选择性**

```python
reaction_smarts = "[C:1]=[C:2]>>[C:1]1O[C:2]1"
```

| 底物 | 现有 R04 产物 | 化学判断 |
|---|---|---|
| 马来酰亚胺 `O=C1C=CC(=O)N1` | `O=C1NC(=O)C2OC12` | ❌ **BMI 体系被打坏** |
| 丙烯酸甲酯 `C=CC(=O)OC` | `COC(=O)C1CO1` | ❌ 吸电子烯烃需碱性 H₂O₂，会破坏体系其他官能团 |

对照组：萘 `c1ccc2ccccc2c1`、蒽、联苯均返回 `[]`（芳构化感知正确，无误伤）。

### 1.4 分布失衡实测（桥联类主导）

**母核供给（无截断，`max_intermediates=10**9`）**：

| 阶段 | 母核类别 | 母核数 | 占比 |
|---|---|---|---|
| 阶段 1 双核桥联 | 拓扑衍生双酚 + 二胺 | **675** | **55.0%** |
| 阶段 1.2 稠环 | 稠环衍生多酚 + 多胺 | 267 | 21.8% |
| 阶段 2 星型 | 拓扑星型多酚 + 多胺 | **23** | **1.9%** |
| 阶段 3 空间四官能 | 特种空间刚性母核 | 95 | 7.7% |
| 阶段 4 杂化 | 不对称杂化母核 | 120 | 9.8% |
| 阶段 5 羧酸 | 多元羧酸衍生母核 | 47 | 3.8% |
| **合计** | | **1227** | 100% |

**产物级分布（3625 产物）**：

| 母核类别 | 母核数 | 产物数 | 占比 |
|---|---|---|---|
| 拓扑衍生双酚/二胺（桥联） | 675 | 1788 | **49.3%** |
| 稠环衍生多酚/多胺 | 267 | 774 | 21.3% |
| 特种空间刚性 | 95 | 458 | 12.6% |
| 不对称杂化 | 120 | 357 | 9.8% |
| 拓扑星型 | 23 | 101 | 2.8% |
| 多元羧酸 | 47 | 57 | 1.6% |

**根因 1｜组合空间先天失衡**
双核模型是 `11 骨架模板 × 17 桥联(valency==2) × 12 R基 = 2244` 的显式三重循环（`molecule_design.py:478` 起）；`poly_cores` 稠环仅 13 个带 `{r}` 槽模板 → 312 空间。前者是后者 7 倍。

**根因 2｜星型/四官能被硬编码为"仅苯环"**（`molecule_design.py:641`、`:676`）

```python
if _linker_enabled("methine_star") and _ring_enabled("benzene"):   # ← 仅苯环能长星型
```

后果：勾选芘/蒽/咔唑/金刚烷时**只能得到双核桥联形态**；更严重的是**取消勾选"苯环"会把所有星型/四官能母核一并清零**。

**根因 3｜`max_intermediates` 截断为先到先得**（`molecule_design.py:1502` 附近的 `len(intermediates) >= max_intermediates` 判定）
生成顺序为 双核 → 稠环 → 星型 → 空间 → 杂化 → 羧酸。实测：

```
max_intermediates=500  →  501 个母核 = 双酚 229 + 二胺 228 + 经典 44
                          （稠环 / 星型 / 空间 / 杂化 / 羧酸 全部饿死为 0）
```

UI 默认 `max_intermediates = max(500, max_total_products // 2)`，故「最大单体产物数」调至 500 时稠环与星型**整类消失**。

**附带缺陷｜4 个骨架环为死选项**
`mesitylene`、`biphenyl_33`、`cardo_bpf`、`s_triazine` 在 `ring_templates` 与 `poly_cores` 中**均无引用**。实测只勾选它们仍"新增 47 个母核"，但全部是**多元羧酸衍生母核**——因为 `include_acids` 段循环（`molecule_design.py:752` 起）**未接入 `_ring_enabled` 门控**，与骨架选择无关。两个缺陷叠加制造了"选项生效"的假象。

---

## 2. 羟基环氧化的化学真相：四条独立路线

"把羟基环氧化"在工业上不是单一路线，而是四条机理、试剂、三废完全不同的路线：

| 路线 | 工业方法 | 关键中间体 | 现状 |
|---|---|---|---|
| **P1** 直接 ECH 醚化 | 酚/醇 + 环氧氯丙烷 / NaOH | —（一步） | 有 R01，但**选择性错**（见 1.3） |
| **P2** 烯丙基化 → 环氧化 | 羟基 + 烯丙基溴 → 过氧酸环氧化 | O-烯丙基醚 | **新增** |
| **P3** 氯代醇法 | 羟基 + ECH 开环 → 闭环脱 HCl | 氯代醇 | **新增** |
| **P4** Sharpless 不对称环氧化 | 烯丙醇 + Ti(OiPr)₄ / (+)DET / TBHP | 环氧醇 | **新增**（**不做手性**，见 4.3） |

### 2.1 已验证 SMARTS（RDKit 2026.03.4 实测通过）

```python
# ── 选择性修复后的 P1（拆分为三条）──
P1a 酚羟基    "[c:1]-[OX2H:2]>>[c:1]-OCC1CO1"
P1b 脂肪醇    "[CX4:1]-[OX2H:2]>>[CX4:1]-OCC1CO1"
P1c 排除羧酸  "[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1"

# ── P2 两步链 ──
P2-S1 烯丙基化   "[c:1][OX2H:2]>>[c:1]OCC=C"
P2-S2 环氧化     "[C:1]=[C:2]>>[C:1]1O[C:2]1"

# ── P3 两步链 ──
P3-S1 ECH 开环   "[#6:1][OX2H:2]>>[#6:1]OCC(O)CCl"
P3-S2 闭环脱HCl  "[OX2:1][CH2:2][CH:3]([OX2H])[CH2:4][Cl:5]>>[OX2:1][CH2:2][CH:3]1[O][CH2:4]1"

# ── P4 单步（消旋）──
P4  Sharpless    "[CX4:1]([OX2H:2])[CX3:3]=[CX3:4]>>[CX4:1]([OX2H:2])[CX3:3]1[O][CX3:4]1"

# ── R04 选择性修复（排除吸电子烯烃）──
R04x "[C;!$(C=O):1]=[C;!$(C=O):2]>>[C:1]1O[C:2]1"
```

### 2.2 实测验证记录

**P1 选择性拆分**：

| 底物 | A1 原版（宽） | A3 仅酚羟基 | A5 仅脂肪醇 | A2 排除羧酸 |
|---|---|---|---|---|
| 对羟基苯甲酸 | `O=C(OCC1CO1)c1ccc(O)cc1` ❌ | `O=C(O)c1ccc(OCC2CO2)cc1` ✅ | `[]` ✅ | `O=C(O)c1ccc(OCC2CO2)cc1` ✅ |
| 己二酸 | `O=C(O)CCCCC(=O)OCC1CO1` ❌ | `[]` ✅ | `[]` ✅ | `[]` ✅ |
| 苯酚 | ✅ | ✅ | `[]` ✅ | ✅ |
| 乙二醇 | ✅ | `[]` ✅ | ✅ | ✅ |
| 叔丁醇 | ✅（不现实） | `[]` ✅ | ✅ | ✅ |

**P3 两步链端到端验证**：

```
苯酚 --P3-S1--> OC(CCl)COc1ccccc1   (氯代醇中间体)
     --P3-S2--> c1ccc(OCC2CO2)cc1
苯酚 --R01单步--> c1ccc(OCC2CO2)cc1
```

**两者 canonical SMILES 完全相同** —— 这是本设计的核心发现，直接决定 §3.3 的去重语义改造。

**P4 验证**：烯丙醇 `OCC=C` → `OCC1CO1` ✅（2-烯丙基苯酚返回 `[]`，因 `[CX4:1]([OX2H:2])[CX3:3]` 要求羟基与双键间为 sp3 碳，符合 Sharpless 底物范围）

**R04x 验证**：`!$(C=O)` 只约束原子自身，而丙烯酸酯的 α-碳**自身不是羰基碳、只是邻接羰基**，因此 `[C;!$(C=O):1]=[C;!$(C=O):2]` 仍会氧化马来酰亚胺/丙烯酸酯/丙烯腈。**必须使用递归 SMARTS 排除邻接吸电子基**，见 §4.2（已实测出可用解）。

---

## 3. 设计方案

### Part A｜骨架多样性修复

| 编号 | 改动 | 位置 |
|---|---|---|
| **A1** | 星型/四官能模型从"仅苯环"泛化为**遍历任意已选环骨架**；`_ring_enabled("benzene")` 门控改为"至少一个已选环可用" | `molecule_design.py:641`、`:676` |
| **A2** | 截断改为**类别配额制**：为 6 个母核类别各设保底配额，替代先到先得 | `generate_scaffold_intermediates` 返回前 |
| **A3** | `include_acids` 段接入 `_ring_enabled` 门控，消除死选项假象 | `molecule_design.py:752` 起 |
| **A4** | 为 4 个死选项补齐模板：`mesitylene`、`biphenyl_33`、`cardo_bpf`、`s_triazine` | `ring_templates` / `poly_cores` |

**A2 配额设计**（默认值，UI 可调）：

| 类别 | 配额占比 | 理由 |
|---|---|---|
| 双核桥联 | 40% | 保留工业主流（双酚A/F/S 体系） |
| 稠环骨架 | 25% | 提升耐热/刚性骨架供给 |
| 空间四官能 | 15% | 卡基/螺环/四苯甲烷 |
| 不对称杂化 | 10% | 苯-萘杂化 |
| 星型拓扑 | 5% | 数量天然少（23 个），按绝对量保底 ≥ 全部 |
| 多元羧酸 | 5% | 酸酐/缩水甘油酯双路线 |

**A4 模板补充**（SMARTS/SMILES 均已实测解析通过，官能度经 `calculate_stoichiometry` 复核）：

```python
# ring_templates 追加（{lk}=桥联, {r}=R基）
("mesitylene",  "均三甲苯基", "mesitylene",  "Cc1cc(C)cc({link})c1",
 "Oc1cc{r}cc(C)c1{lk}c1cc(O)cc(C)c1C", "Nc1cc{r}cc(C)c1{lk}c1cc(N)cc(C)c1C"),
("biphenyl_33", "3,3'-联苯",  "biphenyl_33", "c1cccc(-c2cccc({link})c2)c1",
 "Oc1cccc(-c2cccc({lk}c3cccc(O)c3)c2)c1", "Nc1cccc(-c2cccc({lk}c3cccc(N)c3)c2)c1"),

# poly_cores 追加（注意：r_str 自身已含括号，模板中写 {r} 而非 ({r})，
#           否则产生 "((C))" 双括号导致 SMILES 解析失败）
("cardo_bpf",  "卡基芴",   "cardo_bpf",
 "Oc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1", "Nc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1"),
("s_triazine", "对称三嗪", "s_triazine",
 "Oc1nc(O)nc{r}n1", "Nc1nc(N)nc{r}n1"),
```

实测复核结果：

| 模板 | 产物 | 官能度 | 当量 | MW | 体系 |
|---|---|---|---|---|---|
| mesitylene 双酚 | `Cc1cc(C)c(C(C)(C)c2cc(O)cc(C)c2C)c(O)c1` | 2 | 142.2 | 284.4 | phenol |
| biphenyl_33 双酚 | `CC(C)(c1cccc(O)c1)c1cccc(-c2cccc(O)c2)c1` | 2 | — | — | phenol |
| cardo_bpf 双酚 | `Oc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1` | 2 | — | — | phenol |
| s_triazine 多酚 | `Oc1nc(O)nc(C)n1` | 2 | 63.55 | 127.1 | phenol |
| s_triazine 多胺 | `Nc1nc(N)nc(C)n1` | 4 | 31.28 | 125.14 | amine |

**注意**：`s_triazine` 多酚当量仅 63.55 g/eq，远低于常规环氧固化剂（双酚A 型约 114），单分子含三嗪环密交联点。此为**有意设计**（高交联度阻燃骨架），但需在 UI 中提示其高反应活性。

### Part B｜路线感知重构（核心）

#### 3.1 B1｜新增数据结构

```python
@dataclass
class RouteStep:
    step_index: int
    reaction_id: str
    reaction_smarts: str
    intermediate_smiles: str
    requires_reagent: str            # "环氧氯丙烷" / "烯丙基溴" / "过氧酸" / "TBHP"
    hazard_flags: tuple[str, ...]    # ("剧毒", "含氯废水", "强放热")

@dataclass
class SyntheticRoute:
    route_id: str
    name: str
    chemical_system: str             # "epoxy" | "amine" | ...
    target_role: str
    steps: list[RouteStep]
    n_steps: int
    atom_economy: float              # 原子经济性 (0~1)
    green_score: float               # 绿色化学评分 (0~10，越高越绿)
    default_enabled: bool = True
```

#### 3.2 B2｜`ROUTE_CHAINS` 注册表

将 17 条扁平模板重组为**反应网络**（节点=中间体，边=反应）。羟基环氧化落为 4 条路线：

```python
ROUTE_CHAINS: list[SyntheticRoute] = [
    SyntheticRoute("P1_phenol_ech", "酚羟基直接ECH醚化", "epoxy", "resin",
        steps=[RouteStep(0, "glycidyl_etherification_phenol", P1a, "", "环氧氯丙烷", ("剧毒","含氯废水"))],
        n_steps=1, atom_economy=0.42, green_score=4.5),
    SyntheticRoute("P2_allyl_epoxidation", "烯丙基化-过氧酸环氧化", "epoxy", "resin",
        steps=[RouteStep(0, "allyl_etherification", P2_S1, "", "烯丙基溴", ("催泪",)),
               RouteStep(1, "allyl_epoxidation", P2_S2, "O-烯丙基醚中间体", "过氧酸", ("易爆","强氧化"))],
        n_steps=2, atom_economy=0.55, green_score=6.0),
    SyntheticRoute("P3_chlorohydrin", "氯代醇法闭环", "epoxy", "resin",
        steps=[RouteStep(0, "chlorohydrin_ring_open", P3_S1, "", "环氧氯丙烷", ("剧毒",)),
               RouteStep(1, "chlorohydrin_closure", P3_S2, "氯代醇中间体", "NaOH", ("含盐废水",))],
        n_steps=2, atom_economy=0.48, green_score=5.0),
    SyntheticRoute("P4_sharpless", "Sharpless不对称环氧化(消旋)", "epoxy", "resin",
        steps=[RouteStep(0, "sharpless_epoxidation", P4, "环氧醇中间体", "Ti(OiPr)4/TBHP", ("重金属","强氧化"))],
        n_steps=1, atom_economy=0.60, green_score=5.5),
]
```

#### 3.3 B3｜去重语义：从"丢弃"改为"累加路线"（**关键改动**）

**现状**（`run_combinatorial_monomer_design`）：

```python
if item.product_smiles not in seen_smiles:
    seen_smiles.add(item.product_smiles)
    results.append(item)          # ← 同产物只保留第一条命中，路线信息静默丢失
```

**改为**：

```python
if item.product_smiles in seen_by_smiles:
    existing = seen_by_smiles[item.product_smiles]
    existing.route_ids.append(item.route_id)         # 累加备选路线
    existing.route_steps_min = min(existing.route_steps_min, item.route_steps)
    existing.green_score_best = max(existing.green_score_best, item.green_score)
    existing.route_redundancy = len(set(existing.route_ids))
else:
    seen_by_smiles[item.product_smiles] = item
    results.append(item)
```

**产出新特征**（可直接进入后续筛选排序）：

| 字段 | 含义 | 工业价值 |
|---|---|---|
| `route_ids` | 该产物的全部可行路线列表 | 备选工艺 |
| `route_redundancy` | 路线冗余度（路线条数） | **数值越高，放大风险越低** |
| `route_steps_min` | 最短路线步数 | 成本下限 |
| `green_score_best` | 最优绿色化学评分 | EHS 合规 |

#### 3.4 B4｜中间体入库（用户已确认）

- 中间体以 `role="intermediate"` 单独入库。
- **无需额外门控**：配方筛选只消费 `role=="resin"` 与 `role=="hardener"`（实测 `app_lib.py:25315-25316`），`intermediate` 天然被排除，不会污染树脂/固化剂候选池。
- 中间体仍计算 `sa_score` / `mw` / `formula`，进路线图谱与合成可及性评估。
- 中间体**不参与** EEW/AHEW 当量计算（`functionality` 置 0，`equivalent_weight` 置 0.0）。

#### 3.5 B5｜选择性修复（硬伤，必改）

- R01 拆分为 `glycidyl_etherification_phenol` / `_aliphatic` / `_non_carboxyl` 三条。
- R04 收紧为 `R04x`；**当前 `[C;!$(C=O):1]=[C;!$(C=O):2]` 实测仍会氧化马来酰亚胺与丙烯酸甲酯**，需在实现阶段改用显式白名单式：仅匹配非共轭、非吸电子取代的 C=C（见 §4.2）。

#### 3.6 B6｜UI 扩展（`app_lib.py:27970`）

在现有「细分有机合成转化路径 (可单独微调)」multiselect 下方新增：

1. **路线选择器**：按 `SyntheticRoute` 而非扁平 `reaction_id` 选择，显示步数/绿色评分/危险标记。
2. **路线树可视化**：产物 → 中间体 → 前驱体的有向图。
3. **新增筛选维度**：`route_steps_min`（路线步数上限）、`green_score_best`（绿色评分下限）、`route_redundancy`（路线冗余度下限）。
4. **产物表新增列**：`route_ids`、`route_redundancy`、`route_steps_min`、`green_score_best`。

---

## 4. 实现约束与待办

### 4.1 数据兼容

- `CombinatorialProduct` 新增字段须设默认值，保证 `cache/persisted_designed_monomers.parquet` 旧数据可读（`pd.read_parquet` 后缺失列填默认）。
- `reaction_id` / `reaction_name` **保留不删**，作为 `route_ids[0]` 的向后兼容视图。

### 4.2 R04 选择性收紧（**已实测解决**）

**失败的方案**（实测均 FAIL，仅原子自身约束，无法覆盖 α,β-不饱和体系）：

```python
"[C;!$(C=O):1]=[C;!$(C=O):2]>>[C:1]1O[C:2]1"              # 丙烯酸甲酯/丙烯腈仍被氧化
"[CX4,CX3;!$(C=O):1]=[CX4,CX3;!$(C=O):2]>>..."            # 同上
"[C;!$(C=O);!$(C#N):1]=[C;!$(C=O);!$(C#N):2]>>..."        # 同上
```

**根因**：丙烯酸甲酯 `C=CC(=O)OC` 中，双键的 α-碳（原子 2）**自身不是羰基碳**，只是**邻接**羰基。任何只写在原子上的 `!$(C=O)` 都无法排除它。

**可用解**（RDKit 2026.03.4 实测，负样本 6/6 排除、正样本 7/7 保留）：

```python
# 递归排除：原子自身非羰基碳，且不邻接羰基碳，且不邻接腈基碳
_R04_EXCL = ("[!$([CX3]=[OX1])"
             ";!$([CX4,CX3][CX3]=[OX1])"
             ";!$([CX4,CX3][CX2]#[NX1])")

R04_selective = f"{_R04_EXCL}:1]={_R04_EXCL}:2]>>[C:1]1O[C:2]1"
```

> ⚠️ **原子映射位置关键**：`:1` / `:2` 必须写在**外层方括号的最内层**（即 `[...:1]`），若漏写映射，产物会退化为环氧乙烷 `C1CO1`（实测踩过此坑）。

**回归验证矩阵**：

| 类型 | 底物 | SMILES | 期望 | 实测 |
|---|---|---|---|---|
| 负样本 | 马来酰亚胺 | `O=C1C=CC(=O)N1` | `[]` | `[]` ✅ |
| 负样本 | 丙烯酸甲酯 | `C=CC(=O)OC` | `[]` | `[]` ✅ |
| 负样本 | 肉桂酸甲酯 | `O=C(OC)/C=C/c1ccccc1` | `[]` | `[]` ✅ |
| 负样本 | 肉桂酸 | `O=C(O)/C=C/c1ccccc1` | `[]` | `[]` ✅ |
| 负样本 | 肉桂醛 | `O=C/C=C/c1ccccc1` | `[]` | `[]` ✅ |
| 负样本 | 甲基丙烯酸甲酯 | `C=C(C)C(=O)OC` | `[]` | `[]` ✅ |
| 负样本 | 丙烯腈 | `C=CC#N` | `[]` | `[]` ✅ |
| 负样本 | 甲基乙烯基酮 | `C=CC(C)=O` | `[]` | `[]` ✅ |
| 负样本 | 马来酸酐 | `O=C1OC(=O)C=C1` | `[]` | `[]` ✅ |
| 正样本 | 环己烯 | `C1CC=CCC1` | 环氧环己烷 | `C1CCC2OC2C1` ✅ |
| 正样本 | 烯丙基苯酚 | `C=CCC1=CC=CC=C1O` | 保留酚羟基 | `Oc1ccccc1CC1CO1` ✅ |
| 正样本 | DCPD | `C1CC2C=CC1C2` | 单环氧 | `C1CC2CC1C1OC21` ✅ |
| 正样本 | 苯乙烯 | `C=Cc1ccccc1` | 芳环不受影响 | `c1ccc(C2CO2)cc1` ✅ |
| 正样本 | 1-辛烯 | `CCCCCCC=C` | 端烯环氧化 | `CCCCCCC1CO1` ✅ |
| 正样本 | 异丁烯 | `CC(C)=C` | 端烯环氧化 | `CC1(C)CO1` ✅ |
| 正样本 | 降冰片烯 | `C1CC2CC1C=C2` | 单环氧 | `C1CC2CC1C1OC21` ✅ |

**原版 R04 误伤实证**（对比用，证明修复必要）：

| 底物 | 原版 `[C:1]=[C:2]>>[C:1]1O[C:2]1` 产物 | 后果 |
|---|---|---|
| 真肉桂酸甲酯 | `COC(=O)C1OC1c1ccccc1` | ❌ 不饱和酯被环氧化 |
| 真肉桂酸 | `O=C(O)C1OC1c1ccccc1` | ❌ 不饱和酸被环氧化 |
| 肉桂醛 | `O=CC1OC1c1ccccc1` | ❌ 醛基共存下被环氧化 |
| 甲基丙烯酸甲酯 | `COC(=O)C1(C)CO1` | ❌ 不饱和酯被环氧化 |

> 上述 4 个底物在修复版 `R04_selective` 下均正确返回 `[]`。

### 4.3 P4 手性（用户明确不做）

- `useChirality` 不启用，P4 产物为**消旋体**，不做立体化学标注。
- 路线名标注"(消旋)"，避免误导为不对称合成。

### 4.4 不改动范围

- 不改 `core/reaction_simulator.py`（交联反应模拟，与分子生成解耦）。
- 不改配方级筛选流水线（`core/virtual_screening.py`），仅通过 `role` 字段自然隔离中间体。

---

## 5. 验收标准

| 编号 | 验收项 | 判定方式 |
|---|---|---|
| V1 | R01 不再醚化羧酸 | 己二酸 → `[]`；对羟基苯甲酸 → 仅酚羟基产物 |
| V2 | R04 不再氧化吸电子烯烃 | 马来酰亚胺/丙烯酸甲酯/丙烯腈/马来酸酐 → `[]`；环己烯/烯丙基苯酚/DCPD/降冰片烯 → 正常环氧化（见 §4.2 回归矩阵） |
| V3 | P2/P3 两步链端到端可跑通 | 苯酚经 P3 得 `c1ccc(OCC2CO2)cc1`，与 P1 一致 |
| V4 | 同产物多路线累加 | 苯酚环氧产物 `route_ids` 同时含 P1 与 P3 |
| V5 | 中间体入库且不污染配方筛选 | 产物表含 `role=="intermediate"`；配方筛选候选池不含之 |
| V6 | 星型母核随任意环骨架生成 | 勾选芘/金刚烷时星型母核数 > 0 |
| V7 | 取消勾选苯环不再清零星型 | `selected_rings=["naphthalene_14"]` 时星型母核数 > 0 |
| V8 | 小 `max_intermediates` 不再饿死类别 | `max_intermediates=500` 时稠环/星型/羧酸均 > 0 |
| V9 | 4 个死选项生效 | 只勾 `mesitylene` 时出现含均三甲苯的母核 |
| V10 | 旧 parquet 缓存可读 | 加载旧 `persisted_designed_monomers.parquet` 不抛异常 |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 路线链使产物数暴增，超 `max_total_products` | 路线链独立计数，不占扁平模板配额；UI 分开展示 |
| `route_ids` 累加导致 DataFrame 列含 list，parquet 序列化问题 | 序列化为 `;` 分隔字符串存储，读取时 split |
| 中间体数量大（每步 1 个）拖慢穷举 | 中间体仅在路线链启用时生成；单步路线无中间体 |
| A1 泛化星型后母核数激增 | 受 A2 类别配额约束，总母核数上限不变 |
