# 合成转化路径细分与骨架多样性修复 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将「全维度拓扑演化设计引擎」的转化路径从扁平单步模板升级为路线感知体系（羟基环氧化 P1-P4 路线链 + 中间体入库），并修复产物分布失衡（桥联为主、骨架稀少）。

**Architecture:** 所有核心改动集中在 `core/molecule_design.py`；新增 `SyntheticRoute`/`RouteStep` 数据结构与 `ROUTE_CHAINS` 注册表；产物去重从"丢弃"改为"累加路线"；`generate_scaffold_intermediates` 增加类别配额与环臂模板泛化；UI 在 `app_lib.py:27970` 附近扩展。

**Tech Stack:** Python 3.13 + RDKit 2026.03.4 + pytest 9.1.1 + Streamlit。

**Spec:** `docs/superpowers/specs/2026-09-22-synthetic-route-and-scaffold-diversity-design.md`

## Global Constraints

- 测试命令一律 `python -m pytest <path> -v`（本机 python 直接可用，pytest 9.1.1 已安装）。
- commit 信息风格 `feat: 中文描述`，**只 `git add` 本任务改动的文件**。工作区当前有与本任务无关的未提交修改（`UserPrediction.py`、`core/portal_prediction.py`、`core/prediction_portal.py`、`tests/test_portal_feature_sources.py`、`tests/test_portal_prediction.py`、`.tmp_e2e_tg_check.py`），**任何任务都不得 add/commit 这些文件**。
- RDKit 反应 SMARTS 的原子映射必须写在外层方括号最内层（如 `[...:1]`），漏写映射产物会退化为环氧乙烷 `C1CO1`。
- SMILES 模板中 `{r}` 槽位不加括号——`r_str` 变量自带括号 `(C)`，写成 `({r})` 会产生 `((C))` 解析失败。
- 全部 SMARTS/SMILES 已在本机 RDKit 2026.03.4 实测验证过，见各任务 Step 中的"预期输出"列；实现时以实测为准。

---

### Task 1: R01 选择性修复（保留旧 id，新增两条细分模板）

**Files:**
- Modify: `core/molecule_design.py:205-214`（`SYNTHETIC_REACTION_TEMPLATES` 中 `glycidyl_etherification` 条目）
- Test: `tests/test_reaction_selectivity.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: reaction_id 保持 `glycidyl_etherification`（SMARTS 收紧为排除羧酸版），新增 `glycidyl_etherification_phenol`、`glycidyl_etherification_aliphatic`、`glycidyl_etherification_non_carboxyl`。后续 Task 3 的 P1 路线引用 `glycidyl_etherification_non_carboxyl`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_reaction_selectivity.py`：

```python
"""反应模板选择性回归：R01 羟基醚化不许误伤羧酸，R04 环氧化不许误伤吸电子烯烃。"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from core.molecule_design import SYNTHETIC_REACTION_TEMPLATES


def _rxn(reaction_id):
    tpl = next(t for t in SYNTHETIC_REACTION_TEMPLATES if t.reaction_id == reaction_id)
    return rdChemReactions.ReactionFromSmarts(tpl.reaction_smarts)


def _products(reaction_id: str, smiles: str) -> list[str]:
    """单步应用反应模板，返回产物 canonical SMILES 列表（空=不反应）。"""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"底物解析失败 {smiles}"
    out = []
    for tup in _rxn(reaction_id).RunReactants((mol,)):
        try:
            p = tup[0]
            Chem.SanitizeMol(p)
            out.append(Chem.MolToSmiles(p))
        except Exception:
            continue
    return out


def _has_frag(smiles_prod: str, smarts: str) -> bool:
    m = Chem.MolFromSmiles(smiles_prod)
    return m is not None and m.HasSubstructMatch(Chem.MolFromSmarts(smarts))


def test_r01_no_longer_touches_carboxyl():
    """己二酸必须整类拒绝；对羟基苯甲酸只准醚化酚羟基、保留羧基。"""
    assert _products("glycidyl_etherification", "O=C(O)CCCCC(=O)O") == [], \
        "己二酸被醚化，R01 与 R03 抢活"
    prods = _products("glycidyl_etherification", "O=C(O)c1ccc(O)cc1")
    assert prods, "对羟基苯甲酸的酚羟基应被醚化"
    for p in prods:
        assert _has_frag(p, "C(=O)[OX2H]"), f"羧基被破坏: {p}"


def test_r01a_phenol_only():
    """细分模板：仅酚羟基版不许碰脂肪醇。"""
    assert _products("glycidyl_etherification_phenol", "OCCO") == []
    assert _products("glycidyl_etherification_phenol", "Oc1ccccc1"), "苯酚应被醚化"


def test_r01b_aliphatic_only():
    """细分模板：仅脂肪醇版不许碰酚羟基与羧酸。"""
    assert _products("glycidyl_etherification_aliphatic", "Oc1ccccc1") == []
    assert _products("glycidyl_etherification_aliphatic", "O=C(O)CCCCC(=O)O") == []
    assert _products("glycidyl_etherification_aliphatic", "OCCO"), "乙二醇应被醚化"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_reaction_selectivity.py -v`
Expected: FAIL — `KeyError`/`StopIteration`，因新 reaction_id 尚不存在。

- [ ] **Step 3: 实现**

在 `core/molecule_design.py` 中，把 `glycidyl_etherification` 条目的 `reaction_smarts` 由
`"[#6:1]-[OX2H:2]>>[#6:1]-OCC1CO1"` 改为 `"[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1"`，
并在该条目后追加三条：

```python
    SyntheticReaction(
        reaction_id="glycidyl_etherification_non_carboxyl",
        name="羟基缩水甘油醚化·排除羧酸 (R01c)",
        target_role="resin",
        reaction_smarts="[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1",
        description="酚/脂肪醇缩水甘油醚化，显式排除羧酸（与 R03 分工，供 P1 路线链引用）",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_etherification_phenol",
        name="酚羟基缩水甘油醚化 (R01a)",
        target_role="resin",
        reaction_smarts="[c:1]-[OX2H:2]>>[c:1]-OCC1CO1",
        description="仅酚羟基 + 环氧氯丙烷生成缩水甘油醚",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_etherification_aliphatic",
        name="脂肪醇缩水甘油醚化 (R01b)",
        target_role="resin",
        reaction_smarts="[CX4:1]-[OX2H:2]>>[CX4:1]-OCC1CO1",
        description="仅脂肪醇 + 环氧氯丙烷生成缩水甘油醚",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_reaction_selectivity.py -v`
Expected: 新测试 PASS。再跑 `python -m pytest tests/test_combinatorial_molecule_design.py -v` 确认旧测试不回归（`glycidyl_etherification` id 保留，SMARTS 收紧对脂肪二醇仍有效）。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_reaction_selectivity.py
git commit -m "feat: R01 羟基醚化选择性修复，排除羧酸并细分酚/脂肪醇模板"
```

---

### Task 2: R04 选择性修复（递归 SMARTS 排除吸电子烯烃）

**Files:**
- Modify: `core/molecule_design.py:249`（`olefin_epoxidation` 条目）
- Test: `tests/test_reaction_selectivity.py`（追加）

**Interfaces:**
- Consumes: 无
- Produces: `olefin_epoxidation` 的 SMARTS 收紧；Task 3 的 P2 路线第二步及 Task 5 的多步链复用该反应 id。

- [ ] **Step 1: 写失败测试**

`tests/test_reaction_selectivity.py` 追加：

```python
R04_NEG = {
    "马来酰亚胺": "O=C1C=CC(=O)N1",
    "丙烯酸甲酯": "C=CC(=O)OC",
    "肉桂酸甲酯": "O=C(OC)/C=C/c1ccccc1",
    "肉桂酸": "O=C(O)/C=C/c1ccccc1",
    "肉桂醛": "O=C/C=C/c1ccccc1",
    "丙烯腈": "C=CC#N",
    "甲基乙烯基酮": "C=CC(C)=O",
    "马来酸酐": "O=C1OC(=O)C=C1",
    "甲基丙烯酸甲酯": "C=C(C)C(=O)OC",
}
R04_POS = {
    "环己烯": "C1CC=CCC1",
    "烯丙基苯酚": "C=CCC1=CC=CC=C1O",
    "DCPD": "C1CC2C=CC1C2",
    "苯乙烯": "C=Cc1ccccc1",
    "1-辛烯": "CCCCCCC=C",
    "异丁烯": "CC(C)=C",
    "降冰片烯": "C1CC2CC1C=C2",
}


def test_r04_rejects_electron_poor_alkenes():
    for name, smi in R04_NEG.items():
        prods = _products("olefin_epoxidation", smi)
        assert prods == [], f"{name} 不应被环氧化，实际得到 {prods}"


def test_r04_still_epoxidizes_common_alkenes():
    epoxide_patt = Chem.MolFromSmarts("[O;r3]1[C;r3][C;r3]1")
    for name, smi in R04_POS.items():
        prods = _products("olefin_epoxidation", smi)
        assert prods, f"{name} 应被环氧化"
        for p in prods:
            m = Chem.MolFromSmiles(p)
            assert m is not None and m.HasSubstructMatch(epoxide_patt), f"{name} 产物无环氧: {p}"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_reaction_selectivity.py::test_r04_rejects_electron_poor_alkenes -v`
Expected: FAIL — 马来酰亚胺/丙烯酸酯等被环氧化。实测已知：`O=C1NC(=O)C2OC12`、`COC(=O)C1CO1` 出现。

- [ ] **Step 3: 实现**

把 `olefin_epoxidation` 的 `reaction_smarts` 改为：

```python
reaction_smarts=(
    "[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):1]"
    "=[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):2]"
    ">>[C:1]1O[C:2]1"
),
```

> 化学说明：`!$([CX3]=[OX1])` 排除羰基碳自身；`!$([CX4,CX3][CX3]=[OX1])` 排除**邻接**羰基的 α-碳（丙烯酸酯/肉桂酸酯/马来酰亚胺的根因）；`!$([CX4,CX3][CX2]#[NX1])` 排除邻接腈基。原子映射 `:1`/`:2` 必须写在外层 `]` 之前。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_reaction_selectivity.py -v`
Expected: 全部 PASS（负样本 9/9 排除、正样本 7/7 环氧化）。再跑 `python -m pytest tests/test_combinatorial_molecule_design.py -v` 确认全量旧测试不回归。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_reaction_selectivity.py
git commit -m "feat: R04 环氧化排除吸电子烯烃，BMI/丙烯酸酯不再被误氧化"
```

---

### Task 3: SyntheticRoute/RouteStep 数据结构与 ROUTE_CHAINS 注册表

**Files:**
- Modify: `core/molecule_design.py`（在 `SYNTHETIC_REACTION_TEMPLATES` 定义之后新增一段，约 :370 处）
- Test: `tests/test_synthetic_routes.py`（新建）

**Interfaces:**
- Consumes: `SyntheticReaction`（Task 1/2 之后的模板库）
- Produces: `RouteStep`、`SyntheticRoute`、`ROUTE_CHAINS`、`hydroxyl_epoxidation_routes()`。Task 4/5 消费 `ROUTE_CHAINS`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_synthetic_routes.py`：

```python
"""路线感知体系：ROUTE_CHAINS 注册表、四羟基环氧化路线、端到端等效。"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from core.molecule_design import ROUTE_CHAINS


def test_routes_registered():
    ids = {r.route_id for r in ROUTE_CHAINS}
    assert {"P1_phenol_ech", "P2_allyl_epoxidation", "P3_chlorohydrin", "P4_sharpless"} <= ids


def test_route_smarts_compile():
    for route in ROUTE_CHAINS:
        for step in route.steps:
            if step.reaction_smarts:
                rxn = rdChemReactions.ReactionFromSmarts(step.reaction_smarts)
                assert rxn is not None, f"{route.route_id}.{step.step_index} 编译失败"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_synthetic_routes.py -v`
Expected: FAIL — `ImportError: cannot import name 'ROUTE_CHAINS'`。

- [ ] **Step 3: 实现**

在 `core/molecule_design.py` 的 `SYNTHETIC_REACTION_TEMPLATES` 列表结束之后（`phenolic_hardener_retain` 条目之后、Precursor Catalog 段之前）插入：

```python
# ==============================================================================
# 2b. 路线感知合成路径 (Synthetic Routes)：将扁平反应模板组织为多步反应网络
# ==============================================================================

@dataclass
class RouteStep:
    """合成路线中的单个反应步骤。"""
    step_index: int
    reaction_id: str
    reaction_smarts: str
    intermediate_label: str = ""           # 中间体的可读标签（仅用于 UI 展示，非 SMILES）
    requires_reagent: str = ""             # "环氧氯丙烷" / "烯丙基溴" / "过氧酸" / "NaOH" ...
    hazard_flags: tuple[str, ...] = ()     # ("剧毒", "含氯废水", "强放热")


@dataclass
class SyntheticRoute:
    """从前驱体到目标单体的完整合成路线。

    steps 顺序执行：step[i] 的产物是 step[i+1] 的底物。
    中间体的真实 SMILES 由路径执行器在执行时产出（见 _expand_route_chain），
    不存放在本结构中；step.intermediate_label 仅为 UI 可读标签。
    """
    route_id: str
    name: str
    chemical_system: str                  # "epoxy" | "amine" | ...
    target_role: str                      # "resin" | "hardener"
    steps: list[RouteStep]
    atom_economy: float = 0.5             # 原子经济性 (0~1)
    green_score: float = 5.0              # 绿色化学评分 (0~10)
    default_enabled: bool = True

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# 羟基环氧化四条真实工业路线（P1-P4），均默认启用
_P1_SMARTS = "[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1"

ROUTE_CHAINS: list[SyntheticRoute] = [
    SyntheticRoute(
        route_id="P1_phenol_ech",
        name="P1 直接ECH醚化 (酚/醇→缩水甘油醚)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "glycidyl_etherification_non_carboxyl", _P1_SMARTS,
                      requires_reagent="环氧氯丙烷", hazard_flags=("剧毒", "含氯废水")),
        ],
        atom_economy=0.42, green_score=4.5,
    ),
    SyntheticRoute(
        route_id="P2_allyl_epoxidation",
        name="P2 烯丙基化-环氧化 (两步, 绿色无氯)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "allyl_etherification",
                      "[c:1][OX2H:2]>>[c:1]OCC=C",
                      requires_reagent="烯丙基溴", hazard_flags=("催泪",)),
            RouteStep(1, "olefin_epoxidation_selective",
                      "[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):1]"
                      "=[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):2]"
                      ">>[C:1]1O[C:2]1",
                      intermediate_label="O-烯丙基醚中间体",
                      requires_reagent="过氧酸", hazard_flags=("易爆", "强氧化")),
        ],
        atom_economy=0.55, green_score=6.0,
    ),
    SyntheticRoute(
        route_id="P3_chlorohydrin",
        name="P3 氯代醇法闭环 (两步, 机理清晰)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "chlorohydrin_ether",
                      "[#6;!$(C=O):1][OX2H:2]>>[#6:1]OCC(O)CCl",
                      requires_reagent="环氧氯丙烷", hazard_flags=("剧毒",)),
            RouteStep(1, "chlorohydrin_closure",
                      "[OX2:1][CH2:2][CH:3]([OX2H])[CH2:4][Cl:5]>>[OX2:1][CH2:2][CH:3]1[O][CH2:4]1",
                      intermediate_label="氯代醇中间体",
                      requires_reagent="NaOH", hazard_flags=("含盐废水",)),
        ],
        atom_economy=0.48, green_score=5.0,
    ),
    SyntheticRoute(
        route_id="P4_sharpless",
        name="P4 Sharpless环氧化 (消旋, 烯丙醇→环氧醇)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "sharpless_epoxidation",
                      "[CX4:1]([OX2H:2])[CX3:3]=[CX3:4]>>[CX4:1]([OX2H:2])[CX3:3]1[O][CX3:4]1",
                      intermediate_label="环氧醇",
                      requires_reagent="Ti(OiPr)4/TBHP", hazard_flags=("重金属", "强氧化")),
        ],
        atom_economy=0.60, green_score=5.5,
    ),
]


def hydroxyl_epoxidation_routes() -> list[SyntheticRoute]:
    """返回全部'羟基→环氧'转化路径（供 UI 与流水线引用）。"""
    return [r for r in ROUTE_CHAINS if r.chemical_system == "epoxy"]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_synthetic_routes.py -v`
Expected: 2 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_synthetic_routes.py
git commit -m "feat: 新增 SyntheticRoute/RouteStep 与羟基环氧化四条路线注册表"
```

---

### Task 4: CombinatorialProduct 路线字段 + 路线链多步执行器 + 中间体入库

**Files:**
- Modify: `core/molecule_design.py`（在 `_make_product_record` 之后新增 `_expand_route_chain`；在 `run_combinatorial_monomer_design` 中调用）
- Test: `tests/test_synthetic_routes.py`（追加）

**Interfaces:**
- Consumes: `ROUTE_CHAINS`、`_make_product_record`、`calculate_stoichiometry`、`PrecursorCore`、`CombinatorialProduct`
- Produces: `CombinatorialProduct.route_ids/route_names/route_steps_min/route_redundancy/green_score_best`（dataclass 新字段）、`_make_product_record(..., route=None)`、`_expand_route_chain(parent_mol, route, precursor, mw_range) -> list[CombinatorialProduct]`（含末端产物与中间体）、`_process_single_precursor_reactions(..., enable_route_chains=True, route_chain_ids=None)`、`run_combinatorial_monomer_design(..., enable_route_chains=True, route_chain_ids=None)`

> dataclass 需 `from dataclasses import dataclass, field`（补 `field` 导入）。去重合并逻辑（`_merge_product_records`）留给 Task 5，本任务 mock 时其行为仍为 seen_smiles 首命中去重。

- [ ] **Step 1: 写失败测试**

`tests/test_synthetic_routes.py` 追加：

```python
from core.molecule_design import (
    _expand_route_chain,
    PrecursorCore,
    ROUTE_CHAINS,
)


def test_p3_chain_end_to_end_equals_p1():
    """苯酚 --P3两步--> 与 P1 单步产物 canonical 等价；中间体以 intermediate 角色入库。"""
    phenol = PrecursorCore(core_id="test_phenol", name="苯酚", category="测试",
                           role="resin", smiles="Oc1ccccc1", description="", default_selected=True)
    p1 = next(r for r in ROUTE_CHAINS if r.route_id == "P1_phenol_ech")
    p3 = next(r for r in ROUTE_CHAINS if r.route_id == "P3_chlorohydrin")
    out1 = _expand_route_chain(Chem.MolFromSmiles("Oc1ccccc1"), p1, phenol, mw_range=(50.0, 1500.0))
    out3 = _expand_route_chain(Chem.MolFromSmiles("Oc1ccccc1"), p3, phenol, mw_range=(50.0, 1500.0))
    end1 = [p for p in out1 if p.role != "intermediate"]
    end3 = [p for p in out3 if p.role != "intermediate"]
    assert end1 and end3
    assert end1[0].product_smiles == end3[0].product_smiles == "c1ccc(OCC2CO2)cc1"
    mids = [p for p in out3 if p.role == "intermediate"]
    assert mids, "P3 应产出氯代醇中间体"
    assert mids[0].product_smiles == "OC(CCl)COc1ccccc1"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_synthetic_routes.py::test_p3_chain_end_to_end_equals_p1 -v`
Expected: FAIL — `ImportError: cannot import name '_expand_route_chain'`。

- [ ] **Step 3: 实现**

1) 先扩展 `CombinatorialProduct` dataclass（Task 5 的前置字段，本任务与 Task 5 合并提交）：

```python
    route_ids: list[str] = field(default_factory=list)   # 全部可行路线 id
    route_names: list[str] = field(default_factory=list)
    route_steps_min: int = 1                               # 最短路线步数
    route_redundancy: int = 1                              # 路线冗余度
    green_score_best: float = 0.0                          # 最优绿色评分
```

（`from dataclasses import dataclass, field` 补 `field` 导入；同时给 `_make_product_record` 加尾参 `route: SyntheticRoute | None = None`，把上述 5 个字段写入构造。）

2) 在 `_make_product_record` 之后新增（干净版，无冗余循环）：

```python
_ALLOWED_ELEMENTS_ROUTE = frozenset(('C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si'))


def _expand_route_chain(
    parent_mol,
    route: SyntheticRoute,
    precursor: PrecursorCore,
    mw_range: tuple[float, float],
) -> list[CombinatorialProduct]:
    """沿多步路线顺序执行 Reaction SMARTS。

    - 中间步骤产物 → role="intermediate" 入库（无官能度/当量需求，不进配方筛选）；
    - 最后一步产物 → 以 route.target_role 入库并计算 EEW/AHEW；
    - 任意一步不匹配 → 返回空列表（断链即整体失败，绝不允许未反应底物冒充产物）。
    """
    from .task_manager import is_cancelled

    results: list[CombinatorialProduct] = []
    if not route.steps:
        return results

    compiled = []
    for step in route.steps:
        if not step.reaction_smarts:
            return results
        try:
            compiled.append(rdChemReactions.ReactionFromSmarts(step.reaction_smarts))
        except Exception:
            return results

    current_mols = [parent_mol]
    seen_inter = set()

    for idx, (step, rxn) in enumerate(zip(route.steps, compiled)):
        if is_cancelled():
            return results
        next_mols = []
        for m in current_mols:
            try:
                observed = rxn.RunReactants((m,))
            except Exception:
                continue
            for prod_tuple in observed:
                prod = prod_tuple[0]
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                can = Chem.MolToSmiles(prod, canonical=True)
                if "." in can:
                    continue
                if any(atom.GetSymbol() not in _ALLOWED_ELEMENTS_ROUTE for atom in prod.GetAtoms()):
                    continue
                next_mols.append(prod)
        if not next_mols:
            return results          # 断链：路线不成立

        is_last = idx == len(route.steps) - 1
        for prod in next_mols:
            can = Chem.MolToSmiles(prod, canonical=True)
            if is_last:
                stoich = calculate_stoichiometry(prod, route.target_role, "环氧")
                if stoich["functionality"] < 1:
                    continue
                if not (mw_range[0] <= stoich["mw"] <= mw_range[1]):
                    continue
                results.append(_make_product_record(
                    prod, can, route.target_role, stoich["resin_type"], precursor,
                    route.route_id, route.name, stoich["functionality"],
                    stoich["equivalent_weight"], stoich["mw"],
                    _calculate_sascore(prod), route=route))
            else:
                if can in seen_inter:
                    continue
                seen_inter.add(can)
                results.append(CombinatorialProduct(
                    product_smiles=can, role="intermediate", resin_type="intermediate",
                    precursor_id=precursor.core_id, precursor_name=precursor.name,
                    precursor_smiles=precursor.smiles,
                    reaction_id=step.reaction_id,
                    reaction_name=f"{route.route_id} 中间体",
                    functionality=0, equivalent_weight=0.0,
                    molecular_weight=float(Descriptors.MolWt(prod)),
                    sa_score=round(_calculate_sascore(prod), 2),
                    heavy_atoms=int(prod.GetNumHeavyAtoms()),
                    rotatable_bonds=int(Descriptors.NumRotatableBonds(prod)),
                    aromatic_rings=int(Descriptors.NumAromaticRings(prod)),
                    formula=rdMolDescriptors.CalcMolFormula(prod),
                    route_ids=[route.route_id], route_names=[route.name],
                    route_steps_min=idx + 1, route_redundancy=1,
                    green_score_best=route.green_score,
                ))
        current_mols = next_mols

    return results
```

3) 把路线链展开放进 worker 并行执行——`_process_single_precursor_reactions` 签名追加两个参数并在第 3 段（多通道反应循环）之后展开：

```python
def _process_single_precursor_reactions(
    precursor: PrecursorCore,
    compiled_reactions: list[tuple[SyntheticReaction, Any]],
    reactions: list[SyntheticReaction],
    min_functionality: int,
    max_sa_score: float,
    mw_range: tuple[float, float],
    enable_route_chains: bool = True,
    route_chain_ids: list[str] | None = None,
) -> list[CombinatorialProduct]:
```

函数体末尾（`return local_results` 之前）追加：

```python
    # 4. 路线链多步展开 (P1-P4 羟基环氧化等)：中间体 role=intermediate 入库
    if enable_route_chains:
        chain_seen = set(seen)
        for route in ROUTE_CHAINS:
            if route_chain_ids is not None and route.route_id not in route_chain_ids:
                continue
            chain_prods = _expand_route_chain(parent_mol, route, precursor, mw_range)
            for cp in chain_prods:
                # 与单步模板产物撞车也不丢：路线累加由调用方 _merge_product_records 完成
                local_results.append(cp)
```

> `seen` 语义不变（防 worker 内部重复），跨 workers 的合并与路线累加由调用方完成（Task 5）。

4) `run_combinatorial_monomer_design` 签名新增 `enable_route_chains: bool = True, route_chain_ids: list[str] | None = None`，开头归一化：

```python
    if route_chain_ids is None:
        route_chain_ids = [r.route_id for r in ROUTE_CHAINS if r.default_enabled]
```

并发分支 `executor.submit` 与单核分支调用 `_process_single_precursor_reactions(...)` 处都补传 `enable_route_chains=enable_route_chains, route_chain_ids=route_chain_ids`（已有传参可加或复用关键字）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_synthetic_routes.py -v`
Expected: 全部 PASS。注意 `test_p3_chain_end_to_end_equals_p1` 中 P3 中间体 canonical = `OC(CCl)COc1ccccc1`（本机实测）。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_synthetic_routes.py
git commit -m "feat: 路线链多步执行器与中间体入库（role=intermediate 不进配方筛选）"
```

---

### Task 5: 去重累加语义 + 序列化

**Files:**
- Modify: `core/molecule_design.py`（`run_combinatorial_monomer_design` 去重逻辑与 `asdict` 序列化）
- Test: `tests/test_synthetic_routes.py`（追加）

**Interfaces:**
- Consumes: Task 4 的 `CombinatorialProduct` 路线字段与 `_expand_route_chain`/`enable_route_chains` 签名
- Produces: `_merge_product_records(results, seen_idx, item)`；产物 DataFrame 中 `route_ids`/`route_names` 为 `;` 分隔字符串。Task 6-9 消费输出列。

- [ ] **Step 1: 写失败测试**

`tests/test_synthetic_routes.py` 追加：

```python
import pandas as pd


def test_route_accumulation_columns_exist():
    """产物表必须带路线列：route_ids/route_redundancy/route_steps_min/green_score_best。"""
    from core.molecule_design import run_combinatorial_monomer_design
    df, _ = run_combinatorial_monomer_design(
        enable_scaffold_fission=False, min_functionality=2,
        max_total_products=100, n_jobs=1,
    )
    assert not df.empty
    for col in ("route_ids", "route_names", "route_redundancy", "route_steps_min", "green_score_best"):
        assert col in df.columns, f"缺少路线列 {col}"


def test_same_product_different_routes_accumulate():
    """同一产物若可经多条路线合成，route_ids 必须累加而非丢弃。"""
    import json
    from core.molecule_design import run_combinatorial_monomer_design, PrecursorCore
    phenol = PrecursorCore(core_id="tp", name="苯酚", category="测试", role="resin",
                           smiles="Oc1ccccc1", description="", default_selected=True)
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[phenol], enable_scaffold_fission=False,
        enable_route_chains=True, min_functionality=2, max_total_products=200, n_jobs=1,
    )
    target = df.loc[df["product_smiles"] == "c1ccc(OCC2CO2)cc1"]
    assert not target.empty, "苯基缩水甘油醚必须在产物中"
    routes = str(target.iloc[0]["route_ids"])
    assert "P1_phenol_ech" in routes and "P3_chlorohydrin" in routes, \
        f"路线累加失败: {routes}"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_synthetic_routes.py::test_same_product_different_routes_accumulate -v`
Expected: FAIL — `route_ids` 字符串只含单条路线（缺 `P3_chlorohydrin`；`test_route_accumulation_columns_exist` 在 Task 4 后已 GREEN）。

- [ ] **Step 3: 实现**

1) 新增 `_merge_product_records`（文件级函数，`run_combinatorial_monomer_design` 之前）：

```python
def _merge_product_records(
    results: list[CombinatorialProduct],
    seen_idx: dict[str, int],
    item: CombinatorialProduct,
) -> None:
    """按 canonical SMILES 合并产物：同产物累加路线，不丢弃先行命中的路线信息。"""
    key = item.product_smiles
    idx = seen_idx.get(key)
    if idx is None:
        seen_idx[key] = len(results)
        results.append(item)
        return
    existing = results[idx]
    for rid in item.route_ids:
        if rid not in existing.route_ids:
            existing.route_ids.append(rid)
    for rn in item.route_names:
        if rn not in existing.route_names:
            existing.route_names.append(rn)
    existing.route_redundancy = max(1, len(existing.route_ids))
    existing.route_steps_min = min(existing.route_steps_min, item.route_steps_min)
    existing.green_score_best = max(existing.green_score_best, item.green_score_best)
```

2) `run_combinatorial_monomer_design` 两处收集分支（并发与单核）中的去重替换为：

```python
    seen_idx: dict[str, int] = {}
    ...
    for item in p_res:
        if is_cancelled():
            break
        _merge_product_records(results, seen_idx, item)
```

删除原 `seen_smiles = set()` 与 `if item.product_smiles not in seen_smiles: seen_smiles.add(...)`。排序截断逻辑不变。

3) 末尾序列化改为：

```python
    df = pd.DataFrame([asdict(r) for r in results])
    if not df.empty:
        for col in ("route_ids", "route_names"):
            df[col] = df[col].apply(lambda v: ";".join(v) if isinstance(v, list) else v)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_synthetic_routes.py tests/test_combinatorial_molecule_design.py -v`
Expected: 全部 PASS。再跑 `python -m pytest tests/test_virtual_screening.py -v` 确认配方筛选消费 `role` 列不受影响。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_synthetic_routes.py
git commit -m "feat: 产物去重改为路线累加，新增路线冗余度/步数/绿色评分列"
```

---

### Task 6: A1 星型/四官能环臂泛化

**Files:**
- Modify: `core/molecule_design.py:639-730`（星型与空间四官能组装段）
- Test: `tests/test_scaffold_diversity.py`（新建）

**Interfaces:**
- Consumes: `RING_SCAFFOLDS`、`LINKER_BRIDGES`
- Produces: `RING_ARM_TEMPLATES: dict[str, tuple[str, str]]`（酚臂, 胺臂）；星型/四官能母核随任意已选环生成

- [ ] **Step 1: 写失败测试**

创建 `tests/test_scaffold_diversity.py`：

```python
"""骨架多样性：星型/四官能不再硬编码苯环；配额防饿死；死选项补模板。"""
import pytest
from rdkit import Chem
from core.molecule_design import (
    RING_ARM_TEMPLATES, generate_scaffold_intermediates, PRECURSOR_CATALOG,
)

_BASE = {p.core_id for p in PRECURSOR_CATALOG}


def _new(cores):
    return [p for p in cores if p.core_id not in _BASE]


def test_ring_arm_templates_registered():
    for ring in ("benzene", "naphthalene_14", "anthracene", "pyrene", "fluorene", "carbazole"):
        assert ring in RING_ARM_TEMPLATES, f"缺少环臂模板 {ring}"


def test_star_generates_for_non_benzene_rings():
    """只勾选萘时仍能生成星型母核（旧实现只认苯环，此处必失败）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14"], selected_linkers=["methine_star"],
        max_intermediates=5000,
    )
    star = [p for p in _new(cores) if p.category == "拓扑星型多酚"]
    assert star, "勾选萘时未生成星型母核"


def test_unchecking_benzene_keeps_star():
    """取消苯环不得清零星型（只要还有其他已选环）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14", "fluorene"],
        selected_linkers=["methine_star", "quaternary_carbon"],
        max_intermediates=5000,
    )
    star = [p for p in _new(cores) if "拓扑星型" in p.category]
    assert star, "仅稠环勾选时星型被清零"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_scaffold_diversity.py -v`
Expected: FAIL — `ImportError`（RING_ARM_TEMPLATES 不存在）与星型断言失败。

- [ ] **Step 3: 实现**

1) 在 `generate_scaffold_intermediates` 定义之前新增环臂模板表（连接点碳必须在片段首位，羟基在环上其他位置，否则中心碳会连到氧上形成醚而非 C-C 键——本机实测教训）：

```python
# 星型/四官能组装用"环臂"模板：(酚臂, 胺臂)。片段第一个原子为与中心核的连接点。
RING_ARM_TEMPLATES: dict[str, tuple[str, str]] = {
    "benzene":        ("c1ccc(O)cc1",             "c1ccc(N)cc1"),
    "naphthalene_14": ("c1ccc(O)c2ccccc12",       "c1ccc(N)c2ccccc12"),
    "naphthalene_26": ("c1ccc2cc(O)ccc2c1",       "c1ccc2cc(N)ccc2c1"),
    "anthracene":     ("c1ccc2cc3ccccc3cc2c1O",   "c1ccc2cc3ccccc3cc2c1N"),
    "pyrene":         ("c1cc2ccc3cccc4ccc(O)c1c2c34", "c1cc2ccc3cccc4ccc(N)c1c2c34"),
    "biphenyl_44":    ("c1ccc(-c2ccc(O)cc2)cc1",  "c1ccc(-c2ccc(N)cc2)cc1"),
    "terphenyl":      ("c1ccc(-c2ccc(-c3ccc(O)cc3)cc2)cc1", "c1ccc(-c2ccc(-c3ccc(N)cc3)cc2)cc1"),
    "fluorene":       ("c1ccc2c(c1)Cc1ccc(O)cc12", "c1ccc2c(c1)Cc1ccc(N)cc12"),
    "xanthene":       ("c1ccc2c(c1)Oc1ccc(O)cc1C2", "c1ccc2c(c1)Oc1ccc(N)cc1C2"),
    "carbazole":      ("c1ccc2[nH]c3ccc(O)cc3c2c1", "c1ccc2[nH]c3ccc(N)cc3c2c1"),
    "quinoxaline":    ("c1ccc2nccnc2c1O",          "c1ccc2nccnc2c1N"),
}
```

2) 把星型段（原 :639-668，硬编码 `_ring_enabled("benzene")` + 苯环 SMILES）替换为：

```python
    # 2. 三核星型组装模型 (>CH- 次甲基核)：遍历所有已选环骨架的酚/胺臂
    if _linker_enabled("methine_star"):
        for ring_key, (arm_p, arm_a) in RING_ARM_TEMPLATES.items():
            if not _ring_enabled(ring_key):
                continue
            for r_grp in r_groups:
                if len(intermediates) >= max_intermediates:
                    break
                # R 基替换：苯臂（benzene）模板含 {r} 槽，其他环臂忽略 R 基
                if ring_key == "benzene":
                    r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
                    arm_p_r = f"c1cc{r_str}c(O)cc1"
                    arm_a_r = f"c1cc{r_str}c(N)cc1"
                else:
                    arm_p_r, arm_a_r = arm_p, arm_a
                for s, cat, role, prefix in [
                    (f"C({arm_p_r})({arm_p_r}){arm_p_r}", "拓扑星型多酚", "resin", "三酚"),
                    (f"C({arm_a_r})({arm_a_r}){arm_a_r}", "拓扑星型多胺", "hardener", "三胺"),
                ]:
                    try:
                        m = Chem.MolFromSmiles(s)
                        if m:
                            Chem.SanitizeMol(m)
                            cs = Chem.MolToSmiles(m, canonical=True)
                            if cs not in seen_smiles:
                                seen_smiles.add(cs)
                                intermediates.append(PrecursorCore(
                                    core_id=f"gen_star_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                    name=f"星型{prefix} ({ring_key})",
                                    category=cat, role=role, smiles=cs,
                                    description=f"三官能度星型拓扑核心前驱体（{ring_key} 臂）",
                                    default_selected=True))
                    except Exception:
                        pass
```

> 注：苯臂带 R 基的 `c1cc{r}c(O)cc1` 形式已实测（r=C 时 func=3, mw=334.4），R 基不影响组装。

3) 空间四官能段（原 :671-700 `ethane_tetra`/`quaternary_carbon` 的 benzene 块）替换为遍历臂：

```python
        if _linker_enabled("ethane_tetra"):
            for ring_key, (arm_p, _) in RING_ARM_TEMPLATES.items():
                if not _ring_enabled(ring_key):
                    continue
                spatial_candidates.append(
                    (f"C({arm_p})({arm_p})C({arm_p})({arm_p})", f"四苯乙烷型四酚 ({ring_key})", "resin"))
        if _linker_enabled("quaternary_carbon"):
            for ring_key, (arm_p, arm_a) in RING_ARM_TEMPLATES.items():
                if not _ring_enabled(ring_key):
                    continue
                spatial_candidates.append(
                    (f"C({arm_p})({arm_p})({arm_p}){arm_p}", f"四苯甲烷型四酚 ({ring_key})", "resin"))
                spatial_candidates.append(
                    (f"C({arm_a})({arm_a})({arm_a}){arm_a}", f"四苯甲烷型四胺 ({ring_key})", "hardener"))
```

保留原 fluorene/spirobiindane 特殊块不动。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_scaffold_diversity.py tests/test_combinatorial_molecule_design.py -v`
Expected: 全部 PASS。注意旧测试 `test_ring_selection_and_missing_rings_now_generate` 中"只选苯环+呫吨"的门控断言仍须成立（新实现按 ring_key 过滤，不会泄漏芴）。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_scaffold_diversity.py
git commit -m "feat: 星型/四官能组装泛化到任意环骨架，取消苯环不再清零星型"
```

---

### Task 7: A2 类别配额截断

**Files:**
- Modify: `core/molecule_design.py`（`generate_scaffold_intermediates` 返回前）
- Test: `tests/test_scaffold_diversity.py`（追加）

**Interfaces:**
- Consumes: Task 6 的生成物
- Produces: `_apply_category_quota(intermediates, max_total) -> list[PrecursorCore]`

- [ ] **Step 1: 写失败测试**

`tests/test_scaffold_diversity.py` 追加：

```python
def test_small_budget_keeps_all_categories():
    """max_intermediates=500 时稠环/星型/羧酸类别均不得整类饿死。"""
    from collections import Counter
    cores = generate_scaffold_intermediates(max_intermediates=500)
    cats = Counter(p.category for p in _new(cores))
    for key in ("稠环衍生多酚", "拓扑星型多酚", "多元羧酸衍生母核"):
        assert cats.get(key, 0) > 0, f"{key} 被配额饿死"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_scaffold_diversity.py::test_small_budget_keeps_all_categories -v`
Expected: FAIL — 实测已知 max=500 时 501 母核 = 双酚 229 + 二胺 228 + 经典 44。

- [ ] **Step 3: 实现**

在 `generate_scaffold_intermediates` 的 `return intermediates` 之前插入：

```python
    # 类别配额截断：替代"先到先得"，保证柱后（稠环/星型/酸）类别不被饿死
    intermediates = _apply_category_quota(intermediates, max_intermediates)
```

函数体加在文件内（`generate_scaffold_intermediates` 之前）：

```python
_CATEGORY_QUOTA = {
    "拓扑衍生双酚": 0.18, "拓扑衍生二胺": 0.18,
    "稠环衍生多酚": 0.12, "稠环衍生多胺": 0.12,
    "特种空间刚性母核": 0.12,
    "不对称杂化母核": 0.09,
    "拓扑星型多酚": 0.04, "拓扑星型多胺": 0.04,
    "多元羧酸衍生母核": 0.06,
}


def _apply_category_quota(
    intermediates: list[PrecursorCore], max_total: int
) -> list[PrecursorCore]:
    """按类别配额截断：经典母核恒保留；配额内每类先取；剩余名额按存量回填。

    注意：PrecursorCore 是 @dataclass（eq=True 时不可哈希），判重一律用 core_id 集合，
    严禁 `set(core_mols)`（会 TypeError: unhashable）。
    """
    quota_budget = max(0, max_total - len(PRECURSOR_CATALOG))
    core_ids = {c.core_id for c in PRECURSOR_CATALOG}
    core_mols = [p for p in intermediates if p.core_id in core_ids]
    generated = [p for p in intermediates if p.core_id not in core_ids]
    by_cat: dict[str, list[PrecursorCore]] = {}
    for p in generated:
        by_cat.setdefault(p.category, []).append(p)
    selected: list[PrecursorCore] = []
    for cat, frac in _CATEGORY_QUOTA.items():
        pool = by_cat.get(cat, [])
        take = min(len(pool), int(quota_budget * frac))
        selected.extend(pool[:take])
    # 剩余名额按存量回填（类别顺序保持注册表顺序，保证确定性）
    used = len(selected)
    remaining = quota_budget - used
    for cat in _CATEGORY_QUOTA:
        pool = by_cat.get(cat, [])
        taken = min(len(pool), int(quota_budget * _CATEGORY_QUOTA[cat]))
        extra = min(len(pool) - taken, remaining)
        selected.extend(pool[taken:taken + extra])
        remaining -= extra
        if remaining <= 0:
            break
    # 未列入配额表的类别（如测试类别）全额保留
    for cat, pool in by_cat.items():
        if cat not in _CATEGORY_QUOTA:
            selected.extend(pool)
    return core_mols + selected
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_scaffold_diversity.py tests/test_combinatorial_molecule_design.py -v`
Expected: 全部 PASS。已知新分布（max=500）应为 经典 44 + 各类别均 > 0。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_scaffold_diversity.py
git commit -m "feat: 骨架母核截断改为类别配额制，小预算不再饿死稠环/星型/羧酸"
```

---

### Task 8: A3 include_acids 门控 + A4 补齐 4 个死选项

**Files:**
- Modify: `core/molecule_design.py`（`include_acids` 段加 `_ring_enabled`；`ring_templates` 与 `poly_cores` 追加模板）
- Test: `tests/test_scaffold_diversity.py`（追加）

**Interfaces:**
- Consumes: Task 7 的生成器
- Produces: `mesitylene`/`biphenyl_33`/`cardo_bpf`/`s_triazine` 四个 ring_id 真正参与组装

- [ ] **Step 1: 写失败测试**

`tests/test_scaffold_diversity.py` 追加：

```python
def test_dead_ring_options_now_generate():
    """4 个死选项现在必须产出真实骨架母核（而非酸的假象）。"""
    for rid in ("mesitylene", "biphenyl_33", "cardo_bpf", "s_triazine"):
        cores = generate_scaffold_intermediates(selected_rings=[rid], max_intermediates=5000)
        skeleton = [p for p in _new(cores) if p.category != "多元羧酸衍生母核"]
        assert skeleton, f"{rid} 无任何骨架母核（仍是死选项）"


def test_acid_section_respects_ring_gate():
    """酸段必须受环门控：只勾 non-acid 环时，酸母核数量应显著减少。"""
    all_cores = generate_scaffold_intermediates(max_intermediates=5000)
    gated = generate_scaffold_intermediates(selected_rings=["mesitylene"], max_intermediates=5000)
    n_all = sum(1 for p in _new(all_cores) if p.category == "多元羧酸衍生母核")
    n_gated = sum(1 for p in _new(gated) if p.category == "多元羧酸衍生母核")
    assert n_gated < n_all, f"酸段未受环门控: {n_gated} >= {n_all}"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_scaffold_diversity.py -v`
Expected: FAIL — 实测已知 4 死选项产出 0 骨架母核；酸段不受门控（n_gated == n_all == 47）。

- [ ] **Step 3: 实现**

1) A3：`include_acids` 段开头（`:752` 附近）改为：

```python
    if include_acids and (_ring_enabled("benzene") or _ring_enabled("naphthalene_14") or _ring_enabled("toluene")):
```

> 门控语义：酸母核的模板骨架为苯环系，仅当用户勾选了苯环/萘等芳环时才生成。

2) A4：`ring_templates`（:480 附近列表）追加两条：

```python
        ("mesitylene", "均三甲苯基", "mesitylene", "Cc1cc(C)cc({link})c1",
         "Oc1cc{r}cc(C)c1{lk}c1cc(O)cc(C)c1C", "Nc1cc{r}cc(C)c1{lk}c1cc(N)cc(C)c1C"),
        ("biphenyl_33", "3,3'-联苯", "biphenyl_33", "c1cccc(-c2cccc({link})c2)c1",
         "Oc1cccc(-c2cccc({lk}c3cccc(O)c3)c2)c1", "Nc1cccc(-c2cccc({lk}c3cccc(N)c3)c2)c1"),
```

`poly_cores` 追加两条（模板中 `{r}` 不加括号，r_str 自带括号）：

```python
        ("cardo_bpf", "卡基芴", "cardo_bpf",
         "Oc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1", "Nc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1"),
        ("s_triazine", "对称三嗪", "s_triazine",
         "Oc1nc(O)nc{r}n1", "Nc1nc(N)nc{r}n1"),
```

3) ring_templates 双核循环依赖 `{lk}` 替换 17 种桥——mesitylene/biphenyl_33 模板与现有槽位一致，无需改循环。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_scaffold_diversity.py tests/test_combinatorial_molecule_design.py -v`
Expected: 全部 PASS。本机已实测：mesitylene 双酚（`Cc1cc(C)c(C(C)(C)c2cc(O)cc(C)c2C)c(O)c1`，官能度 2）、biphenyl_33 双酚、cardo_bpf 双酚、s_triazine 多酚（当量 63.55）均可解析。

- [ ] **Step 5: Commit**

```bash
git add core/molecule_design.py tests/test_scaffold_diversity.py
git commit -m "feat: 补齐均三甲苯/3,3-联苯/卡基芴/三嗪骨架模板，酸段接入环门控"
```

---

### Task 9: 数据兼容 + UI 扩展

**Files:**
- Modify: `app_lib.py:27968-27980`（反应路径 multiselect 区域）、`app_lib.py:25308-25320`（配方注入处读旧 parquet）
- Test: `tests/test_molecule_design_ui_contract.py`（新建）

**Interfaces:**
- Consumes: Task 5 的产物新列
- Produces: UI 新增"路线步数/绿色评分"说明与开关；旧 parquet 缓存安全读取

- [ ] **Step 1: 写失败测试**

创建 `tests/test_molecule_design_ui_contract.py`：

```python
"""UI 契约：路线列缺失时旧缓存可读；路线链开关在函数签名中存在。"""
import inspect

import pandas as pd


TEST_LEGACY_ROUTE_COLUMNS = ("route_ids", "route_redundancy", "route_steps_min", "green_score_best")


def test_run_signature_has_route_chain_params():
    from core.molecule_design import run_combinatorial_monomer_design
    sig = inspect.signature(run_combinatorial_monomer_design)
    assert "enable_route_chains" in sig.parameters
    assert "route_chain_ids" in sig.parameters


def test_legacy_parquet_columns_backfilled():
    """旧 parquet（无路线列）经真实回填函数处理后必须补齐默认列，而非抛 KeyError。

    这里调用的是 core.molecule_design.ensure_route_columns 本体（生产代码），
    不是把回填逻辑抄进测试——抄一遍只能证明测试自己写得对。
    """
    from core.molecule_design import ensure_route_columns

    legacy = pd.DataFrame({
        "product_smiles": ["c1ccc(OCC2CO2)cc1", "Oc1ccccc1"],
        "role": ["resin", "hardener"],
    })
    out = ensure_route_columns(legacy)
    for col in TEST_LEGACY_ROUTE_COLUMNS:
        assert col in out.columns, f"旧缓存缺列 {col} 未被回填"
    assert out.loc[0, "route_ids"] == ""
    assert out.loc[0, "route_redundancy"] == 1
    assert out.loc[0, "route_steps_min"] == 1
    assert out.loc[0, "green_score_best"] == 0.0
    # 原有列与数据不得被破坏
    assert out["role"].tolist() == ["resin", "hardener"]
    assert out["product_smiles"].tolist() == ["c1ccc(OCC2CO2)cc1", "Oc1ccccc1"]


def test_route_columns_preserved_when_already_present():
    """新数据（已有路线列）不得被回填覆盖。"""
    from core.molecule_design import ensure_route_columns

    fresh = pd.DataFrame({
        "product_smiles": ["c1ccc(OCC2CO2)cc1"],
        "role": ["resin"],
        "route_ids": ["P1_phenol_ech;P3_chlorohydrin"],
        "route_redundancy": [2],
        "route_steps_min": [1],
        "green_score_best": [6.0],
    })
    out = ensure_route_columns(fresh)
    assert out.loc[0, "route_ids"] == "P1_phenol_ech;P3_chlorohydrin"
    assert out.loc[0, "route_redundancy"] == 2
    assert out.loc[0, "green_score_best"] == 6.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_molecule_design_ui_contract.py -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_route_columns'`。（`test_run_signature_has_route_chain_params` 在 Task 4 后应已 PASS，属正常。）

- [ ] **Step 3: 实现**

1) 在 `core/molecule_design.py` 新增模块级回填函数（**放在 `run_combinatorial_monomer_design` 之前**，与 `_serialize_route_columns` 相邻，同为模块级以避开 Windows-spawn 约束）：

```python
# 产物表路线列及其缺省值（旧 parquet 缓存不含这些列，读取后需回填）
ROUTE_COLUMN_DEFAULTS: tuple[tuple[str, object], ...] = (
    ("route_ids", ""),
    ("route_redundancy", 1),
    ("route_steps_min", 1),
    ("green_score_best", 0.0),
)


def ensure_route_columns(df):
    """为旧产物表回填缺失的路线列（不改动已有列，返回同一个 DataFrame）。

    设计引擎在 Task 4/5 之前持久化的 parquet 不含 route_* 列；
    UI 读取这些缓存时不能抛 KeyError，故在此统一补齐默认值。
    """
    for col, default in ROUTE_COLUMN_DEFAULTS:
        if col not in df.columns:
            df[col] = default
    return df
```

2) `app_lib.py` 配方注入处（`active_designed_source_df` 判空之后、`designed_smiles_col` 之前）调用它：

```python
                if isinstance(active_designed_source_df, pd.DataFrame) and not active_designed_source_df.empty:
                    from core.molecule_design import ensure_route_columns
                    active_designed_source_df = ensure_route_columns(active_designed_source_df)
                    designed_smiles_col = "product_smiles" if ...
```

3) `app_lib.py` 反应路径 multiselect 下方加说明（不新增交互控件，保持最小改动）：

```python
            st.caption(
                "🔬 路线感知已启用：同一产物多路线合成时自动累加备选路线，"
                "并计算冗余度/最短步数/绿色评分（见结果表 route_* 列）。"
                "多步路线（P2/P3/P4）的中间体以 role=intermediate 入库，不参与配方筛选。"
            )
```

结果表渲染处（`_render_molecule_design_engine` 显示 `cached_total_df` 的 `st.dataframe`）若有列清单请保留原样（旧列兼容 + 新列自动显示）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_molecule_design_ui_contract.py tests/test_combinatorial_molecule_design.py -v`
Expected: 全部 PASS。再跑一次完整全量：`python -m pytest tests/ -x -q -k "not melting and not portal"`（必要时）确认无回归。

- [ ] **Step 5: Commit**

```bash
git add app_lib.py tests/test_molecule_design_ui_contract.py
git commit -m "feat: 路线感知 UI 提示与旧缓存列回填"
```

---

## Self-Review

- Spec 覆盖：§3 Part A(A1-A4)→Task 6/7/8；Part B(B1-B5)→Task 1/2/3/4/5；B6 UI→Task 9；§4.1 数据兼容→Task 5/9；§5 验收 V1-V10 分属各任务测试。
- 占位符：无 TBD/TODO。
- 类型一致：`CombinatorialProduct` 路线字段由 Task 4 定义（Task 4/5 已合并为一次提交，消除循环依赖）；`ensure_route_columns` 由 Task 9 定义并被 `app_lib.py` 调用。
- 已知限制：R 基在非苯臂上不替换（Task 6 注）；P4 消旋（spec §4.3）；P2 第二步复用 R04 收紧版 SMARTS。
- 实施中发现的预存缺陷（不在本计划范围）：`benzene_13` 双核模板为 7 元芳环、无法 kekulize，该环双核骨架恒产出 0。