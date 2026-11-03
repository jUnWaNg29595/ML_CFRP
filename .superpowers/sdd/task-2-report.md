# Task 2 Report: R04 选择性修复（递归 SMARTS 排除吸电子烯烃）

**Status: DONE**
**Commit: `149acf1` — feat: R04 环氧化排除吸电子烯烃，BMI/丙烯酸酯不再被误氧化**

## 实现内容

将 `core/molecule_design.py` 中 `olefin_epoxidation` (R04) 的 `reaction_smarts` 从宽泛的
`[C:1]=[C:2]>>[C:1]1O[C:2]1` 收紧为简报指定的递归 SMARTS（逐字一致，已程序化断言校验）：

```python
reaction_smarts=(
    "[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):1]"
    "=[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):2]"
    ">>[C:1]1O[C:2]1"
),
```

- 三个排除条件：羰基碳自身（`!$([CX3]=[OX1])`）、邻接羰基的 α-碳（`!$([CX4,CX3][CX3]=[OX1])`，丙烯酸酯/马来酰亚胺根因）、邻接腈基碳（`!$([CX4,CX3][CX2]#[NX1])`）。
- 原子映射 `:1`/`:2` 按简报要求放在外层 `]` 之前（内层位置），避免产物退化为无映射的 oxirane。
- 除 SMARTS 字符串外，`core/molecule_design.py` 无任何其他改动（description 等保持原样，最小精确编辑）。

## 测试内容与结果

向 Task 1 已有的 `tests/test_reaction_selectivity.py` 追加（复用其 `_products` 辅助函数，未重建文件、未改动既有 3 个 R01 测试与 helpers）：

- `R04_NEG`（9 个负样本）：马来酰亚胺、丙烯酸甲酯、肉桂酸甲酯、肉桂酸、肉桂醛、丙烯腈、甲基乙烯基酮、马来酸酐、甲基丙烯酸甲酯 —— 全部应拒绝环氧化。
- `R04_POS`（7 个正样本）：环己烯、烯丙基苯酚、DCPD、苯乙烯、1-辛烯、异丁烯、降冰片烯 —— 全部应环氧化，且产物含 `[O;r3]1[C;r3][C;r3]1` 三元环氧环。
- `test_r04_rejects_electron_poor_alkenes`、`test_r04_still_epoxidizes_common_alkenes` 两个测试函数，逐字取自简报。

### TDD Evidence

**RED（实现前）：**

```
$ python -m pytest tests/test_reaction_selectivity.py::test_r04_rejects_electron_poor_alkenes -v
>           assert prods == [], f"{name} 不应被环氧化，实际得到 {prods}"
E           AssertionError: 马来酰亚胺 不应被环氧化，实际得到 ['O=C1NC(=O)C2OC12', 'O=C1NC(=O)C2OC12']
FAILED tests/test_reaction_selectivity.py::test_r04_rejects_electron_poor_alkenes
============================== 1 failed in 1.47s ==============================
```

失败产物 `O=C1NC(=O)C2OC12` 与任务描述的已验证失败完全一致——旧 SMARTS 会破坏 BMI 体系。该失败是预期内的：旧模板 `[C:1]=[C:2]` 匹配任何烯烃，无吸电子排除。

**GREEN（实现后）：**

```
$ python -m pytest tests/test_reaction_selectivity.py -v
tests/test_reaction_selectivity.py::test_r01_no_longer_touches_carboxyl PASSED
tests/test_reaction_selectivity.py::test_r01a_phenol_only PASSED
tests/test_reaction_selectivity.py::test_r01b_aliphatic_only PASSED
tests/test_reaction_selectivity.py::test_r04_rejects_electron_poor_alkenes PASSED
tests/test_reaction_selectivity.py::test_r04_still_epoxidizes_common_alkenes PASSED
============================== 5 passed in 1.22s ==============================
```

负样本 9/9 排除、正样本 7/7 环氧化。Task 1 的 3 个 R01 测试不受影响。

**回归检查：**

```
$ python -m pytest tests/test_combinatorial_molecule_design.py -v
============================= 15 passed in 4.95s ==============================
```

全量旧测试无回归。

## 变更文件

- `core/molecule_design.py`：1 处 —— `olefin_epoxidation.reaction_smarts`（+6 −1 行，仅字符串形式换行）
- `tests/test_reaction_selectivity.py`：追加 R04 测试（+38 行）
- 提交仅含上述两个文件；`UserPrediction.py`、`core/portal_prediction.py`、`core/prediction_portal.py`、`tests/test_portal_*.py`、`CHANGELOG.md` 等无关预存修改未被暂存或提交。

## 自审结论

- **完整性**：SMARTS 与简报逐字一致（程序断言校验通过）；两个测试逐字来自简报；原子映射位置符合"内层"要求。
- **质量**：TDD 完整执行（先 RED 后 GREEN），测试输出干净无警告噪声。
- **纪律**：未过度构建——曾顺手微调 description 后主动撤销，最终 diff 严格限于一个 SMARTS 字符串；未新建 helper、未改动既有测试。
- **Git 纪律**：仅 `git add` 两个目标文件，未使用 `git add -A`/`git add .`。

## 问题与关注点

无阻塞问题。一个非阻塞观察：负样本中部分底物（如肉桂酸）在 R04 之外仍可被其他模板（R03 羧酸酯化）命中，属预期分工，与本任务无关；Task 3/5 复用 `olefin_epoxidation` 时将直接受益于本次收紧。
