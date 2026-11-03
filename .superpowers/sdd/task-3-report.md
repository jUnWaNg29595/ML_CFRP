# Task 3 报告：SyntheticRoute/RouteStep 数据结构与 ROUTE_CHAINS 注册表

## 状态

完成（DONE）。提交 `3f21c72`，TDD 全程：RED（ImportError）→ GREEN（2 passed）。

## 实现内容

在 `core/molecule_design.py` 中，`SYNTHETIC_REACTION_TEMPLATES` 列表结束（`phenolic_hardener_retain` 条目与其闭合 `]`）之后、`# 3. 经典工业前驱体母核库 (Precursor Catalog)` 横幅之前，按任务简报**逐字**插入 106 行新增代码（纯新增，0 删除，未改动任何既有代码）：

- `@dataclass RouteStep`：单步反应（step_index / reaction_id / reaction_smarts / intermediate_smiles / requires_reagent / hazard_flags）。
- `@dataclass SyntheticRoute`：完整合成路线，含 `@property n_steps`；字段 route_id / name / chemical_system / target_role / steps / atom_economy / green_score / default_enabled。
- `ROUTE_CHAINS: list[SyntheticRoute]`：四条真实工业羟基→环氧路线：
  - P1 `P1_phenol_ech` 直接 ECH 醚化（单步，复用 Task 1 的 `glycidyl_etherification_non_carboxyl` 对应 SMARTS，剧毒/含氯废水标记）；
  - P2 `P2_allyl_epoxidation` 烯丙基化→环氧化（两步，绿色无氯）；
  - P3 `P3_chlorohydrin` 氯代醇法开环/闭环（两步，NaOH 闭环）；
  - P4 `P4_sharpless` Sharpless 环氧化（单步，名称含「(消旋)」，无手性/无 useChirality——按用户决定不做不对称诱导）。
- `hydroxyl_epoxidation_routes()`：返回全部 `chemical_system == "epoxy"` 的路线。

未新增 import：`dataclass` 已在文件头部导入，`from __future__ import annotations` 已存在。

## 测试与结果

新建 `tests/test_synthetic_routes.py`（18 行，简报原文）：
- `test_routes_registered`：断言四个 route_id 均已注册。
- `test_route_smarts_compile`：遍历所有路线所有步骤，断言 SMARTS 均可被 RDKit 编译。

### TDD 证据

**RED**（实现前）：
```
命令：python -m pytest tests/test_synthetic_routes.py -v
结果：ERROR collecting tests/test_synthetic_routes.py
  tests\test_synthetic_routes.py:5: in <module>
      from core.molecule_design import ROUTE_CHAINS
  E   ImportError: cannot import name 'ROUTE_CHAINS' from 'core.molecule_design'
  1 error in 1.47s
```
失败原因即预期目标：`ROUTE_CHAINS` 尚未实现，ImportError 正是简报 Step 2 预期的 RED 信号。

**GREEN**（实现后）：
```
命令：python -m pytest tests/test_synthetic_routes.py -v
结果：
  tests/test_synthetic_routes.py::test_routes_registered PASSED            [ 50%]
  tests/test_synthetic_routes.py::test_route_smarts_compile PASSED         [100%]
  2 passed in 1.21s
```

### 回归验证

```
python -m pytest tests/test_synthetic_routes.py tests/test_molecule_design.py \
  tests/test_reaction_selectivity.py tests/test_combinatorial_molecule_design.py -q
→ 22 passed, 14 skipped in 5.34s   （skip 为既有条件跳过，非本次引入）
```

补充行为自检（python -c）：4 条路线、accessor 等价于 ROUTE_CHAINS、n_steps ∈ {1,2}、P4 名称含「消旋」且 SMARTS 无 useChirality、全部步骤带 hazard_flags —— 全部通过。

## 已知预存失败（与本任务无关，已逐一核实）

- `tests/test_reaction_network_unit.py` 4 个失败（单元数/连接数/BigSMILES 解析断言）：已用 `git show HEAD:core/molecule_design.py` 临时还原至 Task 2 后状态复跑，**同样 4 failed**，确认为预存问题。
- `tests/test_virtual_screening.py` 13 个失败：根因为 `ModuleNotFoundError: No module named 'streamlit'`（`core/fe_tracker.py:6`），环境缺第三方包，与本次纯新增改动无关。

## 变更文件

- `core/molecule_design.py`（+106，纯新增块）
- `tests/test_synthetic_routes.py`（新建，+18）

提交：`3f21c72 feat: 新增 SyntheticRoute/RouteStep 与羟基环氧化四条路线注册表`（仅含上述两个文件；工作树中 UserPrediction.py、portal 相关文件、CHANGELOG.md、task-2-report.md 等预存改动均未触碰、未暂存）。

## 自审结论

- **完整性**：简报要求的所有接口（RouteStep / SyntheticRoute / ROUTE_CHAINS / hydroxyl_epoxidation_routes）逐字落地，位置精确，无遗漏；P4 保持消旋。
- **纪律**：零额外构建（YAGNI），未重构任何既有结构；未加新 import。
- **测试**：严格 TDD（先 RED 后 GREEN），测试验证的是真实行为（注册完整性 + SMARTS 可编译性）。
- **疑点说明（非阻塞）**：P2/P3/P4 步骤的 `reaction_id`（allyl_etherification、olefin_epoxidation_selective、chlorohydrin_*、sharpless_epoxidation）在 `SYNTHETIC_REACTION_TEMPLATES` 中无同名模板——这是简报原设计：`RouteStep` 自带内联 `reaction_smarts`，`reaction_id` 仅为溯源标签，Task 4 执行器将直接消费步骤级 SMARTS。已按简报逐字执行。

## 遗留事项

无。Task 4（多步执行器）与 Task 5（去重阶段路线累积）将消费 `ROUTE_CHAINS`。
