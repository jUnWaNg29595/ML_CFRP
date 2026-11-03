# CFRP 反应与网络交联特征重构交接文档（第二轮修改·高缺失率文献数据适配版）

## 1. 任务背景与核心目标
本项目致力于构建证据可溯源、物理理论严密的 CFRP 环氧树脂反应网络特征提取与机器学习体系。
在第一轮 9 项验收标准通过的基础上，针对真实文献数据存在严重用量缺失（高缺失率）的现实痛点，推进《第二轮修改报告（高缺失率文献数据适配版）》的重构。

**核心目标**：
在 WSL 环境（/home/wangj/project/CFRP）下，重构 core/reaction_core.py 并协同适配 core/component_physics.py、core/crosslink_physics.py、core/epoxy_mechanism_features.py，使之完全满足 12 项端到端验收测试（test_acceptance_12_criteria.py），严禁修改 Windows 镜像。

---

## 2. 关键设计规范与 12 项验收标准

### 2.1 当量比 r 四级来源优先级（核心防截断机制）
1. literature：文献明确指定（如输入包含 explicit_r 或数据表中存在 formulation_r_value/r_value 且大于 0）。此时 rxn_r_assumed = 0.0。
2. calculated_from_mass：各组分质量与有效当量完整，按实际配方化学计量学计算：
   r = (sum(m_j / AHEW_j)) / (sum(m_i / EEW_i))
   此时 rxn_r_assumed = 0.0。
3. assumed_stoichiometric：部分或全部质量缺失（例如仅有单体 SMILES 无 phr，或仅有单侧树脂用量缺失固化剂用量）。文献中高概率按理论等当量固化，因此假设 r = 1.0，来源标记为 assumed_stoichiometric，rxn_r_assumed = 1.0。绝对禁止将其粗暴计算为 0 或截断至 50/0.01 等失真值。
4. invalid_chemistry：无法构成有效反应体系（如缺少环氧或固化剂，或无有效反应性官能团）。r = NaN，rxn_r_assumed = 0.0。

### 2.2 组分等当量/等摩尔重构规范
当处于 assumed_stoichiometric 模式时，需自洽推算虚拟质量以支持后续数均分子量 Mn、摩尔平均官能度 f_avg、凝胶特征的计算：
- 完全无质量（无 phr）：树脂按 100g 均分（或等摩尔），固化剂按 r=1.0 匹配总当量补齐，记录 composition_assumption = "assumed_equimolar"。
- 单侧有质量（如仅树脂 100g）：按树脂总当量与 r=1.0 补齐固化剂质量，记录 composition_assumption = "assumed_resin_only_reconstructed"。

### 2.3 酸酐当量口径与凝胶机制
- 酸酐 AHEW 计算：必须严格采用化学计量口径 f_stoich = 1.0（1 摩尔酸酐消耗 1 摩尔环氧，AHEW = MW / 1.0 = MW，绝不能使用 f_network = 2.0 使 EW 减半）。
- 酸酐凝胶判定：酸酐固化含酯化、醚化复杂副反应，凝胶假设字段标记为 gel_assumption = "mechanism_approximate"。

### 2.4 Flory-Stockmayer 凝胶分支理论修正
- 分支因子：
  kappa_A = sum(n_i * f_i * (f_i - 1)) / sum(n_i * f_i)
  kappa_B = sum(n_j * g_j * (g_j - 1)) / sum(n_j * g_j)
- 临界凝胶转化率：p_Ac = sqrt(r / (kappa_A * kappa_B))，p_Bc = sqrt(1 / (r * kappa_A * kappa_B))，alpha_gel = 1 / sqrt(kappa_A * kappa_B)。
- 凝胶可行性判定：以分支因子发散为标准：
  kappa_A * kappa_B * min(r, 1/r) >= 1.0 - 1e-4
  不强制全体系分子数摩尔平均官能度 f_avg > 2.0（解决多组分混合体系中微量高官能度引发凝胶但分子数均值受低官能单体拉低的问题）。
- 单官能环氧体系：如 PGE (f_r=1)，分支因子 kappa_A = 0，无法凝胶，gel_possible_ideal = 0.0，Mc_ideal = NaN。

### 2.5 12 项端到端验收清单对照表
| 编号 | 测试用例场景 | 预期物理量行为与断言 |
|---|---|---|
| 1 | DGEBA + EDA (100g : 8.83g) | r 约 1.0，来源为 calculated_from_mass，rxn_r_assumed = 0.0 |
| 2 | DGEBA + EDA (100g : 17.65g) | r 约 2.0（胺过量），不被 r=1 覆盖，来源为 calculated_from_mass |
| 3 | 仅 SMILES 无用量（DGEBA + EDA） | r = 1.0，来源为 assumed_stoichiometric，rxn_r_assumed = 1.0 |
| 4 | 单侧用量缺失（树脂 100g，固化剂缺失） | r = 1.0，来源为 assumed_stoichiometric，禁止计算出 0 或 50 |
| 5 | 文献明确指定 r = 0.5 | 保留 r = 0.5，来源为 literature，不强行配平为 1.0 |
| 6 | 单官能环氧 + 四官能胺 (r=1.0) | 理想网络不可凝胶：gel_possible_ideal == 0.0，Mc_ideal == NaN |
| 7 | 酸酐固化剂（MTHPA） | AHEW 采用 f_stoich = 1.0；gel_assumption == "mechanism_approximate" |
| 8 | 多官能度混合环氧体系 | 凝胶依据分支因子 kappa_A * kappa_B * min(r, 1/r) > 1.0，不强求 f_avg > 2.0 |
| 9 | 有效配方混入未知组分 | 明确记录失配/不支持状态（composition_assumption 或 coverage 标记） |
| 10 | 相同样本跨四个模块计算 | reaction_core, component_physics, crosslink_physics, epoxy_mechanism 对 r、f_avg、凝胶特征严格一致 |
| 11 | 配方质量整体乘 10（等比放大） | 归一化特征（r、f_avg、临界转化率、理论交联特征）数值完全不变 |
| 12 | 自定义 DataFrame 索引（非默认 RangeIndex） | 输出行数与 index 完全保真，不重置不乱序 |

---

## 3. 环境与执行命令规范
- WSL 执行前缀：MSYS_NO_PATHCONV=1 /c/Windows/System32/wsl.exe -d Ubuntu-24.04 -u wangj bash -c "<cmd>"
- 工作目录：/home/wangj/project/CFRP
- Conda 环境路径：/home/wangj/miniconda3/envs/CFRP_env
- Python / pytest 路径：
  - /home/wangj/miniconda3/envs/CFRP_env/bin/python
  - /home/wangj/miniconda3/envs/CFRP_env/bin/pytest
- 备份文件已创建：
  - core/reaction_core.py.bak
  - core/component_physics.py.bak
  - core/crosslink_physics.py.bak_v2
  - core/epoxy_mechanism_features.py.bak_v2

---

## 4. 下一步具体执行步骤
1. 重构 core/reaction_core.py：扩充 NetworkReactionSummary，落地四级当量比优先级及重构模式，更新 Flory-Stockmayer 判据。
2. 适配 core/component_physics.py：对齐无质量时假设 r=1.0 并输出来源标签及 rxn_r_assumed。
3. 适配 core/crosslink_physics.py 与 core/epoxy_mechanism_features.py：统一物理量定义，避免重复计算或逻辑分歧。
4. 编写并运行 test_acceptance_12_criteria.py：确保 12 项验收测试全绿。
