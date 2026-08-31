# 虚拟分子筛选 · 全维度拓扑演化分子设计与配方级高通量衔接引擎设计规范

- **日期**：2026-08-28
- **模块**：`core/molecule_design.py`, `app.py`
- **目标**：解决虚拟分子筛选单体产物库数量少、产物角色分类不清晰、无法直接对接配方级高通量筛选的问题，建立支持环结构、主链桥联、侧链修饰、核心官能团多维演化的高通量组合合成与配方注入引擎。

---

## 1. 架构总览与核心目标

### 1.1 核心问题与痛点
1. **产物库容量受限**：原设计仅依靠 20 余个固定前驱体母核与少量反应，组合空间小（数十至数百个分子），无法满足现代材料基因工程万级搜索空间的需求。
2. **缺乏全维度拓扑演化**：原系统无法自由调节环骨架（芳环/稠环/卡基/脂环/杂环）、主链桥联键（烃基/含氟/极性/杂原子）、侧链 R 基团以及官能团弹头。
3. **产物角色与配方对接断层**：部分反应产物角色标签混淆，缺乏环氧当量 (EEW) 和活性氢当量 (AHEW) 等热固性树脂关键化学计量特征计算，无法一键注入后续的配方级多组分高通量筛选模块。

### 1.2 解决方案：两阶段融合流水线引擎 (Two-Stage Pipeline Engine)
```
[阶段 1: 骨架裂变生成器 (Scaffold Fission)]
环骨架 (24类) × 主链桥联 (20类) × 侧链R基 (12类) × 拓扑连接模式
                  │
                  ▼
   [超大多元前驱体库 (多元酚 / 多元胺 / 多元酸 / 脂环烯烃)]
                  │
                  ▼
[阶段 2: 多通道合成反应流水线 (Multi-Channel Reaction Pipeline)]
12 大工业与前沿反应模板 (Reaction SMARTS) + 反应完全性穷举
                  │
                  ▼
[阶段 3: 物理化学与合成门禁 + 化学计量打标 (Quality Gate & Stoichiometry)]
RDKit 构型校验 + SAScore 难度过滤 + 官能度校验 + EEW/AHEW/Role 自动计算
                  │
                  ▼
[阶段 4: 配方级高通量筛选一键注入 (Seamless Formula HTVS Integration)]
自动分流 Resin / Hardener 库 -> 导入配方笛卡尔积/配比优化器
```

---

## 2. 四层化学积木库与拓扑装配规范

### 2.1 骨架环体系 (Ring Scaffolds - 24 类)
1. **单环与苯系**：Benzene, Toluene, Xylene, Mesitylene
2. **稠环芳烃与联苯系**：Naphthalene (1,4/1,5/2,6/2,7位), Anthracene (9,10位), Phenanthrene, Pyrene, Perylene, Biphenyl (4,4'/3,3'位), p-Terphenyl, m-Terphenyl
3. **卡基 (Cardo) 与大体积三维刚性环**：Fluorene (芴基), 9,9-Bisphenyl Fluorene (双酚芴), Spirobiindane (螺双二氢茚 SBP), Spirobifluorene, Xanthene (呫吨)
4. **含氮/硫杂环 (耐温/阻燃/介电优化)**：s-Triazine (对称三嗪), Pyridine, Carbazole, Quinoxaline, Benzimidazole, Phenothiazine
5. **脂环族与笼状烃 (超低粘度/超低介电 Dk/Df)**：Cyclohexane (1,4/1,3位), Dicyclopentadiene (DCPD), Adamantane (金刚烷 1,3/1,3,5,7位), Norbornane/Norbornene, Isophorone

### 2.2 主链连接桥联库 (Linker Bridges - 20 类)
1. **脂肪/脂环柔性烃基桥**：`-CH2-`, `-C(CH3)2-`, `-C(C2H5)2-`, `-Cyclohexyl-`, `-DCPD-`
2. **含氟/低极性疏水桥**：`-C(CF3)2-` (双酚AF型), `-(CF2)2-`, `-(CF2)4-`
3. **耐热/极性/耐蠕变桥**：`-S(=O)(=O)-` (砜基), `-S(=O)-`, `-C(=O)-` (酮基), `-COO-`, `-CONH-`
4. **杂原子与本征阻燃/增韧桥**：`-O-` (醚键), `-S-` (硫醚), `-P(=O)(Ph)-` (苯基磷氧), `-Si(CH3)2-O-Si(CH3)2-` (硅氧烷)
5. **多支化/星型核心**：`>CH-` (三官能次甲基), `>CH-CH<` (四官能乙烷四基), `>C<` (季碳核心), `>Si<` (硅原子核心)

### 2.3 侧链修饰基团库 (R-Groups - 12 类)
`-H`, `-CH3`, `-C(CH3)3`, `-F`, `-CF3`, `-Cl`, `-Br`, `-OCH3`, `-OCF3`, `-CH=CH2`, `-CH2CH=CH2`, `-CN`

### 2.4 拓扑连接组合模型
- **二官能双核模型 (Dimeric Linear/Angular)**：`Ar1 - Linker - Ar2`（每个芳环带有 1 个活性官能团基团位点，共 2 官能度）
- **三官能星型模型 (Trimeric Star)**：`Core - (Linker - Ar)3`（共 3 官能度）
- **四官能空间模型 (Tetrameric Spatial)**：`Core - (Linker - Ar)4`（共 4 官能度）
- **稠环多取代模型 (Poly-substituted Fused Ring)**：单环/稠环（萘/蒽/芘/三嗪）直接多位取代。

---

## 3. 多通道合成反应模板体系 (12 大 Reaction SMARTS)

| 编号 | 反应通道名称 | 反应物基团 | 目标角色 (Role) | Reaction SMARTS | 产物化学体系 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R01** | 酚羟基缩水甘油醚化 | 多元酚 `-OH` | `resin` | `[c:1][OX2H:2]>>[c:1]OCC1CO1` | DGEBA/DGEBF/多酚缩水甘油醚环氧 |
| **R02** | 芳香胺缩水甘油胺化 | 芳香胺 `-NH2` | `resin` | `[c:1][NX3;H2:2]>>[c:1]N(CC2CO2)CC3CO3` | TGDDM/AFG-90 航空耐温环氧 |
| **R03** | 多元酸缩水甘油酯化 | 羧酸 `-COOH` | `resin` | `[C:1](=O)[OX2H:2]>>[C:1](=O)OCC2CO2` | 缩水甘油酯低粘度树脂 |
| **R04** | 烯烃双键环氧化 | 脂环烯烃 `C=C` | `resin` | `[C:1]1=[C:2]CCCC1>>[C:1]12O[C:2]2CCCC1` | 脂环族环氧 (TDE-85/耐候电绝缘) |
| **R05** | 酚醛胺曼尼希反应 | 多元酚 `-OH` | `hardener`| `[c:1][OX2H:2]>>[c:1](O)CNCCN` | 酚醛胺/改性多胺固化剂 |
| **R06** | 多元胺烷基化/接枝 | 芳香/脂环胺 `-NH2` | `hardener`| `[c:1][NX3;H2:2]>>[c:1]NCC` | 韧性与反应活性调控固化剂 |
| **R07** | 邻二羧酸酸酐化 | 二元酸 `-COOH` | `hardener`| `[C:1](=O)O.[C:2](=O)O>>[C:1](=O)OC(=O)[C:2]` | 环状酸酐固化剂 (MTHPA/NADIC) |
| **R08** | 马来酰亚胺化反应 | 多元胺 `-NH2` | `resin` | `[c:1][NX3;H2:2]>>[c:1]N1C(=O)C=CC1=O` | 双马来酰亚胺 BMI 高耐热基体 |
| **R09** | 酚羟基氰酸酯化 | 多元酚 `-OH` | `resin` | `[c:1][OX2H:2]>>[c:1]OC#N` | 氰酸酯树脂 CE (超低介电 Dk/Df) |
| **R10** | 苯并噁嗪环化反应 | 多元酚 `-OH` | `resin` | `[c:1][OX2H:2]>>[c:1]1OCN(c2ccccc2)Cc1` | 苯并噁嗪 BOZ 树脂 (近零收缩率) |
| **R11** | 酚羟基双炔丙基醚化 | 多元酚 `-OH` | `resin` | `[c:1][OX2H:2]>>[c:1]OCC#C` | 炔丙基醚树脂 (自催化/耐烧蚀) |
| **R12** | 天然母核固化剂继承 | 芳胺/脂环胺/酸酐| `hardener`| 直接保留母核并计算活性氢/酸酐当量 | DDM, 4,4'-DDS, 3,3'-DDS, IPDA, DETDA, PMDA |

---

## 4. 产物数据契约、质量门禁与化学计量计算规范

### 4.1 数据契约 (CombinatorialProduct Schema)
每个生成的单体具备完整、可解释的数据字段：
```python
@dataclass
class CombinatorialProduct:
    product_smiles: str       # 规范化 Canonical SMILES
    role: str                 # "resin" | "hardener"
    resin_type: str           # "epoxy_ether" | "epoxy_amine" | "bmi" | "cyanate" | "benzoxazine" | "amine" | "anhydride"
    precursor_name: str       # 前驱体骨架溯源 (如 "双酚芴-六氟异丙基-4,4'位")
    reaction_name: str        # 合成反应路线 (如 "酚羟基缩水甘油醚化")
    functionality: int        # 交联官能度 (>= 2)
    equivalent_weight: float  # 化学当量 (树脂为 EEW，固化剂为 AHEW)
    molecular_weight: float   # 理论分子量 (g/mol)
    sa_score: float           # 合成可及性评分 (1.0~10.0，越低越易合成)
    heavy_atoms: int          # 重原子数
    rotatable_bonds: int      # 可旋转键数
    aromatic_rings: int       # 芳香环数
    formula: str              # 分子式 (如 C21H24O4)
```

### 4.2 质量门禁与过滤规则
1. **RDKit 化学结构有效性**：严格通过 `Chem.SanitizeMol`，过滤五价碳、未配对自由基等异常结构。
2. **元素白名单**：仅允许 `{H, C, N, O, F, Si, P, S, Cl, Br}`。
3. **交联官能度门禁 (Functionality Gate)**：默认 `functionality >= 2`，排除无法形成 3D 热固性交联网络的单官能度封端小分子。
4. **合成可及性门禁 (SAScore Gate)**：基于环张力、手性中心与重原子数评估，默认 `SAScore <= 6.0`。
5. **分子量范围 (MW Gate)**：默认 `140.0 <= MW <= 1500.0 g/mol`。

### 4.3 化学计量当量自动计算 (Auto Stoichiometry)
- **环氧当量 (EEW)**：$EEW = \frac{Molecular Weight}{Num(Epoxide Groups)}$
- **活性氢当量 (AHEW)**：$AHEW = \frac{Molecular Weight}{Num(Active Hydrogens)}$（芳香伯胺每个 `-NH2` 计 2 个活性氢，仲胺计 1 个）
- **酸酐当量**：$Anhydride EW = \frac{Molecular Weight}{Num(Anhydride Groups)}$

---

## 5. 配方级高通量筛选对接流程 (HTVS Handshake)

```
[分子设计引擎前端]
        │
        ├─ 生成 2,000 ~ 20,000+ 可合成单体库
        ├─ 实时统计分布：树脂库规模、固化剂库规模、EEW/AHEW 分布直方图
        │
        └─ 🔘 点击【一键导入配方级高通量筛选】
                 │
                 ▼
[Session State 共享与协议交付]
st.session_state["vs_designed_resin_library"] = df[role == 'resin']
st.session_state["vs_designed_hardener_library"] = df[role == 'hardener']
                 │
                 ▼
[配方级高通量筛选模块 (Formula HTVS)]
        ├─ 树脂组分选择：自动填充设计的树脂库
        ├─ 固化剂组分选择：自动填充设计的固化剂库
        └─ 配方配比生成器：基于 EEW 与 AHEW 自动按化学计量比 (1:1 当量比 ± 10%) 生成候选配方矩阵！
```

---

## 6. 测试与验证方案

1. **单元测试 (`tests/test_combinatorial_molecule_design.py`)**：
   - 验证 24 种环、20 种桥联、12 种 R 基的组合裂变生成算法；
   - 验证 12 种 Reaction SMARTS 反应的转化准确率与穷举完全性；
   - 验证 EEW、AHEW、官能度与 SAScore 计算逻辑；
   - 验证产物数量规模可从 500 扩展至 10,000+ 且无重复 Canonical SMILES。
2. **端到端集成测试**：
   - 模拟在 `app.py` 中运行分子设计引擎，并一键传递至配方级高通量筛选，成功触发多组分配方预测。
