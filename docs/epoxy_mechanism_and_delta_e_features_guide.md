# 环氧树脂配方机理特征工程 (含 ΔE 能差矩阵与高分子交联网络) 技术白皮书与全流程实现指南

---

## 📌 目录
1. [系统背景与核心解决的问题](#1-系统背景与核心解决的问题)
2. [总体架构与全流程数据流](#2-总体架构与全流程数据流)
3. [核心理论模型与物理数学公式](#3-核心理论模型与物理数学公式)
   - [3.1 量子化学活化能自适应加权与反应概率方程](#31-量子化学活化能自适应加权与反应概率方程)
   - [3.2 动态自适应交联网络理论 (广义 Mc 与交联密度)](#32-动态自适应交联网络理论-广义-mc-与交联密度)
   - [3.3 前线轨道相互作用能差 (ΔE) 多列矩阵架构](#33-前线轨道相互作用能差-δe-多列矩阵架构)
   - [3.4 3D 构象形貌与等效自由体积 (FFV) 代理](#34-3d-构象形貌与等效自由体积-ffv-代理)
   - [3.5 固化原位产物与凝聚态内聚能 (CED) 网络](#35-固化原位产物与凝聚态内聚能-ced-网络)
   - [3.6 微观刚性与宏观交联密度的非线性交叉项](#36-微观刚性与宏观交联密度的非线性交叉项)
4. [面向真实工业数据的鲁棒性处理机制](#4-面向真实工业数据的鲁棒性处理机制)
   - [4.1 多槽位 (1..N) 自动捕获与微量小分子促进剂处理](#41-多槽位-1n-自动捕获与微量小分子促进剂处理)
   - [4.2 反应温度与工艺制度不全的物理平滑反推](#42-反应温度与工艺制度不全的物理平滑反推)
   - [4.3 当量比 (r) 与配比缺失的 4 级回退链](#43-当量比-r-与配比缺失的-4-级回退链)
   - [4.4 零空缺 (Zero-NaN) 防御设计](#44-零空缺-zero-nan-防御设计)
5. [底层代码实现模块解析](#5-底层代码实现模块解析)
   - [5.1 EpoxyMechanismEngine (配方机理引擎)](#51-epoxymechanismengine-配方机理引擎)
   - [5.2 MulticomponentCrosslinkedFeatureExtractor (组合反应模拟器)](#52-multicomponentcrosslinkedfeatureextractor-组合反应模拟器)
   - [5.3 QuickForceFieldFeatureExtractor 与 XTBFeatureExtractor 联立](#53-quickforcefieldfeatureextractor-与-xtbfeatureextractor-联立)
6. [全量提取特征清单 (字典级参数与物理意义)](#6-全量提取特征清单-字典级参数与物理意义)
7. [使用指南与代码调用示例](#7-使用指南与代码调用示例)

---

## 1. 系统背景与核心解决的问题

在复合材料（CFRP）高分子树脂体系的玻璃化转变温度（$T_g$）、热力学及力学性能预测中，传统的分子特征工程（如纯 RDKit 2D 描述符、Morgan 分子指纹）长期存在三大物理信息断层：
1. **割裂了单体与交联产物的关联**：固化是不可逆化学变化，简单使用未开环单体的指纹无法感知开环原位生成的致密氢键网络与交联节点立体拓扑。
2. **割裂了微观刚性与宏观交联网络**：传统模型常将单体的旋转柔顺度与宏观网链分子量（$M_c$）独立输入，树模型很难自发学习高阶非线性乘除耦合，导致误判“纯脂肪族高交联树脂（低 $T_g$）”与“纯刚性低交联树脂（低 $T_g$）”。
3. **生硬的人工经验假设**：传统半经验模型通常对反应基团赋予固定有效常数（如端炔=1.0，内炔=0.15），忽略了取代基电子效应、立体位阻与热活化能垒的连续影响。

本项目研发的**环氧配方机理特征工程体系**，基于第一性原理量子化学（xTB）、分子力场（MMFF94s）与高分子交联动力学，构建了一套连续、自适应、物理自洽的特征计算流水线，彻底弥补了上述物理缺陷。

---

## 2. 总体架构与全流程数据流

整体系统包含“单体量子化学/力场表征”、“多组分微观反应动力学调制”、“宏观交联网络物理拓扑生成”三大层次：

```
                      ┌──────────────────────────────────────┐
                      │    原始输入: 配方表 DataFrame         │
                      │  (多树脂槽位 + 多固化剂槽位 + PHR/r) │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │  树脂端单体表征通道     │                     │  固化剂端单体表征通道   │
    ├─────────────────────────┤                     ├─────────────────────────┤
    │ • ETKDGv3 3D 构象初筛   │                     │ • ETKDGv3 3D 构象初筛   │
    │ • MMFF94s 能量7项拆解   │                     │ • MMFF94s 能量7项拆解   │
    │ • xTB CDFT 5项反应活性  │                     │ • xTB CDFT 5项反应活性  │
    │ • 骨架刚性碳 (sp+sp2)   │                     │ • 活性氢官能团精细检测  │
    │ • 偏离平面度 PBF / 自由体积│                  │ • 偏离平面度 PBF / 自由体积│
    └────────────┬────────────┘                     └────────────┬────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │     动力学与前线轨道相互作用能差 (ΔE) 矩阵    │
                 ├───────────────────────────────────────────────┤
                 │ • 正向主通道: ΔE_fwd = LUMO(Resin) - HOMO(Curer) │
                 │ • 逆向通道: ΔE_rev = LUMO(Curer) - HOMO(Resin)  │
                 │ • 费米-狄拉克自适应反应概率: P_rxn(ΔE)         │
                 │ • 跨分子化学势差: Δμ (电子转移热力学驱动势)   │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌─────────────────────────┐                     ┌─────────────────────────┐
    │   宏观交联物理与产物维  │                     │   非线性物理耦合项      │
    ├─────────────────────────┤                     ├─────────────────────────┤
    │ • 动态自适应 Mc 与交联密度  │                  │ • cross_rigidity_cross  │
    │ • 开环仲醇羟基密度 ρ_OH │                     │   (交联密度 × 刚性碳)   │
    │ • 原位叔胺受体密度 ρ_N  │                     │ • cross_torsion_cross   │
    │ • MMFF 电荷各向异性方差 │                     │   (交联密度 × 扭转势垒) │
    │ • 三嗪环节点密度        │                     │ • cross_fmo_rigidity    │
    └────────────┬────────────┘                     └────────────┬────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    最终输出: 41项机理特征 + 全单体   │
                      │    力场量化特征 (无 NaN、严格对齐)   │
                      └──────────────────────────────────────┘
```

---

## 3. 核心理论模型与物理数学公式

### 3.1 量子化学活化能自适应加权与反应概率方程
系统彻底摒弃了人为固定常数，利用量子化学计算的本征能差 $\Delta E_{\text{fwd}}$，借助阿伦尼乌斯/费米-狄拉克概率因子动态评估每个反应基团的真实转化概率 $P_{\text{rxn}}$：
$$P_{\text{rxn}, i} = \frac{1}{1 + \exp\left( \frac{\Delta E_{\text{gap}, i} - E_0}{\tau} \right)}$$
- $E_0$：特征能垒阈值（主环氧-胺基体系设为 $7.5\,\text{eV}$）；
- $\tau$：活化能温度缩放标度（设为 $1.0\,\text{eV}$）；
- **物理意义**：
  - 端炔基、脂肪胺空间位阻小、能隙低，$P_{\text{rxn}} \to 0.85 \sim 0.98$；
  - 内炔基受芳环/烷基位阻且能隙偏大，$P_{\text{rxn}} \to 0.15 \sim 0.25$（随取代基连续平滑浮动）；
  - 惰性基团能垒极高，$P_{\text{rxn}} \to 0.0$。

### 3.2 动态自适应交联网络理论 (广义 Mc 与交联密度)
利用各反应基团的动态反应概率，加权求得有效官能度：
$$f_{\text{eff}, R} = \sum_i N_{R, i} \times P_{\text{rxn}, R, i},\quad f_{\text{eff}, H} = \sum_j N_{H, j} \times P_{\text{rxn}, H, j}$$
在多槽位非等当量配方下，广义理论网链分子量 $M_c^{\text{adaptive}}$ 计算公式为：
$$M_c^{\text{adaptive}} = \frac{\overline{M}_{\text{formula}}}{\max\left( \frac{f_{\text{eff}, R} + r \cdot f_{\text{eff}, H}}{1.0 + r} - 2.0,\, 0.08 \right)}$$
真实交联密度 $\nu$：
$$\nu = \frac{1000.0}{M_c^{\text{adaptive}}}\quad (\text{mmol/g}),\quad \nu_{\text{vol}} = \frac{\rho \times 10^6}{M_c^{\text{adaptive}}}\quad (\text{mol/m}^3)$$

### 3.3 前线轨道相互作用能差 (ΔE) 多列矩阵架构
针对配方中第 $i$ 树脂与第 $j$ 固化剂：
- **正向主反应通道（固化剂亲核进攻环氧位点）**：
  $$\Delta E_{\text{fwd}, ij} = E_{\text{LUMO}}(R_i) - E_{\text{HOMO}}(H_j)$$
  决定了体系初始开环活化能垒。
- **逆向通道（反向电荷转移）**：
  $$\Delta E_{\text{rev}, ij} = E_{\text{LUMO}}(H_j) - E_{\text{HOMO}}(R_i)$$
- **跨分子化学势驱动差（电子转移本征热力学势差）**：
  $$\Delta \mu = \mu_{\text{Curer}} - \mu_{\text{Resin}} = \chi_{\text{Resin}} - \chi_{\text{Curer}}$$

### 3.4 3D 构象形貌与等效自由体积 (FFV) 代理
基于 MMFF94s 优化后的平衡 3D 构型，提取微观几何量：
1. **回转半径 $R_g$ 与范德华体积 $V_{\text{vdW}}$**：
   $$R_g = \sqrt{\frac{1}{M}\sum m_i (\mathbf{r}_i - \mathbf{r}_{\text{cm}})^2},\quad V_{\text{vdW}} = \text{ComputeMolVolume}(\text{mol})$$
2. **分子紧凑度**：
   $$C_{\text{mol}} = \frac{V_{\text{vdW}}}{R_g^3}$$
3. **偏离平面度 (Plane of Best Fit, PBF)**：
   计算原子坐标相对最佳拟合平面的垂直距离方差。精准捕捉 2,2'-联苯等阻转异构体系的剧烈立体扭转。
4. **等效分数自由体积 (FFV) 代理**：
   $$FFV = 1.0 - 1.3 \times \frac{V_{\text{vdW}} \times 0.6022}{V_{\text{molar}}}$$

### 3.5 固化原位产物与凝聚态内聚能 (CED) 网络
模拟开环后的新化学环境：
1. **原位仲醇羟基密度 $\rho_{\text{OH}}$**：
   $$\rho_{\text{OH}} = \frac{\min(f_{\text{epoxy}},\, f_{\text{amine-H}} \cdot r)}{M_{\text{formula}}} \times 1000.0\quad (\text{mmol/g})$$
2. **固化生成叔胺密度 $\rho_{\text{tert-N}}$**：
   表征仲醇羟基与叔胺形成的密集分子内/分子间氢键网络，该作用力在高温段维持高聚物交联网络不解冻。
3. **静电电荷方差 $\sigma_q^2$**：
   $$\sigma_q^2 = \frac{1}{N}\sum_{i=1}^N (q_i - \bar{q})^2$$
   直接对应凝集态下的偶极内聚能密度（CED）。

### 3.6 微观刚性与宏观交联密度的非线性交叉项
根据 DiBenedetto 理论，通过显式构造物理乘积项消除树模型的学习障碍：
1. **`cross_rigidity_crosslink`**：$\nu \times \text{rigid\_carbon\_ratio}$
   （交联网络密度与主链芳香杂化刚性碳的联合驱动项，彻底打破脂肪族高交联低 $T_g$ 的误判）。
2. **`cross_torsion_crosslink`**：$\nu \times \text{ff\_torsion\_density}$
   （网络节点受限环境下的链段协同旋转势垒）。
3. **`cross_flexibility_index`**：$\frac{N_{\text{rot}}}{N_{\text{arom}} + 1.0} \times \frac{M_c}{100}$
   （链段综合柔顺度因子，与 $T_g$ 呈强烈的负相关关系）。

---

## 4. 面向真实工业数据的鲁棒性处理机制

工业实验数据集往往存在“多槽位、含微量促进剂、反应温度未记录、配比缺失”等不完备性。系统采用以下机制保证**100% 物理自洽且零 NaN 运行**：

### 4.1 多槽位 (1..N) 自动捕获与微量小分子促进剂处理
- **全槽位正则捕获**：
  自动匹配树脂端（`resin_1`, `resin_2`, `epoxy_1`, `diluent_structure` 等）与固化剂端（`curing_agent_1`, `hardener_1`, `accelerator_1` 等）。
- **微量促进剂动力学放大**：
  在组合反应模拟中，微量小分子促进剂（如 1%~2% 咪唑）由于其活化能垒 $\Delta E$ 极低，动力学成键概率被自适应放大：
  $$P_{ij}^{\text{kinetic}} \propto (w_{R,i} \times w_{H,j}) \times \left(0.3 + 0.7 \times \frac{1}{1 + \exp\left(\frac{\Delta E_{ij} - 7.0}{1.2}\right)}\right)$$
  真实反映了“微量促进剂在升温初期抢先开环并引发局部高致密交联网络”的化学本质。

### 4.2 反应温度与工艺制度不全的物理平滑反推
- 当数据缺失真实固化温度时，系统利用 xTB 量化开环能垒 $\Delta E_{\text{fwd}}$ 自适应外推等效工艺温标：
  - 活化能垒高的芳香胺体系（如 DDS）自动推断需要高温活化，并释放其高温深层反应度；
  - 易开环的脂肪胺体系自动匹配中低温活化区间。

### 4.3 当量比 (r) 与配比缺失的 4 级回退链
严格执行 4 级降级保证不崩：
1. **第 1 级**：读取真实标注列（`formulation_r_value`、`r_value` 等）；
2. **第 2 级**：读取各槽位用量 PHR，依据各组分文献/结构当量精确自算 $r = \frac{\sum H_{\text{phr}} / \text{AHEW}}{\sum R_{\text{phr}} / \text{EEW}}$；
3. **第 3 级**：物理区间钳制（约束在 $[0.2, 5.0]$ 避免离群输入炸裂）；
4. **第 4 级**：若完全无任何配比信息，平滑回退到高分子标准**化学计量等当量态（$r=1.0$ 理想态）**，此时对称偏离度 $|r-1.0|=0$。

### 4.4 零空缺 (Zero-NaN) 防御设计
- **废除动态稀疏槽位列**：废除类似 `mech_fmo_delta_e_r2_h2` 等稀疏列，所有输出统一收敛为全配方通用的 41 项标量特征与单体特征；
- **空槽位动态剪枝**：某样本只有 2 个组分时，自动按 2 个槽位截断，绝不针对未提供的槽位输出假特征，确保整个特征矩阵的 NaN 数量严格为 0。

---

## 5. 底层代码实现模块解析

系统核心逻辑分布于以下核心代码模块中：

### 5.1 `core/epoxy_mechanism_features.py`
* **类 `EpoxyMechanismEngine`**：
  - `clean_structure_string(s)`：SMILES 清洗与归一化；
  - `calc_single_molecule_properties(smi, is_resin)`：提取单体的 3D 体积、回转半径 $R_g$、PBF、刚性碳比率及 Gasteiger/MMFF 电荷方差；
  - `compute_formulation_mechanism_features(resins, curers, ...)`：执行多槽位动态自适应加权，输出 41 个 `mech_*`、`geom_*`、`cross_*`、`gen_*` 特征；
  - `enrich_dataframe(df, include_ff_features=True, include_xtb_features=True)`：一键处理整张数据表，联立单体力场分项与 xTB 反应活性。

### 5.2 `core/reaction_simulator.py`
* **类 `MulticomponentCrosslinkedFeatureExtractor`**：
  - `_simulate_combinatorial(...)`：执行多组分优势通道反应模拟，集成活化能能垒二次调制与通道剪枝；
  - `simulate_multicomponent_reaction(...)`：在拓扑上装配代表性产物结构（`product_smiles` / `product_structure`），原位测算交联节点密度与悬挂链比例。

### 5.3 `core/molecular_features.py`
* **类 `QuickForceFieldFeatureExtractor`**：
  - 嵌入 ETKDGv3 5 构象初筛与 MMFF 优化选优，解耦输出伸缩、弯曲、二面角扭转、范德华、静电等 14 项分项能量。
* **类 `XTBFeatureExtractor`**：
  - 封装半经验 GFN2-xTB 计算，衍生化学硬度、软度、电负性、化学势与亲电指数，并内置持久化 SQLite 缓存数据库（`.cache/cfrp_xtb_feature_cache.db`）。

---

## 6. 全量提取特征清单 (字典级参数与物理意义)

一键提取模式下，系统在双组分基准配方上输出的代表性核心特征如下：

| 特征分类 | 代表性特征列名 | 算法/公式定义 | 物理化学意义 |
| :--- | :--- | :--- | :--- |
| **拓扑交联维** | `gen_Mc_adaptive` | $\frac{M_R + r M_H}{f_{\text{eff}, R} + r f_{\text{eff}, H} - 2}$ | 基于量子动态反应概率的理论网链分子量 |
| | `gen_crosslink_density`| $1000.0 / M_c^{\text{adaptive}}$ | 理论真实交联密度 ($\text{mmol/g}$) |
| | `mech_crosslink_density_mol_m3` | $\rho \times 10^6 / M_c$ | 标准 SI 单位交联体积摩尔浓度 ($\text{mol/m}^3$) |
| | `stoich_defect_ratio` | $\vert{}r - 1.0\vert{}$ | 配比对称偏离度（表征网络未交联悬挂链缺陷） |
| | `triazine_node_density`| $(N_{\text{OCN}} / 3) / M_{\text{formula}}$ | 原位三聚均三嗪刚性环节点密度 ($\text{mmol/g}$) |
| **量子与动力学** | `prob_primary_epoxy_amine` | $\frac{1}{1 + \exp((\Delta E - 7.5)/1.0)}$ | 主交联通道前向开环转化概率 |
| | `prob_alkyne_thermal` | 费米概率因子调制 | 体系内炔基高温自发热聚合概率 |
| | `mech_fmo_delta_e_fwd_primary` | $E_{\text{LUMO}}(R_1) - E_{\text{HOMO}}(H_1)$ | 主反应通道前线轨道能差 ($\text{eV}$) |
| | `mech_fmo_delta_mu_weighted` | $\mu_{\text{curer}} - \mu_{\text{resin}}$ | 电子由固化剂流向环氧开环位点的本征驱动势 |
| | `xtb_chemical_hardness_weighted` | $(E_{\text{LUMO}} - E_{\text{HOMO}})/2$ | 配方加权电子化学硬度（抵抗外场变形阻力） |
| **骨架杂化刚度** | `rigid_carbon_ratio` | $\frac{N(sp\text{-C}) + N(sp^2\text{-C})}{N(\text{total-C})}$ | 骨架刚性碳比例（纯连续值，天然区分芳环与烷基） |
| | `ff_torsion_density` | $E_{\text{torsion}} / N_{\text{rot}}$ | MMFF94s 单键旋转扭转能密度（主链构象阻力） |
| | `ff_oop_density` | $E_{\text{oop}} / N_{\text{atoms}}$ | MMFF94s 离面弯曲能密度（共轭平面抗弯刚度） |
| | `flexibility_index` | $\frac{N_{\text{rot}}}{N_{\text{arom}} + 1.0}$ | 链段综合柔顺度因子（与 $T_g$ 强负相关） |
| **3D 形貌与自由体积**| `geom_pbf` | $\text{CalcPBF}(\text{mol})$ | 偏离平面度（捕捉 2,2'-联苯等阻转扭曲位阻） |
| | `geom_vdw_volume_weighted` | $\text{ComputeMolVolume}(\text{mol})$ | 配方加权范德华排除体积 ($\text{Å}^3$) |
| | `geom_compactness_weighted` | $V_{\text{vdW}} / R_g^3$ | 分子致密紧凑度（表征玻璃态抗剪切模量） |
| | `mech_proxy_ffv` | $1.0 - 1.3 \times \frac{V_{\text{vdW}} \times 0.6022}{V_{\text{molar}}}$ | 理论等效分数自由体积代理项（WLF 临界孔隙） |
| **产物内聚能 (CED)**| `post_hbond_donor_density` | $\frac{\min(f_R, f_H \cdot r)}{M_{\text{formula}}} \times 1000$ | 开环生成的仲醇羟基密度 ($\text{mmol/g}$) |
| | `post_tertiary_amine_density` | 固化生成的叔胺氮原子密度 | 叔胺-羟基内部强氢键网络密度 ($\text{mmol/g}$) |
| | `ff_charge_variance` | $\frac{1}{N}\sum(q_i - \bar{q})^2$ | MMFF94s 静电荷各向异性方差（偶极内聚能） |
| **非线性物理耦合项**| `cross_rigidity_crosslink` | $\nu \times \text{rigid\_carbon\_ratio}$ | 宏观交联密度与微观主链刚性的联合协同项 |
| | `cross_torsion_crosslink` | $\nu \times \text{ff\_torsion\_density}$ | 网络受限环境下的链段协同旋转势垒项 |
| | `cross_fmo_rigidity_index` | $\nu / \Delta E_{\text{fwd}}$ | 反应彻底性与网链致密度的协同项 |
| **单体级力场/量化**| `resin_1_ff_energy_torsion`| MMFF 构象分解 | 树脂1 分子内旋转扭转能 |
| | `curing_agent_1_xtb_homo` | xTB 量子化学计算 | 固化剂1 最高占据分子轨道能级 |

---

## 7. 使用指南与代码调用示例

### 7.1 Web 界面端一键提取（推荐）
1. 启动系统后，在左侧导航栏点击进入 **`🧬 分子特征提取`** 页面；
2. 在 **“选择分子特征提取方法”** 中选择：
   👉 **`🧬 环氧配方机理特征 (含 ΔE能差矩阵与交联网络)`**；
3. 系统会自动智能扫描识别表格中的所有树脂槽位与固化剂槽位；
4. 点击 **`🚀 开始提取分子特征`**：系统将在数秒至数分钟内自动完成单体力场能量分解、xTB 量化反应活性、组合反应动力学及交联机理特征计算，生成包含 80+ 项物理特征的高维度特征表。

### 7.2 Python 脚本 / 独立离线调用
如果需要在外部工作流或脱机训练中使用，只需调用 `EpoxyMechanismEngine`：

```python
import pandas as pd
from core.epoxy_mechanism_features import EpoxyMechanismEngine

# 1. 准备多槽位配方数据
df_formulation = pd.DataFrame([{
    "resin_1_structure": "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1", # DGEBA
    "resin_2_structure": "C1OC1CN(CC2CO2)C3=CC=C(CC4=CC=C(N(CC5CO5)CC6CO6)C=C4)C=C3", # TGDDM
    "curing_agent_1_structure": "NC1=CC=C(S(=O)(=O)C2=CC=C(N)C=C2)C=C1", # DDS
    "accelerator_1_structure": "CCC1=NC=C(C)N1", # 2-E-4-MI 咪唑促进剂
    "resin_1_amount_phr": 70.0,
    "resin_2_amount_phr": 30.0,
    "curing_agent_1_amount_phr": 28.0,
    "accelerator_1_phr": 1.5,
    "formulation_r_value": 0.95,
}])

# 2. 初始化高精度机理特征引擎
engine = EpoxyMechanismEngine(verbose=True)

# 3. 一键提取完整物理化学与交联动力学特征
# include_ff_features: 联立输出单体 MMFF 7项分项能量
# include_xtb_features: 联立输出单体 xTB CDFT 5项反应活性
enriched_df = engine.enrich_dataframe(
    df_formulation,
    include_ff_features=True,
    include_xtb_features=True,
)

# 4. 打印提取完成的高分子特征
mech_cols = [c for c in enriched_df.columns if c.startswith(("mech_", "geom_", "cross_", "gen_"))]
print(f"✅ 成功生成 {len(mech_cols)} 个核心配方机理特征！")
print("示例特征值:")
for col in ["gen_Mc_adaptive", "gen_crosslink_density", "rigid_carbon_ratio", "geom_pbf", "mech_proxy_ffv"]:
    print(f"  {col}: {enriched_df[col].iloc[0]}")
```

### 7.3 命令行批量增强已有数据集
系统同时提供了现成的数据集离线批处理脚本：
```bash
python tools/epoxy_qspr_enricher/enrich_qspr_dataset.py \
    --input "C:\Users\wangj\Desktop\ml_dataset\ml_qspr_model_tg_c.csv" \
    --output "C:\Users\wangj\Desktop\ml_dataset\ml_qspr_model_tg_c_enhanced.csv"
```
处理完成后，直接在原数据表右侧自动扩增全量物理交联与 $\Delta E$ 特征，即插即用于 XGBoost、LightGBM、PINN 及高分子高精度回归建模。
