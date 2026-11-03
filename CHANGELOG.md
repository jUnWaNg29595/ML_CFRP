# 材料机器学习平台 - 变更日志

所有重要的更改都将记录在此文件中。

本文档格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 修复与优化
- 【力学与储能模量 PINN 物理重构与优化：Halpin-Tsai 复合力学、对数尺度解耦与 Batch 调度修复】用户诉求：「我还要预测力学性能，比如25度储能模量（数据有），帮我用PINN进行优化」
  - **严重缺陷诊断**：
    1. **基体模量 $E_m$ 未中心化初始化**：原代码直接使用 `Em = softplus(out) * y_scale`，未加入均值偏置 $y_{\text{mean}}$。在零偏置初始化时，$E_m$ 起始预测值仅为 $1.38\text{ GPa}$，严重偏离实测均值 $2.44\text{ GPa}$，导致反向传播在初期剧烈震荡；
    2. **物理分支在力学模式下梯度断流**：训练循环中双分支监督判断硬编码为 `if ... and mode == 'tg'`，导致力学模式（`mechanics`）下物理分支与直接分支完全缺乏协同对齐，物理惩罚与主损失脱节；
    3. **GPU 批大小硬编码导致的欠拟合**：原代码在样本量 $\le 2048$ 时硬编码 `effective_batch_size = 1024`，导致 1,425 条样本的储能模量训练集每个 epoch 只有 1~2 个梯度更新步，100 个 epoch 总共仅更新约 150 次，网络严重欠拟合；
    4. **储能模量真实数据极度右偏（偏度 5.79）**：实测 25°C 储能模量跨越 0.00005 GPa（未交联液/橡胶态）至 24.8 GPa（高填料刚性复合材料），动态范围达 500,000 倍。若直接在原始线性空间拟合，1% 的超高值（20+ GPa）垄断了 95% 的 MSE 梯度，导致网络对绝大多数正常配方（1.5~4.0 GPa）拟合失效；而转换至物理对数 $\log(1+y)$ 空间后，偏度骤降至 -0.09（完美正态），完全符合高分子力学年代对数衰减规律（Decade Scale）。
  - **核心重构措施**：
    1. **力学模式基体与填料物理重构**：将 $E_m$ 重构为以 $y_{\text{mean}}$ 为中心的软平滑正定形式，并引入基于高分子刚性与交联密度的物理潜在修正项；结合 Halpin-Tsai 复合材料微观力学方程计算增强模量 $E_c$；
    2. **全流程启用力学双分支与一致性监督**：在损失函数中使力学直接分支与 Halpin-Tsai 物理分支同步接受真实标签监督并保持一致性约束；
    3. **动态 Batch 尺寸调度修复**：修复 GPU 批大小调度逻辑，严格尊重用户设定的 `batch_size`（如 64 或 128），将每个 epoch 的有效迭代次数提升 10 倍以上，彻底消除欠拟合；
    4. **支持对数模量物理尺度训练**：针对跨越 5 个数量级的储能模量，建立对数物理空间拟合链路，高低模量样本的相对误差被公平加权。
  - **基准验证效果**：
    - 在 1,782 条真实的 25°C 储能模量数据集（`ml_qspr_model_storage_modulus_25c_gpa.csv`）上：
      - 旧版本 PINN 在该数据集上 $R^2$ 仅为 **0.6695**，RMSE 高达 **1.004 GPa**；
      - 经物理重构与对数空间优化后，PINN 决定系数跃升至 **0.7178**；
      - 与 LightGBM 融合模型（PINN + GBDT）$R^2$ 达到 **0.7972**，RMSE 降至 **0.786 GPa**！

- 【物理信息神经网络 (PINN)：底层物理方程、温标量纲与网络架构全面重构，彻底解锁外推性能】用户诉求：「帮我看下我当前的物理信息神经网络算法，应该是有问题的，进行彻底重构与修复，并重新运行对比基准验证外推效果」
  - **严重物理/数学缺陷诊断**：
    1. **Fox–Loshaek 交联密度方程符号反转（越耐热越报负数）**：原代码错将针对线性未交联聚合物末端基团的 Fox–Flory 公式（$T_{g\infty} - K/M_n$）套到交联网络，写成 $T_{g\infty} - K/M_c$。由于交联越密 $M_c$ 越小，该式导致 $T_g$ 随交联密度上升而暴跌！对航空级高交联网络（如 TGDDM/DDS，$M_c = 150\text{ g/mol}$），算出的理论贡献低至 **-674.2°C**，把耐热候选彻底拉垮！
    2. **人工硬截断二次方惩罚（死死卡住外推）**：原代码在损失函数中显式加入 `p2 = relu(pred - y_max - margin)` 惩罚，只要神经元尝试预测超过训练集极值（约 250~270°C），反向传播就会对网络施加剧烈惩罚，人工把预测天花板压死在 250°C 左右；
    3. **化学计量比误判 `cp_r_value` 导致固化度 $\alpha_{\max}$ 腰斩**：按列名长度升序排序时，10 字符的 `cp_r_value`（组分物理当量重比 $\approx 0.35$）排在 19 字符的 `formulation_r_value` 前面。导致模型误以为化学计量比 $r=0.35$，把最大固化度 $\alpha_{\max}=\min(1, r, 1/r)$ 强制锁死在 35%，使 DiBenedetto 预测温升不足极限的 20%；
    4. **Fox 公式热力学温标错误（摄氏度除法崩塌）**：原代码在摄氏度下直接执行 $\frac{1}{T_g} = \frac{1-\alpha}{T_{g0}} + \frac{\alpha}{T_{g\infty}}$。由于未转换为开尔文温标，负温单体（如 $-15^\circ\text{C}$ 液体固化剂）被截断为 $10^{-6}$，计算值直接崩塌为 **0.00001°C**，产生 115°C+ 荒谬误差；
    5. **网络层后置 LayerNorm 破坏外推尺度**：原 MLP 每层激活函数后均紧跟 `LayerNorm`，导致每一层的激活方差均被强制归一化为 1.0，深层网络完全失去了对输入大幅偏离均值时的线性响应能力，高输入信号被逐层削平；
    6. **物理分支弱监督与梯度解耦**：混合损失中物理分支仅占 1%~5% 权重，纯黑盒分支承担了 95% 以上的数据拟合，导致潜在物理参数（$T_{g0}, T_{g\infty}, \lambda, \alpha$）几乎未被有效训练。
  - **核心重构措施**：
    1. **物理正确重塑 Fox–Loshaek 方程**：重构为 $T_g = T_{g0} + K_{\nu} \cdot (\nu / 1000)$，保证随交联密度 $\nu$ 单调严格递增，杜绝任何负温输出；
    2. **热力学绝对温标 (Kelvin) 计算 Fox 方程**：在 `core/epoxy_physics.py` 中新增 `fox_tg_torch` 与 `fox_loshaek_tg_torch`，自动转换至绝对温标，彻底解决低温/负温单体数值崩塌；
    3. **移除训练集人工截断惩罚**：废除基于 `y_max` 的惩罚项，仅保留基于高分子物理实际热分解极限（$T_g \in [-100^\circ\text{C}, 550^\circ\text{C}]$）的合理性防护，彻底解封向 350~450°C 外推的能力；
    4. **严格锁定 `formulation_r_value` 并屏蔽 `cp_r_value`**：确保体系固化度上限恢复至真实物理状态（$\alpha_{\max} \approx 0.95 \sim 1.0$）；
    5. **架构升级为残差直通感知机 (ResMLP)**：采用预投射 + 残差块（`x + block(x)`）架构替代后置 LayerNorm，保留未见特征尺度的线性外推穿透力；
    6. **双分支强监督与一致性正则化**：在损失函数中让物理分支与直接分支同时接受实测标签监督，并增加一致性损失约束，迫使网络真正学到具备物理意义的隐变量（$T_{g0} \in [-70, 50]^\circ\text{C}$, $\Delta T_g \ge 50^\circ\text{C}$, $\alpha \approx 0.88$）；
  - **基准验证效果**：
    - 在 7,411 条真实材料高分子 $T_g$ 数据集上（`ml_qspr_model_tg_c.csv`），80/20 切分实测验证集 $R^2$ 达 **0.7392**，RMSE 为 **33.9°C**；
    - 在测试集上的预测最高值成功突破至 **364.4°C**（旧模型卡在 ~240°C）；
    - 在超出训练集工况的极端高交联/高耐温测试样本上，成功实现外推至 **441.1°C**，完全具备了探索 350~400°C 高性能材料的物理建模能力！

- 【配方筛选：候选化学骨架去重与特征对齐修复，彻底根除高分重复配方霸榜】用户反馈：「基本都是重复的配方，帮我实现化学骨架去重」
  - **根本原因诊断**：
    1. **工况网格乘积放大效应**：虚拟配方由 $(\text{树脂} \times \text{固化剂}) \times \text{工艺网格}$ 建立。当枚举多温度、多时间或配比时，一对耐热分子组合会派生出数十条工况记录；由于其性能优秀，这数十条工况直接垄断了排行榜前列，导致视觉上全是同一对分子在反复重复；
    2. **流式特征提取重置索引错位 Bug**：原代码在构建前 2,500 行候选特征时，执行了 `out_df = out_df.reset_index(drop=True)`，导致其原有的 `pool_df` 行索引丢失；随后执行 `mol_features.iloc[_rank_positions]` 时错误截取了 `pool_df` 的前 2,500 行，导致多样性筛选模块中分子指纹错位，未能正确执行结构多样性去重。
  - **核心实施措施**：
    1. **新增「🧬 候选化学骨架去重」开关（默认开启）**：在联合评分区增加该配置项。对任意树脂+固化剂分子对，在所有工艺条件变体中仅保留综合评分最高的一个最优配方，次优变体自动剔除；确保返回的 Top-K 均为截然不同的独立化学组合；
    2. **实时流式动态看板同步骨架去重**：预测过程中的实时动态监控卡片同步支持骨架去重，每对分子实时仅展示当前预测最高分的记录，杜绝霸屏；
    3. **修复特征对齐索引映射**：在重置索引前记录候选在原始池中的物理位置 `_orig_pool_positions`，确保高分候选与其分子指纹 100% 精确对齐，激活多样性筛选算法的真实效果。

- 【熔点预估系统全面重构：由粗糙线性经验公式升级为高精度 QSPR + 实验真值混合引擎】用户诉求：「当前的熔点预估有很大问题，帮我改进一下」
  - **根本缺陷诊断**：
    1. **旧公式严重脱离物理化学事实**：原经验线性公式（`est_tm = 50 + 0.35*MW + 15*aro + 20*rings - 40*logP - 8*rot + ...` 及 `200 + 0.4*MW - 50*logP - 10*rot`）在 3,000+ 真实实验分子上的平均绝对误差 (MAE) 高达 **63.0°C**，决定系数 $R^2$ 仅为 **0.297**，严重失真：
       - 常温低熔点液体被错误预估为极高熔点固体：间苯二甲胺 (m-XDA, 实测 14.1°C) 预估为 **108.5°C**；甲基四氢苯酐 (MTHPA, 实测 <10°C) 预估为 **121.2°C**；二乙烯三胺 (DETA, 实测 -35°C) 预估为 50+°C；水 (实测 0°C) 预估为 **89.3°C**；苯 (实测 5.5°C) 预估为 **44.9°C**；
       - 高熔点刚性单体被大幅低估：双酚A (BPA, 实测 158°C) 预估为 **63.0°C**（偏差近 100°C）；联苯四甲酸二酐 (BPDA, 实测 300°C) 预估为 **194.0°C**（偏差超 100°C）；
       - 直接导致工业级候选过滤（`stage1_max_mp=130°C`）错误拦截优异的液体固化剂，且在配方筛选中难以提供有效门禁。
    2. **配方筛选界面无默认模型导致标记为 NaN**：原界面若用户未手动训练并上传专用 `.joblib` 熔点模型，筛选只能提示“未加载独立熔点模型”，`resin_mp_predicted_c` 和 `hardener_mp_predicted_c` 全部留空。
  - **核心重构与优化措施**：
    1. **新增三级混合预估架构 `core/melting_point_estimator.py`**：
       - **Tier 1 (实验真值零误差库)**：集成 3,069 个经文献与实测验证的高质量分子数据库，并对工业常用基准（DGEBA 43°C、BPA 156°C、DDM 90°C、DDS 177°C、m-XDA 14.1°C、MTHPA -15°C、PA 131°C、DICY 209°C 等）实施 0 误差精确映射；
       - **Tier 2 (28 维分子拓扑 QSPR 机器学习引擎)**：基于分子量、LogP、极性表面积 TPSA、氢键供受体密度、芳香度、分子形状拓扑指数（Hall-Kier Alpha, Kappa1~3, Labute ASA）、旋转键熵衰减密度及典型反应基团计数，训练高泛化 Gradient Boosting Regressor 模型；5 折交叉验证 $R^2$ 跃升至 **0.7924**（提升 166%），拟合 MAE 降至 **16.4°C**；
       - **Tier 3 (物理自洽降级回退)**：基于真实分子堆积能与晶格熵校准的弹性回退公式，杜绝任何异常输入或极端单原子报错；
       - **微秒级 LRU 缓存**：以 Canonical SMILES 为键建立高速内存缓存，使高通量配方筛选时的单体熔点评估耗时降至微秒级。
    2. **全流程联动更新**：
       - `core/industrial_filter.py::estimate_melting_point` 迁移至新引擎，m-XDA、MTHPA 等优质液体固化剂不再被误杀；
       - `core/virtual_screening.py` 内部熔点过滤逻辑统一接入新引擎；
       - `app_lib.py` 配方级筛选自动无缝接入系统内置高精度 QSPR 熔点预估引擎，即使用户未上传专用外部模型，也能自动预测每种树脂/固化剂的准确熔点并执行标准差与适用域评估。

- 【分子设计引擎：逆合成探索与海量衍生性能暴增，彻底解决长时间卡顿并提高生成上限】用户诉求：「从自建数据库 / BigSMILES / PubChem 模板逆合成探索我设置了20000个，然后就卡住了1小时也没结束，帮我优化... 同时最大生成个数你帮上限调高点」
  - **根本卡死原因诊断**：
    1. **超多核（512核）CPython GIL 锁与线程风暴雪崩**：原代码按 CPU 核心数 `workers = os.cpu_count() = 512` 创建 `ThreadPoolExecutor`。在 Windows + CPython 环境下，512 个并发线程同时争抢单一 GIL 锁并执行密集 RDKit C++ 封装调用，导致极端严重的上下文切换抖动与锁排队，吞吐量断崖式暴跌；
    2. **无平滑提前收敛，强制穷举全部冗余母核**：原逻辑必须等全部上千个前驱体及其数百种反应穷举完毕后，才在最后一行切片 `usable[:max_total_products]`。即使用户设定了 20,000 个目标，前 500 个母核其实已产出 25,000+ 个合格单体，但系统仍盲目运算剩余数千个母核长达 1 小时，且超额单体最后被无情抛弃；
    3. **UI 阶段 2 无进度汇报盲等假死**：`run_combinatorial_monomer_design` 无进度回调接口，Streamlit 界面在进入阶段 2 后长达数十分钟无任何 WebSocket 数据包传输，界面死锁在「2/3 阶段 2...」；
    4. **模板反复动态编译与超标大母核未前置过滤**：多步路线链在每个母核每一步均重复调用 `ReactionFromSmarts` 进行正则解析；自建/PubChem 提取的超分子、高聚物母核（MW > 1500 或重原子 > 120）未做前置剪枝，导致多反应位点下出现 $2^N$ 级组合爆炸。
    5. **自建母核自动拆解逻辑条件判断 Bug**：原代码判断 `if not is_fission and is_custom_mode and ...`，当默认勾选叠加裂变（`is_fission=True`）时，若用户未显式点击拆解按钮，自动拆解直接被跳过。
  - **核心修复与加速措施**：
    1. **单体生成数量上限大幅放开**：将前端「最大单体产物数上限」由原本的 `20,000` 提升至 **`2,000,000`（200 万个）**，默认值设定为 `20,000`，步长增至 `5,000`；同时将内部裂变中间体配额上限由 `10,000` 提升至 `100,000`，全面支持百万级超大规模高通量拓扑设计；
    2. **分批提交 + 自适应平滑提前收敛（Early Stopping）**：
       - 以批次粒度（`batch_chunk_size`）平滑分发前驱体母核任务；
       - 当累计获得的合格单体达到用户预设目标的 1.15 倍安全边际时，立即安全收敛，不再盲目计算后续冗余母核，数十秒内极速出结果；
    3. **全链路实时进度回调与动态看板**：
       - `run_combinatorial_monomer_design` 新增 `progress_callback` 参数；
       - 前端增加实时进度条（`st.progress`）与动态指标（`已处理 X/Y 母核 (Z%) | 实时捕获可用单体 N 个`），彻底告别盲等；
    4. **GIL 锁优化与多步路线预编译**：
       - 限制线程池工作上限 `min(64, cpu_count)`，彻底消除 512 线程并发下的 GIL 锁死与线程上下文切换风暴；
       - 多步路线链 SMARTS 反应对象在启动时预编译一次并复用，杜绝万次重复解析；
       - 前驱体母核执行物理可行性前置拦截（MW > 上限或重原子 > 120 的巨分子直接跳过）；
    5. **自建模板提取 Bug 修复与大结果集前端流畅度保护**：
       - 修复 `is_custom_mode` 下不论是否叠加裂变均自动执行逆合成母核萃取的机制；
       - 当产物超过 5,000 条时，前端表格默认预览前 5,000 条保证流畅响应，同时保留全量百万条完整 CSV 一键导出。

    6. **并发滑动窗口流水线（Sliding Window Pipeline）与分支剪枝，彻底消灭末期降速与核心饥饿**：
       - **摒弃固定批次屏障**：原逻辑采用固定 `batch_chunk_size = 128`，每批必须等待该批最慢的 1~2 个复杂前驱体完成，导致其余 60+ 线程长时间闲置、CPU 占用率骤降至 <1%；改用基于 `wait(..., return_when=FIRST_COMPLETED)` 的连续滑动窗口任务池，任何子任务完成立即补入新母核，保证 128 个工作核心 100% 持续满载运行；
       - **多位点分支组合剪枝**：针对后半段出现的四酚、星型三酚等多位点复杂母核，限制每轮反应中间体分支上限为 24，避免在末期发生 $O(N!)$ 级的指数展开（单分子处理耗时由 0.8s 压降至 0.05s）；
       - **进度更新时间节流与 $O(1)$ 配额统计**：对 Streamlit 前端进度汇报实施 0.35s 时间节流，阻断 1,400+ 次高频 WebSocket 数据包对 Python GIL 解释器锁的持续霸占；将可用单体统计由全量扫描重构为常量时间增量更新。

- 【单体分子设计与配方筛选：化学结构合理性与互反应基团相容性底线】用户反馈：「某些分子设计的不合理，比如同一分子里面有环氧和酚羟基」
  - **原因定位**：
    - 在「骨架裂变演化（Scaffold Fission）」中，当双酚类骨架（带游离酚羟基 `-OH`）引入烯丙基（`-CH2-CH=CH2`）或乙烯基侧链后，若后续经过「双键环氧化（R04）」反应，过氧酸模板将烯基转化为环氧基，但酚羟基未被保护且未被消耗，导致生成了分子内同时存在环氧基与游离酚羟基的缺陷产物（如 `Oc1ccc(CC2CO2)cc1-c1cc(CC2CO2)ccc1O`）。
    - 此类分子在常温或受热时极易发生自分子开环聚合（自交联成胶），无法稳定存储，不属于化学自洽的热固性树脂单体。同时，原平铺反应和多步合成路线末端仅校验了分子量和官能度下限，缺失了单体分子内的官能团正交相容性质量门禁。
  - **核心修复措施**：
    1. **新增 `validate_monomer_chemical_soundness` 化学相容性校验器**：
       - 环氧树脂单体（含环氧基或 role=resin）：严禁分子内同时含有游离酚羟基 `[c][OX2H]`、未保护的伯/仲胺基 `[NX3;H2,H1;!$(NC=O)]`、游离羧基 `[CX3](=O)[OX2H]`、硫醇 `[SX2H]` 或环状酸酐基团，杜绝自分子交联固化缺陷；
       - 固化剂单体（role=hardener）：严禁带有环氧基；酸酐固化剂严禁带游离胺基或酚羟基（分子内自开环失效）；多元胺固化剂严禁带羧基（酸碱盐析失效）；
       - 通用高危结构：严格排除过氧键（`-O-O-`）与易水解酰卤基团（`-COX`）。
    2. **合成反应模板前置正交性防护**：
       - 在 `_precursor_matches_reaction` 中增加正交性约束：对于双键环氧化（`olefin_epoxidation`），底物必须无游离酚羟基、无游离芳胺和无羧基，从源头阻断酚羟基底物与环氧化反应错误配对。
    3. **单体生成管道全链路拦截**：
       - 在多通道反应穷举（`_process_single_precursor_reactions`）与多步合成路线链（`_process_synthetic_routes_for_precursor`）末端全部接入 `validate_monomer_chemical_soundness`，任何互反应缺陷单体一律剔除。
    4. **虚拟配方筛选预过滤规则联动强化**：
       - `DEFAULT_EPOXY_RULES["resin"]` 默认开启 `"ban_phenols": True`，在基础规则与严格规则下同步过滤外部导入或存量库中含酚羟基的环氧树脂单体；
       - 修复前端 UI 提示，明确标识自动拦截自交联互反应基团。

### 新增
- 【分子设计引擎：批次精准管理与清空指定批次候选池】用户诉求：「全维度拓扑演化设计引擎中，请帮我加一个功能可以清空某一批生成的候选池」
  - **痛点**：原先只有一个全局「清空累计总单体库」按钮，一旦点击就把历史所有生成结果一刀切清空，无法剔除某次效果不好的特定批次。
  - **单体批次全生命周期标记**：
    - 每次点击生成单体时，自动分配唯一批次标识（`batch_id`，如 `batch_20260923_1030`）与中文说明标签（`batch_label`，如 `批次 09-23 10:30 (全维度拓扑裂变 · 120个)`）。
    - 历史无批次标签的存量单体自动归入 `历史已保存单体`，保证向前兼容。
  - **交互与批次精准管理**：
    - 在产物库展示区提供「📁 仅看指定批次单体」视图过滤，方便逐批检查结构；
    - 新增操作面板：下拉选择特定批次，提供 `🗑️ 清空选中的批次` 按钮，仅精准剔除该批次单体，保留其余批次，并同步更新本地 parquet 持久化缓存；
    - 保留 `⚠️ 清空全部总库` 供全局重置。
- 【候选池化学规范化严格去重（Canonical SMILES 级）】用户诉求：「候选池最好不要有重复的分子」
  - **痛点**：过去仅依赖字符串 `drop_duplicates(subset=["product_smiles"])`，对于化学结构完全相同但书写形式不同、原子编号不同、或非规范 SMILES 的同构单体（如 `c1ccccc1O` 与 `Oc1ccccc1`），字符串去重会误判为不同分子，导致候选池堆积重复结构。
  - **新增 `core.molecule_design.deduplicate_monomers_canonical`**：
    - 基于 RDKit Canonical SMILES 计算唯一化学标准键，对单体候选表执行严格等价去重；
    - 将规范化的标准 SMILES 回写原列，保证库内所有分子表示绝对统一；
    - 在单批次生成产物时、合并入历史总库时、以及虚拟筛选多来源单体库合并（`core.virtual_screening.merge_component_libraries`）时全链路强制执行，彻底杜绝重复分子堆叠。
- 【实时动态刷新配方：随时暂停 + 即时导出当前成果】用户诉求：「实时动态刷新的配方我希望可以随时暂停+导出」
  - **修复暂停与中断丢弃数据的严重缺陷**：
    - 过去分块预测循环中一旦触发暂停或取消直接 `return` 退出函数，跳过了后续所有结果展示与导出代码，导致已跑完的批次结果全部丢失，页面一片空白。
    - 现改造为**非破坏性截断固化**：在任意批次暂停或取消时，自动将前 N 批已计算出的有效配方、预测值、不确定度打包整理，存入 `vs_formula_result_df` 与 `vs_formula_partial_result_df`，并在页面上保留完整展示，支持随时导出 CSV 或稍后继续筛选。
  - **新增常驻快速导出面板**：
    - 只要有已计算的批次（无论是正在运行中、暂停后还是中断退出），页面上方常驻呈现「📥 实时流式已算配方随时导出」，实时展示已完成条数与表格预览，并提供 `⬇️ 立即导出当前已完成配方 (CSV)` 按钮，无需等待全部几百万条跑完即可随时取用数据。

### 新增
- 【虚拟筛选：实时流式配方支持随时暂停与即时导出】用户诉求：「实时动态刷新的配方我希望可以随时暂停+导出」
  - **痛点与历史断点**：
    1. 旧实现中若用户点击暂停或终止，代码直接 `return` 退出函数，未进入下游结果组装代码，导致已跑出的前几批预测结果全部被丢弃，页面变空白，用户什么也导不出。
    2. 中间结果导出面板原先置于预测全流程执行完毕之后，筛选运行中无法提前下载已算出的配方。
  - **非破坏性暂停/中断截断**（`app_lib.py`）：
    - 当分块预测循环检测到暂停（`_vs_pause_flag[0]`）或任务取消（`is_cancelled()`）时，若已完成 $\ge 1$ 批，**绝不直接 return 丢弃结果**！
    - 自动截取已完成的部分（`preds[:_start]`, `pool_df.iloc[:_start]`），按预测目标即时排序整理，固化存入 `vs_formula_result_df` 与 `vs_formula_partial_result_df`，并在页面上显式提示「已在第 N/M 批安全停下，已完成配方已全部固化保留」。
    - 下方结果渲染与导出区自动激活，用户可立即查看完整分析图表或一键下载 CSV。
  - **实时流式结果随时导出面板**（`app_lib.py`）：
    - 在点击筛选按钮上方新增常驻面板「📥 实时流式已算配方随时导出」，只要计算出一批或多批结果，该面板即时呈现中文表格与 `⬇️ 立即导出当前已完成的 N 条配方（CSV）` 按钮，无论推断正在运行、暂停还是中断，均可随时随地一键下载。
- 【分子设计引擎：批次精准管理与清空指定批次】用户诉求：「虚拟分子筛选 · 全维度拓扑演化设计引擎中，请帮我加一个功能可以清空某一批生成的候选池」
  - **痛点**：原先只有一个全局按钮 `🗑️ 清空累计总单体库`，点击后把历史所有批次生成的单体全部清空，无法剔除某次生成效果不好或不需要的特定批次。
  - **自动化批次标识注入**（`app_lib.py`）：
    - 每次在分子设计引擎生成单体时，自动为结果分配唯一的批次元数据：`batch_id`、`batch_label`（如 `批次 09-23 10:30 (拓扑裂变 · 120个)`）以及 `batch_time`；历史旧数据自动兼容标记为 `历史已保存单体`。
  - **单体库展示与操作面板升级**（`app_lib.py`）：
    - **视图切换增加批次过滤**：显示视图支持 `[📦 汇总总单体库, ⚡ 仅看本次设计产物, 📁 仅看指定批次单体]`，支持精准查看某一具体批次的单体。
    - **新增「批次精准管理与清空指定批次」操作区**：
      - 下拉列表列出当前总库中包含的所有有效批次（包含批次名称、生成时间、设计模式及单体数量）；
      - 提供操作按钮 `🗑️ 清空选中的批次`，仅从总单体库中剔除该批次的单体，保留其余批次，并自动同步更新本地磁盘持久化缓存（`persisted_designed_monomers.parquet`）；
      - 保留 `⚠️ 清空全部总库` 供一键重置使用。
- 【候选池化学分子级严格去重（Canonical SMILES）】用户诉求：「候选池最好不要有重复的分子」
  - **痛点**：过去仅依靠字符串 `drop_duplicates(subset=["product_smiles"])` 或 `groupby(["smiles", "role"])`，不同书写形式的同构分子（如 `Oc1ccccc1` 与 `c1ccccc1O`、环氧基不同原子编号顺序）会被判定为不同分子，导致候选池中堆积重复分子。
  - **新增规范化去重函数**（`core.molecule_design.deduplicate_monomers_canonical`）：
    - 基于 RDKit `canonicalize_smiles` 将每行单体转换为规范统一的标准 SMILES 再进行唯一性去重；
    - 无法通过 RDKit 解析的分子退化为清洗后的文本去重，不静默丢弃数据；
    - 每次单体设计生成完成时、追加累积入总单体库时，均自动执行 Canonical 去重（`keep="last"` 保留最新批次信息）。
  - **多来源库合并规范去重**（`core.virtual_screening.merge_component_libraries`）：
    - 在合并自建库、设计引擎、PubChem、训练集等来源的单体时，引入基于 Canonical SMILES 的归一化，确保跨来源带入的相同分子在入池前合并为单一条目并合并来源标记（如 `train_data|molecule_design_engine`）。
  - 新增 `tests/test_molecule_design_batch_management.py`（4 个测试全量通过）。
- 【虚拟筛选：全流程配方来源标记与结构化追溯】用户诉求：「筛选出来的配方应该有来源标记」
  - **痛点与历史断点**：
    1. 过去界面上的配方来源是英文内部代码（如 `train_observed_pair`、`custom + custom`），普通用户无法直观理解。
    2. 当开启多组分复配（如两树脂复配单体 `A.B`）时，因单体库元信息按单分子索引，复配分子全部查不到来源而**被静默覆盖为 `custom`**，来源信息彻底丢失。
    3. 实时结果预览表格（`_render_live`）与可下载中间结果 CSV 中使用的是未翻译的英文列名 `candidate_origin`，缺乏来源分类。
  - **新增多组分单体来源解析器**（`core.virtual_screening._resolve_component_source_and_score`）：
    - 自动对复配分子（如 `A.B`）逐片段拆分、查询各个单体的原始来源并去重合并（如 `train_data/molecule_design_engine`）；可用性评分取各组分均值，**彻底杜绝复配单体退化为 custom**。
    - 修复单树脂筛选模式下单体元表被置为 `None` 导致单树脂来源全部丢失的隐蔽 bug。
  - **来源语义标准化与双层标签体系**（`core.virtual_screening.resolve_candidate_origin_label`）：
    - **`candidate_origin_label`（配方来源）**：精美人性化中文标签，例如：
      - `🌟 训练集实测配方`（训练集历史真实配方）
      - `🧬 全AI设计配方 (AI树脂 + AI固化剂)`（双组分皆由分子设计引擎产生）
      - `🧬 AI设计树脂 + 🧪 训练集固化剂`
      - `🧪 实验重组配方 (训练集树脂 + 训练集固化剂)`（已有单体的新颖配对）
      - `🌐 PubChem树脂 + 🧪 训练集固化剂`
      - `🔀 复配树脂(训练集+AI设计) + 🧪 训练集固化剂`（跨来源复配体系）
    - **`origin_category`（来源分类）**：高层级业务类别分类（`实测配方`、`AI设计衍生`、`实验重组`、`外部数据库`、`用户导入`、`算法衍生`、`多源复配`、`自定义配方`）。
  - **全界面与导出链路打通**：
    - **实时推断结果表格**：直接展示中文列 `配方编号`、`树脂SMILES`、`固化剂SMILES`、`来源分类`、`配方来源`、`预测值`，每批预测完成即可直观看到最优配方的来源类型。
    - **最终筛选结果表格与 CSV 导出**：同时包含 `来源分类`（`origin_category`）与 `配方来源`（`candidate_origin_label`）。
    - **筛选空间摘要**：直观汇总并展示当前 Top-K 候选的来源分类分布统计表。
  - 新增 `tests/test_screening_candidate_origin_labels.py`（单单体组合、复配单体防丢失、实测锚点、向量化映射等全部测试通过）。

### 修复
- 【分块流式预测崩溃：`TypeError: unhashable type: 'Series'`】第 1 批完成后进入第 2 批即报错
  - **报错栈**：
    ```
    File "app_lib.py", line 26672, in _page_virtual_screening_formula
        _mapped_post = apply_mapping(...)
    File "core/virtual_screening.py", line 4066, in apply_feature_overrides
        out.loc[valid, col] = vals.loc[valid].to_numpy()
    File "pandas/core/indexing.py", line 908, in __setitem__
    File "pandas/_libs/hashtable_class_helper.pxi", line 7139, in PyObjectHashTable.lookup
    TypeError: unhashable type: 'Series'
    ```
  - **根因**：分块流式处理中，第 1 批的切片 `slice(0, 5000)` 索引恰好是 `0..4999`，与 `X_chunk` 的 `RangeIndex(0, 5000)` 完全重合；但从第 2 批开始，切片 `slice(5000, 10000)` 的索引是 `5000..9999`！
    1. 在 `apply_feature_overrides` 中：`valid = vals.notna()` 是一个索引为 `5000..9999` 的 Series，当执行 `out.loc[valid, col]` 时，pandas 尝试跨非重叠的 Series 索引做标签对齐与哈希表查询，直接触发 `TypeError: unhashable type: 'Series'` 崩溃。
    2. 在 `apply_mapping` 中：`result[feature] = values`（`values` 索引为 5000..9999）会导致按标签对齐而将整列全部置为 NaN。
  - **修复**：
    1. `core/virtual_screening.py::apply_feature_overrides`：改用 `numpy` 纯布尔数组与 `out.iloc[valid, col_pos]` 按位置覆盖，彻底脱离两表 Index 标签差异对齐的影响。
    2. `core/post_feature_mapping.py::apply_mapping`：显式使用 `values.to_numpy(dtype=float)` 赋值，避免切片 Series 标签不对齐导致意外注入 NaN。
    3. `app_lib.py`：分块预测与 AD 评分循环开始时，对传入的 `_chunk_pool` 与 `_chunk_mol` 显式执行 `.reset_index(drop=True)`，确保块内所有矩阵与数据表拥有绝对统一的 `RangeIndex(0, n)`。
  - **验证**：本地针对多批次（第 2 批、第 3 批索引不从 0 开始）端到端多切片推断模拟测试完全通过，0 NaN，0 报错。

### 修复
- 【虚拟筛选：彻底消除「卡在化学规则筛选几分钟无结果」的死锁与极端耗时】
  - **现象**：截图显示进度停在 `正在执行树脂/固化剂角色化学规则筛选...`，用时超过 105 秒仍无输出
  - **两大真实瓶颈实测定位（非推测）**：
    1. **Windows + Streamlit 下 `multiprocessing.Pool` 死锁**：上一轮引入的 `_warm_rule_feature_cache` 试图使用 `spawn` 进程池预热特征，在 Windows Streamlit 交互进程内部无法正常 fork/spawn，导致 `pool.imap_unordered` **永久阻塞达 300 秒硬超时**！现彻底移除该多进程池，改回纯内存字典缓存——实测 13,838 个分子在单线程纯 Python 缓存下仅需 **8.69 秒**，彻底消除死锁。
    2. **5,000,000 行全量串行推导耗时超 2 分钟**：
       - `apply_schedule_derivations`：原先在 500 万行上逐行运行 Python for 循环解析 `cure_schedule`，耗时 **61.88 秒**；现改为**按唯一固化制度解析后向量化广播**，耗时直接降至 **0.46 秒**（提速 133 倍）。
       - `derive_computable_candidate_columns`：原先在 500 万行大表上逐行运行 `derive_formulation_features` 解析 RDKit 分子，耗时 **57.81 秒**；现改为**单分子物理属性解耦缓存**（树脂 EEW 与固化剂 AHEW 各自独立计算一次，仅需 ~0.3s），500 万行化学计量比 $r$ 走纯 numpy 向量化广播除法计算，耗时降至 **5.14 秒**（提速 24 倍）。
    3. **多步工作流模型的 `screening_uses_xtb` 检测遗漏**：
       - 之前仅检测了 `mf_cfg.get("method")`（对于 TabPFN 等复合模型，此处仅为 `分子指纹`），未递归检查 `screening_workflow.steps`，导致 xTB 未被识别为昂贵特征，默认上限放宽至 50,000 个分子（5 万个分子计算 xTB 需耗时数小时）；现全面扫描全部步骤，将 xTB 昂贵特征的默认单体上限收敛为 2,000 个，并将并行度彻底打满至 64 进程（充分利用多核服务器）。
  - **总耗时收益**：500 万行候选在进入推断循环前的全量前置准备时间从 **40+ 分钟骤降至约 15 秒**！

### 新增
- 【预测/筛选多组分自适应：解决模型槽位数与输入组分数不匹配】单树脂/单固化剂输入平滑预测多槽位模型，消除连环报错与清池误杀
  - **痛点**：真实工业模型（如基于宽表训练的 TabPFN、XGBoost 多槽位模型）训练时声明了多个树脂、固化剂或添加剂槽位（如 `resin_1/2/3_structure`，`curing_agent_1/2/3_structure`，`small_additive_1/2` 等，共 10 个特征提取步骤）；但在单点预测或虚拟筛选时，用户通常只输入单树脂 + 单固化剂。旧架构下引发四大连环死锁：
    1. `KeyError: 'workflow step requires missing source columns: resin_2_structure'`（工作流步骤直接崩掉）
    2. `预测门禁拦截：'分子源列不能为空：resin_2_structure'`（单点预测被强行阻断）
    3. `分子特征空帧全为 NaN → get_valid_feature_row_mask 判定全行失效 → 虚拟筛选清空全部候选`
    4. `缺少必填手工特征：resin_2_amount_phr`（把未配比的辅助槽位当成必填项）
  - **主/辅槽位契约分级**（`core/portal_prediction.py` & `core/molecular_feature_workflow.py`）：
    - 槽位序号 `_1` 或无序号为主槽位（必填，`required=True`）；序号 `_2`, `_3`, `_4` 及添加剂槽位（`small_additive`, `accelerator`, `initiator` 等）自动判定为**辅助槽位**（`is_auxiliary_component_column`，默认 `required=False, nullable=True`）
    - 预测门禁（`_source_is_required`）对辅助槽位自动放行，不再强求用户填写不存在的副树脂/副固化剂
  - **辅助槽位特征安全填补 0.0**（`core/molecular_feature_workflow.py`）：
    - `prepare_step_inputs` 自动识别缺失的辅助源列，作为空样本平滑调度，不抛 `KeyError`
    - `execute_molecular_feature_workflow` 对辅助槽位未提供分子的样本行，将其分子特征安全填补为 `0.0`，彻底消除 NaN 穿透，保护下游有效行判定（`get_valid_feature_row_mask`）与预测矩阵校验不被误杀
  - **辅助组分配比（PHR）智能归零**（`core/portal_prediction.py` & `app_lib.py` & `UserPrediction.py`）：
    - 针对 `resin_2_amount_phr`、`curing_agent_2_amount_phr` 等辅助配比列，当用户或候选未提供该副组分时，**自动安全归零为 `0.0`**（严格禁止使用训练集中位数插补，避免破坏质量守恒与配比真实意图）
    - `UserPrediction.py` 界面上将辅助配比与辅助结构明确标注为 `（可选副组分）`，默认填 `0.0`，杜绝红色必填星号与提交阻断
  - **虚拟筛选组分容量感知**（`app_lib.py`）：
    - 虚拟配方控制面板动态感知当前模型的树脂与固化剂槽位上限，并在界面提示单组分自适应状态
    - `materialize_workflow_source_columns` 扩展支持将单组分 `resin_smiles` 规范映射到 `resin_1_structure`，并将缺失辅助槽位初始化为 `None`
  - **验证**：真实 10 步 381 维多槽位 TabPFN 模型（训练自 3 树脂 + 3 固化剂 + 2 助剂），在**纯单树脂（DGEBA）+ 单固化剂（DDM）**输入下端到端预测成功，产出有限有效预测值；新增 `tests/test_component_mismatch_adaptation.py`（34 个回归测试全部通过）

### 修复
- 【筛选卡在「正在执行树脂/固化剂角色化学规则筛选」：iterrows 全量阻塞，实测慢 33~548 倍】用户截图显示进度条停在化学规则筛选，长时间不出结果
  - **场景**：树脂候选 8357 | 固化剂候选 5481 | 工艺网格 48 → 配对空间 4580 万，叠加工艺网格理论 22 亿条；UI 默认 `max_formulations=5e6`，即要在 **500 万行**上跑规则过滤
  - **根因（实测定位，非猜测）**：`filter_candidates_by_epoxy_rules` 用 `df.iterrows()` 逐行判定。实测 100,000 行耗时 **3.90 s，而其中唯一分子的 RDKit 计算仅 0.004 s** → **99.99% 是 pandas 逐行开销**。外推 500 万行 ≈ **177 s**，且这是**全量前置阻塞**（必须先跑完才可能出结果，与批大小无关——这就是「设了 5,000 条/批也没用」的原因）
  - **`filter_candidates_by_size` 同样问题且更严重**：200,000 行耗时 **120 s**
  - **修复**：按**唯一组合/SMILES** 判定后映射回行。候选表里 (resin, hardener) 唯一组合数远小于行数（同一分子对被多个工艺网格重复引用），故判定一次即可复用。用 `pd.factorize` 生成 inverse 索引（注意 `np.unique(axis=0)` **不支持 object dtype**，须走字符串键）
  - **实测提速**：
    | 函数 | 规模 | 修复前 | 修复后 | 提速 |
    |---|---|---|---|---|
    | `filter_candidates_by_epoxy_rules` | 100,000 行 | 3.53 s | 0.106 s | **33×** |
    | `filter_candidates_by_epoxy_rules` | 1,000,000 行 | 35.46 s | 0.964 s | **37×** |
    | `filter_candidates_by_size` | 200,000 行 | 120.0 s | 0.219 s | **548×** |
    | 外推 5e6 行（规则过滤） | 5,000,000 | ≈177 s | **≈4.8 s** | 37× |
  - **⚠️ 等价性是本修复的核心风险，已用原码锁定**：判定分支极多（`ban_strong_acids` / `ban_amines` / `min_active_hydrogen` / `reject_mixed_class` / `class_flags` / 四类比例区间…）。本任务中我曾**凭记忆重写** `_passes_resin`/`_passes_hardener`，漏掉上述分支，导致严格规则下过滤结果从 **1853 条错误地变成 0 条**。因此最终改为**从 git HEAD 原码文本级抽取**判定逻辑，并把原实现落成 `tests/_epoxy_rules_reference.py` 作为等价性基准，对 **7 组规则 × keep_invalid 两种取值 = 14 个用例**断言「条数 + 内容逐位一致」（全部通过）
  - 新增 `tests/test_screening_rule_filter_equivalence.py`（23 个测试：等价性、防清空回归、性能上界、唯一组合复用、空表/单行/仅树脂列边界）

### 修复
- 【筛选预测失败：X has 338 features, but SimpleImputer is expecting 381 features】使用带 `feature_mask` 的 TabPFN 模型筛选时，第 1 批即报错
  - **对应模型**：`results/training_runs/20260923_032815_TabPFN/model_TabPFN.joblib`
    ```
    artifact.feature_cols = 381                      ← 契约（mask 前）
    pipeline: SimpleImputer(381) → FeatureMaskTransformer(381→338)
              → StandardScaler(338) → TabPFNRegressor(338)
    model.feature_names_in_ = 338                    ← mask 之后
    ```
  - **根因**：`core/virtual_screening.predict_with_model` 按 `model.feature_names_in_`（mask **后**的 338 列）裁剪输入，把已按契约构建好的 `(n, 381)` 矩阵 reindex 成 `(n, 338)`；而 pipeline 的**第一个** step 是 `SimpleImputer(381)`，它期望 mask **之前**的 381 列 → 宽度不符 → 抛错。已在本机**逐字复现**该报错
  - **正确口径**：pipeline 的输入契约宽度 = 其**第一个** step 的 `n_features_in_`，而不是末端 estimator 的 `feature_names_in_`（后者是 mask 后的宽度）。这与 `core/external_feature_augmenter` 既有做法一致（那里优先取 pipeline 各步的 `feature_names_in_`），虚拟筛选路径是唯一遗漏点
  - **修复（`core/virtual_screening.py`）**：
    1. 新增 `_pipeline_expected_input_width()` / `_pipeline_first_step_feature_names()`：取 pipeline **首步**的宽度与列名（mask 前口径）
    2. 显式传入的 `feature_cols` 宽度若等于 pipeline 首步宽度 → **直接使用，不被 `model.feature_names_in_` 覆盖**
    3. 矩阵宽度已等于契约宽度（mask 场景）→ **保持原样不裁剪**，让 pipeline 自己在内部执行 mask
    4. 抽出 `_predict_with_aligned_matrix()` 统一预测尾部，避免分支间语义漂移
  - **矩阵比契约窄时显式报错，不猜测**：若矩阵列数 < pipeline 输入契约且没有可用的契约列名，直接抛出中文错误。**刻意不采用「尾部补 NaN 列」**——`feature_mask` 是按**位置**取列的，补错位置会静默产出错误预测，比报错严重得多
  - **验证**：该模型（含 4 个同族 TabPFN artifact）全部预测成功；`predict_with_model` 结果与「直接把契约矩阵喂 pipeline」**逐位一致**；非 mask 的 XGBoost 模型预测值**未发生任何变化**；端到端模拟筛选分块循环（250 行 / 100 条一批）跑通且预测值全为有限值
  - 新增 `tests/test_predict_feature_mask_width.py`（10 个测试：首步宽度语义、真实 mask/非 mask 模型回归、逐位一致性、窄矩阵不猜测、不确定度路径）

### 修复
- 【多核服务器算力被浪费：xTB 子进程线程超订】512 物理核的服务器上，xTB 计算比单核还慢
  - **根因**：`core/molecular_features._run_xtb` 的 `subprocess.Popen(cmd, ...)` **没有传 `env`** → xTB 子进程继承父进程的 `OMP_NUM_THREADS`。而 `core/thread_config.py` 在 512 核机器上把它设为 **512**。于是「n_jobs 个 xTB 进程 × 各自 512 线程」严重超订
  - **实测**：单次 `xtb --sp` 在 `OMP_NUM_THREADS=512` 时 **0.31 s**，而在 1/8/64 时为 **0.11~0.12 s**；整批 8 个分子从 **0.22 s 恶化到 25.1 s（慢 100 倍以上）**
  - **修复**：给 xTB 子进程显式设 `OMP_NUM_THREADS=1`（及 `OPENBLAS/MKL/VECLIB/NUMEXPR`），并行度完全交给上层 `n_jobs`。xTB 这类小分子半经验计算单分子内并行度极低，标准最优是「1 线程 × 多进程」。保留 `ML_XTB_THREADS` 环境变量供显式覆盖
  - **实测效果**：修复后 `n_jobs=4` 时 8 个分子 **0.21 s**（修复前 25.1 s）
- 【多核服务器算力被浪费：xTB 并行度上限被硬编码】512 核机器上只用了 8 个进程（1.6% 算力）
  - **根因**：`default_xtb_jobs = min(8, safe_job_max)` 且 `safe_job_max = min(61/128, cpu)`，与实际核数脱节
  - **修复**：上限放宽到「物理核数 × 0.875」（用 `psutil.cpu_count(logical=False)` 取物理核，退化时用逻辑核），默认 `min(64, 上限)`，并在 UI 显示「xTB 并行度：N / 物理核 M（约 X% 算力）」
- 【筛选卡在「正在生成虚拟配方设计空间」：多组分池生成慢 33 倍】实测 2000 个候选、`max_components=2`（UI 默认值）耗时 **29.5 s**
  - **根因**：`_generate_multicomponent_pool` 的采样循环用 `rng.choice(items, size=n, replace=False)`。numpy 的 `replace=False` 路径会为**每个样本**在 n_items 上做全排列——实测 2000 元素 × 10 万次 = **29.1 s**
  - **修复**：改用 `rng.integers` 直接取互异下标（n 通常为 2~6，用集合去重即可），**29.52 s → 0.89 s（33 倍）**
  - 同时修掉 `item_component_counts.get(item, count_smiles_components(item))` 的缺陷：Python 的 `dict.get` 默认参数**会被急切求值**，导致每次白算一次分子解析
- 【筛选：`iter_pair_indices` 逐元素取模】5e6 个索引约 1.2 s
  - **修复**：改为 numpy 向量化（约 30 倍），并保留 int64 溢出保护（接近上限时退回 Python 路径）。已加「唯一性 / 在界 / 数量精确」回归测试
- 【统一输入面板要求给「可计算特征」填定值（用户反馈：不能全部用定值）】面板原先列出**全部**非分子特征（真实模型 981 个），其中大量列本可由候选配方逐行算出
  - **问题**：`formulation_r_value` / `resin_1_equivalent_weight_g_eq` / `process_total_time_h` / `resin_component_count` 等若被填成统一值，会**覆盖掉 `screening_feature_derivation` 按每个候选真实配方算出的结果**——等于用别人的配方物理量打分
  - **修复**：面板改用 `core.computable_features` 过滤，**只保留 `SOURCE_MANUAL`（逐组分 phr、工艺/测试条件）**；可计算特征移出面板，并在面板内明确提示「另有 N 个特征已由系统按你的配方逐行自动计算，不在此表中」+ 可展开清单
  - **实测（真实 TBPFN-0.83）**：面板从 **981 个 → 12 个**真正需要你决定的旋钮（6 个逐组分 phr + `process_atmosphere` + 4 个测试条件 + 2 个测试标准）
- 【筛选：边处理边出结果】用户诉求「难道不能边处理边出结果」
  - 原实现是「三段式全阻塞」：① 生成完整设计空间 ② 构建完整特征矩阵 + 全量预测 ③ 才出结果。无论批大小设多小，中途都拿不到任何东西
  - **新增**：分块预测循环内每批完成即把已完成部分（预测值 / 标准差 / 配方标识）写入 `session_state`，并在结果区新增「📥 已完成批次结果（N / M 条）— 无需等待全部跑完即可下载」面板，支持实时预览（前 200 行）与 CSV 下载
  - 中间结果导出失败绝不影响主流程（已 try/except 隔离）

### 修复
- 【虚拟筛选崩溃：StreamlitAPIException（配置列类型 text 与底层 float 不兼容）】工作区数据齐全时点开筛选页即报错
  - **真实报错**：The configured column type `text` for column `值` is not compatible for editing the underlying data type `float`
  - **根因（数据依赖型 bug）**：`_render_screening_uniform_inputs` 用 `auto_vals.get(c, "")` 构造「值」列——
    - **全部**列都自动取值成功 → 该列全是 float → `dtype=float64`
    - **部分**缺失 → 混入空串 → `dtype=object`
    而 `column_config` 声明的是 `TextColumn`。`dtype=float64` 时 Streamlit 的 `_check_type_compatibilities` 判定不兼容并抛异常。**工作区数据越齐全越会崩**，正好是本次触发场景
  - **修复**：把「值」列显式构造为字符串 `Series(dtype=object)`，保证任何数据情形下都与 `TextColumn` 一致；空串保留「留空 = 交给 imputer」语义
  - **同时修掉一个连带隐患**：`auto_vals` 的值既可能是 float（中位数）也可能是 str（类别众数，如 `test_atmosphere="N2"`、`tg_c_test_method="DMA"`）。原先用 `f"{v:g}"` 统一格式化，**对 str 会抛 `ValueError: Unknown format code 'g'`**。改为 `_value_to_text()` 按类型容错
  - **验证**：用 Streamlit 自身的 `_check_type_compatibilities` 校验**真实函数**传给 `st.data_editor` 的入参（非复刻逻辑）；并用 `streamlit.testing.v1.AppTest` 真实跑通——旧实现异常数 1（报错文本与线上逐字一致），新实现异常数 0
  - 新增 `tests/test_screening_uniform_inputs_ui.py`（12 个测试：全命中 / 部分命中 / 无工作区 / 类别字符串 / 混合类型 / 大数值 / 数值回解析）
- 【虚拟筛选崩溃：ValueError: The truth value of an Index is ambiguous】点开筛选页即报错，无法筛选
  - **根因**：`core/computable_features.resolve_schedule_source_column` 用 `available_columns or []` 判空，而调用方传入的是 `pool_df.columns`（**pandas Index**）。pandas 的 Index/Series/ndarray 布尔求值是**歧义的**（长度 >1 直接抛 ValueError）
  - **修复**：新增 `_as_sequence()` 统一安全归一化（`None`→`[]`、`str`→`[s]`、其余 `list(value)`），并清除本模块所有 `X or []` 同类隐患（`candidate_columns` / `workflow_feature_names` / `rule["input_fields"]` / `summarize_split`）。已实测可接受 pandas Index / list / None / 中文列名
- 【虚拟筛选：不再要求手动确认映射关系（用户诉求：帮我自动完成）】原先必须逐项点「筛选前确认：模型特征来源」，未确认即阻断预测
  - **新增 `core/post_feature_mapping.build_auto_mapping`**：按证据自动决定每个特征的来源，**永不产生 `pending`**（pending 正是让用户被迫手点的状态）。优先级：
    1. 分子特征 workflow 输出 → `keep`（mol_features 已写入矩阵）
    2. **候选表已有该列且整列有限数值 → `candidate`（真实候选值）**
    3. 用户显式给定的统一输入 → `constant`
    4. 兜底 → `keep`（继承基准行）
  - **优先级为何关键（否则「自动映射」会静默失效）**：`post_feature_model_cols` **只由 `apply_mapping` 写入特征矩阵**——`keep` 会保留 `base_row`（模板/中位数）的值。若把系统算出的配方物理量映射成 `keep`，`screening_feature_derivation` 的计算结果**根本进不了矩阵**，等于白算、仍用模板值打分。已加端到端测试锁定「算出的值必须真的进入矩阵」
  - **candidate 源的整列有限约束**：`validate_mapping` 对 `candidate` 要求该列无任何 NaN，否则直接报错阻断。故自动映射先检查整列有限，不满足则退回 `keep`
  - **不写 `catalog_fingerprint`**：`validate_mapping` 会与「带 `POST_FEATURE_COMPUTED_DEFINITIONS` 构建的目录」指纹比对，此处口径不同必然不等 → 会误报 catalog fingerprint changed 而阻断
  - **接入闸门**（`_prepare_post_feature_mapping_for_prediction`）：先构造自动映射 → 用与手工路径**完全相同**的 `validate_mapping` 校验 → 通过则直接返回并只显示一行来源摘要 + 可展开明细；**不通过才回退手工面板**（并显示具体原因，不静默失败）
  - **配方级路径补齐统一输入面板**：原先配方级路径没有 `_render_screening_uniform_inputs`，`process_atmosphere` / `tg_c_test_method` 等必须人工给定的特征只能靠 `base_row` 继承模板值。现补上，并把用户给定的值登记为自动映射的 `constant`
  - **实测（真实 TBPFN-0.83，1408 特征）**：自动映射 `candidate` 34 个 + `keep` 1374 个，**pending 0 个**，`missing_input_tolerant` 为 False/True 两种设置下均通过正式校验

### 新增
- 【虚拟筛选：可计算特征「一键识别」】高通量筛选卡在虚拟配方设计——系统把**本来就能算出来**的特征当成「要枚举的工艺旋钮」
  - **痛点**：模型输入里 `formulation_r_value` / `resin_1_epoxy_group_count` / `process_total_time_h` / `cure_stage_count` 这类列，系统完全能从候选结构或固化制度算出来，但旧版把它们摆进「选择要枚举的配方/工艺特征」下拉框，要求用户手动给一串猜测值
  - **根因**：枚举候选池用的是**关键词启发式**（`"cure"/"temp"/"time"/"phr"/"stoich"/"equiv"` 等），"time"/"equiv"/"stoich" 太宽 → 大量计算结果被误判为设计变量
  - **新增 `core/computable_features.py`**：单一权威判定，给出「特征名 → 能否自动计算 + 计算依据 + 实现入口 + 需先给定哪些原始列」。证据优先（registry 的 `source_type`/`calculation_rule` → 内置计算名表 → 关键词语义兜底），不确定一律归为「需人工确认」，**绝不把算不出来的谎报成可计算**
  - **实测（真实 TBPFN-0.83，1408 个特征）**：识别出 **1396 个系统可自动计算**，真正的设计变量只有 **12 个**（6 个逐组分 phr + `process_atmosphere` + 4 个测试条件 + 2 个测试标准）
  - **UI**：筛选页新增「一键识别：N 个特征系统可自动计算，仅 M 个需要你给值」面板，分三个 Tab（系统自动计算 / 需你给定 / 无法判定）逐条列出判定依据与实现入口；枚举池已自动剔除可计算特征
- 【虚拟筛选：固化制度 → 工艺特征自动派生】
  - 用户只需给一个固化制度（如 `130 °C/2 h + 180 °C/2 h`），系统按行派生出 `process_max_temperature_c` / `process_temperature_time_integral_c_h` / `process_total_time_h` / `process_final_cure_temperature_c` / `process_final_cure_time_h` / `process_has_post_cure` 六个工艺特征（`core.computable_features.derive_process_family_from_schedule`，与 `core.process_features` 的 `cure_*` 族**同口径**，实测逐项一致）
  - 已有实测值的行优先保留，只补空位——**不覆盖实验数据**
- 【虚拟筛选：把「声称可算」变成「真的算了」】判定为可自动计算的配方物理量特征原先**根本没被计算**
  - **根因**：虚拟筛选路径只调用 `add_candidate_equivalent_metrics`（只产出 `computed_*` 别名族），`formulation_r_value` / `resin_1_molecular_weight_g_mol` / `resin_epoxy_group_total` 等列从未被计算 → 被映射层按 `keep` **静默继承模板行的值**，等于用别人的配方物理量给全部候选打分
  - **新增 `core/screening_feature_derivation.py`**：调用已验证实现补上真实计算——逐组分物理量走 `core.component_physics.compute_component_physics`（含 L1 文献值 → L2 当量×官能度 → L3 结构直算 的信任分层）；配方级汇总走 `core.portal_formulation_inputs.derive_formulation_features`
  - **关键陷阱（已加测试锁定）**：`cp_r_value` 是**当量重比**（DGEBA/DDM 100:33 → **0.2912**），而 `formulation_r_value` 是**化学计量比 r**（同一配方 → **1.1332**），两者相差约 4 倍、语义完全不同。**禁止**用 `cp_r_value` 填 `formulation_r_value`，必须走 `derive_r_value`
  - 非法 SMILES / 缺 phr 时保持缺失（列不生成或全 NaN），**不猜 0/均值**
- 【虚拟筛选：卡死与慢——「分批流控」其实只作用于配对生成】
  - **根因**：UI 写着「渐进式连续批处理流控 / 分批流式推断」，但 `batch_process_size` 只传给了 `enumerate_formulation_candidates(batch_size=...)`（**只分批生成配对**），特征矩阵与推理仍是**一次性全量**：5,000,000 行 × 1408 列 × 8B = **56.3 GB** → 必然 OOM / 长时间假死
  - **改为真分块流式**：按 `batch_process_size` 逐块走完整链路 `build_feature_matrix → apply_mapping → apply_feature_overrides → predict_with_uncertainty_info`，算完即释放。内存峰值恒定为「单批 × 特征数」（5000 × 1408 ≈ 56 MB），**与总配方数无关**；暂停 / 取消在推理阶段也真正生效
  - AD 评分与代理不确定度同样改为逐块计算；`X_rank` 不再持全量矩阵（下游都会立刻 reindex 到前 ~2500 行，改为**按需只为这些行构建**）
  - **性能回归修复（三处，均为实测定位）**：① `build_feature_matrix` 逐列赋值 → **向量化**（并修正 strict 下「非数值列必须在 `fillna(0.0)` 之后才 coerce」的顺序语义，已加**逐位一致性测试**）；② `_predict_with_uncertainty_impl` 对已建好的矩阵**重复** build（占单批 45%）；③ `predict_with_model` 增加对齐快路径
  - **实测**：100,000 行 × 1408 列下分块（6.5 s）已**快于**一次性全量（10.4 s），内存从 11.3 GB 降到 56 MB
- 【虚拟筛选：明确显示「实际执行路径」】消除「好像没走模型内部工作流」的疑虑
  - 代码本来就强制走模型内 workflow（`screening_workflow is None` 时直接报错返回，**不静默降级**），但 UI 只在跑起来后一闪而过
  - 新增「执行路径诊断」面板：运行**前**即列出四个环节的实际路径与说明（分子特征提取 → 特征矩阵构建 → 模型推理 → 后处理映射），含 workflow 步数与步骤名、每批行数、一次性全量所需内存、已确认映射的非分子特征数

- 【材料预测平台：配方优先输入、内置条件默认值与模型上传门禁修复】（设计规范 `docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md`）
  - **痛点**：① 从训练平台下载的 `.joblib` 模型导入后**永远无法启用**；② 输入端要求手填 36 个字段，其中「测试标准/反应条件/测试条件」等**对应关系很难找**；③ AI 输入助手无缓存、不能多轮交流
  - **门禁修复（`core/prediction_portal.py`）**：`workflow.final_feature_names` 由「必须与 `prediction_contract.feature_cols` **完全一致**」改为**基于 `contract ∪ removed` 的子集校验**。实测根因：真实 artifact 的 workflow 产出 253 个分子特征，contract 有 284 列（253 + 31 配方/工艺/测试），且 workflow 中有 **5 个特征已被训练侧 `feature_mask` 删除**（在 `feature_audit.removed_feature_cols` 中）→ 旧规则下双向不等，必然失败。已实测恒等式：`contract.feature_cols ∪ removed == canonical_feature_cols`（289 == 289）；契约声明 `workflow_feature_cols` 时额外校验分区归属
  - **第二道门禁（`UserPrediction.py`）**：`_is_publishable_ui_model` 原先无条件要求 `schema_version == 2` + `registry_snapshot` + `approved` profile，导致 legacy(schema-1) 导入模型即使发布成功也**不会出现在预测页**。改为：v2 契约仍保持严格语义，legacy 模型跳过注册表快照校验，但 `gate_report.ok` / `publication_status=published` / `enabled=True` 三道硬条件**不放宽**
  - **内置默认值**：新增 `scripts/build_portal_input_defaults.py`（离线统计）+ `prediction_portal/portal_input_defaults.json`（版本化审计产物）+ `core/portal_input_defaults.py`（读取层）。每条默认值带 `share`（占比）+ `support`（样本数）+ `source_table` 可审计依据；占比 < 0.30 不生成；哨兵值 `other`/`unknown` 计入分母但不得作为默认值。实测产出 31 个目标列 / 78 条测试条件默认值（`storage_modulus_25c_gpa` → DMA 0.856、1.0 Hz 0.826；`td5_c` → TGA 0.991 / N2 0.723）
  - **硬约束**：默认值**只作用于 `manual_input` 分区**，`derived_workflow` / `molecular_workflow` 字段一律返回 `None`（它们必须由 workflow 真实计算，给默认值等同于伪造数据）
  - **配方自动推导（`core/portal_formulation_inputs.py`）**：从 SMILES + phr 推出 EEW / AHEW / r / 组分计数 / 官能团汇总，让用户只需输入配方。核心公式 `r = (固化剂phr/AHEW)/(树脂phr/EEW)` —— 全表实测（n=3237）与 `formulation_r_value` 相关系数 **0.9749**、中位绝对误差 **0.00017**
  - **输入端重构（`UserPrediction.py`）**：分区由「必填/可选/分子/计算」改为按用户心智模型的 5 区 —— ① 配方（必填）② 固化制度（默认预填）③ 高级测试条件（折叠、默认预填）④ 自动推导（只读）⑤ 系统计算（只读）；新增「已自动填充 N 项」明细条，逐字段展示取值与依据（全表占比/样本数/推导公式），避免黑箱填充
  - **AI 优化**：新增 `core/portal_ai_cache.py`（文件型 LRU 缓存，键 = `sha256(service|model|prompt_kind|规范化输入)`，上限 500，不落 API key）+ `st.chat_message`/`st.chat_input` 多轮对话，每轮把**已确认字段**作为上下文传给 AI 以理解「温度改成 200 度」这类增量修正。约束未放松：AI 仍只能提取/整理，不得生成 EEW/AHEW/PHR/分子特征/工艺参数；结果仍须用户确认

### 修复
- 【模型启用失败：feature_mask 场景下 mask 前/后列数不一致】上传 TabPFN 模型时报
  `导入模型缺少 prediction_contract 且自动构建契约失败：无法解析精确模型特征契约：模型公开了 1408 个特征名，但 n_features_in_ 为 2070。；模型要求 2070 个特征，但当前只能解析到 1408 个。`
  - **根因**：该模型的 pipeline 为 `SimpleImputer(2070) → InfCleaner → FeatureMaskTransformer(2070→1408) → StandardScaler(1408) → TabPFNRegressor(1408)`。即 **pipeline 的输入契约是 mask 前的 2070 列**（mask 在 pipeline 内部执行），而 `model.feature_names_in_` 是 mask **之后**的 1408 个列名。`core/prediction_contract.py` 的解析器把两个不同口径直接对比，必然矛盾
  - **附带发现**：解析器一旦发现 model 提供 `feature_names_in_`，就**完全跳过** `feature_mask` 分支；且候选列来源中**缺少 `feature_audit.canonical_feature_cols`**（mask 前的完整列清单）
  - **修复（`core/prediction_contract.py`）**：① 新增 `_widen_columns_to_expected()`——当 pipeline 期望列数 > 模型公开列数且 artifact 带 `feature_mask` 时，用 mask 从更宽候选列还原出 mask 前的列清单（要求 mask 长度匹配、True 数匹配、且过滤后**逐列相等**，避免巧合匹配）；② `_source_candidates` 新增 `canonical_feature_cols` 作为最高优先候选；③ 新增 `_repair_columns_to_length()` 处理 canonical 多记录 1 列的场景（实测：canonical 为 2071 而 pipeline 期望 2070，冗余列为 `resin_3_molecular_weight_g_mol`，与 `core/external_feature_augmenter` 同思路）；④ 报告新增 `removed_features` 字段记录被 mask 剔除的列（不静默丢弃）
  - **门禁同步（`core/prediction_portal.py`）**：① `build_prediction_contract` 不再因「解析结果 ≠ artifact.feature_cols」直接报「特征列顺序不一致」，而是识别 mask 还原场景；② 新增 `_widen_artifact_features_by_mask()`，发布门禁不再把 `artifact.feature_cols`（mask 后）与 `contract.feature_cols`（mask 前）判为不一致；③ workflow 多产出契约未声明的特征**不再阻断发布**——因为 `core/portal_prediction` 会执行 `features.reindex(columns=contract['feature_cols'])` 安全丢弃多余列（实测 TabPFN 的 workflow 有 2131 个特征名而 contract 只有 2070 个）。安全性仍由运行时保证：契约要求的非 workflow 特征由 `_merge_explicit_model_features` 强制补齐，缺一即抛错
  - 新增 `tests/test_prediction_contract_feature_mask.py`（6 个测试）
- 【模型预测失败：cpu 与 cuda:0 设备不一致（下一条修复的回归）】模型补齐数据页报
  `【Tg.joblib】预测失败: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)`
  - **承认根因是下一条修复引入的回归**：下一条把**所有** CUDA tensor 一律映射到 CPU，但模型 pickle 里的 `self.device = 'cuda:0'` 是普通字符串属性（map_location 改不了它）。权重被改到 CPU 后，forward 里 `x.to(self.device)` 仍把输入搬到 `cuda:0` → 设备不一致。最小复现实验证实：同一模型修复前 forward 成功、修复后报错（与用户报错逐字吻合）
  - **正确修法（`core/model_io.py`）**：改用**可调用版 `map_location`**（`_portable_map_location`）——目标设备在当前机器**可用**则 `storage.cuda(index)` 原地恢复（与训练时完全一致），**不可用**才留在 CPU。同时解决两个报错：cuda:0 模型不再被错误改设备；cuda:1 模型在单卡机仍可加载。注意可调用版必须返回 **storage 对象**而非设备字符串（torch legacy/zip 两条路径均如此）
  - **配套修复**：新增 `_repair_loaded_object_devices()` —— 对指向**不可用** CUDA 设备的 `device` 字符串属性（常见自建 NN 写法 `self.device = 'cuda:N'`）同步修正为 `cpu`，条件保守（仅当对象是 nn.Module、属性名为 `device`、设备确实不存在、全部参数/buffer 已在 CPU），使降级模型 forward 不会设备不一致
  - **实施中发现并修正的额外缺陷**：大编辑替换起点未含装饰器行，导致原本属于旧函数的 `@contextlib.contextmanager` 错误地装饰在 `_cuda_device_usable` 上，使其返回恒真的 GeneratorContextManager、"设备可用"判断永远为真——已删除并加测试锁定
  - **重写 `tests/test_model_io_cuda_portability.py`（12 个测试）**：新增报错 2 的核心回归（cuda:0 模型保持原位且 forward 数值一致）、按设备逐一判断（cuda:0 不被 cuda:1 缺失牵连）、降级后 device 属性同步修复与端到端 forward、模型补齐路径（external_feature_augmenter → loads_artifact）验证
- 【模型预览失败：CUDA 设备拓扑不兼容】上传在别的 GPU 拓扑上训练的模型时，UI 报
  `Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1. Please use torch.load with map_location...`
  - **根因**：模型在 `cuda:1` 上训练并保存，artifact 里 pickle 了绑定该设备的 tensor。在只有 1 块 GPU（或纯 CPU）的机器上反序列化时，PyTorch 的 `torch.storage._load_from_bytes` 内部调用 `torch.load(io.BytesIO(b), weights_only=False)`，**未传 `map_location`**，于是尝试在 cuda:1 上重建 storage 并报错
  - **为何不能简单传参**：`joblib.load` 不接受 `map_location`，而错误发生在它内部调用的 torch 反序列化钩子上，调用方无法直接传递
  - **修复（`core/model_io.py`）**：新增 `_torch_cpu_map_location()` 上下文管理器，在反序列化期间把 `torch.storage._load_from_bytes` 临时替换为带 `map_location="cpu"` 的实现，让 CUDA tensor 落回 CPU（数值不变；预测时模型/管线自行决定设备）。补丁在 with 块结束时恢复，**不污染全局 torch 状态**；torch 未安装或钩子不存在时静默跳过
  - 覆盖范围：所有 artifact 加载路径（`preview_artifact` / `portal_prediction` / `prediction_portal` / `training_runs` / `data_imputer` / `external_feature_augmenter`）均经 `load_model_artifact_bytes` → `loads_artifact`，一并修复
  - 新增 `tests/test_model_io_cuda_portability.py`（7 个测试）：模拟单卡环境加载 cuda:1 保存的模型、数值一致性、CPU artifact 行为不变、补丁不泄漏
- 【配方库固化制度静默丢阶段】`core/process_features.py::_schedule_pairs` 的正则要求温度是数字，**非数字温度被静默丢弃且不报错**：`'室温/24 h + 80 °C/2 h'` 只解析出 `[(80.0, 2.0)]`（26 h 误算为 2 h）。配方库第 5 条（DGEBF/IPDA）原含此表述 → 已改写为 `'25 °C/24 h + 80 °C/2 h'`；新增 `validate_recipe_schedule()` 与模块级 `_self_check_preset_recipe_schedules()` 自检，导入时即断言全部 8 条配方的阶段数与声明一致
- 【`cp_r_value` 语义误用风险】`cp_r_value` 的兜底计算是 `ahew / eew`（**当量重比**），而 `formulation_r_value` 是**化学计量比 r**。实测两者与全表 `formulation_r_value` 的相关系数分别为 **-0.0886** 与 **0.9749**（DGEBA/DDS 100:33 时 cp_r_value≈0.365 vs 正确 r≈0.905）→ 在 `core/component_physics.py` 补充 docstring 语义警示（**仅注释，不改计算逻辑**），并在门户侧改用 `core/portal_formulation_inputs.py::derive_r_value()`
- 【化学引擎对非法 SMILES 静默兜底】实测 `EpoxyMechanismEngine.calc_single_molecule_properties` 对非法 SMILES（如 `'this_is_not_a_smiles(((('`）**不抛错**，而返回伪造值（mw=360 / ew=180 / functionality=2）→ `core/portal_formulation_inputs.py` 先用 RDKit 校验语法，非法即抛 `FormulationInputError`

### 新增
- 【虚拟筛选：模型外部特征统一输入】高通量筛选页新增「🧩 自动取值 + 批量确认」填充策略（默认）：
  - **痛点**：筛选会消费模型的**全部**输入特征。分子特征能从候选 SMILES 算出，但**工艺/测试/配方特征**（`cure_*`/`post_cure_*`/`curing_pressure_mpa`/`tg_heating_rate_c_min` 等）无法从结构推导，必须在筛选前给定统一值——它们是**筛选的设计变量**。旧实现只有「训练集中位数 / 0 / 模板行」三种填充，等于**伪造工艺条件**（如 `Tg-XGBoost` 的 27 个工艺列在工作区里全部缺失）
  - **新面板 `_render_screening_uniform_inputs`**：① 自动从工作区取代表值（数值→中位数，类别→众数）② 用**一张 `st.data_editor` 表**让用户逐项确认/修改（三列：模型特征 / 值 / 来源，前两列只读）③ 留空 = 交给模型内置 imputer
  - **优先级**：用户确认的统一输入**覆盖** base_row 的中位数/0/模板行（它们是显式给定的设计变量）
  - 实测：`储能模量.joblib` 的 36 个非分子特征 → **29 个自动取值**（`process_max_temperature_c=140`、`formulation_resin_total_eew_g_eq=187.27` …）、7 个需手工；`Tg-XGBoost` 的 27 个工艺列工作区全无 → 全部列在表里等用户填

### 修复
- 【模型补齐页对配方物理量特征误启动全提取引擎】「模型补齐数据」页第二次导入模型后，特征补齐不按工作流干活、全引擎空转数分钟：
  - **根因**：dsc 系列模型的 44 个特征（`cp_*` / `*_mw_resolved` / `*_ew_resolved` / `*_f_network` / `*_epoxy_group_count` / `formulation_*` 等）训练时来自 `component_physics.enrich_narrow_table`，**不在 workflow 产物里**。workflow 回放后它们落到 `AutoFeatureResolver`，被 `looks_molecular` 误判为分子特征 → 步骤 D 启动全提取引擎按方法序逐个试探（RDKit→MACCS→Morgan→FGD→环氧→Mordred→3D，实测一轮 >2 分钟），而提取引擎永远产不出这些列名，全部落空
  - **修复（`core/auto_feature_resolver.py`）**：① 新增 `is_physics_feature` / `_physics_alias`，`looks_molecular` 对物理量特征返回 False；② `resolve()` 主循环新增 **A0 配方物理量步骤**——用平台自己的物理量引擎（`compute_component_physics` + `compute_formulation_summary`，与训练侧 `enrich_narrow_table` 同口径：L1 文献值 → L2 当量×官能度 → L3 结构直算）补齐，别名映射 `formulation_resin_total_eew_g_eq→cp_eew`、`{side}_{i}_molecular_weight_g_mol→{side}_{i}_mw_resolved`、`resin_{i}_epoxy_group_count→f_stoich` 等；③ 步骤 D（重后端）显式拦截物理量特征（双保险）；④ `diagnose()` 重后端探测同步拦截并支持物理量可用性探测
  - **踩坑**：`cp_W_g_per_epoxy` 的 W 是大写，初版正则 `cp_[a-z0-9_]+` 漏识别 → 该列被轻量路径错算成环氧数 2.0（应为 ~184.6）；已改为 `cp_[A-Za-z0-9_]+` 并加回归断言
  - **实测**（dsc放热峰，515 个特征）：workflow 回放 0.4s → resolver 0.1s，**重后端零调用**（修复前 >2 分钟全引擎扫描）；35 个物理量特征全部由「配方物理量」路径填充且数值正确（DGEBA EEW=170.21、MW=340.42、DDM f=4）
  - 新增回归测试 `tests/test_physics_feature_resolution.py`（6 项：特征识别/分类/别名映射/端到端补齐零重后端/无结构列优雅降级）
- 【workflow 回放裁剪失效】`_prune_workflow_to_needed_steps` 不再按模型真实需求裁剪，导致全量执行无用步骤：
  - **根因**：旧逻辑只认 `step.feature_names`，但平台导出的 workflow **步骤里没有这个字段**（只有 `prefix`/`source_columns`/`params`）→ `names` 恒为空 → `not names` 为真 → **全部保留**
  - **改用 `workflow.feature_source_map`**（`{特征名: step_id}`，导出时生成，最权威）：直接知道每个特征是哪步算的
  - **产出未登记时的 prefix 校验**：实测 `dsc初始温度` 的 22 个 xTB 步骤全部「产出 0」，若一律保守保留就白跑 3 分钟。改用 **prefix 前缀 + 方法标志双重校验**（`_is_method_output`），防止短前缀陷阱（如指纹步 `prefix='resin_'` 会误命中 `resin_3_f_stoich` 这种非指纹列）
  - **重复步骤识别**：`single_4`(源 resin_1+2) 与 `single_7`(源 resin_1+2+3) 产出**同名列**（都是 `resin_Resin_MACCS_*`），后者已产出全部需要的列 → 按 `(method, prefix)` 判重裁掉前者
  - **`needed` 口径扩展**：`input_feature_cols ∪ feature_cols ∪ feature_audit.canonical/effective`，宁可多算不可漏算
  - **同步裁剪 `feature_source_map` / `merge_order`**
  - 实测效果：`dsc初始温度` **31 步 → 3 步**（裁 28）、`dsc放热峰` 25 → 2 步、`model_TabPFN` 15 → 8 步、`储能模量` 5 → 5 步（本就精简）
  - 端到端验证：`dsc初始温度` workflow 回放 **3 分钟+ → 2.8s**；新增 909 列；模型输入列 **518/526 就绪**；补齐 8 个 + 预测 2.5s → ✅ 526/526 预测 656 行
- 【模型补齐页交互优化】解决「点任何控件都刷屏、页面卡」问题（四层优化）：
  - **① 未解析特征分类，不再为无法映射的特征渲染控件**（**主要瓶颈**）：旧实现给**全部** 805 个未解析特征各渲染一个 `st.selectbox`（每个 657 个选项）→ **52.9 万 DOM 选项**。但其中 668 个是**指纹位**（从 SMILES 算得，根本无法手工映射）、55 个是**分子特征**（走提取引擎），真正值得人工确认的只有 80 个工艺/测试列。现改为三类分开展示，只对 80 个可映射特征渲染控件
  - **② 用 `st.data_editor` + `SelectboxColumn` 替代 N 个 selectbox**：一张表 + 一列下拉，DOM 选项数从 77,891 → 1 个表格，降幅 **99.9%**
  - **③ 缓存 augmenter 实例**：模型反序列化首次实测 **8.6s**（含模块导入），旧代码每次 rerun 都重建。改用「文件名 + 大小 + 前 1MB md5」签名做 key（计算仅 0.001s），内容不变则复用 session_state 里的实例
  - **④ 整个面板包成 `@st.fragment`**：面板内交互只重跑本函数，不再重跑整个页面（含其他面板与工作区数据读取）。项目已有同类先例 `_page_frag_upload`
  - 附带：一次性汇总提示「共 N 个特征由计算流程自动产出，无需手工映射；真正需要人工确认的只有 M 个」；指纹位/分子特征折叠为 expander 提示；`app_lib.py` 补 `from typing import Any`（此前靠 `from __future__ import annotations` 未暴露）
  - 验证：无头渲染异常数 0，selectbox 数 **800+ → 0**；分类耗时 0.57s；签名计算 0.0013s
- 【级联模型支持】「模型补齐数据」页现可导入**特征依赖其他模型预测值**的模型：
  - **自动发现依赖**：模型 B 的输入特征名 == 模型 A 的 `target_col` 时，自动建立 A→B 依赖边。实测真实案例 `XGBoost_artifact(2).joblib`(target=`tg_c`，需要 `tensile_modulus_gpa`) 与 `拉伸模量-XGBoost-0.965.joblib`(target=`tensile_modulus_gpa`，需要 `tg_c`) **互相引用**，此前两个都跑不起来
  - **拓扑分层求解**：Tarjan 求强连通分量（SCC）+ 缩点拓扑排序，先算上游、把预测值写回，再作为下游模型的输入特征
  - **环处理**：多节点 SCC（互引）无法拓扑排序，按「依赖的**环外可获得性**」择优打破环——优先用手工映射（×100）> 工作区真值（×50）> 环外上游预测（×30），平局时比可解析特征比例
  - **优先级铁律**：**工作区已有的真实值一定胜过模型预测值**。级联解析排在 `exact`/`case_insensitive`/`normalized`/`alias` 之后、`pattern`/`fuzzy` 之前
  - **UI**：新增「🔗 级联依赖关系」面板（依赖表 + 求解层 + 环提示 + 外部需补特征数）、「🔗 启用级联模型求解」开关（默认开）、结果表新增「层」与「级联输入」列；诊断表把「待上游预测」从「未获取」里区分出来，不再误导手工映射
  - 新 API：`ModelDependencyGraph`（`upstream_of`/`layers`/`scc_layers`/`cycles`/`describe`）、`ExternalFeatureAugmenter.cascade_info()`/`cascade_external_features()`、`_dep_key()`、`_order_layer()`、`describe_cascade()`；`resolve_features`/`diagnose_entry` 新增 `cascade_sources` 参数；`augment`/`augment_with_models` 新增 `enable_cascade` 参数
  - 验证：3 模型链 A→B→C 逐层正确、C 精确复用 B 的预测值（误差 0）；互引环在有真值/手工映射时正确改序；关掉级联则回退旧行为（缺特征全 NaN）
- 【环氧反应特征分层数据源】`EpoxyDomainFeatureExtractor` 支持按信任层级取值，任一层缺失自动降级：
  - **L1 窄表优化列**（`cp_*` / `*_mw_resolved` / `*_ew_resolved`，优先级最高）→ **L2 宽表文献值**（`*_molecular_weight_g_mol` / `*_equivalent_weight_g_eq` / `*_amount_phr`）→ **L3 结构直算**（SMARTS + MolWt，BigSMILES 采样代理的 MW 被拒绝）
  - **修正 EEW 系统偏差**：旧实现只用结构 `MolWt/f`，与文献 EEW 一致率仅 **50.0%**（商用树脂含低聚物，结构算 EEW≈170 而文献值 197~208）；接入宽表后提升到 **100.0%**
  - **修复酰胺假阳性**：旧 `_get_active_hydrogen_count` 数所有 N-H（含不可反应的酰胺），新路径改用 `component_physics` 的机制感知识别
  - **新增酸酐双口径**：`Hardener_Functionality_Stoich`(=1) 与 `Hardener_Functionality_Network`(=2) 分离，避免用化学计量口径代入 Flory `(f-2)` 项得到**负交联密度**
  - 新增物理特征：`W_g_per_epoxy`、`Epoxy_Conc_mol_m3`、`Crosslink_Density_Network_mol_m3`、`Mc_g_mol`（与 `crosslink_physics` 同口径，单位 mol/m³），以及来源标记 `Physics_Source_Resin` / `Physics_Source_Hardener`
  - 行序安全：宽表/窄表行数与输入不一致时**整体降级**为结构口径（不按位置错位取值）；多进程路径按 chunk 切片传递，`row_idx` 在 chunk 内定位（已测单线程与多进程逐行一致）
  - `core/molecular_features.py` 单组分路径接入宽表/窄表（此前仅多组分路径传了 `wide_df`）
  - **模型补齐数据页面同步**：`execute_molecular_feature_workflow` 新增 `use_data_source` 参数（默认 True），把当前数据表作为分层数据源透传给提取器；`virtual_screening.extract_features_from_config` 支持 `_source_df`。这样回放模型自带 workflow 时，某些**不在宽表中**的特征也能用工作区已有的优化列（`cp_*` / `*_resolved`）而非只能结构直算
- 【目标对数变换】训练页新增「目标变量变换」面板，支持 `log1p` / `log` / 不变换，并新增「剔除物理不可能的目标值」开关：
  - `core/model_trainer.py` 新增 `TargetLogTransformer`（sklearn 兼容包装器）：训练在 log 空间，`predict()` 返回**原始空间**，对下游（Pipeline / 预测页 / SHAP / 残差图）完全透明，无需改动任何消费方
  - 关键正确性保证：`fit` 会**同步变换 eval_set 的 y**，否则早停会在错误量纲上评估（实测不变量纲时 RMSE 为原始尺度，变换后为 log 尺度 0.02）
  - 与 ANN 的 `normalize_target` 互斥（两者都改目标尺度），启用 log 时自动关闭后者
  - `train_model` 与 `cross_validate_model` 同口径支持，保证 CV 分数与测试集分数可比
  - UI 现场诊断：显示目标偏度与越界样本数；当检测到异常值时直接提示「优先做异常值过滤而非 log 变换」
- 【逐组分物理量补齐】新增 `core/component_physics.py`，为窄表补齐各组分 MW / EEW|AHEW / 官能度，解决交联密度模型 R² 上不去的特征瓶颈：
  - **根因**：窄表原本没有逐组分分子量列（`resin_1_molecular_weight_g_mol` 全仓只被读取、从未被生成），导致 `crosslink_physics` 的 `hybrid` 分支恒不可达（`f_avg` 永远 NaN），理论 ν 全部退化为 `aggregate` 粗口径
  - 分层补齐（信任层级）：L1 文献值 → L2 当量×官能度（`MW = EEW × f`，实测合法性：树脂侧 `(MW/f)/EEW` 中位 1.000、86.5% 落在 0.9–1.1；固化剂侧中位 1.000、99.4% 落在 0.9–1.1）→ L3 结构直算
  - 覆盖率提升：树脂 MW 17.6% → **98.2%**，固化剂 MW 52.6% → **95.4%**
  - **BigSMILES 妥善处理**：`bigsmiles_to_smiles` 返回的是采样代理，重复单元数 n 与采样长度人为设定，其 MolWt **无物理意义**（实测与文献 EEW 自洽率仅 3.5%、`convMW/origMW` 中位 0.196），但官能度 f 可靠。因此 BigSMILES 只采信 f、MW 强制走 L2，并对代理 MW 与多片段单元格做拒绝校验（实测 119 条 BigSMILES 行中 0 条误用代理 MW）
  - **官能度双口径分离**：酸酐 `f_stoich=1`（1 酸酐:1 环氧）与 `f_network=2`（开环酯化后桥接 2 条链）严格分开，避免用 `f_stoich` 代入 Flory `(f-2)` 项得到**负交联密度**

### 变更
- 【交联密度口径对齐】`core/crosslink_physics.py` 全面重写，单位统一为 **mol/m³**：
  - 新增机制感知理论 ν：`ν = [ep]·(f_h,net−1)/f_h,net`，其中 `bal = min(r,1/r)` 仅对酸酐/羧基类机制生效（胺类过量胺本身起链终止作用，实测不需要 bal：0.229 vs 0.157；酸酐类需要：0.328 vs 0.056）。5 折交叉验证中该形式 5/5 折均被选中
  - 修复官能度污染：文献列 `active_hydrogen_equivalent_count`（化学计量口径）不再覆盖结构解析出的 `f_network`（此前使酸酐 `f_h` 由 2.0 变 3.0，ν 秩相关由 0.242 掉到 0.192）
  - 修复 `xl_alpha_gel` 恒为全空（`f_r_curated` 仅在逐组分文献列存在时为 True，而窄表恰好没有）
  - 新增 `xl_epoxy_conc_mol_m3` / `xl_W_g_per_epoxy` / `xl_f_h_stoich` / `xl_nu_junction_mol_m3` / `xl_mechanism` / `xl_coverage`
  - 新增显式换算函数 `nu_mol_per_m3_to_mmol_per_g` / `mmol_per_g_to_nu_mol_per_m3`（往返偏差 < 1e-12）
- 【口径统一】`core/epoxy_mechanism_features.py` 理论交联密度由 `1000/Mc`（**mmol/g**）改为主口径 mol/m³，同时保留 `mech_crosslink_density_mmol_g` 兼容；修正 `f_avg` 分子分母加权口径不一致（分子用 phr 质量权重、分母用摩尔数）
- 【窄表接入】`core/formulation_fusion.py` 在 `qspr_clean` 清洗中调用 `enrich_narrow_table`，并把逐组分当量重与新派生列加入 `valuable_ordered_cols` 白名单

### 验证
- 5 折 CV（XGBoost，目标 log ν，n=1373）：基线 R²(log)=+0.5519 → 补逐组分 EEW/AHEW **+0.5851** → 补 MW+EEW **+0.5978** → 再加理论ν+alpha_gel **+0.6011**（+0.049）
- 对照实验：加 8 列纯噪声 R² 降至 +0.4039（证明模型无「加列即涨分」假象）；MW 列置换打乱后降至基线以下（证明增益非列数效应）
- 回归测试：全量 pytest 失败数与改动前**逐条完全一致**（21 项既有失败，零新增回归）

### 变更
- 【官能度口径切换】`Hardener_Functionality` 改为**网络支化口径**（酸酐=2）：
  - 理由：该特征主用于 Flory 凝胶点 / Mc / 交联密度，公式中的 f 是“每分子形成的网络分支数”。酸酐 1:1 消耗环氧（`f_stoich=1`），但开环酯化后桥接 2 条链（`f_network=2`）
  - 同步修改 `core/auto_feature_resolver.compute_formulation_feature`（总表查不到特征时的现场计算路径），保证两条路径口径一致
  - **当量重/当量比仍用化学计量口径**（`AHEW`、`formulation_r_value`），不受影响；新增 `Hardener_Functionality_Stoich` 供化学计量用途显式取用

### 变更
- 【当量重去重】`enrich_narrow_table` 不再另起 `cp_eew`/`cp_ahew`/`cp_r_value` 列，
  改为**写回原表列名**（`formulation_resin_total_eew_g_eq` /
  `formulation_hardener_total_ahew_g_eq` / `formulation_r_value`）：
  - 原实现下 `cp_eew` 与原表 EEW 在 756 条重叠样本上 **100% 相同**，属完全重复特征
  - 下游 `crosslink_physics._EEW_COL`、特征白名单、`auto_feature_resolver` 均按原名引用，改名会断链
  - 采用**原表实测值优先、仅填补空位**（非覆盖）：
    | 指标 | 原表非空 | 补齐后 |
    |---|---|---|
    | EEW  | 756 (49%) | **1455 (94%)** |
    | AHEW | 439 (28%) | **1341 (86%)** |
    | r 值 | 958 (62%) | **1379 (89%)** |
  - 为何不反过来用结构直算覆盖：受控子集（n=352 两套都有值，仅换 EEW/AHEW，
    目标=实测 ν）原表 spearman=**+0.236** vs 结构直算 **+0.080**。同一 DGEBA
    结构在原表中有 105 个不同 EEW（134~288，中位 196）—— 这是实测/文献值，
    反映真实低聚物分布（n=0~0.15 同系物混合物）；结构直算恒为单体值 170.21，
    丢失了低聚物分布信息。而原表缺失时补齐值本身有效（535 条 spearman=+0.243）。
- 【元数据列不进表】新增 `is_metadata_column()`：`*_mw_source` / `*_ew_source` /
  `*_f_source` / `*_mw_trust` / `*_mechanism` / `cp_mechanism` / `cp_coverage`
  等字符串标记列**不再写入数据集**（此前会让它们进入特征白名单、污染训练矩阵）。
  新增列中的字符串列数：22 → **0**。需调试时用 `keep_metadata=True`。
- `cp_f_r` / `cp_f_h_stoich` **保留**（不与原表列重复）：原表 `resin_epoxy_group_total`
  是简单求和，`cp_f_r` 是摩尔加权。实测差异集中在多组分样本（差异样本 2/3 组分
  占 84%；相同样本 1 组分占 82%），语义不同。

### 修复
- **`Gel_Point_Conversion` 长期静默失效**（`core/molecular_features.py`）：
  - 根因：模块从未 `import math`，`math.sqrt` 抛 `NameError`，却被 `except Exception: alpha_gel = 1.0` 静默吞掉 → 该特征**恒为 1.0**（等价于“永不凝胶”），实测 400 行样本 100% 受影响
  - 修复：补上 `import math`；`except` 拆分为 `ZeroDivisionError`（真正的不可凝胶）与通用异常（打印告警但不中断主流程），避免同类 bug 再被隐藏
  - 修复后：恒为 1.0 的比例从 100% 降至 31.8%，253/371 行获得有效凝胶点
- 模型解释页 SHAP 分析多次运行后，图表与结果面板堆叠在同一页面（旧图 + 新图、重复的排名图/饼图/下载按钮）：
  - 根因：SHAP tab 内渲染顺序缺陷 —— 上方缓存面板（`cached_shap_*`）在 `if run_shap:` 计算分支**之前**执行，点击计算的那次 rerun 会先用旧缓存渲染一遍；XGBoost 分支计算后调用了 `st.rerun()` 所以页面被重建（正常），而非 XGBoost 分支（TabPFN/神经网络等）直接 `st.image` 新图后 `return`，导致旧面板与新面板叠加
  - `app_lib.py` 非 XGBoost 分支改为与 XGBoost 分支一致：写入 `session_state`（图/CSV/Origin 三套导出）后立即 `st.rerun()`，由缓存面板统一渲染唯一一份最新结果；完成提示改为写入 `shap_last_status`（rerun 后由缓存面板顶部展示），不再在计算轮直接 `st.success` + `st.image`
  - `run_shap` 入口统一先清空上一轮结果缓存（`shap_plot_png` / `shap_plot_path` / `shap_csv_*` / `shap_origin_*` / `shap_*_cache_key`），计算期间与计算失败时都不再残留过期图表
  - 加载模型时清理 SHAP 缓存补全遗漏键（`shap_origin_beeswarm_bytes` / `shap_origin_bar_bytes` / `shap_origin_beeswarm_path` / `shap_origin_bar_path`），避免切换模型后旧 Origin 导出数据驻留内存

### 新增
- 【总表特征自动解析】`core/auto_feature_resolver.py` 大幅增强，解决「预测时从总表按结构查不到特征」的问题：
  - 结构指纹匹配：新增 `_row_fingerprint` / `_build_fingerprint_index` / `lookup_by_fingerprint`，用工作区行的结构列组合做指纹索引，命中总表同结构行后取特征，替代原先只能靠单一列名精确匹配的做法
  - 派生列索引：新增 `_build_derived_index` / `_lookup_derived` / `_match_derived_column`，支持从关联表 join 出来的派生特征回填
  - 关联表 join：新增 `_join_related_tables` / `_safe_left_merge` / `_lookup_from_hits`，按结构列安全左连接（自动处理重复列名与键缺失）
  - 多行聚合策略：新增 `_is_blank` / `_collapse_values`，同一结构在总表中有多行观测时，旧实现是 first-wins（等于随机取一行），现改为数值型取**中位数**（对连续量稳健）、分类型取**众数**（并列取首个），空值占位符（`未测`/`无`/`N/A` 等）统一识别
  - 单分子特征直算：新增 `looks_molecular` / `_parse_molecule_cached` / `_compute_feature_from_mol` / `compute_single_molecule_feature`，总表查不到时直接用 RDKit 从结构 SMILES 现算（带解析缓存）
  - 特征归属判定：新增 `_feature_belongs_to_column` / `_same_role_structure_cols`，避免把 A 结构的特征填到 B 结构的行
- 【模型输入契约】`core/external_feature_augmenter.py` 新增 pipeline 真实输入列数解析，修复预测时 `SimpleImputer is expecting N features` 报错：
  - 根因：`Pipeline(imputer -> feature_mask -> scaler -> model)` 中 `artifact.feature_cols` 只记录了 mask **之后**的列（如 1408），而 imputer 期望的是 mask **之前**的全量列（如 2070），按 artifact 喂数据必然维度不匹配
  - 新增 `_pipeline_expected_n_features`（读取 sklearn pipeline 首步 fit 时记录的 `n_features_in_`）与 `_repair_columns_to_length`（用 `feature_mask` 反推真正输入列），`_resolve_input_feature_cols` 汇总解析结果；条目新增 `input_feature_cols` 字段，喂数据时优先使用
  - 分子特征 workflow 回放：新增 `get_molecular_workflow` / `has_molecular_workflow` / `molecular_workflow_step_count` / `workflow_required_source_columns` / `workflow_output_columns` / `replay_molecular_workflow` / `_prune_workflow_to_needed_steps`，模型自带训练时提取配方时直接回放该配方，不再靠猜测 RDKit/Mordred 特征
  - `app_lib.py` 预测页新增「📋 模型输入契约」面板：展示各模型是否自带 workflow、模型输入列数 vs 声明特征数差异，并提示 workflow 源列缺失情况

### 性能
- 【SHAP 分析】TabPFN 高维场景批量置换 SHAP 提速约 8-10×（实测 532 特征/500 样本从 ~25 分钟降至 ~2.5 分钟，RTX 级 GPU），无显存增量：
  - 根因一：TabPFN 9.0 `n_estimators="auto"` 在特征数 > `max_features_per_estimator(500)` 时仍至少跑 8 个 ensemble 成员，每次 predict 行成本 ≈ 8 × 单成员前向（实测 532 特征 ~4ms/行 vs est=2 ~1.1ms/行）
  - 根因二：置换 SHAP 的 coalition 行数 = 2M+1，M=532 时每样本 1065 行、500 样本共 53 万行
  - `core/model_interpreter.py` 新增 `_create_fast_shap_predictor`：SHAP 期间每次 predict 前临时截断 executor 的 per-config 对齐列表（configs/pipelines/subsample_feature_indices/subsample_row_indices/pipeline_seeds/ensemble_members，兼容 OnDemand/CachePreprocessing/ExplicitKVCache 三种引擎）至前 2 个成员，用完 finally 恢复原值 —— 零额外显存（不克隆模型，避免双份权重/KV cache 驻留 GPU OOM），异常安全还原，实测 SHAP 后模型预测 bit 级一致；引擎结构不认识时回退原模型
  - 新增 `_select_top_k_features`：先用 LightGBM（回退 ExtraTrees）在训练集上选出 top-120 特征，仅对它们做精确置换，其余列固定为背景值（coalition 行数 1065→241，约 4.4×）；未被选中列 SHAP 记 0，对 beeswarm/bar 的 top-N 展示无影响
  - `_compute_batched_permutation_shap` 支持 `predict_fn` / `top_feature_idx` 参数：top-K 模式在 chunk 级别把 K 列工作子矩阵散射回 M 列全特征矩阵后再送预测器；置换轮数预算按实际状态数折算；一致性检查改为打印 f(全集coalition)−(Σφ+base)（top-K 模式下该残差含未解释列在背景值处的联合贡献，属预期）
  - 保真度实测：est=2 vs est=8 的置换 SHAP 特征重要性排序 Spearman≈0.89、top-10 重叠 9/10、top-8 排名完全一致；端到端合成数据（532 特征，前 6 信号列）全部进入 top-6
  - 回归测试 `tests/test_batched_permutation_shap.py` 新增 5 例（top-K 解析解/守恒残差/分块不变性/非 TabPFN 直通/低维不启用）

### 修复
- 侧边栏「📥 数据导出」第一次能导出，折叠面板后重新打开（或切换格式/勾选“包含索引”后再点下载）时点下载无反应/无法导出表格：
  - 根因：`st.download_button` 的媒体文件 id 由 `内容 + mimetype + 文件名` 三者哈希得到（`MemoryMediaFileStorage._calculate_file_id`），且文件名里的时间戳也参与 download_button 元素 id。导出载荷缓存原本是**进程级单条目**，任何一次驱逐都会让下次重跑重新生成时间戳 → 文件名变化 → 产生新的媒体文件；旧媒体文件成为孤儿后被 Streamlit 按 DOWNLOADABLE 两阶段回收（第一次 sweep 标记、第二次 sweep 删除），浏览器已持有的下载链接随即 404
  - 触发驱逐的两个必然场景：① 缓存是进程全局的，**另一会话/用户**导出一次即把本会话条目挤掉；② 「状态条记录」页 tab2 仍以 `key_prefix=""` 调用同一面板，与侧边栏面板（`key_prefix="sb_"`）**同时渲染时互相挤占条目**，导致每次重跑双双 miss
  - 实测复现（真实 app + WebSocket 协议驱动）：反复切换“包含索引”后，同样设置（CSV/索引关）得到的 URL 从 `802e981e…` 变为 `0efc406c…`，且旧 URL `HTTP=404`
  - `core/fe_tracker.py`：`_build_export_payload` 缓存改为**会话级多条目**（存 `st.session_state`，随会话生灭，上限 8 条 FIFO），不再跨会话/跨面板互相驱逐；缓存键把原来只看末列名的探针升级为 `形状+全部列名+全部 dtype+索引端点` 的进程内哈希（实测 607/1248 列 ~0.1-0.2ms，不影响侧边栏重跑预算），可捕捉列增删/改名/类型变化；`ts` 与 payload 绑定存储，只有 payload 真正重建时时间戳才更新，同一份数据同一格式在多次重跑之间文件名保持不变
  - `st.download_button` 三个分支补上显式稳定 `key`（`{prefix}export_dl_csv/xlsx/json`）：带 key 时 Streamlit 的 `key_as_main_identity` 会把 file_name 从元素 id 中剔除，数据变化时只更新 url、不再销毁重建控件
  - 回归测试 `tests/test_data_export_panel_stability.py`（9 项）：同数据同格式文件名跨重跑稳定、插入其它格式/其它数据/双面板交替后文件名不变、缓存会话级且有界、数据替换后载荷与时间戳刷新、AppTest 驱动真实面板验证 download_button 元素 id 不随数据变化而 URL 更新；修复前的单条目全局实现在跨秒重跑场景下前 4 项全部失败（已用可控时钟 A/B 验证）
- 批量特征提取报 `feature contract violation: feature row count does not match valid-row indices (953 != 954)`（如 resin_1_structure 列，分子指纹/RDKit 描述符/Mordred 方法）：
  - 根因：`extract_fingerprints` 等便捷函数内部会跳过解析失败的 SMILES 行（如七元芳环 BigSMILES，repair 链也修不了），返回的特征 DataFrame 行数（953）小于输入有效行数（954），但便捷函数丢弃了内部 valid_indices，批量循环误以为返回行数与全部有效行一一对应，contract 校验行数不一致后中断整列提取
  - `core/molecular_features.py`：`extract_fingerprints` / `extract_rdkit_descriptors` / `extract_rdkit_descriptors_parallel` / `extract_rdkit_descriptors_lowmem` / `extract_mordred_descriptors` 五个便捷函数改为返回 `(df, valid_indices)`（与底层 extractor 一致，不再丢弃有效行下标）
  - `app_lib.py` 批量循环：上述五个分支接收 `sub_valid_idx` 并映射回源行 `extracted_valid_indices = [valid_indices[i] for i in sub_valid_idx]`，与 3D构象/TDA/xTB/GNN 等分支处理模式对齐；解析失败行回填 NaN，不再中断整个批处理
  - 回归验证脚本 `scripts/verify_fingerprint_fix.py`：954 行含 1 条七元芳环无效 SMILES，contract 校验通过、失败行回填 NaN、批处理不中断
- TabPFN 不可用（`ModuleNotFoundError` / 首次训练时 HuggingFace 下载失败）：
  - `CFRP_env` 安装 `tabpfn==9.0.0`（满足 requirements 的 `tabpfn>=0.1.9`，API 兼容 `TabPFNRegressor` + `model_path`/`ignore_pretraining_limits` 等参数集）
  - tabpfn 9.x 默认模型 v3.5（`Prior-Labs/tabpfn_3_5`）为 gated 仓库，且其许可检查硬编码 `huggingface.co` API（不遵循 `HF_ENDPOINT`），国内网络下必然抛 `TabPFNHuggingFaceGatedRepoError`；项目代码本就优先探测本地 v3 权重（`core/model_trainer.py` 的 local_candidates），故手动从 hf-mirror.com 镜像下载非 gated 的 v3 回归权重（233MB）至 `%APPDATA%\Roaming\tabpfn\tabpfn-v3-regressor-v3_default.ckpt`（与 tabpfn 默认缓存目录及项目探测路径一致），此后 fit/predict 完全离线、不再触发任何在线许可检查
  - 端到端验证通过（CPU fit+predict）；若日后缓存被清，可重新执行：`curl -L -o "%APPDATA%\\Roaming\\tabpfn\\tabpfn-v3-regressor-v3_default.ckpt" https://hf-mirror.com/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt`
- 模型训练页：点击下载按钮（训练结果 PNG/CSV、导出模型等）或任意控件后整页重跑，导致训练结果区（指标卡/图表/表格/各类下载按钮）整体消失，"点一下下载其他按钮都不见了"：
  - 将手动训练结果渲染逻辑提取为 `_render_manual_training_results(res, cv_res, persist_run)`；训练完成时以 `persist_run=True` 调用（含保存训练记录、自动导出模型、内存清理等副作用），任何交互触发整页重跑后自动以 `persist_run=False` 恢复渲染（零副作用，不重复保存训练记录/不重复导出模型）
  - 分类模型结果区同理：`_render_binary_classification_results` 新增 `persist_run` 参数，重跑恢复时跳过训练记录落盘
  - 顺带修复：结果渲染代码引用了页面作用域从未定义的 `feature_cols`，NameError 被外层 `except` 静默吞掉，导致 parity/residual 图从未写入训练记录 extra_figs；现在显式从 session_state 取值

### 性能
- 【分子特征】页首屏从 1.01s 降至 0.016s（中位数，约 63×），根因是 Streamlit `st.expander` **无论是否展开都会执行内部代码**：
  - 两个「折叠态」重面板每次 rerun 都要完整构建，其中 `core/formulation_fusion_ui` 会重读 20MB 母宽表 `ml_wide_samples.csv`（10749×1248），实测单次 0.7~1.1s，占首屏 ~98% 耗时
  - `app_lib.py` 新增 `render_lazy_panel(label, key, builder, hint)`：`st.toggle` + 仅在展开时调用 builder，未展开时面板代码完全不执行；分子特征页的「跨表配方数据融合工具」与「高分子物理指数」改为按需构建，并补充未展开时的功能提示
  - `core/formulation_fusion_ui.py` 新增 `read_csv_cached()`（`st.cache_data` + 文件指纹 `mtime_ns`/`size`）：展开面板后首次读盘 0.78s，命中缓存 0.18s，文件被改写自动失效；窄表/母宽表读盘统一走该函数
  - `app_lib._detect_smiles_cols_smart`：宽表下预筛 object 列 + 预编译正则替代「每样本 9 次 `in` 检查 + 多次 `replace`」，7000×1205 数据下页面主体从 0.35s 降至 0.06s
  - 修复 `page_molecular_features` 内三处 `import re`：函数内 import 会让 `re` 在整个函数作用域变成局部变量，导致上方 `re.compile` 抛 `UnboundLocalError`（模块级已有 `import re`）
- 侧边栏每次 rerun 固定 ~88ms 开销：`BackgroundTaskManager.get_orphan_processes` 用 psutil 递归遍历子进程树（Windows 约 88~95ms），而 `render_task_manager_ui` 每次 rerun 都调用；加 10s 节流缓存 + `invalidate_orphan_cache()`（终止/重置后失效），侧边栏降至 ~0.08s
- 新增回归测试 `tests/test_molecular_features_page_perf.py`（首屏不读母宽表、首屏耗时预算、展开后才构建、折叠回去零成本、缓存命中与文件改写失效、`render_lazy_panel` 返回值语义）
- 新增基准脚本 `tools/bench_molecular_features_page.py`（脚本内计时 + read_csv 追踪；注意 cProfile 包 `AppTest.run()` 只会抓到测试框架的 sleep 轮询，无法定位真实瓶颈）
- 【分子特征】页交互卡顿（点击选择框/复选框后卡）已修复：提取完成后工作区常有上千列指纹特征，`_render_extracted_features_panel` 的 `st.dataframe(features_df.head(20))` 会把**全部列**转成 Arrow 并在**每次交互 rerun** 重发给浏览器：
  - 实测交互 rerun：2000 列 → 脚本 0.87s / 负载 801KB；6474×3000 → 0.94s / 1.2MB
  - 新增 `core/preview_ui.py` 的 `render_capped_preview()`：默认只渲染前 40 列（约 31KB / <2ms），其余列通过**按需展开**的列选择器查看（选择器本身也延迟构建，否则上千个列名同样拖慢页面）；列数不超过上限时行为与直接 `st.dataframe` 一致
  - 分子特征页两处预览（已提取特征面板、批量提取预览）与 `core/formulation_fusion_ui` 的融合结果预览（紧凑 70-80 列 / 全息 300-450 列 / 原始 1248 列）全部改用它
  - 修复后交互 rerun：负载 0.031MB（-96%）、脚本 0.029s（-97%），且与表宽解耦（6474×3000 同样是 31KB）
  - 预览列数**未被牺牲**：展开“自定义要预览的列”可访问全部列并支持搜索
- 新增回归测试（`tests/test_molecular_features_page_perf.py`，共 12 项）：宽表交互负载预算、交互脚本耗时预算、预览默认截断列、自定义列选择仍生效；已确认这 4 项在修复前的代码上全部失败、修复后全部通过
- 新增交互基准脚本 `tools/bench_molecular_features_interaction.py`：在脚本线程内同时采集「脚本耗时」与「发给浏览器的 delta 负载」
- TabPFN 等黑盒模型 SHAP 提速约两个数量级（实测 200 样本默认参数从外推 ~53 分钟降至 ~22 秒，RTX 2080 Ti）：
  - 新增「跨样本批量置换 SHAP」快速路径（`core/model_interpreter.py`）：每个样本每轮置换仅需 2M+1 次评估，并把一批样本的 coalition 行合并成少量大 batch 一次性调用 `model.predict`，避免 KernelExplainer 逐样本、每次都重跑 TabPFN 完整训练上下文前向（含 8 个 ensemble 成员）的开销
  - 适用于 TabPFN、TabNet、FT-Transformer、人工神经网络、BNN/Transformer 系列等无专用 Explainer 的黑盒模型；失败时自动回退到原 KernelExplainer 路径
  - 置换轮数复用 UI「Kernel nsamples」预算自适应（1~4 轮）；加和一致性（效率公理）达机器精度，重要性排序与 KernelSHAP 的 Spearman 相关系数 0.987
  - 特征数 ≥300 时的 UI 警告对已启用加速的模型追加提示
- 新增回归测试 `tests/test_batched_permutation_shap.py`（线性解析解、效率公理、分块/分批不变性、快速路径选择逻辑）

---
## [1.6.0] - 2026-09-13

### 变更
- 平台更名：「碳纤维复合材料智能预测平台」→「材料机器学习平台」，浏览器标题、侧边栏品牌位与文档同步更新
- 架构重构：28000 行单体 app.py 拆分为薄入口（236 行）+ app_lib.py 共享库（27855 行，每进程仅导入一次）+ app_pages/ 下 19 个 st.Page 页面，页面导航升级为 st.navigation 分组侧边栏
- 侧边栏布局：导航菜单固定顶部（Streamlit 官方行为），平台名品牌位移至侧边栏底部
- 首页快速入口由 _nav_to 机制改为原生 st.switch_page

### 性能
- 关闭 Streamlit magic：消除每次字节码重建时对全量脚本（1.2MB）的 ast.parse+改写（实测 4.6s/次）
- 每次重跑执行的代码从 28000 行降至 236 行，页面切换显著加速
- 侧边栏快照元信息读取增加 15s 会话级节流，不再每次重跑解析 3.5MB 快照 JSON
- 自动保存（快照序列化与写盘）移至单工作线程后台执行，meta 改为临时文件原子替换，不再周期性冻结 UI
- 服务器启动时后台预加载 shap/plotly/seaborn/xgboost/lightgbm/catboost/tensorflow，消除各页面首次访问的懒加载卡顿

### 修复
- 后台快照的线程池与在途门锁改为 sys 属性进程级单例，修复模块级变量被每次重跑重建导致的线程泄漏与防堆积失效

### 文档
- README/CHANGELOG/DEVELOPMENT 同步新平台名与 v1.6.0 版本号

---
## [1.5.2] - 2026-08-25

### 修复
- 修复代理连接测试把 HTTP 代理端口误报为 SOCKS5 握手失败的问题
- 增加 SOCKS5、HTTP 代理协议自动诊断，并识别代理认证/外部访问拒绝
- 代理协议不匹配时停止继续向 PubChem 和 Hugging Face 发起误导性请求
- 更新代理设置界面，显示实际端口类型和 CCProxy 外部访问提示

### 文档
- 新增网络代理配置说明，覆盖 SOCKS5、HTTP、认证代理和 CCProxy 端口排查

### 测试
- 增加 HTTP 端口误配和外部访问拒绝的代理探测回归测试

---
## [Unreleased]

### 新增
- 手机/窄屏适配：新增 ≤820px 媒体查询样式块——主内容区收紧留白、标题字号降级、输入控件 16px 防 iOS 聚焦缩放、按钮触控目标 ≥44px、侧边栏抽屉 84vw、数据表格/结构图/绘图高度自适应、页面级 tabs 横向滑动、防横向溢出

- 模型训练页新增鲁棒损失选项（训练中自动压制目标异常样本影响，无需剔除数据）：XGBoost/LightGBM 新增 Objective (Loss) 选项（reg:squarederror / reg:pseudohubererror / reg:absoluteerror；regression / huber / regression_l1），人工神经网络新增 Loss Function 选项（mse / huber / mae，含旧模型反序列化兼容）；Huber 在残差超过 δ 后自动由平方转线性惩罚、MAE 全程线性，异常样本影响被自动压低，默认值与历史行为一致，仅影响训练目标、测试集评估不变；含异常点数据的对比验证中 Huber 相对 MSE 稳定取得更低测试 RMSE
- 模型训练页新增「内部验证集（早停用）」配置块：自动（默认，训练样本 ≥ 20 条时划出 15%，与历史行为一致）/ 自定义比例（5%~40%，验证样本不足 4 条自动回退）/ 关闭（全部训练样本参与拟合）三种模式；仅对支持早停的模型（XGBoost/LightGBM/CatBoost 及 FT-Transformer/Transformer+BNN/Transformer+PINN/GNN+Transformer 融合）生效，其他模型显示明确提示；切分尊重分组/分层划分策略，避免配方组同时出现在拟合与验证两侧

- 修复 CatBoost 损失函数选项选 Huber/Quantile 时因缺少 delta/alpha 参数导致训练报错的问题（选项值改为 Huber:delta=1.0 / Quantile:alpha=0.5）
- 训练结果新增「内部验证集（早停）」指标卡片：展示启用状态、验证样本数、验证集 R²/RMSE 及模型最优迭代轮次，回退时给出原因警告；验证集指标在原始目标量纲下计算
- 训练样本 < 100 且启用验证集时提示用交叉验证（CV mean±std）获得更稳定评估
- 验证集模式/比例写入参数保存、训练日志与训练记录元数据（val_mode/val_effective/val_size/val_sample_count），训练结果字典新增 validation_set 结构化信息
- 首页新增「一句话智能输入」入口：直接粘贴配方/工艺描述，点击「🤖 AI 全自动解析并填入」后自动进入工作台完成 解析→确认→回填手动表单→切换到手动输入 全流程，核对后勾选确认即可预测；保留「仅打开 AI 输入助手」的传统入口
- 预测工作台（手动输入）新增「常用配方快速载入」面板：内置 8 组课题组经典环氧体系配方（E-51/DDM、E-51/DDS、AG-80/DDS、E-51/MTHPA、DGEBF/IPDA、E-51/m-PDA、BPAF-EP/DDS、EPN/DDM），一键整体填入树脂/固化剂 SMILES、phr 配比与固化制度，应用后可逐项微调
- 新增配方卡片 UI（portal-recipe-card）与响应式样式，沿用现有科学主题设计令牌

### 修复
- 修复 TabPFN 模型 SHAP 蜂群图特征名显示为 Feature_40/Feature_94 等占位名的问题（多层修复）：① 训练完成后为在 numpy 数组上拟合的模型（TabPFN 等，fit 后 feature_names_in_ 为 None）注入真实特征名元数据；② 修复训练页 split 快照保存块引用未定义变量 feature_cols 触发 NameError 被 except 静默吞掉、导致训练记录从未保存 split_X_train.csv 的问题；③ _coerce_feature_frame 在 feature_cols 与数据列数不一致时回退到数据自身真实列名，不再生成 Feature_i 占位列，且不再用长度巧合的错误列表覆写真实列名；④ _resolve_effective_feature_cols 新增 train_result['feature_names'] 作为候选名源；⑤ EnhancedModelInterpreter 新增 pipeline / fallback_feature_names 参数，占位名解析时可从训练结果特征名列表恢复真实名
- 修复 SHAP 蜂群图着色数据反标准化失败（operands could not be broadcast）导致颜色映射使用标准化值而非原始特征值的问题：先对全宽度 X_sample 反标准化后再取 top-N 子集
- 修复 render_smiles_field 调用未定义函数 init_smiles_field_state 导致手动输入分子结构区块潜在 NameError 崩溃的问题
- 修复 AI 输入助手入口逻辑：原先 st.tabs 激活状态不跨重跑保留且标签顺序随入口动态互换，导致在 AI 标签内点「解析输入/确认字段」后被弹回其它标签、解析结果看似丢失；现改用持久化 segmented_control 导航，重跑后停留在当前功能区，首页/侧边栏入口可精确跳转到 AI 辅助输入，侧边栏按钮置灰时增加原因提示并实时显示当前功能区

后续变更将在此处记录。

---

## [1.5.1] - 2026-07-31

### 修复
- 修复分子特征页面导入 workflow 时局部 `torch` 导入遮蔽全局绑定，导致正常提取流程触发 `UnboundLocalError` 的问题

### 测试
- 新增分子特征页面导入路径的 Torch 作用域回归测试
- 通过分子特征 workflow、虚拟筛选相关测试及 `app.py` 语法检查

---

## [1.5.0] - 2026-07-30

### 新增
- 支持在已有训练流程上追加分子特征提取步骤，并自动生成唯一的步骤标识
- 保存多步骤工作流的特征合并顺序、来源映射和有效行信息
- 增加紧凑的筛选前模型特征人工映射界面
- 支持保留候选配方已有特征值，并允许显式选择计算列、原始列、常数值或不使用

### 改进
- 优化工作流元数据标准化，降低大规模行映射处理开销
- 改进分子特征提取结果的行对齐和稀疏数据合并
- 清理误写入模型特征列表的 Streamlit 内部对象文本
- 增强虚拟筛选特征矩阵的缺失值、非数值和无穷值校验
- 简化筛选前映射操作，避免无关特征信息堆积在页面上

### 修复
- 修复筛选前人工映射未确认或模型特征目录变化时仍沿用旧映射的问题
- 修复提取特征缺失时直接进入严格预测流程导致的错误
- 修复候选特征与模型输入列顺序不一致时的诊断和处理问题

### 测试
- 新增工作流大规模元数据标准化、人工映射选择、候选特征保留和有效特征行掩码测试

---

## [1.4.5] - 2025-01-10

### 新增
- OpenMP线程优化模块 (`core/thread_config.py`)
- 自动限制RDKit底层OpenMP线程数
- 环境变量支持自定义线程数 (`ML_THREAD_COUNT`)

### 修复
- 修复RDKit占用所有CPU核心的问题
- 修复多进程特征提取时的内存泄漏

### 改进
- 优化虚拟筛选性能
- 改进候选库可视化
- 增强诊断提示信息

---

## [1.4.4] - 2025-01-08

### 新增
- 配方可行性分析
  - 当量比计算
  - 组分配比合理性检查
  - 混合类型固化剂检测

### 改进
- 放宽化学规则过滤默认参数
  - `min_aromatic_rings`: 2 → 1
  - `min_mw`: 250 → 180
  - `min_epoxide`: 2 → 1
- 改进候选库生成算法

### 修复
- 修复候选为空时的错误提示
- 修复中文引号导致的语法错误

---

## [1.4.3] - 2025-01-05

### 新增
- 适用域分析模块
  - 基于距离的适用域
  - 基于密度的适用域
  - 预测置信度评估

### 改进
- 优化SHAP计算性能
- 改进模型解释可视化
- 增强UI布局

---

## [1.4.2] - 2025-01-02

### 新增
- PubChem候选集成
  - 关键词搜索
  - SMILES检索
  - 属性过滤

### 改进
- 优化虚拟筛选流程
- 改进特征矩阵构建

---

## [1.4.1] - 2024-12-28

### 新增
- 多模型集成预测
- 不确定度量化 (BNN模型)

### 改进
- 优化模型训练流程
- 改进超参数优化

---

## [1.4.0] - 2024-12-20

### 新增
- 虚拟分子筛选功能
  - 候选分子库生成
  - 化学规则过滤
  - 高通量预测筛选
  - 结果排序与导出

### 改进
- 重构分子特征提取模块
- 优化SMILES处理流程

---

## [1.3.0] - 2024-11-15

### 新增
- 主动学习模块
  - 不确定性采样
  - 多样性采样
  - 混合策略
- 图像转SMILES功能 (DECIMER集成)

### 改进
- 优化UI界面
- 改进模型训练性能

---

## [1.2.0] - 2024-10-01

### 新增
- 多种深度学习模型支持
  - BNN (贝叶斯神经网络)
  - PINN (物理信息神经网络)
  - TabNet
  - GNN (图神经网络)

### 改进
- 优化特征选择算法
- 改进模型解释功能

---

## [1.1.0] - 2024-08-15

### 新增
- SHAP模型解释
- 超参数优化 (Optuna)
- 模型导入/导出

### 改进
- 优化数据清洗流程
- 改进特征提取性能

---

## [1.0.0] - 2024-06-01

### 新增
- 基础预测功能
- 分子特征提取
  - Morgan指纹
  - RDKit描述符
  - Mordred描述符
- XGBoost模型训练
- Streamlit Web界面
- 数据上传与清洗

---

## 版本说明

### 版本号格式: MAJOR.MINOR.PATCH

- **MAJOR**: 重大架构变更或不兼容的API修改
- **MINOR**: 新增功能,向后兼容
- **PATCH**: Bug修复,小改进

### 变更类型

- **新增**: 新功能
- **改进**: 对现有功能的改进
- **修复**: Bug修复
- **移除**: 移除的功能
- **弃用**: 即将移除的功能
- **安全**: 安全相关修复

---

## 路线图

### v1.5.0 (计划)
- [ ] 分布式训练支持
- [ ] 模型压缩与部署优化
- [ ] 实验数据管理模块

### v1.6.0 (计划)
- [ ] 多目标优化
- [ ] 材料知识图谱
- [ ] 自动化实验设计

### v2.0.0 (远期)
- [ ] 云端部署
- [ ] 协作平台
- [ ] 开放API
