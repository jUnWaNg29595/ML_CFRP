# 材料机器学习平台 - 变更日志

所有重要的更改都将记录在此文件中。

本文档格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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
