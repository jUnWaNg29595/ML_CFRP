# -*- coding: utf-8 -*-
"""DECIMER 模型加载性能诊断记录

环境：conda env CFRP_env（主应用运行环境）
日期：2026-08-31

实测分解（冷启动、模型文件已在本地 ~/.data/DECIMER-V2）：
- import tensorflow: ~6.4 秒
- ensure_models（已下载检查）: ~0 秒
- load_tokenizer: ~0 秒
- tf.saved_model.load(DECIMER_model): ~31 秒
- tf.saved_model.load(DECIMER_HandDrawn_model): ~35 秒
- 合计 import DECIMER.decimer（模块顶层即执行 get_models）: ~63-68 秒

关键结构问题：
- DECIMER/decimer.py 第 134 行在**模块顶层**执行
  `tokenizer, DECIMER_V2, DECIMER_Hand_drawn = get_models(model_urls)`，
  即 import 该模块必然加载两个 332MB 的 SavedModel。
- core/image_smiles_extractor._get_decimer_module 有 lru_cache(1)，
  进程内只发生一次，但首停首次进入"图像转SMILES"子页仍需 60+ 秒，
  表现为页面"卡住"。
- app.py 中 st.session_state["_decimer_ready_status"] 只在 session 生命周期内
  缓存 (ok, msg)；新浏览器会话/刷新后 session_state 重置，但
  lru_cache 的模块加载结果仍在 → 不会重复 65 秒。
- 真正的重复加载场景：Streamlit 文件监视器检测到 .py 变更触发
  script rerun（模块重新执行）或多 worker 场景。平时主要为首次进入的 65 秒。
"""

---
## 内存膨胀（21GB）来源分析（追加测量，2026-08-31）

在 CFRP_env 中对**单进程仅 import、不加载数据**实测工作集增长：

| 组件 | 相对增量 |
|---|---|
| 启动基线 | 15 MB |
| pandas + numpy | +63 MB（78 MB）|
| xgboost | +70 MB（148 MB）|
| catboost | +22 MB（170 MB）|
| torch（仅 import） | **+579 MB（749 MB）** |
| rdkit | +18 MB（767 MB）|
| tensorflow | import 时另占（GPU/CUDA 常驻，通常 GB 级随 CUDA 分配浮动）|

结论：
- **torch +579MB 是纯 import torch 就吃掉的**。app.py 本身不顶层 import torch，
  但顶层 `from core.model_trainer / task_manager / molecular_features / ...` 会
  传递 import 大量 `import torch` 的 core 模块 → app 启动即整进 torch（+TF、xgboost 等）。
- 因此主进程基线内存约 1–2GB 几乎全是 ML 库常驻，独立于用户数据。
- **21GB 的绝对大头几乎必然来自用户会话数据**：`st.session_state` 中持久化的
  `X_train / X_test / y_train / y_test / model / scaler / processed_data` 等（训练/预测
  数据集可能数 GB），加上 SHAP 缓存、XGBoost 模型缓存、DECIMER 模型（每次 332MB×2，已懒加载）。
  这些跨 rerun 存活、不会被 GC，是内存膨胀主因。

### 与该页面相关的最优内存/速度权衡
- 图像→SMILES 子页只在**图片转码**时用 DECIMER → 已懒加载，进页面不加载模型（首帧/首切换
  仅首次 ~6s TF import），模型 660MB 只在首次“识别”时进内存。
- 渲染（SMILES→图）与批量检查子页**纯 RDKit**，完全不会触发 TF/DECIMER → 切换为近零开销。
- 若要牺牲内存换“首次识别也不停顿”，可后台调用 `DECIMER.decimer.warmup_models()`，
  代价是把 ~660MB SavedModel 常驻内存（在当前已 21GB 场景不推荐默认开启）。
