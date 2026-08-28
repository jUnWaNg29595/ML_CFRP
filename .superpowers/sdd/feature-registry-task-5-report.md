# Task 5 报告：训练前锁定并贯穿普通训练、CV 和优化

## 变更

- 新增 `core/training_contract.py`：读取并校验 approved registry、profile 和 dataset manifest，冻结 canonical feature 顺序、registry snapshot、registry hash、manifest hash、来源分区和 workflow；训练 context 对未知、重复、缺失列 fail closed；训练结果审计会标记被 feature mask 移除的 canonical 特征并返回 `publishable=False`。
- `EnhancedModelTrainer.train_model`、`cross_validate_model`、`build_regression_cv_pipeline` 增加可选 `feature_contract_context`，在进入模型分支前校验输入列，且不把 context 传给模型构造器。
- `HyperparameterOptimizer.optimize` 及 reliable/exploratory 内部路径将同一个 context 传给每个 trial pipeline 和最终 pipeline；入口先校验输入列。
- `app.py` 在构造 `X/y` 后锁定 session 级 training context，并将同一 context 传入普通训练、CV、优化预检、优化执行以及最佳参数最终训练。
- 修正训练契约测试夹具，写入匹配的 `approval.approved_hash`；补充优化 trial/final pipeline context 复用测试。

## 测试

命令：

```text
C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_training_contract.py tests/test_model_trainer_feature_mask.py tests/test_optimizer.py -q
```

输出：`20 passed, 3 warnings in 16.17s`。

另行执行 `py_compile`：`core/training_contract.py`、`core/model_trainer.py`、`core/optimizer.py`、`app.py` 均通过。

## Self-review / concerns

- 当前默认 `prediction_portal/feature_registry.json` 仍是 draft，且 app 未发现 approved dataset manifest 时会 fail closed；这是门禁要求，但部署前需要由人工批准 registry 并将 manifest 放入 session/config。
- app 现有 raw-frame 专用模型可能使用比基础 `feature_cols` 更宽的原始输入；这些路径若要启用 training context，需要为其准备与 raw 输入列完全匹配的 approved manifest/profile。
- 测试输出中的 3 个 warning 来自第三方依赖弃用提示及既有 sklearn 特征名提示，不是本次失败。
