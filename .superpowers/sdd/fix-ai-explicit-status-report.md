# AI 特征审核显式状态修复报告

## 变更

- `parse_feature_mapping_response()` 对 `suggestions` 中缺失 `status` 的条目直接抛出 `ValueError`，不再静默规范化为 `pending_review`。
- 显式 `pending_review`、`conflict`、`unknown` 仍按原有允许状态解析；`apply_feature_review_decision()` 继续只接受显式 `pending_review`，本地批准输出 `review_status=approved`。
- 增加 parser 和 `request_feature_mapping_review()` 缺失状态回归测试，并覆盖显式状态保留；保留 unknown/conflict 阻断及显式 pending 本地 accept 覆盖。

## TDD 验证

1. 红灯：新增缺失状态测试后运行
   `python -m pytest tests/test_portal_ai_schema.py tests/test_feature_mapping_review.py -q`
   输出 `2 failed, 33 passed`，失败均为预期的“未抛出 ValueError”。
2. 绿灯：修复 parser 后同一命令输出 `35 passed in 0.94s`。

## 覆盖测试与编译

- `C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe -m pytest tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py tests/test_portal_ai_schema.py tests/test_portal_ai.py tests/test_legacy_tg_gate.py -q`
  - 输出：`65 passed in 2.36s`
- `C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe -m compileall -q core tests`
  - 输出：`compileall exit 0`
- `git diff --check`
  - 无 diff 错误（仅报告现有 CRLF 转换提示）。

## 环境说明

使用系统 Python 3.13 运行同一覆盖集时为 `58 passed, 7 failed`；失败来自无关环境依赖（缺少 `streamlit`，以及已安装 SciPy 与 NumPy 的 `np.long` 兼容问题）。项目 `CFRP_env` 解释器运行结果如上且全部通过。
