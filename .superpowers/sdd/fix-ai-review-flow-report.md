# AI 特征审核批准闭环修复报告

## 变更

- `apply_feature_review_decision` 的 `accept`/`edit_accept` 现在要求原始 AI 建议显式为 `status=pending_review`。
- `unknown`、`conflict`、缺失及其他状态（包括 `approved`）均拒绝批准；`reject` 仍只追加审核记录。
- `edit_accept` 的编辑 payload 可省略 `status`，沿用原始 `pending_review`；若显式提供，必须仍为 `pending_review`。
- 本地批准继续输出 `feature_bindings[].review_status=approved`，未放开 AI parser 直接生成 `approved`。

## TDD 验证

1. 先将正常 accept 夹具改为 `pending_review` 并增加编辑后接受成功测试，确认旧实现因仍要求 `approved` 而失败（6 个失败）。
2. 实现最小状态门禁后，审核测试全部通过。

## 测试结果

- `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py tests/test_portal_ai_schema.py tests/test_portal_ai.py tests/test_legacy_tg_gate.py -q`
  - `60 passed`
- 完整特征契约套件（registry、manifest、training、prediction、workflow 及审核相关测试）
  - `139 passed, 2 warnings`
- `C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m compileall -q core tests`
  - 通过

## 提交

提交哈希见 git commit。
