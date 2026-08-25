# Task 2：安全多服务 AI 配置审查修复报告

## 状态

已完成第二轮独立审查修复。工作树为 `C:\Users\wangj\Desktop\CFRP系统-worktrees\portal-ai-ui`，本次提交仅包含：

- `core/portal_ai_config.py`
- `tests/test_portal_ai_config.py`
- `.superpowers/sdd/task-2-report.md`

## 第二轮审查结论

- 复核 `5f78656` 的 `_normalize_url`：当前函数签名为 `(_url, *, reject_sensitive)`，函数体不再引用未定义的 `provider`，不存在该 P0 NameError。
- URL 导出脱敏现在按解码后的 query/fragment 参数值清理已知 secret 和常见 bearer/API key 值，即使参数名任意也不会原样导出。
- 导出/脱敏处理 percent-encoded query/fragment，并重建不含 username/password 的 netloc；userinfo 不会泄露。
- Windows 下 `chmod` 失败会被兼容处理，不阻断原子写入；Unix 仍保持 `0o600` 权限保护。

## 原有实现与本轮修复

- `purpose` 仅接受 `input_parsing`、`result_explanation`、`both`，默认值统一为 `both`。
- 使用 `urllib.parse.urlsplit/urlunsplit` 严格校验 scheme、hostname、IPv4/IPv6、端口范围和 URL credentials；仅移除 path 末尾斜杠。
- 校验阶段拒绝 query/fragment 中的敏感参数；导出/脱敏阶段保留非敏感参数并清理敏感名称和值。
- 统一支持 dataclass、`Mapping`、嵌套 list/tuple 的 validate、redacted 和 exportable，保留未知 JSON 数据且不修改输入对象。
- 原子写入与备份临时文件使用权限保护；写入、fsync 或 replace 失败时旧配置保持不变并清理临时文件。
- 删除重复 URL 处理逻辑，避免校验与导出对 path、query、fragment 的处理不一致。

## 验证

- `python -m pytest -q tests/test_portal_ai_config.py tests/test_portal_ai_schema.py tests/test_prediction_portal.py`：`78 passed`。
- `python -m compileall -q core/portal_ai_config.py tests/test_portal_ai_config.py`：通过。
- `git diff --check`：通过。

## 约束

- 未修改 `app.py`、`UserPrediction.py`、`prediction_config.json` 或其他无关文件。
- 未纳入工作树中预先存在的其他 `.superpowers/sdd` 未跟踪文件。

## 本次复审记录（2026-08-25）

- 修复 P1：`_SECRET_KEY_PATTERN` 现在要求敏感字段完整匹配，并以非字母数字作为分隔边界，避免 `max_tokens` 因包含 `token` 被误判。
- 回归测试确认 `max_tokens` 保留且等于 `2048`，`api_key` 在 exportable 结果中删除、在 redacted 结果中掩码。
- 本次仅修改 `core/portal_ai_config.py`、`tests/test_portal_ai_config.py` 与本报告，不纳入其他预先存在的未跟踪文件。
