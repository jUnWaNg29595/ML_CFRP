# Task 2：安全多服务 AI 配置审查修复报告

## 状态

已完成审查修复。工作树为 `C:\Users\wangj\Desktop\CFRP系统-worktrees\portal-ai-ui`，仅更新以下三个指定文件：

- `core/portal_ai_config.py`
- `tests/test_portal_ai_config.py`（保留并纳入现有审查测试补丁）
- `.superpowers/sdd/task-2-report.md`

## RED/GREEN 证据

### RED

- 使用现有测试补丁执行 `python -m pytest -q tests/test_portal_ai_config.py`。
- 基线为 `15 failed, 21 passed`，失败集中在 purpose 默认值/枚举、URL hostname/port 与凭据参数、递归 dataclass/list 支持及输出形状。

### GREEN

- `python -m pytest -q tests/test_portal_ai_config.py`：`36 passed`。
- `python -m pytest -q tests/test_portal_ai_config.py tests/test_prediction_portal.py`：`62 passed`。
- `python -m compileall -q core/portal_ai_config.py tests/test_portal_ai_config.py`：通过。
- `git diff --check`：通过。

## 修复内容

- `purpose` 仅接受 `input_parsing`、`result_explanation`、`both`，默认值统一为 `both`，dataclass 默认值同步更新。
- 使用 `urllib.parse.urlsplit/urlunsplit` 校验 scheme、hostname、IPv4/IPv6、端口范围和 URL credentials；仅移除 path 末尾斜杠，不影响 query/fragment 中的安全参数。
- URL query/fragment 中的 `api_key`、token、secret、password、key 等敏感参数在校验时拒绝；脱敏/导出时移除敏感参数并保留非敏感参数。
- 以单一递归转换路径统一支持 dataclass、`Mapping`、嵌套 list/tuple 的 validate、redacted 和 exportable，保留未知 JSON 数据且不修改输入对象。
- 原子写入与备份临时文件均先设置 Unix `0o600` 权限；Windows 对 chmod 失败保持兼容，不因权限调用破坏运行。
- 写入、fsync 或 replace 失败时保留旧配置，并在 finally 中清理临时文件；备份仍在正式配置替换前生成。
- 删除重复的 URL 校验/转换逻辑，避免对 path、query、fragment 进行不一致处理。

## 约束与提交

- 未修改门户运行时端口、`prediction_config.json`、`app.py`、`UserPrediction.py` 或其他无关文件。
- 未访问网络或真实模型凭据。
- 修复验证完成后提交为单独的修复 commit；未纳入工作树中其他预先存在的 `.superpowers/sdd` 未跟踪文件。
