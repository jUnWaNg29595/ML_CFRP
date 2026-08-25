# Task 3 完成报告：OpenAI 兼容门户 AI 客户端

## 状态

已完成任务 3，工作树为 `C:\Users\wangj\Desktop\CFRP系统-worktrees\portal-ai-ui`。

## 实现内容

- 新增 `core/portal_ai.py`，提供 `PortalAIClient`、`parse_chat_completion()` 和 `parse_json_or_markdown_json()`。
- 使用 `POST /chat/completions` 调用 OpenAI-compatible 服务；API Key 只进入 `Authorization: Bearer ...` 请求头。
- 注入可测试的 `transport` seam，不在测试中联网。
- 对 timeout、连接异常、408、409、425、429 和 5xx 执行一次重试，最多两次尝试。
- 对 401、403、429、瞬时网络错误和 malformed JSON 分类，错误信息不包含 API Key、请求头或完整请求体。
- 支持纯 JSON 与单个 Markdown fenced JSON；拒绝带解释性 prose 或多重代码围栏的响应。
- AI 输出继续进入既有 `portal_ai_schema` 校验，不执行 Python、命令或其他代码。
- 保留语义兼容异常别名，便于调用方区分认证、请求、限流和响应解析错误。

## 验证

- `python -m pytest -q tests/test_portal_ai.py`：7 passed。
- `python -m pytest -q tests/test_portal_ai.py tests/test_portal_ai_config.py tests/test_portal_ai_schema.py tests/test_prediction_portal.py`：86 passed。
- `python -m compileall -q .`：通过。
- `git diff --check`：通过。

## 变更边界

本任务只提交 `core/portal_ai.py`、`tests/test_portal_ai.py` 和本报告；工作树中预先存在的其他 `.superpowers/sdd` 文件、缓存、模型、数据和备份不纳入提交。

## 安全边界

- AI 只能解析输入和解释结果，不能执行 Python、Shell、模块导入或模型操作。
- API Key 不写入日志、错误文本、导出对象或配置请求体。
- 实际特征工程、xTB、模型加载与预测仍由 Python 后端负责，AI 客户端不绕过确认门禁。
## 独立审查后的修复

- 清除 transport 任意异常的上下文链，避免底层异常文本或请求对象携带 API Key/headers。
- 普通 4xx 错误保留 `PortalAIHTTPError` 类型和 `status_code`。
- 确保最多两次重试，且只在实际重试前 sleep；transport 注入使用 `is None` 判断。
- JSON 解析器拒绝数组、数字、字符串和 `null` 等非对象响应。

复审补测覆盖上述边界，任务 3 与既有门户回归共 `89 passed`。
