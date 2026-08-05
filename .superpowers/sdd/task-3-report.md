# Task 3 完成报告：数据探索页预览与图表快速导出

## 实现内容

- 在数据探索页增加当前处理数据/原始数据预览源选择。
- 增加支持搜索和多选的预览列选择，并在数据源切换后清理失效列。
- 将预览行数限制为最多 5000 行；空列选择只提示“至少选择一列”，不回退显示全部列。
- 增加 `_render_data_explore_preview(raw_df, processed_df)` 辅助函数。
- 增加 `_render_figure_export_controls(figure, data_df, base_name, key_prefix)` 辅助函数。
- 为相关性矩阵、Pearson 分析、分布图、箱线图和缺失值图增加当前图表数据 CSV/Excel 及 HTML/PNG/SVG 快速导出。
- 图表导出数据默认来自当前处理数据；相关性和 Pearson 使用当前图表实际计算数据。
- Excel、PNG、SVG 等单格式依赖错误单独提示，不阻断其他格式。
- 所有图表导出按钮使用稳定的页面级 key 前缀。

## TDD 验证

1. 先新增两个页面源代码回归测试。
2. 首次运行回归测试：2 个新增测试按预期失败，原因是页面缺少预览和快速导出标签。
3. 完成最小实现后运行：
   - `tests/test_app_scope_regressions.py`
   - `tests/test_data_explore_export.py`
4. 最终结果：`26 passed`。
5. 额外运行 `python -m py_compile app.py`，通过。

## 变更边界

- Task 3 自己的修改：`app.py`、`tests/test_app_scope_regressions.py`、本报告。
- `core/model_trainer.py` 保持未暂存、未提交。
- `app.py` 中用户已有的 `StandardScaler/MinMaxScaler/RobustScaler` 无关修改保持未暂存、未提交。
- 缓存、备份和其他未跟踪文件均未加入提交。

## Concerns

- PNG/SVG 是否可用取决于 Plotly 的 `kaleido` 依赖；依赖缺失时页面会仅提示对应格式不可用，HTML 和数据导出仍可继续。
- 未运行完整项目测试套件；本任务使用了 brief 指定的页面回归测试和导出核心测试。

## Task 3 审查修复报告

### 修复内容

- 将数据探索页的预览测试从纯源码字符串断言替换为真实辅助函数交互测试。
- 覆盖预览数据源选择、显式空列选择、最多 5000 行、列选择在数据源切换时保留有效交集，以及图表导出按钮的实际 payload、文件名和稳定 key。
- 修复数据源切换逻辑：保留仍存在的列；仅在切换后没有有效列选择时回退当前数据源前 8 列；显式空选择在同一数据源下仍保持空并隐藏表格。

### TDD 验证

1. RED：
   - 命令：`C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe -m pytest tests\test_app_scope_regressions.py tests\test_data_explore_export.py -q`
   - 输出：`1 failed, 27 passed`
   - 失败：数据源切换后预期保留 `shared`，实际回退为原始数据前 8 列。
2. GREEN：
   - 同一命令输出：`28 passed in 20.79s`
3. 提交前新鲜验证：
   - 同一聚焦测试命令输出：`28 passed in 20.68s`
   - 命令：`C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe -m py_compile app.py`
   - 输出：无；退出码 `0`。

### 边界与顾虑

- 未运行完整项目测试套件；本次仅运行 Task 3 相关回归/导出测试和 `app.py` 编译检查。
- Streamlit 可导入并支持行为级测试；未启动交互式 Streamlit 服务。
- PNG/SVG 仍依赖 Plotly `kaleido`，依赖缺失时由现有逻辑逐格式提示，不影响 HTML、CSV、Excel 按钮。
- `app.py` 中已有的其他未提交修改、缓存、备份和未跟踪文件未纳入本次修复提交。
