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
