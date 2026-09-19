# -*- coding: utf-8 -*-
"""防回归：app_lib.py 中不得存在未定义的全局名。

背景
----
`app_lib.py` 由原单体 `app.py` 拆分而来，拆分过程丢失了若干代码块，
同时历史上遗留了一批"引用了从未定义的名字"的分支。这类问题在
Python 中不会在导入时报错，只在**该分支真正执行时**才抛
`NameError`，因此极易漏测：

* `non_empty_batches`（分子特征页批量模式未选列）——用户实际踩到
* `pending_formula_pool` / `pending_formula_ready` /
  `formula_preflight_mapping`（虚拟筛选页主路径）——整页必然崩溃
* `r2_score` / `confusion_matrix` / `roc_curve` 等 sklearn 指标函数
  ——只存在于 `generate_training_script_code` 的 f-string 模板里，
  模块级从未导入，分类训练结果渲染必崩
* `detected_design_cols`、`Dict` / `List`、`_mordred_progress`

本测试用纯 AST 静态分析（不依赖 pyflakes/flake8 等外部工具）扫描
整个模块，任何未定义的全局名都会失败。

注意：该检查器与 pyflakes 同为"非流敏感"分析——只要名字在函数内
任意位置被绑定，就视为已定义。因此它不会误报
`_mordred_progress if '_mordred_progress' in dir() else None` 这类
写法，但足以拦住真正致命的"全函数未定义"问题。
"""

import ast
import builtins
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app_lib.py"

# 解释器在模块运行时自动注入的 dunder，源码中不会显式绑定
IMPLICIT_MODULE_GLOBALS = frozenset(
    {
        "__file__",
        "__name__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__builtins__",
        "__debug__",
        "__cached__",
    }
)


def _collect_bound_names(node: ast.AST) -> set:
    """收集一棵子树中所有会被绑定的名字（不区分作用域）。"""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                if alias.name == "*":
                    names.add("*")
                else:
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
    return names


def _find_undefined_names(path: Path):
    """返回 [(lineno, function_name, name), ...]。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))

    module_names = _collect_bound_names(tree)
    if "*" in module_names:
        # 存在 star import 时无法静态判定，交由 pyflakes 处理
        pytest.skip("app_lib.py 使用了 star import，静态检查不适用")

    known = module_names | set(dir(builtins)) | IMPLICIT_MODULE_GLOBALS

    problems = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_names = _collect_bound_names(func)
        for child in ast.walk(func):
            if not isinstance(child, ast.Name):
                continue
            if not isinstance(child.ctx, ast.Load):
                continue
            if child.id in local_names or child.id in known:
                continue
            problems.append((child.lineno, func.name, child.id))
    return sorted(set(problems))


def test_app_lib_has_no_undefined_global_names():
    """app_lib.py 中不允许出现未定义的全局名（NameError 隐患）。"""
    problems = _find_undefined_names(APP_PATH)

    assert not problems, (
        "app_lib.py 存在未定义的全局名，会在对应分支执行时抛 NameError：\n"
        + "\n".join(
            f"  line {lineno}: '{name}' 未定义（位于 {func}()）"
            for lineno, func, name in problems
        )
    )


def test_molecular_features_batch_mode_without_selection_does_not_reference_undefined_names():
    """分子特征页：批量模式未选列时必须干净退出（用户实际踩到的崩溃点）。"""
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    start = source.index("def page_molecular_features():")
    end = source.index("\ndef page_molecular_feature_reproduction():", start)
    page = source[start:end]

    assert "请至少选择一个SMILES列进行批量处理" in page, (
        "批量模式未选列时应给出明确提示"
    )

    # 用 AST 判断真实引用（而非注释里提到该名字）
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "page_molecular_features"
    )
    referenced = {
        child.id
        for child in ast.walk(func)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    assert "non_empty_batches" not in referenced, (
        "page_molecular_features 不应引用 non_empty_batches——"
        "那是虚拟筛选页的变量，曾被误粘贴到此处导致批量模式未选列即崩溃"
    )
    assert "vs_formula_batch_export_all" not in page, (
        "分子特征页不应包含虚拟筛选页的下载按钮 key"
    )


def test_virtual_screening_page_defines_pending_pool_before_use():
    """虚拟筛选页：pending_formula_pool 系列变量必须先初始化再使用。"""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8-sig"))

    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_page_virtual_screening_formula"
    )

    required = {"pending_formula_pool", "pending_formula_ready", "formula_preflight_mapping"}

    bound_lines = {}
    used_lines = {}
    for child in ast.walk(func):
        if isinstance(child, ast.Name):
            if child.id in required:
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    bound_lines.setdefault(child.id, child.lineno)
                elif isinstance(child.ctx, ast.Load):
                    used_lines.setdefault(child.id, child.lineno)

    for name in sorted(required):
        assert name in bound_lines, (
            f"_page_virtual_screening_formula 从未绑定 {name}，"
            "该页会在主路径上抛 NameError（预检初始化块曾整体丢失）"
        )
        assert bound_lines[name] < used_lines[name], (
            f"{name} 首次绑定于第 {bound_lines[name]} 行，"
            f"但第 {used_lines[name]} 行就已使用——初始化必须前置"
        )


def test_sklearn_metric_helpers_are_imported_at_module_level():
    """sklearn 指标函数必须在模块级导入（而非仅存在于生成脚本的模板字符串里）。"""
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    required = {
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "auc",
        "average_precision_score",
    }

    imported = set()
    for node in tree.body:  # 只看模块级，不看函数内
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.metrics":
            for alias in node.names:
                imported.add(alias.asname or alias.name)

    missing = required - imported
    assert not missing, (
        "以下 sklearn 指标函数未在 app_lib.py 模块级导入："
        + ", ".join(sorted(missing))
        + "。它们此前只出现在 generate_training_script_code 的 f-string 模板内，"
        "导致分类训练结果渲染与手动训练指标计算抛 NameError。"
    )


def test_typing_helpers_are_imported():
    """Dict / List 等 typing 名字需真实导入（局部注解虽不求值，仍是隐患）。"""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8-sig"))

    imported = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                imported.add(alias.asname or alias.name)

    for name in ("Dict", "List", "Optional"):
        assert name in imported, f"typing.{name} 未导入"


def test_detected_design_cols_is_not_referenced():
    """detected_design_cols 从未定义，应改用 numeric_design_cols。"""
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_page_virtual_screening_formula"
    )
    bound = _collect_bound_names(func)

    assert "detected_design_cols" not in bound, (
        "detected_design_cols 从未被绑定，引用它会抛 NameError；"
        "推荐工艺列请使用 numeric_design_cols"
    )
