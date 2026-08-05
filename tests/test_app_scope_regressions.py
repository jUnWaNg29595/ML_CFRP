import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_molecular_features_page_does_not_bind_torch_locally():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8-sig"))
    page_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "page_molecular_features"
    )

    local_torch_imports = []
    for node in ast.walk(page_function):
        if isinstance(node, ast.Import):
            local_torch_imports.extend(
                alias
                for alias in node.names
                if alias.name == "torch" and (alias.asname or "torch") == "torch"
            )

    assert not local_torch_imports, (
        "page_molecular_features must use the module-level torch binding; "
        "a local import shadows it across the entire function"
    )


def test_hyperparameter_page_uses_persisted_result_without_second_random_split():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    start = source.index("def page_hyperparameter_optimization():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：主动学习",
        start,
    )
    page_source = source[start:end]

    assert "optimization_result" in page_source
    assert "可信优化基线" in page_source
    assert "独立测试集结果" in page_source
    assert "未参与调参" in page_source
    assert "train_test_split(" not in page_source


def test_hyperparameter_page_does_not_apply_reliable_preflight_to_exploration():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    start = source.index("def page_hyperparameter_optimization():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：主动学习",
        start,
    )
    page_source = source[start:end]

    assert "if is_exploratory:" in page_source
    assert "else:" in page_source
    assert "探索模式不使用独立测试集" in page_source
    assert "if use_process_pls and preflight_error is None:" in page_source


def test_data_explore_page_has_searchable_preview_and_explicit_selection():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "数据预览" in page_source
    assert "预览特征/列" in page_source
    assert "至少选择一列" in page_source
    assert "processed_data" in page_source
    assert "原始数据" in page_source


def test_data_explore_page_has_both_quick_export_formats():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "导出图表数据" in page_source
    assert "导出图表" in page_source
    assert "HTML" in page_source
    assert "Excel" in page_source
