import ast
from pathlib import Path

import pandas as pd


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


def test_data_explore_preview_uses_selected_source_and_caps_rows_at_5000(monkeypatch):
    import app

    raw = pd.DataFrame(
        {
            "raw_feature": range(6000),
            "raw_target": range(6000, 12000),
        }
    )
    processed = pd.DataFrame({"processed_feature": range(6000)})
    state = {}
    captured = {}

    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(
        app.st,
        "radio",
        lambda label, options, **kwargs: "原始数据",
    )

    def slider(label, **kwargs):
        captured["slider"] = kwargs
        return kwargs["max_value"]

    def multiselect(label, options, **kwargs):
        captured["options"] = options
        state[kwargs["key"]] = kwargs["default"]
        return kwargs["default"]

    monkeypatch.setattr(app.st, "slider", slider)
    monkeypatch.setattr(app.st, "multiselect", multiselect)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app.st,
        "dataframe",
        lambda frame, **kwargs: captured.update({"frame": frame}),
    )

    app._render_data_explore_preview(raw, processed)

    assert state["_explore_preview_source_key"] == "raw"
    assert captured["options"] == ["raw_feature", "raw_target"]
    assert captured["slider"]["max_value"] == 5000
    assert len(captured["frame"]) == 5000
    assert list(captured["frame"].columns) == ["raw_feature", "raw_target"]


def test_data_explore_preview_keeps_only_valid_selected_columns_after_source_switch(
    monkeypatch,
):
    import app

    raw = pd.DataFrame(
        {
            **{f"raw_{index}": [index] for index in range(8)},
            "shared": [1],
            "raw_extra": [2],
        }
    )
    processed = pd.DataFrame(
        {
            "shared": [1],
            "processed_only": [2],
        }
    )
    state = {}
    source_labels = iter(["当前处理数据", "原始数据"])
    defaults = []
    selections = iter([["shared", "processed_only"], None])

    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(
        app.st,
        "radio",
        lambda label, options, **kwargs: next(source_labels),
    )
    monkeypatch.setattr(
        app.st,
        "slider",
        lambda label, **kwargs: kwargs["min_value"],
    )

    def multiselect(label, options, **kwargs):
        defaults.append(kwargs["default"])
        selection = next(selections)
        if selection is None:
            selection = kwargs["default"]
        state[kwargs["key"]] = selection
        return selection

    monkeypatch.setattr(app.st, "multiselect", multiselect)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "dataframe", lambda *args, **kwargs: None)

    app._render_data_explore_preview(raw, processed)
    app._render_data_explore_preview(raw, processed)

    assert defaults == [["shared", "processed_only"], ["shared"]]


def test_data_explore_preview_keeps_explicit_empty_selection_hidden(monkeypatch):
    import app

    frame = pd.DataFrame({"feature": [1, 2]})
    state = {}
    notices = []
    displayed = []

    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(
        app.st,
        "radio",
        lambda label, options, **kwargs: "当前处理数据",
    )
    monkeypatch.setattr(
        app.st,
        "slider",
        lambda label, **kwargs: kwargs["min_value"],
    )

    def multiselect(label, options, **kwargs):
        state[kwargs["key"]] = []
        return []

    monkeypatch.setattr(app.st, "multiselect", multiselect)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "info", lambda message: notices.append(message))
    monkeypatch.setattr(
        app.st,
        "dataframe",
        lambda *args, **kwargs: displayed.append(args[0]),
    )

    app._render_data_explore_preview(None, frame)

    assert notices == ["至少选择一列后显示预览表。"]
    assert displayed == []
