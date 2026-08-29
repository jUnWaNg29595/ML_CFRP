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


def test_molecular_feature_cleanup_collects_recorded_workflow_and_known_prefix_columns():
    import app

    columns = [
        "resin_smiles",
        "resin_xtb_gap",
        "hardener_maccs_1",
        "custom_embedding_0",
        "temperature",
    ]
    state = {
        "molecular_feature_names": ["resin_xtb_gap"],
        "molecular_feature_workflow": {
            "final_feature_names": ["hardener_maccs_1"],
            "steps": [{"feature_names": ["custom_embedding_0"]}],
        },
        "molecular_feature_config": {"feature_names": ["legacy_feature"]},
        "molecular_features": None,
    }

    assert app._collect_molecular_feature_columns(columns, state) == [
        "resin_xtb_gap",
        "hardener_maccs_1",
        "custom_embedding_0",
    ]


def test_feature_selection_page_exposes_molecular_feature_cleanup_without_touching_raw_data():
    import app

    source = APP_PATH.read_text(encoding="utf-8-sig")
    start = source.index("def page_feature_selection():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：模型训练",
        start,
    )
    page_source = source[start:end]

    assert "清除已提取分子特征" in page_source
    assert "_clear_molecular_features_from_session" in page_source
    assert "st.session_state.data" not in page_source


def test_molecular_feature_cleanup_creates_named_backup_before_mutating_state(monkeypatch):
    import app

    state = {
        "data": pd.DataFrame(
            {
                "resin_smiles": ["CCO"],
                "resin_xtb_gap": [1.2],
                "temperature": [80],
            }
        ),
        "processed_data": pd.DataFrame(
            {
                "resin_smiles": ["CCO"],
                "resin_xtb_gap": [1.2],
                "temperature": [80],
            }
        ),
        "molecular_feature_names": ["resin_xtb_gap"],
        "molecular_features": pd.DataFrame({"resin_xtb_gap": [1.2]}),
        "molecular_feature_workflow": {"final_feature_names": ["resin_xtb_gap"]},
        "molecular_feature_config": {"feature_names": ["resin_xtb_gap"]},
        "molecular_feature_trace": [{"output_columns": ["resin_xtb_gap"]}],
        "feature_cols": ["resin_xtb_gap", "temperature"],
        "multiselect_features": ["resin_xtb_gap"],
        "source_feature_names": ["resin_smiles", "resin_xtb_gap", "temperature"],
    }
    saved = []

    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.setattr(
        app,
        "_save_session_snapshot",
        lambda tag: saved.append(tag) or (True, "ok"),
    )

    removed = app._clear_molecular_features_from_session()

    assert removed == ["resin_xtb_gap"]
    assert saved and saved[0].startswith("before_molecular_feature_clear_")
    assert state["data"].columns.tolist() == ["resin_smiles", "resin_xtb_gap", "temperature"]
    assert state["data"]["resin_xtb_gap"].tolist() == [1.2]
    assert state["processed_data"].columns.tolist() == ["resin_smiles", "temperature"]
    assert state["molecular_feature_clear_backup_tag"] == saved[0]


def test_molecular_feature_cleanup_page_has_restore_entry():
    source = APP_PATH.read_text(encoding="utf-8-sig")
    assert "def _render_molecular_feature_clear_restore_control(" in source
    assert "恢复清除前分子特征" in source
    assert "molecular_feature_clear_backup_tag" in source
    for function_name in ("page_molecular_features", "page_feature_selection"):
        start = source.index(f"def {function_name}():")
        next_function = source.find("\ndef ", start + 5)
        page_source = source[start:next_function if next_function != -1 else len(source)]
        assert "_render_molecular_feature_clear_restore_control" in page_source


def test_data_explore_export_center_supports_raw_and_processed_data():
    source = APP_PATH.read_text(encoding="utf-8-sig")
    start = source.index("def page_data_explore():")
    end = source.index(
        "\n\n# ============================================================\n# 页面：数据清洗",
        start,
    )
    page_source = source[start:end]

    assert "导出数据源" in page_source
    assert "生成统一导出包" in page_source
    assert "build_export_zip" in page_source
    assert "当前处理数据" in page_source
    assert "原始数据" in page_source
    assert "图表数据" in page_source
    assert "图表文件" in page_source
    assert "描述统计" in page_source
    assert "相关性矩阵" in page_source
    assert "分布图" in page_source
    assert "箱线图" in page_source
    assert "缺失值" in page_source
    assert "CSV" in page_source
    assert "Excel" in page_source
    assert "HTML" in page_source
    assert "PNG" in page_source
    assert "SVG" in page_source
    assert "data_explore_export_" in page_source
