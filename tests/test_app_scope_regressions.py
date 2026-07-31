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
