import ast
from pathlib import Path


def test_app_imports_all_scalers_at_module_scope():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8-sig'))

    imported_scalers = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != 'sklearn.preprocessing':
            continue
        imported_scalers.update(
            alias.name
            for alias in node.names
            if alias.name in {'StandardScaler', 'MinMaxScaler', 'RobustScaler'}
        )

    assert imported_scalers == {
        'StandardScaler',
        'MinMaxScaler',
        'RobustScaler',
    }
