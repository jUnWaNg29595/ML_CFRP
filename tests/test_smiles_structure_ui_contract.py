import ast
from pathlib import Path

from core.navigation import NAVIGATION_ALIASES, NAVIGATION_PAGES, resolve_navigation_page


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


def test_structure_image_page_has_new_canonical_name_and_legacy_alias():
    """The merged page gets a descriptive name while old shortcuts keep working."""
    canonical = "🧪 SMILES / BigSMILES 结构图像工具"

    assert canonical in NAVIGATION_PAGES
    assert NAVIGATION_ALIASES["图像转SMILES"] == canonical
    assert resolve_navigation_page(current_page="🏠 首页", pending_page="图像转SMILES") == canonical


def test_main_app_exposes_merged_structure_image_page_entrypoints():
    """The host app must wire the renderer and both conversion directions into one page."""
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "page_smiles_structure_tools" in function_names
    start = source.index("def page_smiles_structure_tools(")
    page_source = source[start:]

    assert "render_structure" in page_source
    assert "SMILES" in page_source
    assert "BigSMILES" in page_source
    assert "page_image_to_smiles" in page_source
    assert "process_table" in page_source
    assert "批量" in page_source
    assert "_render_structure_result" in page_source


def test_bigsmiles_renderer_is_importable_from_host_project():
    """The vendored renderer package remains usable after app integration."""
    import sys

    package_root = ROOT / "bigsmiles_ui"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from bigsmiles_ui.renderer import RenderOptions, render_structure

    result = render_structure("CCO", ROOT / "bigsmiles_ui" / "bigsmiles_ui" / "generated" / "contract", RenderOptions())
    assert result.detected_type == "smiles"
    assert result.parse_status == "valid"
