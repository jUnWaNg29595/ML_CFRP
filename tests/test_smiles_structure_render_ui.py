"""SMILES / BigSMILES 结构渲染页的端到端回归（AppTest，不启动浏览器）。

覆盖三处已知体验问题：
1. 主结构图又小又模糊 → 自适应画布 + 超采样 + 铺满容器 + 矢量 SVG；
2. 顶层 "." 连接的多组分不展示 → 逐组分面板 + 整体视图；
3. "完整外部骨架" 永远只是 {R1} 占位符 → 额外输出重复单元已展开的连通骨架。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGE_SCRIPT = (
    "import sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "from app_lib import page_smiles_structure_tools\n"
    "page_smiles_structure_tools()\n"
)

MULTI_COMPONENT = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1.NCCN"
BIGSMILES = "CC{[>][<]CC(C)[>][<]}CC(C)=C"
#: 化学上合法，但 {…} 未声明端基，且含空方括号 [] → 官方解析器会拒绝。
REJECTED_BIGSMILES = "CC{C1CCC([])(O)CC1}CC"


def _open_render_page(timeout: float = 600.0) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT, default_timeout=timeout)
    at.run()
    assert not at.exception, [error.value for error in at.exception]
    at.radio(key="smiles_tools_sub_mode_v2").set_value("🎨 SMILES / BigSMILES 转图片").run()
    assert not at.exception, [error.value for error in at.exception]
    return at


def _submit(at: AppTest, raw: str, structure_type: str = "自动识别") -> AppTest:
    at.text_area(key="structure_visualization_input").set_value(raw)
    at.selectbox(key="structure_visualization_type").set_value(structure_type)
    at.button[0].click().run()
    assert not at.exception, [error.value for error in at.exception]
    return at


def test_multi_component_smiles_renders_component_panels_and_overall_view():
    at = _submit(_open_render_page(), MULTI_COMPONENT)

    assert any("解析成功" in str(item.value) for item in at.success)
    assert any("2 个顶层组分" in str(item.value) for item in at.warning)
    assert not at.error

    labels = [str(item.label) for item in at.download_button]
    assert any("下载主图 PNG" in label for label in labels)
    # 纯 SMILES 路径必须额外给出矢量下载。
    assert any("下载主图 SVG" in label for label in labels)


def test_svg_is_shown_as_the_main_preview():
    at = _submit(_open_render_page(), MULTI_COMPONENT)
    assert at.image, "主图必须被渲染出来"
    captions = [str(value) for element in at.image for value in (element.captions or [])]
    assert any("矢量 SVG" in value for value in captions), captions
    # 真正的矢量输出（而不是把 PNG 换个名字）。
    assert "data:image/svg+xml" in str(at.image[0].proto)


def test_bigsmiles_shows_expanded_framework_panel():
    at = _submit(_open_render_page(), BIGSMILES, "BigSMILES")

    assert any("重复单元已展开" in str(item.value) for item in at.warning)
    assert not at.error
    labels = [str(item.label) for item in at.download_button]
    assert any("下载主图 PNG" in label for label in labels)


def test_rejected_bigsmiles_reports_actionable_warning_instead_of_success():
    """曾经的体验问题：明明官方解析器拒绝了，界面却显示“解析成功”＋一屏 token dump。"""
    at = _submit(_open_render_page(), REJECTED_BIGSMILES, "BigSMILES")

    # 1) 不能再说“解析成功”。
    assert not any("解析成功" in str(item.value) for item in at.success)

    # 2) 必须告诉用户“官方解析器拒绝了”，而不是“库未找到解析入口”。
    warnings = "\n".join(str(item.value) for item in at.warning)
    assert "官方 BigSMILES 解析器未接受" in warnings
    assert "未找到可用的解析入口" not in warnings

    # 3) 必须给出可照做的修改建议。
    assert "空方括号" in warnings
    assert "端基" in warnings

    # 4) 不再把解析库的原始异常（整段输入回显 + 逐 token dump）倒到界面上。
    assert "Parsing failed on" not in warnings  # 原始异常才有的输入回显前缀
    assert REJECTED_BIGSMILES not in warnings
    assert len(warnings) < 800


def test_renderer_description_does_not_claim_official_parser_when_rejected():
    at = _submit(_open_render_page(), REJECTED_BIGSMILES, "BigSMILES")
    warnings = "\n".join(str(item.value) for item in at.warning)
    assert "Olsen Lab bigsmiles" in warnings  # 版本信息仍然保留，便于排障
    assert "BigSMILES 保守检查" in warnings


def test_static_disclaimers_render_as_caption_not_warning():
    at = _submit(_open_render_page(), BIGSMILES, "BigSMILES")
    captions = "\n".join(str(item.value) for item in at.caption)
    assert "不代表唯一完整聚合物分子" in captions
    assert "不代表真实聚合物构象" in captions

    warnings = "\n".join(str(item.value) for item in at.warning)
    assert "不代表唯一完整聚合物分子" not in warnings


def test_structural_warnings_are_rendered_as_a_list():
    at = _submit(_open_render_page(), REJECTED_BIGSMILES, "BigSMILES")
    warnings = [str(item.value) for item in at.warning]
    assert any(value.lstrip().startswith("需要注意：") and "\n- " in value for value in warnings)


def test_auto_fit_canvas_removes_whitespace_and_keeps_supersampling():
    """长条分子应被收紧到接近自身长宽比，而不是塞进固定高度的大画布。"""
    import app_lib as app
    from PIL import Image

    result, output_dir = app._cached_render_structure(
        "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
        "auto",
        False,
        5,
        42,
        1200,
        800,
        2.0,
        True,
        True,
        True,
    )
    assert result.draw_status == "rendered"

    output_path = Path(output_dir)
    with Image.open(output_path / result.main_image_path) as image:
        width, height = image.size
    assert width == 2400  # 1200px × 2 倍超采样
    assert height < 800  # 自适应后远低于高度上限

    assert result.svg_path
    assert (output_path / result.svg_path).exists()


def test_render_options_are_part_of_the_cache_directory():
    """切换画质/布局后不能命中同一目录，否则会看到上一组选项的旧图。"""
    import app_lib as app

    _result_a, dir_a = app._cached_render_structure(
        "CCO", "auto", False, 5, 42, 1200, 800, 1.0, True, True, True
    )
    _result_b, dir_b = app._cached_render_structure(
        "CCO", "auto", False, 5, 42, 1200, 800, 3.0, True, True, True
    )
    assert dir_a != dir_b
