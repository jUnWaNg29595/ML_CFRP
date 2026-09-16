from pathlib import Path

from PIL import Image

import bigsmiles_ui.renderer as renderer_module
from bigsmiles_ui.renderer import RenderOptions, identify_structure_type, render_structure


BIGSMILES_EXAMPLE = "CC{[>][<]CC(C)[>][<]}CC(C)=C"


def test_empty_input_is_scalar_empty_result(tmp_path: Path):
    result = render_structure("  ", tmp_path, RenderOptions())
    data = result.to_scalar_dict()
    assert data["结构检查_原始字符串"] == "  "
    assert data["结构检查_解析状态"] == "empty"
    assert data["结构检查_识别类型"] == "unknown"
    assert all(not isinstance(value, (list, dict, tuple, set)) for value in data.values())


def test_auto_detection_distinguishes_smiles_and_bigsmiles():
    assert identify_structure_type("CCO", "auto") == "smiles"
    assert identify_structure_type(BIGSMILES_EXAMPLE, "auto") == "bigsmiles"


def test_result_has_fixed_scalar_keys(tmp_path: Path):
    result = render_structure("CCO", tmp_path, RenderOptions())
    keys = set(result.to_scalar_dict())
    assert keys == {
        "结构检查_原始字符串", "结构检查_识别类型", "结构检查_解析状态",
        "结构检查_规范化结构", "结构检查_主图路径", "结构检查_采样图路径",
        "结构检查_绘图状态", "结构检查_错误信息", "结构检查_警告信息",
        "结构检查_渲染器",
    }


def test_valid_smiles_renders_nonempty_png(tmp_path: Path):
    result = render_structure("CCO", tmp_path, RenderOptions())
    assert result.parse_status == "valid"
    assert result.detected_type == "smiles"
    assert result.draw_status == "rendered"
    image_path = tmp_path / result.main_image_path
    assert image_path.exists()
    assert image_path.stat().st_size > 100
    with Image.open(image_path) as image:
        assert image.width > 0 and image.height > 0


def test_invalid_smiles_has_no_fake_image(tmp_path: Path):
    result = render_structure("C1(CC", tmp_path, RenderOptions())
    assert result.parse_status == "invalid"
    assert result.main_image_path == ""
    assert result.error_message


def test_smiles_output_is_deterministic_for_same_input(tmp_path: Path):
    first = render_structure("CCO", tmp_path / "a", RenderOptions())
    second = render_structure("CCO", tmp_path / "b", RenderOptions())
    assert first.normalized_structure == second.normalized_structure
    assert (tmp_path / "a" / first.main_image_path).read_bytes() == (tmp_path / "b" / second.main_image_path).read_bytes()


def test_bigsmiles_is_not_sent_to_rdkit_as_plain_smiles(tmp_path: Path):
    result = render_structure(BIGSMILES_EXAMPLE, tmp_path, RenderOptions())
    assert result.detected_type == "bigsmiles"
    assert result.renderer
    assert result.parse_status in {"valid", "invalid"}
    if result.parse_status == "valid":
        assert result.draw_status in {"rendered", "valid_but_not_renderable", "main_rendered_sample_failed"}


def test_bigsmiles_sample_is_opt_in(tmp_path: Path):
    result = render_structure(BIGSMILES_EXAMPLE, tmp_path, RenderOptions(render_sample_chain=False))
    assert result.sample_image_path == ""
    assert result.draw_status != "main_rendered_sample_failed"


def test_bigsmiles_sample_is_marked_as_representative(tmp_path: Path):
    result = render_structure(
        BIGSMILES_EXAMPLE,
        tmp_path,
        RenderOptions(render_sample_chain=True, repeat_units=5, random_seed=42),
    )
    if result.sample_image_path:
        assert "sample" in result.sample_image_path
    assert "代表性" in result.warning_message or result.sample_image_path == ""


def test_bigsmiles_renderer_uses_rdkit_fragment_layout_when_available(tmp_path: Path):
    result = render_structure(BIGSMILES_EXAMPLE, tmp_path, RenderOptions())
    assert result.parse_status == "valid"
    assert result.draw_status == "rendered"
    assert "RDKit BigSMILES 片段布局" in result.renderer
    assert (tmp_path / result.main_image_path).exists()


def test_bigsmiles_rdkit_layout_preserves_branch_and_ring_fragment(tmp_path: Path):
    value = "CC{[>]C1CC(C)CCC1[<]}CC"
    result = render_structure(value, tmp_path, RenderOptions(requested_type="bigsmiles"))
    assert result.parse_status == "valid"
    assert result.draw_status == "rendered"
    assert "RDKit BigSMILES 片段布局" in result.renderer
    assert result.main_image_path
    with Image.open(tmp_path / result.main_image_path) as image:
        # 面板图的可视宽度固定为 image_width；高度按内容自适应，
        # 不再把分子硬塞进固定画布（否则宽分子会被压得很小）。
        assert image.width == 1000
        assert 200 < image.height < 1400


def test_bigsmiles_main_figure_grows_instead_of_clipping_panels(tmp_path: Path):
    """面板变多时画布要长高，而不是把右侧面板裁掉。"""
    value = "CC{[>][<]CC(C)[>][<]}CC{[>][<]CC[>][<]}CC(C)=C.NCCN"
    result = render_structure(value, tmp_path, RenderOptions(image_width=600))
    assert result.draw_status == "rendered"
    with Image.open(tmp_path / result.main_image_path) as image:
        assert image.width == 600
        assert image.height > 400


def test_bigsmiles_expanded_framework_is_connected_and_highlighted(tmp_path: Path):
    """{重复单元} 会展开成一张连通骨架，并高亮出重复单元原子。"""
    value = "CC{[>][<]CC(C)[>][<]}CC(C)=C"
    specs = renderer_module._bigsmiles_component_specs(value, RenderOptions())
    expanded = [spec for spec in specs if spec.kind == "expanded"]
    assert len(expanded) == 1
    spec = expanded[0]
    assert spec.molecule is not None
    assert spec.raw.count("{") == 0
    assert spec.highlight_atoms
    assert all(0 <= index < spec.molecule.GetNumAtoms() for index in spec.highlight_atoms)
    # 展开后的骨架应当比“占位视图”多出重复单元的原子。
    context = next(item for item in specs if item.kind == "context")
    assert spec.molecule.GetNumAtoms() > context.molecule.GetNumAtoms()


def test_bigsmiles_accepted_by_official_parser_is_attributed_to_it(tmp_path: Path):
    result = render_structure(
        BIGSMILES_EXAMPLE, tmp_path, RenderOptions(requested_type="bigsmiles")
    )
    assert result.parser_status == "accepted"
    assert result.renderer.startswith("Olsen Lab bigsmiles")
    assert "拒绝" not in result.warning_message


def test_rejected_library_error_is_short_single_line_and_not_misattributed(tmp_path: Path):
    """官方解析器拒绝时，界面不能自称是 Olsen Lab 解析的，也不能把 token dump 透传。"""
    value = "CC{C1CCC(O)CC1}CC"  # 合法化学，但 {…} 没写端基
    result = render_structure(value, tmp_path, RenderOptions(requested_type="bigsmiles"))

    assert result.parser_status == "rejected"
    # 图是本平台保守检查画的，渲染器描述必须这么说。
    assert result.renderer.startswith("BigSMILES 保守检查")
    assert "未通过" in result.renderer

    message = result.warning_message
    assert "\n" not in message and "\t" not in message
    assert len(message) < 600
    # 不回显整段输入，也不搬出 token dump。
    assert f"Parsing failed on '{value}'" not in message
    assert "Issue with token" in message or "Stochastic object" in message


def test_empty_brackets_and_missing_end_group_produce_actionable_hints(tmp_path: Path):
    result = render_structure(
        "CC{C1CCC([])(O)CC1}CC", tmp_path, RenderOptions(requested_type="bigsmiles")
    )
    assert "空方括号" in result.warning_message
    assert "端基" in result.warning_message


def test_valid_repeat_unit_without_end_group_still_gets_a_hint(tmp_path: Path):
    result = render_structure(
        "CC{C1CCC(O)CC1}CC", tmp_path, RenderOptions(requested_type="bigsmiles")
    )
    assert "端基" in result.warning_message
    assert "空方括号" not in result.warning_message


def test_static_disclaimers_do_not_drown_actionable_warnings(tmp_path: Path):
    """固定免责说明要和真实问题分开，否则用户会先看到一堆声明而忽略该怎么改。"""
    result = render_structure(BIGSMILES_EXAMPLE, tmp_path, RenderOptions())
    assert result.disclaimers
    assert any("不代表唯一完整聚合物分子" in item for item in result.disclaimers)
    assert any("不代表真实聚合物构象" in item for item in result.disclaimers)
    assert "不代表唯一完整聚合物分子" not in result.warning_message
    assert "不代表真实聚合物构象" not in result.warning_message


def test_summarize_library_error_strips_input_echo_and_collapses_lines():
    raw = (
        "Parsing failed on 'CC{C1}CC'.\n\tIssue with token 'X: {' (token: 3)\n\t\t"
        "Stochastic object starts must be followed an explict or implicit end group."
    )
    summarized = renderer_module._summarize_library_error(raw)
    assert "\n" not in summarized and "\t" not in summarized
    assert "CC{C1}CC" not in summarized
    assert summarized.startswith("Issue with token")


def test_summarize_library_error_truncates_by_length():
    summarized = renderer_module._summarize_library_error("x" * 5000)
    assert len(summarized) == renderer_module._LIBRARY_ERROR_LIMIT
    assert summarized.endswith("…")


def test_summarize_library_error_handles_empty_and_none():
    assert renderer_module._summarize_library_error(None) == ""
    assert renderer_module._summarize_library_error("  \n ") == ""


def test_bigsmiles_algorithmic_renderer_does_not_need_rdkit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(renderer_module, "Chem", None)
    monkeypatch.setattr(renderer_module, "Draw", None)
    monkeypatch.setattr(renderer_module, "rdkit", None)
    result = render_structure(BIGSMILES_EXAMPLE, tmp_path, RenderOptions())
    assert result.parse_status == "valid"
    assert result.draw_status == "rendered"
    assert result.renderer.endswith("BigSMILES 保守示意绘图器")
    assert (tmp_path / result.main_image_path).exists()



def test_embedded_stochastic_object_keeps_complete_rdkit_context(tmp_path: Path):
    value = "O=C(OCC(OC(=O)CCCCCCCC1OC1CCCCCC)COC(=O)CCCCCC1OC1CCCC1OC1CCC)CCCCCCC(O)C(OC(=O){[$1]CC=CC[$1],[$1]CC(C#N)[$1]}C(=O)O)CCCCCC"
    result = render_structure(value, tmp_path, RenderOptions(requested_type="bigsmiles", render_sample_chain=True))
    assert result.parse_status == "valid"
    assert result.draw_status == "rendered"
    assert "RDKit BigSMILES 片段布局" in result.renderer
    assert "完整外部骨架中的 BigSMILES 随机对象已用占位符表示" in result.warning_message
    assert "该片段无法由 RDKit 解析" not in result.warning_message
    assert result.main_image_path
    assert (tmp_path / result.main_image_path).exists()
    assert result.sample_image_path
    assert (tmp_path / result.sample_image_path).exists()

    specs = renderer_module._build_rdkit_bigsmiles_specs(value)
    assert [spec.kind for spec in specs] == ["context", "repeat", "repeat"]
    assert all(spec.molecule is not None for spec in specs)



def test_unsanitized_embedded_context_is_drawable(tmp_path: Path):
    value = "C1CO1CO{[>]c2ccc(C(C)(C)c3ccc(OCC(O)CO[<])cc3)cc2}c4ccc(C(C)(C)c5ccc(OCC6CO6)cc5)cc4"
    result = render_structure(value, tmp_path, RenderOptions(requested_type="bigsmiles", render_sample_chain=True))
    assert result.parse_status == "valid"
    assert result.draw_status == "rendered"
    assert result.main_image_path
    assert (tmp_path / result.main_image_path).exists()
    assert result.sample_image_path
    assert (tmp_path / result.sample_image_path).exists()

    specs = renderer_module._build_rdkit_bigsmiles_specs(value)
    assert [spec.kind for spec in specs] == ["context", "repeat"]
    assert specs[0].molecule is not None
    assert specs[0].parse_mode == "unsanitized_topology"
