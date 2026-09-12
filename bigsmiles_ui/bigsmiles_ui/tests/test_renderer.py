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
        assert image.width == 1000
        assert image.height == 700


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
