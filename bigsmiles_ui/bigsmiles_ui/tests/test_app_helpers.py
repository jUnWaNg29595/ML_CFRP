from bigsmiles_ui.app import format_result_summary
from bigsmiles_ui.renderer import RenderResult


def test_format_result_summary_is_chinese_and_single_line():
    result = RenderResult(
        raw_string="CCO",
        detected_type="smiles",
        parse_status="valid",
        draw_status="rendered",
    )
    text = format_result_summary(result)
    assert "解析状态" in text
    assert "valid" in text
    assert "\n" not in text