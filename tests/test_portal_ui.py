from core.portal_ui import render_stage_timeline, render_status_badge, svg_icon


def test_icons_are_linear_svg_and_timeline_has_trusted_stages():
    markup = svg_icon('molecule', size=20, label='分子')
    assert markup.startswith('<svg') and 'stroke=' in markup and '🧪' not in markup
    html = render_stage_timeline('predicting', 60)
    assert all(x in html for x in ('输入确认', '结构校验', '特征准备', '模型计算', '结果解释'))


def test_status_badge_uses_safe_text():
    html = render_status_badge('completed')
    assert '已完成' in html and 'success' in html
