from UserPrediction import render_result, render_task_snapshot


def test_task_view_shows_id_progress_and_python_result_when_ai_is_unavailable():
    html = render_task_snapshot({'task_id': 'task-1', 'status': 'featuring', 'progress': 42, 'stage_label': '特征准备'})
    assert 'task-1' in html and '42%' in html and '特征准备' in html
    result_html = render_result({'prediction': 123.4, 'unit': '°C', 'explanation': {'status': 'unavailable'}})
    assert '123.4' in result_html and 'AI 解释暂不可用' in result_html
