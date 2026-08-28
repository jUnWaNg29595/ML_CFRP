from pathlib import Path

from UserPrediction import build_ai_confirmation_state, can_submit_ai_prediction, confirm_ai_field, fallback_input_mode, reject_ai_field


def test_ai_suggestions_cannot_submit_without_confirmation():
    state = build_ai_confirmation_state({'recognized_fields': {'resin_smiles': 'CCO'}}, confirmed_fields=set())
    assert can_submit_ai_prediction(state) is False
    state = confirm_ai_field(state, 'resin_smiles', 'CCN')
    assert state['confirmed_fields'] == {'resin_smiles'}
    assert can_submit_ai_prediction(state) is True
    assert fallback_input_mode('authentication_error') == 'manual'


def test_explicit_rejection_resolves_field_without_creating_value():
    state = build_ai_confirmation_state({'recognized_fields': {'phr': 42}})
    state = reject_ai_field(state, 'phr')
    assert can_submit_ai_prediction(state) is True
    assert state['fields']['phr']['value'] is None


def test_failed_portal_task_retry_requires_new_submission():
    source = (Path(__file__).resolve().parents[1] / "UserPrediction.py").read_text(encoding="utf-8")
    assert "manager.retry_task(task_id)" not in source
    assert "重新提交" in source
