import pandas as pd
import pytest


def test_rejected_ai_candidate_does_not_write_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = {"feature_id": "cfrp.tg.pressure", "raw_columns": ["压强"], "source_role": "manual_input", "confidence": 0.96, "rationale_zh": "列名接近但单位未确认"}
    updated = apply_feature_review_decision(manifest, suggestion, "reject", "local-user")
    assert updated["status"] == "draft"
    assert updated["feature_bindings"] == []


def test_accept_action_writes_approved_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    updated = apply_feature_review_decision({"status": "draft", "feature_bindings": []}, {"feature_id": "pressure", "raw_columns": ["pressure_raw"], "source_role": "manual_input", "confidence": 0.8, "rationale_zh": "人工确认"}, "accept", "local-user")
    assert updated["feature_bindings"][0]["feature_id"] == "pressure"
    assert updated["approval"]["approved_by"] == "local-user"


def test_review_context_is_feature_only_and_bounded():
    from core.feature_mapping_review import build_feature_review_context

    frame = pd.DataFrame({"pressure_raw": [1.0, 2.0], "target": [100, 110], "tg_c": [100, 110]})
    registry = {"model_profiles": {"p": {"feature_ids": ["x"], "target_col": "tg_c", "target": "tg"}}, "features": [{"feature_id": "x", "name": "pressure", "source_type": "manual_input", "label": "压力", "review_secret": "must not leave"}]}
    context = build_feature_review_context(frame, registry, "p")
    assert "pressure_raw" in context["raw_columns"]
    assert "target" not in context["raw_columns"]
    assert "tg_c" not in context["raw_columns"]
    assert all("target" not in row and "tg_c" not in row for row in context["sample_rows"])
    assert context["candidate_features"] == [{"feature_id": "x", "name": "pressure", "source_type": "manual_input", "label": "压力"}]
    assert "metrics" not in context
    assert len(context["sample_rows"]) <= 3


def test_review_client_response_is_structured(monkeypatch):
    from core.feature_mapping_review import request_feature_mapping_review

    class Client:
        def review_feature_mapping(self, context):
            return {"suggestions": [{"feature_id": "pressure", "raw_columns": ["p_raw"], "source_role": "manual_input", "confidence": 0.9, "rationale_zh": "单位一致"}], "conflicts": [], "rationale_zh": "仅供审核", "confidence": 0.9}

    result = request_feature_mapping_review(Client(), {"raw_columns": ["p_raw"]})
    assert result["suggestions"][0]["feature_id"] == "pressure"


def test_feature_review_rejects_unbounded_source_role_and_missing_evidence():
    from core.portal_ai_schema import parse_feature_mapping_response

    with pytest.raises(ValueError):
        parse_feature_mapping_response({"suggestions": [{
            "feature_id": "pressure", "raw_columns": ["p_raw"], "source_role": "invented_role",
            "status": "approved", "unit": 123, "rationale_zh": "",
        }], "conflicts": []})
