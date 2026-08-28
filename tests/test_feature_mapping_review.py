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

    updated = apply_feature_review_decision({"status": "draft", "feature_bindings": []}, {"feature_id": "pressure", "raw_columns": ["pressure_raw"], "source_role": "manual_input", "status": "pending_review", "confidence": 0.8, "rationale_zh": "人工确认"}, "accept", "local-user")
    assert updated["feature_bindings"][0]["feature_id"] == "pressure"
    assert updated["feature_bindings"][0]["review_status"] == "approved"
    assert updated["approval"]["approved_by"] == "local-user"


def test_edit_accept_pending_review_without_edited_status_writes_approved_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {
        "feature_id": "pressure",
        "raw_columns": ["pressure_raw"],
        "source_role": "manual_input",
        "status": "pending_review",
    }
    updated = apply_feature_review_decision(
        {"status": "draft", "feature_bindings": []},
        suggestion,
        "edit_accept",
        "local-user",
        edited={"raw_columns": ["pressure_corrected"]},
    )
    assert updated["feature_bindings"][0]["raw_columns"] == ["pressure_corrected"]
    assert updated["feature_bindings"][0]["review_status"] == "approved"


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


def test_empty_profile_does_not_send_full_registry_and_excludes_target():
    from core.feature_mapping_review import build_feature_review_context
    frame = pd.DataFrame({"pressure_raw": [1.0], "tg_c": [100.0]})
    registry = {"model_profiles": {"p": {"feature_ids": [], "target_col": "tg_c", "status": "approved"}},
                "features": [{"feature_id": "pressure", "name": "pressure", "source_type": "manual_input", "status": "approved"},
                             {"feature_id": "unrelated", "name": "secret", "source_type": "manual_input", "status": "approved"}]}
    context = build_feature_review_context(frame, registry, "p")
    assert [item["feature_id"] for item in context["candidate_features"]] == ["pressure"]


def test_apply_review_rejects_invalid_feature_and_source_role_and_requires_edit_payload():
    from core.feature_mapping_review import apply_feature_review_decision
    manifest = {"status": "draft", "feature_bindings": []}
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "other", "raw_columns": ["x"], "source_role": "manual_input"}, "accept", "u", registry={"features": [{"feature_id": "known"}]})
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "target"}, "accept", "u", registry={"features": [{"feature_id": "known"}]})
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input"}, "edit_accept", "u", registry={"features": [{"feature_id": "known"}]})


@pytest.mark.parametrize("status", ["unknown", "conflict", "approved", "draft"])
@pytest.mark.parametrize("action", ["accept", "edit_accept"])
def test_apply_review_rejects_non_pending_suggestion_status(status, action):
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": status}
    edited = {"raw_columns": ["x"]} if action == "edit_accept" else None
    with pytest.raises(ValueError, match="pending_review|状态"):
        apply_feature_review_decision(manifest, suggestion, action, "u", registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]}, edited=edited)
    assert manifest["feature_bindings"] == []


def test_apply_review_rejects_missing_suggestion_status_for_accept_and_edit_accept():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input"}
    manifest = {"status": "draft", "feature_bindings": []}
    registry = {"features": [{"feature_id": "known", "source_type": "manual_input"}]}
    for action, edited in (("accept", None), ("edit_accept", {"raw_columns": ["x"]})):
        with pytest.raises(ValueError, match="pending_review|状态"):
            apply_feature_review_decision(manifest, suggestion, action, "u", registry=registry, edited=edited)
    assert manifest["feature_bindings"] == []


def test_apply_review_requires_explicit_source_role_and_registry_alignment():
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    with pytest.raises(ValueError, match="source_role|来源"):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "status": "pending_review"}, "accept", "u", registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]})
    with pytest.raises(ValueError, match="source|来源"):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"}, "accept", "u", registry={"features": [{"feature_id": "known", "source_type": "molecular_workflow"}]})


def test_apply_review_rejects_nonapproved_edited_status():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"}
    with pytest.raises(ValueError, match="pending_review|状态"):
        apply_feature_review_decision(
            {"status": "draft", "feature_bindings": []}, suggestion, "edit_accept", "u",
            registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]},
            edited={"status": "conflict"},
        )
