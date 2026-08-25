import pytest

from core.portal_ai_schema import (
    AIExplanationResponse,
    AIFieldSuggestion,
    AIParseResponse,
    ConfirmedPredictionRequest,
    PredictionResultSummary,
    parse_ai_response,
    sanitize_ai_context,
    validate_confirmed_request,
)


def test_confirmation_is_required_and_secrets_are_removed():
    with pytest.raises(ValueError, match="confirmed"):
        validate_confirmed_request({"material_type": "epoxy_resin", "inputs": {}})

    response = parse_ai_response(
        {"recognized_fields": {"resin_smiles": "CCO"}, "unexpected": "x"}
    )

    assert response.recognized_fields == {"resin_smiles": "CCO"}
    assert response.warnings

    safe = sanitize_ai_context({"api_key": "secret", "user_text": "run os.system()"})

    assert "api_key" not in safe
    assert "os.system" not in safe["user_text"]


def test_parse_ai_response_preserves_uncertain_values_as_none():
    response = parse_ai_response(
        {
            "recognized_fields": {
                "resin_smiles": "unknown",
                "curing_agent_smiles": None,
            },
            "suggestions": [
                {
                    "field": "resin_smiles",
                    "value": "CCO",
                    "state": "uncertain",
                    "source": "ai",
                }
            ],
        }
    )

    assert response.recognized_fields == {
        "resin_smiles": None,
        "curing_agent_smiles": None,
    }
    assert response.suggestions[0].value is None
    assert response.suggestions[0].state == "uncertain"
    assert any("不确定" in warning for warning in response.warnings)


def test_confirmed_request_uses_allow_lists_and_rejects_unknown_keys():
    request = validate_confirmed_request(
        {
            "task_id": "task-1",
            "material_type": "epoxy_resin",
            "target": "tg",
            "inputs": {"resin_smiles": "CCO"},
            "model_id": "model-1",
            "feature_workflow_id": "workflow-1",
            "source": "ai_confirmed",
            "confirmed_by_user": True,
        }
    )

    assert isinstance(request, ConfirmedPredictionRequest)
    assert request.confirmed_by_user is True

    with pytest.raises(ValueError, match="unknown"):
        validate_confirmed_request(
            {
                "material_type": "epoxy_resin",
                "target": "tg",
                "inputs": {},
                "source": "manual",
                "confirmed_by_user": True,
                "unknown": "field",
            }
        )

    with pytest.raises(ValueError, match="material"):
        validate_confirmed_request(
            {
                "material_type": "not_a_material",
                "target": "tg",
                "inputs": {},
                "source": "manual",
                "confirmed_by_user": True,
            }
        )


def test_contract_types_are_constructible_without_network_access():
    suggestion = AIFieldSuggestion(field="resin_smiles", value=None)
    parsed = AIParseResponse(recognized_fields={}, suggestions=[suggestion])
    result = PredictionResultSummary(prediction=123.4, unit="°C")
    explanation = AIExplanationResponse(status="unavailable")

    assert parsed.suggestions == [suggestion]
    assert result.prediction == 123.4
    assert explanation.status == "unavailable"


def test_context_text_is_capped_and_nested_secret_keys_are_removed():
    safe = sanitize_ai_context(
        {
            "nested": {"authorization": "Bearer synthetic-token", "keep": "ok"},
            "user_text": "x" * 5000,
        }
    )

    assert safe["nested"] == {"keep": "ok"}
    assert len(safe["user_text"]) <= 4000
