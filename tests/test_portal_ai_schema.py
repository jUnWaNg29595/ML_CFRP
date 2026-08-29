import pytest

from core.portal_ai_schema import (
    AIExplanationResponse,
    AIFieldSuggestion,
    AIParseResponse,
    ConfirmedPredictionRequest,
    PredictionResultSummary,
    parse_ai_response,
    parse_feature_mapping_response,
    sanitize_ai_context,
    validate_confirmed_request,
)


def test_feature_mapping_parser_requires_explicit_status():
    with pytest.raises(ValueError, match="status"):
        parse_feature_mapping_response(
            {
                "suggestions": [
                    {
                        "feature_id": "pressure",
                        "raw_columns": ["p_raw"],
                        "source_role": "manual_input",
                    }
                ],
                "conflicts": [],
            }
        )


@pytest.mark.parametrize("status", ["pending_review", "conflict", "unknown"])
def test_feature_mapping_parser_preserves_explicit_status(status):
    response = parse_feature_mapping_response(
        {
            "suggestions": [
                {
                    "feature_id": "pressure",
                    "raw_columns": ["p_raw"],
                    "source_role": "manual_input",
                    "status": status,
                }
            ],
            "conflicts": [],
        }
    )

    assert response["suggestions"][0]["status"] == status


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


def test_confirmed_request_rejects_nested_callable_code_and_unsupported_values():
    base_request = {
        "material_type": "epoxy_resin",
        "target": "tg",
        "source": "manual",
        "confirmed_by_user": True,
    }

    with pytest.raises(ValueError, match="inputs"):
        validate_confirmed_request(
            {
                **base_request,
                "inputs": {"nested": {"callback": lambda: None}},
            }
        )
    with pytest.raises(ValueError, match="inputs"):
        validate_confirmed_request(
            {
                **base_request,
                "inputs": {"nested": [compile("x = 1", "<test>", "exec")]},
            }
        )
    with pytest.raises(ValueError, match="inputs"):
        validate_confirmed_request(
            {
                **base_request,
                "inputs": {"nested": {"unsupported": object()}},
            }
        )


@pytest.mark.parametrize("identifier", ["model_id", "feature_workflow_id"])
def test_confirmed_request_rejects_overlong_identifiers(identifier):
    request = {
        "material_type": "epoxy_resin",
        "target": "tg",
        "inputs": {},
        "source": "manual",
        "confirmed_by_user": True,
        identifier: "x" * 201,
    }

    with pytest.raises(ValueError, match=identifier):
        validate_confirmed_request(request)


def test_context_redacts_credentials_embedded_in_free_text():
    safe = sanitize_ai_context(
        {
            "user_text": "key=synthetic-key password=synthetic-password secret=synthetic-secret token=synthetic-token"
        }
    )

    assert "synthetic-key" not in safe["user_text"]
    assert "synthetic-password" not in safe["user_text"]
    assert "synthetic-secret" not in safe["user_text"]
    assert "synthetic-token" not in safe["user_text"]


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [("state", []), ("source", {})],
)
def test_parse_suggestion_rejects_non_string_state_and_source(field, malformed_value):
    with pytest.raises(ValueError, match=field):
        parse_ai_response(
            {
                "suggestions": [
                    {
                        "field": "resin_smiles",
                        field: malformed_value,
                    }
                ]
            }
        )


# ============================================================
# source_role normalization and safe downgrade tests
# ============================================================

from core.portal_ai_schema import normalize_feature_source_role


def test_normalize_source_role_canonical_values_pass_through():
    assert normalize_feature_source_role("manual_input") == "manual_input"
    assert normalize_feature_source_role("molecular_workflow") == "molecular_workflow"
    assert normalize_feature_source_role("derived_workflow") == "derived_workflow"
    assert normalize_feature_source_role("unknown") == "unknown"


def test_normalize_source_role_limited_aliases():
    assert normalize_feature_source_role("manual") == "manual_input"
    assert normalize_feature_source_role("measured") == "manual_input"
    assert normalize_feature_source_role("experimental") == "manual_input"
    assert normalize_feature_source_role("人工输入") == "manual_input"
    assert normalize_feature_source_role("molecular") == "molecular_workflow"
    assert normalize_feature_source_role("descriptor") == "molecular_workflow"
    assert normalize_feature_source_role("分子特征") == "molecular_workflow"
    assert normalize_feature_source_role("derived") == "derived_workflow"
    assert normalize_feature_source_role("computed") == "derived_workflow"
    assert normalize_feature_source_role("calculated") == "derived_workflow"
    assert normalize_feature_source_role("派生") == "derived_workflow"
    assert normalize_feature_source_role("uncertain") == "unknown"
    assert normalize_feature_source_role("不确定") == "unknown"


def test_normalize_source_role_case_and_whitespace_insensitive():
    assert normalize_feature_source_role("  Manual_Input  ") == "manual_input"
    assert normalize_feature_source_role("MANUAL-INPUT") == "manual_input"
    assert normalize_feature_source_role("molecular_workflow ") == "molecular_workflow"
    assert normalize_feature_source_role(" Derived Workflow ") == "derived_workflow"


def test_normalize_source_role_never_guesses_unknown_strings():
    assert normalize_feature_source_role("completely_made_up_role") is None
    assert normalize_feature_source_role("target") is None
    assert normalize_feature_source_role("metadata") is None
    assert normalize_feature_source_role("这是一段很长的中文句子描述来源") is None
    assert normalize_feature_source_role("") is None
    assert normalize_feature_source_role(None) is None
    assert normalize_feature_source_role(123) is None


def test_parse_feature_mapping_downgrades_unknown_source_role_to_conflict():
    result = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "cfrp.tg.pressure",
            "raw_columns": ["压强"],
            "source_role": "fabricated_role_xyz",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "列名匹配",
        }],
        "conflicts": [],
    })
    sugg = result["suggestions"][0]
    assert sugg["source_role"] == "unknown"
    assert sugg["status"] == "conflict"
    assert sugg["source_role_raw"] == "fabricated_role_xyz"
    assert sugg["source_role_downgraded"] is True
    assert "需要人工审核" in sugg["rationale_zh"]
    assert any("fabricated_role_xyz" in item for item in result["conflicts"])


def test_parse_feature_mapping_target_and_metadata_become_unknown_conflict():
    result = parse_feature_mapping_response({
        "suggestions": [
            {
                "feature_id": "a",
                "raw_columns": ["col_a"],
                "source_role": "target",
                "status": "pending_review",
                "confidence": 0.9,
                "rationale_zh": "x",
            },
            {
                "feature_id": "b",
                "raw_columns": ["col_b"],
                "source_role": "metadata",
                "status": "pending_review",
                "confidence": 0.9,
                "rationale_zh": "y",
            },
        ],
        "conflicts": [],
    })
    for sugg in result["suggestions"]:
        assert sugg["source_role"] == "unknown"
        assert sugg["status"] == "conflict"


def test_parse_feature_mapping_known_alias_is_normalized_not_downgraded():
    result = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "cfrp.tg.pressure",
            "raw_columns": ["压强"],
            "source_role": "computed",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "列名匹配",
        }],
        "conflicts": [],
    })
    sugg = result["suggestions"][0]
    assert sugg["source_role"] == "derived_workflow"
    assert sugg["status"] == "pending_review"
    assert "source_role_raw" not in sugg


def test_parse_feature_mapping_empty_source_role_still_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="source_role"):
        parse_feature_mapping_response({
            "suggestions": [{
                "feature_id": "a",
                "raw_columns": ["col"],
                "source_role": "",
                "status": "pending_review",
                "confidence": 0.9,
                "rationale_zh": "x",
            }],
            "conflicts": [],
        })
