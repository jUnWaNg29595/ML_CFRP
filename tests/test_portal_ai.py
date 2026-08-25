from core.portal_ai import (
    PortalAIAuthenticationError,
    PortalAIClient,
    PortalAIError,
    PortalAIHTTPError,
    PortalAIMalformedResponseError,
    PortalAITransientError,
    parse_chat_completion,
    parse_json_or_markdown_json,
)
from core.portal_ai_config import AIServiceConfig


def _config(api_key="sk-secret"):
    return AIServiceConfig(
        service_id="test",
        provider="deepseek",
        base_url="https://api.example.com/v1",
        model="deepseek-chat",
        purpose="both",
        api_key=api_key,
    )


def test_fenced_json_and_transient_retry_are_supported():
    assert parse_json_or_markdown_json('```json\n{"ok": true}\n```') == {"ok": True}
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PortalAITransientError("timeout")
        return {"choices": [{"message": {"content": '{"recognized_fields": {}}'}}]}

    result = PortalAIClient(_config(), transport=transport, sleep=lambda _: None).parse_input(
        {"material_type": "epoxy_resin", "user_text": "树脂"}
    )
    assert result.recognized_fields == {}
    assert len(calls) == 2
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-secret"


def test_client_posts_to_chat_completions_with_bounded_payload():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"recognized_fields": {}}'}}]}

    PortalAIClient(_config(), transport=transport).parse_input({"user_text": "test"})
    call = calls[0]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["method"] == "POST"
    assert call["timeout"] == 30
    assert call["json"]["max_tokens"] == 2048
    assert call["json"]["messages"][0]["role"] == "system"


def test_authentication_errors_are_not_retried_or_leaked():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"status_code": 401, "body": "api_key=sk-secret"}

    try:
        PortalAIClient(_config(), transport=transport).parse_input({})
    except PortalAIAuthenticationError as exc:
        assert "sk-secret" not in str(exc)
    else:
        raise AssertionError("authentication error was not raised")
    assert len(calls) == 1


def test_transient_status_is_retried_twice_then_fails():
    calls = []
    sleeps = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"status_code": 503}

    try:
        PortalAIClient(_config(), transport=transport, sleep=sleeps.append).parse_input({})
    except PortalAITransientError:
        pass
    else:
        raise AssertionError("transient error was not raised")
    assert len(calls) == 3
    assert sleeps == [0.25, 0.25]


def test_transport_exception_does_not_retain_secret_context():
    def transport(**kwargs):
        raise RuntimeError(f"request failed: {kwargs!r}")

    try:
        PortalAIClient(_config(), transport=transport).parse_input({})
    except PortalAIError as exc:
        assert "sk-secret" not in str(exc)
        assert "Authorization" not in str(exc)
        assert exc.__context__ is None
    else:
        raise AssertionError("transport error was not raised")


def test_non_authentication_http_errors_keep_type_and_status_code():
    def transport(**kwargs):
        return {"status_code": 400, "body": "bad request"}

    try:
        PortalAIClient(_config(), transport=transport).parse_input({})
    except PortalAIHTTPError as exc:
        assert type(exc) is PortalAIHTTPError
        assert exc.status_code == 400
    else:
        raise AssertionError("HTTP error was not raised")


def test_falsey_transport_is_preserved_as_injected_seam():
    class FalseyTransport:
        def __bool__(self):
            return False

        def __call__(self, **kwargs):
            return {"choices": [{"message": {"content": '{"recognized_fields": {}}'}}]}

    transport = FalseyTransport()
    client = PortalAIClient(_config(), transport=transport)

    assert client.transport is transport


def test_malformed_json_and_prose_are_rejected():
    for value in ["here is the answer", "```json\n{bad}\n```", "[]", "1", '"text"', "null"]:
        try:
            parse_json_or_markdown_json(value)
        except PortalAIMalformedResponseError:
            pass
        else:
            raise AssertionError("malformed content was accepted")


def test_explain_result_is_schema_checked_and_context_is_sanitized():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"status":"available","summary":"ok","experiment_suggestions":["重复测试"],"warnings":[]}'
                    }
                }
            ]
        }

    result = PortalAIClient(_config(), transport=transport).explain_result(
        {"prediction": 12.3, "api_key": "sk-secret"}
    )
    assert result.status == "available"
    user_message = calls[0]["json"]["messages"][1]["content"]
    assert "sk-secret" not in user_message


def test_completion_and_json_parser_validate_shapes():
    assert parse_chat_completion({"choices": [{"message": {"content": "{}"}}]}) == "{}"
    for payload in [{}, {"choices": []}, {"choices": [{"message": {}}]}]:
        try:
            parse_chat_completion(payload)
        except PortalAIMalformedResponseError:
            pass
        else:
            raise AssertionError("invalid completion was accepted")
