from core import portal_ai
from core.portal_ai import (
    PortalAIAuthenticationError,
    PortalAIClient,
    PortalAIError,
    PortalAIHTTPError,
    PortalAIMalformedResponseError,
    PortalAITransientError,
    list_models,
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


def test_direct_opener_ignores_process_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:10808")
    opener = portal_ai._direct_opener()
    handlers = [handler for handler in opener.handlers if isinstance(handler, portal_ai.urllib.request.ProxyHandler)]

    assert not handlers or handlers[0].proxies == {}


def test_list_models_parses_openai_compatible_response(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data": [{"id": "gpt-z"}, {"id": "gpt-a", "owned_by": "team"}, {"id": ""}]}'

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(portal_ai, "_direct_opener", lambda: Opener())
    models = list_models(_config())

    assert models == [{"id": "gpt-a", "owned_by": "team"}, {"id": "gpt-z", "owned_by": ""}]


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


def test_transient_gateway_error_exposes_status_without_secret():
    def transport(**kwargs):
        return {"status_code": 502, "body": "upstream unavailable"}

    try:
        PortalAIClient(_config(), transport=transport, sleep=lambda _: None).parse_input({})
    except PortalAITransientError as exc:
        assert exc.status_code == 502
        assert "HTTP 502" in str(exc)
        assert "上游网关" in str(exc)
        assert "sk-secret" not in str(exc)
    else:
        raise AssertionError("gateway error was not raised")


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


def test_parse_json_or_markdown_json_extended_compatibility():
    # 1. Plain JSON
    assert parse_json_or_markdown_json('{"status": "ok"}') == {"status": "ok"}

    # 2. Markdown fenced JSON
    assert parse_json_or_markdown_json('```json\n{"val": 123}\n```') == {"val": 123}

    # 3. With surrounding prose
    prose = "Here is the parsed result:\n```json\n{\"recognized_fields\": {\"temp\": 180}}\n```\nHope this helps!"
    assert parse_json_or_markdown_json(prose) == {"recognized_fields": {"temp": 180}}

    # 4. With <think> tags from reasoning models
    think_text = "<think>Let me reason about the user inputs...</think>\n{\"suggestions\": []}"
    assert parse_json_or_markdown_json(think_text) == {"suggestions": []}

    # 5. Non-fenced JSON embedded in text
    plain_embedded = "Output data: {\"ok\": true, \"confidence\": 0.95} - end"
    assert parse_json_or_markdown_json(plain_embedded) == {"ok": True, "confidence": 0.95}


def test_parse_chat_completion_supports_content_parts_and_rejects_empty():
    # Parts list
    payload_parts = {
        "choices": [{
            "message": {
                "content": [
                    {"type": "text", "text": "{\"recog"},
                    {"type": "output_text", "text": "nized\": true}"},
                ]
            }
        }]
    }
    assert parse_chat_completion(payload_parts) == '{"recognized": true}'

    # Empty content with reasoning only
    payload_reasoning = {
        "choices": [{
            "message": {
                "content": "",
                "reasoning_content": "thinking only...",
            }
        }]
    }
    try:
        parse_chat_completion(payload_reasoning)
    except PortalAIMalformedResponseError as exc:
        assert "reasoning_content" in str(exc)
    else:
        raise AssertionError("reasoning-only payload should be rejected")


def test_response_format_json_mode_downgrade_on_http_400():
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # First call with response_format fails with 400
            return {
                "status_code": 400,
                "body": "Unrecognized parameter: response_format is not supported by model",
            }
        # Second call succeeds
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client = PortalAIClient(_config(), transport=transport, sleep=lambda _: None)
    res = client.health_check()
    assert res.ok is True
    assert len(calls) == 2
    assert "response_format" in calls[0]["json"]
    assert "response_format" not in calls[1]["json"]


def test_health_check_returns_clean_diagnostics_on_natural_language():
    def transport(**kwargs):
        return {"choices": [{"message": {"content": "Sure, here is your prediction for epoxy: it is high"}}]}

    client = PortalAIClient(_config(), transport=transport)
    res = client.health_check()
    assert res.ok is False
    assert res.stage == "模型返回自然语言"
    assert "JSON 契约" in res.message or "自然语言" in res.message or "JSON" in res.message


def test_models_endpoint_two_step_testing_diagnostics():
    from core.portal_ai import test_models_endpoint

    monkeypatch_config = _config()
    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"data": [{"id": "deepseek-chat"}]}'

    class Opener:
        def open(self, request, timeout=None):
            return Resp()

    from core import portal_ai
    orig_opener = portal_ai._direct_opener
    portal_ai._direct_opener = lambda: Opener()
    try:
        res = test_models_endpoint(monkeypatch_config)
        assert res["ok"] is True
        assert res["model_present"] is True
        assert "deepseek-chat" in res["message"] or res["model_present"]
    finally:
        portal_ai._direct_opener = orig_opener


def test_url_normalization_and_endpoint_deduplication():
    from core.portal_ai_config import build_request_url, normalize_endpoint_path

    assert normalize_endpoint_path("chat/completions") == "/chat/completions"
    assert normalize_endpoint_path("/v1/chat/completions/") == "/v1/chat/completions"
    assert build_request_url("https://api.example.com/v1", "/chat/completions") == "https://api.example.com/v1/chat/completions"
    assert build_request_url("https://api.example.com/v1/chat/completions", "/chat/completions") == "https://api.example.com/v1/chat/completions"
    assert build_request_url("https://api.example.com/v1/v1", "/chat/completions") == "https://api.example.com/v1/chat/completions"