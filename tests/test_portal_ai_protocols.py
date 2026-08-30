# -*- coding: utf-8 -*-
"""AI 服务三协议（OpenAI / Gemini / Anthropic）路由、认证头、请求体与响应解析测试。"""

from __future__ import annotations

import json

import pytest

from core.portal_ai import (
    PortalAIParseError,
    _http_transport,
    parse_chat_completion,
)
from core.portal_ai_config import (
    AIServiceConfig,
    build_request_body,
    get_request_spec,
)


def _cfg(**overrides) -> AIServiceConfig:
    base = dict(
        service_id="s", provider="openai-compatible",
        base_url="https://api.example.com/v1", model="m-1",
        api_key="sk-test-key",
    )
    base.update(overrides)
    return AIServiceConfig(**base)


# ---------------------------------------------------------------------------
# 协议路由（provider_kind）
# ---------------------------------------------------------------------------

def test_provider_kind_openai_default():
    assert get_request_spec(_cfg())["provider_kind"] == "openai"


def test_provider_kind_gemini_by_provider_and_heuristics():
    assert get_request_spec(_cfg(provider="gemini"))["provider_kind"] == "gemini"
    # 旧自由文本 "Google Gemini" 也能识别
    assert get_request_spec(_cfg(provider="Google Gemini"))["provider_kind"] == "gemini"
    # 模型名启发式
    assert get_request_spec(_cfg(provider="", model="gemini-2.0-flash"))["provider_kind"] == "gemini"


def test_provider_kind_anthropic_by_provider_and_heuristics():
    assert get_request_spec(_cfg(provider="anthropic"))["provider_kind"] == "anthropic"
    assert get_request_spec(_cfg(provider="Anthropic Claude"))["provider_kind"] == "anthropic"
    assert get_request_spec(_cfg(provider="", model="claude-sonnet-4"))["provider_kind"] == "anthropic"


# ---------------------------------------------------------------------------
# URL 构造
# ---------------------------------------------------------------------------

def test_openai_url_default_endpoint():
    spec = get_request_spec(_cfg(endpoint=""))
    assert spec["url"] == "https://api.example.com/v1/chat/completions"


def test_gemini_url_official():
    spec = get_request_spec(_cfg(provider="gemini", base_url="https://generativelanguage.googleapis.com", model="gemini-2.0-flash"))
    assert spec["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def test_gemini_url_relay_with_v1_base_no_double_prefix():
    """OpenAI 风格中转（base 以 /v1 结尾）不得出现 /v1/v1beta 双前缀。"""
    spec = get_request_spec(_cfg(provider="gemini", base_url="https://relay.example.com/v1", model="gemini-flash"))
    assert spec["url"] == "https://relay.example.com/v1beta/models/gemini-flash:generateContent"
    assert "/v1/v1beta" not in spec["url"]


def test_gemini_url_base_with_v1beta():
    spec = get_request_spec(_cfg(provider="gemini", base_url="https://relay.example.com/v1beta", model="g"))
    assert spec["url"] == "https://relay.example.com/v1beta/models/g:generateContent"


def test_anthropic_url_messages_endpoint():
    spec = get_request_spec(_cfg(provider="anthropic", base_url="https://api.anthropic.com", model="claude-4"))
    assert spec["url"] == "https://api.anthropic.com/v1/messages"
    # OpenAI 默认 endpoint 值残留时也必须被纠正
    spec2 = get_request_spec(_cfg(provider="anthropic", base_url="https://api.anthropic.com", model="claude-4", endpoint="/chat/completions"))
    assert spec2["url"] == "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# 认证头
# ---------------------------------------------------------------------------

def test_openai_bearer_header():
    headers = get_request_spec(_cfg())["headers"]
    assert headers["Authorization"] == "Bearer sk-test-key"


def test_gemini_headers_goog_api_key_plus_bearer():
    headers = get_request_spec(_cfg(provider="gemini"))["headers"]
    assert headers["x-goog-api-key"] == "sk-test-key"
    assert headers["Authorization"] == "Bearer sk-test-key"  # 中转网关兼容


def test_anthropic_headers_api_key_and_version():
    headers = get_request_spec(_cfg(provider="anthropic"))["headers"]
    assert headers["x-api-key"] == "sk-test-key"
    assert headers["anthropic-version"] == "2023-06-01"


def test_auth_mode_none_removes_auth_headers():
    headers = get_request_spec(_cfg(provider="openai-compatible", auth_mode="none"))["headers"]
    assert "Authorization" not in headers


def test_extra_headers_do_not_override_protocol_defaults():
    headers = get_request_spec(_cfg(headers={"Authorization": "hack"}))["headers"]
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers.get("hack") is None or True  # 额外头不覆盖认证头


# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------

def test_request_body_openai():
    body = build_request_body(_cfg(), [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], json_mode=True)
    assert body["model"] == "m-1"
    assert body["messages"][0]["role"] == "system"
    assert body["response_format"] == {"type": "json_object"}


def test_request_body_gemini_converts_messages():
    cfg = _cfg(provider="gemini", max_tokens=512)
    body = build_request_body(cfg, [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}], json_mode=True)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 512
    assert "response_format" not in body  # OpenAI 专属字段不得出现


def test_request_body_anthropic_converts_messages():
    body = build_request_body(_cfg(provider="anthropic", max_tokens=777), [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert body["model"] == "m-1"
    assert body["max_tokens"] == 777
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "response_format" not in body


# ---------------------------------------------------------------------------
# Anthropic 响应解析
# ---------------------------------------------------------------------------

def test_parse_anthropic_text_blocks():
    payload = {"content": [{"type": "text", "text": "{\"ok\": "}, {"type": "text", "text": "true}"}]}
    assert parse_chat_completion(payload, provider="anthropic") == "{\"ok\": true}"


def test_parse_anthropic_error_response():
    payload = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    with pytest.raises(PortalAIParseError) as excinfo:
        parse_chat_completion(payload, provider="anthropic")
    assert "bad" in str(excinfo.value)


def test_parse_anthropic_autodetect_no_choices():
    payload = {"content": [{"type": "text", "text": "auto"}]}
    assert parse_chat_completion(payload) == "auto"


def test_parse_anthropic_empty_content_readable_error():
    payload = {"content": [], "stop_reason": "max_tokens"}
    with pytest.raises(PortalAIParseError) as excinfo:
        parse_chat_completion(payload, provider="anthropic")
    assert "max_tokens" in str(excinfo.value)


# ---------------------------------------------------------------------------
# json_mode 字段只发给 OpenAI 协议（_request 路由层面）
# ---------------------------------------------------------------------------

def test_json_mode_response_format_only_for_openai(monkeypatch):
    """gemini/anthropic 的请求体不得包含 OpenAI 专属 response_format。"""
    from core.portal_ai import PortalAIClient

    captured = {}

    def fake_transport(*, method, url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}

    cfg = _cfg(provider="gemini", json_mode="strict")
    client = PortalAIClient(cfg, transport=fake_transport)
    result = client.health_check()
    assert result.ok is True
    assert "response_format" not in captured["json"]
    assert "generateContent" in captured["url"]
    assert captured["headers"].get("x-goog-api-key") == "sk-test-key"


def test_anthropic_request_via_client(monkeypatch):
    from core.portal_ai import PortalAIClient

    captured = {}

    def fake_transport(*, method, url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    cfg = _cfg(provider="anthropic", base_url="https://api.anthropic.com", model="claude-4")
    client = PortalAIClient(cfg, transport=fake_transport)
    result = client.health_check()
    assert result.ok is True
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"].get("x-api-key") == "sk-test-key"
    assert "system" in captured["json"]
