# -*- coding: utf-8 -*-
"""AI 服务协议扩展测试：Gemini 解析、response_json_path、别名归一、401 分类、get_request_spec。"""

from __future__ import annotations

import json
import urllib.error
from unittest import mock

import pytest

from core.portal_ai import (
    PortalAIAuthError,
    PortalAIParseError,
    _http_transport,
    parse_chat_completion,
    parse_json_or_markdown_json,
)
from core.portal_ai_config import AIServiceConfig, get_request_spec
from core.portal_ai_schema import (
    normalize_feature_mapping_aliases,
    parse_feature_mapping_response,
)


# ---------------------------------------------------------------------------
# Gemini 响应解析
# ---------------------------------------------------------------------------

def test_gemini_response_text_extraction():
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "hello "}, {"text": "world"}]}, "finishReason": "STOP"}
        ]
    }
    assert parse_chat_completion(payload, provider="gemini") == "hello world"


def test_gemini_response_json_content():
    payload = {"candidates": [{"content": {"parts": [{"text": '{"tg_c": 150}'}]}}]}
    text = parse_chat_completion(payload, provider="gemini")
    assert json.loads(text) == {"tg_c": 150}


def test_gemini_response_missing_candidates_is_readable_error():
    with pytest.raises(PortalAIParseError) as excinfo:
        parse_chat_completion({"promptFeedback": {"blockReason": "SAFETY"}}, provider="gemini")
    assert "candidates" in str(excinfo.value)


def test_gemini_autodetect_when_no_choices():
    payload = {"candidates": [{"content": {"parts": [{"text": "auto"}]}}]}
    assert parse_chat_completion(payload) == "auto"


# ---------------------------------------------------------------------------
# response_json_path 提取
# ---------------------------------------------------------------------------

def test_response_json_path_choices_content():
    payload = {"choices": [{"message": {"content": '{"a": 1}'}}]}
    text = parse_chat_completion(payload, response_json_path="choices.0.message.content")
    assert json.loads(text) == {"a": 1}


def test_response_json_path_bracket_syntax():
    payload = {"a": {"b": [{"c": "value"}]}}
    text = parse_chat_completion(payload, response_json_path="a['b'][0].c")
    assert text == "value"


def test_response_json_path_missing_raises_readable_error():
    payload = {"choices": []}
    with pytest.raises(PortalAIParseError) as excinfo:
        parse_chat_completion(payload, response_json_path="choices.0.message.content")
    assert "choices.0.message.content" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 字段别名归一（验收单第 7 项）
# ---------------------------------------------------------------------------

def test_alias_normalization_six_groups():
    entry = {
        "source_type": "manual_input",
        "raw_column": "t",
        "source_fields": ["t"],
        "semantic_feature_id": "f9",
        "units": "°C",
        "accepted_aliases": ["固化温度"],
    }
    normalized = normalize_feature_mapping_aliases(entry)
    assert normalized["source_role"] == "manual_input"
    assert normalized["raw_columns"] == "t"
    assert normalized["source_field"] == ["t"]
    assert normalized["feature_id"] == "f9"
    assert normalized["unit"] == "°C"
    assert normalized["aliases"] == ["固化温度"]
    for alias in ("source_type", "raw_column", "source_fields", "semantic_feature_id", "units", "accepted_aliases"):
        assert alias not in normalized


def test_parse_feature_mapping_response_accepts_aliases():
    response = parse_feature_mapping_response({
        "suggestions": [{
            "semantic_feature_id": "f1",
            "raw_column": "cure_temp",
            "source_type": "manual_input",
            "units": "C",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "工艺字段",
        }],
        "conflicts": [],
    })
    sugg = response["suggestions"][0]
    assert sugg["feature_id"] == "f1"
    assert sugg["raw_columns"] == ["cure_temp"]
    assert sugg["source_role"] == "manual_input"
    assert sugg["unit"] == "C"


def test_parse_feature_mapping_response_raw_columns_comma_string():
    response = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "f2",
            "raw_columns": "a, b ,c",
            "source_role": "manual_input",
            "status": "pending_review",
            "confidence": 0.5,
            "rationale_zh": "ok",
        }],
        "conflicts": [],
    })
    assert response["suggestions"][0]["raw_columns"] == ["a", "b", "c"]


def test_invalid_source_role_does_not_crash_downgrades_to_conflict():
    response = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "f3",
            "raw_columns": ["x"],
            "source_role": "weird_role",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "?",
        }],
        "conflicts": [],
    })
    sugg = response["suggestions"][0]
    assert sugg["source_role"] == "unknown"
    assert sugg["status"] == "conflict"
    assert sugg.get("source_role_raw") == "weird_role"


# ---------------------------------------------------------------------------
# JSON 解析容错（验收单第 9、10 项）
# ---------------------------------------------------------------------------

def test_markdown_json_fenced_parsing():
    text = "说明如下\n```json\n{\"ok\": true}\n```\n以上是结果"
    assert parse_json_or_markdown_json(text) == {"ok": True}


def test_invalid_json_raises_readable_chinese_error():
    with pytest.raises(PortalAIParseError) as excinfo:
        parse_json_or_markdown_json("这不是 JSON")
    assert "JSON" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP 401 → PortalAIAuthError（验收单第 11 项）
# ---------------------------------------------------------------------------

def _http_error(status_code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=status_code,
        msg="Unauthorized",
        hdrs=None,
        fp=None,
    )


def test_http_401_maps_to_auth_error_with_status_code():
    error = _http_error(401, b'{"error": {"message": "bad key"}}')
    error.read = lambda: b'{"error": {"message": "bad key"}}'  # type: ignore[method-assign]
    with mock.patch.object(urllib.request, "OpenerDirector") as opener_cls:
        opener_cls.return_value.open.side_effect = error
        with pytest.raises(PortalAIAuthError) as excinfo:
            _http_transport(
                method="POST",
                url="https://example.invalid/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={"model": "m"},
                timeout=5,
            )
    assert excinfo.value.status_code == 401
    # 摘要脱敏：不应包含完整密钥形态
    excerpt = str(excinfo.value.raw_excerpt or "")
    assert "sk-" not in excerpt


# ---------------------------------------------------------------------------
# get_request_spec：协议分支（验收单第 12 项的服务配置层）
# ---------------------------------------------------------------------------

def _config(**overrides) -> AIServiceConfig:
    base = dict(
        service_id="s1", label="s1", provider="openai-compatible",
        base_url="https://api.example.com/v1", model="gpt-test",
        api_key="sk-test-1234",
    )
    base.update(overrides)
    return AIServiceConfig(**base)


def test_get_request_spec_openai_default():
    spec = get_request_spec(_config())
    assert spec["body_kind"] == "openai"
    assert spec["url"].endswith("/v1/chat/completions")
    assert spec["headers"]["Authorization"] == "Bearer sk-test-1234"


def test_get_request_spec_gemini_generates_generate_content():
    spec = get_request_spec(_config(provider="gemini", base_url="https://generativelanguage.example.com"))
    assert spec["body_kind"] == "gemini"
    assert "models/gpt-test:generateContent" in spec["url"]
    spec2 = get_request_spec(_config(provider="gemini", base_url="https://example.com/v1beta"))
    assert "models/gpt-test:generateContent" in spec2["url"]
    assert "/v1beta//models" not in spec2["url"]


def test_get_request_spec_custom_template():
    spec = get_request_spec(_config(provider="custom", request_template={"model": "{model}"}))
    assert spec["body_kind"] == "template"


def test_get_request_spec_api_key_header_auth_mode():
    spec = get_request_spec(_config(auth_mode="api_key_header"))
    assert spec["headers"].get("X-API-Key") == "sk-test-1234"
    assert "Authorization" not in spec["headers"]


def test_get_request_spec_extra_headers_and_none_auth():
    spec = get_request_spec(_config(auth_mode="none", headers={"X-Custom": "v"}))
    assert "Authorization" not in spec["headers"]
    assert spec["headers"]["X-Custom"] == "v"


def test_service_configs_are_independent_per_service_id():
    """验收单第 12 项：不同 service_id 的配置内容互不影响。"""
    from core.portal_ai_config import _service_mapping
    a = _service_mapping({"service_id": "a", "provider": "gemini", "model": "gemini-1", "base_url": "https://a.example.com"}, index=0)
    b = _service_mapping({"service_id": "b", "provider": "openai-compatible", "model": "gpt-1", "base_url": "https://b.example.com"}, index=1)
    assert a["provider"] != b["provider"]
    assert a["model"] != b["model"]
