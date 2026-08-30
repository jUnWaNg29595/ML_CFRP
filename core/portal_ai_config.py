"""Secure local persistence and redaction for portal AI service configuration."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit


CONFIG_DIRECTORY = "prediction_portal"
CONFIG_FILENAME = "ai_config.json"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.2
DEFAULT_PURPOSE = "both"
SUPPORTED_PURPOSES = {"input_parsing", "result_explanation", "both"}
DEFAULT_ENDPOINT = "/chat/completions"
SUPPORTED_JSON_MODES = {"auto", "strict", "off"}
DEFAULT_JSON_MODE = "auto"
SUPPORTED_NETWORK_MODES = {"auto", "direct", "proxy"}
DEFAULT_NETWORK_MODE = "auto"
SUPPORTED_RESPONSE_PARSE_MODES = {"strict", "compatible"}
DEFAULT_RESPONSE_PARSE_MODE = "compatible"
SUPPORTED_AUTH_MODES = {"bearer", "api_key_header", "none"}
DEFAULT_AUTH_MODE = "bearer"
REQUEST_JSON_PATH_PATTERN = re.compile(r"[A-Za-z0-9_.\[\]]*")
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 300
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 200_000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_RETRY_ATTEMPTS = 0
MAX_RETRY_ATTEMPTS = 5
MAX_TEXT_LENGTH = 200
PRIVATE_FILE_MODE = 0o600

_SECRET_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:api[_-]?key|access[_-]?token|authorization|bearer|"
    r"client[_-]?secret|credential|password|private[_-]?key|secret|token|key)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_HOST_LABEL_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class AIServiceConfig:
    service_id: str
    label: str = ""
    provider: str = ""
    base_url: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    model: str = ""
    purpose: str = DEFAULT_PURPOSE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    enabled: bool = True
    api_key: str | None = None
    json_mode: str = DEFAULT_JSON_MODE
    network_mode: str = DEFAULT_NETWORK_MODE
    response_parse_mode: str = DEFAULT_RESPONSE_PARSE_MODE
    allow_json_mode_downgrade: bool = True
    max_retries: int = 2
    auth_mode: str = DEFAULT_AUTH_MODE
    headers: dict = field(default_factory=dict)
    request_template: dict | None = None
    response_json_path: str | None = None
    anthropic_version: str = "2023-06-01"


def normalize_endpoint_path(endpoint: str) -> str:
    """Normalize an endpoint path: ensure leading slash, no trailing slash, no duplication."""
    value = str(endpoint or "").strip()
    if not value:
        return DEFAULT_ENDPOINT
    if not value.startswith("/"):
        value = "/" + value
    # Collapse repeated slashes
    while "//" in value:
        value = value.replace("//", "/")
    value = value.rstrip("/")
    if not value:
        return DEFAULT_ENDPOINT
    # Avoid duplicating chat/completions
    if value.count("/chat/completions") > 1:
        value = "/chat/completions"
    return value


def build_request_url(base_url: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    """Build the final request URL without /v1/v1 or endpoint duplication."""
    base = str(base_url or "").strip().rstrip("/")
    while "//" in base.replace("://", ""):
        base = base.replace("://", "::").replace("//", "/").replace("::", "://")
    # Strip duplicated /v1 segments
    while base.count("/v1/v1"):
        base = base.replace("/v1/v1", "/v1")
    endpoint_path = normalize_endpoint_path(endpoint)
    # Avoid appending endpoint again if base_url already ends with it
    if base.endswith(endpoint_path):
        final = base
    else:
        # If the base URL already ends with /chat/completions from a previous merge, don't append again
        final = base + endpoint_path
    return final


def key_fingerprint(api_key: str | None) -> str:
    """Return a safe display fingerprint: last 4 chars, or empty when unset."""
    value = str(api_key or "").strip()
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return f"••••{value[-4:]}"


def _config_path(root: Path) -> Path:
    return Path(root) / CONFIG_DIRECTORY / CONFIG_FILENAME


def _default_config() -> dict[str, Any]:
    return {"services": []}


def default_ai_config() -> dict[str, Any]:
    """Return a new empty AI configuration without runtime portal fields."""

    return _default_config()


def get_feature_review_ai_client(
    root: Path | str | None = None,
    *,
    purpose: str = "feature_review",
    preferred_service_id: str | None = None,
) -> tuple[Any | None, str]:
    """Resolve the user-selected feature-review AI client without leaking API keys."""
    from .portal_ai import PortalAIClient

    config_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        config = load_ai_config(config_root)
    except Exception as exc:
        return None, f"AI 配置文件读取失败：{exc}（请检查 {CONFIG_DIRECTORY}/{CONFIG_FILENAME}）"
    services = config.get("services", [])
    if not isinstance(services, list) or not services:
        return None, "未配置任何 AI 服务，请前往左侧边栏【AI 服务管理】添加并启用服务。"
    enabled_services = [s for s in services if isinstance(s, Mapping) and s.get("enabled")]
    if not enabled_services:
        return None, "所有 AI 服务均处于未启用状态，请前往左侧边栏【AI 服务管理】勾选启用。"

    # Honour the explicitly selected feature-review service; never silently switch.
    selected = None
    if preferred_service_id:
        for item in enabled_services:
            if str(item.get("service_id") or "") == str(preferred_service_id):
                selected = item
                break
        if selected is None:
            # Selected service exists but is disabled, or was deleted
            all_ids = {str(item.get("service_id") or "") for item in services}
            if str(preferred_service_id) in all_ids:
                return None, (
                    f"当前特征审核服务 [{preferred_service_id}] 处于未启用状态，"
                    "请前往左侧边栏【AI 服务管理】启用它，或显式切换到其他服务。"
                )
            return None, (
                f"当前特征审核服务 [{preferred_service_id}] 已不存在，"
                "请在左侧边栏【AI 服务管理】重新选择特征审核服务。"
            )
    else:
        selected = enabled_services[0]

    key = str((selected or {}).get("api_key") or "").strip()
    service_id = str((selected or {}).get("service_id") or "unknown")
    if not key:
        return None, (
            f"当前特征审核服务 [{service_id}] 未设置 API Key，"
            "请前往左侧边栏【AI 服务管理】补充 Key。"
        )
    try:
        validated = validate_ai_config({"services": [dict(selected)]})["services"][0]
        client = PortalAIClient(AIServiceConfig(**validated))
        label = str(validated.get("label") or validated.get("service_id") or "AI 服务")
        model = str(validated.get("model") or "")
        return client, f"AI 服务就绪（{label} · 模型：{model or '默认'} · 服务ID: {service_id}）"
    except Exception as exc:
        return None, f"当前特征审核服务 [{service_id}] 配置校验失败：{exc}"



def _plain_value(value: object, *, label: str) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_value(asdict(value), label=label)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string key")
            result[key] = _plain_value(item, label=f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item, label=f"{label}[]") for item in value]
    raise ValueError(f"{label} contains unsupported value type: {type(value).__name__}")


def _as_mapping(value: object, *, label: str) -> dict[str, Any]:
    value = _plain_value(value, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if required and not result:
        raise ValueError(f"{label} must not be empty")
    if len(result) > MAX_TEXT_LENGTH:
        raise ValueError(f"{label} must be at most {MAX_TEXT_LENGTH} characters")
    return result


def _sensitive_parameter_name(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", unquote(value).strip().lower()).strip("_")
    return bool(
        re.search(
            r"(?:^|_)(?:api_?key|access_?token|client_?secret|authorization|bearer|credential|"
            r"private_?key|password|secret|token|key)(?:$|_)",
            normalized,
        )
    )


def _contains_secret_value(value: str, secret_values: set[str]) -> bool:
    decoded = unquote(value)
    if any(secret and secret in decoded for secret in secret_values):
        return True
    return bool(re.search(r"(?i)(?:sk-[A-Za-z0-9_-]{4,}|bearer\s+[A-Za-z0-9._-]{8,})", decoded))


def _clean_url_component(
    component: str,
    *,
    reject_sensitive: bool,
    secret_values: set[str] | None = None,
) -> str:
    if not component:
        return component
    safe_parts = []
    secret_values = secret_values or set()
    for part in component.split("&"):
        parameter_name = part.split("=", 1)[0]
        decoded_part = unquote(part)
        if _sensitive_parameter_name(parameter_name):
            if reject_sensitive:
                raise ValueError("base_url contains sensitive query or fragment credentials")
            continue
        if _contains_secret_value(decoded_part, secret_values):
            continue
        safe_parts.append(part)
    return "&".join(safe_parts)


def _validated_url_parts(url: str) -> tuple[object, str]:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"base_url has an invalid hostname or port: {exc}") from exc

    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("base_url must be an http(s) URL with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if any(character.isspace() for character in hostname):
        raise ValueError("base_url has an invalid hostname")
    if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
        raise ValueError("base_url has an invalid port")

    if ":" in hostname:
        try:
            ipaddress.IPv6Address(hostname)
        except ValueError as exc:
            raise ValueError("base_url has an invalid hostname") from exc
    else:
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            try:
                ascii_hostname = hostname.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("base_url has an invalid hostname") from exc
            hostname_without_dot = ascii_hostname.rstrip(".")
            if not hostname_without_dot or len(hostname_without_dot) > 253:
                raise ValueError("base_url has an invalid hostname")
            if any(
                not label or len(label) > 63 or not _HOST_LABEL_PATTERN.fullmatch(label)
                for label in hostname_without_dot.split(".")
            ):
                raise ValueError("base_url has an invalid hostname")

    return parsed, hostname.lower().rstrip(".")


def _normalize_url(url: str, *, reject_sensitive: bool) -> str:
    parsed, hostname = _validated_url_parts(url)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and hostname not in local_hosts:
        raise ValueError("base_url must use HTTPS for non-local endpoints")


    query = _clean_url_component(parsed.query, reject_sensitive=reject_sensitive)
    fragment = _clean_url_component(parsed.fragment, reject_sensitive=reject_sensitive)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), query, fragment))


def _validate_url(value: object, *, provider: str) -> str:
    url = _text(value, label="base_url")
    return _normalize_url(url, reject_sensitive=True)


def _positive_number(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    result = float(value)
    if not MIN_TEMPERATURE <= result <= MAX_TEMPERATURE:
        raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    return result


def _service_mapping(value: object, *, index: int) -> dict[str, Any]:
    service = _as_mapping(value, label=f"services[{index}]")
    service_id = _text(service.get("service_id"), label=f"services[{index}].service_id")
    label = _text(service.get("label", service_id), label=f"services[{index}].label")
    provider = _text(service.get("provider", "openai-compatible"), label=f"services[{index}].provider")
    base_url = _validate_url(service.get("base_url"), provider=provider)
    endpoint = normalize_endpoint_path(str(service.get("endpoint") or DEFAULT_ENDPOINT))
    model = _text(service.get("model"), label=f"services[{index}].model")
    purpose = _text(service.get("purpose", DEFAULT_PURPOSE), label=f"services[{index}].purpose")
    if purpose not in SUPPORTED_PURPOSES:
        allowed = ", ".join(sorted(SUPPORTED_PURPOSES))
        raise ValueError(f"services[{index}].purpose must be one of: {allowed}")
    timeout_seconds = _positive_number(
        service.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        label=f"services[{index}].timeout_seconds",
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=MAX_TIMEOUT_SECONDS,
    )
    max_tokens = _positive_number(
        service.get("max_tokens", DEFAULT_MAX_TOKENS),
        label=f"services[{index}].max_tokens",
        minimum=MIN_MAX_TOKENS,
        maximum=MAX_MAX_TOKENS,
    )
    temperature = _temperature(service.get("temperature", DEFAULT_TEMPERATURE))
    enabled = service.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"services[{index}].enabled must be a boolean")
    api_key = service.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError(f"services[{index}].api_key must be a string")

    json_mode = str(service.get("json_mode") or DEFAULT_JSON_MODE).strip().lower()
    if json_mode not in SUPPORTED_JSON_MODES:
        json_mode = DEFAULT_JSON_MODE
    network_mode = str(service.get("network_mode") or DEFAULT_NETWORK_MODE).strip().lower()
    if network_mode not in SUPPORTED_NETWORK_MODES:
        network_mode = DEFAULT_NETWORK_MODE
    response_parse_mode = str(service.get("response_parse_mode") or DEFAULT_RESPONSE_PARSE_MODE).strip().lower()
    if response_parse_mode not in SUPPORTED_RESPONSE_PARSE_MODES:
        response_parse_mode = DEFAULT_RESPONSE_PARSE_MODE
    allow_json_mode_downgrade = service.get("allow_json_mode_downgrade", True)
    if not isinstance(allow_json_mode_downgrade, bool):
        allow_json_mode_downgrade = True
    max_retries_raw = service.get("max_retries", 2)
    if isinstance(max_retries_raw, bool) or not isinstance(max_retries_raw, int):
        max_retries = 2
    else:
        max_retries = max(MIN_RETRY_ATTEMPTS, min(MAX_RETRY_ATTEMPTS, max_retries_raw))
    auth_mode = str(service.get("auth_mode") or DEFAULT_AUTH_MODE).strip().lower()
    if auth_mode not in SUPPORTED_AUTH_MODES:
        auth_mode = DEFAULT_AUTH_MODE
    extra_headers_raw = service.get("headers")
    if isinstance(extra_headers_raw, Mapping):
        extra_headers = {
            str(key): str(item)
            for key, item in extra_headers_raw.items()
            if isinstance(key, str) and key.strip()
        }
    else:
        extra_headers = {}
    request_template_raw = service.get("request_template")
    if isinstance(request_template_raw, Mapping):
        request_template: dict[str, Any] | None = _as_mapping(
            request_template_raw, label=f"services[{index}].request_template"
        )
    else:
        request_template = None
    response_json_path_raw = service.get("response_json_path")
    if isinstance(response_json_path_raw, str) and response_json_path_raw.strip() and REQUEST_JSON_PATH_PATTERN.fullmatch(response_json_path_raw.strip()):
        response_json_path: str | None = response_json_path_raw.strip()
    else:
        response_json_path = None

    normalized = dict(service)
    # Backwards-compatible persistence: only persist optional advanced fields when
    # explicitly present in the source config; runtime dataclass supplies defaults.
    normalized.update(
        {
            "service_id": service_id,
            "label": label,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "purpose": purpose,
            "timeout_seconds": timeout_seconds,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enabled": enabled,
        }
    )
    if "endpoint" in service:
        normalized["endpoint"] = endpoint
    if "json_mode" in service:
        normalized["json_mode"] = json_mode
    if "network_mode" in service:
        normalized["network_mode"] = network_mode
    if "response_parse_mode" in service:
        normalized["response_parse_mode"] = response_parse_mode
    if "allow_json_mode_downgrade" in service:
        normalized["allow_json_mode_downgrade"] = allow_json_mode_downgrade
    if "max_retries" in service:
        normalized["max_retries"] = max_retries
    if "auth_mode" in service:
        normalized["auth_mode"] = auth_mode
    if "headers" in service:
        normalized["headers"] = extra_headers
    if "request_template" in service:
        normalized["request_template"] = request_template
    if "response_json_path" in service:
        normalized["response_json_path"] = response_json_path
    if "anthropic_version" in service:
        anthropic_version_raw = service.get("anthropic_version")
        normalized["anthropic_version"] = (
            str(anthropic_version_raw).strip() if isinstance(anthropic_version_raw, str) and str(anthropic_version_raw).strip() else "2023-06-01"
        )
    return normalized


def validate_ai_config(config: object) -> dict[str, Any]:
    """Validate and return a deep, JSON-safe copy of an AI configuration."""

    payload = _as_mapping(config, label="AI config")
    services = payload.get("services", [])
    if not isinstance(services, list):
        raise ValueError("services must be a list")

    normalized = dict(payload)
    normalized_services = []
    service_ids = set()
    for index, service in enumerate(services):
        normalized_service = _service_mapping(service, index=index)
        service_id = normalized_service["service_id"]
        if service_id in service_ids:
            raise ValueError(f"duplicate service_id: {service_id}")
        service_ids.add(service_id)
        normalized_services.append(normalized_service)
    normalized["services"] = normalized_services
    return normalized


def _auth_headers(auth_mode: str, api_key: str | None) -> dict[str, str]:
    if auth_mode == "bearer" and api_key:
        return {"Authorization": f"Bearer {api_key}"}
    if auth_mode == "api_key_header" and api_key:
        return {"X-API-Key": api_key}
    return {}


def _resolve_provider_kind(provider: str, model: str, base_url: str, has_template: bool) -> str:
    """Resolve the wire-protocol family for a service config.

    优先级：provider 显式声明 > URL/模型名启发式。支持
    openai-compatible / gemini / anthropic / custom(template)。
    """
    provider_l = provider.strip().lower()
    model_l = model.strip().lower()
    base_l = base_url.strip().lower()
    if provider_l == "custom" and has_template:
        return "template"
    if "gemini" in provider_l:
        return "gemini"
    if "anthropic" in provider_l or "claude" in provider_l:
        return "anthropic"
    if "openai" in provider_l:
        return "openai"
    # 启发式兜底：provider 自由文本时按模型/URL 猜协议族
    if "gemini" in model_l or "generativelanguage" in base_l:
        return "gemini"
    if "claude" in model_l or "anthropic" in base_l:
        return "anthropic"
    return "openai"


def get_request_spec(config: object) -> dict[str, Any]:
    """Resolve provider-specific request spec: {url, headers, body_kind, endpoint, provider_kind}.

    支持四种协议：openai（/chat/completions，Bearer）、gemini（/v1beta/models/
    {model}:generateContent，x-goog-api-key 头）、anthropic（/v1/messages，
    x-api-key + anthropic-version 头）、custom（request_template）。
    headers = 协议默认认证头 + 配置中的额外头（额外头不覆盖协议默认）。
    """
    provider = str(getattr(config, "provider", "") or "").strip()
    provider_l = provider.lower()
    model = str(getattr(config, "model", "") or "").strip()
    base_url = str(getattr(config, "base_url", "") or "").strip()
    api_key = getattr(config, "api_key", None)
    auth_mode = str(getattr(config, "auth_mode", DEFAULT_AUTH_MODE) or DEFAULT_AUTH_MODE).strip().lower()
    if auth_mode not in SUPPORTED_AUTH_MODES:
        auth_mode = DEFAULT_AUTH_MODE
    has_template = bool(getattr(config, "request_template", None))
    kind = _resolve_provider_kind(provider, model, base_url, has_template)

    if kind == "gemini":
        # 官方 generativelanguage 或兼容中转：/v1beta/models/{model}:generateContent
        # base 已带 /v1beta → 直接拼；带 /v1（OpenAI 风格中转）→ 去掉 /v1 再拼 /v1beta；
        # 其他 → 前缀 /v1beta。
        base_lower = base_url.lower().rstrip("/")
        if "/v1beta" in base_lower:
            endpoint = f"/models/{model}:generateContent"
        elif base_lower.endswith("/v1"):
            trimmed = base_url.rstrip("/")[:-3]
            endpoint = f"/v1beta/models/{model}:generateContent"
            base_url = trimmed
        else:
            endpoint = f"/v1beta/models/{model}:generateContent"
        body_kind = "gemini"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        # Gemini 原生认证用 x-goog-api-key；中转站多用 OpenAI 兼容 Bearer，
        # auth_mode=bearer 时两者都带，保证两种网关都能通。
        if isinstance(api_key, str) and api_key.strip():
            headers["x-goog-api-key"] = api_key.strip()
            if auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {api_key.strip()}"
        elif auth_mode == "bearer" and isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
    elif kind == "anthropic":
        # endpoint 留空或为 OpenAI 默认值时改用 Anthropic 原生路径
        endpoint_raw = str(getattr(config, "endpoint", "") or "").strip()
        if not endpoint_raw or normalize_endpoint_path(endpoint_raw) == DEFAULT_ENDPOINT:
            endpoint = "/v1/messages"
        else:
            endpoint = normalize_endpoint_path(endpoint_raw)
        body_kind = "anthropic"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": str(getattr(config, "anthropic_version", "") or "2023-06-01"),
        }
        if isinstance(api_key, str) and api_key.strip():
            headers["x-api-key"] = api_key.strip()
            if auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {api_key.strip()}"
    elif kind == "template":
        endpoint = normalize_endpoint_path(str(getattr(config, "endpoint", "") or DEFAULT_ENDPOINT))
        body_kind = "template"
        headers = {"Content-Type": "application/json"}
        headers.update(_auth_headers(auth_mode, api_key if isinstance(api_key, str) else None))
    else:
        endpoint = normalize_endpoint_path(str(getattr(config, "endpoint", "") or DEFAULT_ENDPOINT))
        body_kind = "openai"
        headers = {"Content-Type": "application/json"}
        headers.update(_auth_headers(auth_mode, api_key if isinstance(api_key, str) else None))

    extra_headers = getattr(config, "headers", None)
    if isinstance(extra_headers, Mapping):
        for key, value in extra_headers.items():
            if isinstance(key, str) and key.strip() and key not in headers:
                headers[key] = str(value)

    return {
        "url": build_request_url(base_url, endpoint),
        "headers": headers,
        "body_kind": body_kind,
        "endpoint": endpoint,
        "provider_kind": kind,
    }


def build_request_body(config: object, messages: list[dict[str, str]], *, json_mode: bool = False) -> dict[str, Any]:
    """Build the protocol-specific request body for a chat completion.

    openai: {model, messages, temperature, max_tokens[, response_format]}
    gemini: {contents:[{role, parts:[{text}]}], generationConfig{...}[, responseMimeType]}
    anthropic: {model, max_tokens, system, messages:[{role, content}], temperature}
    template: request_template 原样（调用方自行处理占位符替换）。
    """
    kind = _resolve_provider_kind(
        str(getattr(config, "provider", "") or ""),
        str(getattr(config, "model", "") or ""),
        str(getattr(config, "base_url", "") or ""),
        bool(getattr(config, "request_template", None)),
    )
    temperature = getattr(config, "temperature", 0.2)
    try:
        temperature = max(0.0, min(2.0, float(temperature)))
    except (TypeError, ValueError):
        temperature = 0.2
    if kind == "gemini":
        contents = []
        system_text = ""
        for message in messages:
            role = str(message.get("role") or "user").lower()
            text = str(message.get("content") or "")
            if role == "system":
                system_text = (system_text + "\n" + text).strip()
                continue
            contents.append({"role": "user" if role != "assistant" else "model", "parts": [{"text": text}]})
        generation: dict[str, Any] = {"temperature": temperature}
        max_tokens = getattr(config, "max_tokens", None)
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if json_mode:
            generation["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        return body
    if kind == "anthropic":
        system_text = ""
        rest: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").lower()
            text = str(message.get("content") or "")
            if role == "system":
                system_text = (system_text + "\n" + text).strip()
                continue
            rest.append({"role": "user" if role != "assistant" else "assistant", "content": text})
        body = {
            "model": str(getattr(config, "model", "") or ""),
            "max_tokens": int(getattr(config, "max_tokens", 2048) or 2048),
            "temperature": temperature,
            "messages": rest,
        }
        if system_text:
            body["system"] = system_text
        return body
    # openai（默认）
    body = {
        "model": str(getattr(config, "model", "") or ""),
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": int(getattr(config, "max_tokens", 2048) or 2048),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _sanitize_url_for_output(value: str, *, secret_values: set[str]) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if not hostname:
            return "[redacted URL]"
        port = parsed.port
    except ValueError:
        return "[redacted URL]"

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    safe_netloc = hostname
    if port is not None:
        safe_netloc = f"{safe_netloc}:{port}"
    query = _clean_url_component(
        parsed.query,
        reject_sensitive=False,
        secret_values=secret_values,
    )
    fragment = _clean_url_component(
        parsed.fragment,
        reject_sensitive=False,
        secret_values=secret_values,
    )
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path.rstrip("/"), query, fragment))


def _copy_without_secrets(value: object, *, mask: bool, secret_values: set[str]) -> object:
    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            if raw_key.lower() == "base_url" and isinstance(item, str):
                result[raw_key] = _sanitize_url_for_output(item, secret_values=secret_values)
                continue
            if _SECRET_KEY_PATTERN.search(raw_key):
                if mask:
                    result[raw_key] = "••••••••"
                continue
            result[raw_key] = _copy_without_secrets(item, mask=mask, secret_values=secret_values)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_without_secrets(item, mask=mask, secret_values=secret_values) for item in value]
    if isinstance(value, str) and secret_values:
        result = value
        for secret in secret_values:
            if secret:
                result = result.replace(secret, "[redacted]")
        return result
    return value


def _secret_values(config: object) -> set[str]:
    values = set()
    plain_config = _plain_value(config, label="AI config")

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                if isinstance(raw_key, str) and _SECRET_KEY_PATTERN.search(raw_key):
                    if isinstance(item, str) and item:
                        values.add(item)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(plain_config)
    return values


def redacted_ai_config(config: object) -> object:
    """Return a copy suitable for display, with secret values masked."""

    plain_config = _plain_value(config, label="AI config")
    return _copy_without_secrets(plain_config, mask=True, secret_values=_secret_values(plain_config))


def exportable_ai_config(config: object) -> object:
    """Return a copy suitable for export, with all secret fields removed."""

    plain_config = _plain_value(config, label="AI config")
    return _copy_without_secrets(plain_config, mask=False, secret_values=_secret_values(plain_config))


def _protect_file(path: Path) -> None:
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except (NotImplementedError, OSError):
        if os.name != "nt":
            raise


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            _protect_file(temporary_path)
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    candidate = path.with_name(f"{path.stem}.{timestamp}.bak")
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{timestamp}.{uuid.uuid4().hex[:8]}.bak")
    return candidate


def _backup_before_write(path: Path) -> None:
    backup = _backup_path(path)
    if path.is_file():
        _atomic_write(backup, path.read_bytes())
    else:
        _atomic_write(backup, json.dumps(_default_config(), ensure_ascii=False, indent=2).encode("utf-8"))


def load_ai_config(root: Path) -> dict[str, Any]:
    """Load and validate the complete local AI configuration."""

    path = _config_path(root)
    if not path.is_file():
        return _default_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read AI config: {exc}") from exc
    return validate_ai_config(payload)


def save_ai_config(root: Path, config: object) -> dict[str, Any]:
    """Validate, back up, and atomically save the complete local configuration."""

    normalized = validate_ai_config(config)
    try:
        data = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AI config is not JSON serializable: {exc}") from exc

    path = _config_path(root)
    _backup_before_write(path)
    _atomic_write(path, data)
    return normalized


__all__ = [
    "AIServiceConfig",
    "default_ai_config",
    "exportable_ai_config",
    "load_ai_config",
    "redacted_ai_config",
    "save_ai_config",
    "validate_ai_config",
    "normalize_endpoint_path",
    "build_request_url",
    "get_request_spec",
    "build_request_body",
    "_resolve_provider_kind",
    "key_fingerprint",
    "get_feature_review_ai_client",
]
