"""Secure local persistence and redaction for portal AI service configuration."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
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
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 300
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 200_000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_TEXT_LENGTH = 200
PRIVATE_FILE_MODE = 0o600

_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|authorization|bearer|client[_-]?secret|"
    r"credential|password|private[_-]?key|secret|token|key)",
    re.IGNORECASE,
)
_HOST_LABEL_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class AIServiceConfig:
    service_id: str
    label: str = ""
    provider: str = ""
    base_url: str = ""
    model: str = ""
    purpose: str = DEFAULT_PURPOSE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    enabled: bool = True
    api_key: str | None = None


def _config_path(root: Path) -> Path:
    return Path(root) / CONFIG_DIRECTORY / CONFIG_FILENAME


def _default_config() -> dict[str, Any]:
    return {"services": []}


def default_ai_config() -> dict[str, Any]:
    """Return a new empty AI configuration without runtime portal fields."""

    return _default_config()


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


def _clean_url_component(component: str, *, reject_sensitive: bool) -> str:
    if not component:
        return component
    safe_parts = []
    for part in component.split("&"):
        parameter_name = part.split("=", 1)[0]
        if _sensitive_parameter_name(parameter_name):
            if reject_sensitive:
                raise ValueError("base_url contains sensitive query or fragment credentials")
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

    normalized = dict(service)
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


def _sanitize_url_for_output(value: str) -> str:
    try:
        parsed = urlsplit(value)
        query = _clean_url_component(parsed.query, reject_sensitive=False)
        fragment = _clean_url_component(parsed.fragment, reject_sensitive=False)
    except ValueError:
        return "[redacted URL]"
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), query, fragment))


def _copy_without_secrets(value: object, *, mask: bool, secret_values: set[str]) -> object:
    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            if raw_key.lower() == "base_url" and isinstance(item, str):
                result[raw_key] = _sanitize_url_for_output(item)
                continue
            if _SECRET_KEY_PATTERN.search(raw_key.replace(" ", "")):
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
                if isinstance(raw_key, str) and _SECRET_KEY_PATTERN.search(raw_key.replace(" ", "")):
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
]

