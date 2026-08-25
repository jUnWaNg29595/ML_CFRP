import json

import pytest

from core.portal_ai_config import (
    AIServiceConfig,
    exportable_ai_config,
    load_ai_config,
    redacted_ai_config,
    save_ai_config,
    validate_ai_config,
)


def _service(**overrides):
    service = {
        "service_id": "deepseek",
        "label": "DeepSeek",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "purpose": "general",
        "timeout_seconds": 30,
        "max_tokens": 2048,
        "temperature": 0.2,
        "enabled": True,
    }
    service.update(overrides)
    return service


def test_key_is_stored_locally_but_never_exported(tmp_path):
    save_ai_config(
        tmp_path,
        {
            "services": [
                {
                    "service_id": "deepseek",
                    "api_key": "sk-secret",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "enabled": True,
                }
            ]
        },
    )

    payload = load_ai_config(tmp_path)

    assert payload["services"][0]["api_key"] == "sk-secret"
    assert "sk-secret" not in json.dumps(exportable_ai_config(payload))
    assert list((tmp_path / "prediction_portal").glob("ai_config.*.bak"))


def test_second_save_creates_timestamped_backup_of_previous_full_config(tmp_path):
    first = {"services": [_service(api_key="sk-first")], "user_data": {"theme": "dark"}}
    second = {"services": [_service(api_key="sk-second")], "user_data": {"theme": "light"}}

    save_ai_config(tmp_path, first)
    save_ai_config(tmp_path, second)

    backups = sorted((tmp_path / "prediction_portal").glob("ai_config.*.bak"))
    assert backups
    assert json.loads(backups[-1].read_text(encoding="utf-8"))["services"][0]["api_key"] == "sk-first"
    assert load_ai_config(tmp_path)["services"][0]["api_key"] == "sk-second"


def test_redacted_config_masks_keys_without_mutating_input():
    config = {"services": [_service(api_key="sk-secret", nested={"access_token": "token-secret"})]}

    redacted = redacted_ai_config(config)

    assert config["services"][0]["api_key"] == "sk-secret"
    assert redacted["services"][0]["api_key"] != "sk-secret"
    assert redacted["services"][0]["nested"]["access_token"] != "token-secret"
    assert "sk-secret" not in json.dumps(redacted)
    assert "token-secret" not in json.dumps(redacted)


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "http://api.example.com/v1"},
        {"base_url": "not-a-url"},
        {"model": ""},
        {"purpose": ""},
        {"timeout_seconds": 0},
        {"max_tokens": 0},
        {"temperature": 2.1},
    ],
)
def test_validate_rejects_unsafe_or_invalid_service_values(overrides):
    with pytest.raises(ValueError):
        validate_ai_config({"services": [_service(**overrides)]})


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:11434", "http://127.0.0.1:11434/v1"],
)
def test_validate_allows_local_http_endpoints(base_url):
    validated = validate_ai_config(
        {"services": [_service(provider="ollama", base_url=base_url)]}
    )

    assert validated["services"][0]["base_url"] == base_url


def test_validate_rejects_duplicate_service_ids_and_non_boolean_enabled():
    with pytest.raises(ValueError, match="service_id"):
        validate_ai_config({"services": [_service(), _service(label="second")]})

    with pytest.raises(ValueError, match="enabled"):
        validate_ai_config({"services": [_service(enabled=1)]})


def test_save_is_atomic_and_preserves_unknown_json_user_data(tmp_path):
    config = {
        "schema_version": 1,
        "services": [_service(api_key="sk-secret")],
        "user_data": {"favorite_service": "deepseek", "notes": ["keep me"]},
    }

    save_ai_config(tmp_path, config)
    path = tmp_path / "prediction_portal" / "ai_config.json"

    assert json.loads(path.read_text(encoding="utf-8")) == config
    assert not list(path.parent.glob(".ai_config.*.tmp"))


def test_ai_service_config_has_no_runtime_port_or_prediction_fields():
    field_names = {field.name for field in AIServiceConfig.__dataclass_fields__.values()}

    assert {
        "service_id",
        "label",
        "provider",
        "base_url",
        "model",
        "purpose",
        "timeout_seconds",
        "max_tokens",
        "temperature",
        "enabled",
    }.issubset(field_names)
    assert "port" not in field_names
    assert "prediction_config" not in field_names
