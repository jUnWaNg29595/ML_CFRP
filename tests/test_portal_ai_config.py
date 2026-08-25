import json
import os
import stat
from dataclasses import dataclass

import pytest

from core import portal_ai_config
from core.portal_ai_config import (
    AIServiceConfig,
    exportable_ai_config,
    load_ai_config,
    redacted_ai_config,
    save_ai_config,
    validate_ai_config,
)


DEFAULT_PURPOSE = "both"


def _service(**overrides):
    service = {
        "service_id": "deepseek",
        "label": "DeepSeek",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "purpose": DEFAULT_PURPOSE,
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


def test_config_and_backups_have_private_permissions(tmp_path):
    save_ai_config(tmp_path, {"services": [_service(api_key="sk-secret")]})
    save_ai_config(tmp_path, {"services": [_service(api_key="sk-new")]})

    config_path = tmp_path / "prediction_portal" / "ai_config.json"
    backup_paths = list(config_path.parent.glob("ai_config.*.bak"))
    assert backup_paths

    for path in [config_path, *backup_paths]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if os.name == "nt":
            assert mode & stat.S_IRUSR
            assert mode & stat.S_IWUSR
        else:
            assert mode == 0o600


def test_redacted_config_masks_keys_without_mutating_input():
    config = {"services": [_service(api_key="sk-secret", nested={"access_token": "token-secret"})]}

    redacted = redacted_ai_config(config)

    assert config["services"][0]["api_key"] == "sk-secret"
    assert redacted["services"][0]["api_key"] != "sk-secret"
    assert redacted["services"][0]["nested"]["access_token"] != "token-secret"
    assert "sk-secret" not in json.dumps(redacted)
    assert "token-secret" not in json.dumps(redacted)


def test_redaction_and_export_keep_max_tokens_while_handling_api_key():
    config = {"services": [_service(api_key="sk-secret", max_tokens=2048)]}

    redacted = redacted_ai_config(config)
    exported = exportable_ai_config(config)

    assert redacted["services"][0]["max_tokens"] == 2048
    assert redacted["services"][0]["api_key"] == "••••••••"
    assert exported["services"][0]["max_tokens"] == 2048
    assert "api_key" not in exported["services"][0]


def test_redaction_and_export_support_dataclasses_and_nested_lists():
    service = AIServiceConfig(
        service_id="local",
        label="Local",
        provider="ollama",
        base_url="http://localhost:11434/v1",
        model="llama3",
        purpose="input_parsing",
        api_key="sk-dataclass",
    )

    redacted = redacted_ai_config([service, {"nested": [service]}])
    exported = exportable_ai_config([service, {"nested": [service]}])

    assert isinstance(redacted, list)
    assert redacted[0]["api_key"] != "sk-dataclass"
    assert exported[0]["service_id"] == "local"
    assert "api_key" not in exported[0]
    assert "sk-dataclass" not in json.dumps(exported)


@dataclass
class _ConfigBundle:
    services: list[AIServiceConfig]
    metadata: dict[str, object]


def test_validate_supports_dataclass_config_and_nested_lists():
    config = _ConfigBundle(
        services=[
            AIServiceConfig(
                service_id="local",
                label="Local",
                provider="ollama",
                base_url="http://localhost:11434/v1",
                model="llama3",
                purpose="both",
            )
        ],
        metadata={"labels": ["safe", {"values": [1, 2]}]},
    )

    validated = validate_ai_config(config)

    assert validated["services"][0]["purpose"] == "both"
    assert validated["metadata"]["labels"][1]["values"] == [1, 2]


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "http://api.example.com/v1"},
        {"base_url": "not-a-url"},
        {"model": ""},
        {"purpose": "general"},
        {"purpose": "input"},
        {"timeout_seconds": 0},
        {"max_tokens": 0},
        {"temperature": 2.1},
    ],
)
def test_validate_rejects_unsafe_or_invalid_service_values(overrides):
    with pytest.raises(ValueError):
        validate_ai_config({"services": [_service(**overrides)]})


@pytest.mark.parametrize("purpose", ["input_parsing", "result_explanation", "both"])
def test_validate_allows_supported_purposes(purpose):
    validated = validate_ai_config({"services": [_service(purpose=purpose)]})

    assert validated["services"][0]["purpose"] == purpose


def test_default_purpose_is_both():
    validated = validate_ai_config(
        {
            "services": [
                {
                    "service_id": "deepseek",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                }
            ]
        }
    )

    assert validated["services"][0]["purpose"] == DEFAULT_PURPOSE


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:11434", "http://127.0.0.1:11434/v1"],
)
def test_validate_allows_local_http_endpoints(base_url):
    validated = validate_ai_config(
        {"services": [_service(provider="ollama", base_url=base_url)]}
    )

    assert validated["services"][0]["base_url"] == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.com/v1?api_key=secret",
        "https://api.example.com/v1?region=cn#token=secret",
        "https://api.example.com/v1#password=secret",
        "https://api.example.com/v1?client_secret=secret",
    ],
)
def test_validate_rejects_sensitive_url_query_or_fragment_parameters(base_url):
    with pytest.raises(ValueError, match="sensitive|secret|credential"):
        validate_ai_config({"services": [_service(base_url=base_url)]})


def test_exportable_and_redacted_remove_sensitive_url_parameters():
    config = {
        "services": [
            _service(
                base_url="https://api.example.com/v1?region=cn&api_key=secret#view=full&token=secret2"
            )
        ]
    }

    exported = exportable_ai_config(config)
    redacted = redacted_ai_config(config)

    assert "secret" not in json.dumps(exported)
    assert "secret2" not in json.dumps(exported)
    assert "region=cn" in exported["services"][0]["base_url"]
    assert "api_key" not in exported["services"][0]["base_url"]
    assert "secret" not in json.dumps(redacted)
    assert "region=cn" in redacted["services"][0]["base_url"]


def test_exportable_url_removes_secret_values_and_userinfo():
    config = {
        "services": [
            _service(
                base_url="https://user:password@example.com/v1?region=cn&foo=sk-secret#view=full%26bar=sk-secret"
            )
        ]
    }

    exported = exportable_ai_config(config)
    redacted = redacted_ai_config(config)

    for payload in (exported, redacted):
        text = json.dumps(payload)
        assert "password" not in text
        assert "sk-secret" not in text
        assert "user@" not in text
    assert exported["services"][0]["base_url"] == "https://example.com/v1?region=cn"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api. example.com/v1",
        "https://api.example.com:abc/v1",
        "https://api.example.com:0/v1",
        "https://api.example.com:65536/v1",
        "https://[::1/v1",
    ],
)
def test_validate_rejects_malformed_hostname_or_port(base_url):
    with pytest.raises(ValueError, match="base_url"):
        validate_ai_config({"services": [_service(base_url=base_url)]})


def test_validate_only_removes_trailing_path_slashes_and_preserves_query_fragment():
    base_url = "https://api.example.com/v1///?region=cn&mode=chat#view=full/"

    validated = validate_ai_config({"services": [_service(base_url=base_url)]})

    assert validated["services"][0]["base_url"] == (
        "https://api.example.com/v1?region=cn&mode=chat#view=full/"
    )


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

    assert json.loads(path.read_text(encoding="utf-8")) == config | {
        "services": [{**config["services"][0], "purpose": DEFAULT_PURPOSE, "base_url": "https://api.deepseek.com/v1"}]
    }
    assert not list(path.parent.glob(".ai_config.*.tmp"))


def test_replace_failure_preserves_original_and_cleans_temp_file(tmp_path, monkeypatch):
    first = {"services": [_service(api_key="sk-first")]}
    second = {"services": [_service(api_key="sk-second")]}
    save_ai_config(tmp_path, first)
    path = tmp_path / "prediction_portal" / "ai_config.json"
    original = path.read_bytes()
    real_replace = portal_ai_config.os.replace
    calls = 0

    def fail_target_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(portal_ai_config.os, "replace", fail_target_replace)
    with pytest.raises(OSError, match="replace"):
        save_ai_config(tmp_path, second)

    assert path.read_bytes() == original
    assert not list(path.parent.glob(".ai_config.*.tmp"))


def test_atomic_write_failure_preserves_original_and_cleans_temp_file(tmp_path, monkeypatch):
    first = {"services": [_service(api_key="sk-first")]}
    second = {"services": [_service(api_key="sk-second")]}
    save_ai_config(tmp_path, first)
    path = tmp_path / "prediction_portal" / "ai_config.json"
    original = path.read_bytes()
    real_fsync = portal_ai_config.os.fsync
    calls = 0

    def fail_target_fsync(file_descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated atomic write failure")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(portal_ai_config.os, "fsync", fail_target_fsync)
    with pytest.raises(OSError, match="atomic write"):
        save_ai_config(tmp_path, second)

    assert path.read_bytes() == original
    assert not list(path.parent.glob(".ai_config.*.tmp"))


def test_normalize_url_has_no_stale_provider_reference():
    assert portal_ai_config._normalize_url(
        "https://api.example.com/v1///?region=cn#view=full/",
        reject_sensitive=True,
    ) == "https://api.example.com/v1?region=cn#view=full/"


def test_export_removes_known_secret_values_from_arbitrary_encoded_parameters():
    config = {
        "services": [
            _service(
                api_key="sk-secret",
                base_url=(
                    "https://api.example.com/v1?region=cn&custom=sk%2Dsecret"
                    "#state=sk%2Dsecret"
                ),
            )
        ]
    }

    exported = exportable_ai_config(config)
    redacted = redacted_ai_config(config)

    for result in [exported, redacted]:
        url = result["services"][0]["base_url"]
        assert "region=cn" in url
        assert "sk%2Dsecret" not in url
        assert "sk-secret" not in url


def test_export_strips_url_userinfo_without_losing_host():
    config = {
        "services": [
            _service(base_url="https://alice:password@example.com/v1?region=cn")
        ]
    }

    exported = exportable_ai_config(config)
    redacted = redacted_ai_config(config)

    for result in [exported, redacted]:
        url = result["services"][0]["base_url"]
        assert url == "https://example.com/v1?region=cn"
        assert "alice" not in json.dumps(result)
        assert "password" not in json.dumps(result)


def test_windows_chmod_failure_does_not_abort_save(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_ai_config.os, "name", "nt")

    def fail_chmod(*args):
        raise OSError("simulated Windows chmod limitation")

    monkeypatch.setattr(portal_ai_config.os, "chmod", fail_chmod)

    save_ai_config(tmp_path, {"services": [_service(api_key="sk-secret")]})

    assert load_ai_config(tmp_path)["services"][0]["api_key"] == "sk-secret"

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
from pathlib import Path
from core.portal_ai_config import default_ai_config, exportable_ai_config
import json


def test_ai_export_has_no_key_and_runtime_port_is_separate():
    config = {'services': [{'service_id': 'deepseek', 'api_key': 'sk-live-secret', 'model': 'deepseek-chat', 'base_url': 'https://api.deepseek.com/v1'}]}
    assert 'sk-live-secret' not in json.dumps(exportable_ai_config(config))
    assert 'port' not in default_ai_config()
