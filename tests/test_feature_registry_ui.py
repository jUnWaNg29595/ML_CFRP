def test_feature_registry_ui_exports_minimal_renderer():
    from core.feature_registry_ui import render_feature_registry_page

    assert callable(render_feature_registry_page)


def test_app_dispatch_exposes_feature_management_page():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "app.py"
    text = source.read_text(encoding="utf-8")
    assert "🧩 特征管理" in text
    assert "render_feature_registry_page" in text


def test_feature_review_ui_persists_ai_and_local_decision_records():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "feature_registry_ui.py").read_text(encoding="utf-8")
    assert "save_feature_review_record" in source
    assert "feature_reviews" in source
    assert "ai_response" in source
    assert "local_decision" in source


def test_feature_review_ui_exposes_unknown_status_filter():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "feature_registry_ui.py").read_text(encoding="utf-8")
    assert '"unknown"' in source


def test_feature_review_ui_exposes_profile_and_manual_mapping_workspace():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "feature_registry_ui.py").read_text(encoding="utf-8")
    assert "模型 profile" in source
    assert "新建特征映射建议" in source
    assert "保存为待审核建议" in source
    assert '"status": "pending_review"' in source


def test_manual_feature_suggestion_is_pending_and_bounded():
    from core.feature_registry_ui import build_manual_feature_suggestion

    suggestion = build_manual_feature_suggestion(
        feature_id="cfrp.tg.degree_of_cure_pct",
        raw_columns=["固化度"],
        source_role="manual_input",
        unit="%",
        rationale="本地人工确认列含义，等待批准",
    )
    assert suggestion["status"] == "pending_review"
    assert suggestion["feature_id"] == "cfrp.tg.degree_of_cure_pct"
    assert suggestion["raw_columns"] == ["固化度"]
    assert suggestion["source_role"] == "manual_input"
    assert "approved" not in suggestion.values()


def test_frame_column_names_handles_pandas_index():
    import pandas as pd

    from core.feature_registry_ui import frame_column_names

    assert frame_column_names(pd.DataFrame({"pressure": [1.0]})) == ["pressure"]
    assert frame_column_names(None) == []


def test_feature_mapping_candidates_keep_metadata_and_blocked_entries_visible():
    from core.feature_registry_ui import build_feature_mapping_candidates

    registry = {
        "model_profiles": {"p": {"feature_ids": ["legacy", "blocked", "manual"], "status": "blocked"}},
        "features": [
            {"feature_id": "legacy", "name": "legacy_x", "source_type": "metadata", "status": "legacy_observed"},
            {"feature_id": "blocked", "name": "blocked_x", "source_type": "unknown", "status": "blocked"},
            {"feature_id": "manual", "name": "manual_x", "source_type": "manual_input", "status": "draft"},
        ],
    }

    candidates = build_feature_mapping_candidates(registry, "p")
    by_id = {item["feature_id"]: item for item in candidates}
    assert set(by_id) == {"legacy", "blocked", "manual"}
    assert by_id["legacy"]["approval_allowed"] is False
    assert "metadata" in by_id["legacy"]["approval_note"]
    assert by_id["blocked"]["approval_allowed"] is False
    assert "blocked" in by_id["blocked"]["approval_note"]
    assert by_id["manual"]["approval_allowed"] is True


def test_feature_registry_ui_keeps_new_candidate_workspace_when_no_reviewable_feature():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "feature_registry_ui.py").read_text(encoding="utf-8")
    assert "新建规范 feature_id" in source
    assert "新建候选仅保存为 pending_review" in source
    assert "approval_allowed" in source


def test_feature_mapping_candidates_require_reviewable_feature_status():
    from core.feature_registry_ui import build_feature_mapping_candidates

    registry = {
        "model_profiles": {"p": {"feature_ids": ["legacy_role", "missing_status", "draft_role"]}},
        "features": [
            {"feature_id": "legacy_role", "name": "legacy_x", "source_type": "manual_input", "status": "legacy_observed"},
            {"feature_id": "missing_status", "name": "missing_x", "source_type": "derived_workflow"},
            {"feature_id": "draft_role", "name": "draft_x", "source_type": "manual_input", "status": "draft"},
        ],
    }

    candidates = build_feature_mapping_candidates(registry, "p")
    by_id = {item["feature_id"]: item for item in candidates}
    assert by_id["legacy_role"]["approval_allowed"] is False
    assert "legacy_observed" in by_id["legacy_role"]["approval_note"]
    assert by_id["missing_status"]["approval_allowed"] is False
    assert "status=unknown" in by_id["missing_status"]["approval_note"]
    assert by_id["draft_role"]["approval_allowed"] is True


def test_feature_mapping_candidates_keep_missing_profile_references_visible():
    from core.feature_registry_ui import build_feature_mapping_candidates

    registry = {
        "model_profiles": {"p": {"feature_ids": ["missing_feature"]}},
        "features": [],
    }
    candidates = build_feature_mapping_candidates(registry, "p")
    assert candidates == [{
        "feature_id": "missing_feature",
        "name": "missing_feature",
        "source_type": "unknown",
        "status": "unknown",
        "approval_allowed": False,
        "approval_note": "不可直接批准：source_type=unknown，status=unknown",
    }]


def test_feature_review_ai_client_resolver_safe_without_api_key_leaks(tmp_path):
    from core.portal_ai_config import get_feature_review_ai_client, save_ai_config

    client, msg = get_feature_review_ai_client(tmp_path)
    assert client is None
    assert "未配置任何 AI 服务" in msg or "AI 服务" in msg

    # Save a dummy service
    save_ai_config(tmp_path, {
        "services": [{
            "service_id": "test_ai",
            "api_key": "sk-secret-key-12345",
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
            "enabled": True,
            "purpose": "both",
        }]
    })

    client, msg = get_feature_review_ai_client(tmp_path)
    assert client is not None
    assert "sk-secret-key-12345" not in msg
    assert "test_ai" in msg or "test-model" in msg


def test_register_new_feature_and_save_atomic(tmp_path):
    from core.feature_registry import load_registry, register_new_feature, save_registry_atomic, validate_registry

    base_registry = {
        "schema_version": 1,
        "registry_version": "2026.08.27",
        "approval": {"status": "draft"},
        "model_profiles": {"test_profile": {"feature_ids": [], "target_col": "tg", "status": "draft"}},
        "features": [],
    }

    new_feat = {
        "feature_id": "cfrp.custom.cure_temperature",
        "name": "cure_temperature",
        "label": "固化温度",
        "source_type": "manual_input",
        "unit": "℃",
        "status": "draft",
    }

    updated = register_new_feature(
        base_registry,
        new_feat,
        reviewer="auditor-bob",
        target_profile_id="test_profile",
    )
    val = validate_registry(updated)
    assert val["ok"] is True
    assert any(f["feature_id"] == "cfrp.custom.cure_temperature" for f in updated["features"])
    assert "cfrp.custom.cure_temperature" in updated["model_profiles"]["test_profile"]["feature_ids"]

    reg_path = tmp_path / "registry.json"
    save_registry_atomic(reg_path, updated)
    reloaded = load_registry(reg_path)
    assert len(reloaded["features"]) == 1


def test_profile_manifest_and_suggestions_persistence(tmp_path):
    from core.dataset_manifest import compute_dataset_manifest_hash
    from core.feature_mapping_review import (
        load_profile_manifest,
        load_profile_suggestions,
        save_profile_manifest,
        save_profile_suggestions,
    )

    manifest_payload = {
        "schema_version": 1,
        "model_profile_id": "prof_x",
        "dataset_id": "ds_x",
        "status": "draft",
        "feature_bindings": [{
            "feature_id": "f1",
            "raw_columns": ["col_a"],
            "source_role": "manual_input",
            "unit": "%",
            "review_status": "approved",
            "approved_by": "reviewer-alice",
            "approved_at": "2026-08-29T12:00:00Z",
        }],
    }
    save_profile_manifest("prof_x", manifest_payload, tmp_path)
    loaded_manifest = load_profile_manifest("prof_x", tmp_path)
    assert loaded_manifest["model_profile_id"] == "prof_x"
    assert loaded_manifest["feature_bindings"][0]["feature_id"] == "f1"
    assert "manifest_hash" in loaded_manifest
    assert loaded_manifest["manifest_hash"] == compute_dataset_manifest_hash(loaded_manifest)

    suggestions_payload = [{
        "feature_id": "f_prop",
        "raw_columns": ["col_b"],
        "source_role": "manual_input",
        "status": "pending_review",
        "is_new_proposal": True,
    }]
    save_profile_suggestions("prof_x", suggestions_payload, tmp_path)
    loaded_suggestions = load_profile_suggestions("prof_x", tmp_path)
    assert len(loaded_suggestions) == 1
    assert loaded_suggestions[0]["feature_id"] == "f_prop"
    assert loaded_suggestions[0]["is_new_proposal"] is True


def test_feature_registry_ui_import_does_not_raise_name_error():
    import core.feature_registry_ui as ui_module

    assert hasattr(ui_module, "PortalAIError")
    from core.portal_ai import PortalAIError as ImportedError
    assert ui_module.PortalAIError is ImportedError


def test_format_feature_review_error_translates_portal_ai_stages():
    from core.feature_registry_ui import format_feature_review_error
    from core.portal_ai import PortalAIAuthError, PortalAIParseError, PortalAITransientError

    auth = PortalAIAuthError(
        "认证失败（HTTP 401）", stage="authentication", service_id="svc-a",
        status_code=401, suggestion="请检查 API Key。",
    )
    result = format_feature_review_error(auth)
    assert "认证失败" in result["title"]
    assert "401" in result["title"]
    assert "svc-a" in result["title"]
    assert "API Key" in result["suggestion"]

    transient = PortalAITransientError("网络失败", stage="transient_network", service_id="svc-a")
    result = format_feature_review_error(transient)
    assert "网络或代理失败" in result["title"]

    parse = PortalAIParseError("模型返回非 JSON", stage="json_content_parsing", service_id="svc-a")
    result = format_feature_review_error(parse)
    assert "JSON" in result["title"] or "JSON" in result["detail"]


def test_format_feature_review_error_value_error_and_unknown():
    from core.feature_registry_ui import format_feature_review_error

    result = format_feature_review_error(ValueError("feature review status is invalid: approved"))
    assert "校验失败" in result["title"]
    assert "status is invalid" in result["detail"]

    result = format_feature_review_error(RuntimeError("boom"))
    assert "未知程序异常" in result["title"]


def test_format_feature_review_error_never_leaks_api_key():
    from core.feature_registry_ui import format_feature_review_error
    from core.portal_ai import PortalAIError

    secret = "sk-secret-value-987654"
    exc = PortalAIError(f"failed with key {secret}", stage="http_error", raw_excerpt=f"body with {secret}")
    result = format_feature_review_error(exc)
    combined = f"{result['title']} {result['detail']} {result['suggestion']}"
    assert secret not in combined


def test_format_feature_review_error_truncates_long_responses():
    from core.feature_registry_ui import format_feature_review_error
    from core.portal_ai import PortalAIError

    long_body = "x" * 2000
    exc = PortalAIError("failed", stage="http_response_json", raw_excerpt=long_body)
    result = format_feature_review_error(exc)
    assert len(result["detail"]) <= 600


def test_format_feature_review_error_source_role_shows_contract_not_network():
    from core.feature_registry_ui import format_feature_review_error

    exc = ValueError("feature review source_role is invalid")
    result = format_feature_review_error(exc)
    assert "来源类型不符合契约" in result["title"]
    assert "已转为冲突建议" in result["suggestion"] or "人工审核" in result["suggestion"]
    # Must not be presented as network / key failure
    assert "网络" not in result["title"]
    assert "认证" not in result["title"]
