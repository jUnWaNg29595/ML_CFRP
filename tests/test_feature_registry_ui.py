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
