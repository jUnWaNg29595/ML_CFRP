import copy
import json

import pandas as pd
import pytest


def test_rejected_ai_candidate_does_not_write_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = {"feature_id": "cfrp.tg.pressure", "raw_columns": ["压强"], "source_role": "manual_input", "confidence": 0.96, "rationale_zh": "列名接近但单位未确认"}
    updated = apply_feature_review_decision(manifest, suggestion, "reject", "local-user")
    assert updated["status"] == "draft"
    assert updated["feature_bindings"] == []


def test_accept_action_writes_approved_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    updated = apply_feature_review_decision({"status": "draft", "feature_bindings": []}, {"feature_id": "pressure", "raw_columns": ["pressure_raw"], "source_role": "manual_input", "status": "pending_review", "confidence": 0.8, "rationale_zh": "人工确认"}, "accept", "local-user")
    assert updated["feature_bindings"][0]["feature_id"] == "pressure"
    assert updated["feature_bindings"][0]["review_status"] == "approved"
    assert updated["approval"]["approved_by"] == "local-user"


def test_edit_accept_pending_review_without_edited_status_writes_approved_binding():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {
        "feature_id": "pressure",
        "raw_columns": ["pressure_raw"],
        "source_role": "manual_input",
        "status": "pending_review",
    }
    updated = apply_feature_review_decision(
        {"status": "draft", "feature_bindings": []},
        suggestion,
        "edit_accept",
        "local-user",
        edited={"raw_columns": ["pressure_corrected"]},
    )
    assert updated["feature_bindings"][0]["raw_columns"] == ["pressure_corrected"]
    assert updated["feature_bindings"][0]["review_status"] == "approved"


def test_review_context_is_feature_only_and_bounded():
    from core.feature_mapping_review import build_feature_review_context

    frame = pd.DataFrame({"pressure_raw": [1.0, 2.0], "target": [100, 110], "tg_c": [100, 110]})
    registry = {"model_profiles": {"p": {"feature_ids": ["x"], "target_col": "tg_c", "target": "tg"}}, "features": [{"feature_id": "x", "name": "pressure", "source_type": "manual_input", "label": "压力", "review_secret": "must not leave"}]}
    context = build_feature_review_context(frame, registry, "p")
    assert "pressure_raw" in context["raw_columns"]
    assert "target" not in context["raw_columns"]
    assert "tg_c" not in context["raw_columns"]
    assert all("target" not in row and "tg_c" not in row for row in context["sample_rows"])
    assert context["candidate_features"] == [{"feature_id": "x", "name": "pressure", "source_type": "manual_input", "label": "压力"}]
    assert "metrics" not in context
    assert len(context["sample_rows"]) <= 3


def test_review_client_response_is_structured(monkeypatch):
    from core.feature_mapping_review import request_feature_mapping_review

    class Client:
        def review_feature_mapping(self, context):
            return {"suggestions": [{"feature_id": "pressure", "raw_columns": ["p_raw"], "source_role": "manual_input", "status": "pending_review", "confidence": 0.9, "rationale_zh": "单位一致"}], "conflicts": [], "rationale_zh": "仅供审核", "confidence": 0.9}

    result = request_feature_mapping_review(Client(), {"raw_columns": ["p_raw"]})
    assert result["suggestions"][0]["feature_id"] == "pressure"


def test_review_client_rejects_suggestion_without_explicit_status():
    from core.feature_mapping_review import request_feature_mapping_review

    class Client:
        def review_feature_mapping(self, context):
            return {
                "suggestions": [
                    {
                        "feature_id": "pressure",
                        "raw_columns": ["p_raw"],
                        "source_role": "manual_input",
                    }
                ],
                "conflicts": [],
            }

    with pytest.raises(ValueError, match="status"):
        request_feature_mapping_review(Client(), {"raw_columns": ["p_raw"]})


def test_feature_review_rejects_unbounded_source_role_and_missing_evidence():
    from core.portal_ai_schema import parse_feature_mapping_response

    with pytest.raises(ValueError):
        parse_feature_mapping_response({"suggestions": [{
            "feature_id": "pressure", "raw_columns": ["p_raw"], "source_role": "invented_role",
            "status": "approved", "unit": 123, "rationale_zh": "",
        }], "conflicts": []})


def test_empty_profile_does_not_send_full_registry_and_excludes_target():
    from core.feature_mapping_review import build_feature_review_context
    frame = pd.DataFrame({"pressure_raw": [1.0], "tg_c": [100.0]})
    registry = {"model_profiles": {"p": {"feature_ids": [], "target_col": "tg_c", "status": "approved"}},
                "features": [{"feature_id": "pressure", "name": "pressure", "source_type": "manual_input", "status": "approved"},
                             {"feature_id": "unrelated", "name": "secret", "source_type": "manual_input", "status": "approved"}]}
    context = build_feature_review_context(frame, registry, "p")
    assert [item["feature_id"] for item in context["candidate_features"]] == ["pressure"]


def test_apply_review_rejects_invalid_feature_and_source_role_and_requires_edit_payload():
    from core.feature_mapping_review import apply_feature_review_decision
    manifest = {"status": "draft", "feature_bindings": []}
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "other", "raw_columns": ["x"], "source_role": "manual_input"}, "accept", "u", registry={"features": [{"feature_id": "known"}]})
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "target"}, "accept", "u", registry={"features": [{"feature_id": "known"}]})
    with pytest.raises(ValueError):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input"}, "edit_accept", "u", registry={"features": [{"feature_id": "known"}]})


@pytest.mark.parametrize("status", ["unknown", "conflict", "approved", "draft"])
@pytest.mark.parametrize("action", ["accept", "edit_accept"])
def test_apply_review_rejects_non_pending_suggestion_status(status, action):
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": status}
    edited = {"raw_columns": ["x"]} if action == "edit_accept" else None
    with pytest.raises(ValueError, match="pending_review|状态"):
        apply_feature_review_decision(manifest, suggestion, action, "u", registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]}, edited=edited)
    assert manifest["feature_bindings"] == []


def test_apply_review_rejects_missing_suggestion_status_for_accept_and_edit_accept():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input"}
    manifest = {"status": "draft", "feature_bindings": []}
    registry = {"features": [{"feature_id": "known", "source_type": "manual_input"}]}
    for action, edited in (("accept", None), ("edit_accept", {"raw_columns": ["x"]})):
        with pytest.raises(ValueError, match="pending_review|状态"):
            apply_feature_review_decision(manifest, suggestion, action, "u", registry=registry, edited=edited)
    assert manifest["feature_bindings"] == []


def test_apply_review_requires_explicit_source_role_and_registry_alignment():
    from core.feature_mapping_review import apply_feature_review_decision

    manifest = {"status": "draft", "feature_bindings": []}
    with pytest.raises(ValueError, match="source_role|来源"):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "status": "pending_review"}, "accept", "u", registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]})
    with pytest.raises(ValueError, match="source|来源"):
        apply_feature_review_decision(manifest, {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"}, "accept", "u", registry={"features": [{"feature_id": "known", "source_type": "molecular_workflow"}]})


def test_apply_review_requires_profile_when_registry_is_provided():
    from core.feature_mapping_review import apply_feature_review_decision

    with pytest.raises(ValueError, match="profile"):
        apply_feature_review_decision(
            {"status": "draft", "feature_bindings": []},
            {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"},
            "accept",
            "u",
            registry={"features": [{"feature_id": "known", "source_type": "manual_input"}], "model_profiles": {}},
        )


@pytest.mark.parametrize("registry_status", ["legacy_observed", "blocked", "deprecated", "unknown"])
def test_apply_review_rejects_nonreviewable_registry_feature_status(registry_status):
    from core.feature_mapping_review import apply_feature_review_decision

    registry = {
        "features": [{"feature_id": "known", "source_type": "manual_input", "status": registry_status}],
        "model_profiles": {"p": {"feature_ids": ["known"]}},
    }
    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"}
    with pytest.raises(ValueError, match="status"):
        apply_feature_review_decision({"status": "draft", "feature_bindings": []}, suggestion, "accept", "u", registry=registry, profile_id="p")


def test_apply_review_rejects_nonapproved_edited_status():
    from core.feature_mapping_review import apply_feature_review_decision

    suggestion = {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": "pending_review"}
    with pytest.raises(ValueError, match="pending_review|状态"):
        apply_feature_review_decision(
            {"status": "draft", "feature_bindings": []}, suggestion, "edit_accept", "u",
            registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]},
            edited={"status": "conflict"},
        )


# ============================================================
# source_role approval gate tests
# ============================================================

def test_apply_review_rejects_unknown_source_role():
    from core.feature_mapping_review import apply_feature_review_decision

    with pytest.raises(ValueError, match="source_role"):
        apply_feature_review_decision(
            {"status": "draft", "feature_bindings": []},
            {"feature_id": "known", "raw_columns": ["x"], "source_role": "unknown", "status": "pending_review"},
            "accept",
            "u",
            registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]},
        )


def test_apply_review_rejects_conflict_and_unknown_status():
    from core.feature_mapping_review import apply_feature_review_decision

    for status in ("conflict", "unknown", "rejected"):
        with pytest.raises(ValueError, match="pending_review|状态"):
            apply_feature_review_decision(
                {"status": "draft", "feature_bindings": []},
                {"feature_id": "known", "raw_columns": ["x"], "source_role": "manual_input", "status": status},
                "accept",
                "u",
                registry={"features": [{"feature_id": "known", "source_type": "manual_input"}]},
            )


def test_downgraded_ai_suggestion_cannot_be_approved_until_reclassified():
    """An AI suggestion with unknown source_role must not produce an approved binding."""
    from core.feature_mapping_review import apply_feature_review_decision
    from core.portal_ai_schema import parse_feature_mapping_response

    response = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "known",
            "raw_columns": ["x"],
            "source_role": "weird_role",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "evidence",
        }],
        "conflicts": [],
    })
    sugg = response["suggestions"][0]
    assert sugg["status"] == "conflict"
    with pytest.raises(ValueError, match="pending_review|状态"):
        apply_feature_review_decision(
            {"status": "draft", "feature_bindings": []},
            sugg,
            "accept",
            "u",
            registry={"features": [{"feature_id": "known", "source_type": "manual_input"}],
                      "model_profiles": {"p": {"feature_ids": ["known"]}}},
            profile_id="p",
        )


def test_human_reclassified_downgraded_suggestion_can_be_approved():
    """After the human edits source_role to a valid role, approval works."""
    from core.feature_mapping_review import apply_feature_review_decision
    from core.portal_ai_schema import parse_feature_mapping_response

    response = parse_feature_mapping_response({
        "suggestions": [{
            "feature_id": "known",
            "raw_columns": ["x"],
            "source_role": "weird_role",
            "status": "pending_review",
            "confidence": 0.9,
            "rationale_zh": "evidence",
        }],
        "conflicts": [],
    })
    sugg = dict(response["suggestions"][0])
    # Human reclassification: set status back to pending_review after fixing role
    sugg["source_role"] = "manual_input"
    sugg["status"] = "pending_review"
    updated = apply_feature_review_decision(
        {"status": "draft", "feature_bindings": []},
        sugg,
        "accept",
        "u",
        registry={"features": [{"feature_id": "known", "source_type": "manual_input", "status": "draft"}],
                  "model_profiles": {"p": {"feature_ids": ["known"]}}},
        profile_id="p",
    )
    assert updated["feature_bindings"][0]["review_status"] == "approved"


# ============================================================
# Batch approval workflow tests
# ============================================================

def _batch_registry():
    return {
        "features": [
            {"feature_id": "f_safe1", "name": "f_safe1", "source_type": "manual_input", "status": "draft"},
            {"feature_id": "f_safe2", "name": "f_safe2", "source_type": "molecular_workflow", "status": "approved"},
            {"feature_id": "f_unknown", "name": "f_unknown", "source_type": "manual_input", "status": "draft"},
        ],
        "model_profiles": {"p": {"feature_ids": ["f_safe1", "f_safe2", "f_unknown"], "target_col": "tg", "status": "draft"}},
    }


def _safe_suggestion(fid="f_safe1", confidence=0.9, **overrides):
    base = {
        "feature_id": fid,
        "raw_columns": [f"col_{fid}"],
        "source_role": "manual_input" if fid == "f_safe1" else "molecular_workflow",
        "status": "pending_review",
        "confidence": confidence,
        "rationale_zh": "列名与单位一致",
        "unit": "%",
    }
    base.update(overrides)
    return base


def test_classify_suggestions_split_safe_and_attention():
    from core.feature_mapping_review import classify_feature_suggestions

    suggestions = [
        _safe_suggestion("f_safe1", confidence=0.9),
        _safe_suggestion("f_safe2", confidence=0.6),
        {"feature_id": "f_unknown", "raw_columns": ["c"], "source_role": "unknown", "status": "conflict", "confidence": 0.9},
    ]
    safe, attention = classify_feature_suggestions(suggestions, _batch_registry(), "p")
    assert [s["feature_id"] for s in safe] == ["f_safe1"]
    assert len(attention) == 2


def test_classify_new_proposal_and_low_confidence_need_attention():
    from core.feature_mapping_review import classify_feature_suggestions

    suggestions = [
        _safe_suggestion("f_safe1"),
        _safe_suggestion("not_in_profile", confidence=0.99),
        _safe_suggestion("f_safe2", confidence=0.84),
    ]
    safe, attention = classify_feature_suggestions(suggestions, _batch_registry(), "p")
    assert [s["feature_id"] for s in safe] == ["f_safe1"]
    reasons_all = " ".join(str(r) for item in attention for r in item.get("_review_reasons", []))
    assert "不属于当前 profile" in reasons_all or "新特征提案" in reasons_all
    assert any("低于安全阈值" in r for item in attention for r in item.get("_review_reasons", []))


def test_batch_approve_safe_suggestions_writes_all_bindings():
    from core.feature_mapping_review import batch_approve_safe_feature_suggestions

    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    suggestions = [
        _safe_suggestion("f_safe1", confidence=0.92),
        _safe_suggestion("f_safe2", confidence=0.95),
    ]
    updated = batch_approve_safe_feature_suggestions(manifest, suggestions, _batch_registry(), "p", "reviewer-alice")
    # Registry/profile 仍是 draft → manifest 只能 mapped，不得直接声称 approved（可训练）
    assert updated["status"] == "mapped"
    assert updated["approval"]["status"] == "mapped"
    assert updated["approval"]["mapped_by"] == "reviewer-alice"
    assert len(updated["feature_bindings"]) == 2
    for b in updated["feature_bindings"]:
        assert b["review_status"] == "approved"
        assert b["approved_by"] == "reviewer-alice"
        assert b["feature_id"] in {"f_safe1", "f_safe2"}
    assert "manifest_hash" in updated


def test_batch_accept_upgrades_manifest_only_when_registry_and_profile_approved():
    from core.feature_mapping_review import batch_accept_feature_bindings

    registry = _batch_registry()
    registry["approval"] = {"status": "approved"}
    registry["model_profiles"]["p"]["status"] = "approved"
    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    updated = batch_accept_feature_bindings(
        manifest, [_safe_suggestion("f_safe1", confidence=0.95)], registry, "p", "reviewer-alice",
        selected_feature_ids=["f_safe1"],
    )
    assert updated["status"] == "approved"
    assert updated["approval"]["status"] == "approved"
    # Registry 特征状态保持原样（批量接受不修改 Registry）
    assert registry["features"][0]["status"] == "draft"


def test_batch_accept_atomic_when_selected_contains_invalid(monkeypatch):
    """选中列表中混入不可接受项 → 整个批次不写入，原 manifest 不变。"""
    import core.feature_mapping_review as fmr
    from core.feature_mapping_review import batch_accept_feature_bindings

    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    good = _safe_suggestion("f_safe1", confidence=0.95)
    bad = _safe_suggestion("f_safe2", confidence=0.95)
    bad["raw_columns"] = []  # 写时校验失败

    original = fmr.classify_suggestions_with_diagnostics

    def fake_classify(suggestions, registry, profile_id):
        # 模拟 classify 与写时校验不一致：把坏建议也当作 safe
        result = original(suggestions, registry, profile_id)
        bad_sugg = copy.deepcopy(bad)
        bad_sugg["_review_reasons"] = []
        bad_sugg["_diagnostics"] = {"can_batch_accept": True, "reasons": []}
        result["safe"] = [result["safe"][0], bad_sugg]
        return result

    monkeypatch.setattr(fmr, "classify_suggestions_with_diagnostics", fake_classify)
    with pytest.raises(ValueError, match="中止"):
        batch_accept_feature_bindings(
            manifest, [good, bad], _batch_registry(), "p", "reviewer-alice",
            selected_feature_ids=["f_safe1", "f_safe2"],
        )
    assert manifest["status"] == "draft"
    assert manifest["feature_bindings"] == []


def test_batch_accept_requires_reviewer():
    from core.feature_mapping_review import batch_accept_feature_bindings

    with pytest.raises(ValueError, match="审核人"):
        batch_accept_feature_bindings(
            {"status": "draft", "feature_bindings": []},
            [_safe_suggestion()],
            _batch_registry(),
            "p",
            "",
            selected_feature_ids=["f_safe1"],
        )


def test_batch_accept_requires_at_least_one_selection():
    from core.feature_mapping_review import batch_accept_feature_bindings

    with pytest.raises(ValueError, match="至少选择一条"):
        batch_accept_feature_bindings(
            {"status": "draft", "feature_bindings": []},
            [_safe_suggestion()],
            _batch_registry(),
            "p",
            "reviewer-alice",
            selected_feature_ids=[],
        )


def test_batch_approve_with_no_safe_suggestions_raises():
    from core.feature_mapping_review import batch_approve_safe_feature_suggestions

    with pytest.raises(ValueError, match="至少选择一条"):
        batch_approve_safe_feature_suggestions(
            {"status": "draft", "feature_bindings": []},
            [_safe_suggestion("f_unknown", confidence=0.3)],
            _batch_registry(),
            "p",
            "reviewer-alice",
        )


def test_batch_approve_does_not_modify_registry_feature_status():
    from core.feature_mapping_review import batch_approve_safe_feature_suggestions

    registry = _batch_registry()
    batch_approve_safe_feature_suggestions(
        {"status": "draft", "feature_bindings": []},
        [_safe_suggestion("f_safe1", confidence=0.95)],
        registry,
        "p",
        "reviewer-alice",
    )
    assert registry["features"][0]["status"] == "draft"


# ============================================================
# Training manifest sync & blocker diagnostics tests
# ============================================================

def test_training_manifest_sync_via_session_helper():
    """sync_manifest_to_training_state writes training_dataset_manifest for approved manifests only."""
    import core.feature_registry_ui as ui

    class FakeSession(dict):
        pass

    # We cannot use real streamlit; test via monkeypatched session_state
    import streamlit

    class _FakeState(dict):
        pass

    original = streamlit.session_state
    fake = _FakeState()
    try:
        # Patch module-level usage: function uses st.session_state internally
        streamlit.session_state = fake
        approved_manifest = {"schema_version": 1, "status": "approved", "feature_bindings": [{"feature_id": "a"}]}
        ui.sync_manifest_to_training_state(approved_manifest)
        assert fake["training_dataset_manifest"]["status"] == "approved"
        assert fake["feature_mapping_manifest"]["status"] == "approved"

        fake.clear()
        draft_manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
        ui.sync_manifest_to_training_state(draft_manifest)
        assert "training_dataset_manifest" not in fake
        assert fake["feature_mapping_manifest"]["status"] == "draft"
    finally:
        streamlit.session_state = original


def test_diagnose_training_blockers_reports_missing_manifest():
    from core.training_contract import diagnose_training_blockers

    blockers = diagnose_training_blockers("nonexistent/registry.json", None)
    assert any("manifest 不存在" in b for b in blockers)


def test_diagnose_training_blockers_reports_draft_manifest_and_registry(tmp_path):
    from core.training_contract import diagnose_training_blockers

    registry_path = tmp_path / "feature_registry.json"
    registry_path.write_text(json.dumps({
        "schema_version": 1,
        "registry_version": "test",
        "approval": {"status": "draft"},
        "model_profiles": {"p": {"feature_ids": [], "target_col": "tg", "status": "draft"}},
        "features": [],
    }, ensure_ascii=False), encoding="utf-8")

    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    blockers = diagnose_training_blockers(registry_path, manifest)
    joined = " ".join(blockers)
    assert "manifest 仍为 draft" in joined
    assert "Registry 仍为 draft" in joined


def test_diagnose_training_blockers_reports_conflict_bindings(tmp_path):
    from core.training_contract import diagnose_training_blockers

    registry_path = tmp_path / "feature_registry.json"
    registry_path.write_text(json.dumps({
        "schema_version": 1,
        "registry_version": "test",
        "approval": {"status": "draft"},
        "model_profiles": {"p": {"feature_ids": ["f1"], "target_col": "tg", "status": "draft"}},
        "features": [{"feature_id": "f1", "name": "f1", "source_type": "manual_input", "status": "draft", "unit": "%", "default_policy": "explicit_only"}],
    }, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "draft",
        "model_profile_id": "p",
        "feature_bindings": [{
            "feature_id": "f1",
            "raw_columns": ["col_a"],
            "source_role": "manual_input",
            "unit": "%",
            "review_status": "conflict",
        }],
    }
    blockers = diagnose_training_blockers(registry_path, manifest)
    assert any("conflict" in b for b in blockers)
