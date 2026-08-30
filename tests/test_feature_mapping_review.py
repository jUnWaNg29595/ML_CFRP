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
    # Registry/Profile 已批准，但 profile 仍缺少其他必需绑定，因此只能等待补齐。
    assert updated["status"] == "mapped"
    assert updated["approval"]["status"] == "mapped"
    assert updated["approval"]["mapped_by"] == "reviewer-alice"
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


def test_batch_accept_defaults_to_local_user_when_reviewer_empty():
    from core.feature_mapping_review import batch_accept_feature_bindings

    result = batch_accept_feature_bindings(
        {"status": "draft", "feature_bindings": []},
        [_safe_suggestion()],
        _batch_registry(),
        "p",
        "",
        selected_feature_ids=["f_safe1"],
    )
    assert len(result["feature_bindings"]) == 1
    assert result["feature_bindings"][0]["approved_by"] == "local_user"
    assert result["review_records"][0]["reviewer"] == "local_user"



def test_manifest_status_helper_distinguishes_mapping_and_approval_states():
    from core.feature_mapping_review import compute_feature_manifest_status

    registry = {
        "approval": {"status": "draft"},
        "features": [{"feature_id": "f1", "status": "draft", "source_type": "manual_input"},
                      {"feature_id": "f2", "status": "draft", "source_type": "manual_input"}],
        "model_profiles": {"p": {"feature_ids": ["f1", "f2"], "status": "draft"}},
    }
    empty = {"feature_bindings": []}
    partial = {"feature_bindings": [{"feature_id": "f1", "raw_columns": ["a"], "source_role": "manual_input", "review_status": "approved"}]}
    complete = {"feature_bindings": partial["feature_bindings"] + [{"feature_id": "f2", "raw_columns": ["b"], "source_role": "manual_input", "review_status": "approved"}]}
    assert compute_feature_manifest_status(empty, registry, "p") == "draft"
    assert compute_feature_manifest_status(partial, registry, "p") == "mapped"
    assert compute_feature_manifest_status(complete, registry, "p") == "pending_approval"
    registry["approval"]["status"] = "approved"
    registry["model_profiles"]["p"]["status"] = "approved"
    registry["features"][0]["status"] = "approved"
    registry["features"][1]["status"] = "approved"
    assert compute_feature_manifest_status(complete, registry, "p") == "approved"


def test_batch_accept_is_idempotent_for_same_binding():
    from core.feature_mapping_review import batch_accept_feature_bindings

    manifest = {"status": "draft", "feature_bindings": []}
    suggestion = _safe_suggestion("f_safe1", confidence=0.95)
    first = batch_accept_feature_bindings(manifest, [suggestion], _batch_registry(), "p", "local_user", selected_feature_ids=["f_safe1"])
    second = batch_accept_feature_bindings(first, [suggestion], _batch_registry(), "p", "local_user", selected_feature_ids=["f_safe1"])
    assert len(first["feature_bindings"]) == len(second["feature_bindings"]) == 1
    assert len(first["review_records"]) == len(second["review_records"]) == 1
    assert second["manifest_hash"] == first["manifest_hash"]


def test_aliases_and_process_fields_are_classified_with_diagnostics():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    registry = {"approval": {"status": "draft"}, "features": [{"feature_id": "cure_temp", "name": "固化温度", "source_type": "manual_input", "status": "draft"}], "model_profiles": {"p": {"feature_ids": ["cure_temp"], "status": "draft"}}}
    result = classify_suggestions_with_diagnostics([{"feature_id": "cure_temp", "raw_column": "温度", "source_type": "derived", "status": "pending_review", "confidence": 0.99}], registry, "p")
    assert result["attention"][0]["raw_columns"] == ["温度"]
    assert result["attention"][0]["source_role"] == "manual_input"
    assert result["attention"][0]["source_role_raw"] == "derived"
    assert any("工艺/测试" in reason for reason in result["attention"][0]["_review_reasons"])


def test_registry_source_role_mismatch_never_enters_safe_classification():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    registry = {"features": [{"feature_id": "f1", "name": "分子量", "source_type": "molecular_workflow", "status": "approved"}], "model_profiles": {"p": {"feature_ids": ["f1"], "status": "draft"}}}
    result = classify_suggestions_with_diagnostics([_safe_suggestion("f1", source_role="manual_input")], registry, "p")
    assert result["safe"] == []
    assert any("source_type" in reason or "来源类型" in reason for reason in result["attention"][0]["_review_reasons"])



def test_edit_accept_fixes_unknown_source_role_and_approves():
    from core.feature_mapping_review import apply_feature_review_decision

    # AI 建议最初为非法/unknown 来源类型
    bad_suggestion = {
        "feature_id": "f_safe1",
        "raw_columns": ["col_f_safe1"],
        "source_role": "unknown_or_weird_role",
        "status": "pending_review",
    }
    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    registry = _batch_registry()

    # 用户在人工工作区修改为 manual_input 后点击“接受修改后结果”
    updated = apply_feature_review_decision(
        manifest,
        bad_suggestion,
        action="edit_accept",
        reviewer="local_user",
        registry=registry,
        profile_id="p",
        edited={
            "source_role": "manual_input",
            "raw_columns": ["col_f_safe1"],
            "status": "pending_review",
        },
    )

    assert len(updated["feature_bindings"]) == 1
    assert updated["feature_bindings"][0]["feature_id"] == "f_safe1"
    assert updated["feature_bindings"][0]["source_role"] == "manual_input"
    assert updated["feature_bindings"][0]["review_status"] == "approved"


def test_edit_accept_auto_aligns_unknown_registry_feature_and_unblocks_profile():
    from core.feature_mapping_review import apply_feature_review_decision

    # 模拟目标特征在 Registry 中为 unknown/blocked，所属 profile 也处于 blocked
    registry = {
        "features": [
            {"feature_id": "f_blocked_item", "name": "r_val", "source_type": "unknown", "status": "blocked"},
        ],
        "model_profiles": {
            "p": {
                "feature_ids": ["f_blocked_item"],
                "status": "blocked",
                "blocked_feature_ids": ["f_blocked_item"],
            }
        },
        "approval": {"status": "draft"},
    }
    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    bad_suggestion = {
        "feature_id": "f_blocked_item",
        "raw_columns": ["formulation_r_value"],
        "source_role": "unknown",
        "status": "pending_review",
    }

    # 用户在界面上修改为 manual_input 并点击接受
    updated = apply_feature_review_decision(
        manifest,
        bad_suggestion,
        action="edit_accept",
        reviewer="local_user",
        registry=registry,
        profile_id="p",
        edited={
            "source_role": "manual_input",
            "raw_columns": ["formulation_r_value"],
            "status": "pending_review",
        },
    )

    # 验证：1) 绑定成功写入
    assert len(updated["feature_bindings"]) == 1
    assert updated["feature_bindings"][0]["source_role"] == "manual_input"
    assert updated["feature_bindings"][0]["review_status"] == "approved"

    # 验证：2) Registry 特征的 source_type 和 status 被自动对齐并解除 blocked
    reg_feat = registry["features"][0]
    assert reg_feat["source_type"] == "manual_input"
    assert reg_feat["status"] == "draft"

    # 验证：3) Profile 自动解除 blocked 状态变为正常 draft
    assert registry["model_profiles"]["p"]["status"] == "draft"
    assert registry["model_profiles"]["p"]["blocked_feature_ids"] == []


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


def test_save_manifest_hash_is_stable_when_only_save_metadata_changes(tmp_path):
    from core.feature_mapping_review import load_profile_manifest, save_profile_manifest

    payload = {
        "schema_version": 1,
        "dataset_id": "d",
        "model_profile_id": "p",
        "status": "draft",
        "feature_bindings": [],
    }
    save_profile_manifest("p", payload, tmp_path)
    first = load_profile_manifest("p", tmp_path)
    save_profile_manifest("p", first, tmp_path)
    second = load_profile_manifest("p", tmp_path)
    assert second["manifest_hash"] == first["manifest_hash"]


def test_missing_registry_definition_never_enters_safe_classification():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    registry = {"features": [], "model_profiles": {"p": {"feature_ids": ["missing"], "status": "draft"}}}
    result = classify_suggestions_with_diagnostics([_safe_suggestion("missing")], registry, "p")
    assert result["safe"] == []
    assert any("Registry" in reason or "registry" in reason for reason in result["attention"][0]["_review_reasons"])



def test_fast_local_feature_mapping_exact_and_alias_matches():
    from core.feature_mapping_review import fast_local_feature_mapping

    registry = {
        "features": [
            {"feature_id": "f_temp", "name": "cure_temperature", "label": "固化温度", "aliases": ["固化温度(℃)", "curing_temp"], "source_type": "manual_input"},
            {"feature_id": "f_mw", "name": "resin_mw", "label": "树脂分子量", "aliases": ["分子量"], "source_type": "molecular_workflow"},
        ],
        "model_profiles": {"p": {"feature_ids": ["f_temp", "f_mw"]}},
    }
    cols = ["固化温度(℃)", "分子量", "未知列X"]
    matched, unmapped = fast_local_feature_mapping(cols, registry, "p")
    assert len(matched) == 2
    assert {s["feature_id"] for s in matched} == {"f_temp", "f_mw"}
    assert unmapped == ["未知列X"]
    assert matched[0]["confidence"] == 1.0


def test_request_feature_mapping_review_auto_chunking(monkeypatch):
    from core.feature_mapping_review import request_feature_mapping_review

    chunk_calls = []

    class ChunkingMockClient:
        def review_feature_mapping(self, ctx):
            chunk_calls.append(list(ctx.get("raw_columns") or []))
            return {
                "suggestions": [
                    {"feature_id": f"feat_{col}", "raw_columns": [col], "source_role": "manual_input", "status": "pending_review", "confidence": 0.95}
                    for col in ctx.get("raw_columns") or []
                ],
                "conflicts": [],
                "rationale_zh": "分批测试",
                "confidence": 0.95,
            }

    cols = [f"col_{i}" for i in range(75)]
    context = {"profile_id": "p", "raw_columns": cols, "candidate_features": []}
    resp = request_feature_mapping_review(ChunkingMockClient(), context, batch_size=30)
    assert len(chunk_calls) == 3
    assert len(chunk_calls[0]) == 30
    assert len(chunk_calls[1]) == 30
    assert len(chunk_calls[2]) == 15
    assert len(resp["suggestions"]) == 75


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
    import sys
    import types
    import core.feature_registry_ui as ui

    class _FakeState(dict):
        pass

    fake = _FakeState()
    fake_st = types.ModuleType("streamlit")
    fake_st.session_state = fake
    old_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_st
    try:
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
        if old_st is not None:
            sys.modules["streamlit"] = old_st
        else:
            sys.modules.pop("streamlit", None)


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
