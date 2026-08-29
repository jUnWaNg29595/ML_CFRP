# -*- coding: utf-8 -*-
"""特征管理 UI 补全（manifest 导出 / 审核历史 / 提交审批 / 空态健壮性）测试。"""

from __future__ import annotations

import json
from pathlib import Path

from core.feature_registry_ui import (
    build_manifest_download,
    list_review_history,
    submit_manifest_for_approval,
)


def test_build_manifest_download_returns_filename_and_json(tmp_path):
    manifest = {
        "schema_version": 1,
        "dataset_id": "d1",
        "model_profile_id": "p",
        "feature_bindings": [{"feature_id": "x", "raw_columns": ["t"]}],
    }
    result = build_manifest_download("p", manifest, tmp_path)
    assert result is not None
    filename, payload = result
    assert filename == "manifest_p.json"
    parsed = json.loads(payload.decode("utf-8"))
    assert parsed["dataset_id"] == "d1"
    # 原 manifest 未被修改
    assert "approval" not in manifest


def test_build_manifest_download_handles_missing_manifest(tmp_path):
    assert build_manifest_download("p", None, tmp_path) is None
    assert build_manifest_download("p", {}, tmp_path) is None


def test_list_review_history_empty_dir_returns_empty(tmp_path):
    assert list_review_history("p", tmp_path / "not_exist") == []


def test_list_review_history_reads_profile_events(tmp_path):
    root = tmp_path / "reviews"
    root.mkdir()
    (root / "001.json").write_text(
        json.dumps({"profile_id": "p", "event": "local_decision", "action": "accept", "reviewer": "u1"}),
        encoding="utf-8",
    )
    (root / "002.json").write_text(
        json.dumps({"profile_id": "other", "event": "local_decision"}),
        encoding="utf-8",
    )
    (root / "003.json").write_text("not json", encoding="utf-8")
    events = list_review_history("p", root)
    assert len(events) == 1
    assert events[0]["action"] == "accept"


def test_submit_manifest_for_approval_adds_submission_metadata(tmp_path):
    manifest = {"schema_version": 1, "dataset_id": "d1", "feature_bindings": []}
    updated = submit_manifest_for_approval("p", manifest, "reviewer-a", tmp_path)
    assert updated is not None
    approval = updated.get("approval")
    assert approval["submitted"] is True
    assert approval["requested_by"] == "reviewer-a"
    assert approval["submitted_at"]
    # 原 manifest 不被就地修改
    assert "approval" not in manifest


def test_submit_manifest_for_approval_none_manifest(tmp_path):
    assert submit_manifest_for_approval("p", None, "u", tmp_path) is None


def test_submit_then_state_visible_in_download(tmp_path):
    manifest = {"schema_version": 1, "dataset_id": "d1", "feature_bindings": [{"feature_id": "x"}]}
    updated = submit_manifest_for_approval("p", manifest, "u", tmp_path)
    download = build_manifest_download("p", updated, tmp_path)
    assert download is not None
    parsed = json.loads(download[1].decode("utf-8"))
    assert parsed["approval"]["submitted"] is True


# ---------------------------------------------------------------------------
# 空态健壮性（验收单 25：无数据/无 profile/无模型时页面不崩溃）
# 这些由 render_feature_registry_page 的防御逻辑保证；此处验证关键分支的纯函数行为。
# ---------------------------------------------------------------------------

def test_frame_column_names_none_safe():
    from core.feature_registry_ui import frame_column_names
    assert frame_column_names(None) == []
    assert frame_column_names(object()) == []


def test_list_review_history_bad_profile_id(tmp_path):
    root = tmp_path / "reviews"
    root.mkdir()
    (root / "001.json").write_text("{}", encoding="utf-8")
    # 空 profile_id 只匹配空记录，不抛异常
    assert isinstance(list_review_history("", root), list)
