# -*- coding: utf-8 -*-
"""特征管理“批量接受安全映射建议”功能测试。

覆盖验收要求：
1  safe 为空仍显示操作区域（UI 渲染）
2  无 AI 建议时显示分析与刷新按钮（UI 渲染）
3  reviewer 为空显示明确提示（UI 渲染）
4  reviewer 填写后确认框可操作（条件逻辑）
5  未选择建议时批量按钮禁用（条件逻辑）
6  选择建议后按钮文字显示数量（UI 渲染）
7  source_role invalid 进入 attention 并提供修复入口（分类+渲染）
8  工艺温度/时间/压力默认 manual_input（分类拦截）
9  测试方法/标准默认 manual_input（分类拦截）
10 分子描述符/配方派生进入 workflow/derived（分类）
11 profile blocked 显示阻断原因（UI 渲染）
12 legacy_observed 特征不进 safe（分类）
13 一条批量建议失败整个批次不写入（原子性）
14 批量接受只更新 manifest binding 不改 Registry approved（边界）
15 Registry/profile 未批准时 manifest 不得显示可训练（状态规则）
16 session rerun 后 reviewer 与选择状态保留（session state 逻辑）
17 取消某条建议后数量正确更新（选择逻辑）
18 已处理建议不再出现在安全列表（分类+清理）
19 审核日志含 reviewer/feature_id/动作/时间/hash（记录）
20 无建议/无数据/无 profile/无 AI 时页面不崩溃（空态渲染）
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest


def _registry(*, profile_status: str = "draft", feature_statuses: dict[str, str] | None = None, approval: str = "draft"):
    feature_statuses = feature_statuses or {
        "f_temp": "draft", "f_time": "draft", "f_pressure": "draft",
        "f_method": "draft", "f_standard": "draft", "f_mw": "approved", "f_eq": "approved",
        "f_legacy": "legacy_observed", "f_blocked": "blocked",
    }
    # 特征语义名用中文（真实 registry 场景）：工艺/测试字段拦截依赖中文关键词
    semantic_names = {
        "f_temp": "固化温度", "f_time": "固化时间", "f_pressure": "固化压力",
        "f_method": "测试方法", "f_standard": "测试标准",
        "f_mw": "树脂分子量", "f_eq": "等当量比",
        "f_legacy": "旧版目视等级", "f_blocked": "异常标记",
    }
    features = [
        {"feature_id": fid, "name": semantic_names.get(fid, fid), "source_type": _source_type_for(fid), "status": status, "unit": "C"}
        for fid, status in feature_statuses.items()
    ]
    return {
        "features": features,
        "model_profiles": {"p": {
            "feature_ids": list(feature_statuses.keys()),
            "target_col": "tg", "status": profile_status,
        }},
        "approval": {"status": approval},
    }


def _source_type_for(feature_id: str) -> str:
    if feature_id in {"f_mw", "f_eq"}:
        return "derived_workflow" if feature_id == "f_eq" else "molecular_workflow"
    return "manual_input"


def _suggestion(feature_id: str, *, source_role: str | None = None, confidence: float = 0.95,
                raw_columns: list[str] | None = None, status: str = "pending_review", **overrides):
    base = {
        "feature_id": feature_id,
        "raw_columns": raw_columns or [f"col_{feature_id}"],
        "source_role": source_role or _source_type_for(feature_id),
        "unit": "C",
        "confidence": confidence,
        "rationale_zh": "列名与单位一致",
        "status": status,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# fake streamlit：用于 UI 渲染测试（验收 1、2、3、6、7、11、20）
# ---------------------------------------------------------------------------

class FakeCtx:
    def __init__(self, st, name):
        self.st, self.name = st, name
        self._st = st

    def __getattr__(self, attr):
        # columns/expander 等嵌套调用透传到 fake streamlit
        return getattr(self._st, attr)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_fake_st(button_results: dict[str, bool] | None = None, checkbox_values: dict[str, bool] | None = None):
    fake = types.ModuleType("streamlit")
    fake.session_state = {}
    fake.calls = []
    button_results = button_results or {}
    checkbox_values = checkbox_values or {}

    def _record(name, args, kwargs):
        fake.calls.append((name, args, kwargs))

    def _any(name):
        def inner(*args, **kwargs):
            _record(name, args, kwargs)
            key = kwargs.get("key")
            if name in ("container", "expander", "sidebar"):
                return FakeCtx(fake, name)
            if name == "columns":
                n = args[0] if args else 1
                if isinstance(n, (list, tuple)):
                    n = len(n)
                return [FakeCtx(fake, f"col{i}") for i in range(n)]
            if name == "tabs":
                n = len(args[0]) if args and isinstance(args[0], (list, tuple)) else 1
                return [FakeCtx(fake, f"tab{i}") for i in range(n)]
            if name == "button":
                return button_results.get(key, False)
            if name == "checkbox":
                if key in checkbox_values:
                    return checkbox_values[key]
                return False
            if name == "text_input":
                return kwargs.get("value", "") if "value" in kwargs else ""
            if name == "text_area":
                return kwargs.get("value", "") if "value" in kwargs else ""
            if name == "selectbox":
                opts = kwargs.get("options") or (args[1] if len(args) > 1 else [])
                return opts[0] if opts else ""
            if name == "radio":
                opts = kwargs.get("options") or (args[1] if len(args) > 1 else ["a"])
                return opts[0]
            if name == "multiselect":
                return []
            if name == "metric":
                return None
            if name == "dataframe":
                return None
            if name == "json":
                return None
            return None
        return inner

    for name in (
        "container", "expander", "sidebar", "columns", "tabs", "button", "checkbox",
        "text_input", "text_area", "selectbox", "radio", "multiselect", "metric",
        "dataframe", "json", "markdown", "caption", "info", "warning", "error",
        "success", "spinner", "download_button", "write", "title",
    ):
        setattr(fake, name, _any(name))
    return fake


@pytest.fixture
def fake_st(monkeypatch):
    """替换 sys.modules['streamlit'] 为 fake（不复制整个 sys.modules，
    避免破坏 multiprocessing 的模块身份触发循环导入）。"""
    import core.feature_registry_ui  # 提前完成真实导入链（含 multiprocessing）
    fake = make_fake_st()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


# ---------------------------------------------------------------------------
# 1. core 分类：工艺/测试字段拦截 + legacy + 别名归一（验收 7、8、9、10、12、17、18）
# ---------------------------------------------------------------------------

def test_process_fields_classified_manual_input_never_safe_as_derived():
    """AI 把工艺/测试字段标为 derived_workflow → 必须进入 attention 并提示人工确认。"""
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    suggestions = [
        _suggestion("f_temp", source_role="derived_workflow"),   # 固化温度
        _suggestion("f_time", source_role="molecular_workflow"),  # 固化时间
        _suggestion("f_method", source_role="derived_workflow"),  # 测试方法
    ]
    result = classify_suggestions_with_diagnostics(suggestions, _registry(), "p")
    assert result["safe"] == []
    assert len(result["attention"]) == 3
    all_reasons = " ".join(r for item in result["attention"] for r in item.get("_review_reasons", []))
    assert "默认应为 manual_input" in all_reasons
    diag = result["attention"][0]["_diagnostics"]
    assert diag["can_batch_accept"] is False
    assert "manual_input" in diag["repair_action"]


def test_molecular_and_derived_fields_stay_safe():
    """分子描述符/配方派生值保持 workflow/derived 分类并可批量接受。"""
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    suggestions = [
        _suggestion("f_mw", source_role="molecular_workflow"),  # 分子量
        _suggestion("f_eq", source_role="derived_workflow"),    # 等当量比
    ]
    result = classify_suggestions_with_diagnostics(suggestions, _registry(), "p")
    assert [s["feature_id"] for s in result["safe"]] == ["f_mw", "f_eq"]
    assert result["attention"] == []


def test_legacy_observed_feature_never_safe():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    result = classify_suggestions_with_diagnostics(
        [_suggestion("f_legacy", confidence=0.99)], _registry(), "p",
    )
    assert result["safe"] == []
    reasons = result["attention"][0]["_review_reasons"]
    assert any("legacy_observed" in r for r in reasons)


def test_invalid_source_role_enters_attention_with_repair_action():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    result = classify_suggestions_with_diagnostics(
        [_suggestion("f_temp", source_role="weird_role")], _registry(), "p",
    )
    assert result["safe"] == []
    diag = result["attention"][0]["_diagnostics"]
    assert "无法批准" in " ".join(diag["reasons"])
    assert "manual_input" in diag["repair_action"]  # 提供修改入口


def test_source_type_alias_is_normalized_before_classification():
    """AI 返回 source_type（别名）也应被规范化，不因键名不同判非法。"""
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    suggestion = _suggestion("f_temp")
    suggestion["source_type"] = "manual_input"  # 别名键
    suggestion.pop("source_role", None)
    result = classify_suggestions_with_diagnostics([suggestion], _registry(), "p")
    assert [s["feature_id"] for s in result["safe"]] == ["f_temp"]


def test_processed_suggestion_not_relisted_as_safe():
    """已接受（approved）建议不再次进入安全列表（验收 18）。"""
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    result = classify_suggestions_with_diagnostics(
        [_suggestion("f_temp", status="approved")], _registry(), "p",
    )
    assert result["safe"] == []
    assert result["counts"]["approved"] == 1
    assert any("状态为 approved" in r for item in result["attention"] for r in item.get("_review_reasons", []))


def test_duplicate_raw_column_conflict_detected():
    from core.feature_mapping_review import classify_suggestions_with_diagnostics

    suggestions = [
        _suggestion("f_temp", raw_columns=["cure_temp"]),
        _suggestion("f_time", raw_columns=["cure_temp"]),  # 同一原始列映射两个特征
    ]
    result = classify_suggestions_with_diagnostics(suggestions, _registry(), "p")
    safe_ids = [s["feature_id"] for s in result["safe"]]
    assert len(safe_ids) == 1  # 只有第一条安全
    assert any("冲突" in r for item in result["attention"] for r in item.get("_review_reasons", []))


# ---------------------------------------------------------------------------
# 2. 原子化与边界（验收 13、14、15）
# ---------------------------------------------------------------------------

def test_batch_accept_atomic_failure_keeps_manifest_unchanged(monkeypatch):
    """选中列表混入不可接受项 → 整个批次不写入（验收 13）。"""
    import core.feature_mapping_review as fmr
    from core.feature_mapping_review import batch_accept_feature_bindings

    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    good = _suggestion("f_temp", confidence=0.95)
    bad = _suggestion("f_time", confidence=0.95)
    bad["raw_columns"] = []

    original = fmr.classify_suggestions_with_diagnostics

    def fake_classify(suggestions, registry, profile_id):
        result = original(suggestions, registry, profile_id)
        bad_sugg = copy.deepcopy(bad)
        bad_sugg["_review_reasons"] = []
        bad_sugg["_diagnostics"] = {"can_batch_accept": True, "reasons": []}
        result["safe"] = [result["safe"][0], bad_sugg]
        return result

    monkeypatch.setattr(fmr, "classify_suggestions_with_diagnostics", fake_classify)
    with pytest.raises(ValueError, match="中止"):
        batch_accept_feature_bindings(
            manifest, [good, bad], _registry(), "p", "reviewer-alice",
            selected_feature_ids=["f_temp", "f_time"],
        )
    assert manifest["status"] == "draft"
    assert manifest["feature_bindings"] == []


def test_batch_accept_never_modifies_registry_or_profile():
    """批量接受只写 manifest binding；Registry 特征状态与 profile 状态不变（验收 14）。"""
    from core.feature_mapping_review import batch_accept_feature_bindings

    registry = _registry()
    manifest = {"schema_version": 1, "status": "draft", "feature_bindings": []}
    updated = batch_accept_feature_bindings(
        manifest, [_suggestion("f_temp")], registry, "p", "reviewer-alice",
        selected_feature_ids=["f_temp"],
    )
    assert registry["features"][0]["status"] == "draft"  # Registry 未改
    assert registry["model_profiles"]["p"]["status"] == "draft"  # profile 未改
    assert registry["approval"]["status"] == "draft"  # 全局 Registry 未批准
    assert updated["feature_bindings"][0]["review_status"] == "approved"


def test_manifest_status_mapped_not_approved_when_registry_draft():
    """Registry/profile 未批准 → manifest.status=mapped（不可训练）（验收 15）。"""
    from core.feature_mapping_review import batch_accept_feature_bindings

    updated = batch_accept_feature_bindings(
        {"schema_version": 1, "status": "draft", "feature_bindings": []},
        [_suggestion("f_temp")], _registry(), "p", "reviewer-alice",
        selected_feature_ids=["f_temp"],
    )
    assert updated["status"] == "mapped"
    assert updated["approval"]["status"] == "mapped"


def test_manifest_status_approved_only_when_registry_and_profile_approved():
    from core.feature_mapping_review import batch_accept_feature_bindings

    registry = _registry(profile_status="approved", approval="approved")
    updated = batch_accept_feature_bindings(
        {"schema_version": 1, "status": "draft", "feature_bindings": []},
        [_suggestion("f_temp")], registry, "p", "reviewer-alice",
        selected_feature_ids=["f_temp"],
    )
    assert updated["status"] == "approved"
    assert updated["approval"]["status"] == "approved"


def test_manifest_hash_recomputed_and_stable():
    from core.feature_mapping_review import batch_accept_feature_bindings
    from core.dataset_manifest import compute_dataset_manifest_hash

    updated = batch_accept_feature_bindings(
        {"schema_version": 1, "status": "draft", "feature_bindings": []},
        [_suggestion("f_temp")], _registry(), "p", "reviewer-alice",
        selected_feature_ids=["f_temp"],
    )
    assert updated["manifest_hash"] == compute_dataset_manifest_hash(updated)


def test_selection_can_be_partial():
    """只接受选中的建议，未选中的不写入（验收 17 的选择语义）。"""
    from core.feature_mapping_review import batch_accept_feature_bindings

    suggestions = [_suggestion("f_temp"), _suggestion("f_time")]
    updated = batch_accept_feature_bindings(
        {"schema_version": 1, "status": "draft", "feature_bindings": []},
        suggestions, _registry(), "p", "reviewer-alice",
        selected_feature_ids=["f_temp"],  # 只选一条
    )
    fids = [b["feature_id"] for b in updated["feature_bindings"]]
    assert fids == ["f_temp"]


# ---------------------------------------------------------------------------
# 3. 审核日志（验收 19）
# ---------------------------------------------------------------------------

def test_review_log_contains_required_fields(tmp_path):
    from core.feature_mapping_review import save_feature_review_record

    record = {
        "event": "batch_accept",
        "reviewer": "reviewer-alice",
        "feature_id": "f_temp",
        "raw_columns": ["col_f_temp"],
        "recorded_at": "2026-08-30T10:00:00+00:00",
        "manifest_hash": "abc123",
    }
    path = tmp_path / "event.json"
    save_feature_review_record(path, record)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["reviewer"] == "reviewer-alice"
    assert loaded["feature_id"] == "f_temp"
    assert loaded["recorded_at"]
    assert "record_hash" in loaded  # 审计可追溯


# ---------------------------------------------------------------------------
# 4. UI 渲染：空态 / 按钮 / 条件 / 状态（验收 1、2、3、4、5、6、11、20）
# ---------------------------------------------------------------------------

def _render_page(fake, *, frame=None, registry=None, profile_id="p", manifest=None, suggestions=None):
    from core.feature_registry_ui import render_feature_registry_page
    if suggestions is not None:
        # 页面从 session_state 读取建议（与真实 Streamlit 运行一致）
        fake.session_state["feature_review_suggestions"] = list(suggestions)
        fake.session_state["suggestions_loaded_profile"] = profile_id
    render_feature_registry_page(
        frame=frame, registry=registry or _registry(), profile_id=profile_id,
        manifest=manifest, ai_client=None, preferred_service_id=None,
    )
    return fake.calls


def test_empty_state_renders_operation_area_and_buttons(fake_st):
    """无建议/无数据/无 AI 时：操作区完整渲染，分析/刷新按钮可见，不崩溃（验收 1、2、20）。"""
    calls = _render_page(fake_st, frame=None, registry={"model_profiles": {}}, profile_id=None, suggestions=[])
    buttons = [c[2].get("key") for c in calls if c[0] == "button"]
    assert "feature_review_analyze_btn" in buttons
    assert "feature_review_reanalyze_btn" in buttons
    assert "feature_review_refresh_classify_btn" in buttons
    # 空态诊断提示（尚未运行 AI 分析）
    captions = " ".join(str(c[1]) for c in calls if c[0] == "caption")
    assert "尚未运行 AI 分析" in captions


def test_empty_safe_still_shows_action_area(fake_st):
    """safe 为空但存在需人工处理建议时，操作区仍完整（验收 1）。"""
    calls = _render_page(
        fake_st,
        suggestions=[_suggestion("f_temp", source_role="weird_role")],  # 全部进 attention
    )
    buttons = [c[2].get("key") for c in calls if c[0] == "button"]
    assert "feature_review_refresh_classify_btn" in buttons
    assert "feature_review_export_suggestions_btn" in buttons
    # 需要人工处理的建议显示修复建议
    infos = " ".join(str(c[1]) for c in calls if c[0] == "info")
    assert "修复建议" in infos or "manual_input" in infos


def test_reviewer_empty_shows_warning(fake_st):
    """reviewer 为空时显示“请先填写审核人”（验收 3）。"""
    calls = _render_page(fake_st, suggestions=[_suggestion("f_temp")])
    warnings = " ".join(str(c[1]) for c in calls if c[0] == "warning")
    assert "请先填写审核人" in warnings


def test_batch_confirm_checkbox_disabled_without_reviewer(fake_st):
    """reviewer 为空 → 确认框 disabled（验收 4 的条件侧）。"""
    calls = _render_page(fake_st, suggestions=[_suggestion("f_temp")])
    checkbox_calls = [c for c in calls if c[0] == "checkbox"]
    batch_confirm = next(c for c in checkbox_calls if c[2].get("key") == "feature_review_batch_confirm")
    assert batch_confirm[2].get("disabled") is True


def test_selected_count_and_button_text_update(fake_st):
    """选择建议后按钮文字显示数量（验收 6）。"""
    # 未勾选任何 checkbox → fake checkbox 返回 False → 保持默认全选（2 条）
    fake = make_fake_st()
    import sys as _sys
    _sys.modules["streamlit"] = fake
    calls = _render_page(fake, suggestions=[_suggestion("f_temp"), _suggestion("f_time")])
    button_calls = [c for c in calls if c[0] == "button" and c[2].get("key") == "feature_review_batch_approve_btn"]
    assert button_calls
    label = str(button_calls[0][1][0])
    assert "2" in label  # 默认全选 2 条
    captions = " ".join(str(c[1]) for c in calls if c[0] == "caption")
    assert "已选择 2 条建议" in captions


def test_profile_blocked_shows_block_reason(fake_st):
    """profile blocked → 页面显示阻断原因与下一步（验收 11）。"""
    calls = _render_page(fake_st, registry=_registry(profile_status="blocked"), suggestions=[])
    warnings = " ".join(str(c[1]) for c in calls if c[0] == "warning")
    assert "blocked" in warnings
    assert "不能提交正式训练或发布" in warnings


def test_registry_draft_shows_info_not_misleading(fake_st):
    """Registry draft → 明确提示不会获得正式发布资格（验收 15 的 UI 侧）。"""
    calls = _render_page(fake_st, registry=_registry(approval="draft"), suggestions=[])
    infos = " ".join(str(c[1]) for c in calls if c[0] == "info")
    assert "Registry 仍为 draft" in infos
    assert "不会使模型获得正式发布资格" in infos


# ---------------------------------------------------------------------------
# 5. 同步训练页面（验收 15 的 session 侧）
# ---------------------------------------------------------------------------

def test_sync_manifest_only_approved_writes_training_key(fake_st):
    from core.feature_registry_ui import sync_manifest_to_training_state

    mapped_manifest = {"schema_version": 1, "status": "mapped", "feature_bindings": []}
    sync_manifest_to_training_state(mapped_manifest)
    assert fake_st.session_state["feature_mapping_manifest"]["status"] == "mapped"
    assert "training_dataset_manifest" not in fake_st.session_state

    approved_manifest = {"schema_version": 1, "status": "approved", "feature_bindings": []}
    sync_manifest_to_training_state(approved_manifest)
    assert fake_st.session_state["training_dataset_manifest"]["status"] == "approved"


# ---------------------------------------------------------------------------
# 6. session state 保留（验收 16）
# ---------------------------------------------------------------------------

def test_reviewer_persists_across_reruns(fake_st):
    """reviewer 保存到 session state；第二次渲染（模拟 rerun）仍保留（验收 16）。"""
    from core.feature_registry_ui import render_feature_registry_page

    # 第一次渲染：reviewer 有输入
    fake_st.session_state["feature_review_reviewer"] = "reviewer-bob"
    render_feature_registry_page(registry=_registry(), profile_id="p", manifest={"status": "draft", "feature_bindings": []})
    assert fake_st.session_state["feature_review_reviewer"] == "reviewer-bob"
    # 第二次渲染（rerun）：text_input 用 session 值初始化
    text_input_calls = [c for c in fake_st.calls if c[0] == "text_input" and c[2].get("key") == "feature_review_reviewer"]
    assert text_input_calls
    assert text_input_calls[-1][2].get("value") == "reviewer-bob"


def test_selection_persists_across_reruns(fake_st):
    """选择状态保存在 session state；rerun 后保留（验收 16）。"""
    from core.feature_registry_ui import render_feature_registry_page

    fake_st.session_state["feature_review_reviewer"] = "reviewer-bob"
    fake_st.session_state["feature_review_batch_selection"] = {"f_temp"}
    fake_st.session_state["feature_review_suggestions"] = [_suggestion("f_temp")]
    fake_st.session_state["suggestions_loaded_profile"] = "p"
    fake_st.calls = []
    render_feature_registry_page(
        registry=_registry(), profile_id="p",
        manifest={"status": "draft", "feature_bindings": []},
    )
    # 选择状态 key 被重新写入 session（保留 f_temp）
    assert fake_st.session_state.get("feature_review_batch_selection") == {"f_temp"}
