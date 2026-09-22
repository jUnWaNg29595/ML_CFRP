"""AI 输入助手缓存层。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.5

缓存目的
--------
用户反复微调配方时会重复提交相同文本（"E-51/DDS，100:33，180度4小时"）。
命中缓存即复用，不重复消耗 AI 额度。

键的设计
--------
``sha256(service_id | model | prompt_kind | 规范化输入文本)``
—— 不同服务/模型/任务类型**不得串味**，否则会拿到别的模型的结果。

安全约束
--------
缓存**不得**落 API key 或完整敏感输入（沿用 core/portal_tasks.py 的脱敏口径）。
"""

import json
import re
from pathlib import Path

import pytest

from core.portal_ai_cache import (
    PortalAICache,
    cache_key,
    normalize_input_text,
)


@pytest.fixture()
def cache(tmp_path):
    return PortalAICache(root=str(tmp_path), max_entries=5)


# ---------------------------------------------------------------------------
# 键与规范化
# ---------------------------------------------------------------------------

def test_same_input_hits_cache(cache):
    """相同 service/model/prompt_kind/输入 → 命中缓存。"""
    cache.put(service_id="s1", model="m1", prompt_kind="extract", text="E-51/DDS 100:33", value={"a": 1})

    hit = cache.get(service_id="s1", model="m1", prompt_kind="extract", text="E-51/DDS 100:33")

    assert hit is not None
    assert hit["value"] == {"a": 1}


def test_cache_miss_returns_none(cache):
    """未命中返回 None。"""
    assert cache.get(service_id="s1", model="m1", prompt_kind="extract", text="从未见过") is None


def test_different_service_or_model_does_not_collide(cache):
    """不同 service / model / prompt_kind 不得串味。"""
    cache.put(service_id="s1", model="m1", prompt_kind="extract", text="同样的输入", value={"who": "s1m1"})

    assert cache.get(service_id="s2", model="m1", prompt_kind="extract", text="同样的输入") is None
    assert cache.get(service_id="s1", model="m2", prompt_kind="extract", text="同样的输入") is None
    assert cache.get(service_id="s1", model="m1", prompt_kind="other", text="同样的输入") is None


def test_input_normalization_collapses_whitespace_but_keeps_case():
    """空白规范化，但 SMILES 大小写必须保留（大小写有意义）。"""
    assert normalize_input_text("  E-51/DDS   100:33  ") == "E-51/DDS 100:33"
    assert normalize_input_text("c1ccccc1") == "c1ccccc1"
    # 大小写不同 → 不同键
    assert cache_key(service_id="s", model="m", prompt_kind="k", text="CCO") != cache_key(
        service_id="s", model="m", prompt_kind="k", text="cco"
    )


def test_whitespace_variants_share_one_cache_entry(cache):
    """仅空白差异的输入应命中同一缓存条目。"""
    cache.put(service_id="s", model="m", prompt_kind="k", text="E-51 / DDS", value={"v": 1})

    assert cache.get(service_id="s", model="m", prompt_kind="k", text="E-51   /   DDS") is not None


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

def test_lru_evicts_beyond_limit(tmp_path):
    """超过上限时淘汰最久未使用条目。"""
    cache = PortalAICache(root=str(tmp_path), max_entries=3)
    for i in range(4):
        cache.put(service_id="s", model="m", prompt_kind="k", text=f"text-{i}", value={"i": i})

    assert cache.size() == 3
    # 最早的 text-0 应被淘汰
    assert cache.get(service_id="s", model="m", prompt_kind="k", text="text-0") is None
    assert cache.get(service_id="s", model="m", prompt_kind="k", text="text-3") is not None


def test_lru_keeps_recently_used_entry(tmp_path):
    """被读取过的条目应保留（LRU 而非 FIFO）。"""
    cache = PortalAICache(root=str(tmp_path), max_entries=3)
    for i in range(3):
        cache.put(service_id="s", model="m", prompt_kind="k", text=f"t{i}", value={"i": i})
    # 触碰 t0，使其成为最近使用
    cache.get(service_id="s", model="m", prompt_kind="k", text="t0")
    cache.put(service_id="s", model="m", prompt_kind="k", text="t3", value={"i": 3})

    assert cache.get(service_id="s", model="m", prompt_kind="k", text="t0") is not None
    assert cache.get(service_id="s", model="m", prompt_kind="k", text="t1") is None


def test_hit_count_increases(cache):
    """命中次数可统计（用于 UI 展示"已缓存 N 次"）。"""
    cache.put(service_id="s", model="m", prompt_kind="k", text="x", value={"v": 1})
    cache.get(service_id="s", model="m", prompt_kind="k", text="x")
    hit = cache.get(service_id="s", model="m", prompt_kind="k", text="x")

    assert hit["hits"] >= 2


# ---------------------------------------------------------------------------
# 安全
# ---------------------------------------------------------------------------

def test_cache_never_stores_api_key(tmp_path):
    """缓存文件不得包含 API key。"""
    cache = PortalAICache(root=str(tmp_path), max_entries=10)
    cache.put(
        service_id="s",
        model="m",
        prompt_kind="k",
        text="配方文本",
        value={"ok": True},
        api_key="sk-super-secret-value-12345",
    )

    blob = ""
    for path in Path(tmp_path).rglob("*.json"):
        blob += path.read_text(encoding="utf-8")
    assert "sk-super-secret-value-12345" not in blob
    # 键里也不能出现
    assert not re.search(r"sk-[a-z0-9-]{10,}", blob, re.I)


def test_corrupt_cache_files_degrade_gracefully(tmp_path):
    """缓存文件损坏时降级（当未命中），不抛异常。"""
    cache = PortalAICache(root=str(tmp_path), max_entries=10)
    cache.put(service_id="s", model="m", prompt_kind="k", text="x", value={"v": 1})
    for path in Path(tmp_path).rglob("*.json"):
        path.write_text("{ corrupt", encoding="utf-8")

    assert cache.get(service_id="s", model="m", prompt_kind="k", text="x") is None


# ---------------------------------------------------------------------------
# 强制重解析
# ---------------------------------------------------------------------------

def test_force_refresh_bypasses_read_but_still_writes(cache):
    """强制重新解析：跳过读取，但结果仍写入缓存。"""
    cache.put(service_id="s", model="m", prompt_kind="k", text="x", value={"old": True})

    assert cache.get(service_id="s", model="m", prompt_kind="k", text="x", force_refresh=True) is None
    # 强制读取不删除旧值（由调用方写入新值覆盖）
    assert cache.get(service_id="s", model="m", prompt_kind="k", text="x") is not None


def test_clear_removes_all_entries(cache):
    """清空缓存。"""
    cache.put(service_id="s", model="m", prompt_kind="k", text="x", value={"v": 1})
    cache.clear()

    assert cache.size() == 0
    assert cache.get(service_id="s", model="m", prompt_kind="k", text="x") is None


def test_persistence_across_instances(tmp_path):
    """缓存跨实例持久（Streamlit 重跑后仍命中）。"""
    first = PortalAICache(root=str(tmp_path), max_entries=10)
    first.put(service_id="s", model="m", prompt_kind="k", text="x", value={"v": 42})

    second = PortalAICache(root=str(tmp_path), max_entries=10)
    hit = second.get(service_id="s", model="m", prompt_kind="k", text="x")

    assert hit is not None
    assert hit["value"] == {"v": 42}


def test_value_is_json_roundtrippable(cache):
    """缓存值经 JSON 往返后仍等值（复杂结构不丢失）。"""
    payload = {"fields": {"resin": "DGEBA", "phr": 100.0}, "list": [1, 2, 3]}
    cache.put(service_id="s", model="m", prompt_kind="k", text="x", value=payload)

    hit = cache.get(service_id="s", model="m", prompt_kind="k", text="x")
    assert hit["value"] == payload
    assert json.loads(json.dumps(hit["value"])) == payload
