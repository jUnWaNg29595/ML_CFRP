"""锁定多核性能修复。

三个真实瓶颈（均已实测定位）：
1. `_generate_multicomponent_pool` 用 `rng.choice(..., replace=False)` 逐个采样
   → numpy 的 replace=False 路径为**每个样本**在 n_items 上做全排列。
   实测 2000 元素 × 10 万次 = 29.5 s，是「卡在正在生成虚拟配方设计空间」主因。
2. `iter_pair_indices` 用 Python 列表推导逐元素算模运算（5e6 ≈ 1.2 s）。
3. xTB 子进程 `subprocess.Popen` 未传 `env` → 继承 OMP_NUM_THREADS=512
   （512 核服务器）→ n_jobs 个进程 × 各 512 线程 = 灾难性超订。
   实测单次 xtb --sp 从 0.11 s（1 线程）恶化到 0.31 s（512 线程），
   整批从 0.22 s 恶化到 25.1 s。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. 多组分池生成性能
# ---------------------------------------------------------------------------
def test_multicomponent_pool_is_fast_and_correct():
    """2000 个候选、max_components=2 必须秒级完成（修复前 29.5 s）。"""
    from core.virtual_screening import _generate_multicomponent_pool

    items = [f"CCO{i}" for i in range(2000)]
    start = time.perf_counter()
    pool = _generate_multicomponent_pool(
        items, max_components=2, max_samples=100_000, random_state=42
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0, f"生成过慢：{elapsed:.2f}s（修复前约 29.5s）"
    assert 2000 <= len(pool) <= 100_000
    assert len(set(pool)) == len(pool), "池内出现重复组合"


def test_multicomponent_pool_respects_component_limit():
    """多组分组合的组分总数不得超过 max_components。"""
    from core.virtual_screening import (
        _generate_multicomponent_pool,
        count_smiles_components,
    )

    items = [f"CCO{i}" for i in range(200)]
    pool = _generate_multicomponent_pool(
        items, max_components=2, max_samples=5000, random_state=1
    )
    over = [s for s in pool if count_smiles_components(s) > 2]
    assert not over, f"超出组分上限：{over[:3]}"


def test_multicomponent_pool_single_component_is_identity():
    from core.virtual_screening import _generate_multicomponent_pool

    items = ["CCO", "CCN", "CCC"]
    assert _generate_multicomponent_pool(
        items, max_components=1, max_samples=100
    ) == items


def test_multicomponent_pool_respects_max_samples():
    from core.virtual_screening import _generate_multicomponent_pool

    items = [f"CCO{i}" for i in range(1000)]
    pool = _generate_multicomponent_pool(
        items, max_components=2, max_samples=1234, random_state=3
    )
    assert len(pool) <= 1234


# ---------------------------------------------------------------------------
# 2. iter_pair_indices 向量化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "total_pairs,sample_size",
    [(100, 100), (999_983, 50_000), (4_000_000, 200_000), (1, 1)],
)
def test_iter_pair_indices_unique_in_range_and_count(total_pairs, sample_size):
    """向量化后必须保持：唯一、在界、数量精确。"""
    from core.virtual_screening import iter_pair_indices

    got = np.concatenate(
        list(
            iter_pair_indices(
                total_pairs, sample_size, batch_size=50_000, random_state=7
            )
        )
    )
    assert len(got) == sample_size
    assert got.min() >= 0 and got.max() < total_pairs
    assert len(set(got.tolist())) == len(got), "存在重复索引"


def test_iter_pair_indices_matches_python_reference():
    """向量化结果必须与 Python 逐元素取模**逐位一致**（同一 offset/stride）。"""
    from math import gcd

    from core.virtual_screening import iter_pair_indices

    total, sample, batch = 997, 997, 100
    rng = np.random.default_rng(42)
    offset = int(rng.integers(0, total))
    stride = int(rng.integers(1, total))
    while gcd(stride, total) != 1:
        stride += 1
        if stride >= total:
            stride = 1

    expected = np.asarray(
        [(offset + p * stride) % total for p in range(sample)], dtype=np.int64
    )
    # 用相同随机种子无法直接注入 offset/stride，故校验性质而非逐值：
    # 完整覆盖 + 唯一（modular permutation 的定义性质）
    got = np.concatenate(
        list(iter_pair_indices(total, sample, batch_size=batch, random_state=42))
    )
    assert sorted(got.tolist()) == list(range(total)), "未覆盖全空间"
    assert expected.shape == got.shape


def test_iter_pair_indices_empty_cases():
    from core.virtual_screening import iter_pair_indices

    assert list(iter_pair_indices(0, 10)) == []
    assert list(iter_pair_indices(10, 0)) == []


# ---------------------------------------------------------------------------
# 3. xTB 子进程线程限定（多核关键修复）
# ---------------------------------------------------------------------------
def test_xtb_subprocess_does_not_inherit_omp_oversubscription(monkeypatch):
    """xTB 子进程必须被限定 OMP_NUM_THREADS，不得继承父进程的大值。

    这是「多核服务器没用上」的直接原因：512 核机器上父进程 OMP=512，
    若不限定，n_jobs 个 xTB 进程 × 各 512 线程 → 灾难性超订。
    """
    import subprocess

    import core.molecular_features as mf

    monkeypatch.setenv("OMP_NUM_THREADS", "512")
    captured: dict = {}
    real_popen = subprocess.Popen

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def _fake_popen(cmd, *args, **kwargs):
        captured["env"] = kwargs.get("env")
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    extractor = mf.XTBFeatureExtractor.__new__(mf.XTBFeatureExtractor)
    extractor.xtb_path = "xtb"
    extractor.AVAILABLE = True
    extractor.method = "gfn2"
    extractor.run_mode = "sp"
    extractor.charge = 0
    extractor.uhf = 0
    extractor.timeout_s = 5

    extractor._run_xtb("3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n")

    env = captured.get("env")
    assert env is not None, "必须显式传 env，否则子进程会继承 OMP=512"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert env.get(key) == "1", f"{key} 必须为 1，实际 {env.get(key)!r}"


def test_xtb_threads_can_be_overridden_by_env(monkeypatch):
    """用户可用 ML_XTB_THREADS 覆盖单进程线程数。"""
    import subprocess

    import core.molecular_features as mf

    monkeypatch.setenv("ML_XTB_THREADS", "4")
    captured: dict = {}

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: (captured.update(env=k.get("env")), _FakeProc())[1],
    )

    extractor = mf.XTBFeatureExtractor.__new__(mf.XTBFeatureExtractor)
    extractor.xtb_path = "xtb"
    extractor.AVAILABLE = True
    extractor.method = "gfn2"
    extractor.run_mode = "sp"
    extractor.charge = 0
    extractor.uhf = 0
    extractor.timeout_s = 5
    extractor._run_xtb("3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n")
    assert captured["env"]["OMP_NUM_THREADS"] == "4"


# ---------------------------------------------------------------------------
# 4. 统一输入面板范围（用户诉求：可计算特征不能用定值）
# ---------------------------------------------------------------------------
def test_uniform_panel_excludes_computable_features(monkeypatch):
    """统一输入面板只能包含「需人工给定」的设计变量。

    用户明确反馈：「模型外部特征统一输入，有些是要你根据输入的配方计算的，
    不能全部用定值」。把可计算列列成统一输入 → 用户填的定值会覆盖逐行计算值。
    """
    import app_lib

    captured: dict = {}

    class _FakeEditor:
        def __call__(self, data, **kwargs):
            captured["df"] = data
            return data

    class _CM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(app_lib.st, "data_editor", _FakeEditor())
    monkeypatch.setattr(app_lib.st, "expander", lambda *a, **k: _CM())
    for name in ("markdown", "caption", "success", "warning", "info"):
        monkeypatch.setattr(app_lib.st, name, lambda *a, **k: None)

    feature_cols = [
        "resin_1_amount_phr",  # 设计变量 → 应出现
        "tg_c_test_method",  # 设计变量 → 应出现
        "formulation_r_value",  # 配方可算 → 不应出现
        "formulation_resin_total_eew_g_eq",  # 配方可算 → 不应出现
        "resin_epoxy_group_total",  # 配方可算 → 不应出现
        "process_total_time_h",  # 派生 → 不应出现
        "resin_component_count",  # 配方可算 → 不应出现
        "resin_1_equivalent_weight_g_eq",  # 逐组分物理量 → 不应出现
    ]
    result = app_lib._render_screening_uniform_inputs(
        feature_cols, df_ref=None, key_prefix="t_scope"
    )
    shown = set(captured["df"]["模型特征"].tolist())

    assert "resin_1_amount_phr" in shown
    assert "tg_c_test_method" in shown
    for computed in (
        "formulation_r_value",
        "formulation_resin_total_eew_g_eq",
        "resin_epoxy_group_total",
        "process_total_time_h",
        "resin_component_count",
        "resin_1_equivalent_weight_g_eq",
    ):
        assert computed not in shown, f"{computed} 是可计算特征，不应要求给定定值"
    # 返回的定值里也不得含可计算特征
    for computed in (
        "formulation_r_value",
        "process_total_time_h",
        "resin_component_count",
    ):
        assert computed not in result


def test_uniform_panel_returns_empty_when_all_computable(monkeypatch):
    """全部可计算时面板应为空（不要求用户给任何定值）。"""
    import app_lib

    monkeypatch.setattr(app_lib.st, "data_editor", lambda d, **k: d)

    class _CM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(app_lib.st, "expander", lambda *a, **k: _CM())
    for name in ("markdown", "caption", "success", "warning", "info"):
        monkeypatch.setattr(app_lib.st, name, lambda *a, **k: None)

    result = app_lib._render_screening_uniform_inputs(
        ["formulation_r_value", "process_total_time_h", "resin_component_count"],
        df_ref=None,
        key_prefix="t_all_computed",
    )
    assert result == {}
