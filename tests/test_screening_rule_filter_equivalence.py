"""锁定化学规则过滤的向量化：必须与**原始实现**逐位一致。

实测瓶颈（用户截图场景：树脂 8357 × 固化剂 5481，工艺网格 48）：
    filter_candidates_by_epoxy_rules: n=1,000,000 行 iterrows = 35.5 s
        → 其中唯一分子 RDKit 计算仅 0.004 s，99.99% 是 pandas 逐行开销
        → 外推 max_formulations=5e6 ≈ 177 s，且是**全量前置阻塞**
    filter_candidates_by_size:      n=200,000 行 iterrows = 120 s

修复：按**唯一组合/SMILES** 判定后映射回行（同一分子对被多个工艺网格重复引用）。

⚠️ 等价性基准说明（重要）
------------------------
参考实现从 **git HEAD 的原始源码自动抽取**（`tests/_epoxy_rules_reference.py`），
**不是**手工重写。原因：本任务中曾凭记忆重写 `_passes_resin`/`_passes_hardener`，
漏掉 `ban_strong_acids` / `ban_amines` / `min_active_hydrogen` /
`reject_mixed_class` / `class_flags` 等分支，导致过滤结果从 1853 条错误地变成 0 条。
用原码做基准才能真正锁住等价性。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# tests/ 不是包，按文件路径导入参考实现
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _epoxy_rules_reference import (  # noqa: E402
    filter_candidates_by_epoxy_rules as reference_filter,
)

RESINS = [
    "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",  # DGEBA
    "c1ccc(OCC2CO2)cc1",  # 单官能环氧
    "OCC1CO1",  # 脂肪族小分子
    "Nc1ccc(Cc2ccc(N)cc2)cc1",  # 胺（树脂规则应拒）
    "CCCC",  # 无环氧
    "O=C1OC(=O)c2ccccc21",  # 酸酐
    "NCCN",  # 脂肪胺
    "c1ccccc1O",  # 酚
]
HARDENERS = [
    "Nc1ccc(Cc2ccc(N)cc2)cc1",  # DDM
    "NCCN",  # 乙二胺
    "O=C1OC(=O)c2ccccc21",  # 邻苯二甲酸酐
    "Nc1ccccc1",  # 苯胺
    "CS",  # 硫醇
    "CCN(CC)CC",  # 叔胺
    "c1ccncc1",  # 吡啶
    "CCCC",  # 烷烃
]

RULE_CASES = [
    None,
    {
        "global": {"reject_charged": True},
        "resin": {"min_epoxide": 2, "min_aromatic_rings": 1, "ban_amines": True},
        "hardener": {
            "ban_epoxide": True,
            "allowed_classes": [
                "amine",
                "anhydride",
                "phenol",
                "thiol",
                "imidazole",
                "tertiary_amine",
            ],
            "min_active_hydrogen": 1,
        },
    },
    {"resin": {"min_epoxide": 1}, "hardener": {}},
    {
        "resin": {"min_mw": 150.0, "max_mw": 2000.0},
        "hardener": {"min_mw": 40.0, "max_mw": 1000.0},
    },
    {
        "resin": {"min_epoxide": 1},
        "hardener": {"min_active_hydrogen": 2, "ban_strong_acids": True},
    },
    {"pair": {"reject_mixed_class": False}, "hardener": {"allowed_classes": ["amine"]}},
    {
        "resin": {"ban_strong_acids": True, "ban_amines": True},
        "hardener": {"allowed_classes": ["anhydride", "phenol", "thiol"]},
    },
]


def _frame(seed: int, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "resin_smiles": rng.choice(RESINS, n),
            "hardener_smiles": rng.choice(HARDENERS, n),
        }
    )


@pytest.mark.parametrize("case_index", range(len(RULE_CASES)))
@pytest.mark.parametrize("keep_invalid", [False, True])
def test_epoxy_rules_match_original_implementation(case_index, keep_invalid):
    """向量化结果必须与原始实现逐位一致（条数 + 内容 + 顺序）。"""
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    df = _frame(case_index)
    rules = RULE_CASES[case_index]

    got = filter_candidates_by_epoxy_rules(
        df.copy(), rules=rules, keep_invalid=keep_invalid
    )
    expected = reference_filter(
        df.copy(), rules=rules, keep_invalid=keep_invalid
    )
    assert len(got) == len(expected), (
        f"条数不一致：向量化 {len(got)} vs 原始 {len(expected)}"
    )
    assert got.equals(expected), "内容不一致"


def test_epoxy_rules_strict_profile_is_not_empty():
    """防回归：严格角色规则下不得把全部候选过滤成 0 条。

    历史上曾因重构漏掉判定分支，把 1853 条错误地过滤成 0 条。
    """
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    df = _frame(1, n=5000)
    strict = RULE_CASES[1]
    out = filter_candidates_by_epoxy_rules(df.copy(), rules=strict)
    assert len(out) > 0, "严格规则下不应把候选全部清空"
    assert len(out) == len(reference_filter(df.copy(), rules=strict))

    # DGEBA + DDM 必须通过严格规则（经典可用配方）
    dgeba_ddm = pd.DataFrame(
        {"resin_smiles": [RESINS[0]], "hardener_smiles": [HARDENERS[0]]}
    )
    assert len(filter_candidates_by_epoxy_rules(dgeba_ddm.copy(), rules=strict)) == 1


def test_epoxy_rules_no_columns_returns_input():
    """缺少树脂/固化剂列时原样返回（不报错、不丢数据）。"""
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    df = pd.DataFrame({"other": [1, 2, 3]})
    assert filter_candidates_by_epoxy_rules(df) is df


def test_epoxy_rules_is_much_faster_than_rowwise():
    """性能上界：100,000 行必须远快于逐行实现（实测 35s → <1s）。"""
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    df = _frame(1, n=100_000)
    start = time.perf_counter()
    filter_candidates_by_epoxy_rules(df.copy())
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"向量化后仍过慢：{elapsed:.2f}s"


def test_epoxy_rules_reuses_judgement_for_duplicate_pairs():
    """同一分子对重复出现时必须复用判定（这是提速的来源）。"""
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    df = pd.DataFrame(
        {
            "resin_smiles": [RESINS[0], RESINS[1]] * 25_000,
            "hardener_smiles": [HARDENERS[0], HARDENERS[1]] * 25_000,
        }
    )
    start = time.perf_counter()
    out = filter_candidates_by_epoxy_rules(df.copy())
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"唯一组合复用未生效：{elapsed:.2f}s"
    assert 0 <= len(out) <= len(df)
    assert len(out) == len(reference_filter(df.copy()))


def test_epoxy_rules_handles_empty_and_single_row():
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    empty = pd.DataFrame({"resin_smiles": [], "hardener_smiles": []})
    assert len(filter_candidates_by_epoxy_rules(empty.copy())) == 0

    single = pd.DataFrame(
        {"resin_smiles": [RESINS[0]], "hardener_smiles": [HARDENERS[0]]}
    )
    assert filter_candidates_by_epoxy_rules(single.copy()).equals(
        reference_filter(single.copy())
    )


def test_epoxy_rules_only_resin_column():
    """只有树脂列时也必须与原始实现一致。"""
    from core.virtual_screening import filter_candidates_by_epoxy_rules

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"resin_smiles": rng.choice(RESINS, 500)})
    got = filter_candidates_by_epoxy_rules(df.copy(), hardener_col=None)
    expected = reference_filter(df.copy(), hardener_col=None)
    assert got.equals(expected)


# ---------------------------------------------------------------------------
# filter_candidates_by_size 同样向量化
# ---------------------------------------------------------------------------
def test_size_filter_matches_rowwise_reference():
    """尺寸过滤向量化后必须与逐行参考逐位一致。"""
    from core.virtual_screening import calc_mol_basic_stats, filter_candidates_by_size

    df = _frame(0, n=4000)
    rc = {"min_heavy_atoms": 5, "max_mol_wt": 2000}
    hc = {"min_mol_wt": 30}

    got = filter_candidates_by_size(
        df.copy(), resin_constraints=rc, hardener_constraints=hc
    )

    def _passes(smiles, constraints):
        if not constraints:
            return True
        stats = calc_mol_basic_stats(smiles)
        if stats is None:
            return False
        if "min_heavy_atoms" in constraints and stats["heavy_atoms"] < constraints["min_heavy_atoms"]:
            return False
        if "min_mol_wt" in constraints and stats["mol_wt"] < constraints["min_mol_wt"]:
            return False
        if "max_heavy_atoms" in constraints and stats["heavy_atoms"] > constraints["max_heavy_atoms"]:
            return False
        if "max_mol_wt" in constraints and stats["mol_wt"] > constraints["max_mol_wt"]:
            return False
        return True

    keep = [
        bool(_passes(r, rc) and _passes(h, hc))
        for r, h in zip(df["resin_smiles"], df["hardener_smiles"])
    ]
    expected = df.loc[keep].reset_index(drop=True)
    assert len(got) == len(expected)
    assert got.equals(expected)


def test_size_filter_no_constraints_keeps_all():
    from core.virtual_screening import filter_candidates_by_size

    df = pd.DataFrame({"resin_smiles": RESINS, "hardener_smiles": HARDENERS})
    out = filter_candidates_by_size(
        df.copy(), resin_constraints={}, hardener_constraints={}
    )
    assert len(out) == len(df)


def test_size_filter_is_fast():
    """性能上界：200,000 行远快于逐行（实测 120s → <1s）。"""
    from core.virtual_screening import filter_candidates_by_size

    df = _frame(2, n=200_000)
    start = time.perf_counter()
    filter_candidates_by_size(
        df.copy(),
        resin_constraints={"min_heavy_atoms": 5, "max_mol_wt": 2000},
        hardener_constraints={"min_mol_wt": 30},
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"尺寸过滤仍过慢：{elapsed:.2f}s"
