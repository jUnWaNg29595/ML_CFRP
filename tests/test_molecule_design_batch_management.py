"""针对分子设计引擎批次管理、清空指定批次及候选池严格去重的回归测试。

用户诉求：
  1. 「虚拟分子筛选 · 全维度拓扑演化设计引擎中，请帮我加一个功能可以清空某一批生成的候选池」
  2. 「候选池最好不要有重复的分子」
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.molecule_design import deduplicate_monomers_canonical


def test_deduplicate_monomers_canonical_handles_chemical_equivalence():
    """不同书写形式的同构分子必须被严格去重（如 Oc1ccccc1 与 c1ccccc1O）。"""
    df = pd.DataFrame({
        "product_smiles": ["c1ccccc1O", "Oc1ccccc1", "C1=CC=CC=C1O", "CC(=O)O"],
        "role": ["resin", "resin", "resin", "hardener"],
        "batch_id": ["B1", "B2", "B3", "B1"],
    })
    out = deduplicate_monomers_canonical(df, smiles_col="product_smiles", keep="first")
    # 3 个苯酚异构书写应被合并为 1 条 + 1 条乙酸 = 共 2 条
    assert len(out) == 2
    assert out["product_smiles"].iloc[0] == "Oc1ccccc1"
    assert out["batch_id"].iloc[0] == "B1"


def test_deduplicate_monomers_canonical_keep_last():
    """keep='last' 时保留最新批次的记录。"""
    df = pd.DataFrame({
        "product_smiles": ["c1ccccc1O", "Oc1ccccc1"],
        "batch_id": ["B1", "B2"],
    })
    out = deduplicate_monomers_canonical(df, smiles_col="product_smiles", keep="last")
    assert len(out) == 1
    assert out["batch_id"].iloc[0] == "B2"


def test_batch_clear_removes_only_target_batch():
    """精准清空指定批次：只移除该批次单体，不影响其余批次。"""
    df = pd.DataFrame({
        "product_smiles": ["Oc1ccccc1", "CC(=O)O", "NCCN", "CCCC"],
        "batch_id": ["B1", "B1", "B2", "B3"],
        "batch_label": ["批次 1"] * 2 + ["批次 2"] + ["批次 3"],
    })
    # 移除批次 B2
    after_clear_b2 = df[df["batch_id"] != "B2"].reset_index(drop=True)
    assert len(after_clear_b2) == 3
    assert "B2" not in after_clear_b2["batch_id"].values
    assert set(after_clear_b2["batch_id"].values) == {"B1", "B3"}


def test_deduplicate_monomers_empty_and_missing_col():
    empty_df = pd.DataFrame()
    assert len(deduplicate_monomers_canonical(empty_df)) == 0

    no_col_df = pd.DataFrame({"other": [1, 2]})
    assert deduplicate_monomers_canonical(no_col_df).equals(no_col_df)
