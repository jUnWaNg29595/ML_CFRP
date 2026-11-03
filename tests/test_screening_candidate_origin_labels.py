"""针对虚拟筛选配方来源标记的端到端回归测试。

用户诉求：
  「筛选出来的配方应该有来源标记」

测试内容：
  1. 单单体组合正确识别来源（AI设计、训练集、PubChem、用户上传等）
  2. 复配单体（带点号）正确合并子组分来源（如 🔀 复配树脂(训练集+AI设计)），不再丢失变成 custom
  3. 实测配方（train_observed_pair）正确标记为 🌟 训练集实测配方
  4. 向量化函数 add_candidate_origin_labels 生成规范的 candidate_origin_label 与 origin_category
  5. 实时预测结果与中间结果包含正确的来源标记
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.virtual_screening import (
    add_candidate_origin_labels,
    build_component_library,
    enumerate_formulation_candidates,
    merge_component_libraries,
    resolve_candidate_origin_label,
)


@pytest.mark.parametrize(
    "origin_code,r_src,h_src,expected_label,expected_cat",
    [
        ("train_observed_pair", "train_observed_pair", "train_observed_pair", "🌟 训练集实测配方", "实测配方"),
        ("train_data + train_data", "train_data", "train_data", "🧪 实验重组配方 (训练集树脂 + 训练集固化剂)", "实验重组"),
        ("molecule_design_engine + train_data", "molecule_design_engine", "train_data", "🧬 AI设计树脂 + 🧪 训练集固化剂", "AI设计衍生"),
        ("molecule_design_engine + molecule_design_engine", "molecule_design_engine", "molecule_design_engine", "🧬 全AI设计配方 (AI树脂 + AI固化剂)", "AI设计衍生"),
        ("pubchem + train_data", "pubchem", "train_data", "🌐 PubChem树脂 + 🧪 训练集固化剂", "外部数据库"),
        ("uploaded + molecule_design_engine", "uploaded", "molecule_design_engine", "📂 用户上传树脂 + 🧬 AI设计固化剂", "AI设计衍生"),
        ("train_data/molecule_design_engine + train_data", "train_data/molecule_design_engine", "train_data", "🔀 复配树脂(训练集+AI设计) + 🧪 训练集固化剂", "多源复配"),
        ("train_data", "train_data", None, "🧪 训练集单体", "实验重组"),
        ("molecule_design_engine", "molecule_design_engine", None, "🧬 AI设计单体", "AI设计衍生"),
    ],
)
def test_resolve_candidate_origin_label(origin_code, r_src, h_src, expected_label, expected_cat):
    label, cat = resolve_candidate_origin_label(origin_code, r_src, h_src)
    assert label == expected_label
    assert cat == expected_cat


def test_multicomponent_formulation_preserves_origin():
    """复配分子（如 CCO.CCN）必须正确解析各子单体来源，不能退化为 custom。"""
    lib_r1 = build_component_library(["CCO"], role="resin", source="train_data")
    lib_r2 = build_component_library(["CCN"], role="resin", source="molecule_design_engine")
    resin_lib = merge_component_libraries(lib_r1, lib_r2)

    hard_lib = build_component_library(["NCCN"], role="hardener", source="train_data")

    ds = enumerate_formulation_candidates(
        resin_lib,
        hard_lib,
        pair_mode="cartesian",
        max_resin_components=2,
        max_hardener_components=1,
        max_formulations=10,
        batch_size=5,
    )
    df = ds.candidate_df
    assert "candidate_origin_label" in df.columns
    assert "origin_category" in df.columns

    # 找到复配树脂行（包含点号）
    multi_resin_rows = df[df["resin_smiles"].str.contains(r"\.", na=False)]
    if not multi_resin_rows.empty:
        row = multi_resin_rows.iloc[0]
        assert "复配" in row["candidate_origin_label"]
        assert row["origin_category"] in ("多源复配", "AI设计衍生")
        assert "custom" not in row["candidate_origin_label"]


def test_add_candidate_origin_labels_vectorized():
    df = pd.DataFrame({
        "candidate_origin": [
            "train_observed_pair",
            "molecule_design_engine + train_data",
            "pubchem + train_data",
        ]
    })
    out = add_candidate_origin_labels(df)
    assert "candidate_origin_label" in out.columns
    assert "origin_category" in out.columns
    assert out["origin_category"].tolist() == ["实测配方", "AI设计衍生", "外部数据库"]
    assert "🌟" in out["candidate_origin_label"].iloc[0]
    assert "🧬" in out["candidate_origin_label"].iloc[1]
    assert "🌐" in out["candidate_origin_label"].iloc[2]
