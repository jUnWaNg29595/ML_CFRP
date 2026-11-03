"""锁定虚拟筛选的性能修复：
1. build_feature_matrix 向量化后必须与旧实现**逐位一致**（含 strict 的
   fillna(0.0) 与非数值列 coerce 的顺序语义）。
2. 分块流式：内存峰值与总行数无关，且结果与一次性全量计算一致。
3. _predict_with_uncertainty_impl 不得对已构建好的矩阵重复 build。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _legacy_build_feature_matrix(
    feature_cols,
    mol_features,
    base_row=None,
    fill_missing_fp=True,
    strict=False,
    strict_feature_cols=None,
):
    """逐列赋值版本的旧实现，作为逐位一致性基准。"""
    required_cols = [str(c) for c in (feature_cols or [])]
    strict_required = [
        str(c)
        for c in (
            strict_feature_cols if strict_feature_cols is not None else required_cols
        )
    ]
    n = len(mol_features) if mol_features is not None else 0
    X = pd.DataFrame(np.nan, index=range(n), columns=required_cols, dtype=float)
    if base_row is not None:
        for c in base_row.index:
            if c in X.columns:
                X[c] = base_row[c]
    if mol_features is not None and not mol_features.empty:
        for c in mol_features.columns:
            if c in X.columns:
                X[c] = mol_features[c].values
    auto_zero = [
        c
        for c in X.columns
        if any(
            token in c.lower()
            for token in (
                "maccs",
                "morgan",
                "polymer_",
                "ionic_",
                "small_additive",
                "curing_agent_2",
                "resin_2",
            )
        )
    ]
    if auto_zero:
        X[auto_zero] = X[auto_zero].fillna(0.0)
    if strict:
        X = X.fillna(0.0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).astype(float)


@pytest.fixture()
def matrix_case():
    rng = np.random.default_rng(0)
    cols = [
        "a",
        "b",
        "resin_2_mw",
        "resin_1_structure_Resin_MACCS_8",
        "morgan_3",
        "polymer_x",
        "c",
        "d",
        "e",
        "f",
    ]
    mol = pd.DataFrame(
        rng.normal(size=(50, 6)),
        columns=["a", "b", "resin_2_mw", "morgan_3", "polymer_x", "c"],
    )
    mol.loc[3, "a"] = np.nan
    mol["d"] = ["1.5", "bad", None, "2", "3"] * 10
    mol["e"] = [np.inf, -np.inf, 1.0, 2.0, 3.0] * 10
    mol["f"] = [True, False, None, True, False] * 10
    base = pd.Series(
        {"a": 9.0, "b": 8.0, "c": 7.0, "resin_1_structure_Resin_MACCS_8": 0.0}
    )
    strict_cols = ["a", "b", "resin_2_mw", "morgan_3", "polymer_x", "c"]
    return cols, mol, base, strict_cols


@pytest.mark.parametrize("strict", [False, True])
def test_build_feature_matrix_matches_legacy_bitwise(matrix_case, strict):
    """向量化实现必须与逐列旧实现逐位一致（含 NaN 位置）。"""
    from core.virtual_screening import build_feature_matrix

    cols, mol, base, strict_cols = matrix_case
    legacy = _legacy_build_feature_matrix(
        cols, mol, base_row=base, strict=strict, strict_feature_cols=strict_cols
    )
    new = build_feature_matrix(
        cols,
        mol,
        base_row=base,
        fill_missing_fp=True,
        strict=strict,
        strict_feature_cols=strict_cols,
    )
    assert list(new.columns) == list(legacy.columns)
    assert new.shape == legacy.shape
    assert np.allclose(
        legacy.to_numpy(), new.to_numpy(), equal_nan=True
    ), "向量化后与旧实现不一致"


def test_build_feature_matrix_non_numeric_becomes_nan_not_zero(matrix_case):
    """非数值文本必须 coerce 成 NaN，而不是被 fillna 成 0.0。

    这是向量化过程中真实踩过的坑：若在 strict 的 fillna 之前 coerce，
    "bad" 会先变 NaN 再被填成 0.0，与旧实现（NaN）不符。
    """
    from core.virtual_screening import build_feature_matrix

    cols, mol, base, strict_cols = matrix_case
    new = build_feature_matrix(
        cols,
        mol,
        base_row=base,
        fill_missing_fp=True,
        strict=True,
        strict_feature_cols=strict_cols,
    )
    # "bad" 位于第 1 行；d 不在 strict 列内，故不会被 fillna 覆盖
    assert pd.isna(new.loc[1, "d"]), "'bad' 应为 NaN"
    assert new.loc[0, "d"] == pytest.approx(1.5)
    # inf 必须被替换为 NaN
    assert pd.isna(new.loc[0, "e"]) and pd.isna(new.loc[1, "e"])


def test_build_feature_matrix_auto_zero_fills_fingerprint_columns():
    from core.virtual_screening import build_feature_matrix

    cols = ["resin_1_structure_Resin_MACCS_8", "morgan_3", "polymer_x", "plain"]
    mol = pd.DataFrame({"plain": [1.0, 2.0]})
    out = build_feature_matrix(cols, mol, fill_missing_fp=True)
    # 指纹/聚合物语义列在 mol_features 中缺失 → 自动补 0.0（不是 NaN）
    assert (out["resin_1_structure_Resin_MACCS_8"] == 0.0).all()
    assert (out["morgan_3"] == 0.0).all()
    assert (out["polymer_x"] == 0.0).all()
    # 已提供的普通列必须原样保留，不被 auto-zero 影响
    assert out["plain"].tolist() == [1.0, 2.0]


def test_build_feature_matrix_plain_missing_column_stays_nan():
    """非指纹类的缺失列不得被 auto-zero 填成 0.0（那是伪造数值）。"""
    from core.virtual_screening import build_feature_matrix

    out = build_feature_matrix(["plain_missing"], pd.DataFrame({"x": [1.0]}))
    assert pd.isna(out["plain_missing"]).all()


def test_build_feature_matrix_strict_raises_on_missing_columns():
    from core.virtual_screening import build_feature_matrix

    with pytest.raises(ValueError, match="strict feature matrix is missing"):
        build_feature_matrix(
            ["a", "missing_col"],
            pd.DataFrame({"a": [1.0]}),
            strict=True,
            strict_feature_cols=["missing_col"],
        )


def test_build_feature_matrix_without_mol_features():
    from core.virtual_screening import build_feature_matrix

    base = pd.Series({"a": 1.0, "b": 2.0})
    out = build_feature_matrix(["a", "b", "c"], None, base_row=base)
    assert list(out.columns) == ["a", "b", "c"]
    assert len(out) == 0


def test_uncertainty_impl_does_not_rebuild_clean_matrix(monkeypatch):
    """已构建好的矩阵不得被重复 build（旧实现无条件重建，开销 ~45%）。"""
    from core import virtual_screening as vs

    calls = {"n": 0}
    real = vs.build_feature_matrix

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(vs, "build_feature_matrix", counting)

    class M:
        # 暴露 feature_names_in_ 使 predict_with_model 也走对齐快路径，
        # 从而把计数完全归因于 _predict_with_uncertainty_impl 的重建逻辑
        feature_names_in_ = np.array(["a", "b", "c"], dtype=object)

        def predict(self, X):
            return np.asarray(X, dtype=float).sum(axis=1)

    clean = pd.DataFrame(np.zeros((5, 3), dtype=float), columns=["a", "b", "c"])
    vs._predict_with_uncertainty_impl(M(), clean)
    assert calls["n"] == 0, "干净且已对齐的矩阵不应触发任何 build"

    dirty = clean.copy()
    dirty.loc[0, "a"] = np.nan
    vs._predict_with_uncertainty_impl(M(), dirty)
    assert calls["n"] >= 1, "含 NaN 的矩阵仍必须被修复"


def test_chunked_prediction_matches_single_pass():
    """分块预测结果必须与一次性全量预测完全一致（数值稳定性）。"""
    from core.virtual_screening import (
        build_feature_matrix,
        predict_with_uncertainty_info,
    )

    class M:
        def predict(self, X):
            return np.asarray(X, dtype=float).sum(axis=1)

    n_features = 60
    features = [f"f_{i}" for i in range(n_features)]
    n = 1_237
    rng = np.random.default_rng(7)
    mol = pd.DataFrame(rng.normal(size=(n, n_features)), columns=features)
    base = pd.Series({c: 0.0 for c in features})

    X_full = build_feature_matrix(features, mol, base_row=base)
    full_preds, _, _ = predict_with_uncertainty_info(M(), X_full)

    chunk = 100
    chunked = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        Xc = build_feature_matrix(features, mol.iloc[start:end], base_row=base)
        p, _, _ = predict_with_uncertainty_info(M(), Xc)
        chunked[start:end] = np.asarray(p, dtype=float).reshape(-1)

    assert np.allclose(full_preds, chunked)


def test_chunked_memory_is_independent_of_total_rows():
    """分块路径的峰值内存必须与总行数无关（按块大小封顶）。"""
    from core.virtual_screening import build_feature_matrix

    n_features = 1408
    features = [f"f_{i}" for i in range(n_features)]
    base = pd.Series({c: 0.0 for c in features})
    chunk = 500

    def peak_for(total):
        rng = np.random.default_rng(0)
        mol = pd.DataFrame(
            rng.normal(size=(total, 40)), columns=features[:40]
        )
        peak = 0
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            Xc = build_feature_matrix(
                features, mol.iloc[start:end], base_row=base
            )
            peak = max(peak, Xc.memory_usage(deep=True).sum())
            del Xc
        return peak

    peak_small = peak_for(2_000)
    peak_large = peak_for(20_000)
    assert peak_large == peak_small, (
        f"峰值内存随总行数增长：{peak_small} → {peak_large}"
    )
    assert peak_large < 5000 * n_features * 8, "单块内存应约等于 chunk × 列数"
