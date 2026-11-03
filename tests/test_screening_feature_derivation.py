"""锁定「可自动计算」的声明必须与实际计算能力一致。

这是防伪测试：core/computable_features 判定某特征「系统可算」，
那么 core/screening_feature_derivation 就必须**真的**能把它算出来。
否则筛选会把这些列按 keep 静默继承模板行 → 用别人的配方物理量打分。

背景（真实踩过的坑）：cp_r_value 是**当量重比**（DGEBA/DDM 100:33 → 0.2912），
而 formulation_r_value 是**化学计量比 r**（同一配方 → 1.1332），
两者语义完全不同。禁止用 cp_r_value 填 formulation_r_value。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
DDM = "Nc1ccc(Cc2ccc(N)cc2)cc1"
EDA = "NCCN"


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resin_smiles": [DGEBA, DGEBA],
            "hardener_smiles": [DDM, EDA],
            "resin_1_amount_phr": [100.0, 100.0],
            "curing_agent_1_amount_phr": [33.0, 13.0],
        }
    )


# ---------------------------------------------------------------------------
# 真实计算能力
# ---------------------------------------------------------------------------
def test_formulation_columns_are_actually_computed():
    """formulation_* 必须被真正算出（而不是留给映射层 keep）。"""
    from core.screening_feature_derivation import (
        derive_computable_candidate_columns,
    )

    features = [
        "formulation_r_value",
        "formulation_resin_total_eew_g_eq",
        "formulation_hardener_total_ahew_g_eq",
        "formulation_resin_hardener_equivalent_ratio",
        "resin_total_phr",
        "resin_component_count",
        "resin_epoxy_group_total",
        "curing_agent_active_hydrogen_total",
        "curing_agent_equivalent_group_total",
    ]
    out, report = derive_computable_candidate_columns(
        _candidate_frame(), feature_cols=features
    )
    filled = set(report["formulation"])
    missing = [f for f in features if f not in filled]
    assert not missing, f"声明可算但实际未计算：{missing}"
    for feature in features:
        assert out[feature].notna().all(), f"{feature} 仍有缺失值"


def test_r_value_is_stoichiometric_not_equivalent_weight_ratio():
    """核心防伪：r 必须是化学计量比，不是 cp_r_value（当量重比）。"""
    from core.portal_formulation_inputs import derive_r_value
    from core.screening_feature_derivation import (
        derive_formulation_summary_columns,
    )

    out, _ = derive_formulation_summary_columns(
        _candidate_frame(),
        feature_cols=["formulation_r_value", "formulation_resin_total_eew_g_eq"],
    )
    r_actual = float(out["formulation_r_value"].iloc[0])
    eew = float(out["formulation_resin_total_eew_g_eq"].iloc[0])

    expected = derive_r_value(
        resin_phr=100.0, hardener_phr=33.0, resin_eew=eew, hardener_ahew=49.56725
    )
    assert r_actual == pytest.approx(expected, rel=1e-6)

    # cp_r_value（当量重比）= ahew/eew ≈ 0.29，与正确 r ≈ 1.13 相差约 4 倍
    cp_r_like = 49.56725 / eew
    assert r_actual == pytest.approx(1.1332, abs=1e-3)
    assert cp_r_like == pytest.approx(0.2912, abs=1e-3)
    assert abs(r_actual - cp_r_like) > 0.5, "r 不得等于当量重比"


def test_component_physics_columns_are_actually_computed():
    """逐组分物理量必须被真正算出。"""
    from core.screening_feature_derivation import (
        derive_computable_candidate_columns,
    )

    features = [
        "resin_1_molecular_weight_g_mol",
        "resin_1_equivalent_weight_g_eq",
        "resin_1_epoxy_group_count",
        "curing_agent_1_molecular_weight_g_mol",
        "curing_agent_1_active_hydrogen_equivalent_count",
        "resin_2_molecular_weight_g_mol",
    ]
    out, report = derive_computable_candidate_columns(
        _candidate_frame(), feature_cols=features
    )
    filled = set(report["component_physics"])
    # resin_2 在候选表里不存在 → 不应被伪造出来
    assert "resin_2_molecular_weight_g_mol" not in filled
    for feature in features[:5]:
        assert feature in filled, f"{feature} 未计算"
        assert out[feature].notna().all()

    # DGEBA 的 MW/EEW 必须与 component_physics 权威口径一致
    assert float(out["resin_1_molecular_weight_g_mol"].iloc[0]) == pytest.approx(
        340.419, abs=1e-2
    )
    assert float(out["resin_1_equivalent_weight_g_eq"].iloc[0]) == pytest.approx(
        170.2095, abs=1e-3
    )


def test_existing_measured_values_are_not_overwritten():
    """已有实测值优先，只补空位（不得覆盖实验数据）。"""
    from core.screening_feature_derivation import (
        derive_formulation_summary_columns,
    )

    frame = _candidate_frame()
    frame["formulation_r_value"] = [0.99, np.nan]
    out, _ = derive_formulation_summary_columns(
        frame, feature_cols=["formulation_r_value"]
    )
    assert float(out["formulation_r_value"].iloc[0]) == pytest.approx(0.99)
    assert np.isfinite(float(out["formulation_r_value"].iloc[1]))


def test_invalid_smiles_does_not_produce_fabricated_values():
    """非法 SMILES 不得产出伪造值（列要么不存在，要么全 NaN；绝不猜 0/均值）。"""
    from core.screening_feature_derivation import (
        derive_formulation_summary_columns,
    )

    frame = pd.DataFrame(
        {
            "resin_smiles": ["this_is_not_a_smiles(((("],
            "hardener_smiles": [DDM],
            "resin_1_amount_phr": [100.0],
            "curing_agent_1_amount_phr": [33.0],
        }
    )
    out, filled = derive_formulation_summary_columns(
        frame, feature_cols=["formulation_r_value"]
    )
    assert "formulation_r_value" not in filled, "非法 SMILES 不得被算成可用值"
    if "formulation_r_value" in out.columns:
        assert pd.isna(out["formulation_r_value"]).all(), "不得产出伪造的 r 值"


def test_noop_when_no_computable_features_requested():
    from core.screening_feature_derivation import (
        derive_computable_candidate_columns,
    )

    out, report = derive_computable_candidate_columns(
        _candidate_frame(), feature_cols=["some_manual_knob"]
    )
    assert report == {"component_physics": [], "formulation": []}
    assert "formulation_r_value" not in out.columns


def test_missing_hardener_phr_does_not_fabricate():
    """缺固化剂 phr 时不得用 0/默认值算出假的 r。"""
    from core.screening_feature_derivation import (
        derive_formulation_summary_columns,
    )

    frame = pd.DataFrame(
        {"resin_smiles": [DGEBA], "hardener_smiles": [DDM], "resin_1_amount_phr": [100.0]}
    )
    out, filled = derive_formulation_summary_columns(
        frame, feature_cols=["formulation_r_value"]
    )
    assert "formulation_r_value" not in filled
    if "formulation_r_value" in out.columns:
        assert pd.isna(out["formulation_r_value"]).all()


# ---------------------------------------------------------------------------
# 与判定模块的一致性
# ---------------------------------------------------------------------------
def test_claimed_computable_formulation_family_is_covered():
    """判定为「配方可算」的真实模型特征，必须落在实际计算覆盖面内。"""
    import json
    from pathlib import Path

    from core.computable_features import (
        SOURCE_FORMULATION,
        classify_computable_features,
    )
    from core.screening_feature_derivation import (
        _COMPONENT_RE,
        _FORMULATION_DERIVED_KEYS,
    )

    config_path = Path("prediction_portal/prediction_config.json")
    if not config_path.exists():
        pytest.skip("未找到门户模型配置")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = [
        m
        for m in (
            config.get("materials", {})
            .get("epoxy_resin", {})
            .get("targets", {})
            .get("tg", {})
            .get("models", [])
        )
        if m.get("enabled") and m.get("feature_cols")
    ]
    if not models:
        pytest.skip("没有启用的 Tg 模型")

    covered = set(_FORMULATION_DERIVED_KEYS)
    for verdict in classify_computable_features(list(models[0]["feature_cols"])):
        if verdict.source != SOURCE_FORMULATION:
            continue
        # 逐组分物理量（resin_{i}_* 等）由 component_physics 覆盖
        if _COMPONENT_RE.match(verdict.feature):
            continue
        assert verdict.feature in covered, (
            f"{verdict.feature} 被声明为配方可算，但实际计算覆盖面未包含它"
        )
