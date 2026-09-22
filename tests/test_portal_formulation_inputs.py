"""配方自动推导：从 SMILES + phr 推出 EEW / AHEW / r 等特征。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.3

关键回归
--------
``cp_r_value``（当量重比 AHEW/EEW）**不得**被用作 ``formulation_r_value``
（化学计量比 r）。全表实测：正确公式相关系数 0.9749，cp_r_value 仅 -0.0886。
DGEBA/DDS 100:33 时正确 r≈0.905，而 cp_r_value≈0.365。
"""

import pytest

from core.portal_formulation_inputs import (
    FormulationInputError,
    derive_formulation_features,
    derive_r_value,
    parse_component_smiles,
)

DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
DDS = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
MPDA = "Nc1cccc(N)c1"
# IPDA: 3-氨甲基-3,5,5-三甲基环己胺（脂环胺，4 个活泼氢）
IPDA = "CC1(C)CC(N)CC(C)(CN)C1"
# MTHPA: 甲基四氢苯酐（酸酐，1 个酸酐基团）
MTHPA = "CC1CCC2C(C1)C(=O)OC2=O"


# ---------------------------------------------------------------------------
# 解析与公式
# ---------------------------------------------------------------------------

def test_parse_component_smiles_splits_on_dot():
    """多组分 SMILES 以 '.' 分隔。"""
    assert parse_component_smiles("CC.OO") == ["CC", "OO"]
    assert parse_component_smiles(" CC . OO ") == ["CC", "OO"]
    assert parse_component_smiles("") == []
    assert parse_component_smiles("CC") == ["CC"]


def test_derive_r_value_uses_stoichiometric_ratio_formula():
    """r = (固化剂phr/AHEW)/(树脂phr/EEW)。"""
    # DGEBA EEW=170.2095, DDS AHEW=62.07675, 100:33
    r = derive_r_value(
        resin_phr=100.0, hardener_phr=33.0, resin_eew=170.2095, hardener_ahew=62.07675
    )
    assert r == pytest.approx(0.9048, abs=1e-3)


def test_derive_r_value_returns_none_when_inputs_missing():
    """输入不足必须返回 None，不得猜 0 或均值。"""
    assert derive_r_value(resin_phr=100.0, hardener_phr=33.0, resin_eew=None, hardener_ahew=62.0) is None
    assert derive_r_value(resin_phr=100.0, hardener_phr=33.0, resin_eew=170.0, hardener_ahew=None) is None
    assert derive_r_value(resin_phr=0.0, hardener_phr=33.0, resin_eew=170.0, hardener_ahew=62.0) is None
    assert derive_r_value(resin_phr=100.0, hardener_phr=33.0, resin_eew=0.0, hardener_ahew=62.0) is None


# ---------------------------------------------------------------------------
# 端到端推导
# ---------------------------------------------------------------------------

def test_dgeba_dds_derives_expected_values():
    """DGEBA/DDS 100:33 → EEW≈170.2、AHEW≈62.08、r≈0.905。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
    )

    assert features["formulation_resin_total_eew_g_eq"]["value"] == pytest.approx(170.21, abs=0.05)
    assert features["formulation_hardener_total_ahew_g_eq"]["value"] == pytest.approx(62.08, abs=0.05)
    assert features["formulation_r_value"]["value"] == pytest.approx(0.905, abs=0.002)
    assert features["formulation_r_value"]["origin"] == "derived"
    assert features["formulation_r_value"]["detail"]


def test_cp_r_value_is_never_used_as_r_source():
    """断言 cp_r_value 未被用作 formulation_r_value 来源。

    正确 r≈0.905；cp_r_value（AHEW/EEW）≈0.365。若实现误用后者，本测试失败。
    """
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
    )
    r = features["formulation_r_value"]["value"]
    cp_like = 62.07675 / 170.2095  # ≈0.3647

    assert r == pytest.approx(0.905, abs=0.002)
    assert abs(r - cp_like) > 0.5, f"r 疑似误用了 cp_r_value 口径：{r} vs {cp_like}"


def test_equivalent_ratio_matches_r_value():
    """formulation_resin_hardener_equivalent_ratio 与 r 同源（全表实测 100% 相同）。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
    )
    assert (
        features["formulation_resin_hardener_equivalent_ratio"]["value"]
        == features["formulation_r_value"]["value"]
    )


def test_multicomponent_smiles_counted_correctly():
    """多组分 SMILES（'.' 分隔）计数正确。"""
    features = derive_formulation_features(
        resin_smiles=f"{DGEBA}.{DGEBA}",
        hardener_smiles=f"{DDS}.{MPDA}",
        resin_phr=100.0,
        hardener_phr=33.0,
    )

    assert features["resin_component_count"]["value"] == 2
    assert features["curing_agent_component_count"]["value"] == 2
    # 环氧基数：DGEBA×2 = 4；活泼氢：DDS(4) + mPDA(4) = 8
    assert features["resin_epoxy_group_total"]["value"] == pytest.approx(4.0)
    assert features["curing_agent_active_hydrogen_total"]["value"] == pytest.approx(8.0)


def test_invalid_smiles_raises_not_silent():
    """非法 SMILES 必须显式报错，不得静默产出错误值。"""
    with pytest.raises(FormulationInputError):
        derive_formulation_features(
            resin_smiles="this_is_not_a_smiles((((",
            hardener_smiles=DDS,
            resin_phr=100.0,
            hardener_phr=33.0,
        )


def test_missing_recipe_raises():
    """缺树脂或固化剂 SMILES 必须报错（配方必填）。"""
    with pytest.raises(FormulationInputError):
        derive_formulation_features(
            resin_smiles="", hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
        )
    with pytest.raises(FormulationInputError):
        derive_formulation_features(
            resin_smiles=DGEBA, hardener_smiles="", resin_phr=100.0, hardener_phr=33.0
        )


def test_missing_phr_yields_none_r_with_no_guess():
    """缺 phr → r 不产出（不得猜 0/均值），但 EEW/AHEW 仍可推导。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=0.0, hardener_phr=33.0
    )

    assert "formulation_r_value" not in features
    assert features["formulation_resin_total_eew_g_eq"]["value"] == pytest.approx(170.21, abs=0.05)


def test_phr_sums_and_binder_total():
    """phr 求和与粘料总量正确。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA,
        hardener_smiles=DDS,
        resin_phr=100.0,
        hardener_phr=33.0,
        extra_phr={"reactive_diluent": 10.0, "reactive_toughener": 5.0},
    )

    assert features["resin_total_phr"]["value"] == 100.0
    assert features["curing_agent_total_phr"]["value"] == 33.0
    assert features["formulation_epoxy_binder_total_phr"]["value"] == pytest.approx(148.0)

def test_absent_components_count_zero_and_initiator_false():
    """未提供的组分计数为 0，引发剂标记为 False。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
    )

    for name in (
        "accelerator_component_count",
        "catalyst_component_count",
        "initiator_component_count",
        "other_component_count",
        "reactive_diluent_component_count",
        "reactive_toughener_component_count",
        "small_additive_component_count",
    ):
        assert features[name]["value"] == 0
    assert features["initiator_present"]["value"] is False


def test_all_values_carry_derived_origin_and_detail():
    """每个推导值必须带 origin=derived 与可读依据（UI 要展示推导说明）。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=DDS, resin_phr=100.0, hardener_phr=33.0
    )

    assert features
    for name, record in features.items():
        assert record["origin"] == "derived", name
        assert record["detail"], name
        assert "value" in record, name


def test_anhydride_and_amine_hardeners_both_supported():
    """酸酐与胺类固化剂均可推导（活泼氢/官能团口径由化学引擎决定）。"""
    amine = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=MPDA, resin_phr=100.0, hardener_phr=14.0
    )
    anhydride = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=MTHPA, resin_phr=100.0, hardener_phr=85.0
    )

    assert amine["formulation_hardener_total_ahew_g_eq"]["value"] > 0
    assert anhydride["formulation_hardener_total_ahew_g_eq"]["value"] > 0
    assert amine["formulation_r_value"]["value"] != anhydride["formulation_r_value"]["value"]


def test_aliphatic_amine_ipda_supported():
    """脂环胺（IPDA）也应可推导。"""
    features = derive_formulation_features(
        resin_smiles=DGEBA, hardener_smiles=IPDA, resin_phr=100.0, hardener_phr=25.0
    )
    assert features["curing_agent_active_hydrogen_total"]["value"] == pytest.approx(4.0)
    assert features["formulation_r_value"]["value"] > 0
