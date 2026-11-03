import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from core.molecular_features import QuickForceFieldFeatureExtractor, XTBFeatureExtractor
from core.epoxy_mechanism_features import EpoxyMechanismEngine


def test_quick_ff_decomposition_and_multiconf():
    """测试 MMFF 能量分项与多构象初筛"""
    extractor = QuickForceFieldFeatureExtractor(ff_mode="auto", max_iters=50)
    # 测试常规有机小分子（阿司匹林结构）
    df, valid = extractor.smiles_to_ff_features(["CC(=O)OC1=CC=CC=C1C(=O)O"])
    assert len(df) == 1
    assert valid == [0]

    # 验证基础特征存在
    for col in ["ff_energy", "ff_energy_per_atom", "ff_used_mmff", "ff_n_atoms", "ff_success"]:
        assert col in df.columns
        assert not pd.isna(df[col].iloc[0])

    # 验证 MMFF 分项能量与 per_atom 存在且合理
    decomp_terms = ["bond", "angle", "stretchbend", "oop", "torsion", "vdw", "ele"]
    for term in decomp_terms:
        col = f"ff_energy_{term}"
        col_pa = f"ff_energy_{term}_per_atom"
        assert col in df.columns
        assert col_pa in df.columns
        assert not pd.isna(df[col].iloc[0])
        assert not pd.isna(df[col_pa].iloc[0])

    # 验证分项之和约等于总能量
    sum_terms = sum(df[f"ff_energy_{t}"].iloc[0] for t in decomp_terms)
    assert np.isclose(sum_terms, df["ff_energy"].iloc[0], rtol=1e-3)


def test_xtb_cdft_reactivity_and_cache(monkeypatch):
    """测试 xTB 提取器的 CDFT 反应活性衍生特征及持久化缓存"""
    extractor = XTBFeatureExtractor(xtb_path="xtb")
    monkeypatch.setattr(extractor, "AVAILABLE", True)

    # Mock _run_xtb 输出
    def mock_run_xtb(xyz_text, timeout_s=None):
        return {
            "total_energy": -40.123,
            "homo": -8.5,
            "lumo": -1.5,
            "gap": 7.0,
            "dipole": 2.34,
        }

    monkeypatch.setattr(extractor, "_run_xtb", mock_run_xtb)
    # Mock embed_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem
    def mock_embed(mol):
        m = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m, randomSeed=42)
        return m
    monkeypatch.setattr(extractor, "_embed_mol", mock_embed)

    feats = extractor._calc_features("CCO")
    assert feats is not None
    assert feats["xtb_total_energy"] == -40.123
    assert feats["xtb_homo"] == -8.5
    assert feats["xtb_lumo"] == -1.5
    assert feats["xtb_gap"] == 7.0

    # CDFT 计算公式校验:
    # eta = (LUMO - HOMO)/2 = (-1.5 - (-8.5))/2 = 3.5
    # softness = 1/(2*3.5) = 1/7.0
    # chi = -(HOMO + LUMO)/2 = -(-10.0)/2 = 5.0
    # mu = -chi = -5.0
    # omega = mu^2 / (2*eta) = 25.0 / 7.0
    assert np.isclose(feats["xtb_chemical_hardness"], 3.5)
    assert np.isclose(feats["xtb_chemical_softness"], 1.0 / 7.0)
    assert np.isclose(feats["xtb_electronegativity"], 5.0)
    assert np.isclose(feats["xtb_chemical_potential"], -5.0)
    assert np.isclose(feats["xtb_electrophilicity_index"], 25.0 / 7.0)

    # 验证缓存生效 (第二次直接走 cache)
    call_count = {"count": 0}
    def mock_run_xtb_counted(xyz_text, timeout_s=None):
        call_count["count"] += 1
        return mock_run_xtb(xyz_text, timeout_s)
    monkeypatch.setattr(extractor, "_run_xtb", mock_run_xtb_counted)

    # 再次计算相同分子，应该命中缓存不再调用 mock_run_xtb_counted
    feats_cached = extractor._calc_features("CCO")
    assert feats_cached is not None
    assert call_count["count"] == 0


def test_epoxy_mechanism_delta_e_multi_column():
    """测试多组分配方下的 Delta E 多列机制特征衍生"""
    engine = EpoxyMechanismEngine()
    df_formulation = pd.DataFrame([{
        "resin_1_structure": "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1",
        "resin_2_structure": "C1OC1COC1=CC=CC=C1OCC1CO1",
        "curing_agent_1_structure": "NC1=CC=C(CC2=CC=C(N)C=C2)C=C1",
        "curing_agent_2_structure": "NCCN",
        "resin_1_amount_phr": 80.0,
        "resin_2_amount_phr": 20.0,
        "curing_agent_1_amount_phr": 25.0,
        "curing_agent_2_amount_phr": 5.0,
        "resin_1_xtb_lumo": -1.2,
        "resin_2_xtb_lumo": -1.4,
        "resin_1_xtb_homo": -9.5,
        "resin_2_xtb_homo": -9.8,
        "curing_agent_1_xtb_homo": -7.1,
        "curing_agent_2_xtb_homo": -6.8,
        "curing_agent_1_xtb_lumo": 0.8,
        "curing_agent_2_xtb_lumo": 1.2,
        "formulation_r_value": 1.0,
    }])

    enriched = engine.enrich_dataframe(df_formulation)

    # 校验 Delta E 多列架构所有关键字段
    expected_cols = [
        "mech_delta_E_min",
        "mech_delta_E_max",
        "mech_delta_E_span",
        "mech_delta_E_weighted",
        "mech_fmo_delta_e_fwd_primary",
        "mech_fmo_delta_e_rev_primary",
        "mech_fmo_delta_e_fwd_min",
        "mech_fmo_delta_e_fwd_max",
        "mech_fmo_delta_e_fwd_weighted",
        "mech_fmo_delta_e_rev_min",
        "mech_fmo_delta_e_rev_max",
        "mech_fmo_delta_e_rev_weighted",
        "mech_fmo_delta_mu_weighted",
        # 新增 MD 几何与交联产物/交叉项
        "geom_vdw_volume_weighted",
        "geom_rg_weighted",
        "geom_asphericity_weighted",
        "geom_eccentricity_weighted",
        "geom_compactness_weighted",
        "mech_proxy_ffv",
        "post_hbond_donor_density",
        "post_tertiary_amine_density",
        "ff_charge_variance_weighted",
        "cross_torsion_density_index",
        "cross_flexibility_index",
        "cross_fmo_rigidity_index",
    ]

    for col in expected_cols:
        assert col in enriched.columns, f"缺少期望列 {col}"
        val = enriched[col].iloc[0]
        assert not pd.isna(val), f"列 {col} 的值为 NaN"

    # 数值校验：
    # 极值检验
    assert np.isclose(enriched["mech_fmo_delta_e_fwd_min"].iloc[0], 5.4)
    assert np.isclose(enriched["mech_fmo_delta_e_fwd_max"].iloc[0], 5.9)
    # primary 检验 (r1 vs h1)
    assert np.isclose(enriched["mech_fmo_delta_e_fwd_primary"].iloc[0], 5.9)
