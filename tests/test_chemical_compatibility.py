"""
测试分子设计引擎与配方筛选中的化学相容性校验（防止环氧与酚羟基等互反应基团共存）。
"""
import pytest
from rdkit import Chem
from core.molecule_design import (
    validate_monomer_chemical_soundness,
    run_combinatorial_monomer_design,
)
from core.virtual_screening import filter_candidates_by_epoxy_rules
import pandas as pd


def test_validate_monomer_chemical_soundness():
    # 1. 环氧+酚羟基共存（用户提出的典型不合理分子）
    smi_epoxy_phenol = "Oc1ccc(CC2CO2)cc1-c1cc(CC2CO2)ccc1O"
    mol = Chem.MolFromSmiles(smi_epoxy_phenol)
    ok, reason = validate_monomer_chemical_soundness(mol, "resin")
    assert not ok
    assert "酚羟基" in reason

    # 2. 经典标准树脂：DGEBA（纯缩水甘油醚）
    smi_dgeba = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    mol_dgeba = Chem.MolFromSmiles(smi_dgeba)
    ok, _ = validate_monomer_chemical_soundness(mol_dgeba, "resin")
    assert ok

    # 3. 环氧+游离胺基共存（室温自固化胶体）
    smi_epoxy_amine = "Nc1ccc(OCC2CO2)cc1"
    mol_epoxy_amine = Chem.MolFromSmiles(smi_epoxy_amine)
    ok, reason = validate_monomer_chemical_soundness(mol_epoxy_amine, "resin")
    assert not ok
    assert "胺基" in reason

    # 4. 固化剂带环氧基（角色混乱）
    ok, reason = validate_monomer_chemical_soundness(mol_dgeba, "hardener")
    assert not ok
    assert "固化剂角色单体不得含有环氧基" in reason

    # 5. 经典胺固化剂：DDM
    smi_ddm = "Nc1ccc(Cc2ccc(N)cc2)cc1"
    mol_ddm = Chem.MolFromSmiles(smi_ddm)
    ok, _ = validate_monomer_chemical_soundness(mol_ddm, "hardener")
    assert ok


def test_monomer_design_filters_cross_reactive_products():
    """验证包含烯丙基(allyl)/乙烯基(vinyl)修饰的酚醛骨架在衍生时不会产出环氧+酚羟基共存单体。"""
    df, _ = run_combinatorial_monomer_design(
        enable_scaffold_fission=True,
        selected_r_groups=["allyl", "vinyl"],
        min_functionality=2,
        max_sa_score=5.0,
        mw_range=(100, 1000),
        max_total_products=150,
    )
    assert not df.empty

    epoxy_patt = Chem.MolFromSmarts("C1OC1")
    phenol_patt = Chem.MolFromSmarts("[c][OX2H]")

    for _, row in df.iterrows():
        if row["role"] == "resin":
            mol = Chem.MolFromSmiles(row["product_smiles"])
            assert mol is not None
            n_ep = len(mol.GetSubstructMatches(epoxy_patt))
            n_ph = len(mol.GetSubstructMatches(phenol_patt))
            assert not (n_ep > 0 and n_ph > 0), f"非法单体泄露: {row['product_smiles']}"


def test_virtual_screening_filters_epoxy_phenol_resins():
    """验证配方筛选预过滤规则能够剔除带有酚羟基的环氧树脂单体。"""
    pool_df = pd.DataFrame({
        "resin_smiles": [
            "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", # 合格 DGEBA
            "Oc1ccc(CC2CO2)cc1-c1cc(CC2CO2)ccc1O",       # 不合理：环氧+酚羟基
        ],
        "hardener_smiles": [
            "Nc1ccc(Cc2ccc(N)cc2)cc1",
            "Nc1ccc(Cc2ccc(N)cc2)cc1",
        ],
    })

    rules = {
        "resin": {
            "min_epoxide": 1,
            "ban_phenols": True,
        },
        "hardener": {
            "min_mw": 25.0,
            "max_mw": 1500.0,
        }
    }

    filtered_df = filter_candidates_by_epoxy_rules(pool_df, rules=rules)
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["resin_smiles"] == "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
