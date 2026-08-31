import pytest
import pandas as pd
from rdkit import Chem
from core.molecule_design import (
    RING_SCAFFOLDS,
    LINKER_BRIDGES,
    R_GROUPS,
    SYNTHETIC_REACTION_TEMPLATES,
    PRECURSOR_CATALOG,
    generate_scaffold_intermediates,
    run_combinatorial_monomer_design,
    calculate_stoichiometry,
    deconstruct_monomer_to_precursor_core,
    parse_and_extract_custom_precursors,
)


def test_deconstruct_epoxy_monomers_to_precursors():
    """验证逆合成拆解器：将商业环氧树脂/单体精准还原为多元酚/多元胺前驱体母核"""
    dgeba = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    bpa_cores = deconstruct_monomer_to_precursor_core(dgeba)
    assert len(bpa_cores) >= 1
    bpa_mol = Chem.MolFromSmiles(bpa_cores[0])
    assert bpa_mol is not None
    assert "O" in bpa_cores[0]

    tgddm = "C1C(O1)CN(c2ccc(Cc3ccc(N(CC4CO4)CC5CO5)cc3)cc2)CC6CO6"
    ddm_cores = deconstruct_monomer_to_precursor_core(tgddm)
    assert len(ddm_cores) >= 1
    ddm_mol = Chem.MolFromSmiles(ddm_cores[0])
    assert ddm_mol is not None
    assert "N" in ddm_cores[0]


def test_bigsmiles_parsing_and_core_extraction():
    """验证 BigSMILES 聚合物/低聚物语法的解析采样与母核提取"""
    bigsmiles_str = "{[<]CC(C)(c1ccc(O)cc1)c1ccc(O)cc1[>]}"
    records = [{"smiles": bigsmiles_str, "name": "自建双酚A聚合物"}]
    extracted_precursors, logs = parse_and_extract_custom_precursors(records, source_name="测试BigSMILES库")
    
    assert len(extracted_precursors) >= 1
    for p in extracted_precursors:
        mol = Chem.MolFromSmiles(p.smiles)
        assert mol is not None
        assert p.role in ("resin", "hardener", "both")


def test_custom_precursor_combinatorial_design_and_precursor_smiles_field():
    """验证以自建/PubChem提取的前驱体母核为种子，开展正交衍生设计，且输出包含 precursor_smiles"""
    user_inputs = [
        {"smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", "name": "用户环氧A"},
        {"smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "name": "用户固化剂B"},
    ]
    custom_cores, _ = parse_and_extract_custom_precursors(user_inputs, source_name="用户上传库")
    assert len(custom_cores) >= 2

    df, logs = run_combinatorial_monomer_design(
        custom_precursors=custom_cores,
        enable_scaffold_fission=False,
        min_functionality=2,
        max_sa_score=6.0,
        max_total_products=100,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "precursor_smiles" in df.columns
    assert "precursor_name" in df.columns
    assert df["precursor_smiles"].notnull().all()
