import pytest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
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
    count_anhydride_groups,
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


# ==============================================================================
# 化学正确性回归测试（2026 修复：假产物/酸酐当量/确定性/覆盖度）
# ==============================================================================

_MTHPA = "CC1=CCC2C(=O)OC(=O)C2C1"
_PMDA = "O=C1OC(=O)c2cc3C(=O)OC(=O)c3cc21"


def test_anhydride_stoichiometry_1_to_1():
    """酸酐当量约定：1个酸酐基团 = 1个环氧当量（不再×2），且兼容芳构化表示。"""
    mthpa = Chem.MolFromSmiles(_MTHPA)
    st = calculate_stoichiometry(mthpa, "hardener", "多元酸酐")
    assert st["resin_type"] == "anhydride"
    assert st["functionality"] == 1
    assert abs(st["equivalent_weight"] - 166.18) < 1.0  # 文献 AHEW(MTHPA) ≈ 166

    pmda = Chem.MolFromSmiles(_PMDA)
    st2 = calculate_stoichiometry(pmda, "hardener", "多元酸酐")
    assert st2["resin_type"] == "anhydride"
    assert st2["functionality"] == 2  # RDKit 将 PMDA 芳构化，传统模式匹配不上
    assert abs(st2["equivalent_weight"] - 109.06) < 1.0  # 文献 AHEW(PMDA) ≈ 109

    assert count_anhydride_groups(Chem.MolFromSmiles("O=C1OC(=O)c2ccccc21")) == 1
    assert count_anhydride_groups(Chem.MolFromSmiles("O=C(O)c1ccccc1C(=O)O")) == 0


def _catalog_run(max_total: int = 400):
    return run_combinatorial_monomer_design(
        enable_scaffold_fission=False,
        min_functionality=2,
        max_sa_score=6.0,
        max_total_products=max_total,
        n_jobs=1,
    )


def test_no_fallthrough_fake_products():
    """未反应底物不得冒充反应产物：native/phenolic retain 之外，产物必须≠前驱体。"""
    df, _ = _catalog_run()
    assert not df.empty
    retain_ids = {"native_hardener_retain", "phenolic_hardener_retain"}
    reaction_products = df[~df["reaction_id"].isin(retain_ids)]
    assert not reaction_products.empty
    fakes = reaction_products[reaction_products["product_smiles"] == reaction_products["precursor_smiles"]]
    assert fakes.empty, f"发现假产物: {fakes[['reaction_id', 'product_smiles']].head().values.tolist()}"


def test_mannich_products_are_real():
    """R05 曼尼希产物必须含真实邻位氨甲基结构（原模板产物无效且 100% 为假产物）。"""
    df, _ = _catalog_run()
    r05 = df[df["reaction_id"] == "mannich_polyamine"]
    assert not r05.empty
    patt = Chem.MolFromSmarts("cCNCCN")
    for smi in r05["product_smiles"]:
        assert Chem.MolFromSmiles(smi).HasSubstructMatch(patt), f"R05 产物缺少曼尼希碱结构: {smi}"


def test_aliphatic_amine_not_fake_resin():
    """脂肪胺（IPDA）不再穿透芳香胺模板变成假树脂，而是经 R02b 正确缩水甘油胺化。"""
    df, _ = _catalog_run()
    fake = df[(df["product_smiles"] == "CC1(C)CC(C)(CN)CC(N)C1") & (df["role"] == "resin")]
    assert fake.empty, "未反应的 IPDA 不应作为树脂入库"
    aliphatic_gly = df[
        (df["reaction_id"] == "glycidyl_amination_aliphatic")
        & df["product_smiles"].str.contains("N(CC2CO2)CC2CO2", regex=False)
    ]
    assert not aliphatic_gly.empty


def test_aliphatic_polyol_glycidyl_etherification():
    """R01 通用版：脂肪族多元醇（脂环二醇）可被缩水甘油醚化（原先芳香锚点导致整类死路）。"""
    from core.molecule_design import PrecursorCore

    diol = PrecursorCore(
        core_id="test_chex_diol",
        name="环己二醇测试",
        category="测试",
        role="resin",
        smiles="OC1CCC(C2CCC(O)CC2)CC1",
    )
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[diol],
        enable_scaffold_fission=False,
        selected_reaction_ids=["glycidyl_etherification"],
        min_functionality=2,
        max_total_products=50,
        n_jobs=1,
    )
    assert not df.empty
    patt = Chem.MolFromSmarts("[O;r3]1[C;r3][C;r3]1")
    top = df.sort_values("functionality", ascending=False).iloc[0]
    assert top["functionality"] == 2  # 双缩水甘油醚
    assert Chem.MolFromSmiles(top["product_smiles"]).HasSubstructMatch(patt)


def test_tgic_route():
    """酮式氰尿酸经 R13 生成 TGIC（3 个 N-缩水甘油），而非错误互变异构的 O-醚。"""
    from core.molecule_design import PrecursorCore

    cya = PrecursorCore(
        core_id="test_cya",
        name="氰尿酸测试",
        category="测试",
        role="resin",
        smiles="O=C1NC(=O)NC(=O)N1",
    )
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[cya],
        enable_scaffold_fission=False,
        selected_reaction_ids=["isocyanurate_glycidylation"],
        min_functionality=2,
        max_total_products=50,
        n_jobs=1,
    )
    assert not df.empty
    top = df.sort_values("functionality", ascending=False).iloc[0]
    assert top["functionality"] == 3
    assert "n(CC2CO2)" in top["product_smiles"] or "n(CC1CO1)" in top["product_smiles"]


def test_benzoxazine_route():
    """R10 修复后 BPA 可生成双苯并噁嗪（原模板价态错误永不产出）。"""
    from core.molecule_design import PrecursorCore

    bpa = PrecursorCore(
        core_id="test_bpa",
        name="双酚A测试",
        category="测试",
        role="resin",
        smiles="CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    )
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[bpa],
        enable_scaffold_fission=False,
        selected_reaction_ids=["benzoxazine_synthesis"],
        min_functionality=2,
        max_total_products=50,
        n_jobs=1,
    )
    assert not df.empty
    boz_patt = Chem.MolFromSmarts("[OX2]1[CX4][NX3][CX4][c]2[c]1cccc2")
    top = df.sort_values("functionality", ascending=False).iloc[0]
    assert top["functionality"] == 2
    assert Chem.MolFromSmiles(top["product_smiles"]).HasSubstructMatch(boz_patt)


def test_anhydride_route_produces_cyclic_anhydrides():
    """R07a/R07b 修复后邻苯二甲酸可分子内脱水生成环酐（原双反应物模板永不触发）。"""
    from core.molecule_design import PrecursorCore

    pa = PrecursorCore(
        core_id="test_pa",
        name="邻苯二甲酸测试",
        category="测试",
        role="resin",
        smiles="O=C(O)c1ccccc1C(=O)O",
    )
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[pa],
        enable_scaffold_fission=False,
        selected_reaction_ids=["anhydride_cyclization_aromatic"],
        min_functionality=2,
        max_total_products=50,
        n_jobs=1,
    )
    assert not df.empty
    top = df.iloc[0]
    assert top["resin_type"] == "anhydride"
    assert top["functionality"] == 1  # 单酸酐放行（工业标准，如 MTHPA/HHPA）
    assert abs(top["equivalent_weight"] - 148.12) < 1.0


def test_deterministic_truncated_output():
    """相同参数两次运行（含截断）必须得到完全一致的产物集合。"""
    kw = dict(
        enable_scaffold_fission=False,
        min_functionality=2,
        max_sa_score=6.0,
        max_total_products=40,
        n_jobs=-1,
    )
    df1, _ = run_combinatorial_monomer_design(**kw)
    df2, _ = run_combinatorial_monomer_design(**kw)
    assert list(df1["product_smiles"]) == list(df2["product_smiles"])


def test_new_precursor_catalog_entries_valid():
    """新增前驱体（DICY/咪唑/硫醇/氰尿酸/酚醛/多元酸）全部可解析且角色正确。"""
    required = {
        "dicy": "hardener", "mi_2": "hardener", "emi_24": "hardener",
        "pz_2": "hardener", "petmp": "hardener", "melamine": "hardener",
        "cyanuric_acid": "resin", "phthalic_acid": "resin",
        "pyromellitic_acid": "resin", "novolac_trimer": "resin",
    }
    catalog = {p.core_id: p for p in PRECURSOR_CATALOG}
    for core_id, role in required.items():
        assert core_id in catalog, f"缺少前驱体 {core_id}"
        assert catalog[core_id].role == role, f"{core_id} 角色错误"
        mol = Chem.MolFromSmiles(catalog[core_id].smiles)
        assert mol is not None, f"{core_id} SMILES 无法解析"


def test_custom_library_role_inference_by_functional_group():
    """自建库角色推断按官能团：醚键不再让二胺误判为树脂，氨基酚判为双活性。"""
    records = [
        {"smiles": "Nc1ccc(Oc2ccc(N)cc2)cc1", "name": "ODA"},      # 醚键二胺 -> hardener
        {"smiles": "Nc1ccc(O)cc1", "name": "氨基酚"},               # NH2 + 酚OH -> both
        {"smiles": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "name": "BPA"},  # 多元酚 -> resin
    ]
    precursors, _ = parse_and_extract_custom_precursors(records, source_name="测试")
    by_name = {p.name: p for p in precursors}
    assert by_name["ODA_母核"].role == "hardener"
    assert by_name["氨基酚_母核"].role == "both"
    assert by_name["BPA_母核"].role == "resin"


def test_ring_selection_and_missing_rings_now_generate():
    """环系覆盖：咔唑/菲/呫吨/芘/喹喔啉均可生成中间体；环选择门控生效。"""
    inters = generate_scaffold_intermediates(max_intermediates=30000)
    checks = {
        "xanthene": "Oc1ccc2c(c1)Oc1cc(O)ccc1C2",
        "pyrene": "Oc1cc2ccc3cc(O)c4cccc(c1)c4c3-2",
        "quinoxaline": "Oc1nc(O)c2ccccc2n1",
        "carbazole": "Oc1ccc2[nH]c3ccc(O)cc3c2c1",
        "fluorene_27": "Oc1ccc2c(c1)Cc1ccc(O)cc12",
        "terphenyl": "Oc1ccc(-c2ccc(-c3ccc(O)cc3)cc2)cc1",
        "phenanthrene": "Oc1ccc2c3ccc(O)cc3ccc2c1",
    }
    mols = [Chem.MolFromSmiles(p.smiles) for p in inters]
    for ring, frag in checks.items():
        fm = Chem.MolFromSmiles(frag)
        assert fm is not None
        assert any(m is not None and m.HasSubstructMatch(fm) for m in mols), f"{ring} 未参与组装"

    # 门控：只选苯环+呫吨时，2,7-二羟基芴（稠环衍生多酚类）不应生成
    gated = generate_scaffold_intermediates(selected_rings=["benzene", "xanthene"], max_intermediates=30000)
    fm_flu = Chem.MolFromSmarts("Oc1ccc2c(c1)Cc1ccc(O)cc12")
    leaked = [p.smiles for p in gated if p.category == "稠环衍生多酚" and Chem.MolFromSmiles(p.smiles).HasSubstructMatch(fm_flu)]
    assert not leaked, f"未选择的芴环仍参与组装: {leaked[:2]}"
