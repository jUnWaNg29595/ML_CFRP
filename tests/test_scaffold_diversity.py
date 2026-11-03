"""骨架多样性：星型/四官能不再硬编码苯环；配额防饿死；死选项补模板。"""
import pytest
from rdkit import Chem
from core.molecule_design import (
    RING_ARM_TEMPLATES, generate_scaffold_intermediates, PRECURSOR_CATALOG,
    calculate_stoichiometry,
)

_BASE = {p.core_id for p in PRECURSOR_CATALOG}


def _new(cores):
    return [p for p in cores if p.core_id not in _BASE]


def test_ring_arm_templates_registered():
    for ring in ("benzene", "naphthalene_14", "anthracene", "pyrene", "fluorene", "carbazole"):
        assert ring in RING_ARM_TEMPLATES, f"缺少环臂模板 {ring}"


def test_star_generates_for_non_benzene_rings():
    """只勾选萘时仍能生成星型母核（旧实现只认苯环，此处必失败）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14"], selected_linkers=["methine_star"],
        max_intermediates=5000,
    )
    star = [p for p in _new(cores) if p.category == "拓扑星型多酚"]
    assert star, "勾选萘时未生成星型母核"


def test_unchecking_benzene_keeps_star():
    """取消苯环不得清零星型（只要还有其他已选环）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14", "fluorene"], selected_linkers=["methine_star", "quaternary_carbon"],
        max_intermediates=5000,
    )
    star = [p for p in _new(cores) if "拓扑星型" in p.category]
    assert star, "仅稠环勾选时星型被清零"


def test_ethane_tetra_emits_amine_and_phenol_cores():
    """只勾选四苯乙烷时必须同时产出四酚树脂与四胺固化剂母核（旧实现丢弃胺臂）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14"], selected_linkers=["ethane_tetra"],
        max_intermediates=5000,
    )
    new = _new(cores)
    tetra_amine = [p for p in new if "四苯乙烷型四胺" in p.name]
    tetra_phenol = [p for p in new if "四苯乙烷型四酚" in p.name]
    assert tetra_amine, "ethane_tetra 丢失四胺固化剂母核"
    assert all(p.role == "hardener" for p in tetra_amine)
    assert tetra_phenol, "ethane_tetra 未生成四酚树脂母核"
    assert all(p.role == "resin" for p in tetra_phenol)


def test_star_amine_arms_parse_and_carry_functionality():
    """胺臂 SMILES 必须真实可解析且具非零胺官能度（防 except 吞错）。"""
    cores = generate_scaffold_intermediates(
        selected_rings=["naphthalene_14"], selected_linkers=["methine_star"],
        max_intermediates=5000,
    )
    star_amine = [p for p in _new(cores) if "拓扑星型多胺" in p.category]
    assert star_amine, "勾选 methine_star 未生成拓扑星型多胺母核"
    for p in star_amine:
        mol = Chem.MolFromSmiles(p.smiles)
        assert mol is not None, f"胺臂母核 SMILES 解析失败: {p.smiles}"
        sto = calculate_stoichiometry(mol, "hardener", "")
        assert sto["functionality"] >= 2, f"胺官能度异常: {p.smiles} -> {sto}"


def test_small_budget_keeps_all_categories():
    """max_intermediates=500 时稠环/星型/羧酸类别均不得整类饿死。"""
    from collections import Counter
    cores = generate_scaffold_intermediates(max_intermediates=500)
    cats = Counter(p.category for p in _new(cores))
    for key in ("稠环衍生多酚", "拓扑星型多酚", "多元羧酸衍生母核"):
        assert cats.get(key, 0) > 0, f"{key} 被配额饿死"


def test_dead_ring_options_now_generate():
    """4 个死选项现在必须产出真实骨架母核（而非酸的假象）。"""
    for rid in ("mesitylene", "biphenyl_33", "cardo_bpf", "s_triazine"):
        cores = generate_scaffold_intermediates(selected_rings=[rid], max_intermediates=5000)
        skeleton = [p for p in _new(cores) if p.category != "多元羧酸衍生母核"]
        assert skeleton, f"{rid} 无任何骨架母核（仍是死选项）"


def test_revived_rings_resin_cores_have_two_phenolic_oh():
    """复活的 4 个环选项：resin 骨架母核必须含 >=2 个酚羟基（钉住真实官能度）。

    cardo_bpf 曾误用单酚模板（酚羟基数=1，官能度 1，环氧后无法交联，
    被 min_functionality=2 门控过滤）。按 [c][OX2H] SMARTS 计数酚羟基；
    mesitylene/biphenyl_33/cardo_bpf/s_triazine 四个环均预期产出 resin 母核，
    空集视为失败而非静默跳过。
    """
    phenol_oh = Chem.MolFromSmarts("[c][OX2H]")
    for rid in ("mesitylene", "biphenyl_33", "cardo_bpf", "s_triazine"):
        cores = generate_scaffold_intermediates(selected_rings=[rid], max_intermediates=5000)
        resins = [
            p for p in _new(cores)
            if p.role == "resin" and p.category != "多元羧酸衍生母核"
        ]
        assert resins, f"{rid} 未产出任何 resin 骨架母核（不允许空集静默通过）"
        for p in resins:
            mol = Chem.MolFromSmiles(p.smiles)
            assert mol is not None, f"{rid} resin 母核 SMILES 解析失败: {p.smiles}"
            n_oh = len(mol.GetSubstructMatches(phenol_oh))
            assert n_oh >= 2, f"{rid} resin 母核酚羟基数 {n_oh} < 2: {p.smiles}"


def test_acid_section_respects_ring_gate():
    """酸段必须受环门控：只勾 non-acid 环时，酸母核数量应显著减少。"""
    all_cores = generate_scaffold_intermediates(max_intermediates=5000)
    gated = generate_scaffold_intermediates(selected_rings=["mesitylene"], max_intermediates=5000)
    n_all = sum(1 for p in _new(all_cores) if p.category == "多元羧酸衍生母核")
    n_gated = sum(1 for p in _new(gated) if p.category == "多元羧酸衍生母核")
    assert n_gated < n_all, f"酸段未受环门控: {n_gated} >= {n_all}"
