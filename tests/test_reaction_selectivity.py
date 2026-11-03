"""反应模板选择性回归：R01 羟基醚化不许误伤羧酸，R04 环氧化不许误伤吸电子烯烃。"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from core.molecule_design import SYNTHETIC_REACTION_TEMPLATES


def _rxn(reaction_id):
    tpl = next(t for t in SYNTHETIC_REACTION_TEMPLATES if t.reaction_id == reaction_id)
    return rdChemReactions.ReactionFromSmarts(tpl.reaction_smarts)


def _products(reaction_id: str, smiles: str) -> list[str]:
    """单步应用反应模板，返回产物 canonical SMILES 列表（空=不反应）。"""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"底物解析失败 {smiles}"
    out = []
    for tup in _rxn(reaction_id).RunReactants((mol,)):
        try:
            p = tup[0]
            Chem.SanitizeMol(p)
            out.append(Chem.MolToSmiles(p))
        except Exception:
            continue
    return out


def _has_frag(smiles_prod: str, smarts: str) -> bool:
    m = Chem.MolFromSmiles(smiles_prod)
    return m is not None and m.HasSubstructMatch(Chem.MolFromSmarts(smarts))


def test_r01_no_longer_touches_carboxyl():
    """己二酸必须整类拒绝；对羟基苯甲酸只准醚化酚羟基、保留羧基。"""
    assert _products("glycidyl_etherification", "O=C(O)CCCCC(=O)O") == [], \
        "己二酸被醚化，R01 与 R03 抢活"
    prods = _products("glycidyl_etherification", "O=C(O)c1ccc(O)cc1")
    assert prods, "对羟基苯甲酸的酚羟基应被醚化"
    for p in prods:
        assert _has_frag(p, "C(=O)[OX2H]"), f"羧基被破坏: {p}"


def test_r01a_phenol_only():
    """细分模板：仅酚羟基版不许碰脂肪醇。"""
    assert _products("glycidyl_etherification_phenol", "OCCO") == []
    assert _products("glycidyl_etherification_phenol", "Oc1ccccc1"), "苯酚应被醚化"


def test_r01b_aliphatic_only():
    """细分模板：仅脂肪醇版不许碰酚羟基与羧酸。"""
    assert _products("glycidyl_etherification_aliphatic", "Oc1ccccc1") == []
    assert _products("glycidyl_etherification_aliphatic", "O=C(O)CCCCC(=O)O") == []
    assert _products("glycidyl_etherification_aliphatic", "OCCO"), "乙二醇应被醚化"


R04_NEG = {
    "马来酰亚胺": "O=C1C=CC(=O)N1",
    "丙烯酸甲酯": "C=CC(=O)OC",
    "肉桂酸甲酯": "O=C(OC)/C=C/c1ccccc1",
    "肉桂酸": "O=C(O)/C=C/c1ccccc1",
    "肉桂醛": "O=C/C=C/c1ccccc1",
    "丙烯腈": "C=CC#N",
    "甲基乙烯基酮": "C=CC(C)=O",
    "马来酸酐": "O=C1OC(=O)C=C1",
    "甲基丙烯酸甲酯": "C=C(C)C(=O)OC",
}
R04_POS = {
    "环己烯": "C1CC=CCC1",
    "烯丙基苯酚": "C=CCC1=CC=CC=C1O",
    "DCPD": "C1CC2C=CC1C2",
    "苯乙烯": "C=Cc1ccccc1",
    "1-辛烯": "CCCCCCC=C",
    "异丁烯": "CC(C)=C",
    "降冰片烯": "C1CC2CC1C=C2",
}


def test_r04_rejects_electron_poor_alkenes():
    for name, smi in R04_NEG.items():
        prods = _products("olefin_epoxidation", smi)
        assert prods == [], f"{name} 不应被环氧化，实际得到 {prods}"


def test_r04_still_epoxidizes_common_alkenes():
    epoxide_patt = Chem.MolFromSmarts("[O;r3]1[C;r3][C;r3]1")
    for name, smi in R04_POS.items():
        prods = _products("olefin_epoxidation", smi)
        assert prods, f"{name} 应被环氧化"
        for p in prods:
            m = Chem.MolFromSmiles(p)
            assert m is not None and m.HasSubstructMatch(epoxide_patt), f"{name} 产物无环氧: {p}"
