# -*- coding: utf-8 -*-
"""交联网络单元表示的回归测试。

覆盖 2024 修复的两个表示缺陷：
1. "反应后" 代表性单元不应残留环氧基（旧版分支末端以未反应环氧表示悬键）；
2. BigSMILES 连接端数量不应固定为 2 个，应随官能度变化且打在真实连接原子上。
"""
import sys
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reaction_simulator import EpoxyReactionSimulator
from core.bigsmiles_stochastic_graph import _make_unit

EPOXIDE = Chem.MolFromSmarts("[C]1[O][C]1")

RESIN_ALICYCLIC = "C1CC(OCC2CCC3OC3C2)C2OC2C1"   # 双环氧脂环族环氧（用户报告案例）
CURER_THPA = "CC1CCC2C(=O)OC(=O)C2C1"            # 四氢邻苯二甲酸酐
RESIN_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
CURER_DDM = "Nc1ccc(Cc2ccc(N)cc2)cc1"            # 4,4'-二氨基二苯甲烷 (f=4)


@pytest.fixture(scope="module")
def sim():
    return EpoxyReactionSimulator(verbose=False)


def _epoxide_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"单元 SMILES 无法解析: {smiles}"
    return len(mol.GetSubstructMatches(EPOXIDE))


def test_unit_contains_no_residual_epoxide(sim):
    """反应后单元不得残留环氧基（悬键改由 [$] 描述符表达）。"""
    unit = sim.build_network_unit(RESIN_ALICYCLIC, CURER_THPA, target_conversion=0.5)
    assert unit, "网络单元构建失败"
    assert _epoxide_count(unit) == 0


def test_attachment_count_matches_functionality(sim):
    """连接端数量随官能度变化：酸酐单点+悬挂位=2；四官能胺枢纽=4。"""
    pack_thpa = sim.build_network_unit_full(RESIN_ALICYCLIC, CURER_THPA, target_conversion=0.5)
    assert pack_thpa is not None
    _, n_attach_thpa, marked_thpa = pack_thpa
    assert n_attach_thpa == 2 and marked_thpa

    pack_ddm = sim.build_network_unit_full(RESIN_DGEBA, CURER_DDM, target_conversion=0.85)
    assert pack_ddm is not None
    unit, n_attach_ddm, marked_ddm = pack_ddm
    assert _epoxide_count(unit) == 0
    assert n_attach_ddm >= 4, f"四官能胺枢纽连接点应≥4，实际 {n_attach_ddm}"
    assert marked_ddm


def test_bigsmiles_descriptor_count_parsed(sim):
    """生成的 BigSMILES 能被项目解析器解析，且连接点计数与官能度一致。"""
    bs = sim._generate_bigsmiles_network(RESIN_DGEBA, CURER_DDM, target_conversion=0.85)
    assert bs.startswith("{") and bs.endswith("}")
    unit = _make_unit(bs.strip("{}"))
    assert unit.connector_count >= 4
    # 内联描述符应出现在真实连接原子（剩余 N-H / 封端碳）上，而非仅字符串两端
    assert "N([$])" in bs or "([$])" in bs


def test_crosslink_extractor_product_epoxide_free():
    """端到端：CrosslinkedFeatureExtractor 产物不含环氧、结构带描述符。"""
    from core.reaction_simulator import CrosslinkedFeatureExtractor

    ext = CrosslinkedFeatureExtractor(verbose=False)
    feats = ext.extract_crosslink_features(
        RESIN_ALICYCLIC, CURER_THPA, target_conversion=0.5
    )
    assert feats.get("product_smiles"), "产物 SMILES 缺失"
    assert _epoxide_count(feats["product_smiles"]) == 0
    structure = str(feats.get("product_structure"))
    assert structure.startswith("{") and "[$]" in structure
    assert feats.get("product_residual_epoxide") == 0.0
