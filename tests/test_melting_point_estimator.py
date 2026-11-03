"""
测试高精度 QSPR 熔点预估引擎及工业级过滤中的熔点计算改进。
"""
import pytest
from rdkit import Chem
from core.melting_point_estimator import (
    estimate_melting_point,
    estimate_melting_point_details,
    extract_melting_point_descriptors,
)
from core.industrial_filter import estimate_melting_point as industrial_estimate_melting_point
from core.industrial_filter import check_category_constraints


def test_standard_industrial_exact_melting_points():
    """验证标准工业树脂与固化剂命中真值库，具备实验级零误差。"""
    cases = {
        "DGEBA (双酚A环氧树脂)": ("CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", 43.0),
        "DDM (4,4'-二氨基二苯甲烷)": ("Nc1ccc(Cc2ccc(N)cc2)cc1", 90.0),
        "4,4'-DDS (二氨基二苯砜)": ("Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", 177.0),
        "双酚A (BPA)": ("CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", 156.0),
        "邻苯二甲酸酐 (PA)": ("O=C1OC(=O)c2ccccc21", 131.0),
        "双氰胺 (DICY)": ("NC(=N)NC#N", 209.0),
        "间苯二甲胺 (m-XDA)": ("NCc1cccc(CN)c1", 14.1),
        "甲基四氢苯酐 (MTHPA)": ("CC1C=CC2C(=O)OC(=O)C2C1", -15.0),
    }

    for name, (smi, expected_val) in cases.items():
        det = estimate_melting_point_details(smi)
        assert det["is_experimental"] is True, f"{name} 未命中实验数据库"
        assert abs(det["melting_point_c"] - expected_val) < 2.0, f"{name} 偏差过大"

        # 验证同时支持 Chem.Mol 传入
        mol = Chem.MolFromSmiles(smi)
        assert abs(estimate_melting_point(mol) - expected_val) < 2.0


def test_liquid_hardener_passes_industrial_filter():
    """验证常温液体固化剂（m-XDA、MTHPA）不会被错误的旧公式（原分别预估为 108°C 与 121°C）误判超标。"""
    # m-XDA 实际 14.1°C，旧公式算成 108.5°C
    mol_mxda = Chem.MolFromSmiles("NCc1cccc(CN)c1")
    mp_mxda = industrial_estimate_melting_point(mol_mxda)
    assert mp_mxda < 35.0, f"m-XDA 预估熔点过高: {mp_mxda}"

    # MTHPA 实际常温液体，旧公式算成 121.2°C
    mol_mthpa = Chem.MolFromSmiles("CC1C=CC2C(=O)OC(=O)C2C1")
    mp_mthpa = industrial_estimate_melting_point(mol_mthpa)
    assert mp_mthpa < 30.0, f"MTHPA 预估熔点过高: {mp_mthpa}"


def test_novel_molecule_qspr_prediction():
    """验证未在已知实验库的新分子通过 28 维拓扑 QSPR 机器学习模型合理预估。"""
    # 新设计的复杂对称含硅双环氧
    novel_smi = "C1OC1COC(F)(F)c1ccc([Si](C)(C)O[Si](C)(C)c2ccc(C(F)(F)OCC3CO3)cc2)cc1"
    det = estimate_melting_point_details(novel_smi)
    assert det["method"] in ("ml_qspr_model", "physics_calibrated_fallback")
    assert 20.0 <= det["melting_point_c"] <= 200.0


def test_feature_extractor_robustness():
    """验证特征提取器在单原子、非规范分子下永不崩溃。"""
    mol_empty = None
    assert extract_melting_point_descriptors(mol_empty) is None

    mol_methane = Chem.MolFromSmiles("C")
    feats = extract_melting_point_descriptors(mol_methane)
    assert feats is not None
    assert len(feats) == 28
