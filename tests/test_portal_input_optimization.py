import pytest
from UserPrediction import (
    PORTAL_PRESET_RESINS,
    PORTAL_PRESET_HARDENERS,
    _render_2d_molecule_png_b64,
)
from rdkit import Chem


def test_portal_presets_validity():
    assert len(PORTAL_PRESET_RESINS) >= 8
    assert len(PORTAL_PRESET_HARDENERS) >= 8

    # 验证所有预设树脂与固化剂 SMILES 均能被 RDKit 成功解析并生成合法分子
    for name, smiles in PORTAL_PRESET_RESINS.items():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid preset resin SMILES for {name}: {smiles}"
        Chem.SanitizeMol(mol)

    for name, smiles in PORTAL_PRESET_HARDENERS.items():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid preset hardener SMILES for {name}: {smiles}"
        Chem.SanitizeMol(mol)


def test_render_2d_molecule_png_b64():
    # 测试合法 SMILES 渲染
    dgeba_smi = PORTAL_PRESET_RESINS["双酚A二缩水甘油醚 (E-51 / DGEBA)"]
    is_valid, b64_str, formula = _render_2d_molecule_png_b64(dgeba_smi)
    assert is_valid is True
    assert len(b64_str) > 100
    assert "C21H24O4" in formula

    # 测试非法 SMILES 错误捕获
    invalid_smi = "CC(C)(c1ccc(OCC2CO2)cc1)INVALID_GARBAGE"
    is_valid, err_msg, formula = _render_2d_molecule_png_b64(invalid_smi)
    assert is_valid is False
    assert "SMILES 语法无效" in err_msg or "化学解析异常" in err_msg
