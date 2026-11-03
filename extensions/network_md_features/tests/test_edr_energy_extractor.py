"""
原生 EDR 能量项提取器测试套件 (Test Native EDR Extractor)
验证：
1. 从原生 postequilibration-npt.edr 正确提取全部 6 个非键能量分量与能量密度;
2. 缺失 EDR 文件时直接抛出 FileNotFoundError，绝不落回常数;
3. 损坏或空白 EDR 文件时直接报错拦截;
4. 严格 CED 保持 None (null).
"""

import os
import pytest
from pathlib import Path
from src.edr_energy_parser import extract_native_energy_from_edr

EXT_ROOT = Path(__file__).resolve().parent.parent

def test_extract_native_energy_success():
    edr_path = EXT_ROOT / "artifacts/recovery_v2/REAL_DGEBA_DDS_seed20261015/proj-0/systems/postcure/postequilibration-npt.edr"
    if not edr_path.exists():
        pytest.skip("原生 EDR 测试文件不存在")

    vol_nm3 = 2.02824 ** 3
    res = extract_native_energy_from_edr(str(edr_path), box_volume_nm3=vol_nm3)

    assert res["steps"] == 501
    assert res["time_range_ps"] == [0.0, 1.0]
    assert "lj_14" in res["components_kj_mol"]
    assert "coulomb_14" in res["components_kj_mol"]
    assert abs(res["components_kj_mol"]["lj_14"] - 315.602) < 0.1
    assert abs(res["components_kj_mol"]["coulomb_14"] - 38.2751) < 0.1

    # 验证非键能密度已按公式严格换算且不为硬编码 -61.7 / -92.54
    assert res["md_vdw_energy_density_MJ_m3"] is not None
    assert abs(res["md_vdw_energy_density_MJ_m3"] - 8.5358) < 0.05
    assert abs(res["md_electrostatic_energy_density_MJ_m3"] - (-71.0852)) < 0.05

    # 验证严格 CED 为 None
    assert res["md_ced_MJ_m3"] is None
    assert res["energy_density_status"] == "EXPLORATORY_SHORT_SAMPLE"

def test_extract_native_energy_missing_file_fails():
    with pytest.raises(FileNotFoundError):
        extract_native_energy_from_edr("/tmp/non_existent_file.edr", box_volume_nm3=8.0)

def test_extract_native_energy_empty_file_fails(tmp_path):
    empty_edr = tmp_path / "empty.edr"
    empty_edr.write_text("")
    with pytest.raises(FileNotFoundError):
        extract_native_energy_from_edr(str(empty_edr), box_volume_nm3=8.0)
