"""
针对 HTPolyNet 独立审计反例的严密回归测试套件 (Test HT Adversarial Rejection)
覆盖 ht_adversarial.json 中的全部 5 项对抗反例：
1. s1_coordinates_without_chemical_graph: 缺少化学图或元素列表，必须坚决返回 FAIL；
2. s2_incomplete_atomtypes_and_missing_bond_params_no_resp_source: 不完整的 atomtypes 且缺少成键力场参数，必须坚决返回 FAIL；
3. s2_empty_expected_charge_mapping: 声明 DFT-RESP 时 expected_charges_map 为空或未覆盖全部原子，必须坚决返回 FAIL；
4. s3_overlapping_positions_with_distinct_velocities: 带速度的 GRO 中原子空间重叠，定宽坐标解析必须坚决拦截并返回 FAIL；
5. s4_nonconverged_fake_log: 达到最大步数或未收敛的最小化日志，必须坚决返回 FAIL。
"""

import os
import pytest
import numpy as np

from src.chemical_validation import ChemicalValidator

@pytest.fixture
def validator():
    return ChemicalValidator()

def test_adversarial_1_s1_no_chemical_graph_rejected(validator):
    """
    反例 1: 仅有散点坐标与电荷，无化学图 (bonds/smiles/rdkit_mol) 或缺少元素列表
    门禁 S1 必须坚决拦截并返回 FAIL！
    """
    bad_spec_no_graph = [{
        "name": "isolated_points",
        "formal_charge": 0,
        "partial_charges": [0.1, -0.1],
        "coordinates": [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
        "elements": ["C", "C"]
        # 故意不提供 bonds, smiles, rdkit_mol
    }]
    res1 = validator.check_gate_s1(bad_spec_no_graph)
    assert res1.status == "FAIL"
    assert "缺少分子化学连接图定义" in res1.message

    bad_spec_no_elements = [{
        "name": "no_elements",
        "smiles": "CC",
        "formal_charge": 0,
        "partial_charges": [0.1, -0.1],
        "coordinates": [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]
        # 故意不提供 elements
    }]
    res2 = validator.check_gate_s1(bad_spec_no_elements)
    assert res2.status == "FAIL"
    assert "缺少元素组成列表" in res2.message

def test_adversarial_2_s2_incomplete_atomtypes_and_unparameterized_top(validator, tmp_path):
    """
    反例 2: 仅两字段的无效 atomtypes 定义、缺失 bond 参数，即便声明 RESP 也必须 FAIL
    """
    top_path = tmp_path / "incomplete.top"
    top_path.write_text(
        "[ atomtypes ]\n"
        "c3  12.011\n"  # 仅两列，缺少完整非键/力场参数
        "[ moleculetype ]\n"
        "MOL 1\n"
        "[ atoms ]\n"
        "1 c3 1 MOL C1 1 0.15 12.011\n"
        "2 c3 1 MOL C2 1 -0.15 12.011\n"
        "[ bonds ]\n"
        "1 2 1\n"
    )
    res = validator.check_gate_s2_from_top(
        str(top_path),
        declared_charge_protocol="DFT_RESP_B3LYP_631G*",
        expected_charges_map={"C1": 0.15, "C2": -0.15}
    )
    assert res.status == "FAIL"
    assert "atomtypes 定义不完整" in res.message or "缺少真实参数支持" in res.message

def test_adversarial_3_s2_empty_expected_charge_mapping(validator, tmp_path):
    """
    反例 3: 声明 DFT-RESP 电荷协议，但传入空的 expected_charges_map 或缺失映射证据
    门禁 S2 必须坚决拦截并返回 FAIL！
    """
    top_path = tmp_path / "valid_types_but_empty_map.top"
    top_path.write_text(
        "#include \"amber99sb.ff/forcefield.itp\"\n"
        "[ moleculetype ]\n"
        "MOL 1\n"
        "[ atoms ]\n"
        "1 c3 1 MOL C1 1 0.15 12.011\n"
        "2 c3 1 MOL C2 1 -0.15 12.011\n"
        "[ bonds ]\n"
        "1 2 1 0.15 300000\n"
    )
    # 传入空映射
    res = validator.check_gate_s2_from_top(
        str(top_path),
        declared_charge_protocol="DFT_RESP_B3LYP_631G*",
        expected_charges_map={}
    )
    assert res.status == "FAIL"
    assert "expected_charges_map 缺失或为空" in res.message

def test_adversarial_4_s3_overlapping_positions_with_velocities_blocked(validator, tmp_path):
    """
    反例 4: GRO 中两原子在空间上完全重叠 (例如坐标相同)，但末尾速度列数值不同
    定宽格式解析忽略速度列，精准获取真实空间坐标并识别碰撞拦截，坚决返回 FAIL！
    """
    gro_path = tmp_path / "overlap_with_vel.gro"
    # 标准 GRO：第 21-28 列为 x, 29-36 列为 y, 37-44 列为 z; 45-68 列为 vx, vy, vz
    # 两个原子在相同位置 (1.000, 1.000, 1.000)，但速度不同 (0.5 vs -0.5)
    gro_content = (
        "Overlapping test system\n"
        " 2\n"
        "    1DGE      C1    1   1.000   1.000   1.000  0.5000  0.0000  0.0000\n"
        "    1DGE      C2    2   1.000   1.000   1.000 -0.5000  0.0000  0.0000\n"
        "   3.00000   3.00000   3.00000\n"
    )
    gro_path.write_text(gro_content)

    res = validator.check_gate_s3_from_gro(str(gro_path), min_box_nm=2.0)
    assert res.status == "FAIL"
    assert "碰撞重叠" in res.message or "间距" in res.message

def test_adversarial_5_s4_nonconverged_maximum_steps_log_blocked(validator, tmp_path):
    """
    反例 5: 能量最小化日志明确指示达到最大步数未收敛
    必须坚决判定为未收敛 FAIL，严禁放行！
    """
    log_path = tmp_path / "unconverged_max_steps.log"
    log_content = (
        "GROMACS Steepest Descents energy minimization\n"
        "Step = 5000, time = 5.0 ps\n"
        "Energy minimization reached the maximum number of steps\n"
        "Maximum force = 100.0\n"
        "Norm of force = 25.0\n"
    )
    log_path.write_text(log_content)

    res = validator.check_gate_s4_from_log(str(log_path), max_allowed_force=1500.0)
    assert res.status == "FAIL"
    assert "未收敛" in res.message or "maximum number of steps" in res.message
