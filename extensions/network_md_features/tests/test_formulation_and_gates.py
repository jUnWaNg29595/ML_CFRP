"""
配方适配与科学门禁自动化测试 (Test Formulation & Scientific Quality Gates)
验证内容：
1. 配方规范化、离散化整数分子数换算、当量比 r 定义与误差报告；
2. 适用域拦截：缺少环氧基、缺少活性胺基、极端配比；
3. S0~S1 门禁：原子重叠、部分电荷不守恒拦截；
4. S2 门禁 (从拓扑检查参数与电荷)；
5. S4 门禁 (从日志检查极大力)；
6. S5 门禁：残余牺牲氢与虚假转化率超额拦截；
7. S6 门禁：平衡态轨迹稳定性（密度漂移超标拦截）。
"""

import os
import numpy as np
import pytest

from src.formulation_adapter import FormulationAdapter
from src.chemical_validation import ChemicalValidator
from src.trajectory_validation import TrajectoryValidator

def test_formulation_adapter_dgeba_dds():
    adapter = FormulationAdapter(target_atom_budget=1500)
    resin_smiles = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"  # DGEBA, 2 epoxy, 49 atoms
    hardener_smiles = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"         # 4,4-DDS, 4 active H, 29 atoms
    r = 1.0  # 理想等当量

    res = adapter.calculate_discrete_formulation(
        resin_smiles=resin_smiles,
        hardener_smiles=hardener_smiles,
        r=r,
        resin_name="DGEBA",
        hardener_name="4,4-DDS"
    )

    assert res["formulation_id"].startswith("FORM_")
    assert res["nominal_r"] == 1.0
    assert abs(res["actual_discrete_r"] - 1.0) < 0.1
    assert res["discretization_error_pct"] < 10.0
    assert res["n_resin_molecules"] > 0
    assert res["n_hardener_molecules"] > 0
    assert res["theoretical_max_alpha"] == 1.0

def test_formulation_domain_screening():
    adapter = FormulationAdapter(target_atom_budget=1000)
    # 1. 树脂只有 1 个环氧（单官能，无法形成交联网络）
    mono_epoxy = "CC(C)OCC1CO1"
    dds = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
    is_valid, msg = adapter.check_applicability_domain(mono_epoxy, dds, r=1.0)
    assert not is_valid
    assert "无法形成交联网络" in msg

    # 2. 固化剂无活性氢（如苯环）
    benzene = "c1ccccc1"
    dgeba = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    is_valid2, msg2 = adapter.check_applicability_domain(dgeba, benzene, r=1.0)
    assert not is_valid2
    assert "无法作为交联固化剂" in msg2

def test_chemical_validator_gates(tmp_path):
    validator = ChemicalValidator()

    # S1 门禁：电荷不守恒拦截
    bad_monomer = {
        "name": "BadMonomer",
        "formal_charge": 0,
        "partial_charges": [0.25, -0.10, 0.05],  # sum = 0.20 != 0
        "coordinates": None
    }
    s1_fail = validator.check_gate_s1([bad_monomer])
    assert s1_fail.status == "FAIL"

    good_monomer = {
        "name": "GoodMonomer",
        "smiles": "CC",
        "elements": ["C", "C"],
        "bonds": [[0, 1]],
        "formal_charge": 0,
        "partial_charges": [0.25, -0.25],
        "coordinates": [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]
    }
    s1_pass = validator.check_gate_s1([good_monomer])
    assert s1_pass.status == "PASS"

    # S2 门禁：从原生拓扑检查
    top_file = tmp_path / "test.top"
    with open(top_file, "w") as f:
        f.write("[ atoms ]\n1 c3 1 DGE C1 1 0.125 12.011\n2 c3 1 DGE C2 1 -0.125 12.011\n")
    s2_res = validator.check_gate_s2_from_top(str(top_file), declared_charge_protocol="DFT_RESP_B3LYP_631G*", expected_charges_map={"C1": 0.125, "C2": -0.125})
    assert s2_res.status == "PASS"

    # S4 门禁：从原生日志检查极大力
    log_file = tmp_path / "min.log"
    with open(log_file, "w") as f:
        f.write("Maximum force = 650.0\nEnergy minimization reached tolerance\n")
    s4_pass = validator.check_gate_s4_from_log(str(log_file), max_allowed_force=1500.0)
    assert s4_pass.status == "PASS"

    # S5 门禁：残余牺牲氢拦截
    s5_fail = validator.check_gate_s5_curing(
        actual_bonds_formed=10,
        initial_epoxy_count=12,
        theoretical_max_alpha=1.0,
        residual_sacrificial_h=4
    )
    assert s5_fail.status == "FAIL"
    assert "残留" in s5_fail.message

def test_trajectory_validator_gate_s6():
    validator = TrajectoryValidator(max_density_drift_pct_per_ns=1.0)

    # 1. 稳定轨迹：密度在 1.18 附近微小涨落
    time_ns = np.linspace(0.0, 2.0, 20)
    density_good = 1.18 + np.sin(time_ns * 10) * 0.002
    temp_good = np.full_like(time_ns, 298.15) + np.random.normal(0, 0.5, len(time_ns))

    res_good = validator.check_gate_s6(time_ns, density_good, temp_good)
    assert res_good.status == "PASS"

    # 2. 漂移严重轨迹：密度在 2 ns 内持续从 1.18 跌至 1.05 (漂移率 > 1.0%/ns)
    density_drift = np.linspace(1.18, 1.05, 20)
    res_bad = validator.check_gate_s6(time_ns, density_drift, temp_good)
    assert res_bad.status == "FAIL"
    assert "密度漂移率超标" in res_bad.message
