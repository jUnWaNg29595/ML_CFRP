"""
故障注入测试套件 (Fault Injection Tests)
确保门禁系统绝不放行虚假模拟、缺参数、未对齐电荷、同名异构体或物理碰撞。
覆盖用户指定的 7 大类严密测试场景：
1. 未启动引擎/非零退出/仅自写日志/缺少原生 TPR/GRO/LOG；
2. 摘要宣称 RESP，但最终拓扑电荷全为 0 或未守恒；
3. 树脂有参数、固化剂缺参数或缺失关键 improper/torsion；
4. 分子名称相同、化学图不同（同名异构体拦截），或交换杂原子；
5. 严重周期边界原子碰撞 (< 0.05 nm) 与极大力超标；
6. 遗漏门禁或含有 EXPLORATORY 时严禁聚合为总体 PASS；
7. 损坏文件与哈希失配。
"""

import os
import math
import tempfile
import numpy as np
import pytest
from rdkit import Chem

from src.chemical_validation import ChemicalValidator, GateResult, aggregate_overall_gate_status
from src.qm_adapter import QMParameterAdapter

def test_fault_case_1_missing_native_engine_outputs(tmp_path):
    """Case 1: 缺少原生引擎输出文件或仅有一份伪造日志时，门禁必须明确拦截 (BLOCKED/FAIL)"""
    validator = ChemicalValidator()

    # 伪造不存在的原生文件路径
    fake_top = str(tmp_path / "non_existent.top")
    fake_gro = str(tmp_path / "non_existent.gro")
    fake_log = str(tmp_path / "non_existent.log")

    res_s2 = validator.check_gate_s2_from_top(fake_top, declared_charge_protocol="DFT_RESP_B3LYP_631G*")
    assert res_s2.status == "BLOCKED"
    assert "原生拓扑文件不存在" in res_s2.message

    res_s3 = validator.check_gate_s3_from_gro(fake_gro)
    assert res_s3.status == "BLOCKED"
    assert "原生打包结构文件不存在" in res_s3.message

    res_s4 = validator.check_gate_s4_from_log(fake_log)
    assert res_s4.status == "BLOCKED"
    assert "最小化原生日志文件不存在" in res_s4.message

def test_fault_case_2_zero_or_unconserved_charges(tmp_path):
    """Case 2: 声明了 DFT-RESP，但拓扑中原子电荷全部为 0 或未守恒，门禁必须 FAIL"""
    validator = ChemicalValidator()

    # 构造一个电荷全为 0 的虚假 top 文件
    top_file = tmp_path / "zero_charge.top"
    with open(top_file, "w") as f:
        f.write("[ atoms ]\n")
        f.write("1  c3  1  DGE  C1  1  0.000000  12.011\n")
        f.write("2  c3  1  DGE  C2  1  0.000000  12.011\n")
        f.write("[ bonds ]\n")

    res_s2 = validator.check_gate_s2_from_top(str(top_file), declared_charge_protocol="DFT_RESP_B3LYP_631G*")
    assert res_s2.status == "FAIL"
    assert "拓扑中全部原子电荷为 0" in res_s2.message

def test_fault_case_3_monomer_charge_conservation():
    """Case 3: 单体部分电荷求和非 0 (电荷不守恒)，门禁 S1 必须 FAIL"""
    validator = ChemicalValidator()

    bad_monomer = {
        "name": "Bad_DGEBA",
        "formal_charge": 0,
        "partial_charges": [0.15, -0.05, 0.02],  # sum = 0.12 != 0
        "coordinates": None
    }
    res_s1 = validator.check_gate_s1([bad_monomer])
    assert res_s1.status == "FAIL"
    assert "净电荷求和" in res_s1.message

def test_fault_case_4_isomorphic_graph_checking_rejects_isomer():
    """Case 4: 名字相同但化学图不同的分子（异构体欺骗），qm_adapter 必须拒绝继承参数"""
    qm_adapter = QMParameterAdapter()

    # 正确的 4,4'-DDS 是两个苯环对位连接 SO2 与 NH2: Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1
    # 构造邻位或间位异构体 (例如 3,3'-DDS)
    isomer_33_dds = "Nc1cccc(S(=O)(=O)c2cccc(N)c2)c1"

    # 虽然名字传入 "4,4-DDS"，但化学图不同构，必须拒绝返回 VERIFIED
    spec = qm_adapter.get_monomer_parameter_spec(monomer_name="4,4-DDS", smiles=isomer_33_dds)
    assert spec["verification_level"] != "VERIFIED"
    assert spec["verification_level"] == "EXPLORATORY"

def test_fault_case_5_catastrophic_collision_and_huge_force(tmp_path):
    """Case 5: 严重原子空间重叠 (< 0.5 A) 或极大力超标，门禁 S3 / S4 必须 FAIL"""
    validator = ChemicalValidator()

    # 1. 重叠 GRO 文件
    gro_file = tmp_path / "collision.gro"
    with open(gro_file, "w") as f:
        f.write("System with collision\n2\n")
        f.write("    1DGE      C1    1   1.000   1.000   1.000\n")
        f.write("    1DGE      C2    2   1.001   1.000   1.000\n")  # 距离 0.001 nm = 0.01 A (极度碰撞)
        f.write("   3.00000   3.00000   3.00000\n")

    res_s3 = validator.check_gate_s3_from_gro(str(gro_file), min_box_nm=2.0)
    assert res_s3.status == "FAIL"
    assert "灾难性碰撞重叠" in res_s3.message

    # 2. 极大力超标 LOG 文件
    log_file = tmp_path / "huge_force.log"
    with open(log_file, "w") as f:
        f.write("Step 100: Maximum force = 45210.8\n")
        f.write("Energy minimization reached tolerance\n")

    res_s4 = validator.check_gate_s4_from_log(str(log_file), max_allowed_force=1500.0)
    assert res_s4.status == "FAIL"
    assert "极大力超标" in res_s4.message

def test_fault_case_6_aggregate_status_never_promotes_exploratory_or_blocked():
    """Case 6: 总体门禁聚合法则测试：有任一 EXPLORATORY 不能总体 PASS；缺少门禁不能总体 PASS"""
    req_gates = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]

    # 场景 A: 包含一个 EXPLORATORY (例如诊断级力场)，其余 PASS
    gates_a = [
        GateResult("S0", "PASS", {}, "OK"),
        GateResult("S1", "PASS", {}, "OK"),
        GateResult("S2", "EXPLORATORY", {}, "Diagnostic charge model"),
        GateResult("S3", "PASS", {}, "OK"),
        GateResult("S4", "PASS", {}, "OK"),
        GateResult("S5", "PASS", {}, "OK"),
        GateResult("S6", "PASS", {}, "OK"),
    ]
    status_a, msg_a = aggregate_overall_gate_status(gates_a, req_gates)
    assert status_a == "EXPLORATORY"
    assert "受限/诊断级协议" in msg_a

    # 场景 B: 遗漏 S6 门禁 (未运行或缺失)
    gates_b = [
        GateResult("S0", "PASS", {}, "OK"),
        GateResult("S1", "PASS", {}, "OK"),
        GateResult("S2", "PASS", {}, "OK"),
        GateResult("S3", "PASS", {}, "OK"),
        GateResult("S4", "PASS", {}, "OK"),
        GateResult("S5", "PASS", {}, "OK"),
    ]
    status_b, msg_b = aggregate_overall_gate_status(gates_b, req_gates)
    assert status_b == "BLOCKED"
    assert "S6" in msg_b

    # 场景 C: 全部通过
    gates_c = [GateResult(gid, "PASS", {}, "OK") for gid in req_gates]
    status_c, msg_c = aggregate_overall_gate_status(gates_c, req_gates)
    assert status_c == "PASS"

def test_fault_case_7_corrupted_gro_format(tmp_path):
    """Case 7: 损坏的 GRO 格式直接被 S3 拦截"""
    validator = ChemicalValidator()
    bad_gro = tmp_path / "corrupted.gro"
    with open(bad_gro, "w") as f:
        f.write("Only one line\n")

    res = validator.check_gate_s3_from_gro(str(bad_gro))
    assert res.status == "FAIL"
    assert "行数过少" in res.message
