"""
独立复核反例回归测试套件 (Audit Gate Attack Regression Tests)
覆盖审计报告中发现的 4 项严重门禁绕过漏洞 (Counterexamples A, B, C, D):
A. S2: 无力场参数仅含 UNKNOWN 原子类型与占位成键拓扑，标 RESP 必须被拦截 (FAIL);
B. S1: 仅传电荷列表而无 3D 几何坐标与化学图，必须被拦截 (FAIL);
C. S4: 单行伪造日志且无收敛标识 (log_converged=False)，必须被拦截 (FAIL);
D. S3: 202 原子体系中第 201/202 原子重叠 (< 0.05 nm)，废除 200 原子硬截断后必须被全体系检测拦截 (FAIL).
"""

import os
import json
import tempfile
import hashlib
import numpy as np
import pytest
from pathlib import Path

from src.chemical_validation import ChemicalValidator

@pytest.fixture
def validator():
    return ChemicalValidator()

def test_counterexample_a_s2_unknown_types_and_unparameterized_top(validator, tmp_path):
    """反例 A: 拓扑包含 UNKNOWN 类型与无参数成键，即使声明 DFT_RESP_B3LYP_631G* 也必须 FAIL"""
    top_file = tmp_path / "invalid.top"
    top_file.write_text(
        "[ atoms ]\n"
        "1 UNKNOWN 1 FAKE C1 1 0.15 12.011\n"
        "2 UNKNOWN 1 FAKE C2 2 -0.15 12.011\n"
        "[ bonds ]\n"
        "1 2 1\n"
    )
    res = validator.check_gate_s2_from_top(str(top_file), declared_charge_protocol="DFT_RESP_B3LYP_631G*")
    assert res.status == "FAIL"
    assert "UNKNOWN" in res.message or "缺少真实力场参数覆盖" in res.message

def test_counterexample_b_s1_no_graph_no_coords(validator):
    """反例 B: 仅传电荷列表，无化学图与 3D 坐标，门禁 S1 必须 FAIL"""
    bad_input = [{'name': 'not_a_verified_molecule', 'formal_charge': 0, 'partial_charges': [0.15, -0.15]}]
    res = validator.check_gate_s1(bad_input)
    assert res.status == "FAIL"
    assert "缺少 3D 几何坐标" in res.message

def test_counterexample_c_s4_single_line_synthetic_log_unconverged(validator, tmp_path):
    """反例 C: 单行伪造日志，无收敛签名 (log_converged=False)，门禁 S4 必须 FAIL"""
    log_file = tmp_path / "synthetic.log"
    log_file.write_text("Maximum force = 100.0\n")
    res = validator.check_gate_s4_from_log(str(log_file))
    assert res.status == "FAIL"
    assert "伪造" in res.message or "未收敛" in res.message or "缺少真实" in res.message

def test_counterexample_d_s3_overlap_at_atoms_201_202(validator, tmp_path):
    """反例 D: 202 原子 GRO 文件，前 200 个原子正常，第 201 与 202 原子重叠，全体系 PBC 检测必须 FAIL"""
    coords = [(0.2 + (i % 10) * 0.3, 0.2 + (i // 10 % 10) * 0.3, 0.2 + (i // 100) * 0.3) for i in range(200)]
    coords += [(3.5, 3.5, 3.5), (3.5, 3.5, 3.5)]  # 完全重叠的第 201 和 202 原子
    gro_file = tmp_path / "late_collision.gro"
    gro_content = (
        "audit fixture\n"
        "202\n" +
        "".join(f"{1:5d}{'DGE':<5}{'C':>5}{i+1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n" for i, (x, y, z) in enumerate(coords)) +
        "4.00000 4.00000 4.00000\n"
    )
    gro_file.write_text(gro_content)
    res = validator.check_gate_s3_from_gro(str(gro_file))
    assert res.status == "FAIL"
    assert "碰撞重叠" in res.message
    assert res.details.get("clash_count", 0) >= 1
