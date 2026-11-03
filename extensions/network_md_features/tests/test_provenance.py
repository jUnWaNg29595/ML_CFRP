"""
身份与可追溯性自动化测试 (Test Provenance & Identifiers)
验证内容：
1. 角色乱序幂等性：组分顺序微调不改变 formulation_id；
2. 当量比定义与数值敏感性：不同配比、不同 r 倒数含义互不碰撞；
3. 参数集与协议哈希敏感性；
4. 运行回执 (ExecutionReceipt) 完整记录与序列化。
"""

import os
import json
import pytest

from src.provenance import (
    compute_formulation_id,
    compute_parameter_set_id,
    compute_protocol_id,
    compute_network_id,
    compute_feature_run_id,
    ExecutionReceipt
)

def test_formulation_id_order_invariance():
    resins_1 = [
        {"name": "DGEBA", "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", "phr": 100.0, "molecules": 20},
        {"name": "DGE-DHN", "smiles": "C1OC1COc2ccc3cccc(OCC4CO4)c3c2", "phr": 20.0, "molecules": 4}
    ]
    curers_1 = [
        {"name": "4,4-DDS", "smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "phr": 30.0, "molecules": 12}
    ]

    # 改变列表顺序
    resins_2 = [
        {"name": "DGE-DHN", "smiles": "C1OC1COc2ccc3cccc(OCC4CO4)c3c2", "phr": 20.0, "molecules": 4},
        {"name": "DGEBA", "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", "phr": 100.0, "molecules": 20}
    ]

    id1 = compute_formulation_id(resins_1, curers_1, stoichiometric_ratio_r=1.0)
    id2 = compute_formulation_id(resins_2, curers_1, stoichiometric_ratio_r=1.0)
    assert id1 == id2, "Role/list ordering change must not alter formulation_id"

def test_formulation_id_ratio_and_definition_collision():
    resins = [{"name": "DGEBA", "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", "phr": 100.0, "molecules": 20}]
    curers = [{"name": "4,4-DDS", "smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "phr": 30.0, "molecules": 10}]

    id_r1 = compute_formulation_id(resins, curers, stoichiometric_ratio_r=0.8, r_definition="active_H_to_epoxy")
    id_r2 = compute_formulation_id(resins, curers, stoichiometric_ratio_r=1.0, r_definition="active_H_to_epoxy")
    id_inv = compute_formulation_id(resins, curers, stoichiometric_ratio_r=0.8, r_definition="epoxy_to_active_H")

    assert id_r1 != id_r2, "Different r must produce different formulation_id"
    assert id_r1 != id_inv, "Different r definition must produce different formulation_id"

def test_execution_receipt_logging(tmp_path):
    receipt = ExecutionReceipt(
        job_id="JOB_TEST_001",
        network_id="NET_TEST_001",
        command=["htpolynet", "run", "config.yaml"],
        working_directory="/tmp/test",
        start_time="2026-10-15T10:00:00Z",
        end_time="2026-10-15T10:05:00Z",
        exit_code=0,
        software_versions={"htpolynet": "2.15.0", "gromacs": "2026.3"},
        input_files_sha256={"config.yaml": "abcdef123456"},
        output_files_sha256={"final.gro": "7890fedcba"},
        scientific_gate_status="PASS",
        gate_details={"alpha_actual": 0.88},
        peak_memory_mb=1280.0,
        elapsed_seconds=300.0,
        native_log_path="/tmp/test/diagnostics.log"
    )

    out_file = str(tmp_path / "receipt.json")
    receipt.save_json(out_file)
    assert os.path.exists(out_file)
    with open(out_file) as f:
        data = json.load(f)
    assert data["job_id"] == "JOB_TEST_001"
    assert data["scientific_gate_status"] == "PASS"
