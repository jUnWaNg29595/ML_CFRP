"""
严密的数据来源与可追溯性管理模块 (Provenance & Identity Management)
遵循最高审计要求：区分配方身份、参数集身份、协议身份、网络身份与特征分析身份。
真实记录每一次运行的软硬件版本、输入输出 SHA256、执行命令与退出码。
"""

import os
import sys
import json
import hashlib
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

def calculate_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def calculate_sha256_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_formulation_id(
    resins: List[Dict[str, Any]],
    curers: List[Dict[str, Any]],
    stoichiometric_ratio_r: float,
    r_definition: str = "active_H_to_epoxy"
) -> str:
    """
    计算规范配方身份哈希 (formulation_id)。
    对组分结构规范化排序，对角色顺序变化保持幂等，不同配比和倒数定义不发生哈希碰撞。
    """
    norm_resins = sorted([
        {
            "smiles": str(r.get("smiles", "")).strip(),
            "name": str(r.get("name", "")).strip(),
            "phr": round(float(r.get("phr", 100.0)), 4),
            "molecules": int(r.get("molecules", 0))
        }
        for r in resins
    ], key=lambda x: x["smiles"])

    norm_curers = sorted([
        {
            "smiles": str(c.get("smiles", "")).strip(),
            "name": str(c.get("name", "")).strip(),
            "phr": round(float(c.get("phr", 0.0)), 4),
            "molecules": int(c.get("molecules", 0))
        }
        for c in curers
    ], key=lambda x: x["smiles"])

    payload = {
        "resins": norm_resins,
        "curers": norm_curers,
        "r": round(float(stoichiometric_ratio_r), 4),
        "r_definition": r_definition
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return "FORM_" + hashlib.sha256(dumped.encode('utf-8')).hexdigest()[:16]

def compute_parameter_set_id(
    forcefield: str,
    charge_model: str,
    dft_reference_path: str = "",
    product_templates_hash: str = ""
) -> str:
    """计算参数与电荷方案哈希 (parameter_set_id)"""
    payload = {
        "forcefield": forcefield,
        "charge_model": charge_model,
        "dft_reference_path": dft_reference_path,
        "product_templates_hash": product_templates_hash
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return "PARAM_" + hashlib.sha256(dumped.encode('utf-8')).hexdigest()[:16]

def compute_protocol_id(
    target_conversion: float,
    cure_temperature_K: float,
    cure_pressure_bar: float,
    box_initial_density_kg_m3: float,
    min_cutoff_nm: float,
    max_search_radius_nm: float
) -> str:
    """计算模拟制备与交联动力学协议哈希 (protocol_id)"""
    payload = {
        "target_conversion": round(float(target_conversion), 4),
        "temperature_K": round(float(cure_temperature_K), 2),
        "pressure_bar": round(float(cure_pressure_bar), 2),
        "initial_density": round(float(box_initial_density_kg_m3), 2),
        "min_cutoff_nm": round(float(min_cutoff_nm), 4),
        "max_search_radius_nm": round(float(max_search_radius_nm), 4)
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return "PROT_" + hashlib.sha256(dumped.encode('utf-8')).hexdigest()[:16]

def compute_network_id(
    formulation_id: str,
    parameter_set_id: str,
    protocol_id: str,
    seed: int,
    final_graph_hash: str = ""
) -> str:
    """计算独立三维交联网络身份哈希 (network_id)"""
    payload = {
        "formulation_id": formulation_id,
        "parameter_set_id": parameter_set_id,
        "protocol_id": protocol_id,
        "seed": int(seed),
        "final_graph_hash": final_graph_hash
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return "NET_" + hashlib.sha256(dumped.encode('utf-8')).hexdigest()[:16]

def compute_feature_run_id(
    network_id: str,
    extractor_version: str,
    extract_settings_hash: str
) -> str:
    """计算物理特征提取批次哈希 (feature_run_id)"""
    payload = {
        "network_id": network_id,
        "extractor_version": extractor_version,
        "extract_settings_hash": extract_settings_hash
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return "FRUN_" + hashlib.sha256(dumped.encode('utf-8')).hexdigest()[:16]

@dataclass
class ExecutionReceipt:
    job_id: str
    network_id: str
    command: List[str]
    working_directory: str
    start_time: str
    end_time: str
    exit_code: int
    software_versions: Dict[str, str]
    input_files_sha256: Dict[str, str]
    output_files_sha256: Dict[str, str]
    scientific_gate_status: str  # PASS / FAIL / BLOCKED / EXPLORATORY
    gate_details: Dict[str, Any]
    peak_memory_mb: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    native_log_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
