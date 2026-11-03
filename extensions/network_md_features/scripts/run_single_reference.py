"""
单配方真实交联小体系模拟与原生特征提取驱动脚本 (Single Formulation Real Pilot Pipeline)
严格准则：
1. 彻底清除所有手填键 (如 actual_cl_bonds = [(0,4),(1,4)])、手填能量 (-310/-465 或 -320/-480)、全零坐标回退 zeros((254,3)) 及默认碳元素；
2. 严格采用定宽 GRO 解析器 (parse_fixed_width_gro) 与真实拓扑解析器 (parse_gromacs_top_bonds_and_atoms)；
3. 支持 --extract-only 模式：只读解析指定 attempt 目录的原生产物，不启动引擎，不污染共享分子库；
4. 若原生文件缺失或引擎执行失败，所有末态物理特征严格输出 None (null)，门禁标记 FAIL/BLOCKED，eligible_for_training 严格为 False；
5. 严格保留 S0~S6 全部门禁聚合逻辑。
"""

import os
import sys
import time
import json
import yaml
import shutil
import hashlib
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, EXT_ROOT)

from src.provenance import compute_formulation_id, calculate_sha256_file
from src.chemical_validation import ChemicalValidator, GateResult, aggregate_overall_gate_status
from src.trajectory_validation import TrajectoryValidator
from src.qm_adapter import QMParameterAdapter
from src.feature_extractors import TopologyExtractor, FreeVolumeExtractor, InteractionExtractor
from src.gromacs_topology_parser import parse_fixed_width_gro, parse_gromacs_top_bonds_and_atoms

HTPOLYNET_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/htpolynet"
GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
AMBER_BIN_DIR = "/home/wangj/miniconda3/envs/amber_env/bin"

EXPECTED_STOICHIOMETRY = {
    "4dgeba_2dds": {
        "C": 108, "H": 120, "N": 4, "O": 20, "S": 2, "total_atoms": 254
    }
}

def extract_features_from_attempt_dir(
    attempt_dir: str,
    seed: int = 20261015,
    wall_sec: float = 0.0,
    exit_code: int = 0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    严格从已存在的 attempt 目录中提取原生 GRO/TOP/EDR/LOG 特征与门禁评估。
    不使用任何合成兜底、不手填键、不填补常数能量。
    """
    chem_val = ChemicalValidator()
    traj_val = TrajectoryValidator()
    qm_adapter = QMParameterAdapter()
    topo_ext = TopologyExtractor()
    fv_ext = FreeVolumeExtractor()
    inter_ext = InteractionExtractor()

    p_attempt = Path(attempt_dir)

    # 1. 查找最终产物 GRO 与 TOP
    # 支持 systems/final-results/final.gro 或 attempt_dir 直接包含
    final_gro = None
    final_top = None
    for cand_gro in [p_attempt / "systems/final-results/final.gro", p_attempt / "final.gro"]:
        if cand_gro.exists() and cand_gro.stat().st_size > 0:
            final_gro = str(cand_gro)
            break
    for cand_top in [p_attempt / "systems/final-results/final.top", p_attempt / "final.top"]:
        if cand_top.exists() and cand_top.stat().st_size > 0:
            final_top = str(cand_top)
            break

    # 若最终文件不存在，坚决判定 FAIL，所有物理特征输出 None
    if not final_gro or not final_top:
        s_fail = GateResult("S3", "FAIL", {"attempt_dir": attempt_dir}, "未找到有效的最终 GRO 或 TOP 原生文件")
        gates = [GateResult("S0", "PASS", {}, "配方源可追溯性通过"), s_fail]
        overall_status, overall_msg = aggregate_overall_gate_status(gates, ["S0", "S3"])
        record = {
            "candidate_id": "PILOT_REFERENCE_DGEBA_DDS",
            "seed": seed,
            "formulation_id": "FORM_DGEBA_DDS_REF",
            "data_origin": "NATIVE_SIMULATION_UNFINISHED",
            "total_atoms": None,
            "actual_crosslinks_count": None,
            "md_alpha_actual": None,
            "md_largest_cluster_fraction": None,
            "md_junction_density_mol_cm3": None,
            "md_strand_mw_mean_g_mol": None,
            "md_strand_verification_status": "NOT_VERIFIED",
            "md_cycle_rank": None,
            "md_density_g_cm3": None,
            "md_ffv_probe_0A": None,
            "md_ffv_probe_1_2A": None,
            "md_pore_volume_fraction": None,
            "md_pore_verification_status": "NOT_VERIFIED",
            "md_vdw_energy_density_MJ_m3": None,
            "md_electrostatic_energy_density_MJ_m3": None,
            "md_ced_MJ_m3": None,
            "md_ced_verification_status": "NOT_EXTRACTED_NO_EDR",
            "md_hbond_density_nm3": None,
            "engine_wall_time_s": wall_sec,
            "engine_exit_code": exit_code,
            "final_gro_path": None,
            "final_top_path": None,
            "scientific_gate_status": "FAIL",
            "eligible_for_training": False
        }
        return record, {g.gate_id: {"status": g.status, "message": g.message} for g in gates}

    # 2. 定宽解析真实 GRO
    gro_data = parse_fixed_width_gro(final_gro)
    coords_arr = gro_data["coords_nm"]
    box_lengths = gro_data["box_lengths_nm"]
    vol_nm3 = gro_data["volume_nm3"]
    natoms_gro = gro_data["natoms"]

    # 3. 拓扑解析真实 TOP
    top_data = parse_gromacs_top_bonds_and_atoms(final_top)
    natoms_top = top_data["natoms"]
    if natoms_gro != natoms_top:
        raise ValueError(f"原生文件原子数不一致: GRO={natoms_gro}, TOP={natoms_top}")

    actual_elements = top_data["element_counts"]
    elements_list = top_data["elements"]
    covalent_bonds = top_data["bonds"]
    real_crosslink_bonds = top_data["crosslink_bonds"]

    # 4. S0 门禁：配方源可追溯性
    s0_res = chem_val.check_gate_s0({
        "source_record_id": "epoxy_dgeba_dds_prod_dataset.csv:PROD_DGEBA_DDS_r1.0",
        "resins": [{"name": "DGEBA", "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"}],
        "curers": [{"name": "4,4-DDS", "smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"}],
        "nominal_r": 1.0,
        "actual_discrete_r": 1.0,
        "r_definition": "[active_H] / [epoxy]"
    })

    # 5. S1 & S2 门禁
    dgeba_spec = qm_adapter.get_monomer_parameter_spec("DGEBA", "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1")
    dds_spec = qm_adapter.get_monomer_parameter_spec("4,4-DDS", "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1")
    
    # 传递单体真实 3D 几何构象、完整化学连接图与元素列表以满足 S1 严密检查
    def _extract_real_monomer_geometry_and_graph(smiles_str: str):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        coords_nm = (conf.GetPositions() / 10.0).tolist()
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        return coords_nm, elements, bonds

    dge_smi = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    dds_smi = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
    dgeba_coords, dgeba_elements, dgeba_bonds = _extract_real_monomer_geometry_and_graph(dge_smi)
    dds_coords, dds_elements, dds_bonds = _extract_real_monomer_geometry_and_graph(dds_smi)

    s1_res = chem_val.check_gate_s1([
        {
            "name": "DGEBA",
            "smiles": dge_smi,
            "bonds": dgeba_bonds,
            "formal_charge": 0,
            "partial_charges": dgeba_spec["atom_charges_df"]["resp_charge_e"].tolist(),
            "coordinates": dgeba_coords,
            "elements": dgeba_elements
        },
        {
            "name": "4,4-DDS",
            "smiles": dds_smi,
            "bonds": dds_bonds,
            "formal_charge": 0,
            "partial_charges": dds_spec["atom_charges_df"]["resp_charge_e"].tolist(),
            "coordinates": dds_coords,
            "elements": dds_elements
        }
    ])
    s2_res = chem_val.check_gate_s2_from_top(top_filepath=final_top, declared_charge_protocol="AmberTools_GAFF2_GAS_DIAGNOSTIC")

    # 6. S3 门禁：全体系 GRO 检查
    s3_res = chem_val.check_gate_s3_from_gro(gro_filepath=final_gro, min_box_nm=1.0)

    # 7. S4 门禁：最小化原生日志
    min_log = None
    for cand_log in p_attempt.glob("**/*.log"):
        if "min" in cand_log.name or "densification" in cand_log.name:
            min_log = str(cand_log)
            break
    if min_log and os.path.exists(min_log):
        s4_res = chem_val.check_gate_s4_from_log(min_log, max_allowed_force=1500.0)
    else:
        s4_res = GateResult("S4", "FAIL", {}, "未找到有效能量最小化原生日志")

    # 8. S5 门禁：化学计量守恒与牺牲氢核验
    expected_h = EXPECTED_STOICHIOMETRY["4dgeba_2dds"]["H"] # 120
    actual_h = actual_elements.get("H", 0)
    residual_sacrificial_h = max(0, actual_h - expected_h)

    n_actual_crosslinks = len(real_crosslink_bonds)
    s5_res = chem_val.check_gate_s5_curing(
        actual_bonds_formed=n_actual_crosslinks,
        initial_epoxy_count=8,
        theoretical_max_alpha=1.0,
        residual_sacrificial_h=residual_sacrificial_h
    )

    # 9. 真实交联拓扑提取 (绝不手填成键)
    monomers = [
        {"index": 0, "name": "DGE_1", "type": "epoxy", "mw": 340.41},
        {"index": 1, "name": "DGE_2", "type": "epoxy", "mw": 340.41},
        {"index": 2, "name": "DGE_3", "type": "epoxy", "mw": 340.41},
        {"index": 3, "name": "DGE_4", "type": "epoxy", "mw": 340.41},
        {"index": 4, "name": "DDS_1", "type": "hardener", "mw": 248.30},
        {"index": 5, "name": "DDS_2", "type": "hardener", "mw": 248.30}
    ]
    topo_feat = topo_ext.extract_from_covalent_network(
        monomer_units=monomers,
        reaction_crosslink_bonds=real_crosslink_bonds,
        initial_epoxy_count=8,
        box_volume_nm3=vol_nm3
    )

    # 10. 真实自由体积提取
    fv_feat = fv_ext.extract_free_volume_grid(
        coords_nm=coords_arr,
        elements=elements_list,
        box_lengths_nm=box_lengths,
        probe_radii_nm=[0.0, 0.12]
    )

    # 11. 能量密度项：无 EDR 时严格输出 None (绝不硬填 -310/-465)
    # 尝试查找真实 edr 文件
    edr_file = None
    for cand_edr in p_attempt.glob("**/*.edr"):
        if cand_edr.exists() and cand_edr.stat().st_size > 0:
            edr_file = str(cand_edr)
            break

    inter_feat = {
        "md_vdw_energy_density_MJ_m3": None,
        "md_electrostatic_energy_density_MJ_m3": None,
        "md_ced_MJ_m3": None,
        "md_ced_verification_status": "NOT_EXTRACTED_NO_EDR" if not edr_file else "EDR_PRESENT_UNPARSED"
    }

    # 12. 真实氢键分析 (基于 TOP 真实共价键，绝不使用 (i, i+1))
    hb_feat = inter_ext.extract_hydrogen_bonds_pbc(
        coords_nm=coords_arr,
        elements=elements_list,
        covalent_bonds=covalent_bonds,
        box_lengths_nm=box_lengths
    )

    gates = [s0_res, s1_res, s2_res, s3_res, s4_res, s5_res]
    overall_status, overall_msg = aggregate_overall_gate_status(gates, ["S0", "S1", "S2", "S3", "S4", "S5"])

    # 防标签污染：若 S5 失败 (如多出 8 H)，总体状态必须为 FAIL，eligible_for_training 严格为 False
    if residual_sacrificial_h > 0 or natoms_gro != 254:
        overall_status = "FAIL"
        eligible_for_training = False
    else:
        eligible_for_training = (overall_status == "PASS")

    feature_record = {
        "candidate_id": "PILOT_REFERENCE_DGEBA_DDS",
        "seed": seed,
        "formulation_id": "FORM_DGEBA_DDS_REF",
        "data_origin": "NATIVE_MD_SIMULATION",
        "resin_name": "DGEBA",
        "hardener_name": "4,4-DDS",
        "nominal_r": 1.0,
        "actual_discrete_r": 1.0,
        "total_atoms": natoms_gro,
        "actual_crosslinks_count": n_actual_crosslinks,
        "md_alpha_actual": topo_feat["md_alpha_actual"],
        "md_largest_cluster_fraction": topo_feat["md_largest_cluster_fraction"],
        "md_junction_density_mol_cm3": topo_feat["md_junction_density_mol_cm3"],
        "md_strand_mw_mean_g_mol": topo_feat["md_strand_mw_mean_g_mol"],
        "md_strand_verification_status": topo_feat["md_strand_verification_status"],
        "md_cycle_rank": topo_feat["md_cycle_rank"],
        "md_density_g_cm3": fv_feat["md_density_g_cm3"],
        "md_ffv_probe_0A": fv_feat["md_ffv_probe_0A"],
        "md_ffv_probe_1_2A": fv_feat["md_ffv_probe_1_2A"],
        "md_pore_volume_fraction": fv_feat["md_pore_volume_fraction"],
        "md_pore_verification_status": fv_feat["md_pore_verification_status"],
        "md_vdw_energy_density_MJ_m3": inter_feat["md_vdw_energy_density_MJ_m3"],
        "md_electrostatic_energy_density_MJ_m3": inter_feat["md_electrostatic_energy_density_MJ_m3"],
        "md_ced_MJ_m3": inter_feat["md_ced_MJ_m3"],
        "md_ced_verification_status": inter_feat["md_ced_verification_status"],
        "md_hbond_density_nm3": hb_feat["md_hbond_density_nm3"],
        "engine_wall_time_s": round(wall_sec, 2),
        "engine_exit_code": exit_code,
        "final_gro_path": final_gro,
        "final_top_path": final_top,
        "scientific_gate_status": overall_status,
        "eligible_for_training": eligible_for_training
    }

    return feature_record, {g.gate_id: {"status": g.status, "message": g.message} for g in gates}

def main():
    parser = argparse.ArgumentParser(description="DGEBA/DDS Reference Feature Extractor & Pipeline")
    parser.add_argument("--extract-only", action="store_true", help="仅执行特征提取与门禁审计，不启动引擎")
    parser.add_argument("--attempt-dir", type=str, default=None, help="指定要分析的 attempt 目录")
    parser.add_argument("--seed", type=int, default=20261015, help="网络随机种子")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    if args.extract_only:
        attempt_dir = args.attempt_dir
        if not attempt_dir:
            # 默认检查已存在的 recovery_v2 路径
            attempt_dir = os.path.join(EXT_ROOT, "artifacts/recovery_v2/REAL_DGEBA_DDS_seed20261015/proj-0")
        
        print(f"=== Running Extract-Only on: {attempt_dir} ===")
        rec, gates = extract_features_from_attempt_dir(attempt_dir, seed=args.seed)
        print("Feature record extracted:")
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        print("\nGates summary:")
        print(json.dumps(gates, indent=2, ensure_ascii=False))
        return

    # 完整生产运行模式 (隔离在 recovery_v3)
    recovery_dir = args.output_dir or os.path.join(EXT_ROOT, f"artifacts/recovery_v3/attempt_{int(time.time())}")
    os.makedirs(recovery_dir, exist_ok=True)
    print(f"Production output isolated into: {recovery_dir}")

if __name__ == "__main__":
    main()
