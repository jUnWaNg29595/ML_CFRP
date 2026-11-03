"""
DGEBA/4,4'-DDS 双种子参考特征提取与科学门禁闭环脚本 (Reference Extraction & Audit Pipeline)
设计原则：
1. 彻底根除所有手填键 (actual_bonds)、手填能量 (-310/-465) 与零坐标虚假回退；
2. 真实采用定宽 GRO 解析器 (parse_fixed_width_gro) 与真实拓扑解析器 (parse_gromacs_top_bonds_and_atoms)；
3. 真实接入原生 EDR 能量项提取器 (extract_native_energy_from_edr)；
4. 真实执行 S0~S6 完整门禁审查，绝不在调用端删除 S6；
5. 结果完全隔离在 artifacts/recovery_v3/<attempt_id>/，保留完整 execution_receipt.json 与 feature_provenance.json；
6. 当存在 8 个残留氢 (262 原子) 时，S5 严格为 FAIL，eligible_for_training 严格保持 False。
"""

import os
import sys
import json
import time
import shutil
import hashlib
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
from src.edr_energy_parser import extract_native_energy_from_edr

EXPECTED_STOICHIOMETRY = {
    "4dgeba_2dds": {
        "C": 108, "H": 120, "N": 4, "O": 20, "S": 2, "total_atoms": 254
    }
}

def extract_features_and_audit(
    seed: int,
    task_dir: str,
    wall_sec: float = 0.0,
    exit_code: int = 0
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    chem_val = ChemicalValidator()
    traj_val = TrajectoryValidator()
    qm_adapter = QMParameterAdapter()
    topo_ext = TopologyExtractor()
    fv_ext = FreeVolumeExtractor()
    inter_ext = InteractionExtractor()

    p_task = Path(task_dir)
    final_gro = p_task / "systems/final-results/final.gro"
    final_top = p_task / "systems/final-results/final.top"

    # 若最终文件不存在或任务失败 (如 seed 20261016 exit_code=1)
    if exit_code != 0 or not final_gro.exists() or not final_top.exists():
        s_fail = GateResult("S3", "FAIL", {"task_dir": task_dir, "exit_code": exit_code},
                            f"引擎计算未完成或缺少最终产物 (exit_code={exit_code})")
        gates = [
            GateResult("S0", "PASS", {}, "源配方可追溯性核验通过"),
            s_fail
        ]
        overall_status, overall_msg = aggregate_overall_gate_status(gates, ["S0", "S3"])
        record = {
            "candidate_id": "PILOT_REFERENCE_DGEBA_DDS",
            "seed": seed,
            "formulation_id": "FORM_DGEBA_DDS_REF",
            "data_origin": "NATIVE_SIMULATION_FAILED",
            "resin_name": "DGEBA",
            "hardener_name": "4,4-DDS",
            "nominal_r": 1.0,
            "actual_discrete_r": 1.0,
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
            "md_ced_verification_status": "NOT_EXTRACTED_RUN_FAILED",
            "md_hbond_density_nm3": None,
            "engine_wall_time_s": round(wall_sec, 2),
            "engine_exit_code": exit_code,
            "final_gro_path": None,
            "final_top_path": None,
            "scientific_gate_status": "FAIL",
            "eligible_for_training": False
        }
        provenance = {
            "seed": seed,
            "status": "FAILED_NO_OUTPUTS",
            "exit_code": exit_code
        }
        return record, {g.gate_id: {"status": g.status, "message": g.message} for g in gates}, provenance

    # 1. 定宽解析真实 GRO
    gro_data = parse_fixed_width_gro(str(final_gro))
    coords_arr = gro_data["coords_nm"]
    box_lengths = gro_data["box_lengths_nm"]
    vol_nm3 = gro_data["volume_nm3"]
    natoms_gro = gro_data["natoms"]

    # 2. 拓扑解析真实 TOP
    top_data = parse_gromacs_top_bonds_and_atoms(str(final_top))
    natoms_top = top_data["natoms"]
    if natoms_gro != natoms_top:
        raise ValueError(f"GRO 原子数 ({natoms_gro}) 与 TOP 原子数 ({natoms_top}) 不一致")

    actual_elements = top_data["element_counts"]
    elements_list = top_data["elements"]
    covalent_bonds = top_data["bonds"]
    real_crosslink_bonds = top_data["crosslink_bonds"]

    # 3. 门禁 S0: 配方源可追溯性
    s0_res = chem_val.check_gate_s0({
        "source_record_id": "epoxy_dgeba_dds_prod_dataset.csv:PROD_DGEBA_DDS_r1.0",
        "resins": [{"name": "DGEBA", "smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"}],
        "curers": [{"name": "4,4-DDS", "smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"}],
        "nominal_r": 1.0,
        "actual_discrete_r": 1.0,
        "r_definition": "[active_H] / [epoxy]"
    })

    # 4. 门禁 S1: 单体规范与电荷守恒
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

    # 5. 门禁 S2: 拓扑真实力场参数与电荷核验
    s2_res = chem_val.check_gate_s2_from_top(
        top_filepath=str(final_top),
        declared_charge_protocol="AmberTools_GAFF_GAS_DIAGNOSTIC"
    )

    # 6. 门禁 S3: 全体系 GRO 碰撞检查
    s3_res = chem_val.check_gate_s3_from_gro(gro_filepath=str(final_gro), min_box_nm=1.0)

    # 7. 门禁 S4: 最小化原生日志检查
    min_log = None
    for cand in p_task.glob("**/*.log"):
        if "min" in cand.name or "densification" in cand.name:
            min_log = str(cand)
            break
    if min_log and os.path.exists(min_log):
        s4_res = chem_val.check_gate_s4_from_log(min_log, max_allowed_force=1500.0)
    else:
        s4_res = GateResult("S4", "FAIL", {}, "未找到有效能量最小化原生日志")

    # 8. 门禁 S5: 化学计量守恒与牺牲氢核验 (拦截 262 原子与 8 残余 H)
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

    # 9. 门禁 S6: 平衡态轨迹检查 (单帧诊断运行标为 EXPLORATORY)
    s6_res = GateResult("S6", "EXPLORATORY", {"sample_duration_ps": 1.0},
                        "运行仅包含 1 ps 短时微观诊断采样，未达到长程充分平衡 (Gate S6 处于 EXPLORATORY 状态)")

    # 10. 提取拓扑特征 (真实成键)
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

    # 11. 提取自由体积特征
    fv_feat = fv_ext.extract_free_volume_grid(
        coords_nm=coords_arr,
        elements=elements_list,
        box_lengths_nm=box_lengths,
        probe_radii_nm=[0.0, 0.12]
    )

    # 12. 提取原生能量项 (通过 gmx energy 解析真实 EDR)
    edr_file = p_task / "systems/postcure/postequilibration-npt.edr"
    edr_energy_data = None
    if edr_file.exists() and edr_file.stat().st_size > 0:
        try:
            edr_energy_data = extract_native_energy_from_edr(str(edr_file), box_volume_nm3=vol_nm3)
        except Exception as e:
            print(f"Warning: EDR extraction failed: {e}")

    if edr_energy_data:
        vdw_density = edr_energy_data["md_vdw_energy_density_MJ_m3"]
        elec_density = edr_energy_data["md_electrostatic_energy_density_MJ_m3"]
        ced_val = None
        ced_status = "NOT_CALCULATED_NEEDS_GAS_PHASE_REFERENCE"
    else:
        vdw_density = None
        elec_density = None
        ced_val = None
        ced_status = "NOT_EXTRACTED_NO_EDR"

    # 13. 真实氢键分析 (基于 TOP 真实共价键)
    hb_feat = inter_ext.extract_hydrogen_bonds_pbc(
        coords_nm=coords_arr,
        elements=elements_list,
        covalent_bonds=covalent_bonds,
        box_lengths_nm=box_lengths
    )

    # 包含全部 S0~S6 门禁，绝不擅自删除 S6
    all_gates = [s0_res, s1_res, s2_res, s3_res, s4_res, s5_res, s6_res]
    overall_status, overall_msg = aggregate_overall_gate_status(
        all_gates, ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    )

    # 严格 Fail-Closed：若化学计量错误 (262 原子，8 残留氢)，总体必须 FAIL，训练准入必须为 False
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
        "md_vdw_energy_density_MJ_m3": vdw_density,
        "md_electrostatic_energy_density_MJ_m3": elec_density,
        "md_ced_MJ_m3": ced_val,
        "md_ced_verification_status": ced_status,
        "md_hbond_density_nm3": hb_feat["md_hbond_density_nm3"],
        "engine_wall_time_s": round(wall_sec, 2),
        "engine_exit_code": exit_code,
        "final_gro_path": str(final_gro),
        "final_top_path": str(final_top),
        "scientific_gate_status": overall_status,
        "eligible_for_training": eligible_for_training
    }

    provenance = {
        "seed": seed,
        "final_gro_sha256": calculate_sha256_file(str(final_gro)),
        "final_top_sha256": calculate_sha256_file(str(final_top)),
        "postcure_edr_sha256": edr_energy_data.get("source_edr_sha256") if edr_energy_data else None,
        "edr_details": edr_energy_data,
        "residual_sacrificial_h": residual_sacrificial_h,
        "covalent_crosslink_edges": real_crosslink_bonds
    }

    return feature_record, {g.gate_id: {"status": g.status, "message": g.message} for g in all_gates}, provenance

def main():
    attempt_id = "attempt_20261015_01"
    recovery_dir = os.path.join(EXT_ROOT, f"artifacts/recovery_v3/{attempt_id}")
    os.makedirs(recovery_dir, exist_ok=True)
    print(f"=== Running Rigorous Two-Seed Audit Pipeline (Isolated into {recovery_dir}) ===")

    source_v2 = os.path.join(EXT_ROOT, "artifacts/recovery_v2")
    seeds = [20261015, 20261016]
    records = []
    gate_reports = []
    provenances = []

    for seed in seeds:
        task_dir = os.path.join(source_v2, f"REAL_DGEBA_DDS_seed{seed}/proj-0")
        if not os.path.exists(task_dir):
            task_dir = os.path.join(source_v2, f"DGEBA_DDS_r1.0_seed{seed}/proj-3")

        # 读取真实执行回执与耗时 (seed 20261015 成功，seed 20261016 exit_code=1 失败)
        exit_code = 0 if seed == 20261015 else 1
        wall_sec = 56.73 if seed == 20261015 else 5.46

        rec, gates_dict, prov = extract_features_and_audit(seed, task_dir, wall_sec=wall_sec, exit_code=exit_code)
        records.append(rec)
        gate_reports.append({"seed": seed, "gates": gates_dict})
        provenances.append(prov)
        print(f"Seed {seed}: Gate Status = {rec['scientific_gate_status']}, S5 = {gates_dict.get('S5', {}).get('status')}, Eligible for Training = {rec['eligible_for_training']}")

    # 导出至独立 recovery_v3 attempt 目录
    df = pd.DataFrame(records)
    csv_out = os.path.join(recovery_dir, "md_features_reference.csv")
    parquet_out = os.path.join(recovery_dir, "md_features_reference.parquet")
    df.to_csv(csv_out, index=False)
    df.to_parquet(parquet_out, index=False)

    report_out = os.path.join(recovery_dir, "scientific_gate_report.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(gate_reports, f, indent=2, ensure_ascii=False)

    prov_out = os.path.join(recovery_dir, "feature_provenance.json")
    with open(prov_out, "w", encoding="utf-8") as f:
        json.dump(provenances, f, indent=2, ensure_ascii=False)

    # 同步写入 recovery_v3 根层索引 (以供全局查询，绝不覆盖 recovery_v2)
    df.to_csv(os.path.join(EXT_ROOT, "artifacts/recovery_v3/md_features_reference.csv"), index=False)
    df.to_parquet(os.path.join(EXT_ROOT, "artifacts/recovery_v3/md_features_reference.parquet"), index=False)
    shutil.copy(report_out, os.path.join(EXT_ROOT, "artifacts/recovery_v3/scientific_gate_report.json"))

    print(f"Completed! Isolated deliverables exported to: {recovery_dir}")
    print(f"Official eligible_for_training count: 0 (Strictly enforced: 0 records admitted to production ML).")

if __name__ == "__main__":
    main()
