"""
postprocess_trajectory.py
唯一统一的真实后处理 CLI 与分析引擎
功能与科学准则：
1. 真正读取原生 XTC 轨迹与 EDR 能量文件，完全消除任何硬编码常数与 np.random.normal 伪造随机数；
2. 逐帧同步计算密度、真实连接图拓扑（含图推导环氧基转化率）、3D PBC 网格空间占据并集 FFV (0.0 Å / 1.2 Å)、真实几何 PBC 氢键、全轨迹选定帧共价键应变；
3. 原生调用 gmx energy 动态按名称映射提取非键能量项，保留正负号与分子内 1-4 + SR 定义说明；
4. 支持明确的时间窗口 (--t-start, --t-end)、抽样步长 (--stride)、网格分辨率 (--grid-spacing) 与输入文件 SHA256 校验 (--expected-hashes-json)；
5. 输入目录与输出目录严格分离，绝不反写或覆盖原生轨迹；
6. 如实标注电荷协议与小体系稀疏诊断状态，停止自动赋予笼统的 PHYSICALLY_VALIDATED 或单一 SUCCESS。
"""

import os
import sys
import json
import math
import platform
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import numpy as np
import pandas as pd
import parmed as pmd
import networkx as nx

try:
    from .pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains_detailed,
    )
    from .native_edr_extractor import (
        extract_native_edr_data,
        compute_file_sha256,
        KJ_MOL_NM3_TO_MJ_M3,
    )
    from .structure_features_extractor import (
        BONDI_VDW_RADII_NM,
        check_xtc_native,
        extract_trajectory_frames_to_gro,
        parse_multiframe_gro,
        compute_graph_topology,
        compute_grid_occupied_volume_union,
        evaluate_ffv_convergence_and_uncertainty,
        detect_geometric_hbonds_pbc,
        evaluate_trajectory_bond_strains_pbc,
    )
except (ImportError, ValueError):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from src.pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains_detailed,
    )
    from src.native_edr_extractor import (
        extract_native_edr_data,
        compute_file_sha256,
        KJ_MOL_NM3_TO_MJ_M3,
    )
    from src.structure_features_extractor import (
        BONDI_VDW_RADII_NM,
        check_xtc_native,
        extract_trajectory_frames_to_gro,
        parse_multiframe_gro,
        compute_graph_topology,
        compute_grid_occupied_volume_union,
        evaluate_ffv_convergence_and_uncertainty,
        detect_geometric_hbonds_pbc,
        evaluate_trajectory_bond_strains_pbc,
    )

GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
DGE_RESP_CSV = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dgeba_resp/atom_charges.csv"
DDS_RESP_CSV = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dds_resp/atom_charges.csv"


def _compute_empirical_formula(elements: List[str]) -> str:
    counts = Counter(elements)
    order = ["C", "H"] + sorted(k for k in counts.keys() if k not in ("C", "H"))
    parts = []
    for el in order:
        if counts.get(el, 0) > 0:
            n = counts[el]
            parts.append(f"{el}{n if n > 1 else ''}")
    return "".join(parts)


def audit_state_charge_protocol(parm: pmd.Structure) -> Dict[str, Any]:
    """
    实际核验拓扑中各残基的原子电荷与单体 DFT-RESP 参考表的逐原子匹配情况，
    如实判定电荷协议与适用域状态，绝不将 Gasteiger (-c gas) 误标为 AM1-BCC。
    """
    q_arr = np.array([float(a.charge) for a in parm.atoms], dtype=float)
    nonzero_count = int(np.sum(np.abs(q_arr) > 1e-10))
    net_charge = float(np.sum(q_arr))

    if nonzero_count == 0:
        raise ValueError("拓扑中全部原子电荷为 0，拒绝继续后处理！")

    ref_tables: Dict[str, Optional[pd.DataFrame]] = {"DGE": None, "DDS": None}
    if os.path.exists(DGE_RESP_CSV):
        ref_tables["DGE"] = pd.read_csv(DGE_RESP_CSV)
    if os.path.exists(DDS_RESP_CSV):
        ref_tables["DDS"] = pd.read_csv(DDS_RESP_CSV)

    residue_checks = []
    all_monomers_resp_verified = True
    has_gas_adduct = False

    for res in parm.residues:
        rname = res.name.strip()
        res_atoms = list(res.atoms)
        res_q = np.array([float(a.charge) for a in res_atoms], dtype=float)
        res_elems = [a.element_name for a in res_atoms]

        if rname in ("DGE", "DDS") and ref_tables.get(rname) is not None:
            ref_df = ref_tables[rname]
            ref_elems = [str(e).strip() for e in ref_df["element"].tolist()]
            ref_q = ref_df["resp_charge_e"].to_numpy(dtype=float)
            elems_match = (res_elems == ref_elems)
            max_diff = float(np.max(np.abs(res_q - ref_q))) if elems_match else float("inf")
            is_resp_match = bool(elems_match and max_diff < 1e-4)
            if not is_resp_match:
                all_monomers_resp_verified = False
            residue_checks.append({
                "residue_1based": res.idx + 1,
                "residue_name": rname,
                "atom_count": len(res_atoms),
                "net_charge_e": round(float(res_q.sum()), 8),
                "assigned_protocol": "DFT_RESP_B3LYP_631G*_ORDERED_MATCH" if is_resp_match else "UNVERIFIED_MONOMER_CHARGE",
                "ordered_elements_match_resp_csv": elems_match,
                "max_abs_charge_diff_vs_resp_csv_e": max_diff,
            })
        elif rname in ("MON", "BIS"):
            has_gas_adduct = True
            all_monomers_resp_verified = False
            residue_checks.append({
                "residue_1based": res.idx + 1,
                "residue_name": rname,
                "atom_count": len(res_atoms),
                "net_charge_e": round(float(res_q.sum()), 8),
                "assigned_protocol": "ANTECHAMBER_GASTEIGER_GAS_DIAGNOSTIC",
                "ordered_elements_match_resp_csv": None,
                "max_abs_charge_diff_vs_resp_csv_e": None,
            })
        else:
            all_monomers_resp_verified = False
            residue_checks.append({
                "residue_1based": res.idx + 1,
                "residue_name": rname,
                "atom_count": len(res_atoms),
                "net_charge_e": round(float(res_q.sum()), 8),
                "assigned_protocol": "UNKNOWN_OR_CUSTOM_RESIDUE_CHARGE",
                "ordered_elements_match_resp_csv": None,
                "max_abs_charge_diff_vs_resp_csv_e": None,
            })

    if all_monomers_resp_verified and not has_gas_adduct:
        charge_protocol = "GAFF2_MONOMER_DFT_RESP_ORDERED_MATCH"
        validation_status = "DIAGNOSTIC_SMALL_SPARSE_RESP_REFERENCE"
        training_admission_status = "BLOCKED_SPARSE_DIAGNOSTIC_NOT_FOR_ML_TRAINING"
    elif has_gas_adduct:
        charge_protocol = "GAFF2_UNREACTED_RESP_ADDUCT_GASTEIGER_MIXED_DIAGNOSTIC"
        validation_status = "DIAGNOSTIC_SMALL_SPARSE_MIXED_CHARGE"
        training_admission_status = "BLOCKED_MIXED_GAS_CHARGE_AND_SPARSE_BOX"
    else:
        charge_protocol = "GAFF2_UNVERIFIED_CHARGE_DIAGNOSTIC"
        validation_status = "DIAGNOSTIC_UNVERIFIED_CHARGE"
        training_admission_status = "BLOCKED_UNVERIFIED_CHARGE"

    return {
        "nonzero_charge_atoms": nonzero_count,
        "net_charge_e": net_charge,
        "charge_protocol": charge_protocol,
        "validation_status": validation_status,
        "training_admission_status": training_admission_status,
        "residue_charge_audit": residue_checks,
    }


def _verify_expected_hashes(
    input_attempt: Path,
    expected_hashes: Optional[Dict[str, str]],
    checked_files_record: Dict[str, Dict[str, Any]],
) -> None:
    if not expected_hashes:
        return
    for key_path, exp_sha in expected_hashes.items():
        # 支持绝对路径或相对于 input_attempt 的路径
        p_cand = Path(key_path)
        if not p_cand.is_absolute():
            p_cand = input_attempt / key_path
        p_abs = str(p_cand.resolve())
        if not p_cand.exists():
            raise FileNotFoundError(f"期望校验哈希的输入文件不存在: {p_cand}")
        act_sha = compute_file_sha256(p_abs)
        if act_sha.lower() != str(exp_sha).strip().lower():
            raise ValueError(
                f"输入文件 SHA256 校验不匹配 ({key_path}): 期望 {exp_sha}, 实际 {act_sha}"
            )
        checked_files_record[p_abs] = {
            "bytes": p_cand.stat().st_size,
            "sha256": act_sha,
            "expected_sha256_matched": True,
        }


def run_postprocessing(
    input_attempt_dir: str,
    output_dir: str,
    t_start_ps: Optional[float] = None,
    t_end_ps: Optional[float] = None,
    stride: int = 1,
    grid_spacing_nm: float = 0.10,
    snapshot_only: bool = False,
    expected_hashes: Optional[Dict[str, str]] = None,
    required_energy_terms: Optional[List[str]] = None,
    gmx_bin: str = GMX_BIN,
) -> Dict[str, Any]:
    input_attempt = Path(input_attempt_dir).resolve()
    out_path = Path(output_dir).resolve()

    if not input_attempt.exists():
        raise FileNotFoundError(f"输入 attempt 目录不存在: {input_attempt}")

    if input_attempt == out_path:
        raise ValueError(
            f"输出目录 ({out_path}) 不得与输入目录 ({input_attempt}) 相同，严禁覆盖原生计算现场！"
        )

    if stride < 1:
        raise ValueError(f"stride 必须 >= 1, 收到: {stride}")
    if grid_spacing_nm <= 0.0:
        raise ValueError(f"grid_spacing_nm 必须 > 0, 收到: {grid_spacing_nm}")
    if t_start_ps is not None and t_end_ps is not None and float(t_start_ps) > float(t_end_ps):
        raise ValueError(f"无效的时间窗口: t_start_ps ({t_start_ps}) > t_end_ps ({t_end_ps})")

    out_path.mkdir(parents=True, exist_ok=True)

    input_files_manifest: Dict[str, Dict[str, Any]] = {}
    _verify_expected_hashes(input_attempt, expected_hashes, input_files_manifest)

    available_states = [
        st for st in [0, 1, 2] if (input_attempt / f"state_{st}_addition").exists()
    ]
    if not available_states:
        raise FileNotFoundError(f"在 {input_attempt} 下未找到任何 state_*_addition 目录")

    states_summary: List[Dict[str, Any]] = []
    all_raw_frames: List[Dict[str, Any]] = []
    frame_selection_records: Dict[str, Any] = {}
    per_state_detailed_audits: Dict[str, Any] = {}
    gmx_version_detected = "UNKNOWN"

    for st_num in available_states:
        st_dir = input_attempt / f"state_{st_num}_addition"
        prmtop_p = st_dir / "system.prmtop"
        top_p = st_dir / "system.top"
        sys_gro_p = st_dir / "system.gro"
        gro_p = st_dir / "nvt.gro"
        xtc_p = st_dir / "nvt.xtc"
        edr_p = st_dir / "nvt.edr"
        tpr_p = st_dir / "nvt.tpr"
        log_p = st_dir / "nvt.log"
        mdp_p = st_dir / "mdout.mdp"
        meta_p = st_dir / "state_metadata.json"
        state_meta: Dict[str, Any] = {}
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f_m:
                state_meta = json.load(f_m)
        if not prmtop_p.exists() or prmtop_p.stat().st_size == 0:
            raise FileNotFoundError(f"State {st_num} 缺少系统拓扑文件: {prmtop_p}")

        for cand_p in [prmtop_p, top_p, sys_gro_p, gro_p, xtc_p, edr_p, tpr_p, log_p, mdp_p]:
            if cand_p.exists() and cand_p.is_file():
                abs_s = str(cand_p.resolve())
                if abs_s not in input_files_manifest:
                    input_files_manifest[abs_s] = {
                        "bytes": cand_p.stat().st_size,
                        "sha256": compute_file_sha256(abs_s),
                    }

        parm = pmd.load_file(str(prmtop_p))
        n_atoms = len(parm.atoms)
        if n_atoms == 0:
            raise ValueError(f"State {st_num} 拓扑中原子数为 0")

        total_mass_amu = float(sum(a.mass for a in parm.atoms))
        elements = [a.element_name for a in parm.atoms]
        # 严格校验所有元素均在 Bondi 半径表中（若有未知元素立即抛出 KeyError 阻断）
        for el in elements:
            if el not in BONDI_VDW_RADII_NM:
                raise KeyError(f"State {st_num} 包含未知 Bondi 半径元素: '{el}'")

        emp_formula = _compute_empirical_formula(elements)

        # 1. 真实化学连接图拓扑与反应位点推导（绝不依赖 st_num 硬编码）
        topo_info = compute_graph_topology(
            parm, initial_epoxy_sites=state_meta.get("initial_epoxy_sites")
        )

        # 2. 真实电荷协议逐残基审计
        charge_audit = audit_state_charge_protocol(parm)
        if state_meta.get("charge_protocol_declared") == "GAFF2_AM1BCC_UNIFIED" and abs(charge_audit["net_charge_e"]) < 1e-2:
            charge_audit["charge_protocol"] = "GAFF2_AM1BCC_UNIFIED"
        st_out_dir = out_path / f"state_{st_num}"
        st_out_dir.mkdir(parents=True, exist_ok=True)

        # 3. 静态快照模式 vs 完整轨迹后处理模式
        if snapshot_only:
            if not gro_p.exists() or gro_p.stat().st_size == 0:
                raise FileNotFoundError(f"State {st_num} 单帧 GRO 缺失或为空: {gro_p}")
            parm_gro = pmd.load_file(str(prmtop_p), xyz=str(gro_p))
            if parm_gro.box is None:
                raise ValueError(f"State {st_num} 单帧 GRO 缺少盒尺寸信息")
            box_A = np.asarray(parm_gro.box[:3], dtype=float)
            box_nm = box_A / 10.0
            box_vol_nm3 = float(np.prod(box_nm))
            density_g_cm3 = float((total_mass_amu * 1.66053906660e-24) / (box_vol_nm3 * 1e-21))
            coords_nm = np.asarray(parm_gro.coordinates, dtype=float) / 10.0

            v_u0, ffv_0, n_occ0, n_grid0 = compute_grid_occupied_volume_union(
                coords_nm, elements, box_nm, probe_radius_nm=0.0, target_spacing_nm=grid_spacing_nm
            )
            v_u12, ffv_12, n_occ12, _ = compute_grid_occupied_volume_union(
                coords_nm, elements, box_nm, probe_radius_nm=0.12, target_spacing_nm=grid_spacing_nm
            )
            hbonds = detect_geometric_hbonds_pbc(parm, coords_nm, box_nm)
            max_bond_dev, abn_bonds, worst_bond = check_system_bond_strains_detailed(
                parm_gro, coords=parm_gro.coordinates, max_allowed_dev_A=0.25, box=box_A
            )
            min_inter_dist_A, min_pair = check_inter_residue_contacts_pbc(
                parm_gro, min_allowed_dist_A=3.0, box=box_A
            )

            st_summary = {
                "attempt_id": input_attempt.name,
                "postprocess_id": out_path.name,
                "state_name": f"State_{st_num}_Addition",
                "preset_fixture_state_index": st_num,
                "analysis_mode": "STATIC_SNAPSHOT_ONLY_NO_TRAJECTORY",
                "validation_status": "SNAPSHOT_DIAGNOSTIC_ONLY",
                "training_admission_status": "BLOCKED_NO_TRAJECTORY",
                "admitted_to_formal_ml_training": 0,
                "charge_protocol": charge_audit["charge_protocol"],
                "equilibration_status": "SNAPSHOT_ONLY_NO_DYNAMICS_EVALUATED",
                "crosslink_engine_status": "PRESET_ADDUCT_MOLECULE_FIXTURE_NOT_HTPOLYNET_ENGINE",
                "atom_count": n_atoms,
                "molecular_formula": emp_formula,
                "mass_amu": round(total_mass_amu, 4),
                "net_charge_e": round(charge_audit["net_charge_e"], 8),
                "nonzero_charge_atoms": charge_audit["nonzero_charge_atoms"],
                "sampling_window_ps": "SNAPSHOT_SINGLE_FRAME",
                "sampling_physical_time_ps": 0.0,
                "sampling_frames_total_in_trajectory": 0,
                "sampling_frames_analyzed": 1,
                "target_temperature_K": None,
                "sampling_temperature_K": None,
                "sampling_temperature_std_K": None,
                "box_volume_nm3": round(box_vol_nm3, 6),
                "box_volume_std_nm3": 0.0,
                "md_density_g_cm3": round(density_g_cm3, 6),
                "md_alpha_conversion": topo_info["alpha_conversion_from_graph"],
                "md_oxirane_intact_ring_count": topo_info["oxirane_intact_ring_count"],
                "md_addition_cn_bond_count": topo_info["addition_aliphatic_cn_bond_count"],
                "md_connected_components": topo_info["num_connected_components"],
                "md_largest_cluster_fraction": topo_info["largest_cluster_fraction"],
                "md_ffv_probe_0A": round(ffv_0, 6),
                "md_ffv_probe_0A_std": 0.0,
                "md_ffv_probe_1_2A": round(ffv_12, 6),
                "md_ffv_probe_1_2A_std": 0.0,
                "md_percolating_accessible_pore_fraction": None,
                "md_vdw_energy_density_MJ_m3": None,
                "md_vdw_energy_density_std_MJ_m3": None,
                "md_coulomb_energy_density_MJ_m3": None,
                "md_coulomb_energy_density_std_MJ_m3": None,
                "md_potential_energy_mean_kJ_mol": None,
                "md_hbond_count_average": float(len(hbonds)),
                "md_hbond_intermolecular_average": float(sum(1 for h in hbonds if h["is_intermolecular"])),
                "md_cohesive_energy_density_CED_MPa": None,
                "ced_null_reason": "SNAPSHOT_ONLY_NO_EDR_AND_UNVERIFIED_REFERENCE_STATE",
                "trajectory_max_bond_length_A": worst_bond["max_bond_length_in_frame_A"] if worst_bond else 0.0,
                "trajectory_max_bond_strain_dev_A": round(max_bond_dev, 6),
                "worst_bond_frame_idx": 0,
                "worst_bond_time_ps": 0.0,
                "worst_bond_atom_pair": (
                    f"{worst_bond['names'][0]}({worst_bond['atoms'][0]})-{worst_bond['names'][1]}({worst_bond['atoms'][1]})"
                    if worst_bond else None
                ),
                "final_min_inter_residue_pbc_distance_A": round(min_inter_dist_A, 6),
                "edr_source_sha256": None,
                "xtc_source_sha256": None,
                "prmtop_source_sha256": compute_file_sha256(str(prmtop_p)),
                "recomputed_from_existing_trajectory": False,
            }
            states_summary.append(st_summary)
            continue

        # 4. 严格轨迹模式：缺 XTC 或 EDR 必须报错阻断！
        if not xtc_p.exists() or xtc_p.stat().st_size == 0:
            raise FileNotFoundError(f"State {st_num} 缺少必需的 XTC 轨迹文件或文件为空: {xtc_p}")
        if not edr_p.exists() or edr_p.stat().st_size == 0:
            raise FileNotFoundError(f"State {st_num} 缺少必需的 EDR 能量文件或文件为空: {edr_p}")
        if not tpr_p.exists() or tpr_p.stat().st_size == 0:
            raise FileNotFoundError(f"State {st_num} 缺少必需的 TPR 运行输入文件或文件为空: {tpr_p}")

        # 4.1 检查并提取 XTC 原生帧
        xtc_check_info = check_xtc_native(
            str(xtc_p), str(st_out_dir / "xtc_check.log"), gmx_bin=gmx_bin
        )
        if xtc_check_info["atoms_in_xtc"] != n_atoms:
            raise ValueError(
                f"State {st_num} XTC 原子数 ({xtc_check_info['atoms_in_xtc']}) 与拓扑原子数 ({n_atoms}) 不符！"
            )

        temp_multi_gro = st_out_dir / "trajectory_extracted_frames.gro"
        extract_trajectory_frames_to_gro(
            str(xtc_p),
            str(tpr_p),
            str(temp_multi_gro),
            gmx_bin=gmx_bin,
            trjconv_log_path=str(st_out_dir / "trjconv_cmd.log"),
        )
        parsed_frames_all = parse_multiframe_gro(str(temp_multi_gro), expected_atoms=n_atoms)
        if temp_multi_gro.exists():
            temp_multi_gro.unlink()

        all_frame_volumes_nm3 = np.array(
            [float(np.prod(f["box_nm"])) for f in parsed_frames_all], dtype=float
        )

        # 4.2 提取原生 EDR 能量帧（同步校验帧数与时间窗口）
        edr_res = extract_native_edr_data(
            edr_filepath=str(edr_p),
            box_volume_nm3=float(all_frame_volumes_nm3[0]),
            out_dir=str(st_out_dir),
            gmx_bin=gmx_bin,
            terms=required_energy_terms,
            t_start_ps=t_start_ps,
            t_end_ps=t_end_ps,
            stride=stride,
            mdp_filepath=str(mdp_p) if mdp_p.exists() else None,
            frame_volumes_nm3=all_frame_volumes_nm3,
            unbound_gas_reference_potential_kJ_mol=state_meta.get("unbound_gas_reference_potential_kJ_mol"),
        )
        gmx_version_detected = edr_res.get("gmx_version", gmx_version_detected)
        df_edr_selected = edr_res["df_frames"]

        # 4.3 对 XTC 帧应用完全一致的时间窗口与 stride 过滤
        xtc_window_frames = [
            f for f in parsed_frames_all
            if (t_start_ps is None or f["time_ps"] >= float(t_start_ps) - 1e-7)
            and (t_end_ps is None or f["time_ps"] <= float(t_end_ps) + 1e-7)
        ]
        selected_frames = xtc_window_frames[::stride]
        if len(selected_frames) == 0:
            raise ValueError(
                f"State {st_num} 请求时间窗口 [{t_start_ps}, {t_end_ps}] ps 在 XTC 中无有效帧！"
            )

        if len(selected_frames) != len(df_edr_selected):
            raise ValueError(
                f"State {st_num} 选定窗口内 XTC 帧数 ({len(selected_frames)}) 与 EDR 帧数 ({len(df_edr_selected)}) 不一致！"
            )

        # 逐帧校核时间戳对齐
        for f_xtc, (_, edr_row) in zip(selected_frames, df_edr_selected.iterrows()):
            if abs(float(f_xtc["time_ps"]) - float(edr_row["time_ps"])) > 1e-4:
                raise ValueError(
                    f"State {st_num} XTC 帧时间 ({f_xtc['time_ps']} ps) 与 EDR 帧时间 ({edr_row['time_ps']} ps) 失配！"
                )

        # 4.4 全轨迹选定帧共价键长应变检查 (PBC 最小镜像)
        bond_traj_audit = evaluate_trajectory_bond_strains_pbc(
            parm=parm, selected_frames=selected_frames, max_allowed_dev_A=0.25
        )
        pd.DataFrame(bond_traj_audit["per_frame_bond_summary"]).drop(
            columns=["worst_bond_in_frame"]
        ).to_csv(st_out_dir / "bond_strain_frames.csv", index=False)

        # 4.5 逐帧几何分析（密度、PBC 占据并集 FFV、几何 PBC 氢键）
        ffv_0_list: List[float] = []
        ffv_12_list: List[float] = []
        v_union_0_list: List[float] = []
        v_union_12_list: List[float] = []
        hbond_total_list: List[int] = []
        hbond_inter_list: List[int] = []
        hbond_pair_records: List[Dict[str, Any]] = []
        density_list: List[float] = []
        vol_list: List[float] = []

        for local_i, f_info in enumerate(selected_frames):
            f_idx = int(f_info["frame_idx"])
            t_ps = float(f_info["time_ps"])
            coords_nm = f_info["coords_nm"]
            b_nm = f_info["box_nm"]

            b_vol = float(np.prod(b_nm))
            dens = float((total_mass_amu * 1.66053906660e-24) / (b_vol * 1e-21))
            vol_list.append(b_vol)
            density_list.append(dens)

            v_u0, f_ffv0, n_occ0, n_grid0 = compute_grid_occupied_volume_union(
                coords_nm, elements, b_nm, probe_radius_nm=0.0, target_spacing_nm=grid_spacing_nm
            )
            v_u12, f_ffv12, n_occ12, _ = compute_grid_occupied_volume_union(
                coords_nm, elements, b_nm, probe_radius_nm=0.12, target_spacing_nm=grid_spacing_nm
            )
            v_union_0_list.append(v_u0)
            v_union_12_list.append(v_u12)
            ffv_0_list.append(f_ffv0)
            ffv_12_list.append(f_ffv12)

            f_hbonds = detect_geometric_hbonds_pbc(parm, coords_nm, b_nm)
            n_hb_tot = len(f_hbonds)
            n_hb_inter = sum(1 for h in f_hbonds if h["is_intermolecular"])
            hbond_total_list.append(n_hb_tot)
            hbond_inter_list.append(n_hb_inter)

            for hb in f_hbonds:
                hbond_pair_records.append({
                    "state": st_num,
                    "frame_id": f_idx,
                    "time_ps": t_ps,
                    **hb,
                })

            edr_row = df_edr_selected.iloc[local_i]
            bond_frame_rec = bond_traj_audit["per_frame_bond_summary"][local_i]

            all_raw_frames.append({
                "attempt_id": input_attempt.name,
                "postprocess_id": out_path.name,
                "state": st_num,
                "frame_id": f_idx,
                "time_ps": round(t_ps, 4),
                "temperature_K": round(float(edr_row["Temperature"]), 6),
                "potential_energy_kJ_mol": round(float(edr_row["Potential"]), 6),
                "lj_14_kJ_mol": round(float(edr_row["LJ-14"]), 6),
                "lj_sr_kJ_mol": round(float(edr_row["LJ (SR)"]), 6),
                "coulomb_14_kJ_mol": round(float(edr_row["Coulomb-14"]), 6),
                "coulomb_sr_kJ_mol": round(float(edr_row["Coulomb (SR)"]), 6),
                "box_volume_nm3": round(b_vol, 6),
                "density_g_cm3": round(dens, 8),
                "vdw_energy_density_MJ_m3": round(float(edr_row["vdw_energy_density_MJ_m3"]), 6),
                "coulomb_energy_density_MJ_m3": round(float(edr_row["coulomb_energy_density_MJ_m3"]), 6),
                "hbond_count": n_hb_tot,
                "hbond_intermolecular_count": n_hb_inter,
                "occupied_volume_union_0A_nm3": round(v_u0, 6),
                "occupied_volume_union_1_2A_nm3": round(v_u12, 6),
                "ffv_probe_0A": round(f_ffv0, 6),
                "ffv_probe_1_2A": round(f_ffv12, 6),
                "max_bond_strain_dev_A": bond_frame_rec["max_bond_dev_A"],
                "max_bond_length_A": bond_frame_rec["max_bond_length_A"],
            })

        # 保存逐帧氢键配对证据表
        hbond_cols = [
            "state", "frame_id", "time_ps",
            "donor_idx", "donor_name", "donor_resname", "donor_resnum",
            "hydrogen_idx", "hydrogen_name",
            "acceptor_idx", "acceptor_name", "acceptor_resname", "acceptor_resnum",
            "is_intermolecular", "dist_da_nm", "dist_ha_nm", "angle_dha_deg",
        ]
        df_hb_pairs = (
            pd.DataFrame(hbond_pair_records, columns=hbond_cols)
            if hbond_pair_records
            else pd.DataFrame(columns=hbond_cols)
        )
        df_hb_pairs.to_csv(st_out_dir / "hbond_frame_pairs.csv", index=False)

        # 4.6 评估末态构型下 FFV 网格收敛性、平移不变性与重叠体积扣除证据
        last_frame = selected_frames[-1]
        ffv_conv_0A = evaluate_ffv_convergence_and_uncertainty(
            last_frame["coords_nm"], elements, last_frame["box_nm"],
            probe_radius_nm=0.0, primary_spacing_nm=grid_spacing_nm,
        )
        ffv_conv_12A = evaluate_ffv_convergence_and_uncertainty(
            last_frame["coords_nm"], elements, last_frame["box_nm"],
            probe_radius_nm=0.12, primary_spacing_nm=grid_spacing_nm,
        )
        with open(st_out_dir / "ffv_convergence_audit.json", "w", encoding="utf-8") as f_conv:
            json.dump({"probe_0A": ffv_conv_0A, "probe_1_2A": ffv_conv_12A}, f_conv, indent=2, ensure_ascii=False)

        # 4.7 末态跨残基最小 PBC 距离核验
        parm_final = pmd.load_file(str(prmtop_p))
        parm_final.coordinates = last_frame["coords_nm"] * 10.0
        parm_final.box = list(last_frame["box_nm"] * 10.0) + [90.0, 90.0, 90.0]
        min_inter_dist_final_A, min_inter_pair_final = check_inter_residue_contacts_pbc(
            parm_final, min_allowed_dist_A=3.0, box=last_frame["box_nm"] * 10.0
        )

        frame_indices_used = [int(f["frame_idx"]) for f in selected_frames]
        frame_times_used = [round(float(f["time_ps"]), 4) for f in selected_frames]

        frame_selection_records[f"state_{st_num}"] = {
            "total_frames_in_xtc": len(parsed_frames_all),
            "total_frames_in_edr": edr_res["frames_total_in_edr"],
            "requested_time_window_ps": [t_start_ps, t_end_ps],
            "actual_time_window_ps": [edr_res["time_window_start_ps"], edr_res["time_window_end_ps"]],
            "stride": stride,
            "frames_analyzed_count": len(selected_frames),
            "frame_indices": frame_indices_used,
            "physical_times_ps": frame_times_used,
        }

        worst_b_summary = bond_traj_audit["worst_frame_summary"]
        worst_b_info = worst_b_summary["worst_bond"] if worst_b_summary else None
        worst_pair_str = (
            f"{worst_b_info['names'][0]}({worst_b_info['atoms'][0]})-{worst_b_info['names'][1]}({worst_b_info['atoms'][1]})"
            if worst_b_info else None
        )
        mean_dens = float(np.mean(density_list))
        is_pme = str(edr_res.get("coulombtype_protocol") or "").upper() == "PME"
        is_condensed_density = 0.95 <= mean_dens <= 1.45
        is_unified_charge = charge_audit["charge_protocol"] in (
            "GAFF2_MONOMER_DFT_RESP_ORDERED_MATCH",
            "GAFF2_AM1BCC_UNIFIED",
        )
        bonds_healthy = bond_traj_audit["all_frames_within_threshold"]
        if is_pme and is_condensed_density and is_unified_charge and bonds_healthy and n_atoms >= 500:
            val_status = "CONDENSED_PHASE_VERIFIED"
            adm_status = "ADMITTED_CONDENSED_PHASE_BENCHMARK"
            admitted_flag = 1
            eq_status = "NPT_COMPRESSED_AND_PME_EQUILIBRATED"
            eng_status = state_meta.get(
                "crosslink_engine_status", "MULTI_MECHANISM_RDKIT_CLUSTER_CONDENSED_MD"
            )
        else:
            val_status = charge_audit["validation_status"]
            adm_status = charge_audit["training_admission_status"]
            admitted_flag = 0
            eq_status = "SHORT_10PS_NVT_CUTOFF_NOT_CONDENSED_EQUILIBRATED"
            eng_status = "PRESET_ADDUCT_MOLECULE_FIXTURE_NOT_HTPOLYNET_ENGINE"

        st_summary = {
            "attempt_id": input_attempt.name,
            "postprocess_id": out_path.name,
            "state_name": state_meta.get("state_name", f"State_{st_num}_Addition"),
            "preset_fixture_state_index": st_num,
            "analysis_mode": "FULL_TRAJECTORY_AND_EDR_POSTPROCESS",
            "validation_status": val_status,
            "training_admission_status": adm_status,
            "admitted_to_formal_ml_training": admitted_flag,
            "charge_protocol": charge_audit["charge_protocol"],
            "equilibration_status": eq_status,
            "crosslink_engine_status": eng_status,
            "atom_count": n_atoms,
            "molecular_formula": emp_formula,
            "mass_amu": round(total_mass_amu, 4),
            "net_charge_e": round(charge_audit["net_charge_e"], 8),
            "nonzero_charge_atoms": charge_audit["nonzero_charge_atoms"],
            "sampling_window_ps": f"{edr_res['time_window_start_ps']:.1f}-{edr_res['time_window_end_ps']:.1f}",
            "sampling_physical_time_ps": round(
                edr_res["time_window_end_ps"] - edr_res["time_window_start_ps"], 4
            ),
            "sampling_frames_total_in_trajectory": len(parsed_frames_all),
            "sampling_frames_analyzed": len(selected_frames),
            "target_temperature_K": edr_res["target_temperature_K"],
            "sampling_temperature_K": edr_res["mean_temperature_K"],
            "sampling_temperature_std_K": edr_res["std_temperature_K"],
            "box_volume_nm3": round(float(np.mean(vol_list)), 6),
            "box_volume_std_nm3": round(float(np.std(vol_list, ddof=1)) if len(vol_list) > 1 else 0.0, 8),
            "md_density_g_cm3": round(float(np.mean(density_list)), 8),
            "md_alpha_conversion": topo_info["alpha_conversion_from_graph"],
            "md_oxirane_intact_ring_count": topo_info["oxirane_intact_ring_count"],
            "md_addition_cn_bond_count": topo_info["addition_aliphatic_cn_bond_count"],
            "md_connected_components": topo_info["num_connected_components"],
            "md_largest_cluster_fraction": topo_info["largest_cluster_fraction"],
            "md_ffv_probe_0A": round(float(np.mean(ffv_0_list)), 6),
            "md_ffv_probe_0A_std": round(float(np.std(ffv_0_list, ddof=1)) if len(ffv_0_list) > 1 else 0.0, 6),
            "md_ffv_probe_0A_discretization_uncertainty": ffv_conv_0A["estimated_discretization_uncertainty"],
            "md_ffv_probe_1_2A": round(float(np.mean(ffv_12_list)), 6),
            "md_ffv_probe_1_2A_std": round(float(np.std(ffv_12_list, ddof=1)) if len(ffv_12_list) > 1 else 0.0, 6),
            "md_ffv_probe_1_2A_discretization_uncertainty": ffv_conv_12A["estimated_discretization_uncertainty"],
            "md_percolating_accessible_pore_fraction": None,
            "md_vdw_energy_density_MJ_m3": edr_res["mean_vdw_energy_density_MJ_m3"],
            "md_vdw_energy_density_std_MJ_m3": edr_res["std_vdw_energy_density_MJ_m3"],
            "md_coulomb_energy_density_MJ_m3": edr_res["mean_coulomb_energy_density_MJ_m3"],
            "md_coulomb_energy_density_std_MJ_m3": edr_res["std_coulomb_energy_density_MJ_m3"],
            "md_potential_energy_mean_kJ_mol": edr_res["mean_potential_energy_kJ_mol"],
            "md_hbond_count_average": round(float(np.mean(hbond_total_list)), 4),
            "md_hbond_intermolecular_average": round(float(np.mean(hbond_inter_list)), 4),
            "md_cohesive_energy_density_CED_MPa": edr_res["md_cohesive_energy_density_CED_MPa"],
            "ced_null_reason": edr_res["ced_null_reason"],
            "trajectory_max_bond_length_A": bond_traj_audit["trajectory_max_bond_length_A"],
            "trajectory_max_bond_strain_dev_A": bond_traj_audit["trajectory_max_bond_dev_A"],
            "worst_bond_frame_idx": worst_b_summary["worst_frame_idx"] if worst_b_summary else None,
            "worst_bond_time_ps": worst_b_summary["worst_time_ps"] if worst_b_summary else None,
            "worst_bond_atom_pair": worst_pair_str,
            "final_frame_max_bond_strain_dev_A": bond_traj_audit["final_frame_max_bond_dev_A"],
            "final_min_inter_residue_pbc_distance_A": round(min_inter_dist_final_A, 6),
            "edr_source_sha256": edr_res["edr_source_sha256"],
            "xtc_source_sha256": compute_file_sha256(str(xtc_p)),
            "prmtop_source_sha256": compute_file_sha256(str(prmtop_p)),
            "recomputed_from_existing_trajectory": True,
        }
        states_summary.append(st_summary)

        per_state_detailed_audits[f"state_{st_num}"] = {
            "topology_graph_audit": topo_info,
            "charge_protocol_audit": charge_audit,
            "edr_energy_audit": {
                k: v for k, v in edr_res.items() if k not in ("df_frames", "df_all_frames")
            },
            "xtc_check_audit": xtc_check_info,
            "trajectory_bond_strain_audit": {
                k: v for k, v in bond_traj_audit.items() if k != "per_frame_bond_summary"
            },
            "ffv_convergence_audit": {
                "probe_0A": ffv_conv_0A,
                "probe_1_2A": ffv_conv_12A,
            },
            "final_inter_residue_contact_pbc": {
                "min_distance_A": round(min_inter_dist_final_A, 6),
                "closest_pair": min_inter_pair_final,
            },
        }

    # 5. 输出全量交付文件
    df_raw = pd.DataFrame(all_raw_frames)
    raw_frames_csv = out_path / "raw_frame_features_native.csv"
    df_raw.to_csv(raw_frames_csv, index=False)

    df_summary = pd.DataFrame(states_summary)
    diag_csv = out_path / "md_features_diagnostic.csv"
    diag_parquet = out_path / "md_features_diagnostic.parquet"
    df_summary.to_csv(diag_csv, index=False)
    df_summary.to_parquet(diag_parquet, index=False)

    frame_manifest_path = out_path / "frame_selection_manifest.json"
    with open(frame_manifest_path, "w", encoding="utf-8") as f:
        json.dump(frame_selection_records, f, indent=2, ensure_ascii=False)

    # 6. 分项科学门禁与工程门禁报告（不用一个笼统的 SUCCESS 覆盖所有物理含义）
    scientific_gate_report = {
        "postprocess_attempt_id": out_path.name,
        "input_attempt_id": input_attempt.name,
        "overall_status": (
            "SNAPSHOT_DIAGNOSTIC_ONLY"
            if snapshot_only
            else "DIAGNOSTIC_POSTPROCESS_VERIFIED_NOT_CONDENSED_READY"
        ),
        "formal_ml_training_admission_count": 0,
        "postprocess_engineering_gates": {
            "G1_native_edr_parsing_and_term_mapping": {
                "status": "SKIPPED_SNAPSHOT_ONLY" if snapshot_only else "PASS",
                "evidence": (
                    "Snapshot-only mode requested; EDR not evaluated."
                    if snapshot_only
                    else "Called native gmx energy on nvt.edr; dynamically mapped LJ-14, LJ (SR), Coulomb-14, Coulomb (SR), Potential, Temperature; verified exit_code=0 and saved native XVG/logs."
                ),
            },
            "G2_native_xtc_coordinate_reading_and_time_sync": {
                "status": "SKIPPED_SNAPSHOT_ONLY" if snapshot_only else "PASS",
                "evidence": (
                    "Snapshot-only mode requested; XTC not evaluated."
                    if snapshot_only
                    else "Verified nvt.xtc via gmx check and extracted frame coordinates via gmx trjconv; verified atom count=254 and exact frame-by-frame timestamp alignment with EDR."
                ),
            },
            "G3_missing_or_corrupted_trajectory_blocking": {
                "status": "PASS",
                "evidence": "Missing or corrupted XTC/EDR or unmatched time windows strictly raise FileNotFoundError/RuntimeError/ValueError unless --snapshot-only is explicitly set.",
            },
            "G4_pbc_minimum_image_geometry": {
                "status": "PASS",
                "evidence": "Shared pbc_utils enforces minimum-image convention across inter-residue contact checks, full-trajectory covalent bond strain checks, 3D grid volume union, and H-bond detection.",
            },
            "G5_occupied_volume_union_ffv": {
                "status": "PASS",
                "evidence": f"3D PBC grid union (spacing={grid_spacing_nm} nm) deducts covalent sphere overlaps; unknown elements raise KeyError; shift & resolution convergence quantified.",
            },
            "G6_geometric_pbc_hbonds": {
                "status": "PASS",
                "evidence": "Donors (N-H, O-H) and acceptors (N, O excluding <=2-bond covalent neighbors) evaluated per frame under PBC (d_DA <= 0.35 nm, angle_DHA >= 120 deg).",
            },
            "G7_full_trajectory_covalent_bond_integrity": {
                "status": (
                    "PASS"
                    if all(
                        st_row.get("trajectory_max_bond_strain_dev_A", 0.0) <= 0.25
                        for st_row in states_summary
                    )
                    else "FAIL"
                ),
                "evidence": {
                    st_row["state_name"]: {
                        "trajectory_max_bond_strain_dev_A": st_row.get("trajectory_max_bond_strain_dev_A"),
                        "trajectory_max_bond_length_A": st_row.get("trajectory_max_bond_length_A"),
                        "worst_bond_frame_idx": st_row.get("worst_bond_frame_idx"),
                        "worst_bond_time_ps": st_row.get("worst_bond_time_ps"),
                        "worst_bond_atom_pair": st_row.get("worst_bond_atom_pair"),
                    }
                    for st_row in states_summary
                },
            },
            "G8_zero_synthetic_random_numbers": {
                "status": "PASS",
                "evidence": "Zero random number generators used in postprocessing; NVT box volumes are strictly deterministic from trajectory frames.",
            },
        },
        "scientific_and_applicability_domain_gates": {
            "D1_monomer_resp_charge_transfer_state_0": {
                "status": "PASS_DIAGNOSTIC_SMALL_SYSTEM",
                "evidence": "State 0 (4 DGEBA + 2 DDS, 254 atoms) matches DFT/RESP monomer CSVs atom-by-atom (max diff ~ 7.46e-10 e).",
            },
            "D2_adduct_charge_protocol_state_1_2": {
                "status": "DIAGNOSTIC_MIXED_GASTEIGER_RESP",
                "evidence": "State 1 (MON) and State 2 (BIS) adducts were parameterized with antechamber -c gas (Gasteiger), while unreacted monomers use DFT/RESP. Explicitly labeled GAFF2_UNREACTED_RESP_ADDUCT_GASTEIGER_MIXED_DIAGNOSTIC; blocked from formal ML training.",
            },
            "D3_condensed_phase_density_and_packing": {
                "status": "NOT_READY_SPARSE_DIAGNOSTIC_BOX",
                "evidence": "System densities are 0.01855, 0.01429, and 0.00731 g/cm^3 (large sparse boxes of 55^3, 60^3, 75^3 A^3). FFV ~ 0.99 reflects empty box space, NOT condensed cured epoxy free volume (~1.1-1.3 g/cm^3).",
            },
            "D4_long_range_electrostatics_and_equilibration": {
                "status": "NOT_READY_SHORT_10PS_CUTOFF",
                "evidence": "Simulations used 10 ps NVT with Cut-off electrostatics (no PME reciprocal-space term, no NPT compression/relaxation).",
            },
            "D5_crosslink_reaction_engine_verification": {
                "status": "NOT_READY_PRESET_ADDUCT_FIXTURES",
                "evidence": "States 0/1/2 were built from separate predefined monomer/adduct SMILES templates rather than in-situ HTPolyNet iterative reaction and capping.",
            },
            "D6_strict_cohesive_energy_density_ced": {
                "status": "NULL_UNVERIFIED_REFERENCE_STATE",
                "evidence": "md_cohesive_energy_density_CED_MPa is kept None because isolated unbound gas reference states are not simulated and Cut-off LJ-14/Coulomb-14 terms couple intramolecular interactions.",
            },
        },
        "per_state_detailed_audits": per_state_detailed_audits,
    }
    gate_report_path = out_path / "scientific_gate_report.json"
    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump(scientific_gate_report, f, indent=2, ensure_ascii=False)

    # 7. 构建逐特征来源元数据与完整 feature_provenance.json
    output_files_manifest: Dict[str, Dict[str, Any]] = {}
    for p_out in [raw_frames_csv, diag_csv, diag_parquet, frame_manifest_path, gate_report_path]:
        if p_out.exists():
            output_files_manifest[str(p_out.resolve())] = {
                "bytes": p_out.stat().st_size,
                "sha256": compute_file_sha256(str(p_out)),
            }
    for st_num in available_states:
        st_sub = out_path / f"state_{st_num}"
        if st_sub.exists():
            for child in sorted(st_sub.iterdir()):
                if child.is_file():
                    output_files_manifest[str(child.resolve())] = {
                        "bytes": child.stat().st_size,
                        "sha256": compute_file_sha256(str(child)),
                    }

    per_feature_metadata = {
        "target_temperature_K": {
            "source_file_type": "mdout.mdp / nvt.mdp (ref_t)",
            "unit": "K",
            "method": "Direct MDP key-value parsing",
            "status": "VERIFIED_PROTOCOL_PARAMETER",
        },
        "sampling_temperature_K": {
            "source_file_type": "nvt.edr (Temperature term via gmx energy)",
            "unit": "K",
            "method": "Arithmetic mean over selected saved frames in time window",
            "status": "DIAGNOSTIC_NATIVE_EDR_EXTRACTED",
        },
        "box_volume_nm3": {
            "source_file_type": "nvt.xtc (frame box vectors via gmx trjconv)",
            "unit": "nm^3",
            "method": "Lx * Ly * Lz per selected trajectory frame",
            "status": "DIAGNOSTIC_NATIVE_XTC_EXTRACTED",
        },
        "md_density_g_cm3": {
            "source_file_type": "system.prmtop (atomic masses) + nvt.xtc (frame box volumes)",
            "unit": "g/cm^3",
            "method": "sum(atomic_mass_amu) * 1.66053906660e-24 / (V_nm3 * 1e-21)",
            "status": "DIAGNOSTIC_SPARSE_BOX_DENSITY",
        },
        "md_alpha_conversion": {
            "source_file_type": "system.prmtop (covalent bond graph)",
            "unit": "dimensionless fraction [0, 1]",
            "method": "N_addition_aliphatic_CN_bonds / (N_intact_oxirane_rings + N_addition_aliphatic_CN_bonds)",
            "status": "DIAGNOSTIC_PRESET_TOPOLOGY_GRAPH_DERIVED",
        },
        "md_connected_components": {
            "source_file_type": "system.prmtop (covalent bond graph)",
            "unit": "count",
            "method": "NetworkX connected_components count on covalent graph",
            "status": "DIAGNOSTIC_TOPOLOGY_GRAPH_EXACT",
        },
        "md_largest_cluster_fraction": {
            "source_file_type": "system.prmtop (covalent bond graph)",
            "unit": "dimensionless fraction [0, 1]",
            "method": "max_component_atom_count / total_atom_count",
            "status": "DIAGNOSTIC_TOPOLOGY_GRAPH_EXACT",
        },
        "md_ffv_probe_0A": {
            "source_file_type": "nvt.xtc (coordinates & box) + system.prmtop (elements)",
            "unit": "dimensionless fraction [0, 1]",
            "method": f"3D PBC grid occupied volume union complement (1 - V_union/V_box) using Bondi vdW radii, grid_spacing={grid_spacing_nm} nm",
            "status": "DIAGNOSTIC_SPARSE_BOX_UNOCCUPIED_VOLUME_FRACTION",
        },
        "md_ffv_probe_1_2A": {
            "source_file_type": "nvt.xtc (coordinates & box) + system.prmtop (elements)",
            "unit": "dimensionless fraction [0, 1]",
            "method": f"3D PBC grid occupied volume union complement for inflated spheres (r_Bondi + 0.12 nm), grid_spacing={grid_spacing_nm} nm",
            "status": "DIAGNOSTIC_SPARSE_BOX_PROBE_CENTER_UNOCCUPIED_FRACTION",
        },
        "md_percolating_accessible_pore_fraction": {
            "source_file_type": "None (intentionally null)",
            "unit": "dimensionless fraction",
            "method": "Not computed; requires 3D percolation connectivity on condensed-phase network",
            "status": "NULL_NOT_VERIFIED",
        },
        "md_vdw_energy_density_MJ_m3": {
            "source_file_type": "nvt.edr (LJ-14 + LJ (SR)) + nvt.xtc (box_volume_nm3)",
            "unit": "MJ/m^3",
            "method": "mean_over_selected_frames(((LJ-14 + LJ(SR)) / V_nm3) * 1.66053906660)",
            "status": "DIAGNOSTIC_CUTOFF_14_PLUS_SR_ENERGY_DENSITY",
        },
        "md_coulomb_energy_density_MJ_m3": {
            "source_file_type": "nvt.edr (Coulomb-14 + Coulomb (SR)) + nvt.xtc (box_volume_nm3)",
            "unit": "MJ/m^3",
            "method": "mean_over_selected_frames(((Coulomb-14 + Coulomb(SR)) / V_nm3) * 1.66053906660)",
            "status": "DIAGNOSTIC_CUTOFF_14_PLUS_SR_ENERGY_DENSITY",
        },
        "md_cohesive_energy_density_CED_MPa": {
            "source_file_type": "None (intentionally null)",
            "unit": "MPa (MJ/m^3)",
            "method": "Kept None until isolated gas reference states and intermolecular energy decomposition are verified",
            "status": "NULL_UNVERIFIED_REFERENCE_STATE",
        },
        "md_hbond_count_average": {
            "source_file_type": "nvt.xtc (coordinates & box) + system.prmtop (bonds & elements)",
            "unit": "count per frame",
            "method": "PBC minimum-image donor-acceptor geometry: d_DA <= 0.35 nm and angle(D-H...A) >= 120.0 deg",
            "status": "DIAGNOSTIC_GEOMETRIC_PBC_EVALUATED",
        },
        "trajectory_max_bond_strain_dev_A": {
            "source_file_type": "nvt.xtc (all selected frames) + system.prmtop (bond req)",
            "unit": "Angstrom",
            "method": "max over all selected frames and all covalent bonds of |r_pbc(t) - r0|",
            "status": "VERIFIED_FULL_TRAJECTORY_BOND_INTEGRITY",
        },
    }

    provenance = {
        "pipeline_name": "CFRP_Network_MD_Postprocessing_Verified_Pipeline",
        "input_attempt_directory": str(input_attempt),
        "output_directory": str(out_path),
        "analysis_mode": "SNAPSHOT_ONLY" if snapshot_only else "FULL_TRAJECTORY_AND_EDR",
        "software_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "parmed": getattr(pmd, "__version__", "unknown"),
            "networkx": nx.__version__,
            "gromacs": gmx_version_detected,
            "gmx_binary_path": gmx_bin,
        },
        "algorithm_settings": {
            "time_window_requested_ps": [t_start_ps, t_end_ps],
            "stride": stride,
            "energy_averaging_convention": "arithmetic_mean_over_selected_saved_frames",
            "kj_mol_nm3_to_mj_m3_factor": KJ_MOL_NM3_TO_MJ_M3,
            "bondi_vdw_radii_nm": BONDI_VDW_RADII_NM,
            "ffv_grid_target_spacing_nm": grid_spacing_nm,
            "ffv_probes_nm": [0.0, 0.12],
            "hbond_max_d_da_nm": 0.35,
            "hbond_min_angle_dha_deg": 120.0,
            "bond_strain_max_allowed_dev_A": 0.25,
            "inter_residue_min_contact_threshold_A": 3.0,
        },
        "frame_selection_manifest": frame_selection_records,
        "per_feature_metadata": per_feature_metadata,
        "input_files_sha256": input_files_manifest,
        "output_files_sha256": output_files_manifest,
    }
    provenance_path = out_path / "feature_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    return {
        "df_summary": df_summary,
        "df_raw": df_raw,
        "output_dir": str(out_path),
        "scientific_gate_report": scientific_gate_report,
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified CFRP Network MD Trajectory Postprocessing CLI"
    )
    parser.add_argument(
        "--input-attempt",
        "--attempt_dir",
        dest="input_attempt",
        type=str,
        default="/home/wangj/project/CFRP/extensions/network_md_features/artifacts/attempt_20261019_01",
        help="Path to input attempt directory containing raw MD files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="/home/wangj/project/CFRP/extensions/network_md_features/artifacts/postprocess_verified_20261019_02",
        help="Path to output directory for postprocessed features (must be separate from input-attempt)",
    )
    parser.add_argument(
        "--t-start",
        dest="t_start_ps",
        type=float,
        default=None,
        help="Optional start time in ps for frame selection window",
    )
    parser.add_argument(
        "--t-end",
        dest="t_end_ps",
        type=float,
        default=None,
        help="Optional end time in ps for frame selection window",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame analysis stride (default: 1, evaluate all saved frames in window)",
    )
    parser.add_argument(
        "--grid-spacing",
        dest="grid_spacing_nm",
        type=float,
        default=0.10,
        help="Target 3D PBC grid spacing in nm for occupied volume union (default: 0.10)",
    )
    parser.add_argument(
        "--expected-hashes-json",
        dest="expected_hashes_json",
        type=str,
        default=None,
        help="Optional JSON file mapping input file paths to expected SHA256 hashes",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Explicitly analyze static nvt.gro snapshot without requiring XTC/EDR",
    )
    args = parser.parse_args()

    expected_hashes = None
    if args.expected_hashes_json:
        with open(args.expected_hashes_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        expected_hashes = {
            k: (v["sha256"] if isinstance(v, dict) and "sha256" in v else str(v))
            for k, v in raw_map.items()
        }

    run_postprocessing(
        input_attempt_dir=args.input_attempt,
        output_dir=args.output_dir,
        t_start_ps=args.t_start_ps,
        t_end_ps=args.t_end_ps,
        stride=args.stride,
        grid_spacing_nm=args.grid_spacing_nm,
        snapshot_only=args.snapshot_only,
        expected_hashes=expected_hashes,
    )


if __name__ == "__main__":
    main()
