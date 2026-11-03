"""
structure_features_extractor.py
真正读取 XTC 轨迹与拓扑的结构特征提取器
核心算法与科学准则：
1. 真实轨迹帧迭代：通过 gmx check 与 gmx trjconv 从实际保存的 XTC 轨迹中逐帧提取坐标与同步盒矢量，记录命令与日志，校验帧原子数与拓扑一致；
2. 真实图拓扑与反应位点推导：
   - 从体系实际共价键集合 (parm.bonds) 构建无向图，计算连通分量数、各分量大小与最大团簇占比；
   - 从真实分子连接图自动识别未反应三元环氧环 (oxirane C-O-C 3-membered rings) 与开环生成的脂肪碳-胺氮共价加成键 (C(sp3)-N)，由此从真实连接表推导环氧基转化率 alpha_graph，彻底废除 st_num / 8.0 硬编码；
3. PBC 下原子球并集的几何自由体积 (Occupied Volume Union under PBC)：
   - 严格使用一维可分离 PBC 最小镜像网格算法计算 vdW 球 (probe=0.0 Å) 与膨胀探针球 (probe=1.2 Å) 的空间占据并集 V_union；
   - 严禁使用 sum(V_atom) 替代并集；
   - 未知元素无 Bondi 半径时明确抛出 KeyError 阻断，严禁静默回退；
   - 支持网格平移偏移 (grid_shift_fraction) 与多分辨率收敛性评估，记录网格分辨率、总网格点数与离散不确定度；
   - 明确区分几何未占据体积分数 (unoccupied volume fraction) 与连通可达孔隙 (percolating accessible pore volume = None/NOT_VERIFIED)；
4. 真实几何 PBC 氢键分析：
   - 从真实拓扑共价键识别供体对 D-H (N-H, O-H) 与受体 A (N, O，排除自身及同分子 <=2 键共价近邻)；
   - 依据 PBC 最小镜像距离 d_pbc(D, A) <= 0.35 nm 且夹角 angle(D-H-A) >= 120.0 度逐帧判定；
   - 输出每帧具体配对明细，绝不用加成次数代写；
5. 全轨迹共价键完整性与应变检查：
   - 对所有选定轨迹帧逐帧在 PBC 最小镜像下计算全部共价键与平衡键长 r0 的偏差 |r - r0|，记录最严重帧索引、时间及原子对。
"""

import os
import re
import math
import subprocess
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import parmed as pmd
import networkx as nx

try:
    from .pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_system_bond_strains_detailed,
        check_inter_residue_contacts_pbc,
    )
except (ImportError, ValueError):
    from src.pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_system_bond_strains_detailed,
        check_inter_residue_contacts_pbc,
    )
BONDI_VDW_RADII_NM: Dict[str, float] = {
    "H": 0.120,
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "S": 0.180,
    "F": 0.147,
    "Cl": 0.175,
    "Br": 0.185,
    "I": 0.198,
    "P": 0.180,
    "Si": 0.210,
    "B": 0.192,
}

STANDARD_ATOMIC_WEIGHTS_AMU: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
    "F": 18.998,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
    "P": 30.974,
    "Si": 28.085,
    "B": 10.811,
}


def check_xtc_native(
    xtc_path: str,
    out_log_path: str,
    gmx_bin: str = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx",
) -> Dict[str, Any]:
    """
    调用原生 gmx check -f <xtc> 检查轨迹文件完整性并保存日志。
    """
    if not os.path.exists(xtc_path) or os.path.getsize(xtc_path) == 0:
        raise FileNotFoundError(f"XTC 轨迹文件不存在或为空: {xtc_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_log_path)), exist_ok=True)
    cmd = [gmx_bin, "check", "-f", xtc_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    combined = res.stdout + "\n" + res.stderr

    with open(out_log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Returncode: {res.returncode}\n")
        f.write(combined)

    if res.returncode != 0:
        raise RuntimeError(f"gmx check 验证 XTC 失败 (exit {res.returncode}):\n{res.stderr}")

    m_atoms = re.search(r"#\s*Atoms\s+(\d+)", combined)
    m_coords = re.search(r"Coords\s+(\d+)\s+([0-9.]+)", combined)
    atoms_in_xtc = int(m_atoms.group(1)) if m_atoms else None
    frames_in_xtc = int(m_coords.group(1)) if m_coords else None
    dt_ps = float(m_coords.group(2)) if m_coords else None

    if atoms_in_xtc is None or frames_in_xtc is None or frames_in_xtc <= 0:
        raise RuntimeError(f"XTC 文件损坏或未包含有效坐标帧: {xtc_path}")

    return {
        "command": cmd,
        "returncode": res.returncode,
        "log_path": os.path.abspath(out_log_path),
        "atoms_in_xtc": atoms_in_xtc,
        "frames_in_xtc": frames_in_xtc,
        "timestep_ps": dt_ps,
    }


def extract_trajectory_frames_to_gro(
    xtc_path: str,
    tpr_path: str,
    out_gro_path: str,
    gmx_bin: str = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx",
    trjconv_log_path: Optional[str] = None,
) -> str:
    """
    使用 GROMACS 原生 gmx trjconv 将 xtc 轨迹转换为多帧 gro 文件，并保存命令日志。
    """
    if not os.path.exists(xtc_path) or os.path.getsize(xtc_path) == 0:
        raise FileNotFoundError(f"XTC 轨迹文件不存在或为空: {xtc_path}")
    if not os.path.exists(tpr_path) or os.path.getsize(tpr_path) == 0:
        raise FileNotFoundError(f"TPR 文件不存在或为空: {tpr_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_gro_path)), exist_ok=True)
    cmd = [gmx_bin, "trjconv", "-f", xtc_path, "-s", tpr_path, "-o", out_gro_path]
    res = subprocess.run(cmd, input="0\n", capture_output=True, text=True)

    if trjconv_log_path is None:
        trjconv_log_path = os.path.join(os.path.dirname(os.path.abspath(out_gro_path)), "trjconv_cmd.log")
    with open(trjconv_log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Returncode: {res.returncode}\n")
        f.write(f"--- STDOUT ---\n{res.stdout}\n")
        f.write(f"--- STDERR ---\n{res.stderr}\n")

    if res.returncode != 0 or not os.path.exists(out_gro_path) or os.path.getsize(out_gro_path) == 0:
        raise RuntimeError(f"gmx trjconv 执行失败 (exit {res.returncode}):\n{res.stderr}")

    return out_gro_path


def parse_multiframe_gro(gro_path: str, expected_atoms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    定宽解析多帧 gro 文件，提取每帧的物理时间、原子坐标与盒矢量。
    GRO 标准定宽: 列 20:28 为 x, 28:36 为 y, 36:44 为 z (nm)。
    """
    if not os.path.exists(gro_path) or os.path.getsize(gro_path) == 0:
        raise FileNotFoundError(f"多帧 GRO 文件不存在或为空: {gro_path}")

    frames: List[Dict[str, Any]] = []
    with open(gro_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = 0
    total_lines = len(lines)
    while idx < total_lines:
        title_line = lines[idx].strip()
        if not title_line and idx == total_lines - 1:
            break
        if not title_line:
            idx += 1
            continue

        if "t=" in title_line:
            try:
                t_str = title_line.split("t=")[1].split()[0]
                t_ps = float(t_str)
            except Exception as exc:
                raise ValueError(f"无法从 GRO 标题行解析时间 t=: '{title_line}'") from exc
        else:
            t_ps = 0.0 if len(frames) == 0 else float(len(frames) * 0.1)

        idx += 1
        if idx >= total_lines:
            raise ValueError(f"GRO 文件在标题行后截断: {gro_path}")

        try:
            n_atoms = int(lines[idx].strip())
        except ValueError as exc:
            raise ValueError(f"GRO 原子数行格式非法: '{lines[idx]}'") from exc

        if expected_atoms is not None and n_atoms != expected_atoms:
            raise ValueError(
                f"轨迹帧原子数 ({n_atoms}) 与拓扑原子数 ({expected_atoms}) 不匹配！"
            )

        idx += 1
        if idx + n_atoms >= total_lines:
            raise ValueError(f"GRO 文件在帧 {len(frames)} 坐标区截断")

        coords = []
        atom_names = []
        res_names = []

        for _ in range(n_atoms):
            line = lines[idx]
            r_name = line[5:10].strip()
            a_name = line[10:15].strip()
            x_str = line[20:28].strip()
            y_str = line[28:36].strip()
            z_str = line[36:44].strip()
            coords.append([float(x_str), float(y_str), float(z_str)])
            atom_names.append(a_name)
            res_names.append(r_name)
            idx += 1

        box_line = lines[idx].strip().split()
        if len(box_line) < 3:
            raise ValueError(f"GRO 盒尺寸行非法: '{lines[idx]}'")
        box_nm = np.array([float(box_line[0]), float(box_line[1]), float(box_line[2])], dtype=float)
        if np.any(box_nm <= 0.0) or np.any(np.isnan(box_nm)):
            raise ValueError(f"GRO 帧 {len(frames)} 盒矢量非法: {box_nm}")
        idx += 1

        frames.append({
            "frame_idx": len(frames),
            "time_ps": round(t_ps, 4),
            "coords_nm": np.array(coords, dtype=float),
            "box_nm": box_nm,
            "atom_names": atom_names,
            "res_names": res_names,
        })

    if not frames:
        raise ValueError(f"未能从 {gro_path} 解析出任何有效坐标帧")

    return frames


def compute_graph_topology(parm: pmd.Structure, initial_epoxy_sites: Optional[int] = None) -> Dict[str, Any]:
    """
    从体系真实成键连接表 (parm.bonds) 构建无向化学图，计算：
    1. 连通分量数 (num_connected_components)、各分量原子数 (component_sizes)、最大团簇占比 (largest_cluster_fraction)；
    2. 真实三元环氧环数量 (oxirane_ring_count: 1 个 O 与 2 个 C 构成的 3 元环)；
    3. 真实开环加成生成的脂肪碳-胺氮共价键数量 (addition_cn_bond_count: N 与 sp3 C 之间的共价键)；
    4. 基于真实化学图的环氧基开环转化率 alpha_graph = N_add / (N_oxirane + N_add)。
    绝不使用文件夹编号或 st_num 硬编码！
    """
    n_atoms = len(parm.atoms)
    if n_atoms == 0:
        raise ValueError("拓扑中原子数为 0")

    G = nx.Graph()
    for a in parm.atoms:
        G.add_node(a.idx, element=a.element_name, name=a.name, resname=a.residue.name)
    for b in parm.bonds:
        G.add_edge(b.atom1.idx, b.atom2.idx)

    comp_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    num_comp = len(comp_sizes)
    largest_cluster_size = comp_sizes[0]
    largest_cluster_fraction = float(largest_cluster_size / n_atoms)

    # 检测三元环氧环 (1 O + 2 C 构成三角形)
    oxirane_rings: List[Tuple[int, int, int]] = []
    for a in parm.atoms:
        if a.element_name == "O":
            c_nbrs = [nbr for nbr in G.neighbors(a.idx) if parm.atoms[nbr].element_name == "C"]
            if len(c_nbrs) == 2 and G.has_edge(c_nbrs[0], c_nbrs[1]):
                oxirane_rings.append((a.idx + 1, c_nbrs[0] + 1, c_nbrs[1] + 1))

    # 检测环氧基开环与胺基加成生成的 C(sp3)-N 共价键
    # 在 DGEBA/DDS 体系中，芳香胺 N 连接芳香碳 (degree=3)，而环氧开环生成的仲胺/叔胺额外连接脂肪族 CH2 碳 (degree=4)
    addition_cn_bonds: List[Tuple[int, int]] = []
    amine_h_counts: List[int] = []
    for a in parm.atoms:
        if a.element_name == "N":
            h_cnt = sum(1 for nbr in G.neighbors(a.idx) if parm.atoms[nbr].element_name == "H")
            amine_h_counts.append(h_cnt)
            for nbr in G.neighbors(a.idx):
                if parm.atoms[nbr].element_name == "C" and G.degree(nbr) == 4:
                    addition_cn_bonds.append((a.idx + 1, nbr + 1))

    # 检测脂族羟基 O-H 数量 (连接在 sp3 C 上的 -OH)
    aliphatic_oh_count = 0
    for a in parm.atoms:
        if a.element_name == "O":
            nbrs = list(G.neighbors(a.idx))
            has_h = any(parm.atoms[n].element_name == "H" for n in nbrs)
            has_sp3_c = any(parm.atoms[n].element_name == "C" and G.degree(n) == 4 for n in nbrs)
            if has_h and has_sp3_c:
                aliphatic_oh_count += 1

    n_oxirane = len(oxirane_rings)
    n_add = len(addition_cn_bonds)
    inferred_reacted = max(n_add, aliphatic_oh_count)
    if initial_epoxy_sites is not None and initial_epoxy_sites > 0:
        total_epoxy_sites = int(initial_epoxy_sites)
        n_reacted_epoxy = max(0, total_epoxy_sites - n_oxirane)
    else:
        n_reacted_epoxy = inferred_reacted
        total_epoxy_sites = n_oxirane + n_reacted_epoxy
    alpha_graph = float(n_reacted_epoxy / total_epoxy_sites) if total_epoxy_sites > 0 else 0.0
    return {
        "num_connected_components": num_comp,
        "component_sizes": comp_sizes,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": round(largest_cluster_fraction, 6),
        "oxirane_intact_ring_count": n_oxirane,
        "addition_aliphatic_cn_bond_count": n_add,
        "aliphatic_hydroxyl_oh_count": aliphatic_oh_count,
        "total_epoxy_reactive_sites_in_graph": total_epoxy_sites,
        "amine_nh_counts_per_nitrogen": amine_h_counts,
        "alpha_conversion_from_graph": round(alpha_graph, 6),
        "conversion_derivation_method": "TOPOLOGY_GRAPH_OXIRANE_AND_CN_ADDUCT_BOND_COUNT",
    }


def compute_grid_occupied_volume_union(
    coords_nm: np.ndarray,
    elements: List[str],
    box_nm: np.ndarray,
    probe_radius_nm: float = 0.0,
    target_spacing_nm: float = 0.10,
    grid_shift_fraction: Optional[np.ndarray] = None,
) -> Tuple[float, float, int, int]:
    """
    在周期性边界条件 (PBC) 下使用三维空间规则网格离散积分计算原子球占据并集 V_union。
    利用正交网格一维轴向 PBC 距离可分离性 (dx^2 + dy^2 + dz^2 <= R^2) 进行局部包围盒切片，
    数学结果与全网格暴力扫描 100% 恒等，杜绝 sum(V_atom) 球体重叠高估错误！
    遇到未知化学元素直接抛出 KeyError 阻断，严禁静默回退到碳半径。

    Returns:
        v_union_nm3: 真实空间占据并集体积 (nm^3)
        ffv_unoccupied: 几何未占据自由体积分数 = 1.0 - V_union / V_box
        n_occupied: 被占据的网格点数
        n_total_grid: 总网格点数
    """
    for elem in elements:
        if elem not in BONDI_VDW_RADII_NM:
            raise KeyError(f"未知化学元素 '{elem}' 缺失 Bondi 范德华半径定义，严禁静默回退！")

    if probe_radius_nm < 0.0:
        raise ValueError(f"探针半径不能为负数: {probe_radius_nm}")
    if target_spacing_nm <= 0.0:
        raise ValueError(f"网格间距必须 > 0: {target_spacing_nm}")

    box_arr = np.asarray(box_nm, dtype=float).ravel()[:3]
    if box_arr.shape[0] != 3 or np.any(box_arr <= 0.0) or np.any(np.isnan(box_arr)):
        raise ValueError(f"无效的盒尺寸 box_nm: {box_nm}")

    coords_arr = np.asarray(coords_nm, dtype=float)
    if coords_arr.shape != (len(elements), 3):
        raise ValueError(f"坐标形状 {coords_arr.shape} 与元素列表长度 {len(elements)} 不一致")

    v_box_nm3 = float(np.prod(box_arr))

    nx_pts = max(10, int(round(box_arr[0] / target_spacing_nm)))
    ny_pts = max(10, int(round(box_arr[1] / target_spacing_nm)))
    nz_pts = max(10, int(round(box_arr[2] / target_spacing_nm)))
    n_total_grid = int(nx_pts * ny_pts * nz_pts)

    gx = np.linspace(0.0, box_arr[0], nx_pts, endpoint=False)
    gy = np.linspace(0.0, box_arr[1], ny_pts, endpoint=False)
    gz = np.linspace(0.0, box_arr[2], nz_pts, endpoint=False)

    if grid_shift_fraction is not None:
        shift_f = np.asarray(grid_shift_fraction, dtype=float).ravel()[:3]
        dx_step = box_arr[0] / nx_pts
        dy_step = box_arr[1] / ny_pts
        dz_step = box_arr[2] / nz_pts
        gx = np.mod(gx + shift_f[0] * dx_step, box_arr[0])
        gy = np.mod(gy + shift_f[1] * dy_step, box_arr[1])
        gz = np.mod(gz + shift_f[2] * dz_step, box_arr[2])

    radii_nm = np.array([BONDI_VDW_RADII_NM[el] + probe_radius_nm for el in elements], dtype=float)
    occ = np.zeros((nx_pts, ny_pts, nz_pts), dtype=bool)

    for i in range(len(coords_arr)):
        pos = coords_arr[i]
        rc = radii_nm[i]
        rc2 = rc * rc

        dx = np.abs(gx - (pos[0] % box_arr[0]))
        dx = np.minimum(dx, box_arr[0] - dx)
        ix = np.where(dx <= rc)[0]
        if len(ix) == 0:
            continue

        dy = np.abs(gy - (pos[1] % box_arr[1]))
        dy = np.minimum(dy, box_arr[1] - dy)
        iy = np.where(dy <= rc)[0]
        if len(iy) == 0:
            continue

        dz = np.abs(gz - (pos[2] % box_arr[2]))
        dz = np.minimum(dz, box_arr[2] - dz)
        iz = np.where(dz <= rc)[0]
        if len(iz) == 0:
            continue

        d2 = (dx[ix, None, None] ** 2) + (dy[None, iy, None] ** 2) + (dz[None, None, iz] ** 2)
        occ[np.ix_(ix, iy, iz)] |= (d2 <= rc2)

    n_occupied = int(np.sum(occ))
    occ_fraction = float(n_occupied / n_total_grid)
    v_union_nm3 = float(occ_fraction * v_box_nm3)
    ffv_unoccupied = float(max(0.0, 1.0 - occ_fraction))

    return v_union_nm3, ffv_unoccupied, n_occupied, n_total_grid


def evaluate_ffv_convergence_and_uncertainty(
    coords_nm: np.ndarray,
    elements: List[str],
    box_nm: np.ndarray,
    probe_radius_nm: float = 0.0,
    primary_spacing_nm: float = 0.10,
) -> Dict[str, Any]:
    """
    在代表性构型上评估 3D PBC 网格并集算法的：
    1. 半格平移不变性离散不确定度 (shift by [0.5, 0.5, 0.5] grid spacing)；
    2. 粗网格 (1.5x spacing) 与细网格 (0.75x spacing) 的分辨率收敛差；
    3. 对比错误球体直接相加公式 1 - sum(V_sphere)/V_box 的重叠差值 (证明真实扣除了成键原子球重叠)。
    """
    v_u_base, ffv_base, n_occ_base, n_tot_base = compute_grid_occupied_volume_union(
        coords_nm, elements, box_nm, probe_radius_nm=probe_radius_nm, target_spacing_nm=primary_spacing_nm
    )
    _, ffv_half_shift, _, _ = compute_grid_occupied_volume_union(
        coords_nm,
        elements,
        box_nm,
        probe_radius_nm=probe_radius_nm,
        target_spacing_nm=primary_spacing_nm,
        grid_shift_fraction=np.array([0.5, 0.5, 0.5]),
    )
    _, ffv_coarse, _, n_tot_coarse = compute_grid_occupied_volume_union(
        coords_nm, elements, box_nm, probe_radius_nm=probe_radius_nm, target_spacing_nm=primary_spacing_nm * 1.5
    )
    v_u_fine, ffv_fine, _, n_tot_fine = compute_grid_occupied_volume_union(
        coords_nm, elements, box_nm, probe_radius_nm=probe_radius_nm, target_spacing_nm=max(0.06, primary_spacing_nm * 0.75)
    )

    # 计算未扣除重叠的独立球体积之和 sum(4/3 pi r^3)，用于量化成键原子重叠体积 V_overlap = sum(V_i) - V_union
    v_box_nm3 = float(np.prod(box_nm))
    naive_sum_vol_nm3 = float(
        sum((4.0 / 3.0) * math.pi * ((BONDI_VDW_RADII_NM[el] + probe_radius_nm) ** 3) for el in elements)
    )
    naive_ffv_no_overlap = float(max(0.0, 1.0 - naive_sum_vol_nm3 / v_box_nm3))
    overlap_volume_deducted_nm3 = float(naive_sum_vol_nm3 - v_u_fine)

    shift_abs_diff = abs(ffv_base - ffv_half_shift)
    resolution_abs_diff = abs(ffv_base - ffv_fine)
    combined_uncertainty = float(max(shift_abs_diff, resolution_abs_diff))

    return {
        "probe_radius_nm": probe_radius_nm,
        "primary_spacing_nm": primary_spacing_nm,
        "primary_grid_points": n_tot_base,
        "fine_grid_points": n_tot_fine,
        "occupied_grid_points_primary": n_occ_base,
        "v_union_primary_nm3": round(v_u_base, 6),
        "v_union_fine_nm3": round(v_u_fine, 6),
        "naive_sum_spheres_nm3": round(naive_sum_vol_nm3, 6),
        "overlap_volume_deducted_nm3": round(overlap_volume_deducted_nm3, 6),
        "ffv_union_primary": round(ffv_base, 6),
        "ffv_union_half_grid_shifted": round(ffv_half_shift, 6),
        "ffv_union_coarse": round(ffv_coarse, 6),
        "ffv_union_fine": round(ffv_fine, 6),
        "naive_ffv_without_overlap_correction": round(naive_ffv_no_overlap, 6),
        "half_grid_shift_abs_diff": round(shift_abs_diff, 6),
        "resolution_refinement_abs_diff": round(resolution_abs_diff, 6),
        "estimated_discretization_uncertainty": round(combined_uncertainty, 6),
    }


def detect_geometric_hbonds_pbc(
    parm: pmd.Structure,
    coords_nm: np.ndarray,
    box_nm: np.ndarray,
    max_d_da_nm: float = 0.35,
    min_angle_deg: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    从真实共价键供体和受体原子中检测符合几何准则与 PBC 的氢键：
    - 供体 D: 与 H 共价成键的 N 或 O；
    - 受体 A: 拓扑中的 N 或 O（排除自身 D，且排除与 D 直接共价成键或同残基内 1-2/1-3 共价近邻）；
    - 几何准则 (PBC 最小镜像):
      1. d_pbc(D, A) <= max_d_da_nm (0.35 nm)
      2. angle(D - H ... A) >= min_angle_deg (120.0 度)，其中角顶点为 H，即向量 H->D 与 H->A 的夹角 >= 120 度。
    """
    box_arr = np.asarray(box_nm, dtype=float).ravel()[:3]
    coords_arr = np.asarray(coords_nm, dtype=float)
    if coords_arr.shape != (len(parm.atoms), 3):
        raise ValueError(f"coords_nm shape {coords_arr.shape} != ({len(parm.atoms)}, 3)")

    # 构建共价邻接表以排除 1-2 / 1-3 分子内近邻
    adj: Dict[int, set] = {i: set() for i in range(len(parm.atoms))}
    dh_pairs: List[Tuple[int, int]] = []
    for bond in parm.bonds:
        i1, i2 = bond.atom1.idx, bond.atom2.idx
        adj[i1].add(i2)
        adj[i2].add(i1)
        e1, e2 = bond.atom1.element_name, bond.atom2.element_name
        if e1 in ("N", "O") and e2 == "H":
            dh_pairs.append((i1, i2))
        elif e2 in ("N", "O") and e1 == "H":
            dh_pairs.append((i2, i1))

    acceptor_indices = [a.idx for a in parm.atoms if a.element_name in ("N", "O")]
    detected_hbonds: List[Dict[str, Any]] = []

    for d_idx, h_idx in dh_pairs:
        pos_d = coords_arr[d_idx]
        pos_h = coords_arr[h_idx]
        # 排除 D 自身、与 D 直接成键(1-2)或间隔一个原子(1-3)的共价近邻
        excl_12_13 = {d_idx} | adj[d_idx]
        for nbr in list(adj[d_idx]):
            excl_12_13 |= adj[nbr]

        for a_idx in acceptor_indices:
            if a_idx in excl_12_13:
                continue

            pos_a = coords_arr[a_idx]
            dist_da = pbc_distance(pos_d, pos_a, box_arr)
            if dist_da <= max_d_da_nm:
                v_hd = pbc_displacement(pos_d, pos_h, box_arr)
                v_ha = pbc_displacement(pos_a, pos_h, box_arr)
                norm_hd = float(np.linalg.norm(v_hd))
                norm_ha = float(np.linalg.norm(v_ha))
                if norm_hd > 1e-5 and norm_ha > 1e-5:
                    cos_theta = float(np.dot(v_hd, v_ha) / (norm_hd * norm_ha))
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_theta)))
                    if angle_deg >= min_angle_deg:
                        dist_ha = norm_ha
                        is_inter = parm.atoms[d_idx].residue.idx != parm.atoms[a_idx].residue.idx
                        detected_hbonds.append({
                            "donor_idx": d_idx + 1,
                            "donor_name": parm.atoms[d_idx].name,
                            "donor_resname": parm.atoms[d_idx].residue.name,
                            "donor_resnum": parm.atoms[d_idx].residue.idx + 1,
                            "hydrogen_idx": h_idx + 1,
                            "hydrogen_name": parm.atoms[h_idx].name,
                            "acceptor_idx": a_idx + 1,
                            "acceptor_name": parm.atoms[a_idx].name,
                            "acceptor_resname": parm.atoms[a_idx].residue.name,
                            "acceptor_resnum": parm.atoms[a_idx].residue.idx + 1,
                            "is_intermolecular": bool(is_inter),
                            "dist_da_nm": round(float(dist_da), 5),
                            "dist_ha_nm": round(float(dist_ha), 5),
                            "angle_dha_deg": round(float(angle_deg), 2),
                        })

    return detected_hbonds


def evaluate_trajectory_bond_strains_pbc(
    parm: pmd.Structure,
    selected_frames: List[Dict[str, Any]],
    max_allowed_dev_A: float = 0.25,
) -> Dict[str, Any]:
    """
    对选定轨迹帧逐帧在 PBC 最小镜像下检查所有共价键长 r(t) 与平衡键长 r0 的偏差 |r(t) - r0|，
    不仅检查最终 gro，而且覆盖全部选定轨迹帧，记录最严重帧索引、物理时间与原子对。
    """
    if not selected_frames:
        raise ValueError("selected_frames 为空，无法执行轨迹共价键检查")

    per_frame_records: List[Dict[str, Any]] = []
    worst_overall: Optional[Dict[str, Any]] = None
    total_abnormal_frame_count = 0

    for f_info in selected_frames:
        f_idx = int(f_info["frame_idx"])
        t_ps = float(f_info["time_ps"])
        coords_A = np.asarray(f_info["coords_nm"], dtype=float) * 10.0
        box_A = np.asarray(f_info["box_nm"], dtype=float) * 10.0

        max_dev, abnormal_bonds, worst_bond = check_system_bond_strains_detailed(
            parm=parm,
            coords=coords_A,
            max_allowed_dev_A=max_allowed_dev_A,
            box=box_A,
        )
        if len(abnormal_bonds) > 0:
            total_abnormal_frame_count += 1

        frame_rec = {
            "frame_idx": f_idx,
            "time_ps": t_ps,
            "max_bond_dev_A": round(float(max_dev), 6),
            "max_bond_length_A": worst_bond["max_bond_length_in_frame_A"] if worst_bond else 0.0,
            "abnormal_bond_count_gt_threshold": len(abnormal_bonds),
            "worst_bond_in_frame": worst_bond,
        }
        per_frame_records.append(frame_rec)

        if worst_overall is None or max_dev > worst_overall["max_bond_dev_A"]:
            worst_overall = {
                "worst_frame_idx": f_idx,
                "worst_time_ps": t_ps,
                "max_bond_dev_A": round(float(max_dev), 6),
                "max_bond_length_A": worst_bond["max_bond_length_in_frame_A"] if worst_bond else 0.0,
                "worst_bond": worst_bond,
            }

    final_rec = per_frame_records[-1]
    return {
        "frames_checked_count": len(per_frame_records),
        "threshold_dev_A": max_allowed_dev_A,
        "frames_with_abnormal_bonds_count": total_abnormal_frame_count,
        "all_frames_within_threshold": total_abnormal_frame_count == 0,
        "trajectory_max_bond_dev_A": worst_overall["max_bond_dev_A"] if worst_overall else 0.0,
        "trajectory_max_bond_length_A": worst_overall["max_bond_length_A"] if worst_overall else 0.0,
        "worst_frame_summary": worst_overall,
        "final_frame_max_bond_dev_A": final_rec["max_bond_dev_A"],
        "final_frame_max_bond_length_A": final_rec["max_bond_length_A"],
        "per_frame_bond_summary": per_frame_records,
    }
