"""
pbc_utils.py
唯一共享的周期性边界条件 (PBC) 几何计算与接触/键长门禁模块。
严格遵循最小镜像约定 (Minimum Image Convention)，彻底消除欧氏距离漏检跨周期边界碰撞或误算跨边界共价键的缺陷。
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import parmed as pmd


def _validate_box(box: Optional[np.ndarray], parm: Optional[pmd.Structure] = None) -> np.ndarray:
    if box is None and parm is not None and parm.box is not None:
        box = parm.box[:3]
    if box is None:
        raise ValueError("PBC box vectors are required for minimum-image geometry calculation, got None.")
    box_arr = np.asarray(box, dtype=float).ravel()[:3]
    if box_arr.shape[0] != 3 or np.any(np.isnan(box_arr)) or np.any(box_arr <= 0.0):
        raise ValueError(f"Invalid PBC box dimensions: {box}")
    return box_arr


def pbc_displacement(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    计算满足正交周期性边界条件 (PBC) 的三维最小镜像位移矢量 r1 - r2。
    支持一维坐标数组 (3,) 或批量坐标数组 (..., 3)。
    """
    box_arr = _validate_box(box)
    dr = np.asarray(pos1, dtype=float) - np.asarray(pos2, dtype=float)
    return dr - box_arr * np.round(dr / box_arr)


def pbc_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> float:
    """
    计算两点间满足周期性边界条件的标量最小镜像距离。
    """
    dr_pbc = pbc_displacement(pos1, pos2, box)
    return float(np.linalg.norm(dr_pbc))


def pbc_pairwise_distances(coords1: np.ndarray, coords2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    批量计算两组坐标点在 PBC 下的两两最小镜像距离矩阵 (N1, N2)。
    """
    box_arr = _validate_box(box)
    c1 = np.asarray(coords1, dtype=float)
    c2 = np.asarray(coords2, dtype=float)
    diff = c1[:, np.newaxis, :] - c2[np.newaxis, :, :]
    diff_pbc = diff - box_arr * np.round(diff / box_arr)
    return np.linalg.norm(diff_pbc, axis=2)


def check_inter_residue_contacts_pbc(
    parm: pmd.Structure,
    min_allowed_dist_A: float = 3.0,
    box: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[Tuple[int, int, str, str, str, str]]]:
    """
    唯一共享的跨残基原子级接触检查函数（严格使用 PBC 最小镜像距离，单位 Å）。
    Returns:
        min_dist_A: 跨残基原子对最小周期距离 (Å)
        clashing_pair: 最小距离原子对 (atom1_1based, atom2_1based, atom1_name, atom2_name, res1_name, res2_name)
    """
    box_arr = _validate_box(box, parm=parm)
    xyz = np.asarray(coords if coords is not None else parm.coordinates, dtype=float)
    n_atoms = len(parm.atoms)
    if xyz.shape != (n_atoms, 3):
        raise ValueError(f"Coordinates shape {xyz.shape} does not match topology atom count {n_atoms}")

    res_indices = np.array([a.residue.idx for a in parm.atoms], dtype=int)
    min_dist = float("inf")
    clashing_pair = None

    for i in range(n_atoms):
        mask = res_indices[i + 1:] != res_indices[i]
        if not np.any(mask):
            continue
        j_candidates = np.nonzero(mask)[0] + (i + 1)
        dr = xyz[j_candidates] - xyz[i]
        dr_pbc = dr - box_arr * np.round(dr / box_arr)
        dists = np.linalg.norm(dr_pbc, axis=1)
        min_idx_local = int(np.argmin(dists))
        d_min_local = float(dists[min_idx_local])
        if d_min_local < min_dist:
            min_dist = d_min_local
            j = int(j_candidates[min_idx_local])
            clashing_pair = (
                i + 1,
                j + 1,
                parm.atoms[i].name,
                parm.atoms[j].name,
                parm.atoms[i].residue.name,
                parm.atoms[j].residue.name,
            )

    return min_dist, clashing_pair


def check_inter_residue_contacts(
    parm: pmd.Structure,
    min_allowed_dist_A: float = 3.0,
    box: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[Tuple[int, int, str, str, str, str]]]:
    """向后兼容函数名，强制调用 PBC 最小镜像实现。"""
    return check_inter_residue_contacts_pbc(
        parm=parm,
        min_allowed_dist_A=min_allowed_dist_A,
        box=box,
        coords=coords,
    )


def check_system_bond_strains(
    parm: pmd.Structure,
    coords: Optional[np.ndarray] = None,
    max_allowed_dev_A: float = 0.25,
    box: Optional[np.ndarray] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    唯一共享的共价键长应变门禁：在 PBC 最小镜像下核验所有共价键长 r 与平衡键长 r0 的偏差 (单位 Å)。
    返回 (max_dev, abnormal_bonds)。如需最严重键详情，可调用 check_system_bond_strains_detailed。
    """
    max_dev, abnormal_bonds, _ = check_system_bond_strains_detailed(
        parm=parm,
        coords=coords,
        max_allowed_dev_A=max_allowed_dev_A,
        box=box,
    )
    return max_dev, abnormal_bonds


def check_system_bond_strains_detailed(
    parm: pmd.Structure,
    coords: Optional[np.ndarray] = None,
    max_allowed_dev_A: float = 0.25,
    box: Optional[np.ndarray] = None,
) -> Tuple[float, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    返回 (max_dev_A, abnormal_bonds, worst_bond_info)，全程采用 PBC 最小镜像距离。
    """
    xyz = np.asarray(coords if coords is not None else parm.coordinates, dtype=float)
    n_atoms = len(parm.atoms)
    if xyz.shape != (n_atoms, 3):
        raise ValueError(f"Coordinates shape {xyz.shape} does not match topology atom count {n_atoms}")

    box_arr = _validate_box(box, parm=parm)

    max_dev = 0.0
    max_length = 0.0
    abnormal_bonds: List[Dict[str, Any]] = []
    worst_bond: Optional[Dict[str, Any]] = None

    for bond in parm.bonds:
        a1 = bond.atom1
        a2 = bond.atom2
        r0 = float(bond.type.req)  # Angstroms
        k = float(bond.type.k)     # kcal/(mol*A^2)
        actual_r = pbc_distance(xyz[a1.idx], xyz[a2.idx], box_arr)
        dev = abs(actual_r - r0)

        bond_record = {
            "atoms": [a1.idx + 1, a2.idx + 1],
            "names": [a1.name, a2.name],
            "elements": [a1.element_name, a2.element_name],
            "residues": [a1.residue.name, a2.residue.name],
            "residue_indices": [a1.residue.idx + 1, a2.residue.idx + 1],
            "actual_r_A": round(actual_r, 6),
            "r0_A": round(r0, 6),
            "dev_A": round(dev, 6),
            "k": round(k, 2),
        }

        if actual_r > max_length:
            max_length = actual_r

        if dev > max_dev or worst_bond is None:
            max_dev = dev
            worst_bond = dict(bond_record)
            worst_bond["max_bond_length_in_frame_A"] = round(max_length, 6)

        if dev > max_allowed_dev_A:
            abnormal_bonds.append(bond_record)

    if worst_bond is not None:
        worst_bond["max_bond_length_in_frame_A"] = round(max_length, 6)

    return float(max_dev), abnormal_bonds, worst_bond
