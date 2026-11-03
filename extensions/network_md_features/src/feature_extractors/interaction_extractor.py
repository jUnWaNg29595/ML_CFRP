"""
真实相互作用与能量密度特征提取器 (Rigorous Interaction & Energy Density Extractor)
科学准则：
1. 彻底删除任何无物理依据的经验公式（如 abs(vdw) + 0.5*abs(elec)）；
2. 热固性交联网络由于全体系共价连续，未建立严格化学键解聚气相孤立参考态前，
   正式 md_ced_MJ_m3 必须显式输出 None (null)，并标记为 NOT_VERIFIED；
3. 输出物理定义清晰的分项能量密度（单位 MJ/m^3，保留物理正负号）：
   - md_vdw_energy_density_MJ_m3: 范德华相互作用能量除以平衡体积；
   - md_electrostatic_energy_density_MJ_m3: 全长程静电相互作用能量除以平衡体积；
   - 严格物理换算系数: 1 (kJ/mol)/nm^3 = 1.66053906660 MJ/m^3；
4. 氢键密度 md_hbond_density_nm3：
   - 供体 H 严格限定于与 O 或 N 形成共价键的活性质子 (O-H / N-H)；
   - 受体为电负性原子 (O / N)；
   - 几何双判据：供受体距离 D...A <= 0.35 nm 且 D-H...A 夹角 >= 120 度。
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

KJ_MOL_NM3_TO_MJ_M3 = 1.66053906660

class InteractionExtractor:
    def __init__(self):
        pass

    def calculate_energy_densities(
        self,
        vdw_energy_kJ_mol: float,
        elec_energy_kJ_mol: float,
        box_volume_nm3: float
    ) -> Dict[str, Any]:
        """
        从原生能量输出计算范德华与静电分项能量密度 (MJ/m^3)
        保留严格的物理符号（负值代表相互作用吸引能量）
        """
        if box_volume_nm3 <= 0:
            raise ValueError(f"盒体积必须 > 0: {box_volume_nm3}")

        vdw_density = (vdw_energy_kJ_mol / box_volume_nm3) * KJ_MOL_NM3_TO_MJ_M3
        elec_density = (elec_energy_kJ_mol / box_volume_nm3) * KJ_MOL_NM3_TO_MJ_M3

        return {
            "md_vdw_energy_density_MJ_m3": round(float(vdw_density), 2),
            "md_electrostatic_energy_density_MJ_m3": round(float(elec_density), 2),
            "md_ced_MJ_m3": None,
            "md_ced_verification_status": "NOT_VERIFIED",
            "ced_note": "热固性交联网络由于共价全连接，经典小分子气相蒸发参考态不适用。在严格键解聚热力学循环建立前不输出数值。"
        }

    def extract_hydrogen_bonds_pbc(
        self,
        coords_nm: np.ndarray,
        elements: List[str],
        covalent_bonds: List[Tuple[int, int]],
        box_lengths_nm: np.ndarray,
        distance_cutoff_nm: float = 0.35,
        angle_cutoff_deg: float = 120.0
    ) -> Dict[str, Any]:
        """
        基于真实共价拓扑在周期性边界条件 (PBC) 下识别几何氢键
        """
        N = len(elements)
        lx, ly, lz = box_lengths_nm[0], box_lengths_nm[1], box_lengths_nm[2]
        vol_nm3 = lx * ly * lz

        elem_clean = [e.upper().strip() for e in elements]

        # 1. 建立真实的 H -> Donor 对应关系（必须由共价键 covalent_bonds 决定）
        h_to_donor = {}
        for u, v in covalent_bonds:
            if elem_clean[u] == "H" and elem_clean[v] in ("O", "N"):
                h_to_donor[u] = v
            elif elem_clean[v] == "H" and elem_clean[u] in ("O", "N"):
                h_to_donor[v] = u

        # 2. 识别受体（O, N）
        acceptors = [i for i, e in enumerate(elem_clean) if e in ("O", "N")]

        hbond_count = 0
        hbond_pairs = []

        for h_idx, d_idx in h_to_donor.items():
            h_pos = coords_nm[h_idx]
            d_pos = coords_nm[d_idx]

            for a_idx in acceptors:
                if a_idx == d_idx:
                    continue  # 不与自身共价相连的供体成键

                a_pos = coords_nm[a_idx]

                # D...A 距离 (PBC 最小镜像)
                dr_da = a_pos - d_pos
                dr_da -= box_lengths_nm * np.round(dr_da / box_lengths_nm)
                dist_da = np.linalg.norm(dr_da)

                if dist_da > distance_cutoff_nm:
                    continue

                # 夹角 D-H...A 计算
                v_hd = d_pos - h_pos
                v_hd -= box_lengths_nm * np.round(v_hd / box_lengths_nm)

                v_ha = a_pos - h_pos
                v_ha -= box_lengths_nm * np.round(v_ha / box_lengths_nm)

                norm_hd = np.linalg.norm(v_hd)
                norm_ha = np.linalg.norm(v_ha)

                if norm_hd > 0 and norm_ha > 0:
                    cos_dha = np.dot(v_hd, v_ha) / (norm_hd * norm_ha)
                    cos_dha = np.clip(cos_dha, -1.0, 1.0)
                    angle_deg = math.degrees(math.acos(cos_dha))

                    if angle_deg >= angle_cutoff_deg:
                        hbond_count += 1
                        hbond_pairs.append({
                            "donor": d_idx,
                            "hydrogen": h_idx,
                            "acceptor": a_idx,
                            "dist_da_nm": round(float(dist_da), 3),
                            "angle_deg": round(float(angle_deg), 1)
                        })

        hbond_density_nm3 = float(hbond_count) / vol_nm3 if vol_nm3 > 0 else 0.0

        return {
            "md_hbond_density_nm3": round(float(hbond_density_nm3), 4),
            "qc_hbond_count_total": int(hbond_count),
            "qc_active_proton_donors": len(h_to_donor)
        }
