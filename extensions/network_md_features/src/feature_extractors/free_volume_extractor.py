"""
真实自由体积特征提取器 (Rigorous Free Volume Extractor)
科学准则：
1. 严禁使用“探针体积分数 * 0.75”伪造连通孔隙，未做三维网格渗流绕行连通性分析时置为 None (null) 并标为 NOT_VERIFIED；
2. 修复探针键名碰撞：显式命名 md_ffv_probe_0A (0.0 A 几何空隙) 与 md_ffv_probe_1_2A (1.2 A 渗透探针)；
3. 未知元素必须报错，严禁静默回退默认半径；
4. 严格采用 Bondi 范德华半径标准与周期性边界条件 (PBC) 最小镜像距离；
5. 保证三维网格在周期性包裹与平移下物理等价。
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Bondi 范德华半径表 (单位: nm, 来源: Bondi 1964; Mantina et al. 2009)
BONDI_VDW_RADII_NM: Dict[str, float] = {
    "H": 0.120,
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "S": 0.180,
    "CL": 0.175,
    "F": 0.147,
    "P": 0.180,
    "BR": 0.185,
    "I": 0.198
}

STANDARD_ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
    "CL": 35.453,
    "F": 18.998,
    "P": 30.974
}

class FreeVolumeExtractor:
    def __init__(self, grid_resolution_nm: float = 0.04):
        self.grid_resolution_nm = grid_resolution_nm

    def get_vdw_radius(self, element: str) -> float:
        elem_clean = element.upper().strip()
        if elem_clean not in BONDI_VDW_RADII_NM:
            raise ValueError(f"未知元素 '{element}'，无标准 Bondi 范德华半径，禁止使用默认值")
        return BONDI_VDW_RADII_NM[elem_clean]

    def get_atomic_mass(self, element: str) -> float:
        elem_clean = element.upper().strip()
        if elem_clean not in STANDARD_ATOMIC_WEIGHTS:
            raise ValueError(f"未知元素 '{element}'，无标准原子量")
        return STANDARD_ATOMIC_WEIGHTS[elem_clean]

    def calculate_density(
        self,
        coords_nm: np.ndarray,
        elements: List[str],
        box_lengths_nm: np.ndarray,
        atomic_masses: Optional[List[float]] = None
    ) -> float:
        lx, ly, lz = box_lengths_nm[0], box_lengths_nm[1], box_lengths_nm[2]
        vol_nm3 = lx * ly * lz

        if atomic_masses is not None and len(atomic_masses) == len(elements):
            total_mass_amu = sum(atomic_masses)
        else:
            total_mass_amu = sum(self.get_atomic_mass(e) for e in elements)

        # 1 amu = 1.66053906660e-24 g, 1 nm3 = 1e-21 cm3
        mass_g = total_mass_amu * 1.66053906660e-24
        vol_cm3 = vol_nm3 * 1e-21
        density_g_cm3 = mass_g / vol_cm3 if vol_cm3 > 0 else 0.0
        return float(density_g_cm3)

    def extract_free_volume_grid(
        self,
        coords_nm: np.ndarray,
        elements: List[str],
        box_lengths_nm: np.ndarray,
        probe_radii_nm: List[float] = [0.0, 0.12]
    ) -> Dict[str, Any]:
        """
        利用三维网格在周期性边界条件 (PBC) 下精确测定自由体积
        """
        lx, ly, lz = box_lengths_nm[0], box_lengths_nm[1], box_lengths_nm[2]
        vol_box_nm3 = lx * ly * lz

        # 校验未知元素
        radii = np.array([self.get_vdw_radius(e) for e in elements])
        coords = np.mod(coords_nm, box_lengths_nm)

        nx = max(8, int(round(lx / self.grid_resolution_nm)))
        ny = max(8, int(round(ly / self.grid_resolution_nm)))
        nz = max(8, int(round(lz / self.grid_resolution_nm)))
        total_grid_points = nx * ny * nz

        gx = np.linspace(0, lx, nx, endpoint=False)
        gy = np.linspace(0, ly, ny, endpoint=False)
        gz = np.linspace(0, lz, nz, endpoint=False)

        results = {}
        for r_p in probe_radii_nm:
            eff_radii = radii + r_p
            max_r = np.max(eff_radii)

            free_count = 0
            X, Y = np.meshgrid(gx, gy, indexing='ij')

            for iz in range(nz):
                z_val = gz[iz]
                dz = np.abs(coords[:, 2] - z_val)
                dz = np.minimum(dz, lz - dz)
                z_mask = dz <= max_r
                if not np.any(z_mask):
                    free_count += nx * ny
                    continue

                sub_coords = coords[z_mask]
                sub_r_sq = eff_radii[z_mask] ** 2
                sub_dz_sq = dz[z_mask] ** 2
                r_xy_sq_max = sub_r_sq - sub_dz_sq

                valid_xy = r_xy_sq_max > 0
                if not np.any(valid_xy):
                    free_count += nx * ny
                    continue

                sub_coords = sub_coords[valid_xy]
                r_xy_sq = r_xy_sq_max[valid_xy]

                slice_occupied = np.zeros((nx, ny), dtype=bool)
                for at_idx in range(len(sub_coords)):
                    ax, ay = sub_coords[at_idx, 0], sub_coords[at_idx, 1]
                    req_sq = r_xy_sq[at_idx]

                    dx = np.abs(X - ax)
                    dx = np.minimum(dx, lx - dx)
                    dy = np.abs(Y - ay)
                    dy = np.minimum(dy, ly - dy)

                    dist_sq = dx**2 + dy**2
                    slice_occupied |= (dist_sq <= req_sq)
                    if np.all(slice_occupied):
                        break

                free_count += np.sum(~slice_occupied)

            ffv_ratio = float(free_count) / float(total_grid_points)
            # 严格消除键名碰撞，按明确半径命名
            tag = "0A" if abs(r_p - 0.0) < 1e-4 else ("1_2A" if abs(r_p - 0.12) < 1e-3 else f"{r_p*10:.1f}A")
            results[f"probe_{tag}"] = round(ffv_ratio, 4)

        density = self.calculate_density(coords, elements, box_lengths_nm)

        return {
            "md_density_g_cm3": round(density, 4),
            "md_ffv_probe_0A": results.get("probe_0A"),
            "md_ffv_probe_1_2A": results.get("probe_1_2A"),
            "md_pore_volume_fraction": None,
            "md_pore_verification_status": "NOT_VERIFIED",
            "qc_box_size_nm": round(float(np.mean(box_lengths_nm)), 3),
            "qc_grid_points_total": total_grid_points
        }
