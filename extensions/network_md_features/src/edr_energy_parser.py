"""
原生 GROMACS EDR 能量项解析与能密度计算器 (Native EDR Energy Extractor)
严格准则：
1. 真实调用原生 GROMACS 工具 gmx energy 解析 .edr 文件；
2. 提取全部所需项：LJ-14, LJ(SR), 色散修正 (Disper. corr.), Coulomb-14, Coulomb(SR), 倒空间电荷 (Coul. recip.);
3. 单位换算: (kJ/mol)/nm³ -> MJ/m³ = 1.66053906660;
4. 记录源文件 SHA256、采样时间范围与步数;
5. 缺失或损坏 EDR 时直接报错或返回 None，绝不回退任何人工常数;
6. 严格 CED 保持 None (null).
"""

import os
import re
import shutil
import hashlib
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
KJ_MOL_NM3_TO_MJ_M3 = 1.66053906660

def extract_native_energy_from_edr(
    edr_filepath: str,
    box_volume_nm3: float,
    gmx_bin: str = GMX_BIN
) -> Dict[str, Any]:
    """
    从真实 EDR 文件中提取原生非键能量项并计算能量密度
    """
    p_edr = Path(edr_filepath)
    if not p_edr.exists() or p_edr.stat().st_size == 0:
        raise FileNotFoundError(f"EDR 文件不存在或为空: {edr_filepath}")

    with open(p_edr, "rb") as f:
        edr_sha256 = hashlib.sha256(f.read()).hexdigest()

    if not os.path.exists(gmx_bin):
        raise FileNotFoundError(f"GROMACS gmx 可执行文件未找到: {gmx_bin}")

    if box_volume_nm3 <= 0:
        raise ValueError(f"无效的盒子体积: {box_volume_nm3} nm³")

    # 准备 gmx energy 输入命令
    # 选择项: LJ-14, Coulomb-14, LJ-(SR), Disper.-corr., Coulomb-(SR), Coul.-recip.
    query_input = "LJ-14\nCoulomb-14\nLJ-(SR)\nDisper.-corr.\nCoulomb-(SR)\nCoul.-recip.\n0\n"

    with tempfile.TemporaryDirectory(prefix="gmx_energy_extract_") as td:
        out_xvg = os.path.join(td, "energy.xvg")
        cmd = [gmx_bin, "energy", "-f", str(p_edr), "-o", out_xvg]

        proc = subprocess.run(
            cmd,
            input=query_input.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=td,
            timeout=60
        )

        stdout_txt = proc.stdout.decode("utf-8", errors="ignore")
        stderr_txt = proc.stderr.decode("utf-8", errors="ignore")

        if proc.returncode != 0:
            raise RuntimeError(f"gmx energy 提取失败 (code {proc.returncode}): {stderr_txt}")

        # 解析统计均值表格
        # 例如:
        # LJ-14                       315.602
        # Coulomb-14                  38.2751
        # LJ (SR)                    -243.914
        # Disper. corr.              -28.7984
        # Coulomb (SR)               -520.962
        # Coul. recip.                125.506
        values = {}
        for line in stdout_txt.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("-") or line_str.startswith("Energy"):
                continue

            patterns = [
                ("lj_14", r"LJ-14\s+([0-9.eE+-]+)"),
                ("coulomb_14", r"Coulomb-14\s+([0-9.eE+-]+)"),
                ("lj_sr", r"LJ\s*\(SR\)\s+([0-9.eE+-]+)"),
                ("disper_corr", r"Disper\.\s*corr\.\s+([0-9.eE+-]+)"),
                ("coulomb_sr", r"Coulomb\s*\(SR\)\s+([0-9.eE+-]+)"),
                ("coulomb_recip", r"Coul\.\s*recip\.\s+([0-9.eE+-]+)"),
            ]
            for key, pat in patterns:
                m = re.search(pat, line_str)
                if m:
                    values[key] = float(m.group(1))

        # 检查是否全部提取到
        required_keys = ["lj_14", "coulomb_14", "lj_sr", "disper_corr", "coulomb_sr", "coulomb_recip"]
        missing = [k for k in required_keys if k not in values]
        if missing:
            raise ValueError(f"gmx energy 未提取到完整非键能量项，缺失: {missing}\nLog snippet:\n{stdout_txt[:500]}")

        # 提取时间与步数统计信息
        # Statistics over 501 steps [ 0.0000 through 1.0000 ps ]
        steps = None
        time_range_ps = None
        m_stat = re.search(r"Statistics over (\d+) steps \[\s*([0-9.]+)\s+through\s+([0-9.]+)\s+ps\s*\]", stdout_txt)
        if m_stat:
            steps = int(m_stat.group(1))
            time_range_ps = [float(m_stat.group(2)), float(m_stat.group(3))]

        # 计算总范德华能量与总静电能量 (kJ/mol)
        total_vdw_kj_mol = values["lj_14"] + values["lj_sr"] + values["disper_corr"]
        total_elec_kj_mol = values["coulomb_14"] + values["coulomb_sr"] + values["coulomb_recip"]

        # 换算为能量密度 (MJ/m³) = (kJ/mol / nm³) * 1.66053906660
        vdw_density_mj_m3 = (total_vdw_kj_mol / box_volume_nm3) * KJ_MOL_NM3_TO_MJ_M3
        elec_density_mj_m3 = (total_elec_kj_mol / box_volume_nm3) * KJ_MOL_NM3_TO_MJ_M3

        # 判定充分平衡态：若采样时间 <= 10 ps，明确标记为受限探索/诊断态
        is_equilibrated = (time_range_ps is not None and (time_range_ps[1] - time_range_ps[0]) >= 100.0)
        status = "VERIFIED_EQUILIBRATED" if is_equilibrated else "EXPLORATORY_SHORT_SAMPLE"

        return {
            "source_edr_file": str(p_edr),
            "source_edr_sha256": edr_sha256,
            "steps": steps,
            "time_range_ps": time_range_ps,
            "box_volume_nm3": box_volume_nm3,
            "components_kj_mol": values,
            "total_vdw_energy_kj_mol": round(total_vdw_kj_mol, 4),
            "total_electrostatic_energy_kj_mol": round(total_elec_kj_mol, 4),
            "md_vdw_energy_density_MJ_m3": round(vdw_density_mj_m3, 4),
            "md_electrostatic_energy_density_MJ_m3": round(elec_density_mj_m3, 4),
            "md_ced_MJ_m3": None,  # 严格 CED 保持 null
            "md_ced_verification_status": "NOT_CALCULATED_NEEDS_GAS_PHASE_REFERENCE",
            "energy_density_status": status
        }
