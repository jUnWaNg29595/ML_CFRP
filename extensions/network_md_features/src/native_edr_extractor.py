"""
native_edr_extractor.py
唯一的原生 GROMACS EDR 能量项提取器单入口
功能与准则：
1. 真实调用原生 GROMACS gmx energy 命令，严格读取 .edr 二进制文件；
2. 动态解析原生 XVG，按名称映射 LJ-14, LJ (SR), Coulomb-14, Coulomb (SR), Potential, Temperature, Pressure，
   并在 PME / DispCorr 启用时自动提取 Disper.-corr. 与 Coul.-recip.；
3. 严禁按固定列号猜测，严禁制造未声明的项；若显式请求的项在 EDR/XVG 中不存在，立即抛出 KeyError 阻断；
4. 支持明确的时间窗口 [t_start_ps, t_end_ps] 与 stride 筛选；若请求时间窗口无有效帧，立即抛出 ValueError 阻断；
5. 支持输入文件 SHA256 校验（expected_edr_sha256）；
6. 区分目标温度 (target_temperature_K, 来自 mdp ref_t) 与实测平均温度 (mean_temperature_K, 来自 EDR 帧均值)；
7. 当提供经严格验证的同温孤立气相单分子参考势能总和 (unbound_gas_reference_potential_kJ_mol) 时，
   按严格物理定义计算内聚能密度 CED = -(E_pot_condensed - E_pot_gas_sum) / V * 1.66053906660 (MPa)；
   未提供气相参考态时，严格保持 md_cohesive_energy_density_CED_MPa = None。
"""

import os
import re
import math
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
KJ_MOL_NM3_TO_MJ_M3 = 1.66053906660


def compute_file_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_mdp_protocol_metadata(mdp_filepath: Optional[str]) -> Dict[str, Any]:
    """
    从 mdout.mdp 或 nvt.mdp / prod.mdp 中读取真实的模拟设定参数。
    """
    meta: Dict[str, Any] = {
        "target_temperature_K": None,
        "coulombtype": None,
        "dispcorr": None,
        "pcoupl": None,
        "rcoulomb_nm": None,
        "rvdw_nm": None,
        "dt_ps": None,
        "nsteps": None,
        "nstxout_compressed": None,
        "nstenergy": None,
        "mdp_path": mdp_filepath if (mdp_filepath and os.path.exists(mdp_filepath)) else None,
    }
    if not mdp_filepath or not os.path.exists(mdp_filepath):
        return meta

    with open(mdp_filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.split(";")[0].strip()
            if "=" not in clean:
                continue
            k, v = [x.strip() for x in clean.split("=", 1)]
            if not v:
                continue
            k_low = k.lower().replace("_", "-")
            tok = v.split()[0]
            if k_low == "ref-t":
                try:
                    meta["target_temperature_K"] = float(tok)
                except ValueError:
                    pass
            elif k_low == "coulombtype":
                meta["coulombtype"] = tok
            elif k_low == "dispcorr":
                meta["dispcorr"] = tok
            elif k_low == "pcoupl":
                meta["pcoupl"] = tok
            elif k_low == "rcoulomb":
                try:
                    meta["rcoulomb_nm"] = float(tok)
                except ValueError:
                    pass
            elif k_low == "rvdw":
                try:
                    meta["rvdw_nm"] = float(tok)
                except ValueError:
                    pass
            elif k_low == "dt":
                try:
                    meta["dt_ps"] = float(tok)
                except ValueError:
                    pass
            elif k_low == "nsteps":
                try:
                    meta["nsteps"] = int(tok)
                except ValueError:
                    pass
            elif k_low == "nstxout-compressed":
                try:
                    meta["nstxout_compressed"] = int(tok)
                except ValueError:
                    pass
            elif k_low == "nstenergy":
                try:
                    meta["nstenergy"] = int(tok)
                except ValueError:
                    pass
    return meta


def extract_native_edr_data(
    edr_filepath: str,
    box_volume_nm3: float,
    out_dir: str,
    gmx_bin: str = GMX_BIN,
    terms: Optional[List[str]] = None,
    t_start_ps: Optional[float] = None,
    t_end_ps: Optional[float] = None,
    stride: int = 1,
    expected_edr_sha256: Optional[str] = None,
    mdp_filepath: Optional[str] = None,
    frame_volumes_nm3: Optional[np.ndarray] = None,
    unbound_gas_reference_potential_kJ_mol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    从实际 EDR 文件提取原生能量与热力学时间序列并计算统计量。
    """
    p_edr = Path(edr_filepath)
    if not p_edr.exists() or p_edr.stat().st_size == 0:
        raise FileNotFoundError(f"EDR 文件不存在或为空: {edr_filepath}")

    if box_volume_nm3 <= 0 or math.isnan(box_volume_nm3):
        raise ValueError(f"无效的盒体积数值: {box_volume_nm3} nm^3")

    if stride < 1:
        raise ValueError(f"stride 必须 >= 1, 收到: {stride}")

    edr_sha256 = compute_file_sha256(str(p_edr))
    if expected_edr_sha256 is not None and edr_sha256.lower() != expected_edr_sha256.lower():
        raise ValueError(
            f"EDR 输入文件 SHA256 校验失败: 期望 {expected_edr_sha256}, 实际 {edr_sha256} ({edr_filepath})"
        )

    os.makedirs(out_dir, exist_ok=True)

    if mdp_filepath is None:
        for cand_name in ["mdout.mdp", "prod.mdp", "nvt.mdp"]:
            cand_p = p_edr.parent / cand_name
            if cand_p.exists():
                mdp_filepath = str(cand_p)
                break
    mdp_meta = parse_mdp_protocol_metadata(mdp_filepath)

    stem = p_edr.stem
    xvg_path = os.path.join(out_dir, f"{stem}_native_energy.xvg")
    log_path = os.path.join(out_dir, f"{stem}_energy_cmd.log")

    if terms is not None:
        requested_terms = list(terms)
        if not requested_terms:
            raise ValueError("请求的能量项列表 terms 不能为空")
        strict_custom_terms = True
    else:
        strict_custom_terms = False
        requested_terms = [
            "LJ-14",
            "Coulomb-14",
            "LJ-(SR)",
            "Coulomb-(SR)",
            "Potential",
            "Temperature",
            "Pressure",
        ]
        if mdp_meta.get("coulombtype") and mdp_meta["coulombtype"].upper() == "PME":
            requested_terms.append("Coul.-recip.")
        if mdp_meta.get("dispcorr") and mdp_meta["dispcorr"].lower() not in ("no", "none"):
            requested_terms.append("Disper.-corr.")

    input_str = "\n".join(requested_terms) + "\n0\n"
    cmd = [gmx_bin, "energy", "-f", str(p_edr), "-o", xvg_path]
    res = subprocess.run(cmd, input=input_str, capture_output=True, text=True)

    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write(f"Command: {' '.join(cmd)}\n")
        f_log.write(f"Requested_Terms: {requested_terms}\n")
        f_log.write(f"Returncode: {res.returncode}\n")
        f_log.write(f"--- STDOUT ---\n{res.stdout}\n")
        f_log.write(f"--- STDERR ---\n{res.stderr}\n")

    if res.returncode != 0 or not os.path.exists(xvg_path) or os.path.getsize(xvg_path) == 0:
        raise RuntimeError(f"gmx energy 执行失败 (exit {res.returncode}):\n{res.stderr}")

    gmx_version = "UNKNOWN"
    m_ver = re.search(r"GROMACS\s*-\s*gmx\s+\w+,\s*([^\s(]+)", res.stdout + "\n" + res.stderr)
    if m_ver:
        gmx_version = m_ver.group(1)

    col_names = ["time_ps"]
    data_rows = []
    with open(xvg_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("@") and "legend" in line_str and re.search(r"@\s+s\d+\s+legend", line_str):
                m = re.search(r'legend\s+"([^"]+)"', line_str)
                if m:
                    col_names.append(m.group(1).strip())
            elif not line_str.startswith(("@", "#")):
                parts = [float(x) for x in line_str.split()]
                if len(parts) >= 2:
                    data_rows.append(parts)

    if not data_rows:
        raise RuntimeError(f"XVG 文件中未提取到有效数据行: {xvg_path}")

    n_cols = len(data_rows[0])
    if len(col_names) != n_cols:
        raise RuntimeError(
            f"XVG legend 列数 ({len(col_names)}) 与数据列数 ({n_cols}) 不一致: {col_names}"
        )

    def _norm_term(s: str) -> str:
        return re.sub(r"[\s\-_().]+", "", s.lower())

    exported_norm_map = {_norm_term(c): c for c in col_names[1:]}
    missing_requested = [t for t in requested_terms if _norm_term(t) not in exported_norm_map]
    if missing_requested:
        raise KeyError(
            f"请求的能量项在 EDR/XVG 中不存在: {missing_requested} (实际导出列: {col_names[1:]})"
        )

    df_all = pd.DataFrame(data_rows, columns=col_names)
    df_all["edr_frame_idx"] = np.arange(len(df_all), dtype=int)

    def find_col(patterns: List[str], required: bool = True) -> Optional[str]:
        for pat in patterns:
            for c in col_names[1:]:
                if re.search(pat, c, re.IGNORECASE):
                    return c
        if required:
            raise KeyError(f"未在 XVG 列中找到必需的能量项 {patterns} (现有列: {col_names[1:]})")
        return None

    col_lj14 = find_col([r"^LJ-14$", r"LJ.*14"], required=True)
    col_ljsr = find_col([r"^LJ\s*\(SR\)$", r"^LJ-\(SR\)$", r"LJ.*SR"], required=True)
    col_coul14 = find_col([r"^Coulomb-14$", r"Coulomb.*14"], required=True)
    col_coulsr = find_col([r"^Coulomb\s*\(SR\)$", r"^Coulomb-\(SR\)$", r"Coulomb.*SR"], required=True)
    col_pot = find_col([r"^Potential$", r"Potential"], required=True)
    col_temp = find_col([r"^Temperature$", r"Temperature"], required=True)
    col_pres = find_col([r"^Pressure$", r"Pressure"], required=False)
    col_disp = find_col([r"Disper.*corr", r"^Disper\.-corr\.$"], required=False)
    col_coul_recip = find_col([r"Coul.*recip", r"^Coul\.-recip\.$"], required=False)

    if frame_volumes_nm3 is not None:
        vols = np.asarray(frame_volumes_nm3, dtype=float)
        if len(vols) != len(df_all):
            raise ValueError(
                f"frame_volumes_nm3 长度 ({len(vols)}) 与 EDR 总帧数 ({len(df_all)}) 不一致"
            )
        if np.any(vols <= 0) or np.any(np.isnan(vols)):
            raise ValueError("frame_volumes_nm3 含有非正或 NaN 体积")
        df_all["box_volume_nm3"] = vols
    else:
        df_all["box_volume_nm3"] = float(box_volume_nm3)

    # 短程 + 1-4 项与含长程校正项
    df_all["lj_total_cutoff_14_plus_sr_kJ_mol"] = df_all[col_lj14] + df_all[col_ljsr]
    if col_disp is not None:
        df_all["lj_total_with_dispcorr_kJ_mol"] = df_all["lj_total_cutoff_14_plus_sr_kJ_mol"] + df_all[col_disp]
    else:
        df_all["lj_total_with_dispcorr_kJ_mol"] = df_all["lj_total_cutoff_14_plus_sr_kJ_mol"]

    df_all["coulomb_total_cutoff_14_plus_sr_kJ_mol"] = df_all[col_coul14] + df_all[col_coulsr]
    if col_coul_recip is not None:
        df_all["coulomb_total_with_recip_kJ_mol"] = df_all["coulomb_total_cutoff_14_plus_sr_kJ_mol"] + df_all[col_coul_recip]
    else:
        df_all["coulomb_total_with_recip_kJ_mol"] = df_all["coulomb_total_cutoff_14_plus_sr_kJ_mol"]

    # vdw_energy_density_MJ_m3: 对于 Cut-off 为 (LJ-14+LJ(SR))/V；对于启用 DispCorr 的 PME 为含色散校正总非键 LJ 密度
    df_all["vdw_energy_density_MJ_m3"] = (
        df_all["lj_total_with_dispcorr_kJ_mol"] / df_all["box_volume_nm3"]
    ) * KJ_MOL_NM3_TO_MJ_M3
    df_all["coulomb_energy_density_MJ_m3"] = (
        df_all["coulomb_total_with_recip_kJ_mol"] / df_all["box_volume_nm3"]
    ) * KJ_MOL_NM3_TO_MJ_M3

    all_frames_csv = os.path.join(out_dir, f"{stem}_native_energy_frames_all.csv")
    df_all.to_csv(all_frames_csv, index=False)

    mask = np.ones(len(df_all), dtype=bool)
    if t_start_ps is not None:
        mask &= (df_all["time_ps"].to_numpy() >= (float(t_start_ps) - 1e-7))
    if t_end_ps is not None:
        mask &= (df_all["time_ps"].to_numpy() <= (float(t_end_ps) + 1e-7))

    df_window = df_all.loc[mask].iloc[::stride].copy().reset_index(drop=True)
    if len(df_window) == 0:
        raise ValueError(
            f"请求的时间窗口 [{t_start_ps}, {t_end_ps}] ps (stride={stride}) 在 EDR 中没有匹配到任何帧！"
            f" (EDR 实际时间范围: [{df_all['time_ps'].min()}, {df_all['time_ps'].max()}] ps)"
        )

    raw_frames_csv = os.path.join(out_dir, f"{stem}_native_energy_frames.csv")
    df_window.to_csv(raw_frames_csv, index=False)

    n_frames_total = len(df_all)
    n_frames_selected = len(df_window)
    t_start_actual = float(df_window["time_ps"].min())
    t_end_actual = float(df_window["time_ps"].max())

    mean_temp_K = float(df_window[col_temp].mean())
    std_temp_K = float(df_window[col_temp].std(ddof=1)) if n_frames_selected > 1 else 0.0
    mean_pot_kJ_mol = float(df_window[col_pot].mean())
    std_pot_kJ_mol = float(df_window[col_pot].std(ddof=1)) if n_frames_selected > 1 else 0.0
    mean_vdw_density = float(df_window["vdw_energy_density_MJ_m3"].mean())
    std_vdw_density = float(df_window["vdw_energy_density_MJ_m3"].std(ddof=1)) if n_frames_selected > 1 else 0.0
    mean_coulomb_density = float(df_window["coulomb_energy_density_MJ_m3"].mean())
    std_coulomb_density = float(df_window["coulomb_energy_density_MJ_m3"].std(ddof=1)) if n_frames_selected > 1 else 0.0
    mean_pres_bar = float(df_window[col_pres].mean()) if col_pres is not None else None

    component_means_kJ_mol = {
        "LJ-14": float(df_window[col_lj14].mean()),
        "LJ (SR)": float(df_window[col_ljsr].mean()),
        "Coulomb-14": float(df_window[col_coul14].mean()),
        "Coulomb (SR)": float(df_window[col_coulsr].mean()),
        "Potential": mean_pot_kJ_mol,
        "Temperature_K": mean_temp_K,
    }
    if col_disp is not None:
        component_means_kJ_mol["Disper.-corr."] = float(df_window[col_disp].mean())
    if col_coul_recip is not None:
        component_means_kJ_mol["Coul.-recip."] = float(df_window[col_coul_recip].mean())
    if col_pres is not None and mean_pres_bar is not None:
        component_means_kJ_mol["Pressure_bar"] = mean_pres_bar

    # 严格内聚能密度 CED 计算（仅在提供同温孤立气相单分子参考势能总和时计算）
    if unbound_gas_reference_potential_kJ_mol is not None:
        ced_frames = (
            -(df_window[col_pot] - float(unbound_gas_reference_potential_kJ_mol))
            / df_window["box_volume_nm3"]
        ) * KJ_MOL_NM3_TO_MJ_M3
        ced_mean_MPa: Optional[float] = round(float(ced_frames.mean()), 6)
        ced_reason = "VERIFIED_FROM_ISOLATED_UNBOUND_GAS_MOLECULE_REFERENCE_AT_SAME_T"
    else:
        ced_mean_MPa = None
        ced_reason = "UNVERIFIED_UNBOUND_REFERENCE_STATE_AND_CUTOFF_14_COUPLED"

    return {
        "edr_filepath": str(p_edr.resolve()),
        "edr_source_sha256": edr_sha256,
        "gmx_version": gmx_version,
        "command_executed": cmd,
        "command_returncode": res.returncode,
        "xvg_path": os.path.abspath(xvg_path),
        "raw_frames_csv": os.path.abspath(raw_frames_csv),
        "all_frames_csv": os.path.abspath(all_frames_csv),
        "log_path": os.path.abspath(log_path),
        "mdp_protocol_metadata": mdp_meta,
        "target_temperature_K": mdp_meta["target_temperature_K"],
        "coulombtype_protocol": mdp_meta["coulombtype"],
        "dispcorr_protocol": mdp_meta["dispcorr"],
        "pcoupl_protocol": mdp_meta["pcoupl"],
        "frames_total_in_edr": n_frames_total,
        "frames_selected_count": n_frames_selected,
        "frames_saved_count": n_frames_selected,
        "selected_edr_frame_indices": df_window["edr_frame_idx"].astype(int).tolist(),
        "selected_times_ps": [round(float(t), 4) for t in df_window["time_ps"].tolist()],
        "time_window_requested_ps": [t_start_ps, t_end_ps],
        "stride_requested": stride,
        "time_window_start_ps": t_start_actual,
        "time_window_end_ps": t_end_actual,
        "sampling_window_desc": (
            f"{t_start_actual:.1f}-{t_end_actual:.1f} ps "
            f"({n_frames_selected}/{n_frames_total} saved frames, stride={stride})"
        ),
        "averaging_convention": "arithmetic_mean_over_selected_saved_frames",
        "energy_decomposition_note": (
            f"coulombtype={mdp_meta.get('coulombtype')}, dispcorr={mdp_meta.get('dispcorr')}; "
            "LJ_density=(LJ-14 + LJ(SR) [+ Disper.-corr.])/V, Coulomb_density=(Coulomb-14 + Coulomb(SR) [+ Coul.-recip.])/V."
        ),
        "unit_conversion_factor_kJ_mol_nm3_to_MJ_m3": KJ_MOL_NM3_TO_MJ_M3,
        "component_means_selected_frames": component_means_kJ_mol,
        "mean_temperature_K": round(mean_temp_K, 6),
        "std_temperature_K": round(std_temp_K, 6),
        "mean_potential_energy_kJ_mol": round(mean_pot_kJ_mol, 6),
        "std_potential_energy_kJ_mol": round(std_pot_kJ_mol, 6),
        "mean_vdw_energy_density_MJ_m3": round(mean_vdw_density, 6),
        "std_vdw_energy_density_MJ_m3": round(std_vdw_density, 6),
        "mean_coulomb_energy_density_MJ_m3": round(mean_coulomb_density, 6),
        "std_coulomb_energy_density_MJ_m3": round(std_coulomb_density, 6),
        "mean_pressure_bar": round(mean_pres_bar, 4) if mean_pres_bar is not None else None,
        "unbound_gas_reference_potential_kJ_mol": unbound_gas_reference_potential_kJ_mol,
        "md_cohesive_energy_density_CED_MPa": ced_mean_MPa,
        "ced_null_reason": ced_reason,
        "df_frames": df_window,
        "df_all_frames": df_all,
    }
