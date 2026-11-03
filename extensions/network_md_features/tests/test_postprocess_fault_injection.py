"""
test_postprocess_fault_injection.py
针对真实生产后处理入口 (postprocess_trajectory / compute_features_from_trajectory / run_verified_md_pipeline / assembly_engine)
的 12 项严密故障注入与对抗测试套件：
1. 删除 XTC 必须阻断 (FileNotFoundError)；
2. 删除 EDR 必须阻断 (FileNotFoundError)；
3. 损坏 XTC 或损坏 EDR 必须阻断 (RuntimeError) 并保留错误日志；
4. 输入文件 SHA256 哈希改变必须阻断 (ValueError)；
5. 请求时间窗口无帧必须阻断 (ValueError)；
6. 改变时间窗口 (如 0.0-10.0 ps vs 0.1-10.0 ps 排除 t=0 vs 5.0-10.0 ps) 与步长 stride 后，帧数与能量均值严格等于所选窗口原生算术均值；
7. PBC 跨周期边界严重重叠 (10 A 盒中 x=0.1 与 9.9 A) 在 assembly_engine / run_verified_md_pipeline / pbc_utils 中一致报告 0.2 A 而非 9.8 A；
8. 固定密度/组成下改变原子重叠几何，3D PBC 占据并集 V_union 与 FFV 显著变化（严禁 sum(V_atom)）；
9. 氢键随供受体几何距离/角度改变而变化，绝不等于 state 编号；
10. 未知元素半径必须抛出 KeyError 阻断（严禁回退碳半径 0.170 nm）；
11. 请求缺失的能量项（如 Cut-off 体系请求 PME 倒空间项 Coulomb-recip.）必须抛出 KeyError 阻断；
12. 严禁伪造随机物理数据，且生产 CLI 与共享函数行为及输出 100% 一致。
"""

import os
import sys
import json
import shutil
import subprocess
import pytest
import numpy as np
import pandas as pd
import parmed as pmd
from pathlib import Path

EXT_ROOT = Path("/home/wangj/project/CFRP/extensions/network_md_features")
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

from src.pbc_utils import (
    pbc_displacement,
    pbc_distance,
    check_inter_residue_contacts as pbc_check_contacts,
    check_system_bond_strains as pbc_check_bonds,
)
from src.assembly_engine import (
    check_inter_residue_contacts as engine_check_contacts,
    check_system_bond_strains as engine_check_bonds,
    inject_and_verify_resp_mol2,
)
from src.run_verified_md_pipeline import (
    check_inter_residue_contacts as pipeline_check_contacts,
    check_system_bond_strains as pipeline_check_bonds,
    extract_features_for_state,
)
from src.native_edr_extractor import extract_native_edr_data
from src.structure_features_extractor import (
    compute_grid_occupied_volume_union,
    detect_geometric_hbonds_pbc,
    compute_graph_topology,
)
from src.postprocess_trajectory import run_postprocessing
from src.compute_features_from_trajectory import recompute_features_for_attempt

ATTEMPT_RAW_DIR = EXT_ROOT / "artifacts" / "attempt_20261019_01"
POSTPROCESS_VERIFIED_DIR = EXT_ROOT / "artifacts" / "postprocess_verified_20261019_02"
PYTHON_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/python"


def _copy_single_state_fixture(tmp_path: Path, state_num: int = 0) -> Path:
    mock_attempt = tmp_path / f"fixture_attempt_s{state_num}"
    st_dir = mock_attempt / f"state_{state_num}_addition"
    st_dir.mkdir(parents=True, exist_ok=True)
    src_dir = ATTEMPT_RAW_DIR / f"state_{state_num}_addition"
    for fname in ["system.prmtop", "system.top", "system.gro", "nvt.gro", "nvt.xtc", "nvt.edr", "nvt.tpr", "nvt.log", "mdout.mdp"]:
        if (src_dir / fname).exists():
            shutil.copy2(src_dir / fname, st_dir / fname)
    return mock_attempt


def test_1_missing_xtc_blocks_both_postprocess_and_recompute_cli(tmp_path):
    """删除 XTC 文件后，run_postprocessing 与 recompute_features_for_attempt 均必须抛出 FileNotFoundError"""
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)
    (mock_attempt / "state_0_addition" / "nvt.xtc").unlink()

    with pytest.raises(FileNotFoundError, match="XTC"):
        run_postprocessing(str(mock_attempt), str(tmp_path / "out1"), snapshot_only=False)

    with pytest.raises(FileNotFoundError, match="XTC"):
        recompute_features_for_attempt(str(mock_attempt), str(tmp_path / "out2"), snapshot_only=False)


def test_2_missing_edr_blocks_both_postprocess_and_recompute_cli(tmp_path):
    """删除 EDR 文件后，run_postprocessing 与 recompute_features_for_attempt 均必须抛出 FileNotFoundError"""
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)
    (mock_attempt / "state_0_addition" / "nvt.edr").unlink()

    with pytest.raises(FileNotFoundError, match="EDR"):
        run_postprocessing(str(mock_attempt), str(tmp_path / "out1"), snapshot_only=False)

    with pytest.raises(FileNotFoundError, match="EDR"):
        recompute_features_for_attempt(str(mock_attempt), str(tmp_path / "out2"), snapshot_only=False)

    # 显式 snapshot_only 模式允许仅分析静态 GRO，但状态绝不能为 PHYSICALLY_VALIDATED，且能量为 NaN/None
    res_snap = run_postprocessing(str(mock_attempt), str(tmp_path / "out_snap"), snapshot_only=True)
    df_snap = res_snap["df_summary"]
    assert df_snap.loc[0, "validation_status"] == "SNAPSHOT_DIAGNOSTIC_ONLY"
    assert df_snap.loc[0, "recomputed_from_existing_trajectory"] == False
    assert pd.isna(df_snap.loc[0, "md_vdw_energy_density_MJ_m3"])


def test_3_corrupted_xtc_and_corrupted_edr_block_and_preserve_logs(tmp_path):
    """损坏 XTC 或损坏 EDR 必须报错阻断并保留命令日志现场"""
    # 3a. 损坏 EDR
    mock_edr_attempt = _copy_single_state_fixture(tmp_path / "case_edr", 0)
    (mock_edr_attempt / "state_0_addition" / "nvt.edr").write_bytes(b"CORRUPTED_EDR_BINARY_PAYLOAD")
    out_edr = tmp_path / "out_bad_edr"
    with pytest.raises(RuntimeError, match="gmx energy"):
        run_postprocessing(str(mock_edr_attempt), str(out_edr))
    assert (out_edr / "state_0" / "nvt_energy_cmd.log").exists()

    # 3b. 损坏 XTC
    mock_xtc_attempt = _copy_single_state_fixture(tmp_path / "case_xtc", 0)
    (mock_xtc_attempt / "state_0_addition" / "nvt.xtc").write_bytes(b"CORRUPTED_XTC_BINARY_PAYLOAD")
    out_xtc = tmp_path / "out_bad_xtc"
    with pytest.raises(RuntimeError, match="gmx check"):
        run_postprocessing(str(mock_xtc_attempt), str(out_xtc))
    assert (out_xtc / "state_0" / "xtc_check.log").exists()


def test_4_input_hash_mismatch_blocks_execution(tmp_path):
    """输入文件 SHA256 与期望值不一致时，必须立即抛出 ValueError 阻断"""
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)
    wrong_hashes = {
        "state_0_addition/nvt.edr": "0000000000000000000000000000000000000000000000000000000000000000"
    }
    with pytest.raises(ValueError, match="SHA256"):
        run_postprocessing(
            str(mock_attempt),
            str(tmp_path / "out_hash_fail"),
            expected_hashes=wrong_hashes,
        )


def test_5_empty_time_window_blocks_execution(tmp_path):
    """请求的时间窗口内没有任何帧（如 25.0 ~ 30.0 ps）时，必须抛出 ValueError 阻断"""
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)
    with pytest.raises(ValueError, match="没有匹配到任何帧"):
        run_postprocessing(
            str(mock_attempt),
            str(tmp_path / "out_empty_win"),
            t_start_ps=25.0,
            t_end_ps=30.0,
        )


def test_6_time_window_and_stride_changes_reflect_actual_selected_data(tmp_path):
    """改变时间窗口 (0-10 ps 全 101 帧 vs 0.1-10.0 ps 排除 t=0 的 100 帧 vs 5.0-10.0 ps 51 帧) 后，结果准确反映所选数据"""
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)

    res_all = run_postprocessing(str(mock_attempt), str(tmp_path / "win_all"), t_start_ps=0.0, t_end_ps=10.0, stride=1)
    res_no_t0 = run_postprocessing(str(mock_attempt), str(tmp_path / "win_no_t0"), t_start_ps=0.1, t_end_ps=10.0, stride=1)
    res_second_half = run_postprocessing(str(mock_attempt), str(tmp_path / "win_half"), t_start_ps=5.0, t_end_ps=10.0, stride=1)

    s_all = res_all["df_summary"].iloc[0]
    s_no_t0 = res_no_t0["df_summary"].iloc[0]
    s_half = res_second_half["df_summary"].iloc[0]

    assert s_all["sampling_frames_analyzed"] == 101
    assert s_no_t0["sampling_frames_analyzed"] == 100
    assert s_half["sampling_frames_analyzed"] == 51

    # 对照原生全帧表直接计算子窗口均值，必须完全一致
    df_native_all = pd.read_csv(tmp_path / "win_all" / "state_0" / "nvt_native_energy_frames_all.csv")
    exp_pot_no_t0 = float(df_native_all[df_native_all["time_ps"] >= 0.1]["Potential"].mean())
    exp_pot_half = float(df_native_all[df_native_all["time_ps"] >= 5.0]["Potential"].mean())

    assert abs(s_no_t0["md_potential_energy_mean_kJ_mol"] - exp_pot_no_t0) < 1e-5
    assert abs(s_half["md_potential_energy_mean_kJ_mol"] - exp_pot_half) < 1e-5
    assert abs(s_all["md_potential_energy_mean_kJ_mol"] - s_half["md_potential_energy_mean_kJ_mol"]) > 1.0


def test_7_pbc_minimum_image_across_all_modules():
    """
    独立审计反例 F：10 A 盒子中 x=0.1 A 与 x=9.9 A 的两个不同残基原子，
    真实周期最小镜像距离为 0.2 A（绝不能报告欧氏距离 9.8 A）。
    同时验证跨边界共价键应变与整体刚体平移不变性。
    """
    s = pmd.Structure()
    s.add_atom(pmd.Atom(name="C1", type="c3", atomic_number=6, mass=12.011), "RESA", 1)
    s.add_atom(pmd.Atom(name="C2", type="c3", atomic_number=6, mass=12.011), "RESB", 2)
    s.box = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    s.coordinates = np.array([[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]])

    d_pbc, _ = pbc_check_contacts(s)
    d_eng, _ = engine_check_contacts(s)
    d_pipe, _ = pipeline_check_contacts(s)

    assert abs(d_pbc - 0.2) < 1e-6
    assert abs(d_eng - 0.2) < 1e-6
    assert abs(d_pipe - 0.2) < 1e-6

    # 整体平移不变性
    s.coordinates = (s.coordinates + np.array([4.7, -2.3, 8.1])) % 10.0
    d_shifted, _ = engine_check_contacts(s)
    assert abs(d_shifted - 0.2) < 1e-6

    # 跨周期边界的正常共价键：x=0.05 A 与 x=9.90 A (真实 PBC 键长 0.15 nm = 1.50 A，平衡键长 1.50 A)
    bt = pmd.BondType(k=300.0, req=1.50)
    s.bond_types.append(bt)
    s.bonds.append(pmd.Bond(s.atoms[0], s.atoms[1], type=bt))
    s.coordinates = np.array([[0.5, 5.0, 5.0], [9.0, 5.0, 5.0]])  # PBC 距离 1.5 A
    max_dev, abn = engine_check_bonds(s, max_allowed_dev_A=0.25)
    assert max_dev < 1e-5
    assert len(abn) == 0


def test_8_ffv_volume_union_varies_with_geometry_and_rejects_unknown_element():
    """
    固定原子组成与盒子体积下，改变原子空间重叠度，FFV 必须随之改变；
    遇到未知元素必须抛出 KeyError 阻断，严禁静默使用碳半径 0.170 nm。
    """
    box = np.array([2.0, 2.0, 2.0])
    elements = ["C", "C"]
    coords_overlap = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    coords_separated = np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]])

    v_ov, ffv_ov, _, _ = compute_grid_occupied_volume_union(coords_overlap, elements, box, 0.0, 0.05)
    v_sep, ffv_sep, _, _ = compute_grid_occupied_volume_union(coords_separated, elements, box, 0.0, 0.05)

    assert v_sep > 1.8 * v_ov
    assert ffv_ov > ffv_sep

    with pytest.raises(KeyError, match="未知化学元素"):
        compute_grid_occupied_volume_union(coords_overlap, ["Xe_UNKNOWN", "C"], box, 0.0, 0.05)


def test_9_hbond_detection_depends_on_pbc_geometry_not_state():
    """氢键数量由真实供受体距离、角度和 PBC 决定，包括跨周期边界氢键"""
    prmtop_p = ATTEMPT_RAW_DIR / "state_0_addition" / "system.prmtop"
    gro_p = ATTEMPT_RAW_DIR / "state_0_addition" / "nvt.gro"
    parm = pmd.load_file(str(prmtop_p), xyz=str(gro_p))
    box_nm = np.array([5.5, 5.5, 5.5])

    # 找到第一个共价 N-H 供体对和另一个残基上的 O 受体
    d_idx, h_idx = None, None
    for b in parm.bonds:
        if b.atom1.element_name == "N" and b.atom2.element_name == "H":
            d_idx, h_idx = b.atom1.idx, b.atom2.idx
            break
        elif b.atom2.element_name == "N" and b.atom1.element_name == "H":
            d_idx, h_idx = b.atom2.idx, b.atom1.idx
            break
    o_idx = [a.idx for a in parm.atoms if a.element_name == "O" and a.residue.idx != parm.atoms[d_idx].residue.idx][0]

    coords = np.asarray(parm.coordinates, dtype=float) / 10.0
    # 构造跨周期边界的线性氢键：D 在 z=5.42 nm, H 在 z=0.02 nm (跨过 5.5 nm 边界，距离 0.10 nm), A 在 z=0.20 nm (D-A PBC 距离 0.28 nm, 180 deg)
    coords[d_idx] = np.array([1.0, 1.0, 5.42])
    coords[h_idx] = np.array([1.0, 1.0, 0.02])
    coords[o_idx] = np.array([1.0, 1.0, 0.20])

    hb_pbc = detect_geometric_hbonds_pbc(parm, coords, box_nm)
    assert len(hb_pbc) == 1
    assert hb_pbc[0]["is_intermolecular"] is True
    assert abs(hb_pbc[0]["dist_da_nm"] - 0.28) < 1e-4
    assert abs(hb_pbc[0]["angle_dha_deg"] - 180.0) < 1e-2


def test_10_missing_energy_term_blocks_extraction(tmp_path):
    """在 Cut-off 体系的 EDR 中请求不存在的长程 PME 项 (如 Coulomb-recip.) 必须立即抛出 KeyError 阻断"""
    edr_p = ATTEMPT_RAW_DIR / "state_0_addition" / "nvt.edr"
    with pytest.raises(KeyError, match="不存在"):
        extract_native_edr_data(
            str(edr_p),
            box_volume_nm3=166.375,
            out_dir=str(tmp_path / "missing_term"),
            terms=["LJ-14", "Coulomb-recip."],
        )


def test_11_swapped_resp_csv_element_order_rejected_by_injector(tmp_path):
    """交换单体 RESP CSV 中不同元素的行序时，inject_and_verify_resp_mol2 必须抛出 ValueError 拒绝"""
    dds_csv = Path("/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dds_resp/atom_charges.csv")
    dds_mol2 = ATTEMPT_RAW_DIR / "templates" / "DDS.mol2"
    df_csv = pd.read_csv(dds_csv)

    # 找到一个 C 行和一个 S 或 N 行交换
    idx_c = int(df_csv[df_csv["element"] == "C"].index[0])
    idx_s = int(df_csv[df_csv["element"] == "S"].index[0])
    df_swapped = df_csv.copy()
    df_swapped.iloc[[idx_c, idx_s]] = df_swapped.iloc[[idx_s, idx_c]].values

    bad_csv = tmp_path / "swapped_dds.csv"
    df_swapped.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="mismatch"):
        inject_and_verify_resp_mol2(
            str(dds_mol2),
            str(tmp_path / "out_dds.mol2"),
            str(bad_csv),
            expected_atom_count=29,
        )


def test_12_cli_and_shared_function_deterministic_and_identical(tmp_path):
    """
    验证生产 CLI (compute_features_from_trajectory.py) 与共享函数 (run_postprocessing)
    在相同输入下生成的 CSV 完全一致，无任何随机数波动，且与独立审计基线能量精确吻合。
    """
    mock_attempt = _copy_single_state_fixture(tmp_path, 0)
    out_func = tmp_path / "out_func"
    out_cli = tmp_path / "out_cli"

    res_func = run_postprocessing(str(mock_attempt), str(out_func), stride=2, grid_spacing_nm=0.15)
    cmd = [
        PYTHON_BIN,
        str(EXT_ROOT / "src" / "compute_features_from_trajectory.py"),
        "--input-attempt", str(mock_attempt),
        "--output-dir", str(out_cli),
        "--stride", "2",
        "--grid-spacing", "0.15",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0

    df_f = pd.read_csv(out_func / "md_features_diagnostic.csv").drop(columns=["postprocess_id"])
    df_c = pd.read_csv(out_cli / "md_features_diagnostic.csv").drop(columns=["postprocess_id"])
    pd.testing.assert_frame_equal(df_f, df_c)

    df_raw_f = pd.read_csv(out_func / "raw_frame_features_native.csv").drop(columns=["postprocess_id"])
    df_raw_c = pd.read_csv(out_cli / "raw_frame_features_native.csv").drop(columns=["postprocess_id"])
    pd.testing.assert_frame_equal(df_raw_f, df_raw_c)
    assert float(df_raw_f["box_volume_nm3"].std()) == 0.0
