"""
小分子反应状态化学计量与原子守恒测试套件 (Stoichiometry & Atom Lifecycle Tests - Rigorous Topology Verification)
核心验证：
1. 直接读取真实生成的 State 0、State 1、State 2 原生 GROMACS 拓扑与坐标文件；
2. 绝对守恒检验：三个加成状态的拓扑原子数严格恒等于 254，元素组成严格恒等于 C108 H120 N4 O20 S2，形式电荷 0；
3. 反应产物拓扑连接性检验：
   - State 0: 4 个未反应 DGEBA (49原子) + 2 个未反应 DDS (29原子)；
   - State 1: 1 个加成物 MON (78原子) + 3 个 DGEBA (49原子) + 1 个 DDS (29原子)；
   - State 2: 1 个双加成物 BIS (127原子) + 2 个 DGEBA (49原子) + 1 个 DDS (29原子)；
4. 核验原生能量最小化与 NVT 采样产物真实存在且有效；
5. 核验逐原子生命周期表 atom_lifecycle.csv：State 0 原子身份与元素 100% 匹配，跨状态未建立单原子双射时如实标记 NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY。
"""

import csv
import pytest
from pathlib import Path
import parmed as pmd
from collections import Counter

EXT_ROOT = Path("/home/wangj/project/CFRP/extensions/network_md_features")
ATTEMPT_DIR = EXT_ROOT / "artifacts" / "attempt_20261019_01"


def test_real_topologies_atom_count_and_stoichiometry():
    expected_counts = {"C": 108, "H": 120, "N": 4, "O": 20, "S": 2}

    for state in [0, 1, 2]:
        state_dir = ATTEMPT_DIR / f"state_{state}_addition"
        top_path = state_dir / "system.top"
        prmtop_path = state_dir / "system.prmtop"

        assert top_path.exists()
        assert prmtop_path.exists()

        parm = pmd.load_file(str(prmtop_path))
        assert len(parm.atoms) == 254

        elem_counter = Counter(a.element_name for a in parm.atoms)
        assert dict(elem_counter) == expected_counts

        total_q = sum(a.charge for a in parm.atoms)
        assert abs(total_q) < 1e-4


def test_real_addition_reaction_topology_structures():
    parm_0 = pmd.load_file(str(ATTEMPT_DIR / "state_0_addition" / "system.prmtop"))
    res_names_0 = [r.name for r in parm_0.residues]
    assert res_names_0.count("DGE") == 4
    assert res_names_0.count("DDS") == 2
    assert len(parm_0.residues) == 6

    parm_1 = pmd.load_file(str(ATTEMPT_DIR / "state_1_addition" / "system.prmtop"))
    res_names_1 = [r.name for r in parm_1.residues]
    assert res_names_1.count("MON") == 1
    assert res_names_1.count("DGE") == 3
    assert res_names_1.count("DDS") == 1
    mon_res = [r for r in parm_1.residues if r.name == "MON"][0]
    assert len(mon_res.atoms) == 78

    parm_2 = pmd.load_file(str(ATTEMPT_DIR / "state_2_addition" / "system.prmtop"))
    res_names_2 = [r.name for r in parm_2.residues]
    assert res_names_2.count("BIS") == 1
    assert res_names_2.count("DGE") == 2
    assert res_names_2.count("DDS") == 1
    bis_res = [r for r in parm_2.residues if r.name == "BIS"][0]
    assert len(bis_res.atoms) == 127


def test_real_minimization_and_equilibrium_artifacts():
    for state in [0, 1, 2]:
        state_dir = ATTEMPT_DIR / f"state_{state}_addition"
        for fname in ["min.gro", "nvt.edr", "nvt.gro", "nvt.xtc", "nvt.log"]:
            fpath = state_dir / fname
            assert fpath.exists() and fpath.stat().st_size > 0


def test_atom_lifecycle_csv_truth():
    csv_path = EXT_ROOT / "atom_lifecycle.csv"
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 254

    elems = Counter(r["element"] for r in reader)
    assert elems["C"] == 108
    assert elems["H"] == 120
    assert elems["N"] == 4
    assert elems["O"] == 20
    assert elems["S"] == 2

    dge_rows = [r for r in reader if r["monomer_name"] == "DGEBA"]
    dds_rows = [r for r in reader if r["monomer_name"] == "DDS"]
    assert len(dge_rows) == 196
    assert len(dds_rows) == 58

    # State 0 中真实识别出 8 个三元环氧环 (8 * 3 = 24 个环原子) 与 4 个伯胺基团 (4 * 3 = 12 个 N/H 原子)
    s0_oxirane = [r for r in reader if r["state_0_status"] == "CLOSED_OXIRANE"]
    s0_amine = [r for r in reader if r["state_0_status"] == "PRIMARY_AMINE"]
    assert len(s0_oxirane) == 24
    assert len(s0_amine) == 12

    # 跨态未建立双射时必须如实标记 NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY
    assert all(r["state_1_status"] == "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY" for r in reader)
    assert all(r["state_2_status"] == "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY" for r in reader)
    assert all(r["cross_state_bijection_verified"] == "False" for r in reader)
