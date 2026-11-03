"""
run_verified_md_pipeline.py
CFRP 交联网络 MD 特征管线实现（已彻底剔除一切硬编码能量公式、随机数伪造帧表与非 PBC 距离函数）
核心准则：
1. 接触检查与共价键长检查直接复用唯一共享的 pbc_utils / assembly_engine 实现（严格 PBC 最小镜像）；
2. 单体模板与加成物模板制备直接复用 assembly_engine，如实标记未反应单体 DFT/RESP 与加成物 Gasteiger (-c gas) 混合诊断协议；
3. 特征提取与后处理直接调用唯一共享的 postprocess_trajectory.run_postprocessing，绝不硬编码 -15.2*(st+1)、-8.5*(st+1) 或使用 np.random.normal 生成逐帧数据；
4. 生命周期表基于 State 0 真实拓扑连接图识别原子角色，跨状态未建立单原子双射时如实标记 NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY。
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import parmed as pmd
import networkx as nx

try:
    from .pbc_utils import (
        check_inter_residue_contacts,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains,
        check_system_bond_strains_detailed,
    )
    from .assembly_engine import (
        AMBER_BIN,
        GMX_BIN,
        DGE_CSV,
        DDS_CSV,
        prepare_resp_monomer_templates,
        prepare_diagnostic_adduct_templates,
    )
    from .postprocess_trajectory import run_postprocessing
except (ImportError, ValueError):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from src.pbc_utils import (
        check_inter_residue_contacts,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains,
        check_system_bond_strains_detailed,
    )
    from src.assembly_engine import (
        AMBER_BIN,
        GMX_BIN,
        DGE_CSV,
        DDS_CSV,
        prepare_resp_monomer_templates,
        prepare_diagnostic_adduct_templates,
    )
    from src.postprocess_trajectory import run_postprocessing


def build_and_equilibrate_state(state_num: int, templates_dir: str, state_dir: str) -> Dict[str, Any]:
    os.makedirs(state_dir, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GMX_BIN)}:{AMBER_BIN}:{env.get('PATH', '')}"

    dge_mol2 = os.path.join(templates_dir, "DGE.mol2")
    dge_frcmod = os.path.join(templates_dir, "DGE.frcmod")
    dds_mol2 = os.path.join(templates_dir, "DDS.mol2")
    dds_frcmod = os.path.join(templates_dir, "DDS.frcmod")

    leap_in = os.path.join(state_dir, "leap.in")
    prmtop = os.path.join(state_dir, "system.prmtop")
    inpcrd = os.path.join(state_dir, "system.inpcrd")

    with open(leap_in, "w", encoding="utf-8") as f:
        f.write("source leaprc.gaff2\n")
        f.write(f"loadamberparams {dge_frcmod}\n")
        f.write(f"loadamberparams {dds_frcmod}\n")
        f.write(f"DGE = loadmol2 {dge_mol2}\n")
        f.write(f"DDS = loadmol2 {dds_mol2}\n")

        if state_num == 0:
            f.write("d1 = copy DGE\nd2 = copy DGE\nd3 = copy DGE\nd4 = copy DGE\n")
            f.write("s1 = copy DDS\ns2 = copy DDS\n")
            f.write("translate d1 { 12.0 12.0 12.0 }\n")
            f.write("translate d2 { 38.0 12.0 12.0 }\n")
            f.write("translate d3 { 12.0 38.0 12.0 }\n")
            f.write("translate d4 { 38.0 38.0 12.0 }\n")
            f.write("translate s1 { 12.0 12.0 38.0 }\n")
            f.write("translate s2 { 38.0 38.0 38.0 }\n")
            f.write("sys = combine { d1 d2 d3 d4 s1 s2 }\n")
            f.write("set sys box { 55.0 55.0 55.0 }\n")
            f.write(f"saveamberparm sys {prmtop} {inpcrd}\nquit\n")
        elif state_num == 1:
            mono_mol2 = os.path.join(templates_dir, "MON.mol2")
            mono_frcmod = os.path.join(templates_dir, "MON.frcmod")
            f.write(f"loadamberparams {mono_frcmod}\n")
            f.write(f"MON = loadmol2 {mono_mol2}\n")
            f.write("m1 = copy MON\n")
            f.write("d1 = copy DGE\nd2 = copy DGE\nd3 = copy DGE\n")
            f.write("s1 = copy DDS\n")
            f.write("translate m1 { 15.0 15.0 15.0 }\n")
            f.write("translate d1 { 42.0 15.0 15.0 }\n")
            f.write("translate d2 { 15.0 42.0 15.0 }\n")
            f.write("translate d3 { 42.0 42.0 15.0 }\n")
            f.write("translate s1 { 15.0 15.0 42.0 }\n")
            f.write("sys = combine { m1 d1 d2 d3 s1 }\n")
            f.write("set sys box { 60.0 60.0 60.0 }\n")
            f.write(f"saveamberparm sys {prmtop} {inpcrd}\nquit\n")
        elif state_num == 2:
            bis_mol2 = os.path.join(templates_dir, "BIS.mol2")
            bis_frcmod = os.path.join(templates_dir, "BIS.frcmod")
            f.write(f"loadamberparams {bis_frcmod}\n")
            f.write(f"BIS = loadmol2 {bis_mol2}\n")
            f.write("b1 = copy BIS\n")
            f.write("d1 = copy DGE\nd2 = copy DGE\n")
            f.write("s1 = copy DDS\n")
            f.write("translate b1 { 15.0 15.0 15.0 }\n")
            f.write("translate d1 { 55.0 15.0 15.0 }\n")
            f.write("translate d2 { 15.0 55.0 15.0 }\n")
            f.write("translate s1 { 55.0 55.0 55.0 }\n")
            f.write("sys = combine { b1 d1 d2 s1 }\n")
            f.write("set sys box { 75.0 75.0 75.0 }\n")
            f.write(f"saveamberparm sys {prmtop} {inpcrd}\nquit\n")

    subprocess.run([os.path.join(AMBER_BIN, "tleap"), "-f", leap_in], cwd=state_dir, check=True, capture_output=True)

    top_out = os.path.join(state_dir, "system.top")
    gro_out = os.path.join(state_dir, "system.gro")

    parm = pmd.load_file(prmtop, inpcrd)
    assert len(parm.atoms) == 254, f"Atom count mismatch: {len(parm.atoms)} != 254"

    min_inter_dist, clashing_pair = check_inter_residue_contacts(parm, min_allowed_dist_A=3.0)
    assert min_inter_dist >= 3.0, f"Initial PBC contact clash in State {state_num}: {min_inter_dist} < 3.0 A"

    max_bond_dev, abnormal_bonds = check_system_bond_strains(parm, max_allowed_dev_A=0.25)
    assert max_bond_dev <= 0.25, f"Initial abnormal bond strain in State {state_num}: {max_bond_dev} A"

    q_non_zero = sum(1 for a in parm.atoms if abs(a.charge) > 1e-4)
    net_q = sum(a.charge for a in parm.atoms)
    if state_num == 0:
        assert q_non_zero == 254, f"State 0 must have 100% non-zero RESP charges! Got {q_non_zero}"
        assert abs(net_q) < 1e-4, f"State 0 net charge must be 0! Got {net_q}"

    parm.save(top_out, overwrite=True)
    parm.save(gro_out, overwrite=True)

    mdp_min = os.path.join(state_dir, "min.mdp")
    with open(mdp_min, "w", encoding="utf-8") as f:
        f.write("""integrator  = steep
emtol       = 200.0
emstep      = 0.01
nsteps      = 5000
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
""")
    tpr_min = os.path.join(state_dir, "min.tpr")
    log_min = os.path.join(state_dir, "min.log")
    gro_min = os.path.join(state_dir, "min.gro")

    subprocess.run([GMX_BIN, "grompp", "-f", mdp_min, "-c", gro_out, "-p", top_out, "-o", tpr_min, "-maxwarn", "2"], cwd=state_dir, check=True, capture_output=True)
    subprocess.run([GMX_BIN, "mdrun", "-deffnm", "min", "-nt", "4"], cwd=state_dir, check=True, capture_output=True)

    gro_min_obj = pmd.load_file(gro_min)
    coords_min = gro_min_obj.coordinates
    box_min = gro_min_obj.box[:3]
    max_dev_min, _ = check_system_bond_strains(parm, coords=coords_min, max_allowed_dev_A=0.25, box=box_min)
    assert max_dev_min <= 0.25, f"Minimization produced abnormal stretched bonds! Max dev: {max_dev_min:.4f} A"

    mdp_nvt = os.path.join(state_dir, "nvt.mdp")
    with open(mdp_nvt, "w", encoding="utf-8") as f:
        f.write("""integrator  = md
dt          = 0.001
nsteps      = 10000
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = Cut-off
tcoupl      = v-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = 300
gen-vel     = yes
gen-temp    = 300
gen-seed    = 20261019
nstxout-compressed = 100
nstenergy   = 100
pbc         = xyz
""")
    tpr_nvt = os.path.join(state_dir, "nvt.tpr")
    gro_nvt = os.path.join(state_dir, "nvt.gro")
    edr_nvt = os.path.join(state_dir, "nvt.edr")
    log_nvt = os.path.join(state_dir, "nvt.log")
    xtc_nvt = os.path.join(state_dir, "nvt.xtc")

    subprocess.run([GMX_BIN, "grompp", "-f", mdp_nvt, "-c", gro_min, "-p", top_out, "-o", tpr_nvt, "-maxwarn", "2"], cwd=state_dir, check=True, capture_output=True)
    subprocess.run([GMX_BIN, "mdrun", "-deffnm", "nvt", "-nt", "4"], cwd=state_dir, check=True, capture_output=True)

    gro_nvt_obj = pmd.load_file(gro_nvt)
    max_dev_nvt, _ = check_system_bond_strains(parm, coords=gro_nvt_obj.coordinates, max_allowed_dev_A=0.25, box=gro_nvt_obj.box[:3])
    assert max_dev_nvt <= 0.25, f"NVT produced abnormal stretched bonds! Max dev: {max_dev_nvt:.4f} A"

    return {
        "state": state_num,
        "prmtop": prmtop,
        "top": top_out,
        "gro": gro_nvt,
        "xtc": xtc_nvt,
        "edr": edr_nvt,
        "tpr": tpr_nvt,
        "log_min": log_min,
        "log_nvt": log_nvt,
        "min_dist_init_A": min_inter_dist,
        "max_bond_dev_nvt_A": max_dev_nvt,
        "q_non_zero": q_non_zero,
        "net_q": net_q,
    }


def generate_verified_atom_lifecycle_csv(top_path: str, out_csv_path: str) -> pd.DataFrame:
    """
    从 State 0 真实拓扑文件 (system.prmtop / system.top) 的共价连接图构建原子身份表：
    1. 精确读取 State 0 全部 254 个原子的元素、类型、电荷与残基；
    2. 从真实分子连接图自动识别三元环氧环原子 (CLOSED_OXIRANE_RING_ATOM) 与伯胺基团原子 (PRIMARY_AMINE_GROUP_ATOM)，不依赖原子名硬编码；
    3. 由于 State 1 与 State 2 是从独立 SMILES 重新生成的预设加成物拓扑，未建立跨状态单原子双射，
       因此 cross_state_bijection_verified 严格设为 False，state_1_status / state_2_status 如实标记为
       NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY，绝不靠残基编号猜测具体哪个原子发生了反应！
    """
    parm = pmd.load_file(top_path)
    if len(parm.atoms) != 254:
        raise ValueError(f"Atom count != 254: {len(parm.atoms)}")

    G = nx.Graph()
    for a in parm.atoms:
        G.add_node(a.idx)
    for b in parm.bonds:
        G.add_edge(b.atom1.idx, b.atom2.idx)

    # 通过真实连接图识别三元环氧环中的 O, C, C 原子及相连 H
    oxirane_core_atoms = set()
    for a in parm.atoms:
        if a.element_name == "O":
            c_nbrs = [n for n in G.neighbors(a.idx) if parm.atoms[n].element_name == "C"]
            if len(c_nbrs) == 2 and G.has_edge(c_nbrs[0], c_nbrs[1]):
                oxirane_core_atoms.add(a.idx)
                oxirane_core_atoms.update(c_nbrs)

    # 通过真实连接图识别伯胺 -NH2 中的 N 与 H 原子
    primary_amine_atoms = set()
    for a in parm.atoms:
        if a.element_name == "N":
            h_nbrs = [n for n in G.neighbors(a.idx) if parm.atoms[n].element_name == "H"]
            if len(h_nbrs) == 2:
                primary_amine_atoms.add(a.idx)
                primary_amine_atoms.update(h_nbrs)

    rows = []
    for idx, atom in enumerate(parm.atoms):
        global_id = idx + 1
        res_name = atom.residue.name
        res_idx = atom.residue.idx + 1
        elem = atom.element_name

        if res_name == "DGE":
            monomer_name = "DGEBA"
            s0 = "CLOSED_OXIRANE" if idx in oxirane_core_atoms else "BACKBONE_CONSERVED"
        elif res_name == "DDS":
            monomer_name = "DDS"
            s0 = "PRIMARY_AMINE" if idx in primary_amine_atoms else "BACKBONE_CONSERVED"
        else:
            monomer_name = res_name
            s0 = "CONSERVED"

        rows.append({
            "global_atom_id": global_id,
            "origin_atom_id": f"{monomer_name}_{res_idx}_{atom.name}",
            "monomer_name": monomer_name,
            "residue_index": res_idx,
            "atom_name": atom.name,
            "atom_type": atom.type,
            "element": elem,
            "charge_e": round(float(atom.charge), 6),
            "state_0_status": s0,
            "state_1_status": "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY",
            "state_2_status": "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY",
            "cross_state_bijection_verified": False,
            "conserved_in_254_system": 1,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    return df


def extract_features_for_state(state_info: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """
    兼容接口：严禁任何硬编码能量或球体体积求和，直接调用共享的 postprocess_trajectory.run_postprocessing。
    """
    st_num = int(state_info["state"])
    gro_p = Path(state_info["gro"]).resolve()
    attempt_dir = gro_p.parent.parent
    res = run_postprocessing(
        input_attempt_dir=str(attempt_dir),
        output_dir=str(Path(out_dir).resolve()),
    )
    df_sum = res["df_summary"]
    matched = df_sum[df_sum["preset_fixture_state_index"] == st_num]
    if matched.empty:
        raise ValueError(f"State {st_num} not found in postprocessing output")
    return matched.iloc[0].to_dict()


def main():
    """
    默认后处理入口：对已存在的 attempt_20261019_01 执行真实后处理并写入新独立目录，不重跑 MD 也不覆盖旧现场。
    """
    ext_root = Path("/home/wangj/project/CFRP/extensions/network_md_features")
    input_attempt = ext_root / "artifacts" / "attempt_20261019_01"
    output_dir = ext_root / "artifacts" / "postprocess_verified_20261019_02"
    run_postprocessing(
        input_attempt_dir=str(input_attempt),
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
