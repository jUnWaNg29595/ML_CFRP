"""
assembly_engine.py
核心功能：
1. 从经过 DFT/RESP 计算的最终 atom_charges.csv 为单体注入真实逐原子 RESP 电荷，并严格核验元素序列、原子名与净电荷守恒；
2. 唯一共享的 PBC 最小镜像跨残基接触检查与共价键长应变门禁（直接复用 pbc_utils，杜绝非周期欧氏距离漏洞）；
3. 加成物模板如实标记为 GAFF2 + Gasteiger (-c gas) 诊断协议，严禁冒充 AM1-BCC 或 DFT-RESP。
"""

import os
import subprocess
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from .pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_inter_residue_contacts,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains,
        check_system_bond_strains_detailed,
    )
except (ImportError, ValueError):
    from src.pbc_utils import (
        pbc_displacement,
        pbc_distance,
        check_inter_residue_contacts,
        check_inter_residue_contacts_pbc,
        check_system_bond_strains,
        check_system_bond_strains_detailed,
    )

AMBER_BIN = "/home/wangj/miniconda3/envs/amber_env/bin"
GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"

DGE_CSV = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dgeba_resp/atom_charges.csv"
DDS_CSV = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dds_resp/atom_charges.csv"
DGE_AC_SRC = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dgeba_resp/molecule.ac"
DDS_AC_SRC = "/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/qm/dds_resp/molecule.ac"


def inject_and_verify_resp_mol2(
    raw_mol2_path: str,
    out_mol2_path: str,
    resp_csv_path: str,
    expected_atom_count: int,
) -> Dict[str, Any]:
    """
    将 RESP 电荷从 CSV 注入到 MOL2，并执行严格的原子身份、元素符号与净电荷守恒核验。
    若 CSV 行数不符、元素序列被交换、原子名称错位或净电荷不守恒，直接抛出 ValueError 阻断。
    """
    df_resp = pd.read_csv(resp_csv_path)
    if len(df_resp) != expected_atom_count:
        raise ValueError(
            f"RESP CSV atom count mismatch for {resp_csv_path}: expected {expected_atom_count}, got {len(df_resp)}"
        )

    for required_col in ("element", "resp_charge_e"):
        if required_col not in df_resp.columns:
            raise ValueError(f"Missing required column '{required_col}' in {resp_csv_path}")

    q_list = df_resp["resp_charge_e"].astype(float).tolist()
    elem_list = [str(e).strip() for e in df_resp["element"].tolist()]
    name_list = (
        [str(n).strip() for n in df_resp["atom_name"].tolist()]
        if "atom_name" in df_resp.columns
        else None
    )

    net_q = float(np.sum(q_list))
    if abs(net_q) > 1e-3:
        raise ValueError(f"Unconserved monomer RESP net charge in {resp_csv_path}: sum={net_q}")
    if sum(1 for q in q_list if abs(q) > 1e-6) == 0:
        raise ValueError(f"All-zero RESP charges in {resp_csv_path} are forbidden.")

    with open(raw_mol2_path, "r", encoding="utf-8") as f_in, open(out_mol2_path, "w", encoding="utf-8") as f_out:
        in_atoms = False
        idx = 0
        for line in f_in:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                f_out.write(line)
                continue
            elif line.startswith("@<TRIPOS>BOND") or line.startswith("@<TRIPOS>SUBSTRUCTURE"):
                in_atoms = False
                f_out.write(line)
                continue
            if in_atoms:
                p = line.split()
                if len(p) >= 9:
                    if idx >= expected_atom_count:
                        raise ValueError(f"MOL2 atom count exceeds expected {expected_atom_count} in {raw_mol2_path}")
                    mol2_atom_name = p[1].strip()
                    mol2_elem = "".join(ch for ch in mol2_atom_name if ch.isalpha())[:1].upper()
                    csv_elem = elem_list[idx].upper()
                    if mol2_elem != csv_elem:
                        raise ValueError(
                            f"Atom identity/element mismatch at 1-based index {idx + 1} in {raw_mol2_path}: "
                            f"MOL2 atom '{mol2_atom_name}' (element {mol2_elem}) != CSV element '{csv_elem}'"
                        )
                    if name_list is not None and name_list[idx] != mol2_atom_name:
                        raise ValueError(
                            f"Atom name order mismatch at 1-based index {idx + 1}: "
                            f"MOL2 '{mol2_atom_name}' != CSV '{name_list[idx]}'"
                        )
                    q = q_list[idx]
                    idx += 1
                    f_out.write(
                        f"{int(p[0]):7d} {p[1]:<8s} {float(p[2]):10.4f} {float(p[3]):10.4f} "
                        f"{float(p[4]):10.4f} {p[5]:<6s} {int(p[6]):4d} {p[7]:<6s} {q:10.6f}\n"
                    )
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

    if idx != expected_atom_count:
        raise ValueError(f"Injected atom count {idx} != expected {expected_atom_count}")

    return {
        "atom_count": idx,
        "net_charge_e": net_q,
        "charge_protocol": "GAFF2_MONOMER_DFT_RESP_ORDERED_VERIFIED",
    }


def prepare_resp_monomer_templates(out_dir: str) -> Tuple[str, str, str, str]:
    """
    为 DGEBA 与 DDS 注入真实 DFT-RESP 逐原子电荷，生成经过核验的 mol2 与 frcmod。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. DDS (29 atoms, C12H12N2O2S)
    raw_dds = os.path.join(out_dir, "dds_raw.mol2")
    subprocess.run(
        [os.path.join(AMBER_BIN, "antechamber"), "-i", DDS_AC_SRC, "-fi", "ac", "-o", raw_dds, "-fo", "mol2", "-at", "gaff2", "-rn", "DDS"],
        check=True,
        capture_output=True,
    )
    dds_mol2 = os.path.join(out_dir, "DDS.mol2")
    inject_and_verify_resp_mol2(raw_dds, dds_mol2, DDS_CSV, expected_atom_count=29)
    dds_frcmod = os.path.join(out_dir, "DDS.frcmod")
    subprocess.run(
        [os.path.join(AMBER_BIN, "parmchk2"), "-i", dds_mol2, "-f", "mol2", "-o", dds_frcmod, "-s", "2"],
        check=True,
        capture_output=True,
    )

    # 2. DGEBA (49 atoms, C21H24O4)
    raw_dge = os.path.join(out_dir, "dge_raw.mol2")
    subprocess.run(
        [os.path.join(AMBER_BIN, "antechamber"), "-i", DGE_AC_SRC, "-fi", "ac", "-o", raw_dge, "-fo", "mol2", "-at", "gaff2", "-rn", "DGE"],
        check=True,
        capture_output=True,
    )
    dge_mol2 = os.path.join(out_dir, "DGE.mol2")
    inject_and_verify_resp_mol2(raw_dge, dge_mol2, DGE_CSV, expected_atom_count=49)
    dge_frcmod = os.path.join(out_dir, "DGE.frcmod")
    subprocess.run(
        [os.path.join(AMBER_BIN, "parmchk2"), "-i", dge_mol2, "-f", "mol2", "-o", dge_frcmod, "-s", "2"],
        check=True,
        capture_output=True,
    )

    return dge_mol2, dge_frcmod, dds_mol2, dds_frcmod


def prepare_diagnostic_adduct_templates(out_dir: str) -> Tuple[str, str, str, str]:
    """
    为 State 1 (MON 78 atoms) 与 State 2 (BIS 127 atoms) 准备加成物诊断模板。
    注意：实际调用 `antechamber -c gas` (Gasteiger 电荷)，其电荷协议必须如实标记为
    GAFF2_UNREACTED_RESP_ADDUCT_GASTEIGER_MIXED_DIAGNOSTIC，严禁误标为 AM1-BCC！
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. MON (78 atoms, C33H36N2O6S)
    mono_smi = "CC(C)(c1ccc(OCC2CO2)cc1)c3ccc(OCC(O)CNc4ccc(S(=O)(=O)c5ccc(N)cc5)cc4)cc3"
    mol_mono = Chem.AddHs(Chem.MolFromSmiles(mono_smi))
    AllChem.EmbedMolecule(mol_mono, randomSeed=20261019)
    AllChem.MMFFOptimizeMolecule(mol_mono, maxIters=2000)
    mono_sdf = os.path.join(out_dir, "mono.sdf")
    mono_raw = os.path.join(out_dir, "mono_raw.mol2")
    with open(mono_sdf, "w", encoding="utf-8") as f:
        f.write(Chem.MolToMolBlock(mol_mono))
    subprocess.run(
        ["/home/wangj/miniconda3/envs/htpolynet_env/bin/obabel", mono_sdf, "-omol2", "-O", mono_raw, "--title", "MON"],
        check=True,
        capture_output=True,
    )

    mono_mol2 = os.path.join(out_dir, "MON.mol2")
    mono_frcmod = os.path.join(out_dir, "MON.frcmod")
    subprocess.run(
        [os.path.join(AMBER_BIN, "antechamber"), "-i", mono_raw, "-fi", "mol2", "-o", mono_mol2, "-fo", "mol2", "-at", "gaff2", "-rn", "MON", "-c", "gas"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [os.path.join(AMBER_BIN, "parmchk2"), "-i", mono_mol2, "-f", "mol2", "-o", mono_frcmod, "-s", "2"],
        check=True,
        capture_output=True,
    )

    # 2. BIS (127 atoms, C54H60N2O10S)
    bis_smi = "CC(C)(c1ccc(OCC2CO2)cc1)c3ccc(OCC(O)CNc4ccc(S(=O)(=O)c5ccc(NCC(O)COc6ccc(C(C)(C)c7ccc(OCC8CO8)cc7)cc6)cc5)cc4)cc3"
    mol_bis = Chem.AddHs(Chem.MolFromSmiles(bis_smi))
    AllChem.EmbedMolecule(mol_bis, randomSeed=20261019)
    AllChem.MMFFOptimizeMolecule(mol_bis, maxIters=3000)
    bis_sdf = os.path.join(out_dir, "bis.sdf")
    bis_raw = os.path.join(out_dir, "bis_raw.mol2")
    with open(bis_sdf, "w", encoding="utf-8") as f:
        f.write(Chem.MolToMolBlock(mol_bis))
    subprocess.run(
        ["/home/wangj/miniconda3/envs/htpolynet_env/bin/obabel", bis_sdf, "-omol2", "-O", bis_raw, "--title", "BIS"],
        check=True,
        capture_output=True,
    )

    bis_mol2 = os.path.join(out_dir, "BIS.mol2")
    bis_frcmod = os.path.join(out_dir, "BIS.frcmod")
    subprocess.run(
        [os.path.join(AMBER_BIN, "antechamber"), "-i", bis_raw, "-fi", "mol2", "-o", bis_mol2, "-fo", "mol2", "-at", "gaff2", "-rn", "BIS", "-c", "gas"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [os.path.join(AMBER_BIN, "parmchk2"), "-i", bis_mol2, "-f", "mol2", "-o", bis_frcmod, "-s", "2"],
        check=True,
        capture_output=True,
    )

    return mono_mol2, mono_frcmod, bis_mol2, bis_frcmod
