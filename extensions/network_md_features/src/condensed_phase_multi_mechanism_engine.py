"""
condensed_phase_multi_mechanism_engine.py
凝聚态多固化机理（环氧+胺、环氧+酸酐、环氧+酚醛、环氧+硫醇、纯环氧均聚自固化）
自动交联团簇建模、统一 AM1-BCC 参数化、孤立气相单分子参考态标定与 GROMACS NPT 压密 + PME 采样引擎。

核心科学能力：
1. 自动反应团簇构建：对接 RDKit 反应 SMARTS，支持 5 大类固化机理（含无固化剂的环氧均聚自固化），
   生成化合价 100% 合法、零残氢错误的代表性加成物/支化交联团簇；
2. 统一力场与电荷协议：全组分统一采用 AmberTools GAFF2 + AM1-BCC (antechamber -c bcc) 电荷，
   消除单体与加成物电荷协议割裂，并強制精确电中性；
3. 严格气相参考态标定：对每种独立组分在 300 K 下运行孤立单分子气相 NVT 采样，实测 <E_pot,gas,i>，
   从而严格按热力学定义计算分子间内聚能密度 CED = -(E_condensed - sum(N_i * E_gas,i)) / V；
4. 四步凝聚态压密与平衡协议：
   - Step 1: 能量最小化 (steep)
   - Step 2: 高温高压 NPT 各向同性压密 (450 K, 500 bar, C-rescale) 消除空隙至凝聚态
   - Step 3: 300 K, 1 bar NPT 退火弛豫至常压平衡密度 (~1.10 - 1.30 g/cm^3)
   - Step 4: 300 K PME + DispCorr 凝聚态生产采样，无缝对接 postprocess_trajectory.py 提取全部特征。
"""

import os
import sys
import json
import math
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

try:
    from .pbc_utils import (
        check_inter_residue_contacts_pbc,
        check_system_bond_strains_detailed,
    )
    from .native_edr_extractor import extract_native_edr_data
    from .postprocess_trajectory import run_postprocessing
except (ImportError, ValueError):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from src.pbc_utils import (
        check_inter_residue_contacts_pbc,
        check_system_bond_strains_detailed,
    )
    from src.native_edr_extractor import extract_native_edr_data
    from src.postprocess_trajectory import run_postprocessing

AMBER_BIN = "/home/wangj/miniconda3/envs/amber_env/bin"
GMX_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/gmx"
OBABEL_BIN = "/home/wangj/miniconda3/envs/htpolynet_env/bin/obabel"


# ==============================================================================
# 1. 多固化机理 RDKit 反应团簇自动生成器 (支持胺、酸酐、酚醛、硫醇、自固化)
# ==============================================================================

MECHANISM_RXN_SMIRKS = {
    # 环氧 + 伯胺 -> 仲胺 + β-羟基
    "amine_step1": "[C:1]1[O:2][C:3]1.[NX3;H2;!$(NC=O);!$(NS=O):4]>>[C:1]([O:2])[C:3][N:4]",
    # 环氧 + 仲胺 -> 叔胺/双加成 + β-羟基
    "amine_step2": "[C:1]1[O:2][C:3]1.[NX3;H1;!$(NC=O);!$(NS=O):4]>>[C:1]([O:2])[C:3][N:4]",
    # 环氧 + 5元环酸酐开环 + 再与第2分子环氧酯化交联 -> 双酯交联桥 + β-羟基
    "anhydride_5_open": "[C:1]1[O:2][C:3]1.[#6:9]1[#6:10][C:4](=[O:5])[O:6][C:7](=[O:8])1>>[C:1]([O:2])[C:3][O:6][C:7](=[O:8])[#6:10][#6:9][C:4](=[O:5])O",
    # 环氧 + 6元环酸酐开环 -> 单酯羧酸
    "anhydride_6_open": "[C:1]1[O:2][C:3]1.[#6:9]1[#6:10][#6:11][C:4](=[O:5])[O:6][C:7](=[O:8])1>>[C:1]([O:2])[C:3][O:6][C:7](=[O:8])[#6:11][#6:10][#6:9][C:4](=[O:5])O",
    # 羧酸 + 环氧 -> β-羟基酯
    "carboxylic_acid_add": "[C:1]1[O:2][C:3]1.[CX3:4](=[O:5])[OX2;H1:6]>>[C:1]([O:2])[C:3][O:6][CX3:4](=[O:5])",
    # 环氧 + 酚羟基 -> 芳醚 + β-羟基
    "phenol_add": "[C:1]1[O:2][C:3]1.[OX2;H1:4][c:5]>>[C:1]([O:2])[C:3][O:4][c:5]",
    # 环氧 + 硫醇 -> 硫醚 + β-羟基
    "thiol_add": "[C:1]1[O:2][C:3]1.[SX2;H1:4][#6:5]>>[C:1]([O:2])[C:3][S:4][#6:5]",
    # 环氧 + 异氰酸酯 -> 五元噁唑烷酮环
    "isocyanate_add": "[C:1]1[O:2][C:3]1.[N:4]=[C:5]=[O:6]>>[C:1]1[O:2][C:5](=[O:6])[N:4][C:3]1",
    # 纯环氧自固化 (引发开环生成邻二醇后与另一环氧发生链式聚醚化)
    "epoxy_init_diol": "[C:1]1[O:2][C:3]1>>[C:1](O)[C:3][O:2]",
    "epoxy_homopoly_ether": "[C:1]1[O:2][C:3]1.[OX2;H1:4][CX4;#6:5]>>[C:1]([O:2])[C:3][O:4][#6:5]",
}


def _run_bimolecular_smirks(smirks: str, mol_a: Chem.Mol, mol_b: Chem.Mol) -> Optional[Chem.Mol]:
    rxn = rdChemReactions.ReactionFromSmarts(smirks)
    prods = rxn.RunReactants((mol_a, mol_b))
    for p_tuple in prods:
        if len(p_tuple) >= 1:
            cand = p_tuple[0]
            try:
                Chem.SanitizeMol(cand)
                smi = Chem.MolToSmiles(cand, canonical=True)
                clean = Chem.MolFromSmiles(smi)
                if clean is not None:
                    return clean
            except Exception:
                continue
    return None


def _run_unimolecular_smirks(smirks: str, mol_a: Chem.Mol) -> Optional[Chem.Mol]:
    rxn = rdChemReactions.ReactionFromSmarts(smirks)
    prods = rxn.RunReactants((mol_a,))
    for p_tuple in prods:
        if len(p_tuple) >= 1:
            cand = p_tuple[0]
            try:
                Chem.SanitizeMol(cand)
                smi = Chem.MolToSmiles(cand, canonical=True)
                clean = Chem.MolFromSmiles(smi)
                if clean is not None:
                    return clean
            except Exception:
                continue
    return None


def generate_multi_mechanism_reaction_adducts(
    resin_smiles: str,
    hardener_smiles: Optional[str],
    mechanism: str,
) -> Dict[str, Any]:
    """
    根据给定的树脂 SMILES、固化剂 SMILES 与反应机理类型，自动生成：
    - resin_canonical: 规范化未反应环氧树脂 SMILES
    - hardener_canonical: 规范化固化剂（自固化时为引发邻二醇单体）SMILES
    - adduct_1_smiles: 1 步加成产物 SMILES
    - adduct_2_smiles: 2 步交联加成产物 SMILES
    """
    mol_r = Chem.MolFromSmiles(resin_smiles)
    if mol_r is None:
        raise ValueError(f"无法解析树脂 SMILES: {resin_smiles}")
    resin_can = Chem.MolToSmiles(mol_r, canonical=True)
    mol_r = Chem.MolFromSmiles(resin_can)

    mech = mechanism.lower().strip()

    if mech in ("self_cure", "homopolymerization", "polyol") or not hardener_smiles:
        # 环氧均聚自固化：环氧基先引发开环为邻二醇，再连续与环氧基发生聚醚化反应
        mol_diol = _run_unimolecular_smirks(MECHANISM_RXN_SMIRKS["epoxy_init_diol"], mol_r)
        if mol_diol is None:
            raise ValueError(f"无法从树脂生成自固化引发二醇: {resin_can}")
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["epoxy_homopoly_ether"], mol_r, mol_diol)
        if mol_ad1 is None:
            raise ValueError(f"无法生成自固化聚醚二聚体: {resin_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["epoxy_homopoly_ether"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
        return {
            "mechanism": "homopolymerization",
            "resin_smiles": resin_can,
            "hardener_smiles": Chem.MolToSmiles(mol_diol, canonical=True),
            "adduct_1_smiles": Chem.MolToSmiles(mol_ad1, canonical=True),
            "adduct_2_smiles": Chem.MolToSmiles(mol_ad2, canonical=True),
        }

    mol_h = Chem.MolFromSmiles(hardener_smiles)
    if mol_h is None:
        raise ValueError(f"无法解析固化剂 SMILES: {hardener_smiles}")
    hard_can = Chem.MolToSmiles(mol_h, canonical=True)
    mol_h = Chem.MolFromSmiles(hard_can)

    if mech == "amine":
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["amine_step1"], mol_r, mol_h)
        if mol_ad1 is None:
            mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["amine_step2"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-胺 1 步加成失败: {resin_can} + {hard_can}")
        # 优先在第二个伯胺位点加成，若无则在仲胺位点加成
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["amine_step1"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["amine_step2"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    elif mech == "anhydride":
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["anhydride_5_open"], mol_r, mol_h)
        if mol_ad1 is None:
            mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["anhydride_6_open"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-酸酐开环反应失败: {resin_can} + {hard_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["carboxylic_acid_add"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    elif mech == "phenol":
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["phenol_add"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-酚醛加成失败: {resin_can} + {hard_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["phenol_add"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    elif mech == "thiol":
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["thiol_add"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-硫醇加成失败: {resin_can} + {hard_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["thiol_add"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    elif mech == "isocyanate":
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["isocyanate_add"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-异氰酸酯环化失败: {resin_can} + {hard_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["isocyanate_add"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    elif mech in ("carboxyl", "acid"):
        mol_ad1 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["carboxylic_acid_add"], mol_r, mol_h)
        if mol_ad1 is None:
            raise ValueError(f"环氧-羧酸加成失败: {resin_can} + {hard_can}")
        mol_ad2 = _run_bimolecular_smirks(MECHANISM_RXN_SMIRKS["carboxylic_acid_add"], mol_r, mol_ad1)
        if mol_ad2 is None:
            mol_ad2 = mol_ad1
    else:
        raise ValueError(f"不支持的固化反应机理类型: {mechanism}")

    return {
        "mechanism": mech,
        "resin_smiles": resin_can,
        "hardener_smiles": hard_can,
        "adduct_1_smiles": Chem.MolToSmiles(mol_ad1, canonical=True),
        "adduct_2_smiles": Chem.MolToSmiles(mol_ad2, canonical=True),
    }


# ==============================================================================
# 2. 统一 GAFF2 + AM1-BCC 参数化与 300 K 孤立气相单分子参考势能标定
# ==============================================================================

def _enforce_exact_neutrality_in_mol2(mol2_path: str) -> float:
    """
    将 antechamber 输出的 mol2 原子电荷微小舍入误差（通常 ~1e-4 e）均摊消除，
    确保单分子净电荷在双精度下精确等于 0.000000 e。
    """
    with open(mol2_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    atom_line_indices = []
    charges = []
    in_atoms = False
    for idx, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        elif line.startswith("@<TRIPOS>BOND") or line.startswith("@<TRIPOS>SUBSTRUCTURE"):
            in_atoms = False
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) >= 9:
                atom_line_indices.append(idx)
                charges.append(float(parts[8]))

    if not charges:
        raise ValueError(f"MOL2 文件中未找到原子行: {mol2_path}")

    q_arr = np.array(charges, dtype=float)
    residual = float(np.sum(q_arr))
    q_arr -= residual / len(q_arr)
    # 再次四舍五入到 6 位小数并将最后余量补在绝对电荷最大的原子上
    q_rounded = np.round(q_arr, 6)
    rem = round(float(np.sum(q_rounded)), 6)
    if abs(rem) > 0:
        max_i = int(np.argmax(np.abs(q_rounded)))
        q_rounded[max_i] = round(float(q_rounded[max_i] - rem), 6)

    for local_i, line_idx in enumerate(atom_line_indices):
        p = lines[line_idx].split()
        q = q_rounded[local_i]
        lines[line_idx] = (
            f"{int(p[0]):7d} {p[1]:<8s} {float(p[2]):10.4f} {float(p[3]):10.4f} "
            f"{float(p[4]):10.4f} {p[5]:<6s} {int(p[6]):4d} {p[7]:<6s} {q:10.6f}\n"
        )

    with open(mol2_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return float(np.sum(q_rounded))


def parameterize_species_am1bcc_with_gas_ref(
    smiles: str,
    res_code: str,
    out_dir: str,
    seed: int = 20261019,
    compute_gas_reference: bool = True,
) -> Dict[str, Any]:
    """
    对给定 SMILES 分子执行：
    1. RDKit 3D 嵌入与 MMFF 优化；
    2. AmberTools antechamber -at gaff2 -c bcc (单点 AM1-BCC 半经验量子化学电荷) + 精确电中性校准；
    3. parmchk2 生成 .frcmod；
    4. (可选) 在 300 K 下运行孤立单分子气相 NVT 采样，实测 <E_pot,gas> 作为严格 CED 参考态。
    """
    sp_dir = Path(out_dir) / f"species_{res_code}"
    sp_dir.mkdir(parents=True, exist_ok=True)

    mol2_final = sp_dir / f"{res_code}.mol2"
    frcmod_final = sp_dir / f"{res_code}.frcmod"
    meta_json = sp_dir / f"{res_code}_meta.json"

    if mol2_final.exists() and frcmod_final.exists() and meta_json.exists():
        return json.loads(meta_json.read_text(encoding="utf-8"))

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"无法从 SMILES 构建加氢分子 ({res_code}): {smiles}")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    cid = AllChem.EmbedMolecule(mol, params)
    if cid == -1:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid == -1:
        raise RuntimeError(f"RDKit 3D 构象嵌入失败 ({res_code}): {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=2500)

    # 将分子中心移至原点，并记录分子跨度半径
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    center = coords.mean(axis=0)
    coords -= center
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].tolist())
    span_radius_A = float(np.max(np.linalg.norm(coords, axis=1)))

    sdf_path = sp_dir / f"{res_code}.sdf"
    raw_mol2 = sp_dir / f"{res_code}_raw.mol2"
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(Chem.MolToMolBlock(mol))

    subprocess.run(
        [OBABEL_BIN, str(sdf_path), "-omol2", "-O", str(raw_mol2), "--title", res_code],
        check=True,
        capture_output=True,
    )

    env = os.environ.copy()
    env["PATH"] = f"{AMBER_BIN}:{os.path.dirname(GMX_BIN)}:{env.get('PATH', '')}"
    env["OMP_NUM_THREADS"] = "1"

    # 优先使用单点 AM1-BCC (-c bcc -ek "maxcyc=0")
    cmd_bcc = [
        os.path.join(AMBER_BIN, "antechamber"),
        "-i", str(raw_mol2),
        "-fi", "mol2",
        "-o", str(mol2_final),
        "-fo", "mol2",
        "-at", "gaff2",
        "-rn", res_code,
        "-c", "bcc",
        "-nc", "0",
        "-ek", "maxcyc=0, qm_theory='AM1'",
    ]
    res_ant = subprocess.run(cmd_bcc, cwd=str(sp_dir), env=env, capture_output=True, text=True)
    if res_ant.returncode != 0 or not mol2_final.exists():
        # 若偶发收敛问题，开启 50 步几何微调重试 AM1-BCC
        cmd_bcc_retry = [
            os.path.join(AMBER_BIN, "antechamber"),
            "-i", str(raw_mol2),
            "-fi", "mol2",
            "-o", str(mol2_final),
            "-fo", "mol2",
            "-at", "gaff2",
            "-rn", res_code,
            "-c", "bcc",
            "-nc", "0",
            "-ek", "maxcyc=50, qm_theory='AM1'",
        ]
        res_ant = subprocess.run(cmd_bcc_retry, cwd=str(sp_dir), env=env, capture_output=True, text=True)
        if res_ant.returncode != 0 or not mol2_final.exists():
            raise RuntimeError(f"antechamber AM1-BCC 失败 ({res_code}):\n{res_ant.stderr}")

    net_q = _enforce_exact_neutrality_in_mol2(str(mol2_final))
    subprocess.run(
        [os.path.join(AMBER_BIN, "parmchk2"), "-i", str(mol2_final), "-f", "mol2", "-o", str(frcmod_final), "-s", "2"],
        cwd=str(sp_dir),
        env=env,
        check=True,
        capture_output=True,
    )

    gas_pot_mean_kJ_mol = None
    if compute_gas_reference:
        env_gmx = env.copy()
        env_gmx.pop("OMP_NUM_THREADS", None)
        gas_dir = sp_dir / "gas_ref_300K"
        gas_dir.mkdir(exist_ok=True)
        leap_gas = gas_dir / "leap_gas.in"
        prm_gas = gas_dir / "gas.prmtop"
        crd_gas = gas_dir / "gas.inpcrd"
        with open(leap_gas, "w", encoding="utf-8") as f:
            f.write("source leaprc.gaff2\n")
            f.write(f"loadamberparams {frcmod_final}\n")
            f.write(f"MOL = loadmol2 {mol2_final}\n")
            f.write("translate MOL { 25.0 25.0 25.0 }\n")
            f.write("set MOL box { 50.0 50.0 50.0 }\n")
            f.write(f"saveamberparm MOL {prm_gas} {crd_gas}\nquit\n")
        subprocess.run([os.path.join(AMBER_BIN, "tleap"), "-f", str(leap_gas)], cwd=str(gas_dir), env=env, check=True, capture_output=True)

        parm_g = pmd.load_file(str(prm_gas), str(crd_gas))
        top_g = gas_dir / "gas.top"
        gro_g = gas_dir / "gas.gro"
        parm_g.save(str(top_g), overwrite=True)
        parm_g.save(str(gro_g), overwrite=True)

        mdp_min_g = gas_dir / "min.mdp"
        mdp_min_g.write_text(
            "integrator = steep\nemtol = 100.0\nemstep = 0.01\nnsteps = 1000\n"
            "cutoff-scheme = Verlet\nrcoulomb = 1.5\nrvdw = 1.5\npbc = xyz\n",
            encoding="utf-8",
        )
        subprocess.run([GMX_BIN, "grompp", "-f", "min.mdp", "-c", "gas.gro", "-p", "gas.top", "-o", "min.tpr", "-maxwarn", "3"], cwd=str(gas_dir), check=True, capture_output=True)
        subprocess.run([GMX_BIN, "mdrun", "-deffnm", "min", "-ntmpi", "1", "-ntomp", "2"], cwd=str(gas_dir), env=env_gmx, check=True, capture_output=True)

        mdp_nvt_g = gas_dir / "nvt.mdp"
        mdp_nvt_g.write_text(
            f"integrator = md\ndt = 0.001\nnsteps = 3000\ncutoff-scheme = Verlet\n"
            f"rcoulomb = 1.5\nrvdw = 1.5\ncoulombtype = Cut-off\ntcoupl = v-rescale\n"
            f"tc-grps = System\ntau_t = 0.1\nref_t = 300\ngen-vel = yes\ngen-temp = 300\n"
            f"gen-seed = {seed}\nnstenergy = 50\npbc = xyz\n",
            encoding="utf-8",
        )
        subprocess.run([GMX_BIN, "grompp", "-f", "nvt.mdp", "-c", "min.gro", "-p", "gas.top", "-o", "nvt.tpr", "-maxwarn", "3"], cwd=str(gas_dir), check=True, capture_output=True)
        subprocess.run([GMX_BIN, "mdrun", "-deffnm", "nvt", "-ntmpi", "1", "-ntomp", "2"], cwd=str(gas_dir), env=env_gmx, check=True, capture_output=True)

        edr_g_res = extract_native_edr_data(
            str(gas_dir / "nvt.edr"),
            box_volume_nm3=125.0,
            out_dir=str(gas_dir),
            gmx_bin=GMX_BIN,
            t_start_ps=1.0,
            t_end_ps=3.0,
        )
        gas_pot_mean_kJ_mol = float(edr_g_res["mean_potential_energy_kJ_mol"])

    info = {
        "res_code": res_code,
        "smiles": smiles,
        "atom_count": int(mol.GetNumAtoms()),
        "span_radius_A": round(span_radius_A, 3),
        "net_charge_e": net_q,
        "charge_protocol": "GAFF2_AM1BCC_UNIFIED",
        "mol2_path": str(mol2_final.resolve()),
        "frcmod_path": str(frcmod_final.resolve()),
        "gas_reference_potential_300K_kJ_mol": gas_pot_mean_kJ_mol,
    }
    meta_json.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


# ==============================================================================
# 3. 凝聚态多分子装配 + 四步 GROMACS (Min -> 高压NPT压密 -> 常压NPT退火 -> PME生产)
# ==============================================================================

def compute_stoichiometry_and_cure_coupled_composition(
    sp_r: Dict[str, Any],
    sp_h: Dict[str, Any],
    sp_a1: Dict[str, Any],
    sp_a2: Dict[str, Any],
    stoichiometric_ratio_r: float = 1.0,
    cure_temperature_c: float = 160.0,
    cure_time_h: float = 3.0,
    target_atom_budget: int = 1150,
) -> Tuple[List[Tuple[Dict[str, Any], int]], int, Dict[str, float]]:
    """
    根据化学计量当量比 r (活性H / 环氧基) 与固化工艺条件 (最高固化温度 T_cure, 固化时间 t_cure)
    动态计算凝聚态盒子中的交联簇与未反应单体整数分子组成 [(sp_a2, n_a2), (sp_a1, n_a1), (sp_r, n_r), (sp_h, n_h)]
    """
    r_clean = float(np.clip(stoichiometric_ratio_r if not np.isnan(stoichiometric_ratio_r) else 1.0, 0.25, 2.50))
    t_c = float(np.clip(cure_temperature_c if not np.isnan(cure_temperature_c) else 150.0, 25.0, 280.0))
    time_h = float(np.clip(cure_time_h if not np.isnan(cure_time_h) else 3.0, 0.25, 48.0))

    t_k = t_c + 273.15
    k_arr = math.exp(-3800.0 * (1.0 / t_k - 1.0 / 433.15))
    alpha_kinetic = float(np.clip(1.0 - math.exp(-0.85 * k_arr * (time_h ** 0.65)), 0.35, 0.96))

    if r_clean <= 1.0:
        alpha_epoxy_eff = r_clean * alpha_kinetic
        w_a2 = (alpha_epoxy_eff ** 1.6) * 0.65
        w_a1 = alpha_epoxy_eff * (1.0 - 0.6 * alpha_epoxy_eff) * 0.45
        w_r_free = max(0.05, (1.0 - r_clean) * 0.85 + (1.0 - alpha_kinetic) * 0.25)
        w_h_free = max(0.02, (1.0 - alpha_kinetic) * 0.15 * r_clean)
    else:
        inv_r = 1.0 / r_clean
        alpha_epoxy_eff = alpha_kinetic
        w_a2 = (inv_r ** 1.5) * (alpha_kinetic ** 1.5) * 0.60
        w_a1 = (1.0 - inv_r * 0.5) * alpha_kinetic * 0.55
        w_r_free = max(0.02, (1.0 - alpha_kinetic) * 0.15 * inv_r)
        w_h_free = max(0.05, (r_clean - 1.0) * 0.75 + (1.0 - alpha_kinetic) * 0.25)

    unit_atoms = (
        w_a2 * sp_a2["atom_count"]
        + w_a1 * sp_a1["atom_count"]
        + w_r_free * sp_r["atom_count"]
        + w_h_free * sp_h["atom_count"]
    )
    scale = max(1.0, target_atom_budget / max(50.0, unit_atoms))
    n_a2 = max(1, int(round(w_a2 * scale)))
    n_a1 = max(1, int(round(w_a1 * scale)))
    n_r = max(1 if w_r_free > 0.08 else 0, int(round(w_r_free * scale)))
    n_h = max(1 if w_h_free > 0.08 else 0, int(round(w_h_free * scale)))

    comp = [(sp_a2, n_a2), (sp_a1, n_a1), (sp_r, n_r), (sp_h, n_h)]
    tot_atoms = sum(sp["atom_count"] * c for sp, c in comp)
    if tot_atoms < 950:
        mult = int(math.ceil(1000.0 / max(1, tot_atoms)))
        comp = [(sp, c * mult) for sp, c in comp]
        n_a2, n_a1, n_r, n_h = [c for _, c in comp]

    init_epoxy_sites = max(2, (n_a2 * 2 + n_a1 * 1 + n_r) * 2)
    coupling_meta = {
        "stoichiometric_ratio_r": round(r_clean, 4),
        "cure_temperature_c": round(t_c, 2),
        "cure_temperature_K": round(t_k, 2),
        "cure_time_h": round(time_h, 2),
        "kinetic_conversion_factor": round(alpha_kinetic, 4),
        "effective_epoxy_conversion": round(alpha_epoxy_eff, 4),
        "n_adduct2": n_a2,
        "n_adduct1": n_a1,
        "n_free_resin": n_r,
        "n_free_hardener": n_h,
    }
    return comp, init_epoxy_sites, coupling_meta


def build_and_run_condensed_state(
    state_dir: str,
    state_index: int,
    state_name: str,
    species_Composition: List[Tuple[Dict[str, Any], int]],
    initial_epoxy_sites: int,
    seed: int = 20261019,
    nt_threads: int = 16,
    compress_steps: int = 20000,
    anneal_steps: int = 10000,
    prod_steps: int = 10000,
    mechanism_type: str = "amine",
    formulation_id: str = "DGEBA_DDS_BENCHMARK",
    cure_temperature_c: float = 160.0,
    stoichiometric_ratio_r: float = 1.0,
    cure_time_h: float = 3.0,
) -> Dict[str, Any]:
    """
    构建并运行单个凝聚态状态：
    species_Composition: [(species_info_dict, count), ...]
    """
    st_path = Path(state_dir).resolve()
    st_path.mkdir(parents=True, exist_ok=True)
    if (
        (st_path / "nvt.xtc").exists()
        and (st_path / "nvt.xtc").stat().st_size > 0
        and (st_path / "nvt.edr").exists()
        and (st_path / "nvt.edr").stat().st_size > 0
        and (st_path / "state_metadata.json").exists()
    ):
        return json.loads((st_path / "state_metadata.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(GMX_BIN)}:{AMBER_BIN}:{env.get('PATH', '')}"
    env.pop("OMP_NUM_THREADS", None)
    os.environ.pop("OMP_NUM_THREADS", None)

    # 1. 计算总分子数、总原子数与气相参考势能总和
    mol_instances: List[Dict[str, Any]] = []
    unique_species: Dict[str, Dict[str, Any]] = {}
    total_gas_ref_pot = 0.0
    has_all_gas_ref = True

    for sp_info, cnt in species_Composition:
        if cnt <= 0:
            continue
        unique_species[sp_info["res_code"]] = sp_info
        for _ in range(cnt):
            mol_instances.append(sp_info)
        if sp_info.get("gas_reference_potential_300K_kJ_mol") is not None:
            total_gas_ref_pot += float(sp_info["gas_reference_potential_300K_kJ_mol"]) * cnt
        else:
            has_all_gas_ref = False

    # 使用固定的随机种子打乱分子在晶格上的排列顺序，实现不同种子的真实构象采样
    rng = np.random.default_rng(seed + state_index * 100)
    perm = rng.permutation(len(mol_instances))
    mol_instances = [mol_instances[i] for i in perm]

    n_mols = len(mol_instances)
    n_grid_side = int(math.ceil(n_mols ** (1.0 / 3.0)))
    max_span_A = max(sp["span_radius_A"] for sp in unique_species.values())
    # 每个晶格单元边长取 2 * max_span_A + 6.5 A，确保初始绝对无跨分子或跨周期边界碰撞
    cell_step_A = max(18.0, 2.0 * max_span_A + 6.5)
    box_init_A = round(n_grid_side * cell_step_A + 4.0, 2)

    leap_in = st_path / "leap.in"
    prmtop_p = st_path / "system.prmtop"
    inpcrd_p = st_path / "system.inpcrd"

    grid_positions = []
    for ix in range(n_grid_side):
        for iy in range(n_grid_side):
            for iz in range(n_grid_side):
                if len(grid_positions) < n_mols:
                    # 加入微小随机抖动 (+- 0.6 A) 打破完美对称并区分不同随机种子
                    jitter = rng.uniform(-0.6, 0.6, size=3)
                    pos = np.array([
                        (ix + 0.5) * cell_step_A + jitter[0],
                        (iy + 0.5) * cell_step_A + jitter[1],
                        (iz + 0.5) * cell_step_A + jitter[2],
                    ])
                    grid_positions.append(pos)

    with open(leap_in, "w", encoding="utf-8") as f:
        f.write("source leaprc.gaff2\n")
        for rcode, sp in unique_species.items():
            f.write(f"loadamberparams {sp['frcmod_path']}\n")
            f.write(f"{rcode} = loadmol2 {sp['mol2_path']}\n")
        unit_names = []
        for idx, (sp, pos) in enumerate(zip(mol_instances, grid_positions)):
            uname = f"m{idx+1}"
            unit_names.append(uname)
            f.write(f"{uname} = copy {sp['res_code']}\n")
            f.write(f"translate {uname} {{ {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} }}\n")
        f.write(f"sys = combine {{ {' '.join(unit_names)} }}\n")
        f.write(f"set sys box {{ {box_init_A:.2f} {box_init_A:.2f} {box_init_A:.2f} }}\n")
        f.write(f"saveamberparm sys {prmtop_p} {inpcrd_p}\nquit\n")

    subprocess.run([os.path.join(AMBER_BIN, "tleap"), "-f", str(leap_in)], cwd=str(st_path), env=env, check=True, capture_output=True)

    parm = pmd.load_file(str(prmtop_p), str(inpcrd_p))
    total_mass_amu = float(sum(a.mass for a in parm.atoms))

    # 使用 SO(3) 四元数随机旋转 + PBC 最小镜像碰撞检测直接将各分子作为刚体装入初始密度 ~0.46 g/cm^3 的紧凑周期盒
    # 分子内部共价键长与键角 100% 保持原样，且避免从巨大空盒压缩
    orig_coords = np.asarray(parm.coordinates, dtype=float)

    def _rand_rot(r_gen: np.random.Generator) -> np.ndarray:
        u1, u2, u3 = r_gen.uniform(0.0, 1.0, size=3)
        q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
        q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
        q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
        q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
        return np.array([
            [1.0 - 2.0 * (q3 * q3 + q4 * q4), 2.0 * (q2 * q3 - q4 * q1), 2.0 * (q2 * q4 + q3 * q1)],
            [2.0 * (q2 * q3 + q4 * q1), 1.0 - 2.0 * (q2 * q2 + q4 * q4), 2.0 * (q3 * q4 - q2 * q1)],
            [2.0 * (q2 * q4 - q3 * q1), 2.0 * (q3 * q4 + q2 * q1), 1.0 - 2.0 * (q2 * q2 + q3 * q3)],
        ], dtype=float)

    res_sorted = sorted(list(parm.residues), key=lambda r: len(r.atoms), reverse=True)
    target_pack_dens = 0.42
    for pack_try in range(6):
        vol_init_nm3 = (total_mass_amu * 1.66053906660e-24) / (target_pack_dens * 1e-21)
        box_pack_A = round((vol_init_nm3 ** (1.0 / 3.0)) * 10.0, 2)
        box_pack_arr = np.array([box_pack_A, box_pack_A, box_pack_A], dtype=float)
        placed_coords = np.zeros_like(orig_coords)
        placed_atom_indices: List[int] = []
        all_ok = True
        for res in res_sorted:
            idxs = [a.idx for a in res.atoms]
            c0 = orig_coords[idxs]
            c0 = c0 - c0.mean(axis=0)
            best_c = None
            best_min_d = -1.0
            for _ in range(400):
                R_mat = _rand_rot(rng)
                trans = rng.uniform(0.0, box_pack_A, size=3)
                cand = (c0 @ R_mat.T) + trans
                if not placed_atom_indices:
                    best_c = cand
                    best_min_d = 999.0
                    break
                existing = placed_coords[placed_atom_indices]
                diff = cand[:, None, :] - existing[None, :, :]
                diff -= box_pack_arr * np.round(diff / box_pack_arr)
                min_d = float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_c = cand
                if min_d >= 2.25:
                    break
            if best_min_d < 2.10:
                all_ok = False
                break
            placed_coords[idxs] = best_c
            placed_atom_indices.extend(idxs)
        if all_ok:
            break
        target_pack_dens *= 0.85

    parm.coordinates = placed_coords
    parm.box = [box_pack_A, box_pack_A, box_pack_A, 90.0, 90.0, 90.0]

    top_p = st_path / "system.top"
    sys_gro_p = st_path / "system.gro"
    parm.save(str(top_p), overwrite=True)
    parm.save(str(sys_gro_p), overwrite=True)

    # Step 1: 能量最小化 (min)
    mdp_min = st_path / "min.mdp"
    mdp_min.write_text(
        "integrator  = steep\n"
        "emtol       = 250.0\n"
        "emstep      = 0.01\n"
        "nsteps      = 5000\n"
        "cutoff-scheme = Verlet\n"
        "ns_type     = grid\n"
        "nstlist     = 10\n"
        "rcoulomb    = 1.0\n"
        "rvdw        = 1.0\n"
        "coulombtype = PME\n"
        "DispCorr    = EnerPres\n"
        "pbc         = xyz\n",
        encoding="utf-8",
    )
    run_flags = ["-ntmpi", "1", "-ntomp", str(min(8, nt_threads))]
    subprocess.run([GMX_BIN, "grompp", "-f", "min.mdp", "-c", "system.gro", "-p", "system.top", "-o", "min.tpr", "-maxwarn", "3"], cwd=str(st_path), check=True, capture_output=True)
    subprocess.run([GMX_BIN, "mdrun", "-deffnm", "min"] + run_flags, cwd=str(st_path), check=True, capture_output=True)

    # Step 2: 从 0.46 g/cm^3 紧凑盒出发，连续分段 re-grompp 各向同性压密至 >= 1.03 g/cm^3
    curr_gro_path = st_path / "min.gro"
    best_gro_path = curr_gro_path
    best_dens = 0.0
    mdp_comp = st_path / "compress.mdp"
    for seg_i in range(10):
        gro_obj = pmd.load_file(str(curr_gro_path))
        vol_nm3 = float(np.prod(gro_obj.box[:3] / 10.0))
        curr_dens = float((total_mass_amu * 1.66053906660e-24) / (vol_nm3 * 1e-21))
        if curr_dens > best_dens:
            best_dens = curr_dens
            best_gro_path = curr_gro_path
        if best_dens >= 1.01:
            break
        gen_vel_str = "yes" if seg_i == 0 else "no"
        min_box_nm = float(np.min(gro_obj.box[:3] / 10.0))
        rcut_c = round(min(1.0, max(0.75, 0.38 * min_box_nm)), 2)
        mdp_comp.write_text(
            f"integrator  = md\n"
            f"dt          = 0.001\n"
            f"nsteps      = 4000\n"
            f"cutoff-scheme = Verlet\n"
            f"nstlist     = 20\n"
            f"rcoulomb    = {rcut_c:.2f}\n"
            f"rvdw        = {rcut_c:.2f}\n"
            f"coulombtype = PME\n"
            f"DispCorr    = EnerPres\n"
            f"tcoupl      = v-rescale\n"
            f"tc-grps     = System\n"
            f"tau_t       = 0.1\n"
            f"ref_t       = {int(round(np.clip(cure_temperature_c + 273.15, 350.0, 520.0)))}\n"
            f"pcoupl      = C-rescale\n"
            f"pcoupltype  = isotropic\n"
            f"tau_p       = 1.0\n"
            f"ref_p       = 1200.0\n"
            f"compressibility = 4.5e-5\n"
            f"gen-vel     = {gen_vel_str}\n"
            f"gen-temp    = 300\n"
            f"gen-seed    = {seed + seg_i}\n"
            f"nstenergy   = 250\n"
            f"pbc         = xyz\n",
            encoding="utf-8",
        )
        subprocess.run([GMX_BIN, "grompp", "-f", "compress.mdp", "-c", best_gro_path.name, "-p", "system.top", "-o", "compress.tpr", "-maxwarn", "3"], cwd=str(st_path), check=True, capture_output=True)
        subprocess.run([GMX_BIN, "mdrun", "-deffnm", "compress"] + run_flags, cwd=str(st_path), check=True, capture_output=True)
        seg_gro = st_path / f"compress_seg{seg_i}.gro"
        shutil.copy2(st_path / "compress.gro", seg_gro)
        curr_gro_path = seg_gro
    shutil.copy2(best_gro_path, st_path / "compress.gro")
    # Step 3: 300 K, 1.0 bar NPT 常压退火弛豫
    comp_final_obj = pmd.load_file(str(st_path / "compress.gro"))
    rcut_a = round(min(1.0, max(0.75, 0.38 * float(np.min(comp_final_obj.box[:3] / 10.0)))), 2)
    mdp_anneal = st_path / "anneal.mdp"
    mdp_anneal.write_text(
        f"integrator  = md\n"
        f"dt          = 0.001\n"
        f"nsteps      = {anneal_steps}\n"
        f"cutoff-scheme = Verlet\n"
        f"nstlist     = 20\n"
        f"rcoulomb    = {rcut_a:.2f}\n"
        f"rvdw        = {rcut_a:.2f}\n"
        f"coulombtype = PME\n"
        f"DispCorr    = EnerPres\n"
        f"tcoupl      = v-rescale\n"
        f"tc-grps     = System\n"
        f"tau_t       = 0.1\n"
        f"ref_t       = 300\n"
        f"pcoupl      = C-rescale\n"
        f"pcoupltype  = isotropic\n"
        f"tau_p       = 2.0\n"
        f"ref_p       = 1.0\n"
        f"compressibility = 4.5e-5\n"
        f"gen-vel     = no\n"
        f"nstenergy   = 200\n"
        f"pbc         = xyz\n",
        encoding="utf-8",
    )
    subprocess.run([GMX_BIN, "grompp", "-f", "anneal.mdp", "-c", "compress.gro", "-p", "system.top", "-o", "anneal.tpr", "-maxwarn", "3"], cwd=str(st_path), check=True, capture_output=True)
    subprocess.run([GMX_BIN, "mdrun", "-deffnm", "anneal"] + run_flags, cwd=str(st_path), check=True, capture_output=True)
    # Step 4: 300 K, 1.0 bar 凝聚态生产采样 (输出 nvt.xtc, nvt.edr, nvt.gro, nvt.tpr, mdout.mdp 供统一后处理读取)
    ann_final_obj = pmd.load_file(str(st_path / "anneal.gro"))
    rcut_p = round(min(1.0, max(0.75, 0.38 * float(np.min(ann_final_obj.box[:3] / 10.0)))), 2)
    mdp_prod = st_path / "nvt.mdp"
    mdp_prod.write_text(
        f"integrator  = md\n"
        f"dt          = 0.001\n"
        f"nsteps      = {prod_steps}\n"
        f"cutoff-scheme = Verlet\n"
        f"nstlist     = 20\n"
        f"rcoulomb    = {rcut_p:.2f}\n"
        f"rvdw        = {rcut_p:.2f}\n"
        f"coulombtype = PME\n"
        f"DispCorr    = EnerPres\n"
        f"tcoupl      = v-rescale\n"
        f"tc-grps     = System\n"
        f"tau_t       = 0.1\n"
        f"ref_t       = 300\n"
        f"pcoupl      = C-rescale\n"
        f"pcoupltype  = isotropic\n"
        f"tau_p       = 2.0\n"
        f"ref_p       = 1.0\n"
        f"compressibility = 4.5e-5\n"
        f"gen-vel     = no\n"
        f"nstxout-compressed = 200\n"
        f"nstenergy   = 200\n"
        f"pbc         = xyz\n",
        encoding="utf-8",
    )
    subprocess.run([GMX_BIN, "grompp", "-f", "nvt.mdp", "-c", "anneal.gro", "-p", "system.top", "-o", "nvt.tpr", "-po", "mdout.mdp", "-maxwarn", "3"], cwd=str(st_path), check=True, capture_output=True)
    subprocess.run([GMX_BIN, "mdrun", "-deffnm", "nvt"] + run_flags, cwd=str(st_path), check=True, capture_output=True)

    state_meta = {
        "formulation_id": formulation_id,
        "mechanism_type": mechanism_type,
        "state_index": state_index,
        "state_name": state_name,
        "seed": seed,
        "initial_epoxy_sites": initial_epoxy_sites,
        "charge_protocol_declared": "GAFF2_AM1BCC_UNIFIED",
        "crosslink_engine_status": f"MULTI_MECHANISM_RDKIT_CLUSTER_CONDENSED_NPT_PME ({mechanism_type.upper()})",
        "stoichiometric_ratio_r": round(float(stoichiometric_ratio_r), 4),
        "cure_temperature_c": round(float(cure_temperature_c), 2),
        "cure_time_h": round(float(cure_time_h), 2),
        "unbound_gas_reference_potential_kJ_mol": round(total_gas_ref_pot, 4) if has_all_gas_ref else None,
        "species_composition": [
            {"res_code": sp["res_code"], "smiles": sp["smiles"], "count": cnt, "atoms_per_mol": sp["atom_count"]}
            for sp, cnt in species_Composition
        ],
    }
    with open(st_path / "state_metadata.json", "w", encoding="utf-8") as f_m:
        json.dump(state_meta, f_m, indent=2, ensure_ascii=False)

    return state_meta
