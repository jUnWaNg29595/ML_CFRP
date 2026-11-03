# -*- coding: utf-8 -*-
"""
阶段 2：生产级高保真体系装配与前平衡 (stage2_assembly.py)
特性:
- 针对 Direction B 设计: n_dgeba 默认 300, 体系规模约 18,000 ~ 20,000 原子
- 参数化化学计量比 r: 动态计算 DDS 单体数量与目标晶胞体积
- LAMMPS 多线程与 OpenMP 优化: 采用 16-32 线程结合智能退火平衡
"""

import os
import json
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def build_monomer_templates():
    smi_dgeba = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
    mol_dgeba = Chem.MolFromSmiles(smi_dgeba)
    mol_dgeba = Chem.AddHs(mol_dgeba)
    AllChem.EmbedMolecule(mol_dgeba, randomSeed=101)
    AllChem.MMFFOptimizeMolecule(mol_dgeba)

    patt_epoxy = Chem.MolFromSmarts('C1OC1')
    matches_dgeba = mol_dgeba.GetSubstructMatches(patt_epoxy)
    epoxy_c_indices = []
    for match in matches_dgeba:
        c_cands = [idx for idx in match if mol_dgeba.GetAtomWithIdx(idx).GetSymbol() == 'C']
        c_sub = sorted(c_cands, key=lambda idx: len([nbr for nbr in mol_dgeba.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetSymbol() == 'H']), reverse=True)[0]
        epoxy_c_indices.append(c_sub)

    smi_dds = 'Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1'
    mol_dds = Chem.MolFromSmiles(smi_dds)
    mol_dds = Chem.AddHs(mol_dds)
    AllChem.EmbedMolecule(mol_dds, randomSeed=202)
    AllChem.MMFFOptimizeMolecule(mol_dds)

    patt_amine = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    matches_dds = mol_dds.GetSubstructMatches(patt_amine)
    amine_n_indices = [m[0] for m in matches_dds]

    return (mol_dgeba, epoxy_c_indices), (mol_dds, amine_n_indices)

def pack_system_by_ratio(r=1.0, n_dgeba=300, output_dir='/home/wangj/epoxy_prod/data'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 理论活泼氢数 = n_dgeba * 2 * r, DDS 每个分子有 4 个活泼氢
    n_dds = int(round(n_dgeba * r / 2.0))
    phr_value = round((62.08 / 170.21) * 100.0 * r, 2)
    
    # 动态根据总原子数估算晶胞尺寸 (目标初密度 ~ 1.05 g/cm3)
    total_mw = n_dgeba * 340.41 + n_dds * 248.30
    target_density = 1.05
    box_vol_cm3 = (total_mw / 6.022e23) / target_density
    box_size = round((box_vol_cm3 * 1.0e24) ** (1.0 / 3.0) * 1.15, 2) # 留出 15% 余量便于初装配

    (m_dgeba, ep_c_idx), (m_dds, am_n_idx) = build_monomer_templates()
    
    elem_type = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5}
    masses = {1: 12.011, 2: 1.008, 3: 15.999, 4: 14.007, 5: 32.065}

    all_atoms = []
    all_bonds = []
    reactive_c = []
    reactive_n = []

    atom_counter = 1
    bond_counter = 1
    mol_counter = 1

    def add_molecule(mol, reactive_locs, is_dgeba=True):
        nonlocal atom_counter, bond_counter, mol_counter
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        
        rot = np.random.randn(3, 3)
        q, _ = np.linalg.qr(rot)
        shift = np.random.uniform(2.0, box_size - 2.0, size=3)
        
        idx_map = {}
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            xyz = np.array([pos.x, pos.y, pos.z])
            new_xyz = np.dot(q, xyz) + shift
            new_xyz = np.mod(new_xyz, box_size)
            
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            t = elem_type.get(sym, 1)
            
            all_atoms.append({
                'id': atom_counter,
                'mol': mol_counter,
                'type': t,
                'q': 0.0,
                'x': new_xyz[0],
                'y': new_xyz[1],
                'z': new_xyz[2]
            })
            idx_map[i] = atom_counter
            
            if i in reactive_locs:
                if is_dgeba:
                    reactive_c.append(atom_counter)
                else:
                    reactive_n.append(atom_counter)
            atom_counter += 1

        for b in mol.GetBonds():
            a1 = idx_map[b.GetBeginAtomIdx()]
            a2 = idx_map[b.GetEndAtomIdx()]
            all_bonds.append({
                'id': bond_counter,
                'type': 1,
                'a1': a1,
                'a2': a2
            })
            bond_counter += 1
            
        mol_counter += 1

    np.random.seed(42)
    for _ in range(n_dgeba):
        add_molecule(m_dgeba, ep_c_idx, is_dgeba=True)
    for _ in range(n_dds):
        add_molecule(m_dds, am_n_idx, is_dgeba=False)

    suffix = f'_r{r:.2f}'.replace('.', '_')
    data_path = os.path.join(output_dir, f'system_init{suffix}.data')
    meta_path = os.path.join(output_dir, f'system_meta{suffix}.json')

    with open(data_path, 'w') as f:
        f.write("LAMMPS data file - Direction B Production Scale DGEBA/DDS\n\n")
        f.write(f"{len(all_atoms)} atoms\n")
        f.write(f"{len(all_bonds)} bonds\n\n")
        f.write(f"5 atom types\n")
        f.write(f"2 bond types\n\n")
        f.write(f"0.0 {box_size:.3f} xlo xhi\n")
        f.write(f"0.0 {box_size:.3f} ylo yhi\n")
        f.write(f"0.0 {box_size:.3f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in masses.items():
            f.write(f"{t} {m}\n")
        f.write("\nAtoms # full\n\n")
        for at in all_atoms:
            f.write(f"{at['id']} {at['mol']} {at['type']} {at['q']:.4f} {at['x']:.4f} {at['y']:.4f} {at['z']:.4f}\n")
        f.write("\nBonds\n\n")
        for bd in all_bonds:
            f.write(f"{bd['id']} {bd['type']} {bd['a1']} {bd['a2']}\n")

    metadata = {
        'stoichiometric_ratio_r': r,
        'phr': phr_value,
        'n_dgeba': n_dgeba,
        'n_dds': n_dds,
        'box_size': box_size,
        'total_atoms': len(all_atoms),
        'reactive_c_epoxy': reactive_c,
        'reactive_n_amine': reactive_n
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[阶段 2 成功] (r={r:.2f}) 装配生产级体系: DGEBA={n_dgeba}, DDS={n_dds}, phr={phr_value:.2f}, 总原子数={len(all_atoms)}, 盒子尺寸={box_size} A")
    return data_path, meta_path

def run_relaxation(data_path, r=1.0, output_dir='/home/wangj/epoxy_prod/data'):
    suffix = f'_r{r:.2f}'.replace('.', '_')
    in_script = os.path.join(output_dir, f'in_relax{suffix}.lmp')
    relaxed_data = os.path.join(output_dir, f'system_relaxed{suffix}.data')
    
    with open(in_script, 'w') as f:
        f.write(f"""units real
atom_style full
boundary p p p

pair_style lj/cut 10.0
bond_style harmonic
special_bonds lj 0.0 0.0 0.5

read_data {data_path}

pair_coeff * * 0.10 3.4
pair_coeff 2 * 0.03 2.5
pair_coeff 3 * 0.12 3.1
pair_coeff 4 * 0.15 3.3
pair_coeff 5 * 0.25 3.6

bond_coeff 1 350.0 1.45
bond_coeff 2 320.0 1.47

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 100
thermo_style custom step temp press density pe

minimize 1.0e-4 1.0e-5 1000 5000

reset_timestep 0
velocity all create 500.0 12345 dist gaussian

fix 1 all nve/limit 0.1
run 500
unfix 1

fix 2 all nvt temp 500.0 500.0 50.0
run 1000
unfix 2

write_data {relaxed_data}
""")

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'
    cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', in_script]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if res.returncode != 0:
        cmd_fallback = ['/opt/conda/envs/lammps/bin/lmp', '-in', in_script]
        res = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f'LAMMPS 前平衡失败 (r={r}): {res.stderr}')
            
    if os.path.exists(in_script):
        os.remove(in_script)
    print(f"[阶段 2 成功] (r={r:.2f}) 生产级前平衡完成: {relaxed_data}")
    return relaxed_data

def run_stage2(r=1.0, n_dgeba=300):
    data_path, meta_path = pack_system_by_ratio(r=r, n_dgeba=n_dgeba)
    relaxed_data = run_relaxation(data_path, r=r)
    return relaxed_data, meta_path

if __name__ == '__main__':
    run_stage2(r=1.0, n_dgeba=300)
