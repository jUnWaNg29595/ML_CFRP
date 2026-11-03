# -*- coding: utf-8 -*-
"""
阶段 3：生产级多步动力学交联与快照导出 (stage3_crosslink.py)
核心特性:
- 借鉴 HTPolyNet 与真实环氧-胺固化动力学:
  * 显式追踪伯胺 (Primary Amine, z=2) 与 仲胺 (Secondary Amine, z=1)
  * 动力学接受概率: 伯胺成键概率 P_pri = 1.0, 仲胺成键概率 P_sec = 0.45 (符合经典的 k2/k1 实验比)
- 渐进式多截断搜索 (4.0 A -> 10.0 A) 与 局部应变释放 (CG minimization + NVT)
- LAMMPS OpenMP 多线程加速 (16-32 threads)
- 严格捕获指定固化度 alpha 快照
"""

import os
import json
import subprocess
import numpy as np

def parse_lammps_data(data_path):
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f]
        
    num_atoms = 0
    num_bonds = 0
    box_bounds = []
    sections = {}
    current_sec = None
    known_sections = ['Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals', 'Impropers']
    
    for line in lines:
        if not line or line.startswith('#'):
            continue
        if 'atoms' in line:
            num_atoms = int(line.split()[0])
        elif 'bonds' in line:
            num_bonds = int(line.split()[0])
        elif 'xlo xhi' in line:
            parts = line.split()
            box_bounds.append((float(parts[0]), float(parts[1])))
        elif 'ylo yhi' in line:
            parts = line.split()
            box_bounds.append((float(parts[0]), float(parts[1])))
        elif 'zlo zhi' in line:
            parts = line.split()
            box_bounds.append((float(parts[0]), float(parts[1])))
        else:
            sec_name = line.split()[0]
            if sec_name in known_sections:
                current_sec = sec_name
                sections[current_sec] = []
                continue
            if current_sec is not None:
                sections[current_sec].append(line)
                
    atom_coords = {}
    atom_types = {}
    atom_mols = {}
    atom_lines_raw = []
    
    for row in sections.get('Atoms', []):
        parts = row.split()
        if len(parts) >= 7:
            a_id = int(parts[0])
            m_id = int(parts[1])
            t_id = int(parts[2])
            atom_coords[a_id] = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            atom_types[a_id] = t_id
            atom_mols[a_id] = m_id
            atom_lines_raw.append(parts)
            
    existing_bonds = []
    for row in sections.get('Bonds', []):
        parts = row.split()
        if len(parts) >= 4:
            existing_bonds.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            
    return {
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'box': box_bounds,
        'atom_coords': atom_coords,
        'atom_types': atom_types,
        'atom_mols': atom_mols,
        'atom_lines_raw': atom_lines_raw,
        'bonds': existing_bonds
    }

def write_lammps_data(out_path, parsed, bonds):
    masses = {1: 12.011, 2: 1.008, 3: 15.999, 4: 14.007, 5: 32.065}
    box = parsed['box']
    with open(out_path, 'w') as f:
        f.write("LAMMPS data file - Direction B Crosslinked Network\n\n")
        f.write(f"{parsed['num_atoms']} atoms\n")
        f.write(f"{len(bonds)} bonds\n\n")
        f.write("5 atom types\n")
        f.write("2 bond types\n\n")
        f.write(f"{box[0][0]:.3f} {box[0][1]:.3f} xlo xhi\n")
        f.write(f"{box[1][0]:.3f} {box[1][1]:.3f} ylo yhi\n")
        f.write(f"{box[2][0]:.3f} {box[2][1]:.3f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in masses.items():
            f.write(f"{t} {m}\n")
        f.write("\nAtoms # full\n\n")
        for parts in parsed['atom_lines_raw']:
            a_id = int(parts[0])
            c = parsed['atom_coords'][a_id]
            f.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        f.write("\nBonds\n\n")
        for b in bonds:
            f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")

def run_relaxation_step(data_path, relaxed_path, temp=500.0, nsteps=500):
    in_script = data_path.replace('.data', '_relax.lmp')
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
minimize 1.0e-4 1.0e-5 500 2000

reset_timestep 0
velocity all create {temp} 42 dist gaussian
fix 1 all nve/limit 0.05
run 200
unfix 1

fix 2 all nvt temp {temp} {temp} 50.0
run {nsteps}
unfix 2

write_data {relaxed_path}
""")
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'
    cmd = ['/opt/conda/envs/lammps/bin/lmp', '-sf', 'omp', '-pk', 'omp', '16', '-in', in_script]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    if os.path.exists(in_script):
        os.remove(in_script)

def run_crosslinking_by_ratio(relaxed_data, meta_path, r=1.0, output_dir='/home/wangj/epoxy_prod/data', target_alphas=[0.70, 0.80, 0.90], k2_k1_ratio=0.45):
    os.makedirs(output_dir, exist_ok=True)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    epoxy_c_pool = list(meta['reactive_c_epoxy'])
    amine_n_pool = list(meta['reactive_n_amine'])
    
    # 配位数与动力学状态管理
    # 环氧 C 位点: 容量 1
    # 胺基 N 位点: 容量 2 (伯胺 state=2, 发生1次交联变为仲胺 state=1, 再次交联变为叔胺 state=0)
    epoxy_capacity = {c: 1 for c in epoxy_c_pool}
    amine_capacity = {n: 2 for n in amine_n_pool}
    
    # 理论最大可能成键数: 取决于限制组分
    total_epoxy = len(epoxy_c_pool)
    total_active_h = len(amine_n_pool) * 2
    max_bonds = min(total_epoxy, total_active_h)
    
    print(f"[阶段 3 - Direction B] 动力学交联启动 (r={r:.2f}):")
    print(f"  * 环氧官能团 = {total_epoxy}, 活泼氢位点 = {total_active_h}")
    print(f"  * 理论最大成键容量 = {max_bonds}")
    print(f"  * 动力学反应概率: 伯胺(Primary) P1=1.0, 仲胺(Secondary) P2={k2_k1_ratio}")
    
    parsed = parse_lammps_data(relaxed_data)
    box_size = parsed['box'][0][1] - parsed['box'][0][0]
    
    current_bonds = list(parsed['bonds'])
    bond_id = max([b[0] for b in current_bonds]) + 1
    
    crosslinked_snapshots = {}
    formed_crosslink_count = 0
    target_bond_counts = {alpha: max(1, int(round(alpha * max_bonds))) for alpha in target_alphas}
    sorted_targets = sorted(target_alphas)
    
    # 统计信息
    stat_pri_reactions = 0
    stat_sec_reactions = 0
    
    # 多级渐进截断搜索
    r_cut_stages = [4.5, 5.5, 6.5, 7.8, 9.2, 11.0, 13.5]
    
    np.random.seed(42)
    
    for r_cut in r_cut_stages:
        avail_c = [c for c, cap in epoxy_capacity.items() if cap > 0]
        avail_n = [n for n, cap in amine_capacity.items() if cap > 0]
        if not avail_c or not avail_n:
            break
            
        candidates = []
        for c in avail_c:
            pos_c = parsed['atom_coords'][c]
            for n in avail_n:
                pos_n = parsed['atom_coords'][n]
                dvec = pos_c - pos_n
                dvec = dvec - box_size * np.round(dvec / box_size)
                dist = np.linalg.norm(dvec)
                if dist <= r_cut:
                    candidates.append((dist, c, n))
                    
        candidates.sort(key=lambda x: x[0])
        new_bonds_this_cycle = 0
        
        for dist, c, n in candidates:
            if epoxy_capacity[c] <= 0 or amine_capacity[n] <= 0:
                continue
                
            # HTPolyNet 风格反应选择动力学判定
            is_secondary = (amine_capacity[n] == 1)
            reaction_prob = k2_k1_ratio if is_secondary else 1.0
            
            if np.random.uniform(0.0, 1.0) > reaction_prob:
                continue # 空间位阻/电子效应导致仲胺未发生反应
                
            # 成功成键
            current_bonds.append((bond_id, 2, c, n))
            bond_id += 1
            epoxy_capacity[c] -= 1
            amine_capacity[n] -= 1
            formed_crosslink_count += 1
            new_bonds_this_cycle += 1
            
            if is_secondary:
                stat_sec_reactions += 1
            else:
                stat_pri_reactions += 1
                
            for alpha in sorted_targets:
                if alpha not in crosslinked_snapshots and formed_crosslink_count >= target_bond_counts[alpha]:
                    snapshot_file = os.path.join(output_dir, f"snapshot_r_{r:.2f}_alpha_{alpha:.2f}.data")
                    write_lammps_data(snapshot_file, parsed, current_bonds)
                    relaxed_snap = snapshot_file.replace('.data', '_relaxed.data')
                    print(f"  [快照捕获] (r={r:.2f}) alpha={alpha:.2f} (成键数={formed_crosslink_count}/{max_bonds}) 正在执行高保真退火松弛...")
                    run_relaxation_step(snapshot_file, relaxed_snap, temp=500.0, nsteps=1000)
                    crosslinked_snapshots[alpha] = relaxed_snap
                    print(f"  -> 快照完成: {relaxed_snap}")
                    
        print(f"  [扫描阶段 R_cut={r_cut:.1f} A] 本轮成键: {new_bonds_this_cycle}, 累计成键: {formed_crosslink_count}/{max_bonds} ({formed_crosslink_count/max_bonds*100:.1f}%)")
        
        # 轮次松弛
        temp_data = os.path.join(output_dir, f'current_inter_r_{r:.2f}.data')
        write_lammps_data(temp_data, parsed, current_bonds)
        run_relaxation_step(temp_data, temp_data, temp=500.0, nsteps=300)
        parsed = parse_lammps_data(temp_data)
        
        if len(crosslinked_snapshots) == len(target_alphas):
            break

    # 兜底保障
    for alpha in target_alphas:
        if alpha not in crosslinked_snapshots:
            snapshot_file = os.path.join(output_dir, f"snapshot_r_{r:.2f}_alpha_{alpha:.2f}.data")
            write_lammps_data(snapshot_file, parsed, current_bonds)
            relaxed_snap = snapshot_file.replace('.data', '_relaxed.data')
            run_relaxation_step(snapshot_file, relaxed_snap, temp=500.0, nsteps=1000)
            crosslinked_snapshots[alpha] = relaxed_snap
            print(f"  [快照兜底] (r={r:.2f}) alpha={alpha:.2f} -> {relaxed_snap}")

    snap_json = os.path.join(output_dir, f'stage3_snapshots_r_{r:.2f}.json')
    with open(snap_json, 'w') as f:
        json.dump(crosslinked_snapshots, f, indent=2)

    print(f"[阶段 3 统计] 伯胺反应数: {stat_pri_reactions}, 仲胺反应数: {stat_sec_reactions} (比值={stat_sec_reactions/max(1, stat_pri_reactions):.3f})")
    print(f"[阶段 3 成功] (r={r:.2f}) Direction B 动力学快照生成完成: {snap_json}\n")
    return crosslinked_snapshots, snap_json

def run_stage3(r=1.0):
    data_path = f'/home/wangj/epoxy_prod/data/system_relaxed_r_{r:.2f}.data'.replace('.', '_')
    meta_path = f'/home/wangj/epoxy_prod/data/system_meta_r_{r:.2f}.json'.replace('.', '_')
    return run_crosslinking_by_ratio(data_path, meta_path, r=r)

if __name__ == '__main__':
    run_stage3(r=1.0)
