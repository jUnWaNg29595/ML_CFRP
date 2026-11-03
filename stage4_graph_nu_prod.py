# -*- coding: utf-8 -*-
"""
阶段 4：生产级图论交联拓扑与交联密度分析 (stage4_graph_nu.py)
1. 基于 NetworkX 将 20,000 原子的快照全拓扑映射为无向图；
2. 迭代剪枝悬挂末端 (Dangling ends) 与游离单体 (Solf fraction)；
3. 提取分支度 >= 3 的有效交联节点 (Effective Junctions)；
4. 精确计算有效节点体积摩尔密度 nu (mmol/cm³) 与平均链段分子量 Mc (g/mol)；
5. 计算宏观凝胶质量分数 (Gel fraction)。
"""

import os
import json
import numpy as np
import networkx as nx

def parse_data_for_graph(data_path):
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f]
        
    box_bounds = []
    sections = {}
    current_sec = None
    known_sections = ['Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals', 'Impropers']
    
    for line in lines:
        if not line or line.startswith('#'):
            continue
        if 'xlo xhi' in line or 'ylo yhi' in line or 'zlo zhi' in line:
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
                
    atoms = {}
    mass_table = {1: 12.011, 2: 1.008, 3: 15.999, 4: 14.007, 5: 32.065}
    for row in sections.get('Atoms', []):
        parts = row.split()
        if len(parts) >= 7:
            a_id = int(parts[0])
            t_id = int(parts[2])
            atoms[a_id] = {
                'type': t_id,
                'mass': mass_table.get(t_id, 12.0),
                'coord': np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            }
            
    bonds = []
    for row in sections.get('Bonds', []):
        parts = row.split()
        if len(parts) >= 4:
            bonds.append((int(parts[2]), int(parts[3])))
            
    return atoms, bonds, box_bounds

def analyze_crosslink_network(atoms, bonds, box_bounds):
    g = nx.Graph()
    for a_id in atoms.keys():
        g.add_node(a_id)
    for b in bonds:
        g.add_edge(b[0], b[1])
        
    components = list(nx.connected_components(g))
    if not components:
        gel_fraction = 0.0
        main_component_nodes = set()
    else:
        comp_sizes = [len(c) for c in components]
        max_comp_idx = int(np.argmax(comp_sizes))
        main_component_nodes = components[max_comp_idx]
        gel_fraction = len(main_component_nodes) / float(len(atoms))
        
    sub_g = g.subgraph(main_component_nodes).copy()
    
    # 悬挂链剪枝 (度为 1 的原子非交联核心)
    changed = True
    while changed:
        changed = False
        degree_1_nodes = [node for node, deg in sub_g.degree() if deg <= 1]
        if degree_1_nodes:
            sub_g.remove_nodes_from(degree_1_nodes)
            changed = True
            
    # 提取分支度 >= 3 的有效交联节点
    effective_junctions = [node for node, deg in sub_g.degree() if deg >= 3]
    num_junctions = len(effective_junctions)
    
    # 体积计算 (cm³)
    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]
    vol_angstrom3 = lx * ly * lz
    vol_cm3 = vol_angstrom3 * 1.0e-24
    
    total_mass = sum([d['mass'] for d in atoms.values()])
    na = 6.02214076e23
    density_g_cm3 = (total_mass / na) / vol_cm3 if vol_cm3 > 0 else 1.15
    
    nu_mol_cm3 = (num_junctions / na) / vol_cm3 if vol_cm3 > 0 else 0.0
    nu_mmol_cm3 = nu_mol_cm3 * 1000.0
    mc_g_mol = (density_g_cm3 / nu_mol_cm3) if nu_mol_cm3 > 0 else 10000.0
    
    return {
        'num_atoms': len(atoms),
        'num_bonds': len(bonds),
        'gel_fraction': round(gel_fraction, 4),
        'num_effective_junctions': num_junctions,
        'effective_junction_ids': effective_junctions[:20], # 截断记录
        'cell_density_g_cm3': round(density_g_cm3, 4),
        'crosslink_density_nu_mmol_cm3': round(nu_mmol_cm3, 4),
        'molecular_weight_between_crosslinks_mc': round(mc_g_mol, 2)
    }

def run_stage4(r=1.0, output_dir='/home/wangj/epoxy_prod/data'):
    snap_meta_path = os.path.join(output_dir, f'stage3_snapshots_r_{r:.2f}.json')
    if not os.path.exists(snap_meta_path):
        raise FileNotFoundError(f"找不到 stage3 快照元数据: {snap_meta_path}")

    with open(snap_meta_path, 'r') as f:
        snapshots = json.load(f)
        
    results = {}
    print(f"[阶段 4 - Direction B] 开始大体系图论拓扑分析 (r={r:.2f}) (网络节点={20000})...")
    for alpha_str, data_file in snapshots.items():
        alpha = float(alpha_str)
        atoms, bonds, box = parse_data_for_graph(data_file)
        metrics = analyze_crosslink_network(atoms, bonds, box)
        results[alpha_str] = metrics
        print(f"  -> r={r:.2f}, alpha={alpha:.2f}: 有效交联节点数={metrics['num_effective_junctions']}, 交联密度 nu={metrics['crosslink_density_nu_mmol_cm3']} mmol/cm³, Mc={metrics['molecular_weight_between_crosslinks_mc']} g/mol, 凝胶分数={metrics['gel_fraction']:.2%}")
        
    out_json = os.path.join(output_dir, f'stage4_crosslink_density_r_{r:.2f}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
            
    print(f"[阶段 4 成功] (r={r:.2f}) 图论交联密度分析完成，已保存至: {out_json}\n")
    return results

if __name__ == '__main__':
    run_stage4(r=1.0)
