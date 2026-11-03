# -*- coding: utf-8 -*-
"""
阶段 7：生产级多尺度特征融合与结构化数据集输出 (stage7_dataset_export.py)
整合阶段 1-6 的全部物理/化学特征与物性标签，支持化学计量比 r (0.8, 1.0, 1.2)
与固化度 alpha (0.7, 0.8, 0.9) 2D 参数矩阵的全量样本结构化汇出。
"""

import os
import glob
import json
import pandas as pd

def run_stage7(ratios=None, output_dir='/home/wangj/epoxy_prod/data', export_dir='/home/wangj/epoxy_prod'):
    # 1. 读取单体量子化学本征特征 (Stage 1)
    stage1_file = os.path.join(output_dir, 'stage1_monomer_features.json')
    if not os.path.exists(stage1_file):
        raise FileNotFoundError(f"找不到阶段 1 单体特征文件: {stage1_file}")
    with open(stage1_file, 'r', encoding='utf-8') as f:
        m_feat = json.load(f)

    # 确定要处理的 ratios
    if ratios is None:
        meta_files = glob.glob(os.path.join(output_dir, 'system_meta_r*.json'))
        if meta_files:
            found_ratios = []
            for mf in meta_files:
                basename = os.path.basename(mf)
                try:
                    r_str = basename.replace('system_meta_r', '').replace('.json', '').replace('_', '.')
                    found_ratios.append(float(r_str))
                except ValueError:
                    pass
            ratios = sorted(found_ratios)
        else:
            ratios = [1.0]

    print(f"[阶段 7] 开始生产级多尺度特征聚合，待汇聚配比 r 列表: {ratios}")
    rows = []

    for r in ratios:
        suffix = f'_r{r:.2f}'.replace('.', '_')
        meta_file = os.path.join(output_dir, f'system_meta{suffix}.json')
        g_file = os.path.join(output_dir, f'stage4_crosslink_density_r_{r:.2f}.json')
        c_file = os.path.join(output_dir, f'stage5_cluster_td_r_{r:.2f}.json')
        p_file = os.path.join(output_dir, f'stage6_properties_r_{r:.2f}.json')

        if not (os.path.exists(g_file) and os.path.exists(c_file) and os.path.exists(p_file)):
            print(f"  [跳过] r={r:.2f} 的关联特征文件不完备，跳过该配比。")
            continue

        with open(g_file, 'r', encoding='utf-8') as f:
            graph_feat = json.load(f)
        with open(c_file, 'r', encoding='utf-8') as f:
            cluster_feat = json.load(f)
        with open(p_file, 'r', encoding='utf-8') as f:
            macro_props = json.load(f)

        phr_val = round((62.08 / 170.21) * 100.0 * r, 2)
        n_dgeba = 300
        n_dds = int(round(n_dgeba * r / 2.0))
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
                phr_val = meta_info.get('phr', phr_val)
                n_dgeba = meta_info.get('n_dgeba', n_dgeba)
                n_dds = meta_info.get('n_dds', n_dds)

        for alpha_str in sorted(graph_feat.keys()):
            alpha = float(alpha_str)
            g_data = graph_feat.get(alpha_str, graph_feat.get(f"{alpha:.2f}", graph_feat.get(f"{alpha:.1f}", {})))
            c_data = cluster_feat.get(alpha_str, cluster_feat.get(f"{alpha:.2f}", cluster_feat.get(f"{alpha:.1f}", {})))
            p_data = macro_props.get(alpha_str, macro_props.get(f"{alpha:.2f}", macro_props.get(f"{alpha:.1f}", {})))

            sample_id = f"DGEBA_DDS_prod_r{r:.2f}_a{alpha:.2f}".replace('.', '')

            row = {
                # 1. 体系与配方标识 (含显式参数 r)
                'sample_id': sample_id,
                'resin_name': 'DGEBA',
                'resin_smiles': m_feat['DGEBA']['smiles'],
                'hardener_name': '4,4-DDS',
                'hardener_smiles': m_feat['DDS']['smiles'],
                'stoichiometric_ratio_r': round(r, 2),
                'hardener_phr': phr_val,
                'n_dgeba_molecules': n_dgeba,
                'n_dds_molecules': n_dds,
                'molar_ratio_dgeba_to_dds': round(n_dgeba / n_dds, 3) if n_dds > 0 else 0.0,
                
                # 2. 阶段 1：单体级本征量子化学特征
                'resin_homo_ev': m_feat['DGEBA']['homo_ev'],
                'resin_lumo_ev': m_feat['DGEBA']['lumo_ev'],
                'resin_gap_ev': m_feat['DGEBA']['gap_ev'],
                'resin_hardness_ev': m_feat['DGEBA']['chemical_hardness_ev'],
                'resin_dipole_debye': m_feat['DGEBA']['dipole_moment_debye'],
                'resin_electrophilic_fukui': m_feat['DGEBA']['reactive_center_fukui'],
                
                'hardener_homo_ev': m_feat['DDS']['homo_ev'],
                'hardener_lumo_ev': m_feat['DDS']['lumo_ev'],
                'hardener_gap_ev': m_feat['DDS']['gap_ev'],
                'hardener_hardness_ev': m_feat['DDS']['chemical_hardness_ev'],
                'hardener_dipole_debye': m_feat['DDS']['dipole_moment_debye'],
                'hardener_nucleophilic_fukui': m_feat['DDS']['reactive_center_fukui'],
                
                # 3. 核心反应自变量
                'curing_conversion_alpha': alpha,
                
                # 4. 阶段 4：图论拓扑交联特征
                'density_g_cm3': g_data.get('cell_density_g_cm3', 1.15),
                'gel_fraction': g_data.get('gel_fraction', 0.0),
                'effective_crosslink_junctions_count': g_data.get('num_effective_junctions', 0),
                'crosslink_density_nu_mmol_cm3': g_data.get('crosslink_density_nu_mmol_cm3', 0.0),
                'mean_segment_mw_mc_g_mol': g_data.get('molecular_weight_between_crosslinks_mc', 1000.0),
                
                # 5. 阶段 5：凝聚态微团簇逆向切球弱键特征 (Td 代理特征)
                'cluster_atoms_count': c_data.get('total_cluster_atoms', 0),
                'cluster_gap_ev': c_data.get('cluster_gap_ev', 0.0),
                'weak_bond_bde_kcal_mol': c_data.get('weak_bond_dissociation_energy_kcal_mol', 0.0),
                'decomposition_temp_td_celsius': c_data.get('predicted_td_onset_celsius', 0.0),
                
                # 6. 阶段 6：聚集态宏观物性标签
                'glass_transition_temp_tg_celsius': p_data.get('glass_transition_temp_tg_celsius', 0.0),
                'fractional_free_volume_ffv': p_data.get('fractional_free_volume_ffv', 0.0),
                'youngs_modulus_gpa': p_data.get('youngs_modulus_gpa', 0.0),
                'toughness_strain_energy_mj_m3': p_data.get('toughness_strain_energy_mj_m3', 0.0)
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # 导出到 /home/wangj/epoxy_prod 与 Windows 工作区
    csv_path_wsl = os.path.join(export_dir, 'epoxy_dgeba_dds_prod_dataset.csv')
    df.to_csv(csv_path_wsl, index=False, encoding='utf-8-sig')

    win_path = '/mnt/c/Users/wangj/Desktop/CFRP系统/CFRP系统/epoxy_dgeba_dds_prod_dataset.csv'
    try:
        df.to_csv(win_path, index=False, encoding='utf-8-sig')
    except Exception:
        pass

    print(f"\n[阶段 7 成功] Direction B 生产级结构化特征数据集导出完成!")
    print(f"  -> WSL 导出路径: {csv_path_wsl}")
    print(f"  -> 样本总数: {len(df)}, 特征字段总数: {len(df.columns)}")
    print("\n=== Direction B 生产级数据集全量核心特征概览 ===")
    preview_cols = [
        'sample_id', 'stoichiometric_ratio_r', 'hardener_phr', 'curing_conversion_alpha',
        'crosslink_density_nu_mmol_cm3', 'mean_segment_mw_mc_g_mol',
        'glass_transition_temp_tg_celsius', 'youngs_modulus_gpa'
    ]
    if len(df) > 0:
        print(df[preview_cols].to_string(index=False))
    return csv_path_wsl

if __name__ == '__main__':
    run_stage7()
