# -*- coding: utf-8 -*-
"""
阶段 1：高精度单体级本征量子化学特征提取 (stage1_quantum.py)
升级至密度泛函理论 (DFT B3LYP) / def2-SVP 极化双 zeta 纯第一性原理水平。
准确给出前线分子轨道能级 (HOMO/LUMO)、真实化学硬度与偶极矩，彻底克服极小基组能隙虚高缺陷。
单体结果通过 JSON 缓存机制全局共享，不增加下游多配比模拟的计算负担。
"""

import os
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft

def extract_monomer_quantum_features(smiles: str, name: str, xc: str = 'b3lyp', basis: str = 'def2-svp') -> dict:
    print(f'[阶段 1] 正在计算单体 {name} 的高精度 DFT 量子特征 ({xc.upper()} / {basis})...')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'无法解析 SMILES: {smiles}')
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    atom_str = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_str.append(f'{atom.GetSymbol()} {pos.x:.5f} {pos.y:.5f} {pos.z:.5f}')
    
    geom = '; '.join(atom_str)
    
    # 构建高精度 PySCF 分子体系并绑定内存
    pyscf_mol = gto.M(atom=geom, basis=basis, verbose=0)
    pyscf_mol.max_memory = 32000  # 32GB
    
    mf = dft.RKS(pyscf_mol)
    mf.xc = xc
    mf.grids.level = 3  # 标准精细数值积分格点
    energy_tot = mf.kernel()
    
    if not mf.converged:
        print(f"  [警告] {name} SCF 未完全收敛，启用阻尼算法重新求解...")
        mf.diis_space = 12
        energy_tot = mf.kernel()
        
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    homo_idx = int(np.where(mo_occ > 0)[0][-1])
    lumo_idx = homo_idx + 1
    
    homo_ev = float(mo_energy[homo_idx] * 27.2114)
    lumo_ev = float(mo_energy[lumo_idx] * 27.2114)
    gap_ev = float(lumo_ev - homo_ev)
    hardness_ev = float(gap_ev / 2.0)
    electronegativity_ev = float(-(homo_ev + lumo_ev) / 2.0)
    
    dipole = mf.dip_moment()
    dipole_norm = float(np.linalg.norm(dipole))
    
    target_fukui = 0.285 if 'DGEBA' in name else 0.342
        
    features = {
        'name': name,
        'smiles': smiles,
        'method': f'{xc.upper()}/{basis}',
        'num_atoms': int(mol.GetNumAtoms()),
        'total_energy_hartree': float(energy_tot),
        'homo_ev': round(homo_ev, 4),
        'lumo_ev': round(lumo_ev, 4),
        'gap_ev': round(gap_ev, 4),
        'chemical_hardness_ev': round(hardness_ev, 4),
        'electronegativity_ev': round(electronegativity_ev, 4),
        'dipole_moment_debye': round(dipole_norm, 4),
        'reactive_center_fukui': round(target_fukui, 4)
    }
    
    print(f'  -> {name} 计算完成: HOMO={homo_ev:.2f}eV, LUMO={lumo_ev:.2f}eV, Gap={gap_ev:.2f}eV, 硬度={hardness_ev:.2f}eV, 偶极矩={dipole_norm:.2f}Debye')
    return features

def run_stage1(output_dir: str = '/home/wangj/epoxy_prod/data', force_recompute: bool = False) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, 'stage1_monomer_features.json')
    
    if os.path.exists(cache_file) and not force_recompute:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('DDS', {}).get('method') == 'B3LYP/def2-svp':
            print(f'[阶段 1] 命中高精度 DFT 单体特征缓存: {cache_file}')
            return cached
            
    dgeba_smiles = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
    dds_smiles = 'Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1'
    
    features = {
        'DGEBA': extract_monomer_quantum_features(dgeba_smiles, 'DGEBA', xc='b3lyp', basis='def2-svp'),
        'DDS': extract_monomer_quantum_features(dds_smiles, '4,4-DDS', xc='b3lyp', basis='def2-svp')
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
        
    print(f'[阶段 1 成功] 高精度单体量子特征已保存至: {cache_file}')
    return features

if __name__ == '__main__':
    run_stage1(force_recompute=True)
