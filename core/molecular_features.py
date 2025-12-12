# -*- coding: utf-8 -*-
"""分子特征工程模块 - 完整5种提取方法 + 分子指纹 (高性能优化版)"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from rdkit.Chem import MACCSkeys
from tqdm import tqdm
import warnings
import torch
import os  # 新增

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import MACCSkeys

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


# =============================================================================
# 辅助函数：3D 构象生成 (用于多进程)
# =============================================================================
def _generate_3d_data_worker(smiles):
    """
    单个分子的3D生成工作函数
    返回: (atomic_numbers, coordinates) 或 None
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        # 1. 基础转换
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)  # 力场计算必须加氢

        # 2. 生成3D构象
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.numThreads = 1  # 禁用 RDKit 内部线程

        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
            if res != 0:
                return None

        # 3. 初步力场优化 (MMFF)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
        except:
            pass

        # 4. 提取数据
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()

        supported_species = {1, 6, 7, 8, 16, 9, 17}
        if not set(atoms).issubset(supported_species):
            return None

        return (atoms, coords)

    except Exception:
        return None


class RDKitFeatureExtractor:
    """RDKit基础提取器"""

    def __init__(self):
        self.feature_names = None

    def smiles_to_rdkit_features(self, smiles_list):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")

        features_list, valid_indices = [], []
        descriptor_funcs = dict(Descriptors.descList)

        for idx, smiles in enumerate(tqdm(smiles_list, desc="RDKit提取")):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                features = {}
                for name, func in descriptor_funcs.items():
                    try:
                        val = func(mol)
                        features[name] = val if np.isfinite(val) else np.nan
                    except:
                        features[name] = np.nan
                features_list.append(features)
                valid_indices.append(idx)
            except:
                continue

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        self.feature_names = df.columns.tolist()
        return df, valid_indices


class OptimizedRDKitFeatureExtractor:
    """并行版RDKit提取器"""

    def __init__(self, n_jobs=-1, batch_size=1000):
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.batch_size = batch_size
        self.feature_names = None

    @staticmethod
    def _process_batch(args):
        start_idx, smiles_list = args
        if not RDKIT_AVAILABLE:
            return [], []

        descriptor_funcs = dict(Descriptors.descList)
        features_list, valid_indices = [], []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                features = {}
                for name, func in descriptor_funcs.items():
                    try:
                        val = func(mol)
                        features[name] = val if np.isfinite(val) else np.nan
                    except:
                        features[name] = np.nan
                features_list.append(features)
                valid_indices.append(start_idx + i)
            except:
                continue
        return features_list, valid_indices

    def smiles_to_rdkit_features(self, smiles_list):
        batches = [(i, smiles_list[i:i + self.batch_size])
                   for i in range(0, len(smiles_list), self.batch_size)]

        all_features, all_indices = [], []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for features, indices in executor.map(self._process_batch, batches):
                all_features.extend(features)
                all_indices.extend(indices)

        if not all_features:
            return pd.DataFrame(), []

        df = pd.DataFrame(all_features)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        self.feature_names = df.columns.tolist()
        return df, all_indices


class MemoryEfficientRDKitExtractor:
    """内存优化版提取器"""

    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.feature_names = None

    def smiles_to_rdkit_features(self, smiles_list):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")

        all_features, all_indices = [], []
        descriptor_funcs = dict(Descriptors.descList)

        for batch_start in tqdm(range(0, len(smiles_list), self.batch_size), desc="内存优化提取"):
            batch = smiles_list[batch_start:batch_start + self.batch_size]
            for i, smiles in enumerate(batch):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    features = {}
                    for name, func in descriptor_funcs.items():
                        try:
                            val = func(mol)
                            features[name] = val if np.isfinite(val) else np.nan
                        except:
                            features[name] = np.nan
                    all_features.append(features)
                    all_indices.append(batch_start + i)
                except:
                    continue

        if not all_features:
            return pd.DataFrame(), []

        df = pd.DataFrame(all_features)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0]
        df = df.fillna(df.median())

        self.feature_names = df.columns.tolist()
        return df, all_indices


class AdvancedMolecularFeatureExtractor:
    """高级分子特征提取器"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装rdkit")
        self.descriptor_names = []

    def _smiles_to_mol(self, smiles):
        try:
            if pd.isna(smiles):
                return None
            return Chem.MolFromSmiles(str(smiles))
        except:
            return None

    def _process_result(self, features, indices, is_df=False):
        if not features:
            return pd.DataFrame(), []

        if is_df:
            df = features
        else:
            df = pd.DataFrame(features)

        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.var() > 0] if len(df) > 0 else df
        df = df.fillna(df.median())
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        return df, indices

    def smiles_to_rdkit_features(self, smiles_list):
        all_features, valid_indices = [], []
        descriptor_funcs = {name: func for name, func in Descriptors.descList}

        print(f"\n🧬 RDKit特征提取")
        for idx, smiles in enumerate(tqdm(smiles_list, desc="提取中")):
            mol = self._smiles_to_mol(smiles)
            if mol is None:
                continue
            features = {}
            for name, func in descriptor_funcs.items():
                try:
                    val = func(mol)
                    features[name] = val if np.isfinite(val) else np.nan
                except:
                    features[name] = np.nan
            all_features.append(features)
            valid_indices.append(idx)

        return self._process_result(all_features, valid_indices)

    def smiles_to_mordred(self, smiles_list, batch_size=1000):
        """
        Mordred特征提取 - 优化版
        增加了分批处理和Windows环境下的稳定性保护
        """
        if not MORDRED_AVAILABLE:
            raise ImportError("需要安装mordred")

        print(f"\n🔬 Mordred特征提取")

        # 1. 预处理分子
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(tqdm(smiles_list, desc="预处理分子结构")):
            mol = self._smiles_to_mol(smiles)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)

        if not mols:
            return pd.DataFrame(), []

        # 2. 初始化计算器
        calc = Calculator(descriptors, ignore_3D=True)

        # 3. 智能选择进程数
        # Windows 下多进程极其不稳定，强制使用单进程
        is_windows = os.name == 'nt'
        if is_windows:
            print("⚠️ 检测到 Windows 系统，强制使用单进程模式以确保稳定（可能会慢一些）。")
            n_proc = 1
        else:
            n_proc = mp.cpu_count()

        # 4. 分批计算 (Batch Processing)
        # 即使是单进程，分批也能让进度条动起来，并防止内存溢出
        all_dfs = []
        total_mols = len(mols)

        # 主进度条
        pbar = tqdm(total=total_mols, desc="计算Mordred描述符")

        for i in range(0, total_mols, batch_size):
            batch_mols = mols[i: i + batch_size]

            try:
                # 尝试计算当前批次
                # quiet=True 是为了防止 mordred 内部再打印一个进度条干扰我们
                if n_proc > 1:
                    try:
                        df_batch = calc.pandas(batch_mols, n_proc=n_proc, quiet=True)
                    except Exception as e:
                        # 如果并行失败，降级重试该批次
                        if i == 0:
                            print(f"\n⚠️ 并行计算出错 ({str(e)})，自动切换回单进程模式...")
                        n_proc = 1
                        df_batch = calc.pandas(batch_mols, n_proc=1, quiet=True)
                else:
                    df_batch = calc.pandas(batch_mols, n_proc=1, quiet=True)

                all_dfs.append(df_batch)

            except Exception as e:
                print(f"\n❌ 批次 {i // batch_size + 1} 计算失败: {str(e)}")
                # 如果某批次彻底失败，插入全NaN行以保持索引对齐
                empty_df = pd.DataFrame(index=range(len(batch_mols)), columns=[str(d) for d in calc.descriptors])
                all_dfs.append(empty_df)

            finally:
                pbar.update(len(batch_mols))

        pbar.close()

        if not all_dfs:
            return pd.DataFrame(), []

        # 5. 合并与后处理
        try:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # 强制转为数值，非数值转为 NaN
            final_df = final_df.apply(pd.to_numeric, errors='coerce')
            return self._process_result(final_df, valid_indices, is_df=True)
        except Exception as e:
            print(f"❌ 结果合并失败: {str(e)}")
            return pd.DataFrame(), []

    def smiles_to_graph_features(self, smiles_list):
        all_features, valid_indices = [], []

        print(f"\n🕸️ 图特征提取")
        for idx, smiles in enumerate(tqdm(smiles_list, desc="构建图")):
            mol = self._smiles_to_mol(smiles)
            if mol is None:
                continue

            try:
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()
                features = {
                    'graph_num_nodes': num_atoms,
                    'graph_num_edges': num_bonds,
                    'graph_avg_degree': 2 * num_bonds / num_atoms if num_atoms > 0 else 0,
                    'graph_density': num_bonds / (num_atoms * (num_atoms - 1) / 2) if num_atoms > 1 else 0,
                    'num_rings': Chem.GetSSSR(mol).__len__(),
                    'num_aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
                    'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'mol_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                }
                all_features.append(features)
                valid_indices.append(idx)
            except:
                continue

        return self._process_result(all_features, valid_indices)


class MLForceFieldExtractor:
    """机器学习力场特征提取器"""

    def __init__(self, device=None):
        try:
            import torchani
            import torch
            self.torch = torch
            self.torchani = torchani
            self.AVAILABLE = True
        except ImportError:
            self.AVAILABLE = False
            self.feature_names = []
            return

        if device is None:
            self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        try:
            self.model = self.torchani.models.ANI2x().to(self.device)
        except Exception as e:
            print(f"ANI Model load error: {e}")
            self.AVAILABLE = False

        self.feature_names = ['ani_energy', 'ani_energy_per_atom', 'ani_max_force', 'ani_mean_force', 'ani_force_std']

    def smiles_to_ani_features(self, smiles_list, batch_size=32):
        if not self.AVAILABLE:
            raise ImportError("请先安装 torchani: pip install torchani")

        print(f"\n⚛️ 正在并行生成 3D 构象 (这可能需要一些时间)...")

        valid_indices = []
        data_list = []

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_generate_3d_data_worker, smiles_list),
                                total=len(smiles_list),
                                desc="3D Generation"))

        for i, res in enumerate(results):
            if res is not None:
                valid_indices.append(i)
                data_list.append(res)

        if not data_list:
            return pd.DataFrame(), []

        print(f"⚛️ 开始 ANI 批量推理 (Batch Size: {batch_size}, Device: {self.device})...")
        features_list = []

        for i in tqdm(range(0, len(data_list), batch_size), desc="Inference"):
            batch_data = data_list[i: i + batch_size]
            species_list = [self.torch.tensor(d[0], dtype=self.torch.long) for d in batch_data]
            coords_list = [self.torch.tensor(d[1], dtype=self.torch.float32) for d in batch_data]

            species_padded = self.torch.nn.utils.rnn.pad_sequence(species_list, batch_first=True, padding_value=-1).to(
                self.device)
            coords_padded = self.torch.nn.utils.rnn.pad_sequence(coords_list, batch_first=True, padding_value=0.0).to(
                self.device)
            coords_padded.requires_grad_(True)
            mask = (species_padded >= 0)

            try:
                species_safe = species_padded.clone()
                species_safe[~mask] = 0

                energy = self.model((species_safe, coords_padded)).energies
                forces = -self.torch.autograd.grad(energy.sum(), coords_padded, create_graph=False, retain_graph=False)[
                    0]

                energy_np = energy.detach().cpu().numpy()
                forces_np = forces.detach().cpu().numpy()

                _, atomic_energies = self.model((species_safe, coords_padded))
                real_energy = (atomic_energies * mask.float()).sum(dim=1).detach().cpu().numpy()

                for j in range(len(batch_data)):
                    n_atoms = len(batch_data[j][0])
                    e_val = real_energy[j]
                    f_vec = forces_np[j][:n_atoms]
                    f_norm = np.linalg.norm(f_vec, axis=1)

                    feats = {
                        'ani_energy': e_val,
                        'ani_energy_per_atom': e_val / n_atoms,
                        'ani_max_force': np.max(f_norm),
                        'ani_mean_force': np.mean(f_norm),
                        'ani_force_std': np.std(f_norm)
                    }
                    features_list.append(feats)

            except Exception as e:
                print(f"Batch error: {e}, processing individually...")
                for d in batch_data:
                    features_list.append({k: np.nan for k in self.feature_names})

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        return df, valid_indices


class EpoxyDomainFeatureExtractor:
    """环氧树脂领域知识特征提取器 (增强版：加入电子效应模拟)"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")

    def _get_epoxide_count(self, mol):
        patt = Chem.MolFromSmarts("[C]1[O][C]1")
        matches = mol.GetSubstructMatches(patt)
        return len(matches)

    def _get_active_hydrogen_count(self, mol):
        count = 0
        for atom in mol.GetAtoms():
            # 计算与氮原子相连的氢原子数 (胺类固化剂)
            if atom.GetAtomicNum() == 7:
                count += atom.GetTotalNumHs()
        return count

    def _calc_electronic_props(self, mol):
        """计算电子性质 (作为DFT的低成本替代)"""
        try:
            # 计算 Gasteiger 部分电荷
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                # 获取计算出的电荷
                c = atom.GetProp('_GasteigerCharge')
                # 有些原子可能无法计算，返回inf或nan
                if c and not c.lower().startswith('nan') and not c.lower().startswith('inf'):
                    charges.append(float(c))

            if not charges:
                return 0.0, 0.0, 0.0

            max_pos_charge = max(charges)  # 亲电性指标
            max_neg_charge = min(charges)  # 亲核性指标

            # 拓扑极性表面积 (TPSA) - 表征分子极性
            tpsa = Descriptors.TPSA(mol)

            return max_pos_charge, max_neg_charge, tpsa
        except Exception:
            return 0.0, 0.0, 0.0

    def extract_features(self, resin_smiles_list, hardener_smiles_list, stoichiometry_list=None):
        features_list = []
        valid_indices = []

        if len(resin_smiles_list) != len(hardener_smiles_list):
            return pd.DataFrame(), []

        # 遍历每对样本
        for idx, (smi_r, smi_h) in enumerate(zip(resin_smiles_list, hardener_smiles_list)):
            try:
                mol_r = Chem.MolFromSmiles(str(smi_r))
                mol_h = Chem.MolFromSmiles(str(smi_h))

                if mol_r is None or mol_h is None:
                    continue

                # 1. 基础化学计量特征 (原有功能)
                mw_r = Descriptors.MolWt(mol_r)
                mw_h = Descriptors.MolWt(mol_h)
                f_epoxy = self._get_epoxide_count(mol_r)
                f_amine = self._get_active_hydrogen_count(mol_h)

                eew = mw_r / f_epoxy if f_epoxy > 0 else mw_r
                ahew = mw_h / f_amine if f_amine > 0 else mw_h

                # 计算理论配比 (phr)
                theo_phr = (ahew / eew) * 100 if eew > 0 else 0

                # 2. 电子性质特征 (新增功能 - 模拟DFT)
                r_pos_chg, r_neg_chg, r_tpsa = self._calc_electronic_props(mol_r)
                h_pos_chg, h_neg_chg, h_tpsa = self._calc_electronic_props(mol_h)

                features = {
                    'EEW': eew,
                    'AHEW': ahew,
                    'Resin_Functionality': f_epoxy,
                    'Hardener_Functionality': f_amine,
                    'Theoretical_PHR': theo_phr,
                    # 新增特征列
                    'Resin_Max_Pos_Charge': r_pos_chg,
                    'Resin_Max_Neg_Charge': r_neg_chg,
                    'Resin_TPSA': r_tpsa,
                    'Hardener_Max_Pos_Charge': h_pos_chg,
                    'Hardener_TPSA': h_tpsa
                }

                features_list.append(features)
                valid_indices.append(idx)

            except Exception:
                continue

        if not features_list:
            return pd.DataFrame(), []

        return pd.DataFrame(features_list), valid_indices


class FingerprintExtractor:
    """分子指纹提取器：支持 MACCS Keys 和 Morgan Fingerprints (支持双组分拼接)"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")

    def _gen_fp_array(self, mol, fp_type, n_bits, radius):
        """辅助函数：生成单个分子的指纹数组"""
        if fp_type == 'MACCS':
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        elif fp_type == 'Morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        return np.array([])

    def smiles_to_fingerprints(self, smiles_list, smiles_list_2=None, fp_type='MACCS', n_bits=2048, radius=2):
        """
        提取分子指纹。
        Args:
            smiles_list: 树脂/第一组分 SMILES
            smiles_list_2: (可选) 固化剂/第二组分 SMILES。如果提供，将拼接两个指纹。
        """
        all_fps = []
        valid_indices = []

        # 判断是否需要双组分拼接
        is_dual = smiles_list_2 is not None and len(smiles_list_2) == len(smiles_list)

        desc_str = f"提取 {fp_type} 指纹"
        if is_dual:
            desc_str += " (双组分拼接: Resin + Hardener)"

        print(f"\n👆 {desc_str}")

        for idx, smi1 in enumerate(tqdm(smiles_list, desc="指纹提取")):
            try:
                # 1. 处理第一个分子
                mol1 = Chem.MolFromSmiles(str(smi1))
                if mol1 is None:
                    continue

                feat_dict = {}

                # 生成指纹 1
                fp1_arr = self._gen_fp_array(mol1, fp_type, n_bits, radius)
                for i, val in enumerate(fp1_arr):
                    # 特征名加前缀区分
                    feat_dict[f"Resin_{fp_type}_{i}"] = val

                # 2. 处理第二个分子 (如果有)
                if is_dual:
                    smi2 = smiles_list_2[idx]
                    mol2 = Chem.MolFromSmiles(str(smi2))
                    if mol2 is None:
                        # 如果固化剂SMILES无效，您可以选择跳过该样本，或者填0
                        # 这里选择跳过，保证数据质量
                        continue

                        # 生成指纹 2
                    fp2_arr = self._gen_fp_array(mol2, fp_type, n_bits, radius)
                    for i, val in enumerate(fp2_arr):
                        feat_dict[f"Hardener_{fp_type}_{i}"] = val

                all_fps.append(feat_dict)
                valid_indices.append(idx)

            except Exception as e:
                continue

        if not all_fps:
            return pd.DataFrame(), []

        # 转为 DataFrame 并优化内存
        df = pd.DataFrame(all_fps)
        df = df.astype(np.uint8)

        # 移除全为0的列 (无信息量的位)
        df = df.loc[:, (df != 0).any(axis=0)]

        return df, valid_indices

# =============================================================================
# [新增] MACCS 键定义字典 (用于解释器)
# =============================================================================
MACCS_DEFINITIONS = {
    1: "ISOTOPE", 2: "Atomic no > 103", 3: "Group IVa,Va,VIa Rows 4-6", 4: "Actinides", 
    5: "Group IIIA,IVA", 6: "Lanthanides", 7: "Group VA,VIA Rows 4-6", 8: "QAAA@1", 
    9: "Group VIII (Fe...)", 10: "Group IIA", 11: "4M Ring", 12: "Group IB,IIB", 
    13: "ON(C)C", 14: "S-S", 15: "OC(O)O", 16: "Q:Q", 17: "C#C", 18: "Group IIIA", 
    19: "7M Ring", 20: "Si", 21: "C=C(Q)Q", 22: "3M Ring", 23: "NC(O)O", 24: "N-O", 
    25: "NC(N)N", 26: "C$=C($)C($)C", 27: "I", 28: "QCH2Q", 29: "P", 30: "CQ(C)(C)A", 
    31: "QX", 32: "CSN", 33: "NS", 34: "CH2=A", 35: "Group IA", 36: "S Heterocycle", 
    37: "NC(O)N", 38: "NC(C)N", 39: "OS(O)O", 40: "S-O", 41: "C#N", 42: "F", 43: "QHAQH", 
    44: "Other", 45: "C=CN", 46: "Br", 47: "SAN", 48: "OQ(O)O", 49: "C=C", 50: "C=C(C)C", 
    51: "CSO", 52: "NN", 53: "CN(C)C", 54: "C=C(O)C", 55: "OSO", 56: "ON(O)C", 
    57: "O Heterocycle", 58: "QSQ", 59: "Snot%A%A", 60: "S=O", 61: "AS(A)A", 
    62: "A$A!A$A", 63: "N=O", 64: "A-S", 65: "C%N", 66: "CC(C)(C)C", 67: "QSQ", 
    68: "QHQH (&...)", 69: "QQH", 70: "Q-N-Q", 71: "NO", 72: "O-A", 73: "S=A", 
    74: "CH3ACH3", 75: "A!N$A", 76: "C=C(O)O", 77: "NAN", 78: "C=N", 79: "N$A$N", 
    80: "NAAAN", 81: "SA(A)A", 82: "ACH2QA", 83: "QAA@1", 84: "NH2", 85: "CN(C)Q", 
    86: "CH2QCH2", 87: "X!A$A", 88: "S", 89: "OAAAO", 90: "QHAAQH", 91: "QHAAQH", 
    92: "OC(N)C", 93: "QCH3", 94: "QN", 95: "NAAO", 96: "5M Ring", 97: "N A A O", 
    98: "QAAAA@1", 99: "C=C", 100: "ACH2N", 101: "8M Ring", 102: "QO", 103: "Cl", 
    104: "QA(Q)Q", 105: "A$A($)A", 106: "QA(Q)Q", 107: "X (Halogen)", 108: "CH3AAACH2", 
    109: "ACH2O", 110: "NCO", 111: "NAAOH", 112: "AA(A)(A)A", 113: "Onot%A%A", 
    114: "CH3CH2A", 115: "CH3ACH2", 116: "CH3AAO", 117: "NAO", 118: "ACH2CH2A > 1", 
    119: "N=A", 120: "Heterocyclic atom > 1", 121: "N Heterocycle", 122: "AN(A)A", 
    123: "OCO", 124: "QQ", 125: "Aromatic Ring > 1", 126: "A!O!A", 127: "A$A!O > 1", 
    128: "ACH2A > 1", 129: "ACH2A", 130: "QQ > 1", 131: "QH > 1", 132: "OH > 1", 
    133: "A@A!A", 134: "X (Halogen)", 135: "Nnot%A%A", 136: "O=A > 1", 137: "Heterocycle", 
    138: "QCH2Q > 1", 139: "OH", 140: "O > 3", 141: "CH3 > 2", 142: "N > 1", 
    143: "A$A!A$A", 144: "Anot%A%A", 145: "6M ring > 1", 146: "O > 2", 147: "ACH2CH2A", 
    148: "AQ(A)A", 149: "CH3 > 1", 150: "A!A$A!A", 151: "NH", 152: "OC(C)C", 
    153: "QCH2Q", 154: "C=O", 155: "A!CH2!A", 156: "NA(A)A", 157: "C-O", 158: "C-N", 
    159: "O > 1", 160: "CH3", 161: "N", 162: "Aromatic", 163: "6M Ring", 164: "O", 
    165: "Ring", 166: "Fragments"
}

def get_maccs_description(key_idx):
    """根据键索引获取 MACCS 描述"""
    try:
        idx = int(key_idx)
        return MACCS_DEFINITIONS.get(idx, "Unknown Fragment")
    except:
        return "Invalid Key"
