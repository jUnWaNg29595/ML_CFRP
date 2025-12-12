# -*- coding: utf-8 -*-
"""分子特征工程模块 - 完整5种提取方法 + 分子指纹 (高性能优化版)"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import MACCSkeys  # 导入MACCS Keys

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

        # 2. 生成3D构象 (尝试不同参数以提高成功率)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.numThreads = 1  # 禁用 RDKit 内部线程，避免与多进程冲突

        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            # 备用方案
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
            if res != 0:
                return None

        # 3. 初步力场优化 (MMFF)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=50)  # 减少迭代次数以提升速度
        except:
            pass

        # 4. 提取数据
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()

        # 简单过滤：ANI-2x 只支持 H, C, N, O, S, F, Cl
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

    def smiles_to_mordred(self, smiles_list):
        if not MORDRED_AVAILABLE:
            raise ImportError("需要安装mordred")

        print(f"\n🔬 Mordred特征提取 (并行模式)")
        n_cpu = mp.cpu_count()
        mols = []
        valid_indices = []

        for idx, smiles in enumerate(tqdm(smiles_list, desc="预处理分子结构")):
            mol = self._smiles_to_mol(smiles)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)

        if not mols:
            return pd.DataFrame(), []

        calc = Calculator(descriptors, ignore_3D=True)
        try:
            df = calc.pandas(mols, n_proc=n_cpu, quiet=False)
        except:
            print("并行计算失败，回退到单进程...")
            df = calc.pandas(mols, quiet=False)

        df = df.apply(pd.to_numeric, errors='coerce')
        return self._process_result(df, valid_indices, is_df=True)

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
    """环氧树脂领域知识特征提取器"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")

    def _get_epoxide_count(self, mol):
        patt = Chem.MolFromSmarts("[C]1[O][C]1")
        return len(mol.GetSubstructMatches(patt))

    def _get_active_hydrogen_count(self, mol):
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                count += atom.GetTotalNumHs()
        return count

    def _calc_rigidity(self, mol, mw):
        num_aromatic = Descriptors.NumAromaticRings(mol)
        aromatic_density = num_aromatic / mw if mw > 0 else 0
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        rotatable_density = num_rotatable / mw if mw > 0 else 0
        return aromatic_density, rotatable_density

    def extract_features(self, resin_smiles_list, hardener_smiles_list, stoichiometry_list=None):
        features_list = []
        valid_indices = []

        if len(resin_smiles_list) != len(hardener_smiles_list):
            return pd.DataFrame(), []

        for idx, (smi_r, smi_h) in enumerate(zip(resin_smiles_list, hardener_smiles_list)):
            try:
                mol_r = Chem.MolFromSmiles(str(smi_r))
                mol_h = Chem.MolFromSmiles(str(smi_h))

                if mol_r is None or mol_h is None:
                    continue

                mw_r = Descriptors.MolWt(mol_r)
                mw_h = Descriptors.MolWt(mol_h)

                f_epoxy = self._get_epoxide_count(mol_r)
                f_amine = self._get_active_hydrogen_count(mol_h)

                eew = mw_r / f_epoxy if f_epoxy > 0 else mw_r
                ahew = mw_h / f_amine if f_amine > 0 else mw_h

                theo_phr = (ahew / eew) * 100 if eew > 0 else 0

                if stoichiometry_list is not None and idx < len(stoichiometry_list):
                    actual_phr = stoichiometry_list[idx]
                    stoich_deviation = actual_phr / theo_phr if theo_phr > 0 else 0
                else:
                    stoich_deviation = 1.0

                if f_amine > 0 and (mw_r + mw_h) > 0:
                    mass_unit = mw_r + (mw_h * (f_epoxy / f_amine))
                    xd_proxy = f_epoxy / mass_unit
                else:
                    xd_proxy = 0

                r_aro, r_rot = self._calc_rigidity(mol_r, mw_r)
                h_aro, h_rot = self._calc_rigidity(mol_h, mw_h)

                total_mass = mw_r + mw_h
                avg_aromatic_density = (r_aro * mw_r + h_aro * mw_h) / total_mass

                features = {
                    'EEW': eew,
                    'AHEW': ahew,
                    'Resin_Functionality': f_epoxy,
                    'Hardener_Functionality': f_amine,
                    'Theoretical_PHR': theo_phr,
                    'Stoich_Deviation': stoich_deviation,
                    'Crosslink_Density_Proxy': xd_proxy * 1000,
                    'System_Aromatic_Density': avg_aromatic_density,
                    'Resin_Rotatable_Density': r_rot
                }

                features_list.append(features)
                valid_indices.append(idx)

            except Exception:
                continue

        if not features_list:
            return pd.DataFrame(), []

        return pd.DataFrame(features_list), valid_indices


class FingerprintExtractor:
    """分子指纹提取器：支持 MACCS Keys 和 Morgan Fingerprints"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 rdkit")

    def smiles_to_fingerprints(self, smiles_list, fp_type='MACCS', n_bits=2048, radius=2):
        """
        提取分子指纹
        Args:
            smiles_list: SMILES 字符串列表
            fp_type: 'MACCS' 或 'Morgan'
            n_bits: Morgan指纹的位长 (仅对Morgan有效)
            radius: Morgan指纹的半径 (仅对Morgan有效)
        """
        all_fps = []
        valid_indices = []

        desc_str = f"提取 {fp_type} 指纹"
        if fp_type == 'Morgan':
            desc_str += f" (r={radius}, b={n_bits})"

        print(f"\n👆 {desc_str}")

        for idx, smiles in enumerate(tqdm(smiles_list, desc="Processing")):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue

                fp_array = None

                if fp_type == 'MACCS':
                    # MACCS Keys: 167 bits
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    # 转换为 numpy array
                    fp_array = np.array(fp)  # 0-166

                    # 创建特征名字典 (MACCS_0 ... MACCS_166)
                    # 注意：MACCS keys 索引从 1 开始有意义，index 0 通常为 0
                    features = {f"MACCS_{i}": val for i, val in enumerate(fp_array)}

                elif fp_type == 'Morgan':
                    # Morgan (ECFP like): bit vector
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fp_array = np.array(fp)

                    features = {f"Morgan_{i}": val for i, val in enumerate(fp_array)}

                else:
                    continue

                all_fps.append(features)
                valid_indices.append(idx)

            except Exception as e:
                # print(e)
                continue

        if not all_fps:
            return pd.DataFrame(), []

        # 转为 DataFrame
        df = pd.DataFrame(all_fps)

        # 优化内存：指纹是0/1，可以使用 uint8
        df = df.astype(np.uint8)

        # 移除全为0的列 (无信息量的位)
        df = df.loc[:, (df != 0).any(axis=0)]

        return df, valid_indices