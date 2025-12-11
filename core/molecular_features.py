# -*- coding: utf-8 -*-
"""分子特征工程模块 - 完整5种提取方法"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


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

    _DESCRIPTOR_FUNCS = None

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
        """RDKit标准提取"""
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
        """Mordred描述符提取 - 并行优化版"""
        if not MORDRED_AVAILABLE:
            raise ImportError("需要安装mordred")

        # 1. 并行化 SMILES -> Mol 转换
        # 使用 OptimizedRDKitFeatureExtractor 中的 batch 处理逻辑或简单的 map
        # 这里为了简单直接使用多进程池
        print(f"\n🔬 Mordred特征提取 (并行模式)")

        n_cpu = mp.cpu_count()
        mols = []
        valid_indices = []

        # 分批处理转换以节省内存
        batch_size = 1000
        total = len(smiles_list)

        # 定义转换辅助函数 (需放在类外或作为静态方法，这里简化逻辑)
        # 为避免 pickle 问题，我们在单线程做转换，但通常 Mordred 计算才是瓶颈
        # 如果 SMILES 转 Mol 很慢，也可以并行，但 Mordred 自带并行计算

        for idx, smiles in enumerate(tqdm(smiles_list, desc="预处理分子结构")):
            mol = self._smiles_to_mol(smiles)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)

        if not mols:
            return pd.DataFrame(), []

        # 2. 使用 Mordred 的并行计算能力
        # ignore_3D=True 大幅提升速度
        calc = Calculator(descriptors, ignore_3D=True)

        # [优化] 启用 n_proc 进行多进程计算
        # quiet=False 可以看到进度条
        try:
            df = calc.pandas(mols, n_proc=n_cpu, quiet=False)
        except:
            # 如果多进程报错（特定系统环境），回退到单进程
            print("并行计算失败，回退到单进程...")
            df = calc.pandas(mols, quiet=False)

        df = df.apply(pd.to_numeric, errors='coerce')

        return self._process_result(df, valid_indices, is_df=True)

    def smiles_to_graph_features(self, smiles_list):
        """图结构特征提取"""
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
    """
    机器学习力场特征提取器 (基于 TorchANI)
    提取特征：
    1. 势能 (Potential Energy)
    2. 原子平均受力 (Mean Atomic Force)
    3. 分子稳定性指标
    """

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

        # 自动选择设备
        if device is None:
            self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 加载预训练模型 ANI-2x (支持 H, C, N, O, S, F, Cl)
        # periodic=False 表示非周期性边界条件（气相分子）
        self.model = self.torchani.models.ANI2x().to(self.device)
        self.feature_names = ['ani_energy', 'ani_energy_per_atom', 'ani_max_force', 'ani_mean_force', 'ani_force_std']

    def _generate_3d_mol(self, smiles):
        """将SMILES转换为包含3D坐标的RDKit分子"""
        try:
            if not RDKIT_AVAILABLE:
                return None
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            mol = Chem.AddHs(mol)  # 力场计算必须加氢

            # 生成3D构象
            params = Chem.AllChem.ETKDGv3()
            params.useRandomCoords = True
            res = Chem.AllChem.EmbedMolecule(mol, params)

            if res != 0:  # 尝试备用方法
                res = Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if res != 0: return None

            # 简单的力场优化，确保构象合理
            try:
                Chem.AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass  # 如果MMFF失败，使用原始嵌入坐标

            return mol
        except:
            return None

    def smiles_to_ani_features(self, smiles_list):
        if not self.AVAILABLE:
            raise ImportError("请先安装 torchani: pip install torchani")

        features_list = []
        valid_indices = []

        # 元素映射 ANI-2x: {H:1, C:6, N:7, O:8, S:16, F:9, Cl:17}
        supported_species = {1, 6, 7, 8, 16, 9, 17}

        print(f"\n⚛️ 机器学习力场(ANI)特征提取 (Device: {self.device})...")

        for idx, smiles in enumerate(tqdm(smiles_list, desc="ANI Inference")):
            mol = self._generate_3d_mol(smiles)

            if mol is None:
                continue

            # 检查是否包含不支持的元素
            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if not set(atoms).issubset(supported_species):
                # 如果包含 B, P, Si 等 ANI 不支持的元素，跳过
                continue

            try:
                # 准备输入数据
                coordinates = mol.GetConformer().GetPositions()
                coordinates = self.torch.tensor([coordinates], requires_grad=True, device=self.device,
                                                dtype=self.torch.float32)
                species = self.torch.tensor([atoms], device=self.device)

                # 计算能量
                energy = self.model((species, coordinates)).energies

                # 计算力 (能量对坐标的负梯度)
                derivative = self.torch.autograd.grad(energy.sum(), coordinates)[0]
                forces = -derivative

                # 提取标量特征 (转换为 numpy)
                energy_val = energy.item()  # Hartree
                forces_norm = self.torch.norm(forces, dim=2).detach().cpu().numpy()[0]  # [n_atoms]

                features = {
                    'ani_energy': energy_val,  # 总能量
                    'ani_energy_per_atom': energy_val / len(atoms),  # 平均原子能量
                    'ani_max_force': np.max(forces_norm),  # 最大受力点 (通常是不稳定点)
                    'ani_mean_force': np.mean(forces_norm),  # 平均受力
                    'ani_force_std': np.std(forces_norm)  # 受力分布方差
                }

                features_list.append(features)
                valid_indices.append(idx)

            except Exception as e:
                # print(f"Error processing {smiles}: {e}")
                continue

        if not features_list:
            return pd.DataFrame(), []

        # 整理结果
        df = pd.DataFrame(features_list)
        return df, valid_indices