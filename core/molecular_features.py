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
        """Mordred描述符提取"""
        if not MORDRED_AVAILABLE:
            raise ImportError("需要安装mordred")

        mols, valid_indices = [], []
        for idx, smiles in enumerate(smiles_list):
            mol = self._smiles_to_mol(smiles)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)

        if not mols:
            return pd.DataFrame(), []

        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols, quiet=True)
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
