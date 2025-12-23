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
import re  # 新增: 用于分割多组分 SMILES
from functools import partial  # 新增

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors3D, rdMolDescriptors
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
    单个样本的 3D 构象生成工作函数（供多进程调用）

    - 支持多组分/多片段 SMILES：会自动按 ';'、'；'、'|'、带空格的 ' + '、以及 '.' 进行分割
    - 对每个片段分别生成 3D（ETKDGv3）并做轻量优化（MMFF / UFF）
    - 仅保留 ANI2x 支持的元素：H,C,N,O,F,S,Cl

    返回:
        list[tuple[list[int], np.ndarray]]  # [(atomic_numbers, coordinates), ...]
        或 None（任一片段失败则返回 None，保证数据质量）
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            return None
        s = str(smiles).strip()
        if not s:
            return None

        # 1) 智能分割多组分
        # 先按 ; / ； / | 分割
        parts = re.split(r"\s*[;；|]\s*", s)

        # 再按“带空格的 +”分割（避免误伤 [N+] 这类带电荷写法）
        final = []
        for p in parts:
            final.extend(re.split(r"\s+\+\s+", p))

        # 再按 '.' 分割（SMILES 规范的多片段分隔）
        frags = []
        for p in final:
            frags.extend([x.strip() for x in str(p).split('.') if x and str(x).strip()])

        frags = [f for f in frags if f]
        if not frags:
            return None

        frag_data = []

        supported_species = {1, 6, 7, 8, 9, 16, 17}  # H,C,N,O,F,S,Cl (ANI2x)

        for frag in frags:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                return None

            mol = Chem.AddHs(mol)  # 力场/ANI 计算建议加氢

            # 2) 生成 3D 构象（ETKDGv3）
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.numThreads = 1  # 禁用 RDKit 内部线程，避免与多进程冲突

            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                # 兜底：再试一次
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if res != 0:
                    return None

            # 3) 快速几何优化：优先 MMFF，否则 UFF
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=40)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass

            # 4) 提取数据
            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if not set(atoms).issubset(supported_species):
                return None

            coords = mol.GetConformer().GetPositions().astype(np.float32)

            frag_data.append((atoms, coords))

        return frag_data if frag_data else None

    except Exception:
        return None



# =============================================================================
# 3D 描述符：RDKit3D + Coulomb Matrix (可选更前沿的构象表征)
# =============================================================================
def _rdkit3d_feature_worker(smiles, coulomb_top_k: int = 10):
    """
    计算单个样本的 3D 构象描述符（修复版）
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            return None
        s = str(smiles).strip()
        if not s:
            return None

        # --- 预处理：处理聚合物中的 * 号 ---
        # 3D 构象生成不支持 *，将其替换为 C (甲基) 以模拟占位
        if '*' in s:
            s = s.replace('*', 'C')

        # 分割多组分
        parts = re.split(r"\s*[;；|]\s*", s)
        final = []
        for p in parts:
            final.extend(re.split(r"\s+\+\s+", p))
        frags = []
        for p in final:
            frags.extend([x.strip() for x in str(p).split('.') if x and str(x).strip()])
        frags = [f for f in frags if f]
        if not frags:
            return None

        total_atoms = 0
        n_frags = 0
        d3_weighted = {}
        eig_all = []

        for frag in frags:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue  # 解析失败跳过该片段，不要直接返回 None

            # 过滤掉单原子或太小的碎片（通常是离子或杂质），它们很难生成有意义的 3D
            if mol.GetNumAtoms() < 2:
                continue

            mol = Chem.AddHs(mol)

            # --- 生成 3D 构象 (放宽参数) ---
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.numThreads = 1
            params.maxAttempts = 50  # [修改] 增加尝试次数

            # 尝试嵌入
            res = AllChem.EmbedMolecule(mol, params)

            # 如果失败，尝试更激进的随机坐标
            if res != 0:
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                if res != 0:
                    # [修改] 如果该片段生成失败，仅跳过该片段，不放弃整个样本
                    # print(f"⚠️ 3D生成失败 (跳过片段): {frag}")
                    continue

                    # 优化
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
            except Exception:
                pass

            n_atoms = int(mol.GetNumAtoms())
            if n_atoms <= 0:
                continue

            n_frags += 1
            total_atoms += n_atoms

            # RDKit 3D descriptors
            try:
                d3 = Descriptors3D.CalcMolDescriptors3D(mol)  # dict
                for k, v in d3.items():
                    val = float(v)
                    if np.isfinite(val):
                        d3_weighted[k] = d3_weighted.get(k, 0.0) + val * n_atoms
            except Exception:
                pass

            # Coulomb matrix
            try:
                cm = rdMolDescriptors.CalcCoulombMat(mol)
                cm_arr = np.array([list(row) for row in cm], dtype=float)
                eig = np.linalg.eigvalsh(cm_arr)
                eig_all.append(eig)
            except Exception:
                pass

        # [修改] 如果所有片段都失败了，才返回 None
        if total_atoms <= 0:
            # 打开下面的注释可以调试具体是哪个 SMILES 失败了
            # print(f"❌ 所有片段3D生成均失败: {s}")
            return None

        out = {
            "rdkit3d_n_atoms": int(total_atoms),
            "rdkit3d_n_fragments": int(n_frags),
        }

        # 加权平均
        for k, v in d3_weighted.items():
            out[f"rdkit3d_{k}"] = float(v) / float(total_atoms)

        # Coulomb Matrix 处理
        if eig_all:
            eig_concat = np.concatenate(eig_all).astype(float)
            if eig_concat.size > 0:
                eig_sorted = np.sort(eig_concat)[::-1]  # desc
                for i in range(int(coulomb_top_k)):
                    out[f"coulomb_eig_{i + 1}"] = float(eig_sorted[i]) if i < len(eig_sorted) else 0.0
                out["coulomb_eig_mean"] = float(np.mean(eig_concat))
                out["coulomb_eig_std"] = float(np.std(eig_concat))
                out["coulomb_eig_max"] = float(np.max(eig_concat))
                out["coulomb_eig_min"] = float(np.min(eig_concat))
            else:
                _fill_nan(out, coulomb_top_k)
        else:
            _fill_nan(out, coulomb_top_k)

        return out

    except Exception as e:
        # print(f"❌ 3D Worker 异常: {e}") # 调试用
        return None


def _fill_nan(out, k):
    for i in range(int(k)):
        out[f"coulomb_eig_{i + 1}"] = np.nan
    out["coulomb_eig_mean"] = np.nan
    out["coulomb_eig_std"] = np.nan
    out["coulomb_eig_max"] = np.nan
    out["coulomb_eig_min"] = np.nan


class RDKit3DDescriptorExtractor:
    """RDKit 3D 构象描述符提取器（可选更前沿的几何表征）"""

    def __init__(self, coulomb_top_k: int = 10):
        self.coulomb_top_k = int(coulomb_top_k)
        self.feature_names = []  # 运行后才知道完整列名

    def smiles_to_3d_descriptors(self, smiles_list, n_jobs: int | None = None):
        if not RDKIT_AVAILABLE:
            raise ImportError("需要安装 RDKit 才能使用 3D 描述符。")

        if n_jobs is None:
            n_jobs = 1 if os.name == 'nt' else max(1, (mp.cpu_count() or 1) - 1)

        feats = []
        valid_indices = []

        print(f"\n🧊 3D 构象描述符提取 (n_jobs={n_jobs}, coulomb_top_k={self.coulomb_top_k})")

        worker = partial(_rdkit3d_feature_worker, coulomb_top_k=self.coulomb_top_k)

        if n_jobs == 1:
            for idx, s in enumerate(tqdm(smiles_list, desc="3D Descriptors")):
                out = worker(s)
                if out is not None:
                    feats.append(out)
                    valid_indices.append(idx)
        else:
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    for idx, out in enumerate(tqdm(executor.map(worker, smiles_list),
                                                   total=len(smiles_list),
                                                   desc=f"3D Descriptors ({n_jobs} workers)")):
                        if out is not None:
                            feats.append(out)
                            valid_indices.append(idx)
            except Exception as e:
                print(f"⚠️ 3D 并行提取失败，回退单进程：{e}")
                for idx, s in enumerate(tqdm(smiles_list, desc="3D Descriptors (fallback)")):
                    out = worker(s)
                    if out is not None:
                        feats.append(out)
                        valid_indices.append(idx)

        if not feats:
            return pd.DataFrame(), []

        df = pd.DataFrame(feats)
        df = df.apply(pd.to_numeric, errors='coerce')
        self.feature_names = df.columns.tolist()

        return df, valid_indices



# =============================================================================
# 预训练 SMILES Transformer Embedding（可选：需要 transformers）
# =============================================================================
class SmilesTransformerEmbeddingExtractor:
    """
    预训练 SMILES Transformer 表征（例如 ChemBERTa 等）

    - 适合做“前沿特征工程”：不依赖手工描述符，能学习到更抽象的分子语义表示
    - 注意：首次运行会从 HuggingFace 下载模型权重（需要联网）
    """

    _CACHE = {}  # (model_name, device_str) -> (tokenizer, model, hidden_size)

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        pooling: str = "cls",
        max_length: int = 128,
        device=None
    ):
        self.model_name = model_name
        self.pooling = (pooling or "cls").lower()
        self.max_length = int(max_length)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModel = AutoModel
            self.AVAILABLE = True
        except Exception:
            self.AVAILABLE = False
            self.feature_names = []
            return

        if device is None:
            self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        cache_key = (self.model_name, str(self.device))
        if cache_key in self._CACHE:
            self.tokenizer, self.model, self.hidden_size = self._CACHE[cache_key]
        else:
            self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
            # 某些 tokenizer 可能没有 pad_token，做个兜底
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token

            self.model = self.AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # hidden size
            self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)

            self._CACHE[cache_key] = (self.tokenizer, self.model, self.hidden_size)

        # feature names 运行后根据 hidden_size 生成
        self.feature_names = [f"lm_emb_{i}" for i in range(self.hidden_size)] if self.hidden_size else []

    def _pool(self, last_hidden_state, attention_mask):
        # last_hidden_state: (B, L, H)
        if self.pooling == "mean":
            # mean pooling with mask
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        # default: cls pooling (take first token)
        return last_hidden_state[:, 0, :]

    def smiles_to_embeddings(self, smiles_list, batch_size: int = 32):
        if not self.AVAILABLE:
            raise ImportError("需要 transformers：pip install transformers")

        # 过滤空值
        valid_indices = []
        texts = []
        for i, s in enumerate(smiles_list):
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            ss = str(s).strip()
            if not ss:
                continue
            valid_indices.append(i)
            texts.append(ss)

        if not texts:
            return pd.DataFrame(), []

        embs = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Transformer Embedding"):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                pooled = self._pool(last_hidden, inputs.get("attention_mask"))
                embs.append(pooled.detach().cpu().numpy().astype(np.float32))

        emb_mat = np.vstack(embs)
        # 生成列名
        if not self.feature_names or len(self.feature_names) != emb_mat.shape[1]:
            self.feature_names = [f"lm_emb_{i}" for i in range(emb_mat.shape[1])]

        df = pd.DataFrame(emb_mat, columns=self.feature_names)
        return df, valid_indices

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
                # 修改开始：修复 n_proc 参数导致的 TypeError
                if n_proc > 1:
                    try:
                        # 尝试并行
                        df_batch = calc.pandas(batch_mols, n_proc=n_proc, quiet=True)
                    except TypeError:
                        # 如果不支持 n_proc 参数，回退到默认调用
                        if i == 0:
                            print(f"\n⚠️ Mordred版本不支持并行参数，切换至默认模式...")
                        n_proc = 1
                        df_batch = calc.pandas(batch_mols, quiet=True)
                    except Exception as e:
                        # 其他并行错误，回退到单进程
                        if i == 0:
                            print(f"\n⚠️ 并行计算出错 ({str(e)})，自动切换回单进程模式...")
                        n_proc = 1
                        # 单进程模式下不传 n_proc 参数
                        df_batch = calc.pandas(batch_mols, quiet=True)
                else:
                    # 单进程模式：直接不传 n_proc 参数，兼容所有版本
                    df_batch = calc.pandas(batch_mols, quiet=True)
                # 修改结束
                if type(df_batch).__name__ == 'MordredDataFrame':
                    df_batch = pd.DataFrame(df_batch)

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
    """
    机器学习力场特征提取器（TorchANI / ANI2x）

    ✅ 修复点（对应“力场特征总是 0”的常见原因）：
    1) 旧版在 batch padding 后尝试用“拆包”获取 atomic_energies，易与 TorchANI 输出结构不匹配，
       导致能量被错误计算为接近 0（甚至变成全 0）。
    2) 旧版将 padding 原子当作真实原子（或错误 mask），会污染能量/力。
    3) 多组分/多片段 SMILES（A.B 或 A;B）若直接作为一个体系计算，片段间非物理近距离会导致异常。

    本实现策略：
    - 先多进程生成 3D 构象（每个片段独立）
    - 按 “原子数相同” 分组做 batch 推理（无需 padding）
    - 对每个样本把各片段的结果聚合为一个特征向量
    """

    SUPPORTED_SPECIES = {1, 6, 7, 8, 9, 16, 17}  # H,C,N,O,F,S,Cl (ANI2x)

    _HARTREE_TO_KJ_MOL = 2625.499638
    _HARTREE_TO_KCAL_MOL = 627.509474

    def __init__(self, device=None, energy_unit: str = "hartree"):
        """
        Args:
            device: torch.device 或 None（自动选择 cuda/cpu）
            energy_unit: 'hartree' | 'kJ/mol' | 'kcal/mol'
        """
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

        self.energy_unit = (energy_unit or "hartree").lower()

        # ✅ CPU 性能优化：让 Torch 在 CPU 上充分使用线程
        # 注意：在多进程 3D 生成时，ANI 推理通常在主进程执行，因此这里多线程能明显加速
        try:
            if self.device.type == "cpu":
                import os as _os
                n_cpu = _os.cpu_count() or 1
                # 计算线程：尽量用满 CPU；Interop 线程保持较小以减少调度开销
                self.torch.set_num_threads(n_cpu)
                try:
                    self.torch.set_num_interop_threads(min(4, n_cpu))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.model = self.torchani.models.ANI2x().to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"ANI Model load error: {e}")
            self.AVAILABLE = False
            self.feature_names = []
            return

        # 保留旧列名，避免下游逻辑/历史模型不兼容
        self.feature_names = [
            'ani_energy',
            'ani_energy_per_atom',
            'ani_max_force',
            'ani_mean_force',
            'ani_force_std',
            # 新增诊断/结构信息
            'ani_n_atoms',
            'ani_n_fragments',
            'ani_success'
        ]

    def _convert_energy(self, e_hartree: float) -> float:
        if e_hartree is None or (isinstance(e_hartree, float) and (np.isnan(e_hartree) or np.isinf(e_hartree))):
            return np.nan
        if self.energy_unit in ["hartree", "ha"]:
            return float(e_hartree)
        if self.energy_unit in ["kj/mol", "kjmol", "kj"]:
            return float(e_hartree) * self._HARTREE_TO_KJ_MOL
        if self.energy_unit in ["kcal/mol", "kcalmol", "kcal"]:
            return float(e_hartree) * self._HARTREE_TO_KCAL_MOL
        # 未知单位：不转换
        return float(e_hartree)

    def _infer_batch(self, species_np: np.ndarray, coords_np: np.ndarray):
        """
        对同原子数的一组分子做 batch 推理（无 padding）
        species_np: (B, N) int64 原子序数
        coords_np: (B, N, 3) float32 3D 坐标
        返回:
            energies: (B,) float
            forces: (B, N, 3) float
        """
        species = self.torch.tensor(species_np, dtype=self.torch.long, device=self.device)
        coords = self.torch.tensor(coords_np, dtype=self.torch.float32, device=self.device)
        coords.requires_grad_(True)

        energy = self.model((species, coords)).energies  # (B,)
        forces = -self.torch.autograd.grad(
            energy.sum(), coords, create_graph=False, retain_graph=False
        )[0]  # (B, N, 3)

        return (
            energy.detach().cpu().numpy().astype(np.float64),
            forces.detach().cpu().numpy().astype(np.float64)
        )

    def smiles_to_ani_features(self, smiles_list, batch_size: int = 64, n_jobs: int | None = None):
        if not self.AVAILABLE:
            raise ImportError("请先安装 torchani: pip install torchani")

        # -------- 1) 多进程生成 3D 构象（每个样本可能含多个片段）--------
        print(f"\n⚛️ 正在生成 3D 构象（多组分将按片段分别生成）...")

        # Windows 下多进程可能不稳定，默认降为单进程
        if n_jobs is None:
            n_jobs = 1 if os.name == 'nt' else max(1, (mp.cpu_count() or 1) - 1)

        valid_indices = []
        sample_frags = []  # list[list[(atoms, coords)]]

        try:
            if n_jobs == 1:
                # 单进程（更稳）
                for i, s in enumerate(tqdm(smiles_list, desc="3D Generation")):
                    res = _generate_3d_data_worker(s)
                    if res is not None:
                        valid_indices.append(i)
                        sample_frags.append(res)
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    for i, res in enumerate(tqdm(executor.map(_generate_3d_data_worker, smiles_list),
                                                 total=len(smiles_list),
                                                 desc=f"3D Generation ({n_jobs} workers)")):
                        if res is not None:
                            valid_indices.append(i)
                            sample_frags.append(res)
        except Exception as e:
            print(f"⚠️ 3D 并行生成失败，回退到单进程：{e}")
            valid_indices = []
            sample_frags = []
            for i, s in enumerate(tqdm(smiles_list, desc="3D Generation (fallback)")):
                res = _generate_3d_data_worker(s)
                if res is not None:
                    valid_indices.append(i)
                    sample_frags.append(res)

        if not sample_frags:
            return pd.DataFrame(), []

        # -------- 2) 展平片段，按原子数分组 batch 推理（无 padding）--------
        from collections import defaultdict

        frag_records = []  # 每个元素对应一个片段
        for orig_i, frags in zip(valid_indices, sample_frags):
            for atoms, coords in frags:
                frag_records.append({
                    'orig_index': orig_i,
                    'n_atoms': int(len(atoms)),
                    'atoms': atoms,
                    'coords': coords,
                    'energy': np.nan,
                    'forces': None,
                    'failed': False
                })

        groups = defaultdict(list)
        for r in frag_records:
            groups[r['n_atoms']].append(r)

        print(f"⚛️ 开始 ANI 推理（按原子数分组批处理，Batch Size={batch_size}, Device={self.device}）...")

        for n_atoms, recs in groups.items():
            for start in tqdm(range(0, len(recs), batch_size), desc=f"Inference (N={n_atoms})"):
                batch = recs[start:start + batch_size]
                try:
                    species_np = np.asarray([b['atoms'] for b in batch], dtype=np.int64)
                    coords_np = np.stack([b['coords'] for b in batch]).astype(np.float32)

                    energies, forces = self._infer_batch(species_np, coords_np)
                    for k, b in enumerate(batch):
                        b['energy'] = float(energies[k])
                        b['forces'] = forces[k]
                except Exception as e:
                    # 兜底：逐个推理，尽量不让整个批次失败
                    for b in batch:
                        try:
                            species_np = np.asarray([b['atoms']], dtype=np.int64)
                            coords_np = np.asarray([b['coords']], dtype=np.float32)
                            energies, forces = self._infer_batch(species_np, coords_np)
                            b['energy'] = float(energies[0])
                            b['forces'] = forces[0]
                        except Exception:
                            b['failed'] = True
                            b['energy'] = np.nan
                            b['forces'] = None

        # -------- 3) 按样本聚合片段结果，生成特征 --------
        sample_acc = {}
        for idx in valid_indices:
            sample_acc[idx] = {
                'energies': [],
                'force_norms': [],
                'n_atoms': 0,
                'n_frags': 0,
                'failed': False
            }

        for r in frag_records:
            acc = sample_acc.get(r['orig_index'])
            if acc is None:
                continue

            if r.get('failed') or r.get('forces') is None or (not np.isfinite(r.get('energy', np.nan))):
                acc['failed'] = True
                continue

            acc['energies'].append(float(r['energy']))
            norms = np.linalg.norm(np.asarray(r['forces'], dtype=np.float64), axis=1)
            acc['force_norms'].append(norms)
            acc['n_atoms'] += int(r['n_atoms'])
            acc['n_frags'] += 1

        features_list = []
        final_indices = []

        for idx in valid_indices:
            acc = sample_acc[idx]
            if acc['failed'] or acc['n_atoms'] <= 0 or len(acc['energies']) == 0:
                continue

            e_total = float(np.sum(acc['energies']))
            e_total_conv = self._convert_energy(e_total)
            e_per_atom = e_total_conv / acc['n_atoms'] if acc['n_atoms'] > 0 else np.nan

            if acc['force_norms']:
                fn = np.concatenate(acc['force_norms'])
                f_max = float(np.max(fn)) if fn.size else np.nan
                f_mean = float(np.mean(fn)) if fn.size else np.nan
                f_std = float(np.std(fn)) if fn.size else np.nan
            else:
                f_max = f_mean = f_std = np.nan

            feats = {
                'ani_energy': e_total_conv,
                'ani_energy_per_atom': e_per_atom,
                'ani_max_force': f_max,
                'ani_mean_force': f_mean,
                'ani_force_std': f_std,
                'ani_n_atoms': int(acc['n_atoms']),
                'ani_n_fragments': int(acc['n_frags']),
                'ani_success': 1
            }
            features_list.append(feats)
            final_indices.append(idx)

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        return df, final_indices

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

    def extract_features(self, resin_smiles_list, hardener_smiles_list, stoichiometry_list=None, stoich_mode: str = 'Resin/Hardener (总质量比, R/H)'):
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


                # 用户提供的配比（可选）
                # 说明：
                # - stoich_mode = "Resin/Hardener (总质量比, R/H)"：列值为 树脂总量/固化剂总量 (R/H)
                #   则可换算为实际 PHR = 100 / (R/H)
                # - stoich_mode = "PHR (Hardener per 100 Resin)"：列值即为 PHR
                actual_phr = theo_phr
                if stoichiometry_list is not None and idx < len(stoichiometry_list):
                    try:
                        v = float(stoichiometry_list[idx])
                        if v > 0:
                            if stoich_mode.startswith("Resin/Hardener"):
                                # R/H -> PHR = 100 * H/R = 100 / (R/H)
                                actual_phr = 100.0 / v
                            elif stoich_mode.startswith("PHR"):
                                actual_phr = v
                            else:
                                actual_phr = v
                    except Exception:
                        pass

                # 与理论配比的偏离（用于反映固化欠量/过量）
                stoich_ratio = (actual_phr / theo_phr) if theo_phr > 0 else 0.0
                stoich_delta = actual_phr - theo_phr
                # 2. 电子性质特征 (新增功能 - 模拟DFT)
                r_pos_chg, r_neg_chg, r_tpsa = self._calc_electronic_props(mol_r)
                h_pos_chg, h_neg_chg, h_tpsa = self._calc_electronic_props(mol_h)

                features = {
                    'EEW': eew,
                    'AHEW': ahew,
                    'Resin_Functionality': f_epoxy,
                    'Hardener_Functionality': f_amine,
                    'Theoretical_PHR': theo_phr,
                    'Actual_PHR': actual_phr,
                    'Stoich_Ratio': stoich_ratio,
                    'Stoich_Delta': stoich_delta,
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


class FGDFeatureExtractor:
    """
    [增强版] FGD (Functional Group Distinction) 特征提取器
    针对用户数据集进行了定制优化：增加了硫醇、酰肼、二苯甲酮等识别规则。
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("FGD 提取需要 RDKit 支持。")

        # 1. 定义骨架 (Substrates) - 优先级：结构越特异，越靠前
        self.substrates = {
            # --- [新增] 针对您数据中的二苯甲酮环氧 ---
            "Benzophenone": "c1ccc(cc1)C(=O)c2ccc(cc2)",

            "DGEBA": "c1ccc(cc1)C(C)(C)c2ccc(cc2)",  # 双酚A型
            "DGEBF": "c1ccc(cc1)Cc2ccc(cc2)",  # 双酚F型 (也匹配 DDM 固化剂骨架)
            "Novolac": "c1ccc(O)c(c1)Cc2ccccc2",  # 酚醛骨架
            "TDE-85 (Ester)": "C(=O)OC",  # 酯环族/通用酯键
            "Cycloaliphatic": "C1CCCCC1",  # 脂环族 (六元环)
            "Isocyanurate": "N1C(=O)NC(=O)NC1=O",  # 异氰尿酸酯 (TGIC等)
            "Aliphatic Chain": "[CX4,CX3]~[CX4,CX3]~[CX4,CX3]~[CX4,CX3]",  # 长链脂肪族
            "Benzene Ring": "c1ccccc1"  # 简单苯环 (兜底)
        }

        # 2. 定义官能团 (Groups) - 决定反应机理
        self.groups = {
            "Epoxide": "C1OC1",  # 环氧基
            "Anhydride": "C(=O)OC(=O)",  # 酸酐 (如 MTHPA)

            # --- [新增] 针对您数据中的 NNC(=O) ---
            "Hydrazide": "[NX3][NX3]C(=O)",  # 酰肼 (潜伏性固化剂)

            # --- [新增] 针对您数据中的 SCC... ---
            "Thiol": "[#16X2H]",  # 巯基/硫醇 (-SH)

            "Methacrylate": "CC(=C)C(=O)O",  # 甲基丙烯酸酯
            "Acrylate": "C=CC(=O)O",  # 丙烯酸酯
            "Amine (Primary)": "[NX3;H2]",  # 伯胺 (如 DDM)
            "Amine (Secondary)": "[NX3;H1]",  # 仲胺
            "Hydroxyl": "[OX2H]",  # 羟基
            "Vinyl": "C=C",  # 乙烯基 (兜底)
        }

        # 预编译 pattern
        self._sub_pats = {}
        for k, v in self.substrates.items():
            try:
                self._sub_pats[k] = Chem.MolFromSmarts(v)
            except:
                pass

        self._grp_pats = {}
        for k, v in self.groups.items():
            try:
                self._grp_pats[k] = Chem.MolFromSmarts(v)
            except:
                pass

    def _clean_smiles(self, text):
        """清洗混合物SMILES，处理分号等非标准分隔符"""
        if pd.isna(text): return None
        s = str(text).strip()
        # 将分号替换为 RDKit 可识别的点号 (表示非键连混合物)
        s = s.replace(';', '.').replace('；', '.')
        return s

    def categorize_smiles(self, smiles_list):
        """
        输入 SMILES 列表，返回 DataFrame 包含 'FGD_Substrate' 和 'FGD_Group'
        """
        results = []
        valid_indices = []

        print(f"\n📑 正在执行 FGD 官能团分类 (增强版)...")

        for idx, raw_smi in enumerate(tqdm(smiles_list, desc="FGD Classification")):
            try:
                smi = self._clean_smiles(raw_smi)
                if not smi:
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                # 匹配骨架
                sub_type = "Other_Substrate"
                for name, pat in self._sub_pats.items():
                    if pat and mol.HasSubstructMatch(pat):
                        sub_type = name
                        break

                        # 匹配官能团
                func_group = "Other_Group"
                for name, pat in self._grp_pats.items():
                    if pat and mol.HasSubstructMatch(pat):
                        func_group = name
                        break

                results.append({
                    "FGD_Substrate": sub_type,
                    "FGD_Group": func_group
                })
                valid_indices.append(idx)

            except Exception:
                continue

        if not results:
            return pd.DataFrame(), []

        df = pd.DataFrame(results)
        return df, valid_indices
