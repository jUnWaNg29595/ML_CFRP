# -*- coding: utf-8 -*-
"""TDA (Topological Data Analysis) 特征提取模块 (高性能并行版)"""

from __future__ import annotations

# ============================================
# 重要：必须在导入 RDKit 之前导入线程配置！
# ============================================
from . import thread_config

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
from functools import partial  # [新增] 用于固定函数参数

# [新增] 后台任务管理器 - 支持任务取消
from .task_manager import is_cancelled, CancellableProcessPoolExecutor

warnings.filterwarnings("ignore")

# ----------------------------
# 依赖检查
# ----------------------------
try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except Exception:
    ripser = None
    RIPSER_AVAILABLE = False

try:
    from persim import PersImage

    PERSIM_AVAILABLE = True
except Exception:
    PersImage = None
    PERSIM_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    AllChem = None
    RDKIT_AVAILABLE = False

# 关闭 RDKit 繁杂的日志
if RDKIT_AVAILABLE:
    RDLogger.DisableLog('rdApp.*')

# [新增] 支持 SMILES / SELFIES / BigSMILES 输入
from .smiles_utils import convert_to_smiles, normalize_chemical_string, split_smiles_cell

@dataclass
class TDAConfig:
    """TDA 特征配置"""
    maxdim: int = 2
    thresh: Optional[float] = None
    metric: str = "euclidean"
    max_points: Optional[int] = 200  # 限制最大原子数，加速 Ripser
    downsample_seed: int = 42
    do_optimize: bool = False  # 默认关闭力场优化，大幅提速
    use_persistence_image: bool = False
    pim_size: Tuple[int, int] = (10, 10)
    pim_spread: float = 1.0


# ----------------------------
# 核心工作函数 (放在类外以支持多进程 pickle)
# ----------------------------

def _generate_point_cloud_worker(smiles: str, add_hs: bool = False, optimize: bool = False, seed: int = 42) -> Optional[
    np.ndarray]:
    """单样本 3D 生成函数 (Worker)"""
    if not RDKIT_AVAILABLE or not smiles:
        return None

    try:
        # 智能多组分拆分（支持 SMILES / SELFIES / BigSMILES）
        s = convert_to_smiles(smiles, fmt="auto") or str(smiles)
        frags = split_smiles_cell(s)
        all_pts = []

        for frag in frags:
            frag = frag.strip()
            if not frag:
                continue

            mol = Chem.MolFromSmiles(normalize_chemical_string(frag, canonicalize=False, repair=True, keep_largest_frag=False) or "")
            if mol is None:
                continue
            
            # ✅ 检查分子是否有原子
            if mol.GetNumAtoms() == 0:
                continue

            # 处理 Dummy Atoms: 将 * 替换为 Carbon，防止 3D 生成崩溃
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(6)

            if add_hs:
                mol = Chem.AddHs(mol)
            
            # ✅ 再次检查（AddHs后）
            if mol.GetNumAtoms() == 0:
                continue

            # 尝试生成 3D (ETKDGv3)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.randomSeed = seed
            params.numThreads = 1  # Worker 内单线程

            try:
                res = AllChem.EmbedMolecule(mol, params)
            except Exception:
                res = -1

            # 失败兜底 1: 随机坐标
            if res != 0:
                try:
                    res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
                except Exception:
                    res = -1

            # 失败兜底 2: 2D 坐标 (Z=0) -> 保证 TDA 不挂
            if res != 0:
                try:
                    AllChem.Compute2DCoords(mol)
                except Exception:
                    continue  # 彻底失败，跳过这个片段

            # 可选: 力场优化 (极慢，慎用)
            if optimize and res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
                except:
                    pass

            try:
                conf = mol.GetConformer()
                pts = np.asarray(conf.GetPositions(), dtype=np.float32)
            except Exception:
                continue  # 无法获取构象，跳过

            # 仅保留重原子以减少点数 (除非指定 add_hs)
            if not add_hs:
                heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
                if len(heavy_indices) >= 3:  # 至少3个点才能构成面
                    pts = pts[heavy_indices]

            all_pts.append(pts)

        if not all_pts:
            return None

        # 堆叠所有片段的点云
        return np.vstack(all_pts)
    
    except Exception:
        # 捕获所有未预期的异常，返回None而不是崩溃
        return None


class PersistentHomologyFeatureExtractor:
    """并行 TDA 特征提取器"""

    def __init__(self, config: Optional[TDAConfig] = None):
        self.config = config or TDAConfig()
        self.AVAILABLE = bool(RIPSER_AVAILABLE)
        self.feature_names: List[str] = []

    def _point_cloud_to_features(self, points: np.ndarray) -> Dict[str, float]:
        """Ripser 计算核心"""
        # 下采样保护
        if self.config.max_points and points.shape[0] > self.config.max_points:
            idx = np.random.RandomState(self.config.downsample_seed).choice(
                points.shape[0], self.config.max_points, replace=False
            )
            points = points[idx]

        # 运行 Ripser
        try:
            out = ripser(points, maxdim=self.config.maxdim, thresh=self.config.thresh, metric=self.config.metric)
            dgms = out.get("dgms", [])
        except Exception:
            # 极少数情况 (如共线点) ripser 可能失败，返回零特征
            dgms = []

        feat = {}
        # 统计特征提取
        for dim in range(self.config.maxdim + 1):
            diag = dgms[dim] if dim < len(dgms) else np.zeros((0, 2))

            # 清洗 inf
            if len(diag) > 0:
                diag = diag[np.isfinite(diag[:, 1])]

            if len(diag) == 0:
                lifetimes = np.array([0.0])
            else:
                lifetimes = diag[:, 1] - diag[:, 0]

            feat[f"tda_dim{dim}_count"] = float(len(lifetimes))
            feat[f"tda_dim{dim}_max"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
            feat[f"tda_dim{dim}_mean"] = float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0
            feat[f"tda_dim{dim}_sum"] = float(np.sum(lifetimes))
            feat[f"tda_dim{dim}_std"] = float(np.std(lifetimes)) if len(lifetimes) > 0 else 0.0

            # 简单的 Persistence Entropy
            if np.sum(lifetimes) > 0:
                probs = lifetimes / np.sum(lifetimes)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0
            feat[f"tda_dim{dim}_entropy"] = float(entropy)

        # Persistence Image (可选)
        if self.config.use_persistence_image and PERSIM_AVAILABLE:
            pim = PersImage(pixels=self.config.pim_size, spread=self.config.pim_spread, verbose=False)
            for dim in range(min(self.config.maxdim + 1, len(dgms))):
                diag = dgms[dim]
                # PersIm 需要有限值
                if len(diag) > 0:
                    diag = diag[np.isfinite(diag[:, 1])]

                if len(diag) == 0:
                    img_vec = np.zeros(self.config.pim_size[0] * self.config.pim_size[1])
                else:
                    try:
                        img = pim.transform(diag)
                        img_vec = img.flatten()
                    except:
                        img_vec = np.zeros(self.config.pim_size[0] * self.config.pim_size[1])

                for i, val in enumerate(img_vec):
                    feat[f"tda_pim_dim{dim}_{i}"] = val

        return feat

    # ----------------------------
    # [修复] 显式添加参数以匹配 app.py 的调用
    # ----------------------------
    def smiles_to_tda_features(
            self,
            smiles_list: Sequence[str],
            n_jobs: int = -1,
            add_hs: bool = False,
            optimize: Optional[bool] = None,
            seed: int = 42
    ) -> Tuple[pd.DataFrame, List[int]]:
        """并行提取入口"""
        if not self.AVAILABLE:
            print("❌ Error: ripser not installed.")
            return pd.DataFrame(), []

        # 确定并行核数
        if n_jobs < 1:
            n_jobs = max(1, mp.cpu_count() - 2)  # 留2个核给系统

        # 确定 optimize 参数
        do_optimize = self.config.do_optimize if optimize is None else bool(optimize)

        print(f"\n🧩 TDA 提取中 (n_jobs={n_jobs}, max_points={self.config.max_points}, add_hs={add_hs})...")

        valid_indices = []
        features_list = []

        # 1. 并行生成点云 (CPU密集型)
        point_clouds = []

        # 使用 partial 固定 worker 需要的参数
        worker_func = partial(
            _generate_point_cloud_worker,
            add_hs=add_hs,
            optimize=do_optimize,
            seed=seed
        )

        # ✅ 修复：使用 submit + wait 替代 map，添加超时机制避免卡死
        from concurrent.futures import wait, TimeoutError as FuturesTimeoutError
        
        per_molecule_timeout = 30  # 单分子超时时间（秒）
        results_dict = {}  # {index: result}
        total = len(smiles_list)
        timeout_count = 0
        
        # 分批处理
        batch_submit_size = n_jobs * 2
        pbar = tqdm(total=total, desc="生成 3D 点云")
        
        with CancellableProcessPoolExecutor(max_workers=n_jobs, task_name="TDA特征提取") as executor:
            for batch_start in range(0, total, batch_submit_size):
                # 检查是否请求取消
                if is_cancelled():
                    print("⏹️ 任务已取消")
                    pbar.close()
                    break
                    
                batch_end = min(batch_start + batch_submit_size, total)
                batch_smiles = smiles_list[batch_start:batch_end]
                
                # 提交这一批任务
                futures = {
                    executor.submit(worker_func, s): batch_start + j 
                    for j, s in enumerate(batch_smiles)
                }
                
                # 等待这批任务完成，设置超时
                batch_timeout = per_molecule_timeout * len(batch_smiles) / max(1, n_jobs) + 10
                done, not_done = wait(futures.keys(), timeout=batch_timeout)
                
                # 处理完成的任务
                for future in done:
                    idx = futures[future]
                    try:
                        res = future.result(timeout=1)
                        results_dict[idx] = res
                    except Exception:
                        results_dict[idx] = None
                
                # 取消超时的任务
                for future in not_done:
                    future.cancel()
                    idx = futures[future]
                    results_dict[idx] = None
                    timeout_count += 1
                
                pbar.update(len(batch_smiles))
        
        pbar.close()
        
        if timeout_count > 0:
            print(f"⚠️ {timeout_count} 个分子处理超时，已跳过")
        
        # 按索引顺序排列结果
        results = [results_dict.get(i) for i in range(total)]

        # 2. 串行/并行计算 TDA (Ripser 释放 GIL 较好，且通常很快，简单循环即可)
        for idx, pts in enumerate(tqdm(results, desc="计算拓扑特征")):
            # [新增] 检查是否请求取消
            if is_cancelled():
                print("⏹️ 任务已取消")
                break
            if pts is None or pts.shape[0] < 3:
                continue

            try:
                feats = self._point_cloud_to_features(pts)
                features_list.append(feats)
                valid_indices.append(idx)
            except Exception as e:
                continue

        if not features_list:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)
        # 填充 NaN
        df = df.fillna(0.0)
        self.feature_names = df.columns.tolist()
        return df, valid_indices
