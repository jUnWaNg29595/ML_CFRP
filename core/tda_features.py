# -*- coding: utf-8 -*-
"""TDA (Topological Data Analysis) 特征提取模块 (高性能并行版)"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
from functools import partial  # [新增] 用于固定函数参数

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

    # 简单的多组分拆分逻辑
    frags = str(smiles).replace(';', '.').replace('|', '.').split('.')
    all_pts = []

    for frag in frags:
        frag = frag.strip()
        if not frag:
            continue

        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue

        # 处理 Dummy Atoms: 将 * 替换为 Carbon，防止 3D 生成崩溃
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)

        if add_hs:
            mol = Chem.AddHs(mol)

        # 尝试生成 3D (ETKDGv3)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = seed
        params.numThreads = 1  # Worker 内单线程

        res = AllChem.EmbedMolecule(mol, params)

        # 失败兜底 1: 随机坐标
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)

        # 失败兜底 2: 2D 坐标 (Z=0) -> 保证 TDA 不挂
        if res != 0:
            AllChem.Compute2DCoords(mol)

        # 可选: 力场优化 (极慢，慎用)
        if optimize and res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
            except:
                pass

        conf = mol.GetConformer()
        pts = np.asarray(conf.GetPositions(), dtype=np.float32)

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

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # executor.map 只需要传递 smiles_list，其他参数已通过 partial 绑定
            results = list(tqdm(
                executor.map(worker_func, smiles_list),
                total=len(smiles_list),
                desc="生成 3D 点云"
            ))

        # 2. 串行/并行计算 TDA (Ripser 释放 GIL 较好，且通常很快，简单循环即可)
        for idx, pts in enumerate(tqdm(results, desc="计算拓扑特征")):
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