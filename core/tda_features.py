# -*- coding: utf-8 -*-
"""TDA (Topological Data Analysis) 特征提取模块

本模块将“持续同调（Persistent Homology）”作为一种结构表征方式，
把 3D 点云/分子构象中的连通性、环与孔洞信息编码为固定长度的数值特征。

典型用途（高分子/热固性网络）：
    - 交联网络的孔洞/自由体积（Betti-2）
    - 环/隧道结构（Betti-1）
    - 连通分量演化（Betti-0）

⚠️ 依赖说明
    - 计算持久同调推荐使用 ripser：
        pip install ripser persim
    - 本模块在依赖缺失时不会让系统崩溃；但当你真正调用 TDA 提取时会给出明确报错。

接口设计
    - PersistentHomologyFeatureExtractor.smiles_to_tda_features(smiles_list)
        从 SMILES -> RDKit 3D 构象 -> 点云 -> PH -> 特征
    - PersistentHomologyFeatureExtractor.point_clouds_to_tda_features(point_clouds)
        直接从点云列表提取（用于 MD/CT/网络节点坐标）
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------
# Optional dependencies
# ----------------------------
try:
    from ripser import ripser  # type: ignore

    RIPSER_AVAILABLE = True
except Exception:
    ripser = None
    RIPSER_AVAILABLE = False

try:
    from persim import PersImage  # type: ignore

    PERSIM_AVAILABLE = True
except Exception:
    PersImage = None
    PERSIM_AVAILABLE = False

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    AllChem = None
    RDKIT_AVAILABLE = False


def _split_multi_component_smiles(smiles: str) -> List[str]:
    """把单元格里的多组分 SMILES 拆分成列表。

    支持的分隔符：;、；、|、以及“带空格的 +”（避免误伤 [N+]）。
    另外会进一步按 '.' 进行碎片拆分（SMILES 标准多片段分隔）。
    """

    if smiles is None:
        return []

    s = str(smiles).strip()
    if not s or s.lower() == "nan":
        return []

    s = s.replace("；", ";")

    # 先按 ; 或 | 分割
    parts = re.split(r"\s*[;|]\s*", s)

    # 再按“带空格的 +”分割（避免误伤 [N+]）
    final: List[str] = []
    for p in parts:
        final.extend(re.split(r"\s+\+\s+", p))

    # 再按 '.' 分割（SMILES 规范的多片段分隔）
    frags: List[str] = []
    for p in final:
        frags.extend([x.strip() for x in str(p).split(".") if x and str(x).strip()])

    frags = [f for f in frags if f]
    return frags


def _smiles_to_3d_points(smiles: str, *, add_hs: bool = True, seed: int = 42) -> Optional[np.ndarray]:
    """SMILES -> 3D 点云（Nx3）。

    - 支持多组分：各片段分别生成 3D，再把坐标拼接成一个点云。
    - 生成策略：ETKDGv3 + (可选) MMFF/UFF 轻量优化

    Returns:
        points: (N, 3) float32 或 None
    """

    if not RDKIT_AVAILABLE:
        return None

    frags = _split_multi_component_smiles(smiles)
    if not frags:
        return None

    all_points: List[np.ndarray] = []

    for frag in frags:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return None

        if add_hs:
            mol = Chem.AddHs(mol)

        # 生成 3D 构象
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = int(seed)
        params.numThreads = 1

        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            # 兜底：再试一次
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=int(seed))
            if res != 0:
                return None

        # 简单优化（失败也没关系）
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass

        conf = mol.GetConformer()
        pts = conf.GetPositions().astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            return None

        all_points.append(pts)

    if not all_points:
        return None

    return np.vstack(all_points).astype(np.float32)


def _persistence_entropy(pers: np.ndarray, eps: float = 1e-12) -> float:
    """持久性熵（Persistence Entropy）。"""
    pers = np.asarray(pers, dtype=float)
    pers = pers[np.isfinite(pers)]
    pers = pers[pers > 0]
    if pers.size == 0:
        return 0.0
    p = pers / (pers.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())


def _diagram_summary(diag: np.ndarray) -> Dict[str, float]:
    """把单个维度的持久图 (k,2) 转成统计特征。"""
    if diag is None or len(diag) == 0:
        return {
            "count": 0.0,
            "pers_sum": 0.0,
            "pers_mean": 0.0,
            "pers_std": 0.0,
            "pers_max": 0.0,
            "pers_q50": 0.0,
            "pers_q90": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
            "entropy": 0.0,
        }

    diag = np.asarray(diag, dtype=float)
    if diag.ndim != 2 or diag.shape[1] != 2:
        return {
            "count": 0.0,
            "pers_sum": 0.0,
            "pers_mean": 0.0,
            "pers_std": 0.0,
            "pers_max": 0.0,
            "pers_q50": 0.0,
            "pers_q90": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
            "entropy": 0.0,
        }

    births = diag[:, 0]
    deaths = diag[:, 1]
    mask = np.isfinite(births) & np.isfinite(deaths)
    births = births[mask]
    deaths = deaths[mask]

    # ripser 会给 dim0 最后一条 death=inf，直接剔除
    mask_inf = np.isfinite(deaths)
    births = births[mask_inf]
    deaths = deaths[mask_inf]

    if births.size == 0:
        return {
            "count": 0.0,
            "pers_sum": 0.0,
            "pers_mean": 0.0,
            "pers_std": 0.0,
            "pers_max": 0.0,
            "pers_q50": 0.0,
            "pers_q90": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
            "entropy": 0.0,
        }

    pers = deaths - births
    pers = pers[np.isfinite(pers)]
    pers = pers[pers > 0]
    if pers.size == 0:
        pers = np.asarray([0.0])

    return {
        "count": float(len(pers)),
        "pers_sum": float(np.sum(pers)),
        "pers_mean": float(np.mean(pers)),
        "pers_std": float(np.std(pers)),
        "pers_max": float(np.max(pers)),
        "pers_q50": float(np.quantile(pers, 0.5)),
        "pers_q90": float(np.quantile(pers, 0.9)),
        "birth_mean": float(np.mean(births)) if births.size else 0.0,
        "death_mean": float(np.mean(deaths)) if deaths.size else 0.0,
        "entropy": float(_persistence_entropy(pers)),
    }


@dataclass
class TDAConfig:
    """TDA 特征配置。"""

    maxdim: int = 2
    thresh: Optional[float] = None
    metric: str = "euclidean"
    # 可选：把持久图转为 Persistence Image（维度会比较大，默认关闭）
    use_persistence_image: bool = False
    pim_size: Tuple[int, int] = (10, 10)
    pim_spread: float = 1.0


class PersistentHomologyFeatureExtractor:
    """持续同调特征提取器。

    说明：
        - 如果 ripser 未安装，AVAILABLE=False。
        - 如果使用 smiles_to_tda_features，则需要 RDKit。
    """

    def __init__(self, config: Optional[TDAConfig] = None):
        self.config = config or TDAConfig()
        self.AVAILABLE = bool(RIPSER_AVAILABLE)
        self.RDKIT_AVAILABLE = bool(RDKIT_AVAILABLE)
        self.feature_names: List[str] = []

    @staticmethod
    def _ensure_available():
        if not RIPSER_AVAILABLE:
            raise ImportError(
                "未检测到 ripser/persim。请先安装：pip install ripser persim\n"
                "（建议在你的虚拟环境中执行）"
            )

    def _point_cloud_to_features(self, points: np.ndarray) -> Dict[str, float]:
        self._ensure_available()

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points 必须是 (N,3) 且 N>=2")

        # ripser 返回：{'dgms': [D0, D1, ...], ...}
        out = ripser(points, maxdim=int(self.config.maxdim), thresh=self.config.thresh, metric=self.config.metric)
        dgms = out.get("dgms", [])

        feat: Dict[str, float] = {}

        # 1) 统计型特征（固定长度，推荐默认）
        for dim in range(int(self.config.maxdim) + 1):
            diag = dgms[dim] if dim < len(dgms) else np.zeros((0, 2))
            stats = _diagram_summary(diag)
            for k, v in stats.items():
                feat[f"tda_dim{dim}_{k}"] = float(v)

        # 2) 可选：Persistence Image（维度较大，适合下游再做特征选择/降维）
        if self.config.use_persistence_image:
            if not PERSIM_AVAILABLE:
                raise ImportError("use_persistence_image=True 需要安装 persim：pip install persim")
            pim = PersImage(pixels=self.config.pim_size, spread=self.config.pim_spread)
            for dim in range(min(int(self.config.maxdim) + 1, len(dgms))):
                diag = np.asarray(dgms[dim], dtype=float)
                if diag.size == 0:
                    img = np.zeros(self.config.pim_size, dtype=float)
                else:
                    # persim 期望有限 death
                    mask = np.isfinite(diag[:, 1]) & np.isfinite(diag[:, 0])
                    diag2 = diag[mask]
                    img = pim.transform(diag2) if diag2.size else np.zeros(self.config.pim_size, dtype=float)

                # 展平写入特征
                flat = img.reshape(-1)
                for i, val in enumerate(flat):
                    feat[f"tda_pim_dim{dim}_{i}"] = float(val)

        return feat

    def point_clouds_to_tda_features(self, point_clouds: Sequence[np.ndarray]) -> Tuple[pd.DataFrame, List[int]]:
        """从点云列表批量提取特征。

        Args:
            point_clouds: list/tuple，每个元素为 (N,3)

        Returns:
            features_df: DataFrame
            valid_indices: 成功的样本索引
        """
        self._ensure_available()

        rows: List[Dict[str, float]] = []
        valid_indices: List[int] = []

        for i, pts in enumerate(point_clouds):
            try:
                if pts is None:
                    continue
                feat = self._point_cloud_to_features(pts)
                rows.append(feat)
                valid_indices.append(i)
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(), []

        df = pd.DataFrame(rows)
        df = df.select_dtypes(include=[np.number])
        df = df.fillna(0.0)

        self.feature_names = df.columns.tolist()
        return df, valid_indices

    def smiles_to_tda_features(
        self,
        smiles_list: Sequence[str],
        *,
        add_hs: bool = True,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """从 SMILES 列表生成 3D 构象并提取 TDA 特征。"""
        self._ensure_available()
        if not self.RDKIT_AVAILABLE:
            raise ImportError("smiles_to_tda_features 需要 RDKit。")

        point_clouds: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i, smi in enumerate(smiles_list):
            try:
                if smi is None or (isinstance(smi, float) and np.isnan(smi)):
                    continue
                pts = _smiles_to_3d_points(str(smi), add_hs=add_hs, seed=seed)
                if pts is None:
                    continue
                point_clouds.append(pts)
                valid_indices.append(i)
            except Exception:
                continue

        if not point_clouds:
            return pd.DataFrame(), []

        df, inner_valid = self.point_clouds_to_tda_features(point_clouds)
        # inner_valid 是 point_clouds 内部索引，需要映射回原 smiles_list 索引
        mapped_valid = [valid_indices[j] for j in inner_valid]
        return df, mapped_valid
