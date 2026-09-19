# -*- coding: utf-8 -*-
"""
pinn_model.py

Epoxy PINN / Physics-Guided 神经网络回归器（sklearn 风格封装）

- Tg 模式（tg）：DiBenedetto Tg-α 方程 + r-value 约束 α 上限
- 力学模量模式（mechanics）：Halpin–Tsai 复合材料模量模型
- generic：纯 MLP 回归（作为兜底）

说明：
- 本实现支持直接输入原始 DataFrame（包含 nanofiller_content 这类带单位字符串列），内部完成解析/清洗/归一化。
- 为保证模型可序列化（joblib/pickle）与跨机器加载，fit 结束后默认把权重移回 CPU。
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import inspect
import math
import os
import platform
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from .missing_value_handler import MissingValueHandler, build_missing_mask
from . import crosslink_physics as xphy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

try:
    from .task_manager import is_cancelled as _task_manager_is_cancelled
except Exception:
    def _task_manager_is_cancelled() -> bool:
        return False

from .epoxy_physics import (
    parse_first_number,
    parse_percent_to_fraction,
    alpha_max_from_r_torch,
    dibenedetto_tg_torch,
    get_filler_props,
    volume_fraction_from_wt_fraction_torch,
    halpin_tsai_torch,
)


def _set_seed(seed: int = 42):
    if not TORCH_AVAILABLE:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _raise_if_cancelled() -> None:
    if _task_manager_is_cancelled():
        raise RuntimeError("用户取消")


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception:
        pass


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _confidence_to_weight(conf: Union[str, float, int, None]) -> float:
    """r_confidence -> [0,1] 权重（用于物理约束强度的样本级缩放）"""
    if conf is None:
        return 0.6
    if isinstance(conf, (int, float, np.number)):
        v = float(conf)
        if 0.0 <= v <= 1.0:
            return v
        return 0.6
    s = str(conf).strip().lower()
    if s in {"high", "h", "较高", "高"}:
        return 1.0
    if s in {"medium", "mid", "m", "中", "一般"}:
        return 0.6
    if s in {"low", "l", "较低", "低"}:
        return 0.3
    return 0.6


def _infer_mode(mode: str, target_name: Optional[str]) -> str:
    m = (mode or "auto").strip().lower()
    if m != "auto":
        return m
    t = (target_name or "").strip().lower()
    if "tg" in t:
        return "tg"
    # 热稳定类目标（td5/td10/td50/tmax/char yield）：不走 DiBenedetto tg 分支，
    # 交联密度读出头将在后续版本接入 thermal 模式
    if any(k in t for k in ("td5", "td10", "td50", "tmax", "decomposition", "thermal_stability", "char_yield")):
        return "thermal"
    if "modulus" in t or "young" in t or "elastic" in t:
        return "mechanics"
    return "generic"


@dataclass
class _PreprocessPack:
    feature_names: List[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class EpoxyPINNRegressor(BaseEstimator, RegressorMixin):
    """Epoxy PINN 回归器（sklearn 风格）
    
    支持三种模式：
    - tg: 使用 DiBenedetto 方程预测玻璃化转变温度
    - mechanics: 使用 Halpin-Tsai 方程预测复合材料模量
    - generic: 纯神经网络回归（无物理约束）
    
    注意：如果发现预测值被截断，可以尝试：
    1. 设置 mode='generic' 关闭物理约束
    2. 降低 physics_weight 减弱物理约束影响
    """

    def __init__(
        self,
        mode: str = "auto",
        target_name: Optional[str] = None,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.15,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,  # 增大batch_size提高GPU利用率
        epochs: int = 500,
        patience: int = 50,
        physics_weight: float = 0.01,  # 大幅降低物理约束权重
        physics_formula: str = "standard",
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
        missing_value_strategy: str = "bayesian",
        missing_imputer_max_iter: int = 15,
        missing_n_imputations: int = 5,
        crosslink_physics_weight: float = 0.1,
        nu_column: str = "",
        nu_encoder_path: str = "",
        use_polymer_physics: bool = True,
        polymer_physics_features: str = "",
    ):
        self.mode = mode
        self.target_name = target_name
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.physics_weight = physics_weight
        self.physics_formula = physics_formula
        self.grad_clip = grad_clip
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.missing_value_strategy = missing_value_strategy
        self.missing_imputer_max_iter = missing_imputer_max_iter
        self.missing_n_imputations = missing_n_imputations
        # ν（交联密度）中间层：辅助监督权重与实测 ν 列名（空=自动探测 crosslink_density_mol_m3）
        # nu_encoder_path：阶段1冻结的 ν 嵌入模型路径（空=自动发现 models/crosslink_nu_encoder.joblib）
        self.crosslink_physics_weight = crosslink_physics_weight
        self.nu_column = nu_column
        self.nu_encoder_path = nu_encoder_path
        # 高分子物理指数（芳香度/sp3/柔性/BDE/内聚能）——从结构 SMILES 零标签计算
        self.use_polymer_physics = bool(use_polymer_physics)
        self.polymer_physics_features = polymer_physics_features

        # fitted attrs
        self._mode_: Optional[str] = None
        self._prep_: Optional[_PreprocessPack] = None
        self._model_: Optional[nn.Module] = None
        self._device_: str = "cpu"
        self._missing_handler_: Optional[MissingValueHandler] = None
        # ν 中间层状态
        self._nu_active_: bool = False
        self._nu_ch_: int = -1          # ν 残差 logit 通道索引
        self._nu_param_ch_: int = -1    # 斜率/K 通道索引（tg: Fox–Loshaek K；thermal/generic: ν 斜率）
        self._log_nu_ref_: float = float(np.log(2000.0))
        self._log_mc_ref_: float = float(np.log(2000.0))
        self._nu_encoder_ = None        # 阶段1冻结编码器（CrosslinkNuEncoder 或 None）

        # special column names
        self._col_r_: Optional[str] = None
        self._col_r_conf_: Optional[str] = None
        self._col_nf_type_: Optional[str] = None
        self._col_nf_content_: Optional[str] = None
        self._col_is_nf_: Optional[str] = None

    @staticmethod
    def is_available() -> bool:
        return bool(TORCH_AVAILABLE)

    def _select_device(self) -> str:
        if not TORCH_AVAILABLE:
            return "cpu"
        if self.device and self.device.lower() in {"cpu", "cuda"}:
            if self.device.lower() == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return self.device.lower()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _identify_special_columns(self, df: pd.DataFrame):
        cols = list(df.columns)

        # 化学计量比列识别：正则候选覆盖 formulation_r_value / *_equivalent_ratio 等实际命名。
        # alpha_max_from_r 对 r 与 1/r 对称，因此 equivalent_ratio 类列物理上等价可用。
        r_candidates: List[Tuple[int, str]] = []
        for c in cols:
            cl = str(c).strip().lower()
            if cl == "r_value" or cl.endswith("_r_value"):
                r_candidates.append((0, c))  # 显式 r_value 优先
            elif "equivalent_ratio" in cl:
                r_candidates.append((1, c))
            elif "stoich" in cl:
                r_candidates.append((2, c))
        if r_candidates:
            r_candidates.sort(key=lambda t: (t[0], len(t[1])))
            self._col_r_ = r_candidates[0][1]

        for c in ["r_confidence", "R_confidence", "stoich_confidence"]:
            if c in cols:
                self._col_r_conf_ = c
                break

        for c in ["nanofiller_type", "nano_filler_type", "filler_type"]:
            if c in cols:
                self._col_nf_type_ = c
                break
        for c in ["nanofiller_content", "nano_filler_content", "filler_content", "vf", "Vf", "nanofiller_wt"]:
            if c in cols:
                self._col_nf_content_ = c
                break
        for c in ["is_nanofilled", "is_filled", "filled"]:
            if c in cols:
                self._col_is_nf_ = c
                break

    # 非数值文本列黑名单：结构/标识/测试条件等语义列，绝不允许被 parse_first_number 污染成假数值。
    # _ALWAYS_DROP：分子结构标识列 → 永远直接剔除（结构信息由外部分子特征管线负责）。
    # 其余黑名单列：低基数(≤12)→ one-hot；高基数 → 直接剔除。
    _TEXT_COLS_ALWAYS_DROP = ("structure", "smiles", "inchi", "bigsmiles")
    _TEXT_COL_BLACKLIST = (
        "_method", "_atmosphere", "_standard", "mechanism", "_formula",
        "_source", "_basis", "_unit", "layup", "铺层", "工艺路线", "结构",
        "test_method", "specimen",
    )

    def _build_numeric_features(self, df_raw: pd.DataFrame, fit_categoricals: bool = False) -> pd.DataFrame:
        df = df_raw.copy()

        drop_text_cols = set()
        if self._col_nf_type_ is not None:
            drop_text_cols.add(self._col_nf_type_)
        if self._col_r_conf_ is not None:
            drop_text_cols.add(self._col_r_conf_)

        onehot_frames: List[pd.DataFrame] = []
        if fit_categoricals:
            cat_levels: Dict[str, List[str]] = {}
        else:
            cat_levels = dict(getattr(self, "_cat_levels_", {}) or {})

        for c in list(df.columns):
            if not isinstance(c, str) or pd.api.types.is_numeric_dtype(df[c]):
                continue
            cl = c.lower()
            # 分子结构标识列：无论基数大小永远剔除，禁止 one-hot / parse_first_number
            if any(p in cl for p in self._TEXT_COLS_ALWAYS_DROP):
                drop_text_cols.add(c)
                continue
            # 历史逻辑保留：明确含 smiles/inchi 的非数值列直接剔除
            if "smiles" in cl or "inchi" in cl:
                drop_text_cols.add(c)
                continue
            # 黑名单语义列：禁止落入 parse_first_number
            if any(p in cl for p in self._TEXT_COL_BLACKLIST):
                drop_text_cols.add(c)
                s = df[c].astype(str)
                nuniq = int(s.nunique(dropna=True))
                if 2 <= nuniq <= 12:
                    if fit_categoricals:
                        levels = sorted(s.dropna().unique().tolist())[:12]
                        cat_levels[c] = levels
                    else:
                        levels = cat_levels.get(c)
                        if not levels:
                            continue
                    cat_idx = pd.Categorical(
                        s.where(s.isin(levels), other="__other__"),
                        categories=list(levels) + ["__other__"],
                    )
                    dummies = pd.get_dummies(cat_idx, prefix=c, dtype=float)
                    # 关键：pd.get_dummies(Categorical) 返回 RangeIndex，必须回填原索引，
                    # 否则 concat(axis=1) 按索引对齐会导致行膨胀/错位（非连续索引输入时）
                    dummies.index = df.index
                    onehot_frames.append(dummies)
                # 高基数（如 SMILES 结构列）或常数列：直接剔除
                continue

        if drop_text_cols:
            df = df.drop(columns=[c for c in drop_text_cols if c in df.columns], errors="ignore")
        if onehot_frames:
            df = pd.concat([df] + onehot_frames, axis=1)
        if fit_categoricals:
            self._cat_levels_ = cat_levels

        # 尝试将剩余的 object 列转为数值（处理混杂的字符串）
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue

            s = df[c]
            converted = pd.to_numeric(s, errors="coerce")
            ok_ratio = float(converted.notna().mean()) if len(converted) else 0.0

            if ok_ratio >= 0.50:
                df[c] = converted
            else:
                df[c] = s.apply(parse_first_number)

        # 最后的兜底转换
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1, how="all")
        return df

    def _fit_preprocess(self, df_num: pd.DataFrame) -> _PreprocessPack:
        """拟合预处理参数（缺失填充 + 标准化），并自动剔除常数/全空特征。"""
        # Defensive: require some numeric features
        if df_num is None or df_num.shape[1] == 0:
            raise ValueError(
                "未检测到任何可用的数值特征（可能只选择了 SMILES/文本列，或所有数值列均为空）。\n"
                "请先在【分子特征】或【数据清洗】中生成/保留数值特征，再训练 Epoxy PINN。"
            )

        # Clean inf and drop all-NaN cols
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.dropna(axis=1, how="all")
        if df_num.shape[1] == 0:
            raise ValueError(
                "所有特征列均为 NaN/Inf，无法训练 Epoxy PINN。\n"
                "建议：检查特征列是否提取成功，或在特征选择中去掉全空列。"
            )

        X_raw = df_num.to_numpy(dtype=float)
        # 记录插补器拟合时的完整列集合：后续 _transform 需要先展开到同一列集再插补
        # （下方会剔除常数列，但插补器是按全列拟合的，直接喂子集会列数不匹配）
        self._imputer_feature_names_ = list(df_num.columns)
        self._missing_handler_ = MissingValueHandler(
            strategy=self.missing_value_strategy,
            random_state=self.seed,
            max_iter=self.missing_imputer_max_iter,
            n_imputations=self.missing_n_imputations,
        )
        X_imp = self._missing_handler_.fit_transform(X_raw)
        df_imp = pd.DataFrame(X_imp, columns=df_num.columns)

        median_s = df_imp.median(numeric_only=True)
        mean_s = df_imp.mean(numeric_only=True)
        std_s = df_imp.std(numeric_only=True, ddof=0)

        feature_names_all = list(df_num.columns)
        median = median_s.to_numpy(dtype=float)
        mean = mean_s.to_numpy(dtype=float)
        std = std_s.to_numpy(dtype=float)

        nonconst_mask = np.isfinite(std) & (std >= 1e-8)
        if int(nonconst_mask.sum()) == 0:
            raise ValueError(
                "所有数值特征在有效样本上都是常数或无效（标准差≈0 或 NaN）。\n"
                "这会导致 PINN 只能输出常数。请重新选择/生成更有信息量的特征。"
            )

        if int(nonconst_mask.sum()) < len(feature_names_all):
            feature_names = [f for f, keep in zip(feature_names_all, nonconst_mask) if keep]
            median = median[nonconst_mask]
            mean = mean[nonconst_mask]
            std = std[nonconst_mask]
        else:
            feature_names = feature_names_all

        std = np.where(std < 1e-8, 1.0, std)
        return _PreprocessPack(feature_names=feature_names, median=median, mean=mean, std=std)

    def _df_cache_key(self, df_raw: pd.DataFrame) -> Optional[str]:
        """数据内容指纹，用于 _transform 结果缓存（相同 DataFrame 直接复用插补/标准化结果）。"""
        try:
            h = pd.util.hash_pandas_object(df_raw, index=True).to_numpy(dtype=np.int64)
            return hashlib.md5(h.tobytes()).hexdigest()
        except Exception:
            return None

    def _transform(
        self,
        df_raw: pd.DataFrame,
        return_missing_mask: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]] | Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        if self._prep_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if self._missing_handler_ is None:
            raise RuntimeError("Missing-value handler is not fitted yet.")

        # ---- 结果缓存：fit 内部 + predict(train)/predict(test) 的重复变换直接命中 ----
        cache: "OrderedDict[str, Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]" = getattr(
            self, "_transform_cache_", None
        )
        key = self._df_cache_key(df_raw)
        if cache is not None and key is not None and key in cache:
            x_out, aux_c, mask_c = cache[key]
            return (x_out, aux_c, mask_c) if return_missing_mask else (x_out, aux_c)

        n = len(df_raw)
        aux: Dict[str, np.ndarray] = {}

        r = np.ones(n, dtype=float)
        if self._col_r_ is not None and self._col_r_ in df_raw.columns:
            r = df_raw[self._col_r_].apply(parse_first_number).to_numpy(dtype=float)
            r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
        aux["r"] = r.astype(np.float32)

        conf_w = np.full(n, 0.6, dtype=float)
        if self._col_r_conf_ is not None and self._col_r_conf_ in df_raw.columns:
            conf_w = df_raw[self._col_r_conf_].apply(_confidence_to_weight).to_numpy(dtype=float)
        aux["r_conf_w"] = np.clip(conf_w, 0.0, 1.0).astype(np.float32)

        nf_w = np.zeros(n, dtype=float)
        if self._col_nf_content_ is not None and self._col_nf_content_ in df_raw.columns:
            nf_w = df_raw[self._col_nf_content_].apply(parse_percent_to_fraction).to_numpy(dtype=float)
            nf_w = np.where(np.isfinite(nf_w), nf_w, 0.0)
        aux["nf_w"] = np.clip(nf_w, 0.0, 0.95).astype(np.float32)

        is_nf = np.zeros(n, dtype=float)
        if self._col_is_nf_ is not None and self._col_is_nf_ in df_raw.columns:
            is_nf = df_raw[self._col_is_nf_].apply(parse_first_number).to_numpy(dtype=float)
            is_nf = np.where(np.isfinite(is_nf), is_nf, 0.0)
        aux["is_nf"] = (is_nf > 0.5).astype(np.float32)

        nf_type = [""] * n
        if self._col_nf_type_ is not None and self._col_nf_type_ in df_raw.columns:
            nf_type = df_raw[self._col_nf_type_].astype(str).fillna("").tolist()

        Ef = np.zeros(n, dtype=float)
        rho_f = np.zeros(n, dtype=float)
        for i, t in enumerate(nf_type):
            props = get_filler_props(t)
            Ef[i] = props.E_gpa
            rho_f[i] = props.rho_g_cm3
        aux["Ef"] = Ef.astype(np.float32)
        aux["rho_f"] = rho_f.astype(np.float32)

        # ν（交联密度）物理基线量：理论 ν 与 Mc（log 空间，供潜变量残差与 Fox–Loshaek 使用）
        xl = xphy.compute_crosslink_features(df_raw)
        if len(xl.columns) > 0:
            nu_th = xl["xl_nu_theory_mol_m3"].to_numpy(dtype=float)
            mc = xl["xl_Mc_g_mol"].to_numpy(dtype=float)
        else:
            nu_th = np.full(n, np.nan)
            mc = np.full(n, np.nan)
        aux["log_nu_theory"] = np.where(
            np.isfinite(nu_th) & (nu_th > 0), np.log(nu_th), np.nan
        ).astype(np.float32)
        aux["log_Mc"] = np.where(
            np.isfinite(mc) & (mc > 0), np.log(mc), np.nan
        ).astype(np.float32)
        aux["alpha_gel"] = xl["xl_alpha_gel"].to_numpy(dtype=np.float32) if "xl_alpha_gel" in xl.columns else np.full(n, np.nan, dtype=np.float32)
        aux["alpha_max"] = xl["xl_alpha_max"].to_numpy(dtype=np.float32) if "xl_alpha_max" in xl.columns else np.full(n, np.nan, dtype=np.float32)

        df_num = self._build_numeric_features(df_raw)

        # 展开到插补器拟合时的完整列集合（缺失列补 NaN，多余列丢弃）
        full_cols = list(getattr(self, "_imputer_feature_names_", None) or self._prep_.feature_names)
        for c in full_cols:
            if c not in df_num.columns:
                df_num[c] = np.nan
        df_num = df_num[full_cols].copy()

        X = df_num.to_numpy(dtype=float)
        missing_mask = build_missing_mask(X)
        X = np.where(np.isfinite(X), X, np.nan)
        X = self._missing_handler_.transform(X)

        # 仅保留非常数特征子集（与 _prep_.feature_names 对齐）再做标准化
        prep_names = list(self._prep_.feature_names)
        if prep_names != full_cols:
            keep_idx = [full_cols.index(c) for c in prep_names]
            X = X[:, keep_idx]
            missing_mask = missing_mask[:, keep_idx]

        X = (X - self._prep_.mean) / self._prep_.std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_out = X.astype(np.float32)

        if cache is not None and key is not None:
            try:
                cache[key] = (X_out, aux, missing_mask.astype(np.float32))
                while len(cache) > 2:
                    cache.popitem(last=False)
            except Exception:
                pass

        if return_missing_mask:
            return X_out, aux, missing_mask.astype(np.float32)
        return X_out, aux

    def _uses_missing_mask(self) -> bool:
        return False

    def _output_dim_for(self, mode: str) -> int:
        """输出通道布局：基座通道 + (ν 残差 logit + 物理参数通道)。Transformer 子类共用。"""
        if mode == "tg":
            base = 5  # [0]直接 | [1]Tg0 | [2]ΔTg | [3]λ | [4]α
        elif mode == "mechanics":
            base = 2  # [0]Em | [1]ξ
        else:
            base = 1  # [0]直接
        if self._nu_active_:
            base += 2  # [base]=ν 残差 logit, [base+1]=物理参数(斜率或 K)
        return base

    def _make_model(self, input_dim: int) -> nn.Module:
        mode = self._mode_ or "generic"
        out_dim = self._output_dim_for(mode)
        return _MLP(input_dim=input_dim, hidden_dim=int(self.hidden_dim), n_layers=int(self.n_layers),
                    dropout=float(self.dropout), out_dim=out_dim)

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("EpoxyPINNRegressor 需要 torch，请先安装 torch>=2.1.0")

        _set_seed(int(self.seed))

        # 新一轮 fit 清空 transform 缓存（插补器/预处理全部重建）
        self._transform_cache_ = OrderedDict()

        if isinstance(X, pd.DataFrame):
            df_raw = X.copy()
        else:
            X_arr = np.asarray(X)
            df_raw = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])

        y_arr = np.asarray(y).reshape(-1).astype(np.float32)
        valid = np.isfinite(y_arr)
        if valid.sum() < 20:
            raise ValueError("有效样本过少（<20），请检查目标列是否包含大量缺失/非数值。")
        df_raw = df_raw.iloc[valid].reset_index(drop=True)

        # 高分子物理指数自动注入（零标签；幂等；无结构列时静默跳过）
        try:
            from .polymer_physics import augment_polymer_physics
            _pf = [s.strip() for s in (self.polymer_physics_features or "").split(",") if s.strip()]
            df_raw = augment_polymer_physics(
                df_raw, features=_pf or None, enabled=bool(self.use_polymer_physics)
            )
        except Exception:
            pass
        y_arr = y_arr[valid]

        # 防御：目标变量几乎为常数时，任何模型都只能学到常数输出
        y_std = float(np.nanstd(y_arr))
        if (not np.isfinite(y_std)) or (y_std < 1e-8):
            raise ValueError(
                "目标变量几乎是常数（std≈0），PINN无法学习有效映射。请检查目标列或过滤口径/异常值。"
            )

        self._identify_special_columns(df_raw)

        # ---- ν 实测列：从特征中剥离（防泄漏），作为 ν 潜变量的辅助监督 ----
        nu_col = (self.nu_column or "").strip()
        if not nu_col:
            nu_col = "crosslink_density_mol_m3" if "crosslink_density_mol_m3" in df_raw.columns else ""
        nu_measured = None
        if nu_col and nu_col in df_raw.columns:
            nu_measured = pd.to_numeric(df_raw[nu_col], errors="coerce").to_numpy(dtype=float)
            df_raw = df_raw.drop(columns=[nu_col])

        df_num = self._build_numeric_features(df_raw, fit_categoricals=True)
        self._prep_ = self._fit_preprocess(df_num)

        X_scaled, aux, missing_mask = self._transform(df_raw, return_missing_mask=True)

        self._mode_ = _infer_mode(self.mode, self.target_name)
        device = self._select_device()
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

        X_tensor = torch.from_numpy(X_scaled)  # 保持在 CPU 上
        X_missing_tensor = torch.from_numpy(missing_mask)
        # 为了稳定训练：loss 统一在标准化 y 空间计算（预测与目标同尺度，消除梯度条件数问题）
        y_scale = float(np.nanstd(y_arr))
        if (not np.isfinite(y_scale)) or (y_scale < 1e-8):
            y_scale = 1.0
        self._y_scale_ = y_scale
        self._y_mean_ = float(np.nanmean(y_arr))
        
        # 计算 Tg 范围缩放因子（用于放宽物理约束）
        y_range = float(np.nanmax(y_arr) - np.nanmin(y_arr))
        if (not np.isfinite(y_range)) or (y_range < 1.0):
            y_range = 200.0  # 默认范围
        self._tg_delta_scale_ = max(y_range * 1.5, 200.0)  # 至少200°C，或数据范围的1.5倍
        
        # 记录目标变量统计信息（用于调试）
        self._y_min_ = float(np.nanmin(y_arr))
        self._y_max_ = float(np.nanmax(y_arr))

        y_tensor = torch.from_numpy(y_arr).view(-1, 1)  # 保持在 CPU 上

        r_tensor = torch.from_numpy(aux["r"]).view(-1, 1)
        conf_tensor = torch.from_numpy(aux["r_conf_w"]).view(-1, 1)
        nf_w_tensor = torch.from_numpy(aux["nf_w"]).view(-1, 1)
        is_nf_tensor = torch.from_numpy(aux["is_nf"]).view(-1, 1)
        Ef_tensor = torch.from_numpy(aux["Ef"]).view(-1, 1)
        rho_f_tensor = torch.from_numpy(aux["rho_f"]).view(-1, 1)

        self._identify_special_columns(df_raw)

        # ---- ν 中间层张量（在 _transform 之前确定 active 状态，供 out_dim 使用）----
        log_nu_th = aux["log_nu_theory"].astype(float)
        log_mc = aux["log_Mc"].astype(float)
        self._log_nu_ref_ = float(np.nanmedian(log_nu_th)) if np.isfinite(log_nu_th).any() else float(np.log(2000.0))
        self._log_mc_ref_ = float(np.nanmedian(log_mc)) if np.isfinite(log_mc).any() else float(np.log(2000.0))

        # ---- 阶段1冻结编码器：为全部行提供 ν 物理基线（两阶段架构核心）----
        encoder = getattr(self, "_nu_encoder_", None)
        if encoder is None:
            enc_path = (self.nu_encoder_path or "").strip()
            if not enc_path:
                try:
                    from .crosslink_nu_model import default_encoder_path
                    enc_path = default_encoder_path()
                except Exception:
                    enc_path = ""
            if enc_path and os.path.exists(enc_path):
                try:
                    from .crosslink_nu_model import CrosslinkNuEncoder
                    encoder = CrosslinkNuEncoder.load(enc_path)
                    self._nu_encoder_ = encoder
                except Exception:
                    encoder = None
        log_nu_enc: Optional[np.ndarray] = None
        if encoder is not None:
            try:
                log_nu_enc = np.asarray(encoder.predict_log_nu(df_raw), dtype=float)
                if len(log_nu_enc) != len(df_raw):
                    log_nu_enc = None
            except Exception:
                log_nu_enc = None

        # ν 基线优先级：冻结编码器 > 理论公式 > 参考中位数
        if log_nu_enc is not None:
            log_nu_base = np.where(np.isfinite(log_nu_enc), log_nu_enc, log_nu_th)
        else:
            log_nu_base = log_nu_th

        theory_frac = float(np.isfinite(log_nu_th).mean())
        meas_frac = 0.0 if nu_measured is None else float(np.isfinite(nu_measured).mean())
        enc_frac = 0.0 if log_nu_enc is None else float(np.isfinite(log_nu_enc).mean())
        self._nu_active_ = bool(
            float(self.crosslink_physics_weight) > 0.0
            and (encoder is not None or max(theory_frac, meas_frac) >= 0.2)
        )
        if self._nu_active_:
            base_dim = self._output_dim_for(self._mode_ or "generic") - 2
            self._nu_ch_ = base_dim
            self._nu_param_ch_ = base_dim + 1

        nub_np = np.where(np.isfinite(log_nu_base), log_nu_base, self._log_nu_ref_).astype(np.float32)
        mcb_np = np.where(np.isfinite(log_mc), log_mc, self._log_mc_ref_).astype(np.float32)
        if nu_measured is not None:
            with np.errstate(all="ignore"):
                log_meas = np.where(
                    np.isfinite(nu_measured) & (nu_measured > 0),
                    np.log(np.where(nu_measured > 0, nu_measured, 1.0)),
                    np.nan,
                )
            nut_np = np.where(np.isfinite(log_meas), log_meas, nub_np).astype(np.float32)
            # 实测监督权重 1.0；编码器/理论锚 0.1/0.3；两者皆无 0
            anchor_w = 0.1 if encoder is not None else 0.3
            nuw_np = np.where(
                np.isfinite(log_meas), 1.0, np.where(np.isfinite(log_nu_base), anchor_w, 0.0)
            ).astype(np.float32)
        else:
            nut_np = nub_np.copy()
            anchor_w = 0.1 if encoder is not None else 0.3
            nuw_np = np.where(np.isfinite(log_nu_base), anchor_w, 0.0).astype(np.float32)

        nub_tensor = torch.from_numpy(nub_np).view(-1, 1)
        nut_tensor = torch.from_numpy(nut_np).view(-1, 1)
        nuw_tensor = torch.from_numpy(nuw_np).view(-1, 1)
        mcb_tensor = torch.from_numpy(mcb_np).view(-1, 1)

        # split train/val（在 CPU 上进行索引操作）
        n = X_tensor.shape[0]
        idx = torch.randperm(n)  # 在 CPU 上
        val_size = max(1, int(0.1 * n))
        val_idx = idx[:val_size]
        tr_idx = idx[val_size:]

        def _sub(t: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
            return t.index_select(0, inds)

        train_tensors = [
            _sub(X_tensor, tr_idx),
            _sub(X_missing_tensor, tr_idx),
            _sub(y_tensor, tr_idx),
            _sub(r_tensor, tr_idx),
            _sub(conf_tensor, tr_idx),
            _sub(nf_w_tensor, tr_idx),
            _sub(is_nf_tensor, tr_idx),
            _sub(Ef_tensor, tr_idx),
            _sub(rho_f_tensor, tr_idx),
            _sub(nub_tensor, tr_idx),
            _sub(nut_tensor, tr_idx),
            _sub(nuw_tensor, tr_idx),
            _sub(mcb_tensor, tr_idx),
        ]
        val_tensors = [
            _sub(X_tensor, val_idx),
            _sub(X_missing_tensor, val_idx),
            _sub(y_tensor, val_idx),
            _sub(r_tensor, val_idx),
            _sub(conf_tensor, val_idx),
            _sub(nf_w_tensor, val_idx),
            _sub(is_nf_tensor, val_idx),
            _sub(Ef_tensor, val_idx),
            _sub(rho_f_tensor, val_idx),
            _sub(nub_tensor, val_idx),
            _sub(nut_tensor, val_idx),
            _sub(nuw_tensor, val_idx),
            _sub(mcb_tensor, val_idx),
        ]

        # 根据系统类型配置 num_workers
        # 可通过环境变量 PYTORCH_NUM_WORKERS 覆盖
        if "PYTORCH_NUM_WORKERS" in os.environ:
            try:
                num_workers = max(0, int(os.environ["PYTORCH_NUM_WORKERS"]))
            except ValueError:
                num_workers = 0
        elif platform.system() == 'Windows':
            num_workers = 0  # Windows 默认禁用（避免多进程问题）
        else:
            # Linux/Mac 使用多进程数据加载
            cpu_count = os.cpu_count() or 1
            num_workers = min(4, max(2, cpu_count // 4))  # 更保守的默认值

        # 动态调整 batch_size：优化 GPU 利用率
        n_samples = X_tensor.shape[0]
        # 对于 GPU，使用更大的 batch_size 以提高利用率
        if device == "cuda":
            # GPU: 小数据集直接用全部数据，大数据集分批
            if n_samples <= 512:
                effective_batch_size = n_samples  # 小数据集用全batch
            elif n_samples <= 2048:
                effective_batch_size = min(1024, n_samples)
            else:
                effective_batch_size = int(self.batch_size)
        else:
            # CPU: 使用较小的 batch_size
            effective_batch_size = min(int(self.batch_size), max(32, n_samples // 8))
        
        # 梯度累积：模拟更大batch
        accumulation_steps = max(1, 1024 // effective_batch_size)

        # 小数据集可直接缓存到GPU，减少CPU->GPU拷贝
        data_on_gpu = False
        if device == "cuda":
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                n_features = int(X_tensor.shape[1])
                est_bytes = int(n_samples) * (n_features * 2 + 7) * 4
                if est_bytes < int(total_mem * 0.35):
                    data_on_gpu = True
            except Exception:
                data_on_gpu = False
        force_gpu_cache = str(os.environ.get("PINN_DATA_ON_GPU", "")).strip().lower() in {"1", "true", "yes"}
        if device == "cuda" and force_gpu_cache:
            data_on_gpu = True

        train_tensors_cpu = train_tensors
        val_tensors_cpu = val_tensors
        if data_on_gpu:
            try:
                train_tensors = [t.to(device, non_blocking=True) for t in train_tensors_cpu]
                val_tensors = [t.to(device, non_blocking=True) for t in val_tensors_cpu]
                num_workers = 0
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    data_on_gpu = False
                    train_tensors = train_tensors_cpu
                    val_tensors = val_tensors_cpu
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                else:
                    raise

        train_ds = TensorDataset(*train_tensors)
        val_ds = TensorDataset(*val_tensors)

        train_steps = int(math.ceil(len(train_tensors[0]) / float(effective_batch_size)))
        val_steps = int(math.ceil(len(val_tensors[0]) / float(effective_batch_size)))

        train_loader = None
        val_loader = None
        if not data_on_gpu:
            loader_kwargs = {
                "batch_size": effective_batch_size,
                "drop_last": False,
                "pin_memory": (device == "cuda" and not data_on_gpu),
                "num_workers": num_workers,
                "persistent_workers": (num_workers > 0) if num_workers > 0 else False,
            }
            if num_workers > 0:
                try:
                    if "prefetch_factor" in inspect.signature(DataLoader).parameters:
                        loader_kwargs["prefetch_factor"] = 2
                except (TypeError, ValueError):
                    pass

            train_loader = DataLoader(
                train_ds,
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                val_ds,
                shuffle=False,
                **loader_kwargs,
            )

            train_steps = len(train_loader)
            val_steps = len(val_loader)

        self._model_ = self._make_model(input_dim=X_tensor.shape[1]).to(device)
        use_compile = False
        if device == "cuda":
            compile_flag = str(os.environ.get("PINN_TORCH_COMPILE", "")).strip().lower()
            if compile_flag in {"1", "true", "yes"} and hasattr(torch, "compile"):
                compile_mode = os.environ.get("PINN_TORCH_COMPILE_MODE", "reduce-overhead")
                try:
                    self._model_ = torch.compile(self._model_, mode=compile_mode)
                    use_compile = True
                except Exception:
                    use_compile = False
        
        # 计算模型参数量
        n_params = sum(p.numel() for p in self._model_.parameters())
        
        # 打印训练配置信息（始终打印，帮助用户了解训练状态）
        print(f"\n{'='*55}")
        print(f"  [EpoxyPINN] 训练配置")
        print(f"{'='*55}")
        print(f"  设备: {device.upper()}", end="")
        if device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f" ({gpu_name}, {gpu_mem:.1f}GB)")
            except:
                print(" ✓")
        else:
            print(" (无GPU加速)")
        print(f"  样本数: {n_samples} (训练: {len(tr_idx)}, 验证: {len(val_idx)})")
        print(f"  Batch大小: {effective_batch_size} | 梯度累积: {accumulation_steps}步")
        print(f"  每epoch迭代: {train_steps} | 模型参数: {n_params:,}")
        print(f"  特征维度: {X_tensor.shape[1]} | DataLoader workers: {num_workers}")
        _nu_status = "已嵌入冻结编码器 ✓" if getattr(self, "_nu_encoder_", None) is not None else (
            "理论基线" if self._nu_active_ else "未激活")
        print(f"  ν中间层: {_nu_status} | ν监督权重: {float(self.crosslink_physics_weight):.2f}")
        print(f"  混合精度(AMP): {'已启用 ⚡' if device == 'cuda' else '不适用'}")
        if device == "cuda":
            print(f"  torch.compile: {'已启用 ⚡' if use_compile else '未启用'}")
        
        # 对小数据集给出明确提示
        if n_samples < 500:
            print(f"\n  ⚠️ 数据量较小 ({n_samples}样本)")
            print(f"     - GPU/CPU利用率低是正常现象")
            print(f"     - 每个batch计算很快完成，大部分时间在等待")
            print(f"     - 建议：增加数据量或使用CPU训练（效率相近）")
        elif n_samples < 2000:
            print(f"\n  ℹ️ 中等数据集({n_samples}样本)，GPU利用率可能偏低")
        print(f"{'='*55}\n")
        
        opt_name = str(self.optimizer).strip().lower()
        optimizer_kwargs = {"lr": float(self.lr), "weight_decay": float(self.weight_decay)}

        optimizer = None
        if opt_name == "adamw":
            if device == "cuda":
                try:
                    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                        optimizer_kwargs["fused"] = True
                except (TypeError, ValueError):
                    pass
            try:
                optimizer = torch.optim.AdamW(self._model_.parameters(), **optimizer_kwargs)
            except (TypeError, RuntimeError):
                optimizer_kwargs.pop("fused", None)
                optimizer = torch.optim.AdamW(self._model_.parameters(), **optimizer_kwargs)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self._model_.parameters(), **optimizer_kwargs)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self._model_.parameters(),
                lr=float(self.lr),
                momentum=0.9,
                nesterov=True,
                weight_decay=float(self.weight_decay),
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self._model_.parameters(), **optimizer_kwargs)

        if optimizer is None:
            print(f"⚠ 未识别的 PINN 优化器 '{self.optimizer}', 已回退到 AdamW")
            optimizer = torch.optim.AdamW(self._model_.parameters(), **optimizer_kwargs)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.6, patience=8, verbose=False
            )
        except TypeError:
            # 兼容旧版本 torch：不支持 verbose 关键字参数
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.6, patience=8
            )

        huber = nn.SmoothL1Loss()

        # 标准化 y 空间的常量张量（loss 与预测同尺度）
        y_mean_t = torch.tensor(float(getattr(self, "_y_mean_", 0.0)), device=device)
        y_scale_t = torch.tensor(float(getattr(self, "_y_scale_", 1.0)), device=device)

        # 训练曲线（用于 UI 展示）
        self.train_mse_curve = []
        self.test_mse_curve = []
        self.train_losses = []
        self.val_losses = []

        best_val = float("inf")
        best_state = None
        no_improve = 0

        phys_w = float(self.physics_weight)
        grad_clip = float(self.grad_clip) if self.grad_clip is not None else 0.0

        # 混合精度训练（AMP）- 显著提高GPU利用率和训练速度
        use_amp = False
        scaler = None
        autocast_ctx = contextlib.nullcontext
        if device == "cuda":
            try:
                # PyTorch 1.6+ 支持 AMP
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                use_amp = True
                autocast_ctx = torch.cuda.amp.autocast
            except ImportError:
                pass

        def _iter_gpu_batches(tensors, batch_size, shuffle):
            n_local = int(tensors[0].shape[0])
            if shuffle:
                order = torch.randperm(n_local, device=device)
            else:
                order = torch.arange(n_local, device=device)
            for start in range(0, n_local, batch_size):
                idx = order[start:start + batch_size]
                yield [t.index_select(0, idx) for t in tensors]

        try:
            for epoch in range(int(self.epochs)):
                _raise_if_cancelled()
                self._model_.train()
                # 使用GPU tensor累加，避免每batch同步到CPU
                train_loss_accum = torch.tensor(0.0, device=device)
                train_mse_accum = torch.tensor(0.0, device=device)
                train_n = 0

                optimizer.zero_grad(set_to_none=True)
                if data_on_gpu:
                    batch_iter = _iter_gpu_batches(train_tensors, effective_batch_size, shuffle=True)
                else:
                    batch_iter = train_loader

                for step_idx, batch in enumerate(batch_iter, start=1):
                    _raise_if_cancelled()
                    if data_on_gpu:
                        xb, xmb, yb, rb, confb, nfw, isnf, Ef, rhof, nub, nut, nuw, mcb = batch
                    else:
                        # 将数据移到GPU（pin_memory会加速这个过程）
                        xb, xmb, yb, rb, confb, nfw, isnf, Ef, rhof, nub, nut, nuw, mcb = [t.to(device, non_blocking=True) for t in batch]

                    # 使用混合精度训练
                    with autocast_ctx():
                        pred, phys_pen, log_nu_pred = self._forward_with_physics(xb, xmb, rb, confb, nfw, isnf, Ef, rhof, nub, mcb)
                        # 统一在标准化 y 空间计算 loss：预测与目标同尺度
                        loss_data = huber((pred - y_mean_t) / y_scale_t, (yb - y_mean_t) / y_scale_t)
                        # tg 分支的 phys_pen 已在内部按 y_scale² 自归一，这里不再重复除
                        loss = loss_data + phys_w * phys_pen
                        if log_nu_pred is not None:
                            # ν 潜变量辅助监督：实测标签(1.0) / 理论锚(0.3) 加权 Huber（log 空间）
                            nu_err = F.smooth_l1_loss(log_nu_pred.float(), nut.float(), reduction="none")
                            wsum = nuw.float().sum().clamp(min=1.0)
                            loss = loss + float(self.crosslink_physics_weight) * (nu_err * nuw.float()).sum() / wsum

                    loss_to_backward = loss / float(accumulation_steps)
                    if use_amp:
                        scaler.scale(loss_to_backward).backward()
                    else:
                        loss_to_backward.backward()

                    if step_idx % accumulation_steps == 0 or step_idx == train_steps:
                        if grad_clip and grad_clip > 0:
                            if use_amp:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self._model_.parameters(), max_norm=grad_clip)
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    # 在GPU上累加，不立即同步到CPU
                    batch_size = len(yb)
                    with torch.no_grad():
                        train_loss_accum += loss.detach() * batch_size
                        train_mse_accum += torch.mean((pred.detach() - yb) ** 2) * batch_size
                    train_n += batch_size

                # epoch结束时才同步到CPU（只同步一次）
                train_loss = float(train_loss_accum.cpu().item()) / max(1, train_n)
                train_mse = float(train_mse_accum.cpu().item()) / max(1, train_n)

                _raise_if_cancelled()
                self._model_.eval()
                val_loss_accum = torch.tensor(0.0, device=device)
                val_mse_accum = torch.tensor(0.0, device=device)
                val_n = 0
                with torch.no_grad():
                    if data_on_gpu:
                        batch_iter = _iter_gpu_batches(val_tensors, effective_batch_size, shuffle=False)
                    else:
                        batch_iter = val_loader

                    for batch in batch_iter:
                        _raise_if_cancelled()
                        # 将数据移到GPU
                        if data_on_gpu:
                            xb, xmb, yb, rb, confb, nfw, isnf, Ef, rhof, nub, nut, nuw, mcb = batch
                        else:
                            xb, xmb, yb, rb, confb, nfw, isnf, Ef, rhof, nub, nut, nuw, mcb = [t.to(device, non_blocking=True) for t in batch]
                        
                        with autocast_ctx():
                            pred, phys_pen, log_nu_pred = self._forward_with_physics(xb, xmb, rb, confb, nfw, isnf, Ef, rhof, nub, mcb)
                            # 同训练循环：标准化 y 空间 loss + ν 辅助监督
                            loss_data = huber((pred - y_mean_t) / y_scale_t, (yb - y_mean_t) / y_scale_t)
                            loss = loss_data + phys_w * phys_pen
                            if log_nu_pred is not None:
                                nu_err = F.smooth_l1_loss(log_nu_pred.float(), nut.float(), reduction="none")
                                wsum = nuw.float().sum().clamp(min=1.0)
                                loss = loss + float(self.crosslink_physics_weight) * (nu_err * nuw.float()).sum() / wsum

                        batch_size = len(yb)
                        val_loss_accum += loss * batch_size
                        val_mse_accum += torch.mean((pred - yb) ** 2) * batch_size
                        val_n += batch_size

                val_loss = float(val_loss_accum.cpu().item()) / max(1, val_n)
                val_mse = float(val_mse_accum.cpu().item()) / max(1, val_n)
                scheduler.step(val_loss)

                if self.verbose:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(f"[EpoxyPINN] epoch={epoch+1:03d} train={train_loss:.4f} val={val_loss:.4f} lr={lr_now:.2e}")

                self.train_mse_curve.append(train_mse)
                self.test_mse_curve.append(val_mse)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(self.patience):
                        break

            if best_state is not None:
                self._model_.load_state_dict(best_state)

            return self
        finally:
            # 默认将权重移回 CPU，便于 joblib/pickle 保存与跨机器加载
            if self._model_ is not None:
                try:
                    self._model_.to("cpu")
                except Exception:
                    pass
            self._device_ = "cpu"
            _clear_cuda_memory()

    def _forward_with_physics(
        self,
        Xb: torch.Tensor,
        Xmb: torch.Tensor,
        rb: torch.Tensor,
        confb: torch.Tensor,
        nfw: torch.Tensor,
        isnf: torch.Tensor,
        Ef: torch.Tensor,
        rhof: torch.Tensor,
        nub: Optional[torch.Tensor] = None,
        mcb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """返回 (pred, phys_pen, log_nu_pred)。log_nu_pred 仅在 ν 中间层激活时非 None。"""
        assert self._model_ is not None
        mode = self._mode_ or "generic"

        if self._uses_missing_mask():
            out = self._model_(Xb, Xmb)
        else:
            out = self._model_(Xb)

        # ν 潜变量：log 空间残差（正定 + 物理底座 + 可外推），ν ∈ [50, 1e5]
        log_nu_pred: Optional[torch.Tensor] = None
        if self._nu_active_ and self._nu_ch_ >= 0 and out.shape[1] > self._nu_ch_ and nub is not None:
            log_nu_pred = torch.clamp(out[:, self._nu_ch_:self._nu_ch_ + 1] + nub, 3.9, 11.5)

        if mode == "tg":
            # ============================================================
            # 混合预测策略：结合直接神经网络输出和物理约束
            # 输出统一在标准化 y 空间参数化：out[:,0] 直接是标准化预测
            # ============================================================
            y_scale = getattr(self, '_y_scale_', 100.0)
            y_mean = getattr(self, '_y_mean_', 0.0)
            y_min = getattr(self, '_y_min_', -50.0)
            y_max = getattr(self, '_y_max_', 300.0)
            y_range = max(y_max - y_min, 100.0)
            
            # 分支1：直接神经网络预测（标准化输出 → 原始空间）
            direct_pred = out[:, 0:1] * y_scale + y_mean
            
            # 分支2：DiBenedetto 物理约束预测（辅助分支）
            tg0_raw = out[:, 1:2] * y_scale + y_mean
            tg_delta = F.softplus(out[:, 2:3]) * (y_scale * 1.5) + 10.0
            tginf = tg0_raw + tg_delta
            lam = torch.sigmoid(out[:, 3:4]) * 0.8 + 0.1
            
            # α: 固化度（独立通道，修复与 λ 共用通道的参数化 bug）
            has_r_info = (confb > 0.5).float()
            amax = alpha_max_from_r_torch(rb)
            amax = has_r_info * amax + (1.0 - has_r_info) * 0.999
            alpha_idx = min(4, out.shape[1] - 1)
            alpha = amax * torch.sigmoid(out[:, alpha_idx:alpha_idx+1])
            
            physics_pred = dibenedetto_tg_torch(tg0=tg0_raw, tginf=tginf, lam=lam, alpha=alpha)

            formula = str(getattr(self, "physics_formula", "standard")).strip().lower()
            if formula in {"advanced", "enhanced"}:
                # Fox 公式（以固化度 alpha 近似质量分数）作为额外物理先验
                tg0_safe = torch.clamp(tg0_raw, min=1e-6)
                tginf_safe = torch.clamp(tginf, min=tg0_safe + 1e-6)
                alpha_clip = torch.clamp(alpha, 0.0, 0.999)
                fox_pred = 1.0 / ((1.0 - alpha_clip) / tg0_safe + alpha_clip / tginf_safe)
                physics_pred = 0.5 * physics_pred + 0.5 * fox_pred

            # Fox–Loshaek 交联密度通道：Tg = Tg∞ − K/Mc（ν↑ → Tg↑，单调由参数化保证）
            if log_nu_pred is not None and mcb is not None:
                k_raw = out[:, self._nu_param_ch_:self._nu_param_ch_ + 1]
                k_const = F.softplus(k_raw) * y_scale * 2.0  # °C·kg/mol 量级
                mc_kg_mol = torch.exp(mcb) / 1000.0
                fl_pred = tginf - k_const / torch.clamp(mc_kg_mol, min=0.05)
                physics_pred = 0.5 * physics_pred + 0.5 * fl_pred
            
            # 混合预测：以直接预测为主，物理预测为辅
            # physics_weight 控制物理约束的影响程度（可配置至 60%）
            phys_mix = min(float(self.physics_weight), 0.6)
            tg_pred = (1.0 - phys_mix) * direct_pred + phys_mix * physics_pred
            
            # 软正则化：鼓励预测在合理范围内
            penalty_margin = y_range * 0.15
            p1 = F.relu(y_min - penalty_margin - direct_pred)
            p2 = F.relu(direct_pred - y_max - penalty_margin)
            phys_pen = (p1 * p1 + p2 * p2).mean() / (y_scale ** 2 + 1e-8) * 0.1

            if formula in {"advanced", "enhanced"}:
                # 额外约束：Tg 应在 Tg0 与 Tg∞ 合理范围内
                lower = torch.minimum(tg0_raw, tginf)
                upper = torch.maximum(tg0_raw, tginf)
                p_low = F.relu(lower - tg_pred)
                p_high = F.relu(tg_pred - upper)
                phys_pen = phys_pen + (p_low * p_low + p_high * p_high).mean() / (y_scale ** 2 + 1e-8) * 0.05

            return tg_pred, phys_pen, log_nu_pred

        if mode == "mechanics":
            Em = F.softplus(out[:, 0:1]) + 1e-6
            xi = F.softplus(out[:, 1:2]) + 0.1

            vf = volume_fraction_from_wt_fraction_torch(
                wt_frac=torch.clamp(nfw, 0.0, 0.95),
                rho_f=rhof,
                rho_m=1.20,
            )

            Ef_t = torch.clamp(Ef, min=1e-6)
            Ec = halpin_tsai_torch(Em, Ef_t, xi, vf)

            pred = torch.where(isnf > 0.5, Ec, Em)
            phys_pen = (0.0005 * (xi * xi).mean())

            formula = str(getattr(self, "physics_formula", "standard")).strip().lower()
            if formula in {"advanced", "enhanced"}:
                # Voigt/Reuss 界作为额外物理约束
                vf_c = torch.clamp(vf, 0.0, 0.95)
                Em_safe = torch.clamp(Em, min=1e-6)
                Ef_safe = torch.clamp(Ef_t, min=1e-6)
                e_voigt = Em_safe * (1.0 - vf_c) + Ef_safe * vf_c
                e_reuss = 1.0 / ((1.0 - vf_c) / Em_safe + vf_c / Ef_safe)
                lower = torch.minimum(e_voigt, e_reuss)
                upper = torch.maximum(e_voigt, e_reuss)
                p_low = F.relu(lower - pred)
                p_high = F.relu(pred - upper)
                phys_pen = phys_pen + (p_low * p_low + p_high * p_high).mean() * 0.05

            return pred, phys_pen, log_nu_pred

        # generic / thermal 模式：纯 MLP 回归（标准化输出 → 原始空间）
        y_scale = getattr(self, '_y_scale_', 1.0)
        y_mean = getattr(self, '_y_mean_', 0.0)

        pred = out[:, 0:1] * y_scale + y_mean
        phys_pen = torch.zeros((), device=Xb.device)

        # ν 物理读出：y_phys = ȳ + softplus(w)·y_scale·0.5·(log ν − log ν_ref)
        # softplus 斜率保证对 ν 单调递增（强度/热稳定性随交联密度上升）
        if log_nu_pred is not None:
            slope = F.softplus(out[:, self._nu_param_ch_:self._nu_param_ch_ + 1]) * y_scale * 0.5
            y_phys = y_mean + slope * (log_nu_pred - float(self._log_nu_ref_))
            pw = min(float(self.physics_weight), 0.6)
            pred = (1.0 - pw) * pred + pw * y_phys

        return pred, phys_pen, log_nu_pred

    def predict(self, X):
        if not TORCH_AVAILABLE:
            raise ImportError("EpoxyPINNRegressor 需要 torch")
        if self._model_ is None or self._prep_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            df_raw = X.copy()
        else:
            X_arr = np.asarray(X)
            df_raw = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])

        # ν 实测列若存在则剥离（与 fit 一致，防止泄入特征）
        nu_col = (self.nu_column or "").strip()
        if not nu_col:
            nu_col = "crosslink_density_mol_m3" if "crosslink_density_mol_m3" in df_raw.columns else ""
        if nu_col and nu_col in df_raw.columns:
            df_raw = df_raw.drop(columns=[nu_col])

        # 高分子物理指数（与拟合时一致；幂等）
        try:
            from .polymer_physics import augment_polymer_physics
            _pf = [s.strip() for s in (self.polymer_physics_features or "").split(",") if s.strip()]
            df_raw = augment_polymer_physics(
                df_raw, features=_pf or None, enabled=bool(self.use_polymer_physics)
            )
        except Exception:
            pass

        self._identify_special_columns(df_raw)

        X_scaled, aux, missing_mask = self._transform(df_raw, return_missing_mask=True)

        device = getattr(self, "_device_", "cpu") or "cpu"
        self._model_.to(device)

        X_tensor = torch.from_numpy(X_scaled).to(device)
        X_missing_tensor = torch.from_numpy(missing_mask).to(device)
        r_tensor = torch.from_numpy(aux["r"]).to(device).view(-1, 1)
        conf_tensor = torch.from_numpy(aux["r_conf_w"]).to(device).view(-1, 1)
        nf_w_tensor = torch.from_numpy(aux["nf_w"]).to(device).view(-1, 1)
        is_nf_tensor = torch.from_numpy(aux["is_nf"]).to(device).view(-1, 1)
        Ef_tensor = torch.from_numpy(aux["Ef"]).to(device).view(-1, 1)
        rho_f_tensor = torch.from_numpy(aux["rho_f"]).to(device).view(-1, 1)

        # ν 物理基线（log 空间）：冻结编码器 > 理论公式 > 拟合时参考值
        log_nu_th = aux["log_nu_theory"].astype(float)
        log_mc = aux["log_Mc"].astype(float)
        log_nu_enc = None
        encoder = getattr(self, "_nu_encoder_", None)
        if encoder is not None:
            try:
                log_nu_enc = np.asarray(encoder.predict_log_nu(df_raw), dtype=float)
                if len(log_nu_enc) != len(df_raw):
                    log_nu_enc = None
            except Exception:
                log_nu_enc = None
        if log_nu_enc is not None:
            log_nu_base = np.where(np.isfinite(log_nu_enc), log_nu_enc, log_nu_th)
        else:
            log_nu_base = log_nu_th
        nub_np = np.where(np.isfinite(log_nu_base), log_nu_base, self._log_nu_ref_).astype(np.float32)
        mcb_np = np.where(np.isfinite(log_mc), log_mc, self._log_mc_ref_).astype(np.float32)
        nub_tensor = torch.from_numpy(nub_np).to(device).view(-1, 1)
        mcb_tensor = torch.from_numpy(mcb_np).to(device).view(-1, 1)

        self._model_.eval()
        preds = []
        nu_out = []
        with torch.no_grad():
            bs = max(256, int(self.batch_size))
            for i in range(0, len(X_tensor), bs):
                xb = X_tensor[i:i+bs]
                xmb = X_missing_tensor[i:i+bs]
                rb = r_tensor[i:i+bs]
                cb = conf_tensor[i:i+bs]
                nfw = nf_w_tensor[i:i+bs]
                isnf = is_nf_tensor[i:i+bs]
                Ef = Ef_tensor[i:i+bs]
                rhof = rho_f_tensor[i:i+bs]
                nub = nub_tensor[i:i+bs]
                mcb = mcb_tensor[i:i+bs]
                pred, _, log_nu_pred = self._forward_with_physics(xb, xmb, rb, cb, nfw, isnf, Ef, rhof, nub, mcb)
                preds.append(pred.detach().cpu().numpy())
                if log_nu_pred is not None:
                    nu_out.append(torch.exp(log_nu_pred).detach().cpu().numpy())
        y_pred = np.vstack(preds).reshape(-1)
        if nu_out:
            self.nu_pred_ = np.vstack(nu_out).reshape(-1)  # 预测的交联密度（可解释中间量）
        return y_pred
