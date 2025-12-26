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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

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
    """Epoxy PINN 回归器（sklearn 风格）"""

    def __init__(
        self,
        mode: str = "auto",
        target_name: Optional[str] = None,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 200,
        patience: int = 25,
        physics_weight: float = 1.0,
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.mode = mode
        self.target_name = target_name
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.physics_weight = physics_weight
        self.grad_clip = grad_clip
        self.device = device
        self.seed = seed
        self.verbose = verbose

        # fitted attrs
        self._mode_: Optional[str] = None
        self._prep_: Optional[_PreprocessPack] = None
        self._model_: Optional[nn.Module] = None
        self._device_: str = "cpu"

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
        cols = set(df.columns)

        for c in ["r_value", "stoich_ratio", "stoichiometric_ratio", "Stoich_Ratio", "StoichRatio"]:
            if c in cols:
                self._col_r_ = c
                break

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

    def _build_numeric_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        drop_text_cols = set()
        if self._col_nf_type_ is not None:
            drop_text_cols.add(self._col_nf_type_)
        if self._col_r_conf_ is not None:
            drop_text_cols.add(self._col_r_conf_)

        for c in df.columns:
            if isinstance(c, str) and ("smiles" in c.lower() or "inchi" in c.lower()):
                drop_text_cols.add(c)

        if drop_text_cols:
            df = df.drop(columns=[c for c in drop_text_cols if c in df.columns], errors="ignore")

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

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def _fit_preprocess(self, df_num: pd.DataFrame) -> _PreprocessPack:
        median = df_num.median(numeric_only=True).to_numpy(dtype=float)
        df_imp = df_num.fillna(df_num.median(numeric_only=True))

        mean = df_imp.mean(numeric_only=True).to_numpy(dtype=float)
        std = df_imp.std(numeric_only=True).to_numpy(dtype=float)
        std = np.where(std < 1e-8, 1.0, std)

        return _PreprocessPack(feature_names=list(df_num.columns), median=median, mean=mean, std=std)

    def _transform(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._prep_ is None:
            raise RuntimeError("Model is not fitted yet.")

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

        df_num = self._build_numeric_features(df_raw)

        for c in self._prep_.feature_names:
            if c not in df_num.columns:
                df_num[c] = np.nan
        df_num = df_num[self._prep_.feature_names].copy()

        X = df_num.to_numpy(dtype=float)
        med = self._prep_.median
        if med.shape[0] == X.shape[1]:
            nan_mask = ~np.isfinite(X)
            if nan_mask.any():
                for j in range(X.shape[1]):
                    col_nan = nan_mask[:, j]
                    if col_nan.any():
                        X[col_nan, j] = med[j]
        else:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X = (X - self._prep_.mean) / self._prep_.std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32), aux

    def _make_model(self, input_dim: int) -> nn.Module:
        mode = self._mode_ or "generic"
        if mode == "tg":
            out_dim = 4
        elif mode == "mechanics":
            out_dim = 2
        else:
            out_dim = 1
        return _MLP(input_dim=input_dim, hidden_dim=int(self.hidden_dim), n_layers=int(self.n_layers),
                    dropout=float(self.dropout), out_dim=out_dim)

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("EpoxyPINNRegressor 需要 torch，请先安装 torch>=2.1.0")

        _set_seed(int(self.seed))

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
        y_arr = y_arr[valid]

        self._identify_special_columns(df_raw)

        df_num = self._build_numeric_features(df_raw)
        self._prep_ = self._fit_preprocess(df_num)

        X_scaled, aux = self._transform(df_raw)

        self._mode_ = _infer_mode(self.mode, self.target_name)
        device = self._select_device()

        X_tensor = torch.from_numpy(X_scaled).to(device)
        y_tensor = torch.from_numpy(y_arr).to(device).view(-1, 1)

        r_tensor = torch.from_numpy(aux["r"]).to(device).view(-1, 1)
        conf_tensor = torch.from_numpy(aux["r_conf_w"]).to(device).view(-1, 1)
        nf_w_tensor = torch.from_numpy(aux["nf_w"]).to(device).view(-1, 1)
        is_nf_tensor = torch.from_numpy(aux["is_nf"]).to(device).view(-1, 1)
        Ef_tensor = torch.from_numpy(aux["Ef"]).to(device).view(-1, 1)
        rho_f_tensor = torch.from_numpy(aux["rho_f"]).to(device).view(-1, 1)

        # split train/val
        n = X_tensor.shape[0]
        idx = torch.randperm(n, device=device)
        val_size = max(1, int(0.1 * n))
        val_idx = idx[:val_size]
        tr_idx = idx[val_size:]

        def _sub(t: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
            return t.index_select(0, inds)

        train_ds = TensorDataset(
            _sub(X_tensor, tr_idx),
            _sub(y_tensor, tr_idx),
            _sub(r_tensor, tr_idx),
            _sub(conf_tensor, tr_idx),
            _sub(nf_w_tensor, tr_idx),
            _sub(is_nf_tensor, tr_idx),
            _sub(Ef_tensor, tr_idx),
            _sub(rho_f_tensor, tr_idx),
        )
        val_ds = TensorDataset(
            _sub(X_tensor, val_idx),
            _sub(y_tensor, val_idx),
            _sub(r_tensor, val_idx),
            _sub(conf_tensor, val_idx),
            _sub(nf_w_tensor, val_idx),
            _sub(is_nf_tensor, val_idx),
            _sub(Ef_tensor, val_idx),
            _sub(rho_f_tensor, val_idx),
        )

        train_loader = DataLoader(train_ds, batch_size=int(self.batch_size), shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=int(self.batch_size), shuffle=False, drop_last=False)

        self._model_ = self._make_model(input_dim=X_tensor.shape[1]).to(device)
        optimizer = torch.optim.AdamW(self._model_.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
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

        for epoch in range(int(self.epochs)):
            self._model_.train()
            train_loss = 0.0
            n_batches = 0
            train_mse_sum = 0.0
            train_n = 0

            for batch in train_loader:
                xb, yb, rb, confb, nfw, isnf, Ef, rhof = batch
                optimizer.zero_grad(set_to_none=True)

                pred, phys_pen = self._forward_with_physics(xb, rb, confb, nfw, isnf, Ef, rhof)

                loss_data = huber(pred, yb)
                loss = loss_data + phys_w * phys_pen
                loss.backward()

                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model_.parameters(), max_norm=grad_clip)

                optimizer.step()

                # 记录 MSE 曲线
                mse_batch = torch.mean((pred - yb) ** 2)
                train_mse_sum += float(mse_batch.detach().cpu().item()) * int(len(yb))
                train_n += int(len(yb))

                train_loss += float(loss.detach().cpu().item())
                n_batches += 1

            train_loss /= max(1, n_batches)
            train_mse = train_mse_sum / max(1, train_n)

            self._model_.eval()
            val_loss = 0.0
            n_valb = 0
            val_mse_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb, rb, confb, nfw, isnf, Ef, rhof = batch
                    pred, phys_pen = self._forward_with_physics(xb, rb, confb, nfw, isnf, Ef, rhof)
                    loss_data = huber(pred, yb)
                    loss = loss_data + phys_w * phys_pen

                    mse_batch = torch.mean((pred - yb) ** 2)
                    val_mse_sum += float(mse_batch.detach().cpu().item()) * int(len(yb))
                    val_n += int(len(yb))

                    val_loss += float(loss.detach().cpu().item())
                    n_valb += 1

            val_loss /= max(1, n_valb)
            val_mse = val_mse_sum / max(1, val_n)
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

        # 默认将权重移回 CPU，便于 joblib/pickle 保存与跨机器加载
        self._model_.to("cpu")
        self._device_ = "cpu"

        return self

    def _forward_with_physics(
        self,
        Xb: torch.Tensor,
        rb: torch.Tensor,
        confb: torch.Tensor,
        nfw: torch.Tensor,
        isnf: torch.Tensor,
        Ef: torch.Tensor,
        rhof: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model_ is not None
        mode = self._mode_ or "generic"

        out = self._model_(Xb)

        if mode == "tg":
            tg0 = out[:, 0:1]
            tg_delta = F.softplus(out[:, 1:2]) + 1e-3
            tginf = tg0 + tg_delta
            lam = torch.sigmoid(out[:, 2:3])

            amax = alpha_max_from_r_torch(rb)
            amax = confb * amax + (1.0 - confb) * 1.0
            alpha = amax * torch.sigmoid(out[:, 3:4])

            tg_pred = dibenedetto_tg_torch(tg0=tg0, tginf=tginf, lam=lam, alpha=alpha)

            p1 = F.relu(tg0 - tg_pred)
            p2 = F.relu(tg_pred - tginf)
            phys_pen = (p1 * p1 + p2 * p2).mean()

            return tg_pred, phys_pen

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

            return pred, phys_pen

        pred = out[:, 0:1]
        phys_pen = torch.zeros((), device=Xb.device)
        return pred, phys_pen

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

        self._identify_special_columns(df_raw)

        X_scaled, aux = self._transform(df_raw)

        device = getattr(self, "_device_", "cpu") or "cpu"
        self._model_.to(device)

        X_tensor = torch.from_numpy(X_scaled).to(device)
        r_tensor = torch.from_numpy(aux["r"]).to(device).view(-1, 1)
        conf_tensor = torch.from_numpy(aux["r_conf_w"]).to(device).view(-1, 1)
        nf_w_tensor = torch.from_numpy(aux["nf_w"]).to(device).view(-1, 1)
        is_nf_tensor = torch.from_numpy(aux["is_nf"]).to(device).view(-1, 1)
        Ef_tensor = torch.from_numpy(aux["Ef"]).to(device).view(-1, 1)
        rho_f_tensor = torch.from_numpy(aux["rho_f"]).to(device).view(-1, 1)

        self._model_.eval()
        preds = []
        with torch.no_grad():
            bs = max(256, int(self.batch_size))
            for i in range(0, len(X_tensor), bs):
                xb = X_tensor[i:i+bs]
                rb = r_tensor[i:i+bs]
                cb = conf_tensor[i:i+bs]
                nfw = nf_w_tensor[i:i+bs]
                isnf = is_nf_tensor[i:i+bs]
                Ef = Ef_tensor[i:i+bs]
                rhof = rho_f_tensor[i:i+bs]
                pred, _ = self._forward_with_physics(xb, rb, cb, nfw, isnf, Ef, rhof)
                preds.append(pred.detach().cpu().numpy())
        y_pred = np.vstack(preds).reshape(-1)
        return y_pred
