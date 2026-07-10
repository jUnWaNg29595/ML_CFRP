# -*- coding: utf-8 -*-
"""FT-Transformer style regressor with a feature sub-attention network."""

from __future__ import annotations

import copy
import gc
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from .missing_value_handler import MissingValueHandler, build_missing_mask

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    FT_TRANSFORMER_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    FT_TRANSFORMER_AVAILABLE = False

try:
    from .task_manager import is_cancelled as _task_manager_is_cancelled
except Exception:
    def _task_manager_is_cancelled() -> bool:
        return False


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _parse_hidden_layers(value: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value if int(v) > 0]
    parts = [part.strip() for part in str(value).split(",")]
    return [int(part) for part in parts if part]


def _make_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    factory = mapping[name]
    return factory() if callable(factory) else factory()


def _move_module_to_cpu(module: nn.Module | None) -> nn.Module | None:
    if module is None:
        return None
    module.to(torch.device("cpu"))
    return module


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


class NumericalFeatureTokenizer(nn.Module):
    """Per-feature learnable affine projection into token space."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FeatureSubAttention(nn.Module):
    """Small attention sub-network that scores feature importance."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float,
        gate_type: str,
        temperature: float,
        use_missingness: bool = False,
    ):
        super().__init__()
        hidden_dim = max(4, int(hidden_dim))
        self.gate_type = gate_type
        self.temperature = max(float(temperature), 1e-3)
        self.use_missingness = bool(use_missingness)
        input_dim = int(n_features) * (2 if self.use_missingness else 1)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_missingness:
            if missing_mask is None:
                missing_mask = torch.zeros_like(x)
            x = torch.cat([x, missing_mask.to(dtype=x.dtype)], dim=-1)
        logits = self.net(x) / self.temperature
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        if self.gate_type == "softmax":
            return torch.softmax(logits, dim=-1)
        raise ValueError(f"Unsupported gate_type: {self.gate_type}")


class AttentionPool(nn.Module):
    """Learnable attention pooling for token sequences."""

    def __init__(self, d_token: int):
        super().__init__()
        self.score = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_hidden_dim: int,
        ffn_dropout: float,
        activation: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(d_token, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden_dim),
            _make_activation(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_dim, d_token),
        )
        self.dropout2 = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout1(attn_out)
        ffn_in = self.norm2(x)
        x = x + self.dropout2(self.ffn(ffn_in))
        return x


class FTTransformerBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_hidden_dim: int,
        ffn_dropout: float,
        token_dropout: float,
        activation: str,
        sub_attention_dim: int,
        sub_attention_dropout: float,
        sub_attention_temperature: float,
        feature_gate_type: str,
        feature_gate_scale: float,
        use_feature_residual: bool,
        pooling: str,
        head_hidden_dim: int,
        layer_norm_eps: float,
        use_missingness_embedding: bool = False,
        use_missingness_attention: bool = False,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.pooling = pooling
        self.feature_gate_scale = float(feature_gate_scale)
        self.use_feature_residual = bool(use_feature_residual)
        self.use_missingness_embedding = bool(use_missingness_embedding)
        self.use_missingness_attention = bool(use_missingness_attention)

        self.tokenizer = NumericalFeatureTokenizer(self.n_features, d_token)
        self.sub_attention = FeatureSubAttention(
            n_features=self.n_features,
            hidden_dim=sub_attention_dim,
            dropout=sub_attention_dropout,
            gate_type=feature_gate_type,
            temperature=sub_attention_temperature,
            use_missingness=self.use_missingness_attention,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_features + 1, d_token))
        if self.use_missingness_embedding:
            self.missing_embedding = nn.Parameter(torch.zeros(1, self.n_features, d_token))
        else:
            self.register_parameter("missing_embedding", None)
        self.token_dropout = nn.Dropout(token_dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_dropout=ffn_dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.final_norm = nn.LayerNorm(d_token, eps=layer_norm_eps)
        self.attention_pool = AttentionPool(d_token)

        if int(head_hidden_dim) > 0:
            self.head = nn.Sequential(
                nn.Linear(d_token, int(head_hidden_dim)),
                _make_activation(activation),
                nn.Dropout(ffn_dropout),
                nn.Linear(int(head_hidden_dim), 1),
            )
        else:
            self.head = nn.Linear(d_token, 1)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        if self.missing_embedding is not None:
            nn.init.normal_(self.missing_embedding, std=0.02)

    def encode(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None, return_gates: bool = False):
        if missing_mask is None:
            missing_mask = torch.zeros_like(x)
        missing_mask = missing_mask.to(dtype=x.dtype)
        gates = self.sub_attention(
            x,
            missing_mask=missing_mask if self.use_missingness_attention else None,
        )
        tokens = self.tokenizer(x)
        if self.missing_embedding is not None:
            tokens = tokens + missing_mask.unsqueeze(-1) * self.missing_embedding
        gate_scale = gates.unsqueeze(-1)
        if self.use_feature_residual:
            tokens = tokens * (1.0 + self.feature_gate_scale * gate_scale)
        else:
            tokens = tokens * (self.feature_gate_scale * gate_scale)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        tokens = self.token_dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)
        if return_gates:
            return tokens, gates
        return tokens

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return tokens[:, 0, :]
        feature_tokens = tokens[:, 1:, :]
        if self.pooling == "mean":
            return feature_tokens.mean(dim=1)
        if self.pooling == "attention":
            return self.attention_pool(feature_tokens)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.encode(x, missing_mask=missing_mask)
        pooled = self.pool_tokens(tokens)
        return self.head(pooled).squeeze(-1)


@dataclass
class _TrainingBatch:
    features: torch.Tensor
    target: torch.Tensor


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible FT-Transformer with feature sub-attention."""

    def __init__(
        self,
        d_token: int = 128,
        n_blocks: int = 4,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        ffn_d_hidden: int = 256,
        ffn_dropout: float = 0.1,
        token_dropout: float = 0.05,
        activation: str = "gelu",
        sub_attention_dim: int = 64,
        sub_attention_dropout: float = 0.1,
        sub_attention_temperature: float = 1.0,
        feature_gate_type: str = "softmax",
        feature_gate_scale: float = 1.0,
        use_feature_residual: bool = True,
        pooling: str = "cls",
        head_hidden_dim: int = 128,
        layer_norm_eps: float = 1e-5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 150,
        patience: int = 20,
        validation_split: float = 0.1,
        loss_name: str = "mse",
        gradient_clip_norm: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
        external_preprocess: bool = False,
        missing_value_strategy: str = "bayesian",
        missing_imputer_max_iter: int = 15,
        missing_n_imputations: int = 5,
        use_missingness_embedding: bool = True,
        use_missingness_attention: bool = True,
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.token_dropout = token_dropout
        self.activation = activation
        self.sub_attention_dim = sub_attention_dim
        self.sub_attention_dropout = sub_attention_dropout
        self.sub_attention_temperature = sub_attention_temperature
        self.feature_gate_type = feature_gate_type
        self.feature_gate_scale = feature_gate_scale
        self.use_feature_residual = use_feature_residual
        self.pooling = pooling
        self.head_hidden_dim = head_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.loss_name = loss_name
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.external_preprocess = external_preprocess
        self.missing_value_strategy = missing_value_strategy
        self.missing_imputer_max_iter = missing_imputer_max_iter
        self.missing_n_imputations = missing_n_imputations
        self.use_missingness_embedding = use_missingness_embedding
        self.use_missingness_attention = use_missingness_attention

        self.model_ = None
        self.imputer_ = MissingValueHandler(
            strategy=self.missing_value_strategy,
            random_state=self.seed,
            max_iter=self.missing_imputer_max_iter,
            n_imputations=self.missing_n_imputations,
        )
        self.scaler_ = StandardScaler()
        self.validation_data = None
        self.train_loss_history = []
        self.test_loss_history = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self._device_used_ = "cpu"

    @staticmethod
    def is_available() -> bool:
        return FT_TRANSFORMER_AVAILABLE

    def _select_device(self) -> torch.device:
        requested = str(self.device or "auto").lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_features(self, X, fit: bool = False) -> np.ndarray:
        X_arr, _ = self._prepare_features_and_mask(X, fit=fit)
        return X_arr

    def _prepare_features_and_mask(self, X, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        missing_mask = build_missing_mask(X)
        X = np.where(np.isfinite(X), X, np.nan)
        if self.external_preprocess:
            X_out = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return np.asarray(X_out, dtype=np.float32), missing_mask
        if fit:
            X = self.imputer_.fit_transform(X)
            X = self.scaler_.fit_transform(X)
        else:
            X = self.imputer_.transform(X)
            X = self.scaler_.transform(X)
        return np.asarray(X, dtype=np.float32), missing_mask

    def _prepare_target(self, y) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        return np.asarray(y, dtype=np.float32).ravel()

    def _build_model(self, n_features: int) -> FTTransformerBackbone:
        if int(self.d_token) % int(self.attention_n_heads) != 0:
            raise ValueError("d_token must be divisible by attention_n_heads")
        return FTTransformerBackbone(
            n_features=n_features,
            d_token=int(self.d_token),
            n_blocks=int(self.n_blocks),
            n_heads=int(self.attention_n_heads),
            attention_dropout=float(self.attention_dropout),
            residual_dropout=float(self.residual_dropout),
            ffn_hidden_dim=int(self.ffn_d_hidden),
            ffn_dropout=float(self.ffn_dropout),
            token_dropout=float(self.token_dropout),
            activation=str(self.activation),
            sub_attention_dim=int(self.sub_attention_dim),
            sub_attention_dropout=float(self.sub_attention_dropout),
            sub_attention_temperature=float(self.sub_attention_temperature),
            feature_gate_type=str(self.feature_gate_type),
            feature_gate_scale=float(self.feature_gate_scale),
            use_feature_residual=bool(self.use_feature_residual),
            pooling=str(self.pooling),
            head_hidden_dim=int(self.head_hidden_dim),
            layer_norm_eps=float(self.layer_norm_eps),
            use_missingness_embedding=bool(self.use_missingness_embedding),
            use_missingness_attention=bool(self.use_missingness_attention),
        )

    def _build_loss(self):
        loss_name = str(self.loss_name).lower()
        if loss_name == "mse":
            return nn.MSELoss()
        if loss_name == "mae":
            return nn.L1Loss()
        if loss_name == "huber":
            return nn.SmoothL1Loss(beta=1.0)
        raise ValueError(f"Unsupported loss_name: {self.loss_name}")

    def _split_validation(
        self, X: np.ndarray, missing_mask: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            X_val, mask_val = self._prepare_features_and_mask(X_val, fit=False)
            y_val = self._prepare_target(y_val)
            return X, missing_mask, y, X_val, mask_val, y_val

        val_ratio = float(self.validation_split)
        if val_ratio <= 0.0 or len(X) < 20:
            return X, missing_mask, y, None, None, None

        rng = np.random.RandomState(int(self.seed))
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split_idx = max(1, int(round(len(X) * (1.0 - val_ratio))))
        split_idx = min(split_idx, len(X) - 1)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return (
            X[train_idx],
            missing_mask[train_idx],
            y[train_idx],
            X[val_idx],
            missing_mask[val_idx],
            y[val_idx],
        )

    def _make_loader(self, X: np.ndarray, missing_mask: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(missing_mask).float(),
            torch.from_numpy(y).float().view(-1, 1),
        )
        return DataLoader(
            dataset,
            batch_size=max(8, int(self.batch_size)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self._device_used_ == "cuda"),
        )

    def fit(self, X, y):
        if not FT_TRANSFORMER_AVAILABLE:
            raise ImportError("PyTorch is required for FT-Transformer")

        _set_global_seed(int(self.seed))
        X_train, mask_train = self._prepare_features_and_mask(X, fit=True)
        y_train = self._prepare_target(y)
        self.n_features_in_ = X_train.shape[1]

        X_fit, mask_fit, y_fit, X_val, mask_val, y_val = self._split_validation(X_train, mask_train, y_train)
        self._device_used_ = str(self._select_device())
        device = torch.device(self._device_used_)

        self.model_ = self._build_model(self.n_features_in_).to(device)
        try:
            optimizer = torch.optim.AdamW(
                self.model_.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.6,
                patience=max(3, int(self.patience) // 3),
                min_lr=1e-6,
            )
            loss_fn = self._build_loss()

            train_loader = self._make_loader(X_fit, mask_fit, y_fit, shuffle=True)
            val_loader = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loader = self._make_loader(X_val, mask_val, y_val, shuffle=False)

            self.train_loss_history = []
            self.test_loss_history = []
            best_state = None
            best_score = math.inf
            stale_epochs = 0

            use_amp = device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for epoch in range(int(self.epochs)):
                _raise_if_cancelled()
                self.model_.train()
                running_loss = 0.0
                count = 0
                for batch_x, batch_mask, batch_y in train_loader:
                    _raise_if_cancelled()
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_mask = batch_mask.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True).squeeze(-1)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = self.model_(batch_x, batch_mask)
                        loss = loss_fn(pred, batch_y)
                    scaler.scale(loss).backward()
                    if float(self.gradient_clip_norm) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model_.parameters(),
                            max_norm=float(self.gradient_clip_norm),
                        )
                    scaler.step(optimizer)
                    scaler.update()

                    batch_size = batch_x.size(0)
                    running_loss += float(loss.detach().cpu().item()) * batch_size
                    count += batch_size

                train_loss = running_loss / max(1, count)
                self.train_loss_history.append(train_loss)

                val_loss = train_loss
                if val_loader is not None:
                    self.model_.eval()
                    val_running = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for batch_x, batch_mask, batch_y in val_loader:
                            _raise_if_cancelled()
                            batch_x = batch_x.to(device, non_blocking=True)
                            batch_mask = batch_mask.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True).squeeze(-1)
                            pred = self.model_(batch_x, batch_mask)
                            loss = loss_fn(pred, batch_y)
                            batch_size = batch_x.size(0)
                            val_running += float(loss.detach().cpu().item()) * batch_size
                            val_count += batch_size
                    val_loss = val_running / max(1, val_count)
                    self.test_loss_history.append(val_loss)

                scheduler.step(val_loss)

                metric_to_track = val_loss if val_loader is not None else train_loss
                if metric_to_track < best_score - 1e-8:
                    best_score = metric_to_track
                    best_state = copy.deepcopy(self.model_.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if self.verbose:
                    msg = (
                        f"[FT-Transformer] epoch={epoch + 1}/{self.epochs} "
                        f"train_loss={train_loss:.6f}"
                    )
                    if val_loader is not None:
                        msg += f" val_loss={val_loss:.6f}"
                    print(msg)

                if stale_epochs >= int(self.patience):
                    break

            if best_state is not None:
                self.model_.load_state_dict(best_state)

            _raise_if_cancelled()
            self.model_.eval()
            with torch.no_grad():
                probe = torch.from_numpy(X_fit[: min(len(X_fit), 2048)]).float().to(device)
                probe_mask = torch.from_numpy(mask_fit[: min(len(mask_fit), 2048)]).float().to(device)
                _, gates = self.model_.encode(probe, missing_mask=probe_mask, return_gates=True)
                self.feature_importances_ = gates.mean(dim=0).detach().cpu().numpy()
            return self
        finally:
            _move_module_to_cpu(self.model_)
            self._device_used_ = "cpu"
            _clear_cuda_memory()

    def _predict_numpy(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model is not fitted yet")
        X, missing_mask = self._prepare_features_and_mask(X, fit=False)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(
                torch.from_numpy(X).float(),
                torch.from_numpy(missing_mask).float(),
            ).detach().cpu().numpy().ravel()
        return pred.astype(np.float64, copy=False)

    def predict(self, X):
        return self._predict_numpy(X)

    def get_training_history(self) -> dict:
        history = {"loss": list(self.train_loss_history)}
        if self.test_loss_history:
            history["val_loss"] = list(self.test_loss_history)
            history["test_mse"] = list(self.test_loss_history)
        if self.train_loss_history:
            history["train_mse"] = list(self.train_loss_history)
        return history

    def score(self, X, y):
        pred = self.predict(X)
        y_true = self._prepare_target(y).astype(np.float64, copy=False)
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom <= 0:
            return 0.0
        return 1.0 - float(np.sum((y_true - pred) ** 2) / denom)
