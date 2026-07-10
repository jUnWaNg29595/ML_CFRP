# -*- coding: utf-8 -*-
"""FT-Transformer + Bayesian regression head for tabular uncertainty modeling."""

from __future__ import annotations

import copy
import gc
import math
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from .missing_value_handler import MissingValueHandler, build_missing_mask

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    from .fttransformer_model import FTTransformerBackbone

    FT_BACKBONE_AVAILABLE = True
except Exception:
    FTTransformerBackbone = None
    FT_BACKBONE_AVAILABLE = False

try:
    from .task_manager import is_cancelled as _task_manager_is_cancelled
except Exception:
    def _task_manager_is_cancelled() -> bool:
        return False


TRANSFORMER_BNN_AVAILABLE = bool(TORCH_AVAILABLE and FT_BACKBONE_AVAILABLE)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    factory = mapping[name]
    return factory() if callable(factory) else factory()


def _enable_dropout_only(module: nn.Module) -> None:
    module.eval()
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()


def _is_cuda_oom_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        for device_idx in range(torch.cuda.device_count()):
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass


def _raise_if_cancelled() -> None:
    if _task_manager_is_cancelled():
        raise RuntimeError("用户取消")


class ProbabilisticFTTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_d_hidden: int,
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
        use_missingness_embedding: bool,
        use_missingness_attention: bool,
    ):
        super().__init__()
        self.backbone = FTTransformerBackbone(
            n_features=int(input_dim),
            d_token=int(d_token),
            n_blocks=int(n_blocks),
            n_heads=int(attention_n_heads),
            attention_dropout=float(attention_dropout),
            residual_dropout=float(residual_dropout),
            ffn_hidden_dim=int(ffn_d_hidden),
            ffn_dropout=float(ffn_dropout),
            token_dropout=float(token_dropout),
            activation=str(activation),
            sub_attention_dim=int(sub_attention_dim),
            sub_attention_dropout=float(sub_attention_dropout),
            sub_attention_temperature=float(sub_attention_temperature),
            feature_gate_type=str(feature_gate_type),
            feature_gate_scale=float(feature_gate_scale),
            use_feature_residual=bool(use_feature_residual),
            pooling=str(pooling),
            head_hidden_dim=0,
            layer_norm_eps=float(layer_norm_eps),
            use_missingness_embedding=bool(use_missingness_embedding),
            use_missingness_attention=bool(use_missingness_attention),
        )
        if int(head_hidden_dim) > 0:
            self.shared_head = nn.Sequential(
                nn.Linear(int(d_token), int(head_hidden_dim)),
                _make_activation(activation),
                nn.Dropout(float(ffn_dropout)),
            )
            final_dim = int(head_hidden_dim)
        else:
            self.shared_head = nn.Identity()
            final_dim = int(d_token)
        self.mean_head = nn.Linear(final_dim, 1)
        self.logvar_head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.encode(x, missing_mask=missing_mask)
        pooled = self.backbone.pool_tokens(tokens)
        shared = self.shared_head(pooled)
        mean = self.mean_head(shared).squeeze(-1)
        logvar = self.logvar_head(shared).squeeze(-1)
        return mean, logvar


class TransformerBNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible FT-Transformer with MC dropout uncertainty."""

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
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        batch_size: int = 256,
        epochs: int = 200,
        patience: int = 30,
        validation_split: float = 0.1,
        mc_samples: int = 50,
        loss_name: str = "gaussian_nll",
        min_logvar: float = -6.0,
        max_logvar: float = 3.0,
        gradient_clip_norm: float = 1.0,
        scheduler_factor: float = 0.6,
        scheduler_patience: int = 8,
        min_learning_rate: float = 1e-6,
        external_preprocess: bool = False,
        device: str = "auto",
        use_data_parallel: bool = False,
        gpu_ids: str = "0,1",
        random_state: int = 42,
        verbose: bool = True,
        missing_value_strategy: str = "bayesian",
        missing_imputer_max_iter: int = 15,
        missing_n_imputations: int = 5,
        use_missingness_embedding: bool = True,
        use_missingness_attention: bool = True,
        epoch_callback=None,
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
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.mc_samples = mc_samples
        self.loss_name = loss_name
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_learning_rate = min_learning_rate
        self.external_preprocess = external_preprocess
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = gpu_ids
        self.random_state = random_state
        self.verbose = verbose
        self.missing_value_strategy = missing_value_strategy
        self.missing_imputer_max_iter = missing_imputer_max_iter
        self.missing_n_imputations = missing_n_imputations
        self.use_missingness_embedding = use_missingness_embedding
        self.use_missingness_attention = use_missingness_attention
        self.epoch_callback = epoch_callback

        self.model_ = None
        self.imputer_ = MissingValueHandler(
            strategy=self.missing_value_strategy,
            random_state=self.random_state,
            max_iter=self.missing_imputer_max_iter,
            n_imputations=self.missing_n_imputations,
        )
        self.scaler_ = StandardScaler()
        self.validation_data = None
        self.train_loss_history = []
        self.test_loss_history = []
        self.epoch_time_history = []
        self.elapsed_time_history = []
        self.eta_history = []
        self.estimated_total_time_history = []
        self.learning_rate_history = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self._device_used_ = "cpu"
        self._training_device_used_ = "cpu"
        self._pin_memory_enabled_ = False
        self._parallel_device_ids_ = []
        self.epochs_completed_ = 0
        self.effective_batch_size_ = max(1, int(batch_size))
        self._effective_missing_value_strategy_ = str(self.missing_value_strategy or "median").lower()
        self._missing_strategy_note_ = ""
        self._has_missing_values_ = True
        self._mask_zero_mean_ = None
        self._mask_zero_scale_ = None

    @staticmethod
    def is_available() -> bool:
        return TRANSFORMER_BNN_AVAILABLE

    def _select_device(self) -> torch.device:
        device, _ = self._resolve_training_devices()
        return device

    def _parse_gpu_ids(self) -> list[int]:
        raw = str(self.gpu_ids or "").strip()
        if not raw:
            return []
        gpu_ids: list[int] = []
        for part in raw.split(","):
            item = part.strip()
            if not item:
                continue
            try:
                gpu_idx = int(item)
            except ValueError:
                continue
            if gpu_idx >= 0 and gpu_idx not in gpu_ids:
                gpu_ids.append(gpu_idx)
        return gpu_ids

    def _get_cuda_free_bytes(self, device_idx: int) -> int:
        if torch is None or not torch.cuda.is_available():
            return 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_idx)
            return int(free_bytes)
        except TypeError:
            try:
                with torch.cuda.device(device_idx):
                    free_bytes, _ = torch.cuda.mem_get_info()
                return int(free_bytes)
            except Exception:
                pass
        except Exception:
            pass
        try:
            props = torch.cuda.get_device_properties(device_idx)
            allocated = max(
                int(torch.cuda.memory_allocated(device_idx)),
                int(torch.cuda.memory_reserved(device_idx)),
            )
            return int(max(0, int(props.total_memory) - allocated))
        except Exception:
            return 0

    def _filter_low_memory_devices(self, device_ids: list[int], min_free_gb: float = 2.0) -> list[int]:
        threshold_bytes = int(float(min_free_gb) * (1024 ** 3))
        valid_ids = [int(idx) for idx in device_ids if int(idx) >= 0]
        if not valid_ids:
            return []
        eligible_ids = [idx for idx in valid_ids if self._get_cuda_free_bytes(idx) >= threshold_bytes]
        return eligible_ids or valid_ids

    def _estimate_effective_batch_size(self, n_features: int) -> int:
        requested = max(4, int(self.batch_size))
        n_features = max(1, int(n_features))
        if n_features >= 8192:
            cap = 4
        elif n_features >= 4096:
            cap = 8
        elif n_features >= 2048:
            cap = 16
        elif n_features >= 1024:
            cap = 32
        elif n_features >= 512:
            cap = 64
        else:
            cap = requested
        return max(4, min(requested, cap))

    def _build_batch_schedule(self, start_batch_size: int) -> list[int]:
        schedule: list[int] = []
        current = max(4, int(start_batch_size))
        while True:
            if current not in schedule:
                schedule.append(current)
            if current <= 4:
                break
            next_batch = max(4, current // 2)
            if next_batch == current:
                break
            current = next_batch
        return schedule

    def _cleanup_failed_attempt(self) -> None:
        try:
            if self.model_ is not None:
                try:
                    self.model_.to(torch.device("cpu"))
                except Exception:
                    pass
        finally:
            self.model_ = None
            self._device_used_ = "cpu"
            self._training_device_used_ = "cpu"
            self._parallel_device_ids_ = []
            self._pin_memory_enabled_ = False
            _clear_cuda_memory()

    def _resolve_training_devices(self, allow_data_parallel: bool | None = None) -> tuple[torch.device, list[int]]:
        requested = str(self.device or "auto").lower()
        if requested == "cpu":
            return torch.device("cpu"), []
        if not torch.cuda.is_available():
            return torch.device("cpu"), []

        allow_dp = bool(self.use_data_parallel if allow_data_parallel is None else allow_data_parallel)
        if requested in {"auto", "cuda"}:
            primary_idx = None
        elif requested.startswith("cuda:"):
            try:
                primary_idx = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {self.device}") from exc
        else:
            primary_idx = 0

        n_cuda = int(torch.cuda.device_count())
        requested_ids = [idx for idx in self._parse_gpu_ids() if idx < n_cuda]
        if primary_idx is None:
            candidate_ids = requested_ids or list(range(n_cuda))
            candidate_ids = self._filter_low_memory_devices(candidate_ids)
            primary_idx = max(candidate_ids, key=self._get_cuda_free_bytes)
        if primary_idx >= n_cuda:
            raise ValueError(f"Requested GPU cuda:{primary_idx} is not available; detected {n_cuda} GPU(s)")

        parallel_ids: list[int] = []
        if allow_dp and n_cuda > 1:
            candidate_parallel_ids = requested_ids or list(range(n_cuda))
            candidate_parallel_ids = [idx for idx in candidate_parallel_ids if idx < n_cuda]
            remaining_ids = [idx for idx in candidate_parallel_ids if idx != primary_idx]
            remaining_ids = self._filter_low_memory_devices(remaining_ids, min_free_gb=1.5)
            remaining_ids = sorted(remaining_ids, key=self._get_cuda_free_bytes, reverse=True)
            parallel_ids = [primary_idx] + remaining_ids
            if len(parallel_ids) < 2:
                parallel_ids = []

        return torch.device(f"cuda:{primary_idx}"), parallel_ids

    def _unwrap_model(self) -> nn.Module | None:
        if self.model_ is None:
            return None
        if isinstance(self.model_, nn.DataParallel):
            return self.model_.module
        return self.model_

    def _emit_status(self, phase: str, message: str, progress_ratio: float | None = None, **extra) -> None:
        if not callable(self.epoch_callback):
            return
        payload = {
            "phase": str(phase or "").strip().lower() or "status",
            "message": str(message or ""),
        }
        if progress_ratio is not None:
            try:
                payload["progress_ratio"] = float(progress_ratio)
            except Exception:
                pass
        payload.update(extra)
        try:
            self.epoch_callback(payload)
        except Exception:
            pass

    def _resolve_effective_missing_strategy(self, X: np.ndarray, missing_mask: np.ndarray) -> tuple[str, str]:
        requested = str(self.missing_value_strategy or "median").strip().lower()
        n_samples = int(X.shape[0]) if X.ndim >= 1 else 0
        n_features = int(X.shape[1]) if X.ndim >= 2 else 0
        total_cells = max(1, n_samples * max(1, n_features))
        missing_fraction = float(np.mean(missing_mask)) if missing_mask.size else 0.0

        if missing_fraction <= 0.0:
            return "none", "No missing values detected; skipping imputation."

        if requested == "mask_zero":
            return "mask_zero", "Using missing-value mask with zero placeholder; statistical imputation is skipped."

        if requested == "median":
            return "median", "Using median imputation."

        if n_features >= 256:
            return (
                "median",
                f"Requested {requested} imputation is too expensive for {n_features} features; using median instead.",
            )

        if requested == "multiple_bayesian" and total_cells >= 120000:
            return (
                "median",
                f"Requested {requested} imputation is too expensive for {n_samples}x{n_features} tabular input; using median instead.",
            )

        if requested == "bayesian" and total_cells >= 220000:
            return (
                "median",
                f"Requested {requested} imputation is too expensive for {n_samples}x{n_features} tabular input; using median instead.",
            )

        return requested, f"Using {requested} imputation."

    def _fit_mask_zero_scaler(self, X: np.ndarray) -> np.ndarray:
        mean = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, 1.0)
        self._mask_zero_mean_ = np.asarray(mean, dtype=np.float32)
        self._mask_zero_scale_ = np.asarray(scale, dtype=np.float32)
        X_scaled = (X - self._mask_zero_mean_) / self._mask_zero_scale_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(X_scaled, dtype=np.float32)

    def _transform_mask_zero_scaled(self, X: np.ndarray) -> np.ndarray:
        if self._mask_zero_mean_ is None or self._mask_zero_scale_ is None:
            raise ValueError("Mask-zero scaler statistics are not fitted")
        X_scaled = (X - self._mask_zero_mean_) / self._mask_zero_scale_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(X_scaled, dtype=np.float32)

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
            effective_strategy, note = self._resolve_effective_missing_strategy(X, missing_mask)
            self._effective_missing_value_strategy_ = effective_strategy
            self._missing_strategy_note_ = note
            self._has_missing_values_ = effective_strategy != "none"
            self._emit_status(
                "preprocessing",
                "preparing tabular features",
                progress_ratio=0.03,
                n_samples=int(X.shape[0]),
                n_features=int(X.shape[1]) if X.ndim >= 2 else 0,
                missing_fraction=float(np.mean(missing_mask)) if missing_mask.size else 0.0,
                missing_strategy=requested if (requested := str(self.missing_value_strategy or "median").strip().lower()) else "median",
                effective_missing_strategy=effective_strategy,
                note=note,
            )
            if self.verbose:
                print(
                    "[Transformer+BNN] preprocessing "
                    f"| samples={int(X.shape[0])} features={int(X.shape[1]) if X.ndim >= 2 else 0} "
                    f"| missing_fraction={float(np.mean(missing_mask)) if missing_mask.size else 0.0:.4f} "
                    f"| strategy={requested} -> {effective_strategy}"
                )

            if effective_strategy != "none":
                if effective_strategy == "mask_zero":
                    self._emit_status(
                        "preprocessing",
                        "applying missing-value mask without statistical imputation",
                        progress_ratio=0.06,
                        effective_missing_strategy=effective_strategy,
                        note=note,
                    )
                else:
                    if effective_strategy != str(self.imputer_.strategy or "").strip().lower():
                        self.imputer_ = MissingValueHandler(
                            strategy=effective_strategy,
                            random_state=self.random_state,
                            max_iter=self.missing_imputer_max_iter,
                            n_imputations=self.missing_n_imputations,
                        )
                    self._emit_status(
                        "preprocessing",
                        f"imputing missing values ({effective_strategy})",
                        progress_ratio=0.06,
                        effective_missing_strategy=effective_strategy,
                        note=note,
                    )
                    X = self.imputer_.fit_transform(X)
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            self._emit_status(
                "preprocessing",
                "scaling input features",
                progress_ratio=0.10,
                effective_missing_strategy=effective_strategy,
            )
            if effective_strategy == "mask_zero":
                X = self._fit_mask_zero_scaler(X)
            else:
                X = self.scaler_.fit_transform(X)
            self._emit_status(
                "preprocessing",
                "feature preprocessing completed",
                progress_ratio=0.14,
                effective_missing_strategy=effective_strategy,
                note=note,
            )
            return np.asarray(X, dtype=np.float32), missing_mask

        if not bool(self._has_missing_values_):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = self.scaler_.transform(X)
            return np.asarray(X, dtype=np.float32), missing_mask

        if str(self._effective_missing_value_strategy_ or "").strip().lower() == "mask_zero":
            X = self._transform_mask_zero_scaled(X)
            return np.asarray(X, dtype=np.float32), missing_mask

        X = self.imputer_.transform(X)
        X = self.scaler_.transform(X)
        return np.asarray(X, dtype=np.float32), missing_mask

    def _prepare_target(self, y) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        return np.asarray(y, dtype=np.float32).ravel()

    def _build_optimizer(self):
        name = str(self.optimizer).lower()
        if name == "adam":
            return torch.optim.Adam
        if name == "adamw":
            return torch.optim.AdamW
        if name == "sgd":
            return lambda params, lr, weight_decay: torch.optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _build_model(self, input_dim: int) -> ProbabilisticFTTransformer:
        if int(self.d_token) % int(self.attention_n_heads) != 0:
            raise ValueError("d_token must be divisible by attention_n_heads")
        return ProbabilisticFTTransformer(
            input_dim=int(input_dim),
            d_token=int(self.d_token),
            n_blocks=int(self.n_blocks),
            attention_n_heads=int(self.attention_n_heads),
            attention_dropout=float(self.attention_dropout),
            residual_dropout=float(self.residual_dropout),
            ffn_d_hidden=int(self.ffn_d_hidden),
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

        rng = np.random.RandomState(int(self.random_state))
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

    def _loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=float(self.min_logvar), max=float(self.max_logvar))
        if str(self.loss_name).lower() == "mse":
            return torch.mean((mean - target) ** 2)
        var = torch.exp(logvar)
        return 0.5 * torch.mean(logvar + ((target - mean) ** 2) / var)

    def _make_loader(
        self,
        X: np.ndarray,
        missing_mask: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(missing_mask).float(),
            torch.from_numpy(y).float().view(-1, 1),
        )
        return DataLoader(
            dataset,
            batch_size=max(1, int(self.batch_size if batch_size is None else batch_size)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=bool(self._pin_memory_enabled_),
        )

    def _fit_single_attempt(
        self,
        X_fit: np.ndarray,
        mask_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray | None,
        mask_val: np.ndarray | None,
        y_val: np.ndarray | None,
        epoch_callback,
        effective_batch_size: int,
        allow_data_parallel: bool,
    ) -> None:
        device, parallel_ids = self._resolve_training_devices(allow_data_parallel=allow_data_parallel)
        self._parallel_device_ids_ = list(parallel_ids)
        self._pin_memory_enabled_ = device.type == "cuda"
        self.effective_batch_size_ = max(1, int(effective_batch_size))
        self._training_device_used_ = str(device)
        if parallel_ids:
            gpu_text = ",".join(str(idx) for idx in parallel_ids)
            self._training_device_used_ = f"{device} [data_parallel:{gpu_text}]"
        self._device_used_ = self._training_device_used_

        base_model = self._build_model(self.n_features_in_).to(device)
        self.model_ = base_model
        if parallel_ids:
            self.model_ = nn.DataParallel(
                base_model,
                device_ids=parallel_ids,
                output_device=parallel_ids[0],
            )
        elif self.verbose and bool(self.use_data_parallel) and device.type == "cuda" and allow_data_parallel:
            print("[Transformer+BNN] DataParallel requested but fewer than 2 valid CUDA devices were found; using single GPU.")
        try:
            optimizer_factory = self._build_optimizer()
            optimizer = optimizer_factory(
                self.model_.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.scheduler_factor),
                patience=max(2, int(self.scheduler_patience)),
                min_lr=float(self.min_learning_rate),
            )

            train_loader = self._make_loader(X_fit, mask_fit, y_fit, shuffle=True, batch_size=effective_batch_size)
            val_loader = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loader = self._make_loader(X_val, mask_val, y_val, shuffle=False, batch_size=effective_batch_size)

            self.train_loss_history = []
            self.test_loss_history = []
            self.epoch_time_history = []
            self.elapsed_time_history = []
            self.eta_history = []
            self.estimated_total_time_history = []
            self.learning_rate_history = []

            best_state = None
            best_score = math.inf
            stale_epochs = 0
            core_model = self._unwrap_model()

            use_amp = device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            fit_start_time = time.perf_counter()

            for epoch in range(int(self.epochs)):
                _raise_if_cancelled()
                epoch_start_time = time.perf_counter()
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
                        mean, logvar = self.model_(batch_x, batch_mask)
                        loss = self._loss(mean, logvar, batch_y)
                    scaler.scale(loss).backward()
                    if float(self.gradient_clip_norm) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model_.parameters(),
                            max_norm=float(self.gradient_clip_norm),
                        )
                    scaler.step(optimizer)
                    scaler.update()

                    current_batch_size = batch_x.size(0)
                    running_loss += float(loss.detach().cpu().item()) * current_batch_size
                    count += current_batch_size

                train_loss = running_loss / max(1, count)
                self.train_loss_history.append(train_loss)

                val_loss = None
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
                            mean, logvar = self.model_(batch_x, batch_mask)
                            loss = self._loss(mean, logvar, batch_y)
                            current_batch_size = batch_x.size(0)
                            val_running += float(loss.detach().cpu().item()) * current_batch_size
                            val_count += current_batch_size
                    val_loss = val_running / max(1, val_count)
                    self.test_loss_history.append(val_loss)

                tracked = val_loss if val_loss is not None else train_loss
                scheduler.step(tracked)

                if tracked < best_score - 1e-8:
                    best_score = tracked
                    best_state = copy.deepcopy(core_model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                epoch_seconds = time.perf_counter() - epoch_start_time
                elapsed_seconds = time.perf_counter() - fit_start_time
                avg_epoch_seconds = elapsed_seconds / max(1, epoch + 1)
                estimated_total_seconds = avg_epoch_seconds * max(1, int(self.epochs))
                eta_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))

                self.epoch_time_history.append(epoch_seconds)
                self.elapsed_time_history.append(elapsed_seconds)
                self.eta_history.append(eta_seconds)
                self.estimated_total_time_history.append(estimated_total_seconds)
                self.learning_rate_history.append(current_lr)
                self.epochs_completed_ = epoch + 1

                if callable(epoch_callback):
                    callback_payload = {
                        "epoch": epoch + 1,
                        "total_epochs": int(self.epochs),
                        "train_loss": float(train_loss),
                        "val_loss": None if val_loss is None else float(val_loss),
                        "epoch_seconds": float(epoch_seconds),
                        "elapsed_seconds": float(elapsed_seconds),
                        "eta_seconds": float(eta_seconds),
                        "estimated_total_seconds": float(estimated_total_seconds),
                        "learning_rate": float(current_lr),
                        "device": self._training_device_used_,
                        "data_parallel_enabled": bool(parallel_ids),
                        "gpu_ids": list(parallel_ids),
                        "effective_batch_size": int(self.effective_batch_size_),
                        "stale_epochs": int(stale_epochs),
                        "patience": int(self.patience),
                    }
                    try:
                        epoch_callback(callback_payload)
                    except Exception:
                        pass

                if self.verbose:
                    message = (
                        f"[Transformer+BNN] epoch={epoch + 1}/{self.epochs} "
                        f"train_loss={train_loss:.6f}"
                    )
                    if val_loss is not None:
                        message += f" val_loss={val_loss:.6f}"
                    message += (
                        f" batch_size={self.effective_batch_size_} "
                        f"epoch_time={epoch_seconds:.2f}s eta={eta_seconds:.2f}s "
                        f"device={self._training_device_used_}"
                    )
                    print(message)

                if stale_epochs >= int(self.patience):
                    break

            if best_state is not None and core_model is not None:
                core_model.load_state_dict(best_state)

            _raise_if_cancelled()
            self.model_.eval()
            core_model = self._unwrap_model()
            with torch.no_grad():
                probe_rows = min(len(X_fit), max(64, min(512, int(self.effective_batch_size_) * 4)))
                probe = torch.from_numpy(X_fit[:probe_rows]).float().to(device)
                probe_mask = torch.from_numpy(mask_fit[:probe_rows]).float().to(device)
                _, gates = core_model.backbone.encode(probe, missing_mask=probe_mask, return_gates=True)
                self.feature_importances_ = gates.mean(dim=0).detach().cpu().numpy()
        finally:
            if self.model_ is not None:
                if isinstance(self.model_, nn.DataParallel):
                    self.model_ = self.model_.module
                try:
                    self.model_.to(torch.device("cpu"))
                    self.model_.eval()
                except Exception:
                    pass
            self._device_used_ = "cpu"
            _clear_cuda_memory()

    def fit(self, X, y):
        if not TRANSFORMER_BNN_AVAILABLE:
            raise ImportError("PyTorch and FT-Transformer backbone are required")

        _set_seed(int(self.random_state))
        self._emit_status("preprocessing", "checking input data", progress_ratio=0.01)
        X_train, mask_train = self._prepare_features_and_mask(X, fit=True)
        y_train = self._prepare_target(y)
        self.n_features_in_ = X_train.shape[1]

        self._emit_status(
            "preprocessing",
            "building validation split",
            progress_ratio=0.16,
            n_features=int(self.n_features_in_),
            effective_missing_strategy=self._effective_missing_value_strategy_,
        )
        X_fit, mask_fit, y_fit, X_val, mask_val, y_val = self._split_validation(X_train, mask_train, y_train)
        epoch_callback = self.epoch_callback
        requested_batch_size = max(4, int(self.batch_size))
        initial_batch_size = self._estimate_effective_batch_size(self.n_features_in_)
        if self.verbose and initial_batch_size < requested_batch_size:
            print(
                f"[Transformer+BNN] detected high feature dimensionality ({self.n_features_in_}); "
                f"starting with safer batch_size={initial_batch_size} instead of requested {requested_batch_size}."
            )
        batch_schedule = self._build_batch_schedule(initial_batch_size)
        if bool(self.use_data_parallel):
            attempt_settings = [(True, bs) for bs in batch_schedule] + [(False, bs) for bs in batch_schedule]
        else:
            attempt_settings = [(False, bs) for bs in batch_schedule]

        deduped_attempts: list[tuple[bool, int]] = []
        for setting in attempt_settings:
            if setting not in deduped_attempts:
                deduped_attempts.append(setting)

        last_oom_error: RuntimeError | None = None
        _clear_cuda_memory()
        try:
            self._emit_status(
                "preprocessing",
                "starting training loop",
                progress_ratio=0.20,
                effective_batch_size=int(initial_batch_size),
                effective_missing_strategy=self._effective_missing_value_strategy_,
            )
            for attempt_idx, (allow_data_parallel, effective_batch_size) in enumerate(deduped_attempts, start=1):
                _raise_if_cancelled()
                if self.verbose and len(deduped_attempts) > 1:
                    mode_label = "multi_gpu" if allow_data_parallel else "single_gpu"
                    print(
                        f"[Transformer+BNN] training attempt {attempt_idx}/{len(deduped_attempts)} "
                        f"| mode={mode_label} | batch_size={effective_batch_size}"
                    )
                try:
                    self._fit_single_attempt(
                        X_fit,
                        mask_fit,
                        y_fit,
                        X_val,
                        mask_val,
                        y_val,
                        epoch_callback,
                        effective_batch_size=effective_batch_size,
                        allow_data_parallel=allow_data_parallel,
                    )
                    return self
                except RuntimeError as exc:
                    if not _is_cuda_oom_error(exc):
                        self._cleanup_failed_attempt()
                        raise
                    last_oom_error = exc
                    if self.verbose:
                        print(
                            f"[Transformer+BNN] CUDA OOM on attempt {attempt_idx}/{len(deduped_attempts)} "
                            f"(mode={'multi_gpu' if allow_data_parallel else 'single_gpu'}, "
                            f"batch_size={effective_batch_size}). Cleaning cache and retrying..."
                        )
                    self._cleanup_failed_attempt()
                except Exception:
                    self._cleanup_failed_attempt()
                    raise

            guidance = (
                "CUDA out of memory after automatic retries. "
                f"Requested batch_size={requested_batch_size}, last attempted batch_size={deduped_attempts[-1][1]}, "
                f"n_features={self.n_features_in_}. Reduce batch_size, lower d_token / n_blocks / ffn_d_hidden, "
                "or disable multi-GPU DataParallel."
            )
            raise RuntimeError(guidance) from last_oom_error
        finally:
            self.epoch_callback = None

    def _predict_raw(self, X, mc_samples: int | None = None):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet")

        X, missing_mask = self._prepare_features_and_mask(X, fit=False)
        x_tensor = torch.from_numpy(X).float()
        mask_tensor = torch.from_numpy(missing_mask).float()
        samples = max(1, int(mc_samples if mc_samples is not None else self.mc_samples))

        self.model_.to(torch.device("cpu"))
        mean_preds = []
        aleatoric_vars = []
        with torch.no_grad():
            if samples <= 1:
                self.model_.eval()
                mean, logvar = self.model_(x_tensor, mask_tensor)
                mean_preds.append(mean.detach().cpu().numpy())
                aleatoric_vars.append(
                    np.exp(
                        np.clip(
                            logvar.detach().cpu().numpy(),
                            float(self.min_logvar),
                            float(self.max_logvar),
                        )
                    )
                )
            else:
                for _ in range(samples):
                    _enable_dropout_only(self.model_)
                    mean, logvar = self.model_(x_tensor, mask_tensor)
                    mean_preds.append(mean.detach().cpu().numpy())
                    aleatoric_vars.append(
                        np.exp(
                            np.clip(
                                logvar.detach().cpu().numpy(),
                                float(self.min_logvar),
                                float(self.max_logvar),
                            )
                        )
                    )
                self.model_.eval()

        mean_preds = np.stack(mean_preds, axis=0)
        aleatoric_vars = np.stack(aleatoric_vars, axis=0)
        pred_mean = mean_preds.mean(axis=0)
        epistemic_var = mean_preds.var(axis=0)
        aleatoric_var = aleatoric_vars.mean(axis=0)
        pred_std = np.sqrt(np.maximum(epistemic_var + aleatoric_var, 1e-12))
        return pred_mean.ravel().astype(np.float64), pred_std.ravel().astype(np.float64)

    def predict_with_uncertainty(self, X, n_samples: int | None = None):
        return self._predict_raw(X, mc_samples=n_samples)

    def predict(self, X):
        mean, _ = self._predict_raw(X, mc_samples=self.mc_samples)
        return mean

    def get_training_history(self) -> dict:
        history = {"loss": list(self.train_loss_history)}
        if self.train_loss_history:
            history["train_mse"] = list(self.train_loss_history)
        if self.test_loss_history:
            history["val_loss"] = list(self.test_loss_history)
            history["test_mse"] = list(self.test_loss_history)
        if self.epoch_time_history:
            history["epoch_seconds"] = list(self.epoch_time_history)
        if self.elapsed_time_history:
            history["elapsed_seconds"] = list(self.elapsed_time_history)
        if self.eta_history:
            history["eta_seconds"] = list(self.eta_history)
        if self.estimated_total_time_history:
            history["estimated_total_seconds"] = list(self.estimated_total_time_history)
        if self.learning_rate_history:
            history["learning_rate"] = list(self.learning_rate_history)
        return history

    def score(self, X, y):
        pred = self.predict(X)
        y_true = self._prepare_target(y).astype(np.float64, copy=False)
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom <= 0:
            return 0.0
        return 1.0 - float(np.sum((y_true - pred) ** 2) / denom)
