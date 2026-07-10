# -*- coding: utf-8 -*-
"""Approximate Bayesian neural network regressor based on MC dropout."""

from __future__ import annotations

import copy
import gc
import math
import random

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    BNN_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    BNN_AVAILABLE = False

try:
    from .task_manager import is_cancelled as _task_manager_is_cancelled
except Exception:
    def _task_manager_is_cancelled() -> bool:
        return False


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


def _parse_hidden_layers(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value if int(v) > 0]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _enable_dropout_only(module: nn.Module) -> None:
    module.eval()
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()


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


class ProbabilisticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        activation: str,
        dropout_rate: float,
        use_batch_norm: bool,
        batch_norm_momentum: float,
    ):
        super().__init__()
        layers = []
        prev = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(int(hidden_dim), momentum=float(batch_norm_momentum)))
            layers.append(_make_activation(activation))
            layers.append(nn.Dropout(float(dropout_rate)))
            prev = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar


class BayesianNeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible BNN using heteroscedastic loss and MC dropout."""

    def __init__(
        self,
        hidden_layer_sizes_str: str = "256,128,64",
        activation: str = "relu",
        dropout_rate: float = 0.15,
        use_batch_norm: bool = False,
        batch_norm_momentum: float = 0.05,
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
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.hidden_layer_sizes_str = hidden_layer_sizes_str
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
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
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.validation_data = None
        self.train_loss_history = []
        self.test_loss_history = []
        self.n_features_in_ = None
        self._device_used_ = "cpu"

    @staticmethod
    def is_available() -> bool:
        return BNN_AVAILABLE

    def _select_device(self) -> torch.device:
        requested = str(self.device or "auto").lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_features(self, X, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.external_preprocess:
            return X
        if fit:
            X = self.imputer_.fit_transform(X)
            X = self.scaler_.fit_transform(X)
        else:
            X = self.imputer_.transform(X)
            X = self.scaler_.transform(X)
        return np.asarray(X, dtype=np.float32)

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

    def _build_model(self, input_dim: int) -> ProbabilisticMLP:
        hidden_layers = _parse_hidden_layers(self.hidden_layer_sizes_str)
        if not hidden_layers:
            hidden_layers = [128, 64]
        return ProbabilisticMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation=self.activation,
            dropout_rate=float(self.dropout_rate),
            use_batch_norm=bool(self.use_batch_norm),
            batch_norm_momentum=float(self.batch_norm_momentum),
        )

    def _split_validation(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            X_val = self._prepare_features(X_val, fit=False)
            y_val = self._prepare_target(y_val)
            return X, y, X_val, y_val

        val_ratio = float(self.validation_split)
        if val_ratio <= 0.0 or len(X) < 20:
            return X, y, None, None

        rng = np.random.RandomState(int(self.random_state))
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split_idx = max(1, int(round(len(X) * (1.0 - val_ratio))))
        split_idx = min(split_idx, len(X) - 1)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=float(self.min_logvar), max=float(self.max_logvar))
        if str(self.loss_name).lower() == "mse":
            return torch.mean((mean - target) ** 2)
        var = torch.exp(logvar)
        return 0.5 * torch.mean(logvar + ((target - mean) ** 2) / var)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
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
        if not BNN_AVAILABLE:
            raise ImportError("PyTorch is required for Bayesian Neural Network")

        _set_seed(int(self.random_state))
        X_train = self._prepare_features(X, fit=True)
        y_train = self._prepare_target(y)
        self.n_features_in_ = X_train.shape[1]

        X_fit, y_fit, X_val, y_val = self._split_validation(X_train, y_train)
        device = self._select_device()
        self._device_used_ = str(device)

        self.model_ = self._build_model(self.n_features_in_).to(device)
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

            train_loader = self._make_loader(X_fit, y_fit, shuffle=True)
            val_loader = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loader = self._make_loader(X_val, y_val, shuffle=False)

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

                for batch_x, batch_y in train_loader:
                    _raise_if_cancelled()
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True).squeeze(-1)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        mean, logvar = self.model_(batch_x)
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
                        for batch_x, batch_y in val_loader:
                            _raise_if_cancelled()
                            batch_x = batch_x.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True).squeeze(-1)
                            mean, logvar = self.model_(batch_x)
                            loss = self._loss(mean, logvar, batch_y)
                            batch_size = batch_x.size(0)
                            val_running += float(loss.detach().cpu().item()) * batch_size
                            val_count += batch_size
                    val_loss = val_running / max(1, val_count)
                    self.test_loss_history.append(val_loss)

                scheduler.step(val_loss)

                tracked = val_loss if val_loader is not None else train_loss
                if tracked < best_score - 1e-8:
                    best_score = tracked
                    best_state = copy.deepcopy(self.model_.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if self.verbose:
                    message = (
                        f"[BNN] epoch={epoch + 1}/{self.epochs} "
                        f"train_loss={train_loss:.6f}"
                    )
                    if val_loader is not None:
                        message += f" val_loss={val_loss:.6f}"
                    print(message)

                if stale_epochs >= int(self.patience):
                    break

            if best_state is not None:
                self.model_.load_state_dict(best_state)
            return self
        finally:
            if self.model_ is not None:
                try:
                    self.model_.to(torch.device("cpu"))
                except Exception:
                    pass
            self._device_used_ = "cpu"
            _clear_cuda_memory()

    def _predict_raw(self, X, mc_samples: int | None = None):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet")

        X = self._prepare_features(X, fit=False)
        x_tensor = torch.from_numpy(X).float()
        samples = max(1, int(mc_samples if mc_samples is not None else self.mc_samples))

        self.model_.to(torch.device("cpu"))
        mean_preds = []
        aleatoric_vars = []
        with torch.no_grad():
            if samples <= 1:
                self.model_.eval()
                mean, logvar = self.model_(x_tensor)
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
                    mean, logvar = self.model_(x_tensor)
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
        return history

    def score(self, X, y):
        pred = self.predict(X)
        y_true = self._prepare_target(y).astype(np.float64, copy=False)
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom <= 0:
            return 0.0
        return 1.0 - float(np.sum((y_true - pred) ** 2) / denom)
