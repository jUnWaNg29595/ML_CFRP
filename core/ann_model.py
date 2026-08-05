# -*- coding: utf-8 -*-
"""sklearn-compatible feed-forward ANN regressor."""

import copy
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from .task_manager import is_cancelled
except ImportError:
    def is_cancelled():
        return False

from .multi_gpu_utils import (
    get_model_for_inference,
    print_gpu_info,
    unwrap_data_parallel,
    wrap_model_for_multi_gpu,
)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}
_OPTIMIZERS = {"adam", "adamw", "rmsprop", "sgd"}
_SCHEDULERS = {"none", "reduce_on_plateau", "cosine_annealing"}


def _normalise_activation(activation):
    value = str(activation or "relu").strip().lower().replace("-", "_")
    if value not in _ACTIVATIONS:
        raise ValueError(
            "activation must be one of relu, gelu, silu, elu, tanh, leaky_relu"
        )
    return value


def _parse_hidden_layers(hidden_layers):
    if isinstance(hidden_layers, str):
        parts = [part.strip() for part in hidden_layers.split(",") if part.strip()]
    else:
        try:
            parts = list(hidden_layers)
        except TypeError as error:
            raise ValueError("hidden_layer_sizes_str must contain positive integers") from error
    if not parts:
        raise ValueError("hidden_layer_sizes_str must contain positive integers")
    try:
        parsed = [int(part) for part in parts]
    except (TypeError, ValueError) as error:
        raise ValueError("hidden_layer_sizes_str must contain positive integers") from error
    if any(size <= 0 for size in parsed):
        raise ValueError("hidden_layer_sizes_str must contain positive integers")
    return parsed


def _validate_non_negative_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative number") from error
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return numeric


class FFN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        activation="relu",
        dropout_rate=0.0,
        output_dim=1,
    ):
        super().__init__()
        activation_name = _normalise_activation(activation)
        dropout_rate = _validate_non_negative_number(dropout_rate, "dropout_rate")
        if dropout_rate > 0.8:
            raise ValueError("dropout_rate must be between 0 and 0.8")
        layers = []
        previous_dim = int(input_dim)
        for layer_size in _parse_hidden_layers(hidden_layers):
            layers.append(nn.Linear(previous_dim, layer_size))
            layers.append(_ACTIVATIONS[activation_name]())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            previous_dim = layer_size
        layers.append(nn.Linear(previous_dim, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes_str="256,128,64",
        activation="relu",
        dropout_rate=0.0,
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=512,
        epochs=100,
        validation_split=0.0,
        early_stopping=True,
        patience=10,
        min_delta=0.0,
        lr_scheduler="none",
        scheduler_factor=0.5,
        min_learning_rate=1e-6,
        gradient_clip=0.0,
        use_amp=True,
        verbose=True,
        random_state=42,
        external_preprocess=False,
        device="auto",
        use_data_parallel=True,
        validation_data=None,
    ):
        self.hidden_layer_sizes_str = hidden_layer_sizes_str
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.lr_scheduler = lr_scheduler
        self.scheduler_factor = scheduler_factor
        self.min_learning_rate = min_learning_rate
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.verbose = verbose
        self.random_state = random_state
        self.external_preprocess = external_preprocess
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.validation_data = validation_data

        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.train_loss_history = []
        self.validation_loss_history = []
        self.test_loss_history = self.validation_loss_history
        self.training_metadata_ = {}
        self.history_ = {}
        self._device_used_ = None
        self._is_data_parallel = False
        self._device_fallback_reasons_ = []

    def _validate_params(self):
        _parse_hidden_layers(self.hidden_layer_sizes_str)
        dropout_rate = _validate_non_negative_number(self.dropout_rate, "dropout_rate")
        if dropout_rate > 0.8:
            raise ValueError("dropout_rate must be between 0 and 0.8")

        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("learning_rate must be positive") from error
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        weight_decay = _validate_non_negative_number(self.weight_decay, "weight_decay")
        try:
            min_learning_rate = float(self.min_learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("min_learning_rate must be positive") from error
        if not np.isfinite(min_learning_rate) or min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be positive")
        if min_learning_rate > learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")

        optimizer = str(self.optimizer or "").strip().lower()
        if optimizer not in _OPTIMIZERS:
            raise ValueError("optimizer must be one of adam, adamw, rmsprop, sgd")

        try:
            validation_split = float(self.validation_split)
        except (TypeError, ValueError) as error:
            raise ValueError("validation_split must be between 0 and 0.4") from error
        if not np.isfinite(validation_split) or not 0 <= validation_split <= 0.4:
            raise ValueError("validation_split must be between 0 and 0.4")

        try:
            patience = int(self.patience)
        except (TypeError, ValueError) as error:
            raise ValueError("patience must be a non-negative integer") from error
        if isinstance(self.patience, bool) or patience != self.patience:
            raise ValueError("patience must be a non-negative integer")
        if patience < 0:
            raise ValueError("patience must be a non-negative integer")

        min_delta = _validate_non_negative_number(self.min_delta, "min_delta")
        scheduler = str(self.lr_scheduler or "none").strip().lower()
        if scheduler not in _SCHEDULERS:
            raise ValueError(
                "lr_scheduler must be one of none, reduce_on_plateau, cosine_annealing"
            )
        try:
            scheduler_factor = float(self.scheduler_factor)
        except (TypeError, ValueError) as error:
            raise ValueError("scheduler_factor must be greater than 0 and less than 1") from error
        if (
            not np.isfinite(scheduler_factor)
            or scheduler_factor <= 0
            or scheduler_factor >= 1
        ):
            raise ValueError("scheduler_factor must be greater than 0 and less than 1")

        if isinstance(self.gradient_clip, bool):
            raise ValueError("gradient_clip must be non-negative")
        gradient_clip = _validate_non_negative_number(self.gradient_clip, "gradient_clip")

        try:
            batch_size = int(self.batch_size)
        except (TypeError, ValueError) as error:
            raise ValueError("batch_size must be a positive integer") from error
        if isinstance(self.batch_size, bool) or batch_size != self.batch_size:
            raise ValueError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            epochs = int(self.epochs)
        except (TypeError, ValueError) as error:
            raise ValueError("epochs must be a non-negative integer") from error
        if isinstance(self.epochs, bool) or epochs != self.epochs:
            raise ValueError("epochs must be a non-negative integer")
        if epochs < 0:
            raise ValueError("epochs must be a non-negative integer")

        return {
            "activation": _normalise_activation(self.activation),
            "dropout_rate": dropout_rate,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "validation_split": validation_split,
            "patience": patience,
            "min_delta": min_delta,
            "lr_scheduler": scheduler,
            "scheduler_factor": scheduler_factor,
            "min_learning_rate": min_learning_rate,
            "gradient_clip": gradient_clip,
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_layer_sizes": _parse_hidden_layers(self.hidden_layer_sizes_str),
        }

    def _select_device(self):
        self._device_fallback_reasons_ = []
        requested = str(self.device or "auto").strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" or requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                self._device_fallback_reasons_.append(
                    f"requested {requested}, CUDA unavailable; fell back to CPU"
                )
                return torch.device("cpu")
            try:
                selected = torch.device(requested)
                if selected.index is not None and selected.index >= torch.cuda.device_count():
                    self._device_fallback_reasons_.append(
                        f"requested {requested}, CUDA index unavailable; fell back to CPU"
                    )
                    return torch.device("cpu")
                return selected
            except (RuntimeError, ValueError):
                self._device_fallback_reasons_.append(
                    f"requested {requested}, invalid CUDA device; fell back to CPU"
                )
                return torch.device("cpu")
        raise ValueError("device must be one of auto, cpu, cuda, or cuda:<index>")

    @staticmethod
    def _as_feature_array(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        return array

    @staticmethod
    def _as_target_array(y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        array = np.asarray(y, dtype=np.float32).ravel()
        if array.ndim != 1:
            raise ValueError("y must be a one-dimensional target array")
        return array

    def _prepare_features(self, X, fit=False):
        X = self._as_feature_array(X)
        if self.external_preprocess:
            return np.nan_to_num(
                X.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        if fit:
            X = self.imputer.fit_transform(X)
            return self.scaler.fit_transform(X).astype(np.float32)
        X = self.imputer.transform(X)
        return self.scaler.transform(X).astype(np.float32)

    def _split_training_data(self, X, y, validation_split):
        if self.validation_data is not None:
            return X, y, self.validation_data, np.arange(len(y), dtype=int)
        if validation_split <= 0 or len(y) < 2:
            return X, y, None, np.arange(len(y), dtype=int)
        validation_count = int(round(len(y) * validation_split))
        validation_count = min(max(1, validation_count), len(y) - 1)
        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(y))
        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]
        return (
            X[training_indices],
            y[training_indices],
            (X[validation_indices], y[validation_indices]),
            training_indices,
        )

    @staticmethod
    def _make_optimizer(name, parameters, learning_rate, weight_decay):
        kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
        if name == "adam":
            return torch.optim.Adam(parameters, **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(parameters, **kwargs)
        if name == "rmsprop":
            return torch.optim.RMSprop(parameters, **kwargs)
        return torch.optim.SGD(parameters, **kwargs)

    def _make_scheduler(self, optimizer, config):
        if config["lr_scheduler"] == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min", factor=config["scheduler_factor"],
                patience=max(1, config["patience"]),
                min_lr=config["min_learning_rate"],
            )
        if config["lr_scheduler"] == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config["epochs"]),
                eta_min=config["min_learning_rate"],
            )
        return None

    @staticmethod
    def _make_amp_tools(device, use_amp):
        if device.type != "cuda" or not use_amp:
            return False, None, None
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            autocast = lambda: torch.amp.autocast(device_type="cuda", enabled=True)
        except (AttributeError, TypeError):
            from torch.cuda.amp import GradScaler, autocast as cuda_autocast
            scaler = GradScaler(enabled=True)
            autocast = lambda: cuda_autocast(enabled=True)
        return True, scaler, autocast

    def _ensure_compat_attributes(self):
        defaults = {
            "activation": "relu",
            "dropout_rate": 0.0,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "validation_split": 0.0,
            "early_stopping": True,
            "patience": 10,
            "min_delta": 0.0,
            "lr_scheduler": "none",
            "scheduler_factor": 0.5,
            "min_learning_rate": 1e-6,
            "gradient_clip": 0.0,
            "use_amp": True,
            "validation_data": None,
            "training_metadata_": {},
            "history_": {},
            "validation_loss_history": [],
            "device": "auto",
            "use_data_parallel": True,
            "external_preprocess": False,
            "random_state": 42,
            "batch_size": 512,
            "epochs": 100,
            "learning_rate": 0.001,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, copy.deepcopy(value))
        if not hasattr(self, "test_loss_history"):
            self.test_loss_history = self.validation_loss_history

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_compat_attributes()

    def fit(self, X, y, sample_weight=None):
        self._ensure_compat_attributes()
        config = self._validate_params()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        X_array = self._as_feature_array(X)
        y_array = self._as_target_array(y)
        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of samples")
        if len(X_array) == 0:
            raise ValueError("cannot train ANN with no samples")

        if sample_weight is None:
            weights = None
        else:
            weights = np.asarray(sample_weight, dtype=np.float32).ravel()
            if len(weights) != len(y_array):
                raise ValueError("sample_weight must match the number of samples")
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("sample_weight must contain finite non-negative values")

        X_scaled = self._prepare_features(X_array, fit=True)
        train_X, train_y, validation, training_indices = self._split_training_data(
            X_scaled,
            y_array,
            config["validation_split"],
        )
        train_weights = None if weights is None else weights[training_indices]

        validation_scaled = None
        if validation is not None:
            validation_X, validation_y = validation
            if self.validation_data is not None:
                validation_scaled = (
                    self._prepare_features(validation_X, fit=False),
                    self._as_target_array(validation_y),
                )
            else:
                validation_scaled = (
                    np.asarray(validation_X, dtype=np.float32),
                    self._as_target_array(validation_y),
                )

        device = self._select_device()
        self._device_used_ = str(device)
        input_dim = train_X.shape[1]
        hidden_layers = config["hidden_layer_sizes"]
        model = FFN(
            input_dim,
            hidden_layers,
            activation=config["activation"],
            dropout_rate=config["dropout_rate"],
        )
        requested_parallel = bool(self.use_data_parallel)
        if device.type == "cuda" and device.index not in (None, 0) and requested_parallel:
            requested_parallel = False
            self._device_fallback_reasons_.append(
                f"requested DataParallel on non-zero CUDA device {device}; "
                "disabled DataParallel and used the selected single GPU"
            )
        try:
            model = model.to(device)
            model, is_parallel, gpu_count = wrap_model_for_multi_gpu(
                model,
                device,
                requested_parallel,
                verbose=self.verbose,
            )
        except Exception as error:
            self._device_fallback_reasons_.append(
                f"DataParallel/device initialization failed ({type(error).__name__}: {error}); "
                "fell back to single GPU/CPU"
            )
            is_parallel = False
            gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
            try:
                model = model.to(device)
            except Exception as device_error:
                self._device_fallback_reasons_.append(
                    f"single-device initialization failed ({type(device_error).__name__}); "
                    "fell back to CPU"
                )
                device = torch.device("cpu")
                self._device_used_ = str(device)
                model = model.to(device)
        self.model = model
        self._is_data_parallel = is_parallel

        train_tensors = [
            torch.from_numpy(np.asarray(train_X, dtype=np.float32)),
            torch.from_numpy(np.asarray(train_y, dtype=np.float32)).view(-1, 1),
        ]
        if train_weights is not None:
            train_tensors.append(
                torch.from_numpy(np.asarray(train_weights, dtype=np.float32)).view(-1, 1)
            )
        dataset = TensorDataset(*train_tensors)
        generator = torch.Generator()
        generator.manual_seed(int(self.random_state) + 1)
        loader = DataLoader(
            dataset,
            batch_size=min(config["batch_size"], len(dataset)),
            shuffle=True,
            generator=generator,
            pin_memory=device.type == "cuda",
            num_workers=0 if platform.system() == "Windows" else 0,
        )

        criterion = nn.MSELoss(reduction="none")
        optimizer = self._make_optimizer(
            config["optimizer"],
            self.model.parameters(),
            config["learning_rate"],
            config["weight_decay"],
        )
        scheduler = self._make_scheduler(optimizer, config)
        amp_enabled, amp_scaler, autocast = self._make_amp_tools(device, self.use_amp)

        if self.verbose:
            print_gpu_info(device, gpu_count)
            print(
                f"[ANN] samples={len(train_y)}, batch_size={loader.batch_size}, "
                f"device={device}, amp={amp_enabled}, data_parallel={is_parallel}"
            )

        self.train_loss_history = []
        self.validation_loss_history = []
        self.test_loss_history = self.validation_loss_history
        best_state = None
        best_validation_loss = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopped = False
        early_stop_reason = "disabled"

        iterator = range(config["epochs"])
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Training ANN ({self._device_used_})")

        for epoch in iterator:
            if is_cancelled():
                early_stop_reason = "cancelled"
                break
            self.model.train()
            loss_sum = 0.0
            weight_sum = 0.0
            for batch in loader:
                batch_X, batch_y = batch[:2]
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_weights = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    with autocast():
                        predictions = self.model(batch_X)
                        losses = criterion(predictions, batch_y)
                        if batch_weights is None:
                            loss = losses.mean()
                        else:
                            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
                    amp_scaler.scale(loss).backward()
                    if config["gradient_clip"] > 0:
                        amp_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config["gradient_clip"]
                        )
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    predictions = self.model(batch_X)
                    losses = criterion(predictions, batch_y)
                    if batch_weights is None:
                        loss = losses.mean()
                    else:
                        loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
                    loss.backward()
                    if config["gradient_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config["gradient_clip"]
                        )
                    optimizer.step()
                batch_weight_sum = float(
                    batch_weights.sum().detach().cpu().item()
                    if batch_weights is not None
                    else len(batch_X)
                )
                loss_sum += float(loss.detach().cpu().item()) * batch_weight_sum
                weight_sum += batch_weight_sum

            train_loss = loss_sum / max(weight_sum, 1.0)
            self.train_loss_history.append(train_loss)

            validation_loss = None
            if validation_scaled is not None and len(validation_scaled[1]) > 0:
                self.model.eval()
                validation_X, validation_y = validation_scaled
                with torch.no_grad():
                    predictions = self.model(
                        torch.from_numpy(validation_X).float().to(device)
                    )
                    validation_loss = float(
                        nn.functional.mse_loss(
                            predictions,
                            torch.from_numpy(validation_y).float().view(-1, 1).to(device),
                        ).cpu().item()
                    )
                self.validation_loss_history.append(validation_loss)

            if scheduler is not None:
                if config["lr_scheduler"] == "reduce_on_plateau":
                    scheduler.step(validation_loss if validation_loss is not None else train_loss)
                else:
                    scheduler.step()

            if validation_loss is None:
                if self.early_stopping and validation_scaled is None:
                    early_stop_reason = "no_validation"
                continue

            if validation_loss < best_validation_loss - config["min_delta"]:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                if self.early_stopping:
                    best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.early_stopping and epochs_without_improvement >= config["patience"]:
                    early_stopped = True
                    early_stop_reason = "patience_exceeded"
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if not best_epoch and self.train_loss_history:
            best_epoch = len(self.train_loss_history)
        if early_stop_reason == "disabled" and self.early_stopping and validation_scaled is None:
            early_stop_reason = "no_validation"
        if early_stop_reason == "disabled" and self.early_stopping and validation_scaled is not None:
            early_stop_reason = "max_epochs"
        if not self.early_stopping and validation_scaled is not None:
            early_stop_reason = "disabled"

        final_learning_rate = float(optimizer.param_groups[0]["lr"])
        self.model = unwrap_data_parallel(self.model)
        self._is_data_parallel = False
        self.model.to(torch.device("cpu"))
        self.history_ = {
            "train_loss": list(self.train_loss_history),
            "validation_loss": list(self.validation_loss_history),
        }
        self.training_metadata_ = {
            "config": {
                "hidden_layer_sizes_str": self.hidden_layer_sizes_str,
                "activation": config["activation"],
                "dropout_rate": config["dropout_rate"],
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "validation_split": config["validation_split"],
                "early_stopping": self.early_stopping,
                "patience": config["patience"],
                "min_delta": config["min_delta"],
                "lr_scheduler": config["lr_scheduler"],
                "scheduler_factor": config["scheduler_factor"],
                "min_learning_rate": config["min_learning_rate"],
                "gradient_clip": config["gradient_clip"],
                "use_amp": self.use_amp,
                "device": self.device,
                "use_data_parallel": self.use_data_parallel,
            },
            "device": self._device_used_,
            "requested_device": self.device,
            "effective_batch_size": loader.batch_size,
            "gpu_count": gpu_count,
            "amp_enabled": amp_enabled,
            "data_parallel_enabled": is_parallel,
            "fallback_reasons": list(self._device_fallback_reasons_),
            "n_samples": len(y_array),
            "n_train_samples": len(train_y),
            "n_validation_samples": 0 if validation_scaled is None else len(validation_scaled[1]),
            "best_epoch": best_epoch,
            "best_validation_loss": (
                None if not np.isfinite(best_validation_loss) else best_validation_loss
            ),
            "final_learning_rate": final_learning_rate,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "train_loss_history": list(self.train_loss_history),
            "validation_loss_history": list(self.validation_loss_history),
        }
        return self

    def predict(self, X):
        self._ensure_compat_attributes()
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        X_scaled = self._prepare_features(X, fit=False)
        device = self._select_device()
        self.model.eval()
        model_for_prediction = get_model_for_inference(self.model, device)
        with torch.no_grad():
            predictions = model_for_prediction(
                torch.from_numpy(X_scaled).float().to(device)
            )
            result = predictions.detach().cpu().numpy().ravel()
        self.model = unwrap_data_parallel(self.model)
        self.model.to(torch.device("cpu"))
        return result

    def score(self, X, y):
        y_array = self._as_target_array(y)
        return -np.mean((y_array - self.predict(X)) ** 2)
