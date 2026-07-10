# -*- coding: utf-8 -*-
"""Physics-informed Transformer regressor for epoxy property prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from .fttransformer_model import FTTransformerBackbone

    FT_BACKBONE_AVAILABLE = True
except Exception:
    FTTransformerBackbone = None
    FT_BACKBONE_AVAILABLE = False

from .pinn_model import EpoxyPINNRegressor


TRANSFORMER_PINN_AVAILABLE = bool(TORCH_AVAILABLE and FT_BACKBONE_AVAILABLE)


class _TransformerPINNCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
            self.head = nn.Sequential(
                nn.Linear(int(d_token), int(head_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(ffn_dropout)),
                nn.Linear(int(head_hidden_dim), int(output_dim)),
            )
        else:
            self.head = nn.Linear(int(d_token), int(output_dim))

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.backbone.encode(x, missing_mask=missing_mask)
        pooled = self.backbone.pool_tokens(tokens)
        return self.head(pooled)


class TransformerPINNRegressor(EpoxyPINNRegressor):
    """Epoxy PINN variant with FT-Transformer tabular encoder."""

    def __init__(
        self,
        mode: str = "auto",
        target_name: str | None = None,
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
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 500,
        patience: int = 50,
        physics_weight: float = 0.01,
        physics_formula: str = "standard",
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
        missing_value_strategy: str = "bayesian",
        missing_imputer_max_iter: int = 15,
        missing_n_imputations: int = 5,
        use_missingness_embedding: bool = True,
        use_missingness_attention: bool = True,
    ):
        super().__init__(
            mode=mode,
            target_name=target_name,
            hidden_dim=max(int(head_hidden_dim), int(d_token)),
            n_layers=max(1, int(n_blocks)),
            dropout=float(ffn_dropout),
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            physics_weight=physics_weight,
            physics_formula=physics_formula,
            grad_clip=grad_clip,
            device=device,
            seed=seed,
            verbose=verbose,
            missing_value_strategy=missing_value_strategy,
            missing_imputer_max_iter=missing_imputer_max_iter,
            missing_n_imputations=missing_n_imputations,
        )
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
        self.use_missingness_embedding = use_missingness_embedding
        self.use_missingness_attention = use_missingness_attention
        self.feature_importances_ = None

    @staticmethod
    def is_available() -> bool:
        return TRANSFORMER_PINN_AVAILABLE

    def _uses_missing_mask(self) -> bool:
        return True

    def _make_model(self, input_dim: int) -> nn.Module:
        if int(self.d_token) % int(self.attention_n_heads) != 0:
            raise ValueError("d_token must be divisible by attention_n_heads")
        mode = self._mode_ or "generic"
        if mode == "tg":
            output_dim = 4
        elif mode == "mechanics":
            output_dim = 2
        else:
            output_dim = 1
        return _TransformerPINNCore(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
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

    def fit(self, X, y):
        if not TRANSFORMER_PINN_AVAILABLE:
            raise ImportError("PyTorch and FT-Transformer backbone are required")
        result = super().fit(X, y)
        try:
            if isinstance(X, pd.DataFrame):
                df_raw = X.copy()
            else:
                X_arr = np.asarray(X)
                df_raw = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])
            y_arr = np.asarray(y).reshape(-1).astype(np.float32)
            valid = np.isfinite(y_arr)
            df_raw = df_raw.iloc[valid].reset_index(drop=True)
            X_scaled, _, missing_mask = self._transform(df_raw, return_missing_mask=True)
            probe = torch.from_numpy(X_scaled[: min(len(X_scaled), 2048)]).float()
            probe_mask = torch.from_numpy(missing_mask[: min(len(missing_mask), 2048)]).float()
            self._model_.eval()
            _, gates = self._model_.backbone.encode(probe, missing_mask=probe_mask, return_gates=True)
            self.feature_importances_ = gates.mean(dim=0).detach().cpu().numpy()
        except Exception:
            self.feature_importances_ = None
        return result
