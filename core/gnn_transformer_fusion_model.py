# -*- coding: utf-8 -*-
"""GNN + FT-Transformer fusion regressor for SMILES and tabular features."""

from __future__ import annotations

import copy
import gc
import math
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from .missing_value_handler import MissingValueHandler, build_missing_mask

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        GINConv,
        SAGEConv,
        NNConv,
        global_add_pool,
        global_mean_pool,
    )

    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    GATConv = None
    GCNConv = None
    GINConv = None
    SAGEConv = None
    NNConv = None
    global_add_pool = None
    global_mean_pool = None
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from .graph_utils import (
        smiles_to_pyg_graph,
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        RDKIT_AVAILABLE,
    )

    GRAPH_UTILS_AVAILABLE = True
except Exception:
    smiles_to_pyg_graph = None
    ATOM_FEATURE_DIM = 0
    BOND_FEATURE_DIM = 0
    RDKIT_AVAILABLE = False
    GRAPH_UTILS_AVAILABLE = False

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


GNN_TRANSFORMER_FUSION_AVAILABLE = bool(
    TORCH_GEOMETRIC_AVAILABLE and GRAPH_UTILS_AVAILABLE and RDKIT_AVAILABLE and FT_BACKBONE_AVAILABLE
)


def _set_seed(seed: Optional[int]) -> None:
    if torch is None or seed is None:
        return
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _resolve_device(device: str):
    if torch is None:
        return None
    device = str(device or "auto").lower()
    if device == "cpu":
        return torch.device("cpu")
    if device in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class _GraphEncoder(nn.Module):
    def __init__(
        self,
        model_type: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling: str,
        gat_heads: int,
    ):
        super().__init__()
        self.model_type = str(model_type or "gcn").lower()
        self.dropout = float(dropout)
        self.pooling = str(pooling or "mean").lower()
        self.use_edge_attr = self.model_type in {"mpnn"}

        layers = []
        in_channels = int(ATOM_FEATURE_DIM)

        if self.model_type == "gcn":
            for _ in range(int(num_layers)):
                layers.append(GCNConv(in_channels, int(hidden_dim)))
                in_channels = int(hidden_dim)
        elif self.model_type == "gat":
            heads = max(1, int(gat_heads))
            head_dim = max(1, int(hidden_dim) // heads)
            for _ in range(int(num_layers)):
                layers.append(GATConv(in_channels, head_dim, heads=heads, dropout=self.dropout))
                in_channels = head_dim * heads
        elif self.model_type == "gin":
            for _ in range(int(num_layers)):
                mlp = nn.Sequential(
                    nn.Linear(in_channels, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                layers.append(GINConv(mlp))
                in_channels = int(hidden_dim)
        elif self.model_type == "graphsage":
            for _ in range(int(num_layers)):
                layers.append(SAGEConv(in_channels, int(hidden_dim)))
                in_channels = int(hidden_dim)
        elif self.model_type == "mpnn":
            for _ in range(int(num_layers)):
                out_size = int(hidden_dim) * in_channels
                bottleneck = max(64, min(int(hidden_dim), max(64, out_size // 4)))
                edge_nn = nn.Sequential(
                    nn.Linear(int(BOND_FEATURE_DIM), bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, out_size),
                )
                layers.append(NNConv(in_channels, int(hidden_dim), edge_nn, aggr="mean"))
                in_channels = int(hidden_dim)
        else:
            raise ValueError(f"unsupported gnn model_type: {model_type}")

        self.output_dim = int(in_channels)
        self.convs = nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv in self.convs:
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == "add":
            return global_add_pool(x, batch)
        return global_mean_pool(x, batch)


class _FusionRegressor(nn.Module):
    def __init__(
        self,
        n_tabular_features: int,
        graph_model_type: str,
        graph_hidden_dim: int,
        graph_num_layers: int,
        graph_dropout: float,
        graph_pooling: str,
        gat_heads: int,
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
        transformer_pooling: str,
        fusion_hidden_dim: int,
        layer_norm_eps: float,
        use_missingness_embedding: bool,
        use_missingness_attention: bool,
    ):
        super().__init__()
        self.graph_encoder = _GraphEncoder(
            model_type=graph_model_type,
            hidden_dim=int(graph_hidden_dim),
            num_layers=int(graph_num_layers),
            dropout=float(graph_dropout),
            pooling=str(graph_pooling),
            gat_heads=int(gat_heads),
        )
        self.tabular_encoder = FTTransformerBackbone(
            n_features=int(n_tabular_features),
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
            pooling=str(transformer_pooling),
            head_hidden_dim=0,
            layer_norm_eps=float(layer_norm_eps),
            use_missingness_embedding=bool(use_missingness_embedding),
            use_missingness_attention=bool(use_missingness_attention),
        )
        fusion_in = int(self.graph_encoder.output_dim) + int(d_token)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, int(fusion_hidden_dim)),
            _make_activation(activation),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(int(fusion_hidden_dim), 1),
        )

    def forward(self, data):
        graph_vec = self.graph_encoder(data)
        tab_missing_mask = getattr(data, "tabular_missing_mask", None)
        tab_tokens = self.tabular_encoder.encode(data.tabular, missing_mask=tab_missing_mask)
        tab_vec = self.tabular_encoder.pool_tokens(tab_tokens)
        fused = torch.cat([graph_vec, tab_vec], dim=1)
        return self.head(fused).view(-1)


class GNNTransformerFusionRegressor(BaseEstimator, RegressorMixin):
    """Fuse graph embeddings from SMILES with tabular FT-Transformer embeddings."""

    def __init__(
        self,
        smiles_col: str = "smiles",
        graph_model_type: str = "gcn",
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 3,
        graph_dropout: float = 0.1,
        graph_pooling: str = "mean",
        gat_heads: int = 4,
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
        transformer_pooling: str = "cls",
        fusion_hidden_dim: int = 256,
        layer_norm_eps: float = 1e-5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 120,
        patience: int = 15,
        validation_split: float = 0.1,
        loss_name: str = "mse",
        gradient_clip_norm: float = 1.0,
        scheduler_factor: float = 0.6,
        scheduler_patience: int = 6,
        min_learning_rate: float = 1e-6,
        num_workers: int = 0,
        device: str = "auto",
        random_state: Optional[int] = 42,
        verbose: int = 0,
        missing_value_strategy: str = "bayesian",
        missing_imputer_max_iter: int = 15,
        missing_n_imputations: int = 5,
        use_missingness_embedding: bool = True,
        use_missingness_attention: bool = True,
    ):
        self.smiles_col = smiles_col
        self.graph_model_type = graph_model_type
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_num_layers = graph_num_layers
        self.graph_dropout = graph_dropout
        self.graph_pooling = graph_pooling
        self.gat_heads = gat_heads
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
        self.transformer_pooling = transformer_pooling
        self.fusion_hidden_dim = fusion_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.loss_name = loss_name
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_learning_rate = min_learning_rate
        self.num_workers = num_workers
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.missing_value_strategy = missing_value_strategy
        self.missing_imputer_max_iter = missing_imputer_max_iter
        self.missing_n_imputations = missing_n_imputations
        self.use_missingness_embedding = use_missingness_embedding
        self.use_missingness_attention = use_missingness_attention

        self.model_ = None
        self.device_ = None
        self.numeric_feature_names_ = None
        self.imputer_ = MissingValueHandler(
            strategy=self.missing_value_strategy,
            random_state=int(self.random_state or 42),
            max_iter=self.missing_imputer_max_iter,
            n_imputations=self.missing_n_imputations,
        )
        self.scaler_ = StandardScaler()
        self.validation_data = None
        self.train_loss_history = []
        self.test_loss_history = []
        self.feature_importances_ = None

    @staticmethod
    def is_available() -> bool:
        return GNN_TRANSFORMER_FUSION_AVAILABLE

    def _extract_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GNN + Transformer Fusion requires a DataFrame input")
        if self.smiles_col not in X.columns:
            raise ValueError(f"smiles_col '{self.smiles_col}' not found in input columns")
        return X.copy()

    def _prepare_numeric_block(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_num, _ = self._prepare_numeric_block_with_mask(df, fit=fit)
        return X_num

    def _prepare_numeric_block_with_mask(self, df: pd.DataFrame, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if fit:
            self.numeric_feature_names_ = [c for c in df.columns if c != self.smiles_col]
        if not self.numeric_feature_names_:
            raise ValueError("GNN + Transformer Fusion requires at least one numeric feature column")
        df_num = df[self.numeric_feature_names_].copy()
        for c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        X_num = df_num.to_numpy(dtype=np.float32)
        missing_mask = build_missing_mask(X_num)
        X_num = np.where(np.isfinite(X_num), X_num, np.nan)
        if fit:
            X_num = self.imputer_.fit_transform(X_num)
            X_num = self.scaler_.fit_transform(X_num)
        else:
            X_num = self.imputer_.transform(X_num)
            X_num = self.scaler_.transform(X_num)
        return np.asarray(X_num, dtype=np.float32), missing_mask

    def transform_tabular(self, X) -> np.ndarray:
        df = self._extract_frame(X)
        return self._prepare_numeric_block(df, fit=False)

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        y_vals: Optional[np.ndarray],
        fit_numeric: bool,
    ):
        X_num, missing_mask = self._prepare_numeric_block_with_mask(df, fit=fit_numeric)
        smiles_list = df[self.smiles_col].astype(str).tolist()

        graphs = []
        valid_idx = []
        for idx, smi in enumerate(smiles_list):
            data = smiles_to_pyg_graph(smi, add_hs=True) if smiles_to_pyg_graph else None
            if data is None:
                continue
            data.tabular = torch.tensor(X_num[idx], dtype=torch.float32).view(1, -1)
            data.tabular_missing_mask = torch.tensor(missing_mask[idx], dtype=torch.float32).view(1, -1)
            if y_vals is not None:
                data.y = torch.tensor([float(y_vals[idx])], dtype=torch.float32)
            graphs.append(data)
            valid_idx.append(idx)
        return graphs, valid_idx

    def _build_model(self, n_tabular_features: int):
        if int(self.d_token) % int(self.attention_n_heads) != 0:
            raise ValueError("d_token must be divisible by attention_n_heads")
        return _FusionRegressor(
            n_tabular_features=int(n_tabular_features),
            graph_model_type=str(self.graph_model_type),
            graph_hidden_dim=int(self.graph_hidden_dim),
            graph_num_layers=int(self.graph_num_layers),
            graph_dropout=float(self.graph_dropout),
            graph_pooling=str(self.graph_pooling),
            gat_heads=int(self.gat_heads),
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
            transformer_pooling=str(self.transformer_pooling),
            fusion_hidden_dim=int(self.fusion_hidden_dim),
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

    def _split_validation(self, graphs):
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            df_val = self._extract_frame(X_val)
            y_arr = np.asarray(y_val).ravel().astype(float)
            mask = np.isfinite(y_arr)
            df_val = df_val.loc[mask].reset_index(drop=True)
            y_arr = y_arr[mask]
            val_graphs, _ = self._prepare_dataset(df_val, y_arr, fit_numeric=False)
            return graphs, val_graphs

        val_ratio = float(self.validation_split)
        if val_ratio <= 0.0 or len(graphs) < 20:
            return graphs, []
        rng = np.random.RandomState(int(self.random_state or 42))
        indices = np.arange(len(graphs))
        rng.shuffle(indices)
        split_idx = max(1, int(round(len(graphs) * (1.0 - val_ratio))))
        split_idx = min(split_idx, len(graphs) - 1)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        return train_graphs, val_graphs

    def fit(self, X, y):
        if not GNN_TRANSFORMER_FUSION_AVAILABLE:
            raise ImportError("torch_geometric, RDKit, and FT-Transformer backbone are required")

        df = self._extract_frame(X)
        y_arr = np.asarray(y).ravel().astype(float)
        mask = np.isfinite(y_arr)
        df = df.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if len(df) == 0:
            raise ValueError("no valid samples for fusion training")

        _set_seed(self.random_state)
        device = _resolve_device(self.device)
        self.device_ = device

        graphs, _ = self._prepare_dataset(df, y_arr, fit_numeric=True)
        if not graphs:
            raise ValueError("all SMILES failed to parse into graphs")

        train_graphs, val_graphs = self._split_validation(graphs)
        if not train_graphs:
            raise ValueError("no valid training graphs after validation split")

        train_loader = DataLoader(
            train_graphs,
            batch_size=int(max(1, self.batch_size)),
            shuffle=True,
            num_workers=int(max(0, self.num_workers)),
        )
        val_loader = None
        if val_graphs:
            val_loader = DataLoader(
                val_graphs,
                batch_size=int(max(1, self.batch_size)),
                shuffle=False,
                num_workers=int(max(0, self.num_workers)),
            )

        self.model_ = self._build_model(len(self.numeric_feature_names_)).to(device)
        try:
            optimizer = torch.optim.AdamW(
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
            loss_fn = self._build_loss()

            self.train_loss_history = []
            self.test_loss_history = []
            best_state = None
            best_score = math.inf
            stale_epochs = 0

            for epoch in range(int(max(1, self.epochs))):
                _raise_if_cancelled()
                self.model_.train()
                total_loss = 0.0
                total_count = 0
                for batch in train_loader:
                    _raise_if_cancelled()
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    pred = self.model_(batch)
                    loss = loss_fn(pred, batch.y.view(-1))
                    loss.backward()
                    if float(self.gradient_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_.parameters(),
                            max_norm=float(self.gradient_clip_norm),
                        )
                    optimizer.step()
                    batch_size = int(batch.y.view(-1).shape[0])
                    total_loss += float(loss.detach().cpu().item()) * batch_size
                    total_count += batch_size

                train_loss = total_loss / max(1, total_count)
                self.train_loss_history.append(train_loss)

                val_loss = train_loss
                if val_loader is not None:
                    self.model_.eval()
                    val_total = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            _raise_if_cancelled()
                            batch = batch.to(device)
                            pred = self.model_(batch)
                            loss = loss_fn(pred, batch.y.view(-1))
                            batch_size = int(batch.y.view(-1).shape[0])
                            val_total += float(loss.detach().cpu().item()) * batch_size
                            val_count += batch_size
                    val_loss = val_total / max(1, val_count)
                    self.test_loss_history.append(val_loss)

                scheduler.step(val_loss)

                tracked = val_loss if val_loader is not None else train_loss
                if tracked < best_score - 1e-8:
                    best_score = tracked
                    best_state = copy.deepcopy(self.model_.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if int(self.verbose) > 0:
                    message = (
                        f"[GNN+Transformer] epoch={epoch + 1}/{self.epochs} "
                        f"train_loss={train_loss:.6f}"
                    )
                    if val_loader is not None:
                        message += f" val_loss={val_loss:.6f}"
                    print(message)

                if stale_epochs >= int(self.patience):
                    break

            if best_state is not None:
                self.model_.load_state_dict(best_state)

            _raise_if_cancelled()
            self.model_.eval()
            with torch.no_grad():
                probe, probe_mask = self._prepare_numeric_block_with_mask(df, fit=False)
                probe = torch.from_numpy(probe[: min(len(probe), 2048)]).float().to(device)
                probe_mask = torch.from_numpy(probe_mask[: min(len(probe_mask), 2048)]).float().to(device)
                _, gates = self.model_.tabular_encoder.encode(probe, missing_mask=probe_mask, return_gates=True)
                self.feature_importances_ = gates.mean(dim=0).detach().cpu().numpy()
            return self
        finally:
            if self.model_ is not None:
                try:
                    self.model_.to(torch.device("cpu"))
                except Exception:
                    pass
            self.device_ = torch.device("cpu")
            _clear_cuda_memory()

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("model is not fitted")
        df = self._extract_frame(X)
        graphs, valid_idx = self._prepare_dataset(df, None, fit_numeric=False)
        preds = np.full(len(df), np.nan, dtype=float)
        if not graphs:
            return preds

        loader = DataLoader(
            graphs,
            batch_size=int(max(1, self.batch_size)),
            shuffle=False,
            num_workers=int(max(0, self.num_workers)),
        )
        device = self.device_ or _resolve_device(self.device)
        self.model_.to(device)
        self.model_.eval()
        out_vals: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = self.model_(batch).view(-1)
                out_vals.extend(pred.detach().cpu().numpy().tolist())
        for i, idx in enumerate(valid_idx):
            if i < len(out_vals):
                preds[idx] = float(out_vals[i])
        self.model_.to(torch.device("cpu"))
        self.device_ = torch.device("cpu")
        return preds

    def get_training_history(self) -> dict:
        history = {"loss": list(self.train_loss_history)}
        if self.train_loss_history:
            history["train_mse"] = list(self.train_loss_history)
        if self.test_loss_history:
            history["val_loss"] = list(self.test_loss_history)
            history["test_mse"] = list(self.test_loss_history)
        return history
