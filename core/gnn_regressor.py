# -*- coding: utf-8 -*-
"""PyG-based graph regressors for SMILES -> property modeling."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

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
    TORCH_GEOMETRIC_AVAILABLE = False
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

try:
    from torch_geometric.nn.models import AttentiveFP
    ATTENTIVEFP_AVAILABLE = True
except Exception:
    AttentiveFP = None
    ATTENTIVEFP_AVAILABLE = False

try:
    from torch_geometric.nn.models import DMPNN
    DMPNN_AVAILABLE = True
except Exception:
    DMPNN = None
    DMPNN_AVAILABLE = False

try:
    from .graph_utils import (
        smiles_to_pyg_graph,
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        RDKIT_AVAILABLE as GRAPH_RDKIT_AVAILABLE,
    )
    GRAPH_UTILS_AVAILABLE = True
except Exception:
    smiles_to_pyg_graph = None
    ATOM_FEATURE_DIM = 0
    BOND_FEATURE_DIM = 0
    GRAPH_RDKIT_AVAILABLE = False
    GRAPH_UTILS_AVAILABLE = False

PYG_AVAILABLE = bool(TORCH_GEOMETRIC_AVAILABLE and GRAPH_UTILS_AVAILABLE and GRAPH_RDKIT_AVAILABLE and torch is not None)


def _resolve_device(device: str):
    if torch is None:
        return None
    device = str(device or "auto").lower()
    if device in {"cpu"}:
        return torch.device("cpu")
    if device in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: Optional[int]):
    if torch is None:
        return
    if seed is None:
        return
    try:
        seed = int(seed)
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class _SimpleGNNRegressor(nn.Module):
    def __init__(
        self,
        model_type: str,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        gat_heads: int = 4,
    ):
        super().__init__()
        self.model_type = str(model_type or "gcn").lower()
        self.dropout = float(dropout)
        self.pooling = str(pooling or "mean").lower()
        self.use_edge_attr = self.model_type in {"mpnn"}

        layers = []
        in_channels = int(in_dim)

        if self.model_type == "gcn":
            conv_cls = GCNConv
            for _ in range(int(num_layers)):
                layers.append(conv_cls(in_channels, hidden_dim))
                in_channels = hidden_dim
        elif self.model_type == "gat":
            heads = max(1, int(gat_heads))
            head_dim = max(1, int(hidden_dim) // heads)
            for _ in range(int(num_layers)):
                layers.append(GATConv(in_channels, head_dim, heads=heads, dropout=self.dropout))
                in_channels = head_dim * heads
        elif self.model_type == "gin":
            for _ in range(int(num_layers)):
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                layers.append(GINConv(mlp))
                in_channels = hidden_dim
        elif self.model_type == "graphsage":
            for _ in range(int(num_layers)):
                layers.append(SAGEConv(in_channels, hidden_dim))
                in_channels = hidden_dim
        elif self.model_type == "mpnn":
            for _ in range(int(num_layers)):
                out_size = hidden_dim * in_channels
                bottleneck = max(128, min(hidden_dim, out_size // 4))
                edge_nn = nn.Sequential(
                    nn.Linear(edge_dim, bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, out_size),
                )
                layers.append(NNConv(in_channels, hidden_dim, edge_nn, aggr="mean"))
                in_channels = hidden_dim
        else:
            raise ValueError(f"unsupported gnn model_type: {model_type}")

        self.convs = nn.ModuleList(layers)
        self.readout = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),
        )

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
            pooled = global_add_pool(x, batch)
        else:
            pooled = global_mean_pool(x, batch)
        out = self.readout(pooled)
        return out.view(-1)


class _AttentiveFPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_timesteps: int,
    ):
        super().__init__()
        self.backbone = AttentiveFP(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_layers=int(num_layers),
            num_timesteps=int(num_timesteps),
            dropout=float(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(x).view(-1)


class _DMPNNRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = DMPNN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(x).view(-1)


class PyGGraphRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-like wrapper for PyG GNN regressors."""

    def __init__(
        self,
        model_type: str = "gcn",
        smiles_col: str = "smiles",
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 80,
        batch_size: int = 32,
        num_workers: int = 0,
        gat_heads: int = 4,
        num_timesteps: int = 2,
        device: str = "auto",
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ):
        self.model_type = model_type
        self.smiles_col = smiles_col
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gat_heads = gat_heads
        self.num_timesteps = num_timesteps
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.device_ = None

    def _build_model(self):
        model_type = str(self.model_type or "gcn").lower()
        if model_type == "attentivefp":
            if not ATTENTIVEFP_AVAILABLE:
                raise ImportError("torch_geometric AttentiveFP not available")
            return _AttentiveFPRegressor(
                in_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                hidden_dim=int(self.hidden_dim),
                num_layers=int(self.num_layers),
                dropout=float(self.dropout),
                num_timesteps=int(self.num_timesteps),
            )
        if model_type == "dmpnn":
            if not DMPNN_AVAILABLE:
                raise ImportError("torch_geometric DMPNN not available")
            return _DMPNNRegressor(
                in_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                hidden_dim=int(self.hidden_dim),
                num_layers=int(self.num_layers),
                dropout=float(self.dropout),
            )
        return _SimpleGNNRegressor(
            model_type=model_type,
            in_dim=ATOM_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
            pooling=str(self.pooling),
            gat_heads=int(self.gat_heads),
        )

    def _prepare_graphs(self, smiles_list: List[str], y_vals: Optional[np.ndarray] = None):
        graphs = []
        valid_idx = []
        for idx, smi in enumerate(smiles_list):
            data = smiles_to_pyg_graph(smi, add_hs=True) if smiles_to_pyg_graph else None
            if data is None:
                continue
            if y_vals is not None:
                data.y = torch.tensor([float(y_vals[idx])], dtype=torch.float32)
            graphs.append(data)
            valid_idx.append(idx)
        return graphs, valid_idx

    def _extract_smiles(self, X) -> List[str]:
        if isinstance(X, (list, tuple, np.ndarray)):
            return [str(x) for x in X]
        if hasattr(X, "columns"):
            col = str(self.smiles_col)
            if col not in X.columns:
                raise ValueError(f"smiles_col '{col}' not found in input columns")
            return X[col].astype(str).tolist()
        raise ValueError("input X must be list-like or DataFrame with smiles_col")

    def fit(self, X, y):
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric + RDKit required for graph models")

        smiles_list = self._extract_smiles(X)
        y_arr = np.asarray(y).ravel()
        mask = np.isfinite(y_arr)
        if not mask.all():
            y_arr = y_arr[mask]
            smiles_list = [s for i, s in enumerate(smiles_list) if mask[i]]

        if len(smiles_list) == 0:
            raise ValueError("no valid samples for graph training")

        _set_seed(self.random_state)
        device = _resolve_device(self.device)
        self.device_ = device

        graphs, valid_idx = self._prepare_graphs(smiles_list, y_arr)
        if not graphs:
            raise ValueError("all SMILES failed to parse into graphs")

        loader = DataLoader(
            graphs,
            batch_size=int(max(1, self.batch_size)),
            shuffle=True,
            num_workers=int(max(0, self.num_workers)),
        )

        model = self._build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(int(max(1, self.epochs))):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = loss_fn(pred, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if int(self.verbose) > 0 and (epoch % int(self.verbose) == 0):
                avg_loss = total_loss / max(1, len(loader))
                print(f"[PyG] epoch={epoch} loss={avg_loss:.4f}")

        self.model_ = model
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("model is not fitted")
        smiles_list = self._extract_smiles(X)
        graphs, valid_idx = self._prepare_graphs(smiles_list, None)

        preds = np.full(len(smiles_list), np.nan, dtype=float)
        if not graphs:
            return preds

        loader = DataLoader(
            graphs,
            batch_size=int(max(1, self.batch_size)),
            shuffle=False,
            num_workers=int(max(0, self.num_workers)),
        )
        device = self.device_ or _resolve_device(self.device)
        self.model_.eval()
        out_vals = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = self.model_(batch).view(-1)
                out_vals.extend(pred.detach().cpu().numpy().tolist())
        for i, idx in enumerate(valid_idx):
            if i < len(out_vals):
                preds[idx] = float(out_vals[i])
        return preds
