# -*- coding: utf-8 -*-
"""图神经网络工具模块"""

# ============================================
# 重要：必须在导入 RDKit 之前导入线程配置！
# ============================================
from . import thread_config

import numpy as np
import os
import random
from collections import OrderedDict
from tqdm import tqdm

# 导入 GPU 自动恢复模块
try:
    from .gpu_auto_recovery import auto_recovery_wrapper, GPUAutoRecoveryContext, force_cleanup_all_gpus
    GPU_AUTO_RECOVERY_AVAILABLE = True
except ImportError:
    GPU_AUTO_RECOVERY_AVAILABLE = False
    auto_recovery_wrapper = None
    GPUAutoRecoveryContext = None
    force_cleanup_all_gpus = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        MessagePassing,
        global_add_pool,
        global_mean_pool,
        GATConv,
        GCNConv,
        GINConv,
        SAGEConv,
        NNConv,
    )
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

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
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .smiles_utils import normalize_chemical_string
from .bigsmiles_stochastic_graph import looks_like_bigsmiles, sample_bigsmiles_realizations
# [新增] 支持 SMILES / SELFIES / BigSMILES 输入

# 1. 扩展原子列表 (可选，覆盖更多元素)
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',
    'Fe', 'Zn', 'Cu', 'Mn', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Se', 'Li',
    'Unknown'
]

if RDKIT_AVAILABLE:
    HYBRIDIZATION_LIST = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
    CHIRALITY_LIST = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    BOND_STEREO_LIST = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
else:
    HYBRIDIZATION_LIST = [0, 1, 2, 3, 4, 5]
    CHIRALITY_LIST = [0, 1, 2, 3]
    BOND_STEREO_LIST = [0, 1, 2, 3, 4, 5]

ATOM_FEATURE_DIM = len(ATOM_SYMBOLS) + 8 + len(HYBRIDIZATION_LIST) + len(CHIRALITY_LIST)
BOND_FEATURE_DIM = 4 + 3 + len(BOND_STEREO_LIST)


def get_atom_features(atom):
    """提取原子特征 (扩展维度 + 归一化)"""
    features = []
    # One-hot 原子符号
    symbol = atom.GetSymbol()
    symbol_vec = [0] * len(ATOM_SYMBOLS)
    if symbol in ATOM_SYMBOLS:
        symbol_vec[ATOM_SYMBOLS.index(symbol)] = 1
    else:
        symbol_vec[-1] = 1
    features.extend(symbol_vec)

    # 归一化数值特征
    features.append(float(atom.GetAtomicNum()) / 100.0)
    features.append(float(atom.GetDegree()) / 10.0)
    features.append(float(atom.GetTotalNumHs()) / 8.0)
    features.append(float(atom.GetTotalValence()) / 10.0)
    features.append(float(atom.GetFormalCharge()) / 5.0)
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    features.append(float(atom.GetMass()) / 200.0)

    # Hybridization one-hot
    hyb = atom.GetHybridization()
    for h in HYBRIDIZATION_LIST:
        features.append(1.0 if hyb == h else 0.0)

    # Chirality one-hot
    ch = atom.GetChiralTag()
    for c in CHIRALITY_LIST:
        features.append(1.0 if ch == c else 0.0)

    # 当前总维度: len(ATOM_SYMBOLS) + 8 + len(HYBRIDIZATION_LIST) + len(CHIRALITY_LIST)
    return features


def get_bond_features(bond):
    """提取键特征 (扩展维度)"""
    features = []
    bt = bond.GetBondType()
    features.append(1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0)
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    features.append(float(bond.GetBondTypeAsDouble()) / 3.0)

    stereo = bond.GetStereo()
    for s in BOND_STEREO_LIST:
        features.append(1.0 if stereo == s else 0.0)

    # 总维度: 4 + 3 + len(BOND_STEREO_LIST)
    return features


def smiles_to_pyg_graph(smiles, add_hs: bool = True):
    """将SMILES转换为PyG图"""
    if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        return None

    try:
        if smiles is None:
            return None
        s = str(smiles).strip()
        if not s or s.lower() in {"nan", "none", "<na>"}:
            return None
        mol = Chem.MolFromSmiles(normalize_chemical_string(s, canonicalize=False, repair=True, keep_largest_frag=False) or "")
        if mol is None:
            return None
        if add_hs:
            mol = Chem.AddHs(mol)

        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float32)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
            bf = get_bond_features(bond)
            edge_attrs += [bf, bf]

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


def validate_smiles(smiles_string):
    """验证SMILES字符串"""
    if not RDKIT_AVAILABLE:
        return False
    try:
        if not isinstance(smiles_string, str):
            return False
        mol = Chem.MolFromSmiles(normalize_chemical_string(smiles_string, canonicalize=False, repair=True, keep_largest_frag=False) or "")
        return mol is not None
    except:
        return False


def _normalize_gnn_model_type(model_type):
    mt = str(model_type or "gnn3d").strip().lower()
    mt = mt.replace(" ", "").replace("_", "").replace("-", "")
    if mt in {"gnn3d", "moleculargnn3d"}:
        return "gnn3d"
    if mt in {"gcn"}:
        return "gcn"
    if mt in {"gat"}:
        return "gat"
    if mt in {"gin"}:
        return "gin"
    if mt in {"graphsage"}:
        return "graphsage"
    if mt in {"mpnn"}:
        return "mpnn"
    if mt in {"attentivefp"}:
        return "attentivefp"
    if mt in {"dmpnn"}:
        return "dmpnn"
    return mt


if TORCH_GEOMETRIC_AVAILABLE:
    class TsubakiInteractionLayer(MessagePassing):
        """Tsubaki交互层"""

        def __init__(self, in_dim, edge_dim, out_dim):
            super().__init__(aggr='add')
            self.edge_dim = edge_dim # 记录 edge_dim
            self.message_mlp = nn.Sequential(
                nn.Linear(in_dim + edge_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
            self.update_mlp = nn.Sequential(
                nn.Linear(in_dim + out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_j, edge_attr):
            # 修复: 确保 edge_attr 存在且维度正确
            if edge_attr is None:
                # 使用 self.edge_dim 而不是硬编码的 8
                edge_attr = torch.zeros((x_j.size(0), self.edge_dim), device=x_j.device)
            return self.message_mlp(torch.cat([x_j, edge_attr], dim=1))

        def update(self, aggr_out, x):
            return self.update_mlp(torch.cat([x, aggr_out], dim=1))


    class MolecularGNN3D(nn.Module):
        """分子GNN模型（2D图 + 可选池化）"""
        def __init__(self, node_dim=None, edge_dim=None, hidden_dim=64, output_dim=128, num_layers=3, pooling="sum", use_checkpoint=False):
            super().__init__()
            node_dim = int(node_dim or ATOM_FEATURE_DIM)
            edge_dim = int(edge_dim or BOND_FEATURE_DIM)
            self.pooling = str(pooling or "sum").lower()
            self.use_checkpoint = use_checkpoint
            self.embedding = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList([
                TsubakiInteractionLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            h = self.embedding(x)
            for layer in self.layers:
                # 使用 gradient checkpointing 减少显存占用
                if self.use_checkpoint and self.training:
                    h = h + torch.utils.checkpoint.checkpoint(layer, h, edge_index, edge_attr, use_reentrant=False)
                else:
                    h = h + layer(h, edge_index, edge_attr)
            if self.pooling == "mean":
                pooled = global_mean_pool(h, batch)
            else:
                pooled = global_add_pool(h, batch)
            return self.readout(pooled)


    class _SimpleGNNEncoder(nn.Module):
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
            output_dim: int = None,
        ):
            super().__init__()
            self.model_type = str(model_type or "gcn").lower()
            self.dropout = float(dropout)
            self.pooling = str(pooling or "mean").lower()
            self.use_edge_attr = self.model_type in {"mpnn"}
            self.output_dim = int(output_dim or hidden_dim)

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
                nn.Linear(hidden_dim, self.output_dim),
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
            return self.readout(pooled)


    class _AttentiveFPEncoder(nn.Module):
        def __init__(
            self,
            in_dim: int,
            edge_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            num_timesteps: int,
            output_dim: int = None,
        ):
            super().__init__()
            if AttentiveFP is None:
                raise ImportError("torch_geometric AttentiveFP not available")
            self.backbone = AttentiveFP(
                in_channels=in_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim,
                num_layers=int(num_layers),
                num_timesteps=int(num_timesteps),
                dropout=float(dropout),
            )
            out_dim = int(output_dim or hidden_dim)
            self.proj = nn.Identity() if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim)

        def forward(self, data):
            x = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.proj(x)


    class _DMPNNEncoder(nn.Module):
        def __init__(
            self,
            in_dim: int,
            edge_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            output_dim: int = None,
        ):
            super().__init__()
            if DMPNN is None:
                raise ImportError("torch_geometric DMPNN not available")
            self.backbone = DMPNN(
                in_channels=in_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim,
                num_layers=int(num_layers),
                dropout=float(dropout),
            )
            out_dim = int(output_dim or hidden_dim)
            self.proj = nn.Identity() if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim)

        def forward(self, data):
            x = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.proj(x)


class GNNFeaturizer:
    """GNN特征提取器 - 批量推理优化版"""

    def __init__(
        self,
        model=None,
        model_type: str = None,
        device=None,
        seed: int = 42,
        deterministic: bool = True,
        add_hs: bool = True,
        pooling: str = "sum",
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
        gat_heads: int = None,
        num_timesteps: int = None,
        output_dim: int = None,
        cache_graphs: bool = True,
        max_cache_size: int = 5000,
        num_workers: int = 0,
        pin_memory: bool = None,
        chunk_size: int = 0,
        model_state_path: str = None,
        bigsmiles_mode: str = "auto",
        bigsmiles_num_samples: int = 4,
        bigsmiles_min_repeat_units: int = 2,
        bigsmiles_max_repeat_units: int = 6,
        bigsmiles_seed: int = 42,
        bigsmiles_cache_size: int = 512,
    ):
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装torch_geometric")

        self.add_hs = bool(add_hs)
        self.model_type = model_type
        self.pooling = str(pooling or "sum")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gat_heads = gat_heads
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.cache_graphs = bool(cache_graphs)
        self.max_cache_size = int(max_cache_size) if max_cache_size is not None else 0
        self.num_workers = int(num_workers or 0)
        self.chunk_size = int(chunk_size or 0)
        self.pin_memory = pin_memory
        self.bigsmiles_mode = str(bigsmiles_mode or "auto").strip().lower()
        self.bigsmiles_num_samples = max(1, int(bigsmiles_num_samples or 1))
        self.bigsmiles_min_repeat_units = max(1, int(bigsmiles_min_repeat_units or 1))
        self.bigsmiles_max_repeat_units = max(
            self.bigsmiles_min_repeat_units,
            int(bigsmiles_max_repeat_units or self.bigsmiles_min_repeat_units),
        )
        self.bigsmiles_seed = int(bigsmiles_seed if bigsmiles_seed is not None else (seed if seed is not None else 42))
        self.bigsmiles_cache_size = int(bigsmiles_cache_size or 0)

        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.model = model or self._build_default_model()
        if model_state_path:
            try:
                if os.path.isfile(model_state_path):
                    state = torch.load(model_state_path, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state.get("state_dict")
                    if isinstance(state, dict):
                        self.model.load_state_dict(state, strict=False)
            except Exception:
                pass

        # 设备与 eval 模式
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # 尝试将模型移到 GPU，显存不足时报错而不是自动降级
        try:
            # [新增] 在移动模型前先检查显存
            if torch.cuda.is_available() and self.device.type == 'cuda':
                device_id = self.device.index if self.device.index is not None else 0
                props = torch.cuda.get_device_properties(device_id)
                total_mem = props.total_memory / 1024**3
                allocated_mem = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved_mem = torch.cuda.memory_reserved(device_id) / 1024**3
                free_mem = total_mem - allocated_mem

                # 估算模型大小（粗略估计）
                model_params = sum(p.numel() for p in self.model.parameters())
                estimated_model_size_gb = model_params * 4 / 1024**3  # 4 bytes per float32

                print(f"📊 GPU {device_id} 显存状态:")
                print(f"   总显存: {total_mem:.2f} GB")
                print(f"   已分配: {allocated_mem:.2f} GB")
                print(f"   已保留: {reserved_mem:.2f} GB")
                print(f"   可用: {free_mem:.2f} GB")
                print(f"   模型大小(估算): {estimated_model_size_gb:.2f} GB")

                # 如果可用显存不足模型大小的 3 倍，给出警告
                if free_mem < estimated_model_size_gb * 3:
                    print(f"⚠️ 显存可能不足，建议:")
                    print(f"   1. 运行 'python force_clear_gpu.py' 清理显存")
                    print(f"   2. 降低 Batch Size (推荐 4-8)")
                    print(f"   3. 降低 Chunk Size (推荐 128-256)")
                    print(f"   4. 关闭其他占用 GPU 的程序")

            # [自动恢复] 使用上下文管理器保护模型加载
            if GPU_AUTO_RECOVERY_AVAILABLE:
                with GPUAutoRecoveryContext(cleanup_on_exit=False, cleanup_on_error=True, verbose=False):
                    self.model.to(self.device)
            else:
                self.model.to(self.device)

            # [新增] 检查可用显存，如果不足 4GB 则警告
            if torch.cuda.is_available() and self.device.type == 'cuda':
                device_id = self.device.index if self.device.index is not None else 0
                free_mem = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                free_mem_gb = free_mem / 1024**3
                if free_mem_gb < 4:
                    print(f"⚠️ GPU {device_id} 可用显存较低 ({free_mem_gb:.1f} GB)，建议:")
                    print("   1. 降低 Batch Size (推荐 4-8)")
                    print("   2. 降低 Chunk Size (推荐 128-256)")
                    print("   3. 减少 DataLoader Workers (推荐 0)")
                    print("   4. 运行 force_clear_gpu.py 清理显存")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 不自动降级，而是给出明确的错误信息
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"❌ GPU 显存不足，无法初始化 GNN 模型。\n"
                    f"错误: {e}\n\n"
                    f"解决方案:\n"
                    f"  1. 运行 'python force_clear_gpu.py' 清理显存\n"
                    f"  2. 降低 Batch Size (当前建议: 4-8)\n"
                    f"  3. 降低 Chunk Size (当前建议: 128-256)\n"
                    f"  4. 减少 DataLoader Workers (建议设为 0)\n"
                    f"  5. 关闭其他占用 GPU 的程序\n"
                    f"  6. 如需使用 CPU 模式，请在 UI 中选择 'CPU' 设备"
                ) from e
            else:
                raise

        self.model.eval()

        if self.pin_memory is None:
            self.pin_memory = bool(getattr(self.device, "type", "") == "cuda")

        self._graph_cache = OrderedDict()
        self._bigsmiles_realization_cache = OrderedDict()

    def _build_default_model(self):
        model_type = _normalize_gnn_model_type(self.model_type)
        if model_type in {"gnn3d"}:
            kwargs = {"pooling": self.pooling}
            # 降低默认 hidden_dim 以减少显存占用
            if self.hidden_dim is not None:
                kwargs["hidden_dim"] = int(self.hidden_dim)
            else:
                kwargs["hidden_dim"] = 32  # 从 64 降低到 32
            if self.output_dim is not None:
                kwargs["output_dim"] = int(self.output_dim)
            else:
                kwargs["output_dim"] = 64  # 从 128 降低到 64
            if self.num_layers is not None:
                kwargs["num_layers"] = int(self.num_layers)
            else:
                kwargs["num_layers"] = 2  # 从 3 降低到 2
            return MolecularGNN3D(**kwargs)

        # 降低默认参数以减少显存占用
        hidden_dim = int(self.hidden_dim) if self.hidden_dim is not None else 64  # 从 128 降低到 64
        num_layers = int(self.num_layers) if self.num_layers is not None else 2  # 从 3 降低到 2
        dropout = float(self.dropout) if self.dropout is not None else 0.1
        output_dim = int(self.output_dim) if self.output_dim is not None else hidden_dim
        gat_heads = int(self.gat_heads) if self.gat_heads is not None else 4
        num_timesteps = int(self.num_timesteps) if self.num_timesteps is not None else 2
        pooling = "add" if self.pooling in {"sum", "add"} else "mean"

        if model_type in {"gcn", "gat", "gin", "graphsage", "mpnn"}:
            return _SimpleGNNEncoder(
                model_type=model_type,
                in_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                gat_heads=gat_heads,
                output_dim=output_dim,
            )
        if model_type == "attentivefp":
            return _AttentiveFPEncoder(
                in_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_timesteps=num_timesteps,
                output_dim=output_dim,
            )
        if model_type == "dmpnn":
            if DMPNN is None:
                return _SimpleGNNEncoder(
                    model_type="mpnn",
                    in_dim=ATOM_FEATURE_DIM,
                    edge_dim=BOND_FEATURE_DIM,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    pooling=pooling,
                    gat_heads=gat_heads,
                    output_dim=output_dim,
                )
            return _DMPNNEncoder(
                in_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=output_dim,
            )
        raise ValueError(f"unsupported gnn model_type: {model_type}")

    def _cache_get(self, smiles: str):
        if not self.cache_graphs:
            return None
        if smiles in self._graph_cache:
            self._graph_cache.move_to_end(smiles)
            return self._graph_cache[smiles]
        return None

    def _cache_put(self, smiles: str, graph):
        if not self.cache_graphs or self.max_cache_size <= 0:
            return
        self._graph_cache[smiles] = graph
        self._graph_cache.move_to_end(smiles)
        while len(self._graph_cache) > self.max_cache_size:
            self._graph_cache.popitem(last=False)

    def _bigsmiles_cache_get(self, smiles: str):
        if self.bigsmiles_cache_size <= 0:
            return None
        if smiles in self._bigsmiles_realization_cache:
            self._bigsmiles_realization_cache.move_to_end(smiles)
            return self._bigsmiles_realization_cache[smiles]
        return None

    def _bigsmiles_cache_put(self, smiles: str, realizations):
        if self.bigsmiles_cache_size <= 0:
            return
        self._bigsmiles_realization_cache[smiles] = realizations
        self._bigsmiles_realization_cache.move_to_end(smiles)
        while len(self._bigsmiles_realization_cache) > self.bigsmiles_cache_size:
            self._bigsmiles_realization_cache.popitem(last=False)

    def _sample_bigsmiles_smiles(self, smiles: str):
        cached = self._bigsmiles_cache_get(smiles)
        if cached is not None:
            return cached
        realizations = sample_bigsmiles_realizations(
            smiles,
            n_samples=self.bigsmiles_num_samples,
            min_repeat_units=self.bigsmiles_min_repeat_units,
            max_repeat_units=self.bigsmiles_max_repeat_units,
            random_state=self.bigsmiles_seed,
        )
        if realizations:
            self._bigsmiles_cache_put(smiles, realizations)
        return realizations

    def _build_graph(self, smiles: str):
        graph = self._cache_get(smiles)
        if graph is not None:
            return graph
        graph = smiles_to_pyg_graph(smiles, add_hs=self.add_hs)
        if graph is not None:
            self._cache_put(smiles, graph)
        return graph

    def _build_graph_variants(self, smiles: str):
        if smiles is None:
            return []
        s = str(smiles).strip()
        if not s:
            return []

        mode = self.bigsmiles_mode
        if mode in {"off", "disable", "disabled", "none"} or not looks_like_bigsmiles(s):
            graph = self._build_graph(s)
            return [graph] if graph is not None else []

        if mode in {"sample", "single"}:
            n_samples = 1
        elif mode in {"ensemble", "multi", "stochastic"}:
            n_samples = self.bigsmiles_num_samples
        else:
            n_samples = self.bigsmiles_num_samples if self.bigsmiles_num_samples > 1 else 1

        realizations = self._sample_bigsmiles_smiles(s)[:n_samples]
        graphs = []
        for sample_smiles in realizations:
            graph = self._build_graph(sample_smiles)
            if graph is not None:
                graphs.append(graph)
        if graphs:
            return graphs

        graph = self._build_graph(s)
        return [graph] if graph is not None else []

    def featurize(self, smiles_list, batch_size=32, chunk_size=None, num_workers=None, show_progress=True):
        """批量提取特征（支持分块、缓存和多进程加载）- 带自动恢复"""
        # 使用 GPU 自动恢复上下文
        if GPU_AUTO_RECOVERY_AVAILABLE:
            with GPUAutoRecoveryContext(cleanup_on_exit=False, cleanup_on_error=True, verbose=True):
                return self._featurize_impl(smiles_list, batch_size, chunk_size, num_workers, show_progress)
        else:
            return self._featurize_impl(smiles_list, batch_size, chunk_size, num_workers, show_progress)

    def _featurize_impl(self, smiles_list, batch_size=32, chunk_size=None, num_workers=None, show_progress=True):
        """批量提取特征的实际实现"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装torch_geometric")

        if smiles_list is None:
            return np.array([]), []

        if num_workers is None:
            num_workers = self.num_workers
        num_workers = int(max(0, num_workers))

        if chunk_size is None:
            chunk_size = self.chunk_size
        chunk_size = int(chunk_size or 0)
        if chunk_size <= 0:
            chunk_size = max(int(batch_size) * 16, 256)

        all_features = []
        valid_indices = []

        total = len(smiles_list)
        chunk_iter = range(0, total, chunk_size)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="图结构生成(分块)")

        for start in chunk_iter:
            end = min(start + chunk_size, total)
            graph_data_list = []
            chunk_valid = []
            graph_owner_indices = []

            # [优化] 并行构建图结构，提升 GPU 利用率
            if num_workers > 0 and (end - start) > 100:
                # 使用多进程并行构建图
                from concurrent.futures import ThreadPoolExecutor
                import threading

                def build_graph_safe(idx):
                    try:
                        smiles = smiles_list[idx]
                        graphs = self._build_graph_variants(smiles)
                        return (idx, graphs)
                    except:
                        return (idx, [])

                with ThreadPoolExecutor(max_workers=min(num_workers, 8)) as executor:
                    futures = [executor.submit(build_graph_safe, idx) for idx in range(start, end)]
                    for future in futures:
                        idx, graphs = future.result()
                        if graphs:
                            chunk_valid.append(idx)
                            for graph in graphs:
                                graph_data_list.append(graph)
                                graph_owner_indices.append(idx)
            else:
                # 串行构建（小批量或 num_workers=0）
                for idx in range(start, end):
                    smiles = smiles_list[idx]
                    graphs = self._build_graph_variants(smiles)
                    if graphs:
                        chunk_valid.append(idx)
                        for graph in graphs:
                            graph_data_list.append(graph)
                            graph_owner_indices.append(idx)

            if not graph_data_list:
                continue

            # [优化] 配置 DataLoader 以提升 GPU 利用率
            loader_kwargs = {
                'batch_size': int(batch_size),
                'shuffle': False,
                'num_workers': 0,  # 强制使用单进程以减少内存占用
                'pin_memory': False,  # 禁用 pin_memory 以减少内存占用
            }

            # 仅在 num_workers > 0 且显存充足时启用优化
            if num_workers > 0 and torch.cuda.is_available():
                device_id = self.device.index if self.device.index is not None else 0
                free_mem = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
                free_mem_gb = free_mem / 1024**3
                if free_mem_gb > 10:  # 只有在显存充足时才启用多进程
                    loader_kwargs['num_workers'] = min(num_workers, 4)
                    loader_kwargs['pin_memory'] = self.pin_memory
                    loader_kwargs['persistent_workers'] = True
                    loader_kwargs['prefetch_factor'] = 2  # 从 4 降低到 2

            loader = DataLoader(graph_data_list, **loader_kwargs)

            if show_progress:
                loader_iter = tqdm(loader, desc="GNN推理", leave=False)
            else:
                loader_iter = loader

            sample_feature_map = {}
            sample_order = list(dict.fromkeys(chunk_valid))
            graph_offset = 0
            with torch.no_grad():
                for batch in loader_iter:
                    try:
                        batch_n = int(getattr(batch, "num_graphs", 0) or 0)
                        if batch_n <= 0:
                            try:
                                batch_n = int(batch.batch.max().item()) + 1
                            except Exception:
                                batch_n = len(graph_owner_indices[graph_offset:])
                        batch_owner_indices = graph_owner_indices[graph_offset:graph_offset + batch_n]
                        graph_offset += batch_n
                        batch = batch.to(self.device)
                        out = self.model(batch)
                        out_np = out.cpu().numpy()
                        for owner_idx, vec in zip(batch_owner_indices, out_np):
                            sample_feature_map.setdefault(owner_idx, []).append(vec)

                        # [修复] 每个 batch 处理完立即清理 GPU 显存
                        del batch, out
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except RuntimeError as e:
                        # 显存不足时报错，不自动降级
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            raise RuntimeError(
                                f"❌ GPU 显存不足，推理过程中断。\n"
                                f"当前配置: Batch Size={batch_size}, Chunk Size={chunk_size}\n\n"
                                f"解决方案:\n"
                                f"  1. 降低 Batch Size (建议: {batch_size // 2})\n"
                                f"  2. 降低 Chunk Size (建议: {chunk_size // 2})\n"
                                f"  3. 减少 DataLoader Workers\n"
                                f"  4. 运行 'python force_clear_gpu.py' 清理显存\n"
                                f"  5. 如需使用 CPU 模式，请在 UI 中选择 'CPU' 设备"
                            ) from e
                        else:
                            raise

            for owner_idx in sample_order:
                feats = sample_feature_map.get(owner_idx)
                if not feats:
                    continue
                feats_np = np.vstack(feats)
                all_features.append(np.mean(feats_np, axis=0)[None, :])
                valid_indices.append(owner_idx)

            # [修复] 每个 chunk 处理完清理
            del graph_data_list, loader

            # [关键修复] 清理图缓存，防止内存泄漏
            if self.cache_graphs and len(self._graph_cache) > self.max_cache_size:
                # 只保留最近使用的一半
                while len(self._graph_cache) > self.max_cache_size // 2:
                    self._graph_cache.popitem(last=False)

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_features:
            return np.array([]), []

        all_features = np.concatenate(all_features, axis=0)
        return all_features, valid_indices
