# -*- coding: utf-8 -*-
"""图神经网络工具模块"""

import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Unknown']


def get_atom_features(atom):
    """提取原子特征"""
    features = []
    symbol = atom.GetSymbol()
    symbol_vec = [0] * len(ATOM_SYMBOLS)
    if symbol in ATOM_SYMBOLS:
        symbol_vec[ATOM_SYMBOLS.index(symbol)] = 1
    else:
        symbol_vec[-1] = 1
    features.extend(symbol_vec)

    features.append(atom.GetAtomicNum())
    features.append(atom.GetDegree())
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    features.append(atom.GetTotalValence())
    return features


def get_bond_features(bond):
    """提取键特征"""
    features = []
    bt = bond.GetBondType()
    features.append(1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0)
    features.append(1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0)
    features.append(int(bond.IsInRing()))
    features.append(int(bond.GetIsConjugated()))
    return features


def smiles_to_pyg_graph(smiles):
    """将SMILES转换为PyG图"""
    if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
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
            edge_attr = torch.empty((0, 6), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except:
        return None


def validate_smiles(smiles_string):
    """验证SMILES字符串"""
    if not RDKIT_AVAILABLE:
        return False
    try:
        if not isinstance(smiles_string, str):
            return False
        mol = Chem.MolFromSmiles(smiles_string)
        return mol is not None
    except:
        return False


if TORCH_GEOMETRIC_AVAILABLE:
    class TsubakiInteractionLayer(MessagePassing):
        """Tsubaki交互层"""

        def __init__(self, in_dim, edge_dim, out_dim):
            super().__init__(aggr='add')
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
            if edge_attr is None:
                edge_attr = torch.zeros((x_j.size(0), 8), device=x_j.device)
            return self.message_mlp(torch.cat([x_j, edge_attr], dim=1))

        def update(self, aggr_out, x):
            return self.update_mlp(torch.cat([x, aggr_out], dim=1))


    class MolecularGNN3D(nn.Module):
        """分子GNN模型"""

        def __init__(self, node_dim=29, edge_dim=8, hidden_dim=64, output_dim=128, num_layers=3):
            super().__init__()
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
                h = h + layer(h, edge_index, edge_attr)
            return self.readout(global_add_pool(h, batch))


class GNNFeaturizer:
    """GNN特征提取器"""

    def __init__(self, model=None):
        self.model = model
        if TORCH_GEOMETRIC_AVAILABLE and model is None:
            self.model = MolecularGNN3D()

    def featurize(self, smiles_list):
        """批量提取特征"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装torch_geometric")

        features, valid_indices = [], []
        for idx, smiles in enumerate(tqdm(smiles_list, desc="GNN特征提取")):
            graph = smiles_to_pyg_graph(smiles)
            if graph is None:
                continue
            try:
                self.model.eval()
                with torch.no_grad():
                    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                    feat = self.model(graph).numpy().flatten()
                features.append(feat)
                valid_indices.append(idx)
            except:
                continue

        return np.array(features) if features else np.array([]), valid_indices
