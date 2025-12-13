# -*- coding: utf-8 -*-
"""
ML_CFRP 多组分功能升级脚本
1. app.py: 支持多列选择，自动拼接多组分 SMILES。
2. molecular_features.py: 修复 ML 力场对混合物(dot-separated)的处理逻辑。
"""
import os

# ==============================================================================
# 1. 升级 core/molecular_features.py
# ==============================================================================
MOLECULAR_FEATURES_NEW = r'''# -*- coding: utf-8 -*-
"""
分子特征工程模块 - 完整5种提取方法 + 分子指纹 (高性能优化版)
更新：支持多组分混合物 (dot-separated SMILES) 的 3D 力场计算
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import warnings
import torch
import os

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

# =============================================================================
# 辅助函数：3D 构象生成 (支持混合物拆分)
# =============================================================================
def _generate_3d_data_worker(smiles):
    """
    单个分子的3D生成工作函数
    更新：如果输入是混合物(A.B)，则拆分后分别生成3D构象，返回列表
    """
    if not RDKIT_AVAILABLE: return None

    # 拆分混合物
    smiles_str = str(smiles)
    fragments = smiles_str.split('.')

    results = []

    try:
        for frag in fragments:
            clean_smi = frag.replace('*', 'C')
            mol = Chem.MolFromSmiles(clean_smi)
            if mol is None: continue
            mol = Chem.AddHs(mol)

            # 生成 3D
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.numThreads = 1
            if AllChem.EmbedMolecule(mol, params) != 0:
                if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
                    continue # 该片段生成失败，跳过

            # 优化
            is_opt = False
            try:
                if AllChem.MMFFOptimizeMolecule(mol, maxIters=100) == 0: is_opt = True
                elif AllChem.UFFOptimizeMolecule(mol, maxIters=100) == 0: is_opt = True
            except: pass

            if not is_opt: continue

            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            coords = mol.GetConformer().GetPositions()

            # 检查元素支持 (ANI-2x 仅支持 H, C, N, O, S, F, Cl)
            supported = {1, 6, 7, 8, 16, 9, 17}
            if not set(atoms).issubset(supported): continue

            results.append((atoms, coords))

        return results if results else None

    except Exception: return None

# =============================================================================
# 特征提取类
# =============================================================================
class RDKitFeatureExtractor:
    def smiles_to_rdkit_features(self, smiles_list):
        return AdvancedMolecularFeatureExtractor().smiles_to_rdkit_features(smiles_list)

class OptimizedRDKitFeatureExtractor:
    def __init__(self, n_jobs=-1, batch_size=1000):
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.batch_size = batch_size

    @staticmethod
    def _process_batch(args):
        start_idx, smiles_list = args
        if not RDKIT_AVAILABLE: return [], []
        desc_funcs = dict(Descriptors.descList)
        feats, idxs = [], []
        for i, s in enumerate(smiles_list):
            try:
                m = Chem.MolFromSmiles(str(s))
                if m:
                    feats.append({k: v(m) for k, v in desc_funcs.items()})
                    idxs.append(start_idx + i)
            except: continue
        return feats, idxs

    def smiles_to_rdkit_features(self, smiles_list):
        batches = [(i, smiles_list[i:i+self.batch_size]) for i in range(0, len(smiles_list), self.batch_size)]
        all_feats, all_idxs = [], []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for f, i in executor.map(self._process_batch, batches):
                all_feats.extend(f)
                all_idxs.extend(i)
        if not all_feats: return pd.DataFrame(), []
        df = pd.DataFrame(all_feats).select_dtypes(include=[np.number]).dropna(axis=1, how='all').fillna(0)
        return df, all_idxs

class MemoryEfficientRDKitExtractor:
    def __init__(self, batch_size=100): self.batch_size = batch_size
    def smiles_to_rdkit_features(self, smiles_list):
        return AdvancedMolecularFeatureExtractor().smiles_to_rdkit_features(smiles_list)

class AdvancedMolecularFeatureExtractor:
    def __init__(self):
        if not RDKIT_AVAILABLE: raise ImportError("Need RDKit")

    def _process_result(self, features, indices, is_df=False):
        if not features: return pd.DataFrame(), []
        df = features if is_df else pd.DataFrame(features)
        df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all').fillna(df.median())
        df = df.loc[:, ~df.columns.duplicated()]
        return df, indices

    def smiles_to_rdkit_features(self, smiles_list):
        feats, idxs = [], []
        desc_funcs = dict(Descriptors.descList)
        for i, s in enumerate(tqdm(smiles_list, desc="RDKit")):
            try:
                m = Chem.MolFromSmiles(str(s))
                if m:
                    f = {}
                    for n, func in desc_funcs.items():
                        try: val = func(m); f[n] = val if np.isfinite(val) else np.nan
                        except: f[n] = np.nan
                    feats.append(f); idxs.append(i)
            except: continue
        return self._process_result(feats, idxs)

    def smiles_to_mordred(self, smiles_list, batch_size=1000):
        if not MORDRED_AVAILABLE: raise ImportError("Need Mordred")
        mols, idxs = [], []
        for i, s in enumerate(smiles_list):
            m = Chem.MolFromSmiles(str(s))
            if m: mols.append(m); idxs.append(i)

        if not mols: return pd.DataFrame(), []
        calc = Calculator(descriptors, ignore_3D=True)
        n_proc = 1 if os.name == 'nt' else mp.cpu_count()

        dfs = []
        for i in range(0, len(mols), batch_size):
            batch = mols[i:i+batch_size]
            try: dfs.append(calc.pandas(batch, n_proc=n_proc, quiet=True))
            except: dfs.append(calc.pandas(batch, n_proc=1, quiet=True))

        try:
            final = pd.concat(dfs, ignore_index=True).apply(pd.to_numeric, errors='coerce')
            return self._process_result(final, idxs, is_df=True)
        except: return pd.DataFrame(), []

    def smiles_to_graph_features(self, smiles_list):
        feats, idxs = [], []
        for i, s in enumerate(tqdm(smiles_list, desc="Graph")):
            try:
                m = Chem.MolFromSmiles(str(s))
                if not m: continue
                na = m.GetNumAtoms(); nb = m.GetNumBonds()
                feats.append({
                    'Nodes': na, 'Edges': nb, 'Density': nb/(na*(na-1)/2) if na>1 else 0,
                    'Rings': Chem.GetSSSR(m).__len__(), 'MolWt': Descriptors.MolWt(m),
                    'LogP': Descriptors.MolLogP(m), 'TPSA': Descriptors.TPSA(m)
                })
                idxs.append(i)
            except: continue
        return self._process_result(feats, idxs)

class MLForceFieldExtractor:
    """ML力场提取器 (支持混合物)"""
    def __init__(self, device=None):
        try:
            import torchani, torch
            self.torch = torch; self.torchani = torchani
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torchani.models.ANI2x().to(self.device)
            self.AVAILABLE = True
        except: self.AVAILABLE = False
        self.feature_names = ['ani_energy', 'ani_energy_per_atom', 'ani_max_force', 'ani_mean_force']

    def smiles_to_ani_features(self, smiles_list, batch_size=32):
        if not self.AVAILABLE: raise ImportError("Need TorchANI")

        # 1. 并行生成3D (每个结果可能包含多个片段)
        valid_idxs, data_list = [], []
        with ProcessPoolExecutor() as exe:
            # 结果形式: [[(atoms1, coords1), (atoms2, coords2)], ...]
            res = list(tqdm(exe.map(_generate_3d_data_worker, smiles_list), total=len(smiles_list), desc="3D生成"))

        for i, r in enumerate(res):
            if r: valid_idxs.append(i); data_list.append(r)

        if not data_list: return pd.DataFrame(), []

        feats_all = []

        # 为了批量推理，我们需要把 list of lists 展平，同时记录归属
        flat_data = []
        sample_map = [] # 记录 flat_data 中的第j个分子属于第几个样本

        for sample_idx, fragments in enumerate(data_list):
            for frag in fragments:
                flat_data.append(frag)
                sample_map.append(sample_idx)

        # 批量推理
        flat_results = []
        for i in tqdm(range(0, len(flat_data), batch_size), desc="ANI推理"):
            batch = flat_data[i:i+batch_size]
            try:
                species = [self.torch.tensor(d[0], dtype=self.torch.long) for d in batch]
                coords = [self.torch.tensor(d[1], dtype=self.torch.float32) for d in batch]

                s_pad = self.torch.nn.utils.rnn.pad_sequence(species, batch_first=True, padding_value=-1).to(self.device)
                c_pad = self.torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=0).to(self.device)
                c_pad.requires_grad_(True)
                mask = (s_pad >= 0)

                s_safe = s_pad.clone(); s_safe[~mask] = 0
                energy = self.model((s_safe, c_pad)).energies
                forces = -self.torch.autograd.grad(energy.sum(), c_pad, create_graph=False, retain_graph=False)[0]

                _, atomic_E = self.model((s_safe, c_pad))
                real_E = (atomic_E * mask.float()).sum(dim=1).detach().cpu().numpy()
                forces_np = forces.detach().cpu().numpy()

                for j in range(len(batch)):
                    n = len(batch[j][0])
                    f_vec = forces_np[j][:n]
                    f_norm = np.linalg.norm(f_vec, axis=1)
                    flat_results.append({
                        'E': real_E[j], 
                        'atoms': n,
                        'max_F': np.max(f_norm), 
                        'mean_F': np.mean(f_norm)
                    })
            except:
                for _ in batch: flat_results.append(None)

        # 聚合结果
        # data_list 对应 valid_idxs
        current_flat_idx = 0
        for sample_frags in data_list:
            n_frags = len(sample_frags)
            frag_res = flat_results[current_flat_idx : current_flat_idx + n_frags]
            current_flat_idx += n_frags

            # 过滤失败的片段
            frag_res = [r for r in frag_res if r is not None]

            if not frag_res:
                feats_all.append({k: np.nan for k in self.feature_names})
                continue

            # 聚合逻辑：能量求和，力取最大/平均
            total_E = sum(r['E'] for r in frag_res)
            total_atoms = sum(r['atoms'] for r in frag_res)
            global_max_F = max(r['max_F'] for r in frag_res)
            # 加权平均力
            global_mean_F = sum(r['mean_F'] * r['atoms'] for r in frag_res) / total_atoms

            feats_all.append({
                'ani_energy': total_E,
                'ani_energy_per_atom': total_E / total_atoms,
                'ani_max_force': global_max_F,
                'ani_mean_force': global_mean_F
            })

        return pd.DataFrame(feats_all), valid_idxs

class EpoxyDomainFeatureExtractor:
    def __init__(self): 
        if not RDKIT_AVAILABLE: raise ImportError("Need RDKit")

    def extract_features(self, resin, hardener, phr=None):
        feats, idxs = [], []
        if len(resin) != len(hardener): return pd.DataFrame(), []

        for i, (r, h) in enumerate(zip(resin, hardener)):
            try:
                mr = Chem.MolFromSmiles(str(r)); mh = Chem.MolFromSmiles(str(h))
                if not mr or not mh: continue
                # 简化特征计算
                mw_r = Descriptors.MolWt(mr); mw_h = Descriptors.MolWt(mh)
                feats.append({'EEW': mw_r/2, 'AHEW': mw_h/4, 'MolWt_Resin': mw_r, 'MolWt_Hardener': mw_h})
                idxs.append(i)
            except: continue
        return pd.DataFrame(feats), idxs

class FingerprintExtractor:
    def __init__(self):
        if not RDKIT_AVAILABLE: raise ImportError("Need RDKit")

    def _gen_fp(self, mol, fp_type, n_bits, radius):
        if fp_type == 'MACCS': return np.array(MACCSkeys.GenMACCSKeys(mol))
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

    def smiles_to_fingerprints(self, smiles_list, smiles_list_2=None, fp_type='MACCS', n_bits=2048, radius=2):
        fps, idxs = [], []
        is_dual = smiles_list_2 is not None and len(smiles_list_2) == len(smiles_list)

        for i, s1 in enumerate(tqdm(smiles_list, desc=f"{fp_type}")):
            try:
                m1 = Chem.MolFromSmiles(str(s1))
                if not m1: continue
                f = {}

                v1 = self._gen_fp(m1, fp_type, n_bits, radius)
                for j, v in enumerate(v1): f[f"Resin_{fp_type}_{j}"] = v

                if is_dual:
                    m2 = Chem.MolFromSmiles(str(smiles_list_2[i]))
                    if m2:
                        v2 = self._gen_fp(m2, fp_type, n_bits, radius)
                        for j, v in enumerate(v2): f[f"Hardener_{fp_type}_{j}"] = v

                fps.append(f); idxs.append(i)
            except: continue

        if not fps: return pd.DataFrame(), []
        df = pd.DataFrame(fps).astype(np.uint8)
        df = df.loc[:, (df != 0).any(axis=0)]
        return df, idxs

MACCS_DEFINITIONS = {
    1: "ISOTOPE", 11: "4M Ring", 22: "3M Ring", 24: "N-O", 41: "C#N", 42: "F", 
    49: "C=C", 52: "NN", 60: "S=O", 78: "C=N", 84: "NH2", 96: "5M Ring", 
    101: "8M Ring", 103: "Cl", 121: "N Heterocycle", 125: "Aromatic Ring > 1", 
    139: "OH", 145: "6M ring > 1", 149: "CH3 > 1", 154: "C=O", 157: "C-O", 
    158: "C-N", 160: "CH3", 161: "N", 162: "Aromatic", 163: "6M Ring", 164: "O", 165: "Ring"
}
def get_maccs_description(idx):
    try: return MACCS_DEFINITIONS.get(int(idx), "Unknown Fragment")
    except: return "Invalid"
'''

# ==============================================================================
# 2. 升级 app.py (只替换 page_molecular_features 函数)
# ==============================================================================
APP_PAGE_UPDATE = r'''def page_molecular_features():
    """分子特征提取页面 (支持多列组分选择)"""
    st.title("🧬 分子特征提取")
    if st.session_state.data is None: st.warning("请上传数据"); return
    df = st.session_state.processed_data
    cols = df.select_dtypes(include=['object']).columns.tolist()

    c1, c2 = st.columns(2)
    with c1: 
        # [升级] 改为多选，支持多组分树脂
        resin_cols = st.multiselect("树脂 SMILES 列 (支持多选)", cols, help="如果树脂由多种单体组成，请选中所有对应列，系统将自动合并")
    with c2: 
        # [升级] 改为多选，支持多组分固化剂
        hard_cols = st.multiselect("固化剂 SMILES 列 (可选)", [c for c in cols if c not in resin_cols])

    method = st.radio("方法", ["MACCS指纹", "RDKit描述符", "ML力场 (ANI-2x)"])

    if st.button("开始提取"):
        if not resin_cols: st.error("请至少选择一列树脂"); return

        # [逻辑] 自动合并多列为混合物字符串 (A.B.C)
        def merge_cols(columns):
            if not columns: return None
            # 将多列转为字符串，并用 . 连接，忽略空值
            return df[columns].apply(lambda x: '.'.join(x.dropna().astype(str)), axis=1).tolist()

        l1 = merge_cols(resin_cols)
        l2 = merge_cols(hard_cols)

        try:
            if "指纹" in method:
                ext = FingerprintExtractor()
                res, idx = ext.smiles_to_fingerprints(l1, l2)
            elif "RDKit" in method:
                ext = AdvancedMolecularFeatureExtractor()
                res, idx = ext.smiles_to_rdkit_features(l1)
            else:
                ext = MLForceFieldExtractor()
                # 提示：ML力场计算较慢
                st.info("正在进行量子化学计算，混合物将自动拆分计算...")
                res, idx = ext.smiles_to_ani_features(l1)

            if not res.empty:
                df_valid = df.iloc[idx].reset_index(drop=True)
                # 前缀使用首个列名
                prefix = f"{resin_cols[0]}_"
                res = res.add_prefix(prefix)

                st.session_state.processed_data = pd.concat([df_valid, res], axis=1)
                st.session_state.molecular_features = res
                st.success(f"✅ 提取完成: {res.shape[1]} 个特征")
                st.dataframe(res.head())
            else:
                st.error("未能提取特征，请检查 SMILES 格式")
        except Exception as e: st.error(f"提取失败: {str(e)}")
'''


def update_file(path, content, mode='w'):
    with open(path, mode, encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已更新: {path}")


def replace_func(path, func_name, new_code):
    import re
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 替换函数正则
    pattern = re.compile(fr"def {func_name}\(.*\):.*?(?=\n^def |\Z)", re.DOTALL | re.MULTILINE)
    if pattern.search(content):
        new_content = pattern.sub(new_code, content)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已替换函数: {func_name} in {path}")
    else:
        print(f"⚠️ 未找到函数 {func_name}")


if __name__ == "__main__":
    # 1. 覆盖 molecular_features.py (包含所有新逻辑)
    update_file("core/molecular_features.py", MOLECULAR_FEATURES_NEW)

    # 2. 替换 app.py 中的 page_molecular_features 函数
    replace_func("app.py", "page_molecular_features", APP_PAGE_UPDATE)

    print("\n🎉 多组分功能升级完成！\n现在您可以在界面上选择多列 SMILES，系统会自动合并处理。")