# -*- coding: utf-8 -*-
"""特征选择模块 - 优化版 V2
核心改进：
1. 智能前缀分析：自动检测重复模式的特征组（如 fp_0, fp_1, ...）
2. 手动纠错机制：允许用户调整特征分类
3. 确保所有特征可选：不遗漏任何数值列
4. 简化操作：默认全选，方便剔除不需要的原始特征
5. 多核并行处理：使用 joblib 加速特征分类和数据处理
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import warnings
import re
from collections import Counter, defaultdict
from joblib import Parallel, delayed
import multiprocessing
from .model_interpreter import compute_xgboost_native_shap
from .process_pls import ProcessPLSTransformer, process_pls_config_to_dict

warnings.filterwarnings('ignore')

# 获取 CPU 核心数
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统


def infer_process_feature_candidates(
    frame,
    original_features,
    molecular_features,
    target_col,
):
    """Infer numeric process-feature candidates without mixing molecular inputs."""
    if frame is None:
        return []
    molecular = set(molecular_features or [])
    excluded = {str(target_col)} if target_col else set()
    result = []
    for column in list(original_features or []):
        if column not in getattr(frame, "columns", []):
            continue
        column_name = str(column)
        if column in molecular or column_name in excluded:
            continue
        if column_name.lower().endswith(("_smiles", "_bigsmiles")):
            continue
        if frame[column].dtype == "object":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        result.append(column)
    return result


def build_process_pls_config(process_feature_cols, random_state=42):
    """Build a serializable, unfitted process-PLS workflow config."""
    return process_pls_config_to_dict({
        "schema_version": 1,
        "enabled": True,
        "process_feature_cols": list(process_feature_cols or []),
        "max_components": 8,
        "vip_top_k": 8,
        "missing_threshold": 0.85,
        "cv_splits": 5,
        "random_state": int(random_state),
        "selection_mode": "auto_combined_score",
    })


# ==========================================
# 1. 智能特征分类器
# ==========================================

class SmartFeatureClassifier:
    """智能特征分类器 - 基于模式检测自动识别分子特征"""
    
    # 分子特征的典型前缀（已知的）
    KNOWN_MOLECULAR_PREFIXES = {
        'fp', 'maccs', 'morgan', 'ecfp', 'fcfp', 'topological', 'fingerprint',
        'avalon', 'atompair', 'rdk', 'rdkfp', 'rdkit', 'mordred', 'rdkit3d',
        'fgd', 'mlff', 'transformer', 'tda', 'graph', 'gnn', 'lm_emb',
        'chemberta', 'bert', 'embedding', 'coulomb', 'cm', '3dconf', 'ani',
        'persistence', 'betti', 'diagram', 'crosslink', 'polymer'
    }
    
    # 原始特征的典型关键词
    ORIGINAL_FEATURE_KEYWORDS = {
        'temperature', 'temp', 'pressure', 'time', 'duration',
        'concentration', 'ratio', 'percent', 'phr', 'content',
        'density', 'viscosity', 'modulus', 'strength', 'toughness',
        'hardness', 'elongation', 'strain', 'stress', 'cure', 'curing',
        'heating', 'cooling', 'mixing', 'stirring', 'speed', 'rate',
        'flow', 'test', 'sample', 'specimen', 'batch', 'lot',
        'id', 'index', 'idx', 'no', 'number', 'amount', 'loading',
        'filler', 'additive', 'weight', 'mass', 'volume'
    }
    
    def __init__(self, feature_list: list, recorded_molecular_features: set = None, source_feature_names: set = None):
        """
        初始化分类器
        
        Parameters
        ----------
        feature_list : list
            所有特征名列表
        recorded_molecular_features : set
            已记录的分子特征名集合（来自session_state）
        """
        self.feature_list = feature_list
        self.source_features = set(source_feature_names or set())
        self.recorded_mol = set(recorded_molecular_features or set()) - self.source_features
        
        # 分析结果
        self.prefix_groups = {}  # {prefix: [features]}
        self.prefix_counts = {}  # {prefix: count}
        self.molecular_features = []
        self.original_features = []
        self.uncertain_features = []  # 不确定的特征
        
        # 执行分析
        self._analyze_prefixes()
        self._classify_features()
    
    def _extract_prefix(self, feature_name: str) -> str:
        """提取特征名的前缀"""
        # 策略1: 用下划线分割，取第一部分
        parts = feature_name.split('_')
        if len(parts) >= 2:
            # 检查是否是 "prefix_数字" 模式
            if parts[-1].isdigit():
                return '_'.join(parts[:-1])
            # 检查是否是 "prefix_描述符名" 模式
            return parts[0]
        
        # 策略2: 用数字分割
        match = re.match(r'^([a-zA-Z_]+)', feature_name)
        if match:
            return match.group(1).rstrip('_')
        
        return feature_name
    
    def _analyze_prefixes(self):
        """分析所有特征的前缀模式"""
        # 这里的字符串拆分本身非常轻，线程调度反而更慢。
        prefix_to_features = defaultdict(list)
        for feat in self.feature_list:
            prefix = self._extract_prefix(feat)
            prefix_to_features[prefix].append(feat)

        self.prefix_groups = dict(prefix_to_features)
        self.prefix_counts = {p: len(feats) for p, feats in prefix_to_features.items()}
    
    def _is_molecular_prefix(self, prefix: str, count: int) -> bool:
        """判断一个前缀是否代表分子特征组"""
        prefix_lower = prefix.lower()
        prefix_tokens = {tok for tok in re.split(r'[^a-z0-9]+', prefix_lower) if tok}
        if prefix_lower in self.KNOWN_MOLECULAR_PREFIXES:
            return True
        if prefix_tokens & self.KNOWN_MOLECULAR_PREFIXES:
            return True

        # 1. 已知的分子特征前缀
        for known in ():
            if known in prefix_lower:
                return True

        # 2. 大量重复的特征组（>5个）很可能是分子特征
        if False and count >= 5:
            # 但要排除明显的原始特征关键词
            for kw in self.ORIGINAL_FEATURE_KEYWORDS:
                if kw in prefix_lower:
                    return False
            return True

        return False

    def _is_molecular_feature(self, feature_name: str) -> bool:
        """判断单个特征名是否为分子特征（用于环氧树脂反应特征等）"""
        feat_lower = feature_name.lower()
        strong_patterns = (
            r'^(fp|maccs|morgan|ecfp|fcfp|rdkit|rdkfp|mordred|rdkit3d|graph|gnn|crosslink|polymer|'
            r'chemberta|transformer|embedding|lm_emb|coulomb|ani|tda)(_|$)',
        )
        if any(re.search(pattern, feat_lower) for pattern in strong_patterns):
            return True
        if any(token in feat_lower for token in (
            '_gnn_', '_graph_', '_polymer_', '_crosslink_', '_morgan_', '_maccs_', '_rdkit_', '_mordred_'
        )):
            return True

        # 环氧树脂反应特征的典型特征名
        epoxy_keywords = {
            'crosslink', 'curer_type'
        }

        for kw in epoxy_keywords:
            if kw in feat_lower:
                return True

        return False
    
    def _classify_features(self):
        """对所有特征进行分类"""
        self.molecular_features = []
        self.original_features = []
        self.uncertain_features = []

        # 已处理的特征
        processed = set()

        # 1. 首先处理已记录的分子特征
        for feat in self.feature_list:
            if feat in self.recorded_mol:
                self.molecular_features.append(feat)
                processed.add(feat)

        for feat in self.feature_list:
            if feat in processed:
                continue
            if feat in self.source_features:
                self.original_features.append(feat)
                processed.add(feat)

        # 2. 按前缀组处理
        for prefix, features in self.prefix_groups.items():
            count = len(features)
            is_mol = any(f in self.recorded_mol for f in features) or self._is_molecular_prefix(prefix, count)

            for feat in features:
                if feat in processed:
                    continue

                # 先检查特征名本身是否为分子特征（环氧树脂反应特征等）
                if self._is_molecular_feature(feat):
                    self.molecular_features.append(feat)
                elif is_mol:
                    self.molecular_features.append(feat)
                else:
                    # 进一步检查是否包含原始特征关键词
                    feat_lower = feat.lower()
                    is_original = False
                    for kw in self.ORIGINAL_FEATURE_KEYWORDS:
                        if kw in feat_lower:
                            is_original = True
                            break

                    if is_original:
                        self.original_features.append(feat)
                    else:
                        # 单个特征，可能是原始特征
                        self.original_features.append(feat)
                    if False:
                        # 不确定的归为分子特征（因为有多个同前缀的）
                        self.molecular_features.append(feat)

                processed.add(feat)

        # 3. 处理遗漏的特征（保底）
        for feat in self.feature_list:
            if feat not in processed:
                # 最后检查一次是否为分子特征
                if self._is_molecular_feature(feat):
                    self.molecular_features.append(feat)
                else:
                    self.original_features.append(feat)
    
    def get_classification(self):
        """获取分类结果"""
        return {
            'molecular': self.molecular_features.copy(),
            'original': self.original_features.copy(),
            'prefix_groups': self.prefix_groups.copy(),
            'prefix_counts': self.prefix_counts.copy()
        }
    
    def get_prefix_summary(self) -> pd.DataFrame:
        """获取前缀统计摘要"""
        data = []
        for prefix, count in sorted(self.prefix_counts.items(), key=lambda x: -x[1]):
            is_mol = self._is_molecular_prefix(prefix, count)
            data.append({
                '前缀': prefix,
                '特征数量': count,
                '分类': '🧬 分子特征' if is_mol else '📊 原始特征',
                '示例': self.prefix_groups[prefix][0] if self.prefix_groups[prefix] else ''
            })
        return pd.DataFrame(data)
    
    def move_to_molecular(self, features: list):
        """将特征移动到分子特征组"""
        for feat in features:
            if feat in self.original_features:
                self.original_features.remove(feat)
            if feat not in self.molecular_features:
                self.molecular_features.append(feat)
    
    def move_to_original(self, features: list):
        """将特征移动到原始特征组"""
        for feat in features:
            if feat in self.molecular_features:
                self.molecular_features.remove(feat)
            if feat not in self.original_features:
                self.original_features.append(feat)


def _smart_feature_classifier_is_molecular_prefix_conservative(self, prefix: str, count: int) -> bool:
    """Conservative prefix rule: only explicit molecular prefixes are molecular."""
    prefix_lower = prefix.lower()
    prefix_tokens = {tok for tok in re.split(r'[^a-z0-9]+', prefix_lower) if tok}
    if prefix_lower in self.KNOWN_MOLECULAR_PREFIXES:
        return True
    if prefix_tokens & self.KNOWN_MOLECULAR_PREFIXES:
        return True
    return False


def _smart_feature_classifier_classify_features_conservative(self):
    """
    Conservative feature classification:
    1. session-recorded molecular features stay molecular
    2. source/import-time features stay original
    3. only high-confidence molecular names/prefixes become molecular
    4. everything else defaults to original
    """
    self.molecular_features = []
    self.original_features = []
    self.uncertain_features = []

    processed = set()

    for feat in self.feature_list:
        if feat in self.recorded_mol:
            self.molecular_features.append(feat)
            processed.add(feat)

    for feat in self.feature_list:
        if feat in processed:
            continue
        if feat in self.source_features:
            self.original_features.append(feat)
            processed.add(feat)

    for prefix, features in self.prefix_groups.items():
        count = len(features)
        is_mol_group = any(f in self.recorded_mol for f in features) or self._is_molecular_prefix(prefix, count)

        for feat in features:
            if feat in processed:
                continue
            if self._is_molecular_feature(feat) or is_mol_group:
                self.molecular_features.append(feat)
            else:
                self.original_features.append(feat)
            processed.add(feat)

    for feat in self.feature_list:
        if feat in processed:
            continue
        if self._is_molecular_feature(feat):
            self.molecular_features.append(feat)
        else:
            self.original_features.append(feat)


SmartFeatureClassifier._is_molecular_prefix = _smart_feature_classifier_is_molecular_prefix_conservative
SmartFeatureClassifier._classify_features = _smart_feature_classifier_classify_features_conservative


# ==========================================
# 2. 回调函数定义区
# ==========================================

def _update_selection_state(new_selection):
    """通用回调：更新特征选择状态"""
    st.session_state.feature_cols = new_selection
    st.session_state.multiselect_features = new_selection


def _apply_variance_filter_callback(df, candidates, threshold_key):
    """方差筛选回调"""
    try:
        threshold = st.session_state[threshold_key]
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df.fillna(0))
        selected = [candidates[i] for i in selector.get_support(indices=True)]
        _update_selection_state(selected)
        st.session_state['feature_selector_msg'] = f"✅ 方差筛选完成：剩余 {len(selected)} 个特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_correlation_filter_callback(df, target_series, k_key):
    """相关性筛选回调"""
    try:
        k = st.session_state[k_key]
        corrs = df.corrwith(target_series).abs().sort_values(ascending=False)
        selected = corrs.head(int(k)).index.tolist()
        _update_selection_state(selected)
        st.session_state['feature_selector_msg'] = f"✅ 相关性筛选完成：已选 Top-{k} 特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _build_rfe_estimator(estimator_name: str, random_state: int = 42):
    """根据名称构建RFE/RFECV可用的估计器"""
    name = (estimator_name or "").strip()
    if name in ["Ridge线性回归", "Ridge"]:
        return Ridge(alpha=1.0), True
    if name in ["Lasso稀疏回归", "Lasso"]:
        return Lasso(alpha=0.001, max_iter=5000, random_state=random_state), True
    if name in ["线性SVR", "SVR(linear)"]:
        return SVR(kernel="linear", C=1.0), True
    if name in ["ExtraTrees", "极端随机森林"]:
        return ExtraTreesRegressor(n_estimators=400, random_state=random_state, n_jobs=-1), False
    if name in ["梯度提升", "GBDT", "GradientBoosting"]:
        return GradientBoostingRegressor(random_state=random_state), False
    return RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1), False


def _apply_rfe_filter_callback(df, target_series, candidates,
                              mode_key, est_key, k_key, step_key,
                              cv_key, scoring_key, min_k_key):
    """RFE / RFECV 递归特征消除回调"""
    try:
        mode = st.session_state.get(mode_key, "RFECV自动（推荐）")
        est_name = st.session_state.get(est_key, "随机森林")
        estimator, needs_scaling = _build_rfe_estimator(est_name, random_state=42)

        X = df[candidates].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(numeric_only=True))
        X = X.select_dtypes(include=np.number)
        used_candidates = X.columns.tolist()

        if len(used_candidates) < 2:
            st.session_state['feature_selector_error'] = "可用于RFE的数值特征少于2个。"
            return

        y = pd.Series(target_series).copy()
        y = y.replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            y = y.fillna(method="ffill").fillna(method="bfill")

        if needs_scaling:
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
        else:
            X_values = X.values

        step = max(1, int(st.session_state.get(step_key, 1)))

        if str(mode).startswith("RFECV"):
            cv = max(2, min(10, int(st.session_state.get(cv_key, 5))))
            min_k = max(1, min(len(used_candidates) - 1, int(st.session_state.get(min_k_key, 10))))
            scoring = st.session_state.get(scoring_key, "r2")
            selector = RFECV(
                estimator=estimator,
                step=step,
                cv=KFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=scoring,
                min_features_to_select=min_k,
                n_jobs=-1
            )
        else:
            k = max(1, min(len(used_candidates), int(st.session_state.get(k_key, 20))))
            selector = RFE(estimator=estimator, n_features_to_select=k, step=step)

        selector.fit(X_values, y.values.ravel())

        selected = [used_candidates[i] for i, flag in enumerate(selector.support_) if flag]
        _update_selection_state(selected)

        ranking_df = pd.DataFrame({
            "特征": used_candidates,
            "RFE排名": selector.ranking_
        }).sort_values("RFE排名", ascending=True)
        st.session_state["rfe_ranking_df"] = ranking_df

        n_selected = int(getattr(selector, "n_features_", len(selected)))
        if str(mode).startswith("RFECV"):
            st.session_state['feature_selector_msg'] = f"✅ RFECV完成：CV最优特征数 = {n_selected}"
        else:
            st.session_state['feature_selector_msg'] = f"✅ RFE完成：已选 {n_selected} 个特征"

    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_importance_filter_callback_v2(sorted_features, candidates, k_key):
    """模型重要性筛选回调"""
    try:
        k = st.session_state.get(k_key, 20)
        top_f = sorted_features[:int(k)]
        valid_selected = [f for f in top_f if f in candidates]
        _update_selection_state(valid_selected)
        st.session_state['feature_selector_msg'] = f"✅ 模型筛选完成：已选 {len(valid_selected)} 个特征"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _impute_df(df, strategy: str):
    X = df.copy()
    try:
        X = X.apply(pd.to_numeric, errors="coerce")
    except Exception:
        pass
    X = X.replace([np.inf, -np.inf], np.nan)

    if strategy == "median":
        X = X.fillna(X.median())
    elif strategy == "zero":
        X = X.fillna(0)
    else:
        X = X.fillna(X.mean())

    if X.isna().values.any():
        X = X.fillna(0.0)
    return X


def _build_scaler(mode: str):
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler()
    return None


def _ensure_finite_matrix(X, columns=None):
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if np.isfinite(arr).all():
        return arr, []

    bad_cols = []
    if columns is not None and arr.ndim == 2 and len(columns) == arr.shape[1]:
        finite_by_col = np.isfinite(arr).all(axis=0)
        bad_cols = [str(columns[i]) for i, ok in enumerate(finite_by_col) if not ok]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, bad_cols


def _warn_pca_fallback(stage: str, bad_cols):
    if not bad_cols:
        return
    preview = ", ".join(list(bad_cols)[:8])
    if len(bad_cols) > 8:
        preview += f" 等 {len(bad_cols)} 列"
    st.warning(f"⚠️ {stage} 前检测到非有限值列，系统已自动按 0 兜底处理：{preview}")


def _apply_pca_callback(pca, scaler, numeric_df, current_df, feature_candidates, fill_strategy: str = "mean", pc_prefix: str = "PC"):
    """PCA应用回调"""
    try:
        X = _impute_df(numeric_df.copy(), fill_strategy)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        X_scaled, bad_cols = _ensure_finite_matrix(X_scaled, X.columns.tolist())
        _warn_pca_fallback("PCA应用", bad_cols)
        X_pca = pca.transform(X_scaled)

        pc_names = [f"{pc_prefix}{i + 1}" for i in range(pca.n_components_)]
        df_pca = pd.DataFrame(X_pca, columns=pc_names, index=current_df.index)

        df_rest = current_df.drop(columns=feature_candidates)
        df_new = pd.concat([df_rest, df_pca], axis=1)

        st.session_state.processed_data = df_new
        _update_selection_state(pc_names)

        st.session_state.pop('_pca_model', None)
        st.session_state.pop('_pca_scaler', None)
        st.session_state.pop('_pca_ready', None)

        st.session_state['feature_selector_msg'] = "✅ PCA 转换已应用！数据集已更新。"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


def _apply_batch_pca_callback(
    current_df,
    numeric_df,
    feature_candidates,
    prefix_groups,
    selected_prefixes,
    n_components,
    scaler_mode: str,
    fill_strategy: str,
    svd_solver: str,
    whiten: bool,
    random_state: int,
    iterated_power,
    keep_original: bool = False,
):
    """按前缀组批量PCA降维"""
    try:
        df_new = current_df.copy()
        removed_cols = []
        added_cols = []
        reports = []

        for prefix in selected_prefixes:
            group_feats = [f for f in prefix_groups.get(prefix, []) if f in feature_candidates]
            if len(group_feats) < 2:
                continue

            X_group = _impute_df(numeric_df[group_feats].copy(), fill_strategy)
            scaler = _build_scaler(scaler_mode)
            if scaler is not None:
                X_scaled = scaler.fit_transform(X_group)
            else:
                X_scaled = X_group.values
            X_scaled, bad_cols = _ensure_finite_matrix(X_scaled, X_group.columns.tolist())
            _warn_pca_fallback(f"批量PCA（{prefix}）", bad_cols)

            pca = PCA(
                n_components=n_components,
                whiten=bool(whiten),
                svd_solver=svd_solver,
                random_state=int(random_state) if random_state is not None else None,
                iterated_power=iterated_power,
            )
            pca.fit(X_scaled)
            X_pca = pca.transform(X_scaled)

            safe_prefix = re.sub(r"\s+", "_", str(prefix)).strip("_") or "group"
            pc_names = [f"{safe_prefix}_PC{i + 1}" for i in range(pca.n_components_)]
            df_pca = pd.DataFrame(X_pca, columns=pc_names, index=current_df.index)

            if not keep_original:
                df_new = df_new.drop(columns=group_feats, errors="ignore")
                removed_cols.extend(group_feats)
            df_new = pd.concat([df_new, df_pca], axis=1)
            added_cols.extend(pc_names)

            explained = float(np.sum(pca.explained_variance_ratio_)) if hasattr(pca, "explained_variance_ratio_") else 0.0
            reports.append({
                "前缀": prefix,
                "原始特征数": int(len(group_feats)),
                "主成分数": int(getattr(pca, "n_components_", X_pca.shape[1])),
                "累计解释方差": explained,
                "_variance_ratio": pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else [],
            })

        if not added_cols:
            st.session_state['feature_selector_error'] = "未生成任何主成分，请检查所选前缀组。"
            return

        st.session_state.processed_data = df_new

        current_selected = st.session_state.get('feature_cols') or feature_candidates
        valid_selected = [f for f in current_selected if f in df_new.columns and f not in removed_cols]
        for c in added_cols:
            if c not in valid_selected:
                valid_selected.append(c)
        _update_selection_state(valid_selected)

        st.session_state["pca_batch_report"] = pd.DataFrame(reports)
        st.session_state['feature_selector_msg'] = f"✅ 批量PCA完成：新增 {len(added_cols)} 个主成分"
    except Exception as e:
        st.session_state['feature_selector_error'] = str(e)


# ==========================================
# 3. 主渲染函数
# ==========================================

def render_feature_selector():
    """渲染特征选择界面 - 优化版V2"""
    st.markdown("## 🎯 特征选择与筛选")

    def _count_inf_values(df_block: pd.DataFrame) -> int:
        if df_block is None or df_block.empty:
            return 0
        try:
            arr = df_block.to_numpy(dtype=np.float64, na_value=np.nan)
        except TypeError:
            arr = df_block.astype(float).to_numpy()
        return int(np.isinf(arr).sum())

    # 性能提示
    if st.session_state.get('_show_perf_tip', True):
        with st.expander("💡 性能优化提示", expanded=False):
            st.markdown(f"""
            **系统信息：**
            - 使用 {N_JOBS} 个 CPU 核心进行并行处理
            - 已启用智能缓存机制

            **如果界面卡顿，可以尝试：**
            1. 使用筛选功能缩小特征范围（类型/关键词/前缀）
            2. 避免频繁切换筛选条件
            3. 使用快速操作按钮代替手动多选
            4. 关闭调试模式
            """)
            if st.button("不再显示", key="hide_perf_tip"):
                st.session_state['_show_perf_tip'] = False
                st.rerun()

    # [新增] 调试模式开关
    col_debug1, col_debug2 = st.columns([4, 1])
    with col_debug2:
        debug_mode = st.checkbox("🔍 调试模式", value=st.session_state.get('_fs_debug_mode', False), key='_fs_debug_mode')

    # 显示消息
    if 'feature_selector_msg' in st.session_state:
        st.success(st.session_state.pop('feature_selector_msg'))
    if 'feature_selector_error' in st.session_state:
        st.error(st.session_state.pop('feature_selector_error'))

    # 数据源
    if st.session_state.get('processed_data') is not None:
        current_df = st.session_state.processed_data
    elif st.session_state.get('data') is not None:
        current_df = st.session_state.data
        st.info("ℹ️ 使用原始数据")
    else:
        st.warning("⚠️ 请先上传数据")
        return

    # 重复列名检查
    if current_df.columns.duplicated().any():
        current_df = current_df.loc[:, ~current_df.columns.duplicated()]
        st.session_state.processed_data = current_df
    
    all_columns = current_df.columns.tolist()

    # 目标变量选择
    col1, col2 = st.columns([1, 2])
    with col1:
        default_idx = 0
        if st.session_state.get('target_col') and st.session_state.target_col in all_columns:
            default_idx = all_columns.index(st.session_state.target_col)
        target_col = st.selectbox("🎯 选择目标变量 (Y)", options=all_columns, index=default_idx)
        st.session_state.target_col = target_col

    # 获取所有数值特征（包含布尔/扩展整型）；布尔列转换为0/1避免被遗漏
    numeric_df = current_df.select_dtypes(include=[np.number, "number", "bool", "boolean"]).copy()
    bool_cols = numeric_df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    if bool_cols:
        numeric_df[bool_cols] = numeric_df[bool_cols].astype(int)

    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    
    feature_candidates = numeric_df.columns.tolist()
    total_features = len(feature_candidates)

    with col2:
        st.metric("可用数值特征", f"{total_features} 个")

    # 检查无穷大值（避免每次交互都全表扫描）
    if total_features > 0:
        data_shape_sig = (current_df.shape, len(current_df.columns))
        cached_sig = st.session_state.get("_fs_inf_signature")
        check_mode = st.session_state.get("_fs_inf_check_mode", "sample")
        large_matrix = (current_df.shape[0] * max(1, total_features)) >= 2_000_000 or total_features >= 1200

        if cached_sig != data_shape_sig:
            if large_matrix:
                sample_rows = max(1, min(len(numeric_df), 512))
                check_inf = _count_inf_values(numeric_df.head(sample_rows))
                check_mode = "sample"
            else:
                check_inf = _count_inf_values(numeric_df)
                check_mode = "full"
            st.session_state["_fs_inf_signature"] = data_shape_sig
            st.session_state["_fs_inf_count"] = check_inf
            st.session_state["_fs_inf_check_mode"] = check_mode
        else:
            check_inf = int(st.session_state.get("_fs_inf_count", 0))
            check_mode = st.session_state.get("_fs_inf_check_mode", "sample")

        with st.expander("🩺 数据异常值检查", expanded=False):
            if check_mode == "sample":
                if check_inf > 0:
                    st.warning("⚠️ 采样检查发现疑似 Inf 值。为避免首屏卡顿，系统尚未执行全表扫描。")
                else:
                    st.caption("已跳过全表 Inf 扫描以提升加载速度。当前结果来自采样检查。")

                if st.button("🔎 执行完整 Inf 检查", key="run_full_inf_check_btn"):
                    with st.spinner("正在执行完整 Inf 检查..."):
                        full_inf_count = _count_inf_values(numeric_df)
                    st.session_state["_fs_inf_signature"] = data_shape_sig
                    st.session_state["_fs_inf_count"] = int(full_inf_count)
                    st.session_state["_fs_inf_check_mode"] = "full"
                    st.rerun()
            elif check_inf > 0:
                st.error(f"⚠️ 检测到 {check_inf} 个无穷大数值 (Inf)")
            else:
                st.success("✅ 未检测到 Inf 值")

            if check_mode == "full" and check_inf > 0:
                if st.button("🧹 一键修复异常值", type="primary", key="fix_inf_btn"):
                    with st.spinner("正在修复异常值..."):
                        clean_df = current_df.copy()
                        all_numeric_cols = clean_df.select_dtypes(include=[np.number, "number"]).columns.tolist()

                        def fix_column(col):
                            """修复单列的 Inf 和 NaN 值"""
                            col_data = clean_df[col].replace([np.inf, -np.inf], np.nan)
                            col_mean = col_data.mean()
                            if pd.isna(col_mean):
                                return col_data.fillna(0)
                            return col_data.fillna(col_mean)

                        if len(all_numeric_cols) > 20:
                            fixed_cols = Parallel(n_jobs=N_JOBS, backend='threading')(
                                delayed(fix_column)(col) for col in all_numeric_cols
                            )
                            for col, fixed_data in zip(all_numeric_cols, fixed_cols):
                                clean_df[col] = fixed_data
                        else:
                            for col in all_numeric_cols:
                                clean_df[col] = fix_column(col)

                        remaining_inf = _count_inf_values(clean_df.select_dtypes(include=[np.number, "number"]))

                        if remaining_inf > 0:
                            st.error(f"❌ 修复失败，仍有 {remaining_inf} 个 Inf 值")
                        else:
                            st.session_state.data = clean_df
                            st.session_state.processed_data = clean_df
                            st.session_state.pop("_fs_inf_signature", None)
                            st.session_state.pop("_fs_inf_count", None)
                            st.session_state.pop("_fs_inf_check_mode", None)
                            st.success("✅ 修复完成！已清除所有 Inf 值")
                            st.rerun()

    st.markdown("---")

    # ==========================================
    # Tabs 布局 - 先选择标签，再加载对应内容
    # ==========================================
    tab_titles = [
        "👆 手动选择",
        "🔧 特征分类纠错",
        "📉 方差筛选",
        "🔗 相关性筛选",
        "🌀 RFE递归消除",
        "🎯 FSFS去冗余",
        "🧩 PCA降维",
        "⚗️ 工艺PLS",
        "⭐ 模型重要性",
        "🏷️ 前缀管理",
    ]

    # 使用 session_state 记住当前标签，避免重复渲染
    if 'feature_selector_tab_index' not in st.session_state:
        st.session_state.feature_selector_tab_index = 0

    tab_choice = st.radio(
        "功能区",
        tab_titles,
        index=st.session_state.feature_selector_tab_index,
        horizontal=True,
        label_visibility="collapsed",
        key="feature_selector_tab"
    )

    # 更新标签索引
    st.session_state.feature_selector_tab_index = tab_titles.index(tab_choice)

    # ==========================================
    # 懒加载：只在需要时执行特征分类
    # ==========================================
    def get_feature_classification():
        """获取特征分类结果 - 懒加载"""
        # 获取已记录的分子特征（来自特征提取时保存的）
        recorded_mol_features = set(st.session_state.get('molecular_feature_names', []))
        source_feature_names = set(st.session_state.get('source_feature_names', []))
        base_df_for_origin = st.session_state.get('data')
        if isinstance(base_df_for_origin, pd.DataFrame):
            source_feature_names.update(map(str, base_df_for_origin.columns.tolist()))
        recorded_mol_features -= source_feature_names

        # [调试] 显示已记录的分子特征数量 - 仅在调试模式下显示
        debug_mode = st.session_state.get('_fs_debug_mode', False)
        if debug_mode and recorded_mol_features:
            with st.expander("🔍 分子特征记录状态", expanded=False):
                if recorded_mol_features:
                    st.success(f"✅ 从 session_state 读取到 {len(recorded_mol_features)} 个已记录的分子特征")
                    if len(recorded_mol_features) <= 30:
                        st.write(f"**已记录特征:**")
                        st.write(list(recorded_mol_features))
                    else:
                        st.write(f"**已记录特征（前30个）:**")
                        st.write(list(recorded_mol_features)[:30])

                    # 检查环氧树脂特征
                    epoxy_keywords = ['eew', 'ahew', 'stoich', 'equiv', 'phr', 'crosslink', 'curer', 'functionality', 'alpha_max', 'charge', 'tpsa']
                    epoxy_features = [f for f in recorded_mol_features if any(kw in f.lower() for kw in epoxy_keywords)]
                    if epoxy_features:
                        st.success(f"✅ 包含 {len(epoxy_features)} 个环氧树脂反应特征")
                        st.write(f"**环氧树脂特征:**")
                        st.write(epoxy_features)
                else:
                    st.warning("⚠️ molecular_feature_names 为空或不存在")
                    st.info("💡 提示：如果刚提取了特征但这里显示为空，可能是：\n1. 特征提取时没有正确保存\n2. 页面刷新导致状态丢失\n3. 使用了'清除特征'功能")

        # ✅ 检测数据是否发生变化（使用更高效的签名）
        cached_classification = st.session_state.get('feature_classification', None)
        need_reclassify = False
        new_features = set()  # 新增特征集合

        # 使用特征数量和前几个特征名作为快速签名
        current_signature = (
            "feature_origin_v3",
            tuple(feature_candidates),
            tuple(sorted(recorded_mol_features & set(feature_candidates))),
            tuple(sorted(source_feature_names & set(feature_candidates))),
        )
        cached_signature = st.session_state.get('_feature_signature', None)

        if cached_classification is not None and cached_signature == current_signature:
            # 签名匹配，使用缓存
            need_reclassify = False
        elif cached_classification is not None:
            # 签名不匹配，检查详细差异
            cached_all = set(cached_classification.get('molecular', [])) | set(cached_classification.get('original', []))
            current_all = set(feature_candidates)

            # 如果特征列发生变化，需要重新分类
            if cached_all != current_all:
                need_reclassify = True
                new_features = current_all - cached_all
                removed_features = cached_all - current_all
                if new_features:
                    st.info(f"🔄 检测到 {len(new_features)} 个新特征，正在重新分类...")
                    # ✅ 关键修复：自动将新特征添加到已选择的特征列表中
                    existing_feature_cols = set(st.session_state.get('feature_cols', []))
                    updated_feature_cols = list(existing_feature_cols | new_features)
                    st.session_state.feature_cols = updated_feature_cols
                    st.session_state.multiselect_features = updated_feature_cols.copy()
                    st.success(f"✅ 已自动将 {len(new_features)} 个新提取的特征添加到选择中")
                if removed_features:
                    st.info(f"🔄 检测到 {len(removed_features)} 个特征已移除")
                    # 清理已选择列表中已移除的特征
                    existing_feature_cols = st.session_state.get('feature_cols', [])
                    st.session_state.feature_cols = [f for f in existing_feature_cols if f not in removed_features]
                    st.session_state.multiselect_features = st.session_state.feature_cols.copy()
        else:
            # 没有缓存，需要分类
            need_reclassify = True
            existing_feature_cols = set(st.session_state.get('feature_cols', []))
            current_all = set(feature_candidates)

            # ✅ 关键修复：检测所有新增的特征（包括 one-hot 编码、分子特征等）
            # 这里比较当前可用特征与已选择特征的差异
            new_features_detected = current_all - existing_feature_cols

            if new_features_detected and existing_feature_cols:
                # 有已选择的特征，但发现了新增特征（可能是 one-hot 编码等）
                st.info(f"🔄 检测到 {len(new_features_detected)} 个新特征，正在自动添加...")
                updated_feature_cols = list(existing_feature_cols | new_features_detected)
                st.session_state.feature_cols = updated_feature_cols
                st.session_state.multiselect_features = updated_feature_cols.copy()
                st.success(f"✅ 已自动将 {len(new_features_detected)} 个新特征添加到选择中")
            elif recorded_mol_features:
                # 首次进入时，检查是否有已记录的分子特征需要自动添加
                new_mol_in_candidates = recorded_mol_features & current_all
                if new_mol_in_candidates - existing_feature_cols:
                    # 有已记录但未选择的分子特征
                    updated_feature_cols = list(existing_feature_cols | new_mol_in_candidates)
                    st.session_state.feature_cols = updated_feature_cols
                    st.session_state.multiselect_features = updated_feature_cols.copy()
                    st.info(f"🔄 已自动将 {len(new_mol_in_candidates - existing_feature_cols)} 个已提取的分子特征添加到选择中")

        # 使用智能分类器（缓存优化 + 并行处理）
        @st.cache_data(show_spinner=False, ttl=3600)  # 缓存1小时
        def _get_classification(features_tuple, recorded_tuple, source_tuple):
            """获取特征分类结果 - 使用缓存和并行处理"""
            classifier = SmartFeatureClassifier(list(features_tuple), set(recorded_tuple), set(source_tuple))
            return classifier.get_classification(), classifier

        # 只在需要时重新分类
        if need_reclassify:
            with st.spinner("正在分类特征..."):
                classification, classifier = _get_classification(
                    tuple(feature_candidates),
                    tuple(recorded_mol_features),
                    tuple(source_feature_names),
                )
            # 使用新分类结果
            molecular_features = classification['molecular']
            original_features = classification['original']
            st.session_state.feature_classification = {
                'molecular': molecular_features.copy(),
                'original': original_features.copy()
            }
            st.session_state['_feature_signature'] = current_signature
        else:
            # 使用缓存的分类（保留用户的纠错）
            molecular_features = list(cached_classification.get('molecular', []))
            original_features = list(cached_classification.get('original', []))

            # 验证缓存的特征是否仍然存在（使用集合加速）
            feature_set = set(feature_candidates)
            molecular_features = [f for f in molecular_features if f in feature_set]
            original_features = [f for f in original_features if f in feature_set]

            # 需要获取 classifier 用于后续操作
            classification, classifier = _get_classification(
                tuple(feature_candidates),
                tuple(recorded_mol_features),
                tuple(source_feature_names),
            )

        return molecular_features, original_features, classifier, recorded_mol_features

    # 只在需要特征分类的标签时才执行分类
    # 标签 0 (手动选择)、1 (特征分类纠错) 和 7 (工艺PLS) 需要分类结果
    needs_classification = tab_choice in [tab_titles[0], tab_titles[1], tab_titles[7]]

    if needs_classification:
        # 懒加载：只在需要时才执行特征分类
        molecular_features, original_features, classifier, recorded_mol_features = get_feature_classification()

        # 验证分类完整性
        classified_count = len(molecular_features) + len(original_features)
        classified_set = set(molecular_features) | set(original_features)

        if classified_count != total_features:
            # 找出遗漏的特征
            missing = [f for f in feature_candidates if f not in classified_set]

            if missing:
                st.warning(f"⚠️ 发现 {len(missing)} 个未分类特征，已自动归为原始特征")
                original_features.extend(missing)

        # 显示分类统计（调试信息）
        if st.session_state.get('_show_classification_debug', False):
            st.write(f"📊 分类统计: 总特征={total_features}, 分子特征={len(molecular_features)}, 原始特征={len(original_features)}")
            st.write(f"📊 recorded_mol_features: {len(recorded_mol_features)} 个")

    # --- Tab 1: 手动选择 ---
    if tab_choice == tab_titles[0]:
        st.markdown("#### 手动选择特征")
        
        # 显示分类统计
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("📊 总特征数", total_features)
        with col_stat2:
            st.metric("🧬 分子特征", len(molecular_features))
        with col_stat3:
            st.metric("📋 原始特征", len(original_features))
        with col_stat4:
            current_selected = len(st.session_state.get('feature_cols', []))
            st.metric("✅ 已选特征", current_selected)
        
        # 调试信息（可展开）
        with st.expander("🔍 调试信息", expanded=False):
            st.write(f"**recorded_mol_features (from molecular_feature_names):** {len(recorded_mol_features)} 个")
            if recorded_mol_features:
                st.write(f"前5个: {list(recorded_mol_features)[:5]}")
            st.write(f"**feature_candidates (当前数值列):** {total_features} 个")
            st.write(f"前5个: {feature_candidates[:5]}")
            st.write(f"**molecular_features (分类结果):** {len(molecular_features)} 个")
            if molecular_features:
                st.write(f"前5个: {molecular_features[:5]}")
            st.write(f"**original_features (分类结果):** {len(original_features)} 个")
            if original_features:
                st.write(f"前5个: {original_features[:5]}")
        
        st.markdown("---")
        
        # 快捷操作按钮
        st.markdown("##### ⚡ 快捷操作")
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)

        with col_btn1:
            def _select_all():
                st.session_state.feature_cols = feature_candidates.copy()
                st.session_state.multiselect_features = feature_candidates.copy()
                st.session_state['feature_selector_msg'] = f"✅ 已选择全部 {len(feature_candidates)} 个特征"
            st.button("✅ 全选所有", on_click=_select_all, type="primary")

        with col_btn2:
            def _select_molecular_only():
                st.session_state.feature_cols = molecular_features.copy()
                st.session_state.multiselect_features = molecular_features.copy()
                st.session_state['feature_selector_msg'] = f"✅ 已选择 {len(molecular_features)} 个分子特征"
            st.button("🧬 仅分子特征", on_click=_select_molecular_only)

        with col_btn3:
            def _select_original_only():
                st.session_state.feature_cols = original_features.copy()
                st.session_state.multiselect_features = original_features.copy()
                st.session_state['feature_selector_msg'] = f"✅ 已选择 {len(original_features)} 个原始特征"
            st.button("📋 仅原始特征", on_click=_select_original_only)

        with col_btn4:
            def _clear_all():
                st.session_state.feature_cols = []
                st.session_state.multiselect_features = []
                st.session_state['feature_selector_msg'] = "✅ 已清空所有选择"
            st.button("🗑️ 清空选择", on_click=_clear_all)

        with col_btn5:
            def _refresh_classification():
                # 强制重新分类
                if 'feature_classification' in st.session_state:
                    del st.session_state['feature_classification']
                st.session_state['feature_selector_msg'] = "🔄 已刷新特征分类"
            st.button("🔄 刷新分类", on_click=_refresh_classification,
                     help="提取新特征后点击此按钮刷新分类")

        with col_btn6:
            # [新增] 导出当前选择的特征
            current_selected_features = st.session_state.get('feature_cols', [])
            if current_selected_features:
                import io
                # 创建导出内容
                export_content = "# 当前选择的特征列表\n"
                export_content += f"# 导出时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_content += f"# 总特征数: {len(current_selected_features)}\n"
                export_content += f"# 目标变量: {target_col}\n\n"

                # 按分类导出
                selected_molecular = [f for f in current_selected_features if f in molecular_features]
                selected_original = [f for f in current_selected_features if f in original_features]

                if selected_molecular:
                    export_content += f"## 分子特征 ({len(selected_molecular)} 个)\n"
                    for feat in selected_molecular:
                        export_content += f"{feat}\n"
                    export_content += "\n"

                if selected_original:
                    export_content += f"## 原始特征 ({len(selected_original)} 个)\n"
                    for feat in selected_original:
                        export_content += f"{feat}\n"

                # 提供下载按钮
                st.download_button(
                    label="📥 导出特征",
                    data=export_content,
                    file_name=f"selected_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="导出当前选择的特征列表到文本文件"
                )
            else:
                st.button("📥 导出特征", disabled=True, help="请先选择特征")

        st.markdown("---")

        # [新增] 导入特征列表
        with st.expander("📤 导入特征列表", expanded=False):
            st.caption("上传之前导出的特征列表文件，快速恢复特征选择")

            uploaded_file = st.file_uploader(
                "选择特征列表文件",
                type=['txt', 'csv'],
                key="feature_import_file",
                help="支持 .txt 和 .csv 格式"
            )

            if uploaded_file is not None:
                try:
                    # 读取文件内容
                    content = uploaded_file.read().decode('utf-8')

                    # 解析特征名称（跳过注释行和空行）
                    imported_features = []
                    for line in content.split('\n'):
                        line = line.strip()
                        # 跳过注释、空行、标题行
                        if line and not line.startswith('#') and not line.startswith('##'):
                            # 处理可能的CSV格式
                            if ',' in line:
                                imported_features.extend([f.strip() for f in line.split(',') if f.strip()])
                            else:
                                imported_features.append(line)

                    # 去重
                    imported_features = list(dict.fromkeys(imported_features))

                    # 验证特征是否存在于当前数据集
                    valid_features = [f for f in imported_features if f in feature_candidates]
                    invalid_features = [f for f in imported_features if f not in feature_candidates]

                    # 显示导入统计
                    col_import1, col_import2, col_import3 = st.columns(3)
                    with col_import1:
                        st.metric("📋 文件中特征", len(imported_features))
                    with col_import2:
                        st.metric("✅ 有效特征", len(valid_features))
                    with col_import3:
                        st.metric("❌ 无效特征", len(invalid_features))

                    if invalid_features:
                        with st.expander("⚠️ 查看无效特征", expanded=False):
                            st.warning(f"以下 {len(invalid_features)} 个特征在当前数据集中不存在：")
                            st.code('\n'.join(invalid_features[:20]))
                            if len(invalid_features) > 20:
                                st.caption(f"... 还有 {len(invalid_features) - 20} 个")

                    if valid_features:
                        col_action1, col_action2, col_action3 = st.columns(3)

                        with col_action1:
                            if st.button("✅ 应用导入", type="primary", key="apply_import"):
                                st.session_state.feature_cols = valid_features.copy()
                                st.session_state.multiselect_features = valid_features.copy()
                                st.session_state['feature_selector_msg'] = f"✅ 已导入 {len(valid_features)} 个特征"
                                st.rerun()

                        with col_action2:
                            if st.button("➕ 追加到现有", key="append_import"):
                                existing = set(st.session_state.get('feature_cols', []))
                                combined = list(existing | set(valid_features))
                                st.session_state.feature_cols = combined
                                st.session_state.multiselect_features = combined.copy()
                                added_count = len(combined) - len(existing)
                                st.session_state['feature_selector_msg'] = f"✅ 已追加 {added_count} 个新特征（总计 {len(combined)} 个）"
                                st.rerun()

                        with col_action3:
                            if st.button("🔍 预览特征", key="preview_import"):
                                st.write("**有效特征列表：**")
                                st.code('\n'.join(valid_features[:50]))
                                if len(valid_features) > 50:
                                    st.caption(f"... 还有 {len(valid_features) - 50} 个")
                    else:
                        st.error("❌ 文件中没有有效的特征名称")

                except Exception as e:
                    st.error(f"❌ 文件解析失败: {str(e)}")
                    st.caption("请确保文件格式正确（每行一个特征名，或逗号分隔）")

        st.markdown("---")

        # 特征筛选器（用 form 包裹避免频繁 rerun）
        st.markdown("##### 🔍 特征筛选")
        with st.form("feature_filter_form"):
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                filter_type = st.selectbox(
                    "显示类型",
                    ["全部特征", "仅分子特征", "仅原始特征"],
                    index=st.session_state.get("_filter_type_idx", 0),
                    key="feature_filter_type_v2"
                )

            with col_f2:
                search_keyword = st.text_input(
                    "搜索特征名",
                    value=st.session_state.get("_search_keyword", ""),
                    placeholder="输入关键词...",
                    key="feature_search_v2"
                )

            with col_f3:
                prefix_filter = st.selectbox(
                    "按前缀筛选",
                    ["全部"] + list(classifier.prefix_counts.keys()),
                    index=st.session_state.get("_prefix_filter_idx", 0),
                    key="prefix_filter_v2"
                )

            apply_filter = st.form_submit_button("🔍 应用筛选")

        if apply_filter:
            st.session_state["_filter_type_idx"] = ["全部特征", "仅分子特征", "仅原始特征"].index(filter_type)
            st.session_state["_search_keyword"] = search_keyword
            prefix_options = ["全部"] + list(classifier.prefix_counts.keys())
            st.session_state["_prefix_filter_idx"] = prefix_options.index(prefix_filter)
            st.session_state["feature_page_num_v2"] = 1
            # 清除 multiselect widget 缓存，避免旧选项与新 options 冲突
            if "feature_multiselect_v2" in st.session_state:
                del st.session_state["feature_multiselect_v2"]

        # 应用筛选
        display_candidates = feature_candidates.copy()

        if filter_type == "仅分子特征":
            display_candidates = molecular_features
        elif filter_type == "仅原始特征":
            display_candidates = original_features

        if search_keyword:
            display_candidates = [f for f in display_candidates if search_keyword.lower() in f.lower()]

        if prefix_filter != "全部":
            display_candidates = [f for f in display_candidates if f in classifier.prefix_groups.get(prefix_filter, [])]

        # 判断是否有筛选条件
        has_filter = (filter_type != "全部特征" or search_keyword or prefix_filter != "全部")

        st.caption(f"当前显示: {len(display_candidates)} / {total_features} 个特征")

        # 快速操作按钮 - 使用 key 避免重复渲染
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if has_filter and display_candidates:
                if st.button("➕ 添加筛选结果", type="primary",
                           help="将当前筛选出的特征添加到已选列表",
                           use_container_width=True,
                           key="btn_add_filtered"):
                    current = set(st.session_state.get('feature_cols', []))
                    to_add = set(display_candidates)
                    added_count = len(to_add - current)
                    new_selection = list(current | to_add)
                    st.session_state.feature_cols = new_selection
                    st.session_state.multiselect_features = new_selection
                    if added_count > 0:
                        st.toast(f"✅ 已添加 {added_count} 个特征", icon="✅")

        with col_btn2:
            if has_filter and display_candidates:
                if st.button("🗑️ 剔除筛选结果", type="secondary",
                           help="从已选列表中移除当前筛选出的特征",
                           use_container_width=True,
                           key="btn_remove_filtered"):
                    current = set(st.session_state.get('feature_cols', []))
                    to_remove = set(display_candidates)
                    removed_count = len(current & to_remove)
                    new_selection = list(current - to_remove)
                    st.session_state.feature_cols = new_selection
                    st.session_state.multiselect_features = new_selection
                    if removed_count > 0:
                        st.toast(f"✅ 已剔除 {removed_count} 个特征", icon="🗑️")

        with col_btn3:
            if has_filter and display_candidates:
                if st.button("🎯 仅保留筛选结果", type="secondary",
                           help="清空其他选择，仅保留当前筛选结果",
                           use_container_width=True,
                           key="btn_only_filtered"):
                    new_selection = display_candidates.copy()
                    st.session_state.feature_cols = new_selection
                    st.session_state.multiselect_features = new_selection
                    st.toast(f"✅ 已设置为仅选中 {len(new_selection)} 个特征", icon="🎯")

        with col_btn4:
            if st.button("🧹 清空选择", type="secondary",
                       help="清空所有已选特征",
                       use_container_width=True,
                       key="btn_clear_all"):
                st.session_state.feature_cols = []
                st.session_state.multiselect_features = []
                st.toast("✅ 已清空所有选择", icon="🧹")

        # 初始化选择状态 - 改进版：处理空列表和无效特征
        if 'feature_cols' not in st.session_state or not st.session_state.feature_cols:
            # 首次进入或列表为空时，默认全选
            st.session_state.feature_cols = feature_candidates.copy()
        else:
            # 清理feature_cols中不存在的特征（使用集合加速）
            feature_set = set(feature_candidates)
            valid_feature_cols = [f for f in st.session_state.feature_cols if f in feature_set]
            if len(valid_feature_cols) != len(st.session_state.feature_cols):
                st.session_state.feature_cols = valid_feature_cols

        if 'multiselect_features' not in st.session_state or not st.session_state.multiselect_features:
            st.session_state.multiselect_features = st.session_state.feature_cols.copy()

        # 清理无效特征（使用集合加速）
        feature_set = set(feature_candidates)
        st.session_state.multiselect_features = [
            f for f in st.session_state.get('multiselect_features', [])
            if f in feature_set
        ]

        # 同步：如果multiselect_features为空但feature_cols不为空，则同步
        if not st.session_state.multiselect_features and st.session_state.feature_cols:
            st.session_state.multiselect_features = st.session_state.feature_cols.copy()

        # 特征多选（只显示筛选后的特征）
        display_set = set(display_candidates)
        current_display_selection = [f for f in st.session_state.multiselect_features if f in display_set]

        # 大特征集使用分页，而不是直接截断前 200 个
        if len(display_candidates) > 100:
            page_size_options = [100, 200, 500, 1000]
            default_page_size = 200 if len(display_candidates) > 200 else 100
            if default_page_size not in page_size_options:
                default_page_size = page_size_options[0]

            col_page1, col_page2, col_page3 = st.columns([1, 1, 2])
            with col_page1:
                page_size = st.selectbox(
                    "每页显示",
                    options=page_size_options,
                    index=page_size_options.index(st.session_state.get("feature_page_size_v2", default_page_size))
                    if st.session_state.get("feature_page_size_v2", default_page_size) in page_size_options
                    else page_size_options.index(default_page_size),
                    key="feature_page_size_v2",
                )

            total_pages = max(1, (len(display_candidates) - 1) // max(1, int(page_size)) + 1) if display_candidates else 1
            stored_page_num = int(st.session_state.get("feature_page_num_v2", 1) or 1)
            if stored_page_num > total_pages:
                stored_page_num = total_pages
                st.session_state["feature_page_num_v2"] = total_pages

            with col_page2:
                page_num = st.number_input(
                    "页码",
                    min_value=1,
                    max_value=total_pages,
                    value=stored_page_num,
                    step=1,
                    key="feature_page_num_v2",
                )

            start_idx = (int(page_num) - 1) * int(page_size)
            end_idx = min(len(display_candidates), start_idx + int(page_size))
            display_candidates_limited = display_candidates[start_idx:end_idx]
            current_display_selection = [f for f in current_display_selection if f in set(display_candidates_limited)]

            with col_page3:
                if display_candidates:
                    st.caption(
                        f"当前显示第 {int(page_num)}/{total_pages} 页，范围 {start_idx + 1}-{end_idx}，共 {len(display_candidates)} 个特征"
                    )
                else:
                    st.caption("当前筛选结果为空")
        else:
            display_candidates_limited = display_candidates

        page_signature = (
            len(display_candidates_limited),
            tuple(display_candidates_limited[:5]),
            tuple(display_candidates_limited[-5:]) if len(display_candidates_limited) > 5 else tuple(display_candidates_limited),
        )
        if st.session_state.get("_feature_multiselect_signature") != page_signature:
            st.session_state.pop("feature_multiselect_v2", None)
            st.session_state["_feature_multiselect_signature"] = page_signature

        with st.form("feature_selection_form_v2"):
            selected_in_display = st.multiselect(
                f"选择特征（当前显示 {len(display_candidates_limited)} 个）",
                options=display_candidates_limited,
                default=current_display_selection,
                key="feature_multiselect_v2",
                help="提示：如果选项过多导致卡顿，请使用上方筛选功能缩小范围"
            )
            confirm_sel = st.form_submit_button("💾 确认选择", type="primary")

        if confirm_sel:
            # 保留不在当前显示中的已选特征（使用集合加速）
            display_set = set(display_candidates_limited)
            hidden_selected = [f for f in st.session_state.multiselect_features if f not in display_set]
            new_selection = list(dict.fromkeys(selected_in_display + hidden_selected))
            st.session_state.feature_cols = new_selection
            st.session_state.multiselect_features = new_selection
            st.success(f"✅ 已确认选择 {len(new_selection)} 个特征")

        # 显示当前选择摘要（使用集合加速）
        if st.session_state.get('feature_cols'):
            st.markdown("---")
            st.markdown(f"**当前已选择 {len(st.session_state.feature_cols)} 个特征**")

            molecular_set = set(molecular_features)
            original_set = set(original_features)
            selected_mol = [f for f in st.session_state.feature_cols if f in molecular_set]
            selected_orig = [f for f in st.session_state.feature_cols if f in original_set]
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.info(f"🧬 分子特征: {len(selected_mol)} 个")
            with col_s2:
                st.info(f"📋 原始特征: {len(selected_orig)} 个")

    # --- Tab 2: 特征分类纠错 ---
    # --- Tab 2: 特征分类纠错 ---
    if tab_choice == tab_titles[1]:
        # 懒加载：只在特征分类纠错标签时才执行特征分类
        molecular_features, original_features, classifier, recorded_mol_features = get_feature_classification()

        st.markdown("#### 🔧 特征分类纠错")
        st.caption("如果自动分类有误，可以在这里手动调整。调整后会立即生效。")
        
        # 显示前缀统计
        st.markdown("##### 📊 前缀模式分析")
        st.caption("系统会自动将同一前缀下有大量特征（≥5个）的组识别为分子特征")
        
        prefix_df = classifier.get_prefix_summary()
        st.dataframe(prefix_df, use_container_width=True, height=300)
        
        st.markdown("---")
        
        # 批量调整
        st.markdown("##### 🔄 批量调整分类")
        
        col_adj1, col_adj2 = st.columns(2)
        
        with col_adj1:
            st.markdown("**将原始特征 → 分子特征**")
            orig_to_move = st.multiselect(
                "选择要移动的原始特征",
                options=original_features,
                key="orig_to_mol_v2"
            )
            if st.button("➡️ 移至分子特征", key="btn_move_to_mol"):
                if orig_to_move:
                    for f in orig_to_move:
                        if f in original_features:
                            original_features.remove(f)
                        if f not in molecular_features:
                            molecular_features.append(f)
                    st.session_state.feature_classification = {
                        'molecular': molecular_features,
                        'original': original_features
                    }
                    st.success(f"✅ 已将 {len(orig_to_move)} 个特征移至分子特征")
                    st.rerun()
        
        with col_adj2:
            st.markdown("**将分子特征 → 原始特征**")
            mol_to_move = st.multiselect(
                "选择要移动的分子特征",
                options=molecular_features,
                key="mol_to_orig_v2"
            )
            if st.button("➡️ 移至原始特征", key="btn_move_to_orig"):
                if mol_to_move:
                    for f in mol_to_move:
                        if f in molecular_features:
                            molecular_features.remove(f)
                        if f not in original_features:
                            original_features.append(f)
                    st.session_state.feature_classification = {
                        'molecular': molecular_features,
                        'original': original_features
                    }
                    st.success(f"✅ 已将 {len(mol_to_move)} 个特征移至原始特征")
                    st.rerun()
        
        st.markdown("---")
        
        # 按前缀批量调整（系统识别的前缀）
        st.markdown("##### 📦 按系统前缀批量调整")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            prefix_to_mol = st.selectbox(
                "选择前缀（移至分子特征）",
                options=[""] + list(classifier.prefix_counts.keys()),
                key="prefix_to_mol_v2"
            )
            if prefix_to_mol and st.button("📦 整组移至分子特征", key="btn_prefix_to_mol"):
                features_to_move = classifier.prefix_groups.get(prefix_to_mol, [])
                for f in features_to_move:
                    if f in original_features:
                        original_features.remove(f)
                    if f not in molecular_features:
                        molecular_features.append(f)
                st.session_state.feature_classification = {
                    'molecular': molecular_features,
                    'original': original_features
                }
                st.success(f"✅ 已将前缀 '{prefix_to_mol}' 下的 {len(features_to_move)} 个特征移至分子特征")
                st.rerun()
        
        with col_p2:
            prefix_to_orig = st.selectbox(
                "选择前缀（移至原始特征）",
                options=[""] + list(classifier.prefix_counts.keys()),
                key="prefix_to_orig_v2"
            )
            if prefix_to_orig and st.button("📦 整组移至原始特征", key="btn_prefix_to_orig"):
                features_to_move = classifier.prefix_groups.get(prefix_to_orig, [])
                for f in features_to_move:
                    if f in molecular_features:
                        molecular_features.remove(f)
                    if f not in original_features:
                        original_features.append(f)
                st.session_state.feature_classification = {
                    'molecular': molecular_features,
                    'original': original_features
                }
                st.success(f"✅ 已将前缀 '{prefix_to_orig}' 下的 {len(features_to_move)} 个特征移至原始特征")
                st.rerun()
        
        st.markdown("---")
        
        # ==========================================
        # 自定义前缀/模式匹配功能
        # ==========================================
        st.markdown("##### ✏️ 自定义前缀/模式匹配")
        st.caption("输入自定义前缀或关键词来匹配特征，支持多种匹配模式")
        
        col_custom1, col_custom2 = st.columns([2, 1])
        
        with col_custom1:
            custom_pattern = st.text_input(
                "输入前缀/关键词",
                value="",
                placeholder="例如: resin_, hardener_, fp_, MolWt...",
                key="custom_prefix_pattern",
                help="输入要匹配的前缀或关键词"
            )
        
        with col_custom2:
            match_mode = st.selectbox(
                "匹配模式",
                options=["前缀匹配", "包含匹配", "后缀匹配", "正则表达式"],
                index=0,
                key="custom_match_mode",
                help="前缀匹配：特征名以输入内容开头\n包含匹配：特征名包含输入内容\n后缀匹配：特征名以输入内容结尾\n正则表达式：使用正则表达式匹配"
            )
        
        # 执行匹配
        matched_features = []
        if custom_pattern:
            pattern = custom_pattern.strip()
            try:
                if match_mode == "前缀匹配":
                    matched_features = [f for f in feature_candidates if f.startswith(pattern) or f.lower().startswith(pattern.lower())]
                elif match_mode == "包含匹配":
                    matched_features = [f for f in feature_candidates if pattern in f or pattern.lower() in f.lower()]
                elif match_mode == "后缀匹配":
                    matched_features = [f for f in feature_candidates if f.endswith(pattern) or f.lower().endswith(pattern.lower())]
                elif match_mode == "正则表达式":
                    import re as regex_module
                    compiled = regex_module.compile(pattern, regex_module.IGNORECASE)
                    matched_features = [f for f in feature_candidates if compiled.search(f)]
            except Exception as e:
                st.error(f"匹配错误: {e}")
                matched_features = []
        
        # 显示匹配结果
        if custom_pattern:
            if matched_features:
                # 统计匹配结果中的分类情况
                matched_mol = [f for f in matched_features if f in molecular_features]
                matched_orig = [f for f in matched_features if f in original_features]
                
                st.success(f"✅ 匹配到 **{len(matched_features)}** 个特征")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("匹配总数", len(matched_features))
                with col_stat2:
                    st.metric("当前为分子特征", len(matched_mol))
                with col_stat3:
                    st.metric("当前为原始特征", len(matched_orig))
                
                # 预览匹配的特征（可展开）
                with st.expander(f"📋 查看匹配的特征列表（前50个）", expanded=False):
                    preview_list = matched_features[:50]
                    preview_df = pd.DataFrame({
                        '特征名': preview_list,
                        '当前分类': ['🧬 分子' if f in molecular_features else '📊 原始' for f in preview_list]
                    })
                    st.dataframe(preview_df, use_container_width=True, height=200)
                    if len(matched_features) > 50:
                        st.caption(f"... 还有 {len(matched_features) - 50} 个特征未显示")
                
                # 操作按钮
                col_act1, col_act2 = st.columns(2)
                
                with col_act1:
                    if st.button("🧬 全部移至分子特征", key="btn_custom_to_mol", type="primary"):
                        move_count = 0
                        for f in matched_features:
                            if f in original_features:
                                original_features.remove(f)
                                move_count += 1
                            if f not in molecular_features:
                                molecular_features.append(f)
                        st.session_state.feature_classification = {
                            'molecular': molecular_features,
                            'original': original_features
                        }
                        st.success(f"✅ 已将 {len(matched_features)} 个匹配特征移至分子特征（新移动 {move_count} 个）")
                        st.rerun()
                
                with col_act2:
                    if st.button("📊 全部移至原始特征", key="btn_custom_to_orig"):
                        move_count = 0
                        for f in matched_features:
                            if f in molecular_features:
                                molecular_features.remove(f)
                                move_count += 1
                            if f not in original_features:
                                original_features.append(f)
                        st.session_state.feature_classification = {
                            'molecular': molecular_features,
                            'original': original_features
                        }
                        st.success(f"✅ 已将 {len(matched_features)} 个匹配特征移至原始特征（新移动 {move_count} 个）")
                        st.rerun()
            else:
                st.warning(f"⚠️ 未匹配到任何特征，请检查输入的前缀/关键词")
        
        st.markdown("---")
        
        # ==========================================
        # 批量自定义前缀（多个前缀一次性处理）
        # ==========================================
        st.markdown("##### 📝 批量前缀处理")
        st.caption("一次性输入多个前缀，用逗号或换行分隔")
        
        batch_prefixes = st.text_area(
            "输入多个前缀（逗号或换行分隔）",
            value="",
            placeholder="例如:\nresin_fp_, resin_maccs_, resin_rdkit_\nhardener_fp_, hardener_maccs_",
            height=100,
            key="batch_prefixes_input"
        )
        
        if batch_prefixes.strip():
            # 解析多个前缀
            prefixes = []
            for line in batch_prefixes.split('\n'):
                for part in line.split(','):
                    p = part.strip()
                    if p:
                        prefixes.append(p)
            
            if prefixes:
                # 匹配所有前缀
                all_matched = set()
                prefix_match_counts = {}
                for prefix in prefixes:
                    matched = [f for f in feature_candidates if f.startswith(prefix) or f.lower().startswith(prefix.lower())]
                    prefix_match_counts[prefix] = len(matched)
                    all_matched.update(matched)
                
                all_matched = list(all_matched)
                
                st.info(f"📊 共输入 {len(prefixes)} 个前缀，匹配到 {len(all_matched)} 个特征")
                
                # 显示每个前缀的匹配情况
                with st.expander("查看各前缀匹配详情", expanded=False):
                    match_summary = pd.DataFrame([
                        {'前缀': p, '匹配数量': c} for p, c in prefix_match_counts.items()
                    ]).sort_values('匹配数量', ascending=False)
                    st.dataframe(match_summary, use_container_width=True)
                
                col_batch1, col_batch2 = st.columns(2)
                
                with col_batch1:
                    if st.button("🧬 批量移至分子特征", key="btn_batch_to_mol", type="primary"):
                        move_count = 0
                        for f in all_matched:
                            if f in original_features:
                                original_features.remove(f)
                                move_count += 1
                            if f not in molecular_features:
                                molecular_features.append(f)
                        st.session_state.feature_classification = {
                            'molecular': molecular_features,
                            'original': original_features
                        }
                        st.success(f"✅ 已将 {len(all_matched)} 个特征移至分子特征")
                        st.rerun()
                
                with col_batch2:
                    if st.button("📊 批量移至原始特征", key="btn_batch_to_orig"):
                        move_count = 0
                        for f in all_matched:
                            if f in molecular_features:
                                molecular_features.remove(f)
                                move_count += 1
                            if f not in original_features:
                                original_features.append(f)
                        st.session_state.feature_classification = {
                            'molecular': molecular_features,
                            'original': original_features
                        }
                        st.success(f"✅ 已将 {len(all_matched)} 个特征移至原始特征")
                        st.rerun()
        
        # 重置分类
        st.markdown("---")
        if st.button("🔄 重置为自动分类", key="btn_reset_classification"):
            new_classifier = SmartFeatureClassifier(
                feature_candidates,
                recorded_mol_features,
                set(st.session_state.get('source_feature_names', [])),
            )
            new_classification = new_classifier.get_classification()
            st.session_state.feature_classification = {
                'molecular': new_classification['molecular'],
                'original': new_classification['original']
            }
            st.success("✅ 已重置为自动分类")
            st.rerun()

    # --- Tab 3: 方差筛选 ---
    if tab_choice == tab_titles[2]:
        st.markdown("#### 📉 方差筛选")
        st.caption("移除方差过低的特征（常量或近常量特征）")
        
        threshold = st.slider("方差阈值", 0.0, 1.0, 0.01, 0.001, key="var_threshold_v2")
        
        if st.button("应用方差筛选", key="btn_var_filter_v2"):
            _apply_variance_filter_callback(
                numeric_df[feature_candidates], 
                feature_candidates, 
                "var_threshold_v2"
            )
            st.rerun()

    # --- Tab 4: 相关性筛选 ---
    if tab_choice == tab_titles[3]:
        st.markdown("#### 🔗 相关性筛选")
        st.caption("选择与目标变量相关性最高的 Top-K 特征")
        
        k_corr = st.slider("Top-K", 5, min(500, total_features), min(50, total_features), key="corr_k_v2")
        
        if st.button("应用相关性筛选", key="btn_corr_filter_v2"):
            target_series = current_df[target_col]
            _apply_correlation_filter_callback(
                numeric_df[feature_candidates],
                target_series,
                "corr_k_v2"
            )
            st.rerun()

    # --- Tab 5: RFE递归消除 ---
    if tab_choice == tab_titles[4]:
        st.markdown("#### 🌀 RFE 递归特征消除")
        
        col_rfe1, col_rfe2 = st.columns(2)
        with col_rfe1:
            rfe_mode = st.selectbox("模式", ["RFECV自动（推荐）", "RFE手动指定"], key="rfe_mode_v2")
            rfe_estimator = st.selectbox("基估计器", ["随机森林", "ExtraTrees", "Ridge", "Lasso"], key="rfe_est_v2")
        
        with col_rfe2:
            if rfe_mode == "RFE手动指定":
                rfe_k = st.number_input("目标特征数", 5, total_features, min(30, total_features), key="rfe_k_v2")
            else:
                rfe_cv = st.number_input("CV折数", 2, 10, 5, key="rfe_cv_v2")
                rfe_min_k = st.number_input("最小特征数", 1, total_features, 5, key="rfe_min_k_v2")
            rfe_step = st.number_input("每步消除数", 1, 10, 1, key="rfe_step_v2")
        
        if st.button("运行RFE", type="primary", key="btn_rfe_v2"):
            with st.spinner("正在运行RFE..."):
                _apply_rfe_filter_callback(
                    numeric_df, current_df[target_col], feature_candidates,
                    "rfe_mode_v2", "rfe_est_v2", "rfe_k_v2", "rfe_step_v2",
                    "rfe_cv_v2", "r2", "rfe_min_k_v2"
                )
            st.rerun()

    # --- Tab 6: FSFS去冗余 ---
    if tab_choice == tab_titles[5]:
        # 懒加载：只在 FSFS 标签时才执行特征分类
        molecular_features, original_features, classifier, recorded_mol_features = get_feature_classification()

        st.markdown("#### 🎯 FSFS 特征选择（去除冗余特征）")

        st.markdown("""
        **FSFS (Feature Selection via Feature Similarity)** 算法通过分析特征相似性来选择特征：

        ✅ **优势：**
        - 选择与目标变量相关性高的特征
        - 自动去除冗余特征（高度相似的特征）
        - 保持特征多样性，避免信息重复

        💡 **适用场景：**
        - 特征间存在高度相关性
        - 需要减少特征数量但保持模型性能
        - 想要理解哪些特征是冗余的
        """)

        # [新增] 选择筛选范围
        col_scope1, col_scope2 = st.columns(2)

        with col_scope1:
            fsfs_scope = st.radio(
                "筛选范围",
                ["全部特征", "已选择的特征", "按前缀筛选"],
                index=0,
                horizontal=False,
                key="fsfs_scope",
                help="选择在哪些特征中进行FSFS筛选"
            )

        with col_scope2:
            # 前缀筛选选项
            if fsfs_scope == "按前缀筛选":
                # 获取所有前缀
                prefix_list = sorted(classifier.prefix_counts.keys())
                if prefix_list:
                    selected_prefixes = st.multiselect(
                        "选择要筛选的前缀",
                        options=prefix_list,
                        default=[],
                        key="fsfs_prefix_filter",
                        help="选择一个或多个前缀，只对这些前缀的特征进行FSFS筛选"
                    )
                    # 显示每个前缀的特征数量
                    if selected_prefixes:
                        prefix_info = []
                        for prefix in selected_prefixes:
                            count = classifier.prefix_counts.get(prefix, 0)
                            prefix_info.append(f"{prefix}: {count}个")
                        st.caption("📊 " + ", ".join(prefix_info))
                else:
                    st.warning("⚠️ 未检测到特征前缀")
                    selected_prefixes = []

        # 根据选择确定候选特征
        if fsfs_scope == "已选择的特征":
            current_selected = st.session_state.get('feature_cols', [])
            if not current_selected:
                st.warning("⚠️ 尚未选择任何特征，请先在「手动选择」标签页选择特征")
                st.info("💡 或者选择「全部特征」进行筛选")
                fsfs_candidates = []
            else:
                fsfs_candidates = [f for f in current_selected if f in feature_candidates]
                st.info(f"📊 将从已选择的 {len(fsfs_candidates)} 个特征中筛选")
        elif fsfs_scope == "按前缀筛选":
            if not selected_prefixes:
                st.warning("⚠️ 请至少选择一个前缀")
                fsfs_candidates = []
            else:
                # 获取选中前缀的所有特征
                fsfs_candidates = []
                for prefix in selected_prefixes:
                    prefix_features = classifier.prefix_groups.get(prefix, [])
                    fsfs_candidates.extend([f for f in prefix_features if f in feature_candidates])
                fsfs_candidates = list(set(fsfs_candidates))  # 去重
                st.info(f"📊 将从选中前缀的 {len(fsfs_candidates)} 个特征中筛选")
        else:
            fsfs_candidates = feature_candidates
            st.info(f"📊 将从全部 {len(fsfs_candidates)} 个数值特征中筛选")

        if not fsfs_candidates:
            return

        col_fsfs1, col_fsfs2 = st.columns(2)

        with col_fsfs1:
            fsfs_n_features = st.number_input(
                "选择特征数量",
                min_value=1,
                max_value=len(fsfs_candidates),
                value=min(20, len(fsfs_candidates)),
                key="fsfs_n_features",
                help="要保留的特征数量"
            )

            fsfs_threshold = st.slider(
                "相似度阈值",
                0.5, 0.95, 0.8, 0.05,
                key="fsfs_threshold",
                help="特征相似度超过此值将被视为冗余。推荐：0.7-0.9"
            )

        with col_fsfs2:
            fsfs_importance = st.selectbox(
                "重要性度量",
                ["correlation", "mutual_info", "variance"],
                index=0,
                key="fsfs_importance",
                help="correlation: 相关系数（快速，推荐）\nmutual_info: 互信息（慢但准确）\nvariance: 方差（最快）"
            )

            fsfs_similarity = st.selectbox(
                "相似度度量",
                ["correlation", "spearman", "cosine"],
                index=0,
                key="fsfs_similarity",
                help="correlation: Pearson相关（推荐）\nspearman: Spearman秩相关\ncosine: 余弦相似度"
            )

        # [新增] 快速模式选项
        if len(fsfs_candidates) > 1000:
            st.warning(f"⚠️ 特征数量较多（{len(fsfs_candidates)}个），建议启用快速模式")

            use_fast_mode = st.checkbox(
                "启用快速模式（先预筛选再FSFS）",
                value=True,
                key="fsfs_fast_mode",
                help="对于大量特征，先用快速方法预筛选到30%，再用FSFS精选。可大幅提升速度。"
            )

            if use_fast_mode:
                prefilter_ratio = st.slider(
                    "预筛选保留比例",
                    0.1, 0.5, 0.3, 0.05,
                    key="fsfs_prefilter_ratio",
                    help="预筛选阶段保留的特征比例。例如0.3表示保留30%的特征。"
                )
                st.info(f"💡 将先筛选到约 {int(len(fsfs_candidates) * prefilter_ratio)} 个特征，再用FSFS精选")
        else:
            use_fast_mode = False
            prefilter_ratio = 0.3

        if st.button("运行FSFS", type="primary", key="btn_fsfs"):
            try:
                from core.fsfs_selector import FSFSSelector
                import os
                import multiprocessing

                with st.spinner("正在运行FSFS特征选择..."):
                    # 准备数据 - 使用fsfs_candidates而不是feature_candidates
                    X = numeric_df[fsfs_candidates]
                    y = current_df[target_col]

                    # [性能修复] 临时解除线程限制，启用真正的多线程
                    original_omp = os.environ.get('OMP_NUM_THREADS')
                    original_mkl = os.environ.get('MKL_NUM_THREADS')
                    original_openblas = os.environ.get('OPENBLAS_NUM_THREADS')

                    try:
                        # 设置为使用所有CPU核心
                        n_cores = str(multiprocessing.cpu_count())
                        os.environ['OMP_NUM_THREADS'] = n_cores
                        os.environ['MKL_NUM_THREADS'] = n_cores
                        os.environ['OPENBLAS_NUM_THREADS'] = n_cores

                        st.info(f"⚡ 已启用多线程加速（{n_cores}核心）")

                        # 显示数据规模信息
                        st.info(f"数据规模: {X.shape[0]} 样本 × {X.shape[1]} 特征")

                        # [新增] 快速模式：先预筛选
                        if use_fast_mode and X.shape[1] > 1000:
                            from fast_fsfs import fast_fsfs_with_prefilter

                            st.info(f"🚀 使用快速模式（预筛选比例: {prefilter_ratio*100:.0f}%）")

                            # 拟合并选择特征（启用详细输出）
                            import sys
                            from io import StringIO

                            # 捕获详细输出
                            old_stdout = sys.stdout
                            sys.stdout = mystdout = StringIO()

                            final_indices, fsfs, selected_features = fast_fsfs_with_prefilter(
                                X, y,
                                n_features=fsfs_n_features,
                                similarity_threshold=fsfs_threshold,
                                prefilter_ratio=prefilter_ratio,
                                importance_metric=fsfs_importance,
                                similarity_metric=fsfs_similarity,
                                n_jobs=-1,
                                verbose=True
                            )

                            # [调试] 验证选择的特征是否在候选范围内
                            if fsfs_scope == "已选择的特征":
                                invalid_features = [f for f in selected_features if f not in fsfs_candidates]
                                if invalid_features:
                                    st.error(f"⚠️ 检测到 {len(invalid_features)} 个不在候选范围内的特征")
                                    st.write("不在范围内的特征:", invalid_features[:10])
                                    # 过滤掉不在范围内的特征
                                    selected_features = [f for f in selected_features if f in fsfs_candidates]
                                    st.info(f"已过滤，剩余 {len(selected_features)} 个特征")

                        else:
                            # 标准模式
                            # 创建FSFS选择器（启用并行计算）
                            fsfs = FSFSSelector(
                                n_features=fsfs_n_features,
                                similarity_threshold=fsfs_threshold,
                                importance_metric=fsfs_importance,
                                similarity_metric=fsfs_similarity,
                                task_type='regression',
                                random_state=42,
                                n_jobs=-1  # 使用所有CPU核心
                            )

                            # 拟合并选择特征（启用详细输出）
                            import sys
                            from io import StringIO

                            # 捕获详细输出
                            old_stdout = sys.stdout
                            sys.stdout = mystdout = StringIO()

                            fsfs.fit(X, y, verbose=True)

                            selected_features = fsfs.selected_features_

                            # [调试] 验证选择的特征是否在候选范围内
                            if fsfs_scope == "已选择的特征":
                                invalid_features = [f for f in selected_features if f not in fsfs_candidates]
                                if invalid_features:
                                    st.error(f"⚠️ 检测到 {len(invalid_features)} 个不在候选范围内的特征")
                                    st.write("不在范围内的特征:", invalid_features[:10])
                                    # 过滤掉不在范围内的特征
                                    selected_features = [f for f in selected_features if f in fsfs_candidates]
                                    st.info(f"已过滤，剩余 {len(selected_features)} 个特征")

                    finally:
                        # [性能修复] 恢复原始线程设置
                        if original_omp:
                            os.environ['OMP_NUM_THREADS'] = original_omp
                        if original_mkl:
                            os.environ['MKL_NUM_THREADS'] = original_mkl
                        if original_openblas:
                            os.environ['OPENBLAS_NUM_THREADS'] = original_openblas

                    # 恢复标准输出
                    sys.stdout = old_stdout
                    verbose_output = mystdout.getvalue()

                    # 显示详细信息
                    if verbose_output:
                        with st.expander("查看详细执行信息"):
                            st.code(verbose_output)

                    # 更新session_state
                    st.session_state['feature_cols'] = selected_features
                    st.session_state['feature_selector_msg'] = f"✅ FSFS完成：已选 {len(selected_features)} 个特征"

                    # 显示结果
                    st.success(f"✅ 已选择 {len(selected_features)} 个特征")

                    # 显示特征信息
                    feature_info = fsfs.get_feature_info()
                    st.markdown("##### 选择的特征及重要性")
                    st.dataframe(
                        feature_info,
                        use_container_width=True,
                        height=300
                    )

                    # 显示冗余特征信息
                    if len(selected_features) < len(feature_candidates):
                        st.markdown("##### 被排除的冗余特征（前10个）")
                        redundancy_info = fsfs.get_redundancy_info(top_n=10)
                        if len(redundancy_info) > 0:
                            st.dataframe(
                                redundancy_info,
                                use_container_width=True,
                                height=200
                            )
                            st.caption("这些特征因与已选特征高度相似而被排除")

                    # 可视化特征相似度矩阵
                    with st.expander("📊 查看特征相似度矩阵", expanded=False):
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        selected_indices = fsfs.selected_indices_
                        sim_matrix = fsfs.similarity_matrix_[np.ix_(selected_indices, selected_indices)]

                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            sim_matrix,
                            xticklabels=selected_features,
                            yticklabels=selected_features,
                            cmap='coolwarm',
                            center=0,
                            vmin=-1,
                            vmax=1,
                            square=True,
                            ax=ax,
                            cbar_kws={'label': 'Similarity'}
                        )
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.title('Feature Similarity Matrix (Selected Features)')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                st.rerun()

            except Exception as e:
                st.error(f"❌ FSFS运行失败: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())

    # --- Tab 7: PCA降维 ---
    if tab_choice == tab_titles[6]:
        # 懒加载：只在 PCA 标签时才执行特征分类
        molecular_features, original_features, classifier, recorded_mol_features = get_feature_classification()

        st.markdown("#### 🧩 PCA 主成分分析")

        # ==========================================
        # 🎯 RDKit特征一键优化
        # ==========================================
        with st.expander("🚀 RDKit描述符一键优化（仅针对描述符，不含指纹）", expanded=False):
            st.markdown("""
            **智能识别RDKit描述符并自动降维**

            ✅ **适用于：** RDKit 2D/3D描述符（如MolWt, TPSA, NumHDonors, fr_NH2等）

            ❌ **不适用于：** 指纹特征（MACCS, Morgan, ECFP等）

            💡 **提示：** 如果你有指纹特征，请使用下方的"批量PCA降维（按前缀组）"

            **功能：**
            - 自动检测RDKit描述符列
            - 保留95%方差，大幅减少特征数量
            - 保留其他特征（指纹、工艺参数等）不变
            - 生成详细的降维分析报告
            """)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                rdkit_variance = st.slider(
                    "累计解释方差阈值",
                    0.80, 0.99, 0.95, 0.01,
                    key="rdkit_pca_variance",
                    help="保留多少比例的信息"
                )
            with col_opt2:
                rdkit_min_pc = st.number_input(
                    "最小主成分数",
                    2, 20, 5,
                    key="rdkit_pca_min_pc",
                    help="至少保留多少个主成分"
                )

            rdkit_prefix = st.text_input(
                "主成分前缀",
                value="RDKit_PC",
                key="rdkit_pca_prefix",
                help="生成的主成分列名前缀"
            )

            if st.button("🎯 一键优化RDKit描述符", type="primary", key="btn_rdkit_optimize"):
                try:
                    from core.rdkit_pca_optimizer import RDKitPCAOptimizer

                    with st.spinner("正在分析和优化RDKit描述符..."):
                        # 创建优化器
                        optimizer = RDKitPCAOptimizer(
                            variance_threshold=rdkit_variance,
                            min_components=int(rdkit_min_pc)
                        )

                        # 执行优化
                        df_optimized, stats = optimizer.fit_transform(
                            current_df,
                            rdkit_cols=None,  # 自动检测
                            prefix=rdkit_prefix
                        )

                        # 显示检测到的特征
                        detected_cols = optimizer.rdkit_cols
                        st.info(f"🔍 检测到 {len(detected_cols)} 个RDKit描述符特征")

                        if len(detected_cols) > 0:
                            with st.expander("查看检测到的特征", expanded=False):
                                st.write(detected_cols[:50])  # 显示前50个
                                if len(detected_cols) > 50:
                                    st.caption(f"...还有 {len(detected_cols) - 50} 个特征")

                        # 显示未检测到的数值特征（可能遗漏的RDKit特征）
                        all_numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
                        if target_col in all_numeric_cols:
                            all_numeric_cols.remove(target_col)

                        undetected_cols = [c for c in all_numeric_cols if c not in detected_cols]

                        if undetected_cols:
                            st.warning(f"⚠️ 有 {len(undetected_cols)} 个数值特征未被识别为RDKit描述符")
                            with st.expander("查看未识别的特征（可能遗漏）", expanded=False):
                                st.write(undetected_cols[:50])
                                if len(undetected_cols) > 50:
                                    st.caption(f"...还有 {len(undetected_cols) - 50} 个特征")

                                st.info("""
                                **如果这些特征确实是RDKit描述符：**
                                1. 可以手动指定特征列表
                                2. 或者使用"标准PCA"功能手动选择特征
                                3. 或者联系开发者更新识别规则
                                """)

                        # 显示统计信息
                        st.success("✅ RDKit描述符优化完成！")

                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("原始RDKit特征", stats['original_rdkit_features'])
                        col_s2.metric("降维后主成分", stats['n_components'])
                        col_s3.metric("压缩比", f"{stats['compression_ratio']:.1f}x")
                        col_s4.metric("保留信息", f"{stats['total_variance_explained']:.1%}")

                        # 生成可视化
                        fig = optimizer.plot_analysis()
                        st.pyplot(fig, use_container_width=True)

                        # 显示特征重要性
                        with st.expander("📊 特征重要性 (Top 20)", expanded=False):
                            top_features = optimizer.get_top_features(20)
                            st.dataframe(top_features, use_container_width=True, height=400)

                        # 保存到session_state供后续应用
                        st.session_state['_rdkit_opt_result'] = {
                            'df_optimized': df_optimized,
                            'stats': stats,
                            'optimizer': optimizer
                        }

                except ImportError as e:
                    st.error(f"❌ 导入失败: {e}")
                except ValueError as e:
                    st.warning(f"⚠️ {e}")
                    st.info("""
                    **未检测到RDKit描述符特征**

                    RDKit描述符通常包含这些名称：
                    - 分子性质：MolWt, MolLogP, TPSA
                    - 拓扑指数：Chi, Kappa, BalabanJ
                    - 功能团计数：fr_NH2, fr_COO, fr_benzene

                    如果你的特征是指纹（MACCS, Morgan等），请使用下方的"批量PCA降维"功能。
                    """)
                except Exception as e:
                    st.error(f"❌ 优化失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # 应用优化结果（移到外面，避免嵌套）
            if st.session_state.get('_rdkit_opt_result') is not None:
                result = st.session_state['_rdkit_opt_result']
                st.info("💡 已生成优化结果，点击下方按钮应用到数据")

                if st.button("✅ 应用RDKit优化结果到数据", type="primary", key="btn_apply_rdkit_opt_final"):
                    df_optimized = result['df_optimized']
                    stats = result['stats']
                    optimizer = result['optimizer']

                    # 更新数据
                    st.session_state.processed_data = df_optimized

                    # 更新特征列表
                    new_feature_cols = [c for c in df_optimized.columns if c != target_col]
                    st.session_state.feature_cols = new_feature_cols
                    st.session_state.multiselect_features = new_feature_cols.copy()

                    # 清除特征分类缓存，强制重新分类
                    if 'feature_classification' in st.session_state:
                        del st.session_state['feature_classification']

                    # 保存优化器（用于后续transform）
                    st.session_state['rdkit_pca_optimizer'] = optimizer

                    # 清除优化结果缓存
                    del st.session_state['_rdkit_opt_result']

                    st.success(f"✅ 已应用优化！特征数: {stats['original_rdkit_features']} → {stats['total_features']}")
                    st.info("🔄 页面将刷新以显示新特征...")
                    st.rerun()

        st.markdown("---")

        scope_mode = st.radio(
            "降维范围",
            ["全部数值特征", "仅当前已选特征"],
            horizontal=True,
            key="pca_scope_v2"
        )
        if scope_mode == "仅当前已选特征":
            selected = st.session_state.get('feature_cols') or []
            pca_features = [f for f in feature_candidates if f in selected]
        else:
            pca_features = feature_candidates.copy()

        if len(pca_features) < 2:
            st.warning("⚠️ 可用于PCA的特征数不足（至少需要2个特征）")
        else:
            n_samples = numeric_df.shape[0]
            max_components = min(50, len(pca_features), max(2, n_samples - 1))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pca_mode = st.radio(
                    "主成分设置",
                    ["指定数量", "累计解释方差", "自动MLE"],
                    horizontal=True,
                    key="pca_mode_v2"
                )
            with col_p2:
                pc_prefix = st.text_input("主成分前缀", value="PC", key="pca_prefix_v2")

            n_components = None
            if pca_mode == "指定数量":
                n_components = st.slider("主成分数量", 2, max_components, min(10, max_components), key="pca_n_v2")
            elif pca_mode == "累计解释方差":
                var_ratio = st.slider("累计解释方差阈值", 0.50, 0.99, 0.95, 0.01, key="pca_var_v2")
                n_components = float(var_ratio)
            else:
                n_components = "mle"

            with st.expander("高级参数", expanded=False):
                scaler_mode_ui = st.selectbox(
                    "特征缩放",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler", "不缩放"],
                    index=0,
                    key="pca_scale_v2"
                )
                fill_strategy_ui = st.selectbox(
                    "缺失值处理",
                    ["均值", "中位数", "填0"],
                    index=0,
                    key="pca_fill_v2"
                )
                svd_solver = st.selectbox(
                    "SVD求解器",
                    ["auto", "full", "randomized", "arpack", "covariance_eigh"],
                    index=0,
                    key="pca_solver_v2"
                )
                whiten = st.checkbox("Whiten", value=False, key="pca_whiten_v2")
                random_state = st.number_input("随机种子", 0, 1000000, 42, key="pca_rs_v2")
                iterated_power_ui = st.selectbox("iterated_power(仅randomized)", ["auto", 2, 3, 5, 7], key="pca_iter_v2")

            scaler_mode = "standard"
            if scaler_mode_ui == "MinMaxScaler":
                scaler_mode = "minmax"
            elif scaler_mode_ui == "RobustScaler":
                scaler_mode = "robust"
            elif scaler_mode_ui == "不缩放":
                scaler_mode = "none"

            fill_strategy = "mean"
            if fill_strategy_ui == "中位数":
                fill_strategy = "median"
            elif fill_strategy_ui == "填0":
                fill_strategy = "zero"

            solver = svd_solver
            if n_components == "mle":
                solver = "full"
            if isinstance(n_components, float) and solver == "arpack":
                solver = "full"

            iterated_power = iterated_power_ui if iterated_power_ui == "auto" else int(iterated_power_ui)

            if st.button("分析PCA", key="btn_pca_analyze_v2"):
                X = _impute_df(numeric_df[pca_features].copy(), fill_strategy)
                scaler = _build_scaler(scaler_mode)
                if scaler is not None:
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                X_scaled, bad_cols = _ensure_finite_matrix(X_scaled, X.columns.tolist())
                _warn_pca_fallback("PCA分析", bad_cols)

                if n_components == "mle" and X_scaled.shape[0] <= X_scaled.shape[1]:
                    st.error("❌ 自动MLE需要样本数大于特征数，请调整参数")
                    return

                pca = PCA(
                    n_components=n_components,
                    whiten=bool(whiten),
                    svd_solver=solver,
                    random_state=int(random_state),
                    iterated_power=iterated_power,
                )
                pca.fit(X_scaled)

                explained = pca.explained_variance_ratio_
                cum_explained = np.cumsum(explained)
                st.markdown(f"**累计解释方差: {cum_explained[-1]:.2%}**")
                st.caption(f"主成分数: {len(explained)} / 特征数: {len(pca_features)}")

                # --- 碎石图 (Scree Plot) ---
                import matplotlib.pyplot as plt
                try:
                    from .plot_style import apply_global_style
                    apply_global_style()
                except Exception:
                    pass

                n_show = min(30, len(explained))
                fig_scree, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # 左图：碎石图（柱状 + 折线）
                x_pos = np.arange(1, n_show + 1)
                ax1.bar(x_pos, explained[:n_show], alpha=0.7, color='steelblue',
                        edgecolor='black', linewidth=0.5, label='单个方差')
                ax1.plot(x_pos, explained[:n_show], 'o-', color='darkred',
                         markersize=4, linewidth=1.5, label='趋势线')
                # 标注前5个
                for i in range(min(5, n_show)):
                    ax1.text(i + 1, explained[i], f'{explained[i]:.1%}',
                             ha='center', va='bottom', fontsize=8)
                ax1.set_xlabel('主成分', fontsize=11)
                ax1.set_ylabel('解释方差比例', fontsize=11)
                ax1.set_title('碎石图 (Scree Plot)', fontsize=13, fontweight='bold')
                ax1.legend(fontsize=9)
                ax1.grid(axis='y', alpha=0.3)
                if n_show > 20:
                    ax1.set_xticks(x_pos[::2])

                # 右图：累计解释方差曲线
                n_cum = len(cum_explained)
                x_cum = np.arange(1, n_cum + 1)
                ax2.plot(x_cum, cum_explained, 'o-', color='darkred',
                         markersize=4, linewidth=2)
                ax2.fill_between(x_cum, cum_explained, alpha=0.15, color='darkred')
                # 阈值线
                for thr, ls, clr in [(0.90, ':', 'orange'), (0.95, '--', 'green'), (0.99, '-.', 'blue')]:
                    if cum_explained[-1] >= thr:
                        idx_thr = int(np.searchsorted(cum_explained, thr))
                        ax2.axhline(y=thr, color=clr, linestyle=ls, linewidth=1.5,
                                    label=f'{thr:.0%} → {idx_thr + 1} 个PC')
                        ax2.axvline(x=idx_thr + 1, color=clr, linestyle=ls,
                                    linewidth=1, alpha=0.5)
                ax2.set_xlabel('主成分数量', fontsize=11)
                ax2.set_ylabel('累计解释方差', fontsize=11)
                ax2.set_title('累计解释方差曲线', fontsize=13, fontweight='bold')
                ax2.legend(fontsize=9, loc='lower right')
                ax2.grid(alpha=0.3)
                ax2.set_ylim(0, 1.05)

                fig_scree.tight_layout()
                st.pyplot(fig_scree, use_container_width=True)
                plt.close(fig_scree)

                # 数据表格（可折叠）
                with st.expander("📋 查看各主成分方差明细"):
                    chart_df = pd.DataFrame({
                        "主成分": [f"PC{i+1}" for i in range(len(explained))],
                        "解释方差": [f"{v:.4%}" for v in explained],
                        "累计方差": [f"{v:.4%}" for v in cum_explained],
                    })
                    st.dataframe(chart_df, use_container_width=True, hide_index=True)

                st.session_state['_pca_model'] = pca
                st.session_state['_pca_scaler'] = scaler
                st.session_state['_pca_ready'] = True
                st.session_state['_pca_features'] = pca_features
                st.session_state['_pca_fill'] = fill_strategy
                st.session_state['_pca_prefix'] = (pc_prefix or "PC").strip()

            if st.session_state.get('_pca_ready'):
                st.warning("⚠️ 应用PCA将替换所选特征为主成分")
                if st.button("🚀 应用PCA转换", type="primary", key="btn_pca_apply_v2"):
                    _apply_pca_callback(
                        st.session_state['_pca_model'],
                        st.session_state['_pca_scaler'],
                        numeric_df[st.session_state.get('_pca_features', pca_features)],
                        current_df,
                        st.session_state.get('_pca_features', pca_features),
                        fill_strategy=st.session_state.get('_pca_fill', "mean"),
                        pc_prefix=st.session_state.get('_pca_prefix', "PC")
                    )
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 📦 批量PCA降维（按前缀组）")
            st.caption("按前缀分组批量降维，适合指纹/描述符等高维特征组。")

            min_group_size = st.slider("最小组内特征数", 2, 500, 10, key="pca_batch_min_v2")
            group_scope = st.selectbox(
                "前缀范围",
                ["全部前缀", "仅分子特征前缀", "仅原始特征前缀"],
                index=1,
                key="pca_batch_scope_v2"
            )

            prefix_options = []
            for pfx, cnt in classifier.prefix_counts.items():
                if cnt < int(min_group_size):
                    continue
                is_mol = classifier._is_molecular_prefix(pfx, cnt)
                if group_scope == "仅分子特征前缀" and not is_mol:
                    continue
                if group_scope == "仅原始特征前缀" and is_mol:
                    continue
                prefix_options.append(pfx)

            default_prefixes = prefix_options[:5] if len(prefix_options) > 5 else prefix_options
            selected_prefixes = st.multiselect(
                "选择前缀组",
                options=prefix_options,
                default=default_prefixes,
                key="pca_batch_prefixes_v2"
            )

            batch_mode = st.radio(
                "批量主成分设置",
                ["统一数量", "统一解释方差"],
                horizontal=True,
                key="pca_batch_mode_v2"
            )
            if batch_mode == "统一数量":
                batch_n = st.number_input("每组保留主成分数", 2, 500, 10, key="pca_batch_n_v2")
                batch_n_components = int(batch_n)
            else:
                batch_ratio = st.slider("累计解释方差阈值", 0.50, 0.99, 0.95, 0.01, key="pca_batch_ratio_v2")
                batch_n_components = float(batch_ratio)

            keep_original = st.checkbox("保留原始特征（仅追加主成分）", value=False, key="pca_batch_keep_v2")

            if st.button("🚀 执行批量PCA降维", key="btn_pca_batch_apply_v2"):
                if not selected_prefixes:
                    st.error("❌ 请选择至少一个前缀组")
                else:
                    solver_batch = svd_solver
                    if isinstance(batch_n_components, float) and solver_batch == "arpack":
                        solver_batch = "full"
                    _apply_batch_pca_callback(
                        current_df=current_df,
                        numeric_df=numeric_df,
                        feature_candidates=feature_candidates,
                        prefix_groups=classifier.prefix_groups,
                        selected_prefixes=selected_prefixes,
                        n_components=batch_n_components,
                        scaler_mode=scaler_mode,
                        fill_strategy=fill_strategy,
                        svd_solver=solver_batch,
                        whiten=bool(whiten),
                        random_state=int(random_state),
                        iterated_power=iterated_power,
                        keep_original=bool(keep_original),
                    )
                    st.rerun()

            if st.session_state.get("pca_batch_report") is not None:
                st.markdown("##### 📊 批量PCA摘要")
                batch_report = st.session_state.get("pca_batch_report")
                # 显示表格（隐藏内部字段）
                display_cols = [c for c in batch_report.columns if not c.startswith("_")]
                st.dataframe(batch_report[display_cols], use_container_width=True)

                # 碎石图
                variance_data = batch_report.get("_variance_ratio")
                if variance_data is not None and len(batch_report) > 0:
                    import matplotlib.pyplot as plt
                    try:
                        from .plot_style import apply_global_style
                        apply_global_style()
                    except Exception:
                        pass

                    n_groups = len(batch_report)
                    n_cols = min(3, n_groups)
                    n_rows = (n_groups + n_cols - 1) // n_cols
                    fig_batch, axes = plt.subplots(n_rows, n_cols,
                                                    figsize=(5 * n_cols, 4 * n_rows),
                                                    squeeze=False)
                    for idx, row in batch_report.iterrows():
                        vr = row.get("_variance_ratio", [])
                        if vr is None:
                            continue
                        if isinstance(vr, str):
                            import ast
                            try:
                                vr = ast.literal_eval(vr)
                            except Exception:
                                continue
                        if not hasattr(vr, '__len__') or len(vr) == 0:
                            continue
                        vr = np.array(vr, dtype=float)
                        cum_vr = np.cumsum(vr)
                        r, c = divmod(idx, n_cols)
                        ax = axes[r][c]
                        n_show = min(20, len(vr))
                        x_pos = np.arange(1, n_show + 1)
                        ax.bar(x_pos, vr[:n_show], alpha=0.7, color='steelblue',
                               edgecolor='black', linewidth=0.5)
                        ax_twin = ax.twinx()
                        ax_twin.plot(x_pos, cum_vr[:n_show], 'o-', color='darkred',
                                     markersize=3, linewidth=1.5)
                        ax_twin.set_ylabel('累计方差', fontsize=9, color='darkred')
                        ax_twin.set_ylim(0, 1.05)
                        ax.set_title(f"{row['前缀']}  ({row['原始特征数']}→{row['主成分数']})",
                                     fontsize=10, fontweight='bold')
                        ax.set_xlabel('PC', fontsize=9)
                        ax.set_ylabel('方差比例', fontsize=9)
                        ax.grid(axis='y', alpha=0.3)
                    # 隐藏多余子图
                    for idx in range(n_groups, n_rows * n_cols):
                        r, c = divmod(idx, n_cols)
                        axes[r][c].set_visible(False)
                    fig_batch.tight_layout()
                    st.pyplot(fig_batch, use_container_width=True)
                    plt.close(fig_batch)

    # --- Tab 8: 工艺特征 PLS ---
    if tab_choice == tab_titles[7]:
        molecular_features, original_features, classifier, recorded_mol_features = get_feature_classification()

        st.markdown("#### ⚗️ 工艺特征 PLS")
        st.caption("用于把稀疏工艺特征压缩为少量监督主成分；此处只做探索性预览，正式训练会重新在训练折内拟合以避免数据泄漏。")

        process_candidates = infer_process_feature_candidates(
            current_df,
            original_features=original_features,
            molecular_features=molecular_features,
            target_col=target_col,
        )

        if not process_candidates:
            st.warning("⚠️ 未检测到可用于 PLS 的工艺数值候选列。请先在“特征分类纠错”中把工艺列归为原始特征。")
        else:
            preview_rows = []
            for column in process_candidates:
                numeric = pd.to_numeric(current_df[column], errors="coerce")
                preview_rows.append({
                    "候选列": column,
                    "缺失率": float(numeric.isna().mean()),
                    "有效样本数": int(numeric.notna().sum()),
                })
            candidate_preview = pd.DataFrame(preview_rows)

            selected_process_cols = st.multiselect(
                "选择工艺候选列",
                options=process_candidates,
                default=[],
                key="process_pls_selected_cols",
                help="默认不自动全选，避免把无关原始特征误纳入 PLS。建议至少选择 2 个连续/数值工艺变量。",
            )

            st.markdown("##### 候选列预览")
            display_preview = candidate_preview.copy()
            display_preview["缺失率"] = display_preview["缺失率"].map(lambda value: f"{value:.1%}")
            st.dataframe(display_preview, hide_index=True, height=260)

            y_numeric = pd.to_numeric(current_df[target_col], errors="coerce") if target_col in current_df.columns else pd.Series(dtype=float)
            y_mask = y_numeric.notna() & np.isfinite(y_numeric)
            y_available = bool(y_mask.sum() >= 2)

            col_diag, col_lock, col_clear = st.columns([1.2, 1.2, 1.0])
            with col_diag:
                if st.button("生成 PLS 诊断", key="btn_process_pls_preview"):
                    if len(selected_process_cols) < 2:
                        st.error("至少选择 2 个工艺数值特征")
                    elif not y_available:
                        st.error("当前数据没有可用于 PLS 诊断的数值目标列")
                    else:
                        try:
                            fit_frame = current_df.loc[y_mask, selected_process_cols].copy()
                            transformer = ProcessPLSTransformer(
                                process_feature_cols=selected_process_cols,
                                max_components=8,
                                vip_top_k=8,
                                missing_threshold=0.85,
                                cv_splits=5,
                                random_state=42,
                            ).fit(fit_frame, y_numeric.loc[y_mask].to_numpy(dtype=float))

                            cv_rows = transformer.cv_report_.get("candidates", [])
                            cv_report = pd.DataFrame(cv_rows)
                            if not cv_report.empty:
                                cv_report = cv_report.rename(columns={
                                    "n_components": "候选成分数",
                                    "cv_r2_mean": "CV R²",
                                    "cv_r2_std": "R²折间标准差",
                                    "cv_rmse_mean": "CV RMSE",
                                    "cv_rmse_std": "RMSE折间标准差",
                                    "rmse_improvement": "RMSE改善",
                                    "selection_score": "综合得分",
                                })

                            vip_report = pd.DataFrame({
                                "特征": transformer.kept_process_feature_cols_,
                                "VIP": transformer.vip_scores_,
                            }).sort_values("VIP", ascending=False).head(8)

                            st.session_state.process_pls_preview_report = {
                                "selected_cols": list(selected_process_cols),
                                "selected_n_components": int(transformer.n_components_),
                                "cv_report": cv_report.to_dict("records"),
                                "vip_top": vip_report.to_dict("records"),
                                "workflow_hash_preview": transformer.workflow_hash_,
                            }
                            st.success("✅ PLS 诊断已生成（此预览不用于正式训练）")
                        except Exception as exc:
                            st.session_state.process_pls_preview_report = None
                            st.error(f"PLS 诊断失败: {exc}")

            preview_report = st.session_state.get("process_pls_preview_report")
            with col_lock:
                if st.button("锁定工艺 PLS 工作流", key="btn_process_pls_lock"):
                    if len(selected_process_cols) < 2:
                        st.error("至少选择 2 个工艺数值特征")
                    elif not y_available:
                        st.error("当前数据没有可用于 PLS 诊断的数值目标列")
                    elif preview_report is None:
                        st.error("请先生成 PLS 诊断")
                    elif list(preview_report.get("selected_cols", [])) != list(selected_process_cols):
                        st.error("当前选择与诊断结果不一致，请重新生成 PLS 诊断")
                    else:
                        st.session_state.process_pls_workflow = build_process_pls_config(
                            selected_process_cols,
                            random_state=42,
                        )
                        st.session_state.process_pls_enabled_default = False
                        st.success("✅ 已锁定工艺 PLS 工作流；正式训练时默认关闭，需在训练页显式启用。")

            with col_clear:
                if st.button("清除锁定工作流", key="btn_process_pls_clear"):
                    st.session_state.process_pls_workflow = None
                    st.session_state.process_pls_preview_report = None
                    st.session_state.process_pls_enabled_default = False
                    st.success("已清除工艺 PLS 锁定配置")

            preview_report = st.session_state.get("process_pls_preview_report")
            if preview_report:
                st.info("此预览不用于正式训练；正式训练会仅在训练数据内重新拟合 imputer、scaler 与 PLS。")
                cv_display = pd.DataFrame(preview_report.get("cv_report", []))
                if not cv_display.empty:
                    st.markdown("##### PLS 诊断表")
                    st.dataframe(cv_display, hide_index=True, height=260)
                vip_display = pd.DataFrame(preview_report.get("vip_top", []))
                if not vip_display.empty:
                    st.markdown("##### VIP Top 结果")
                    st.dataframe(vip_display, hide_index=True, height=240)

            locked_workflow = st.session_state.get("process_pls_workflow")
            if locked_workflow:
                locked_cols = locked_workflow.get("process_feature_cols", [])
                st.success(
                    f"当前锁定：{len(locked_cols)} 个工艺列 | "
                    f"max_components={locked_workflow.get('max_components')} | "
                    f"vip_top_k={locked_workflow.get('vip_top_k')} | "
                    f"hash={str(locked_workflow.get('workflow_hash', ''))[:12]}"
                )

    # --- Tab 9: 模型重要性 ---
    if tab_choice == tab_titles[8]:
        st.markdown("#### ⭐ 模型重要性筛选")

        model = st.session_state.get('model')
        if model is None:
            st.info("⚠️ 请先训练模型，然后返回此处根据特征重要性筛选")
        else:
            model_name = st.session_state.get('model_name', 'Unknown')
            st.success(f"当前模型: **{model_name}**")

            # 选择重要性计算方法
            importance_method = st.radio(
                "重要性计算方法",
                ["模型内置重要性（快速）", "SHAP重要性（准确）"],
                help="模型内置：基于增益/系数，速度快但可能不准确\nSHAP：基于博弈论，考虑特征交互，更准确但较慢",
                horizontal=True,
                key="importance_method"
            )

            importances = None
            feature_names = None

            if importance_method == "模型内置重要性（快速）":
                # 原有的模型内置重要性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_).flatten()

                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif st.session_state.get('feature_cols'):
                    feature_names = np.array(st.session_state.feature_cols)

                if importances is not None and feature_names is not None and len(importances) == len(feature_names):
                    imp_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    st.info("📊 使用模型内置重要性（基于增益/系数）")
                    st.bar_chart(imp_df.set_index('Feature').head(20))

                    top_k = st.slider("保留Top-K", 5, len(feature_names), min(30, len(feature_names)), key="imp_k_v2")

                    if st.button("应用重要性筛选", key="btn_imp_v2"):
                        _apply_importance_filter_callback_v2(
                            imp_df['Feature'].tolist(),
                            feature_candidates,
                            "imp_k_v2"
                        )
                        st.rerun()
                else:
                    st.warning("当前模型不支持提取特征重要性")

            else:  # SHAP重要性
                st.info("🔬 使用SHAP重要性（基于Shapley值，考虑特征交互）")

                # 获取训练和测试数据
                X_train = st.session_state.get('X_train')
                X_test = st.session_state.get('X_test')

                if X_train is None or X_test is None:
                    st.warning("⚠️ 缺少训练/测试数据，请先完成模型训练")
                else:
                    # SHAP计算参数
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        max_samples = st.number_input(
                            "SHAP计算样本数",
                            min_value=50,
                            max_value=min(5000, len(X_test)),
                            value=min(1000, len(X_test)),
                            step=100,
                            help="样本数越多越准确，512核服务器可以处理更多样本"
                        )
                    with col2:
                        background_samples = st.number_input(
                            "背景样本数（仅KernelExplainer）",
                            min_value=20,
                            max_value=min(500, len(X_train)),
                            value=min(100, len(X_train)),
                            step=20,
                            help="仅用于非树模型，样本数越多越准确但越慢"
                        )
                    with col3:
                        n_jobs = st.number_input(
                            "并行核心数",
                            min_value=1,
                            max_value=512,
                            value=min(64, 512),
                            step=8,
                            help="512核服务器建议使用64-128核进行并行计算"
                        )

                    if st.button("🚀 计算SHAP重要性", key="btn_compute_shap"):
                        with st.spinner("正在计算SHAP值，请稍候..."):
                            try:
                                # 先禁用transformers检查，避免导入冲突
                                import os
                                import sys

                                # 方法1: 设置环境变量
                                os.environ['SHAP_DISABLE_TRANSFORMERS_CHECK'] = '1'

                                # 设置多线程环境变量，充分利用多核
                                os.environ['OMP_NUM_THREADS'] = str(int(n_jobs))
                                os.environ['MKL_NUM_THREADS'] = str(int(n_jobs))
                                os.environ['OPENBLAS_NUM_THREADS'] = str(int(n_jobs))

                                # 方法2: 彻底Mock transformers模块及其子模块
                                import types

                                # 创建假的transformers模块
                                if 'transformers' not in sys.modules:
                                    fake_transformers = types.ModuleType('transformers')
                                    sys.modules['transformers'] = fake_transformers

                                # Mock transformers.utils模块
                                if 'transformers.utils' not in sys.modules:
                                    fake_utils = types.ModuleType('transformers.utils')
                                    sys.modules['transformers.utils'] = fake_utils

                                # Mock transformers.utils.import_utils
                                if 'transformers.utils.import_utils' not in sys.modules:
                                    fake_import_utils = types.ModuleType('transformers.utils.import_utils')
                                    sys.modules['transformers.utils.import_utils'] = fake_import_utils

                                # 方法3: Monkey patch SHAP的transformers检查函数
                                import shap
                                if hasattr(shap.utils, 'transformers'):
                                    # 禁用transformers检查
                                    shap.utils.transformers.is_transformers_lm = lambda x: False

                                from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
                                import multiprocessing as mp

                                # 采样数据
                                X_sample = X_test.sample(n=int(max_samples), random_state=42) if len(X_test) > max_samples else X_test.copy()

                                st.info(f"🚀 使用 {int(n_jobs)} 个核心进行并行计算，样本数: {len(X_sample)}")

                                # 根据模型类型选择Explainer
                                tree_models = ['XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'Extra Trees', '梯度提升树']

                                if model_name in tree_models or hasattr(model, 'feature_importances_'):
                                    st.info(f"✓ 检测到树模型，使用TreeExplainer（快速）")
                                    try:
                                        # 对于XGBoost，尝试多种方式
                                        if model_name == 'XGBoost':
                                            shap_values, X_sample, _ = compute_xgboost_native_shap(
                                                model,
                                                X_sample,
                                                feature_names=list(X_sample.columns),
                                            )
                                            st.info("⚡ 使用 XGBoost 原生 pred_contribs 计算 SHAP...")
                                        else:
                                            explainer = shap.TreeExplainer(model, feature_names=list(X_sample.columns))

                                            # TreeExplainer支持批量计算，直接计算所有样本
                                            st.info("⚡ TreeExplainer支持高效批量计算...")
                                            # 禁用SHAP内部进度条，避免与Streamlit冲突
                                            shap_values = explainer.shap_values(X_sample, check_additivity=False)

                                    except Exception as tree_err:
                                        st.warning(f"TreeExplainer失败: {str(tree_err)[:100]}")
                                        st.info("尝试使用KernelExplainer作为备选...")
                                        background = shap.sample(X_train, min(100, len(X_train)))
                                        explainer = shap.KernelExplainer(model.predict, background)

                                        # KernelExplainer使用并行计算
                                        st.info(f"⚡ 使用 {int(n_jobs)} 核并行计算KernelExplainer...")

                                        # 分批并行计算
                                        batch_size = max(1, len(X_sample) // int(n_jobs))
                                        batches = [X_sample.iloc[i:i+batch_size] for i in range(0, len(X_sample), batch_size)]

                                        def compute_batch_shap(batch):
                                            # 禁用SHAP内部进度条，避免与Streamlit冲突
                                            return explainer.shap_values(batch, silent=True)

                                        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
                                            batch_results = list(executor.map(compute_batch_shap, batches))

                                        shap_values = np.vstack(batch_results)

                                elif model_name in ['线性回归', 'Ridge回归', 'Lasso回归', 'ElasticNet']:
                                    st.info(f"✓ 检测到线性模型，使用LinearExplainer")
                                    background = shap.sample(X_train, min(200, len(X_train)))
                                    explainer = shap.LinearExplainer(model, background)
                                    # 禁用SHAP内部进度条，避免与Streamlit冲突
                                    shap_values = explainer.shap_values(X_sample)

                                else:
                                    st.info(f"✓ 使用KernelExplainer（较慢，适用于任意模型）")
                                    background = shap.sample(X_train, int(background_samples))
                                    explainer = shap.KernelExplainer(model.predict, background)

                                    # KernelExplainer并行计算
                                    st.info(f"⚡ 使用 {int(n_jobs)} 核并行计算...")
                                    batch_size = max(1, len(X_sample) // int(n_jobs))
                                    batches = [X_sample.iloc[i:i+batch_size] for i in range(0, len(X_sample), batch_size)]

                                    def compute_batch_shap(batch):
                                        # 禁用SHAP内部进度条，避免与Streamlit冲突
                                        return explainer.shap_values(batch, silent=True)

                                    with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
                                        batch_results = list(executor.map(compute_batch_shap, batches))

                                    shap_values = np.vstack(batch_results)

                                # 计算平均绝对SHAP值作为重要性
                                if isinstance(shap_values, list):  # 多输出情况
                                    shap_values = shap_values[0]

                                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                                # 获取特征名
                                if hasattr(model, 'feature_names_in_'):
                                    feature_names = model.feature_names_in_
                                elif st.session_state.get('feature_cols'):
                                    feature_names = np.array(st.session_state.feature_cols)
                                else:
                                    feature_names = X_sample.columns.values

                                # 创建重要性DataFrame
                                shap_imp_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'SHAP_Importance': mean_abs_shap
                                }).sort_values('SHAP_Importance', ascending=False)

                                # 保存到session_state
                                st.session_state['shap_importance_df'] = shap_imp_df
                                st.session_state['shap_values'] = shap_values
                                st.session_state['shap_X_sample'] = X_sample

                                st.success(f"✅ SHAP计算完成！基于 {len(X_sample)} 个样本")

                            except Exception as e:
                                st.error(f"❌ SHAP计算失败: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                    # 显示SHAP重要性结果
                    if 'shap_importance_df' in st.session_state:
                        shap_imp_df = st.session_state['shap_importance_df']

                        st.markdown("##### 📊 SHAP特征重要性排名")
                        st.bar_chart(shap_imp_df.set_index('Feature').head(20))

                        # 显示详细数据
                        with st.expander("📋 查看完整SHAP重要性数据"):
                            st.dataframe(shap_imp_df, use_container_width=True)

                        # 特征筛选
                        top_k_shap = st.slider(
                            "保留Top-K特征",
                            5,
                            len(shap_imp_df),
                            min(30, len(shap_imp_df)),
                            key="shap_k_v2"
                        )

                        if st.button("应用SHAP筛选", key="btn_apply_shap"):
                            _apply_importance_filter_callback_v2(
                                shap_imp_df['Feature'].tolist(),
                                feature_candidates,
                                "shap_k_v2"
                            )
                            st.rerun()
                    else:
                        st.info("👆 点击上方按钮开始计算SHAP重要性")

    # --- Tab 9: 前缀管理 ---
    if tab_choice == tab_titles[9]:
        from core.prefix_manager import render_prefix_manager
        render_prefix_manager()


# 导出主函数
__all__ = ['render_feature_selector', 'SmartFeatureClassifier', 'SmartFeatureSelector', 
           'SmartSparseDataSelector', 'show_robust_feature_selection']


# ==========================================
# 4. 兼容性类（保留旧接口）
# ==========================================

class SmartFeatureSelector:
    """智能特征选择器（兼容旧接口）"""
    
    MISSING_VALUE_TOLERANT_MODELS = {'XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'ExtraTrees'}
    
    def __init__(self, data, feature_cols, target_col, model_name=None):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_name = model_name
        self.missing_info = {}
    
    def analyze_missing_values(self):
        selected = self.data[self.feature_cols + [self.target_col]]
        total = selected.size
        missing = selected.isnull().sum().sum()
        col_missing = selected.isnull().sum()
        
        self.missing_info = {
            'total_cells': total,
            'missing_cells': missing,
            'missing_rate': missing / total if total > 0 else 0,
            'column_missing': col_missing,
            'column_missing_rate': col_missing / len(selected),
            'rows_with_missing': (selected.isnull().sum(axis=1) > 0).sum(),
            'columns_with_high_missing': col_missing[col_missing / len(selected) > 0.3].index.tolist()
        }
        return self.missing_info
    
    def recommend_strategy(self):
        self.analyze_missing_values()
        recommendations = []
        
        if self.model_name in self.MISSING_VALUE_TOLERANT_MODELS:
            recommendations.append({
                'strategy': 'model_native',
                'priority': 1,
                'reason': f'{self.model_name}原生支持缺失值'
            })
        
        recommendations.append({
            'strategy': 'median',
            'priority': 2,
            'reason': '中位数填充，对异常值稳健'
        })
        
        return recommendations


class SmartSparseDataSelector:
    """智能稀疏数据选择器（兼容旧接口）"""
    
    def __init__(self, data: pd.DataFrame, auto_analyze: bool = True):
        self.data = data
        self.numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        self.sparsity_info = None
        if auto_analyze:
            self._ensure_analyzed()
    
    def _ensure_analyzed(self):
        if self.sparsity_info is None:
            self.sparsity_info = self._analyze()
        return self.sparsity_info
    
    def _analyze(self):
        if not self.numeric_cols:
            return {}
        
        df_num = self.data[self.numeric_cols]
        non_null_count = df_num.notna().sum()
        non_null_ratio = df_num.notna().mean()
        null_count = df_num.isna().sum()
        
        info = {}
        for col in self.numeric_cols:
            info[col] = {
                'non_null_count': int(non_null_count[col]),
                'non_null_ratio': float(non_null_ratio[col]),
                'null_count': int(null_count[col]),
            }
        return info
    
    def get_target_analysis(self):
        df = self.data[self.numeric_cols]
        
        analysis = []
        for col in self.numeric_cols:
            non_null = df[col].notna().sum()
            valid_ratio = non_null / len(df)
            analysis.append({
                'column': col,
                'valid_samples': non_null,
                'valid_ratio': valid_ratio,
                'mean': df[col].mean(),
                'std': df[col].std()
            })
        
        return pd.DataFrame(analysis).sort_values('valid_ratio', ascending=False)
    
    def get_valid_samples_for_target(self, target_col):
        return self.data[target_col].notna()
    
    def analyze_features_for_target(self, target_col):
        self._ensure_analyzed()
        
        target_series = self.data[target_col]
        valid_mask = target_series.notna()
        target_series = target_series[valid_mask]
        
        analysis = []
        for col in self.numeric_cols:
            if col == target_col:
                continue
            
            col_data = self.data[col][valid_mask]
            valid_samples = col_data.notna().sum()
            valid_ratio = valid_samples / len(target_series)
            
            corr = None
            if valid_samples > 10:
                try:
                    corr = col_data.corr(target_series)
                except:
                    corr = None
            
            sparsity = self.sparsity_info.get(col, {})
            
            analysis.append({
                'feature': col,
                '特征': col,
                'valid_samples': valid_samples,
                'valid_ratio': valid_ratio,
                'correlation': corr,
                'missing_ratio': 1 - sparsity.get('non_null_ratio', 0)
            })
        
        df_analysis = pd.DataFrame(analysis)
        
        df_analysis['correlation'] = pd.to_numeric(df_analysis['correlation'], errors='coerce')
        df_analysis['score'] = (
            df_analysis['valid_ratio'] * 0.6 +
            df_analysis['correlation'].abs().fillna(0) * 0.4
        )
        
        if len(df_analysis) > 0:
            score_threshold = df_analysis['score'].median()
            df_analysis['推荐'] = df_analysis.apply(
                lambda row: '✓' if row['score'] >= score_threshold and row['missing_ratio'] < 0.5 else '',
                axis=1
            )
        else:
            df_analysis['推荐'] = ''
        
        return df_analysis.sort_values('score', ascending=False)


def show_robust_feature_selection():
    """显示特征选择界面（兼容旧接口，调用新版render函数）"""
    render_feature_selector()
