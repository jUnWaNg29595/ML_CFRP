# -*- coding: utf-8 -*-
"""模型训练模块

增强点（面向 Tg / 力学等小样本回归任务的稳健训练）：
1) 支持多种划分策略：随机 / 回归分箱分层 / 按配方分组
2) 支持 Repeated KFold / GroupKFold 的交叉验证，并输出 OOF 预测
3) 统一用 Pipeline 保存 imputer + scaler + model，避免预测阶段漏变换
"""

import time
import gc
from typing import Optional
import numpy as np
import os
import pandas as pd
import shutil
import inspect

from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    GroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    balanced_accuracy_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)

# [修复] 导入自定义 ANN 模型
try:
    from .ann_model import ANNRegressor
    ANN_AVAILABLE = True
except Exception:
    ANN_AVAILABLE = False
    ANNRegressor = None

try:
    from .bnn_model import BayesianNeuralNetworkRegressor, BNN_AVAILABLE
except Exception:
    BNN_AVAILABLE = False
    BayesianNeuralNetworkRegressor = None

try:
    from .missing_value_handler import MissingValueHandler
except Exception:
    MissingValueHandler = None

try:
    from .process_pls import PROCESS_PLS_SCHEMA_VERSION, ProcessPLSTransformer
except Exception:
    PROCESS_PLS_SCHEMA_VERSION = 1
    ProcessPLSTransformer = None


# [新增] Epoxy PINN (Physics-Informed) 模型
try:
    from .pinn_model import EpoxyPINNRegressor
    PINN_AVAILABLE = True
except Exception:
    PINN_AVAILABLE = False
    EpoxyPINNRegressor = None

# [新增] TabNet模型
try:
    from .tabnet_model import TabNetRegressor
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False
    TabNetRegressor = None

# [新增] FT-Transformer模型
try:
    from .fttransformer_model import FTTransformerRegressor
    FT_TRANSFORMER_AVAILABLE = True
except Exception:
    FT_TRANSFORMER_AVAILABLE = False
    FTTransformerRegressor = None

try:
    from .transformer_bnn_model import TransformerBNNRegressor, TRANSFORMER_BNN_AVAILABLE
except Exception:
    TRANSFORMER_BNN_AVAILABLE = False
    TransformerBNNRegressor = None

try:
    from .transformer_pinn_model import TransformerPINNRegressor, TRANSFORMER_PINN_AVAILABLE
except Exception:
    TRANSFORMER_PINN_AVAILABLE = False
    TransformerPINNRegressor = None

try:
    from .gnn_transformer_fusion_model import (
        GNNTransformerFusionRegressor,
        GNN_TRANSFORMER_FUSION_AVAILABLE,
    )
except Exception:
    GNN_TRANSFORMER_FUSION_AVAILABLE = False
    GNNTransformerFusionRegressor = None

try:
    from .tf_model import TFSequentialRegressor, TENSORFLOW_AVAILABLE
except Exception:
    TENSORFLOW_AVAILABLE = False
    TFSequentialRegressor = None

# 训练曲线工具（尽量不影响核心训练流程）
from .training_curves import (
    extract_history_from_fitted_model,
    build_holdout_learning_curve,
    history_to_frame,
)


def _safe_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), True
    except Exception:
        return None, False


XGBRegressor, XGBOOST_AVAILABLE = _safe_import('xgboost', 'XGBRegressor')
XGBClassifier, _XGB_CLASSIFIER_AVAILABLE = _safe_import('xgboost', 'XGBClassifier')
LGBMRegressor, LIGHTGBM_AVAILABLE = _safe_import('lightgbm', 'LGBMRegressor')
LGBMClassifier, _LGBM_CLASSIFIER_AVAILABLE = _safe_import('lightgbm', 'LGBMClassifier')
CatBoostRegressor, CATBOOST_AVAILABLE = _safe_import('catboost', 'CatBoostRegressor')
CatBoostClassifier, _CATBOOST_CLASSIFIER_AVAILABLE = _safe_import('catboost', 'CatBoostClassifier')
TabPFNRegressor, TABPFN_AVAILABLE = _safe_import('tabpfn', 'TabPFNRegressor')
AutoSklearnRegressor, AUTOSKLEARN_AVAILABLE = _safe_import('autosklearn.regression', 'AutoSklearnRegressor')
TPOTRegressor, TPOT_AVAILABLE = _safe_import('tpot', 'TPOTRegressor')
AutoML, FLAML_AVAILABLE = _safe_import('flaml', 'AutoML')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except Exception:
    AUTOGLUON_AVAILABLE = False

try:
    from .gnn_regressor import (
        PyGGraphRegressor,
        PYG_AVAILABLE,
        ATTENTIVEFP_AVAILABLE,
        DMPNN_AVAILABLE,
    )
except Exception:
    PyGGraphRegressor = None
    PYG_AVAILABLE = False
    ATTENTIVEFP_AVAILABLE = False
    DMPNN_AVAILABLE = False

GRAPH_MODEL_NAMES = {
    "AttentiveFP",
    "GAT",
    "GIN",
    "GCN",
    "GraphSAGE",
    "MPNN",
    "D-MPNN",
}

RAW_FRAME_MODEL_NAMES = {
    "FT-Transformer",
    "Transformer + BNN",
    "Transformer + PINN",
    "GNN + Transformer Fusion",
}

RAW_FRAME_MODELS_WITH_SMILES = {
    "GNN + Transformer Fusion",
}

TARGET_NAME_MODEL_NAMES = {
    "Epoxy PINN (Physics-Informed)",
    "Transformer + PINN",
}

AUTOML_MODEL_NAMES = {
    "Auto-sklearn",
    "TPOT",
    "FLAML",
}

CLASSIFICATION_MODEL_NAMES = {
    "逻辑回归分类",
    "随机森林分类",
    "XGBoost分类",
    "LightGBM分类",
    "CatBoost分类",
}

TREE_CLASSIFICATION_MODEL_NAMES = {
    "随机森林分类",
    "XGBoost分类",
    "LightGBM分类",
    "CatBoost分类",
}


def _should_pass_target_name(model_name: str) -> bool:
    return str(model_name) in TARGET_NAME_MODEL_NAMES


def _is_classification_model(model_name: str) -> bool:
    return str(model_name) in CLASSIFICATION_MODEL_NAMES


def _sorted_class_labels(labels):
    labels = list(labels)
    try:
        numeric = [float(v) for v in labels]
    except Exception:
        return sorted(labels, key=lambda v: str(v))
    order = np.argsort(numeric)
    return [labels[i] for i in order]


def _encode_binary_target(y_values):
    y_series = pd.Series(np.asarray(y_values).ravel())
    valid_mask = ~y_series.isna()
    y_series = y_series.loc[valid_mask].reset_index(drop=True)
    if y_series.empty:
        raise ValueError("目标列没有可用的非空样本，无法执行二分类训练。")

    unique_labels = _sorted_class_labels(pd.unique(y_series))
    if len(unique_labels) != 2:
        raise ValueError(
            f"二分类任务要求目标列恰好包含2个有效类别，当前检测到 {len(unique_labels)} 个：{unique_labels}"
        )

    positive_label = None
    for candidate in unique_labels:
        text = str(candidate).strip().lower()
        if text in {"1", "true", "yes", "positive", "pos"}:
            positive_label = candidate
            break
    if positive_label is None:
        positive_label = unique_labels[-1]
    negative_label = unique_labels[0] if positive_label != unique_labels[0] else unique_labels[1]

    class_labels = [negative_label, positive_label]
    class_to_int = {negative_label: 0, positive_label: 1}
    y_encoded = y_series.map(class_to_int).astype(int).to_numpy()
    return y_series, y_encoded, class_labels, class_to_int, positive_label


def _safe_predict_proba(estimator, X):
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        proba = estimator.predict_proba(X)
    except Exception:
        return None
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1].astype(float)
    return None


def _compute_binary_classification_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "log_loss": float("nan"),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba).ravel().astype(float)
        finite_mask = np.isfinite(y_proba)
        if finite_mask.sum() == len(y_true):
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                pass
            try:
                clipped = np.clip(y_proba, 1e-7, 1.0 - 1e-7)
                metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
            except Exception:
                pass
    return metrics


class InfCleaner(BaseEstimator, TransformerMixin):
    """清理无穷大值的转换器，用于 Pipeline 中"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clean = np.where(np.isinf(X), np.nan, X)
        return X_clean

    def __getstate__(self):
        """支持pickle序列化"""
        return {}

    def __setstate__(self, state):
        """支持pickle反序列化"""
        pass


class AllNaNColumnDropper(BaseEstimator, TransformerMixin):
    """删除全NaN列的转换器，确保训练集和测试集特征数一致"""

    def __init__(self):
        self.keep_mask_ = None

    def fit(self, X, y=None):
        # 记录训练集中哪些列不是全NaN
        X_arr = np.asarray(X)
        all_nan_mask = np.isnan(X_arr).all(axis=0)
        self.keep_mask_ = ~all_nan_mask

        if not self.keep_mask_.all():
            n_dropped = (~self.keep_mask_).sum()
            print(f"⚠️ AllNaNColumnDropper: 发现 {n_dropped} 个全NaN列，将删除")

        return self

    def transform(self, X):
        if self.keep_mask_ is None:
            raise RuntimeError("AllNaNColumnDropper not fitted yet")

        X_arr = np.asarray(X)
        return X_arr[:, self.keep_mask_]


class FeatureMaskTransformer(BaseEstimator, TransformerMixin):
    """应用特征掩码的转换器，用于删除训练时被移除的特征"""

    def __init__(self, feature_mask=None):
        self.feature_mask = feature_mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_mask is None:
            return X

        X_arr = np.asarray(X)
        feature_mask = np.asarray(self.feature_mask, dtype=bool)

        if X_arr.shape[1] != len(feature_mask):
            raise ValueError(
                f"特征数量不匹配: 输入数据有 {X_arr.shape[1]} 个特征，"
                f"但特征掩码期望 {len(feature_mask)} 个特征"
            )

        return X_arr[:, feature_mask]


def apply_feature_mask(X, feature_mask):
    """
    应用特征掩码，过滤掉训练时被删除的特征

    Parameters
    ----------
    X : array-like
        输入特征矩阵
    feature_mask : array-like of bool or None
        特征掩码，True表示保留，False表示删除
        如果为None，则不做任何过滤

    Returns
    -------
    X_filtered : array-like
        过滤后的特征矩阵
    """
    if feature_mask is None:
        return X

    X_arr = np.asarray(X)
    feature_mask = np.asarray(feature_mask, dtype=bool)

    if X_arr.shape[1] != len(feature_mask):
        raise ValueError(
            f"特征数量不匹配: 输入数据有 {X_arr.shape[1]} 个特征，"
            f"但特征掩码期望 {len(feature_mask)} 个特征"
        )

    return X_arr[:, feature_mask]



def _make_y_bins(y: np.ndarray, n_bins: int = 10):
    """把连续 y 分箱用于"回归分层划分"。

    返回:
        bins (np.ndarray[int]) 或 None（表示无法分箱，需回退随机划分）
    """
    y = np.asarray(y).ravel()
    if len(y) < 3:
        return None

    n_bins = int(max(2, n_bins))
    try:
        bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates='drop')
        bins = np.asarray(bins)
        if np.unique(bins).size < 2:
            return None
        return bins
    except Exception:
        try:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
            if len(edges) < 3:
                return None
            bins = np.digitize(y, edges[1:-1], right=True)
            if np.unique(bins).size < 2:
                return None
            return bins
        except Exception:
            return None


def _build_target_balance_info(
    y_train,
    enabled=True,
    n_bins=10,
    max_weight=3.0,
    random_state=42,
):
    y_arr = pd.to_numeric(
        pd.Series(np.asarray(y_train).ravel()),
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_mask = np.isfinite(y_arr)
    y_arr = y_arr[valid_mask]
    info = {
        "enabled": bool(enabled),
        "method": "disabled",
        "n_bins": int(max(1, n_bins)),
        "max_weight": float(max(1.0, max_weight)),
        "random_state": int(random_state),
        "weights": np.ones(len(y_arr), dtype=float),
        "bin_ids": np.zeros(len(y_arr), dtype=int),
        "bin_edges": [],
        "bin_counts": [],
        "bin_mean_weights": [],
        "train_sample_count": int(len(y_arr)),
        "resampled_sample_count": int(len(y_arr)),
        "fallback_reason": None,
    }

    def disable(reason):
        info["enabled"] = False
        info["fallback_reason"] = reason
        return info

    if not enabled:
        return disable("用户关闭目标分布平衡")
    if len(y_arr) < 20:
        return disable("有效训练样本少于20，跳过分箱")
    if np.unique(y_arr).size < 2 or np.isclose(np.ptp(y_arr), 0.0):
        return disable("目标值近似常量，无法建立密度差异")

    requested_bins = int(np.clip(n_bins, 4, 20))
    finite_min = float(np.min(y_arr))
    finite_max = float(np.max(y_arr))
    raw_edges = np.linspace(finite_min, finite_max, requested_bins + 1)
    edges = np.unique(raw_edges)
    if len(edges) < 3:
        return disable("目标值边界不足，无法形成至少两个分箱")

    bin_ids = np.digitize(y_arr, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(int)
    nonempty = counts > 0
    if int(nonempty.sum()) < 2:
        return disable("有效非空分箱少于2个")

    occupied_ids = np.flatnonzero(nonempty)
    remap = {old_id: new_id for new_id, old_id in enumerate(occupied_ids)}
    bin_ids = np.asarray([remap[int(value)] for value in bin_ids], dtype=int)
    occupied_edges = np.concatenate(
        ([edges[occupied_ids[0]]], edges[occupied_ids + 1])
    )
    counts = np.bincount(bin_ids, minlength=len(occupied_ids)).astype(int)

    if len(counts) > 2:
        while len(counts) > 2 and int(np.min(counts)) < 2:
            sparse_id = int(np.argmin(counts))
            if sparse_id == 0:
                bin_ids[bin_ids == sparse_id] = 1
                bin_ids[bin_ids > sparse_id] -= 1
                boundary_index = sparse_id + 1
            else:
                merge_to = sparse_id - 1
                bin_ids[bin_ids == sparse_id] = merge_to
                bin_ids[bin_ids > sparse_id] -= 1
                boundary_index = sparse_id
            counts = np.bincount(
                bin_ids,
                minlength=int(np.max(bin_ids)) + 1,
            ).astype(int)
            occupied_edges = np.delete(occupied_edges, boundary_index)

    density = counts.astype(float) / max(1, len(y_arr))
    smoothed_density = density + (1.0 / max(1, len(y_arr)))
    raw_weights_by_bin = 1.0 / smoothed_density
    raw_weights_by_bin /= np.average(raw_weights_by_bin, weights=counts)

    weight_cap = float(max(1.0, max_weight))
    weights_by_bin = raw_weights_by_bin.copy()
    capped = np.zeros(len(weights_by_bin), dtype=bool)
    total_weight = float(np.sum(counts))
    while True:
        uncapped = ~capped
        fixed_weight = float(np.sum(counts[capped] * weight_cap))
        remaining_weight = max(0.0, total_weight - fixed_weight)
        raw_uncapped_weight = float(np.sum(counts[uncapped] * raw_weights_by_bin[uncapped]))
        scale = remaining_weight / max(raw_uncapped_weight, 1e-12)
        weights_by_bin[uncapped] = raw_weights_by_bin[uncapped] * scale
        weights_by_bin[capped] = weight_cap
        newly_capped = uncapped & (weights_by_bin > weight_cap)
        if not np.any(newly_capped):
            break
        capped[newly_capped] = True

    weights = weights_by_bin[bin_ids]
    info.update(
        {
            "method": "ready",
            "weights": weights.astype(float),
            "bin_ids": bin_ids,
            "bin_edges": occupied_edges.astype(float).tolist(),
            "bin_counts": counts.astype(int).tolist(),
            "bin_mean_weights": [
                float(np.mean(weights[bin_ids == bin_id]))
                for bin_id in range(len(counts))
            ],
        }
    )
    return info


def _weighted_resample_indices(weights, random_state=42):
    weights_arr = np.asarray(weights, dtype=float).ravel()
    if len(weights_arr) == 0:
        return np.asarray([], dtype=int)
    probabilities = np.where(
        np.isfinite(weights_arr) & (weights_arr > 0),
        weights_arr,
        0.0,
    )
    if float(probabilities.sum()) <= 0.0:
        probabilities = np.ones(len(weights_arr), dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(int(random_state))
    indices = rng.choice(
        np.arange(len(weights_arr), dtype=int),
        size=len(weights_arr),
        replace=True,
        p=probabilities,
    )
    if (
        len(indices) > 1
        and len(np.unique(indices)) == len(indices)
        and np.ptp(probabilities) > 1e-12
    ):
        highest_probability_index = int(np.argmax(probabilities))
        lowest_sample_position = int(
            np.argmin(probabilities[indices])
        )
        indices[lowest_sample_position] = highest_probability_index
    return indices


def _model_accepts_sample_weight(model):
    try:
        signature = inspect.signature(model.fit)
    except Exception:
        return False
    parameters = signature.parameters
    return (
        "sample_weight" in parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    )


def _prepare_balanced_fit_data(model, X_train, y_train, balance_info, random_state=42):
    y_arr = np.asarray(y_train).ravel()
    method = balance_info.get("method")
    if method not in {"ready", "disabled"}:
        raise ValueError("balance_info 只接受 method 为 ready 或 disabled")

    weights = np.asarray(
        balance_info.get("weights", np.ones(len(y_arr))),
        dtype=float,
    ).ravel()
    if len(weights) != len(y_arr):
        weights = np.ones(len(y_arr), dtype=float)

    if method == "disabled":
        return {
            "X": X_train,
            "y": y_arr,
            "sample_weight": None,
            "method": "disabled",
            "resampled_indices": np.arange(len(y_arr), dtype=int),
            "fallback_reason": balance_info.get("fallback_reason"),
        }

    if _model_accepts_sample_weight(model):
        return {
            "X": X_train,
            "y": y_arr,
            "sample_weight": weights,
            "method": "sample_weight",
            "resampled_indices": np.arange(len(y_arr), dtype=int),
            "fallback_reason": None,
        }

    indices = _weighted_resample_indices(weights, random_state=random_state)
    if isinstance(X_train, pd.DataFrame):
        X_resampled = X_train.iloc[indices].reset_index(drop=True)
    else:
        X_resampled = np.asarray(X_train)[indices]
    return {
        "X": X_resampled,
        "y": y_arr[indices],
        "sample_weight": None,
        "method": "weighted_resample",
        "resampled_indices": indices,
        "fallback_reason": None,
    }


def _compute_regression_bin_metrics(y_true, y_pred, bin_edges):
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    edges = np.asarray(bin_edges, dtype=float).ravel()
    if len(edges) < 2:
        return []

    bin_ids = np.digitize(y_true_arr, edges[1:-1], right=False)
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = (
            (bin_ids == bin_id)
            & np.isfinite(y_true_arr)
            & np.isfinite(y_pred_arr)
        )
        if not np.any(mask):
            continue
        y_bin = y_true_arr[mask]
        pred_bin = y_pred_arr[mask]
        rows.append(
            {
                "bin": f"[{edges[bin_id]:.6g}, {edges[bin_id + 1]:.6g})",
                "sample_count": int(mask.sum()),
                "r2": (
                    float(r2_score(y_bin, pred_bin))
                    if len(y_bin) >= 2
                    else float("nan")
                ),
                "rmse": float(np.sqrt(mean_squared_error(y_bin, pred_bin))),
                "mae": float(mean_absolute_error(y_bin, pred_bin)),
            }
        )
    return rows


def _finalize_target_balance_result(
    balance_info,
    actual_method=None,
    fit_sample_count=None,
    early_stopping_validation_count=0,
):
    """压缩目标平衡元数据，避免把逐样本数组写入模型结果。"""
    result = {
        key: value
        for key, value in dict(balance_info or {}).items()
        if key not in {"weights", "bin_ids"}
    }
    weights = np.asarray((balance_info or {}).get("weights", []), dtype=float)
    result["method"] = actual_method or result.get("method", "disabled")
    result["weights"] = None
    result["weight_min"] = (
        float(np.min(weights)) if len(weights) else 1.0
    )
    result["weight_max"] = (
        float(np.max(weights)) if len(weights) else 1.0
    )
    result["fit_sample_count"] = int(
        len(weights) if fit_sample_count is None else fit_sample_count
    )
    result["resampled_sample_count"] = int(result["fit_sample_count"])
    result["early_stopping_validation_count"] = int(
        early_stopping_validation_count
    )
    return result


def _sanitize_feature_frame(df_in: pd.DataFrame, model_name_in: str) -> pd.DataFrame:
    """Replace inf/too-large values with NaN for safer model fitting."""
    df_out = df_in.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df_out

    df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan)

    if model_name_in in {"XGBoost", "LightGBM", "CatBoost", "XGBoost分类", "LightGBM分类", "CatBoost分类"}:
        max_f32 = np.finfo(np.float32).max
        too_large = df_out[num_cols].abs() > max_f32
        if bool(too_large.any().any()):
            df_out[num_cols] = df_out[num_cols].where(~too_large, np.nan)
    return df_out


def _validate_process_pls_config(config, feature_columns) -> None:
    if not isinstance(config, dict):
        raise ValueError("已启用工艺 PLS，但没有有效的工艺 PLS 配置")
    if config.get("schema_version") != PROCESS_PLS_SCHEMA_VERSION:
        raise ValueError("工艺 PLS 工作流版本不匹配，请回到特征选择页面重新锁定")
    process_feature_cols = list(config.get("process_feature_cols") or [])
    if not process_feature_cols:
        raise ValueError("已启用工艺 PLS，但工作流没有工艺特征列")
    if not set(process_feature_cols).issubset(set(feature_columns or [])):
        raise ValueError("工艺 PLS 所需原始列不完整，请检查特征选择和数据列映射")


def _config_to_process_pls_kwargs(config) -> dict:
    return {
        "process_feature_cols": list(config.get("process_feature_cols") or []),
        "max_components": int(config.get("max_components", 8) or 8),
        "vip_top_k": int(config.get("vip_top_k", 8) or 8),
        "missing_threshold": float(config.get("missing_threshold", 0.85) or 0.85),
        "random_state": int(config.get("random_state", 42) or 42),
        "cv_splits": int(config.get("cv_splits", 5) or 5),
    }


def _make_process_pls_step(process_pls_config, enabled, feature_columns=None):
    if not enabled:
        return None
    if ProcessPLSTransformer is None:
        raise ValueError("工艺 PLS 模块不可用，请检查 core/process_pls.py")
    if feature_columns is not None:
        _validate_process_pls_config(process_pls_config, feature_columns)
    elif not isinstance(process_pls_config, dict):
        raise ValueError("已启用工艺 PLS，但没有有效的工艺 PLS 配置")
    return (
        "process_pls",
        ProcessPLSTransformer(**_config_to_process_pls_kwargs(process_pls_config)),
    )


def _filter_fit_kwargs_by_signature(model, fit_kwargs: dict) -> dict:
    try:
        sig = inspect.signature(model.fit)
    except Exception:
        return fit_kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fit_kwargs
    return {k: v for k, v in fit_kwargs.items() if k in params}


def _safe_xgb_fit(model, X, y, fit_kwargs: dict):
    """XGBoost 安全训练，防止 512 核机器崩溃

    注意: 保留early_stopping_rounds和eval_set参数,确保早停机制生效
    """
    import gc

    if not fit_kwargs:
        model.fit(X, y)
        gc.collect()  # 训练后立即清理
        return

    filtered = _filter_fit_kwargs_by_signature(model, fit_kwargs)

    # 第一次尝试: 使用所有参数(包括early_stopping)
    try:
        model.fit(X, y, **filtered)
        gc.collect()  # 训练后立即清理
        return
    except TypeError as e:
        # 如果失败,记录错误但继续尝试
        print(f"[DEBUG] XGBoost fit with all params failed: {e}")
        pass

    # 第二次尝试: 只移除eval_metric(通过set_params设置)
    cleaned = dict(filtered)
    if "eval_metric" in cleaned:
        try:
            model.set_params(eval_metric=cleaned.get("eval_metric"))
            cleaned.pop("eval_metric", None)
            model.fit(X, y, **cleaned)
            gc.collect()
            return
        except Exception as e:
            print(f"[DEBUG] XGBoost fit with eval_metric in set_params failed: {e}")
            # 恢复eval_metric到fit参数
            cleaned["eval_metric"] = filtered.get("eval_metric")

    # 第三次尝试: 保留early_stopping_rounds和eval_set,只移除verbose
    # 这是最重要的,确保早停机制生效!
    if "verbose" in cleaned:
        trial = dict(cleaned)
        trial.pop("verbose", None)
        try:
            model.fit(X, y, **trial)
            gc.collect()
            return
        except TypeError as e:
            print(f"[DEBUG] XGBoost fit without verbose failed: {e}")
            pass

    # 第四次尝试: 移除eval_metric,但保留early_stopping和eval_set
    if "eval_metric" in cleaned:
        trial = dict(cleaned)
        trial.pop("eval_metric", None)
        try:
            model.fit(X, y, **trial)
            gc.collect()
            return
        except TypeError as e:
            print(f"[DEBUG] XGBoost fit without eval_metric failed: {e}")
            pass

    # 最后尝试: 如果以上都失败,才移除early_stopping相关参数
    print("[WARNING] XGBoost早停参数不兼容,将禁用早停机制")
    for key in ("early_stopping_rounds", "eval_set", "verbose", "eval_metric"):
        cleaned.pop(key, None)

    model.fit(X, y, **cleaned)
    gc.collect()  # 训练后立即清理


# --- AutoGluon 适配器 ---
class AutoGluonWrapper(BaseEstimator, RegressorMixin):
    """将 AutoGluon 封装为 Scikit-Learn 风格的 Estimator"""

    def __init__(self, time_limit=60, presets='medium_quality', num_gpus=None, num_cpus=None, **kwargs):
        self.time_limit = time_limit
        self.presets = presets
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.kwargs = kwargs
        self.predictor = None
        self.label_col = 'target'
        self.save_path = f"AutogluonModels/ag-{int(time.time())}"

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            train_data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            train_data = pd.DataFrame(X).copy()

        self.feature_names_ = train_data.columns.tolist()
        train_data[self.label_col] = y

        fit_kwargs = dict(self.kwargs)
        fit_kwargs.setdefault("time_limit", self.time_limit)
        fit_kwargs.setdefault("presets", self.presets)
        if self.num_gpus is not None:
            fit_kwargs.setdefault("num_gpus", self.num_gpus)
        if self.num_cpus is not None:
            fit_kwargs.setdefault("num_cpus", self.num_cpus)

        def _do_fit(kwargs):
            return TabularPredictor(
                label=self.label_col,
                path=self.save_path,
                verbosity=0
            ).fit(
                train_data,
                **kwargs
            )

        try:
            self.predictor = _do_fit(fit_kwargs)
        except TypeError:
            if "num_gpus" in fit_kwargs or "num_cpus" in fit_kwargs:
                ag_args_fit = fit_kwargs.get("ag_args_fit")
                if not isinstance(ag_args_fit, dict):
                    ag_args_fit = {}
                if "num_gpus" in fit_kwargs:
                    ag_args_fit.setdefault("num_gpus", fit_kwargs.pop("num_gpus"))
                if "num_cpus" in fit_kwargs:
                    ag_args_fit.setdefault("num_cpus", fit_kwargs.pop("num_cpus"))
                fit_kwargs["ag_args_fit"] = ag_args_fit
                self.predictor = _do_fit(fit_kwargs)
            else:
                raise
        return self

    def predict(self, X):
        if self.predictor is None:
            raise RuntimeError("AutoGluon model not fitted yet.")

        if isinstance(X, np.ndarray):
            test_data = pd.DataFrame(X, columns=self.feature_names_)
        else:
            test_data = pd.DataFrame(X)
            if test_data.shape[1] == len(self.feature_names_):
                test_data.columns = self.feature_names_

        return self.predictor.predict(test_data).values

    def __del__(self):
        pass


class AutoSklearnWrapper(BaseEstimator, RegressorMixin):
    """Auto-sklearn wrapper (if autosklearn is installed)."""

    def __init__(
        self,
        time_left_for_this_task: int = 120,
        per_run_time_limit: int = 30,
        ensemble_size: int = 50,
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.seed = seed
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        if not AUTOSKLEARN_AVAILABLE or AutoSklearnRegressor is None:
            raise ImportError("autosklearn 未安装，请运行: pip install auto-sklearn")
        self.model = AutoSklearnRegressor(
            time_left_for_this_task=int(self.time_left_for_this_task),
            per_run_time_limit=int(self.per_run_time_limit),
            ensemble_size=int(self.ensemble_size),
            n_jobs=self.n_jobs,
            seed=self.seed,
            **(self.kwargs or {}),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Auto-sklearn model not fitted yet.")
        return self.model.predict(X)


class TPOTWrapper(BaseEstimator, RegressorMixin):
    """TPOT wrapper (if tpot is installed)."""

    def __init__(
        self,
        max_time_mins: int = 5,
        max_eval_time_mins: int = 2,
        generations: int = 5,
        population_size: int = 50,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.generations = generations
        self.population_size = population_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def _resolve_search_budget(self, X):
        try:
            n_samples = int(getattr(X, "shape", [len(X)])[0])
        except Exception:
            n_samples = 0
        try:
            max_time = float(self.max_time_mins)
        except Exception:
            max_time = 5.0
        try:
            max_eval = float(self.max_eval_time_mins)
        except Exception:
            max_eval = 2.0
        try:
            generations = int(self.generations)
        except Exception:
            generations = 5
        try:
            population_size = int(self.population_size)
        except Exception:
            population_size = 50

        if n_samples >= 5000:
            max_time = min(max_time, 3.0)
            max_eval = min(max_eval, 1.0)
            generations = min(generations, 3)
            population_size = min(population_size, 20)
        elif n_samples >= 1000:
            max_time = min(max_time, 5.0)
            max_eval = min(max_eval, 2.0)
            generations = min(generations, 4)
            population_size = min(population_size, 30)

        max_time = max(0.5, float(max_time))
        max_eval = max(0.1, float(max_eval))
        generations = max(1, int(generations))
        population_size = max(2, int(population_size))
        return max_time, max_eval, generations, population_size

    def fit(self, X, y):
        if not TPOT_AVAILABLE or TPOTRegressor is None:
            raise ImportError("TPOT 未安装，请运行: pip install tpot")
        max_time, max_eval, generations, population_size = self._resolve_search_budget(X)
        self.model = TPOTRegressor(
            max_time_mins=float(max_time),
            max_eval_time_mins=float(max_eval),
            generations=int(generations),
            population_size=int(population_size),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **(self.kwargs or {}),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("TPOT model not fitted yet.")
        return self.model.predict(X)


class FLAMLWrapper(BaseEstimator, RegressorMixin):
    """FLAML wrapper (if flaml is installed)."""

    def __init__(
        self,
        time_budget: int = 60,
        metric: str = "r2",
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.time_budget = time_budget
        self.metric = metric
        self.n_jobs = n_jobs
        self.seed = seed
        self.kwargs = kwargs
        self.automl = None

    def fit(self, X, y):
        if not FLAML_AVAILABLE or AutoML is None:
            raise ImportError("FLAML 未安装，请运行: pip install flaml")
        self.automl = AutoML()
        fit_kwargs = dict(self.kwargs or {})
        fit_kwargs.setdefault("task", "regression")
        fit_kwargs.setdefault("time_budget", float(self.time_budget))
        fit_kwargs.setdefault("metric", self.metric)
        if self.n_jobs is not None:
            fit_kwargs.setdefault("n_jobs", int(self.n_jobs))
        if self.seed is not None:
            fit_kwargs.setdefault("seed", int(self.seed))
        self.automl.fit(X_train=X, y_train=y, **fit_kwargs)
        return self

    def predict(self, X):
        if self.automl is None:
            raise RuntimeError("FLAML model not fitted yet.")
        return self.automl.predict(X)


class ChemSLRegressor(BaseEstimator, RegressorMixin):
    """Chemical SuperLearner (stacking ensemble) for tabular features."""

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = 42,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        fast_mode: bool = True,
        min_estimators: int = 50,
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.fast_mode = fast_mode
        self.min_estimators = min_estimators
        self.model = None

    def _resolve_n_estimators(self, n_samples: int) -> int:
        try:
            base = int(self.n_estimators)
        except Exception:
            base = 200
        if self.fast_mode:
            if n_samples >= 20000:
                base = min(base, 80)
            elif n_samples >= 5000:
                base = min(base, 120)
            elif n_samples >= 1000:
                base = min(base, 160)
        try:
            base = max(int(self.min_estimators), int(base))
        except Exception:
            base = max(50, int(base))
        return base

    def _build_estimators(self, n_samples: int):
        estimators = []
        rs = self.random_state
        n_estimators = self._resolve_n_estimators(n_samples)
        max_depth = self.max_depth
        if max_depth in (None, "None", 0, "0"):
            max_depth = None
        estimators.append((
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=rs,
                n_jobs=self.n_jobs,
            ),
        ))
        estimators.append((
            "et",
            ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=rs,
                n_jobs=self.n_jobs,
            ),
        ))
        estimators.append(("gbrt", GradientBoostingRegressor(random_state=rs)))
        if XGBOOST_AVAILABLE:
            estimators.append((
                "xgb",
                XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=6 if max_depth is None else int(max_depth),
                    learning_rate=0.05,
                    random_state=rs,
                    n_jobs=self.n_jobs,
                ),
            ))
        if LIGHTGBM_AVAILABLE:
            estimators.append((
                "lgb",
                LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=-1 if max_depth is None else int(max_depth),
                    learning_rate=0.05,
                    random_state=rs,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                    force_col_wise=True,
                ),
            ))
        if CATBOOST_AVAILABLE:
            estimators.append((
                "cat",
                CatBoostRegressor(
                    iterations=n_estimators,
                    depth=6 if max_depth is None else int(max_depth),
                    learning_rate=0.05,
                    random_state=rs,
                    verbose=0,
                ),
            ))
        return estimators

    def fit(self, X, y):
        n_samples = int(getattr(X, "shape", [len(X)])[0])
        estimators = self._build_estimators(n_samples)
        if not estimators:
            estimators = [("rf", RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=self.n_jobs))]
        final_estimator = Ridge()
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("ChemSL model not fitted yet.")
        return self.model.predict(X)


class FastPropRegressor(BaseEstimator, RegressorMixin):
    """FastProp baseline using HistGradientBoostingRegressor."""

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 6,
        l2_regularization: float = 0.0,
        random_state: Optional[int] = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = HistGradientBoostingRegressor(
            max_iter=int(self.max_iter),
            learning_rate=float(self.learning_rate),
            max_depth=None if self.max_depth in (None, "None") else int(self.max_depth),
            l2_regularization=float(self.l2_regularization),
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("FastProp model not fitted yet.")
        return self.model.predict(X)


def _build_gpr_kernel(kernel_name="RBF + White", length_scale=1.0, noise_level=0.1, nu=1.5):
    """Build a stable sklearn Gaussian-process kernel from UI-friendly params."""
    try:
        length_scale = float(length_scale)
    except Exception:
        length_scale = 1.0
    try:
        noise_level = float(noise_level)
    except Exception:
        noise_level = 0.1
    try:
        nu = float(nu)
    except Exception:
        nu = 1.5

    length_scale = max(length_scale, 1e-6)
    noise_level = max(noise_level, 1e-10)
    kernel_label = str(kernel_name or "RBF + White").lower()

    signal = ConstantKernel(1.0, (1e-3, 1e3))
    if "matern" in kernel_label:
        base = Matern(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3), nu=nu)
    elif "rational" in kernel_label or "quadratic" in kernel_label:
        base = RationalQuadratic(
            length_scale=length_scale,
            alpha=1.0,
            length_scale_bounds=(1e-3, 1e3),
            alpha_bounds=(1e-3, 1e3),
        )
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))

    if "white" in kernel_label or "noise" in kernel_label:
        return signal * base + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1e2))
    return signal * base



def _normalize_train_n_jobs(train_n_jobs):
    """Normalize user-specified parallelism.
    -1 -> all cores; None/0/invalid -> -1; positive int -> that many.
    """
    if train_n_jobs is None:
        return -1
    try:
        v = int(train_n_jobs)
    except Exception:
        return -1
    if v == 0:
        return -1
    if v < -1:
        return -1
    return v


def _resolve_catboost_bootstrap_conflict(params_clean: dict) -> dict:
    """Normalize CatBoost bootstrap parameters to avoid invalid UI combinations.

    CatBoost's default CPU bootstrap type is Bayesian, which supports
    ``bagging_temperature`` but not ``subsample``. The current UI exposes both,
    so this helper resolves the conflict conservatively.
    """
    params_clean = dict(params_clean or {})
    bootstrap_type = str(params_clean.get("bootstrap_type", "") or "").strip().lower()
    has_subsample = "subsample" in params_clean and params_clean.get("subsample") is not None
    has_bagging_temperature = (
        "bagging_temperature" in params_clean
        and params_clean.get("bagging_temperature") is not None
    )

    if bootstrap_type == "bayesian":
        if has_subsample:
            params_clean.pop("subsample", None)
            print("⚠️ CatBoost bootstrap_type=Bayesian does not support subsample; subsample was dropped.")
        return params_clean

    if bootstrap_type:
        if has_bagging_temperature and bootstrap_type not in {"bayesian"}:
            params_clean.pop("bagging_temperature", None)
            print(
                f"⚠️ CatBoost bootstrap_type={bootstrap_type} does not use bagging_temperature; "
                "bagging_temperature was dropped."
            )
        return params_clean

    if has_subsample and has_bagging_temperature:
        params_clean.pop("subsample", None)
        print(
            "⚠️ CatBoost UI passed both subsample and bagging_temperature. "
            "Kept CatBoost default Bayesian bootstrap and dropped subsample."
        )
        return params_clean

    return params_clean


def _build_missing_value_imputer(
    strategy: str = "median",
    random_state: int = 42,
    max_iter: int = 15,
    n_imputations: int = 5,
):
    strategy_norm = str(strategy or "median").strip().lower()
    if strategy_norm not in {"median", "bayesian", "multiple_bayesian"}:
        strategy_norm = "median"
    if MissingValueHandler is None:
        if strategy_norm != "median":
            print(
                f"⚠️ MissingValueHandler unavailable, fallback to median imputation "
                f"(requested={strategy_norm})"
            )
        return SimpleImputer(strategy="median")
    return MissingValueHandler(
        strategy=strategy_norm,
        random_state=int(random_state),
        max_iter=max(5, int(max_iter)),
        n_imputations=max(1, int(n_imputations)),
    )


def _get_mem_available_bytes():
    """Best-effort读取可用内存（Linux /proc/meminfo）。"""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _detect_gpu_availability():
    """检测GPU可用性并返回设备信息"""
    gpu_info = {
        'cuda_available': False,
        'xgb_gpu': False,
        'lgb_gpu': False,
        'catboost_gpu': False,
        'tensorflow_gpu': False,
        'torch_gpu': False,
        'device_count': 0,
        'device_name': None
    }
    
    # 检测CUDA (PyTorch)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['torch_gpu'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    
    # 检测TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info['tensorflow_gpu'] = True
            gpu_info['device_count'] = max(gpu_info['device_count'], len(gpus))
    except Exception:
        pass
    
    # XGBoost GPU支持
    try:
        import xgboost as xgb
        # XGBoost 2.0+ 使用 device="cuda"
        gpu_info['xgb_gpu'] = gpu_info['cuda_available']
    except Exception:
        pass
    
    # LightGBM GPU支持
    try:
        import lightgbm as lgb
        gpu_info['lgb_gpu'] = gpu_info['cuda_available']
    except Exception:
        pass
    
    # CatBoost GPU支持
    try:
        import catboost
        gpu_info['catboost_gpu'] = gpu_info['cuda_available']
    except Exception:
        pass
    
    return gpu_info


# 全局GPU信息缓存
_GPU_INFO_CACHE = None

def get_gpu_info(refresh=False):
    """获取GPU信息（带缓存）"""
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is None or refresh:
        _GPU_INFO_CACHE = _detect_gpu_availability()
    return _GPU_INFO_CACHE


def _apply_parallel_settings(model_name: str, model_params: dict, train_n_jobs: int, use_gpu: bool = True):
    """Inject model-specific parallel params into model_params (in-place).
    
    Parameters
    ----------
    model_name : str
        模型名称
    model_params : dict
        模型参数字典
    train_n_jobs : int
        并行作业数
    use_gpu : bool
        是否启用GPU加速（默认True，自动检测可用性）
    
    Returns
    -------
    dict : 更新后的模型参数
    """
    train_n_jobs = _normalize_train_n_jobs(train_n_jobs)
    # 限制最大线程数，避免512核机器OOM崩溃
    max_safe_threads = min(16, os.cpu_count() or 16)
    threads = max_safe_threads if train_n_jobs == -1 else max(1, min(int(train_n_jobs), max_safe_threads))
    
    # 获取GPU信息
    gpu_info = get_gpu_info() if use_gpu else {'cuda_available': False}

    # sklearn-style n_jobs models（支持多核CPU）
    if model_name in {"线性回归", "随机森林", "Extra Trees"}:
        safe_n_jobs = train_n_jobs
        if safe_n_jobs == -1:
            safe_n_jobs = min(16, os.cpu_count() or 16)
        model_params.setdefault("n_jobs", safe_n_jobs)
    
    # 决策树和梯度提升树不支持并行，但可以通过其他方式优化
    if model_name == "梯度提升树":
        # GradientBoostingRegressor 不支持 n_jobs，但可以设置 n_iter_no_change
        pass
    
    # AdaBoost 不支持并行
    if model_name == "AdaBoost":
        pass
    
    # SVR 不支持并行，但可以使用 LinearSVR 替代大数据集
    if model_name == "SVR":
        pass
    
    # 多层感知器（sklearn）不直接支持 GPU，但可以优化线程
    if model_name == "多层感知器":
        pass

    if model_name in {"Chemical SuperLearner (ChemSL)", "Auto-sklearn", "TPOT", "FLAML"}:
        # [性能修复] 使用全部CPU核心，不再限制为16
        safe_n_jobs = train_n_jobs
        if safe_n_jobs == -1:
            # 使用全部核心（移除16核限制）
            safe_n_jobs = os.cpu_count() or -1
        model_params.setdefault("n_jobs", safe_n_jobs)

    if model_name in GRAPH_MODEL_NAMES or model_name in RAW_FRAME_MODELS_WITH_SMILES:
        if int(train_n_jobs) > 0:
            model_params.setdefault("num_workers", int(train_n_jobs))
        else:
            model_params.setdefault("num_workers", 0)
    
    # XGBoost 多核和GPU优化
    if model_name == "XGBoost":
        try:
            import xgboost as xgb
            xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
        except Exception:
            xgb_version = (1, 0)

        # GPU训练时CPU线程数限制为4，避免256核机器OOM崩溃
        xgb_use_gpu = gpu_info.get('xgb_gpu') and use_gpu
        if xgb_use_gpu:
            model_params.setdefault("n_jobs", 4)
        else:
            # 限制最大线程数，避免512核机器OOM崩溃
            safe_n_jobs = train_n_jobs
            if safe_n_jobs == -1:
                safe_n_jobs = min(16, os.cpu_count() or 16)
            model_params.setdefault("n_jobs", safe_n_jobs)
        # 不设置 nthread，避免与 n_jobs 冲突
        
        # tree_method='hist' 是 CPU 上最快的方法，支持更好的多线程
        # 'approx' 和 'exact' 在大数据集上较慢
        if 'tree_method' not in model_params and 'device' not in model_params:
            model_params.setdefault("tree_method", "hist")
        
        # === GPU 加速 ===
        if gpu_info.get('xgb_gpu') and use_gpu:
            # 检查是否指定了GPU设备ID
            gpu_device_id = model_params.pop('gpu_device_id', None)

            if xgb_version >= (2, 0):
                # XGBoost 2.0+ 使用统一的 device 参数
                if gpu_device_id is not None:
                    model_params["device"] = f"cuda:{gpu_device_id}"
                    print(f"✓ XGBoost 将使用 GPU {gpu_device_id}")
                else:
                    model_params["device"] = "cuda"
                    print(f"✓ XGBoost 将使用默认 GPU")
                model_params.pop("tree_method", None)  # device="cuda" 自动选择最优方法
            else:
                # XGBoost 1.x 使用 tree_method
                model_params["tree_method"] = "gpu_hist"
                model_params.setdefault("predictor", "gpu_predictor")
                if gpu_device_id is not None:
                    model_params["gpu_id"] = gpu_device_id
                    print(f"✓ XGBoost 将使用 GPU {gpu_device_id}")
                else:
                    model_params["n_gpus"] = -1
                    print(f"✓ XGBoost 将使用所有可用 GPU")
        
        # === 额外性能优化参数 ===
        # 禁用不必要的输出
        model_params.setdefault("verbosity", 0)
        # 对于 CPU 训练，使用 grow_policy='lossguide' 可能更快
        if not (gpu_info.get('xgb_gpu') and use_gpu):
            # 仅在非GPU模式下设置
            model_params.setdefault("max_bin", 256)  # 减少bin数可加速，但可能影响精度
    
    # LightGBM 多核和GPU优化
    if model_name == "LightGBM":
        # === 多核 CPU 优化 ===
        # 限制最大线程数，避免512核机器OOM崩溃
        safe_n_jobs = train_n_jobs
        if safe_n_jobs == -1:
            safe_n_jobs = min(16, os.cpu_count() or 16)
        model_params.setdefault("n_jobs", safe_n_jobs)
        model_params.setdefault("num_threads", threads)  # LightGBM 原生线程参数

        # === GPU 加速 ===
        if gpu_info.get('lgb_gpu') and use_gpu:
            gpu_device_id = model_params.pop('gpu_device_id', None)
            model_params.setdefault("device", "gpu")
            model_params.setdefault("gpu_platform_id", 0)
            if gpu_device_id is not None:
                model_params["gpu_device_id"] = gpu_device_id
                print(f"✓ LightGBM 将使用 GPU {gpu_device_id}")
            else:
                model_params.setdefault("gpu_device_id", 0)
                print(f"✓ LightGBM 将使用默认 GPU")

        # === 额外性能优化参数 ===
        model_params.setdefault("verbose", -1)
        # 使用直方图算法以获得更好的多核性能
        model_params.setdefault("boosting_type", "gbdt")
        # force_col_wise 通常比 force_row_wise 更快（特别是特征数较多时）
        model_params.setdefault("force_col_wise", True)
        # 增加 max_bin 可以提高精度，但会稍微降低速度（默认255）
        # model_params.setdefault("max_bin", 255)
        # 使用更快的直方图构建算法
        model_params.setdefault("histogram_pool_size", -1)  # 自动选择最优内存池大小

    # CatBoost GPU加速
    if model_name == "CatBoost":
        gpu_device_id = model_params.pop('gpu_device_id', None)
        if gpu_info.get('catboost_gpu') and use_gpu:
            model_params.setdefault("thread_count", min(16, threads))  # GPU模式下CPU线程不需要太多
            model_params.setdefault("task_type", "GPU")
            if gpu_device_id is not None:
                model_params["devices"] = str(gpu_device_id)
                print(f"✓ CatBoost 将使用 GPU {gpu_device_id}")
            else:
                dev_cnt = int(gpu_info.get('device_count') or 1)
                model_params.setdefault("devices", ":".join(str(i) for i in range(dev_cnt)))
                print(f"✓ CatBoost 将使用所有可用 GPU")
        else:
            model_params.setdefault("thread_count", threads)

    # AutoGluon GPU/CPU 资源设置
    if model_name == "AutoGluon":
        model_params.setdefault("num_cpus", min(16, threads))  # Windows限制，避免卡死
        if gpu_info.get('cuda_available') and use_gpu:
            dev_cnt = int(gpu_info.get('device_count') or 1)
            model_params.setdefault("num_gpus", max(1, dev_cnt))
        else:
            model_params.setdefault("num_gpus", 0)

    # Best-effort BLAS thread env (may help some solvers / MLP / numpy ops)
    try:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    except Exception:
        pass

    return model_params


def _apply_gpu_settings_for_neural_networks(model_name: str, model_params: dict, use_gpu: bool = True):
    """为神经网络模型应用GPU设置
    
    Parameters
    ----------
    model_name : str
        模型名称
    model_params : dict
        模型参数字典
    use_gpu : bool
        是否启用GPU
    
    Returns
    -------
    dict : 更新后的模型参数
    """
    gpu_info = get_gpu_info() if use_gpu else {'cuda_available': False}
    gpu_device_id = model_params.pop('gpu_device_id', None)
    
    # ANN: 只支持 device 参数，不支持 use_gpu
    if model_name in {
        "人工神经网络",
        "Bayesian Neural Network (BNN)",
        "FT-Transformer",
        "Transformer + BNN",
        "Transformer + PINN",
        "GNN + Transformer Fusion",
    }:
        if gpu_info.get('cuda_available') and use_gpu:
            if gpu_device_id is not None:
                device_str = f"cuda:{int(gpu_device_id)}"
                model_params["device"] = device_str
                if "gpu_ids" in model_params:
                    current_ids = [s.strip() for s in str(model_params.get("gpu_ids", "")).split(",") if s.strip()]
                    selected_id = str(int(gpu_device_id))
                    reordered_ids = [selected_id] + [gpu_id for gpu_id in current_ids if gpu_id != selected_id]
                    model_params["gpu_ids"] = ",".join(reordered_ids)
            else:
                model_params.setdefault("device", "cuda")
        else:
            model_params.setdefault("device", "cpu")
    
    # PINN: 强制覆盖 device，不用 setdefault（避免 UI 传入的 'auto'/'cpu' 阻止GPU）
    elif model_name == "Epoxy PINN (Physics-Informed)":
        if gpu_info.get('cuda_available') and use_gpu:
            if gpu_device_id is not None:
                model_params["device"] = f"cuda:{int(gpu_device_id)}"
            else:
                model_params["device"] = "cuda"
        else:
            model_params["device"] = "cpu"
            model_params.setdefault("device", "cpu")

    elif model_name in GRAPH_MODEL_NAMES:
        if gpu_info.get('cuda_available') and use_gpu:
            if gpu_device_id is not None:
                model_params["device"] = f"cuda:{int(gpu_device_id)}"
            else:
                model_params.setdefault("device", "cuda")
        else:
            model_params.setdefault("device", "cpu")
    
    # TensorFlow: 自动检测GPU，不需要任何参数
    # elif model_name == "TensorFlow Sequential":
    #     pass  # TensorFlow 自动处理 GPU
    
    return model_params


class EnhancedModelTrainer:
    """增强版模型训练器 - 支持GPU加速和优化的并行训练"""

    def __init__(self, use_gpu: bool = True):
        """初始化训练器
        
        Parameters
        ----------
        use_gpu : bool
            是否启用GPU加速（默认True，自动检测可用性）
        """
        self.use_gpu = use_gpu
        self.gpu_info = get_gpu_info()
        # 统一用 catalog 维护"是否可用 + 缺失原因"，避免 UI 侧难以解释
        self.model_catalog = self._get_model_catalog()
        self.available_models = [m for m, meta in self.model_catalog.items() if meta.get('available', True)]
    
    def get_gpu_status(self):
        """获取GPU状态信息"""
        return {
            'gpu_enabled': self.use_gpu,
            'gpu_available': self.gpu_info.get('cuda_available', False),
            'device_name': self.gpu_info.get('device_name', 'N/A'),
            'device_count': self.gpu_info.get('device_count', 0),
            'xgb_gpu': self.gpu_info.get('xgb_gpu', False),
            'lgb_gpu': self.gpu_info.get('lgb_gpu', False),
            'catboost_gpu': self.gpu_info.get('catboost_gpu', False),
            'tensorflow_gpu': self.gpu_info.get('tensorflow_gpu', False),
            'torch_gpu': self.gpu_info.get('torch_gpu', False),
        }








    def _get_model_catalog(self):
        """返回模型目录：{model_name: {available: bool, reason: str}}。

        设计目标：
        - UI 可以“始终显示入口”，即使依赖未安装也能给出明确原因
        - 保持核心训练逻辑不变（缺依赖时在 _get_model 中抛更清晰错误）
        """
        catalog = {}

        # --- 基础 sklearn 模型（默认可用） ---
        base_models = [
            "线性回归", "Ridge回归", "Lasso回归", "ElasticNet",
            "决策树", "随机森林", "Extra Trees", "梯度提升树",
            "AdaBoost", "SVR", "多层感知器", "Gaussian Process (GPR)",
        ]
        for m in base_models:
            catalog[m] = {"available": True, "reason": ""}

        catalog["逻辑回归分类"] = {"available": True, "reason": ""}
        catalog["随机森林分类"] = {"available": True, "reason": ""}

        # --- 可选依赖模型 ---
        catalog["XGBoost"] = {
            "available": bool(XGBOOST_AVAILABLE),
            "reason": "" if XGBOOST_AVAILABLE else "未安装 xgboost（pip install xgboost）",
        }
        catalog["XGBoost分类"] = {
            "available": bool(XGBOOST_AVAILABLE and _XGB_CLASSIFIER_AVAILABLE),
            "reason": "" if (XGBOOST_AVAILABLE and _XGB_CLASSIFIER_AVAILABLE) else "未安装 xgboost（pip install xgboost）",
        }
        catalog["LightGBM"] = {
            "available": bool(LIGHTGBM_AVAILABLE),
            "reason": "" if LIGHTGBM_AVAILABLE else "未安装 lightgbm（pip install lightgbm）",
        }
        catalog["LightGBM分类"] = {
            "available": bool(LIGHTGBM_AVAILABLE and _LGBM_CLASSIFIER_AVAILABLE),
            "reason": "" if (LIGHTGBM_AVAILABLE and _LGBM_CLASSIFIER_AVAILABLE) else "未安装 lightgbm（pip install lightgbm）",
        }
        catalog["CatBoost"] = {
            "available": bool(CATBOOST_AVAILABLE),
            "reason": "" if CATBOOST_AVAILABLE else "未安装 catboost（pip install catboost）",
        }

        catalog["CatBoost分类"] = {
            "available": bool(CATBOOST_AVAILABLE and _CATBOOST_CLASSIFIER_AVAILABLE),
            "reason": "" if (CATBOOST_AVAILABLE and _CATBOOST_CLASSIFIER_AVAILABLE) else "未安装 catboost（pip install catboost）",
        }

        # TensorFlow Sequential (TFS)
        catalog["TensorFlow Sequential"] = {
            "available": bool(TENSORFLOW_AVAILABLE),
            "reason": "" if TENSORFLOW_AVAILABLE else "未安装 TensorFlow（pip install tensorflow）",
        }

        # 自定义 ANN
        catalog["人工神经网络"] = {
            "available": bool(ANN_AVAILABLE),
            "reason": "" if ANN_AVAILABLE else "ANNRegressor 不可用（检查 core/ann_model.py 依赖）",
        }

        catalog["Bayesian Neural Network (BNN)"] = {
            "available": bool(BNN_AVAILABLE),
            "reason": "" if BNN_AVAILABLE else "BNN 模块不可用（需要 PyTorch 环境）",
        }

        catalog["Transformer + BNN"] = {
            "available": bool(TRANSFORMER_BNN_AVAILABLE),
            "reason": "" if TRANSFORMER_BNN_AVAILABLE else "Transformer + BNN 模块不可用（需要 PyTorch + FT-Transformer 主干）",
        }


        # Epoxy PINN (Physics-Informed)
        catalog["Epoxy PINN (Physics-Informed)"] = {
            "available": bool(PINN_AVAILABLE),
            "reason": "" if PINN_AVAILABLE else "未安装 torch 或 PINN 模块不可用（需要 torch>=2.1.0）",
        }

        catalog["Transformer + PINN"] = {
            "available": bool(TRANSFORMER_PINN_AVAILABLE),
            "reason": "" if TRANSFORMER_PINN_AVAILABLE else "Transformer + PINN 模块不可用（需要 PyTorch + FT-Transformer 主干）",
        }

        # TabNet
        catalog["TabNet"] = {
            "available": bool(TABNET_AVAILABLE),
            "reason": "" if TABNET_AVAILABLE else "未安装 pytorch-tabnet（pip install pytorch-tabnet）",
        }

        # FT-Transformer
        catalog["FT-Transformer"] = {
            "available": bool(FT_TRANSFORMER_AVAILABLE),
            "reason": "" if FT_TRANSFORMER_AVAILABLE else "FT-Transformer 模块不可用（需要 PyTorch 环境）",
        }

        catalog["GNN + Transformer Fusion"] = {
            "available": bool(GNN_TRANSFORMER_FUSION_AVAILABLE),
            "reason": "" if GNN_TRANSFORMER_FUSION_AVAILABLE else "GNN + Transformer Fusion 不可用（需要 torch_geometric + RDKit + PyTorch）",
        }

        # TabPFN / AutoGluon
        catalog["TabPFN"] = {
            "available": bool(TABPFN_AVAILABLE),
            "reason": "" if TABPFN_AVAILABLE else "未安装 tabpfn（pip install tabpfn）",
        }
        catalog["AutoGluon"] = {
            "available": bool(AUTOGLUON_AVAILABLE),
            "reason": "" if AUTOGLUON_AVAILABLE else "未安装 autogluon.tabular（pip install autogluon.tabular）",
        }

        catalog["Chemical SuperLearner (ChemSL)"] = {
            "available": True,
            "reason": "",
        }
        catalog["fastprop"] = {
            "available": True,
            "reason": "",
        }
        catalog["Auto-sklearn"] = {
            "available": bool(AUTOSKLEARN_AVAILABLE),
            "reason": "" if AUTOSKLEARN_AVAILABLE else "未安装 auto-sklearn（pip install auto-sklearn）",
        }
        catalog["TPOT"] = {
            "available": bool(TPOT_AVAILABLE),
            "reason": "" if TPOT_AVAILABLE else "未安装 tpot（pip install tpot）",
        }
        catalog["FLAML"] = {
            "available": bool(FLAML_AVAILABLE),
            "reason": "" if FLAML_AVAILABLE else "未安装 flaml（pip install flaml）",
        }

        graph_feature_reason = "图神经网络模型已迁移至特征提取：分子特征提取 -> 图神经网络特征"
        for graph_name in ["GCN", "GAT", "GIN", "GraphSAGE", "MPNN", "AttentiveFP", "D-MPNN"]:
            catalog[graph_name] = {
                "available": False,
                "reason": graph_feature_reason,
            }

        # 过滤掉不可用但用户可能不需要的项：这里不做过滤，交给 UI 决定
        return catalog

    def get_model_catalog(self):
        """获取模型目录（包含可用性与缺失依赖原因）。"""
        return dict(self.model_catalog)

    def get_available_models(self, include_unavailable: bool = False):
        """返回模型列表。

        Parameters
        ----------
        include_unavailable : bool
            True: 返回所有模型（含不可用项，便于 UI 显示入口）
            False: 仅返回可用模型
        """
        if include_unavailable:
            return list(self.model_catalog.keys())
        return self.available_models.copy()

    def _get_model(self, model_name: str, random_state: int = 42, **params):
        """
        根据模型名称返回模型实例（内部方法）
        
        Parameters
        ----------
        model_name : str
            模型名称
        random_state : int
            随机种子
        **params : dict
            模型参数
            
        Returns
        -------
        model : estimator
            sklearn 兼容的模型实例
        """
        # 清理参数中的 random_state 和通用控制参数（避免透传给 sklearn 模型）
        _internal_keys = {'random_state', 'use_gpu', 'gpu_device_id', 'train_n_jobs',
                          'scaler_type', 'normalize_target', 'external_preprocess'}
        params_clean = {k: v for k, v in params.items() if k not in _internal_keys}

        if model_name in GRAPH_MODEL_NAMES:
            raise ValueError("图神经网络模型已迁移至特征提取，请在分子特征提取中使用。")

        if model_name == "线性回归":
            n_jobs = params_clean.pop("n_jobs", None)
            if n_jobs is None:
                return LinearRegression()
            return LinearRegression(n_jobs=n_jobs)

        elif model_name == "Ridge回归":
            params_clean.pop("n_jobs", None)
            if bool(params_clean.get("positive", False)):
                if str(params_clean.get("solver", "auto")).lower() != "lbfgs":
                    print("[Ridge] positive=True requires solver='lbfgs'; overriding solver automatically")
                params_clean["solver"] = "lbfgs"
                params_clean.setdefault("max_iter", 1000)
            return Ridge(random_state=random_state, **params_clean)

        elif model_name == "Lasso回归":
            params_clean.pop("n_jobs", None)
            return Lasso(random_state=random_state, **params_clean)

        elif model_name == "ElasticNet":
            params_clean.pop("n_jobs", None)
            return ElasticNet(random_state=random_state, **params_clean)

        elif model_name == "决策树":
            return DecisionTreeRegressor(random_state=random_state, **params_clean)

        elif model_name == "随机森林":
            n_jobs = params_clean.pop("n_jobs", -1)
            verbose = params_clean.pop("verbose", 1)
            warm_start = params_clean.pop("warm_start", False)
            if params_clean.get("max_depth") == 0:
                params_clean["max_depth"] = None
            rf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs,
                                       verbose=verbose, warm_start=warm_start, **params_clean)
            print(f"[随机森林] n_estimators={rf.n_estimators}, max_depth={rf.max_depth}, "
                  f"n_jobs={n_jobs}, max_features={rf.max_features}, "
                  f"min_samples_split={rf.min_samples_split}, min_samples_leaf={rf.min_samples_leaf}")
            return rf

        elif model_name == "Extra Trees":
            n_jobs = params_clean.pop("n_jobs", -1)
            verbose = params_clean.pop("verbose", 1)
            warm_start = params_clean.pop("warm_start", False)
            if params_clean.get("max_depth") == 0:
                params_clean["max_depth"] = None
            et = ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs,
                                     verbose=verbose, warm_start=warm_start, **params_clean)
            print(f"[Extra Trees] n_estimators={et.n_estimators}, max_depth={et.max_depth}, "
                  f"n_jobs={n_jobs}, max_features={et.max_features}")
            return et

        elif model_name == "梯度提升树":
            verbose = params_clean.pop("verbose", 1)
            gb = GradientBoostingRegressor(random_state=random_state, verbose=verbose, **params_clean)
            print(f"[梯度提升树] n_estimators={gb.n_estimators}, max_depth={gb.max_depth}, "
                  f"learning_rate={gb.learning_rate}")
            return gb

        elif model_name == "AdaBoost":
            return AdaBoostRegressor(random_state=random_state, **params_clean)

        elif model_name == "SVR":
            # 大数据集优化：限制最大样本数
            if hasattr(self, 'X_train') and len(self.X_train) > 5000:
                print(f"[SVR 优化] 数据量大({len(self.X_train)}样本)，建议使用线性核或减少样本")
                if params_clean.get('kernel') == 'rbf':
                    print("[SVR 优化] RBF核在大数据集上很慢，建议切换为'linear'核")

            # 设置缓存大小加速
            if 'cache_size' not in params_clean:
                params_clean['cache_size'] = 1000  # 默认1GB缓存

            return SVR(**params_clean)

        elif model_name == "多层感知器":
            mlp_params = params_clean.copy()
            hidden_layers = mlp_params.get("hidden_layer_sizes")
            if isinstance(hidden_layers, str):
                try:
                    parsed_layers = tuple(
                        int(part.strip()) for part in hidden_layers.split(",") if part.strip()
                    )
                except ValueError as exc:
                    raise ValueError("多层感知器的 hidden_layer_sizes 格式应为如 100,50 的整数列表") from exc
                if parsed_layers:
                    mlp_params["hidden_layer_sizes"] = parsed_layers
                else:
                    mlp_params.pop("hidden_layer_sizes", None)

            mlp_params.setdefault("random_state", random_state)
            mlp_params.setdefault("max_iter", 1000)
            return MLPRegressor(**mlp_params)

        elif model_name == "Gaussian Process (GPR)":
            kernel_name = params_clean.pop("kernel", "RBF + White")
            length_scale = params_clean.pop("length_scale", 1.0)
            noise_level = params_clean.pop("noise_level", 0.1)
            nu = params_clean.pop("nu", 1.5)
            params_clean.pop("n_jobs", None)
            kernel = _build_gpr_kernel(
                kernel_name=kernel_name,
                length_scale=length_scale,
                noise_level=noise_level,
                nu=nu,
            )
            params_clean.setdefault("alpha", 1e-6)
            params_clean.setdefault("normalize_y", True)
            params_clean.setdefault("n_restarts_optimizer", 3)
            return GaussianProcessRegressor(
                kernel=kernel,
                random_state=random_state,
                **params_clean,
            )

        elif model_name == "逻辑回归分类":
            logreg_params = params_clean.copy()
            penalty = str(logreg_params.get("penalty", "l2")).lower()
            solver = str(logreg_params.get("solver", "lbfgs")).lower()
            if penalty == "l1" and solver not in {"liblinear", "saga"}:
                logreg_params["solver"] = "liblinear"
            logreg_params.setdefault("max_iter", 2000)
            return LogisticRegression(random_state=random_state, **logreg_params)

        elif model_name == "随机森林分类":
            n_jobs = params_clean.pop("n_jobs", -1)
            verbose = params_clean.pop("verbose", 0)
            warm_start = params_clean.pop("warm_start", False)
            if params_clean.get("max_depth") == 0:
                params_clean["max_depth"] = None
            return RandomForestClassifier(
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
                warm_start=warm_start,
                **params_clean,
            )

        elif model_name == "XGBoost分类":
            if not (XGBOOST_AVAILABLE and _XGB_CLASSIFIER_AVAILABLE):
                raise ImportError("XGBoost 分类模型不可用，请安装 xgboost")

            n_jobs = params_clean.pop("n_jobs", -1)
            params_clean.pop("nthread", None)
            if n_jobs == -1:
                n_jobs = min(16, os.cpu_count() or 16)

            early_stopping_rounds = params_clean.pop("early_stopping_rounds", None)

            xgb_params = {
                'random_state': random_state,
                'n_jobs': n_jobs,
                'verbosity': params_clean.pop('verbosity', 0),
                'use_label_encoder': False,
            }

            if early_stopping_rounds is not None:
                try:
                    xgb_params['early_stopping_rounds'] = int(early_stopping_rounds)
                except Exception:
                    pass

            xgb_params.update(params_clean)
            return XGBClassifier(**xgb_params)

        elif model_name == "LightGBM分类":
            if not (LIGHTGBM_AVAILABLE and _LGBM_CLASSIFIER_AVAILABLE):
                raise ImportError("LightGBM 分类模型不可用，请安装 lightgbm")
            params_clean.setdefault('verbose', -1)
            n_jobs = params_clean.pop("n_jobs", -1)
            return LGBMClassifier(random_state=random_state, n_jobs=n_jobs, **params_clean)

        elif model_name == "CatBoost分类":
            if not (CATBOOST_AVAILABLE and _CATBOOST_CLASSIFIER_AVAILABLE):
                raise ImportError("CatBoost 分类模型不可用，请安装 catboost")
            params_clean = _resolve_catboost_bootstrap_conflict(params_clean)
            params_clean.setdefault('verbose', 0)
            thread_count = params_clean.pop("thread_count", None)
            if thread_count is not None:
                return CatBoostClassifier(random_state=random_state, thread_count=int(thread_count), **params_clean)
            return CatBoostClassifier(random_state=random_state, **params_clean)

        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost 未安装，请运行: pip install xgboost")

            # 只使用 n_jobs，移除 nthread 避免 Windows 上 OpenMP 线程冲突
            n_jobs = params_clean.pop("n_jobs", -1)
            params_clean.pop("nthread", None)
            # 256核机器上 n_jobs=-1 会OOM崩溃，限制最多16线程
            if n_jobs == -1:
                n_jobs = min(16, os.cpu_count() or 16)

            # 提取early_stopping_rounds参数(如果有)
            early_stopping_rounds = params_clean.pop("early_stopping_rounds", None)

            xgb_params = {
                'random_state': random_state,
                'n_jobs': n_jobs,
                'verbosity': params_clean.pop('verbosity', 0),
            }

            # XGBoost 2.0+ 支持在初始化时设置early_stopping_rounds
            if early_stopping_rounds is not None:
                try:
                    xgb_params['early_stopping_rounds'] = int(early_stopping_rounds)
                    print(f"[DEBUG] 在XGBRegressor初始化时设置early_stopping_rounds={early_stopping_rounds}")
                except Exception:
                    pass

            xgb_params.update(params_clean)
            return XGBRegressor(**xgb_params)

        elif model_name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
            params_clean.setdefault('verbose', -1)
            n_jobs = params_clean.pop("n_jobs", -1)
            return LGBMRegressor(random_state=random_state, n_jobs=n_jobs, **params_clean)

        elif model_name == "CatBoost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost 未安装，请运行: pip install catboost")
            params_clean = _resolve_catboost_bootstrap_conflict(params_clean)
            params_clean.setdefault('verbose', 0)
            thread_count = params_clean.pop("thread_count", None)
            if thread_count is not None:
                return CatBoostRegressor(random_state=random_state, thread_count=int(thread_count), **params_clean)
            return CatBoostRegressor(random_state=random_state, **params_clean)

        elif model_name == "人工神经网络":
            if not ANN_AVAILABLE:
                raise ImportError("ANN 模块不可用")
            # 训练器内部已用 Pipeline 做缺失填充 + 标准化，避免 ANN 内部重复预处理
            params_clean.setdefault('external_preprocess', True)
            # 参数名映射：UI使用 hidden_layer_sizes，模型使用 hidden_layer_sizes_str
            if 'hidden_layer_sizes' in params_clean:
                params_clean['hidden_layer_sizes_str'] = str(params_clean.pop('hidden_layer_sizes'))
            # 移除ANN不支持的参数（UI配置可能包含这些）
            ann_unsupported = ['activation', 'dropout_rate', 'use_gpu', 'scaler_type', 'normalize_target']
            for key in ann_unsupported:
                params_clean.pop(key, None)
            return ANNRegressor(random_state=random_state, **params_clean)

        elif model_name == "Bayesian Neural Network (BNN)":
            if not BNN_AVAILABLE:
                raise ImportError("BNN 模块不可用（需要 PyTorch 环境）")
            params_clean.setdefault('external_preprocess', True)
            if 'hidden_layer_sizes' in params_clean:
                params_clean['hidden_layer_sizes_str'] = str(params_clean.pop('hidden_layer_sizes'))
            bnn_unsupported = [
                'use_gpu',
                'scaler_type',
                'normalize_target',
                'missing_value_strategy',
                'missing_imputer_max_iter',
                'missing_n_imputations',
            ]
            for key in bnn_unsupported:
                params_clean.pop(key, None)
            return BayesianNeuralNetworkRegressor(random_state=random_state, **params_clean)

        elif model_name == "Transformer + BNN":
            if not TRANSFORMER_BNN_AVAILABLE:
                raise ImportError("Transformer + BNN 模块不可用（需要 PyTorch + FT-Transformer 主干）")
            params_clean.setdefault('external_preprocess', False)
            tbnn_unsupported = ['use_gpu', 'scaler_type', 'normalize_target']
            for key in tbnn_unsupported:
                params_clean.pop(key, None)
            return TransformerBNNRegressor(random_state=random_state, **params_clean)

        elif model_name == "Epoxy PINN (Physics-Informed)":

            if not PINN_AVAILABLE:

                raise ImportError("Epoxy PINN 不可用（需要安装 torch>=2.1.0）")

            # 将 random_state 映射为 PINN 的 seed（保持与其它模型一致）

            if "seed" not in params_clean:

                params_clean["seed"] = int(random_state)

            # external_preprocess 对 PINN 不适用（模型内部已完成解析/归一化）

            params_clean.pop("external_preprocess", None)
            
            # 移除PINN不支持的参数
            params_clean.pop("use_gpu", None)

            return EpoxyPINNRegressor(**params_clean)

        elif model_name == "Transformer + PINN":

            if not TRANSFORMER_PINN_AVAILABLE:

                raise ImportError("Transformer + PINN 模块不可用（需要 PyTorch + FT-Transformer 主干）")

            if "seed" not in params_clean:

                params_clean["seed"] = int(random_state)

            params_clean.pop("external_preprocess", None)

            params_clean.pop("use_gpu", None)

            return TransformerPINNRegressor(**params_clean)


        elif model_name == "TensorFlow Sequential":
            # 训练器内部已用 Pipeline 做缺失填充 + 标准化，避免 TFS 内部重复预处理
            # 若未安装 TensorFlow，训练时给出明确提示（模型仍可在 UI 中选择）
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")
            params_clean.setdefault('external_preprocess', True)
            # 移除TensorFlow不支持的参数
            tf_unsupported = ['use_gpu', 'device']
            for key in tf_unsupported:
                params_clean.pop(key, None)
            return TFSequentialRegressor(random_state=random_state, **params_clean)

        elif model_name == "TabNet":
            if not TABNET_AVAILABLE:
                raise ImportError("TabNet 未安装，请运行: pip install pytorch-tabnet")
            # 将 random_state 映射为 TabNet 的 seed
            if "seed" not in params_clean:
                params_clean["seed"] = int(random_state)

            # 处理学习率参数 (从UI的learning_rate转换为optimizer_params)
            if "learning_rate" in params_clean:
                lr = params_clean.pop("learning_rate")
                if "optimizer_params" not in params_clean:
                    params_clean["optimizer_params"] = {}
                params_clean["optimizer_params"]["lr"] = lr

            # 移除TabNet不支持的参数
            tabnet_unsupported = ['external_preprocess', 'use_gpu']
            for key in tabnet_unsupported:
                params_clean.pop(key, None)
            return TabNetRegressor(**params_clean)

        elif model_name == "FT-Transformer":
            if not FT_TRANSFORMER_AVAILABLE:
                raise ImportError("FT-Transformer 模块不可用（需要 PyTorch 环境）")
            # 将 random_state 映射为 FT-Transformer 的 seed
            if "seed" not in params_clean:
                params_clean["seed"] = int(random_state)
            params_clean.setdefault('external_preprocess', False)
            # 移除FT-Transformer不支持的参数
            ft_unsupported = ['use_gpu', 'scaler_type', 'normalize_target']
            for key in ft_unsupported:
                params_clean.pop(key, None)
            return FTTransformerRegressor(**params_clean)

        elif model_name == "GNN + Transformer Fusion":
            if not GNN_TRANSFORMER_FUSION_AVAILABLE:
                raise ImportError("GNN + Transformer Fusion 不可用（需要 torch_geometric + RDKit + PyTorch）")
            fusion_unsupported = ['use_gpu', 'scaler_type', 'normalize_target']
            for key in fusion_unsupported:
                params_clean.pop(key, None)
            params_clean.setdefault("random_state", int(random_state))
            return GNNTransformerFusionRegressor(**params_clean)

        elif model_name == "TabPFN":
            if not TABPFN_AVAILABLE:
                raise ImportError("TabPFN 未安装，请运行: pip install tabpfn")
            return TabPFNRegressor(**params_clean)

        elif model_name == "AutoGluon":
            if not AUTOGLUON_AVAILABLE:
                raise ImportError("AutoGluon 未安装")
            # 移除AutoGluon不支持的参数
            ag_unsupported = ['use_gpu', 'gpu_device_id']
            for key in ag_unsupported:
                params_clean.pop(key, None)
            return AutoGluonWrapper(**params_clean)

        elif model_name == "Chemical SuperLearner (ChemSL)":
            return ChemSLRegressor(
                n_jobs=params_clean.get("n_jobs"),
                random_state=random_state,
                n_estimators=params_clean.get("n_estimators", 200),
                max_depth=params_clean.get("max_depth"),
                fast_mode=params_clean.get("fast_mode", True),
                min_estimators=params_clean.get("min_estimators", 50),
            )

        elif model_name == "fastprop":
            return FastPropRegressor(random_state=random_state, **params_clean)

        elif model_name == "Auto-sklearn":
            if not AUTOSKLEARN_AVAILABLE:
                raise ImportError("auto-sklearn 未安装，请运行: pip install auto-sklearn")
            params_clean.setdefault("seed", int(random_state))
            return AutoSklearnWrapper(**params_clean)

        elif model_name == "TPOT":
            if not TPOT_AVAILABLE:
                raise ImportError("TPOT 未安装，请运行: pip install tpot")
            params_clean.setdefault("random_state", int(random_state))
            return TPOTWrapper(**params_clean)

        elif model_name == "FLAML":
            if not FLAML_AVAILABLE:
                raise ImportError("FLAML 未安装，请运行: pip install flaml")
            params_clean.setdefault("seed", int(random_state))
            return FLAMLWrapper(**params_clean)

        elif model_name in GRAPH_MODEL_NAMES:
            if not PYG_AVAILABLE or PyGGraphRegressor is None:
                raise ImportError("torch_geometric 或 RDKit 未安装，无法使用图神经网络模型")
            model_map = {
                "GCN": "gcn",
                "GAT": "gat",
                "GIN": "gin",
                "GraphSAGE": "graphsage",
                "MPNN": "mpnn",
                "AttentiveFP": "attentivefp",
                "D-MPNN": "dmpnn",
            }
            params_clean.setdefault("model_type", model_map.get(model_name, "gcn"))
            params_clean.setdefault("random_state", int(random_state))
            return PyGGraphRegressor(**params_clean)

        else:
            raise ValueError(f"未知模型: {model_name}")

    def get_model(self, model_name: str, random_state: int = 42, **params):
        """
        公开的获取模型方法
        
        Parameters
        ----------
        model_name : str
            模型名称
        random_state : int
            随机种子
        **params : dict
            模型参数
        """
        return self._get_model(model_name, random_state, **params)

    def _resolve_split(self, X, y, test_size, random_state, split_strategy='random', n_bins=10, groups=None):
        """根据 split_strategy 生成 train/test 索引"""
        n = len(y)
        idx = np.arange(n)

        split_strategy = (split_strategy or 'random').lower()

        if split_strategy in ['random', '随机', 'random_split']:
            tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
            return np.array(tr), np.array(te)

        if split_strategy in ['stratified', '分层', 'stratified_split']:
            bins = _make_y_bins(y, n_bins=n_bins)
            if bins is None:
                tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
                return np.array(tr), np.array(te)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr, te = next(sss.split(idx.reshape(-1, 1), bins))
            return np.array(tr), np.array(te)

        if split_strategy in ['group', '分组', 'group_split']:
            if groups is None:
                raise ValueError("split_strategy=group 需要提供 groups")
            groups = np.asarray(groups)
            if groups.shape[0] != n:
                raise ValueError("groups 长度必须与样本数一致")

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr, te = next(gss.split(idx.reshape(-1, 1), y, groups))
            return np.array(tr), np.array(te)

        # fallback
        tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
        return np.array(tr), np.array(te)

    def _train_pinn_special(

        self,

        X,

        y,

        model_name,

        test_size=0.2,

        random_state=42,

        split_strategy='random',

        n_bins=10,

        groups=None,

        target_balance_enabled=True,

        balance_n_bins=10,

        balance_max_weight=3.0,

        **params

    ):

        """Epoxy PINN 专用训练分支：允许输入包含带单位字符串列（如 nanofiller_content），由模型内部解析。"""

        if isinstance(X, pd.DataFrame):

            X_df = X.copy()

            feature_names = X_df.columns.tolist()

        else:

            X_arr = np.asarray(X)

            feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]

            X_df = pd.DataFrame(X_arr, columns=feature_names)


        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)


        mask = np.isfinite(y_arr)

        X_df = X_df.loc[mask].reset_index(drop=True)

        y_arr = y_arr[mask]

        if groups is not None:

            groups = np.asarray(groups)[mask]


        if len(y_arr) == 0:

            raise ValueError("所有样本的目标变量均无效（NaN/Inf），无法训练 Epoxy PINN")


        train_idx, test_idx = self._resolve_split(

            X_df, y_arr, test_size=test_size, random_state=random_state,

            split_strategy=split_strategy, n_bins=n_bins, groups=groups

        )


        X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)

        X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)

        y_train = y_arr[train_idx]

        y_test = y_arr[test_idx]

        balance_info = _build_target_balance_info(
            y_train,
            enabled=target_balance_enabled,
            n_bins=balance_n_bins,
            max_weight=balance_max_weight,
            random_state=random_state,
        )


        params.pop('train_n_jobs', None)

        model_params = params.copy()

        model_params.setdefault("seed", int(random_state))

        if _should_pass_target_name(model_name) and "target_name" not in model_params and hasattr(y, "name"):

            model_params["target_name"] = getattr(y, "name", None)


        model = self._get_model(model_name, random_state=int(random_state), **model_params)


        start_time = time.time()

        fit_data = _prepare_balanced_fit_data(
            model,
            X_train_raw,
            y_train,
            balance_info,
            random_state=random_state,
        )
        fit_kwargs = {}
        if fit_data["sample_weight"] is not None:
            fit_kwargs["sample_weight"] = fit_data["sample_weight"]
        model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)

        train_time = time.time() - start_time


        pipeline = Pipeline(steps=[('model', model)])


        y_pred_test = pipeline.predict(X_test_raw)

        y_pred_train = pipeline.predict(X_train_raw)


        metrics = {

            'train_r2': float(r2_score(y_train, y_pred_train)),

            'test_r2': float(r2_score(y_test, y_pred_test)),

            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),

            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),

            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),

            'test_mae': float(mean_absolute_error(y_test, y_pred_test)),

            'train_time': float(train_time),

        }


        training_history = extract_history_from_fitted_model(

            model_name=model_name,

            model=model,

            X_train_scaled=None,

            y_train=np.asarray(y_train),

            X_test_scaled=None,

            y_test=np.asarray(y_test),

        )

        balance_result = _finalize_target_balance_result(
            balance_info,
            actual_method=fit_data["method"],
            fit_sample_count=len(fit_data["y"]),
        )
        test_bin_metrics = _compute_regression_bin_metrics(
            y_test,
            y_pred_test,
            balance_info.get("bin_edges", []),
        )


        return {
            'model': model,
            'pipeline': pipeline,
            'scaler': None,
            'imputer': None,
            # 供 UI/分析使用：返回与普通模型一致的“数值化+标准化”特征矩阵（来自 PINN 内部预处理）
            'X_train': pd.DataFrame(model._transform(X_train_raw)[0], columns=getattr(getattr(model, "_prep_", None), "feature_names", None)
                                    or [f"feat_{i}" for i in range(model._transform(X_train_raw)[0].shape[1])]),
            'X_test': pd.DataFrame(model._transform(X_test_raw)[0], columns=getattr(getattr(model, "_prep_", None), "feature_names", None)
                                   or [f"feat_{i}" for i in range(model._transform(X_test_raw)[0].shape[1])]),
            # 原始数据（含字符串列）保留，供 PINN 预测/解析使用
            'X_train_raw': X_train_raw,
            'X_test_raw': X_test_raw,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            # 与 app.py 指标展示对齐（Test 指标）
            'r2': float(r2_score(y_test, y_pred_test)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'train_time': float(train_time),
            'split_strategy': split_strategy,
            'n_bins': int(n_bins),
            'train_indices': train_idx,
            'test_indices': test_idx,
            'training_history': training_history,
            'training_history_df': history_to_frame(training_history) if training_history else pd.DataFrame(),
            'target_balance': balance_result,
            'test_bin_metrics': test_bin_metrics,


        }

    def _resolve_smiles_col(self, X: pd.DataFrame, smiles_col: Optional[str] = None) -> Optional[str]:
        if smiles_col and smiles_col in X.columns:
            return smiles_col
        for c in X.columns:
            if "smiles" in str(c).lower():
                return c
        text_cols = [c for c in X.columns if X[c].dtype == object or X[c].dtype.name == "string"]
        if text_cols:
            return text_cols[0]
        return None

    def _train_graph_model(
        self,
        X,
        y,
        model_name,
        test_size=0.2,
        random_state=42,
        split_strategy="random",
        n_bins=10,
        groups=None,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        **params,
    ):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("图神经网络模型需要 DataFrame 输入，并包含 SMILES 列")

        smiles_col = self._resolve_smiles_col(X, params.pop("smiles_col", None))
        if not smiles_col:
            raise ValueError("未检测到 SMILES 列，请在训练参数中指定 smiles_col")

        X_df = X[[smiles_col]].copy()
        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)
        mask = np.isfinite(y_arr)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if groups is not None:
            groups = np.asarray(groups)[mask]

        if len(y_arr) == 0:
            raise ValueError("所有样本的目标变量均无效（NaN/Inf），无法训练图模型")

        train_idx, test_idx = self._resolve_split(
            X_df,
            y_arr,
            test_size=test_size,
            random_state=random_state,
            split_strategy=split_strategy,
            n_bins=n_bins,
            groups=groups,
        )

        X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        balance_info = _build_target_balance_info(
            y_train,
            enabled=target_balance_enabled,
            n_bins=balance_n_bins,
            max_weight=balance_max_weight,
            random_state=random_state,
        )

        train_n_jobs = params.pop("train_n_jobs", -1)
        model_params = params.copy()
        requested_use_gpu = bool(model_params.get("use_gpu", self.use_gpu))
        _apply_parallel_settings(model_name, model_params, train_n_jobs, use_gpu=requested_use_gpu)
        _apply_gpu_settings_for_neural_networks(model_name, model_params, use_gpu=requested_use_gpu)
        model_params.setdefault("smiles_col", smiles_col)
        model_params.setdefault("random_state", int(random_state))

        model = self._get_model(model_name, random_state=int(random_state), **model_params)

        start_time = time.time()
        fit_data = _prepare_balanced_fit_data(
            model,
            X_train_raw,
            y_train,
            balance_info,
            random_state=random_state,
        )
        fit_kwargs = {}
        if fit_data["sample_weight"] is not None:
            fit_kwargs["sample_weight"] = fit_data["sample_weight"]
        model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)
        train_time = time.time() - start_time

        pipeline = Pipeline(steps=[("model", model)])
        y_pred_test = np.asarray(pipeline.predict(X_test_raw)).ravel()
        y_pred_train = np.asarray(pipeline.predict(X_train_raw)).ravel()

        def _safe_metrics(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                return float("nan"), float("nan"), float("nan")
            return (
                float(r2_score(y_true[mask], y_pred[mask])),
                float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                float(mean_absolute_error(y_true[mask], y_pred[mask])),
            )

        r2_val, rmse_val, mae_val = _safe_metrics(y_test, y_pred_test)
        balance_result = _finalize_target_balance_result(
            balance_info,
            actual_method=fit_data["method"],
            fit_sample_count=len(fit_data["y"]),
        )
        test_bin_metrics = _compute_regression_bin_metrics(
            y_test,
            y_pred_test,
            balance_info.get("bin_edges", []),
        )

        return {
            "model": model,
            "pipeline": pipeline,
            "scaler": None,
            "imputer": None,
            "X_train": X_train_raw,
            "X_test": X_test_raw,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred_test,
            "y_pred_test": y_pred_test,
            "y_pred_train": y_pred_train,
            "r2": r2_val,
            "rmse": rmse_val,
            "mae": mae_val,
            "train_time": float(train_time),
            "split_strategy": split_strategy,
            "n_bins": int(n_bins),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "training_history": None,
            "training_history_df": pd.DataFrame(),
            "target_balance": balance_result,
            "test_bin_metrics": test_bin_metrics,
        }

    def _train_raw_frame_model(
        self,
        X,
        y,
        model_name,
        test_size=0.2,
        random_state=42,
        split_strategy="random",
        n_bins=10,
        groups=None,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        process_pls_config=None,
        use_process_pls=False,
        **params,
    ):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            X_df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])

        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)
        mask = np.isfinite(y_arr)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if groups is not None:
            groups = np.asarray(groups)[mask]

        if len(y_arr) == 0:
            raise ValueError(f"所有样本的目标变量均无效（NaN/Inf），无法训练 {model_name}")

        train_n_jobs = params.pop("train_n_jobs", -1)
        model_params = params.copy()
        requested_use_gpu = bool(model_params.get("use_gpu", self.use_gpu))
        _apply_parallel_settings(model_name, model_params, train_n_jobs, use_gpu=requested_use_gpu)
        _apply_gpu_settings_for_neural_networks(model_name, model_params, use_gpu=requested_use_gpu)

        if model_name in RAW_FRAME_MODELS_WITH_SMILES:
            smiles_col = self._resolve_smiles_col(X_df, model_params.pop("smiles_col", None))
            if not smiles_col:
                raise ValueError("未检测到 SMILES 列，请在训练参数中指定 smiles_col")
            model_params["smiles_col"] = smiles_col

        train_idx, test_idx = self._resolve_split(
            X_df,
            y_arr,
            test_size=test_size,
            random_state=random_state,
            split_strategy=split_strategy,
            n_bins=n_bins,
            groups=groups,
        )

        X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        if len(y_train) >= 20:
            fit_idx, early_idx = train_test_split(
                np.arange(len(y_train), dtype=int),
                test_size=0.15,
                random_state=random_state,
            )
            if len(early_idx) < 4:
                fit_idx = np.arange(len(y_train), dtype=int)
                early_idx = np.asarray([], dtype=int)
        else:
            fit_idx = np.arange(len(y_train), dtype=int)
            early_idx = np.asarray([], dtype=int)

        if use_process_pls and model_name in RAW_FRAME_MODELS_WITH_SMILES:
            raise ValueError("工艺 PLS 暂不支持含 SMILES 的原始帧融合模型，请先关闭工艺 PLS")

        process_pls_step = _make_process_pls_step(
            process_pls_config,
            use_process_pls,
            X_df.columns.tolist(),
        )
        pipeline_steps = []
        X_train_model_frame = X_train_raw
        X_test_model_frame = X_test_raw
        if process_pls_step is not None:
            process_pls = process_pls_step[1]
            process_pls.fit(X_train_raw, y_train)
            X_train_model_frame = process_pls.transform(X_train_raw)
            X_test_model_frame = process_pls.transform(X_test_raw)
            pipeline_steps.append(process_pls_step)

        X_fit_raw = X_train_model_frame.iloc[fit_idx].reset_index(drop=True)
        X_early_valid_raw = X_train_model_frame.iloc[early_idx].reset_index(drop=True)
        y_fit = y_train[fit_idx]
        y_early_valid = y_train[early_idx]
        balance_info = _build_target_balance_info(
            y_fit,
            enabled=target_balance_enabled,
            n_bins=balance_n_bins,
            max_weight=balance_max_weight,
            random_state=random_state,
        )

        if _should_pass_target_name(model_name) and "target_name" not in model_params and hasattr(y, "name"):
            model_params["target_name"] = getattr(y, "name", None)

        model = self._get_model(model_name, random_state=int(random_state), **model_params)

        try:
            if hasattr(model, "validation_data") and len(early_idx) >= 4:
                setattr(model, "validation_data", (X_early_valid_raw, y_early_valid))
        except Exception:
            pass

        start_time = time.time()
        fit_data = _prepare_balanced_fit_data(
            model,
            X_fit_raw,
            y_fit,
            balance_info,
            random_state=random_state,
        )
        fit_kwargs = {}
        if fit_data["sample_weight"] is not None:
            fit_kwargs["sample_weight"] = fit_data["sample_weight"]
        model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)
        train_time = time.time() - start_time

        pipeline = Pipeline(steps=pipeline_steps + [("model", model)])
        y_pred_test = np.asarray(pipeline.predict(X_test_raw)).ravel()
        y_pred_train = np.asarray(pipeline.predict(X_train_raw)).ravel()

        def _safe_metrics(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            metric_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if metric_mask.sum() == 0:
                return float("nan"), float("nan"), float("nan")
            return (
                float(r2_score(y_true[metric_mask], y_pred[metric_mask])),
                float(np.sqrt(mean_squared_error(y_true[metric_mask], y_pred[metric_mask]))),
                float(mean_absolute_error(y_true[metric_mask], y_pred[metric_mask])),
            )

        r2_val, rmse_val, mae_val = _safe_metrics(y_test, y_pred_test)

        X_train_view = X_train_model_frame
        X_test_view = X_test_model_frame
        if hasattr(model, "_transform"):
            try:
                X_train_arr, _ = model._transform(X_train_model_frame)
                X_test_arr, _ = model._transform(X_test_model_frame)
                columns = getattr(getattr(model, "_prep_", None), "feature_names", None)
                if not columns:
                    columns = [f"feat_{i}" for i in range(X_train_arr.shape[1])]
                X_train_view = pd.DataFrame(X_train_arr, columns=columns)
                X_test_view = pd.DataFrame(X_test_arr, columns=columns)
            except Exception:
                X_train_view = X_train_raw
                X_test_view = X_test_raw
        elif hasattr(model, "_prepare_features_and_mask"):
            try:
                X_train_arr, _ = model._prepare_features_and_mask(X_train_model_frame, fit=False)
                X_test_arr, _ = model._prepare_features_and_mask(X_test_model_frame, fit=False)
                if isinstance(X_train_model_frame, pd.DataFrame):
                    columns = list(X_train_model_frame.columns)
                else:
                    columns = [f"feat_{i}" for i in range(X_train_arr.shape[1])]
                X_train_view = pd.DataFrame(X_train_arr, columns=columns)
                X_test_view = pd.DataFrame(X_test_arr, columns=columns)
            except Exception:
                X_train_view = X_train_raw
                X_test_view = X_test_raw
        elif hasattr(model, "transform_tabular"):
            try:
                X_train_arr = model.transform_tabular(X_train_model_frame)
                X_test_arr = model.transform_tabular(X_test_model_frame)
                columns = list(getattr(model, "numeric_feature_names_", None) or [f"feat_{i}" for i in range(X_train_arr.shape[1])])
                X_train_view = pd.DataFrame(X_train_arr, columns=columns)
                X_test_view = pd.DataFrame(X_test_arr, columns=columns)
            except Exception:
                X_train_view = X_train_raw
                X_test_view = X_test_raw

        training_history = extract_history_from_fitted_model(
            model_name=model_name,
            model=model,
            X_train_scaled=None,
            y_train=np.asarray(y_train),
            X_test_scaled=None,
            y_test=np.asarray(y_test),
        )
        balance_result = _finalize_target_balance_result(
            balance_info,
            actual_method=fit_data["method"],
            fit_sample_count=len(fit_data["y"]),
            early_stopping_validation_count=len(y_early_valid),
        )
        test_bin_metrics = _compute_regression_bin_metrics(
            y_test,
            y_pred_test,
            balance_info.get("bin_edges", []),
        )

        return {
            "model": model,
            "pipeline": pipeline,
            "scaler": None,
            "imputer": None,
            "X_train": X_train_view,
            "X_test": X_test_view,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "feature_names": list(X_train_model_frame.columns),
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred_test,
            "y_pred_test": y_pred_test,
            "y_pred_train": y_pred_train,
            "r2": r2_val,
            "rmse": rmse_val,
            "mae": mae_val,
            "train_time": float(train_time),
            "split_strategy": split_strategy,
            "n_bins": int(n_bins),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "training_history": training_history,
            "training_history_df": history_to_frame(training_history) if training_history else pd.DataFrame(),
            "target_balance": balance_result,
            "test_bin_metrics": test_bin_metrics,
        }


    def _prepare_binary_classification_inputs(self, X, y, model_name, groups=None):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            X_df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])

        try:
            for c in X_df.columns:
                if X_df[c].dtype == "object":
                    X_df[c] = X_df[c].replace(
                        {
                            "True": 1,
                            "true": 1,
                            "TRUE": 1,
                            "False": 0,
                            "false": 0,
                            "FALSE": 0,
                        }
                    )
            X_df = X_df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            for c in X_df.columns:
                if X_df[c].dtype == "object":
                    X_df[c] = X_df[c].replace(
                        {
                            "True": 1,
                            "true": 1,
                            "TRUE": 1,
                            "False": 0,
                            "false": 0,
                            "FALSE": 0,
                        }
                    )
                X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

        X_df = _sanitize_feature_frame(X_df.replace([np.inf, -np.inf], np.nan), model_name)
        dropped_all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
        if dropped_all_nan_cols:
            X_df = X_df.drop(columns=dropped_all_nan_cols)

        y_series = pd.Series(np.asarray(y).ravel())
        valid_target_mask = ~y_series.isna()
        X_df = X_df.loc[valid_target_mask].reset_index(drop=True)
        y_series = y_series.loc[valid_target_mask].reset_index(drop=True)
        if groups is not None:
            groups = np.asarray(groups)[valid_target_mask.to_numpy()]

        if X_df.shape[1] == 0:
            raise ValueError("当前输入特征在数值化后为空，无法执行二分类训练。")

        y_series, y_encoded, class_labels, _, positive_label = _encode_binary_target(y_series)
        X_arr = X_df.to_numpy(dtype=float, copy=False)

        return {
            "X_df": X_df,
            "X_arr": X_arr,
            "y_raw": y_series.to_numpy(),
            "y_encoded": y_encoded,
            "class_labels": list(class_labels),
            "positive_label": positive_label,
            "groups": groups,
            "feature_names": list(X_df.columns),
        }

    def _resolve_classification_split(self, X, y, test_size, random_state, split_strategy="random", groups=None):
        n = len(y)
        idx = np.arange(n)
        split_strategy = str(split_strategy or "random").lower()

        if split_strategy == "group" and groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups))
            return train_idx, test_idx

        class_counts = np.bincount(np.asarray(y, dtype=int))
        stratify_labels = y if class_counts.size >= 2 and np.min(class_counts) >= 2 else None

        if split_strategy == "stratified" and stratify_labels is not None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(np.zeros((n, 1)), y))
            return train_idx, test_idx

        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        return np.asarray(train_idx), np.asarray(test_idx)

    def _train_classification_model(
        self,
        X,
        y,
        model_name,
        test_size=0.2,
        random_state=42,
        split_strategy="random",
        n_bins=10,
        groups=None,
        drop_missing_rows=True,
        **params,
    ):
        prepared = self._prepare_binary_classification_inputs(X, y, model_name, groups=groups)
        X_df = prepared["X_df"]
        y_raw = prepared["y_raw"]
        y_encoded = prepared["y_encoded"]
        class_labels = prepared["class_labels"]
        positive_label = prepared["positive_label"]
        groups = prepared["groups"]
        feature_names = prepared["feature_names"]

        if len(np.unique(y_encoded)) != 2:
            raise ValueError("划分前清洗数据后，目标列不再是有效的二分类标签。")

        train_idx, test_idx = self._resolve_classification_split(
            X_df.to_numpy(dtype=float, copy=False),
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            split_strategy=split_strategy,
            groups=groups,
        )

        X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]
        y_train_raw = y_raw[train_idx]
        y_test_raw = y_raw[test_idx]

        train_n_jobs = params.pop("train_n_jobs", -1)
        model_params = params.copy()
        requested_use_gpu = bool(model_params.get("use_gpu", self.use_gpu))
        parallel_alias = {
            "随机森林分类": "随机森林",
            "XGBoost分类": "XGBoost",
            "LightGBM分类": "LightGBM",
            "CatBoost分类": "CatBoost",
        }.get(model_name, model_name)
        _apply_parallel_settings(parallel_alias, model_params, train_n_jobs, use_gpu=requested_use_gpu)

        base_model = self._get_model(model_name, random_state=int(random_state), **model_params)

        missing_tolerant_models = {"XGBoost分类", "LightGBM分类", "CatBoost分类"}
        use_imputer = model_name not in missing_tolerant_models
        use_scaler = model_name == "逻辑回归分类"
        imputer = SimpleImputer(strategy="median") if use_imputer else None
        scaler = StandardScaler() if use_scaler else None

        X_train_proc = X_train_raw.to_numpy(dtype=float, copy=True)
        X_test_proc = X_test_raw.to_numpy(dtype=float, copy=True)
        if imputer is not None:
            missing_count_train = int(np.isnan(X_train_proc).sum())
            missing_count_test = int(np.isnan(X_test_proc).sum())
            total_count_train = int(X_train_proc.size) if X_train_proc.size else 0
            total_count_test = int(X_test_proc.size) if X_test_proc.size else 0
            imputer_name = type(imputer).__name__
            print(
                f"[DEBUG] Starting imputer={imputer_name} | "
                f"train_missing={missing_count_train}/{total_count_train} | "
                f"test_missing={missing_count_test}/{total_count_test}"
            )
            X_train_proc = imputer.fit_transform(X_train_proc)
            X_test_proc = imputer.transform(X_test_proc)
        if scaler is not None:
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_test_proc = scaler.transform(X_test_proc)

        start_time = time.time()
        if model_name == "XGBoost分类":
            _safe_xgb_fit(
                base_model,
                X_train_proc,
                y_train,
                {"eval_set": [(X_test_proc, y_test)], "verbose": False},
            )
        elif model_name == "LightGBM分类":
            try:
                base_model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)])
            except TypeError:
                base_model.fit(X_train_proc, y_train)
        elif model_name == "CatBoost分类":
            try:
                base_model.fit(X_train_proc, y_train, eval_set=(X_test_proc, y_test), verbose=False)
            except TypeError:
                base_model.fit(X_train_proc, y_train)
        else:
            base_model.fit(X_train_proc, y_train)
        train_time = time.time() - start_time

        steps = []
        if imputer is not None:
            steps.append(("imputer", imputer))
        steps.append(("inf_cleaner", InfCleaner()))
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", base_model))
        pipeline = Pipeline(steps)

        y_pred_test_encoded = np.asarray(pipeline.predict(X_test_raw)).ravel().astype(int)
        y_pred_train_encoded = np.asarray(pipeline.predict(X_train_raw)).ravel().astype(int)
        y_pred_proba_test = _safe_predict_proba(pipeline, X_test_raw)
        y_pred_proba_train = _safe_predict_proba(pipeline, X_train_raw)

        y_pred_test = np.asarray([class_labels[int(v)] for v in y_pred_test_encoded], dtype=object)
        y_pred_train = np.asarray([class_labels[int(v)] for v in y_pred_train_encoded], dtype=object)

        metrics = _compute_binary_classification_metrics(y_test, y_pred_test_encoded, y_pred_proba_test)

        X_train_view = pd.DataFrame(X_train_proc, columns=feature_names)
        X_test_view = pd.DataFrame(X_test_proc, columns=feature_names)

        return {
            "task_kind": "classification",
            "model": base_model,
            "pipeline": pipeline,
            "scaler": scaler,
            "imputer": imputer,
            "X_train": X_train_view,
            "X_test": X_test_view,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "y_train": y_train_raw,
            "y_test": y_test_raw,
            "y_train_encoded": y_train,
            "y_test_encoded": y_test,
            "y_pred": y_pred_test,
            "y_pred_test": y_pred_test,
            "y_pred_train": y_pred_train,
            "y_pred_test_encoded": y_pred_test_encoded,
            "y_pred_train_encoded": y_pred_train_encoded,
            "y_pred_proba_test": y_pred_proba_test,
            "y_pred_proba_train": y_pred_proba_train,
            "class_labels": class_labels,
            "positive_label": positive_label,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
            "r2": None,
            "rmse": None,
            "mae": None,
            "train_time": float(train_time),
            "split_strategy": split_strategy,
            "n_bins": int(n_bins),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "training_history": None,
            "training_history_df": pd.DataFrame(),
        }

    def _cross_validate_classification_model(
        self,
        X,
        y,
        model_name,
        cv_strategy="repeated_kfold",
        n_splits=5,
        n_repeats=5,
        random_state=42,
        groups=None,
        n_bins=10,
        drop_missing_rows=True,
        **params,
    ):
        prepared = self._prepare_binary_classification_inputs(X, y, model_name, groups=groups)
        X_df = prepared["X_df"]
        y_raw = prepared["y_raw"]
        y_encoded = prepared["y_encoded"]
        class_labels = prepared["class_labels"]
        positive_label = prepared["positive_label"]
        groups = prepared["groups"]

        n = len(y_encoded)
        if n < max(4, n_splits):
            raise ValueError("有效样本过少，无法执行二分类交叉验证。")

        cv_strategy = str(cv_strategy or "repeated_kfold").lower()
        n_splits = int(max(2, n_splits))
        n_repeats = int(max(1, n_repeats))

        if cv_strategy in {"group_kfold", "group", "分组"}:
            if groups is None:
                raise ValueError("group_kfold 需要 groups。")
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(np.zeros((n, 1)), y_encoded, groups)
        elif cv_strategy in {"stratified_kfold", "stratified", "分层"}:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iter = splitter.split(np.zeros((n, 1)), y_encoded)
        else:
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            split_iter = splitter.split(np.zeros((n, 1)), y_encoded)

        train_n_jobs = params.pop("train_n_jobs", -1)
        base_params = params.copy()
        requested_use_gpu = bool(base_params.get("use_gpu", self.use_gpu))
        parallel_alias = {
            "随机森林分类": "随机森林",
            "XGBoost分类": "XGBoost",
            "LightGBM分类": "LightGBM",
            "CatBoost分类": "CatBoost",
        }.get(model_name, model_name)

        missing_tolerant_models = {"XGBoost分类", "LightGBM分类", "CatBoost分类"}
        use_imputer = model_name not in missing_tolerant_models
        use_scaler = model_name == "逻辑回归分类"

        oof_pred_encoded = np.full(n, -1, dtype=int)
        oof_proba = np.full(n, np.nan, dtype=float)
        fold_accuracy = []
        fold_precision = []
        fold_recall = []
        fold_f1 = []
        fold_roc_auc = []

        for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
            fold_params = base_params.copy()
            _apply_parallel_settings(parallel_alias, fold_params, train_n_jobs, use_gpu=requested_use_gpu)
            model = self._get_model(model_name, random_state=int(random_state + fold_i), **fold_params)

            X_train = X_df.iloc[tr_idx].to_numpy(dtype=float, copy=True)
            X_valid = X_df.iloc[va_idx].to_numpy(dtype=float, copy=True)
            y_train = y_encoded[tr_idx]
            y_valid = y_encoded[va_idx]

            imputer = SimpleImputer(strategy="median") if use_imputer else None
            scaler = StandardScaler() if use_scaler else None
            if imputer is not None:
                X_train = imputer.fit_transform(X_train)
                X_valid = imputer.transform(X_valid)
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_valid = scaler.transform(X_valid)

            if model_name == "XGBoost分类":
                _safe_xgb_fit(model, X_train, y_train, {"eval_set": [(X_valid, y_valid)], "verbose": False})
            elif model_name == "LightGBM分类":
                try:
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                except TypeError:
                    model.fit(X_train, y_train)
            elif model_name == "CatBoost分类":
                try:
                    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_pred = np.asarray(model.predict(X_valid)).ravel().astype(int)
            y_proba = _safe_predict_proba(model, X_valid)
            fold_metrics = _compute_binary_classification_metrics(y_valid, y_pred, y_proba)

            oof_pred_encoded[va_idx] = y_pred
            if y_proba is not None:
                oof_proba[va_idx] = y_proba

            fold_accuracy.append(fold_metrics["accuracy"])
            fold_precision.append(fold_metrics["precision"])
            fold_recall.append(fold_metrics["recall"])
            fold_f1.append(fold_metrics["f1"])
            fold_roc_auc.append(fold_metrics["roc_auc"])

        valid_mask = oof_pred_encoded >= 0
        oof_metrics = _compute_binary_classification_metrics(
            y_encoded[valid_mask],
            oof_pred_encoded[valid_mask],
            oof_proba[valid_mask] if np.isfinite(oof_proba[valid_mask]).all() else None,
        )
        oof_pred = np.asarray([class_labels[int(v)] for v in oof_pred_encoded], dtype=object)

        return {
            "task_kind": "classification",
            "cv_strategy": cv_strategy,
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "class_labels": class_labels,
            "positive_label": positive_label,
            "fold_accuracy": fold_accuracy,
            "fold_precision": fold_precision,
            "fold_recall": fold_recall,
            "fold_f1": fold_f1,
            "fold_roc_auc": fold_roc_auc,
            "cv_accuracy_mean": float(np.nanmean(fold_accuracy)) if fold_accuracy else float("nan"),
            "cv_accuracy_std": float(np.nanstd(fold_accuracy, ddof=1)) if len(fold_accuracy) > 1 else 0.0,
            "cv_f1_mean": float(np.nanmean(fold_f1)) if fold_f1 else float("nan"),
            "cv_f1_std": float(np.nanstd(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
            "cv_roc_auc_mean": float(np.nanmean(fold_roc_auc)) if fold_roc_auc else float("nan"),
            "cv_roc_auc_std": float(np.nanstd(fold_roc_auc, ddof=1)) if len(fold_roc_auc) > 1 else 0.0,
            "oof_pred": oof_pred,
            "oof_pred_encoded": oof_pred_encoded,
            "oof_true": y_raw,
            "oof_true_encoded": y_encoded,
            "oof_proba": oof_proba,
            "oof_accuracy": oof_metrics["accuracy"],
            "oof_balanced_accuracy": oof_metrics["balanced_accuracy"],
            "oof_precision": oof_metrics["precision"],
            "oof_recall": oof_metrics["recall"],
            "oof_f1": oof_metrics["f1"],
            "oof_roc_auc": oof_metrics["roc_auc"],
            "oof_log_loss": oof_metrics["log_loss"],
        }

    def build_regression_cv_pipeline(
        self,
        model_name,
        feature_columns,
        *,
        random_state=42,
        process_pls_config=None,
        use_process_pls=False,
        **params,
    ):
        if str(model_name) in RAW_FRAME_MODEL_NAMES:
            raise ValueError("当前模型不支持通用回归优化 pipeline，请在训练页使用专用训练流程")
        if _is_classification_model(model_name):
            raise ValueError("可信超参数优化当前仅支持回归模型")

        columns = list(feature_columns or [])
        process_pls_step = _make_process_pls_step(
            process_pls_config,
            bool(use_process_pls),
            columns,
        )
        model = self._get_model(
            model_name,
            random_state=int(random_state),
            **dict(params),
        )
        steps = []
        if process_pls_step is not None:
            steps.append(process_pls_step)
        steps.extend([
            ("inf_cleaner", InfCleaner()),
            ("imputer", SimpleImputer(strategy="median")),
            ("nan_col_dropper", AllNaNColumnDropper()),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        return Pipeline(steps=steps)

    def train_model(
        self,
        X,
        y,
        model_name,
        test_size=0.2,
        random_state=42,
        split_strategy='random',
        n_bins=10,
        groups=None,
        drop_missing_rows=True,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        process_pls_config=None,
        use_process_pls=False,
        **params
    ):
        """训练单个模型（支持随机/分层/分组划分）"""
        # Epoxy PINN 专用分支：允许原始字符串列，由模型内部解析（用于 Tg / 力学等物理约束）
        if str(model_name) == "Epoxy PINN (Physics-Informed)":
            return self._train_pinn_special(
                X=X,
                y=y,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                split_strategy=split_strategy,
                n_bins=n_bins,
                groups=groups,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                **params
            )

        if _is_classification_model(model_name):
            return self._train_classification_model(
                X=X,
                y=y,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                split_strategy=split_strategy,
                n_bins=n_bins,
                groups=groups,
                drop_missing_rows=drop_missing_rows,
                **params,
            )

        if str(model_name) in RAW_FRAME_MODEL_NAMES:
            return self._train_raw_frame_model(
                X=X,
                y=y,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                split_strategy=split_strategy,
                n_bins=n_bins,
                groups=groups,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                process_pls_config=process_pls_config,
                use_process_pls=use_process_pls,
                **params,
            )

        if str(model_name) in GRAPH_MODEL_NAMES:
            return self._train_graph_model(
                X=X,
                y=y,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                split_strategy=split_strategy,
                n_bins=n_bins,
                groups=groups,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                **params,
            )


        # 1) 输入统一为 numpy，并做 y 清洗
        feature_names = None
        X_df = None
        if isinstance(X, pd.DataFrame):
            # 保留列名，但对训练输入做一次“数值化 + 清洗”，避免出现全 NaN/非数值列导致后续形状不一致
            X_df = X.copy()

            # 将非数值列强制转为 NaN（例如误选了 smiles 文本列 / object 列）
            try:
                # 先处理字符串形式的布尔值
                for c in X_df.columns:
                    if X_df[c].dtype == 'object':
                        # 尝试转换字符串布尔值
                        X_df[c] = X_df[c].replace({'True': 1, 'true': 1, 'TRUE': 1,
                                                     'False': 0, 'false': 0, 'FALSE': 0})

                X_df = X_df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                # 某些极端 object 列（如嵌套结构）转换可能失败，退化为逐列转换
                for c in X_df.columns:
                    if X_df[c].dtype == 'object':
                        X_df[c] = X_df[c].replace({'True': 1, 'true': 1, 'TRUE': 1,
                                                     'False': 0, 'false': 0, 'FALSE': 0})
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

            # 将 ±inf 视为缺失
            X_df = X_df.replace([np.inf, -np.inf], np.nan)

            # 丢弃”整列都是 NaN”的特征列（常见于大量缺失/拼接后空列）
            dropped_all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
            if dropped_all_nan_cols:
                X_df = X_df.drop(columns=dropped_all_nan_cols)

            feature_names = X_df.columns.tolist()
            X_arr = X_df.values
        else:
            X_arr = np.asarray(X)
            feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=feature_names)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        # 目标强制转数值
        y_arr = pd.to_numeric(pd.Series(y_arr), errors='coerce').values
        mask = (~np.isnan(y_arr)) & (~np.isinf(y_arr))

        if np.sum(~mask) > 0:
            X_arr = X_arr[mask]
            X_df = X_df.loc[mask].reset_index(drop=True)

        # 过滤无效 y 之后，再次丢弃“整列都是 NaN”的特征列（避免训练/回传 DataFrame 形状不一致）
        if X_df is not None:
            X_df = _sanitize_feature_frame(X_df, model_name)
            dropped_all_nan_cols_2 = X_df.columns[X_df.isna().all()].tolist()
            if dropped_all_nan_cols_2:
                X_df = X_df.drop(columns=dropped_all_nan_cols_2)
                feature_names = X_df.columns.tolist()
            X_arr = X_df.values
            y_arr = y_arr[mask]
            if groups is not None:
                groups = np.asarray(groups)[mask]

        if len(y_arr) == 0:
            raise ValueError("所有样本的目标变量均无效（NaN/Inf），无法训练模型，请检查数据")

        print(f"✓ 有效样本数: {len(y_arr)} 行（已删除目标列缺失值）")

        # 确保 X_arr 是数值类型（所有模型都需要）
        if X_df is not None:
            try:
                X_arr = X_arr.astype(float)
            except (ValueError, TypeError) as e:
                raise ValueError(f"特征数据包含非数值类型，无法训练。请检查特征选择是否包含了文本列（如 SMILES）。错误: {e}")
        raw_feature_names = list(X_df.columns)
        process_pls_step = _make_process_pls_step(
            process_pls_config,
            use_process_pls,
            raw_feature_names,
        )

        # 缺失值策略只作用于目标列：无效目标样本已在上方从本次训练视图中排除，
        # 原始 X 数据不因输入特征缺失而丢行。不能原生接收 NaN 的模型在后续使用插补。

        # 2) 划分索引
        train_idx, test_idx = self._resolve_split(
            X_arr, y_arr, test_size=test_size, random_state=random_state,
            split_strategy=split_strategy, n_bins=n_bins, groups=groups
        )

        if process_pls_step is not None:
            X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
            X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
        else:
            X_train_raw = X_arr[train_idx]
            X_test_raw = X_arr[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        early_stop_models = {"XGBoost", "LightGBM", "CatBoost"}
        use_internal_validation = (
            model_name in early_stop_models
            and len(y_train) >= 20
        )
        if use_internal_validation:
            fit_idx, early_idx = train_test_split(
                np.arange(len(y_train), dtype=int),
                test_size=0.15,
                random_state=random_state,
            )
            if len(early_idx) < 4:
                fit_idx = np.arange(len(y_train), dtype=int)
                early_idx = np.asarray([], dtype=int)
        else:
            fit_idx = np.arange(len(y_train), dtype=int)
            early_idx = np.asarray([], dtype=int)

        y_fit = y_train[fit_idx]
        y_early_valid = y_train[early_idx]
        balance_info = _build_target_balance_info(
            y_fit,
            enabled=target_balance_enabled,
            n_bins=balance_n_bins,
            max_weight=balance_max_weight,
            random_state=random_state,
        )


        # 3) 模型参数注入 random_state（对不支持的模型跳过）
        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]
        # 并行训练设置（UI：训练并行核数）
        train_n_jobs = params.pop('train_n_jobs', -1)
        train_n_jobs = _normalize_train_n_jobs(train_n_jobs)
        model_params = params.copy()
        progress_callback = model_params.get("epoch_callback")
        xgb_fit_params = {}
        if model_name == "XGBoost":
            # 提取XGBoost的fit参数，如果没有设置则使用默认值
            xgb_fit_params["early_stopping_rounds"] = model_params.pop("early_stopping_rounds", 500)  # 默认500轮早停
            xgb_fit_params["eval_metric"] = model_params.pop("eval_metric", None)
            xgb_fit_params["verbose_eval"] = model_params.pop("verbose_eval", None)
        _apply_parallel_settings(model_name, model_params, train_n_jobs, use_gpu=self.use_gpu)
        _apply_gpu_settings_for_neural_networks(model_name, model_params, use_gpu=self.use_gpu)
        if model_name not in NO_SEED_MODELS:
            model_params.setdefault('random_state', random_state)

        model_handles_preprocessing = model_name == "Transformer + BNN"
        if model_handles_preprocessing:
            model_params.setdefault("missing_value_strategy", "mask_zero")
            model_params.setdefault("loss_name", "mse")

        base_model = self._get_model(model_name, **model_params)

        def emit_transformer_postprocessing(message, progress_ratio, **extra):
            if model_name != "Transformer + BNN" or not callable(progress_callback):
                return
            payload = {
                "phase": "postprocessing",
                "message": str(message),
                "progress_ratio": float(progress_ratio),
            }
            payload.update(extra)
            try:
                progress_callback(payload)
            except Exception:
                pass

        # 4) 预处理：按模型类型选择（树模型可跳过标准化，部分模型原生支持缺失值）
        missing_tolerant_models = {"XGBoost", "LightGBM", "CatBoost"}
        scale_free_models = {"XGBoost", "LightGBM", "CatBoost", "随机森林", "Extra Trees", "梯度提升树", "决策树"}

        use_scaler = model_name not in scale_free_models and not model_handles_preprocessing
        bnn_missing_strategy = str(model_params.get("missing_value_strategy", "median") or "median").strip().lower()
        bnn_missing_max_iter = model_params.get("missing_imputer_max_iter", 15)
        bnn_missing_n_imputations = model_params.get("missing_n_imputations", 5)

        if model_handles_preprocessing:
            # Transformer + BNN 内部必须看到原始 NaN 和 missing mask。
            # 外层 SimpleImputer/StandardScaler 会破坏 mask_zero 的语义，
            # 还会让模型内部再次缩放，造成重复预处理。
            imputer = None
            scaler = None
            print("🧩 Transformer + BNN: 保留原始缺失值，由模型内部 mask_zero/缩放流程处理")
        elif model_name == "Bayesian Neural Network (BNN)":
            imputer = _build_missing_value_imputer(
                strategy=bnn_missing_strategy,
                random_state=random_state,
                max_iter=bnn_missing_max_iter,
                n_imputations=bnn_missing_n_imputations,
            )
            use_imputer = True
            print(
                f"🧩 BNN missing-value strategy: {bnn_missing_strategy} "
                f"(max_iter={int(bnn_missing_max_iter)}, n_imputations={int(bnn_missing_n_imputations)})"
            )
        elif model_name in missing_tolerant_models:
            imputer = None
        else:
            imputer = SimpleImputer(strategy='median')

        # ANN 支持自定义 scaler 类型和目标归一化
        scaler_type = model_params.pop('scaler_type', 'standard') if model_name == "人工神经网络" else 'standard'
        normalize_target = model_params.pop('normalize_target', False) if model_name == "人工神经网络" else False

        if use_scaler:
            if scaler_type == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
        else:
            scaler = None

        y_scaler = StandardScaler() if normalize_target else None

        if process_pls_step is not None:
            process_pls = process_pls_step[1]
            process_pls.fit(X_train_raw, y_train)
            X_train_pls = process_pls.transform(X_train_raw)
            X_test_pls = process_pls.transform(X_test_raw)
            feature_names = process_pls.get_feature_names_out().tolist()
            X_train_proc = X_train_pls.to_numpy(dtype=float, copy=False)
            X_test_proc = X_test_pls.to_numpy(dtype=float, copy=False)
        else:
            X_train_proc = X_train_raw
            X_test_proc = X_test_raw

        # [修复] 初始化特征掩码，用于跟踪哪些特征被保留
        # 这个掩码将被保存，以便在预测时应用相同的特征选择
        original_n_features = X_train_proc.shape[1]
        feature_mask = np.ones(original_n_features, dtype=bool)

        print(f"[DEBUG] 初始特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")

        if imputer is not None:
            X_train_proc = imputer.fit_transform(X_train_proc)
            X_test_proc = imputer.transform(X_test_proc)
            print(f"[DEBUG] Imputer后特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")

        # [修复] 检查并删除全NaN列（确保训练集和测试集特征数一致）
        # 这可能发生在数据划分后，某些列在训练集或测试集中全是NaN
        if X_train_proc.shape[1] > 0 and not model_handles_preprocessing:
            # 检查训练集中的全NaN列
            train_all_nan_mask = np.isnan(X_train_proc).all(axis=0)
            if train_all_nan_mask.any():
                nan_col_indices = np.where(train_all_nan_mask)[0]
                print(f"⚠️ 训练集中发现 {len(nan_col_indices)} 个全NaN列，将从训练集和测试集中同时删除")
                # 从训练集和测试集中删除这些列
                keep_mask = ~train_all_nan_mask
                X_train_proc = X_train_proc[:, keep_mask]
                X_test_proc = X_test_proc[:, keep_mask]

                # [修复] 更新全局特征掩码
                feature_mask[feature_mask] = keep_mask

                # 更新特征名称
                if feature_names:
                    removed_features = [feature_names[i] for i in nan_col_indices]
                    feature_names = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]
                    print(f"   删除的特征: {removed_features}")
            print(f"[DEBUG] 删除全NaN列后特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")

        # 清理无穷大值（防止标准化或模型计算产生非法结果）；输入缺失行仍然保留。
        X_train_proc = np.where(np.isinf(X_train_proc), np.nan, X_train_proc)
        X_test_proc = np.where(np.isinf(X_test_proc), np.nan, X_test_proc)

        if imputer is not None and np.isnan(X_train_proc).any():
            X_train_proc = imputer.fit_transform(X_train_proc)
            X_test_proc = imputer.transform(X_test_proc)
        print(f"[DEBUG] 清理inf并完成缺失值处理后特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")

        if scaler is not None:
            # [修复] 标准化前再次检查全NaN列（可能在inf清理后产生）
            if X_train_proc.shape[1] > 0:
                train_all_nan_mask = np.isnan(X_train_proc).all(axis=0)
                if train_all_nan_mask.any():
                    nan_col_indices = np.where(train_all_nan_mask)[0]
                    print(f"⚠️ 标准化前发现 {len(nan_col_indices)} 个全NaN列，将从训练集和测试集中同时删除")
                    keep_mask = ~train_all_nan_mask
                    X_train_proc = X_train_proc[:, keep_mask]
                    X_test_proc = X_test_proc[:, keep_mask]

                    # [修复] 更新全局特征掩码
                    feature_mask[feature_mask] = keep_mask

                    if feature_names:
                        removed_features = [feature_names[i] for i in nan_col_indices]
                        feature_names = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]
                        print(f"   删除的特征: {removed_features}")

            # [修复] 标准化前检查方差为0的特征（会导致除以0产生无穷大）
            train_std = np.std(X_train_proc, axis=0)
            zero_var_mask = train_std == 0

            if zero_var_mask.any():
                zero_var_indices = np.where(zero_var_mask)[0]
                print(f"[WARNING] 发现 {len(zero_var_indices)} 个方差为0的特征列（所有值相同），将移除这些列以避免标准化产生无穷大值")
                print(f"[WARNING] 移除的特征索引: {zero_var_indices.tolist()}")

                # 移除方差为0的列
                keep_mask = ~zero_var_mask
                X_train_proc = X_train_proc[:, keep_mask]
                X_test_proc = X_test_proc[:, keep_mask]

                # [修复] 更新全局特征掩码
                feature_mask[feature_mask] = keep_mask

                # 更新特征名称
                if feature_names:
                    removed_features = [feature_names[i] for i in zero_var_indices]
                    feature_names = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]
                    print(f"[WARNING] 移除的特征名称: {removed_features}")

            print(f"[DEBUG] 标准化前特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_test_proc = scaler.transform(X_test_proc)
            print(f"[DEBUG] 标准化后特征数: 训练集={X_train_proc.shape[1]}, 测试集={X_test_proc.shape[1]}")

            # [修复] 标准化后可能产生无穷大值（方差为0的特征会导致除以0）
            # 将无穷大值替换为 NaN，然后用中位数填充
            if np.isinf(X_train_proc).any() or np.isinf(X_test_proc).any():
                X_train_proc = np.where(np.isinf(X_train_proc), np.nan, X_train_proc)
                X_test_proc = np.where(np.isinf(X_test_proc), np.nan, X_test_proc)

                # 用中位数填充新产生的 NaN
                if np.isnan(X_train_proc).any():
                    post_imputer = SimpleImputer(strategy='median')
                    X_train_proc = post_imputer.fit_transform(X_train_proc)
                    X_test_proc = post_imputer.transform(X_test_proc)
                    # 如果原来没有 imputer，现在需要一个
                    if imputer is None:
                        imputer = post_imputer

        # 目标变量归一化（ANN专用）
        y_train_fit = y_train
        if y_scaler is not None:
            y_train_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            # [修复] 检查目标变量归一化后是否产生无穷大值
            if np.isinf(y_train_fit).any():
                raise ValueError("目标变量归一化后产生无穷大值，可能是目标变量方差为0或存在极端值，请检查数据")

        X_fit_proc = X_train_proc[fit_idx]
        X_early_valid_proc = (
            X_train_proc[early_idx] if len(early_idx) >= 4 else None
        )
        y_fit_model = (
            y_train_fit[fit_idx]
            if model_name == "人工神经网络"
            else y_fit
        )
        y_early_valid_model = (
            y_train_fit[early_idx]
            if model_name == "人工神经网络" and len(early_idx) >= 4
            else y_early_valid
        )
        fit_data = _prepare_balanced_fit_data(
            base_model,
            X_fit_proc,
            y_fit_model,
            balance_info,
            random_state=random_state,
        )
        X_model_fit = fit_data["X"]
        y_model_fit = fit_data["y"]
        sample_weight = fit_data["sample_weight"]
        actual_balance_method = fit_data["method"]

        # TensorFlow Sequential: 预先转为 float32 以降低内存峰值
        if model_name == "TensorFlow Sequential":
            X_train_proc = np.asarray(X_train_proc, dtype=np.float32)
            X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32).ravel()
            y_test = np.asarray(y_test, dtype=np.float32).ravel()

            # [修复] 检查转换为 float32 后是否产生无穷大值
            if np.isinf(X_train_proc).any() or np.isinf(X_test_proc).any():
                raise ValueError("数据转换为 float32 后产生无穷大值，数值超出 float32 范围，请检查数据或使用其他模型")

            mem_avail = _get_mem_available_bytes()
            if mem_avail is not None:
                est_bytes = (
                    getattr(X_train_proc, "nbytes", 0)
                    + getattr(X_test_proc, "nbytes", 0)
                    + getattr(y_train, "nbytes", 0)
                    + getattr(y_test, "nbytes", 0)
                )
                if est_bytes > mem_avail * 0.8:
                    raise MemoryError(
                        "TensorFlow Sequential 训练数据过大，可能触发系统 OOM。"
                        "请减少特征/样本或降低 batch_size 后重试。"
                    )

        # 5) 训练模型（对可提供迭代日志的模型，尽量注入 eval_set）
        # [修复] 训练前最终检查：确保没有无穷大值或超出 float32 范围的值
        max_float32 = np.finfo(np.float32).max
        # 使用更保守的阈值：float32 最大值的 1%
        safe_threshold = max_float32 * 0.01

        print(f"[DEBUG] 训练前数据检查:")
        print(f"  X_train_proc shape: {X_train_proc.shape}, dtype: {X_train_proc.dtype}")
        print(f"  X_train_proc 统计: min={np.min(X_train_proc):.4e}, max={np.max(X_train_proc):.4e}, mean={np.mean(X_train_proc):.4e}")
        print(f"  包含 inf: {np.isinf(X_train_proc).any()}, 包含 nan: {np.isnan(X_train_proc).any()}")
        print(f"  超出 float32 范围: {(np.abs(X_train_proc) > max_float32).any()}")
        print(f"  超出安全阈值 ({safe_threshold:.2e}): {(np.abs(X_train_proc) > safe_threshold).any()}")

        # 检查 X_train_proc 和 X_test_proc
        if np.isinf(X_train_proc).any():
            inf_count = np.isinf(X_train_proc).sum()
            inf_cols = np.where(np.isinf(X_train_proc).any(axis=0))[0]
            raise ValueError(f"训练数据包含 {inf_count} 个无穷大值，位于列索引: {inf_cols.tolist()}")

        if np.isinf(X_test_proc).any():
            inf_count = np.isinf(X_test_proc).sum()
            inf_cols = np.where(np.isinf(X_test_proc).any(axis=0))[0]
            raise ValueError(f"测试数据包含 {inf_count} 个无穷大值，位于列索引: {inf_cols.tolist()}")

        # 检查是否超出安全阈值，用更合理的值替换极端值
        if (np.abs(X_train_proc) > safe_threshold).any():
            large_count = (np.abs(X_train_proc) > safe_threshold).sum()
            large_cols = np.where((np.abs(X_train_proc) > safe_threshold).any(axis=0))[0]
            print(f"[WARNING] 训练数据有 {large_count} 个极端值（超出安全阈值），位于列索引: {large_cols.tolist()}")

            # 对每个包含极端值的列，用该列的 99 百分位数替换极端值
            for col_idx in large_cols:
                col_data = X_train_proc[:, col_idx]
                mask = np.abs(col_data) > safe_threshold

                # 计算该列非极端值的 99 百分位数
                normal_values = col_data[~mask]
                if len(normal_values) > 0:
                    p99 = np.percentile(np.abs(normal_values), 99)
                    replacement_value = p99
                else:
                    # 如果所有值都是极端值，用中位数
                    replacement_value = np.median(col_data[np.isfinite(col_data)])

                # 替换极端值，保持符号
                X_train_proc[mask, col_idx] = np.sign(col_data[mask]) * replacement_value
                print(f"[WARNING] 列 {col_idx}: 将 {mask.sum()} 个极端值替换为 ±{replacement_value:.4e}")

        if (np.abs(X_test_proc) > safe_threshold).any():
            large_count = (np.abs(X_test_proc) > safe_threshold).sum()
            large_cols = np.where((np.abs(X_test_proc) > safe_threshold).any(axis=0))[0]
            print(f"[WARNING] 测试数据有 {large_count} 个极端值（超出安全阈值），位于列索引: {large_cols.tolist()}")

            # 对测试集使用训练集的统计信息
            for col_idx in large_cols:
                col_data_test = X_test_proc[:, col_idx]
                col_data_train = X_train_proc[:, col_idx]
                mask = np.abs(col_data_test) > safe_threshold

                # 使用训练集的 99 百分位数
                p99 = np.percentile(np.abs(col_data_train), 99)
                X_test_proc[mask, col_idx] = np.sign(col_data_test[mask]) * p99
                print(f"[WARNING] 列 {col_idx}: 将 {mask.sum()} 个极端值替换为 ±{p99:.4e}")

        # 检查 y_train_fit
        if np.isinf(y_train_fit).any():
            raise ValueError("目标变量包含无穷大值，无法训练模型")

        print(f"[DEBUG] 数据检查完成，准备训练模型...")
        print(f"[DEBUG] 处理后统计: min={np.min(X_train_proc):.4e}, max={np.max(X_train_proc):.4e}")

        start_time = time.time()

        fit_kwargs = {}
        try:
            if model_name == "XGBoost" and XGBOOST_AVAILABLE:
                # XGBoost: 支持 eval_set + eval_metric，训练后可读取 evals_result
                early_stop = xgb_fit_params.get("early_stopping_rounds")
                verbose_eval = xgb_fit_params.get("verbose_eval") or 10  # 每10轮显示一次

                eval_sets = []
                if len(y_early_valid_model) >= 4 and X_early_valid_proc is not None:
                    eval_sets.append((X_early_valid_proc, y_early_valid_model))

                # 使用单个内置指标，避免部分 XGBoost 版本在预测阶段解析多指标失败
                eval_metric = "rmse"

                fit_kwargs = {
                    "eval_metric": eval_metric,
                    "verbose": int(verbose_eval) if int(verbose_eval) > 0 else False,
                }
                if eval_sets:
                    fit_kwargs["eval_set"] = eval_sets

                print(f"✓ 训练指标: RMSE (MAE和R²将从预测结果计算)")

                # 早停配置：默认启用，50轮不提升则停止
                early_stop_rounds = 500  # 默认500轮早停
                if eval_sets and early_stop is not None and int(early_stop) > 0:
                    early_stop_rounds = int(early_stop)
                    print(f"✓ 启用早停机制: {early_stop_rounds} 轮不提升则停止训练")
                elif eval_sets:
                    print(f"✓ 启用默认早停机制: {early_stop_rounds} 轮不提升则停止训练")

                # 检测XGBoost版本,使用正确的早停方式
                try:
                    import xgboost
                    xgb_version = tuple(map(int, xgboost.__version__.split('.')[:2]))

                    if eval_sets and xgb_version >= (2, 0):
                        # XGBoost 2.0+: 使用callbacks
                        print(f"[DEBUG] XGBoost {xgboost.__version__}: 使用callbacks方式")
                        try:
                            from xgboost.callback import EarlyStopping
                            # 通过模型的callbacks参数设置早停
                            if not hasattr(base_model, 'callbacks') or base_model.callbacks is None:
                                base_model.set_params(callbacks=[EarlyStopping(rounds=early_stop_rounds, save_best=True)])
                            print(f"[DEBUG] ✓ 已设置EarlyStopping callback")
                        except ImportError:
                            print(f"[WARNING] 无法导入EarlyStopping,尝试传统方式")
                            fit_kwargs["early_stopping_rounds"] = early_stop_rounds
                    elif eval_sets:
                        # XGBoost 1.x: 使用fit参数
                        print(f"[DEBUG] XGBoost {xgboost.__version__}: 使用fit参数方式")
                        fit_kwargs["early_stopping_rounds"] = early_stop_rounds
                except Exception as e:
                    if eval_sets:
                        print(f"[WARNING] 无法检测XGBoost版本: {e}, 使用传统方式")
                        fit_kwargs["early_stopping_rounds"] = early_stop_rounds
                if sample_weight is not None:
                    fit_kwargs["sample_weight"] = sample_weight
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                fit_kwargs = {}
                if len(y_early_valid_model) >= 4 and X_early_valid_proc is not None:
                    fit_kwargs = {
                        "eval_set": [(X_early_valid_proc, y_early_valid_model)],
                        "eval_names": ["internal_validation"],
                        "eval_metric": ["rmse", "mae", "l2"],
                        "callbacks": [],
                    }
                    try:
                        from lightgbm import early_stopping, log_evaluation
                        fit_kwargs["callbacks"].append(early_stopping(stopping_rounds=50, verbose=True))
                        fit_kwargs["callbacks"].append(log_evaluation(period=10))
                        print(f"✓ 启用LightGBM内部验证早停机制: 50 轮不提升则停止训练")
                    except ImportError:
                        fit_kwargs["early_stopping_rounds"] = 50
                        fit_kwargs["verbose"] = 10
                if sample_weight is not None:
                    fit_kwargs["sample_weight"] = sample_weight
            elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
                fit_kwargs = {
                    "verbose": 10,
                    "plot": False,
                    "metric_period": 10,
                }
                if len(y_early_valid_model) >= 4 and X_early_valid_proc is not None:
                    from catboost import Pool
                    fit_kwargs["eval_set"] = Pool(
                        X_early_valid_proc,
                        y_early_valid_model,
                    )
                    fit_kwargs["early_stopping_rounds"] = 50
                    print(f"✓ 启用CatBoost内部验证早停机制: 50 轮不提升则停止训练")
                if sample_weight is not None:
                    fit_kwargs["sample_weight"] = sample_weight
        except Exception:
            fit_kwargs = {}

        if (
            sample_weight is not None
            and model_name not in {"XGBoost", "LightGBM", "CatBoost"}
        ):
            fit_kwargs["sample_weight"] = sample_weight

        # 有些模型不接受额外 kwargs，做一次安全回退
        # 神经网络验证数据只来自训练集内部切分
        try:
            if model_name in {
                "人工神经网络",
                "TensorFlow Sequential",
                "Bayesian Neural Network (BNN)",
                "FT-Transformer",
            } and len(y_early_valid_model) >= 4 and X_early_valid_proc is not None:
                setattr(base_model, "validation_data", (X_early_valid_proc, y_early_valid_model))
                # TF 模型内部若使用 validation_split，会导致 Test 曲线不是同一批数据；这里优先用外部 validation_data
                if model_name == "TensorFlow Sequential" and hasattr(base_model, "validation_split"):
                    base_model.validation_split = 0.0
        except Exception:
            pass

        if model_name == "XGBoost" and XGBOOST_AVAILABLE:
            # 强制确保early_stopping生效
            print(f"[DEBUG] XGBoost训练参数: {fit_kwargs}")

            # 直接调用fit,不使用_safe_xgb_fit(它会过滤掉early_stopping)
            try:
                # 方法1: 直接传入所有参数
                base_model.fit(X_model_fit, y_model_fit, **fit_kwargs)
                print("[DEBUG] ✓ XGBoost训练成功(方法1: 直接fit)")
            except TypeError as e:
                print(f"[DEBUG] 方法1失败: {e}")
                if "sample_weight" in fit_kwargs:
                    fallback_indices = _weighted_resample_indices(
                        balance_info.get("weights", []),
                        random_state=random_state,
                    )
                    X_model_fit = X_fit_proc[fallback_indices]
                    y_model_fit = y_fit_model[fallback_indices]
                    fit_kwargs.pop("sample_weight", None)
                    actual_balance_method = "weighted_resample"
                    balance_info["fallback_reason"] = f"模型拒绝sample_weight: {e}"
                # 方法2: 移除eval_metric,保留early_stopping
                try:
                    fit_kwargs_backup = dict(fit_kwargs)
                    eval_metric = fit_kwargs_backup.pop("eval_metric", None)
                    if eval_metric:
                        base_model.set_params(eval_metric=eval_metric)
                    base_model.fit(X_model_fit, y_model_fit, **fit_kwargs_backup)
                    print("[DEBUG] ✓ XGBoost训练成功(方法2: eval_metric通过set_params)")
                except Exception as e2:
                    print(f"[DEBUG] 方法2失败: {e2}")
                    # 方法3: 使用_safe_xgb_fit作为最后手段
                    print("[WARNING] 使用_safe_xgb_fit,早停可能不生效")
                    _safe_xgb_fit(base_model, X_model_fit, y_model_fit, fit_kwargs or {})

            # 检查是否真的使用了早停
            if hasattr(base_model, 'best_iteration'):
                print(f"[DEBUG] ✓ 早停生效! 最佳迭代: {base_model.best_iteration}, 总迭代: {base_model.n_estimators}")
                if base_model.best_iteration < base_model.n_estimators - 50:
                    print(f"[DEBUG] ✓ 早停正常工作,节省了 {base_model.n_estimators - base_model.best_iteration} 轮训练")
                else:
                    print(f"[WARNING] 早停可能未生效,训练了几乎所有轮次")
            else:
                print("[WARNING] 模型没有best_iteration属性,早停可能未生效")

            import gc
            gc.collect()


            # [关键修复] 在清理之前先提取训练历史
            training_history_early = extract_history_from_fitted_model(
                model_name=model_name,
                model=base_model,
                X_train_scaled=X_train_proc,
                y_train=y_train,
                X_test_scaled=X_test_proc,
                y_test=y_test,
            )

            # [DEBUG] 检查训练历史中是否包含R²数据
            if training_history_early:
                available_metrics = [k for k in training_history_early.keys() if k not in ['kind', 'step']]
                print(f"[DEBUG] 训练历史包含指标: {', '.join(available_metrics)}")
                if 'train_r2' in training_history_early or 'test_r2' in training_history_early:
                    print(f"[DEBUG] ✓ R²指标已成功记录")
                else:
                    print(f"[DEBUG] ⚠️ 未找到R²指标，可能需要检查XGBoost版本")
            else:
                print(f"[DEBUG] ⚠️ 训练历史为空")

            # [关键修复] XGBoost 模型序列化问题：清理所有可能导致崩溃的内部引用
            try:
                import gc
                # 清理 eval_set 等大对象引用
                attrs_to_clean = [
                    '_eval_set', 'eval_set', '_validation_data',
                    'evals_result_', '_evals_result',
                    'feature_importances_', '_feature_importances'
                ]
                for attr in attrs_to_clean:
                    if hasattr(base_model, attr):
                        try:
                            delattr(base_model, attr)
                        except Exception:
                            pass

                # 强制垃圾回收
                gc.collect()
                print(f"[DEBUG] ✓ XGBoost model cleaned for serialization")
            except Exception as e:
                print(f"[DEBUG] ⚠️ XGBoost cleanup warning: {e}")
        else:
            _y_fit = y_train_fit if model_name == "人工神经网络" else y_train

            # [修复] 确保数据类型正确，sklearn 的随机森林等模型对 float64 更友好
            if X_train_proc.dtype != np.float64:
                print(f"[DEBUG] 转换 X_train_proc 从 {X_train_proc.dtype} 到 float64")
                X_train_proc = X_train_proc.astype(np.float64)
                X_test_proc = X_test_proc.astype(np.float64)

            # [修复] 再次检查转换后是否有问题
            if np.isinf(X_train_proc).any():
                raise ValueError("转换为 float64 后仍包含无穷大值")

            try:
                base_model.fit(X_model_fit, y_model_fit, **(fit_kwargs or {}))
            except TypeError as error:
                if "sample_weight" not in fit_kwargs:
                    base_model.fit(X_model_fit, y_model_fit)
                else:
                    fallback_indices = _weighted_resample_indices(
                        balance_info.get("weights", []),
                        random_state=random_state,
                    )
                    X_model_fit = X_fit_proc[fallback_indices]
                    y_model_fit = y_fit_model[fallback_indices]
                    fit_kwargs = dict(fit_kwargs)
                    fit_kwargs.pop("sample_weight", None)
                    actual_balance_method = "weighted_resample"
                    balance_info["fallback_reason"] = f"模型拒绝sample_weight: {error}"
                    base_model.fit(X_model_fit, y_model_fit, **fit_kwargs)

        train_time = time.time() - start_time

        model = base_model
        best_iteration = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)
        if best_score is not None:
            try:
                best_score = float(best_score)
            except Exception:
                pass

        # 6) 组装 Pipeline（不再重新 fit，用于后续 predict 保持一致）
        steps = []
        if process_pls_step is not None:
            steps.append(process_pls_step)
        if imputer is not None:
            steps.append(('imputer', imputer))
        # 添加 inf 清理步骤（即使训练时已清理，预测时也需要）
        steps.append(('inf_cleaner', InfCleaner()))
        # [修复] 添加特征掩码转换器，确保预测时应用相同的特征选择
        if not np.all(feature_mask):
            steps.append(('feature_mask', FeatureMaskTransformer(feature_mask)))
        if scaler is not None:
            steps.append(('scaler', scaler))
        steps.append(('model', model))
        pipeline = Pipeline(steps=steps)

        # 7) 预测与评估（用 pipeline 预测，保证一致）
        emit_transformer_postprocessing(
            "running deterministic evaluation",
            0.91,
            epochs_completed=int(getattr(model, "epochs_completed_", 0) or 0),
            total_epochs=int(getattr(model, "epochs", 0) or 0),
        )
        y_pred_test = pipeline.predict(X_test_raw)
        y_pred_train = pipeline.predict(X_train_raw)
        emit_transformer_postprocessing(
            "deterministic evaluation completed",
            0.94,
            epochs_completed=int(getattr(model, "epochs_completed_", 0) or 0),
            total_epochs=int(getattr(model, "epochs", 0) or 0),
        )

        y_std_test = None
        y_std_train = None
        if model_name == "Gaussian Process (GPR)" and hasattr(model, "predict"):
            try:
                _, y_std_test = model.predict(X_test_proc, return_std=True)
                _, y_std_train = model.predict(X_train_proc, return_std=True)
            except Exception:
                y_std_test = None
                y_std_train = None
        elif hasattr(model, "predict_with_uncertainty"):
            try:
                # 训练后的不确定度只用于评估展示，不应默认重复 50 次完整前向。
                # 保留用户较小的 mc_samples 设置，默认上限为 8 次以避免早停后长时间无反馈。
                evaluation_mc_samples = max(
                    1,
                    min(int(getattr(model, "mc_samples", 8) or 8), 8),
                )
                emit_transformer_postprocessing(
                    "estimating predictive uncertainty",
                    0.95,
                    mc_samples=int(evaluation_mc_samples),
                    epochs_completed=int(getattr(model, "epochs_completed_", 0) or 0),
                    total_epochs=int(getattr(model, "epochs", 0) or 0),
                )
                _, y_std_test = model.predict_with_uncertainty(
                    X_test_proc,
                    n_samples=evaluation_mc_samples,
                )
                _, y_std_train = model.predict_with_uncertainty(
                    X_train_proc,
                    n_samples=evaluation_mc_samples,
                )
                emit_transformer_postprocessing(
                    "evaluation completed",
                    0.98,
                    mc_samples=int(evaluation_mc_samples),
                    epochs_completed=int(getattr(model, "epochs_completed_", 0) or 0),
                    total_epochs=int(getattr(model, "epochs", 0) or 0),
                )
            except Exception:
                y_std_test = None
                y_std_train = None

        # 目标变量反归一化（ANN专用）
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
            y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

        # 8) 训练曲线提取（尽量不额外训练）
        # [修复] 对于 XGBoost，使用提前提取的训练历史
        if model_name == "XGBoost" and 'training_history_early' in locals() and training_history_early:
            training_history = training_history_early
        else:
            training_history = extract_history_from_fitted_model(
                model_name=model_name,
                model=model,
                X_train_scaled=X_train_proc,
                y_train=y_train,
                X_test_scaled=X_test_proc,
                y_test=y_test,
            )

        # 9) holdout-learning-curve（Train size -> Train/Test 指标）
        # - 默认：仅在无法提取训练历史时回退（更快）
        # - 可通过环境变量 ML_FORCE_LC=1 强制开启
        FORCE_LC = {"XGBoost", "LightGBM", "CatBoost", "多层感知器"}
        EXCLUDE = {"AutoGluon", "TabPFN", "Auto-sklearn", "TPOT", "FLAML", "Chemical SuperLearner (ChemSL)", "TabNet", "FT-Transformer", "Bayesian Neural Network (BNN)", "Transformer + BNN", "Transformer + PINN", "GNN + Transformer Fusion"}  # 排除深度学习模型，避免重复训练导致的崩溃
        force_lc = str(os.environ.get("ML_FORCE_LC", "")).strip().lower() in {"1", "true", "yes"}
        if (model_name in FORCE_LC) and force_lc:
            if model_name not in EXCLUDE:
                try:
                    training_history = build_holdout_learning_curve(
                        make_model=lambda: self._get_model(model_name, **model_params),
                        X_train_raw=X_train_raw,
                        y_train=y_train,
                        X_test_raw=X_test_raw,
                        y_test=y_test,
                        imputer_factory=(lambda: SimpleImputer(strategy='median')) if imputer is not None else None,
                        scaler_factory=(lambda: StandardScaler()) if scaler is not None else None,
                        random_state=random_state,
                    )
                except Exception:
                    # 若学习曲线构建失败，则保留原始 history（或空）
                    pass

        # XGBoost 特殊处理：训练完成后立即清理，防止 512 核机器崩溃
        if model_name == "XGBoost":
            try:
                import gc
                # 强制释放 XGBoost 内部缓存
                gc.collect()
            except Exception:
                pass

        balance_result = _finalize_target_balance_result(
            balance_info,
            actual_method=actual_balance_method,
            fit_sample_count=len(y_model_fit),
            early_stopping_validation_count=len(y_early_valid),
        )
        test_bin_metrics = _compute_regression_bin_metrics(
            y_test,
            y_pred_test,
            balance_info.get("bin_edges", []),
        )

        # [优化] 对于大数据集（>5000样本），不返回 raw 数据副本以节省内存
        skip_raw = (model_name == "XGBoost" and len(y_train) > 5000)

        print(f"[DEBUG] Creating return DataFrames (skip_raw={skip_raw})...")

        cols = feature_names if (feature_names is not None and len(feature_names) == X_train_proc.shape[1]) else [f"feat_{i}" for i in range(X_train_proc.shape[1])]
        raw_cols = raw_feature_names if process_pls_step is not None else cols
        print(f"[DEBUG] Using {len(cols)} column names")

        try:
            X_train_df = pd.DataFrame(X_train_proc, columns=cols, copy=False)
            print(f"[DEBUG] ✓ X_train_df created: {X_train_df.shape}")

            X_test_df = pd.DataFrame(X_test_proc, columns=cols, copy=False)
            print(f"[DEBUG] ✓ X_test_df created: {X_test_df.shape}")

            if skip_raw:
                X_train_raw_df = None
                X_test_raw_df = None
                print(f"[DEBUG] ✓ Skipped raw DataFrames")
            else:
                if isinstance(X_train_raw, pd.DataFrame):
                    X_train_raw_df = X_train_raw.copy()
                else:
                    X_train_raw_df = pd.DataFrame(X_train_raw, columns=raw_cols, copy=False)
                print(f"[DEBUG] ✓ X_train_raw_df created: {X_train_raw_df.shape}")
                if isinstance(X_test_raw, pd.DataFrame):
                    X_test_raw_df = X_test_raw.copy()
                else:
                    X_test_raw_df = pd.DataFrame(X_test_raw, columns=raw_cols, copy=False)
                print(f"[DEBUG] ✓ X_test_raw_df created: {X_test_raw_df.shape}")

            X_train_result = X_train_df
            X_test_result = X_test_df
            X_train_raw_result = X_train_raw_df
            X_test_raw_result = X_test_raw_df

        except Exception as e:
            print(f"[DEBUG] ⚠️ DataFrame creation failed: {e}, using arrays")
            X_train_result = X_train_proc
            X_test_result = X_test_proc
            X_train_raw_result = None if skip_raw else X_train_raw
            X_test_raw_result = None if skip_raw else X_test_raw

        print(f"[DEBUG] Preparing return dictionary...")

        # [关键修复] XGBoost 训练完成后，立即保存所有数据到临时文件
        # 避免在内存中传递大对象导致Streamlit崩溃
        temp_model_path = None
        temp_data_path = None
        if model_name == "XGBoost" and XGBOOST_AVAILABLE:
            try:
                import tempfile
                import joblib
                print(f"[DEBUG] Saving XGBoost results to temporary files...")
                temp_dir = tempfile.gettempdir()

                # 保存模型
                temp_model_path = os.path.join(temp_dir, f"xgb_model_{id(model)}.joblib")
                joblib.dump(model, temp_model_path)
                print(f"[DEBUG] ✓ Model saved to: {temp_model_path}")

                # 保存所有训练数据到一个文件
                temp_data_path = os.path.join(temp_dir, f"xgb_data_{id(model)}.joblib")
                data_bundle = {
                    'X_train': X_train_result,
                    'X_test': X_test_result,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'pipeline': pipeline,
                    'scaler': scaler,
                    'imputer': imputer,
                    'feature_mask': feature_mask,  # [修复] 保存特征掩码
                }
                joblib.dump(data_bundle, temp_data_path)
                print(f"[DEBUG] ✓ Data saved to: {temp_data_path}")

                # 从文件重新加载模型（确保干净）
                model = joblib.load(temp_model_path)
                print(f"[DEBUG] ✓ Model reloaded from file")

            except Exception as e:
                error_msg = str(e)
                if "pickle" in error_msg.lower():
                    print(f"[DEBUG] ⚠️ XGBoost序列化警告（不影响使用）: Pipeline包含无法序列化的组件")
                    print(f"[DEBUG]    提示: 模型仍可正常使用，只是无法保存到某些格式")
                else:
                    print(f"[DEBUG] ⚠️ XGBoost temp file save failed: {e}")
                temp_model_path = None
                temp_data_path = None

        # 对于XGBoost，如果保存成功，返回轻量级结果
        if model_name == "XGBoost" and temp_data_path:
            print(f"[DEBUG] Creating lightweight result dictionary for XGBoost...")
            result = {
                'model': model,
                'pipeline': None,  # 已保存到文件
                'scaler': None,
                'imputer': None,
                'feature_mask': feature_mask,  # [修复] 保存特征掩码
                'feature_names': list(cols),
                'X_train': None,  # 不返回大数据
                'X_test': None,
                'X_train_raw': None,
                'X_test_raw': None,
                'y_train': None,
                 'y_test': y_test,
                'y_pred': y_pred_test,
                'y_pred_test': y_pred_test,
                'y_pred_train': None,
                'y_std_test': y_std_test,
                'y_std_train': y_std_train,
                'r2': r2_score(y_test, y_pred_test),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test)),
                'train_time': float(train_time),
                'best_iteration': best_iteration,
                'best_score': best_score,
                'split_strategy': split_strategy,
                'n_bins': int(n_bins),
                'train_indices': None,
                'test_indices': None,
                 'training_history': training_history,
                 'training_history_df': history_to_frame(training_history) if training_history else pd.DataFrame(),
                 'target_balance': balance_result,
                 'test_bin_metrics': test_bin_metrics,
                 '_xgb_temp_model_path': temp_model_path,
                '_xgb_temp_data_path': temp_data_path,
                '_xgb_lightweight_mode': True,  # 标记为轻量级模式
            }
            print(f"[DEBUG] ✓ Lightweight result dictionary created")
            print(f"[DEBUG] About to return result...")
            import sys
            sys.stdout.flush()  # 强制刷新输出
        else:
            # 其他模型正常返回
            result = {
                'model': model,
                'pipeline': pipeline,
                'scaler': scaler,
            'imputer': imputer,
            'feature_mask': feature_mask,  # [修复] 保存特征掩码
            'feature_names': list(cols),
            'X_train': X_train_result,
            'X_test': X_test_result,
            'X_train_raw': X_train_raw_result,
            'X_test_raw': X_test_raw_result,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_std_test': y_std_test,
            'y_std_train': y_std_train,
            'r2': r2_score(y_test, y_pred_test),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'train_time': float(train_time),
            'best_iteration': best_iteration,
            'best_score': best_score,
            'split_strategy': split_strategy,
            'n_bins': int(n_bins),
            'train_indices': train_idx,
            'test_indices': test_idx,
             'training_history': training_history,
             'training_history_df': history_to_frame(training_history) if training_history else pd.DataFrame(),
             'target_balance': balance_result,
             'test_bin_metrics': test_bin_metrics,
             '_xgb_temp_model_path': temp_model_path,  # XGBoost临时文件路径
        }

        print(f"[DEBUG] ✓ Return dictionary ready, returning...")
        import sys
        sys.stdout.flush()  # 强制刷新输出
        print(f"[DEBUG] Result keys: {list(result.keys())}")
        print(f"[DEBUG] Model type: {type(result.get('model'))}")
        sys.stdout.flush()
        return result

    def _cross_validate_pinn_special(

        self,

        X,

        y,

        model_name,

        cv_strategy: str = 'repeated_kfold',

        n_splits: int = 5,

        n_repeats: int = 3,

        random_state: int = 42,

        groups=None,

        n_bins: int = 10,

        target_balance_enabled=True,

        balance_n_bins=10,

        balance_max_weight=3.0,

        **params

    ):

        """Epoxy PINN 专用交叉验证：保留 DataFrame 原始字符串列给模型解析。"""

        if isinstance(X, pd.DataFrame):

            X_df = X.copy()

        else:

            X_arr = np.asarray(X)

            X_df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])


        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)


        mask = np.isfinite(y_arr)

        X_df = X_df.loc[mask].reset_index(drop=True)

        y_arr = y_arr[mask]

        if groups is not None:

            groups = np.asarray(groups)[mask]


        n = len(y_arr)

        if n < 20:

            raise ValueError("有效样本过少，无法进行 CV")


        cv_strategy = (cv_strategy or 'repeated_kfold').lower()


        if cv_strategy == 'group_kfold':

            if groups is None:

                raise ValueError("group_kfold 需要 groups")

            splitter = GroupKFold(n_splits=n_splits)

            split_iter = splitter.split(np.zeros((n, 1)), y_arr, groups)

        elif cv_strategy == 'stratified_kfold':

            y_bins = _make_y_bins(y_arr, n_bins=n_bins)

            if y_bins is None:

                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

                split_iter = splitter.split(np.zeros((n, 1)), y_arr)

            else:

                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                split_iter = splitter.split(np.zeros((n, 1)), y_bins)

        else:

            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

            split_iter = splitter.split(np.zeros((n, 1)), y_arr)


        oof_sum = np.zeros(n, dtype=float)

        oof_cnt = np.zeros(n, dtype=int)

        fold_scores, fold_rmse, fold_mae = [], [], []
        fold_target_balance = []


        params.pop('train_n_jobs', None)


        for fold_i, (tr_idx, va_idx) in enumerate(split_iter):

            model_params = params.copy()

            model_params.setdefault("seed", int(random_state + fold_i))

            if _should_pass_target_name(model_name) and "target_name" not in model_params and hasattr(y, "name"):

                model_params["target_name"] = getattr(y, "name", None)


            base_model = self._get_model(model_name, random_state=int(random_state + fold_i), **model_params)
            fold_balance = _build_target_balance_info(
                y_arr[tr_idx],
                enabled=target_balance_enabled,
                n_bins=balance_n_bins,
                max_weight=balance_max_weight,
                random_state=int(random_state + fold_i),
            )
            fit_data = _prepare_balanced_fit_data(
                base_model,
                X_df.iloc[tr_idx].reset_index(drop=True),
                y_arr[tr_idx],
                fold_balance,
                random_state=int(random_state + fold_i),
            )
            fit_kwargs = {}
            if fit_data["sample_weight"] is not None:
                fit_kwargs["sample_weight"] = fit_data["sample_weight"]
            base_model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)

            pred = base_model.predict(X_df.iloc[va_idx].reset_index(drop=True))

            fold_target_balance.append(
                {
                    key: value
                    for key, value in fold_balance.items()
                    if key not in {"weights", "bin_ids"}
                }
                | {
                    "method": fit_data["method"],
                    "fit_sample_count": int(len(fit_data["y"])),
                }
            )

            oof_sum[va_idx] += pred

            oof_cnt[va_idx] += 1


            fold_scores.append(float(r2_score(y_arr[va_idx], pred)))

            fold_rmse.append(float(np.sqrt(mean_squared_error(y_arr[va_idx], pred))))

            fold_mae.append(float(mean_absolute_error(y_arr[va_idx], pred)))


        valid_mask = oof_cnt > 0

        oof_pred = np.zeros(n, dtype=float)

        oof_pred[valid_mask] = oof_sum[valid_mask] / oof_cnt[valid_mask]


        oof_r2 = float(r2_score(y_arr[valid_mask], oof_pred[valid_mask]))

        oof_rmse = float(np.sqrt(mean_squared_error(y_arr[valid_mask], oof_pred[valid_mask])))

        oof_mae = float(mean_absolute_error(y_arr[valid_mask], oof_pred[valid_mask]))


        return {

            'model_name': model_name,

            'cv_strategy': cv_strategy,

            'n_splits': int(n_splits),

            'n_repeats': int(n_repeats),

            'fold_r2': fold_scores,

            'fold_rmse': fold_rmse,

            'fold_mae': fold_mae,

            'cv_r2_mean': float(np.mean(fold_scores)) if len(fold_scores) else float('nan'),

            'cv_r2_std': float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,

            'oof_pred': oof_pred,

            'oof_true': y_arr,

            'oof_r2': oof_r2,

            'oof_rmse': oof_rmse,

            'oof_mae': oof_mae,

            'target_balance_enabled': bool(target_balance_enabled),

            'fold_target_balance': fold_target_balance,

        }

    def _cross_validate_graph_model(
        self,
        X,
        y,
        model_name,
        cv_strategy: str = "repeated_kfold",
        n_splits: int = 5,
        n_repeats: int = 5,
        random_state: int = 42,
        groups=None,
        n_bins: int = 10,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        **params,
    ):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("图神经网络模型需要 DataFrame 输入，并包含 SMILES 列")

        smiles_col = self._resolve_smiles_col(X, params.pop("smiles_col", None))
        if not smiles_col:
            raise ValueError("未检测到 SMILES 列，请在训练参数中指定 smiles_col")

        X_df = X[[smiles_col]].copy()
        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)
        mask = np.isfinite(y_arr)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if groups is not None:
            groups = np.asarray(groups)[mask]

        n = len(y_arr)
        if n < 3:
            raise ValueError("样本数太少，无法进行交叉验证")

        cv_strategy = (cv_strategy or "repeated_kfold").lower()
        n_splits = int(max(2, n_splits))
        n_repeats = int(max(1, n_repeats))

        splitter = None
        y_bins = None
        if cv_strategy in ["group_kfold", "group", "分组"]:
            if groups is None:
                raise ValueError("group_kfold 需要提供 groups")
            splitter = GroupKFold(n_splits=n_splits)
        elif cv_strategy in ["stratified_kfold", "stratified", "分层"]:
            y_bins = _make_y_bins(y_arr, n_bins=n_bins)
            if y_bins is None:
                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                cv_strategy = "repeated_kfold"
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        oof_sum = np.zeros(n, dtype=float)
        oof_cnt = np.zeros(n, dtype=int)

        fold_scores = []
        fold_rmse = []
        fold_mae = []
        fold_target_balance = []

        params.pop("train_n_jobs", None)
        model_params = params.copy()
        model_params.setdefault("smiles_col", smiles_col)
        model_params.setdefault("random_state", int(random_state))

        if isinstance(splitter, GroupKFold):
            split_iter = splitter.split(np.zeros((n, 1)), y_arr, groups)
        elif y_bins is not None:
            split_iter = splitter.split(np.zeros((n, 1)), y_bins)
        else:
            split_iter = splitter.split(np.zeros((n, 1)), y_arr)

        for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
            base_model = self._get_model(model_name, random_state=int(random_state + fold_i), **model_params)
            fold_balance = _build_target_balance_info(
                y_arr[tr_idx],
                enabled=target_balance_enabled,
                n_bins=balance_n_bins,
                max_weight=balance_max_weight,
                random_state=int(random_state + fold_i),
            )
            fit_data = _prepare_balanced_fit_data(
                base_model,
                X_df.iloc[tr_idx].reset_index(drop=True),
                y_arr[tr_idx],
                fold_balance,
                random_state=int(random_state + fold_i),
            )
            fit_kwargs = {}
            if fit_data["sample_weight"] is not None:
                fit_kwargs["sample_weight"] = fit_data["sample_weight"]
            base_model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)
            pred = np.asarray(base_model.predict(X_df.iloc[va_idx].reset_index(drop=True))).ravel()
            fold_target_balance.append(
                {
                    key: value
                    for key, value in fold_balance.items()
                    if key not in {"weights", "bin_ids"}
                }
                | {
                    "method": fit_data["method"],
                    "fit_sample_count": int(len(fit_data["y"])),
                }
            )

            y_fold = y_arr[va_idx]
            mask = np.isfinite(y_fold) & np.isfinite(pred)
            if mask.sum() == 0:
                fold_scores.append(float("nan"))
                fold_rmse.append(float("nan"))
                fold_mae.append(float("nan"))
                continue

            oof_sum[va_idx[mask]] += pred[mask]
            oof_cnt[va_idx[mask]] += 1

            fold_scores.append(r2_score(y_fold[mask], pred[mask]))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_fold[mask], pred[mask]))))
            fold_mae.append(float(mean_absolute_error(y_fold[mask], pred[mask])))

        oof_pred = np.zeros(n, dtype=float)
        valid_mask = oof_cnt > 0
        oof_pred[valid_mask] = oof_sum[valid_mask] / oof_cnt[valid_mask]

        if valid_mask.sum() == 0:
            oof_r2 = float("nan")
            oof_rmse = float("nan")
            oof_mae = float("nan")
        else:
            oof_r2 = r2_score(y_arr[valid_mask], oof_pred[valid_mask])
            oof_rmse = float(np.sqrt(mean_squared_error(y_arr[valid_mask], oof_pred[valid_mask])))
            oof_mae = float(mean_absolute_error(y_arr[valid_mask], oof_pred[valid_mask]))

        return {
            "cv_strategy": cv_strategy,
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "fold_r2": fold_scores,
            "fold_rmse": fold_rmse,
            "fold_mae": fold_mae,
            "cv_r2_mean": float(np.mean(fold_scores)) if len(fold_scores) else float("nan"),
            "cv_r2_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "oof_pred": oof_pred,
            "oof_true": y_arr,
            "oof_r2": float(oof_r2),
            "oof_rmse": float(oof_rmse),
            "oof_mae": float(oof_mae),
            "target_balance_enabled": bool(target_balance_enabled),
            "fold_target_balance": fold_target_balance,
        }

    def _cross_validate_raw_frame_model(
        self,
        X,
        y,
        model_name,
        cv_strategy: str = "repeated_kfold",
        n_splits: int = 5,
        n_repeats: int = 5,
        random_state: int = 42,
        groups=None,
        n_bins: int = 10,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        process_pls_config=None,
        use_process_pls=False,
        **params,
    ):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            X_df = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])

        y_arr = pd.to_numeric(np.asarray(y).ravel(), errors="coerce").astype(float)
        mask = np.isfinite(y_arr)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if groups is not None:
            groups = np.asarray(groups)[mask]

        if len(y_arr) < 3:
            raise ValueError("样本数太少，无法进行交叉验证")

        params.pop("train_n_jobs", None)
        model_params = params.copy()
        if model_name in RAW_FRAME_MODELS_WITH_SMILES:
            smiles_col = self._resolve_smiles_col(X_df, model_params.pop("smiles_col", None))
            if not smiles_col:
                raise ValueError("未检测到 SMILES 列，请在训练参数中指定 smiles_col")
            model_params["smiles_col"] = smiles_col
        if use_process_pls and model_name in RAW_FRAME_MODELS_WITH_SMILES:
            raise ValueError("工艺 PLS 暂不支持含 SMILES 的原始帧融合模型，请先关闭工艺 PLS")
        _make_process_pls_step(
            process_pls_config,
            use_process_pls,
            X_df.columns.tolist(),
        )

        cv_strategy = (cv_strategy or "repeated_kfold").lower()
        n_splits = int(max(2, n_splits))
        n_repeats = int(max(1, n_repeats))

        splitter = None
        y_bins = None
        if cv_strategy in ["group_kfold", "group", "分组"]:
            if groups is None:
                raise ValueError("group_kfold 需要提供 groups")
            splitter = GroupKFold(n_splits=n_splits)
        elif cv_strategy in ["stratified_kfold", "stratified", "分层"]:
            y_bins = _make_y_bins(y_arr, n_bins=n_bins)
            if y_bins is None:
                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                cv_strategy = "repeated_kfold"
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        if isinstance(splitter, GroupKFold):
            split_iter = splitter.split(np.zeros((len(y_arr), 1)), y_arr, groups)
        elif y_bins is not None:
            split_iter = splitter.split(np.zeros((len(y_arr), 1)), y_bins)
        else:
            split_iter = splitter.split(np.zeros((len(y_arr), 1)), y_arr)

        oof_sum = np.zeros(len(y_arr), dtype=float)
        oof_cnt = np.zeros(len(y_arr), dtype=int)
        fold_scores = []
        fold_rmse = []
        fold_mae = []
        fold_target_balance = []

        for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
            fold_params = model_params.copy()
            if _should_pass_target_name(model_name) and "target_name" not in fold_params and hasattr(y, "name"):
                fold_params["target_name"] = getattr(y, "name", None)
            base_model = self._get_model(model_name, random_state=int(random_state + fold_i), **fold_params)
            X_train_fold = X_df.iloc[tr_idx].reset_index(drop=True)
            X_valid_fold = X_df.iloc[va_idx].reset_index(drop=True)
            if use_process_pls:
                process_pls = _make_process_pls_step(
                    process_pls_config,
                    True,
                    X_df.columns.tolist(),
                )[1]
                process_pls.fit(X_train_fold, y_arr[tr_idx])
                X_train_fold = process_pls.transform(X_train_fold)
                X_valid_fold = process_pls.transform(X_valid_fold)
            fold_balance = _build_target_balance_info(
                y_arr[tr_idx],
                enabled=target_balance_enabled,
                n_bins=balance_n_bins,
                max_weight=balance_max_weight,
                random_state=int(random_state + fold_i),
            )
            fit_data = _prepare_balanced_fit_data(
                base_model,
                X_train_fold,
                y_arr[tr_idx],
                fold_balance,
                random_state=int(random_state + fold_i),
            )
            fit_kwargs = {}
            if fit_data["sample_weight"] is not None:
                fit_kwargs["sample_weight"] = fit_data["sample_weight"]
            base_model.fit(fit_data["X"], fit_data["y"], **fit_kwargs)
            pred = np.asarray(base_model.predict(X_valid_fold)).ravel()
            fold_target_balance.append(
                {
                    key: value
                    for key, value in fold_balance.items()
                    if key not in {"weights", "bin_ids"}
                }
                | {
                    "method": fit_data["method"],
                    "fit_sample_count": int(len(fit_data["y"])),
                }
            )
            y_fold = y_arr[va_idx]
            metric_mask = np.isfinite(y_fold) & np.isfinite(pred)
            if metric_mask.sum() == 0:
                fold_scores.append(float("nan"))
                fold_rmse.append(float("nan"))
                fold_mae.append(float("nan"))
                continue

            safe_idx = va_idx[metric_mask]
            oof_sum[safe_idx] += pred[metric_mask]
            oof_cnt[safe_idx] += 1
            fold_scores.append(float(r2_score(y_fold[metric_mask], pred[metric_mask])))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_fold[metric_mask], pred[metric_mask]))))
            fold_mae.append(float(mean_absolute_error(y_fold[metric_mask], pred[metric_mask])))

        oof_pred = np.zeros(len(y_arr), dtype=float)
        valid_mask = oof_cnt > 0
        oof_pred[valid_mask] = oof_sum[valid_mask] / oof_cnt[valid_mask]

        if valid_mask.sum() == 0:
            oof_r2 = float("nan")
            oof_rmse = float("nan")
            oof_mae = float("nan")
        else:
            oof_r2 = float(r2_score(y_arr[valid_mask], oof_pred[valid_mask]))
            oof_rmse = float(np.sqrt(mean_squared_error(y_arr[valid_mask], oof_pred[valid_mask])))
            oof_mae = float(mean_absolute_error(y_arr[valid_mask], oof_pred[valid_mask]))

        return {
            "cv_strategy": cv_strategy,
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "fold_r2": fold_scores,
            "fold_rmse": fold_rmse,
            "fold_mae": fold_mae,
            "cv_r2_mean": float(np.mean(fold_scores)) if len(fold_scores) else float("nan"),
            "cv_r2_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "oof_pred": oof_pred,
            "oof_true": y_arr,
            "oof_r2": oof_r2,
            "oof_rmse": oof_rmse,
            "oof_mae": oof_mae,
            "target_balance_enabled": bool(target_balance_enabled),
            "fold_target_balance": fold_target_balance,
        }


    def cross_validate_model(
        self,
        X,
        y,
        model_name,
        cv_strategy: str = 'repeated_kfold',
        n_splits: int = 5,
        n_repeats: int = 5,
        random_state: int = 42,
        groups=None,
        n_bins: int = 10,
        drop_missing_rows=True,
        target_balance_enabled=True,
        balance_n_bins=10,
        balance_max_weight=3.0,
        process_pls_config=None,
        use_process_pls=False,
        **params
    ):
        """交叉验证（输出每折分数 + OOF 预测）

        cv_strategy:
            - repeated_kfold: RepeatedKFold
            - stratified_kfold: 对 y 分箱后用 StratifiedKFold
            - group_kfold: GroupKFold（需要 groups）
        """
        # Epoxy PINN 专用分支：允许原始字符串列，由模型内部解析（用于 Tg / 力学等物理约束）
        if str(model_name) == "Epoxy PINN (Physics-Informed)":
            return self._cross_validate_pinn_special(
                X=X,
                y=y,
                model_name=model_name,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                groups=groups,
                n_bins=n_bins,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                **params
            )

        if _is_classification_model(model_name):
            return self._cross_validate_classification_model(
                X=X,
                y=y,
                model_name=model_name,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                groups=groups,
                n_bins=n_bins,
                drop_missing_rows=drop_missing_rows,
                **params,
            )

        if str(model_name) in RAW_FRAME_MODEL_NAMES:
            return self._cross_validate_raw_frame_model(
                X=X,
                y=y,
                model_name=model_name,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                groups=groups,
                n_bins=n_bins,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                process_pls_config=process_pls_config,
                use_process_pls=use_process_pls,
                **params,
            )

        if str(model_name) in GRAPH_MODEL_NAMES:
            return self._cross_validate_graph_model(
                X=X,
                y=y,
                model_name=model_name,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                groups=groups,
                n_bins=n_bins,
                target_balance_enabled=target_balance_enabled,
                balance_n_bins=balance_n_bins,
                balance_max_weight=balance_max_weight,
                **params,
            )



        feature_names = None
        X_df = None
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            try:
                # 先处理字符串形式的布尔值
                for c in X_df.columns:
                    if X_df[c].dtype == 'object':
                        X_df[c] = X_df[c].replace({'True': 1, 'true': 1, 'TRUE': 1,
                                                     'False': 0, 'false': 0, 'FALSE': 0})

                X_df = X_df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                for c in X_df.columns:
                    if X_df[c].dtype == 'object':
                        X_df[c] = X_df[c].replace({'True': 1, 'true': 1, 'TRUE': 1,
                                                     'False': 0, 'false': 0, 'FALSE': 0})
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            dropped_all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
            if dropped_all_nan_cols:
                X_df = X_df.drop(columns=dropped_all_nan_cols)
            feature_names = X_df.columns.tolist()
            X_arr = X_df.values
        else:
            X_arr = np.asarray(X)
            feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=feature_names)
            try:
                X_df = X_df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                for c in X_df.columns:
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            dropped_all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
            if dropped_all_nan_cols:
                X_df = X_df.drop(columns=dropped_all_nan_cols)
            feature_names = X_df.columns.tolist()
            X_arr = X_df.values

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        y_arr = pd.to_numeric(pd.Series(y_arr), errors='coerce').values
        mask = (~np.isnan(y_arr)) & (~np.isinf(y_arr))
        if np.sum(~mask) > 0:
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            if X_df is not None:
                X_df = X_df.loc[mask].reset_index(drop=True)
            if groups is not None:
                groups = np.asarray(groups)[mask]

        if X_df is not None:
            X_df = _sanitize_feature_frame(X_df, model_name)
            dropped_all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
            if dropped_all_nan_cols:
                X_df = X_df.drop(columns=dropped_all_nan_cols)
                feature_names = X_df.columns.tolist()
            X_arr = X_df.values

        print(f"✓ 交叉验证有效样本数: {len(y_arr)} 行（已删除目标列缺失值）")

        # 确保 X_arr 是数值类型（所有模型都需要）
        if X_df is not None:
            try:
                X_arr = X_arr.astype(float)
            except (ValueError, TypeError) as e:
                raise ValueError(f"特征数据包含非数值类型，无法训练。请检查特征选择是否包含了文本列（如 SMILES）。错误: {e}")

        process_pls_enabled = bool(use_process_pls)
        _make_process_pls_step(
            process_pls_config,
            process_pls_enabled,
            feature_names,
        )

        # 只排除目标列缺失的样本；输入特征缺失行保留到各折，
        # 由折内插补器或模型自身的缺失值能力处理。

        n = len(y_arr)
        if n < 3:
            raise ValueError("样本数太少，无法进行交叉验证")

        cv_strategy = (cv_strategy or 'repeated_kfold').lower()
        n_splits = int(max(2, n_splits))
        n_repeats = int(max(1, n_repeats))

        splitter = None
        y_bins = None

        if cv_strategy in ['group_kfold', 'group', '分组']:
            if groups is None:
                raise ValueError("group_kfold 需要提供 groups")
            splitter = GroupKFold(n_splits=n_splits)
        elif cv_strategy in ['stratified_kfold', 'stratified', '分层']:
            y_bins = _make_y_bins(y_arr, n_bins=n_bins)
            if y_bins is None:
                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                cv_strategy = 'repeated_kfold'
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        # OOF：重复 CV 时每个样本会预测多次，这里取平均
        oof_sum = np.zeros(n, dtype=float)
        oof_cnt = np.zeros(n, dtype=int)

        fold_scores = []
        fold_rmse = []
        fold_mae = []
        fold_target_balance = []

        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]
        # 并行训练设置（UI：训练并行核数）
        train_n_jobs = params.pop('train_n_jobs', -1)
        train_n_jobs = _normalize_train_n_jobs(train_n_jobs)
        model_params = params.copy()
        _apply_parallel_settings(model_name, model_params, train_n_jobs, use_gpu=self.use_gpu)
        _apply_gpu_settings_for_neural_networks(model_name, model_params, use_gpu=self.use_gpu)
        if model_name not in NO_SEED_MODELS:
            model_params.setdefault('random_state', random_state)
        xgb_fit_params = {}
        if model_name == "XGBoost":
            xgb_fit_params["early_stopping_rounds"] = model_params.pop("early_stopping_rounds", None)
            xgb_fit_params["eval_metric"] = model_params.pop("eval_metric", None)
            xgb_fit_params["verbose_eval"] = model_params.pop("verbose_eval", None)

        # 根据 splitter 类型选择正确的 split 调用方式
        if isinstance(splitter, GroupKFold):
            split_iter = list(splitter.split(X_arr, y_arr, groups))
        elif y_bins is not None:
            split_iter = list(splitter.split(X_arr, y_bins))
        else:
            split_iter = list(splitter.split(X_arr, y_arr))

        # ============ 多GPU并行交叉验证 ============
        # 检测是否支持多GPU并行
        use_multi_gpu = False
        available_gpus = []

        if model_name == "XGBoost" and XGBOOST_AVAILABLE and not process_pls_enabled:
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    available_gpus = list(range(torch.cuda.device_count()))
                    use_multi_gpu = True
                    print(f"✓ 检测到 {len(available_gpus)} 个GPU，启用并行交叉验证")
            except:
                pass

        if use_multi_gpu and len(split_iter) > 1:
            # 并行训练每一折
            from joblib import Parallel, delayed

            def train_single_fold(fold_i, tr_idx, va_idx, gpu_id):
                """训练单个fold"""
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                # 复制模型参数并设置GPU
                fold_params = model_params.copy()
                try:
                    import xgboost as xgb
                    xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
                    if xgb_version >= (2, 0):
                        fold_params["device"] = f"cuda:{gpu_id}"
                    else:
                        fold_params["tree_method"] = "gpu_hist"
                        fold_params["gpu_id"] = gpu_id
                except:
                    pass

                base_model = self._get_model(model_name, **fold_params)
                fold_balance = _build_target_balance_info(
                    y_arr[tr_idx],
                    enabled=target_balance_enabled,
                    n_bins=balance_n_bins,
                    max_weight=balance_max_weight,
                    random_state=int(random_state + fold_i),
                )
                fit_data = _prepare_balanced_fit_data(
                    base_model,
                    X_arr[tr_idx],
                    y_arr[tr_idx],
                    fold_balance,
                    random_state=int(random_state + fold_i),
                )

                fit_kwargs = {
                    "eval_set": [(X_arr[va_idx], y_arr[va_idx])],
                    "eval_metric": xgb_fit_params.get("eval_metric") or "rmse",
                }
                verbose_eval = xgb_fit_params.get("verbose_eval") or 0
                fit_kwargs["verbose"] = int(verbose_eval) if int(verbose_eval) > 0 else False
                early_stop = xgb_fit_params.get("early_stopping_rounds")
                if early_stop is not None and int(early_stop) > 0:
                    fit_kwargs["early_stopping_rounds"] = int(early_stop)
                if fit_data["sample_weight"] is not None:
                    fit_kwargs["sample_weight"] = fit_data["sample_weight"]

                _safe_xgb_fit(base_model, fit_data["X"], fit_data["y"], fit_kwargs)
                pred = base_model.predict(X_arr[va_idx])

                return (
                    fold_i,
                    va_idx,
                    pred,
                    y_arr[va_idx],
                    fold_balance,
                    fit_data["method"],
                    int(len(fit_data["y"])),
                )

            # 并行执行，每个fold分配到不同GPU
            print(f"🚀 开始并行训练 {len(split_iter)} 折...")
            results = Parallel(n_jobs=len(available_gpus), backend='threading')(
                delayed(train_single_fold)(
                    fold_i, tr_idx, va_idx, available_gpus[fold_i % len(available_gpus)]
                )
                for fold_i, (tr_idx, va_idx) in enumerate(split_iter)
            )

            # 收集结果
            for fold_i, va_idx, pred, y_true, fold_balance, fit_method, fit_sample_count in results:
                oof_sum[va_idx] += pred
                oof_cnt[va_idx] += 1
                fold_scores.append(r2_score(y_true, pred))
                fold_rmse.append(float(np.sqrt(mean_squared_error(y_true, pred))))
                fold_mae.append(float(mean_absolute_error(y_true, pred)))
                fold_target_balance.append(
                    {
                        key: value
                        for key, value in fold_balance.items()
                        if key not in {"weights", "bin_ids"}
                    }
                    | {
                        "method": fit_method,
                        "fit_sample_count": fit_sample_count,
                    }
                )
                print(f"  Fold {fold_i+1}: R²={fold_scores[-1]:.4f}")

        else:
            # 串行训练（原有逻辑）
            for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
                base_model = self._get_model(model_name, **model_params)
                if process_pls_enabled:
                    X_train_fold = X_df.iloc[tr_idx].reset_index(drop=True)
                    X_valid_fold = X_df.iloc[va_idx].reset_index(drop=True)
                    process_pls = _make_process_pls_step(
                        process_pls_config,
                        True,
                        feature_names,
                    )[1]
                    process_pls.fit(X_train_fold, y_arr[tr_idx])
                    X_train_fold = process_pls.transform(X_train_fold)
                    X_valid_fold = process_pls.transform(X_valid_fold)
                else:
                    X_train_fold = X_arr[tr_idx]
                    X_valid_fold = X_arr[va_idx]
                fold_balance = _build_target_balance_info(
                    y_arr[tr_idx],
                    enabled=target_balance_enabled,
                    n_bins=balance_n_bins,
                    max_weight=balance_max_weight,
                    random_state=int(random_state + fold_i),
                )

                if model_name == "XGBoost" and XGBOOST_AVAILABLE:
                    # XGBoost early stopping requires an eval_set in CV folds.
                    fit_kwargs = {
                        "eval_set": [(X_valid_fold, y_arr[va_idx])],
                        "eval_metric": xgb_fit_params.get("eval_metric") or "rmse",
                    }
                    verbose_eval = xgb_fit_params.get("verbose_eval") or 0
                    fit_kwargs["verbose"] = int(verbose_eval) if int(verbose_eval) > 0 else False
                    early_stop = xgb_fit_params.get("early_stopping_rounds")
                    if early_stop is not None and int(early_stop) > 0:
                        fit_kwargs["early_stopping_rounds"] = int(early_stop)

                    fit_data = _prepare_balanced_fit_data(
                        base_model,
                        X_train_fold,
                        y_arr[tr_idx],
                        fold_balance,
                        random_state=int(random_state + fold_i),
                    )
                    if fit_data["sample_weight"] is not None:
                        fit_kwargs["sample_weight"] = fit_data["sample_weight"]
                    _safe_xgb_fit(base_model, fit_data["X"], fit_data["y"], fit_kwargs)
                    pred = base_model.predict(X_valid_fold)
                else:
                    pipe = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('inf_cleaner', InfCleaner()),
                        ('nan_col_dropper', AllNaNColumnDropper()),
                        ('scaler', StandardScaler()),
                        ('model', base_model)
                    ])

                    fit_data = _prepare_balanced_fit_data(
                        base_model,
                        X_train_fold,
                        y_arr[tr_idx],
                        fold_balance,
                        random_state=int(random_state + fold_i),
                    )
                    fit_kwargs = {}
                    if fit_data["sample_weight"] is not None:
                        fit_kwargs["model__sample_weight"] = fit_data["sample_weight"]
                    pipe.fit(fit_data["X"], fit_data["y"], **fit_kwargs)
                    pred = pipe.predict(X_valid_fold)

                fold_target_balance.append(
                    {
                        key: value
                        for key, value in fold_balance.items()
                        if key not in {"weights", "bin_ids"}
                    }
                    | {
                        "method": fit_data["method"],
                        "fit_sample_count": int(len(fit_data["y"])),
                    }
                )

                oof_sum[va_idx] += pred
                oof_cnt[va_idx] += 1

                fold_scores.append(r2_score(y_arr[va_idx], pred))
                fold_rmse.append(float(np.sqrt(mean_squared_error(y_arr[va_idx], pred))))
                fold_mae.append(float(mean_absolute_error(y_arr[va_idx], pred)))

        # 汇总 OOF
        oof_pred = np.zeros(n, dtype=float)
        valid_mask = oof_cnt > 0
        oof_pred[valid_mask] = oof_sum[valid_mask] / oof_cnt[valid_mask]

        oof_r2 = r2_score(y_arr[valid_mask], oof_pred[valid_mask])
        oof_rmse = float(np.sqrt(mean_squared_error(y_arr[valid_mask], oof_pred[valid_mask])))
        oof_mae = float(mean_absolute_error(y_arr[valid_mask], oof_pred[valid_mask]))

        return {
            'cv_strategy': cv_strategy,
            'n_splits': int(n_splits),
            'n_repeats': int(n_repeats),
            'fold_r2': fold_scores,
            'fold_rmse': fold_rmse,
            'fold_mae': fold_mae,
            'cv_r2_mean': float(np.mean(fold_scores)) if len(fold_scores) else float('nan'),
            'cv_r2_std': float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            'oof_pred': oof_pred,
            'oof_true': y_arr,
            'oof_r2': float(oof_r2),
            'oof_rmse': float(oof_rmse),
            'oof_mae': float(oof_mae),
            'target_balance_enabled': bool(target_balance_enabled),
            'fold_target_balance': fold_target_balance,
        }
