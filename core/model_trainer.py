# -*- coding: utf-8 -*-
"""模型训练模块

增强点（面向 Tg / 力学等小样本回归任务的稳健训练）：
1) 支持多种划分策略：随机 / 回归分箱分层 / 按配方分组
2) 支持 Repeated KFold / GroupKFold 的交叉验证，并输出 OOF 预测
3) 统一用 Pipeline 保存 imputer + scaler + model，避免预测阶段漏变换
"""

import time
import numpy as np
import pandas as pd
import shutil

from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    GroupKFold,
    RepeatedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# [修复] 导入自定义 ANN 模型
try:
    from .ann_model import ANNRegressor
    ANN_AVAILABLE = True
except Exception:
    ANN_AVAILABLE = False
    ANNRegressor = None


def _safe_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), True
    except Exception:
        return None, False


XGBRegressor, XGBOOST_AVAILABLE = _safe_import('xgboost', 'XGBRegressor')
LGBMRegressor, LIGHTGBM_AVAILABLE = _safe_import('lightgbm', 'LGBMRegressor')
CatBoostRegressor, CATBOOST_AVAILABLE = _safe_import('catboost', 'CatBoostRegressor')
TabPFNRegressor, TABPFN_AVAILABLE = _safe_import('tabpfn', 'TabPFNRegressor')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except Exception:
    AUTOGLUON_AVAILABLE = False


def _make_y_bins(y: np.ndarray, n_bins: int = 10):
    """把连续 y 分箱用于“回归分层划分”。

    返回:
        bins (np.ndarray[int]) 或 None（表示无法分箱，需回退随机划分）
    """
    y = np.asarray(y).ravel()
    if len(y) < 3:
        return None

    n_bins = int(max(2, n_bins))
    # qcut 在重复值多时会自动 drop bins（duplicates='drop'）
    try:
        bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates='drop')
        bins = np.asarray(bins)
        if np.unique(bins).size < 2:
            return None
        return bins
    except Exception:
        # 兜底：用分位数手动构箱
        try:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
            if len(edges) < 3:
                return None
            # digitize 生成 1..len(edges)-1
            bins = np.digitize(y, edges[1:-1], right=True)
            if np.unique(bins).size < 2:
                return None
            return bins
        except Exception:
            return None


# --- [新增] AutoGluon 适配器 ---
class AutoGluonWrapper(BaseEstimator, RegressorMixin):
    """将 AutoGluon 封装为 Scikit-Learn 风格的 Estimator"""

    def __init__(self, time_limit=60, presets='medium_quality', **kwargs):
        self.time_limit = time_limit
        self.presets = presets
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

        self.predictor = TabularPredictor(
            label=self.label_col,
            path=self.save_path,
            verbosity=0
        ).fit(
            train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            **self.kwargs
        )
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
        # (可选) 清理临时模型文件，防止磁盘占满
        # try:
        #     shutil.rmtree(self.save_path)
        # except:
        #     pass
        pass


class EnhancedModelTrainer:
    """增强版模型训练器"""

    def __init__(self):
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        models = [
            "线性回归", "Ridge回归", "Lasso回归", "ElasticNet",
            "决策树", "随机森林", "Extra Trees", "梯度提升树", "AdaBoost", "SVR", "多层感知器"
        ]
        if XGBOOST_AVAILABLE:
            models.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            models.append("LightGBM")
        if CATBOOST_AVAILABLE:
            models.append("CatBoost")
        if TABPFN_AVAILABLE:
            models.append("TabPFN")
        if AUTOGLUON_AVAILABLE:
            models.append("AutoGluon")
        if ANN_AVAILABLE:
            models.append("人工神经网络")
        return models

    def get_available_models(self):
        return self.available_models.copy()

    def _get_model(self, name, **params):
        models = {
            "线性回归": LinearRegression,
            "Ridge回归": Ridge,
            "Lasso回归": Lasso,
            "ElasticNet": ElasticNet,
            "决策树": DecisionTreeRegressor,
            "随机森林": RandomForestRegressor,
            "Extra Trees": ExtraTreesRegressor,
            "梯度提升树": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor,
            "SVR": SVR,
            "多层感知器": MLPRegressor,
        }

        if name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBRegressor(**params)
        elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
            params.setdefault('verbose', -1)
            return LGBMRegressor(**params)
        elif name == "CatBoost" and CATBOOST_AVAILABLE:
            params.setdefault('verbose', 0)
            return CatBoostRegressor(**params)
        elif name == "TabPFN" and TABPFN_AVAILABLE:
            return TabPFNRegressor(**params)
        elif name == "AutoGluon" and AUTOGLUON_AVAILABLE:
            params.setdefault('time_limit', 30)
            return AutoGluonWrapper(**params)
        elif name == "人工神经网络" and ANN_AVAILABLE:
            return ANNRegressor(**params)
        elif name in models:
            return models[name](**params)
        else:
            raise ValueError(f"Unknown model: {name}")

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
        **params
    ):
        """训练单个模型（支持随机/分层/分组划分）"""

        # 1) 输入统一为 numpy，并做 y 清洗
        feature_names = None
        X_df = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_df = X.copy()
            X_arr = X.values
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
            y_arr = y_arr[mask]
            if groups is not None:
                groups = np.asarray(groups)[mask]

        if len(y_arr) == 0:
            raise ValueError("所有样本的目标变量均无效（NaN/Inf），无法训练模型，请检查数据")

        # 2) 划分索引
        train_idx, test_idx = self._resolve_split(
            X_arr, y_arr, test_size=test_size, random_state=random_state,
            split_strategy=split_strategy, n_bins=n_bins, groups=groups
        )

        X_train_raw = X_arr[train_idx]
        X_test_raw = X_arr[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        # 3) 模型参数注入 random_state（对不支持的模型跳过）
        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]
        model_params = params.copy()
        if model_name not in NO_SEED_MODELS:
            model_params.setdefault('random_state', random_state)

        base_model = self._get_model(model_name, **model_params)

        # 4) Pipeline：imputer + scaler + model
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        start_time = time.time()
        pipeline.fit(X_train_raw, y_train)
        train_time = time.time() - start_time

        # 5) 取出拟合后的组件
        imputer = pipeline.named_steps['imputer']
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['model']

        # 6) 生成“缩放后”的训练/测试特征（用于解释器/可视化）
        X_train_scaled = scaler.transform(imputer.transform(X_train_raw))
        X_test_scaled = scaler.transform(imputer.transform(X_test_raw))

        # 7) 预测与评估（用 pipeline 预测，保证一致）
        y_pred_test = pipeline.predict(X_test_raw)
        y_pred_train = pipeline.predict(X_train_raw)

        return {
            'model': model,
            'pipeline': pipeline,
            'scaler': scaler,
            'imputer': imputer,
            'X_train': pd.DataFrame(X_train_scaled, columns=feature_names),
            'X_test': pd.DataFrame(X_test_scaled, columns=feature_names),
            'X_train_raw': pd.DataFrame(X_train_raw, columns=feature_names),
            'X_test_raw': pd.DataFrame(X_test_raw, columns=feature_names),
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'r2': r2_score(y_test, y_pred_test),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'train_time': float(train_time),
            'split_strategy': split_strategy,
            'n_bins': int(n_bins),
            'train_indices': train_idx,
            'test_indices': test_idx,
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
        **params
    ):
        """交叉验证（输出每折分数 + OOF 预测）

        cv_strategy:
            - repeated_kfold: RepeatedKFold
            - stratified_kfold: 对 y 分箱后用 StratifiedKFold
            - group_kfold: GroupKFold（需要 groups）
        """

        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        y_arr = pd.to_numeric(pd.Series(y_arr), errors='coerce').values
        mask = (~np.isnan(y_arr)) & (~np.isinf(y_arr))
        if np.sum(~mask) > 0:
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            if groups is not None:
                groups = np.asarray(groups)[mask]

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

        NO_SEED_MODELS = ["线性回归", "SVR", "TabPFN", "AutoGluon"]
        model_params = params.copy()
        if model_name not in NO_SEED_MODELS:
            model_params.setdefault('random_state', random_state)

        # 使用与 train_model 相同的预处理流程
        for fold_i, (tr_idx, va_idx) in enumerate(
            splitter.split(X_arr, y_bins if y_bins is not None else y_arr, groups)
            if groups is not None and isinstance(splitter, GroupKFold)
            else splitter.split(X_arr, y_bins if y_bins is not None else y_arr)
        ):
            base_model = self._get_model(model_name, **model_params)

            pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', base_model)
            ])

            pipe.fit(X_arr[tr_idx], y_arr[tr_idx])
            pred = pipe.predict(X_arr[va_idx])

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
        }
