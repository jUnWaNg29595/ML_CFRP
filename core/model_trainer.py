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
LGBMRegressor, LIGHTGBM_AVAILABLE = _safe_import('lightgbm', 'LGBMRegressor')
CatBoostRegressor, CATBOOST_AVAILABLE = _safe_import('catboost', 'CatBoostRegressor')
TabPFNRegressor, TABPFN_AVAILABLE = _safe_import('tabpfn', 'TabPFNRegressor')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except Exception:
    AUTOGLUON_AVAILABLE = False


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


# --- AutoGluon 适配器 ---
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
        pass


class EnhancedModelTrainer:
    """增强版模型训练器"""

    def __init__(self):
        # 统一用 catalog 维护“是否可用 + 缺失原因”，避免 UI 侧难以解释
        self.model_catalog = self._get_model_catalog()
        self.available_models = [m for m, meta in self.model_catalog.items() if meta.get('available', True)]

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
            "AdaBoost", "SVR", "多层感知器",
        ]
        for m in base_models:
            catalog[m] = {"available": True, "reason": ""}

        # --- 可选依赖模型 ---
        catalog["XGBoost"] = {
            "available": bool(XGBOOST_AVAILABLE),
            "reason": "" if XGBOOST_AVAILABLE else "未安装 xgboost（pip install xgboost）",
        }
        catalog["LightGBM"] = {
            "available": bool(LIGHTGBM_AVAILABLE),
            "reason": "" if LIGHTGBM_AVAILABLE else "未安装 lightgbm（pip install lightgbm）",
        }
        catalog["CatBoost"] = {
            "available": bool(CATBOOST_AVAILABLE),
            "reason": "" if CATBOOST_AVAILABLE else "未安装 catboost（pip install catboost）",
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

        # TabPFN / AutoGluon
        catalog["TabPFN"] = {
            "available": bool(TABPFN_AVAILABLE),
            "reason": "" if TABPFN_AVAILABLE else "未安装 tabpfn（pip install tabpfn）",
        }
        catalog["AutoGluon"] = {
            "available": bool(AUTOGLUON_AVAILABLE),
            "reason": "" if AUTOGLUON_AVAILABLE else "未安装 autogluon.tabular（pip install autogluon.tabular）",
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
        # 清理参数中的 random_state（避免重复传递）
        params_clean = {k: v for k, v in params.items() if k != 'random_state'}

        if model_name == "线性回归":
            return LinearRegression()

        elif model_name == "Ridge回归":
            return Ridge(random_state=random_state, **params_clean)

        elif model_name == "Lasso回归":
            return Lasso(random_state=random_state, **params_clean)

        elif model_name == "ElasticNet":
            return ElasticNet(random_state=random_state, **params_clean)

        elif model_name == "决策树":
            return DecisionTreeRegressor(random_state=random_state, **params_clean)

        elif model_name == "随机森林":
            return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params_clean)

        elif model_name == "Extra Trees":
            return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **params_clean)

        elif model_name == "梯度提升树":
            return GradientBoostingRegressor(random_state=random_state, **params_clean)

        elif model_name == "AdaBoost":
            return AdaBoostRegressor(random_state=random_state, **params_clean)

        elif model_name == "SVR":
            return SVR(**params_clean)

        elif model_name == "多层感知器":
            return MLPRegressor(random_state=random_state, max_iter=1000, **params_clean)

        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost 未安装，请运行: pip install xgboost")
            return XGBRegressor(random_state=random_state, n_jobs=-1, **params_clean)

        elif model_name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
            params_clean.setdefault('verbose', -1)
            return LGBMRegressor(random_state=random_state, n_jobs=-1, **params_clean)

        elif model_name == "CatBoost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost 未安装，请运行: pip install catboost")
            params_clean.setdefault('verbose', 0)
            return CatBoostRegressor(random_state=random_state, **params_clean)

        elif model_name == "人工神经网络":
            if not ANN_AVAILABLE:
                raise ImportError("ANN 模块不可用")
            # 训练器内部已用 Pipeline 做缺失填充 + 标准化，避免 ANN 内部重复预处理
            params_clean.setdefault('external_preprocess', True)
            return ANNRegressor(random_state=random_state, **params_clean)

        elif model_name == "TensorFlow Sequential":
            # 训练器内部已用 Pipeline 做缺失填充 + 标准化，避免 TFS 内部重复预处理
            # 若未安装 TensorFlow，训练时给出明确提示（模型仍可在 UI 中选择）
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")
            params_clean.setdefault('external_preprocess', True)
            return TFSequentialRegressor(random_state=random_state, **params_clean)

        elif model_name == "TabPFN":
            if not TABPFN_AVAILABLE:
                raise ImportError("TabPFN 未安装，请运行: pip install tabpfn")
            return TabPFNRegressor(**params_clean)

        elif model_name == "AutoGluon":
            if not AUTOGLUON_AVAILABLE:
                raise ImportError("AutoGluon 未安装")
            return AutoGluonWrapper(**params_clean)

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

        # 4) 预处理：imputer + scaler（先拟合，再训练模型）
        #    这样可以在部分模型中传入 eval_set，获得 per-iter 训练曲线。
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_test_imputed = imputer.transform(X_test_raw)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # 5) 训练模型（对可提供迭代日志的模型，尽量注入 eval_set）
        start_time = time.time()

        fit_kwargs = {}
        try:
            if model_name == "XGBoost" and XGBOOST_AVAILABLE:
                # XGBoost: 支持 eval_set + eval_metric，训练后可读取 evals_result
                fit_kwargs = {
                    "eval_set": [(X_train_scaled, y_train), (X_test_scaled, y_test)],
                    # 同时记录 RMSE/MAE，便于生成多指标训练曲线
                    "eval_metric": ["rmse", "mae"],
                    "verbose": False,
                }
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                fit_kwargs = {
                    "eval_set": [(X_test_scaled, y_test)],
                    "eval_metric": ["rmse", "mae"],
                    "verbose": -1,
                }
            elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
                # CatBoost: eval_set 可提供验证曲线
                fit_kwargs = {
                    "eval_set": (X_test_scaled, y_test),
                    "verbose": False,
                }
        except Exception:
            fit_kwargs = {}

        # 有些模型不接受额外 kwargs，做一次安全回退
        # 对神经网络类模型：把测试集作为 validation_data，便于记录 Test 的 MAE/MSE 曲线
        try:
            if model_name in {"人工神经网络", "TensorFlow Sequential"}:
                setattr(base_model, "validation_data", (X_test_scaled, y_test))
                # TF 模型内部若使用 validation_split，会导致 Test 曲线不是同一批数据；这里优先用外部 validation_data
                if model_name == "TensorFlow Sequential" and hasattr(base_model, "validation_split"):
                    base_model.validation_split = 0.0
        except Exception:
            pass

        try:
            base_model.fit(X_train_scaled, y_train, **(fit_kwargs or {}))
        except TypeError:
            base_model.fit(X_train_scaled, y_train)

        train_time = time.time() - start_time

        model = base_model

        # 6) 组装 Pipeline（不再重新 fit，用于后续 predict 保持一致）
        pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler),
            ('model', model)
        ])

        # 7) 预测与评估（用 pipeline 预测，保证一致）
        y_pred_test = pipeline.predict(X_test_raw)
        y_pred_train = pipeline.predict(X_train_raw)

        # 8) 训练曲线提取（尽量不额外训练）
        training_history = extract_history_from_fitted_model(
            model_name=model_name,
            model=model,
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_test_scaled=X_test_scaled,
            y_test=y_test,
        )

        # 9) holdout-learning-curve（Train size -> Train/Test 指标）
        # - 对 XGBoost/LightGBM/CatBoost：强制用 learning-curve，保证可以输出 R^2 / MAE / MSE 曲线
        # - 对其它“一次性拟合”模型：作为回退（避免 UI 卡死，AutoGluon/TabPFN 默认跳过）
        FORCE_LC = {"XGBoost", "LightGBM", "CatBoost", "多层感知器"}
        EXCLUDE = {"AutoGluon", "TabPFN"}
        if (model_name in FORCE_LC) or (not training_history):
            if model_name not in EXCLUDE:
                try:
                    training_history = build_holdout_learning_curve(
                        make_model=lambda: self._get_model(model_name, **model_params),
                        X_train_raw=X_train_raw,
                        y_train=y_train,
                        X_test_raw=X_test_raw,
                        y_test=y_test,
                        imputer_factory=lambda: SimpleImputer(strategy='median'),
                        scaler_factory=lambda: StandardScaler(),
                        random_state=random_state,
                    )
                except Exception:
                    # 若学习曲线构建失败，则保留原始 history（或空）
                    pass

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
            # 训练曲线/记录
            'training_history': training_history,
            'training_history_df': history_to_frame(training_history) if training_history else pd.DataFrame(),
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

        # 根据 splitter 类型选择正确的 split 调用方式
        if isinstance(splitter, GroupKFold):
            split_iter = splitter.split(X_arr, y_arr, groups)
        elif y_bins is not None:
            split_iter = splitter.split(X_arr, y_bins)
        else:
            split_iter = splitter.split(X_arr, y_arr)

        for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
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
