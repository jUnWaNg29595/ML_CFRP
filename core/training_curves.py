# -*- coding: utf-8 -*-
"""训练曲线：提取 + 绘制

目标：
- 尽可能为“所有模型训练”提供可视化曲线：
  - 迭代/epoch/boosting-iteration：直接读取模型训练历史
  - 非迭代一次性拟合模型：回退到 holdout-learning-curve（训练集增量 -> 测试集得分）

约定：
history 是一个 dict，其中包含：
- kind: str  ("iter" | "learning_curve" | "single")
- step: list[int] 或 train_size: list[int]
- 其它键：loss/val_loss/r2/train_r2/test_r2/rmse/...（按可用性填充）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _to_1d(a) -> np.ndarray:
    return np.asarray(a).ravel()


def extract_history_from_fitted_model(
    model_name: str,
    model: Any,
    X_train_scaled: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_test_scaled: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """从已拟合模型中尽量提取训练历史（不做额外拟合）。"""
    hist: Dict[str, Any] = {}

    # 0) 若模型已经显式记录 train/test MAE/MSE 曲线
    for train_mse_attr, test_mse_attr, train_mae_attr, test_mae_attr in (
        ("train_mse_curve", "test_mse_curve", "train_mae_curve", "test_mae_curve"),
        ("train_mse", "test_mse", "train_mae", "test_mae"),
    ):
        if hasattr(model, train_mse_attr):
            try:
                train_mse = list(getattr(model, train_mse_attr) or [])
                if len(train_mse) > 0:
                    steps = list(range(1, len(train_mse) + 1))
                    hist = {"kind": "iter", "step": steps, "train_mse": train_mse}
                    if hasattr(model, test_mse_attr):
                        v = list(getattr(model, test_mse_attr) or [])
                        if len(v) == len(train_mse):
                            hist["test_mse"] = v
                    if hasattr(model, train_mae_attr):
                        v = list(getattr(model, train_mae_attr) or [])
                        if len(v) == len(train_mse):
                            hist["train_mae"] = v
                    if hasattr(model, test_mae_attr):
                        v = list(getattr(model, test_mae_attr) or [])
                        if len(v) == len(train_mse):
                            hist["test_mae"] = v
                    return hist
            except Exception:
                pass

    # 1) PyTorch ANNRegressor: train_losses（loss= MSE）
    if hasattr(model, "train_losses") and isinstance(getattr(model, "train_losses"), (list, tuple)):
        losses = list(getattr(model, "train_losses"))
        if len(losses) > 0:
            hist = {
                "kind": "iter",
                "step": list(range(1, len(losses) + 1)),
                "train_mse": losses,
            }
            return hist

    # 2) TFSequentialRegressor: get_training_history
    if hasattr(model, "get_training_history"):
        try:
            th = model.get_training_history()
            if th and len(th.get("loss", [])) > 0:
                hist = {"kind": "iter", "step": list(range(1, len(th.get("loss", [])) + 1)), **th}
                return hist
        except Exception:
            pass

    # 3) sklearn MLPRegressor: loss_curve_
    if hasattr(model, "loss_curve_"):
        try:
            lc = list(getattr(model, "loss_curve_"))
            if len(lc) > 0:
                hist = {
                    "kind": "iter",
                    "step": list(range(1, len(lc) + 1)),
                    "loss": lc,
                }
                return hist
        except Exception:
            pass

    # 4) GradientBoostingRegressor: train_score_ / staged_predict
    if hasattr(model, "staged_predict") and callable(getattr(model, "staged_predict")):
        # staged_predict 适用于 GBDT / AdaBoost 等
        try:
            if X_train_scaled is not None and y_train is not None:
                y_train = _to_1d(y_train)
            if X_test_scaled is not None and y_test is not None:
                y_test = _to_1d(y_test)

            train_r2, test_r2 = [], []
            train_rmse, test_rmse = [], []
            train_mae, test_mae = [], []
            train_mse, test_mse = [], []
            steps = []

            if X_train_scaled is not None and y_train is not None and X_test_scaled is not None and y_test is not None:
                for i, (p_tr, p_te) in enumerate(zip(model.staged_predict(X_train_scaled), model.staged_predict(X_test_scaled)), start=1):
                    p_tr = _to_1d(p_tr)
                    p_te = _to_1d(p_te)
                    steps.append(i)
                    train_r2.append(float(r2_score(y_train, p_tr)))
                    test_r2.append(float(r2_score(y_test, p_te)))
                    train_mae.append(float(mean_absolute_error(y_train, p_tr)))
                    test_mae.append(float(mean_absolute_error(y_test, p_te)))
                    train_mse.append(float(mean_squared_error(y_train, p_tr)))
                    test_mse.append(float(mean_squared_error(y_test, p_te)))
                    train_rmse.append(float(np.sqrt(mean_squared_error(y_train, p_tr))))
                    test_rmse.append(float(np.sqrt(mean_squared_error(y_test, p_te))))

                if len(steps) > 1:
                    hist = {
                        "kind": "iter",
                        "step": steps,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                    }
                    return hist
        except Exception:
            pass

    # 5) XGBoost / LightGBM: evals_result_
    # XGBoost sklearn wrapper: evals_result() or evals_result_
    for getter in ("evals_result", "evals_result_"):
        if hasattr(model, getter):
            try:
                er = getattr(model, getter)
                er = er() if callable(er) else er
                if isinstance(er, dict) and len(er) > 0:
                    # 结构一般：
                    # - XGBoost: {'validation_0': {'rmse':[...], 'mae':[...]} , 'validation_1': {...}}
                    # - LightGBM: {'training': {...}, 'valid_1': {...}}
                    def _split_to_prefix(s: str) -> Optional[str]:
                        sl = str(s).lower()
                        if sl in {"validation_0", "train", "training", "learn"}:
                            return "train"
                        if sl in {"validation_1", "validation", "valid", "valid_0", "valid_1", "test", "eval"}:
                            return "test"
                        return None

                    def _metric_to_key(prefix: str, mname: str) -> str:
                        ml = str(mname).lower()
                        if ml in {"rmse", "l2_root"}:
                            return f"{prefix}_rmse"
                        if ml in {"mae", "l1"}:
                            return f"{prefix}_mae"
                        if ml in {"mse", "l2"}:
                            return f"{prefix}_mse"
                        if ml in {"r2", "rsquared"}:
                            return f"{prefix}_r2"
                        return f"{prefix}_{ml}"

                    metrics: Dict[str, List[float]] = {}
                    for split_name, metric_dict in er.items():
                        if not isinstance(metric_dict, dict):
                            continue
                        prefix = _split_to_prefix(split_name)
                        # 未识别的 split 保留原名（不会影响主曲线，但可用于导出）
                        for mname, values in metric_dict.items():
                            try:
                                values_list = list(values)
                            except Exception:
                                continue
                            if prefix:
                                key = _metric_to_key(prefix, mname)
                            else:
                                key = f"{split_name}_{mname}"
                            metrics[key] = values_list

                    if metrics:
                        steps = list(range(1, len(next(iter(metrics.values()))) + 1))
                        hist = {"kind": "iter", "step": steps, **metrics}
                        return hist
            except Exception:
                pass

    # CatBoost: get_evals_result()
    if hasattr(model, "get_evals_result"):
        try:
            er = model.get_evals_result()
            if isinstance(er, dict) and len(er) > 0:
                # 结构：{'learn': {'RMSE':[...]}, 'validation': {'RMSE':[...]}}
                def _split_to_prefix(s: str) -> Optional[str]:
                    sl = str(s).lower()
                    if sl == "learn":
                        return "train"
                    if sl in {"validation", "valid", "test"}:
                        return "test"
                    return None

                def _metric_to_key(prefix: str, mname: str) -> str:
                    ml = str(mname).lower()
                    if ml == "rmse":
                        return f"{prefix}_rmse"
                    if ml == "mae":
                        return f"{prefix}_mae"
                    if ml == "mse":
                        return f"{prefix}_mse"
                    if ml in {"r2", "rsquared"}:
                        return f"{prefix}_r2"
                    return f"{prefix}_{ml}"

                metrics: Dict[str, List[float]] = {}
                for split_name, metric_dict in er.items():
                    if not isinstance(metric_dict, dict):
                        continue
                    prefix = _split_to_prefix(split_name)
                    for mname, values in metric_dict.items():
                        try:
                            values_list = list(values)
                        except Exception:
                            continue
                        if prefix:
                            key = _metric_to_key(prefix, mname)
                        else:
                            key = f"{split_name}_{mname}"
                        metrics[key] = values_list
                if metrics:
                    steps = list(range(1, len(next(iter(metrics.values()))) + 1))
                    hist = {"kind": "iter", "step": steps, **metrics}
                    return hist
        except Exception:
            pass

    return {}


def build_holdout_learning_curve(
    make_model: Callable[[], Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    imputer_factory: Callable[[], Any],
    scaler_factory: Callable[[], Any],
    fractions: Optional[List[float]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """对一次性拟合模型：用训练集增量构造学习曲线（holdout）。

不使用 CV（更快），但能给出“数据量 -> 泛化”趋势。

说明：
- x 轴“取样点”来自不同的 train_size；越密集意味着需要更多次“重新训练并评估”，
  运行时间也会线性增加。
"""
    rng = np.random.RandomState(int(random_state))

    X_train_raw = np.asarray(X_train_raw)
    y_train = _to_1d(y_train)
    X_test_raw = np.asarray(X_test_raw)
    y_test = _to_1d(y_test)

    n = len(y_train)
    if n < 5:
        return {
            "kind": "single",
            "train_size": [n],
            "test_r2": [float("nan")],
        }

    # 默认用更密集的 train_size 取样点（但做上限，避免过慢）
    # - n≈1000 时大约 14 个点（0.2~1.0 等距）
    # - n 较小时仍保证至少 8 个点
    if fractions is None:
        n_points = int(np.clip(int(round(n / 80)), 8, 20))
        fractions = np.linspace(0.2, 1.0, n_points).tolist()

    fractions = [float(f) for f in fractions if 0 < f <= 1.0]

    # 将 fractions 映射到“唯一且递增”的 train_size，避免 round 后重复点
    sizes = sorted({min(n, max(2, int(round(n * f)))) for f in fractions})
    if sizes[-1] != n:
        sizes.append(n)

    indices = np.arange(n)
    rng.shuffle(indices)

    train_sizes: List[int] = []
    train_r2: List[float] = []
    test_r2: List[float] = []
    train_mae: List[float] = []
    test_mae: List[float] = []
    train_mse: List[float] = []
    test_mse: List[float] = []

    for k in sizes:
        sub_idx = indices[:k]
        X_sub = X_train_raw[sub_idx]
        y_sub = y_train[sub_idx]

        imputer = imputer_factory()
        scaler = scaler_factory()

        X_sub_s = scaler.fit_transform(imputer.fit_transform(X_sub))
        # 对训练子集自身预测（与 fit 时同一套变换）
        X_tr_full_s = X_sub_s
        X_te_s = scaler.transform(imputer.transform(X_test_raw))

        m = make_model()
        m.fit(X_sub_s, y_sub)

        p_tr = _to_1d(m.predict(X_tr_full_s))
        p_te = _to_1d(m.predict(X_te_s))

        train_sizes.append(int(k))
        train_r2.append(float(r2_score(y_sub, p_tr)))
        test_r2.append(float(r2_score(y_test, p_te)))
        train_mae.append(float(mean_absolute_error(y_sub, p_tr)))
        test_mae.append(float(mean_absolute_error(y_test, p_te)))
        train_mse.append(float(mean_squared_error(y_sub, p_tr)))
        test_mse.append(float(mean_squared_error(y_test, p_te)))

    return {
        "kind": "learning_curve",
        "train_size": train_sizes,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }


def history_to_frame(history: Dict[str, Any]) -> pd.DataFrame:
    """history dict -> DataFrame（用于表格/导出）"""
    if not history:
        return pd.DataFrame()

    # 优先 step，其次 train_size
    x_key = "step" if "step" in history else ("train_size" if "train_size" in history else None)
    if x_key is None:
        return pd.DataFrame(history)

    df = pd.DataFrame({x_key: history.get(x_key, [])})
    for k, v in history.items():
        if k in ("kind", x_key):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == len(df):
                df[k] = list(v)
    return df


def plot_history(history: Dict[str, Any], title: str = "Training Curves"):
    """绘制训练曲线，返回 (fig, export_df)。"""
    from .plot_style import apply_global_style, style_axes, TRAIN_COLOR, TEST_COLOR

    apply_global_style()

    import matplotlib.pyplot as plt

    if not history:
        fig, ax = plt.subplots(figsize=(7, 4))
        style_axes(ax, title=title)
        # 避免服务器环境缺少中文字体导致方块
        ax.text(0.5, 0.5, "No training curve available", ha="center", va="center")
        plt.tight_layout()
        return fig, pd.DataFrame()

    kind = history.get("kind", "")
    x_key = "step" if "step" in history else ("train_size" if "train_size" in history else "step")
    x = history.get(x_key, [])

    # ---- 统一：优先输出 R^2/MAE/MSE（三张子图），否则回退为“把所有可画的 key 画出来” ----
    def _get_pair(train_keys: Tuple[str, ...], test_keys: Tuple[str, ...]):
        tr = next((k for k in train_keys if k in history), None)
        te = next((k for k in test_keys if k in history), None)
        return tr, te

    # 兼容 Keras: loss/val_loss 作为 MSE
    if "train_mse" not in history and "loss" in history:
        history = dict(history)
        history["train_mse"] = history.get("loss")
    if "test_mse" not in history and "val_loss" in history:
        history = dict(history)
        history["test_mse"] = history.get("val_loss")
    if "train_mse" not in history and "mse" in history:
        history = dict(history)
        history["train_mse"] = history.get("mse")
    if "test_mse" not in history and "val_mse" in history:
        history = dict(history)
        history["test_mse"] = history.get("val_mse")
    # 若只有 RMSE，推导 MSE
    if "train_mse" not in history and "train_rmse" in history:
        history = dict(history)
        history["train_mse"] = [float(v) ** 2 for v in history.get("train_rmse", [])]
    if "test_mse" not in history and "test_rmse" in history:
        history = dict(history)
        history["test_mse"] = [float(v) ** 2 for v in history.get("test_rmse", [])]

    r2_tr, r2_te = _get_pair(("train_r2", "r2"), ("test_r2", "val_r2"))
    mae_tr, mae_te = _get_pair(("train_mae", "mae"), ("test_mae", "val_mae"))
    mse_tr, mse_te = _get_pair(("train_mse",), ("test_mse",))

    metric_specs = []
    if r2_tr or r2_te:
        metric_specs.append((r2_tr, r2_te, r"$R^2$"))
    if mae_tr or mae_te:
        metric_specs.append((mae_tr, mae_te, "MAE"))
    if mse_tr or mse_te:
        metric_specs.append((mse_tr, mse_te, "MSE"))

    if metric_specs:
        def _fmt(v: float) -> str:
            # 和系统中 parity plot 保持一致（默认三位小数）
            try:
                return f"{float(v):.3f}"
            except Exception:
                return "nan"

        nrows = len(metric_specs)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7, 3.2 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        fig.suptitle(title, y=0.985, fontweight="bold")

        # x 轴标签：learning-curve 固定为 Train size，否则为 Epoch/Iter
        x_label = "Train size" if kind == "learning_curve" else ("Epoch/Iter" if x_key == "step" else "Train size")

        for ax, (k_tr, k_te, ylab) in zip(axes, metric_specs):
            # 统一图例：Train/Test + 最后一个点的数值
            if ylab == r"$R^2$":
                tr_prefix = r"Train ($R^2$="
                te_prefix = r"Test ($R^2$="
                suffix = ")"
            else:
                tr_prefix = f"Train ({ylab}="
                te_prefix = f"Test ({ylab}="
                suffix = ")"

            if k_tr and isinstance(history.get(k_tr), (list, tuple, np.ndarray)) and len(history.get(k_tr)) == len(x):
                y_tr = list(history[k_tr])
                label_tr = f"{tr_prefix}{_fmt(y_tr[-1])}{suffix}"
                ax.plot(
                    x,
                    y_tr,
                    label=label_tr,
                    color=TRAIN_COLOR,
                    marker="o",
                    markersize=4,
                    alpha=0.9,
                )
            if k_te and isinstance(history.get(k_te), (list, tuple, np.ndarray)) and len(history.get(k_te)) == len(x):
                y_te = list(history[k_te])
                label_te = f"{te_prefix}{_fmt(y_te[-1])}{suffix}"
                ax.plot(
                    x,
                    y_te,
                    label=label_te,
                    color=TEST_COLOR,
                    marker="d",
                    markersize=4,
                    alpha=0.9,
                )

            style_axes(ax, title=None, xlabel=None, ylabel=ylab)
            ax.legend(loc="best", frameon=False)

        axes[-1].set_xlabel(x_label)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, history_to_frame(history)

    # 回退：把所有可画的 key 画在一张图上
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for k, v in history.items():
        if k in ("kind", x_key):
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == len(x):
            ax.plot(x, v, label=k)
            plotted += 1

    xlabel = "Epoch/Iter" if x_key == "step" else "Train size"
    style_axes(ax, title=title, xlabel=xlabel, ylabel="Metric")
    if plotted > 0:
        ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    return fig, history_to_frame(history)
