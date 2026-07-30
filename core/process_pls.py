"""Leakage-safe process-feature PLS utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted


PROCESS_PLS_SCHEMA_VERSION = 1


def _coerce_numeric_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("process PLS input must be a 2D table")
        frame = pd.DataFrame(array, columns=[f"feat_{idx}" for idx in range(array.shape[1])])

    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].replace({
                "True": 1,
                "true": 1,
                "TRUE": 1,
                "False": 0,
                "false": 0,
                "FALSE": 0,
            })
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _coerce_finite_target(y: Any) -> np.ndarray:
    y_array = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy(dtype=float)
    if len(y_array) == 0 or not np.isfinite(y_array).all():
        raise ValueError("process PLS target contains missing or non-finite values")
    return y_array


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def fingerprint_process_pls_workflow(payload: Mapping[str, Any]) -> str:
    """Create an order-sensitive workflow fingerprint."""
    normalized = {
        key: value
        for key, value in dict(payload or {}).items()
        if key != "workflow_hash"
    }
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_vip_scores(pls_model: Any) -> np.ndarray:
    """Compute Wold VIP scores for a fitted PLS model."""
    weights = np.asarray(pls_model.x_weights_, dtype=float)
    scores = np.asarray(pls_model.x_scores_, dtype=float)
    y_loadings = np.asarray(pls_model.y_loadings_, dtype=float)

    if weights.ndim != 2 or scores.ndim != 2 or y_loadings.ndim != 2:
        raise ValueError("invalid PLS model arrays for VIP computation")

    n_features = weights.shape[0]
    if n_features == 0:
        return np.asarray([], dtype=float)

    explained = np.sum(scores ** 2, axis=0) * np.sum(y_loadings ** 2, axis=0)
    total_explained = float(np.sum(explained))
    if not np.isfinite(total_explained) or total_explained <= 0:
        return np.zeros(n_features, dtype=float)

    weight_norm = np.sum(weights ** 2, axis=0)
    weight_norm = np.where(weight_norm == 0, np.nan, weight_norm)
    weighted = (weights ** 2) / weight_norm
    vip = np.sqrt(n_features * np.nansum(weighted * explained, axis=1) / total_explained)
    return np.nan_to_num(vip, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_normalize(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=float)
    clean = array.copy()
    fallback = np.nanmedian(clean[finite])
    clean[~finite] = fallback
    minimum = float(np.min(clean))
    maximum = float(np.max(clean))
    if np.isclose(maximum, minimum):
        return np.ones_like(clean, dtype=float)
    normalized = (clean - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1.0 - normalized


def _finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _finite_std(values: Sequence[float], default: float = 0.0) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size <= 1:
        return float(default)
    return float(np.std(finite, ddof=1))


def select_pls_components_cv(
    X: Any,
    y: Any,
    max_components: int,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[int, dict[str, Any]]:
    """Select PLS components using training data only."""
    X_array = np.asarray(X, dtype=float)
    y_array = _coerce_finite_target(y)
    if X_array.ndim != 2:
        raise ValueError("process PLS component selection requires a 2D matrix")
    if len(y_array) != X_array.shape[0]:
        raise ValueError("process PLS X and y row counts do not match")

    n_samples, n_features = X_array.shape
    if n_samples < 2 or n_features < 1:
        raise ValueError("not enough samples or features for process PLS")
    max_candidate = int(min(max(1, max_components), n_features, max(1, n_samples - 1)))
    if max_candidate < 1:
        raise ValueError("not enough samples or features for process PLS")

    n_splits = int(min(max(2, cv_splits), n_samples))
    if n_splits < 2:
        n_splits = 2
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_array, np.repeat(np.mean(y_array), len(y_array)))))

    rows: list[dict[str, Any]] = []
    for n_components in range(1, max_candidate + 1):
        fold_r2: list[float] = []
        fold_rmse: list[float] = []
        for train_idx, valid_idx in splitter.split(X_array, y_array):
            fold_components = int(min(n_components, len(train_idx) - 1, n_features))
            if fold_components < 1:
                continue
            model = PLSRegression(n_components=fold_components)
            model.fit(X_array[train_idx], y_array[train_idx])
            pred = model.predict(X_array[valid_idx]).ravel()
            fold_r2.append(
                float(r2_score(y_array[valid_idx], pred))
                if len(valid_idx) >= 2
                else np.nan
            )
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_array[valid_idx], pred))))
        if not fold_rmse:
            continue
        rmse_mean = _finite_mean(fold_rmse)
        rows.append({
            "n_components": int(n_components),
            "cv_r2_mean": _finite_mean(fold_r2),
            "cv_r2_std": _finite_std(fold_r2),
            "cv_rmse_mean": rmse_mean,
            "cv_rmse_std": _finite_std(fold_rmse),
            "rmse_improvement": float(baseline_rmse - rmse_mean),
        })

    if not rows:
        raise ValueError("process PLS component selection failed for all candidates")

    r2_norm = _safe_normalize([row["cv_r2_mean"] for row in rows], higher_is_better=True)
    rmse_norm = _safe_normalize([row["rmse_improvement"] for row in rows], higher_is_better=True)
    stability_norm = _safe_normalize([row["cv_rmse_std"] for row in rows], higher_is_better=False)
    complexity = np.asarray([1.0 - (row["n_components"] / max_candidate) for row in rows], dtype=float)
    scores = 0.45 * r2_norm + 0.30 * rmse_norm + 0.15 * stability_norm + 0.10 * complexity
    for row, score in zip(rows, scores):
        row["selection_score"] = float(score)

    best_idx = int(np.argmax(scores))
    return int(rows[best_idx]["n_components"]), {
        "candidates": rows,
        "selected_n_components": int(rows[best_idx]["n_components"]),
        "baseline_rmse": float(baseline_rmse),
    }


def process_pls_config_to_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config or {})
    payload.setdefault("schema_version", PROCESS_PLS_SCHEMA_VERSION)
    payload["workflow_hash"] = fingerprint_process_pls_workflow(payload)
    return payload


class ProcessPLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        process_feature_cols,
        max_components=8,
        vip_top_k=8,
        missing_threshold=0.85,
        random_state=42,
        cv_splits=5,
    ):
        self.process_feature_cols = process_feature_cols
        self.max_components = max_components
        self.vip_top_k = vip_top_k
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        self.cv_splits = cv_splits

    def fit(self, X, y):
        frame = _coerce_numeric_frame(X)
        self.input_feature_cols_ = frame.columns.tolist()
        self.feature_names_in_ = np.asarray(self.input_feature_cols_, dtype=object)
        self.n_features_in_ = len(self.input_feature_cols_)
        self.process_feature_cols_ = list(self.process_feature_cols or [])
        missing = [column for column in self.process_feature_cols_ if column not in frame.columns]
        if missing:
            raise ValueError(
                f"missing required process columns: {', '.join(missing[:12])}"
            )

        self.kept_process_feature_cols_ = [
            column
            for column in self.process_feature_cols_
            if frame[column].notna().mean() >= 1.0 - float(self.missing_threshold)
        ]
        if not self.kept_process_feature_cols_:
            raise ValueError("no process feature remains after missingness filtering")

        y_array = _coerce_finite_target(y)
        if len(y_array) != len(frame):
            raise ValueError("process PLS X and y row counts do not match")

        process_frame = frame[self.kept_process_feature_cols_].copy()
        self.missing_mask_cols_ = [
            f"{column}__missing" for column in self.kept_process_feature_cols_
        ]
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(process_frame)
        self.scaler_ = RobustScaler()
        scaled = self.scaler_.fit_transform(imputed)
        self.n_components_, self.cv_report_ = select_pls_components_cv(
            scaled,
            y_array,
            max_components=int(self.max_components),
            cv_splits=int(self.cv_splits),
            random_state=int(self.random_state),
        )
        self.pls_ = PLSRegression(n_components=self.n_components_)
        self.pls_.fit(scaled, y_array)
        self.vip_scores_ = compute_vip_scores(self.pls_)
        order = np.argsort(-self.vip_scores_)
        self.selected_original_features_ = [
            self.kept_process_feature_cols_[index]
            for index in order[: min(self.vip_top_k, len(order))]
        ]
        passthrough_cols = [
            column
            for column in self.input_feature_cols_
            if column not in self.kept_process_feature_cols_
        ]
        self.output_feature_names_ = (
            [f"process_pls_{index + 1}" for index in range(self.n_components_)]
            + self.selected_original_features_
            + self.missing_mask_cols_
            + passthrough_cols
        )
        self.workflow_hash_ = fingerprint_process_pls_workflow(self.to_workflow_dict())
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ["input_feature_cols_", "imputer_", "scaler_", "pls_", "output_feature_names_"],
        )
        frame = _coerce_numeric_frame(X)
        missing = [column for column in self.process_feature_cols_ if column not in frame.columns]
        if missing:
            raise ValueError(
                f"missing required process columns: {', '.join(missing[:12])}"
            )
        frame = frame.reindex(columns=self.input_feature_cols_)
        process_frame = frame[self.kept_process_feature_cols_]
        masks = process_frame.isna().astype(float)
        imputed = self.imputer_.transform(process_frame)
        imputed_df = pd.DataFrame(
            imputed,
            index=frame.index,
            columns=self.kept_process_feature_cols_,
        )
        scaled = self.scaler_.transform(imputed)
        components = self.pls_.transform(scaled)
        output = pd.DataFrame(
            components,
            index=frame.index,
            columns=[f"process_pls_{index + 1}" for index in range(self.n_components_)],
        )
        passthrough_cols = [
            column
            for column in self.input_feature_cols_
            if column not in self.kept_process_feature_cols_
        ]
        output = pd.concat(
            [
                output.reset_index(drop=True),
                imputed_df[self.selected_original_features_].reset_index(drop=True),
                masks.set_axis(self.missing_mask_cols_, axis=1).reset_index(drop=True),
                frame[passthrough_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        return output.reindex(columns=self.output_feature_names_).replace(
            [np.inf, -np.inf], np.nan
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["output_feature_names_"])
        return np.asarray(self.output_feature_names_, dtype=object)

    def to_workflow_dict(self):
        return {
            "schema_version": PROCESS_PLS_SCHEMA_VERSION,
            "process_feature_cols": list(self.process_feature_cols_),
            "kept_process_feature_cols": list(self.kept_process_feature_cols_),
            "selected_original_features": list(self.selected_original_features_),
            "missing_mask_cols": list(self.missing_mask_cols_),
            "output_feature_names": list(self.output_feature_names_),
            "n_components": int(self.n_components_),
            "max_components": int(self.max_components),
            "vip_top_k": int(self.vip_top_k),
            "missing_threshold": float(self.missing_threshold),
            "random_state": int(self.random_state),
            "workflow_hash": getattr(self, "workflow_hash_", None),
        }
