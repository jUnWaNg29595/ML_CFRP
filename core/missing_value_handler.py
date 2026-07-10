# -*- coding: utf-8 -*-
"""Missing-value utilities for tabular neural models."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.impute import SimpleImputer

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    ITERATIVE_IMPUTER_AVAILABLE = True
except Exception:
    IterativeImputer = None
    BayesianRidge = None
    ITERATIVE_IMPUTER_AVAILABLE = False


def build_missing_mask(X) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    return (~np.isfinite(arr)).astype(np.float32)


class MissingValueHandler:
    """Wrapper around median / Bayesian / multiple Bayesian imputation."""

    def __init__(
        self,
        strategy: str = "median",
        random_state: int = 42,
        max_iter: int = 15,
        n_imputations: int = 5,
    ):
        self.strategy = str(strategy or "median").strip().lower()
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.n_imputations = max(1, int(n_imputations))
        self.strategy_used_ = None
        self.imputer_ = None
        self.imputers_ = None
        self.empty_feature_mask_ = None

    @staticmethod
    def is_iterative_available() -> bool:
        return bool(ITERATIVE_IMPUTER_AVAILABLE)

    def _sanitize(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    def _make_iterative_imputer(self, seed: int, sample_posterior: bool):
        if not ITERATIVE_IMPUTER_AVAILABLE or IterativeImputer is None or BayesianRidge is None:
            raise ImportError("IterativeImputer with BayesianRidge is unavailable")
        return IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max(5, int(self.max_iter)),
            random_state=int(seed),
            sample_posterior=bool(sample_posterior),
            initial_strategy="median",
            imputation_order="ascending",
            skip_complete=True,
        )

    def _fit_single(self, X: np.ndarray, seed: int, sample_posterior: bool) -> np.ndarray:
        imputer = self._make_iterative_imputer(seed=seed, sample_posterior=sample_posterior)
        X_out = imputer.fit_transform(X)
        return np.asarray(X_out, dtype=np.float64), imputer

    @staticmethod
    def _finalize(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _split_empty_features(self, X: np.ndarray, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("MissingValueHandler expects a 2D feature matrix")
        if fit:
            self.empty_feature_mask_ = np.all(np.isnan(X), axis=0)
        if self.empty_feature_mask_ is None:
            raise ValueError("Empty-feature mask is not fitted")
        mask = np.asarray(self.empty_feature_mask_, dtype=bool)
        if mask.shape[0] != X.shape[1]:
            raise ValueError(
                f"Feature count mismatch: fitted on {mask.shape[0]} columns, got {X.shape[1]} columns"
            )
        return X[:, ~mask], mask

    @staticmethod
    def _restore_empty_features(X_core: np.ndarray, empty_mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        empty_mask = np.asarray(empty_mask, dtype=bool)
        n_rows = int(X_core.shape[0]) if X_core.ndim >= 1 else 0
        restored = np.full((n_rows, empty_mask.shape[0]), float(fill_value), dtype=np.float64)
        if X_core.size > 0:
            restored[:, ~empty_mask] = np.asarray(X_core, dtype=np.float64)
        return restored

    def fit_transform(self, X) -> np.ndarray:
        X_in = self._sanitize(X)
        X_core, empty_mask = self._split_empty_features(X_in, fit=True)
        strategy = self.strategy

        if X_core.shape[1] == 0:
            self.imputer_ = None
            self.imputers_ = None
            self.strategy_used_ = f"{strategy}_empty_only"
            return self._finalize(self._restore_empty_features(X_core, empty_mask))

        if strategy == "median":
            self.imputer_ = SimpleImputer(strategy="median")
            self.strategy_used_ = "median"
            X_out = self.imputer_.fit_transform(X_core)
            return self._finalize(self._restore_empty_features(X_out, empty_mask))

        if strategy == "bayesian":
            if not ITERATIVE_IMPUTER_AVAILABLE:
                warnings.warn(
                    "IterativeImputer is unavailable; falling back to median imputation.",
                    RuntimeWarning,
                )
                self.imputer_ = SimpleImputer(strategy="median")
                self.strategy_used_ = "median"
                X_out = self.imputer_.fit_transform(X_core)
                return self._finalize(self._restore_empty_features(X_out, empty_mask))
            start_time = time.time()
            print(
                f"[MissingValueHandler] Bayesian imputation started: "
                f"samples={X_core.shape[0]}, features={X_core.shape[1]}, max_iter={self.max_iter}"
            )
            X_out, imputer = self._fit_single(X_core, seed=self.random_state, sample_posterior=False)
            self.imputer_ = imputer
            self.strategy_used_ = "bayesian"
            print(
                f"[MissingValueHandler] Bayesian imputation finished in "
                f"{time.time() - start_time:.2f}s"
            )
            return self._finalize(self._restore_empty_features(X_out, empty_mask))

        if strategy == "multiple_bayesian":
            if not ITERATIVE_IMPUTER_AVAILABLE:
                warnings.warn(
                    "IterativeImputer is unavailable; falling back to median imputation.",
                    RuntimeWarning,
                )
                self.imputer_ = SimpleImputer(strategy="median")
                self.strategy_used_ = "median"
                X_out = self.imputer_.fit_transform(X_core)
                return self._finalize(self._restore_empty_features(X_out, empty_mask))

            self.imputers_ = []
            imputations = []
            total_rounds = max(2, self.n_imputations)
            start_time = time.time()
            print(
                f"[MissingValueHandler] Multiple Bayesian imputation started: "
                f"samples={X_core.shape[0]}, features={X_core.shape[1]}, "
                f"max_iter={self.max_iter}, rounds={total_rounds}"
            )
            for idx in range(total_rounds):
                round_start = time.time()
                print(
                    f"[MissingValueHandler] Multiple Bayesian round "
                    f"{idx + 1}/{total_rounds} started"
                )
                X_out, imputer = self._fit_single(
                    X_core,
                    seed=self.random_state + idx,
                    sample_posterior=True,
                )
                self.imputers_.append(imputer)
                imputations.append(X_out)
                print(
                    f"[MissingValueHandler] Multiple Bayesian round "
                    f"{idx + 1}/{total_rounds} finished in {time.time() - round_start:.2f}s"
                )
            self.strategy_used_ = "multiple_bayesian"
            X_out = np.mean(np.stack(imputations, axis=0), axis=0)
            print(
                f"[MissingValueHandler] Multiple Bayesian imputation finished in "
                f"{time.time() - start_time:.2f}s"
            )
            return self._finalize(self._restore_empty_features(X_out, empty_mask))

        raise ValueError(f"Unsupported missing-value strategy: {self.strategy}")

    def transform(self, X) -> np.ndarray:
        if self.strategy_used_ is None:
            raise ValueError("MissingValueHandler is not fitted")

        X_in = self._sanitize(X)
        X_core, empty_mask = self._split_empty_features(X_in, fit=False)
        if X_core.shape[1] == 0:
            return self._finalize(self._restore_empty_features(X_core, empty_mask))
        if self.strategy_used_ == "multiple_bayesian":
            if not self.imputers_:
                raise ValueError("Multiple imputers are not fitted")
            imputations = [imp.transform(X_core) for imp in self.imputers_]
            X_out = np.mean(np.stack(imputations, axis=0), axis=0)
            return self._finalize(self._restore_empty_features(X_out, empty_mask))

        if self.imputer_ is None:
            raise ValueError("Imputer is not fitted")
        X_out = self.imputer_.transform(X_core)
        return self._finalize(self._restore_empty_features(X_out, empty_mask))
