# -*- coding: utf-8 -*-
"""Isolated XGBoost surface prediction worker to protect the Streamlit process."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import traceback

import numpy as np
import pandas as pd


def _prepare_xgboost_model(model) -> None:
    try:
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
    except Exception:
        pass

    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            booster.set_param("nthread", 1)
    except Exception:
        pass


def _safe_model_predict(model, X):
    try:
        return model.predict(X)
    except Exception as e:
        params = getattr(model, "get_xgb_params", lambda: {})()
        eval_metric = params.get("eval_metric") if isinstance(params, dict) else None
        if "Unknown metric function" not in str(e) or not isinstance(eval_metric, list):
            raise
        model.set_params(eval_metric="rmse")
        return model.predict(X)


def _predict_xgboost_chunk(model, X_chunk: np.ndarray) -> np.ndarray:
    X_chunk = np.ascontiguousarray(X_chunk, dtype=np.float32)

    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            preds = booster.inplace_predict(X_chunk, validate_features=False)
            return np.asarray(preds, dtype=np.float64).ravel()
        except Exception:
            pass

    preds = _safe_model_predict(model, X_chunk)
    return np.asarray(preds, dtype=np.float64).ravel()


def _predict_grid(model, grid_df, feature_cols, imputer=None, scaler=None, batch_size: int = 2048):
    preds = []
    total = len(grid_df)
    batch_size = max(128, int(batch_size))

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch_df = grid_df.iloc[start:stop].loc[:, feature_cols].copy()
        for col in batch_df.columns:
            batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce")

        X_batch = batch_df.to_numpy(dtype=np.float64, copy=True)
        if imputer is not None:
            X_batch = imputer.transform(X_batch)
        if scaler is not None:
            X_batch = scaler.transform(X_batch)

        preds.append(_predict_xgboost_chunk(model, X_batch))

    if not preds:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(preds, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XGBoost surface prediction in a subprocess")
    parser.add_argument("--job", required=True, help="Path to the pickled job payload")
    args = parser.parse_args()

    with open(args.job, "rb") as f:
        payload = pickle.load(f)

    result = {
        "ok": False,
        "predictions_path": payload.get("predictions_path"),
    }

    try:
        model = payload["model"]
        grid_df = payload["grid_df"]
        feature_cols = list(payload["feature_cols"])
        imputer = payload.get("imputer")
        scaler = payload.get("scaler")
        batch_size = int(payload.get("batch_size", 2048))

        _prepare_xgboost_model(model)
        preds = _predict_grid(
            model=model,
            grid_df=grid_df,
            feature_cols=feature_cols,
            imputer=imputer,
            scaler=scaler,
            batch_size=batch_size,
        )

        np.save(payload["predictions_path"], np.asarray(preds, dtype=np.float64))
        result["ok"] = True
        result["n_predictions"] = int(len(preds))
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        gc.collect()

    os.makedirs(os.path.dirname(payload["result_path"]), exist_ok=True)
    with open(payload["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
